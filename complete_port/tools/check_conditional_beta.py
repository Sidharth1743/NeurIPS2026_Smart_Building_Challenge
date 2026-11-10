#!/usr/bin/env python3
"""Compile actual conditional trainer kernels and check their probabilities/gradients on GPU."""
from pathlib import Path
import argparse, subprocess
from check_ppo_math import definition
P=argparse.ArgumentParser();P.add_argument('--checkout',required=True);P.add_argument('--out',required=True);P.add_argument('--gpu',default='0');a=P.parse_args()
checkout=Path(a.checkout).resolve(); out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
s=(checkout/'src/sb_r110_conditional_kernels.cuh').read_text()
source=r'''
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "sb_r110_conditional_head.cuh"
#define SB_CONT 256
#define NUM_ATNS 272
using precision_t=float;
#define to_float(x) (x)
#define from_float(x) (x)
struct Prec{float* data;long shape[8];};
__device__ float d_beta_floor=1.0f,d_beta_conc_max=12.0f;
__device__ int d_beta_mean_precision=0;
__device__ float beta_clamp_x(float x){return fminf(.99999994f,fmaxf(1.e-12f,x));}
__device__ void rec_put(int,int,float){}
__device__ void rec_put_conc(int,int,float){}
'''
for n in ['cb_rho','cb_fail','cb_clear_terminal','cb_reencode','cb_sample','cb_score_rows','cb_reduce_ratio','cb_shift_grads']:
 source+=definition(s,n)
source+=r'''
#define OK(x) do {cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e));return 1;}}while(0)
template<class T>T* alloc(size_t n){T* p=nullptr;if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess)abort();memset(p,0,n*sizeof(T));return p;}
__global__ void init_rng(curandStatePhilox4_32_10_t* p){int i=threadIdx.x;curand_init(11075,i,0,&p[i]);}
int check_mode(int mp){
 OK(cudaMemcpyToSymbol(d_beta_mean_precision,&mp,sizeof(mp)));
 const int A=288,P=A+1;
 auto dec=alloc<float>(4*P),d0=alloc<float>(2*P),d1=alloc<float>(2*P),mask=alloc<float>(2*A),prefix=alloc<float>(2*P),term=alloc<float>(2);
 auto act=alloc<float>(2*NUM_ATNS),ea=alloc<float>(2*NUM_ATNS),oldf=alloc<float>(2),value=alloc<float>(2); auto old=alloc<double>(2),y=alloc<double>(2*SB_CONT);
 auto livey=alloc<double>(SB_CONT),z=alloc<double>(SB_CONT);auto valid=alloc<unsigned char>(SB_CONT),pvalid=alloc<unsigned char>(SB_CONT);
 auto score=alloc<float>(2*SB_CONT*4),diff=alloc<float>(2),imp=alloc<float>(2); auto lp=alloc<double>(2*SB_CONT);
 auto pg=alloc<float>(2*SB_CONT*2),gu=alloc<float>(2*A),gv=alloc<float>(2*A),pgu=alloc<float>(A),pgv=alloc<float>(A); auto bad=alloc<int>(1);
 auto rng=alloc<curandStatePhilox4_32_10_t>(SB_CONT);
 int heads[]={0,5,180};for(int h:heads){mask[h]=mask[A+h]=1;d0[h]=.2f+.01f*h;d0[P+h]=-.5f;d1[h]=-.4f;d1[P+h]=1.2f;}
 if(mp){d0[5]=-7.5f;d0[180]=7.5f;d1[5]=-6.5f;d1[180]=6.5f;}
 init_rng<<<1,SB_CONT>>>(rng);
 Prec x0={d0,{1,P}},x1={d1,{1,P}};
 cb_sample<<<1,SB_CONT>>>(x0,mask,act,ea,oldf,value,old,y,livey,z,valid,rng,A,1,expf(-1.f/12),expf(-1.f/48),bad);
 cb_sample<<<1,SB_CONT>>>(x1,mask+A,act+NUM_ATNS,ea+NUM_ATNS,oldf+1,value+1,old+1,y+SB_CONT,livey,z,valid,rng,A,1,expf(-1.f/12),expf(-1.f/48),bad);
 OK(cudaDeviceSynchronize());if(*bad){fprintf(stderr,"sampler bad\\n");return 1;}
 for(int h:heads){dec[h]=d0[h];dec[P+h]=d1[h];dec[2*P+h]=d0[P+h];dec[3*P+h]=d1[P+h];}
 cb_score_rows<<<2,SB_CONT>>>(dec,prefix,mask,term,y,livey,pvalid,lp,score,1,2,A,expf(-1.f/12),expf(-1.f/48),bad);
 cb_reduce_ratio<<<2,SB_CONT>>>(lp,old,diff,imp,1,2);
 OK(cudaDeviceSynchronize());
 if(fabs(diff[0])>2.e-4 || fabs(diff[1])>2.e-4){fprintf(stderr,"behavior/replay log ratios %g %g\\n",diff[0],diff[1]);return 2;}
 for(int h:heads){
   double rr=h==180?(double)expf(-1.f/48):(double)expf(-1.f/12);
   double u0=d0[h],v0=d0[P+h],u1=d1[h],v1=d1[P+h],yy0=y[h],yy1=y[SB_CONT+h],eps=1.e-5;
   auto f=[&](double aa,double bb,double cc,double dd){auto s0=sb_cond_head::score(aa,bb,yy0,0,0,0,rr,1,12,false,mp);auto s1=sb_cond_head::score(cc,dd,yy1,aa,bb,yy0,rr,1,12,true,mp);if(!s0.ok||!s1.ok)abort();return s0.logp+s1.logp;};
   double fd[4]={(f(u0+eps,v0,u1,v1)-f(u0-eps,v0,u1,v1))/(2*eps),
                 (f(u0,v0+eps,u1,v1)-f(u0,v0-eps,u1,v1))/(2*eps),
                 (f(u0,v0,u1+eps,v1)-f(u0,v0,u1-eps,v1))/(2*eps),
                 (f(u0,v0,u1,v1+eps)-f(u0,v0,u1,v1-eps))/(2*eps)};
   float* s0=score+h*4;float* s1=score+(SB_CONT+h)*4;
   double got[]={s0[0]+s1[2],s0[1]+s1[3],s1[0],s1[1]};
   for(int k=0;k<4;k++)if(fabs(fd[k]-got[k])>3e-4*(1+fabs(fd[k]))){fprintf(stderr,"grad h%d k%d finite %g GPU %g\\n",h,k,fd[k],got[k]);return 3;}
   if(!(ea[h]>=-1 && ea[h]<=1 && ea[NUM_ATNS+h]>=-1 && ea[NUM_ATNS+h]<=1))return 4;
   pg[(SB_CONT+h)*2]=s1[2];pg[(SB_CONT+h)*2+1]=s1[3];pg[h*2]=.3f;pg[h*2+1]=-.2f;
 }
 cb_shift_grads<<<2,SB_CONT>>>(gu,gv,pg,pgu,pgv,1,2,A);OK(cudaDeviceSynchronize());
 for(int h:heads){if(fabs(gu[h]-score[(SB_CONT+h)*4+2])>1e-6||fabs(pgu[h]-.3f)>1e-6)return 5;}
 // Prefix re-encoding under a changed actor snapshot, then compare sampler to replay.
 auto pref_y=alloc<double>(SB_CONT);
 for(int h:heads){prefix[h]=.6f;prefix[P+h]=-.2f;pvalid[h]=1;pref_y[h]=livey[h];}
 cb_reencode<<<1,SB_CONT>>>(prefix,pref_y,pvalid,z,1,A,bad);
 cb_sample<<<1,SB_CONT>>>(x1,mask,act,ea,oldf,value,old,y,livey,z,valid,rng,A,1,expf(-1.f/12),expf(-1.f/48),bad);
 cb_score_rows<<<1,SB_CONT>>>(d1,prefix,mask,term,y,pref_y,pvalid,lp,score,1,1,A,expf(-1.f/12),expf(-1.f/48),bad);
 cb_reduce_ratio<<<1,SB_CONT>>>(lp,old,diff,imp,1,1);
 OK(cudaDeviceSynchronize());if(fabs(diff[0])>2.e-4){fprintf(stderr,"boundary ratio %g\\n",diff[0]);return 6;}
 for(int h:heads){
   double rr=h==180?(double)expf(-1.f/48):(double)expf(-1.f/12),eps=1.e-5;
   auto f=[&](double a,double b){return sb_cond_head::score(d1[h],d1[P+h],y[h],a,b,pref_y[h],rr,1,12,true,mp).logp;};
   double da=(f(prefix[h]+eps,prefix[P+h])-f(prefix[h]-eps,prefix[P+h]))/(2*eps);
   double db=(f(prefix[h],prefix[P+h]+eps)-f(prefix[h],prefix[P+h]-eps))/(2*eps);
   if(fabs(da-score[4*h+2])>3e-4*(1+fabs(da))||fabs(db-score[4*h+3])>3e-4*(1+fabs(db)))return 7;
 }
 // An in-rollout terminal must clear the action history before sampling and
 // must suppress the previous actor gradient during replay. Capture the actual
 // kernels in a CUDA graph so this also covers the production execution path.
 OK(cudaMemset(valid,0,SB_CONT));OK(cudaMemset(pvalid,0,SB_CONT));
 term[0]=0.f;term[1]=1.f;init_rng<<<1,SB_CONT>>>(rng);OK(cudaDeviceSynchronize());
 cudaStream_t st;cudaGraph_t graph;cudaGraphExec_t graph_exec;OK(cudaStreamCreate(&st));
 OK(cudaStreamBeginCapture(st,cudaStreamCaptureModeGlobal));
 cb_clear_terminal<<<1,SB_CONT,0,st>>>(valid,term,1);
 cb_sample<<<1,SB_CONT,0,st>>>(x0,mask,act,ea,oldf,value,old,y,livey,z,valid,rng,A,1,expf(-1.f/12),expf(-1.f/48),bad);
 cb_clear_terminal<<<1,SB_CONT,0,st>>>(valid,term+1,1);
 cb_sample<<<1,SB_CONT,0,st>>>(x1,mask+A,act+NUM_ATNS,ea+NUM_ATNS,oldf+1,value+1,old+1,y+SB_CONT,livey,z,valid,rng,A,1,expf(-1.f/12),expf(-1.f/48),bad);
 cb_score_rows<<<2,SB_CONT,0,st>>>(dec,prefix,mask,term,y,livey,pvalid,lp,score,1,2,A,expf(-1.f/12),expf(-1.f/48),bad);
 cb_reduce_ratio<<<2,SB_CONT,0,st>>>(lp,old,diff,imp,1,2);
 OK(cudaStreamEndCapture(st,&graph));OK(cudaGraphInstantiate(&graph_exec,graph,nullptr,nullptr,0));
 OK(cudaGraphLaunch(graph_exec,st));OK(cudaStreamSynchronize(st));
 if(*bad||fabs(diff[0])>2.e-4||fabs(diff[1])>2.e-4){fprintf(stderr,"terminal graph sampler/replay ratio %g %g\\n",diff[0],diff[1]);return 8;}
 for(int h:heads){
   auto expected=sb_cond_head::score(d1[h],d1[P+h],y[SB_CONT+h],0,0,0,h==180?expf(-1.f/48):expf(-1.f/12),1,12,false,mp);
   const float* s1=score+(SB_CONT+h)*4;
   if(!expected.ok||fabs(s1[0]-expected.du)>3e-4*(1+fabs(expected.du))||fabs(s1[1]-expected.dv)>3e-4*(1+fabs(expected.dv))||s1[2]!=0||s1[3]!=0){fprintf(stderr,"terminal row kept prefix gradient on head %d\\n",h);return 9;}
 }
 // A terminal at the beginning of an async slot must be honored even if the
 // saved prefix validity was set by an earlier episode.
 for(int h:heads)pvalid[h]=1;
 term[0]=1.f;
 cb_score_rows<<<1,SB_CONT>>>(d1,prefix,mask,term,y+SB_CONT,pref_y,pvalid,lp,score,1,1,A,expf(-1.f/12),expf(-1.f/48),bad);
 OK(cudaDeviceSynchronize());for(int h:heads)if(score[4*h+2]!=0||score[4*h+3]!=0){fprintf(stderr,"terminal prefix gradient on head %d\\n",h);return 10;}
 auto two_valid=alloc<unsigned char>(2*SB_CONT);memset(two_valid,1,2*SB_CONT);term[0]=1.f;term[1]=0.f;
 cb_clear_terminal<<<2,SB_CONT>>>(two_valid,term,2);OK(cudaDeviceSynchronize());
 for(int h=0;h<SB_CONT;h++)if(two_valid[h]!=0||two_valid[SB_CONT+h]!=1){fprintf(stderr,"terminal reset leaked across agents\\n");return 11;}
 OK(cudaGraphExecDestroy(graph_exec));OK(cudaGraphDestroy(graph));OK(cudaStreamDestroy(st));
 printf("conditional mode %d sampler/replay, masks, finite-difference derivatives, terminal and graph reset: PASS\\n",mp);
 return 0;
}
int main(){int r=check_mode(0);if(r)return r;return check_mode(1);}
'''
(out/'check.cu').write_text(source)
cmd=['docker','run','--rm','--gpus',f'device={a.gpu}','-v',f'{checkout}:/pl:ro','-v',f'{out}:/out','-w','/out','pufferai/puffertank:5.0','bash','-lc','nvcc -O2 -std=c++17 -I/pl/src check.cu -o check && ./check']
r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);(out/'result.log').write_text(r.stdout);print(r.stdout,end='');raise SystemExit(r.returncode)
