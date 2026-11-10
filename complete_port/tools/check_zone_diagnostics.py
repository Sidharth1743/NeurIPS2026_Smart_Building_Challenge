#!/usr/bin/env python3
"""Exercise the actual passive zone diagnostic CUDA kernel with known data."""
import argparse
from pathlib import Path
import subprocess

CP=Path(__file__).resolve().parents[1]
SOURCE=r'''
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define OBS_SIZE 1024
#define CN_OG 964
#define CN_FZ 9
#define CN_ZS 90
#define CN_OA 810
#define CN_FA 4
#define CN_K 9
#define CN_AA 180
#define CN_AH 216
using precision_t=float;
#define to_float(x) (x)
__device__ __forceinline__ void sbs_beta_params(const float*,int,int,int,int h,double* a,double* b){
    if(h==0){*a=3.;*b=1.;}else if(h==2){*a=1.;*b=3.;}else {*a=2.;*b=2.;}
}
#include "sb_zone_diagnostics.cuh"
// GENERATED_ZONE_KERNELS
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e));exit(2);}}while(0)
template<class T>T* managed(size_t n){T* p=nullptr;CK(cudaMallocManaged(&p,n*sizeof(T)));for(size_t i=0;i<n;i++)p[i]=0;return p;}
void near(const char* name,double actual,double expected){
    if(!std::isfinite(actual)||fabs(actual-expected)>1e-5){fprintf(stderr,"FAIL %s got %.12g expected %.12g\n",name,actual,expected);exit(1);}
}
int main(){
    constexpr int N=2,T=36,W=288,A=272,ZW=90;
    float* obs=managed<float>((size_t)N*T*OBS_SIZE);float* mask=managed<float>((size_t)N*T*W);
    float* acts=managed<float>((size_t)N*T*A);float* terms=managed<float>(N*T);
    float* dec=managed<float>((size_t)N*T*(W+1)*2);float* pred=managed<float>((size_t)N*T*ZW);
    float* ret=managed<float>((size_t)N*T*ZW);double* out=managed<double>((size_t)N*ZD_WIDTH);
    for(int b=0;b<N;b++)for(int t=0;t<T;t++){
        int row=b*T+t;float* o=obs+(size_t)row*OBS_SIZE;
        float hour=b?14.f:8.f;float angle=hour*6.28318530718f/24.f;
        o[CN_OG]=sinf(angle);o[CN_OG+4]=cosf(angle);o[CN_OG+16]=1.f;o[CN_OG+18]=b?-.49f:-.5f;
        o[0]=1.f;o[2]=b?1.f:-1.f;o[3]=1.f;
        if(!b){o[9]=1.f;o[11]=1.f;o[12]=-1.f;}
        for(int h=0;h<(b?2:4);h++){
            if(b && h==0)continue;
            mask[(size_t)row*W+h]=1.f;acts[(size_t)row*A+h]=h==0?.02f:(h==1?.98f:(h==2?.25f:.75f));
        }
        pred[(size_t)row*ZW]=2;ret[(size_t)row*ZW]=3;
        pred[(size_t)row*ZW+1]=4;ret[(size_t)row*ZW+1]=2;
    }
    terms[12]=1.f; // the sampled t=12 transition crosses an episode boundary
    acts[(size_t)23*A]=.60f; // the sampled t=24 transition is valid
    mask[(size_t)23*W+3]=0.f; // absent previous head must not count a transition
    zd_collect_k<<<N,128>>>(obs,mask,acts,terms,dec,pred,ret,out,N,T,W,A,ZW,0,-.5f);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    auto bin=[](int time,int occ,int state){return (time*ZD_OCC+occ)*ZD_STATE+state;};
    double* cold=out+ZD_ACTION_OFF+((bin(1,0,0)*ZD_HEAD+0)*ZD_ACTION);
    near("cold_count",cold[0],3);near("cold_mu",cold[1],2.25);near("cold_mu2",cold[2],1.6875);
    near("cold_draw",cold[3],.06);near("draw_low",cold[7],3);
    near("terminal_mask_transition_count",cold[9],1);near("delta_absolute",cold[10],.58);near("delta_square",cold[11],.3364);
    double* hot=out+ZD_ACTION_OFF+((bin(1,0,2)*ZD_HEAD+1)*ZD_ACTION);
    near("hot_count",hot[0],3);near("masked_previous_transition_count",hot[9],0);
    double* value=out+ZD_VALUE_OFF+bin(1,0,0)*ZD_VALUE;
    near("critic_count",value[0],3);near("critic_prediction",value[1],6);near("critic_target",value[2],9);
    near("critic_target2",value[3],27);near("critic_residual",value[4],-3);near("critic_residual2",value[5],3);
    double* spread=out+ZD_SPREAD_OFF+((1*ZD_OCC+0)*ZD_HEAD+0)*ZD_SPREAD;
    near("spread_count",spread[0],3);near("same_time_spread",spread[1],.75);near("same_time_spread2",spread[2],.1875);
    double* other=out+ZD_WIDTH+ZD_ACTION_OFF+((bin(2,1,1)*ZD_HEAD+1)*ZD_ACTION);
    near("occupied_afternoon_band_count",other[0],3);near("high_draw",other[8],3);
    near("masked_damper",out[ZD_WIDTH+ZD_ACTION_OFF+((bin(2,1,1)*ZD_HEAD)*ZD_ACTION)],0);
    near("nonfinite_agent0",out[ZD_BAD_OFF],0);near("nonfinite_agent1",out[ZD_WIDTH+ZD_BAD_OFF],0);
    // Real AHU/plant diagnostic: masks, one-hour samples and terminal-safe jitter.
    double* ap=managed<double>((size_t)N*AP_WIDTH);
    for(int b=0;b<N;b++)for(int t=0;t<T;t++){
        int row=b*T+t;
        mask[(size_t)row*W+CN_AA]=1;acts[(size_t)row*A+CN_AA]=.75f;
        mask[(size_t)row*W+CN_AH]=1;acts[(size_t)row*A+CN_AH]=.25f;
    }
    acts[(size_t)23*A+CN_AA]=.25f;
    ap_collect_k<<<N,128>>>(obs,mask,acts,terms,dec,ap,N,T,W,A,0,-.5f);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    double* apahu=ap+((1*ZD_OCC+0)*AP_HEADS)*AP_COLS;
    near("ahu n",apahu[0],3);near("ahu means",apahu[1],1.5);near("ahu draws",apahu[3],2.25);
    near("ahu transitions terminal masked",apahu[9],1);near("ahu jitter",apahu[10],.5);
    double* app=ap+((1*ZD_OCC+0)*AP_HEADS+4)*AP_COLS;
    near("plant n",app[0],3);near("plant draws",app[3],.75);
    near("masked ahu action",ap[((1*ZD_OCC+0)*AP_HEADS+1)*AP_COLS],0);
    near("AP nonfinite",ap[AP_WIDTH-1],0);
    // Exercise the 48-step diagnostic with one terminal and one bootstrap.
    constexpr int T48=72;
    float* o48=managed<float>((size_t)T48*OBS_SIZE);float* m48=managed<float>((size_t)T48*W);
    float* done48=managed<float>(T48);float* r48=managed<float>((size_t)T48*ZW);
    float* v48=managed<float>((size_t)T48*ZW);double* out48=managed<double>(ZD_WIDTH);
    for(int t=0;t<T48;t++){
        float* o=o48+(size_t)t*OBS_SIZE;o[0]=1.f;o[2]=-1.f;o[3]=1.f;
        float hr=t==12?14.f:8.f,angle=hr*6.28318530718f/24.f;
        o[CN_OG]=sinf(angle);o[CN_OG+4]=cosf(angle);o[CN_OG+16]=1.f;o[CN_OG+18]=-.5f;
        m48[(size_t)t*W]=1.f;r48[(size_t)t*ZW]=1.f;v48[(size_t)t*ZW]=4.f;
    }
    done48[5]=1.f;
    zd_mc48_k<<<1,128>>>(o48,m48,done48,r48,v48,out48,1,T48,W,ZW,0,-.5f,.9f);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    double* terminal=out48+ZD_MC48_OFF+bin(1,0,0)*ZD_VALUE;
    double* bootstrap=out48+ZD_MC48_OFF+bin(2,0,0)*ZD_VALUE;
    double term_target=1.+.9+.81+.729+.6561;
    double boot_target=(1.-pow(.9,48))/.1+4.*pow(.9,48);
    near("mc48 terminal count",terminal[0],1);near("mc48 terminal target",terminal[2],term_target);
    near("mc48 terminal residual",terminal[4],4.-term_target);
    near("mc48 bootstrap count",bootstrap[0],1);near("mc48 bootstrap target",bootstrap[2],boot_target);
    near("mc48 rollout tail excluded",terminal[0]+bootstrap[0],2);
    near("mc48 nonfinite",out48[ZD_BAD_OFF],0);

    // Actual cache diagnostic: two slots and two passes; untouched slots stay zero.
    float* stats=managed<float>(2);stats[0]=2.5f;stats[1]=2.f;
    float* marks=managed<float>(8);int* counts=managed<int>(4);
    double* zd=managed<double>((size_t)2*2*2*ZP_WIDTH);
    zpass_mark_k<<<1,1>>>(stats,marks,counts,1,1,2);
    zpass_diag_k<<<N,128>>>(obs,mask,pred,ret,pred,ret,stats,zd,N,T,W,ZW,0,N,1,1,2,2,1,-.5f);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    near("slot isolation count",counts[0],0);near("refresh count",counts[3],1);
    near("saved mean",marks[6],2.5);near("saved inverse std",marks[7],2);
    near("untouched slot",zd[bin(1,0,0)*ZP_COLS],0);
    double* pair=zd+((1*2+1)*2)*ZP_WIDTH+bin(1,0,0)*ZP_COLS;
    near("paired count",pair[0],3);near("cached normalized",pair[1],-3);
    near("dynamic normalized",pair[2],3);near("difference absolute",pair[5],6);
    near("paired sign change",pair[7],3);near("return difference",pair[8],3);
    double* second=zd+((1*2+1)*2+1)*ZP_WIDTH+bin(2,1,1)*ZP_COLS;
    near("second agent partially masked head remains present",second[0],3);
    // Exercise the production GAE kernel: reward at t+1 and terminal mask.
    constexpr int GH=4,GZ=1;float* gv=managed<float>(GH);float* gr=managed<float>(GH);
    float* gd=managed<float>(GH);float* ga=managed<float>(GH);float* gt=managed<float>(GH);
    for(int t=0;t<GH;t++){gv[t]=t+1;gr[t]=1;}gd[2]=1;
    sbs_zone_gae_k<<<1,256>>>(gv,gr,gd,ga,gt,1,GH,.9f,.8f,GZ);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    near("gae terminal advantage",ga[1],-1);near("gae before terminal",ga[0],1+.9*2-1-.72);
    near("gae last bootstrap",gt[3],4);near("gae terminal target",gt[1],1);
    // Production present-zone statistics ignore absent slots, even if their values are large.
    float* sobs=managed<float>(2*OBS_SIZE);float* sa=managed<float>(2*CN_ZS);
    float* pairs=managed<float>(3);float* norm=managed<float>(2);
    sobs[0]=1;sobs[OBS_SIZE]=1;sa[0]=1;sa[CN_ZS]=3;sa[1]=10000;
    sbs_zone_stat_k<<<1,256>>>(sa,sobs,OBS_SIZE,1,2,pairs,CN_ZS,0);
    sbs_zone_stat_finish_k<<<1,1>>>(pairs,1,norm);
    CK(cudaGetLastError());CK(cudaDeviceSynchronize());
    near("present zone count",pairs[2],2);near("present zone mean",norm[0],2);
    near("present zone inverse sd",norm[1],1);

    puts("PASS actual zone diagnostic kernel: time, occupancy, masks, terminal resets, action moments, critic and spatial spread");
}
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',default='0');a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    shared=(CP/'patches/src/sb_shared_policy.cuh').read_text()
    gae=shared[shared.index('__global__ void sbs_zone_gae_k'):shared.index('void sbs_zone_gae(')]
    stat=shared[shared.index('__global__ void sbs_zone_stat_k'):shared.index('void sbs_zone_stat(')]
    finish=shared[shared.index('__global__ void sbs_zone_stat_finish_k'):shared.index('void sbs_zone_stat_finish(')]
    (out/'check.cu').write_text(SOURCE.replace('// GENERATED_ZONE_KERNELS',gae+'\n'+stat+'\n'+finish))
    rel=out.relative_to(CP)
    docker=['docker','run','--rm','--gpus',f'device={a.gpu}','-v',f'{CP}:/cp','-w','/cp','pufferai/puffertank:5.0']
    subprocess.run(docker+['nvcc','-O2','-std=c++17','-arch=sm_86','-I/cp/patches/src',f'/cp/{rel}/check.cu','-o',f'/cp/{rel}/check'],check=True)
    done=subprocess.run(docker+[f'/cp/{rel}/check'],text=True,capture_output=True)
    (out/'result.txt').write_text(done.stdout+done.stderr)
    print(done.stdout,end='');print(done.stderr,end='')
    raise SystemExit(done.returncode)
if __name__=='__main__':main()
