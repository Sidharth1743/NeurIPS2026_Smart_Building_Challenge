// Conditional raw-action Beta research path. Only for the canonical GPU policy.
#include "sb_r110_conditional_head.cuh"
static int g_cb_enabled=0;
static int g_cb_B=0,g_cb_T=0,g_cb_L=0,g_cb_H=0,g_cb_A=0,g_cb_slots=0;
static double *g_cb_raw_roll=nullptr,*g_cb_raw_bt=nullptr,*g_cb_old_roll=nullptr,*g_cb_old_bt=nullptr;
static double *g_cb_z=nullptr,*g_cb_live_y=nullptr,*g_cb_pref_y=nullptr,*g_cb_head_lp=nullptr;
static unsigned char *g_cb_live_valid=nullptr,*g_cb_pref_valid=nullptr;
static precision_t *g_cb_live_obs=nullptr,*g_cb_pref_obs=nullptr,*g_cb_live_st=nullptr,*g_cb_pref_st=nullptr,*g_cb_actor_st=nullptr;
static float *g_cb_live_feat=nullptr,*g_cb_pref_feat=nullptr,*g_cb_score=nullptr,*g_cb_prev_grad=nullptr;
static curandStatePhilox4_32_10_t *g_cb_rng=nullptr;
static int *g_cb_bad=nullptr;
static float g_cb_rho=0,g_cb_rho_ahu=0;
static const double *g_cb_loss_old=nullptr;
static const float *g_cb_loss_score=nullptr;
static float *g_cb_loss_prev=nullptr;

__device__ inline double cb_rho(int h,float r,float ra){return h>=180&&h<216?(double)ra:(double)r;}
__device__ inline void cb_fail(int* err){atomicExch(err,1); asm("trap;");}
// The conditional exploration state is per episode, just like the recurrent
// policy state. Clear before the first-row prefix is snapshotted as well as
// before ordinary sampling, so an asynchronous slot cannot retain old history.
__global__ void cb_clear_terminal(unsigned char* valid,const float* terminals,int B){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*SB_CONT)return;
 if(terminals[i/SB_CONT]!=0.f)valid[i]=0;
}
__global__ void cb_reencode(const precision_t* dec,const double* y,const unsigned char* valid,
                            double* z,int B,int A,int* bad){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*SB_CONT)return;
 int b=i/SB_CONT,h=i%SB_CONT;if(!valid[i])return;
 double u=to_float(dec[(size_t)b*(A+1)+h]);
 double v=to_float(dec[(size_t)B*(A+1)+(size_t)b*(A+1)+h]);
 auto sh=sb_cond_head::shapes(u,v,d_beta_floor,d_beta_conc_max,d_beta_mean_precision);
 auto l=sb_cond_beta::latent(sh.a,sh.b,y[i]);
 if(!l.ok){cb_fail(bad);return;}z[i]=l.z;
}
// One block per environment, one thread per continuous head. The RNG is private
// per head; absent heads emit zeros. Discrete slots are required to be masked.
__global__ void cb_sample(Prec dec,const precision_t* mask,float* acts,float* envact,
                          precision_t* old_f,precision_t* value,double* old_d,double* raw,
                          double* live_y,double* z,unsigned char* valid,
                          curandStatePhilox4_32_10_t* rng,int A,int B,float rho,float rho_ahu,int* bad){
 int b=blockIdx.x,h=threadIdx.x;if(b>=B)return;
 __shared__ double p[SB_CONT];
 if(h<SB_CONT){
  int i=b*SB_CONT+h;double lp=0.;
  if(to_float(mask[(size_t)b*A+h])!=0.f){
   double u=to_float(dec.data[(size_t)b*(A+1)+h]);
   double v=to_float(dec.data[(size_t)B*(A+1)+(size_t)b*(A+1)+h]);
   auto sh=sb_cond_head::shapes(u,v,d_beta_floor,d_beta_conc_max,d_beta_mean_precision);
   auto state=rng[i];double eps=curand_normal_double(&state);rng[i]=state;
   double r=cb_rho(h,rho,rho_ahu),prior=z[i],zz=valid[i]?r*prior+sqrt(1-r*r)*eps:eps;
   auto q=sb_cond_beta::inverse_logit(sh.a,sh.b,zz);
   if(!q.ok){cb_fail(bad);return;}
   double y=q.y;
   lp=sh.a*sb_cond_beta::logsigmoid(y)+sh.b*sb_cond_beta::logsigmoid(-y)
      -lgamma(sh.a)-lgamma(sh.b)+lgamma(sh.a+sh.b);
   if(valid[i]){double v0=1-r*r,d=zz-r*prior;lp+=-.5*d*d/v0+.5*zz*zz-.5*log(v0);}
   if(!sb_cond_beta::finite(lp)){cb_fail(bad);return;}
   float x=beta_clamp_x((float)sb_cond_head::sigmoid(y));
   acts[(size_t)b*NUM_ATNS+h]=x;envact[(size_t)b*NUM_ATNS+h]=2*x-1;
   raw[i]=y;live_y[i]=y;z[i]=zz;valid[i]=1;
   rec_put(b,h,2*beta_clamp_x((float)(sh.a/(sh.a+sh.b)))-1);
   rec_put_conc(b,h,(float)(sh.a+sh.b));
  }else{
   acts[(size_t)b*NUM_ATNS+h]=0;envact[(size_t)b*NUM_ATNS+h]=0;raw[i]=0;live_y[i]=0;valid[i]=0;
  }
  p[h]=lp;
 }
 for(int h2=SB_CONT+h;h2<NUM_ATNS;h2+=blockDim.x){
   int off=SB_CONT+2*(h2-SB_CONT);
   if(to_float(mask[(size_t)b*A+off])!=0.f || to_float(mask[(size_t)b*A+off+1])!=0.f){cb_fail(bad);return;}
   acts[(size_t)b*NUM_ATNS+h2]=0;envact[(size_t)b*NUM_ATNS+h2]=0;
 }
 __syncthreads();
 for(int stride=SB_CONT/2;stride;stride>>=1){if(h<stride)p[h]+=p[h+stride];__syncthreads();}
 if(h==0){old_d[b]=p[0];old_f[b]=from_float((float)p[0]);value[b]=dec.data[(size_t)b*(A+1)+A];}
}
__global__ void cb_transpose(const double* src,double* dst,int T,int B,int C){
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)T*B*C;if(i>=n)return;
 int c=i%C,b=(i/C)%B,t=i/(C*B);dst[((size_t)b*T+t)*C+c]=src[i];
}
// One thread per active head; output current and previous actor derivatives.
__global__ void cb_score_rows(const precision_t* dec,const precision_t* prefix,const precision_t* mask,
 const precision_t* terminals,
 const double* y,const double* py0,const unsigned char* valid0,double* lp,float* score,
 int N,int T,int A,float rho,float rho_ahu,int* bad){
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)N*T*SB_CONT;if(i>=n)return;
 int h=i%SB_CONT,row=i/SB_CONT,t=row%T,b=row/T;size_t j=(size_t)row*A+h;
 if(to_float(mask[j])==0){lp[i]=0;for(int k=0;k<4;k++)score[i*4+k]=0;return;}
 int P=A+1;const precision_t *second=dec+(size_t)N*T*P;
 double u=to_float(dec[(size_t)row*P+h]),v=to_float(second[(size_t)row*P+h]);
 bool prev=(terminals==nullptr || to_float(terminals[row])==0.f) &&
     (t>0 ? (to_float(mask[(size_t)(row-1)*A+h])!=0.f) : (valid0[(size_t)b*SB_CONT+h]!=0));
 double pu=0,pv=0,py=0;
 if(t>0 && prev){pu=to_float(dec[(size_t)(row-1)*P+h]);pv=to_float(second[(size_t)(row-1)*P+h]);py=y[i-SB_CONT];}
 else if(t==0 && prev){pu=to_float(prefix[(size_t)b*P+h]);pv=to_float(prefix[(size_t)N*P+(size_t)b*P+h]);py=py0[(size_t)b*SB_CONT+h];}
 auto s=sb_cond_head::score(u,v,y[i],pu,pv,py,cb_rho(h,rho,rho_ahu),d_beta_floor,d_beta_conc_max,prev,d_beta_mean_precision);
 if(!s.ok){cb_fail(bad);return;}
 lp[i]=s.logp;score[i*4]=(float)s.du;score[i*4+1]=(float)s.dv;
 score[i*4+2]=(float)s.pdu;score[i*4+3]=(float)s.pdv;
}
__global__ void cb_reduce_ratio(const double* lp,const double* old,float* ratio,precision_t* imp,int N,int T){
 int row=blockIdx.x,h=threadIdx.x;if(row>=N*T)return;__shared__ double x[SB_CONT];
 if(h<SB_CONT)x[h]=lp[(size_t)row*SB_CONT+h];__syncthreads();
 for(int s=SB_CONT/2;s;s>>=1){if(h<s)x[h]+=x[h+s];__syncthreads();}
 if(h==0){double lr=x[0]-old[row];ratio[row]=(float)lr;imp[row]=from_float((float)exp(fmin(20.,fmax(-20.,lr))));}
}
__global__ void cb_shift_grads(float* gu,float* gv,const float* pg,float* pgu,float* pgv,int N,int T,int A){
 size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,n=(size_t)N*T*SB_CONT;if(i>=n)return;
 int h=i%SB_CONT,row=i/SB_CONT,t=row%T,b=row/T;
 if(t+1<T){size_t k=(i+SB_CONT)*2;gu[(size_t)row*A+h]+=pg[k];gv[(size_t)row*A+h]+=pg[k+1];}
 if(t==0){size_t k=i*2;pgu[(size_t)b*A+h]=pg[k];pgv[(size_t)b*A+h]=pg[k+1];}
}
__global__ void cb_sum_grad(precision_t* main,const precision_t* prefix,long n){
 long i=(long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)main[i]=from_float(to_float(main[i])+to_float(prefix[i]));
}
static Allocator g_cb_pacts_alloc={},g_cb_pgrads_alloc={};
static Activations g_cb_pacts={};static Weights g_cb_pweights={};static MinGRUWeights g_cb_gru={};
static Float g_cb_pgu={},g_cb_pgv={},g_cb_pvalue={};static Prec g_cb_pterms={};
template<class T> static void cb_malloc(T** p,size_t n){
 if(cudaMalloc((void**)p,n*sizeof(T))!=cudaSuccess){fprintf(stderr,"conditional Beta allocation failed\n");abort();}
 if(cudaMemset(*p,0,n*sizeof(T))!=cudaSuccess){fprintf(stderr,"conditional Beta memset failed\n");abort();}
}
// Read-only per-group rollout statistics; policy samples are in [0,1].
static double *g_cb_diag=nullptr;static int g_cb_group_size=0,g_cb_groups=0;
__global__ void cb_diag_k(const float* act,const precision_t* mask,const double* old,double* out,
                          int B,int T,int A,int group_size){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*SB_CONT)return;
 int b=i/SB_CONT,h=i%SB_CONT,r=h<180?0:h<216?1:2,g=b/group_size;
 double s[12]={};
 for(int t=0;t<T;++t){
  size_t row=(size_t)b*T+t;if(to_float(mask[row*A+h])==0)continue;
  double x=act[row*NUM_ATNS+h];s[0]++;s[1]+=x;s[2]+=x*x;s[3]+=(x<=1.000001e-12f);s[4]+=(x>=.99999994f);
  if(t && to_float(mask[(row-1)*A+h])!=0){double p=act[(row-1)*NUM_ATNS+h];
   s[5]++;s[6]+=fabs(x-p);s[7]+=p;s[8]+=p*x;s[9]+=x;s[10]+=p*p;s[11]+=x*x;}
 }
 for(int k=0;k<12;++k)atomicAdd(out+(size_t)g*40+r*12+k,s[k]);
 if(h==0){double l=0;for(int t=0;t<T;++t)l+=old[(size_t)b*T+t];
  atomicAdd(out+(size_t)g*40+36,l);atomicAdd(out+(size_t)g*40+37,(double)T);}
}
