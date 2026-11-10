// Passive, first-replay-visit diagnostics for the existing shared Beta actor.
// Nothing in this file is used by the optimizer, policy, observations or reward.
#pragma once

// 5 observed schedule/time bins x 2 observed building-occupancy bins x 4
// mutually exclusive sensor states. The fourth state records overlapping
// heating/cooling thresholds instead of silently classifying it as cold.
constexpr int ZD_TIME=5, ZD_OCC=2, ZD_STATE=4, ZD_HEAD=2, ZD_ACTION=12, ZD_VALUE=6;
constexpr int ZD_BIN=ZD_TIME*ZD_OCC*ZD_STATE, ZD_SAMPLE_STRIDE=12, ZD_MAX_SAMPLE=32;
constexpr int ZD_ACTION_OFF=0;
constexpr int ZD_VALUE_OFF=ZD_BIN*ZD_HEAD*ZD_ACTION;
constexpr int ZD_SPREAD_OFF=ZD_VALUE_OFF+ZD_BIN*ZD_VALUE;
constexpr int ZD_SPREAD=3;
constexpr int ZD_BAD_OFF=ZD_SPREAD_OFF+ZD_TIME*ZD_OCC*ZD_HEAD*ZD_SPREAD;
constexpr int ZD_MC48_OFF=ZD_BAD_OFF+1;
constexpr int ZD_WIDTH=ZD_MC48_OFF+ZD_BIN*ZD_VALUE;
static double* g_zd_acc=nullptr;
static float g_zd_empty_obs=0.f;

__device__ __forceinline__ int zd_time_bin(const precision_t* o){
    float x=to_float(o[CN_OG]),y=to_float(o[CN_OG+4]);
    if (!isfinite(x)||!isfinite(y)) return -1;
    float a=atan2f(x,y);if(a<0.f)a+=6.28318530718f;
    int minute=(int)lrintf(a*(1440.f/6.28318530718f));minute%=1440;
    if(minute<0)minute+=1440;
    if(minute<360||minute>=1140)return 0;
    float comfort=to_float(o[CN_OG+16]);if(!isfinite(comfort))return -1;
    if(comfort<=0.5f)return 4;
    return minute<720?1:(minute<960?2:3);
}
__device__ __forceinline__ int zd_occupancy_bin(const precision_t* o,float empty){
    float x=to_float(o[CN_OG+18]);
    if(!isfinite(x)||x<empty-1e-5f)return -1;
    return x<=empty?0:1;
}
__device__ __forceinline__ int zd_state_bin(const precision_t* o,int z){
    float c=to_float(o[CN_FZ*z+2]),h=to_float(o[CN_FZ*z+3]);
    if(!isfinite(c)||!isfinite(h))return -1;
    bool cold=c<0.f,hot=h<0.f;
    return cold?(hot?3:0):(hot?2:1);
}

// One block per agent. Samples every hour on the first replay pass, so the
// policy mean is the current replay policy, while x is the behavior draw that
// was executed by the simulator. The beta cap leaves the mean unchanged.
__global__ void zd_collect_k(const precision_t* obs,const precision_t* mask,const float* actions,
        const precision_t* terminals,const precision_t* dec,const float* zpred,const float* zret,
        double* out,int N,int T,int width,int na,int zw,int agent0,float empty){
    const int agent=blockIdx.x,tid=threadIdx.x;
    if(agent>=N)return;
    __shared__ double sh[ZD_WIDTH];
    __shared__ double spatial[ZD_MAX_SAMPLE*ZD_HEAD*3]; // n, sum(m), sum(m^2)
    for(int i=tid;i<ZD_WIDTH;i+=blockDim.x)sh[i]=0.;
    for(int i=tid;i<ZD_MAX_SAMPLE*ZD_HEAD*3;i+=blockDim.x)spatial[i]=0.;
    __syncthreads();
    const int ns=(T+ZD_SAMPLE_STRIDE-1)/ZD_SAMPLE_STRIDE;
    if(ns>ZD_MAX_SAMPLE){asm("trap;");return;}
    for(int i=tid;i<ns*CN_ZS;i+=blockDim.x){
        const int sample=i/CN_ZS,z=i%CN_ZS,t=sample*ZD_SAMPLE_STRIDE;
        const int row=agent*T+t;const precision_t* o=obs+(size_t)row*OBS_SIZE;
        if(to_float(o[CN_FZ*z])==0.f)continue;
        const int tb=zd_time_bin(o),ob=zd_occupancy_bin(o,empty),st=zd_state_bin(o,z);
        if(tb<0||ob<0||st<0){atomicAdd(sh+ZD_BAD_OFF,1.);continue;}
        const int bin=(tb*ZD_OCC+ob)*ZD_STATE+st;
        bool any=false;
        for(int k=0;k<ZD_HEAD;++k){
            int h=2*z+k;
            if(h>=width||to_float(mask[(size_t)row*width+h])==0.f)continue;
            any=true;double al,be;sbs_beta_params(dec,N*T,width,row,h,&al,&be);
            const double mu=al/(al+be),x=actions[(size_t)row*na+h];
            if(!isfinite(mu)||!isfinite(x)||mu<0.||mu>1.||x<0.||x>1.){
                atomicAdd(sh+ZD_BAD_OFF,1.);continue;
            }
            double* q=sh+ZD_ACTION_OFF+((size_t)bin*ZD_HEAD+k)*ZD_ACTION;
            atomicAdd(q+0,1.);atomicAdd(q+1,mu);atomicAdd(q+2,mu*mu);
            atomicAdd(q+3,x);atomicAdd(q+4,x*x);
            atomicAdd(q+5,mu<.025?1.:0.);atomicAdd(q+6,mu>.975?1.:0.);
            atomicAdd(q+7,x<.025?1.:0.);atomicAdd(q+8,x>.975?1.:0.);
            if(t>0 && to_float(terminals[row])==0.f && to_float(mask[(size_t)(row-1)*width+h])!=0.f){
                const double prev=actions[(size_t)(row-1)*na+h];
                if(isfinite(prev)&&prev>=0.&&prev<=1.){
                    const double d=x-prev;atomicAdd(q+9,1.);atomicAdd(q+10,fabs(d));atomicAdd(q+11,d*d);
                }else atomicAdd(sh+ZD_BAD_OFF,1.);
            }
            double* p=spatial+((sample*ZD_HEAD+k)*3);
            atomicAdd(p+0,1.);atomicAdd(p+1,mu);atomicAdd(p+2,mu*mu);
        }
        if(any && zpred && zret && z<zw){
            const double pred=zpred[(size_t)row*zw+z],target=zret[(size_t)row*zw+z];
            if(!isfinite(pred)||!isfinite(target)){atomicAdd(sh+ZD_BAD_OFF,1.);continue;}
            const double residual=pred-target;double* q=sh+ZD_VALUE_OFF+(size_t)bin*ZD_VALUE;
            atomicAdd(q+0,1.);atomicAdd(q+1,pred);atomicAdd(q+2,target);
            atomicAdd(q+3,target*target);atomicAdd(q+4,residual);atomicAdd(q+5,residual*residual);
        }
    }
    __syncthreads();
    // Same-time spread is calculated across zones, not across time/agents.
    for(int i=tid;i<ns*ZD_HEAD;i+=blockDim.x){
        int sample=i/ZD_HEAD,k=i%ZD_HEAD,t=sample*ZD_SAMPLE_STRIDE,row=agent*T+t;
        const precision_t* o=obs+(size_t)row*OBS_SIZE;
        const int tb=zd_time_bin(o),ob=zd_occupancy_bin(o,empty);
        if(tb<0||ob<0)continue;
        const double* p=spatial+i*3;const double n=p[0];if(n<=0.)continue;
        double mu=p[1]/n,sd=sqrt(fmax(0.,p[2]/n-mu*mu));
        double* q=sh+ZD_SPREAD_OFF+((tb*ZD_OCC+ob)*ZD_HEAD+k)*ZD_SPREAD;
        atomicAdd(q+0,1.);atomicAdd(q+1,sd);atomicAdd(q+2,sd*sd);
    }
    __syncthreads();
    double* dst=out+(size_t)(agent0+agent)*ZD_WIDTH;
    for(int k=tid;k<ZD_WIDTH;k+=blockDim.x)dst[k]+=sh[k];
}

// Before this rollout's optimizer visit: compare the *stored rollout value*
// with 48 rewards and the stored value at t+48. The reward at t+1 follows the
// action at t. Stop at episode boundaries; exclude incomplete rollout tails.
// This is a bootstrapped diagnostic, not an independent Monte Carlo return.
__global__ void zd_mc48_k(const precision_t* obs,const precision_t* mask,
        const precision_t* terminals,const float* rewards,const float* oldv,
        double* out,int N,int T,int width,int zw,int agent0,float empty,float gamma){
    const int agent=blockIdx.x,tid=threadIdx.x;if(agent>=N||T<=48)return;
    __shared__ double sh[ZD_BIN*ZD_VALUE+1];
    for(int i=tid;i<ZD_BIN*ZD_VALUE+1;i+=blockDim.x)sh[i]=0.;
    __syncthreads();
    const int ns=(T-48+ZD_SAMPLE_STRIDE-1)/ZD_SAMPLE_STRIDE;
    for(int i=tid;i<ns*CN_ZS;i+=blockDim.x){
        int sample=i/CN_ZS,z=i%CN_ZS,t=sample*ZD_SAMPLE_STRIDE;
        int row=agent*T+t;const precision_t* o=obs+(size_t)row*OBS_SIZE;
        if(z>=zw||to_float(o[CN_FZ*z])==0.f)continue;
        if(to_float(mask[(size_t)row*width+2*z])==0.f &&
           to_float(mask[(size_t)row*width+2*z+1])==0.f)continue;
        int tb=zd_time_bin(o),ob=zd_occupancy_bin(o,empty),st=zd_state_bin(o,z);
        if(tb<0||ob<0||st<0){atomicAdd(sh+ZD_BIN*ZD_VALUE,1.);continue;}
        double target=0.,discount=1.;bool terminal=false;
        for(int k=1;k<=48;k++){
            int next=row+k;double r=rewards[(size_t)next*zw+z];
            double d=to_float(terminals[next]);
            if(!isfinite(r)||!isfinite(d)){target=NAN;break;}
            target+=discount*r;
            if(d!=0.){terminal=true;break;}
            discount*=gamma;
        }
        if(!terminal&&isfinite(target))target+=discount*oldv[(size_t)(row+48)*zw+z];
        double pred=oldv[(size_t)row*zw+z];
        if(!isfinite(pred)||!isfinite(target)){atomicAdd(sh+ZD_BIN*ZD_VALUE,1.);continue;}
        double residual=pred-target;
        double* q=sh+((tb*ZD_OCC+ob)*ZD_STATE+st)*ZD_VALUE;
        atomicAdd(q,1.);atomicAdd(q+1,pred);atomicAdd(q+2,target);
        atomicAdd(q+3,target*target);atomicAdd(q+4,residual);atomicAdd(q+5,residual*residual);
    }
    __syncthreads();
    double* dst=out+(size_t)(agent0+agent)*ZD_WIDTH;
    for(int k=tid;k<ZD_BIN*ZD_VALUE;k+=blockDim.x)dst[ZD_MC48_OFF+k]+=sh[k];
    if(tid==0)dst[ZD_BAD_OFF]+=sh[ZD_BIN*ZD_VALUE];
}

// r118: paired, hourly in-pass drift diagnostics. Cache and current estimates
// use the same pass-start normalization; no optimizer inputs are modified.
constexpr int ZP_COLS=10, ZP_WIDTH=ZD_BIN*ZP_COLS;
static double* g_zpass_diag=nullptr;
static float* g_zpass_stats=nullptr;
static int* g_zpass_counts=nullptr;
static int g_zpass_passes=0, g_zpass_assert=0;
__global__ void zpass_mark_k(const float* stats,float* saved,int* count,int slot,int pass,int passes){
    if(blockIdx.x || threadIdx.x)return;
    int i=slot*passes+pass;saved[2*i]=stats[0];saved[2*i+1]=stats[1];count[i]++;
}
__global__ void zpass_diag_k(const precision_t* obs,const precision_t* mask,
        const float* cached,const float* dynamic,const float* cached_ret,const float* dynamic_ret,
        const float* stats,double* out,int N,int T,int width,int zw,int agent0,int agents,
        int slot,int pass,int passes,int groups,int group_size,float empty){
    int agent=blockIdx.x,tid=threadIdx.x;if(agent>=N)return;
    __shared__ double sh[ZP_WIDTH];
    for(int k=tid;k<ZP_WIDTH;k+=blockDim.x)sh[k]=0.;
    __syncthreads();
    int ns=(T+ZD_SAMPLE_STRIDE-1)/ZD_SAMPLE_STRIDE;
    for(int i=tid;i<ns*CN_ZS;i+=blockDim.x){
        int z=i%CN_ZS,t=(i/CN_ZS)*ZD_SAMPLE_STRIDE,row=agent*T+t;
        const precision_t* o=obs+(size_t)row*OBS_SIZE;
        if(z>=zw || to_float(o[CN_FZ*z])==0.f)continue;
        if(to_float(mask[(size_t)row*width+2*z])==0.f && to_float(mask[(size_t)row*width+2*z+1])==0.f)continue;
        int tb=zd_time_bin(o),ob=zd_occupancy_bin(o,empty),st=zd_state_bin(o,z);
        if(tb<0||ob<0||st<0){asm("trap;");continue;}
        double a=cached[(size_t)row*zw+z],b=dynamic[(size_t)row*zw+z];
        double ar=cached_ret[(size_t)row*zw+z],br=dynamic_ret[(size_t)row*zw+z];
        double m=stats[0],scale=stats[1];
        if(!isfinite(a)||!isfinite(b)||!isfinite(ar)||!isfinite(br)||!isfinite(m)||!isfinite(scale)){
            asm("trap;");continue;
        }
        a=(a-m)*scale;b=(b-m)*scale;double d=b-a;
        double* q=sh+((tb*ZD_OCC+ob)*ZD_STATE+st)*ZP_COLS;
        atomicAdd(q,1.);atomicAdd(q+1,a);atomicAdd(q+2,b);
        atomicAdd(q+3,a*a);atomicAdd(q+4,b*b);atomicAdd(q+5,fabs(d));
        atomicAdd(q+6,d*d);atomicAdd(q+7,(a<0.&&b>0.)||(a>0.&&b<0.)?1.:0.);
        atomicAdd(q+8,fabs(br-ar));atomicAdd(q+9,br-ar);
    }
    __syncthreads();
    double* dst=out+(((size_t)slot*passes+pass)*groups+(agent0+agent)/group_size)*ZP_WIDTH;
    for(int k=tid;k<ZP_WIDTH;k+=blockDim.x)atomicAdd(dst+k,sh[k]);
}
__global__ void zpass_verify_k(const float* a,const float* b,const float* ca,const float* cb,size_t n){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
        if(__float_as_uint(a[i])!=__float_as_uint(ca[i]) || __float_as_uint(b[i])!=__float_as_uint(cb[i]))asm("trap;");
}

// First-visit AHU/plant action moments on the same time/occupancy bins as zone
// statistics. No product actions, observations, losses or rewards are altered.
constexpr int AP_HEADS=7,AP_COLS=12,AP_WIDTH=ZD_TIME*ZD_OCC*AP_HEADS*AP_COLS+1;
static double* g_ap_acc=nullptr;
__global__ void ap_collect_k(const precision_t* obs,const precision_t* mask,const float* actions,
        const precision_t* terminals,const precision_t* dec,double* out,
        int N,int T,int width,int na,int agent0,float empty){
    int agent=blockIdx.x,tid=threadIdx.x;if(agent>=N)return;
    __shared__ double sh[AP_WIDTH];
    for(int k=tid;k<AP_WIDTH;k+=blockDim.x)sh[k]=0.;
    __syncthreads();
    int ns=(T+ZD_SAMPLE_STRIDE-1)/ZD_SAMPLE_STRIDE;
    const int items=4*CN_K+3;
    for(int i=tid;i<ns*items;i+=blockDim.x){
        int t=(i/items)*ZD_SAMPLE_STRIDE,item=i%items,row=agent*T+t;
        int h=item<4*CN_K ? CN_AA+item : CN_AH+item-4*CN_K;
        int role=item<4*CN_K ? item%4 : 4+item-4*CN_K;
        if(h>=width || h>=na || to_float(mask[(size_t)row*width+h])==0.f)continue;
        const precision_t* o=obs+(size_t)row*OBS_SIZE;
        int tb=zd_time_bin(o),ob=zd_occupancy_bin(o,empty);
        if(tb<0||ob<0){atomicAdd(sh+AP_WIDTH-1,1.);continue;}
        double al,be;sbs_beta_params(dec,N*T,width,row,h,&al,&be);
        double mu=al/(al+be),x=actions[(size_t)row*na+h];
        if(!isfinite(mu)||!isfinite(x)||mu<0.||mu>1.||x<0.||x>1.){atomicAdd(sh+AP_WIDTH-1,1.);continue;}
        double* q=sh+((tb*ZD_OCC+ob)*AP_HEADS+role)*AP_COLS;
        atomicAdd(q,1.);atomicAdd(q+1,mu);atomicAdd(q+2,mu*mu);
        atomicAdd(q+3,x);atomicAdd(q+4,x*x);
        atomicAdd(q+5,mu<.025?1.:0.);atomicAdd(q+6,mu>.975?1.:0.);
        atomicAdd(q+7,x<.025?1.:0.);atomicAdd(q+8,x>.975?1.:0.);
        if(t>0 && to_float(terminals[row])==0.f && to_float(mask[(size_t)(row-1)*width+h])!=0.f){
            double prev=actions[(size_t)(row-1)*na+h];
            if(!isfinite(prev)||prev<0.||prev>1.){atomicAdd(sh+AP_WIDTH-1,1.);continue;}
            double d=x-prev;atomicAdd(q+9,1.);atomicAdd(q+10,fabs(d));atomicAdd(q+11,d*d);
        }
    }
    __syncthreads();
    double* dst=out+(size_t)(agent0+agent)*AP_WIDTH;
    for(int k=tid;k<AP_WIDTH;k+=blockDim.x)dst[k]+=sh[k];
}
