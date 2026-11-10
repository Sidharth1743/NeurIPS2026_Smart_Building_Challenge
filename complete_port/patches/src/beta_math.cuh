/* Numerical primitives prototype; NOT part of the trainer or a gate.
 * Doubles preserve tail probabilities and shape derivatives; no action clamp.
 * Compiles as host C++ or CUDA.  The censored-history filter is separate.
 */
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define SB_HD __host__ __device__
#else
#define SB_HD
#endif
namespace sb_cond_beta {
constexpr double LN2=0.693147180559945309417232121458;
constexpr double LOG2PI=1.837877066409345483560659472811;
constexpr double SQRT2=1.414213562373095048801688724210;
SB_HD inline bool finite(double x){return x==x && ::fabs(x)<INFINITY;}
struct Dual { double v,a,b; };
SB_HD inline Dual v(double x){return {x,0,0};}
SB_HD inline Dual add(Dual x,Dual y){return {x.v+y.v,x.a+y.a,x.b+y.b};}
SB_HD inline Dual sub(Dual x,Dual y){return {x.v-y.v,x.a-y.a,x.b-y.b};}
SB_HD inline Dual mul(Dual x,Dual y){return {x.v*y.v,x.a*y.v+x.v*y.a,x.b*y.v+x.v*y.b};}
SB_HD inline Dual div(Dual x,Dual y){double u=1.0/y.v;return {x.v*u,(x.a-x.v*u*y.a)*u,(x.b-x.v*u*y.b)*u};}
SB_HD inline Dual keep(Dual x){return ::fabs(x.v)>=1e-280?x:Dual{x.v>=0?1e-280:-1e-280,0,0};}
SB_HD inline double digamma(double x){
    double r=0;while(x<12.0){r-=1.0/x;x+=1.0;}
    double q=1.0/(x*x);
    return r+::log(x)-.5/x-q*(1.0/12.0-q*(1.0/120.0-q*(1.0/252.0-q*(1.0/240.0-q*(1.0/132.0)))));
}
SB_HD inline double logsigmoid(double x){return x>=0?-::log1p(::exp(-x)):x-::log1p(::exp(x));}
SB_HD inline double log1mexp(double x){return x> -LN2 ? ::log(-::expm1(x)) : ::log1p(-::exp(x));}
struct CFResult { Dual f; int n; bool ok; };
SB_HD inline CFResult fraction(double a,double b,double x){
    Dual A{a,1,0},B{b,0,1},qab=add(A,B),qap=add(A,v(1)),qam=sub(A,v(1));
    Dual c=v(1),d=div(v(1),keep(sub(v(1),div(mul(qab,v(x)),qap)))),h=d;
    for(int m=1;m<=2000;++m){
        int mm=2*m;
        Dual aa=div(mul(mul(v(m*x),sub(B,v(m))),v(1)),mul(add(qam,v(mm)),add(A,v(mm))));
        d=div(v(1),keep(add(v(1),mul(aa,d))));c=keep(add(v(1),div(aa,c)));h=mul(mul(h,d),c);
        aa=div(mul(v(-x),mul(add(A,v(m)),add(qab,v(m)))),mul(add(A,v(mm)),add(qap,v(mm))));
        d=div(v(1),keep(add(v(1),mul(aa,d))));c=keep(add(v(1),div(aa,c)));Dual delta=mul(d,c);h=mul(h,delta);
        if(::fabs(delta.v-1)<3e-14 && ::fabs(delta.a)<3e-12 && ::fabs(delta.b)<3e-12)return {h,m,true};
    }
    return {h,2000,false};
}
struct Tail {double logv, da, db; int iterations; bool ok;};
SB_HD inline Tail small_tail(double a,double b,double logx,double log1mx){
    double x=logx>-745?::exp(logx):0.;CFResult cf=fraction(a,b,x);
    if(!cf.ok||!(cf.f.v>0))return {0,0,0,cf.n,false};
    double ab=digamma(a+b);
    double value=a*logx+b*log1mx-::lgamma(a)-::lgamma(b)+::lgamma(a+b)-::log(a)+::log(cf.f.v);
    double da=logx-digamma(a)+ab-1/a+cf.f.a/cf.f.v;
    double db=log1mx-digamma(b)+ab+cf.f.b/cf.f.v;
    return {value,da,db,cf.n,finite(value)&&value<0&&finite(da)&&finite(db)};
}
struct LogTails {double lf,ls,fa,fb,sa,sb;int iterations;bool ok;};
SB_HD inline LogTails logtails(double a,double b,double y){
    if(!(a>0&&b>0))return {0,0,0,0,0,0,0,false};
    double lx=logsigmoid(y),l1=logsigmoid(-y);
    if(lx < ::log((a+1)/(a+b+2))){
        Tail t=small_tail(a,b,lx,l1);if(!t.ok)return {0,0,0,0,0,0,t.iterations,false};
        double other=log1mexp(t.logv),scale=-::exp(t.logv-other);
        return {t.logv,other,t.da,t.db,scale*t.da,scale*t.db,t.iterations,true};
    }
    Tail t=small_tail(b,a,l1,lx);if(!t.ok)return {0,0,0,0,0,0,t.iterations,false};
    double other=log1mexp(t.logv),scale=-::exp(t.logv-other);
    return {other,t.logv,scale*t.db,scale*t.da,t.db,t.da,t.iterations,true};
}
SB_HD inline double normal_logcdf(double z){
    if(z>0)return ::log1p(-.5*::erfc(z/SQRT2));
    if(z>=-30)return ::log(.5*::erfc(-z/SQRT2));
    double q=1/(z*z),term=1,sum=1;
    for(int k=1;k<=12;++k){term*=-(2*k-1)*q;sum+=term;if(::fabs(term)<1e-17*::fabs(sum))break;}
    return -.5*z*z-.5*LOG2PI-::log(-z)+::log(sum);
}
SB_HD inline double normal_logpdf(double z){return -.5*z*z-.5*LOG2PI;}
/* logp <= log(0.5), invert Phi(z)=exp(logp), z <= 0. */
SB_HD inline double normal_quantile_log(double logp){
    if(!(logp<=-LN2))return NAN;
    double z=-::sqrt(-2*logp);
    for(int i=0;i<32;++i){
        double f=normal_logcdf(z)-logp,deriv=::exp(normal_logpdf(z)-normal_logcdf(z));
        double dz=f/deriv;z-=dz;
        if(::fabs(dz)<2e-14*(1+::fabs(z)))break;
    }
    return z;
}
struct Latent {double z,da,db;bool ok;};
SB_HD inline Latent latent(double a,double b,double y){
    LogTails t=logtails(a,b,y);if(!t.ok)return {0,0,0,false};
    if(t.lf<=t.ls){
        double z=normal_quantile_log(t.lf),fac=::exp(t.lf-normal_logpdf(z));
        return {z,fac*t.fa,fac*t.fb,finite(z)&&finite(fac*t.fa)&&finite(fac*t.fb)};
    }
    double z=-normal_quantile_log(t.ls),fac=-::exp(t.ls-normal_logpdf(z));
    return {z,fac*t.sa,fac*t.sb,finite(z)&&finite(fac*t.sa)&&finite(fac*t.sb)};
}
/* Raw internal action y=logit(x). A fixed, parameter-independent clamp maps it
 * to the submitted float action; PPO operates on this continuous auxiliary action.
 * Both current and preceding shape derivatives are required. */
struct Conditional {double logp,da,db,dpa,dpb;bool ok;};
SB_HD inline Conditional raw_likelihood(double a,double b,double y,double pa,double pb,double py,double rho,bool have_prev){
    if(!(a>0&&b>0&&rho>=0&&rho<1))return {0,0,0,0,0,false};
    double lx=logsigmoid(y),l1=logsigmoid(-y);
    double base=a*lx+b*l1-::lgamma(a)-::lgamma(b)+::lgamma(a+b);
    double ba=lx-digamma(a)+digamma(a+b),bb=l1-digamma(b)+digamma(a+b);
    if(!have_prev)return {base,ba,bb,0,0,finite(base)&&finite(ba)&&finite(bb)};
    Latent cur=latent(a,b,y),pre=latent(pa,pb,py);
    if(!cur.ok||!pre.ok)return {0,0,0,0,0,false};
    double variance=1-rho*rho,d=cur.z-rho*pre.z;
    double L=base-.5*d*d/variance+.5*cur.z*cur.z-.5*::log(variance);
    double gc=cur.z-d/variance,gp=rho*d/variance;
    return {L,ba+gc*cur.da,bb+gc*cur.db,gp*pre.da,gp*pre.db,
            finite(L)&&finite(ba+gc*cur.da)&&finite(bb+gc*cur.db)&&finite(gp*pre.da)&&finite(gp*pre.db)};
}
struct Inverse {double y;int iterations;bool ok;};
SB_HD inline Inverse inverse_logit(double a,double b,double z){
    if(!(a>0&&b>0&&finite(z)))return {0,0,false};
    bool left=z<0;double target=normal_logcdf(left?z:-z);
    double lb=::lgamma(a)+::lgamma(b)-::lgamma(a+b);
    double y0=left?(target+::log(a)+lb)/a:-(target+::log(b)+lb)/b;
    double lo=left?::fmin(-1.,2*y0) : ::fmin(-1.,y0-2.);
    double hi=left?::fmax(1.,y0+2.) : ::fmax(1.,2*y0);
    for(int i=0;i<128;++i){LogTails t=logtails(a,b,lo);if(!t.ok)return {0,i,false};
        bool below=left?t.lf<=target:t.ls>=target;if(below)break;lo=2*lo-1;}
    for(int i=0;i<128;++i){LogTails t=logtails(a,b,hi);if(!t.ok)return {0,i,false};
        bool below=left?t.lf<=target:t.ls>=target;if(!below)break;hi=2*hi+1;}
    double y=::fmax(lo,::fmin(hi,y0));
    for(int i=0;i<128;++i){
        LogTails t=logtails(a,b,y);if(!t.ok)return {y,i,false};
        double L=left?t.lf:t.ls,res=L-target;
        if(::fabs(res)<2e-11*(1+::fabs(target)))return {y,i+1,true};
        bool below=left?L<=target:L>=target;if(below)lo=y;else hi=y;
        double dl=::exp(a*logsigmoid(y)+b*logsigmoid(-y)-lb-L);if(!left)dl=-dl;
        double p=dl!=0?y-res/dl:NAN;
        y=(p>lo&&p<hi)?p:(lo+hi)/2;
    }
    return {y,128,false};
}
}
