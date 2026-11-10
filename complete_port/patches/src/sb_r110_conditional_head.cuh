/* CPU/CUDA prototype only. Not connected to PufferLib or a training gate.
 * Raw logit-action likelihood including both time-adjacent actor derivatives.
 */
#pragma once
#include "beta_math.cuh"
namespace sb_cond_head {
using sb_cond_beta::finite;
struct Shapes { double a,b, au,av,bu,bv; };
SB_HD inline double softplus(double x){return fmax(x,0.)+log1p(exp(-fabs(x)));}
SB_HD inline double sigmoid(double x){return x>=0.?1./(1.+exp(-x)):exp(x)/(1.+exp(x));}
SB_HD inline Shapes shapes(double u,double v,double floor,double cap,int mean_precision=0){
    if(mean_precision){
        // Independent location and total concentration. Compute both sigmoid
        // tails directly to retain a positive small shape near either boundary.
        const double mu=sigmoid(u),other=sigmoid(-u),q=sigmoid(v),qo=sigmoid(-v);
        const double k=2.0+10.0*q,dk=10.0*q*qo,dmu=mu*other;
        return {k*mu,k*other,k*dmu,mu*dk,-k*dmu,other*dk};
    }
    const double a=softplus(u)+floor,b=softplus(v)+floor,du=sigmoid(u),dv=sigmoid(v),s=a+b;
    if(cap>0 && s>cap){
        const double k=cap/s,ka=k*a/s,kb=k*b/s;
        return {k*a,k*b,kb*du,-ka*dv,-kb*du,ka*dv};
    }
    return {a,b,du,0.,0.,dv};
}
struct Score {double logp,du,dv,pdu,pdv;bool ok;};
SB_HD inline Score score(double u,double v,double y,double pu,double pv,double py,
                         double rho,double floor,double cap,bool has_prev,int mean_precision=0){
    Shapes c=shapes(u,v,floor,cap,mean_precision),p=shapes(pu,pv,floor,cap,mean_precision);
    auto r=sb_cond_beta::raw_likelihood(c.a,c.b,y,p.a,p.b,py,rho,has_prev);
    Score s={r.logp,r.da*c.au+r.db*c.bu,r.da*c.av+r.db*c.bv,
             r.dpa*p.au+r.dpb*p.bu,r.dpa*p.av+r.dpb*p.bv,r.ok};
    s.ok=s.ok&&finite(s.logp)&&finite(s.du)&&finite(s.dv)&&finite(s.pdu)&&finite(s.pdv);
    return s;
}
}
