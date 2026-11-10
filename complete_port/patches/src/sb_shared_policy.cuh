// Pooled recurrent context with shared local actuator heads.
// Included by ocean.cu only for SB_SHARED_POLICY. Physics and the action API
// are unchanged. Inputs are sensors, schedule and metadata-derived slot masks;
// no PID proposal or PID state is read. Parameters start randomly.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>
#include <unistd.h>

constexpr int SBS_POOL = 80;
constexpr int SBS_FEAT = 11;
// Same-handler sibling block. Order is the weight layout: temp, damper,
// reheat, presence. Slots are CN_ZPA*k+j. A present sibling writes its three
// canonical sensors and 1. The zone's own slot is 0,0,0,0. An empty slot is
// 0,0,0,0. Do not reorder these four fields.
constexpr int SBS_SIB = 4;
constexpr int SBS_LZ = 17 + SBS_FEAT + CN_ZPA * SBS_SIB;
constexpr int SBS_LA = 32;
constexpr int SBS_LH = 16;
constexpr int SBS_LOCAL[3] = {SBS_LZ, SBS_LA, SBS_LH};
constexpr int SBS_NODES[3] = {CN_ZS, CN_K, 1};
constexpr int SBS_OUTPUTS[3] = {2, 4, 3};
// Host tensor descriptor only. Encoder then decoder is the architecture's
// forward contract; graph capture records the device addresses, not this TLS.
static thread_local Prec sbs_raw_input;
// Training switches. Off, the actor is the original shared net plus zero
// features, and the critic is one head on the pooled state. value_delay
// keeps that one head and appends the delay summary, so the advantage stays
// the scalar return. component heads are a separate path and are not combined
// with the delay summary.
int g_sbs_components = 0;
int g_sbs_timeout = 0;
int g_sbs_value_extra = 0;
// Explicit compatibility with the scalar recurrent critic used by r40.
int g_sbs_value_shared = 0;
// Per-rollout audit of zone rewards versus the building reward. The occupancy
// split uses the recorded observation; it can differ by one step at transitions.
static int g_sbs_za_group=0, g_sbs_za_groups=0;
static double* g_sbs_za=nullptr;
__global__ void sbs_zone_audit_k(const float* rz,const precision_t* rewards,const precision_t* obs,
        double* out,int B,int T,int width,int group_size){
    int b=blockIdx.x,tid=threadIdx.x;if(b>=B)return;
    __shared__ double acc[8][128];double v[8]={};
    for(int t=tid;t<T;t+=128){
        size_t row=(size_t)b*T+t;
        int k=to_float(obs[row*OBS_SIZE+CN_OG+18])==0.f?0:4;
        double z=0;for(int s=0;s<CN_ZS;++s)z+=rz[row*width+s];
        double r=to_float(rewards[row]);
        v[k]+=1.;v[k+1]+=z;v[k+2]+=r;v[k+3]+=fabs(z-r);
    }
    for(int k=0;k<8;++k)acc[k][tid]=v[k];__syncthreads();
    for(int d=64;d;d>>=1){if(tid<d)for(int k=0;k<8;++k)acc[k][tid]+=acc[k][tid+d];__syncthreads();}
    if(tid==0)for(int k=0;k<8;++k)atomicAdd(out+(size_t)(b/group_size)*8+k,acc[k][0]);
}
// One value per canonical slot. Off, the value row stays the scalar critic.
int g_sbs_zone_critic = 0;
// [train] valve_zone (zone_critic only): damper and reheat take the whitened zone advantage
// alone instead of 0.70 building + 0.30 zone. Plant and AHU heads keep the building advantage.
int g_sbs_valve_zone = 0;
// Training-only, state-conditioned control variate for the whole-building GAE.
// The readout and its optimizer are deliberately outside the policy parameter
// buffer: its regression cannot backpropagate into the actor or its clip norm.
int g_sbs_team = 0;
// [policy] no_cmd_echo (r97; 0 off, 1 zone valves, 2 every head): the actor's per-node local block stops
// reading the observation columns that are that head's OWN previous command instead of physical state.
// Level 1: zone local f=4,5 = supply_air_damper_percentage_command, reheat_valve_percentage_command.
// Level 2 adds AHU local f=4,5,7 = supply_air_temperature_setpoint, recirculation_percentage_command,
// supervisor_run_command, and plant local f=1,3,4 = supply_water_temperature_setpoint,
// differential_pressure_setpoint, run_status. Those columns were measured to be bit-exact copies of the
// previous executed action (zone reheat: max error 0.0 over a whole month), so under policy.beta_cn_tau the
// readout can use them to copy its own AR(1) exploration draw: in r94b's 40.11M checkpoint the zone reheat
// row puts +3.545 on the reheat column against -0.100 on (T - Hs) and the valve stops tracking the room.
// Sensors stay: temperatures, setpoints of other devices, zone pressures, air and water flows, boiler
// temperature, and the whole pooled trunk (whose building-level command aggregates are unchanged). The
// removed value stays reachable through the recurrent state, which already carries the last action.
int g_sbs_no_echo = 0;
// Shared nonlinear zone actor branch; the affine head remains as a skip path.
int g_sbs_zone_mlp = 0;
// One shared AHU actor residual, default off; inputs and outputs stay canonical.
int g_sbs_ahu_mlp = 0;
constexpr int SBS_AHU_MLP_HIDDEN = 32;
precision_t* g_sbs_ahu_mlp_grad = nullptr;
long g_sbs_ahu_mlp_off = 0;
int g_sbs_ahu_mlp_n1 = 0, g_sbs_ahu_mlp_n2 = 0;
constexpr int SBS_ZONE_MLP_HIDDEN = 32;
// r117: a local, shared value residual. The pooled slot values stay intact;
// this branch has no gradient into its policy/trunk input.
int g_sbs_zone_value_residual = 0;
constexpr int SBS_ZVR_HIDDEN = 32;
precision_t* g_sbs_zvr_grad = nullptr;
long g_sbs_zvr_off = 0;
int g_sbs_zvr_n1 = 0, g_sbs_zvr_n2 = 0;
// Diagnostic copy of the most recent raw gradients, taken before Muon clears
// the role-Adam range. This buffer never participates in the update.
precision_t* g_sbs_zone_mlp_grad = nullptr;
long g_sbs_zone_mlp_off = 0;
int g_sbs_zone_mlp_n1 = 0, g_sbs_zone_mlp_n2 = 0;
// r100: the zone reheat valves share the Bernoulli head the run commands use (canonical slots 2s+1).
// Declared here rather than in the pufferlib patch so the flag lives with the policy it belongs to.
int g_binary_reheat = 0;
// [train] value_adam (r101): put the value matrix on the same Adam range as the role readouts. Muon
// normalises a rank-1 gradient (which is what the value matrix has when the per-zone rows are dead) to a
// constant step, so the building value cannot follow the returns.
int g_sbs_value_adam = 0;
__device__ int d_sbs_no_echo = 0;
// [train] ahu_critic (zone_critic only): CN_K more value rows, one per AHU, trained on the env's per-AHU
// return (env.ahu_critic). The AHU heads take that AHU's whitened advantage. g_sbs_zw is the critic width.
int g_sbs_ahu_critic = 0;
int g_sbs_zw = CN_ZS;
// One value for every zone, read from that zone's own 11 features. The weights
// are a 1D vector, so Muon does not pin their scale. The gradient stops here:
// the features are not the trunk. 11 weights plus a bias.
int g_sbs_room_value = 0;
constexpr int SBS_ROOM = SBS_FEAT + 1;
// [train] zone_value (r98, zone_critic only): one value per present zone, read from that zone's OWN
// representation — role[0].input = [trunk h (H); its 68 local features], which sbs_role_input_k has
// already built and asinh-transformed. The pooled rows (base.weight 91 x 264) can only see the
// building, so V_z could not baseline a zone's own transition; this row can. One shared vector of
// H + SBS_LZ weights plus one bias per slot (324 + 90 = 414 parameters, not 90 x 264 = 23,760),
// which transfers to any building's zone layout. The gradient stops at the input, so the zone-value
// path does not turn the trunk (the building row still does, through base.weight).
int g_sbs_zone_value = 0;
// The slot bias is padded to a multiple of 4 floats: every tensor is padded to 16 bytes while Muon
// advances by the element count, so a width that is not a multiple of 4 shifts every later slot.
constexpr int SBS_ZVB = ((CN_ZS + 3) / 4) * 4;
// [train] value_trunk_cap: clamp the value path's contribution to the shared trunk gradient at
// the policy path's magnitude, measured once per minibatch visit. The value head's own gradient,
// the actor heads and the policy gradient are untouched. 0 leaves the trunk as
// (role gradients + grad_value_h), bit for bit. On the separate-critic path grad_value_h is zero
// by construction, so the cap cannot bite there.
int g_sbs_vtc = 0;
// Device telemetry, 4 floats: [0] visits where the cap bit, [1] visits, [2] tightest scale used
// this epoch (starts at 1.0, so 1.0 means it never bit).
float* g_sbs_vtc_stat = nullptr;
// Device scratch for the per-block partial sums of ||g_pi||^2 and ||g_V||^2.
float* g_sbs_vtc_part = nullptr;
// [train] value_spike_guard: clamp the trunk-side value tangent (grad_value_h) at q times its own
// trailing scale, and only when it surges: s = min(1, q*M/||g_V||), with M the trailing geometric
// mean of ||g_V|| (an EMA of ln||g_V|| whose halflife is one epoch of visits, so a short storm
// barely moves it and a real shift is followed). Applied per minibatch visit, to grad_value_h
// only: the policy path into the trunk, the critic's own row (wgrad_value), the value loss and the
// actor heads are untouched, so a chronic ratio below q passes through at s = 1 by construction.
int g_sbs_vsg = 0;
float g_sbs_vsg_q = 3.0f;       // the multiple of the trailing scale that is allowed through
float g_sbs_vsg_a = 0.005404f;  // EMA coefficient in ln space; set from the visits per epoch
// Device telemetry, 32 floats: [0] visits, [1] bites, [2] sum of s, [3] min s, [4] max ratio,
// [5] sum of ratio, [6] max ||g_V||, [7] sum of ln||g_V||, [8..31] log2(ratio/q) histogram.
float* g_sbs_vsg_stat = nullptr;
// Persistent EMA state, 3 floats: [0] lnM, [1] initialised flag, [2] the scale this visit applied.
float* g_sbs_vsg_ema = nullptr;
// Device scratch for the per-block sum of squares of ||g_V||.
float* g_sbs_vsg_part = nullptr;
// [train] lastlayer_stepsize (C14 / P28+P29): per-weight step sizes with step-size
// optimization on the actor's last layer — the three role readout matrices
// (SBSWeights::role[0..2]: zones, AHUs, plant). The method is IDBD, Sutton 1992
// ("Adapting Bias by Gradient Descent", Fig. 2), instantiated per weight on the
// summed minibatch gradient (delta*x_i -> -g_ij), with SwiftTD's robustness
// mechanisms from the official implementation (github.com/kjaved0/swifttd,
// SwiftTD.cpp): a hard step-size bound eta with a floor eta_min, and the decay
// rule that wipes the trace and shrinks the step size when a weight's effective
// step alpha*g^2 exceeds eta. SwiftTD's true-online traces (z, z_bar, v_old) are
// TD machinery and do not apply to a policy-gradient minibatch; the per-weight
// trace h and the normalized meta step (the kappa/alpha factor, whose dalpha is
// alpha-independent) are SwiftTD's, carried over as-is:
//   alpha = exp(beta);  beta initialised to log(alpha0)
//   beta  += kappa / alpha * (-g) * h
//   clamp alpha into [eta_min, eta]
//   w     -= alpha * g
//   h     = h * max(0, 1 - alpha*g^2) - alpha*g   (IDBD eq. 6: a trace of recent
//            weight changes; the decay term is IDBD's exact d w / d beta factor
//            with the gradient standing in for the LMS input)
//   if alpha*g^2 > eta: h = 0 and beta += log(decay) * g^2
// g is the minibatch gradient at the scale of the global L2 clip Muon applies (the
// same scale, computed here over the full gradient so both optimizers see one
// clip). The role weights are removed from Muon's step: their gradient is zeroed
// inside muon_step after the clip norm is computed, so the trunk's update is
// bit-identical to the Muon-for-roles control given the same weights and the only
// difference is who steps the role matrices. Off (default), nothing launches and
// muon_step is called with the freeze range 0. Constants: kappa 0.01 and alpha0
// 1e-3 (register P28); eta 0.1, eta_min 1e-10, decay 0.999 (the official SwiftTD
// defaults).
int g_sbs_idbd = 0;
// policy.occ_asinh: pool feature 69 (building occupancy) goes through asinh like the zone,
// plant and demand columns. 0 keeps it raw.
int g_sbs_occ_asinh = 0;
// train.head_adam: role[0] (the zone head) is stepped by Adam (beta 0.9/0.999, eps 1e-8, bias
// corrected, the run's lr) on the clipped gradient; muon_step gets role[0] as its freeze range.
int g_sbs_head_adam = 0;
// train.role_adam: the same Adam on all three role readouts, role[0..2] (zone, AHU, plant; under policy.beta
// both the u and the v rows), one contiguous range. Muon keeps every other matrix. Muon orthogonalises each
// matrix's update, so on these 4-12 row readouts every output direction takes a unit step whatever its signal;
// Adam scales each weight's step by its own gradient history. Uses the head_adam state below.
int g_sbs_role_adam = 0;
long g_sbs_hadam_off = 0; int g_sbs_hadam_len = 0;
float* g_sbs_hadam_m = nullptr; float* g_sbs_hadam_v = nullptr;
float* g_sbs_hadam_bc = nullptr; int* g_sbs_hadam_t = nullptr;
// Role/zone Adam is separate from Muon's .mb state. For r117 persist the
// complete range, including the critic residual, with strict layout checks.
static char g_sbs_hadam_restore_path[4096] = {};
struct SBSHadamHeader { char magic[8]; uint32_t version; int32_t len; int64_t off; int32_t step; uint32_t reserved; };
static void sbs_hadam_save(const char* path){
    if(!g_sbs_hadam_m || !g_sbs_hadam_v || !g_sbs_hadam_t || g_sbs_hadam_len<=0)abort();
    SBSHadamHeader h={};memcpy(h.magic,"SBHADAM1",8);h.version=1;h.len=g_sbs_hadam_len;h.off=g_sbs_hadam_off;
    assert(cudaMemcpy(&h.step,g_sbs_hadam_t,sizeof(h.step),cudaMemcpyDeviceToHost)==cudaSuccess);
    std::vector<float> state((size_t)2*h.len);
    assert(cudaMemcpy(state.data(),g_sbs_hadam_m,(size_t)h.len*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(state.data()+h.len,g_sbs_hadam_v,(size_t)h.len*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    for(float x:state)if(!isfinite(x)){fprintf(stderr,"pufferl: nonfinite role Adam state\n");abort();}
    char final[4200],part[4200];snprintf(final,sizeof(final),"%s.hadam",path);
    snprintf(part,sizeof(part),"%s.part.%d",final,(int)getpid());
    FILE* f=fopen(part,"wb");if(!f){perror("role Adam checkpoint");abort();}
    if(fwrite(&h,sizeof(h),1,f)!=1 || fwrite(state.data(),sizeof(float),state.size(),f)!=state.size() || fclose(f)!=0){
        fprintf(stderr,"pufferl: failed writing role Adam checkpoint\n");abort();
    }
    if(rename(part,final)!=0){perror("role Adam checkpoint rename");abort();}
}
static void sbs_hadam_restore(){
    const char* path=g_sbs_hadam_restore_path;if(!path[0])return;
    FILE* f=fopen(path,"rb");if(!f){fprintf(stderr,"pufferl: missing role Adam state %s\n",path);abort();}
    SBSHadamHeader h={};
    if(fread(&h,sizeof(h),1,f)!=1 || memcmp(h.magic,"SBHADAM1",8)!=0 || h.version!=1 ||
       h.len!=g_sbs_hadam_len || h.off!=g_sbs_hadam_off || h.step<0 || h.reserved!=0){
        fprintf(stderr,"pufferl: incompatible role Adam checkpoint %s\n",path);abort();
    }
    std::vector<float> state((size_t)2*h.len);
    if(fread(state.data(),sizeof(float),state.size(),f)!=state.size() || fgetc(f)!=EOF || fclose(f)!=0){
        fprintf(stderr,"pufferl: truncated or oversized role Adam checkpoint %s\n",path);abort();
    }
    for(size_t i=0;i<state.size();++i)if(!isfinite(state[i]) || (i>=(size_t)h.len && state[i]<0.f)){
        fprintf(stderr,"pufferl: invalid role Adam checkpoint %s\n",path);abort();
    }
    assert(cudaMemcpy(g_sbs_hadam_m,state.data(),(size_t)h.len*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(g_sbs_hadam_v,state.data()+h.len,(size_t)h.len*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(g_sbs_hadam_t,&h.step,sizeof(h.step),cudaMemcpyHostToDevice)==cudaSuccess);
    std::vector<float> check(state.size());int step=-1;
    assert(cudaMemcpy(check.data(),g_sbs_hadam_m,(size_t)h.len*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(check.data()+h.len,g_sbs_hadam_v,(size_t)h.len*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(&step,g_sbs_hadam_t,sizeof(step),cudaMemcpyDeviceToHost)==cudaSuccess);
    if(memcmp(check.data(),state.data(),state.size()*sizeof(float))!=0 || step!=h.step)abort();
    fprintf(stderr,"pufferl: restored role Adam state %s (%d weights, step %d)\n",path,h.len,h.step);
    g_sbs_hadam_restore_path[0]='\0';
}
// train.trunk_log: read-only per-visit ||g_pi|| and ||g_V|| at the trunk output h.
// stat: [0] sum ||g_pi||, [1] sum ||g_V||, [2] sum ln(||g_V||/||g_pi||), [3] visits,
// [4] max ||g_V||/||g_pi||, [5] max ||g_V||, [6] visits with both norms > 0.
int g_sbs_tl = 0;
float* g_sbs_tl_stat = nullptr; float* g_sbs_tl_part = nullptr;
constexpr float SBS_IDBD_KAPPA = 0.01f;
constexpr float SBS_IDBD_ALPHA0 = 1e-3f;
constexpr float SBS_IDBD_ETA = 0.1f;
constexpr float SBS_IDBD_ETA_MIN = 1e-10f;
constexpr float SBS_IDBD_DECAY = 0.999f;
// Persistent optimizer state, one entry per role weight. Not saved in checkpoints:
// the arm trains from scratch and is never resumed.
float* g_sbs_idbd_beta = nullptr;   // log of the step size
float* g_sbs_idbd_h = nullptr;      // trace of recent weight changes
int g_sbs_idbd_n = 0;               // total role weights
long g_sbs_idbd_off[3] = {0, 0, 0}; // param offsets of role[0..2]
int g_sbs_idbd_len[3] = {0, 0, 0};  // numels of role[0..2]
int g_sbs_idbd_fz0 = 0, g_sbs_idbd_fz1 = 0;  // the freeze range handed to muon_step
// Per-epoch telemetry, 8 floats: [0] sum alpha, [1] sum effective step alpha*g^2,
// [2] sum |h|, [3] weights at the eta cap, [4] weights at eta_min, [5] weights
// seen, [6] decay triggers, [7] sum beta. Plus the gradient-norm scratch.
float* g_sbs_idbd_stat = nullptr;
float* g_sbs_idbd_part = nullptr;   // block partials
float* g_sbs_idbd_norm = nullptr;   // sum of squares of the full minibatch gradient

// Sum of squares per block, ascending within the block. Same shape as the Muon
// norm partials so the reduce and the clip scale are computed the same way.
__global__ void sbs_idbd_sumsq_partials_k(const precision_t* __restrict__ g,
        float* __restrict__ out, int n) {
    __shared__ float s[256];
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(g[i]);
        acc += v * v;
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        for (int k = 0; k < 256; ++k) t += s[k];
        out[blockIdx.x] = t;
    }
}
__global__ void sbs_idbd_sumsq_reduce_k(const float* __restrict__ partials,
        float* __restrict__ out, int nb) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float t = 0.0f;
    for (int k = 0; k < nb; ++k) t += partials[k];
    *out = t;
}
// One IDBD visit for one role tensor. One thread per weight; no races (each thread
// touches only its own w/beta/h). Telemetry uses atomics and is order-independent.
__global__ void sbs_idbd_update_k(precision_t* __restrict__ grad, float* __restrict__ w,
        float* __restrict__ beta, float* __restrict__ h, const float* __restrict__ sumsq,
        float* __restrict__ stat, float max_norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // The clip Muon applies: max_norm / (sqrt(sum g^2) + 1e-6), capped at 1.
    float clip = fminf(max_norm / (sqrtf(*sumsq) + 1e-6f), 1.0f);
    float g = to_float(grad[i]) * clip;
    atomicAdd(&stat[5], 1.0f);
    if (!isfinite(g)) return;   // a non-finite visit is skipped, as in the loss
    float b = beta[i];
    float hv = h[i];
    b += SBS_IDBD_KAPPA / expf(b) * (-g) * hv;
    if (b > logf(SBS_IDBD_ETA)) b = logf(SBS_IDBD_ETA);            // SwiftTD bound
    if (!(b > logf(SBS_IDBD_ETA_MIN))) b = logf(SBS_IDBD_ETA_MIN); // floor; drops NaN
    float a = expf(b);
    float dw = -a * g;
    w[i] += dw;
    hv = hv * fmaxf(0.0f, 1.0f - a * g * g) + dw;   // IDBD eq. 6
    float eff = a * g * g;
    if (eff > SBS_IDBD_ETA) {                       // SwiftTD: too hot -> wipe and decay
        hv = 0.0f;
        b += logf(SBS_IDBD_DECAY) * g * g;
    }
    beta[i] = b;
    h[i] = hv;
    atomicAdd(&stat[0], a);
    atomicAdd(&stat[1], eff);
    atomicAdd(&stat[2], fabsf(hv));
    atomicAdd(&stat[3], (float)(a >= SBS_IDBD_ETA * 0.99999f));
    atomicAdd(&stat[4], (float)(a <= SBS_IDBD_ETA_MIN * 1.00001f));
    atomicAdd(&stat[6], (float)(eff > SBS_IDBD_ETA));
    atomicAdd(&stat[7], b);
}
float* g_sbs_zpred = nullptr;
float* g_sbs_zgrad = nullptr;
constexpr int SBS_VEXTRA = 8;
float* g_sbs_feat = nullptr;
float* g_sbs_feat_batch = nullptr;
const float* g_sbs_aux = nullptr;
const float* g_sbs_critic = nullptr;
precision_t* g_sbs_head_dst = nullptr;
float* g_sbs_head_grad = nullptr;
extern "C" void sb_set_feat(float* p) { g_sbs_feat = p; }

__device__ float sbs_zstat(const precision_t* o, int first, int count, int feature, int stat) {
    float value = 0.0f; int n = 0;
    for (int s = first; s < first + count; ++s) {
        if (to_float(o[CN_FZ*s]) == 0.0f) continue;
        float x = to_float(o[CN_FZ*s + feature]);
        if (n == 0) value = x;
        else if (stat == 1) value = fminf(value,x);
        else if (stat == 2) value = fmaxf(value,x);
        else value += x;
        ++n;
    }
    return stat == 0 && n ? value/n : value;
}
__device__ float sbs_demand(const precision_t* o, int feature, int kind) {
    float value=0; int n=0;
    for(int s=0;s<CN_ZS;++s) if(to_float(o[CN_FZ*s])!=0){
        float x=fmaxf(0.0f,-to_float(o[CN_FZ*s+feature]));
        if(kind==1)value=fmaxf(value,x);
        else if(kind==2)value+=x>0;
        else if(kind==3)value+=x*x;
        else value+=x;
        ++n;
    }
    return kind!=1 && n ? value/n : value;
}
__device__ float sbs_pool_feature(const precision_t* o, int f) {
    /* r99 level 3: the pooled command aggregates carry the same self-echo, building-wide:
     * f 12..17 = all-zone mean/min/max of the damper and reheat command fields,
     * f 30..35 = per-AHU supply-air-temperature setpoint and recirculation,
     * f 39..41 = per-AHU run command. */
    if(d_sbs_no_echo>=3){
        if(f>=12&&f<18)return 0.0f;
        if((f>=30&&f<36)||(f>=39&&f<42))return 0.0f;
    }
    if(f<18)return sbs_zstat(o,0,CN_ZS,1+f/3,f%3);
    if(f<45){
        int q=(f-18)/3,stat=(f-18)%3,n=0;float value=0;
        for(int k=0;k<CN_K;++k)if(to_float(o[CN_OA+CN_FA*k])!=0){
            float x=q<8?to_float(o[CN_OA+CN_FA*k+1+q]):sbs_zstat(o,CN_ZPA*k,CN_ZPA,1,0);
            if(n==0)value=x;
            else if(stat==1)value=fminf(value,x);
            else if(stat==2)value=fmaxf(value,x);
            else value+=x;
            ++n;
        }
        return stat==0 && n ? value/n : value;
    }
    if(f<51)return to_float(o[CN_OH+1+f-45]);
    if(f<70)return to_float(o[CN_OG+f-51]);
    if(f==70){float n=0;for(int s=0;s<CN_ZS;++s)n+=to_float(o[CN_FZ*s])!=0;return n/CN_ZS;}
    if(f==71){float n=0;for(int k=0;k<CN_K;++k)n+=to_float(o[CN_OA+CN_FA*k])!=0;return n/CN_K;}
    if(f==72)return sbs_demand(o,2,0);
    if(f==73)return sbs_demand(o,3,0);
    if(f==74)return sbs_demand(o,2,1);
    if(f==75)return sbs_demand(o,3,1);
    if(f==76)return sbs_demand(o,2,2);
    if(f==77)return sbs_demand(o,3,2);
    if(f==78)return sbs_demand(o,2,3);
    return sbs_demand(o,3,3);
}
__global__ void sbs_pool_k(const precision_t* raw, precision_t* pool, int B, int stride, int occ){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*SBS_POOL)return;
    int f=i%SBS_POOL;float x=sbs_pool_feature(raw+(size_t)(i/SBS_POOL)*stride,f);
    // Smooth compression preserves information in large cold-start deviations.
    pool[i]=from_float((f<51 || f>=72 || (occ && f==69))?asinhf(x):x);
}
// Critic input. The pool is the same fixed summary the encoder reads. The eight
// delay numbers are the value_delay summary. Neither is a learned policy layer.
__global__ void sbs_value_features_k(const precision_t* raw, const float* feat,
        precision_t* dst, int B, int extra, int occ){
    int din=SBS_POOL+extra,i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*din)return;
    int col=i%din,b=i/din;
    const precision_t* o=raw+(size_t)b*OBS_SIZE;
    if(col<SBS_POOL){
        float x=sbs_pool_feature(o,col);
        dst[i]=from_float((col<51 || col>=72 || (occ && col==69))?asinhf(x):x);
        return;
    }
    const float* base=feat?feat+(size_t)b*CN_ZS*SBS_FEAT:nullptr;
    float mx0=0,oat=0,mx4=0,mx6=0,mn7=0,mx9=0,mean9=0,mx10=0;int n=0;
    if(base){
        for(int z=0;z<CN_ZS;++z){
            if(to_float(o[CN_FZ*z])==0.0f)continue;
            const float* f=base+(size_t)z*SBS_FEAT;
            if(n==0){oat=f[3];mn7=f[7];}
            /* fabsf: the delay summary stays a magnitude summary when the deficits are signed (r99) */
            mx0=fmaxf(mx0,f[0]);mx4=fmaxf(mx4,fabsf(f[4]));mx6=fmaxf(mx6,fabsf(f[6]));
            mn7=fminf(mn7,f[7]);mx9=fmaxf(mx9,fabsf(f[9]));mx10=fmaxf(mx10,f[10]);
            mean9+=fabsf(f[9]);++n;
        }
    }
    float s[SBS_VEXTRA]={mx0,oat,mx4,mx6,mn7,mx9,n?mean9/(float)n:0.0f,mx10};
    dst[i]=from_float(asinhf(s[col-SBS_POOL]));
}
__global__ void sbs_tanh_k(precision_t* x, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    x[i]=from_float(tanhf(to_float(x[i])));
}
__global__ void sbs_tanh_bwd_k(precision_t* grad, const precision_t* y, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float h=to_float(y[i]);
    grad[i]=from_float(to_float(grad[i])*(1.0f-h*h));
}
__global__ void sbs_add_k(precision_t* dst,const precision_t* src,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=from_float(to_float(dst[i])+to_float(src[i]));
}
__device__ float sbs_pop_sigma(float mu, float nu){
    return sqrtf(fmaxf(nu-mu*mu, 1.0f));
}
// V = σ (w·h + b) + μ. van Hasselt et al. 2016, PopArt. σ and μ are the
// return scale. The normalized loss below divides by σ, so the last row is
// not asked to grow with the return.
__global__ void sbs_popart_fwd_k(precision_t* value, const precision_t* vb,
        const precision_t* pop, int B){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B)return;
    float mu=to_float(pop[0]),sig=sbs_pop_sigma(mu,to_float(pop[1]));
    float n=to_float(value[i])+to_float(vb[0]);
    value[i]=from_float(sig*n+mu);
}
__global__ void sbs_popart_bwd_k(precision_t* g, const float* src, const precision_t* pop, int B){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B)return;
    float inv=1.0f/sbs_pop_sigma(to_float(pop[0]),to_float(pop[1]));
    g[i]=from_float(src[i]*inv);
}
__global__ void sbs_pop_init_k(precision_t* pop){
    if(threadIdx.x==0&&blockIdx.x==0){
        pop[0]=from_float(0.0f);pop[1]=from_float(1.0f);
        pop[2]=from_float(0.0f);pop[3]=from_float(0.0f);
    }
}
__global__ void sbs_bias_grad_k(const precision_t* g, precision_t* dst, int n){
    float sum=0.0f;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)sum+=to_float(g[i]);
    __shared__ float buf[256];
    buf[threadIdx.x]=sum;
    __syncthreads();
    for(int t=blockDim.x/2;t>0;t>>=1){
        if(threadIdx.x<t)buf[threadIdx.x]+=buf[threadIdx.x+t];
        __syncthreads();
    }
    if(threadIdx.x==0){dst[0]=from_float(buf[0]);dst[1]=dst[2]=dst[3]=from_float(0.0f);}
}
// One update per epoch. pairs are the per-block (sum, sum of squares) of the
// GAE return in true units. Preserve: w' = w σ/σ', b' = (σ b + μ − μ')/σ',
// and the same factor on the Muon momentum, so V is unchanged.
__device__ int d_sbs_popart_tick=0;
__global__ void sbs_popart_commit_k(const float* pairs, int nblocks, int rows,
        float* master, float* mb, precision_t* param,
        long off_w3, int n_w3, long off_vb, long off_pop){
    if(threadIdx.x!=0||blockIdx.x!=0)return;
    double sum=0.0,sumsq=0.0;
    for(int b=0;b<nblocks;++b){sum+=pairs[2*b];sumsq+=pairs[2*b+1];}
    double N=(double)nblocks*(double)rows;
    float mean=(float)(sum/N),second=(float)(sumsq/N);
    float mu=master[off_pop],nu=master[off_pop+1];
    float sig=sbs_pop_sigma(mu,nu);
    float beta=d_sbs_popart_tick==0?1.0f:0.1f;
    float mu2=(1.0f-beta)*mu+beta*mean;
    float nu2=(1.0f-beta)*nu+beta*second;
    float sig2=sbs_pop_sigma(mu2,nu2);
    float alpha=sig/sig2;
    for(int i=0;i<n_w3;++i){master[off_w3+i]*=alpha;mb[off_w3+i]*=alpha;}
    float b=master[off_vb];
    master[off_vb]=(sig*b+mu-mu2)/sig2;
    mb[off_vb]*=alpha;
    master[off_pop]=mu2;master[off_pop+1]=nu2;
    mb[off_pop]=0.0f;mb[off_pop+1]=0.0f;
    for(int i=0;i<n_w3;++i)param[off_w3+i]=from_float(master[off_w3+i]);
    param[off_vb]=from_float(master[off_vb]);
    param[off_pop]=from_float(mu2);param[off_pop+1]=from_float(nu2);
    d_sbs_popart_tick+=1;
}
Prec sbs_encoder_forward(void* w,void* acts,Prec input,cudaStream_t stream){
    auto* ew=(EncoderWeights*)w;auto* a=(EncoderActivations*)acts;
    sbs_raw_input=input;
    sbs_pool_k<<<grid_size(input.shape[0]*SBS_POOL),BLOCK_SIZE,0,stream>>>(
        input.data,a->saved_input.data,input.shape[0],input.shape[1],g_sbs_occ_asinh);
    puf_mm(&a->saved_input,&ew->weight,&a->out,stream);return a->out;
}
void sbs_encoder_rollout(void* w,void* acts,Allocator* alloc,int B){
    encoder_reg_rollout(w,acts,alloc,B);auto* a=(EncoderActivations*)acts;
    a->saved_input={.shape={B,SBS_POOL}};alloc_register(alloc,&a->saved_input);
}
static void create_sbs_encoder(Encoder* e){
    assert(e->in_dim==OBS_SIZE);e->in_dim=SBS_POOL;
    e->forward=sbs_encoder_forward;e->reg_rollout=sbs_encoder_rollout;
}

__device__ float sbs_local_feature(const precision_t* o,int role,int node,int f,const float* zone_feat){
    if(role==0){
        /* r97: the zone's own valve commands are its previous action, not state. */
        if(f<6){if(d_sbs_no_echo>=1&&f>=4)return 0.0f;return to_float(o[CN_FZ*node+1+f]);}
        if(f<14)return to_float(o[CN_OA+CN_FA*(node/CN_ZPA)+1+f-6]);
        if(f<16)return fmaxf(0.0f,-to_float(o[CN_FZ*node+2+f-14]));
        if(f==16)return 1.0f;
        if(f<17+SBS_FEAT)return zone_feat ? zone_feat[f - 17] : 0.0f;
        int rel=f-(17+SBS_FEAT),j=rel/SBS_SIB,field=rel%SBS_SIB;
        if(j==node%CN_ZPA)return 0.0f;
        int s=CN_ZPA*(node/CN_ZPA)+j;
        if(to_float(o[CN_FZ*s])==0.0f)return 0.0f;
        if(field==3)return 1.0f;
        int col=field==0?1:field==1?5:6;
        return to_float(o[CN_FZ*s+col]);
    }
    if(role==1){
        /* r97 level 2: AHU SAT setpoint, recirculation and run commands are the previous action. */
        if(f<8){if(d_sbs_no_echo>=2&&(f==4||f==5||f==7))return 0.0f;return to_float(o[CN_OA+CN_FA*node+1+f]);}
        if(f<26)return sbs_zstat(o,CN_ZPA*node,CN_ZPA,1+(f-8)/3,(f-8)%3);
        if(f==26){float n=0;for(int j=0;j<CN_ZPA;++j)n+=to_float(o[CN_FZ*(CN_ZPA*node+j)])!=0;return n/CN_ZPA;}
        if(f<29){float v=0;for(int j=0;j<CN_ZPA;++j){int s=CN_ZPA*node+j;
            if(to_float(o[CN_FZ*s])!=0)v=fmaxf(v,-to_float(o[CN_FZ*s+2+f-27]));}return v;}
        if(f==29)return 1.0f;
        return to_float(o[CN_OH+1+f-30]);
    }
    if(f<6){/* r97 level 2: plant water setpoint, dp setpoint and run status are the previous action. */
        if(d_sbs_no_echo>=2&&(f==1||f==3||f==4))return 0.0f;
        return to_float(o[CN_OH+1+f]);}
    if(f<8)return sbs_demand(o,2+f-6,0);
    if(f<10)return sbs_demand(o,2+f-8,1);
    if(f<12)return sbs_demand(o,2+f-10,2);
    if(f==12)return to_float(o[CN_OG+18]);
    if(f<15)return to_float(o[CN_OG+16+f-13]);
    return 1.0f;
}
__global__ void sbs_role_input_k(const precision_t* raw,const precision_t* h,precision_t* dst,
        int B,int stride,int H,int role,int nodes,int local,const float* feat_base){
    int i=blockIdx.x*blockDim.x+threadIdx.x;int D=H+local;if(i>=B*nodes*D)return;
    int col=i%D,row=i/D,node=row%nodes,b=row/nodes;
    const precision_t* o=raw+(size_t)b*stride;
    bool exists=role==0?to_float(o[CN_FZ*node])!=0:
                role==1?to_float(o[CN_OA+CN_FA*node])!=0:to_float(o[CN_OH])!=0;
    if(!exists){dst[i]=from_float(0);return;}
    const float* zf = (role==0 && feat_base) ? feat_base + ((size_t)b*CN_ZS + node)*SBS_FEAT : nullptr;
    float x=col<H?to_float(h[(size_t)b*H+col]):sbs_local_feature(o,role,node,col-H,zf);
    dst[i]=from_float(col<H?x:asinhf(x));
}
// policy.beta: a role row holds 2*base outputs; the second base go to the second block at B*(width+1).
__global__ void sbs_scatter_k(const precision_t* src,precision_t* out,int B,int nodes,int outputs,int stride,int role,int width,int base){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*nodes*outputs)return;
    int c=i%outputs,node=(i/outputs)%nodes,b=i/(nodes*outputs);
    int block=c/base;c%=base;
    int action=role==0?2*node+c:role==1?CN_AA+4*node+c:CN_AH+c;
    /* stride >= outputs: role 0 carries the value row in its last column, which is not an action */
    out[(size_t)block*B*(width+1)+(size_t)b*(width+1)+action]=src[(size_t)(b*nodes+node)*stride+(i%outputs)];
}
__global__ void sbs_value_scatter_k(const precision_t* value,precision_t* out,int B,int width,int stride){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b<B)out[(size_t)b*(width+1)+width]=value[(size_t)b*stride];
}
__global__ void sbs_zpred_k(const precision_t* value,float* dst,int B,int stride){
    const int zw=stride-1;
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*zw)return;
    int s=i%zw,b=i/zw;
    dst[i]=to_float(value[(size_t)b*stride+1+s]);
}
__global__ void sbs_pack_gvalue_k(precision_t* gvalue,const float* building,const float* zone,int B,int rows){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;
    gvalue[(size_t)b*rows]=from_float(building?building[b]:0.0f);
    for(int s=0;s<rows-1;++s)
        gvalue[(size_t)b*rows+1+s]=from_float(zone?zone[(size_t)b*(rows-1)+s]:0.0f);
}
__global__ void sbs_role_grad_k(const float* grad,const float* grad2,const precision_t* raw,precision_t* dst,
        int B,int nodes,int outputs,int stride,int role,int width,int base,const float* zgrad,int nz){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*nodes*stride)return;
    int c=i%stride,row=i/stride,node=row%nodes,b=row/nodes;
    const precision_t* o=raw+(size_t)b*OBS_SIZE;
    bool exists=role==0?to_float(o[CN_FZ*node])!=0:
                role==1?to_float(o[CN_OA+CN_FA*node])!=0:to_float(o[CN_OH])!=0;
    float v=0.0f;
    if(exists){
        if(c<outputs){
            const float* g=c<base?grad:grad2;int cc=c%base;
            int action=role==0?2*node+cc:role==1?CN_AA+4*node+cc:CN_AH+cc;
            v=g[(size_t)b*width+action];
        }else if(c==outputs&&zgrad!=nullptr){
            v=zgrad[(size_t)b*nz+node];   /* the zone value row's gradient */
        }
    }
    dst[i]=from_float(v);
}
// Zero the value column of a role's gradient, so the zone-value path does not differentiate the input
// (it stops at the readout, exactly like the r53 room-value head).
__global__ void sbs_zv_zerocol_k(precision_t* g,int rows,int stride,int col){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows)return;
    g[(size_t)i*stride+col]=from_float(0.0f);
}
// The zone value is the value column of the role-0 readout: copy it into a->value rows 1..CN_ZS.
__global__ void sbs_zv_pick_k(const precision_t* rout,const precision_t* raw,const precision_t* zvb,
        precision_t* value,int B,int stride,int col,int rows){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*CN_ZS)return;
    int s=i%CN_ZS;if(s>=rows-1)return;
    int b=i/CN_ZS;
    if(to_float(raw[(size_t)b*OBS_SIZE+CN_FZ*s])==0.0f){value[(size_t)b*rows+1+s]=from_float(0);return;}
    value[(size_t)b*rows+1+s]=from_float(to_float(rout[(size_t)i*stride+col])+to_float(zvb[s]));
}
// value_trunk_cap phase 1: per-block partial sums of ||g_pi||^2 (the policy path into the trunk:
// the three role gradients) and ||g_V||^2 (the value path into the trunk: grad_value_h). The
// policy sum is accumulated exactly as sbs_context_grad_k composes it, so the norm matches the
// term that kernel adds.
__global__ void sbs_vtc_parts_k(const precision_t* z,const precision_t* a,const precision_t* hw,
        const precision_t* value,float* parts,int B,int H){
    __shared__ double ps[2][256];int t=threadIdx.x;
    double sp=0.0,sv=0.0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<B*H;i+=blockDim.x*gridDim.x){
        int b=i/H,c=i%H;
        float sum=0.0f;
        for(int s=0;s<CN_ZS;++s)sum+=to_float(z[((size_t)b*CN_ZS+s)*(H+SBS_LZ)+c]);
        for(int k=0;k<CN_K;++k)sum+=to_float(a[((size_t)b*CN_K+k)*(H+SBS_LA)+c]);
        sum+=to_float(hw[(size_t)b*(H+SBS_LH)+c]);
        float v=to_float(value[i]);
        sp+=(double)sum*(double)sum;sv+=(double)v*(double)v;
    }
    ps[0][t]=sp;ps[1][t]=sv;__syncthreads();
    for(int d=128;d;d>>=1){if(t<d){ps[0][t]+=ps[0][t+d];ps[1][t]+=ps[1][t+d];}__syncthreads();}
    if(t==0){parts[(size_t)blockIdx.x*2]=(float)ps[0][0];parts[(size_t)blockIdx.x*2+1]=(float)ps[1][0];}
}
// phase 2: one block reduces the partials and writes the scale to stat[2], plus this visit to
// stat[1] and (when the cap bit) to stat[0]. A scale of exactly 1 leaves the trunk unchanged.
__global__ void sbs_vtc_scale_k(const float* parts,int nb,float* stat){
    __shared__ double ps[2][256];int t=threadIdx.x;
    double sp=0.0,sv=0.0;
    for(int b=t;b<nb;b+=blockDim.x){sp+=(double)parts[(size_t)b*2];sv+=(double)parts[(size_t)b*2+1];}
    ps[0][t]=sp;ps[1][t]=sv;__syncthreads();
    for(int d=128;d;d>>=1){if(t<d){ps[0][t]+=ps[0][t+d];ps[1][t]+=ps[1][t+d];}__syncthreads();}
    if(t==0){
        float pi=sqrtf((float)ps[0][0]),gv=sqrtf((float)ps[1][0]);
        float s=(gv>pi&&gv>0.0f)?pi/gv:1.0f;
        // stat[2] keeps this epoch's tightest scale (1.0 = the cap never bit on any visit).
        stat[2]=fminf(stat[2],s);
        atomicAdd(stat+1,1.0f);
        if(s<1.0f)atomicAdd(stat,1.0f);
    }
}
__global__ void sbs_tl_acc_k(const float* parts,int nb,float* stat){
    __shared__ double ps[2][256];int t=threadIdx.x;
    double sp=0.0,sv=0.0;
    for(int b=t;b<nb;b+=blockDim.x){sp+=(double)parts[(size_t)b*2];sv+=(double)parts[(size_t)b*2+1];}
    ps[0][t]=sp;ps[1][t]=sv;__syncthreads();
    for(int d=128;d;d>>=1){if(t<d){ps[0][t]+=ps[0][t+d];ps[1][t]+=ps[1][t+d];}__syncthreads();}
    if(t==0){
        float pi=sqrtf((float)ps[0][0]),gv=sqrtf((float)ps[1][0]);
        stat[0]+=pi;stat[1]+=gv;stat[3]+=1.0f;stat[5]=fmaxf(stat[5],gv);
        if(pi>0.0f&&gv>0.0f){stat[2]+=logf(gv/pi);stat[4]=fmaxf(stat[4],gv/pi);stat[6]+=1.0f;}
    }
}
__global__ void sbs_head_adam_bump_k(int* t,float* bc){
    int s=*t+1;*t=s;bc[0]=(float)(1.0-pow(0.9,(double)s));bc[1]=(float)(1.0-pow(0.999,(double)s));
}
// One Adam step on the zone head. The clip is the one Muon applies to the whole gradient.
__global__ void sbs_head_adam_k(const precision_t* __restrict__ grad,float* __restrict__ w,
        float* __restrict__ m,float* __restrict__ v,const float* __restrict__ bc,
        const float* __restrict__ lr,const float* __restrict__ sumsq,float max_norm,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float clip=fminf(max_norm/(sqrtf(*sumsq)+1e-6f),1.0f);
    float g=to_float(grad[i])*clip;
    if(!isfinite(g))return;
    float mi=0.9f*m[i]+0.1f*g,vi=0.999f*v[i]+0.001f*g*g;
    m[i]=mi;v[i]=vi;
    w[i]-=(*lr)*(mi/bc[0])/(sqrtf(vi/bc[1])+1e-8f);
}
// SB_ROLESTEP=1 (diagnostic, needs train.update_diagnostics=1): per role readout (zone, AHU, plant), the sum over the
// epoch's minibatch visits of ||pre-clip gradient|| and ||applied step||, the visit count, and at the epoch print
// the net displacement ||W_now - W_epoch_start||. net / sum(step) is 1 when every step points the same way and about
// 1/sqrt(visits) for a random walk. Read-only. acc[3*r + {0: sum ||g||, 1: sum ||dW||, 2: visits}].
float* g_sbs_rl_acc = nullptr; float* g_sbs_rl_before = nullptr; float* g_sbs_rl_start = nullptr;
float* g_sbs_rl_net = nullptr; long g_sbs_rl_off = 0; int g_sbs_rl_len[3] = {0, 0, 0};
__device__ __forceinline__ int sbs_rl_role(int i,int l0,int l1){return i<l0?0:(i<l0+l1?1:2);}
__global__ void sbs_rl_pre_k(const float* __restrict__ w,const precision_t* __restrict__ g,float* before,
        float* acc,int l0,int l1,int l2){
    __shared__ double ps[3][256];int t=threadIdx.x;double s[3]={0,0,0};
    for(int i=t;i<l0+l1+l2;i+=blockDim.x){before[i]=w[i];double x=to_float(g[i]);s[sbs_rl_role(i,l0,l1)]+=x*x;}
    for(int k=0;k<3;++k)ps[k][t]=s[k];__syncthreads();
    for(int d=128;d;d>>=1){if(t<d)for(int k=0;k<3;++k)ps[k][t]+=ps[k][t+d];__syncthreads();}
    if(t==0)for(int k=0;k<3;++k)acc[3*k]+=(float)sqrt(ps[k][0]);
}
__global__ void sbs_rl_post_k(const float* __restrict__ w,const float* __restrict__ ref,float* acc,
        int l0,int l1,int l2,int visit){
    __shared__ double ps[3][256];int t=threadIdx.x;double s[3]={0,0,0};
    for(int i=t;i<l0+l1+l2;i+=blockDim.x){double x=(double)w[i]-ref[i];s[sbs_rl_role(i,l0,l1)]+=x*x;}
    for(int k=0;k<3;++k)ps[k][t]=s[k];__syncthreads();
    for(int d=128;d;d>>=1){if(t<d)for(int k=0;k<3;++k)ps[k][t]+=ps[k][t+d];__syncthreads();}
    if(t==0)for(int k=0;k<3;++k){
        if(visit){acc[3*k+1]+=(float)sqrt(ps[k][0]);acc[3*k+2]+=1.0f;}else acc[k]=(float)sqrt(ps[k][0]);
    }
}
__global__ void sbs_rl_copy_k(const float* __restrict__ w,float* dst,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)dst[i]=w[i];
}
// phase 3: scale the value path's trunk gradient in place. stat[2]==1.0f is an exact identity.
__global__ void sbs_vtc_apply_k(precision_t* value,const float* stat,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    value[i]=from_float(to_float(value[i])*stat[2]);
}
// value_spike_guard phase 1: per-block sum of squares of the value tangent (grad_value_h). This is
// a pass over B*H entries, not over the role gradients, so the guard is nearly free.
__global__ void sbs_vsg_parts_k(const precision_t* value,float* parts,int n){
    __shared__ double ps[1][256];int t=threadIdx.x;double s=0.0;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=blockDim.x*gridDim.x){
        double x=to_float(value[i]);s+=x*x;
    }
    ps[0][t]=s;__syncthreads();
    for(int d=128;d;d>>=1){if(t<d)ps[0][t]+=ps[0][t+d];__syncthreads();}
    if(t==0)parts[blockIdx.x]=(float)ps[0][0];
}
// phase 2: one block. Reduce to ||g_V||, advance the trailing log-scale EMA, compute the applied
// scale, and record this visit. The EMA is state, not a per-epoch statistic: it spans epochs, so
// the threshold is the trailing baseline and the earlier running-min defect cannot recur.
__global__ void sbs_vsg_scale_k(const float* parts,int nb,float* stat,float* ema,float q,float a){
    __shared__ double ps[1][256];int t=threadIdx.x;double s=0.0;
    for(int b=t;b<nb;b+=blockDim.x)s+=(double)parts[b];
    ps[0][t]=s;__syncthreads();
    for(int d=128;d;d>>=1){if(t<d)ps[0][t]+=ps[0][t+d];__syncthreads();}
    if(t==0){
        float gv=sqrtf((float)ps[0][0]);
        float lnv=logf(gv>1.0e-30f?gv:1.0e-30f);
        float sc=1.0f,ratio=0.0f;
        if(ema[1]==0.0f){
            ema[0]=lnv;ema[1]=1.0f;         // first visit: no trailing scale yet, s = 1
        }else{
            float M=expf(ema[0]);           // the trailing geometric mean of ||g_V||
            ratio=gv/M;
            float want=q*M;
            if(gv>1.0e-30f&&want<gv)sc=want/gv;
            ema[0]=(1.0f-a)*ema[0]+a*lnv;   // advance after the decision: this visit used the
        }                                   // baseline it did not help move
        ema[2]=sc;
        atomicAdd(stat+0,1.0f);
        if(sc<1.0f)atomicAdd(stat+1,1.0f);
        atomicAdd(stat+2,sc);
        if(sc<stat[3])stat[3]=sc;
        if(ratio>stat[4])stat[4]=ratio;
        atomicAdd(stat+5,ratio);
        if(gv>stat[6])stat[6]=gv;
        atomicAdd(stat+7,lnv);
        // Histogram of log2(ratio/q): bin 8 is the guard threshold itself, so the buckets read as
        // "below q", "q to 2q", and so on. This is the shape of the ||g_V|| surges, not just their
        // worst value.
        float rq=ratio/(q>0.0f?q:1.0f);
        int bin=(int)floorf(log2f(rq>1.0e-30f?rq:1.0e-30f))+8;
        if(bin<0)bin=0;if(bin>23)bin=23;
        atomicAdd(stat+8+bin,1.0f);
    }
}
// phase 3: apply the scale to grad_value_h only. ema[2]==1.0f is an exact identity.
__global__ void sbs_vsg_apply_k(precision_t* value,const float* ema,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    value[i]=from_float(to_float(value[i])*ema[2]);
}
__global__ void sbs_context_grad_k(const precision_t* z,const precision_t* a,const precision_t* hw,
        const precision_t* value,precision_t* out,int B,int H){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*H)return;int b=i/H,c=i%H;
    float sum=to_float(value[i]);
    for(int s=0;s<CN_ZS;++s)sum+=to_float(z[((size_t)b*CN_ZS+s)*(H+SBS_LZ)+c]);
    for(int k=0;k<CN_K;++k)sum+=to_float(a[((size_t)b*CN_K+k)*(H+SBS_LA)+c]);
    sum+=to_float(hw[(size_t)b*(H+SBS_LH)+c]);out[i]=from_float(sum);
}
__global__ void sbs_pack_value_k(const precision_t* h,const float* critic,precision_t* dst,int B,int H,int extra){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*(H+extra))return;
    int col=i%(H+extra),b=i/(H+extra);
    float x=col<H?to_float(h[(size_t)b*H+col]):(critic?critic[(size_t)b*extra+col-H]:0.0f);
    dst[i]=from_float(x);
}
__global__ void sbs_combine_k(const precision_t* heads,const float* aux,precision_t* value,int B){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;
    float vp=to_float(heads[b*4]),ve=to_float(heads[b*4+1]),vc=to_float(heads[b*4+2]);
    (void)aux;
    /* Competition value: 0.6 productivity - 0.2 energy - 0.2 carbon. The comfort potential and the
     * comfort-start head stay out of this number; that head still has its own loss. */
    value[b]=from_float(0.6f*vp-0.2f*ve-0.2f*vc);
}
__global__ void sbs_head_grad_k(const float* scalar_g,const float* extra,precision_t* dst,int B){
    int b=blockIdx.x*blockDim.x+threadIdx.x;if(b>=B)return;
    float g=scalar_g?scalar_g[b]:0.0f;
    float e0=extra?extra[b*4]:0.0f,e1=extra?extra[b*4+1]:0.0f,e2=extra?extra[b*4+2]:0.0f,e3=extra?extra[b*4+3]:0.0f;
    dst[b*4]=from_float(0.6f*g+e0);dst[b*4+1]=from_float(-0.2f*g+e1);
    dst[b*4+2]=from_float(-0.2f*g+e2);dst[b*4+3]=from_float(e3);
}
__global__ void sbs_take_h_k(const precision_t* wide,precision_t* h,int B,int H,int din){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*H)return;
    int b=i/H,c=i%H;h[i]=wide[(size_t)b*din+c];
}
/* Building-level delay, from the same per-zone features the actor reads.
 * minutes to comfort, outside-air change, worst current deficit, worst 4-hour
 * deficit, coldest slope, worst arrival deficit, mean arrival deficit, worst
 * idle-damper path. asinh keeps a 10 from dominating the hidden state. */
__global__ void sbs_pack_delay_k(const precision_t* h,const precision_t* raw,const float* feat,
        precision_t* dst,int B,int H){
    int i=blockIdx.x*blockDim.x+threadIdx.x;int din=H+SBS_VEXTRA;if(i>=B*din)return;
    int col=i%din,b=i/din;
    if(col<H){dst[i]=h[(size_t)b*H+col];return;}
    const precision_t* o=raw+(size_t)b*OBS_SIZE;
    const float* base=feat?feat+(size_t)b*CN_ZS*SBS_FEAT:nullptr;
    float mx0=0,oat=0,mx4=0,mx6=0,mn7=0,mx9=0,mean9=0,mx10=0;int n=0;
    if(base){
        for(int z=0;z<CN_ZS;++z){
            if(to_float(o[CN_FZ*z])==0.0f)continue;
            const float* f=base+(size_t)z*SBS_FEAT;
            if(n==0){oat=f[3];mn7=f[7];}
            /* fabsf: the delay summary stays a magnitude summary when the deficits are signed (r99) */
            mx0=fmaxf(mx0,f[0]);mx4=fmaxf(mx4,fabsf(f[4]));mx6=fmaxf(mx6,fabsf(f[6]));
            mn7=fminf(mn7,f[7]);mx9=fmaxf(mx9,fabsf(f[9]));mx10=fmaxf(mx10,f[10]);
            mean9+=fabsf(f[9]);++n;
        }
    }
    float s[SBS_VEXTRA]={mx0,oat,mx4,mx6,mn7,mx9,n?mean9/(float)n:0.0f,mx10};
    dst[i]=from_float(asinhf(s[col-H]));
}
static int sbs_value_din(int H){
    if(g_sbs_components)return H+CN_ZS*2;
    if(g_sbs_value_extra)return H+SBS_VEXTRA;
    return H;
}
static int sbs_value_rows(){
    if(g_sbs_components)return 4;
    if(g_sbs_zone_critic)return 1+g_sbs_zw;
    return 1;
}
struct SBSWeights{DecoderWeights base;Prec role[3];Prec zone_mlp1,zone_mlp2;Prec ahu_mlp1,ahu_mlp2;Prec zvr1,zvr2;Prec room;Prec zv,zvb;Prec vw1,vw2,vw3,vb,pop;};
// Rows of role r's readout: one per command, two per command under policy.beta (u rows, then v rows).
static int sbs_role_out(int r){return SBS_OUTPUTS[r]*(g_beta?2:1);}
// Rows of role r's MATRIX. With r98's per-zone value the last row of role 0 is the value row, so the
// zone value is produced by the actor's own matmul (no second pass over the 324-dim zone input) and its
// gradient comes out of the same matmul in the backward. sbs_role_out stays the number of ACTION rows.
static int sbs_role_rows(int r){return sbs_role_out(r)+((r==0&&g_sbs_zone_value)?1:0);}
// A resumed checkpoint already holds PopArt mean and second moment. Leaving the
// tick at 0 makes the next commit use beta 1, which replaces those statistics
// and rescales the value row and its momentum.
void sbs_popart_mark_resumed() {
    int one = 1;
    cudaMemcpyToSymbol(d_sbs_popart_tick, &one, sizeof(one));
}
void sbs_popart_commit(void* decoder, float* master, float* momentum,
        precision_t* param, const float* pairs, int nblocks, int rows, cudaStream_t stream){
    if(g_sbs_components||decoder==nullptr||param==nullptr||master==nullptr)return;
    auto* w=(SBSWeights*)decoder;
    if(w->pop.data==nullptr||w->vw3.data==nullptr||w->vb.data==nullptr)return;
    long off_w3=w->vw3.data-param,off_vb=w->vb.data-param,off_pop=w->pop.data-param;
    sbs_popart_commit_k<<<1,1,0,stream>>>(pairs,nblocks,rows,master,momentum,param,
        off_w3,(int)numel(w->vw3.shape),off_vb,off_pop);
}
// Index of the PopArt statistics in the flat parameter list. They are state,
// updated once per epoch, and their backprop gradient is zero on purpose.
int g_sbs_popart_reg=-1;
struct SBSRoleActs{Prec input,out,gout,gin,wgrad;};
struct SBSActs{Prec out,saved_h,raw,grad_h,value,gvalue,grad_value_h,wgrad_value,grad_std,room_g,zv_g,zvb_g,
    value_in,heads,gheads,value_gin,vin,vh1,vh2,vdh1,vdh2,vgw1,vgw2,vgw3,vgb,vgp,
    zone_hidden,zone_branch,zone_dhidden,zone_dinput,zone_gw1,zone_gw2,
    ahu_hidden,ahu_branch,ahu_dhidden,ahu_dinput,ahu_gw1,ahu_gw2,
    zvr_hidden,zvr_out,zvr_gout,zvr_dhidden,zvr_gw1,zvr_gw2;SBSRoleActs role[3];};
static int sbs_value_mlp_in(){return SBS_POOL+(g_sbs_value_extra?SBS_VEXTRA:0);}
void* sbs_decoder_create(void* self){
    static int announced=0;
    if(!announced){
        fprintf(stderr,"pufferl: same-handler sibling block: temp, damper, reheat, presence\n");
        announced=1;
    }
    auto* d=(Decoder*)self;auto* w=(SBSWeights*)calloc(1,sizeof(SBSWeights));
    w->base.hidden_dim=d->hidden_dim;w->base.output_dim=d->output_dim;w->base.continuous=d->continuous;return w;
}
void sbs_decoder_params(void* ptr,Allocator* alloc){
    auto* w=(SBSWeights*)ptr;int H=w->base.hidden_dim;
    int nhead=sbs_value_rows(),din=sbs_value_din(H);
    g_sbs_popart_reg=-1;
    w->base.weight={.shape={nhead,din}};alloc_register(alloc,&w->base.weight);
    w->base.logstd={.shape={1,w->base.output_dim}};alloc_register(alloc,&w->base.logstd);
    for(int r=0;r<3;++r){w->role[r]={.shape={sbs_role_rows(r),H+SBS_LOCAL[r]}};alloc_register(alloc,&w->role[r]);}
    if(g_sbs_zone_mlp){
        w->zone_mlp1={.shape={SBS_ZONE_MLP_HIDDEN,H+SBS_LZ}};
        w->zone_mlp2={.shape={sbs_role_out(0),SBS_ZONE_MLP_HIDDEN}};
        alloc_register(alloc,&w->zone_mlp1);alloc_register(alloc,&w->zone_mlp2);
    }
    if(g_sbs_ahu_mlp){
        w->ahu_mlp1={.shape={SBS_AHU_MLP_HIDDEN,H+SBS_LA}};
        w->ahu_mlp2={.shape={sbs_role_out(1),SBS_AHU_MLP_HIDDEN}};
        alloc_register(alloc,&w->ahu_mlp1);alloc_register(alloc,&w->ahu_mlp2);
    }
    if(g_sbs_zone_value_residual){
        w->zvr1={.shape={SBS_ZVR_HIDDEN,H+SBS_LZ}};
        w->zvr2={.shape={1,SBS_ZVR_HIDDEN}};
        alloc_register(alloc,&w->zvr1);alloc_register(alloc,&w->zvr2);
    }
    if(g_sbs_room_value){w->room={.shape={SBS_ROOM}};alloc_register(alloc,&w->room);}
    if(g_sbs_zone_value){w->zvb={.shape={SBS_ZVB}};alloc_register(alloc,&w->zvb);}
    if(!g_sbs_components && !g_sbs_value_shared){
        int in=sbs_value_mlp_in();
        w->vw1={.shape={H,in}};w->vw2={.shape={H,H}};w->vw3={.shape={1,H}};
        // Rank-1, four wide. Muon orthogonalizes every matrix, which pinned the
        // old 1×256 output row to one learning-rate step. A width of 1 would
        // also break the packed walk: each tensor is padded to 16 bytes while
        // Muon advances by the element count. The bias is element 0. The
        // statistics are μ, ν in elements 0 and 1. The rest stay zero.
        w->vb={.shape={4}};w->pop={.shape={4}};
        alloc_register(alloc,&w->vw1);alloc_register(alloc,&w->vw2);alloc_register(alloc,&w->vw3);
        alloc_register(alloc,&w->vb);
        g_sbs_popart_reg=alloc->num_regs;
        alloc_register(alloc,&w->pop);
    }
}
void sbs_decoder_init(void* ptr,ulong* seed,cudaStream_t stream){
    auto* w=(SBSWeights*)ptr;int H=w->base.hidden_dim;
    puf_kaiming_init(&w->base.weight,1.0f,(*seed)++,stream);
    for(int r=0;r<3;++r)puf_kaiming_init(&w->role[r],0.1f,(*seed)++,stream);
    if(g_sbs_zone_mlp){
        // Do not advance the seed for the existing recurrent network: seed 75
        // must initialize the unchanged layers exactly as in the control.
        const ulong zone_seed=(*seed)^0x9e3779b97f4a7c15ULL;
        puf_kaiming_init(&w->zone_mlp1,1.0f,zone_seed,stream);
        fill_prec_kernel<<<grid_size(numel(w->zone_mlp2.shape)),BLOCK_SIZE,0,stream>>>(
            w->zone_mlp2.data,0.0f,numel(w->zone_mlp2.shape));
    }
    if(g_sbs_ahu_mlp){
        // Independent stream: do not consume the existing policy's seed.
        const ulong seed_ahu=(*seed)^0xa0761d6478bd642fULL;
        puf_kaiming_init(&w->ahu_mlp1,1.0f,seed_ahu,stream);
        fill_prec_kernel<<<grid_size(numel(w->ahu_mlp2.shape)),BLOCK_SIZE,0,stream>>>(
            w->ahu_mlp2.data,0.0f,numel(w->ahu_mlp2.shape));
    }
    if(g_sbs_zone_value_residual){
        // Do not consume the control network's initialization stream.
        const ulong seed_zvr=(*seed)^0xd1b54a32d192ed03ULL;
        puf_kaiming_init(&w->zvr1,1.0f,seed_zvr,stream);
        fill_prec_kernel<<<grid_size(numel(w->zvr2.shape)),BLOCK_SIZE,0,stream>>>(
            w->zvr2.data,0.0f,numel(w->zvr2.shape));
    }
    if(g_sbs_room_value)fill_prec_kernel<<<1,SBS_ROOM,0,stream>>>(w->room.data,0.0f,SBS_ROOM);
    if(g_sbs_zone_value){   /* zero start: V_z = 0, the zone advantage is r_z unmodified */
        fill_prec_kernel<<<grid_size(SBS_ZVB),BLOCK_SIZE,0,stream>>>(w->zvb.data,0.0f,SBS_ZVB);
        /* the value row of role 0 also starts at 0 (the matrix was kaiming-initialised) */
        float* r0=(float*)w->role[0].data+(size_t)(sbs_role_out(0))*(H+SBS_LZ);
        fill_prec_kernel<<<grid_size(H+SBS_LZ),BLOCK_SIZE,0,stream>>>(r0,0.0f,H+SBS_LZ);
    }
    if(!g_sbs_components && !g_sbs_value_shared){
        puf_kaiming_init(&w->vw1,1.0f,(*seed)++,stream);
        puf_kaiming_init(&w->vw2,1.0f,(*seed)++,stream);
        puf_kaiming_init(&w->vw3,1.0f,(*seed)++,stream);
        fill_prec_kernel<<<1,4,0,stream>>>(w->vb.data,0.0f,4);
        sbs_pop_init_k<<<1,1,0,stream>>>(w->pop.data);
    }
    fill_prec_kernel<<<grid_size(w->base.output_dim),BLOCK_SIZE,0,stream>>>(w->base.logstd.data,g_init_logstd,w->base.output_dim);
}
void sbs_register_acts(void* ptr,void* act,Allocator* acts,Allocator* grads,int B){
    auto* w=(SBSWeights*)ptr;auto* a=(SBSActs*)act;int H=w->base.hidden_dim,W=w->base.output_dim;
    int nrows=sbs_value_rows();
    a->out={.shape={B*(g_beta?2:1),W+1}};a->value={.shape={B,nrows}};
    alloc_register(acts,&a->out);alloc_register(acts,&a->value);
    if(g_sbs_components){
        int din=sbs_value_din(H);
        a->value_in={.shape={B,din}};a->heads={.shape={B,4}};
        alloc_register(acts,&a->value_in);alloc_register(acts,&a->heads);
    }else if(g_sbs_value_extra){
        a->value_in={.shape={B,sbs_value_din(H)}};
        alloc_register(acts,&a->value_in);
    }
    if(grads){
        a->saved_h={.shape={B,H}};a->grad_h={.shape={B,H}};
        a->gvalue={.shape={B,nrows}};a->grad_value_h={.shape={B,H}};
        for(Prec* x:{&a->saved_h,&a->grad_h,&a->gvalue,&a->grad_value_h})alloc_register(acts,x);
        int nhead=nrows,din=sbs_value_din(H);
        a->wgrad_value={.shape={nhead,din}};alloc_register(grads,&a->wgrad_value);
        a->grad_std={.shape={1,W}};alloc_register(grads,&a->grad_std);
        if(g_sbs_components || g_sbs_value_extra){
            a->value_gin={.shape={B,din}};
            alloc_register(acts,&a->value_gin);
        }
        if(g_sbs_components){
            a->gheads={.shape={B,4}};
            alloc_register(acts,&a->gheads);
        }
    }
    for(int r=0;r<3;++r){auto* v=&a->role[r];int N=SBS_NODES[r],F=H+SBS_LOCAL[r],O=sbs_role_rows(r);
        v->input={.shape={B*N,F}};v->out={.shape={B*N,O}};
        alloc_register(acts,&v->input);alloc_register(acts,&v->out);
        if(grads){v->gout={.shape={B*N,O}};v->gin={.shape={B*N,F}};v->wgrad={.shape={O,F}};
            alloc_register(acts,&v->gout);alloc_register(acts,&v->gin);alloc_register(grads,&v->wgrad);}
    }
    if(g_sbs_zone_mlp){
        int rows=B*CN_ZS,F=H+SBS_LZ,O=sbs_role_out(0);
        a->zone_hidden={.shape={rows,SBS_ZONE_MLP_HIDDEN}};
        a->zone_branch={.shape={rows,O}};
        alloc_register(acts,&a->zone_hidden);alloc_register(acts,&a->zone_branch);
        if(grads){
            a->zone_dhidden={.shape={rows,SBS_ZONE_MLP_HIDDEN}};
            a->zone_dinput={.shape={rows,F}};
            a->zone_gw1={.shape={SBS_ZONE_MLP_HIDDEN,F}};
            a->zone_gw2={.shape={O,SBS_ZONE_MLP_HIDDEN}};
            alloc_register(acts,&a->zone_dhidden);alloc_register(acts,&a->zone_dinput);
            alloc_register(grads,&a->zone_gw1);alloc_register(grads,&a->zone_gw2);
        }
    }
    if(g_sbs_ahu_mlp){
        int rows=B*CN_K,F=H+SBS_LA,O=sbs_role_out(1);
        a->ahu_hidden={.shape={rows,SBS_AHU_MLP_HIDDEN}};
        a->ahu_branch={.shape={rows,O}};
        alloc_register(acts,&a->ahu_hidden);alloc_register(acts,&a->ahu_branch);
        if(grads){
            a->ahu_dhidden={.shape={rows,SBS_AHU_MLP_HIDDEN}};
            a->ahu_dinput={.shape={rows,F}};
            a->ahu_gw1={.shape={SBS_AHU_MLP_HIDDEN,F}};
            a->ahu_gw2={.shape={O,SBS_AHU_MLP_HIDDEN}};
            alloc_register(acts,&a->ahu_dhidden);alloc_register(acts,&a->ahu_dinput);
            alloc_register(grads,&a->ahu_gw1);alloc_register(grads,&a->ahu_gw2);
        }
    }
    if(g_sbs_zone_value_residual){
        int rows=B*CN_ZS,F=H+SBS_LZ;
        a->zvr_hidden={.shape={rows,SBS_ZVR_HIDDEN}};
        a->zvr_out={.shape={rows,1}};
        alloc_register(acts,&a->zvr_hidden);alloc_register(acts,&a->zvr_out);
        if(grads){
            a->zvr_gout={.shape={rows,1}};
            a->zvr_dhidden={.shape={rows,SBS_ZVR_HIDDEN}};
            a->zvr_gw1={.shape={SBS_ZVR_HIDDEN,F}};
            a->zvr_gw2={.shape={1,SBS_ZVR_HIDDEN}};
            alloc_register(acts,&a->zvr_gout);alloc_register(acts,&a->zvr_dhidden);
            alloc_register(grads,&a->zvr_gw1);alloc_register(grads,&a->zvr_gw2);
        }
    }
    if(g_sbs_room_value && grads){a->room_g={.shape={SBS_ROOM}};alloc_register(grads,&a->room_g);}
    if(g_sbs_zone_value && grads){a->zvb_g={.shape={SBS_ZVB}};alloc_register(grads,&a->zvb_g);}
    if(!g_sbs_components && !g_sbs_value_shared){
        int in=sbs_value_mlp_in();
        a->vin={.shape={B,in}};a->vh1={.shape={B,H}};a->vh2={.shape={B,H}};
        alloc_register(acts,&a->vin);alloc_register(acts,&a->vh1);alloc_register(acts,&a->vh2);
        if(grads){
            a->vdh1={.shape={B,H}};a->vdh2={.shape={B,H}};
            a->vgw1={.shape={H,in}};a->vgw2={.shape={H,H}};a->vgw3={.shape={1,H}};
            a->vgb={.shape={4}};a->vgp={.shape={4}};
            alloc_register(acts,&a->vdh1);alloc_register(acts,&a->vdh2);
            alloc_register(grads,&a->vgw1);alloc_register(grads,&a->vgw2);alloc_register(grads,&a->vgw3);
            alloc_register(grads,&a->vgb);alloc_register(grads,&a->vgp);
        }
    }
}
void sbs_decoder_train(void* w,void* a,Allocator* acts,Allocator* grads,int B){sbs_register_acts(w,a,acts,grads,B);}
void sbs_decoder_rollout(void* w,void* a,Allocator* acts,int B){sbs_register_acts(w,a,acts,nullptr,B);}
// V_z = bias + w · asinh(features_z). An absent slot is 0, so it cannot pull the bias.
__global__ void sbs_room_fwd_k(const precision_t* raw, const float* feat, const precision_t* w,
        float* dst, int B){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*CN_ZS)return;
    int s=i%CN_ZS,b=i/CN_ZS;
    const precision_t* o=raw+(size_t)b*OBS_SIZE;
    if(feat==nullptr || to_float(o[CN_FZ*s])==0.0f){dst[i]=0.0f;return;}
    const float* f=feat+((size_t)b*CN_ZS+s)*SBS_FEAT;
    float y=to_float(w[SBS_FEAT]);
    for(int k=0;k<SBS_FEAT;++k)y+=to_float(w[k])*asinhf(f[k]);
    dst[i]=y;
}
// Mean over present slots, same clip as the scalar value loss. Absent slots stay out.
__global__ void sbs_room_vloss_k(const float* pred, const float* oldv, const float* ret,
        const precision_t* obs, float* err, int n, int obs_dim, float clip, float coef){
    if(threadIdx.x!=0||blockIdx.x!=0)return;
    int present=0;
    for(int i=0;i<n*CN_ZS;++i){
        int s=i%CN_ZS,b=i/CN_ZS;
        if(to_float(obs[(size_t)b*obs_dim+CN_FZ*s])==0.0f){err[i]=0.0f;continue;}
        float vp=pred[i],ov=oldv[i],rt=ret[i];
        if(!isfinite(vp)||!isfinite(ov)||!isfinite(rt)){err[i]=0.0f;continue;}
        float vc=ov+fminf(clip,fmaxf(-clip,vp-ov));
        float d=( (vc-rt)*(vc-rt) > (vp-rt)*(vp-rt) ) ? 0.0f : (vp-rt);
        if(!isfinite(d))d=0.0f;
        err[i]=d;++present;
    }
    float scale=present?coef/(float)present:0.0f;
    for(int i=0;i<n*CN_ZS;++i)err[i]*=scale;
}
void sbs_room_vloss(const float* pred, const float* oldv, const float* ret,
        const precision_t* obs, float* err, int n, int obs_dim, float clip, float coef,
        cudaStream_t stream){
    sbs_room_vloss_k<<<1,1,0,stream>>>(pred,oldv,ret,obs,err,n,obs_dim,clip,coef);
}
// dL/dw = sum_z (dL/dV_z) asinh(f_z). The trunk is not an input.
__global__ void sbs_room_grad_k(const precision_t* raw, const float* feat, const float* err,
        precision_t* wgrad, int B){
    if(threadIdx.x!=0||blockIdx.x!=0)return;
    float g[SBS_ROOM];
    for(int k=0;k<SBS_ROOM;++k)g[k]=0.0f;
    if(feat!=nullptr && err!=nullptr){
        for(int i=0;i<B*CN_ZS;++i){
            int s=i%CN_ZS,b=i/CN_ZS;
            if(to_float(raw[(size_t)b*OBS_SIZE+CN_FZ*s])==0.0f)continue;
            float d=err[i];
            const float* f=feat+((size_t)b*CN_ZS+s)*SBS_FEAT;
            for(int k=0;k<SBS_FEAT;++k)g[k]+=d*asinhf(f[k]);
            g[SBS_FEAT]+=d;
        }
    }
    for(int k=0;k<SBS_ROOM;++k)wgrad[k]=from_float(g[k]);
}
__global__ void sbs_zv_bgrad_k(const precision_t* raw,const float* gz,precision_t* bgrad,int B,int nz){
    int s=blockIdx.x*blockDim.x+threadIdx.x;if(s>=CN_ZS)return;
    float g=0.0f;
    for(int b=0;b<B;++b){
        if(to_float(raw[(size_t)b*OBS_SIZE+CN_FZ*s])==0.0f)continue;
        g+=gz[(size_t)b*nz+s];
    }
    bgrad[s]=from_float(g);
}
__global__ void sbs_zvr_add_k(const precision_t* raw,const precision_t* residual,
        precision_t* values,int B,int rows){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*CN_ZS)return;
    int z=i%CN_ZS,b=i/CN_ZS;
    if(to_float(raw[(size_t)b*OBS_SIZE+CN_FZ*z])==0.f)return;
    size_t j=(size_t)b*rows+1+z;
    values[j]=from_float(to_float(values[j])+to_float(residual[i]));
}
__global__ void sbs_zvr_mask_grad_k(const precision_t* raw,const float* gz,
        precision_t* dst,int B,int zw){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=B*CN_ZS)return;
    int z=i%CN_ZS,b=i/CN_ZS;
    dst[i]=from_float(to_float(raw[(size_t)b*OBS_SIZE+CN_FZ*z])!=0.f ? gz[(size_t)b*zw+z] : 0.f);
}
Prec sbs_decoder_forward(void* ptr,void* act,Prec input,cudaStream_t stream){
    auto* w=(SBSWeights*)ptr;auto* a=(SBSActs*)act;int B=input.shape[0],H=w->base.hidden_dim,W=w->base.output_dim;
    assert(sbs_raw_input.shape[0]==B && sbs_raw_input.shape[1]==OBS_SIZE);
    a->raw=sbs_raw_input;
    if(a->saved_h.data)puf_copy(&a->saved_h,&input,stream);
    cudaMemsetAsync(a->out.data,0,(size_t)B*(W+1)*(g_beta?2:1)*sizeof(precision_t),stream);
    for(int r=0;r<3;++r){auto* v=&a->role[r];int N=SBS_NODES[r],L=SBS_LOCAL[r],O=sbs_role_out(r);
        const float* feat=g_sbs_feat_batch?g_sbs_feat_batch:g_sbs_feat;
        sbs_role_input_k<<<grid_size(B*N*(H+L)),BLOCK_SIZE,0,stream>>>(sbs_raw_input.data,input.data,v->input.data,B,OBS_SIZE,H,r,N,L,feat);
        puf_mm(&v->input,&w->role[r],&v->out,stream);
        if(r==0&&g_sbs_zone_mlp){
            puf_mm(&v->input,&w->zone_mlp1,&a->zone_hidden,stream);
            sbs_tanh_k<<<grid_size(B*N*SBS_ZONE_MLP_HIDDEN),BLOCK_SIZE,0,stream>>>(
                a->zone_hidden.data,B*N*SBS_ZONE_MLP_HIDDEN);
            puf_mm(&a->zone_hidden,&w->zone_mlp2,&a->zone_branch,stream);
            sbs_add_k<<<grid_size(B*N*O),BLOCK_SIZE,0,stream>>>(v->out.data,a->zone_branch.data,B*N*O);
        }
        if(r==1&&g_sbs_ahu_mlp){
            puf_mm(&v->input,&w->ahu_mlp1,&a->ahu_hidden,stream);
            sbs_tanh_k<<<grid_size(B*N*SBS_AHU_MLP_HIDDEN),BLOCK_SIZE,0,stream>>>(
                a->ahu_hidden.data,B*N*SBS_AHU_MLP_HIDDEN);
            puf_mm(&a->ahu_hidden,&w->ahu_mlp2,&a->ahu_branch,stream);
            sbs_add_k<<<grid_size(B*N*O),BLOCK_SIZE,0,stream>>>(v->out.data,a->ahu_branch.data,B*N*O);
        }
        int R=sbs_role_rows(r);
        sbs_scatter_k<<<grid_size(B*N*O),BLOCK_SIZE,0,stream>>>(v->out.data,a->out.data,B,N,O,R,r,W,SBS_OUTPUTS[r]);
    }
    if(g_sbs_components && a->heads.data){
        int din=H+CN_ZS*2;
        sbs_pack_value_k<<<grid_size(B*din),BLOCK_SIZE,0,stream>>>(input.data,g_sbs_critic,a->value_in.data,B,H,CN_ZS*2);
        puf_mm(&a->value_in,&w->base.weight,&a->heads,stream);
        sbs_combine_k<<<grid_size(B),BLOCK_SIZE,0,stream>>>(a->heads.data,g_sbs_aux,a->value.data,B);
        if(g_sbs_head_dst)cudaMemcpyAsync(g_sbs_head_dst,a->heads.data,(size_t)B*4*sizeof(precision_t),cudaMemcpyDeviceToDevice,stream);
    }else if(g_sbs_value_shared){
        if(g_sbs_value_extra){
            const float* feat=g_sbs_feat_batch?g_sbs_feat_batch:g_sbs_feat;
            sbs_pack_delay_k<<<grid_size(B*(H+SBS_VEXTRA)),BLOCK_SIZE,0,stream>>>(
                input.data,sbs_raw_input.data,feat,a->value_in.data,B,H);
            puf_mm(&a->value_in,&w->base.weight,&a->value,stream);
            /* r98: the zone rows are the value column of the role-0 readout plus the slot bias. This runs
             * AFTER the pooled matmul, which would otherwise overwrite rows 1..CN_ZS. */
            if(g_sbs_zone_value)
                sbs_zv_pick_k<<<grid_size(B*CN_ZS),BLOCK_SIZE,0,stream>>>(a->role[0].out.data,a->raw.data,
                    w->zvb.data,a->value.data,B,sbs_role_rows(0),sbs_role_out(0),sbs_value_rows());
        }else puf_mm(&input,&w->base.weight,&a->value,stream);
    }else{
        // Separate value MLP. Andrychowicz et al. 2020 (arXiv:2006.05990, C47):
        // no learned layer is shared with the policy. The fixed pool and the
        // delay summary are inputs, not policy weights, so the value gradient
        // cannot turn the trunk.
        const float* feat=g_sbs_feat_batch?g_sbs_feat_batch:g_sbs_feat;
        int extra=g_sbs_value_extra?SBS_VEXTRA:0;
        sbs_value_features_k<<<grid_size(B*(SBS_POOL+extra)),BLOCK_SIZE,0,stream>>>(
            sbs_raw_input.data,feat,a->vin.data,B,extra,g_sbs_occ_asinh);
        puf_mm(&a->vin,&w->vw1,&a->vh1,stream);
        sbs_tanh_k<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->vh1.data,B*H);
        puf_mm(&a->vh1,&w->vw2,&a->vh2,stream);
        sbs_tanh_k<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->vh2.data,B*H);
        puf_mm(&a->vh2,&w->vw3,&a->value,stream);
        sbs_popart_fwd_k<<<grid_size(B),BLOCK_SIZE,0,stream>>>(a->value.data,w->vb.data,w->pop.data,B);
    }
    if(g_sbs_zone_value_residual){
        // role[0].input is already constructed above. This value branch never
        // contributes to the actor output and its input is detached in backward.
        puf_mm(&a->role[0].input,&w->zvr1,&a->zvr_hidden,stream);
        sbs_tanh_k<<<grid_size(B*CN_ZS*SBS_ZVR_HIDDEN),BLOCK_SIZE,0,stream>>>(
            a->zvr_hidden.data,B*CN_ZS*SBS_ZVR_HIDDEN);
        puf_mm(&a->zvr_hidden,&w->zvr2,&a->zvr_out,stream);
        sbs_zvr_add_k<<<grid_size(B*CN_ZS),BLOCK_SIZE,0,stream>>>(
            a->raw.data,a->zvr_out.data,a->value.data,B,sbs_value_rows());
    }
    if(g_sbs_zone_critic && g_sbs_zpred)
        sbs_zpred_k<<<grid_size(B*g_sbs_zw),BLOCK_SIZE,0,stream>>>(a->value.data,g_sbs_zpred,B,sbs_value_rows());
    if(g_sbs_room_value && g_sbs_zpred){
        const float* feat=g_sbs_feat_batch?g_sbs_feat_batch:g_sbs_feat;
        sbs_room_fwd_k<<<grid_size(B*CN_ZS),BLOCK_SIZE,0,stream>>>(
            sbs_raw_input.data,feat,w->room.data,g_sbs_zpred,B);
    }
    sbs_value_scatter_k<<<grid_size(B),BLOCK_SIZE,0,stream>>>(a->value.data,a->out.data,B,W,sbs_value_rows());
    Prec out=a->out;out.shape[0]=B;return out;
}
Prec sbs_decoder_backward(void* ptr,void* act,Float grad_logits,Float grad_std,Float grad_value,cudaStream_t stream){
    auto* w=(SBSWeights*)ptr;auto* a=(SBSActs*)act;int B=a->saved_h.shape[0],H=w->base.hidden_dim,W=w->base.output_dim;
    if(g_sbs_room_value && a->room_g.data && g_sbs_zgrad){
        const float* feat=g_sbs_feat_batch?g_sbs_feat_batch:g_sbs_feat;
        sbs_room_grad_k<<<1,1,0,stream>>>(a->raw.data,feat,g_sbs_zgrad,a->room_g.data,B);
    }
    if(g_sbs_zone_value && a->zvb_g.data && g_sbs_zgrad){
        int nz=sbs_value_rows()-1;
        sbs_zv_bgrad_k<<<grid_size(CN_ZS),BLOCK_SIZE,0,stream>>>(a->raw.data,g_sbs_zgrad,a->zvb_g.data,B,nz);
    }
    if(g_sbs_components && a->heads.data){
        int din=H+CN_ZS*2;
        sbs_head_grad_k<<<grid_size(B),BLOCK_SIZE,0,stream>>>(grad_value.data,g_sbs_head_grad,a->gheads.data,B);
        puf_mm_tn(&a->gheads,&a->value_in,&a->wgrad_value,stream);
        puf_mm_nn(&a->gheads,&w->base.weight,&a->value_gin,stream);
        sbs_take_h_k<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->value_gin.data,a->grad_value_h.data,B,H,din);
    }else if(g_sbs_value_shared){
        if(g_sbs_zone_critic)
            sbs_pack_gvalue_k<<<grid_size(B),BLOCK_SIZE,0,stream>>>(a->gvalue.data,grad_value.data,
                g_sbs_zone_value?nullptr:g_sbs_zgrad,B,sbs_value_rows());   /* r98: pooled zone rows are dead */
        else
            cast<<<grid_size(B),BLOCK_SIZE,0,stream>>>(a->gvalue.data,grad_value.data,B);
        if(g_sbs_value_extra){
            int din=H+SBS_VEXTRA;
            puf_mm_tn(&a->gvalue,&a->value_in,&a->wgrad_value,stream);
            puf_mm_nn(&a->gvalue,&w->base.weight,&a->value_gin,stream);
            sbs_take_h_k<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->value_gin.data,a->grad_value_h.data,B,H,din);
        }else{
            puf_mm_tn(&a->gvalue,&a->saved_h,&a->wgrad_value,stream);
            puf_mm_nn(&a->gvalue,&w->base.weight,&a->grad_value_h,stream);
        }
    }else{
        sbs_popart_bwd_k<<<grid_size(B),BLOCK_SIZE,0,stream>>>(a->gvalue.data,grad_value.data,w->pop.data,B);
        sbs_bias_grad_k<<<1,256,0,stream>>>(a->gvalue.data,a->vgb.data,B);
        cudaMemsetAsync(a->vgp.data,0,4*sizeof(precision_t),stream);
        puf_mm_tn(&a->gvalue,&a->vh2,&a->vgw3,stream);
        puf_mm_nn(&a->gvalue,&w->vw3,&a->vdh2,stream);
        sbs_tanh_bwd_k<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->vdh2.data,a->vh2.data,B*H);
        puf_mm_tn(&a->vdh2,&a->vh1,&a->vgw2,stream);
        puf_mm_nn(&a->vdh2,&w->vw2,&a->vdh1,stream);
        sbs_tanh_bwd_k<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->vdh1.data,a->vh1.data,B*H);
        puf_mm_tn(&a->vdh1,&a->vin,&a->vgw1,stream);
        cudaMemsetAsync(a->wgrad_value.data,0,numel(a->wgrad_value.shape)*sizeof(precision_t),stream);
        cudaMemsetAsync(a->grad_value_h.data,0,numel(a->grad_value_h.shape)*sizeof(precision_t),stream);
    }
    // policy.beta: grad_std carries the per-row v gradients, and logstd is unused, so its gradient is 0.
    if(g_beta)cudaMemsetAsync(a->grad_std.data,0,(size_t)W*sizeof(precision_t),stream);
    else sum_rows_to_precision_kernel<<<grid_size(W),BLOCK_SIZE,0,stream>>>(a->grad_std.data,grad_std.data,B,W);
    if(g_sbs_zone_value_residual){
        assert(g_sbs_zgrad!=nullptr);
        sbs_zvr_mask_grad_k<<<grid_size(B*CN_ZS),BLOCK_SIZE,0,stream>>>(
            a->raw.data,g_sbs_zgrad,a->zvr_gout.data,B,g_sbs_zw);
        puf_mm_tn(&a->zvr_gout,&a->zvr_hidden,&a->zvr_gw2,stream);
        puf_mm_nn(&a->zvr_gout,&w->zvr2,&a->zvr_dhidden,stream);
        sbs_tanh_bwd_k<<<grid_size(B*CN_ZS*SBS_ZVR_HIDDEN),BLOCK_SIZE,0,stream>>>(
            a->zvr_dhidden.data,a->zvr_hidden.data,B*CN_ZS*SBS_ZVR_HIDDEN);
        puf_mm_tn(&a->zvr_dhidden,&a->role[0].input,&a->zvr_gw1,stream);
        // Deliberately no derivative into role[0].input or actor trunk.
    }
    for(int r=0;r<3;++r){auto* v=&a->role[r];int N=SBS_NODES[r],O=sbs_role_out(r),R=sbs_role_rows(r);
        int nz=sbs_value_rows()-1;
        sbs_role_grad_k<<<grid_size(B*N*R),BLOCK_SIZE,0,stream>>>(grad_logits.data,grad_std.data,a->raw.data,
            v->gout.data,B,N,O,R,r,W,SBS_OUTPUTS[r],(r==0&&g_sbs_zone_value)?g_sbs_zgrad:nullptr,nz);
        puf_mm_tn(&v->gout,&v->input,&v->wgrad,stream);           /* the value row's gradient lands here */
        if(r==0&&g_sbs_zone_value)
            sbs_zv_zerocol_k<<<grid_size(B*N),BLOCK_SIZE,0,stream>>>(v->gout.data,B*N,R,O);
        puf_mm_nn(&v->gout,&w->role[r],&v->gin,stream);
        if(r==1&&g_sbs_ahu_mlp){
            puf_mm_tn(&v->gout,&a->ahu_hidden,&a->ahu_gw2,stream);
            puf_mm_nn(&v->gout,&w->ahu_mlp2,&a->ahu_dhidden,stream);
            sbs_tanh_bwd_k<<<grid_size(B*N*SBS_AHU_MLP_HIDDEN),BLOCK_SIZE,0,stream>>>(
                a->ahu_dhidden.data,a->ahu_hidden.data,B*N*SBS_AHU_MLP_HIDDEN);
            puf_mm_tn(&a->ahu_dhidden,&v->input,&a->ahu_gw1,stream);
            puf_mm_nn(&a->ahu_dhidden,&w->ahu_mlp1,&a->ahu_dinput,stream);
            sbs_add_k<<<grid_size(B*N*(H+SBS_LA)),BLOCK_SIZE,0,stream>>>(
                v->gin.data,a->ahu_dinput.data,B*N*(H+SBS_LA));
        }
        if(r==0&&g_sbs_zone_mlp){
            puf_mm_tn(&v->gout,&a->zone_hidden,&a->zone_gw2,stream);
            puf_mm_nn(&v->gout,&w->zone_mlp2,&a->zone_dhidden,stream);
            sbs_tanh_bwd_k<<<grid_size(B*N*SBS_ZONE_MLP_HIDDEN),BLOCK_SIZE,0,stream>>>(
                a->zone_dhidden.data,a->zone_hidden.data,B*N*SBS_ZONE_MLP_HIDDEN);
            puf_mm_tn(&a->zone_dhidden,&v->input,&a->zone_gw1,stream);
            puf_mm_nn(&a->zone_dhidden,&w->zone_mlp1,&a->zone_dinput,stream);
            sbs_add_k<<<grid_size(B*N*(H+SBS_LZ)),BLOCK_SIZE,0,stream>>>(
                v->gin.data,a->zone_dinput.data,B*N*(H+SBS_LZ));
        }
    }
    if(g_sbs_tl && g_sbs_tl_stat!=nullptr && g_sbs_tl_part!=nullptr){
        int nb=grid_size(B*H);
        sbs_vtc_parts_k<<<nb,BLOCK_SIZE,0,stream>>>(a->role[0].gin.data,a->role[1].gin.data,
            a->role[2].gin.data,a->grad_value_h.data,g_sbs_tl_part,B,H);
        sbs_tl_acc_k<<<1,BLOCK_SIZE,0,stream>>>(g_sbs_tl_part,nb,g_sbs_tl_stat);
    }
    // [train] value_trunk_cap: clamp the value path's share of the trunk gradient at the policy
    // path's magnitude, once per minibatch visit. Everything rides `stream`, so the order holds
    // inside the captured train graph.
    if(g_sbs_vtc && g_sbs_vtc_stat!=nullptr && g_sbs_vtc_part!=nullptr){
        int nb=grid_size(B*H);
        sbs_vtc_parts_k<<<nb,BLOCK_SIZE,0,stream>>>(a->role[0].gin.data,a->role[1].gin.data,
            a->role[2].gin.data,a->grad_value_h.data,g_sbs_vtc_part,B,H);
        sbs_vtc_scale_k<<<1,BLOCK_SIZE,0,stream>>>(g_sbs_vtc_part,nb,g_sbs_vtc_stat);
        sbs_vtc_apply_k<<<nb,BLOCK_SIZE,0,stream>>>(a->grad_value_h.data,g_sbs_vtc_stat,B*H);
    }
    // [train] value_spike_guard: the relative guard. It reads only ||g_V||, so it costs a pass over
    // grad_value_h rather than over the role gradients. Per visit, on grad_value_h only.
    if(g_sbs_vsg && g_sbs_vsg_stat!=nullptr && g_sbs_vsg_ema!=nullptr && g_sbs_vsg_part!=nullptr){
        int nb=grid_size(B*H);
        sbs_vsg_parts_k<<<nb,BLOCK_SIZE,0,stream>>>(a->grad_value_h.data,g_sbs_vsg_part,B*H);
        sbs_vsg_scale_k<<<1,BLOCK_SIZE,0,stream>>>(g_sbs_vsg_part,nb,g_sbs_vsg_stat,g_sbs_vsg_ema,
            g_sbs_vsg_q,g_sbs_vsg_a);
        sbs_vsg_apply_k<<<nb,BLOCK_SIZE,0,stream>>>(a->grad_value_h.data,g_sbs_vsg_ema,B*H);
    }
    sbs_context_grad_k<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->role[0].gin.data,a->role[1].gin.data,a->role[2].gin.data,
        a->grad_value_h.data,a->grad_h.data,B,H);return a->grad_h;
}
// Same recurrence as puff_advantage with importance 1. The reward stored at
// index t+1 is the reward of the action at t. The last index stays 0.
// Slot is the fast dimension.
__global__ void sbs_zone_gae_k(const float* v, const float* r, const precision_t* dones,
        float* adv, float* ret, int rows, int H, float gamma, float lambda, int ZW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ZW || H < 1) return;
    int s = i % ZW;
    int row = i / ZW;
    int last = row * H + (H - 1);
    float next_v = v[(size_t)last * ZW + s];
    float next_d = to_float(dones[last]);
    float next_r = r[(size_t)last * ZW + s];
    float lastlam = 0.0f;
    adv[(size_t)last * ZW + s] = 0.0f;
    ret[(size_t)last * ZW + s] = next_v;
    for (int t = H - 2; t >= 0; --t) {
        int cur = row * H + t;
        float live = 1.0f - next_d;
        float delta = next_r + gamma * next_v * live - v[(size_t)cur * ZW + s];
        lastlam = delta + gamma * lambda * lastlam * live;
        adv[(size_t)cur * ZW + s] = lastlam;
        ret[(size_t)cur * ZW + s] = v[(size_t)cur * ZW + s] + lastlam;
        next_v = v[(size_t)cur * ZW + s];
        next_d = to_float(dones[cur]);
        next_r = r[(size_t)cur * ZW + s];
    }
}
void sbs_zone_gae(const float* v, const float* r, const precision_t* dones,
        float* adv, float* ret, int rows, int H, float gamma, float lambda, cudaStream_t stream) {
    int n = rows * g_sbs_zw;
    sbs_zone_gae_k<<<(n + 255) / 256, 256, 0, stream>>>(v, r, dones, adv, ret, rows, H, gamma, lambda, g_sbs_zw);
}
__global__ void sbs_zone_transpose_k(const float* src, float* dst, int T, int B, int ZW) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)T * B * ZW;
    if (i >= n) return;
    int s = (int)(i % ZW);
    size_t te = i / ZW;
    int e = (int)(te % (size_t)B);
    int t = (int)(te / (size_t)B);
    dst[((size_t)e * T + t) * ZW + s] = src[i];
}
void sbs_zone_transpose(const float* src, float* dst, int T, int B, cudaStream_t stream) {
    size_t n = (size_t)T * B * g_sbs_zw;
    sbs_zone_transpose_k<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(src, dst, T, B, g_sbs_zw);
}
// Present slots only. Absent slots stay out of the scale.
// One block: each thread strides over (row, t), the block sums in double in a fixed tree order.
__global__ void sbs_zone_stat_k(const float* adv, const precision_t* obs, int obs_dim,
        int rows, int H, float* pair, int ZW, int ahu) {
    __shared__ double part[3][256];
    const int t0 = threadIdx.x;
    double sum = 0.0, sumsq = 0.0, count = 0.0;
    for (int rt = t0; rt < rows * H; rt += blockDim.x) {
        const precision_t* o = obs + (size_t)rt * obs_dim;
        const int s0 = ahu ? CN_ZS : 0, s1 = ahu ? CN_ZS + CN_K : CN_ZS;
        for (int s = s0; s < s1; ++s) {
            float present = ahu ? to_float(o[CN_OA + CN_FA * (s - CN_ZS)]) : to_float(o[CN_FZ * s]);
            if (present == 0.0f) continue;
            double a = adv[(size_t)rt * ZW + s];
            sum += a;
            sumsq += a * a;
            count += 1.0;
        }
    }
    part[0][t0] = sum; part[1][t0] = sumsq; part[2][t0] = count;
    __syncthreads();
    for (int d = blockDim.x / 2; d; d >>= 1) {
        if (t0 < d) for (int k = 0; k < 3; ++k) part[k][t0] += part[k][t0 + d];
        __syncthreads();
    }
    if (t0 == 0) {
        pair[0] = (float)part[0][0];
        pair[1] = (float)part[1][0];
        pair[2] = (float)part[2][0];
    }
}
void sbs_zone_stat(const float* adv, const precision_t* obs, int obs_dim,
        int rows, int H, float* pair, cudaStream_t stream, int ahu = 0) {
    sbs_zone_stat_k<<<1, 256, 0, stream>>>(adv, obs, obs_dim, rows, H, pair, g_sbs_zw, ahu);
}
__global__ void sbs_zone_stat_finish_k(const float* pairs, int nblocks, float* stats) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    double sum = 0.0, sumsq = 0.0, count = 0.0;
    for (int b = 0; b < nblocks; ++b) {
        sum += pairs[3 * b];
        sumsq += pairs[3 * b + 1];
        count += pairs[3 * b + 2];
    }
    float mean = count > 0.0 ? (float)(sum / count) : 0.0f;
    float var = count > 0.0 ? (float)(sumsq / count) - mean * mean : 0.0f;
    if (var < 0.0f) var = 0.0f;
    stats[0] = mean;
    stats[1] = count > 0.0 ? 1.0f / sqrtf(var + 1.0e-8f) : 0.0f;
}
void sbs_zone_stat_finish(const float* pairs, int nblocks, float* stats, cudaStream_t stream) {
    sbs_zone_stat_finish_k<<<1, 1, 0, stream>>>(pairs, nblocks, stats);
}
// Same clip as the scalar value loss. The average is over batch steps and slots,
// so ninety slot heads together have the weight of the one building head. The AHU rows
// (ahu_critic) are averaged on their own, so together they also weigh one head.
__global__ void sbs_zone_vloss_k(const float* pred, const float* oldv, const float* ret,
        float* grad, int n, float clip, float coef, int ZW) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * ZW) return;
    const int nslot = (i % ZW) < CN_ZS ? n * CN_ZS : n * (ZW - CN_ZS);
    float vp = pred[i], ov = oldv[i], rt = ret[i];
    if (!isfinite(vp) || !isfinite(ov) || !isfinite(rt)) { grad[i] = 0.0f; return; }
    float vc = ov + fminf(clip, fmaxf(-clip, vp - ov));
    float unclipped = (vp - rt) * (vp - rt);
    float clipped = (vc - rt) * (vc - rt);
    float d = (clipped > unclipped) ? 0.0f : (vp - rt);
    if (!isfinite(d)) d = 0.0f;
    grad[i] = (coef / (float)nslot) * d;
}
void sbs_zone_vloss(const float* pred, const float* oldv, const float* ret,
        float* grad, int n, float clip, float coef, cudaStream_t stream) {
    int nslot = n * g_sbs_zw;
    sbs_zone_vloss_k<<<(nslot + 255) / 256, 256, 0, stream>>>(pred, oldv, ret, grad, n, clip, coef, g_sbs_zw);
}

// One shared affine control variate c(h,x_z) with a separate Adam state. The
// role-0 tensor already contains a constant and zeroes absent zones. No
// derivative from this auxiliary regression enters that tensor or the actor.
// The training-only state is checkpointed separately and is not needed at eval.
static Prec g_team_w={},g_team_pred={},g_team_err={},g_team_grad={};
static float *g_team_m=nullptr,*g_team_v=nullptr,*g_team_bc=nullptr,*g_team_count=nullptr;
static int* g_team_step=nullptr;
static int g_team_features=0, g_team_capacity=0;
constexpr int SBS_TEAM_DIAG=16;
static double* g_team_diag=nullptr;
static int g_team_agents=0,g_team_group=0;

__global__ void sbs_team_residual_k(const precision_t* pred,const precision_t* adv,
        const precision_t* obs,float* out,int rows,int obs_dim,int zw){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*CN_ZS)return;
    int s=i%CN_ZS,r=i/CN_ZS;
    if(to_float(obs[(size_t)r*obs_dim+CN_FZ*s])==0.0f){out[(size_t)r*zw+s]=0.f;return;}
    float a=to_float(adv[r]),c=to_float(pred[i]);
    out[(size_t)r*zw+s]=a-c;
}
__global__ void sbs_team_count_k(const precision_t* obs,int rows,int obs_dim,float* count){
    __shared__ float sums[256];int tid=threadIdx.x;float n=0.f;
    for(int i=tid;i<rows*CN_ZS;i+=256){int s=i%CN_ZS,r=i/CN_ZS;
        n+=to_float(obs[(size_t)r*obs_dim+CN_FZ*s])!=0.f;}
    sums[tid]=n;__syncthreads();
    for(int d=128;d;d>>=1){if(tid<d)sums[tid]+=sums[tid+d];__syncthreads();}
    if(tid==0)*count=sums[0];
}
__global__ void sbs_team_error_k(const precision_t* pred,const precision_t* adv,
        const precision_t* obs,precision_t* err,const float* count,int rows,int obs_dim){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*CN_ZS)return;
    int s=i%CN_ZS,r=i/CN_ZS;
    float scale=*count>0.f?1.f/(*count):0.f;
    float d=to_float(obs[(size_t)r*obs_dim+CN_FZ*s])!=0.f
        ? (to_float(pred[i])-to_float(adv[r]))*scale : 0.f;
    err[i]=from_float(d);
}
__global__ void sbs_team_adam_k(const precision_t* grad,precision_t* w,float* m,float* v,
        const float* bc,const float* lr,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float g=to_float(grad[i]);if(!isfinite(g)){asm("trap;");return;}
    float mi=.9f*m[i]+.1f*g,vi=.999f*v[i]+.001f*g*g;m[i]=mi;v[i]=vi;
    w[i]=from_float(to_float(w[i])-(*lr)*(mi/bc[0])/(sqrtf(vi/bc[1])+1e-8f));
}
static void sbs_team_alloc(int rows,int H,int agents,int group){
    if(!g_sbs_team || g_team_w.data)return;
    if(USE_BF16 || rows<=0 || H<=0 || agents<=0 || group<=0 || agents%group){
        fprintf(stderr,"pufferl: team baseline needs float policy and valid equal-sized groups\n");abort();
    }
    g_team_features=H+SBS_LZ;g_team_capacity=rows*CN_ZS;g_team_agents=agents;g_team_group=group;
    g_team_w.shape[0]=1;g_team_w.shape[1]=g_team_features;
    g_team_grad.shape[0]=1;g_team_grad.shape[1]=g_team_features;
    g_team_pred.shape[0]=g_team_capacity;g_team_pred.shape[1]=1;
    g_team_err.shape[0]=g_team_capacity;g_team_err.shape[1]=1;
    assert(cudaMalloc(&g_team_w.data,(size_t)g_team_features*sizeof(precision_t))==cudaSuccess);
    assert(cudaMalloc(&g_team_grad.data,(size_t)g_team_features*sizeof(precision_t))==cudaSuccess);
    assert(cudaMalloc(&g_team_pred.data,(size_t)g_team_capacity*sizeof(precision_t))==cudaSuccess);
    assert(cudaMalloc(&g_team_err.data,(size_t)g_team_capacity*sizeof(precision_t))==cudaSuccess);
    assert(cudaMalloc(&g_team_m,(size_t)g_team_features*sizeof(float))==cudaSuccess);
    assert(cudaMalloc(&g_team_v,(size_t)g_team_features*sizeof(float))==cudaSuccess);
    assert(cudaMalloc(&g_team_bc,2*sizeof(float))==cudaSuccess);
    assert(cudaMalloc(&g_team_count,sizeof(float))==cudaSuccess);
    assert(cudaMalloc(&g_team_step,sizeof(int))==cudaSuccess);
    assert(cudaMalloc(&g_team_diag,(size_t)agents*SBS_TEAM_DIAG*sizeof(double))==cudaSuccess);
    assert(cudaMemset(g_team_w.data,0,(size_t)g_team_features*sizeof(precision_t))==cudaSuccess);
    assert(cudaMemset(g_team_m,0,(size_t)g_team_features*sizeof(float))==cudaSuccess);
    assert(cudaMemset(g_team_v,0,(size_t)g_team_features*sizeof(float))==cudaSuccess);
    assert(cudaMemset(g_team_step,0,sizeof(int))==cudaSuccess);
    assert(cudaMemset(g_team_diag,0,(size_t)agents*SBS_TEAM_DIAG*sizeof(double))==cudaSuccess);
    fprintf(stderr,"pufferl: valve team baseline on (%d shared detached features, separate Adam; %d agents/group)\n",g_team_features,group);
    fprintf(stderr,"SB_TEAM columns: epoch group n sum_A sum_A2 sum_c sum_c2 sum_Ac sum_res sum_res2 "
        "sum_Au sum_Au2 sum_resu sum_resu2 sum_Av sum_Av2 sum_resv sum_resv2; "
        "pre-update marginal Beta score proxy, every 24th time and fourth zone\n");
}
static void sbs_team_apply(SBSActs* a,const precision_t* adv,const precision_t* obs,
        float* zadv,int rows,int obs_dim,int zw,cudaStream_t stream){
    assert(g_team_w.data && rows*CN_ZS<=g_team_capacity && zw==CN_ZS);
    Prec x=a->role[0].input; x.shape[0]=rows*CN_ZS;x.shape[1]=g_team_features;
    Prec pred=g_team_pred;pred.shape[0]=rows*CN_ZS;pred.shape[1]=1;
    puf_mm(&x,&g_team_w,&pred,stream);
    sbs_team_residual_k<<<grid_size(rows*CN_ZS),BLOCK_SIZE,0,stream>>>(
        pred.data,adv,obs,zadv,rows,obs_dim,zw);
}
static void sbs_team_fit(SBSActs* a,const precision_t* adv,const precision_t* obs,
        int rows,int obs_dim,cudaStream_t stream){
    Prec x=a->role[0].input;x.shape[0]=rows*CN_ZS;x.shape[1]=g_team_features;
    Prec err=g_team_err;err.shape[0]=rows*CN_ZS;err.shape[1]=1;
    sbs_team_count_k<<<1,256,0,stream>>>(obs,rows,obs_dim,g_team_count);
    sbs_team_error_k<<<grid_size(rows*CN_ZS),BLOCK_SIZE,0,stream>>>(
        g_team_pred.data,adv,obs,err.data,g_team_count,rows,obs_dim);
    puf_mm_tn(&err,&x,&g_team_grad,stream);
}
static void sbs_team_update(const float* lr,cudaStream_t stream){
    sbs_head_adam_bump_k<<<1,1,0,stream>>>(g_team_step,g_team_bc);
    sbs_team_adam_k<<<grid_size(g_team_features),BLOCK_SIZE,0,stream>>>(
        g_team_grad.data,g_team_w.data,g_team_m,g_team_v,g_team_bc,lr,g_team_features);
}
static void sbs_team_save(const char* path){
    if(!g_sbs_team)return;
    char side[4200];snprintf(side,sizeof(side),"%s.team",path);
    FILE* f=fopen(side,"wbx");if(!f){fprintf(stderr,"pufferl: cannot create %s\n",side);abort();}
    std::vector<float> data((size_t)g_team_features*3);int step=0;
    assert(cudaMemcpy(data.data(),g_team_w.data,(size_t)g_team_features*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(data.data()+g_team_features,g_team_m,(size_t)g_team_features*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(data.data()+2*g_team_features,g_team_v,(size_t)g_team_features*sizeof(float),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(cudaMemcpy(&step,g_team_step,sizeof(step),cudaMemcpyDeviceToHost)==cudaSuccess);
    assert(fwrite(&g_team_features,sizeof(int),1,f)==1 && fwrite(&step,sizeof(int),1,f)==1);
    assert(fwrite(data.data(),sizeof(float),data.size(),f)==data.size());assert(fclose(f)==0);
}
static void create_sbs_decoder(Decoder* d){
    d->forward=sbs_decoder_forward;d->backward=sbs_decoder_backward;d->init_weights=sbs_decoder_init;
    d->reg_params=sbs_decoder_params;d->reg_train=sbs_decoder_train;d->reg_rollout=sbs_decoder_rollout;
    d->create_weights=sbs_decoder_create;d->activation_size=sizeof(SBSActs);
}
