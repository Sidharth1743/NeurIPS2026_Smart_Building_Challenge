/* PufferLib 5.0 environment for the NeurIPS 2026 Smart Buildings competition,
 * GPU path.
 *
 * The whole simulator steps on the device (see sb_gpu_kernels.cuh: occupancy,
 * HVAC/pre, batched cuDSS FDM solve, air side, second cuDSS solve, scatter,
 * sensed means, post phase, convection, reward and observation).  This file is
 * the host core: it loads a baked scenario, uploads the shared tables, sets up
 * cuDSS once, allocates the per-env state, and drives one batched step.
 *
 * Built two ways from the same source:
 *   - SB_TEST defined: a standalone driver that replays a recorded trace and
 *     diffs the device against it (the parity gate);
 *   - otherwise: the PufferLib env (puf_vec_create / bind_stream / init /
 *     reset / step / close), mirroring the CPU reference env's observable
 *     behaviour exactly (clamped continuous heads, discrete run bits, the same
 *     reward/observation ordering, auto-reset at the episode end).
 */
#include "sb_gpu_kernels.cuh"
#include "sb_rng.h"

#include <dirent.h>
#include <unistd.h>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <array>
#include <string>
#include <vector>

/* action-space shape the trainer is told about (same numbers as the CPU env) */
#define SB_OBS 1024
#ifdef SB_CONT_BIG          /* buildings with up to 320 continuous actions (e.g. 4-floor generated ones) */
#define SB_CONT 320
#else
#define SB_CONT 256
#endif
#define SB_DISC 16
#define SB_NE 512
#define SB_NATIVE_MAX (SB_CONT + SB_DISC)
#define SB_MAX_BLOBS 64
#define SB_PATH_MAX 512
#define SB_ONE_1 1
#define SB_ONE_2 SB_ONE_1, SB_ONE_1
#define SB_ONE_4 SB_ONE_2, SB_ONE_2
#define SB_ONE_8 SB_ONE_4, SB_ONE_4
#define SB_ONE_16 SB_ONE_8, SB_ONE_8
#define SB_ONE_32 SB_ONE_16, SB_ONE_16
#define SB_ONE_64 SB_ONE_32, SB_ONE_32
#define SB_ONE_128 SB_ONE_64, SB_ONE_64
#define SB_ONE_256 SB_ONE_128, SB_ONE_128
#define SB_TWO_1 2
#define SB_TWO_2 SB_TWO_1, SB_TWO_1
#define SB_TWO_4 SB_TWO_2, SB_TWO_2
#define SB_TWO_8 SB_TWO_4, SB_TWO_4
#define SB_TWO_16 SB_TWO_8, SB_TWO_8
#define OBS_SIZE SB_OBS
#define NUM_ATNS (SB_CONT + SB_DISC)
#ifdef SB_CONT_BIG
#define ACT_SIZES { SB_ONE_256, SB_ONE_64, SB_TWO_16 }
#else
#define ACT_SIZES { SB_ONE_256, SB_TWO_16 }
#endif
#define SB_MASK_WIDTH (SB_CONT + 2 * SB_DISC)
/* The trainer fills its action mask through puf_mask_init: a head whose mask column is 0 contributes
 * nothing to the policy loss and does not count in ratio_per_dim.  Defined for both layouts (the
 * entity block below overrides the row width); the canonical row is a slot the building lacks
 * (no damper / no reheat / no AHU / ...). */
#define PUF_HAS_MASK_INIT

#ifdef SB_ENTITY
/* One policy agent per zone slot.  The trainer's MLP is then shared across zones (and across buildings).
 * AHU and hot-water actions are left at the base controller.  OBS = 9 zone features + 19 global features. */
#define SB_NENT 90
#define SB_ENT_OBS 28
#define SB_ENT_ACT 2
#undef OBS_SIZE
#undef NUM_ATNS
#undef ACT_SIZES
#undef SB_MASK_WIDTH
#define OBS_SIZE SB_ENT_OBS
#define NUM_ATNS SB_ENT_ACT
#define ACT_SIZES {1, 1}
#define SB_MASK_WIDTH SB_ENT_ACT
#define PUF_HAS_MASK_INIT
#endif

#ifndef SB_TEST
typedef float obs_t;

/* GPU-backed env: the step runs the whole vector inside the env */
#define PUF_BACKEND PUF_GPU
#define PUF_HAS_LIVE_LOG
#include "pufferenv.h"
#include "sb_full_io.cuh"

/* The framework forward-declares Log and Env; each env supplies them.  The
 * trainer's irreducible columns are perf/score/episode_return/episode_length. */
struct Log {
    float perf;
    float score;
    float episode_return;
    float episode_length;
    float comfort_dev_K;       /* episode mean of the per-step mean zone deviation */
    float worst_dev_K;         /* episode max zone deviation */
    float out_of_band_frac;    /* episode mean fraction of zones outside the band */
    float productivity;        /* episode mean normalized productivity term */
    float energy_cost_norm;    /* episode mean normalized energy cost */
    float carbon_norm;         /* episode mean normalized carbon */
    float elec_kwh;            /* episode total */
    float gas_kwh;             /* episode total */
    float n;
};
#define SB_NPARTS_ENV 8
struct Env {
#ifdef SB_ENTITY
    Agent agents[SB_NENT];
#else
    Agent agents[1];
#endif
    Log log;
    float ep_ret;                     /* running episode sums (never in Log: the trainer */
    float ep_part[SB_NPARTS_ENV];     /* clears Log mid-episode); filled by finalize_k   */
    float ep_worst;
    int num_agents;
    int tag;
    int boundary_reached;
    int step;
    int max_steps;
    int start_offset, start_day;      /* elapsed step stays independent of calendar */
    unsigned long long start_rng;     /* per-env random-start stream */
    unsigned int start_draws;
    int n_act, n_cont, n_disc;
    int loaded;
    unsigned int rng;     /* the trainer seeds these; must exist in Env */
    int seed;
    double* cont;
    double* disc;
    double* native;
    int env_id;
};
#endif

extern "C" void sb_decode_native(int n_act, const uint8_t* is_disc,
                                 const double* min_native, const double* max_native,
                                 const double* min_norm, const double* max_norm,
                                 const double* cont, const double* disc, double* out);

#include "sb_gpu_base.inc"   /* canonical layout, on-device pid_water, residual decode */
#include "sb_zone_command.cuh"
#ifdef SB_ENTITY
static_assert(SB_ENT_OBS == CN_FZ + CN_FG, "entity observation must be the zone features plus the global features");
static_assert(SB_NENT == CN_ZS, "entity count must be the canonical zone-slot count");
#endif

namespace sbgpu {

/* ---------------------------------------------------------------- state --- */
/* One scenario group.  A batch is loaded from one blob (the gateway runs one
 * scenario per episode, so this is also the submission shape); the struct is
 * dimensioned so more groups can be added without touching the kernels. */
static SbBlob blob;
static int nenv;                 /* envs in the batch */
static int n_rooms_c;
static double conv_p;
static int conv_dist;

/* blob scalars / shapes */
static int nr, nc, ncv, n, k, n_ahu, n_vav, n_zones, n_action;
static int n_dev_, n_aux_, obs_len_, max_room, conv_words;
static int cv2_ok = 0;            /* convect2_k's limits hold for this scenario */
static size_t cv2_smem = 0;
static int air_global = 0;
static double* d_airM;
static int cv2_global = 0;        /* huge rooms: swap list / permutation in global memory */
static unsigned int* d_cv_lr;
static unsigned short* d_cv_p;
static double* d_cvscr;           /* gather scratch for the permutation apply */
static int *d_cvdst, *d_cvsrc, *d_cvcnt;   /* the producer's permutation lists */
static int cv_ns = 1, cv_cap = 0;          /* producer streams, list capacity */
static cudaStream_t stream2;               /* convection producer (side stream) */
static cudaEvent_t ev_app, ev_prod;
static int nassign, occ_ea, occ_la, occ_ed, occ_ld;
static double occ_step, p_arr, p_dep, gas_carb_, pump_eff_;
static double dt_sec_;

/* shared (per-scenario) device tables */
static const int64_t* d_hour_m1;
static const uint8_t* d_wd_m1;
static double *sh_zarea, *sh_ahu, *sh_vav, *sh_sc, *sh_sh, *sh_scl, *sh_amb, *sh_boil;
static const int64_t* sh_hour;
static const uint8_t* sh_wd;
static const int *sh_roo, *sh_rcc, *sh_coo, *sh_cii;
static const int *sh_zvo, *sh_zvi, *sh_vai, *sh_cvi, *sh_pri, *sh_ro, *sh_rc;
static const int *sh_aidx;
static const uint8_t *sh_akind, *sh_aact;
static const double *sh_rws, *sh_ewd, *sh_ewe, *sh_ec, *sh_gp;
static const int64_t *sh_rwh, *sh_rwm;
static const uint8_t* sh_rww;
static const uint8_t *sh_fk, *sh_fid;
static const int *sh_fix;
static const double *sh_fme, *sh_fva, *sh_axt;
static const int *sd_i;
static const int *sd_sp, *sd_si, *sd_ap, *sd_ai;
static const double *sd_cap, *sd_dir, *sd_mc, *sd_a0, *sd_a1, *sd_hor, *sd_ver, *sd_con;
static const double *sd_ret, *sd_gain, *sd_spx, *sd_apx;

/* device buffers */
static cudaStream_t stream;
static GpuSim* d_sims;
static GpuRng *d_rngc, *d_rngo;
static double *d_native, *d_V, *d_temp, *d_mass_temp, *d_obs_d, *d_rew, *d_tmean, *d_rparts;
static double* d_phi;          /* env.comfort_shape: previous step's mean zone shortfall, per env */
static float* d_shape;         /* env.comfort_shape: this step's shaping bonus, per env (trainer only) */
static float* d_epret;
static float* d_obs;
/* Device-resident targets the framework hands over: a GPU env's agent buffers
 * live on the device, so the step must write them in place. */
static float *g_obs, *g_rewards, *g_terminals, *g_actions;
static const uint8_t* d_act_disc;
static const double *d_act_mn, *d_act_mx, *d_act_mnr, *d_act_mxr;
static int* d_reset;
#ifndef SB_TEST
static Env* g_envs;             /* the framework's managed Env batch */
static float *g_rew_fw, *g_term_fw;       /* framework reward / terminal rows (entity mode scatters into these) */
static float *d_canon_obs, *d_entact;     /* per group: canonical observation scratch, gathered canonical actions */
static float* d_cmdact;                   /* per group: zone-command expansion, canonical width */
static float* d_probeact;                 /* per group: probe action vector, canonical width */
static float *d_rew_learn, *d_term_learn; /* per group: one reward and terminal per env, before the zone scatter */
#else
static int g_test_ep_len = 0;   /* gate only: in-step restart after this many steps */
static int g_test_start = 0;    /* gate only: compare a shifted reset with the oracle */
__global__ void test_start_k(GpuSim* sims, int* stepv, int nenv, int offset, const int* gate) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv || (gate != NULL && (gate[0] == 0 || gate[1 + e] == 0))) return;
    stepv[e] = offset;
    sims[e].step_index = offset;
    sims[e].start_index = offset;
}
__global__ void test_done_k(const int* stepv, int ep_len, int* reset, int nenv) {
    int any = 0;
    for (int e = 0; e < nenv; e++) { reset[1 + e] = (stepv[e] >= ep_len) ? 1 : 0; any |= reset[1 + e]; }
    reset[0] = any;
}
#endif
static int *d_stepv, *d_occstate, *d_consumed, *d_left, *d_right;
static int* d_coff;             /* room-cell -> control-volume offsets */
static unsigned int* d_words;
static double *d_apc_keep, *d_yrk_keep;
static int *d_rrp, *d_rrc;          /* R = sensed P_c U^-1, CSR over the k sensors */
static double* d_rrx;
static long nnz_z, nnz_r;
static int *d_rseg_p, *d_rrow_seg, n_rseg;
static int* h_tlev;                           /* level of each group-local solve ticket (2n) */   /* R cut into SB_RSEG-entry segments (sensed_R_part_k) */
static double* d_rpart;                       /* [segment][ld] partial sums */
static double *d_tinit, *d_minit, *d_exdir;
/* the device replay of the baked SuperLU solve (sb_gpu_lusolve.cuh) */
static LuDev g_lu;
static int lu_rpl = 1, lu_ld = 32;
static sb_wt *d_lu_work, *d_lu_z;                /* forward result y / backward result z */
static unsigned int *d_lu_fl, *d_lu_fu, *d_lu_tk, *d_lu_ep;
static int *d_rowpos, *d_pc;
static int* d_cellzone;
static int tmean_needed = 1;     /* the field mean only feeds the reward for an AHU without VAV area */
static int *d_nonsolved, n_nonsolved;      /* cells the FDM does not solve */
static double *d_qair, *d_sr;
static double *d_g, *d_t2;   /* air side, read by the gate */
static int *sh_tp, *sh_ti;
static double* sh_tx;
static size_t smem;
static int th, zth, cth, nth2, kth;

static int* d_envscen;            /* per env: index of its scenario in the batch (per-scenario live metrics) */
static std::vector<std::string> g_scen_names;   /* scenario names, batch-wide (not per group) */
static double g_last_sc[SB_MAX_BLOBS] = {};     /* carried episode scores between live-log windows */
static float g_rew_scale = 1.0f;   /* env.reward_scale: multiplies the reward handed to the trainer (batch-wide) */
/* env.rew_norm = 1 (PopArt-lite, van Hasselt et al. 2016): the learning reward becomes
 * (r - mu_s) / sigma_s with per-scenario running mean/variance of the raw reward, updated at
 * live-log time.  Affine per scenario => action preferences WITHIN a scenario are unchanged;
 * it equalizes the ~1000x cross-scenario return range so one shared critic can fit all of them.
 * Logs, metrics and episode records keep the raw reward (only rewards[] to the trainer changes). */
static float g_rew_norm = 0.0f;
/* env.rew_center = 1: on-policy reward centering (Naik, Wan, Tomar, Sutton, RLC 2024, eq 4).
 * The learning reward is r - mu_s, with mu_s a per-scenario EMA of the raw reward, updated once
 * per live-log window: mu <- (1-beta)*mu + beta*window_mean.  inv_std stays 1 (no division: the
 * PopArt path's moving std made the critic chase a target).  mu starts at 0 so a warm critic,
 * which predicts uncentered returns, is not shifted by mu/(1-gamma) on the first update.
 * Equation 10 (step mu by the advantage) is the wrong estimator here: once explained variance is
 * ~1 the advantage mean is ~0, so that step never removes the constant r_bar/(1-gamma).
 * Logs, metrics and episode records keep the raw reward.  rew_norm wins if both are set. */
static float g_rew_center = 0.0f;
static float g_center_beta = 0.01f;
static float* d_rew_norm;          /* per scenario [2]: mean, inv_std (device mirror, updated host-side) */
static double g_rn_mu[SB_MAX_BLOBS], g_rn_var[SB_MAX_BLOBS];   /* host EMA state across live-log calls */
/* env.comfort_shape = K > 0: train-only potential-based shaping of the comfort term.
 * The learning reward handed to the trainer gains F_t = shape_gamma*Phi(s_t) - Phi(s_{t-1}) with
 * Phi(s) = -K * mean_z shortfall_z(s), computed from the SAME per-zone deviation the reward uses
 * but WITHOUT the occupancy weight.  A setback/plant shutdown is then charged as the zones drift
 * (the step that causes it) instead of when occupancy returns (12 h later, where (gamma*lambda)^144
 * is numerically absent for the estimator).  Summed over an episode it telescopes to
 * shape_gamma^T*Phi_T - Phi_0: the objective is unchanged (up to a negligible boundary term), only
 * the credit assignment moves.  Only rewards[] to the trainer changes: rew_out, rp[], ep_ret, the
 * live metrics and the eval dump keep the official reward. */
static float g_comfort_shape = 0.0f;
static float g_shape_gamma = 0.99f;   /* env.shape_gamma: the shaping's gamma (match train.gamma) */
/* env.canonical / env.residual / env.res_scale / env.res_dscale (batch-wide; see sb_gpu_base.inc) */
static int g_canon = 0, g_resid = 0;
static int g_random_start = 0;      /* batch-wide; separate from desync */
/* env.plant_pair: odd envs repeat the even env's action with the plant setpoints raised.
 * The trainer reads the two official returns. Off unless the run asks for it. */
static int g_plant_pair = 0;
static float g_plant_eps = 0.25f;
/* env.zone_command: one squashed sample per zone becomes damper and reheat.
 * u = 0 is half reheat at the damper floor. r61_mod_band: the warm stroke is
 * joint-proportional (coil and damper open together), and the hot-water cap is the
 * demand reset (mean zone shortfall, 4 K band) instead of the zones' own mean coil
 * request — saturation can no longer command boiler max. Off leaves every head
 * independent. */
static int g_zone_command = 0;
/* env.probe (diagnostic, never a scored run): 1 = the hot-water command is fixed at probe_water
 * (a = 0 -> 330 K through the contract map), 2 = reheat shut for every zone above its cooling
 * setpoint, 3 = both, applied on top of the policy's own actions. Batch-wide like the other env
 * flags: it prices the two moves the policy does not make. */
static int g_probe = 0;
static float g_probe_water = 0.0f;
static float g_res_scale = 0.5f, g_res_dscale = 1.5f;
static float g_guard_d0 = 1.0f, g_guard_d1 = 0.0f;   /* env.guard_d0 / env.guard_d1 (comfort guard; d1 <= 0: off) */
static int g_guard_hot = 0;                          /* env.guard_hot: 1 = the guard also acts above the band */
/* per group: canonical maps, base controller state and outputs */
static CanonMap* h_cmap;
static CanonMap* d_cmap;
static int cn_ok;
static double* d_pid;
static float *d_base, *d_ahuinfo;
static float* d_trust;            /* comfort guard: per env and native action, the residual's authority (NULL = off) */
static float* d_obsraw;           /* canonical mode: the oracle observation (g_obs points here) */
static float* g_obs_pol;          /* canonical mode: the policy's observation (the framework's buffer) */
static int* d_epcount;            /* per env: full episodes completed (eval records only the first) */
static double* d_epretd;          /* per env: running episode return in double */
static char g_snap_blob[512];      /* absolute blob path, recorded in every snapshot */
/* ---- several weathers of one building in one group ----
 * Blobs of the same building baked for different weather stations differ only in the sections
 * listed in sb_weather_sections (checked byte for byte at start-up, see sb_weather_compatible).
 * Such blobs form one group: one geometry, one solver pattern and ordering, one convection
 * producer; the per-weather values are laid out [w][...] and every env reads its own weather's
 * (envw).  Envs are weather-major unless env.interleave round-robins them. */
static int nw = 1;                       /* weathers in this group */
static int g_interleave = 0;             /* env.interleave: round-robin weather across env index */
static SbBlob* wblobs;                   /* nw open blobs, wblobs[0] is `blob` */
static int *h_envw, *d_envw;             /* weather of env e (d_envw has lu_ld entries, padding 0) */
typedef char SbPath[SB_PATH_MAX];
static SbPath* w_paths;                  /* absolute blob path per weather (snapshots) */
static long amb_len, wd_len, wdm1_len, rwm_len, rww_len, hour_len, hourm1_len, rwh_len, axt_len;   /* per-weather table strides */
static const char* const sb_weather_sections[] = {
    /* used, laid out per weather */
    "convection_coefficient", "lu_L_data", "lu_U_data", "weather_ambient", "occ_workday",
    "occ_workday_m1", "rew_month", "rew_workday", "occ_hour", "occ_hour_m1", "rew_hour", "aux_time",
    /* baked for the CPU tools / the oracle only; the GPU env never reads them */
    "A_system_data", "solref_x", "start_epoch", "weather_convection", NULL};
static int g_ep_len = 8640;        /* episode length shown in snapshots */

/* The dynamic shared-memory limit of a kernel is a per-kernel (not per-group) setting: only ever
 * raise it, so a later group with smaller needs cannot break an earlier group's launches. */
template <typename F>
static cudaError_t sb_raise_smem(F fn, size_t bytes) {
    static std::vector<std::pair<const void*, size_t>> done;
    for (auto& d : done)
        if (d.first == (const void*)fn) {
            if (d.second >= bytes) return cudaSuccess;
            d.second = bytes;
            return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
        }
    done.push_back({(const void*)fn, bytes});
    return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
}

/* ---- scenario groups ------------------------------------------------------
 * The state above is ONE scenario.  A batch of several scenarios keeps each scenario's copy
 * as a byte image of exactly the variables listed in sb_group_state.inc (generated by
 * tools/gen_group_state.py, checked by verify_all.sh) and loads it before touching that group:
 * kernel arguments are read at launch, so loading a group and then issuing its launches (also
 * while a CUDA graph is being captured) binds them to that group's buffers. */
static void sb_group_save(std::vector<unsigned char>& b) {
    size_t tot = 0;
#define SBG(x) tot += sizeof(x);
#include "sb_group_state.inc"
#undef SBG
    b.resize(tot);
    unsigned char* p = b.data();
#define SBG(x) memcpy(p, (const void*)&x, sizeof(x)); p += sizeof(x);
#include "sb_group_state.inc"
#undef SBG
}
static void sb_group_load(const std::vector<unsigned char>& b) {
    const unsigned char* p = b.data();
#define SBG(x) memcpy((void*)&x, p, sizeof(x)); p += sizeof(x);
#include "sb_group_state.inc"
#undef SBG
}
struct SbGroup {
    std::vector<unsigned char> st;
    int off, n;               /* first env of the group in the framework's batch, env count */
    char name[96];            /* scenario, for metric keys */
    cudaStream_t s;           /* the group's own stream: groups step concurrently (fork/join on the rollout stream) */
    cudaEvent_t done, ev_a, ev_b;
};
static cudaEvent_t g_fork, g_fork2, g_fork3;
/* Fused solve of all groups (batch-wide, not per group): one persistent launch per half-solve over
 * a merged level order of every group's rows (lu_solve_mg).  On when the batch has several groups
 * with the same envs-per-lane, when SB_FUSE=1 (off by default, see sb_mg_setup). */
static bool g_mg_on = false;
static bool g_mg_force = false;      /* the mgtest gate turns the fused solve on regardless of SB_FUSE */
static int g_mg_mw = 0;
static int g_mg_rpl = 1;
static int2 *d_mg_ordF, *d_mg_ordB;
static unsigned int n_mgF, n_mgB;
static unsigned int *d_mg_ep, *d_mg_tk;
static std::vector<SbGroup> g_groups;

static size_t sb_dtype_size(uint32_t t) {
    switch (t) { case SB_F64: case SB_I64: return 8; case SB_F32: case SB_I32: case SB_U32: return 4; default: return 1; }
}
/* 1 if blob b can join a group whose first blob is a: same sections, same shapes, and byte-identical
 * contents everywhere except the per-weather sections.  *why names the first difference. */
static int sb_weather_compatible(const SbBlob* a, const SbBlob* b, const char** why) {
    *why = "section count";
    if (a->n_sec != b->n_sec) return 0;
    for (int i = 0; i < a->n_sec; i++) {
        const SbSection* x = &a->sec[i];
        const SbSection* y = sb_find(b, x->name);
        *why = x->name;
        if (y == NULL || y->dtype != x->dtype || y->count != x->count) return 0;
        bool per_weather = false;
        for (int t = 0; sb_weather_sections[t] != NULL; t++)
            if (strcmp(x->name, sb_weather_sections[t]) == 0) per_weather = true;
        if (per_weather) continue;
        if (memcmp(sb_data(a, x), sb_data(b, y), x->count * sb_dtype_size(x->dtype)) != 0) return 0;
    }
    *why = NULL;
    return 1;
}

/* upload a baked section to the device, failing loudly if it is absent */
#define UPT(dst, getter, name, type)                                            \
    do {                                                                        \
        const type* sb_p_ = getter(&blob, name);                                \
        size_t sb_c_ = sb_find(&blob, name)->count;                             \
        type* sb_d_ = NULL;                                                     \
        if (!sb_p_) {                                                           \
            fprintf(stderr, "smart_buildings: blob section %s missing\n", name); \
            return 1;                                                           \
        }                                                                       \
        CU(cudaMalloc((void**)&sb_d_, sizeof(type) * sb_c_));                   \
        CU(cudaMemcpy(sb_d_, sb_p_, sizeof(type) * sb_c_,                       \
                      cudaMemcpyHostToDevice));                                 \
        (dst) = sb_d_;                                                          \
    } while (0)

/* the same section of every weather, concatenated [w][count]; *len = count */
#define UPTW(dst, getter, name, type, len)                                      \
    do {                                                                        \
        const SbSection* sb_s_ = sb_find(&blob, name);                          \
        if (!sb_s_ || !getter(&blob, name)) {                                   \
            fprintf(stderr, "smart_buildings: blob section %s missing\n", name); \
            return 1;                                                           \
        }                                                                       \
        const size_t sb_c_ = sb_s_->count;                                      \
        type* sb_d_ = NULL;                                                     \
        CU(cudaMalloc((void**)&sb_d_, sizeof(type) * sb_c_ * nw));              \
        for (int w_ = 0; w_ < nw; w_++)                                         \
            CU(cudaMemcpy(sb_d_ + sb_c_ * w_, getter(&wblobs[w_], name),        \
                          sizeof(type) * sb_c_, cudaMemcpyHostToDevice));       \
        (dst) = sb_d_;                                                          \
        (len) = (long)sb_c_;                                                    \
    } while (0)

/* --------------------------------------------------------------- helpers --- */
static int endsWith(const char* s, const char* suf) {
    size_t a = strlen(s), b = strlen(suf);
    return a >= b && strcmp(s + a - b, suf) == 0;
}

static int list_blobs(const char* dir, char (*out)[SB_PATH_MAX]) {
    DIR* d = opendir(dir);
    if (d == NULL) return 0;
    int nf = 0;
    struct dirent* e;
    while ((e = readdir(d)) != NULL) {
        if (endsWith(e->d_name, ".sbb") && nf < SB_MAX_BLOBS) {
            snprintf(out[nf], SB_PATH_MAX, "%s/%s", dir, e->d_name);
            nf++;
        }
    }
    closedir(d);
    for (int i = 1; i < nf; i++) {          /* sorted, so the pick is stable */
        char tmp[SB_PATH_MAX];
        snprintf(tmp, SB_PATH_MAX, "%s", out[i]);
        int j = i - 1;
        while (j >= 0 && strcmp(out[j], tmp) > 0) {
            snprintf(out[j + 1], SB_PATH_MAX, "%s", out[j]);
            j--;
        }
        snprintf(out[j + 1], SB_PATH_MAX, "%s", tmp);
    }
    return nf;
}

/* ----------------------------------------------------------------- setup --- */
/* ------------------------------------------------------------ the solve --- */
static void sb_lu_launch(LuDev lu, int rpl, int ld, sb_wt* work, sb_wt* zout, cudaStream_t st) {
    lu_prep_k<<<1, 1, 0, st>>>(d_lu_ep, d_lu_tk, d_lu_fl, d_lu_fu, n);
    const int blocks = 164, tpb = 128;
    lu_launch_solve(rpl, lu, work, zout, ld, d_lu_fl, d_lu_fu, d_lu_tk, d_lu_ep, NULL, blocks, tpb, st);
}
/* resident blocks of the persistent solve (SB_LU_BLOCKS overrides; experiment knob) */
static int sb_lu_blocks(void) {
    static int b = -1;
    if (b < 0) b = getenv("SB_LU_BLOCKS") ? atoi(getenv("SB_LU_BLOCKS")) : 164;
    return b;
}
/* threads per block of the persistent solve (SB_LU_TPB overrides; experiment knob).  The solve
 * is latency-bound at a low warp occupancy (ncu on train_1 @256: long_scoreboard ~55%, 16% of
 * the warps active with 164 blocks of 128), so a wider block doubles the warps per SM at the
 * same block count.  Nothing in the kernel is block-scoped (lanes are threadIdx.x & 31). */
static int sb_lu_tpb(void) {
    static int t = -1;
    if (t < 0) t = getenv("SB_LU_TPB") ? atoi(getenv("SB_LU_TPB")) : 128;
    return t;
}
static int sb_lu_rpl_for(int ncol) { return std::max(1, std::min(16, (ncol + 31) / 32)); }   /* envs per lane: no padded lanes beyond the last 32-env row */

/* Upload the baked factors in the layout the solver walks (strict-lower CSR with
 * ascending columns, strict-upper CSR with DEscending columns, the diagonal of U) and
 * the ticket order (rows sorted by dependency level), and set the sensed response
 * S = sensed A^-1 applied from one batched solve of the `applied` columns. */
static int sb_build_nonsolved(void) {
    const int64_t* sol = sb_i64(&blob, "solved");
    std::vector<char> is(ncv, 0);
    for (int t = 0; t < n; t++) is[(size_t)sol[t]] = 1;
    std::vector<int> ns;
    for (int p = 0; p < ncv; p++) if (!is[p]) ns.push_back(p);
    n_nonsolved = (int)ns.size();
    {   /* tmean_k is only read when some AHU has no VAV area; areas are static, so decide once */
        const int* zvo = sb_i32(&blob, "vav_zone_offsets");
        const int* zvi = sb_i32(&blob, "vav_zone_idx");
        const int* vai = sb_i32(&blob, "vav_ahu_idx");
        const double* za = sb_f64(&blob, "zone_area");
        std::vector<double> atot(n_ahu, 0.0);
        for (int v = 0; v < n_vav; v++) {
            double area = 0.0;
            for (int t = zvo[v]; t < zvo[v + 1]; t++) area += za[zvi[t]];
            if (area > 0.0) atot[vai[v]] += area;
        }
        tmean_needed = 0;
        for (int a = 0; a < n_ahu; a++) if (!(atot[a] > 0.0)) tmean_needed = 1;
        if (getenv("SB_VERBOSE")) printf("tmean_k %s\n", tmean_needed ? "needed" : "skipped (every AHU has VAV area)");
    }
    {   /* cell -> zone map for the internal gains (zones' room cells must not overlap) */
        const int* pri = sb_i32(&blob, "phys_room_idx");
        const int* ro = sb_i32(&blob, "room_offsets");
        const int* rc = sb_i32(&blob, "room_cells");
        std::vector<int> cz(ncv, -1);
        for (int z = 0; z < n_zones; z++)
            for (int t = ro[pri[z]]; t < ro[pri[z] + 1]; t++) {
                const int cell = rc[2 * t] * nc + rc[2 * t + 1];
                if (cz[cell] >= 0) { fprintf(stderr, "smart_buildings: zones %d and %d share cell %d\n", cz[cell], z, cell); return 1; }
                cz[cell] = z;
            }
        CU(cudaMalloc(&d_cellzone, 4 * (size_t)ncv));
        CU(cudaMemcpy(d_cellzone, cz.data(), 4 * (size_t)ncv, cudaMemcpyHostToDevice));
    }
    CU(cudaMalloc(&d_nonsolved, 4 * (size_t)(n_nonsolved ? n_nonsolved : 1)));
    if (n_nonsolved) CU(cudaMemcpy(d_nonsolved, ns.data(), 4 * ns.size(), cudaMemcpyHostToDevice));
    return 0;
}
static int sb_lu_setup(void) {
    if (sb_build_nonsolved() != 0) return 1;
    const int *Lp = sb_i32(&blob, "lu_L_indptr"), *Li = sb_i32(&blob, "lu_L_indices");
    const int *Up = sb_i32(&blob, "lu_U_indptr"), *Ui = sb_i32(&blob, "lu_U_indices");
    const int *pr = sb_i32(&blob, "lu_perm_r"), *pc = sb_i32(&blob, "lu_perm_c");
    if (!Lp || !Li || !sb_f64(&blob, "lu_L_data") || !Up || !Ui || !sb_f64(&blob, "lu_U_data") || !pr || !pc) {
        fprintf(stderr, "smart_buildings: blob has no baked LU factors\n");
        return 1;
    }
    std::vector<int> lrp(n + 1, 0), urp(n + 1, 0);
    for (int j = 0; j < n; j++) {
        for (int p = Lp[j]; p < Lp[j + 1]; p++) if (Li[p] != j) lrp[Li[p] + 1]++;
        for (int p = Up[j]; p < Up[j + 1]; p++) if (Ui[p] != j) urp[Ui[p] + 1]++;
    }
    for (int i = 0; i < n; i++) { lrp[i + 1] += lrp[i]; urp[i + 1] += urp[i]; }
    const long nL = lrp[n], nU = urp[n];
    /* the pattern (and so the solver layout) is shared by every weather of the group; the values
     * are laid out as nw consecutive copies */
    std::vector<int> lci(nL), uci(nU);
    std::vector<double> lcx((size_t)nw * nL), ucx((size_t)nw * nU), ud((size_t)nw * n, 0.0), uinv((size_t)nw * n, 0.0);
    for (int w = 0; w < nw; w++) {
        const double* Lx = sb_f64(&wblobs[w], "lu_L_data");
        const double* Ux = sb_f64(&wblobs[w], "lu_U_data");
        std::vector<int> lf(lrp.begin(), lrp.end() - 1), uf(urp.begin(), urp.end() - 1);
        double *lx = &lcx[(size_t)w * nL], *ux = &ucx[(size_t)w * nU], *dw = &ud[(size_t)w * n];
        for (int j = 0; j < n; j++)
            for (int p = Lp[j]; p < Lp[j + 1]; p++)
                if (Li[p] != j) { int i = Li[p]; lci[lf[i]] = j; lx[lf[i]] = Lx[p]; lf[i]++; }
        for (int j = n - 1; j >= 0; j--)
            for (int p = Up[j]; p < Up[j + 1]; p++) {
                int i = Ui[p];
                if (i == j) { dw[j] = Ux[p]; continue; }
                uci[uf[i]] = j; ux[uf[i]] = Ux[p]; uf[i]++;
            }
        for (int i = 0; i < n; i++) uinv[(size_t)w * n + i] = 1.0 / dw[i];
    }
    /* ticket order: rows by dependency level, stable in index */
    std::vector<int> ordL(n), ordU(n);
    {
        std::vector<int> lv(n, 0);
        int nl = 0;
        for (int i = 0; i < n; i++) {
            int m = 0;
            for (int p = lrp[i]; p < lrp[i + 1]; p++) m = std::max(m, lv[lci[p]] + 1);
            lv[i] = m; nl = std::max(nl, m + 1);
        }
        std::vector<int> cnt(nl + 1, 0);
        for (int i = 0; i < n; i++) cnt[lv[i] + 1]++;
        for (int l = 0; l < nl; l++) cnt[l + 1] += cnt[l];
        for (int i = 0; i < n; i++) ordL[cnt[lv[i]]++] = i;
        std::vector<int> lu2(n, 0);
        int nl2 = 0;
        for (int i = n - 1; i >= 0; i--) {
            int m = 0;
            for (int p = urp[i]; p < urp[i + 1]; p++) m = std::max(m, lu2[uci[p]] + 1);
            lu2[i] = m; nl2 = std::max(nl2, m + 1);
        }
        std::vector<int> cnt2(nl2 + 1, 0);
        for (int i = 0; i < n; i++) cnt2[lu2[i] + 1]++;
        for (int l = 0; l < nl2; l++) cnt2[l + 1] += cnt2[l];
        for (int i = n - 1; i >= 0; i--) ordU[cnt2[lu2[i]]++] = i;
        if (getenv("SB_VERBOSE")) printf("lu: %d forward levels, %d backward levels\n", nl, nl2);
        /* dependency level of every group-local ticket (fused multi-group solve order) */
        h_tlev = (int*)malloc(sizeof(int) * 2 * (size_t)n);
        for (int t = 0; t < n; t++) { h_tlev[t] = lv[ordL[t]]; h_tlev[n + t] = lu2[ordU[t]]; }
    }
    auto up = [&](void** d, const void* h, size_t bytes) {
        cudaMalloc(d, bytes ? bytes : 4);
        if (bytes) cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice);
    };
    int *d_lrp, *d_lci, *d_urp, *d_uci, *d_ordL, *d_ordU;
    double *d_lcx, *d_ucx, *d_ud, *d_uinv;
    up((void**)&d_lrp, lrp.data(), 4 * (size_t)(n + 1));
    up((void**)&d_lci, lci.data(), 4 * lci.size());
    up((void**)&d_lcx, lcx.data(), 8 * lcx.size());
    up((void**)&d_urp, urp.data(), 4 * (size_t)(n + 1));
    up((void**)&d_uci, uci.data(), 4 * uci.size());
    up((void**)&d_ucx, ucx.data(), 8 * ucx.size());
    up((void**)&d_ud, ud.data(), 8 * ud.size());
    up((void**)&d_uinv, uinv.data(), 8 * uinv.size());
    up((void**)&d_ordL, ordL.data(), 4 * (size_t)n);
    up((void**)&d_ordU, ordU.data(), 4 * (size_t)n);
    up((void**)&d_rowpos, pr, 4 * (size_t)n);      /* b index t -> solver row perm_r[t] */
    up((void**)&d_pc, pc, 4 * (size_t)n);
    g_lu = {n, d_lrp, d_lci, d_lcx, d_urp, d_uci, d_ucx, d_ud, d_uinv, d_ordL, d_ordU};
    g_lu.nw = nw; g_lu.nnzL = nL; g_lu.nnzU = nU;
    CU(cudaMalloc(&d_lu_fl, 4 * (size_t)n));
    CU(cudaMalloc(&d_lu_fu, 4 * (size_t)n));
    CU(cudaMemset(d_lu_fl, 0, 4 * (size_t)n));
    CU(cudaMemset(d_lu_fu, 0, 4 * (size_t)n));
    CU(cudaMalloc(&d_lu_tk, 8));   /* [0] forward / whole solve, [1] backward */
    CU(cudaMalloc(&d_lu_ep, 4));
    CU(cudaMemset(d_lu_ep, 0, 4));

    /* Per weather, from one batched solve of the `applied` columns with that weather's factors:
     *   S = sensed A^-1 applied (k x k, the air side's matrix), via Y = A^-1 applied;
     *   Z = L^-1 P_r applied  (the forward half, still in wk), kept sparse;
     *   R = sensed P_c U^-1   (host triangular solve with U^T), kept sparse.
     * Z and R of all weathers share one pattern (the union), values [w][nnz]. */
    CU(cudaMalloc(&d_apc_keep, 8 * (size_t)n * k));
    CU(cudaMalloc(&d_yrk_keep, 8 * (size_t)n * k));
    CU(cudaMalloc(&d_sr, 8 * (size_t)nw * k * k));
    CU(cudaMemset(d_apc_keep, 0, 8 * (size_t)n * k));
    applied_cols_k<<<k, 128, 0, stream>>>(sd_ap, sd_ai, sd_apx, d_apc_keep, k, n);
    const int rk = sb_lu_rpl_for(k), ldk = 32 * rk;
    sb_wt *wk, *zk;
    CU(cudaMalloc(&wk, sizeof(sb_wt) * (size_t)n * ldk));
    CU(cudaMalloc(&zk, sizeof(sb_wt) * (size_t)n * ldk));
    /* Structural patterns (from the L / U / applied / sensed patterns only, so identical for every
     * weather of the building): Zs[i] = columns j reachable in L from applied column j's rows,
     * Rs[r] = sensors i whose U^-T solve reaches solver row r.  Numerical zeros inside the
     * pattern are stored, so a group of several weathers sums exactly the same entries in exactly
     * the same order as each weather alone. */
    const int KW = (k + 63) / 64 > 0 ? (k + 63) / 64 : 1;
    std::vector<uint64_t> Zs((size_t)n * KW, 0), Rs((size_t)n * KW, 0);
    {
        const int *ap = sb_i32(&blob, "applied_indptr"), *ai = sb_i32(&blob, "applied_indices");
        for (int j = 0; j < k; j++)
            for (int p = ap[j]; p < ap[j + 1]; p++) Zs[(size_t)pr[ai[p]] * KW + j / 64] |= 1ull << (j % 64);
        for (int i = 0; i < n; i++)
            for (int p = lrp[i]; p < lrp[i + 1]; p++)
                for (int w = 0; w < KW; w++) Zs[(size_t)i * KW + w] |= Zs[(size_t)lci[p] * KW + w];
        const int *sp0 = sb_i32(&blob, "sensed_indptr"), *si0 = sb_i32(&blob, "sensed_indices");
        for (int i = 0; i < k; i++)
            for (int p = sp0[i]; p < sp0[i + 1]; p++) Rs[(size_t)pc[si0[p]] * KW + i / 64] |= 1ull << (i % 64);
        for (int r = 0; r < n; r++) {
            bool any = false;
            for (int w = 0; w < KW; w++) any |= Rs[(size_t)r * KW + w] != 0;
            if (!any) continue;
            for (int p = urp[r]; p < urp[r + 1]; p++)
                for (int w = 0; w < KW; w++) Rs[(size_t)uci[p] * KW + w] |= Rs[(size_t)r * KW + w];
        }
    }
    auto inZ = [&](int i, int j) { return (Zs[(size_t)i * KW + j / 64] >> (j % 64)) & 1ull; };
    auto inR = [&](int r, int i) { return (Rs[(size_t)r * KW + i / 64] >> (i % 64)) & 1ull; };
    struct Trip { int r, c; double v; };
    std::vector<std::vector<Trip>> zt(nw), rt(nw);
    const int *sp = sb_i32(&blob, "sensed_indptr"), *si = sb_i32(&blob, "sensed_indices");
    const double* sx = sb_f64(&blob, "sensed_data");
    /* wk is sb_wt on the device (fp64 by default, fp32 under SB_SOLVE_F32): read it back in its
     * own type and widen in the host loop. */
    std::vector<sb_wt> hw_raw((size_t)n * ldk);
    for (int w = 0; w < nw; w++) {
        LuDev v = g_lu;                 /* this weather's values, single-weather view */
        v.nw = 1;
        v.lcx = d_lcx + (size_t)w * nL; v.ucx = d_ucx + (size_t)w * nU;
        v.udiag = d_ud + (size_t)w * n; v.uinv = d_uinv + (size_t)w * n;
        lu_perm_in_k<<<(unsigned)(((size_t)n * ldk + 255) / 256), 256, 0, stream>>>(
            d_apc_keep, d_rowpos, wk, n, k, ldk);
        sb_lu_launch(v, rk, ldk, wk, zk, stream);
        lu_perm_out_k<<<(unsigned)(((size_t)n * k + 255) / 256), 256, 0, stream>>>(
            zk, d_pc, d_yrk_keep, n, k, ldk);
        sensed_resp_k<<<(k * k + 127) / 128, 128, 0, stream>>>(sd_sp, sd_si, sd_spx, d_yrk_keep,
                                                              d_sr + (size_t)w * k * k, k, n);
        CU(cudaStreamSynchronize(stream));
        CU(cudaMemcpy(hw_raw.data(), wk, sizeof(sb_wt) * hw_raw.size(), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) {
                const double x = (double)hw_raw[(size_t)i * ldk + j];
                if (inZ(i, j)) zt[w].push_back({i, j, x});
                else if (x != 0.0) { fprintf(stderr, "smart_buildings: Z entry outside its structural pattern\n"); return 1; }
            }
        if (k > 0) {   /* R row i = U^-T s'_i, s'_i[pc[t]] = sensed[i][t]: push form over the rows of U */
            const double* ux = &ucx[(size_t)w * nU];
            const double* dw = &ud[(size_t)w * n];
            std::vector<double> X((size_t)n * k, 0.0);
            for (int i = 0; i < k; i++)
                for (int p = sp[i]; p < sp[i + 1]; p++) X[(size_t)pc[si[p]] * k + i] = sx[p];
            for (int r = 0; r < n; r++) {
                double* xr = &X[(size_t)r * k];
                bool any = false;
                for (int i = 0; i < k; i++) if (xr[i] != 0.0) { xr[i] = xr[i] / dw[r]; any = true; }
                if (!any) continue;
                for (int p = urp[r]; p < urp[r + 1]; p++) {
                    double* xc = &X[(size_t)uci[p] * k];
                    const double u = ux[p];
                    for (int i = 0; i < k; i++) xc[i] -= u * xr[i];
                }
            }
            for (int r = 0; r < n; r++)
                for (int i = 0; i < k; i++) {
                    const double x = X[(size_t)r * k + i];
                    if (inR(r, i)) rt[w].push_back({i, r, x});
                    else if (x != 0.0) { fprintf(stderr, "smart_buildings: R entry outside its structural pattern\n"); return 1; }
                }
        }
    }
    CU(cudaFree(wk));
    CU(cudaFree(zk));
    /* the dense n x k Y is only needed for S above */
    CU(cudaFree(d_apc_keep)); d_apc_keep = NULL;
    CU(cudaFree(d_yrk_keep)); d_yrk_keep = NULL;
    /* union patterns: CSR over `nrow` rows, columns ascending; values [w][nnz] (0 where a weather
     * has no entry, which adds an exact +0 term) */
    auto build = [&](std::vector<std::vector<Trip>>& tr, int nrow, int ncol, std::vector<int>& rp,
                     std::vector<int>& ci, std::vector<double>& cx) {
        std::vector<std::vector<int>> cols(nrow);
        for (int w = 0; w < nw; w++) for (auto& t : tr[w]) cols[t.r].push_back(t.c);
        rp.assign(nrow + 1, 0);
        for (int r = 0; r < nrow; r++) {
            auto& c = cols[r];
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
            rp[r + 1] = rp[r] + (int)c.size();
        }
        ci.resize(rp[nrow]);
        for (int r = 0; r < nrow; r++) std::copy(cols[r].begin(), cols[r].end(), ci.begin() + rp[r]);
        const size_t nz = ci.size();
        cx.assign((size_t)nw * (nz ? nz : 1), 0.0);
        for (int w = 0; w < nw; w++)
            for (auto& t : tr[w]) {
                const int* b0 = ci.data() + rp[t.r];
                const int* e0 = ci.data() + rp[t.r + 1];
                cx[(size_t)w * nz + (std::lower_bound(b0, e0, t.c) - ci.data())] = t.v;
            }
        (void)ncol;
    };
    {
        std::vector<int> zrp, zci; std::vector<double> zcx;
        build(zt, n, k, zrp, zci, zcx);
        int* d_zrp; int* d_zci; double* d_zcx;
        up((void**)&d_zrp, zrp.data(), 4 * zrp.size());
        up((void**)&d_zci, zci.data(), 4 * zci.size());
        up((void**)&d_zcx, zcx.data(), 8 * zcx.size());
        g_lu.zrp = d_zrp; g_lu.zci = d_zci; g_lu.zcx = d_zcx;
        nnz_z = (long)zci.size();
        g_lu.nnzZ = nnz_z;
    }
    if (k > 0) {
        std::vector<int> rrp, rrc; std::vector<double> rrx;
        build(rt, k, n, rrp, rrc, rrx);
        up((void**)&d_rrp, rrp.data(), 4 * rrp.size());
        up((void**)&d_rrc, rrc.data(), 4 * rrc.size());
        up((void**)&d_rrx, rrx.data(), 8 * rrx.size());
        nnz_r = (long)rrc.size();
        /* segments: [start, end) pairs, SB_RSEG entries each, never spanning two sensors */
        std::vector<int> seg, rowseg(k + 1, 0);
        for (int i = 0; i < k; i++) {
            rowseg[i] = (int)(seg.size() / 2);
            for (int p0 = rrp[i]; p0 < rrp[i + 1]; p0 += SB_RSEG) {
                seg.push_back(p0);
                seg.push_back(std::min(p0 + SB_RSEG, rrp[i + 1]));
            }
        }
        n_rseg = (int)(seg.size() / 2);
        rowseg[k] = n_rseg;
        up((void**)&d_rrow_seg, rowseg.data(), 4 * rowseg.size());
        up((void**)&d_rseg_p, seg.data(), 4 * seg.size());
    }
    if (getenv("SB_VERBOSE"))
        printf("air side: Z = L^-1 applied %ld nnz, R = sensed U^-1 %ld nnz (dense Y was %ld), %d weather(s)\n",
               nnz_z, nnz_r, (long)n * k, nw);
    return 0;
}

/* Episode length: the scenario's steps_per_episode (obs_meta[3] = 8640 for the shipped set;
 * the per-step tables carry one extra row). */
static int sb_blob_ep_len(void) {
    const SbSection* m = sb_find(&blob, "obs_meta");
    if (m != NULL && m->count > 3) {
        int v = (int)sb_f64(&blob, "obs_meta")[3];
        if (v > 0) return v;
    }
    const SbSection* o = sb_find(&blob, "occ_hour");
    return (o != NULL && o->count > 1) ? (int)o->count - 1 : 8640;
}
/* Canonical maps (sb_gpu_base.inc), built from the blob by NAME (the observation field names and the action
 * owner tables), i.e. from the same information the submission reads from building_meta_data.  Always built;
 * cn_ok = 0 when the building exceeds the canonical slots (more than CN_K AHUs or CN_ZPA zones on one AHU). */
static int sb_canon_setup(void) {
    std::vector<std::string> names;
    {
        uint64_t len = 0;
        const char* fs = sb_str(&blob, "field_names", &len);
        if (fs == NULL) { fprintf(stderr, "smart_buildings: blob has no field_names\n"); return 1; }
        std::string cur;
        for (uint64_t i = 0; i < len; i++) {
            if (fs[i] == '\n') { names.push_back(cur); cur.clear(); } else cur += fs[i];
        }
        if (!cur.empty()) names.push_back(cur);
    }
    auto find = [&](const std::string& nm) -> int {
        for (size_t i = 0; i < names.size(); i++) if (names[i] == nm) return (int)i;
        return -1;
    };
    CanonMap* cm = (CanonMap*)calloc(1, sizeof(CanonMap));
    h_cmap = cm;
    cn_ok = 1;
    for (int i = 0; i < CN_ZS; i++) { cm->zT[i] = cm->zH[i] = cm->zC[i] = cm->zD[i] = cm->zR[i] = cm->zAd[i] = cm->zAr[i] = cm->zV[i] = -1; }
    for (int k = 0; k < CN_K; k++) {
        cm->aOAT[k] = cm->aSP[k] = cm->aSF[k] = cm->aOF[k] = cm->aSAT[k] = cm->aRC[k] = cm->aCR[k] = cm->aRUNo[k] = -1;
        cm->aT[k] = cm->aP[k] = cm->aRec[k] = cm->aRun[k] = -1;
    }
    cm->hW = cm->hDPa = cm->hRun = -1;
    cm->K = n_ahu;
    if (n_ahu > CN_K) cn_ok = 0;
    const uint8_t* fk = sb_u8(&blob, "f_dev_kind");
    const int* fi = sb_i32(&blob, "f_dev_idx");
    const uint8_t* fo = sb_u8(&blob, "f_obs_id");
    const double* fmn = sb_f64(&blob, "f_norm_mean");
    const double* fvr = sb_f64(&blob, "f_norm_var");
    const int* vai = sb_i32(&blob, "vav_ahu_idx");
    const int* vzo = sb_i32(&blob, "vav_zone_offsets");
    const int* vzi = sb_i32(&blob, "vav_zone_idx");
    std::vector<int> vav_slot(n_vav, -1);
    /* zones: VAV temperature fields in observation order */
    cm->nz_total = 0;
    for (int i = 0; i < n_dev_; i++) {
        if (fk[i] != 1 || fo[i] != 5) continue;
        const int v = fi[i], a = vai[v];
        if (a < 0 || a >= CN_K) { cn_ok = 0; continue; }
        if (cm->zc[a] >= CN_ZPA) { cn_ok = 0; continue; }
        const int sl = CN_ZPA * a + cm->zc[a]++;
        vav_slot[v] = sl;
        /* zV indexes avg_temp and occupancy, which are per zone, not per VAV. A VAV serving
         * several zones has no single zone: leave the slot out of the per-zone signals. */
        cm->zV[sl] = (vzo != NULL && vzi != NULL && vzo[v + 1] - vzo[v] == 1) ? vzi[vzo[v]] : -1;
        cm->nz_total++;
        const std::string pre = names[i].substr(0, names[i].size() - strlen("zone_air_temperature_sensor"));
        cm->zT[sl] = i;
        cm->zH[sl] = find(pre + "zone_air_heating_temperature_setpoint_sensor");
        cm->zC[sl] = find(pre + "zone_air_cooling_temperature_setpoint_sensor");
        cm->zD[sl] = find(pre + "supply_air_damper_percentage_command");
        cm->zR[sl] = find(pre + "reheat_valve_percentage_command");
    }
    for (int k = 0; k < n_ahu && k < CN_K; k++) {
        const std::string pre = "ahs_ahu_" + std::to_string(k + 1) + "_";
        cm->aOAT[k] = find(pre + "outside_air_temperature_sensor");
        cm->aSP[k] = find(pre + "supply_air_static_pressure_sensor");
        cm->aSF[k] = find(pre + "supply_air_flowrate_sensor");
        cm->aOF[k] = find(pre + "outside_air_flowrate_sensor");
        cm->aSAT[k] = find(pre + "supply_air_temperature_setpoint");
        cm->aRC[k] = find(pre + "recirculation_percentage_command");
        cm->aCR[k] = find(pre + "cooling_request_count");
        cm->aRUNo[k] = find(pre + "supervisor_run_command");
    }
    cm->hWT = find("hws_supply_water_temperature_sensor");
    cm->hWS = find("hws_supply_water_temperature_setpoint");
    cm->hDP = find("hws_differential_pressure_sensor");
    cm->hDS = find("hws_differential_pressure_setpoint");
    cm->hRS = find("hws_run_status");
    cm->hHR = find("hws_heating_request_count");
    for (int h = 0; h < 4; h++) {
        char b1[32];
        snprintf(b1, sizeof b1, "hod_sin_%03d", h); cm->g_hs[h] = find(b1);
        snprintf(b1, sizeof b1, "hod_cos_%03d", h); cm->g_hc[h] = find(b1);
        snprintf(b1, sizeof b1, "dow_sin_%03d", h); cm->g_ds[h] = find(b1);
        snprintf(b1, sizeof b1, "dow_cos_%03d", h); cm->g_dc[h] = find(b1);
    }
    cm->g_cn = find("comfort_mode_now");
    cm->g_cs = find("comfort_mode_soon");
    cm->g_oc = find("num_occupants");
    if (cm->g_hs[0] < 0 || cm->g_hc[0] < 0 || cm->g_ds[0] < 0 || cm->g_dc[0] < 0) cn_ok = 0;
    /* native value of each observation entry: device fields are z-scored by the blob's constants */
    for (int i = 0; i < SB_OBS; i++) { cm->fmean[i] = 0.0f; cm->fstd[i] = 1.0f; }
    for (int i = 0; i < n_dev_; i++)
        if (fvr[i] > 0.0) { cm->fmean[i] = (float)fmn[i]; cm->fstd[i] = (float)sqrt(fvr[i]); }
    if (cm->zT[0] >= 0 || cm->nz_total > 0) {
        int t0 = -1;
        for (int i = 0; i < CN_ZS && t0 < 0; i++) if (cm->zT[i] >= 0) t0 = cm->zT[i];
        cm->tmean = cm->fmean[t0]; cm->tstd = cm->fstd[t0];
    }
    const double* sc = sb_f64(&blob, "sched_scalars");
    if (sc == NULL) { cn_ok = 0; } else {
        cm->morning = (int)sc[0]; cm->evening = (int)sc[1];
        cm->comfort_lo = sc[2]; cm->comfort_hi = sc[3]; cm->eco_lo = sc[4]; cm->eco_hi = sc[5];
        const SbSection* hol = sb_find(&blob, "sched_holidays");
        if (hol != NULL && hol->count > 0) {
            const int64_t* h = sb_i64(&blob, "sched_holidays");
            cm->n_holiday = (int)hol->count;
            if (cm->n_holiday > 32) cm->n_holiday = 32;
            for (int i = 0; i < cm->n_holiday; i++) cm->holiday[i] = (int)h[i];
        }
        time_t sec = (time_t)sb_f64_1(&blob, "start_epoch", 0);
        struct tm civil{};
        gmtime_r(&sec, &civil);
        int doy = civil.tm_yday + 1;
        int diff = cm->morning - civil.tm_hour;
        if (diff > 12) diff -= 24;
        if (diff < -12) diff += 24;
        if (civil.tm_hour + diff < 0) doy -= 1;
        else if (civil.tm_hour + diff >= 24) doy += 1;
        if (doy < 1) doy = 1;
        cm->start_doy = doy;
    }
    /* actions */
    const uint8_t* ok_ = sb_u8(&blob, "act_owner_kind");
    const int* oi = sb_i32(&blob, "act_owner_index");
    const uint8_t* oa = sb_u8(&blob, "act_actuator");
    for (int a = 0; a < SB_NATIVE_MAX; a++) cm->slot_of_action[a] = -1;
    for (int a = 0; a < n_action; a++) {
        if (ok_[a] == 1) {
            const int sl = vav_slot[oi[a]];
            if (sl < 0) continue;
            if (oa[a] == 0) { cm->zAd[sl] = a; cm->slot_of_action[a] = 2 * sl; }
            else { cm->zAr[sl] = a; cm->slot_of_action[a] = 2 * sl + 1; }
        } else if (ok_[a] == 0) {
            const int k = oi[a];
            if (k < 0 || k >= CN_K) continue;
            if (oa[a] == 0) cm->aT[k] = a; else if (oa[a] == 1) cm->aP[k] = a; else if (oa[a] == 2) cm->aRec[k] = a; else cm->aRun[k] = a;
            cm->slot_of_action[a] = CN_AA + 4 * k + oa[a];
        } else {
            if (oa[a] == 0) cm->hW = a; else if (oa[a] == 1) cm->hDPa = a; else cm->hRun = a;
            cm->slot_of_action[a] = CN_AH + oa[a];
        }
    }
    CU(cudaMalloc(&d_cmap, sizeof(CanonMap)));
    CU(cudaMemcpy(d_cmap, cm, sizeof(CanonMap), cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_pid, 8 * (size_t)nenv * 2 * CN_K));
    CU(cudaMemset(d_pid, 0, 8 * (size_t)nenv * 2 * CN_K));
    CU(cudaMalloc(&d_base, 4 * (size_t)nenv * (n_action ? n_action : 1)));
    CU(cudaMemset(d_base, 0, 4 * (size_t)nenv * (n_action ? n_action : 1)));
    CU(cudaMalloc(&d_ahuinfo, 4 * (size_t)nenv * 2 * CN_K));
    CU(cudaMemset(d_ahuinfo, 0, 4 * (size_t)nenv * 2 * CN_K));
    d_trust = NULL;
    if (g_guard_d1 > 0.0f) {
        CU(cudaMalloc(&d_trust, 4 * (size_t)nenv * (n_action ? n_action : 1)));
        CU(cudaMemset(d_trust, 0, 4 * (size_t)nenv * (n_action ? n_action : 1)));
    }
    CU(cudaMalloc(&d_epcount, 4 * (size_t)nenv));
    CU(cudaMemset(d_epcount, 0, 4 * (size_t)nenv));
    CU(cudaMalloc(&d_epretd, 8 * (size_t)nenv));
    CU(cudaMemset(d_epretd, 0, 8 * (size_t)nenv));
    if (g_canon) {
        if (!cn_ok) { fprintf(stderr, "smart_buildings: building exceeds the canonical slots (%d AHUs, max %d; <= %d zones per AHU)\n", n_ahu, CN_K, CN_ZPA); return 1; }
        CU(cudaMalloc(&d_obsraw, 4 * (size_t)nenv * SB_OBS));
        CU(cudaMemset(d_obsraw, 0, 4 * (size_t)nenv * SB_OBS));
#ifndef SB_TEST
        if (g_zone_command) {
            CU(cudaMalloc(&d_cmdact, 4 * (size_t)nenv * CN_ACT));
            CU(cudaMemset(d_cmdact, 0, 4 * (size_t)nenv * CN_ACT));
        }
        if (g_probe) {
            CU(cudaMalloc(&d_probeact, 4 * (size_t)nenv * CN_ACT));
            CU(cudaMemset(d_probeact, 0, 4 * (size_t)nenv * CN_ACT));
        }
#endif
#ifdef SB_ENTITY
        CU(cudaMalloc(&d_canon_obs, 4 * (size_t)nenv * SB_OBS));
        CU(cudaMemset(d_canon_obs, 0, 4 * (size_t)nenv * SB_OBS));
        CU(cudaMalloc(&d_entact, 4 * (size_t)nenv * CN_ACT));
        CU(cudaMemset(d_entact, 0, 4 * (size_t)nenv * CN_ACT));
        CU(cudaMalloc(&d_rew_learn, 4 * (size_t)nenv));
        CU(cudaMemset(d_rew_learn, 0, 4 * (size_t)nenv));
        CU(cudaMalloc(&d_term_learn, 4 * (size_t)nenv));
        CU(cudaMemset(d_term_learn, 0, 4 * (size_t)nenv));
#endif
    }
    if (getenv("SB_VERBOSE")) {
        printf("canonical: ok=%d K=%d zones=%d (per AHU:", cn_ok, cm->K, cm->nz_total);
        for (int k = 0; k < cm->K && k < CN_K; k++) printf(" %d", cm->zc[k]);
        printf(")\n");
    }
    return 0;
}

/* canonical mode, after the observation of a step / reset is complete: the base controller's proposal for the
 * next step (and its state update), then the policy's observation */
static void sb_canon_tail(void) {
    if (!g_canon) return;
#ifndef SB_SHARED_POLICY
    /* The shared actor reads sensors only. Do not run pid_water or write its proposal. */
    base_pidw_k<<<(nenv + 63) / 64, 64, 0, stream>>>(d_cmap, g_obs, nenv, n_action, d_pid, d_base, d_ahuinfo);
#endif
#ifdef SB_ENTITY
    canon_obs_k<<<(nenv * CN_OBS + 255) / 256, 256, 0, stream>>>(d_cmap, g_obs, d_base, d_ahuinfo, d_pid, nenv,
                                                               n_action, d_canon_obs);
    entity_pack_k<<<(nenv * CN_ZS + 127) / 128, 128, 0, stream>>>(d_canon_obs, g_obs_pol, nenv);
#else
    canon_obs_k<<<(nenv * CN_OBS + 255) / 256, 256, 0, stream>>>(d_cmap, g_obs, d_base, d_ahuinfo, d_pid, nenv,
                                                               n_action, g_obs_pol);
#endif
    if (d_trust != NULL)
        guard_k<<<(nenv + 63) / 64, 64, 0, stream>>>(d_cmap, g_obs, nenv, n_action, g_guard_d0, g_guard_d1, g_guard_hot, d_trust);
}

/* ids[w] repeated cnt[w] times. Round-robin when asked, otherwise weather-major. */
static void sb_rr_ids(std::vector<int>& dst, const std::vector<int>& id, const std::vector<int>& cnt,
                      int round_robin) {
    int n = 0;
    for (int c : cnt) n += c;
    dst.clear();
    dst.reserve(n > 0 ? n : 1);
    if (!round_robin || cnt.size() < 2) {
        for (size_t w = 0; w < cnt.size(); w++)
            for (int c = 0; c < cnt[w]; c++) dst.push_back(id[w]);
        return;
    }
    std::vector<int> left = cnt;
    size_t w = 0;
    int spins = 0;
    while ((int)dst.size() < n) {
        if (left[w] > 0) { dst.push_back(id[w]); left[w]--; }
        w = (w + 1) % cnt.size();
        if (++spins > n * (int)cnt.size() + 8) {
            fprintf(stderr, "smart_buildings: weather interleave did not fill\n");
            abort();
        }
    }
}
static int sb_gpu_setup_w(const char* const* paths, int nwx, const int* wcnt);
static int sb_gpu_setup(const char* path) { return sb_gpu_setup_w(&path, 1, NULL); }
/* Set up one group: nwx weathers of one building (paths[0..nwx-1]), cnt[w] envs of weather w
 * (wcnt == NULL: one weather, nenv envs).  nenv must already hold the group's total. */
static int sb_gpu_setup_w(const char* const* paths, int nwx, const int* wcnt) {
    nw = nwx;
    wblobs = (SbBlob*)calloc(nw, sizeof(SbBlob));
    w_paths = (SbPath*)calloc(nw, sizeof(SbPath));
    for (int w = 0; w < nw; w++) {
        char rp_[4096];
        snprintf(w_paths[w], SB_PATH_MAX, "%s", realpath(paths[w], rp_) != NULL ? rp_ : paths[w]);
        if (sb_open(&wblobs[w], paths[w]) != 0) {
            fprintf(stderr, "smart_buildings: cannot open blob %s\n", paths[w]);
            return 1;
        }
        const char* why = NULL;
        if (w > 0 && !sb_weather_compatible(&wblobs[0], &wblobs[w], &why)) {
            fprintf(stderr, "smart_buildings: %s cannot share a group with %s (section %s differs)\n",
                    paths[w], paths[0], why);
            return 1;
        }
    }
    blob = wblobs[0];
    snprintf(g_snap_blob, sizeof(g_snap_blob), "%s", w_paths[0]);
    h_envw = (int*)calloc(nenv, sizeof(int));
    {
        std::vector<int> id(nw), cnt(nw), order;
        for (int w = 0; w < nw; w++) {
            id[w] = w;
            cnt[w] = wcnt ? wcnt[w] : nenv;
        }
        sb_rr_ids(order, id, cnt, g_interleave && nw > 1);
        if ((int)order.size() != nenv) {
            fprintf(stderr, "smart_buildings: weather env counts do not add up\n");
            return 1;
        }
        for (int e = 0; e < nenv; e++) h_envw[e] = order[e];
    }
    /* solver columns: envs per lane and the padded column count, also the stride of the field */
    lu_rpl = sb_lu_rpl_for(nenv);
    lu_ld = 32 * lu_rpl;
    const int64_t* gr = sb_i64(&blob, "grid");
    nr = (int)gr[0]; nc = (int)gr[1]; ncv = nr * nc;
    n = (int)sb_find(&blob, "solved")->count;
    k = (int)sb_find(&blob, "coupled_vav_idx")->count;
    n_ahu = (int)(sb_find(&blob, "ahu_const")->count / 9);
    n_vav = (int)(sb_find(&blob, "vav_zone_offsets")->count - 1);
    n_zones = (int)sb_find(&blob, "zone_area")->count;
    n_action = (int)sb_find(&blob, "act_owner_kind")->count;
    n_rooms_c = (int)sb_find(&blob, "conv_room_offsets")->count - 1;
    conv_p = sb_f64(&blob, "conv_p")[0];
    conv_dist = (int)sb_f64(&blob, "conv_distance")[0];
    conv_words = 6 * ncv;
    n_dev_ = (int)sb_i64_1(&blob, "n_dev", 0);
    n_aux_ = (int)sb_i64_1(&blob, "n_aux", 0);
    obs_len_ = n_dev_ + n_aux_;
    gas_carb_ = sb_f64_1(&blob, "gas_carbon_rate", 0.0);
    pump_eff_ = sb_f64(&blob, "pump_const") ? sb_f64(&blob, "pump_const")[1] : 1.0;
    {
        const uint8_t* isd = sb_u8(&blob, "act_is_discrete");
        int nc_ = 0, nd_ = 0;
        for (int i = 0; i < n_action; i++) { if (isd[i]) nd_++; else nc_++; }
        if (nc_ > SB_CONT || nd_ > SB_DISC) {
            fprintf(stderr, "smart_buildings: scenario needs %d continuous / %d discrete actions, the env pads to %d / %d (SB_CONT / SB_DISC)\n", nc_, nd_, SB_CONT, SB_DISC);
            return 1;
        }
    }
    if (obs_len_ > SB_OBS || n_ahu > 256 || nenv < 1 || nenv > SB_NE) {
        fprintf(stderr, "smart_buildings: scenario out of range (obs %d)\n", obs_len_);
        return 1;
    }
    const double* op = sb_f64(&blob, "occ_params");
    nassign = (int)op[0];
    occ_ea = (int)op[1]; occ_la = (int)op[2]; occ_ed = (int)op[3]; occ_ld = (int)op[4];
    occ_step = op[5];
    p_arr = 1.0 / ((occ_la - occ_ea) * 3600.0 / occ_step / 2.0);
    p_dep = 1.0 / ((occ_ld - occ_ed) * 3600.0 / occ_step / 2.0);
    dt_sec_ = occ_step;

    /* ---- shared tables (one upload set per scenario) ---- */
    UPT(sh_zarea, sb_f64, "zone_area", double);
    UPT(sh_ahu, sb_f64, "ahu_const", double);
    UPT(sh_vav, sb_f64, "vav_const", double);
    UPT(sh_sc, sb_f64, "sched_comfort", double);
    UPT(sh_sh, sb_f64, "sched_heat_sp", double);
    UPT(sh_scl, sb_f64, "sched_cool_sp", double);
    UPTW(sh_amb, sb_f64, "weather_ambient", double, amb_len);
    UPT(sh_boil, sb_f64, "boiler_const", double);
    UPTW(sh_hour, sb_i64, "occ_hour", int64_t, hour_len);
    UPTW(sh_wd, sb_u8, "occ_workday", uint8_t, wd_len);
    UPT(sh_roo, sb_i32, "conv_room_offsets", int);
    UPT(sh_rcc, sb_i32, "conv_room_cells", int);
    UPT(sh_coo, sb_i32, "conv_cand_offsets", int);
    UPT(sh_cii, sb_i32, "conv_cand_idx", int);
    UPT(sh_zvo, sb_i32, "vav_zone_offsets", int);
    UPT(sh_zvi, sb_i32, "vav_zone_idx", int);
    UPT(sh_vai, sb_i32, "vav_ahu_idx", int);
    UPT(sh_cvi, sb_i32, "coupled_vav_idx", int);
    UPT(sh_pri, sb_i32, "phys_room_idx", int);
    UPT(sh_ro, sb_i32, "room_offsets", int);
    UPT(sh_rc, sb_i32, "room_cells", int);
    UPT(sh_akind, sb_u8, "act_owner_kind", uint8_t);
    UPT(sh_aact, sb_u8, "act_actuator", uint8_t);
    UPT(sh_aidx, sb_i32, "act_owner_index", int);
    UPT(sh_rws, sb_f64, "reward_scalars", double);
    UPTW(sh_rwh, sb_i64, "rew_hour", int64_t, rwh_len);
    UPTW(sh_rwm, sb_i64, "rew_month", int64_t, rwm_len);
    UPTW(sh_rww, sb_u8, "rew_workday", uint8_t, rww_len);
    UPT(sh_ewd, sb_f64, "elec_weekday_price", double);
    UPT(sh_ewe, sb_f64, "elec_weekend_price", double);
    UPT(sh_ec, sb_f64, "elec_carbon_rate", double);
    UPT(sh_gp, sb_f64, "gas_month_price", double);
    UPT(sh_fk, sb_u8, "f_dev_kind", uint8_t);
    UPT(sh_fid, sb_u8, "f_obs_id", uint8_t);
    UPT(sh_fix, sb_i32, "f_dev_idx", int);
    UPT(sh_fme, sb_f64, "f_norm_mean", double);
    UPT(sh_fva, sb_f64, "f_norm_var", double);
    UPTW(sh_axt, sb_f64, "aux_time", double, axt_len);
    UPTW(d_hour_m1, sb_i64, "occ_hour_m1", int64_t, hourm1_len);
    UPTW(d_wd_m1, sb_u8, "occ_workday_m1", uint8_t, wdm1_len);
    /* FDM tables */
    {   /* the solved-cell indices are baked as int64; the kernels index with int */
        const int64_t* sv = sb_i64(&blob, "solved");
        std::vector<int> si(n);
        for (int i = 0; i < n; i++) si[i] = (int)sv[i];
        int* d_si;
        CU(cudaMalloc(&d_si, 4 * (size_t)n));
        CU(cudaMemcpy(d_si, si.data(), 4 * (size_t)n, cudaMemcpyHostToDevice));
        sd_i = d_si;
    }
    UPT(sd_cap, sb_f64, "capacitance_rate", double);
    UPT(sd_dir, sb_f64, "dirichlet_conductance", double);
    UPT(sd_mc, sb_f64, "mass_coupling", double);
    UPT(sd_a0, sb_f64, "ambient_face_areas_0", double);
    UPT(sd_a1, sb_f64, "ambient_face_areas_1", double);
    UPT(sd_hor, sb_f64, "horizontal", double);
    UPT(sd_ver, sb_f64, "vertical", double);
    UPT(sd_con, sb_f64, "conductivity", double);
    UPT(sd_ret, sb_f64, "mass_retained", double);
    UPT(sd_gain, sb_f64, "mass_gained", double);
    UPT(sd_sp, sb_i32, "sensed_indptr", int);
    UPT(sd_si, sb_i32, "sensed_indices", int);
    UPT(sd_spx, sb_f64, "sensed_data", double);
    UPT(sd_ap, sb_i32, "applied_indptr", int);
    UPT(sd_ai, sb_i32, "applied_indices", int);
    UPT(sd_apx, sb_f64, "applied_data", double);

    /* the transposed air-side sensed matrix (k rows, entries by ascending p) */
    const int* ap_p = sb_i32(&blob, "air_side_sensed_indptr");
    const int* ap_i = sb_i32(&blob, "air_side_sensed_indices");
    const double* ap_x = sb_f64(&blob, "air_side_sensed_data");
    int nsens = ap_p[ncv];
    std::vector<int> cnt(k, 0), nxt(k, 0), tp(k + 1, 0), tpi(nsens);
    std::vector<double> tx(nsens);
    for (int p = 0; p < ncv; p++)
        for (int q = ap_p[p]; q < ap_p[p + 1]; q++) cnt[ap_i[q]]++;
    for (int j = 0; j < k; j++) tp[j + 1] = tp[j] + cnt[j];
    for (int j = 0; j < k; j++) nxt[j] = tp[j];
    for (int p = 0; p < ncv; p++)
        for (int q = ap_p[p]; q < ap_p[p + 1]; q++) {
            int j = ap_i[q];
            tpi[nxt[j]] = p; tx[nxt[j]] = ap_x[q]; nxt[j]++;
        }
    CU(cudaMalloc(&sh_tp, 4 * (size_t)(k + 1)));
    CU(cudaMalloc(&sh_ti, 4 * (size_t)nsens));
    CU(cudaMalloc(&sh_tx, 8 * (size_t)nsens));
    CU(cudaMemcpy(sh_tp, tp.data(), 4 * (size_t)(k + 1), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(sh_ti, tpi.data(), 4 * (size_t)nsens, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(sh_tx, tx.data(), 8 * (size_t)nsens, cudaMemcpyHostToDevice));

    /* max convolution room, for the staged shared buffer */
    max_room = 1;
    {
        const int* co = sb_i32(&blob, "conv_room_offsets");
        for (int i = 0; i < n_rooms_c; i++) {
            int v = co[i + 1] - co[i];
            if (v > max_room) max_room = v;
        }
    }

    if (stream == NULL) CU(cudaStreamCreate(&stream));  /* a bound stream wins */
    /* restart flags: [0] any env of the group restarts this step, [1 + e] env e restarts */
    CU(cudaMalloc((void**)&d_reset, 4 * (size_t)(nenv + 1)));
    CU(cudaMemset(d_reset, 0, 4 * (size_t)(nenv + 1)));
    {
        CU(cudaMalloc((void**)&d_act_disc, (size_t)n_action));
        CU(cudaMemcpy((void*)d_act_disc, sb_u8(&blob, "act_is_discrete"),
                      (size_t)n_action, cudaMemcpyHostToDevice));
        const char* nm[4] = {"act_min_native", "act_max_native", "act_min_norm",
                             "act_max_norm"};
        const double** dst[4] = {&d_act_mn, &d_act_mx, &d_act_mnr, &d_act_mxr};
        for (int t = 0; t < 4; t++) {
            CU(cudaMalloc((void**)dst[t], 8 * (size_t)n_action));
            CU(cudaMemcpy((void*)*dst[t], sb_f64(&blob, nm[t]), 8 * (size_t)n_action,
                          cudaMemcpyHostToDevice));
        }
    }
    {
        /* convect2_k: u16 room/candidate indices and a shared footprint that
         * scales with the largest room; outside those limits the original
         * kernel (same results, slower) runs instead. */
        const int* cof = sb_i32(&blob, "conv_cand_offsets");
        const long ncell_c = (long)sb_find(&blob, "conv_cand_offsets")->count - 1;
        int max_cand = 0;
        for (long i = 0; i < ncell_c; i++)
            if (cof[i + 1] - cof[i] > max_cand) max_cand = cof[i + 1] - cof[i];
        const int mr2_ = (max_room + 1) & ~1;
        cv2_smem = (size_t)(CV_RING + 4 + 2 * CV_BW + 2 * MT_N + max_room) * 4 + (size_t)2 * mr2_;
        /* rooms too big for the shared swap list: keep only the u16 candidate counts in shared */
        const size_t cv2g_smem = (size_t)(CV_RING + 4 + 2 * CV_BW + 2 * MT_N) * 4 + (size_t)2 * mr2_;
        int lim2 = 0;
        CU(cudaDeviceGetAttribute(&lim2, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        const bool lim_ok = max_room <= 65535 && max_cand <= 65535 && getenv("SB_OLDCONV") == NULL;
        cv2_global = getenv("SB_CONV_GLOBAL") != NULL || (long)cv2_smem > (long)lim2;
        if (cv2_global) cv2_smem = cv2g_smem;
        cv2_ok = lim_ok && (long)cv2_smem <= (long)lim2;
        if (cv2_ok) {
            if (cv2_global) {
                CU(sb_raise_smem(convect2g_k, cv2_smem));
                CU(cudaMalloc(&d_cv_lr, 4 * (size_t)max_room));
                CU(cudaMalloc(&d_cv_p, 2 * (size_t)max_room));
            } else {
                CU(sb_raise_smem(convect2_k, cv2_smem));
            }
            /* The swap decisions depend only on the generator stream and the room
             * tables, never on the temperatures, so the producer runs on a side
             * stream concurrently with the FDM work and the step just applies
             * the resulting permutation.  Every env of a batch loads the same
             * scenario and therefore the same seed, so their streams are
             * identical: one producer block serves the whole batch
             * (SB_CONV_PER_ENV=1 gives each env its own). */
            cv_ns = getenv("SB_CONV_PER_ENV") ? nenv : 1;
            cv_cap = 2 * (int)(sb_find(&blob, "conv_room_cells")->count / 2);
            if (cv2_global && cv_ns > 1) {          /* per-env producers need their own scratch */
                CU(cudaFree(d_cv_lr)); CU(cudaFree(d_cv_p));
                CU(cudaMalloc(&d_cv_lr, 4 * (size_t)cv_ns * max_room));
                CU(cudaMalloc(&d_cv_p, 2 * (size_t)cv_ns * max_room));
            }
            CU(cudaMalloc(&d_cvdst, 4 * (size_t)cv_ns * cv_cap));
            CU(cudaMalloc(&d_cvsrc, 4 * (size_t)cv_ns * cv_cap));
            CU(cudaMalloc(&d_cvcnt, 4 * (size_t)cv_ns));
            CU(cudaMemset(d_cvcnt, 0, 4 * (size_t)cv_ns));
            CU(cudaMalloc(&d_cvscr, 8 * (size_t)lu_ld * cv_cap));
            CU(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
            CU(cudaEventCreateWithFlags(&ev_app, cudaEventDisableTiming));
            CU(cudaEventCreateWithFlags(&ev_prod, cudaEventDisableTiming));
        }
        if (getenv("SB_VERBOSE"))
            printf("convection: %s (max_room=%d max_cand=%d smem=%zu)\n",
                   cv2_ok ? (cv2_global ? "convect2g_k" : "convect2_k") : "convect_k", max_room, max_cand, cv2_smem);
    }
    {
        /* the fallback kernel (used only when convect2 does not fit): its dynamic shared memory plus its static
         * shared arrays must fit the opt-in limit, or cudaFuncSetAttribute fails ("invalid argument") and set-up
         * aborted even for buildings that never run this kernel (max_room ~11 300 cells, e.g. gen-val building_20001) */
        const size_t csm_ = (size_t)max_room * sizeof(double);
        int lim = 0;
        CU(cudaDeviceGetAttribute(&lim, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        cudaFuncAttributes fa_;
        CU(cudaFuncGetAttributes(&fa_, convect_k));
        if ((long)(csm_ + fa_.sharedSizeBytes) <= (long)lim)
            CU(sb_raise_smem(convect_k, csm_));
    }

    /* cell -> control-volume offsets for the kernels that walk `room_cells`
     * (avg_temp, input_q); the convection has its own conv_room_cells table and
     * keeps its own indexing.  This is exactly room_cells[2t]*nc+room_cells[2t+1],
     * precomputed so those kernels stop chasing two dependent loads per cell. */
    {
        const int* rcc = sb_i32(&blob, "room_cells");
        int n_rc = (int)(sb_find(&blob, "room_cells")->count / 2);
        std::vector<int> coff((size_t)(n_rc > 0 ? n_rc : 1), 0);
        for (int q = 0; q < n_rc; q++)
            coff[q] = rcc[2 * q] * nc + rcc[2 * q + 1];
        CU(cudaMalloc((void**)&d_coff, 4 * (size_t)(n_rc > 0 ? n_rc : 1)));
        CU(cudaMemcpy(d_coff, coff.data(), 4 * (size_t)n_rc, cudaMemcpyHostToDevice));
    }

    /* ---- the baked factorization, on the device ---- */
    if (sb_lu_setup() != 0) return 1;

    /* ---- per-env state (nenv envs of this scenario) ---- */
    CU(cudaMalloc(&d_sims, sizeof(GpuSim) * nenv));
    CU(cudaMalloc(&d_native, 8 * (size_t)nenv * n_action));
    CU(cudaMalloc(&d_V, 8 * (size_t)nenv * n_zones));
    CU(cudaMemset(d_V, 0, 8 * (size_t)nenv * n_zones));
    CU(cudaMalloc(&d_occstate, 4 * (size_t)nenv * n_zones * 2));
    CU(cudaMemset(d_occstate, 0, 4 * (size_t)nenv * n_zones * 2));
    CU(cudaMalloc(&d_stepv, 4 * (size_t)nenv));
    /* field and mass temperatures: [cell][lu_ld], the solver's column layout (padding columns 0) */
    CU(cudaMalloc(&d_temp, 8 * (size_t)lu_ld * ncv));
    CU(cudaMalloc(&d_mass_temp, 8 * (size_t)lu_ld * n));
    CU(cudaMemset(d_temp, 0, 8 * (size_t)lu_ld * ncv));
    CU(cudaMemset(d_mass_temp, 0, 8 * (size_t)lu_ld * n));
    CU(cudaMalloc(&d_obs_d, 8 * (size_t)nenv * n_zones));
    CU(cudaMalloc(&d_rew, 8 * (size_t)nenv));
    CU(cudaMalloc(&d_phi, 8 * (size_t)nenv));
    CU(cudaMalloc(&d_shape, 4 * (size_t)nenv));
    CU(cudaMemset(d_phi, 0xFF, 8 * (size_t)nenv));     /* NaN: no previous step, skip the bonus */
    CU(cudaMemset(d_shape, 0, 4 * (size_t)nenv));
    CU(cudaMalloc(&d_rparts, 8 * (size_t)nenv * SB_NPARTS));
    CU(cudaMalloc(&d_epret, 4 * (size_t)nenv));
    CU(cudaMemset(d_epret, 0, 4 * (size_t)nenv));
    CU(cudaMemset(d_rparts, 0, 8 * (size_t)nenv * SB_NPARTS));
    CU(cudaMalloc(&d_tmean, 8 * (size_t)nenv));
    CU(cudaMalloc(&d_obs, 4 * (size_t)nenv * SB_OBS));
    CU(cudaMemset(d_obs, 0, 4 * (size_t)nenv * SB_OBS));
    if (!cv2_ok) {      /* only the fallback convection kernel needs these */
        CU(cudaMalloc(&d_left, 4 * (size_t)nenv * max_room));
        CU(cudaMalloc(&d_right, 4 * (size_t)nenv * max_room));
        CU(cudaMalloc(&d_consumed, 4 * (size_t)nenv));
        CU(cudaMalloc(&d_words, 4 * (size_t)nenv * conv_words));
    }
    CU(cudaMalloc(&d_rngc, sizeof(GpuRng) * nenv));
    CU(cudaMalloc(&d_rngo, sizeof(GpuRng) * nenv));
    CU(cudaMalloc(&d_qair, 8 * (size_t)(k ? k : 1) * lu_ld));     /* [j][ld], padding lanes stay 0 */
    CU(cudaMalloc(&d_rpart, 8 * (size_t)(n_rseg ? n_rseg : 1) * lu_ld));
    {   /* weather of each solver column (padding columns: weather 0, their values are never used) */
        std::vector<int> ew(lu_ld, 0);
        for (int e = 0; e < lu_ld; e++) ew[e] = h_envw[e < nenv ? e : nenv - 1];   /* padding: the last weather */
        g_lu.slab_uniform = 1;
        for (int e = 0; e < lu_ld; e++) if (ew[e] != ew[e & ~31]) g_lu.slab_uniform = 0;
        CU(cudaMalloc(&d_envw, 4 * (size_t)lu_ld));
        CU(cudaMemcpy(d_envw, ew.data(), 4 * (size_t)lu_ld, cudaMemcpyHostToDevice));
        g_lu.envw = d_envw;
    }
    CU(cudaMemset(d_qair, 0, 8 * (size_t)(k ? k : 1) * lu_ld));
    CU(cudaMalloc(&d_lu_work, sizeof(sb_wt) * (size_t)n * lu_ld));
    CU(cudaMalloc(&d_lu_z, sizeof(sb_wt) * (size_t)n * lu_ld));
    CU(cudaMemset(d_lu_work, 0, sizeof(sb_wt) * (size_t)n * lu_ld));
    CU(cudaMemset(d_lu_z, 0, sizeof(sb_wt) * (size_t)n * lu_ld));

    th = 256;
    zth = (nenv * n_zones + th - 1) / th;
    cth = (nenv * ncv + th - 1) / th;
    nth2 = (nenv * n + th - 1) / th;
    kth = (nenv * k + th - 1) / th;
    smem = (size_t)k * k * 8 + (size_t)k * 2 * 8;
    {
        int lim = 0;
        CU(cudaDeviceGetAttribute(&lim, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        air_global = smem > (size_t)lim;
        if (air_global) {               /* q and the multipliers stay in shared, the matrix goes to global */
            smem = (size_t)k * 2 * 8;
            CU(cudaMalloc(&d_airM, 8 * (size_t)nenv * k * k));
        } else if (smem > 48 * 1024)
            CU(sb_raise_smem(airside_k, smem));
    }

    /* per-env scratch the HVAC/reward code needs */
    double *d_zq, *d_zqd, *d_sm, *d_co;
    CU(cudaMalloc(&d_zq, 8 * (size_t)nenv * n_zones));
    CU(cudaMalloc(&d_zqd, 8 * (size_t)nenv * n_zones));
    CU(cudaMemset(d_zqd, 0, 8 * (size_t)nenv * n_zones));
    CU(cudaMalloc(&d_sm, 8 * (size_t)nenv * k));
    CU(cudaMalloc(&d_g, 8 * (size_t)nenv * k));
    CU(cudaMalloc(&d_t2, 8 * (size_t)nenv * k));
    CU(cudaMalloc(&d_co, 8 * (size_t)nenv * k));
    double *d_at, *d_sen, *d_va, *d_aw, *d_aa, *d_sv;
    int *d_ar, *d_acq, *d_vm, *d_vpv, *d_vds, *d_vrs;
    double *d_vrec, *d_vsat, *d_vfan, *d_vflow, *d_vdam, *d_vreh, *d_vrat, *d_vhsp,
        *d_vcsp, *d_vdcmd, *d_vrcmd;
    CU(cudaMalloc(&d_at, 8 * (size_t)nenv * n_zones));
    CU(cudaMalloc(&d_sen, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_va, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_aw, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_aa, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_sv, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_ar, 4 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_acq, 4 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_vm, 4 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vpv, 4 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vds, 4 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vrs, 4 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vrec, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_vsat, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_vfan, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_vflow, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_vdam, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vreh, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vrat, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vhsp, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vcsp, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vdcmd, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vrcmd, 8 * (size_t)nenv * n_vav));
    double *d_zo_sp, *d_zo_act;
    CU(cudaMalloc(&d_zo_sp, 8 * (size_t)nenv * n_zones));
    CU(cudaMalloc(&d_zo_act, 8 * (size_t)nenv * n_zones));
    CU(cudaMemset(d_acq, 0, 4 * (size_t)nenv * n_ahu));
    CU(cudaMemset(d_at, 0, 8 * (size_t)nenv * n_zones));
    CU(cudaMemset(d_sen, 0, 8 * (size_t)nenv * n_vav));
    CU(cudaMemset(d_va, 0, 8 * (size_t)nenv * n_vav));

    /* initial field / mass / RNG state, and the GpuSim structs */
    const double* ti = sb_f64(&blob, "temp_init");
    const double* mi = sb_f64(&blob, "mass_temp_init");
    const double* ahu_const = sb_f64(&blob, "ahu_const");
    const double* vav_const = sb_f64(&blob, "vav_const");
    const double* boil_host = sb_f64(&blob, "boiler_const");
    if (!ti || !mi || !ahu_const || !vav_const || !boil_host) {
        fprintf(stderr, "smart_buildings: missing initial state\n");
        return 1;
    }
    std::vector<double> t0(ncv), m0(n);
    memcpy(t0.data(), ti, 8 * (size_t)ncv);
    memcpy(m0.data(), mi, 8 * (size_t)n);
    std::vector<int> ahu_run0((size_t)nenv * n_ahu, 1);
    std::vector<double> arec((size_t)nenv * n_ahu), asat((size_t)nenv * n_ahu),
        afan((size_t)nenv * n_ahu), aflow((size_t)nenv * n_ahu);
    std::vector<double> vdam((size_t)nenv * n_vav), vreh((size_t)nenv * n_vav),
        vrat((size_t)nenv * n_vav), vhsp((size_t)nenv * n_vav),
        vcsp((size_t)nenv * n_vav), vdcmd((size_t)nenv * n_vav),
        vrcmd((size_t)nenv * n_vav);
    for (int e = 0; e < nenv; e++) {
        for (int a = 0; a < n_ahu; a++) {
            const double* c = &ahu_const[a * 9];
            ahu_run0[e * n_ahu + a] = (int)c[8];
            arec[e * n_ahu + a] = c[0]; asat[e * n_ahu + a] = c[6];
            afan[e * n_ahu + a] = c[7]; aflow[e * n_ahu + a] = 0.0;
        }
        for (int v = 0; v < n_vav; v++) {
            const double* c = &vav_const[v * 9];
            vdam[e * n_vav + v] = c[7]; vreh[e * n_vav + v] = c[8];
            vrat[e * n_vav + v] = 1.0; vhsp[e * n_vav + v] = c[5];
            vcsp[e * n_vav + v] = c[6]; vdcmd[e * n_vav + v] = 0.0;
            vrcmd[e * n_vav + v] = 0.0;
        }
    }
    double *d_arec, *d_asat, *d_afan, *d_aflow, *d_vdam2, *d_vreh2, *d_vrat2, *d_vhsp2,
        *d_vcsp2, *d_vdcmd2, *d_vrcmd2;
    CU(cudaMalloc(&d_arec, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_asat, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_afan, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_aflow, 8 * (size_t)nenv * n_ahu));
    CU(cudaMalloc(&d_vdam2, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vreh2, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vrat2, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vhsp2, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vcsp2, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vdcmd2, 8 * (size_t)nenv * n_vav));
    CU(cudaMalloc(&d_vrcmd2, 8 * (size_t)nenv * n_vav));
    CU(cudaMemcpy(d_arec, arec.data(), 8 * (size_t)nenv * n_ahu, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_asat, asat.data(), 8 * (size_t)nenv * n_ahu, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_afan, afan.data(), 8 * (size_t)nenv * n_ahu, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_aflow, aflow.data(), 8 * (size_t)nenv * n_ahu, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_vdam2, vdam.data(), 8 * (size_t)nenv * n_vav, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_vreh2, vreh.data(), 8 * (size_t)nenv * n_vav, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_vrat2, vrat.data(), 8 * (size_t)nenv * n_vav, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_vhsp2, vhsp.data(), 8 * (size_t)nenv * n_vav, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_vcsp2, vcsp.data(), 8 * (size_t)nenv * n_vav, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_vdcmd2, vdcmd.data(), 8 * (size_t)nenv * n_vav, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_vrcmd2, vrcmd.data(), 8 * (size_t)nenv * n_vav, cudaMemcpyHostToDevice));

    std::vector<GpuRng> hc(nenv), ho(nenv);
    SbRng tr;
    sb_rng_seed(&tr, (uint32_t)sb_i64_1(&blob, "conv_seed", 0));
    for (int e = 0; e < nenv; e++) { memcpy(hc[e].mt, tr.mt, sizeof(tr.mt)); hc[e].idx = tr.idx; }
    sb_rng_seed(&tr, (uint32_t)sb_i64_1(&blob, "occ_seed", 17321));
    for (int e = 0; e < nenv; e++) { memcpy(ho[e].mt, tr.mt, sizeof(tr.mt)); ho[e].idx = tr.idx; }
    CU(cudaMemcpy(d_rngc, hc.data(), sizeof(GpuRng) * nenv, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_rngo, ho.data(), sizeof(GpuRng) * nenv, cudaMemcpyHostToDevice));

    std::vector<GpuSim> hs(nenv);
    for (int e = 0; e < nenv; e++) {
        GpuSim* s = &hs[e];
        memset(s, 0, sizeof(*s));
        s->n_ahu = n_ahu; s->n_vav = n_vav; s->n_zones = n_zones;
        s->n_rooms = n_rooms_c; s->k = k; s->nr = nr; s->nc = nc; s->n_cvs = ncv;
        s->n_solved = n; s->n_action = n_action; s->step_index = 0;
        s->vav_zone_off = sh_zvo; s->vav_zone_idx = sh_zvi; s->vav_ahu_idx = sh_vai;
        s->coupled_idx = sh_cvi; s->phys_room_idx = sh_pri;
        s->room_off = sh_ro; s->room_cells = sh_rc;
        s->zone_area = sh_zarea; s->ahu_const = sh_ahu; s->vav_const = sh_vav;
        s->act_kind = sh_akind; s->act_act = sh_aact; s->act_idx = sh_aidx;
        s->sched_comfort = sh_sc; s->sched_heat_sp = sh_sh; s->sched_cool_sp = sh_scl;
        const int w = h_envw[e];
        s->amb = sh_amb + (size_t)w * amb_len; s->boiler_c = sh_boil;
        s->internal_w_m2 = sb_f64_1(&blob, "internal_gain_w_per_m2", 0.0);
        s->standby_frac = sb_f64_1(&blob, "internal_gain_standby_fraction", 0.0);
        s->occupant_w = sb_f64_1(&blob, "occupant_sensible_heat_w", 0.0);
        s->cv_area_m2 = sb_f64_1(&blob, "cv_size_cm", 0.0) / 100.0;
        s->cv_area_m2 *= s->cv_area_m2;
        s->h = sb_f64_1(&wblobs[w], "convection_coefficient", 8.8);
        s->header_res = sb_f64_1(&blob, "hws_header_resistance", 0.0);
        {
            const double* pc = sb_f64(&blob, "pump_const");
            s->pump_head = pc ? pc[0] : 0.0;
        }
        s->dt_sec = dt_sec_;
        s->ahu_run = d_ar + (size_t)e * n_ahu; s->ahu_coolreq = d_acq + (size_t)e * n_ahu;
        s->vav_mode = d_vm + (size_t)e * n_vav; s->vav_prev_valid = d_vpv + (size_t)e * n_vav;
        s->vav_dcmd_set = d_vds + (size_t)e * n_vav; s->vav_rcmd_set = d_vrs + (size_t)e * n_vav;
        s->ahu_recirc = d_arec + (size_t)e * n_ahu; s->ahu_sat = d_asat + (size_t)e * n_ahu;
        s->ahu_fan = d_afan + (size_t)e * n_ahu; s->ahu_flow = d_aflow + (size_t)e * n_ahu;
        s->vav_damper = d_vdam2 + (size_t)e * n_vav; s->vav_reheat = d_vreh2 + (size_t)e * n_vav;
        s->vav_ratio = d_vrat2 + (size_t)e * n_vav; s->vav_heatsp = d_vhsp2 + (size_t)e * n_vav;
        s->vav_coolsp = d_vcsp2 + (size_t)e * n_vav; s->vav_dcmd = d_vdcmd2 + (size_t)e * n_vav;
        s->vav_rcmd = d_vrcmd2 + (size_t)e * n_vav;
        s->avg_temp = d_at + (size_t)e * n_zones; s->sensing = d_sen + (size_t)e * n_vav;
        s->vav_area = d_va + (size_t)e * n_vav; s->ahu_wsum = d_aw + (size_t)e * n_ahu;
        s->ahu_atotal = d_aa + (size_t)e * n_ahu; s->ahu_satv = d_sv + (size_t)e * n_ahu;
        s->zone_q = d_zq + (size_t)e * n_zones; s->zq_div = d_zqd + (size_t)e * n_zones;
        s->cell_zone = d_cellzone;
        s->sensed_mean = d_sm + (size_t)e * k; s->air_g = d_g + (size_t)e * k;
        s->air_T = d_t2 + (size_t)e * k; s->coil = d_co + (size_t)e * k;
        s->temp = d_temp + e; s->mass_temp = d_mass_temp + e; s->ts = lu_ld;
        s->zone_vav_off = sh_zvo; s->zone_vav_idx = sh_zvi;
        s->rew_hour = sh_rwh + (size_t)w * rwh_len; s->rew_month = sh_rwm + (size_t)w * rwm_len; s->rew_wd = sh_rww + (size_t)w * rww_len;
        s->rw_scalars = sh_rws; s->elec_wd = sh_ewd; s->elec_we = sh_ewe;
        s->elec_c = sh_ec; s->gas_p = sh_gp; s->gas_carb_rate = gas_carb_;
        s->pump_eff = pump_eff_;
        s->f_kind = sh_fk; s->f_obs_id = sh_fid; s->f_idx = sh_fix;
        s->f_mean = sh_fme; s->f_var = sh_fva; s->aux_time = sh_axt + (size_t)w * axt_len;
        s->n_dev = n_dev_; s->n_aux = n_aux_; s->obs_len = SB_OBS;   /* per-env STRIDE of the framework's observation buffer */
        s->zo_sp = d_zo_sp + (size_t)e * n_zones;
        s->zo_act = d_zo_act + (size_t)e * n_zones;
        s->rew_out = d_rew + e;
        s->rparts = d_rparts + (size_t)e * SB_NPARTS;
        s->boiler_setpoint = boil_host[0]; s->boiler_return = boil_host[1];
        s->boiler_current = boil_host[0]; s->boiler_step_change = 0.0;
        s->hws_run = ON; s->pump_run = ON; s->hws_heating_requests = 0;
    }
    CU(cudaMemcpy(d_sims, hs.data(), sizeof(GpuSim) * nenv, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_exdir, 8 * (size_t)n * nw));      /* [w][n] */
    for (int w = 0; w < nw; w++)
        exdir_k<<<(n + 255) / 256, 256, 0, stream>>>(n, sd_i, sd_dir, sd_a0, sd_a1, sd_hor, sd_ver,
                                                     sd_con, sb_f64_1(&wblobs[w], "convection_coefficient", 8.8),
                                                     d_exdir + (size_t)w * n);
    CU(cudaStreamSynchronize(stream));
    CU(cudaMalloc(&d_tinit, 8 * (size_t)ncv));
    CU(cudaMalloc(&d_minit, 8 * (size_t)n));
    CU(cudaMemcpy(d_tinit, t0.data(), 8 * (size_t)ncv, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_minit, m0.data(), 8 * (size_t)n, cudaMemcpyHostToDevice));
    {
        std::vector<double> tt((size_t)lu_ld * ncv, 0.0), mm((size_t)lu_ld * n, 0.0);
        for (int p = 0; p < ncv; p++) for (int e = 0; e < nenv; e++) tt[(size_t)p * lu_ld + e] = t0[p];
        for (int t = 0; t < n; t++) for (int e = 0; e < nenv; e++) mm[(size_t)t * lu_ld + e] = m0[t];
        CU(cudaMemcpy(d_temp, tt.data(), 8 * tt.size(), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_mass_temp, mm.data(), 8 * mm.size(), cudaMemcpyHostToDevice));
    }
    if (sb_canon_setup() != 0) return 1;
    return 0;
}

/* ------------------------------------------------------------- batched API --- */
/* Episode start for every env: primed occupant chains, interval 0 ready, and
 * the reset observation.  Mirrors sb_sim_reset + sb_occup_prime + the interval-0
 * advance in the CPU env's puf_reset. */
#ifndef SB_TEST
/* the Env batch's own step counters (only the env build has an Env array) */
__global__ void reset_envstep_k(Env* envs, int nenv, const int* gate) {
    if (gate != NULL && gate[0] == 0) return;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < nenv && (gate == NULL || gate[1 + e] != 0)) envs[e].step = 0;
}

/* SplitMix64: independent per-env state advances only when that env resets.
 * This executes on every graph replay, not only during graph capture. */
__host__ __device__ static unsigned long long sb_start_mix(unsigned long long x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
__device__ double g_start_draws[SB_MAX_BLOBS][2][7];  /* initial and later reset draws */
__global__ void random_start_k(Env* envs, GpuSim* sims, int* stepv, int nenv,
                               const int* gate, const int* scenario) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv || (gate != NULL && (gate[0] == 0 || gate[1 + e] == 0))) return;
    Env* env = &envs[e];
    env->start_rng += 0x9e3779b97f4a7c15ULL;
    int day = (int)(sb_start_mix(env->start_rng) % 7ULL);
    env->start_day = day;
    env->start_offset = day * 288;
    const int scen = scenario[e];
    if (scen >= 0 && scen < SB_MAX_BLOBS)
        atomicAdd(&g_start_draws[scen][env->start_draws == 0 ? 0 : 1][day], 1.0);
    env->start_draws++;
    stepv[e] = env->start_offset;
    sims[e].step_index = env->start_offset;
    sims[e].start_index = env->start_offset;
}
#endif
/* The episode restart, as device work.  Order and content follow the CPU env's
 * puf_reset: sb_sim_reset (+ priming the sensed temperatures), occupant prime, the
 * reset OBSERVATION (which sees the primed occupancy), then the interval-0 advance.
 * gate == NULL restarts every env unconditionally (the host reset); a device flag
 * restarts only when it is set, which is how the in-step restart at the end of an
 * episode uses it.  The convection / occupancy generators and the occupant chains
 * keep their state, exactly as the CPU env leaves them. */
#ifndef SB_TEST
__global__ void zm_copy_step_k(Env* envs, const int* stepv, int nenv, const int* gate) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    if (gate != NULL && (gate[0] == 0 || gate[1 + e] == 0)) return;
    envs[e].step = stepv[e];
}
#endif
#include "sb_zone_memory.cuh"

static void sb_enqueue_reset(const int* gate) {
    base_reset_k<<<(nenv + 127) / 128, 128, 0, stream>>>(d_pid, nenv, gate);
    reset_sim_k<<<nenv, 128, 0, stream>>>(d_sims, nenv, gate);
    reset_field_k<<<(int)(((size_t)nenv * (ncv > n ? ncv : n) + 255) / 256), 256, 0, stream>>>(
        d_sims, nenv, ncv, n, d_tinit, d_minit, gate);
    zm_pre_temp(gate);
    avg_temp_k<<<(nenv * n_zones + 31) / 32, 32, 0, stream>>>(d_sims, nenv, d_coff, gate);
    sense_k<<<(nenv * n_vav + 127) / 128, 128, 0, stream>>>(d_sims, nenv, gate);
    reset_stepv_k<<<(nenv + 255) / 256, 256, 0, stream>>>(d_stepv, nenv, gate);
#ifndef SB_TEST
    if (g_envs != NULL)
        reset_envstep_k<<<(nenv + 255) / 256, 256, 0, stream>>>(g_envs, nenv, gate);
    if (g_random_start && g_envs != NULL)
        random_start_k<<<(nenv + 255) / 256, 256, 0, stream>>>(g_envs, d_sims, d_stepv, nenv, gate, d_envscen);
#else
    if (g_test_start > 0)
        test_start_k<<<(nenv + 255) / 256, 256, 0, stream>>>(d_sims, d_stepv, nenv, g_test_start, gate);
#endif
    zm_pre_clock(gate);
    occup_k<<<nenv, 32, 0, stream>>>(d_hour_m1, hourm1_len, d_wd_m1, d_envw, wdm1_len, d_stepv, n_zones, nassign,
                                     occ_ea, occ_la, occ_ed, occ_ld, p_arr, p_dep,
                                     d_occstate, d_V, d_rngo, gate);
    obs_dev_k<<<(nenv * n_dev_ + 127) / 128, 128, 0, stream>>>(d_sims, nenv, d_stepv,
                                                               g_obs, gate);
    obs_aux_k<<<(nenv * n_aux_ + 127) / 128, 128, 0, stream>>>(d_sims, d_V, nenv,
                                                               d_stepv, g_obs, gate);
    occup_k<<<nenv, 32, 0, stream>>>(sh_hour, hour_len, sh_wd, d_envw, wd_len, d_stepv, n_zones, nassign,
                                     occ_ea, occ_la, occ_ed, occ_ld, p_arr, p_dep,
                                     d_occstate, d_V, d_rngo, gate);
    zm_reset(gate);
    zm_update();
}

static int sb_gpu_reset(void) {
    /* the kernels write only the first obs_len entries of each env's SB_OBS slot */
    CU(cudaMemsetAsync(g_obs, 0, 4 * (size_t)nenv * SB_OBS, stream));
#ifdef SB_ENTITY
    if (g_canon) CU(cudaMemsetAsync(g_obs_pol, 0, 4 * (size_t)nenv * SB_NENT * SB_ENT_OBS, stream));
#else
    if (g_canon) CU(cudaMemsetAsync(g_obs_pol, 0, 4 * (size_t)nenv * SB_OBS, stream));
#endif
    sb_enqueue_reset(NULL);
    sb_canon_tail();
    CU(cudaStreamSynchronize(stream));
    return 0;
}

/* One batched step: state, then observation + occupancy advance + reward. */
#ifndef SB_TEST
/* Action decode: sb_decode_native's rules applied to the framework's packed
 * action buffer, so puf_step needs no host round trip.  Also the per-env step
 * counter the schedule lookups use (occ_next == step). */
__global__ void decode_k(Env* envs, int n_act, int astride, const float* actions,
                         const uint8_t* is_disc, const double* mn, const double* mx,
                         const double* mnr, const double* mxr, double* native,
                         int* stepv) {
    int e = blockIdx.x;
    envs[e].step += 1;
    stepv[e] = envs[e].start_offset + envs[e].step;
    const float* a = actions + (size_t)e * astride;
    double* out = native + (size_t)e * n_act;
    int ci = 0, di = 0;
    for (int i = 0; i < n_act; i++) {
        if (is_disc[i]) {
            double dv = (double)a[SB_CONT + di];
            di++;
            double merged = (dv == 1.0) ? 1.0 : -1.0;
            out[i] = (merged - mnr[i]) / (mxr[i] - mnr[i]) * (mx[i] - mn[i]) + mn[i];
        } else {
            float m = a[ci];
            ci++;
            if (m > 1.0f) m = 1.0f;
            else if (m < -1.0f) m = -1.0f;
            float rng_in = (float)(mxr[i] - mnr[i]);
            float rng_out = (float)(mx[i] - mn[i]);
            out[i] = (double)((m - (float)mnr[i]) / rng_in * rng_out + (float)mn[i]);
        }
    }
}

/* canonical mode: the framework's step counters (decode_k does this for the native layout) */
__global__ void envstep_k(Env* envs, int nenv, int* stepv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    envs[e].step += 1;
    stepv[e] = envs[e].start_offset + envs[e].step;
}
/* Odd env executes the even env's action, with supply-air and hot-water setpoints raised by eps. */
__global__ void plant_pair_k(float* actions, int stride, int n_pairs, float eps, const CanonMap* cm) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    float* src = actions + (size_t)(2 * p) * stride;
    float* dst = actions + (size_t)(2 * p + 1) * stride;
    for (int i = 0; i < stride; i++) dst[i] = src[i];
    for (int k = 0; k < CN_K; k++) {
        if (cm->aT[k] < 0) continue;
        const int sl = CN_AA + 4 * k;
        float v = dst[sl] + eps;
        dst[sl] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
    }
    if (cm->hW >= 0) {
        float v = dst[CN_AH] + eps;
        dst[CN_AH] = v > 1.0f ? 1.0f : (v < -1.0f ? -1.0f : v);
    }
}
/* Same occupants and the same convection noise as the partner. Temperatures are not copied. */
__global__ void plant_sync_k(GpuRng* rngc, GpuRng* rngo, int* occ, int occ_stride, int n_pairs) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_pairs) return;
    rngc[2 * p + 1] = rngc[2 * p];
    rngo[2 * p + 1] = rngo[2 * p];
    int* s = occ + (size_t)(2 * p) * occ_stride;
    int* d = occ + (size_t)(2 * p + 1) * occ_stride;
    for (int i = 0; i < occ_stride; i++) d[i] = s[i];
}
/* env.probe (diagnostic, never used for a scored run): measure the environment's response to two
 * actuator changes applied on top of the policy's own actions, from the observation the policy sees.
 *   1 = the hot-water command is fixed at a = 0 through the contract's affine map (330 K);
 *   2 = the reheat command is shut (-1, closed) for every zone above its cooling setpoint;
 *   3 = both.
 * The point is to price the two moves the policy does not make (raise the water, close reheat where the
 * room is already too warm) without training anything. probe = 0 is the untouched action path. */
__global__ void probe_k(const CanonMap* cm, const float* src, const float* obs, float* dst,
                        int nenv, int stride, int mode, float water) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    const float* s = src + (size_t)e * stride;
    const float* o = obs + (size_t)e * SB_OBS;
    float* d = dst + (size_t)e * CN_ACT;
    for (int i = 0; i < CN_ACT; i++) d[i] = s[i];
    if ((mode == 1 || mode == 3) && cm->hW >= 0) d[CN_AH] = water;
    if (mode == 2 || mode == 3) {
        for (int sl = 0; sl < CN_ZS; sl++) {
            if (cm->zAr[sl] < 0 || cm->zT[sl] < 0 || cm->zC[sl] < 0) continue;
            if (cn_nat(cm, o, cm->zT[sl]) > cn_nat(cm, o, cm->zC[sl])) d[2 * sl + 1] = -1.0f;
        }
    }
}
/* the step's action decode on `stream`: native layout (decode_k) or canonical / residual (decode_res_k) */

__global__ void zone_command_k(const CanonMap* __restrict__ cm, const float* __restrict__ src,
                               const float* __restrict__ obs, float* __restrict__ dst, int nenv, int stride) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    const float* s = src + (size_t)e * stride;
    const float* o = obs + (size_t)e * SB_OBS;
    float* d = dst + (size_t)e * CN_ACT;
    for (int i = 0; i < CN_ACT; i++) d[i] = s[i];
    float sum = 0.0f;
    float sum_def = 0.0f;
    int n = 0;
    for (int sl = 0; sl < CN_ZS; sl++) {
        if (cm->zAd[sl] < 0) continue;
        const float u = s[2 * sl];
        int cool = 0;
        if (u < 0.0f && cm->zT[sl] >= 0 && cm->zC[sl] >= 0) {
            const int k = sl / CN_ZPA;
            if (k < CN_K && cm->aSAT[k] >= 0) {
                const float zone = cn_nat(cm, o, cm->zT[sl]);
                const float csp = cn_nat(cm, o, cm->zC[sl]);
                const float sat = cn_nat(cm, o, cm->aSAT[k]);
                if (sat < zone && zone > csp) cool = 1;
            }
        }
        float damper, reheat;
        sb_zone_apply(u, cool, &damper, &reheat);
        d[2 * sl] = damper;
        if (cm->zAr[sl] >= 0) d[2 * sl + 1] = reheat;
        sum += sb_heat_fraction(reheat);
        if (cm->zT[sl] >= 0 && cm->zH[sl] >= 0) {
            const float def = cn_nat(cm, o, cm->zH[sl]) - cn_nat(cm, o, cm->zT[sl]);
            if (def > 0.0f) sum_def += def;
        }
        n += 1;
    }
    const float mean = n > 0 ? sum / (float)n : 0.0f;
    float dp, run;
    sb_plant_from_heat(mean, &dp, &run);
    if (cm->hDPa >= 0) d[CN_AH + 1] = dp;
    if (cm->hRun >= 0) d[CN_AH + 2] = run;
    if (cm->hW >= 0) {
        /* r61_mod_band: the water cap is the demand reset (mean zone shortfall below
         * the heating setpoint, 4 K band), not "the zones asked, so boiler max". */
        const float cap = sb_water_reset_cap(n > 0 ? sum_def / (float)n : 0.0f);
        if (d[CN_AH] > cap) d[CN_AH] = cap;
    }
}
static void sb_zone_command_mask(unsigned char* m, const CanonMap* cm) {
    if (!g_zone_command || m == NULL || cm == NULL) return;
    for (int sl = 0; sl < CN_ZS; sl++) if (cm->zAr[sl] >= 0) m[2 * sl + 1] = 0;
    if (cm->hDPa >= 0) m[CN_AH + 1] = 0;
    if (cm->hRun >= 0) m[CN_AH + 2] = 0;
}
static void sb_decode(void) {
    if (g_canon) {
        if (g_plant_pair) {
            const int pairs = nenv / 2;
            plant_sync_k<<<(pairs + 63) / 64, 64, 0, stream>>>(d_rngc, d_rngo, d_occstate, n_zones * 2, pairs);
            plant_pair_k<<<(pairs + 63) / 64, 64, 0, stream>>>(g_actions, SB_CONT + SB_DISC, pairs, g_plant_eps, d_cmap);
        }
        envstep_k<<<(nenv + 127) / 128, 128, 0, stream>>>(g_envs, nenv, d_stepv);
#ifdef SB_ENTITY
        entity_gather_k<<<(nenv + 63) / 64, 64, 0, stream>>>(g_actions, d_entact, nenv);
        decode_res_k<<<(nenv * n_action + 127) / 128, 128, 0, stream>>>(d_cmap, d_base, d_entact, CN_ACT,
            n_action, nenv, d_act_disc, d_act_mn, d_act_mx, d_act_mnr, d_act_mxr, g_resid, g_res_scale, g_res_dscale,
            d_trust, d_native);
#else
        const float* act = g_actions;
        int astride = SB_CONT + SB_DISC;
#ifndef SB_TEST
        if (g_zone_command) {
            if (d_cmdact == NULL) {
                fprintf(stderr, "smart_buildings: zone command buffer was not allocated\n");
                abort();
            }
            zone_command_k<<<(nenv + 63) / 64, 64, 0, stream>>>(d_cmap, g_actions, g_obs, d_cmdact, nenv, astride);
            act = d_cmdact;
            astride = CN_ACT;
        }
        if (g_probe) {
            if (d_probeact == NULL) {
                fprintf(stderr, "smart_buildings: probe buffer was not allocated\n");
                abort();
            }
            probe_k<<<(nenv + 63) / 64, 64, 0, stream>>>(d_cmap, act, g_obs, d_probeact, nenv, astride,
                                                        g_probe, g_probe_water);
            act = d_probeact;
            astride = CN_ACT;
        }
#endif
        decode_res_k<<<(nenv * n_action + 127) / 128, 128, 0, stream>>>(d_cmap, d_base, act, astride,
            n_action, nenv, d_act_disc, d_act_mn, d_act_mx, d_act_mnr, d_act_mxr, g_resid, g_res_scale, g_res_dscale,
            d_trust, d_native);
#endif
    } else {
        decode_k<<<nenv, 1, 0, stream>>>(g_envs, n_action, SB_CONT + SB_DISC, g_actions,
                                         d_act_disc, d_act_mn, d_act_mx, d_act_mnr,
                                         d_act_mxr, d_native, d_stepv);
    }
}

/* Rewards, terminals and the episode log, on the device (log_reduce reads
 * envs[i].log in a kernel, so this is the intended home for it). */
__device__ double g_live[1 + 1 + SB_NPARTS_ENV];   /* step-level sums, read+reset at log time */
__device__ double g_live_grp[SB_MAX_BLOBS][4];      /* per scenario: reward, mean comfort deviation, steps, reward^2 */
/* Batch-wide, scenario-indexed passive step telemetry. Bins: night, weekday morning,
 * weekday afternoon, weekday evening, weekend daytime, empty, occupied. Columns:
 * count, productivity, energy, carbon, mean deviation, out-of-band, electricity kW, gas kW.
 * Read/reset only after rollout synchronization; never touches managed Env on the host. */
__device__ double g_detail_scen[SB_MAX_BLOBS][7][8];
__device__ double g_ep_scen[SB_MAX_BLOBS][2];       /* per scenario: sum of full-episode mean rewards, count (windowed) */
/* Completed episodes by scenario and start-day offset: count, reward, three
 * score components, and episode electricity/gas. No training signal uses this. */
__device__ double g_ep_start[SB_MAX_BLOBS][7][7];
__device__ double g_first_scen[SB_MAX_BLOBS][5];    /* per scenario, first full episode of each env: mean reward,
                                                       productivity, energy cost, carbon (episode means), count */
__device__ double g_znclip[3];                       /* steps whose learning reward |z| exceeded 1 / 3 / 5 sigma */
__device__ int d_znclip = 0;                         /* 1 only for rew_norm: those thresholds are z-scores */

__global__ void finalize_k(Env* envs, const double* rew, const double* rparts, int nenv,
                           float* rewards, float* terminals, int* reset, float* ep_ret_out, const int* envscen,
                           int ep_len, float rew_scale, int* epcount, double* epret_d,
                           const float* rnorm, const float* shape, const GpuSim* sims,
                           const double* occupancy, const int* calendar_step, int detail, int random_start) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nenv) return;
    float r = (float)rew[i];
    const int gidx = envscen[i];      /* the env's scenario (a group may hold several weathers) */
    /* the learning signal; logs and metrics keep the raw reward.  rnorm (env.rew_norm): the
     * per-scenario running mean/inv_std replace rew_scale entirely (set reward_scale = 1).
     * shape (env.comfort_shape): the train-only potential-based bonus from reward_k; it enters
     * rewards[] (and only that: ep_ret, rp[], the live metrics, the eval dump stay official). */
    const float rl = r + (shape != NULL ? shape[i] : 0.0f);
    rewards[i] = rnorm != NULL
        ? (rl - rnorm[gidx * 2]) * rnorm[gidx * 2 + 1]
        : rl * rew_scale;
    if (rnorm != NULL && d_znclip) {
        /* how much of the learning signal sits outside the +/-1 sigma window the trainer used to
         * clamp to (train.rew_clamp <= 0 disables that clamp; these fractions say what it hid).
         * Centering stores inv_std = 1, so |r - mu| is not a z-score and must not be counted here. */
        const float az = fabsf(rewards[i]);
        if (az > 1.0f) atomicAdd(&g_znclip[0], 1.0);
        if (az > 3.0f) atomicAdd(&g_znclip[1], 1.0);
        if (az > 5.0f) atomicAdd(&g_znclip[2], 1.0);
    }
    float st = (float)envs[i].step;
    int done = (envs[i].step >= envs[i].max_steps);
    terminals[i] = done ? 1.0f : 0.0f;
    Env* e = &envs[i];
    const double* rp = rparts + (size_t)i * SB_NPARTS_ENV;
    e->ep_ret += r;
    epret_d[i] += rew[i];         /* double: the per-scenario episode records (eval JSON, live log) */
    for (int k = 0; k < SB_NPARTS_ENV; k++) e->ep_part[k] += (float)rp[k];
    if ((float)rp[4] > e->ep_worst) e->ep_worst = (float)rp[4];
    atomicAdd(&g_live[0], 1.0);
    atomicAdd(&g_live_grp[gidx][0], (double)r);
    atomicAdd(&g_live_grp[gidx][1], rp[3]);
    atomicAdd(&g_live_grp[gidx][2], 1.0);
    atomicAdd(&g_live_grp[gidx][3], (double)r * (double)r);
    atomicAdd(&g_live[1], (double)r);
    for (int k = 0; k < SB_NPARTS_ENV; k++) atomicAdd(&g_live[2 + k], rp[k]);
    if (detail && gidx >= 0 && gidx < SB_MAX_BLOBS) {
        const GpuSim* sim = &sims[i];
        const int kv = calendar_step[i];
        const int hour = (int)sim->rew_hour[kv];
        const bool weekday = sim->rew_wd[kv] != 0;
        const int time_bin = (hour < 6 || hour >= 19) ? 0 :
            (!weekday ? 4 : (hour < 12 ? 1 : (hour < 16 ? 2 : 3)));
        double occ = 0.0;
        for (int z = 0; z < sim->n_zones; ++z)
            occ += occupancy[(size_t)i * sim->n_zones + z];
        const int occ_bin = occ > 0.0 ? 6 : 5;
        const int parts[7] = {0, 1, 2, 3, 5, 6, 7};
        for (int j = 0; j < 2; ++j) {
            const int bin = j == 0 ? time_bin : occ_bin;
            atomicAdd(&g_detail_scen[gidx][bin][0], 1.0);
            for (int k = 0; k < 7; ++k)
                atomicAdd(&g_detail_scen[gidx][bin][1+k],
                    rp[parts[k]] * (k >= 5 ? 3600.0 / sim->dt_sec : 1.0));
        }
    }
    if (done) {
        /* Only whole episodes enter the episode log: with desynchronised starts an env's first
         * episode is cut short at a random step (a truncation, not a real episode). */
        if (st >= (float)ep_len) {
            float mean = e->ep_ret / st;
            const double mean_d = epret_d[i] / (double)st;
            /* per scenario: every full episode (windowed, read by puf_live_log) and, separately, each env's FIRST
             * full episode (the only one equivalent to a fresh gateway episode: read by the eval dump) */
            atomicAdd(&g_ep_scen[gidx][0], mean_d);
            atomicAdd(&g_ep_scen[gidx][1], 1.0);
            if (random_start && gidx >= 0 && gidx < SB_MAX_BLOBS) {
                const int d = e->start_day;
                atomicAdd(&g_ep_start[gidx][d][0], 1.0);
                atomicAdd(&g_ep_start[gidx][d][1], mean_d);
                for (int k = 0; k < 3; ++k)
                    atomicAdd(&g_ep_start[gidx][d][2 + k], (double)e->ep_part[k] / st);
                atomicAdd(&g_ep_start[gidx][d][5], (double)e->ep_part[6]);
                atomicAdd(&g_ep_start[gidx][d][6], (double)e->ep_part[7]);
            }
            if (epcount[i] == 0) {
                atomicAdd(&g_first_scen[gidx][0], mean_d);
                for (int k = 0; k < 3; k++) atomicAdd(&g_first_scen[gidx][1 + k], (double)(e->ep_part[k] / st));
                atomicAdd(&g_first_scen[gidx][4], 1.0);
            }
            epcount[i] += 1;
            e->log.perf += mean;
            e->log.score += mean;
            e->log.episode_return += e->ep_ret;
            e->log.episode_length += st;
            e->log.productivity += e->ep_part[0] / st;
            e->log.energy_cost_norm += e->ep_part[1] / st;
            e->log.carbon_norm += e->ep_part[2] / st;
            e->log.comfort_dev_K += e->ep_part[3] / st;
            e->log.worst_dev_K += e->ep_worst;
            e->log.out_of_band_frac += e->ep_part[5] / st;
            e->log.elec_kwh += e->ep_part[6];
            e->log.gas_kwh += e->ep_part[7];
            e->log.n += 1.0f;
        }
        e->max_steps = ep_len;          /* every later episode has the scenario's full length */

        e->ep_ret = 0.0f;
        epret_d[i] = 0.0;
        for (int k = 0; k < SB_NPARTS_ENV; k++) e->ep_part[k] = 0.0f;
        e->ep_worst = 0.0f;
        reset[1 + i] = 1;
        reset[0] = 1;
    }
    ep_ret_out[i] = e->ep_ret;   /* plain device copy for the snapshot (host must not touch managed Env) */
}

/* Managed memory may not fault inside a captured stream, so every 4 KB page of
 * the Env batch is touched from the device once before the rollout is captured. */
__global__ void touch_envs_k(Env* envs, size_t nbytes) {
    size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 4096;
    if (i >= nbytes) return;
    volatile unsigned char* p = (volatile unsigned char*)envs;
    p[i] = p[i];
}
#endif
/* The step in three phases around the two halves of the triangular solve, so a batch of several
 * groups can run one fused solve (lu_solve_mg) for all of them:
 *   pre  (state -> forcing in the solver layout) | forward solve
 *   mid  (air side: sensed values off z, q)      | backward solve (starts from z + Z q)
 *   post (field, sensing, convection, observation, occupancy, reward, episode restart) */
static int sb_step_pre(void) {
    CU(cudaMemsetAsync(d_reset, 0, 4 * (size_t)(nenv + 1), stream));
    if (cv2_ok) {
        /* the producer needs nothing from this step's state: launch it now, on the
         * side stream, so its serial chain hides behind the FP64 solve */
        CU(cudaEventRecord(ev_app, stream));
        CU(cudaStreamWaitEvent(stream2, ev_app, 0));
        if (cv2_global)
            convect2g_k<<<cv_ns, 256, cv2_smem, stream2>>>(sh_roo, sh_rcc, sh_coo, sh_cii,
                n_rooms_c, nc, max_room, conv_p, conv_dist, d_rngc, d_cvdst, d_cvsrc, d_cvcnt,
                cv_cap, d_cv_lr, d_cv_p);
        else
            convect2_k<<<cv_ns, 256, cv2_smem, stream2>>>(sh_roo, sh_rcc, sh_coo, sh_cii,
                n_rooms_c, nc, max_room, conv_p, conv_dist, d_rngc, d_cvdst, d_cvsrc, d_cvcnt,
                cv_cap, NULL, NULL);
        CU(cudaEventRecord(ev_prod, stream2));
    }
    seq_k<<<nenv, 32, 0, stream>>>(d_sims, d_V, d_native, n_action, 1);
    /* avg_temp of the field at this step's start is the one the previous step's
     * tail (or the episode reset) already computed: the field is unchanged since. */
    seq_k<<<nenv, 32, 0, stream>>>(d_sims, d_V, d_native, n_action, 0);
    zone_qdiv_k<<<(nenv * n_zones + 127) / 128, 128, 0, stream>>>(d_sims, nenv);
    fdm_forcing_k<<<(unsigned)(((size_t)n * lu_ld + 255) / 256), 256, 0, stream>>>(
        d_sims, sd_i, sd_cap, d_exdir, d_envw, sd_mc, d_rowpos, n, nenv, lu_ld, d_lu_work);
    return 0;
}
static void sb_step_mid(void) {
    if (k > 0) {
        sensed_R_part_k<<<dim3(n_rseg, lu_ld / 32), 32, 0, stream>>>(d_rseg_p, d_rrc, d_rrx, nnz_r, d_envw,
                                                                    d_lu_work, lu_ld, d_rpart);
        sensed_R_sum_k<<<(k * nenv + 127) / 128, 128, 0, stream>>>(d_sims, d_rrow_seg, d_rpart, nenv, k, lu_ld);
        if (air_global)
            airside_gk<<<nenv, 256, smem, stream>>>(d_sims, sd_sp, sd_si, sd_spx, sd_ap, sd_ai, sd_apx,
                                                   d_sr, d_envw, k, lu_ld, d_qair, d_airM);
        else
            airside_k<<<nenv, 256, smem, stream>>>(d_sims, sd_sp, sd_si, sd_spx, sd_ap, sd_ai, sd_apx,
                                                  d_sr, d_envw, k, lu_ld, d_qair, NULL);
    }
}
static int sb_step_post(void) {
    fill_amb_k<<<(unsigned)(((size_t)nenv * n_nonsolved + 255) / 256), 256, 0, stream>>>(
        d_sims, d_nonsolved, n_nonsolved, nenv);
    lu_out_k<<<(unsigned)(((size_t)n * lu_ld + 255) / 256), 256, 0, stream>>>(
        d_sims, d_lu_z, d_pc, sd_i, sd_ret, sd_gain, n, nenv, lu_ld, d_temp, d_mass_temp);
    sensed_mean_k<<<(nenv * k + 31) / 32, 32, 0, stream>>>(d_sims, sh_tp, sh_ti, sh_tx, nenv, k);
    post_seq_k<<<nenv, 32, 0, stream>>>(d_sims, nenv);
    if (cv2_ok) {
        CU(cudaStreamWaitEvent(stream, ev_prod, 0));
        if (cv_ns == 1) {
            conv_gather_sh_k<<<328, 256, 0, stream>>>(d_temp, lu_ld, nenv, d_cvsrc, d_cvcnt, d_cvscr);
            conv_scatter_sh_k<<<328, 256, 0, stream>>>(d_temp, lu_ld, nenv, d_cvdst, d_cvcnt, d_cvscr);
        } else {   /* SB_CONV_PER_ENV: a list per env (diagnostic) */
            conv_gather_k<<<dim3(32, nenv), 256, 0, stream>>>(d_temp, lu_ld, d_cvsrc, d_cvcnt, cv_cap,
                                                             0, d_cvscr, cv_cap);
            conv_scatter_k<<<dim3(32, nenv), 256, 0, stream>>>(d_temp, lu_ld, d_cvdst, d_cvcnt, cv_cap,
                                                              0, d_cvscr, cv_cap);
        }
        cudaError_t e_ = cudaGetLastError();
        if (e_ != cudaSuccess)
            fprintf(stderr, "convect2 launch: %s (smem=%zu)\n", cudaGetErrorString(e_),
                    cv2_smem);
    } else {
        const size_t csmem = (size_t)max_room * sizeof(double);
        CU(cudaMemsetAsync(d_consumed, 0, 4 * (size_t)nenv, stream));
        conv_gen_k<<<nenv, 256, 0, stream>>>(d_rngc, d_words, conv_words);
        convect_k<<<nenv, 256, csmem, stream>>>(sh_roo, sh_rcc, sh_coo, sh_cii,
                                               n_rooms_c, nc, ncv, max_room, conv_p,
                                               conv_dist, d_temp, d_left, d_right,
                                               d_words, conv_words, d_consumed, lu_ld);
        conv_advance_k<<<nenv, 256, 0, stream>>>(d_rngc, d_consumed);
        {
            cudaError_t e_ = cudaGetLastError();
            if (e_ != cudaSuccess)
                fprintf(stderr, "conv launch: %s (csmem=%zu max_room=%d)\n",
                        cudaGetErrorString(e_), csmem, max_room);
        }
    }
    avg_temp_k<<<(nenv * n_zones + 31) / 32, 32, 0, stream>>>(d_sims, nenv, d_coff, NULL);
    obs_dev_k<<<(nenv * n_dev_ + 127) / 128, 128, 0, stream>>>(d_sims, nenv, d_stepv,
                                                               g_obs, NULL);
    obs_aux_k<<<(nenv * n_aux_ + 127) / 128, 128, 0, stream>>>(d_sims, d_V, nenv,
                                                               d_stepv, g_obs, NULL);
    zm_update();
    occup_k<<<nenv, 32, 0, stream>>>(sh_hour, hour_len, sh_wd, d_envw, wd_len, d_stepv, n_zones, nassign,
                                     occ_ea, occ_la, occ_ed, occ_ld, p_arr, p_dep,
                                     d_occstate, d_V, d_rngo, NULL);
    if (tmean_needed) tmean_k<<<nenv, 256, 2048, stream>>>(d_sims, d_tmean);
    reward_k<<<nenv, 32, 0, stream>>>(d_sims, d_tmean, d_V, nenv, d_stepv,
                                      g_comfort_shape > 0.0f ? d_phi : NULL,
                                      g_comfort_shape > 0.0f ? d_shape : NULL,
                                      g_comfort_shape, g_shape_gamma);
    zm_zone_share();
    zm_zone_reward();
    zm_zone_check();
#ifndef SB_TEST
    if (g_envs != NULL) {
        static int detail = -1;
        if (detail < 0) {
            const char* flag = getenv("SB_DETAIL_METRICS");
            detail = flag != NULL && strcmp(flag, "1") == 0;
            if (detail) fprintf(stderr, "smart_buildings: scenario/time/occupancy detail metrics on\n");
        }
        finalize_k<<<(nenv + 127) / 128, 128, 0, stream>>>(g_envs, d_rew, d_rparts, nenv, g_rewards,
                                                           g_terminals, d_reset, d_epret, d_envscen,
                                                           g_ep_len, g_rew_scale, d_epcount, d_epretd,
                                                           (g_rew_norm > 0.5f || g_rew_center > 0.5f) ? d_rew_norm : NULL,
                                                           g_comfort_shape > 0.0f ? d_shape : NULL,
                                                           d_sims, d_V, d_stepv, detail, g_random_start);
#ifdef SB_ENTITY
        entity_scatter_k<<<(nenv + 63) / 64, 64, 0, stream>>>(d_cmap, g_rewards, g_terminals, g_rew_fw, g_term_fw, nenv);
#endif
        sb_enqueue_reset(d_reset);
    }
#else
    if (g_test_ep_len > 0) {
        /* parity gate: restart after g_test_ep_len steps through the same in-step
         * path the trainer uses (flag set on the device, gated reset kernels) */
        test_done_k<<<1, 1, 0, stream>>>(d_stepv, g_test_ep_len, d_reset, nenv);
        sb_enqueue_reset(d_reset);
    }
#endif
    sb_canon_tail();
    return 0;
}
/* One group's whole step on `stream`.  Forward solve z = L^-1 P_r b; the air side reads its sensed
 * values off z (R z) and solves for q; the backward solve starts from z + Z q, so one triangular
 * solve carries the oracle's second (correction) solve.  The two halves take their tickets from
 * separate counters. */
static int sb_gpu_run_step(void) {
    if (sb_step_pre() != 0) return 1;
    lu_prep_k<<<1, 1, 0, stream>>>(d_lu_ep, d_lu_tk, d_lu_fl, d_lu_fu, n, 1);
    const int lub = sb_lu_blocks();
    const int lub_t = sb_lu_tpb();
    lu_launch_range(lu_rpl, g_lu, d_lu_work, d_lu_z, lu_ld, d_lu_fl, d_lu_fu, d_lu_tk, d_lu_ep, NULL,
                    lub, lub_t, stream, 0u, (unsigned)n, NULL);
    sb_step_mid();
    lu_launch_range(lu_rpl, g_lu, d_lu_work, d_lu_z, lu_ld, d_lu_fl, d_lu_fu, d_lu_tk + 1, d_lu_ep, NULL,
                    lub, lub_t, stream, (unsigned)n, 2u * (unsigned)n, k > 0 ? d_qair : NULL);
    return sb_step_post();
}


/* ------------------------------------------------------------- snapshot --- */
/* Host-side, outside any capture: copy one env's visible state to a file the viewer
 * (tools/watch.sh) reads.  Layout (little-endian), parsed by viewer/sb_viewer.py:
 *   "SBS1", i32[12] {version=1, step, max_steps, env, nz, nvav, nahu, rows, cols, ncv,
 *   n_action, 0}, f64[12] {reward, episode_return, boiler_setpoint, boiler_return,
 *   rparts[8]}, char blob[512], f32 field[ncv] (K, display precision), f64 zone_temp[nz],
 *   occ[nz], zo_sp[nz], zo_act[nz], f64 vav_heatsp/coolsp/damper/reheat[nvav],
 *   f64 ahu_sat/flow[nahu], i32 ahu_run[nahu], f64 native[n_action].
 * Written to <path>.tmp then renamed, so a reader never sees a torn file. */
/* envs e0..e0+ne-1 of the field, env-major [e][ncv] (the device keeps it [cell][lu_ld]) */
static void sb_field_get(std::vector<double>& out, int e0, int ne) {
    std::vector<double> all((size_t)lu_ld * ncv);
    cudaMemcpy(all.data(), d_temp, 8 * all.size(), cudaMemcpyDeviceToHost);
    out.assign((size_t)ne * ncv, 0.0);
    for (int e = 0; e < ne; e++)
        for (int p = 0; p < ncv; p++) out[(size_t)e * ncv + p] = all[(size_t)p * lu_ld + e0 + e];
}
/* SB_RECORD companion: each env's reward parts of the step just taken (productivity, energy_cost, carbon,
 * comfort_dev_K, worst_dev_K, out_of_band, elec_kwh, gas_kwh), env-major.  One group only: with several,
 * d_rparts is per group.  Returns the number of doubles written, or -1. */
extern "C" int sb_rec_parts(double* host, int capacity) {
    if (g_groups.size() > 1 || (long)nenv * SB_NPARTS > capacity) return -1;
    cudaStreamSynchronize(stream);
    if (cudaMemcpy(host, d_rparts, 8 * (size_t)nenv * SB_NPARTS, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return nenv * SB_NPARTS;
}
template <typename T>
static void snap_get(std::vector<unsigned char>& out, const T* dev, size_t n) {
    size_t off = out.size();
    out.resize(off + n * sizeof(T));
    if (n) cudaMemcpy(out.data() + off, dev, n * sizeof(T), cudaMemcpyDeviceToHost);
}
static int sb_snapshot_write(int e, const char* path) {
    if (e < 0 || e >= nenv) return 1;
    cudaStreamSynchronize(stream);
    GpuSim h;
    if (cudaMemcpy(&h, d_sims + e, sizeof(GpuSim), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    int step = 0;
    cudaMemcpy(&step, d_stepv + e, 4, cudaMemcpyDeviceToHost);
    step -= h.start_index;   /* snapshot shows elapsed steps, not calendar-table index */
    double rew = 0.0, rp[SB_NPARTS];
    float epr = 0.0f;
    cudaMemcpy(&epr, d_epret + e, 4, cudaMemcpyDeviceToHost);
    double ep_ret = epr;
    cudaMemcpy(&rew, d_rew + e, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(rp, d_rparts + (size_t)e * SB_NPARTS, 8 * SB_NPARTS, cudaMemcpyDeviceToHost);
    std::vector<unsigned char> o;
    o.insert(o.end(), {'S', 'B', 'S', '1'});
    int ints[12] = {1, step, g_ep_len, e, n_zones, n_vav, n_ahu, nr, nc, ncv, n_action, 0};
    o.insert(o.end(), (unsigned char*)ints, (unsigned char*)ints + sizeof(ints));
    double sc[12] = {rew, ep_ret, h.boiler_setpoint, h.boiler_return};
    for (int i = 0; i < SB_NPARTS; i++) sc[4 + i] = rp[i];
    o.insert(o.end(), (unsigned char*)sc, (unsigned char*)sc + sizeof(sc));
    char bp[512] = {0};
    snprintf(bp, sizeof(bp), "%s", w_paths[h_envw[e]]);   /* the env's own scenario */
    o.insert(o.end(), (unsigned char*)bp, (unsigned char*)bp + sizeof(bp));
    std::vector<float> fld(ncv);
    {
        std::vector<double> t(ncv);
        cudaMemcpy2D(t.data(), 8, h.temp, 8 * (size_t)h.ts, 8, ncv, cudaMemcpyDeviceToHost);
        for (int i = 0; i < ncv; i++) fld[i] = (float)t[i];
    }
    o.insert(o.end(), (unsigned char*)fld.data(), (unsigned char*)fld.data() + 4 * (size_t)ncv);
    snap_get(o, h.avg_temp, n_zones);
    snap_get(o, d_V + (size_t)e * n_zones, n_zones);
    snap_get(o, h.zo_sp, n_zones);
    snap_get(o, h.zo_act, n_zones);
    snap_get(o, h.vav_heatsp, n_vav);
    snap_get(o, h.vav_coolsp, n_vav);
    snap_get(o, h.vav_damper, n_vav);
    snap_get(o, h.vav_reheat, n_vav);
    snap_get(o, h.ahu_sat, n_ahu);
    snap_get(o, h.ahu_flow, n_ahu);
    snap_get(o, h.ahu_run, n_ahu);
    snap_get(o, d_native + (size_t)e * n_action, n_action);
    char tmp[1100];
    snprintf(tmp, sizeof(tmp), "%s.tmp", path);
    FILE* fp = fopen(tmp, "wb");
    if (fp == NULL) return 1;
    size_t wr = fwrite(o.data(), 1, o.size(), fp);
    int cl = fclose(fp);
    if (wr != o.size() || cl != 0) { remove(tmp); return 1; }
    return rename(tmp, path) != 0;
}
/* SB_SNAPSHOT=<file> turns it on; SB_SNAP_ENV (default 0) picks the env; SB_SNAP_EVERY
 * (default 1) writes every Nth step. */
struct SnapCfg { const char* path; int env, every; };
static SnapCfg snap_cfg(void) {
    static SnapCfg c = {NULL, 0, 1};
    static int init = 0;
    if (!init) {
        init = 1;
        const char* p = getenv("SB_SNAPSHOT");
        if (p != NULL && p[0] != '\0') c.path = p;
        if (getenv("SB_SNAP_ENV")) c.env = atoi(getenv("SB_SNAP_ENV"));
        if (getenv("SB_SNAP_EVERY")) c.every = atoi(getenv("SB_SNAP_EVERY"));
        if (c.every < 1) c.every = 1;
    }
    return c;
}

static void sb_mg_setup(void) {
    g_mg_on = false;
    /* opt-in (SB_FUSE=1): bit-identical, but measured slower in the trainer than the groups' own
     * concurrent solves (960 agents: 10.8K vs 11.1K SPS); kept for experiments and the mgtest gate */
    if (g_groups.size() < 2 || (getenv("SB_FUSE") == NULL && !g_mg_force)) return;
    std::vector<LuGrp> gr;
    std::vector<std::array<int, 3>> f, b;       /* (level, group, group-local ticket) */
    std::vector<std::array<int, 2>> depth;      /* deepest forward / backward level per group */
    int rpl0 = -1;
    int mw = 0;            /* the fused launch's value mode: the most general any group needs */
    bool any_mixed = false;
    for (size_t gi = 0; gi < g_groups.size(); gi++) {
        sb_group_load(g_groups[gi].st);
        if (rpl0 < 0) rpl0 = lu_rpl;
        if (lu_rpl != rpl0) {
            fprintf(stderr, "smart_buildings: groups differ in envs per lane: separate solves\n");
            return;
        }
        if (nw > 1) { if (g_lu.slab_uniform) mw = 2; else any_mixed = true; }
        gr.push_back({g_lu, d_lu_work, d_lu_z, d_lu_fl, d_lu_fu, k > 0 ? d_qair : NULL, lu_ld});
        /* the fused solve runs its own epochs: no flag may hold a value of one of them */
        cudaMemset(d_lu_fl, 0, 4 * (size_t)n);
        cudaMemset(d_lu_fu, 0, 4 * (size_t)n);
        /* Order: SB_MG_ORDER=1 (default) aligns the groups' DAGs at their END (key = level + the
         * difference between the deepest group's depth and this group's), so the deepest group
         * starts first and every group's narrow tail overlaps the others' (critical path first);
         * 0 aligns them at the start (plain level).  Either keeps a row's dependencies (lower level,
         * same group) on earlier tickets. */
        int mxf = 0, mxb = 0;
        for (int t = 0; t < n; t++) { mxf = std::max(mxf, h_tlev[t]); mxb = std::max(mxb, h_tlev[n + t]); }
        depth.push_back({mxf, mxb});
        for (int t = 0; t < n; t++) f.push_back({h_tlev[t], (int)gi, t});
        for (int t = 0; t < n; t++) b.push_back({h_tlev[n + t], (int)gi, n + t});
    }
    const int mode = getenv("SB_MG_ORDER") ? atoi(getenv("SB_MG_ORDER")) : 1;
    if (mode == 1) {
        int Df = 0, Db = 0;
        for (auto& d : depth) { Df = std::max(Df, d[0]); Db = std::max(Db, d[1]); }
        for (auto& x : f) x[0] += Df - depth[x[1]][0];
        for (auto& x : b) x[0] += Db - depth[x[1]][1];
    }
    if (any_mixed) mw = 1;     /* per-lane values work for every group */
    std::sort(f.begin(), f.end());
    std::sort(b.begin(), b.end());
    std::vector<int2> of(f.size()), ob(b.size());
    for (size_t i = 0; i < f.size(); i++) of[i] = make_int2(f[i][1], f[i][2]);
    for (size_t i = 0; i < b.size(); i++) ob[i] = make_int2(b[i][1], b[i][2]);
    if (gr.size() > LU_MG_MAX) { fprintf(stderr, "smart_buildings: more than %d groups: separate solves\n", LU_MG_MAX); return; }
    cudaMemcpyToSymbol(c_lugrp, gr.data(), sizeof(LuGrp) * gr.size());
    cudaMalloc(&d_mg_ordF, sizeof(int2) * of.size());
    cudaMemcpy(d_mg_ordF, of.data(), sizeof(int2) * of.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_mg_ordB, sizeof(int2) * ob.size());
    cudaMemcpy(d_mg_ordB, ob.data(), sizeof(int2) * ob.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_mg_ep, 4); cudaMemset(d_mg_ep, 0, 4);
    cudaMalloc(&d_mg_tk, 8); cudaMemset(d_mg_tk, 0, 8);
    n_mgF = (unsigned)of.size(); n_mgB = (unsigned)ob.size();
    g_mg_rpl = rpl0; g_mg_mw = mw;
    for (auto& g : g_groups) {
        cudaEventCreateWithFlags(&g.ev_a, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&g.ev_b, cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&g_fork, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_fork2, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_fork3, cudaEventDisableTiming);
    cudaDeviceSynchronize();
    g_mg_on = true;
    fprintf(stderr, "smart_buildings: fused solve over %zu groups (%u + %u tickets, %d envs per lane%s)\n",
            g_groups.size(), n_mgF, n_mgB, rpl0, mw == 2 ? ", weather per slab" : (mw == 1 ? ", weather per lane" : ""));
}

/* The solve's layout is what sets the step's cost (docs/HISTORY.md, "the step is the wall"): its state
 * loads are one value per (term, env), so the step is bandwidth-bound once each term's work is spread
 * over enough columns.  Value mode 2 = one weather per 32-column slab (the fast path); mode 1 =
 * weathers share slabs, so a term's constant is loaded per lane instead of shuffled; mode 0 = one
 * weather.  >= 32 envs PER WEATHER is what keeps mode 2 (sb_gpu_setup_w). */
static void sb_print_solve_layout(const char* label, int envs, int nscen) {
    const int mw = nw > 1 ? (g_lu.slab_uniform ? 2 : 1) : 0;
    fprintf(stderr, "smart_buildings: %s: %d envs, %d scenario(s), ld %d, %d env(s)/lane, value mode %d%s\n",
            label, envs, nscen, lu_ld, lu_rpl, mw,
            (nw > 1 && mw == 1) ? " (weathers share 32-env slabs: per-lane values)" : "");
    if (envs >= 32 && lu_rpl < 5)
        fprintf(stderr, "smart_buildings:   %d env(s)/lane (< 5): the solve amortises each term over too few "
                        "columns; prefer >= 160 envs per group%s\n", lu_rpl,
                nw > 1 ? " (>= 32 per weather)" : "");
}

/* One step of every group with the fused solve, enqueued on main_stream (graph-capturable):
 * every group's pre phase on its own stream, one fused forward solve, every group's air side, one
 * fused backward solve, every group's post phase.  decode: run the trainer's action decode first. */
static void sb_mg_step(cudaStream_t main_stream, bool decode) {
    /* phase-wise: every group's pre phase on its own stream, one fused forward solve, every
     * group's air side, one fused backward solve, every group's post phase */
    auto fork = [&](cudaEvent_t ev, auto&& body, bool rec_a) {
        cudaEventRecord(ev, main_stream);
        for (auto& g : g_groups) {
            sb_group_load(g.st);
            zm_base = g.off;
            stream = g.s;
            cudaStreamWaitEvent(stream, ev, 0);
            body();
            cudaEventRecord(rec_a ? g.ev_a : g.done, g.s);
        }
        stream = main_stream;
        for (auto& g : g_groups) cudaStreamWaitEvent(main_stream, rec_a ? g.ev_a : g.done, 0);
    };
    fork(g_fork, [&]() {
#ifndef SB_TEST
        if (decode)
            sb_decode();
#else
        (void)decode;
#endif
        if (sb_step_pre() != 0) abort();
    }, true);
    lu_prep_k<<<1, 1, 0, main_stream>>>(d_mg_ep, d_mg_tk, NULL, NULL, 0, 1);
    lu_launch_mg(g_mg_rpl, g_mg_mw, d_mg_ordF, d_mg_tk, d_mg_ep, n_mgF, sb_lu_blocks(), 128, main_stream);
    fork(g_fork2, [&]() { sb_step_mid(); }, true);
    lu_launch_mg(g_mg_rpl, g_mg_mw, d_mg_ordB, d_mg_tk + 1, d_mg_ep, n_mgB, sb_lu_blocks(), 128, main_stream);
    fork(g_fork3, [&]() { if (sb_step_post() != 0) abort(); }, false);
    stream = main_stream;
}
#ifndef SB_TEST
/* ------------------------------------------------------------ PufferLib env --- */


static char g_blob_path[SB_PATH_MAX];

/* Order the blobs so a prefix of any length spans as many buildings as possible: all buildings'
 * first weather, then all buildings' second weather, ...  (name = <building>__<weather>.sbb). */
static void order_round_robin(char (*p)[SB_PATH_MAX], int nf) {
    std::vector<std::string> keys, out;
    std::vector<std::string> blds;
    for (int i = 0; i < nf; i++) {
        std::string f = p[i];
        std::string base = f.substr(f.find_last_of('/') + 1);
        std::string b = base.substr(0, base.find("__"));
        keys.push_back(b);
        bool seen = false;
        for (auto& x : blds) seen |= (x == b);
        if (!seen) blds.push_back(b);
    }
    std::vector<int> used(nf, 0);
    for (int round = 0; (int)out.size() < nf; round++)
        for (auto& b : blds) {
            int hit = 0;
            for (int i = 0; i < nf; i++) {
                if (used[i] || keys[i] != b) continue;
                if (hit == 0) { out.push_back(p[i]); used[i] = 1; }
                hit++;
            }
        }
    for (int i = 0; i < nf; i++) snprintf(p[i], SB_PATH_MAX, "%s", out[i].c_str());
}

/* Refuse old or incomplete blobs before any random index reaches a kernel. */
static bool sb_random_start_blob_ok(const SbBlob* b, const char* path) {
    const SbSection* offsets = sb_find(b, "random_start_offsets");
    const int64_t* off = sb_i64(b, "random_start_offsets");
    const double* meta = sb_f64(b, "obs_meta");
    const SbSection* metas = sb_find(b, "obs_meta");
    const double* op = sb_f64(b, "occ_params");
    const SbSection* ops = sb_find(b, "occ_params");
    if (!offsets || offsets->count != 7 || !off || !meta || !metas || metas->count < 4 ||
        !op || !ops || ops->count < 6 || op[5] != 300.0 || meta[3] < 1.0) {
        fprintf(stderr, "smart_buildings: random_start requires a seven-day random-start blob with five-minute steps: %s\n", path);
        return false;
    }
    for (int d = 0; d < 7; ++d) if (off[d] != d * 288) {
        fprintf(stderr, "smart_buildings: random_start offsets malformed: %s\n", path);
        return false;
    }
    const uint64_t rows = (uint64_t)meta[3] + 6 * 288 + 1;
    const char* names[] = {"weather_ambient", "sched_comfort", "sched_heat_sp", "sched_cool_sp",
        "occ_hour", "occ_workday", "occ_hour_m1", "occ_workday_m1",
        "rew_hour", "rew_month", "rew_workday", NULL};
    for (int i = 0; names[i] != NULL; ++i) {
        const SbSection* s = sb_find(b, names[i]);
        if (!s || s->count < rows) {
            fprintf(stderr, "smart_buildings: random_start section %s is too short in %s\n", names[i], path);
            return false;
        }
    }
    const int64_t na = sb_i64_1(b, "n_aux", 0);
    const SbSection* aux = sb_find(b, "aux_time");
    if (na < 1 || !aux || aux->count < rows * (uint64_t)(na - 1)) {
        fprintf(stderr, "smart_buildings: random_start auxiliary calendar is too short in %s\n", path);
        return false;
    }
    return true;
}

Env* puf_vec_create(int n, Dict* env_kwargs, obs_t* observations, float* actions,
                    float* rewards, float* terminals) {
#ifdef SB_ENTITY
    /* The trainer counts agents.  Each building env owns SB_NENT zone agents, packed contiguously. */
    if (n % SB_NENT != 0) {
        fprintf(stderr, "smart_buildings: entity mode needs total_agents divisible by %d (zone slots)\n", SB_NENT);
        abort();
    }
    n /= SB_NENT;
#endif
    /* ---- which scenarios: env.blob (one) or env.blob_dir (every .sbb in it, one group each) ---- */
    static char paths[SB_MAX_BLOBS][SB_PATH_MAX];
    int nf = 0;
    DictItem* one = dict_find(env_kwargs, "blob");
    if (one != NULL && one->str != NULL && one->str[0] != '\0') {
        snprintf(paths[0], SB_PATH_MAX, "%s", one->str);
        nf = 1;
    } else {
        DictItem* di = dict_find(env_kwargs, "blob_dir");
        const char* dir = (di != NULL && di->str != NULL && di->str[0] != '\0')
                              ? di->str : getenv("SB_BLOB_DIR");
        nf = (dir != NULL && dir[0] != '\0') ? list_blobs(dir, paths) : 0;
        if (nf == 0) {
            fprintf(stderr, "smart_buildings: no scenario. Set env.blob or env.blob_dir "
                            "(or SB_BLOB_DIR).\n");
            abort();
        }
        order_round_robin(paths, nf);
    }
    /* scenarios: env.groups > 0 fixes how many (a prefix of the building-first order); 0 = as many
     * as fit with >= 32 envs each */
    DictItem* gk = dict_find(env_kwargs, "groups");
    int G = (gk != NULL) ? (int)gk->value : 0;
    if (G <= 0) G = n / 32;
    if (G > nf) G = nf;
    if (G < 1) G = 1;
    if (G > n) G = n;
    std::vector<int> scnt(G);
    for (int i = 0; i < G; i++) scnt[i] = n / G + (i < n % G ? 1 : 0);
    /* Weathers of one building share a group (one solve with full warps, one geometry): scenarios
     * whose blobs differ only in the per-weather sections are merged, found by comparing the blobs
     * (nothing is keyed on names).  env.merge_weathers = 0 keeps one group per scenario.  A group
     * holds at most SB_NE envs (the solver's lane limit). */
    DictItem* mk = dict_find(env_kwargs, "merge_weathers");
    const bool merge = (mk == NULL) || ((int)mk->value != 0);
    std::vector<std::vector<int>> clusters;
    {
        std::vector<SbBlob> bases;
        std::vector<int> csize;
        for (int i = 0; i < G; i++) {
            SbBlob bi;
            if (sb_open(&bi, paths[i]) != 0) { fprintf(stderr, "smart_buildings: cannot open %s\n", paths[i]); abort(); }
            int hit = -1;
            if (merge)
                for (size_t c = 0; c < clusters.size() && hit < 0; c++) {
                    const char* why = NULL;
                    if (csize[c] + scnt[i] <= SB_NE && sb_weather_compatible(&bases[c], &bi, &why)) hit = (int)c;
                }
            if (hit >= 0) { clusters[hit].push_back(i); csize[hit] += scnt[i]; sb_close(&bi); }
            else { clusters.push_back({i}); csize.push_back(scnt[i]); bases.push_back(bi); }
        }
        for (auto& b : bases) sb_close(&b);
    }
    const int NG = (int)clusters.size();
    for (int c = 0; c < NG; c++) {
        int tot = 0;
        for (int i : clusters[c]) tot += scnt[i];
        if (tot < 32 && NG > 1)
            fprintf(stderr, "smart_buildings: warning: group %d has %d envs (< 32): the solve wastes warp lanes\n", c, tot);
    }
    g_scen_names.clear();
    for (int i = 0; i < G; i++) {
        const char* b0 = strrchr(paths[i], '/');
        std::string nm = b0 ? b0 + 1 : paths[i];
        size_t dot = nm.find(".sbb");
        if (dot != std::string::npos) nm = nm.substr(0, dot);
        g_scen_names.push_back(nm);
    }
    /* managed: the trainer may see this array as the batch base */
    Env* envs = NULL;
    cudaMallocManaged((void**)&envs, sizeof(Env) * n);
    memset(envs, 0, sizeof(Env) * n);
    /* The GPU vector never goes through puf_init per env: the episode length must be set
     * here or every step is terminal.  0 = the scenario's own length; env.max_steps
     * shortens it (smoke runs, curriculum). */
    DictItem* ms = dict_find(env_kwargs, "max_steps");
    /* env.desync = 1: every env's FIRST episode ends at a random step in [1, episode length], so the
     * envs of a batch spread over the whole calendar (an epoch then sees all times of day / days of
     * the week instead of the same few hours everywhere).  Later episodes have the full length;
     * the cut first episodes are truncations and stay out of the episode log. */
    DictItem* dk = dict_find(env_kwargs, "desync");
    DictItem* random_k = dict_find(env_kwargs, "random_start");
    g_random_start = random_k != NULL && (int)random_k->value != 0;
    const bool desync = !g_random_start && dk != NULL && (int)dk->value != 0;
    DictItem* seed_k = dict_find(env_kwargs, "random_start_seed");
    const unsigned long long start_seed = seed_k ? (unsigned long long)(long long)seed_k->value : 0ULL;
    if (g_random_start)
        fprintf(stderr, "smart_buildings: random_start on, seven start-day offsets; desync disabled; calendar seed %llu\n", start_seed);
    DictItem* rk = dict_find(env_kwargs, "reward_scale");
    g_rew_scale = (rk != NULL && rk->value > 0.0) ? (float)rk->value : 1.0f;
    {
        DictItem* x = dict_find(env_kwargs, "rew_norm");
        g_rew_norm = (x != NULL && (int)x->value != 0) ? 1.0f : 0.0f;
        x = dict_find(env_kwargs, "rew_center");
        g_rew_center = (x != NULL && (int)x->value != 0) ? 1.0f : 0.0f;
        x = dict_find(env_kwargs, "center_beta");
        if (x != NULL && x->value > 0.0 && x->value <= 1.0) g_center_beta = (float)x->value;
        if (g_rew_norm > 0.5f && g_rew_center > 0.5f) {
            fprintf(stderr, "smart_buildings: rew_norm and rew_center both set; rew_norm wins\n");
            g_rew_center = 0.0f;
        }
    }
    {
        DictItem* x;
        g_canon = (x = dict_find(env_kwargs, "canonical")) != NULL && (int)x->value != 0;
        g_resid = (x = dict_find(env_kwargs, "residual")) != NULL && (int)x->value != 0;
        g_plant_pair = (x = dict_find(env_kwargs, "plant_pair")) != NULL && (int)x->value != 0;
        if ((x = dict_find(env_kwargs, "plant_eps")) != NULL && x->value > 0.0) g_plant_eps = (float)x->value;
        if (g_plant_pair) {
            if (!g_canon || g_resid) {
                fprintf(stderr, "smart_buildings: plant_pair needs canonical absolute actions\n");
                abort();
            }
            if ((n % 2) != 0) {
                fprintf(stderr, "smart_buildings: plant_pair needs an even agent count\n");
                abort();
            }
            fprintf(stderr, "smart_buildings: plant pair on (odd env = even action, supply air and hot water +%.3f)\n",
                    g_plant_eps);
        }
        if ((x = dict_find(env_kwargs, "res_scale")) != NULL && x->value > 0.0) g_res_scale = (float)x->value;
        if ((x = dict_find(env_kwargs, "res_dscale")) != NULL && x->value > 0.0) g_res_dscale = (float)x->value;
        if ((x = dict_find(env_kwargs, "guard_d0")) != NULL && x->value >= 0.0) g_guard_d0 = (float)x->value;
        if ((x = dict_find(env_kwargs, "guard_d1")) != NULL) g_guard_d1 = (float)x->value;
        g_guard_hot = (x = dict_find(env_kwargs, "guard_hot")) != NULL && (int)x->value != 0;
        if (!g_resid) g_guard_d1 = 0.0f;
        if ((x = dict_find(env_kwargs, "comfort_shape")) != NULL && x->value > 0.0) g_comfort_shape = (float)x->value;
        if ((x = dict_find(env_kwargs, "shape_gamma")) != NULL && x->value > 0.0) g_shape_gamma = (float)x->value;
        g_zone_memory = (x = dict_find(env_kwargs, "zone_memory")) != NULL && (int)x->value != 0;
        g_zone_adv = (x = dict_find(env_kwargs, "zone_adv")) != NULL && (int)x->value != 0;
        g_zone_critic = (x = dict_find(env_kwargs, "zone_critic")) != NULL && (int)x->value != 0;
        g_ahu_critic = (x = dict_find(env_kwargs, "ahu_critic")) != NULL && (int)x->value != 0;
        if (g_ahu_critic && !g_zone_critic) {
            fprintf(stderr, "smart_buildings: ahu_critic needs zone_critic=1\n");
            abort();
        }
        if (g_ahu_critic)
            fprintf(stderr, "smart_buildings: AHU critic on (per-AHU return: own zones' productivity, whole building energy and carbon)\n");
        g_interleave = (x = dict_find(env_kwargs, "interleave")) != NULL && (int)x->value != 0;
        if (g_zone_adv && !g_zone_memory) {
            fprintf(stderr, "smart_buildings: zone_adv needs zone_memory=1\n");
            abort();
        }
        if (g_zone_critic && !g_zone_memory) {
            fprintf(stderr, "smart_buildings: zone_critic needs zone_memory=1\n");
            abort();
        }
        if (g_zone_critic && g_zone_adv) {
            fprintf(stderr, "smart_buildings: zone_critic and zone_adv are different corrections\n");
            abort();
        }
        if (g_zone_adv)
            fprintf(stderr, "smart_buildings: zone advantage on (reheat and damper use that zone's productivity share)\n");
        if (g_zone_critic)
            fprintf(stderr, "smart_buildings: zone critic on (per-zone return, fuel split by occupancy)\n");
        if (g_interleave)
            fprintf(stderr, "smart_buildings: weathers interleaved round-robin\n");
        g_prestart = (x = dict_find(env_kwargs, "prestart_explore")) != NULL && (int)x->value != 0;
        if (g_random_start && (g_prestart || g_plant_pair || g_comfort_shape > 0.0f)) {
            fprintf(stderr, "smart_buildings: random_start cannot be combined with prestart_explore, plant_pair or comfort_shape\n");
            abort();
        }
        g_zone_command = (x = dict_find(env_kwargs, "zone_command")) != NULL && (int)x->value != 0;
        if (g_zone_command && (!g_canon || g_resid)) {
            fprintf(stderr, "smart_buildings: zone_command needs canonical absolute actions\n");
            abort();
        }
#ifdef SB_ENTITY
        if (g_zone_command) {
            fprintf(stderr, "smart_buildings: zone_command is not an entity-mode action\n");
            abort();
        }
#endif
        if (g_zone_memory) fprintf(stderr, "smart_buildings: zone memory on (anticipation from the observation)\n");
        g_zone_signed = (x = dict_find(env_kwargs, "zone_signed")) != NULL && (int)x->value != 0;
        if (cudaMemcpyToSymbol(d_zone_signed, &g_zone_signed, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "smart_buildings: zone_signed could not reach the device\n");
            abort();
        }
        if (g_zone_signed)
            fprintf(stderr, "smart_buildings: zone memory deficits signed (cold +, hot -; the critic keeps magnitudes)\n");
        g_zone_night_share = (x = dict_find(env_kwargs, "zone_night_share")) != NULL && (int)x->value != 0;
        if (cudaMemcpyToSymbol(d_zone_night_share, &g_zone_night_share, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "smart_buildings: zone_night_share could not reach the device\n");
            abort();
        }
        if (g_zone_night_share) {
            if (!g_zone_critic || g_zone_adv) {
                fprintf(stderr, "smart_buildings: zone_night_share needs zone_critic=1 and zone_adv=0\n");
                abort();
            }
            fprintf(stderr, "smart_buildings: empty-building fuel billed to the zone slots (equal share at night)\n");
        }
        g_probe = (x = dict_find(env_kwargs, "probe")) != NULL ? (int)x->value : 0;
        if (g_probe) {
            if (!g_canon || g_resid) {
                fprintf(stderr, "smart_buildings: env.probe needs canonical absolute actions\n");
                abort();
            }
            if (g_probe < 1 || g_probe > 3) {
                fprintf(stderr, "smart_buildings: env.probe must be 1, 2 or 3\n");
                abort();
            }
            DictItem* pw = dict_find(env_kwargs, "probe_water");
            g_probe_water = pw ? (float)pw->value : 0.0f;
            fprintf(stderr, "smart_buildings: PROBE %d (diagnostic, never a scored run): %s%s%s, water action %.3f\n",
                    g_probe, (g_probe & 1) ? "water fixed" : "", g_probe == 3 ? " + " : "",
                    (g_probe & 2) ? "reheat shut above the cooling setpoint" : "", (double)g_probe_water);
        }
        if (g_prestart) fprintf(stderr, "smart_buildings: pre-dawn exploration on (02:00-06:00, zone temperature noise)\n");
        if (g_resid && !g_canon) { fprintf(stderr, "smart_buildings: env.residual needs env.canonical = 1\n"); abort(); }
#ifdef SB_ENTITY
        if (!g_canon || !g_resid) {
            fprintf(stderr, "smart_buildings: entity mode requires env.canonical = 1 and env.residual = 1\n");
            abort();
        }
        fprintf(stderr, "smart_buildings: entity mode, %d zone agents per building, obs %d, actions %d\n",
                SB_NENT, SB_ENT_OBS, SB_ENT_ACT);
#endif
        if (g_canon)
            fprintf(stderr, "smart_buildings: canonical layout (%d obs, %d continuous heads), %s%s\n", CN_OBS, CN_ACT,
                    g_resid ? "residual over the on-device pid_water base" : "absolute actions",
                    g_zone_command ? "; zone command on (proportional band, pump follows the average, water reset 4.0 K)" : "");
        if (g_resid) fprintf(stderr, "smart_buildings: residual scale %.3f (continuous), %.3f (run bits)\n", g_res_scale, g_res_dscale);
        if (g_guard_d1 > 0.0f)
            fprintf(stderr, "smart_buildings: comfort guard: residual authority 1 -> 0 over %.2f .. %.2f K %s the band\n",
                    g_guard_d0, g_guard_d0 + g_guard_d1, g_guard_hot ? "outside" : "below");
        if (g_comfort_shape > 0.0f)
            fprintf(stderr, "smart_buildings: comfort shaping %.3f (train-only potential, occupancy-free, gamma %.3f)\n",
                    g_comfort_shape, g_shape_gamma);
    }
    static std::vector<unsigned char> pristine;     /* the state before any scenario was set up */
    if (pristine.empty()) sb_group_save(pristine);
    g_groups.clear();
    int off = 0;
    for (int gi = 0; gi < NG; gi++) {
        std::vector<const char*> gp;
        std::vector<int> id, cnt, es;
        int ng = 0;
        for (int i : clusters[gi]) {
            gp.push_back(paths[i]); id.push_back(i); cnt.push_back(scnt[i]); ng += scnt[i];
        }
        sb_rr_ids(es, id, cnt, g_interleave && id.size() > 1);
        sb_group_load(pristine);          /* no leftovers from the previous group */
        nenv = ng;
        if (sb_gpu_setup_w(gp.data(), (int)gp.size(), cnt.data()) != 0) abort();
        if (g_random_start) {
            for (size_t w = 0; w < gp.size(); ++w)
                if (!sb_random_start_blob_ok(&wblobs[w], gp[w])) abort();
        }
        if (cudaMalloc(&d_envscen, 4 * (size_t)ng) != cudaSuccess ||
            cudaMemcpy(d_envscen, es.data(), 4 * (size_t)ng, cudaMemcpyHostToDevice) != cudaSuccess) abort();
        if (g_rew_norm > 0.5f) {
            /* Learning-reward normalization stats: mu=0, inv_std at the steady scale.  The old init
             * (var=100 -> inv_std=0.1) was meant to scale the first rollout's raw rewards
             * conservatively, but the per-window EMA (beta 0.9) then took ~40 windows to converge:
             * measured in r8, norm/reward_inv_std ramped x7.5-x16.7 across every 8M-step segment
             * (post-mortem §12), the early advantages were under-scaled by that factor (so the
             * effective policy step was tiny exactly when the policy had to move) and loss/value
             * spiked to 1.1e4 while the critic chased the moving target.  Initialising at 1.0 (r9a)
             * left a transient: measured inside r9a the fixed point itself moves as the policy
             * tightens - inv_std 1.89 at 2.5M (var 0.28), 2.26 at 5.2M, 2.15 at 6.8M.  Start a run
             * (or a continuation) at the endpoint of the run it follows; the EMA then only tracks
             * what the policy's own improvement does to the reward scale. */
            const float rn_var0 = 0.217f; /* measured endpoint of r9a_s00: inv_std 2.147 (var 0.217) */
            float rn_init[2 * SB_MAX_BLOBS];
            for (int i = 0; i < (int)(2 * SB_MAX_BLOBS); i++)
                rn_init[i] = (i % 2) ? (float)(1.0 / sqrt((double)rn_var0)) : 0.0f;
            if (cudaMalloc(&d_rew_norm, 4 * 2 * (size_t)SB_MAX_BLOBS) != cudaSuccess ||
                cudaMemcpy(d_rew_norm, rn_init, 4 * 2 * (size_t)SB_MAX_BLOBS,
                           cudaMemcpyHostToDevice) != cudaSuccess) abort();
            for (int i = 0; i < (int)SB_MAX_BLOBS; i++) { g_rn_mu[i] = 0.0; g_rn_var[i] = rn_var0; }
            int zn = 1;
            cudaMemcpyToSymbol(d_znclip, &zn, sizeof(int));
        } else if (g_rew_center > 0.5f && d_rew_norm == NULL) {
            /* mu = 0, inv_std = 1.  A loaded critic predicts uncentered returns; subtracting the
             * true mean immediately would shift the value target by mu/(1-gamma) (~90 on fairbanks)
             * and the grad clip would starve the policy.  The EMA below creeps mu from 0. */
            float rn_init[2 * SB_MAX_BLOBS];
            for (int i = 0; i < (int)(2 * SB_MAX_BLOBS); i++)
                rn_init[i] = (i % 2) ? 1.0f : 0.0f;
            if (cudaMalloc(&d_rew_norm, 4 * 2 * (size_t)SB_MAX_BLOBS) != cudaSuccess ||
                cudaMemcpy(d_rew_norm, rn_init, 4 * 2 * (size_t)SB_MAX_BLOBS,
                           cudaMemcpyHostToDevice) != cudaSuccess) abort();
            for (int i = 0; i < (int)SB_MAX_BLOBS; i++) { g_rn_mu[i] = 0.0; g_rn_var[i] = 1.0; }
            fprintf(stderr, "smart_buildings: reward centering on (per-scenario running mean, beta %.4f, inv_std 1)\n",
                    g_center_beta);
        }
        int ep_len = sb_blob_ep_len();
        if (ms != NULL && (int)ms->value > 0 && (int)ms->value < ep_len) ep_len = (int)ms->value;
        g_ep_len = ep_len;
        for (size_t w = 0; w < gp.size(); w++)
            fprintf(stderr, "smart_buildings: group %d/%d, weather %zu: %d envs, episode length %d steps, %s\n",
                    gi + 1, NG, w, cnt[w], ep_len, gp[w]);
        for (int i = off; i < off + ng; i++) {
#ifdef SB_ENTITY
            envs[i].num_agents = SB_NENT;
#else
            envs[i].num_agents = 1;
#endif
            envs[i].max_steps = ep_len;
            envs[i].start_rng = sb_start_mix(start_seed ^
                (0x9e3779b97f4a7c15ULL * (unsigned long long)(i + 1)));
            if (desync) {
                const int id = (g_plant_pair && (i & 1)) ? (i - 1) : i;
                unsigned long long h = 0x9E3779B97F4A7C15ULL * (unsigned long long)(id + 1);
                h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 29;
                envs[i].max_steps = 1 + (int)(h % (unsigned long long)ep_len);
            }
            envs[i].env_id = i;
            envs[i].n_act = n_action;
            envs[i].n_cont = 0;
            envs[i].n_disc = 0;
            for (int j = 0; j < n_action; j++) {
                if (sb_u8(&blob, "act_is_discrete")[j]) envs[i].n_disc++;
                else envs[i].n_cont++;
            }
            envs[i].cont = (double*)calloc(SB_CONT, sizeof(double));
            envs[i].disc = (double*)calloc(SB_DISC, sizeof(double));
            envs[i].native = (double*)calloc(n_action ? n_action : 1, sizeof(double));
#ifdef SB_ENTITY
            for (int s = 0; s < SB_NENT; s++) {
                const size_t a = (size_t)i * SB_NENT + s;
                envs[i].agents[s].observations = (float*)observations + a * SB_ENT_OBS;
                envs[i].agents[s].actions = actions + a * SB_ENT_ACT;
                envs[i].agents[s].rewards = rewards + a;
                envs[i].agents[s].terminals = terminals + a;
            }
#else
            envs[i].agents[0].observations = (float*)observations + (size_t)i * SB_OBS;
            envs[i].agents[0].actions = actions + (size_t)i * (SB_CONT + SB_DISC);
            envs[i].agents[0].rewards = rewards + i;
            envs[i].agents[0].terminals = terminals + i;
#endif
            envs[i].loaded = 1;
        }
        g_envs = envs + off;
        if (g_canon) {          /* the step keeps the oracle observation internally and hands the policy its own */
            g_obs = d_obsraw;
            g_obs_pol = envs[off].agents[0].observations;
        } else {
            g_obs = envs[off].agents[0].observations;
        }
#ifdef SB_ENTITY
        g_rew_fw = envs[off].agents[0].rewards;
        g_term_fw = envs[off].agents[0].terminals;
        g_rewards = d_rew_learn;       /* finalize writes one reward per env; entity_scatter_k spreads it */
        g_terminals = d_term_learn;
#else
        g_rewards = envs[off].agents[0].rewards;
        g_terminals = envs[off].agents[0].terminals;
#endif
        g_actions = envs[off].agents[0].actions;
        SbGroup g;
        snprintf(g.name, sizeof(g.name), "%s", g_scen_names[clusters[gi][0]].c_str());
        sb_group_save(g.st);
        g.off = off;
        g.n = ng;
        if (NG > 1) {
            cudaStreamCreateWithFlags(&g.s, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&g.done, cudaEventDisableTiming);
        }
        g_groups.push_back(std::move(g));
        {
            char lab[64];
            snprintf(lab, sizeof(lab), "group %d/%d", gi + 1, NG);
            sb_print_solve_layout(lab, ng, (int)gp.size());
        }
        off += ng;
    }
    {
        const char* traj = getenv("SB_TRAJ");
        if ((g_zone_memory || g_zone_adv || g_zone_critic || g_prestart || (traj != NULL && traj[0] != '\0')) && off > 0) zm_alloc(off);
    }
    sb_mg_setup();
    return envs;
}

/* The framework captures the whole horizon on its own stream: every launch and
 * the cuDSS handle must be on that stream or nothing is captured. */
void puf_bind_stream(cudaStream_t s) {
    if (s == NULL) return;
    stream = s;
}
void puf_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->max_steps = sb_blob_ep_len();
    memset(&env->log, 0, sizeof(env->log));
}
void puf_reset(Env* envs) {
    for (auto& g : g_groups) {
        sb_group_load(g.st);
        zm_base = g.off;
        if (sb_gpu_reset() != 0) abort();
        cudaMemset(d_epretd, 0, 8 * (size_t)nenv);
        for (int i = g.off; i < g.off + g.n; i++) {
            envs[i].step = 0;
            envs[i].ep_ret = 0.0f; envs[i].ep_worst = 0.0f;
            for (int k2 = 0; k2 < SB_NPARTS_ENV; k2++) envs[i].ep_part[k2] = 0.0f;
            for (int a = 0; a < envs[i].n_act; a++) envs[i].native[a] = 0.0;
            unsigned char* m = envs[i].agents[0].action_mask;
            if (m != NULL && g_canon) {
                const CanonMap* cm = h_cmap;
                for (int h = 0; h < SB_MASK_WIDTH; h++) m[h] = 0;
                for (int sl = 0; sl < CN_ZS; sl++) {
                    if (cm->zAd[sl] >= 0) m[2 * sl] = 1;
                    if (cm->zAr[sl] >= 0) m[2 * sl + 1] = 1;
                }
                for (int k = 0; k < cm->K && k < CN_K; k++) {
                    if (cm->aT[k] >= 0) m[CN_AA + 4 * k] = 1;
                    if (cm->aP[k] >= 0) m[CN_AA + 4 * k + 1] = 1;
                    if (cm->aRec[k] >= 0) m[CN_AA + 4 * k + 2] = 1;
                    if (cm->aRun[k] >= 0) m[CN_AA + 4 * k + 3] = 1;
                }
                if (cm->hW >= 0) m[CN_AH] = 1;
                if (cm->hDPa >= 0) m[CN_AH + 1] = 1;
                if (cm->hRun >= 0) m[CN_AH + 2] = 1;
                sb_zone_command_mask(m, cm);
            } else if (m != NULL) {
                for (int h = 0; h < SB_CONT; h++) m[h] = (h < envs[i].n_cont) ? 1 : 0;
                for (int h = 0; h < SB_DISC; h++) {
                    unsigned char v = (h < envs[i].n_disc) ? 1 : 0;
                    m[SB_CONT + 2 * h] = v;
                    m[SB_CONT + 2 * h + 1] = v;
                }
            }
        }
    }
#ifndef SB_TEST
    /* The step writes envs[i].step / envs[i].log on the device inside the capture, where a
     * managed page fault is not permitted: make the whole array device-resident before the
     * rollout starts. */
    int ntot = 0;
    for (auto& g : g_groups) ntot += g.n;
    touch_envs_k<<<(unsigned)((sizeof(Env) * ntot) / 4096 / 256 + 1), 256, 0, stream>>>(
        envs, sizeof(Env) * ntot);
    cudaStreamSynchronize(stream);
#endif
}
/* Snapshot of env `genv` (index in the whole batch): load its group, write, done. */
static int sb_snapshot_global(int genv, const char* path) {
    for (auto& g : g_groups)
        if (genv >= g.off && genv < g.off + g.n) {
            sb_group_load(g.st);
            return sb_snapshot_write(genv - g.off, path);
        }
    return 1;
}

static void sb_eval_dump(const char* path) {
    cudaDeviceSynchronize();
    static double h[SB_MAX_BLOBS][5];
    if (cudaMemcpyFromSymbol(h, g_first_scen, sizeof(h)) != cudaSuccess) return;
    std::string tmp = std::string(path) + ".tmp";
    FILE* f = fopen(tmp.c_str(), "w");
    if (f == NULL) return;
    fprintf(f, "{\"scenarios\": {");
    double ssum = 0.0; int ns = 0, done = 1;
    for (size_t i = 0; i < g_scen_names.size() && i < SB_MAX_BLOBS; i++) {
        const double c = h[i][4];
        if (c <= 0.0) { done = 0; continue; }
        const double r = h[i][0] / c;
        fprintf(f, "%s\"%s\": {\"episodes\": %.0f, \"mean_reward\": %.10g, \"productivity\": %.10g, \"energy_cost\": %.10g, \"carbon\": %.10g}",
                ns ? ", " : "", g_scen_names[i].c_str(), c, r, h[i][1] / c, h[i][2] / c, h[i][3] / c);
        ssum += fabs(r); ns++;
    }
    fprintf(f, "}, \"n_scenarios_done\": %d, \"n_scenarios\": %zu, \"complete\": %s, \"score\": %.10g}\n", ns,
            g_scen_names.size(), (done && ns == (int)g_scen_names.size()) ? "true" : "false", ns ? ssum / ns : 0.0);
    fclose(f);
    rename(tmp.c_str(), path);
}

void puf_step(Env* envs) {
    /* Device-only.  The framework captures the whole horizon into a CUDA graph
     * on its own stream, so nothing here may touch the host, a different
     * stream, a page-migrating managed access, or a per-step cudaFuncSetAttribute.
     * The action decode, the step, the rewards/terminals/episode log and the
     * episode restart are all kernels on that stream.  Groups (scenarios) step one after
     * the other on the same stream. */
    (void)envs;
#ifndef SB_TEST
    /* Several groups: each steps on its own stream, forked from and joined back into the rollout
     * stream with events (legal inside a graph capture).  Every launch reads the `stream` static
     * at call time, so pointing it at the group's stream binds the whole group step to it.
     * SB_GROUP_SERIAL=1 keeps them on one stream (A/B timing, debugging). */
    static const bool serial = getenv("SB_GROUP_SERIAL") != NULL;
    const bool par = g_groups.size() > 1 && !serial;
    cudaStream_t main_stream = stream;
    if (g_mg_on && par) {
        sb_mg_step(main_stream, true);
    } else {
    if (par) {
        if (g_fork == NULL) cudaEventCreateWithFlags(&g_fork, cudaEventDisableTiming);
        cudaEventRecord(g_fork, main_stream);
    }
    for (auto& g : g_groups) {
        sb_group_load(g.st);
        zm_base = g.off;
        if (par) {
            stream = g.s;
            cudaStreamWaitEvent(stream, g_fork, 0);
        }
        sb_decode();
        if (sb_gpu_run_step() != 0) abort();
        if (par) cudaEventRecord(g.done, g.s);
    }
    stream = main_stream;
    if (par)
        for (auto& g : g_groups) cudaStreamWaitEvent(main_stream, g.done, 0);
    }
    /* Per-step snapshots only when this call is not being captured (graphs off: eval,
     * smoke runs); under graphs puf_live_log writes one per epoch instead. */
    SnapCfg sc = snap_cfg();
    if (sc.path != NULL) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cs);
        static int cnt = 0;
        if (cs == cudaStreamCaptureStatusNone && ++cnt % sc.every == 0)
            sb_snapshot_global(sc.env, sc.path);
    }
    /* SB_EVAL_OUT=<file.json>: per scenario, each env's FIRST full episode (= a fresh gateway episode when the run
     * starts every env at step 0, i.e. env.desync = 0): mean per-step reward and reward terms, written every 288
     * steps when graphs are off. puf_log also exports after each rollout, including graph replay. */
    static const char* eval_out = getenv("SB_EVAL_OUT");
    if (eval_out != NULL) {
        cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
        cudaStreamIsCapturing(stream, &cs);
        static long ecnt = 0;
        if (cs == cudaStreamCaptureStatusNone && ++ecnt % 288 == 0) sb_eval_dump(eval_out);
    }
#endif
}

void puf_render(Env* env) { (void)env; }
void puf_close(Env* env) {
    if (env != NULL) env->loaded = 0;
}
void puf_log(Log* log, Dict* out) {
#ifndef SB_TEST
    // Called after rollout synchronization with a host copy of the device Log.
    // Read only g_first_scen (a device diagnostic symbol), never managed Env.
    // Unlike puf_step's host callback, this boundary runs under graph replay too.
    const char* eval_out = getenv("SB_EVAL_OUT");
    if (eval_out != NULL) sb_eval_dump(eval_out);
    zm_zone_check_report();
#endif
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "comfort_dev_K", log->comfort_dev_K);
    dict_set(out, "worst_dev_K", log->worst_dev_K);
    dict_set(out, "out_of_band_frac", log->out_of_band_frac);
    dict_set(out, "productivity", log->productivity);
    dict_set(out, "energy_cost_norm", log->energy_cost_norm);
    dict_set(out, "carbon_norm", log->carbon_norm);
    dict_set(out, "elec_kwh", log->elec_kwh);
    dict_set(out, "gas_kwh", log->gas_kwh);
    dict_set(out, "n", log->n);
}
/* Step-level metrics since the previous call (the per-episode Log only fills every
 * 8640 steps).  Called by the trainer once per epoch, outside any capture. */
void puf_live_log(Env* envs, Dict* out) {
    (void)envs;
    cudaDeviceSynchronize();
    SnapCfg sc = snap_cfg();
    if (sc.path != NULL) sb_snapshot_global(sc.env, sc.path);
    double h[1 + 1 + SB_NPARTS_ENV];
    double z[1 + 1 + SB_NPARTS_ENV] = {0};
    if (cudaMemcpyFromSymbol(h, g_live, sizeof(h)) != cudaSuccess) return;
    cudaMemcpyToSymbol(g_live, z, sizeof(z));
    {
        double zn[3] = {0}, zz[3] = {0};
        if (cudaMemcpyFromSymbol(zn, g_znclip, sizeof(zn)) == cudaSuccess) {
            cudaMemcpyToSymbol(g_znclip, zz, sizeof(zz));
            if (h[0] > 0.0) {
                dict_set(out, "live/rew_z_gt1_frac", zn[0] / h[0]);
                dict_set(out, "live/rew_z_gt3_frac", zn[1] / h[0]);
                dict_set(out, "live/rew_z_gt5_frac", zn[2] / h[0]);
            }
        }
    }
    {
        static double hg[SB_MAX_BLOBS][4], zg[SB_MAX_BLOBS][4];
        if (cudaMemcpyFromSymbol(hg, g_live_grp, sizeof(hg)) == cudaSuccess) {
            cudaMemcpyToSymbol(g_live_grp, zg, sizeof(zg));
            if (g_scen_names.size() > 1)
                for (size_t gi = 0; gi < g_scen_names.size() && gi < SB_MAX_BLOBS; gi++) {
                    if (hg[gi][2] <= 0.0) continue;
                    char key[160];
                    snprintf(key, sizeof(key), "scenario/%s/reward", g_scen_names[gi].c_str());
                    dict_set(out, key, hg[gi][0] / hg[gi][2]);
                    snprintf(key, sizeof(key), "scenario/%s/comfort_dev_K", g_scen_names[gi].c_str());
                    dict_set(out, key, hg[gi][1] / hg[gi][2]);
                }
            /* Refresh the per-scenario learning-reward stats from this window and push the device
             * mirror.  rew_norm steps mean and variance by 0.1; rew_center steps the mean only
             * (env.center_beta) and keeps inv_std at 1.  One live-log lag is intentional:
             * finalize_k must never read stats that include the step being scaled. */
            if ((g_rew_norm > 0.5f || g_rew_center > 0.5f) && g_scen_names.size() > 0 && d_rew_norm != NULL) {
                float rn[2 * SB_MAX_BLOBS];
                double mu_sum = 0.0, sd_sum = 0.0;
                int ns = 0;
                /* rew_norm: beta 0.1 on mean and variance (the historical PopArt-lite).
                 * rew_center: beta on the mean only, inv_std fixed at 1 (equation 4). */
                const double b = g_rew_center > 0.5f ? (double)g_center_beta : 0.1;
                for (size_t gi = 0; gi < g_scen_names.size() && gi < SB_MAX_BLOBS; gi++) {
                    if (hg[gi][2] > 0.0) {
                        const double mw = hg[gi][0] / hg[gi][2];
                        g_rn_mu[gi] = (1.0 - b) * g_rn_mu[gi] + b * mw;
                        if (g_rew_norm > 0.5f) {
                            double vw = hg[gi][3] / hg[gi][2] - mw * mw;
                            if (vw < 1e-12) vw = 1e-12;
                            g_rn_var[gi] = (1.0 - b) * g_rn_var[gi] + b * vw;
                        }
                    }
                    const double var = g_rn_var[gi] < 1e-12 ? 1e-12 : g_rn_var[gi];
                    rn[2 * gi] = (float)g_rn_mu[gi];
                    rn[2 * gi + 1] = g_rew_center > 0.5f ? 1.0f : (float)(1.0 / sqrt(var));
                    mu_sum += g_rn_mu[gi];
                    sd_sum += rn[2 * gi + 1];
                    ns++;
                }
                if (ns > 0) {
                    cudaMemcpy(d_rew_norm, rn, 4 * 2 * (size_t)SB_MAX_BLOBS, cudaMemcpyHostToDevice);
                    dict_set(out, "norm/reward_mean", mu_sum / ns);
                    dict_set(out, "norm/reward_inv_std", sd_sum / ns);
                }
            }
        }
    }
    {
        static double he[SB_MAX_BLOBS][2], ze[SB_MAX_BLOBS][2];
        if (cudaMemcpyFromSymbol(he, g_ep_scen, sizeof(he)) == cudaSuccess) {
            cudaMemcpyToSymbol(g_ep_scen, ze, sizeof(ze));
            double ss = 0.0; int ns = 0;
            for (size_t gi = 0; gi < g_scen_names.size() && gi < SB_MAX_BLOBS; gi++) {
                if (he[gi][1] > 0.0) g_last_sc[gi] = -he[gi][0] / he[gi][1];
                char key[160];
                snprintf(key, sizeof(key), "scenario/%s/episode_score", g_scen_names[gi].c_str());
                dict_set(out, key, g_last_sc[gi]);
                ss += g_last_sc[gi]; ns++;
            }
            if (ns > 0) { dict_set(out, "env/episode_score_scen_mean", ss / ns); dict_set(out, "env/episode_score_n_scen", ns); }
        }
    }
    if (g_random_start) {
        static double hd[SB_MAX_BLOBS][2][7], zd[SB_MAX_BLOBS][2][7];
        if (cudaMemcpyFromSymbol(hd, g_start_draws, sizeof(hd)) == cudaSuccess) {
            cudaMemcpyToSymbol(g_start_draws, zd, sizeof(zd));
            for (size_t gi = 0; gi < g_scen_names.size() && gi < SB_MAX_BLOBS; ++gi) {
                for (int first = 0; first < 2; ++first) for (int d = 0; d < 7; ++d) {
                    if (hd[gi][first][d] <= 0.0) continue;
                    char key[160];
                    snprintf(key, sizeof(key), "scenario/%s/start_day_%d/%s_draws",
                             g_scen_names[gi].c_str(), d, first == 0 ? "initial" : "later");
                    dict_set(out, key, hd[gi][first][d]);
                }
            }
        }
        static double hs[SB_MAX_BLOBS][7][7], zs[SB_MAX_BLOBS][7][7];
        if (cudaMemcpyFromSymbol(hs, g_ep_start, sizeof(hs)) == cudaSuccess) {
            cudaMemcpyToSymbol(g_ep_start, zs, sizeof(zs));
            const char* cols[6] = {"score", "productivity", "energy_cost", "carbon", "elec_kwh", "gas_kwh"};
            for (size_t gi = 0; gi < g_scen_names.size() && gi < SB_MAX_BLOBS; ++gi) {
                for (int d = 0; d < 7; ++d) {
                    double count = hs[gi][d][0];
                    if (count <= 0.0) continue;
                    char key[160];
                    snprintf(key, sizeof(key), "scenario/%s/start_day_%d/count", g_scen_names[gi].c_str(), d);
                    dict_set(out, key, count);
                    for (int k = 0; k < 6; ++k) {
                        snprintf(key, sizeof(key), "scenario/%s/start_day_%d/%s", g_scen_names[gi].c_str(), d, cols[k]);
                        dict_set(out, key, (k == 0 ? -1.0 : 1.0) * hs[gi][d][k + 1] / count);
                    }
                }
            }
        }
    }
    {
        const char* enabled = getenv("SB_DETAIL_METRICS");
        if (enabled != NULL && strcmp(enabled, "1") == 0) {
            static double hd[SB_MAX_BLOBS][7][8], zd[SB_MAX_BLOBS][7][8];
            if (cudaMemcpyFromSymbol(hd, g_detail_scen, sizeof(hd)) == cudaSuccess) {
                cudaMemcpyToSymbol(g_detail_scen, zd, sizeof(zd));
                const char* bins[7] = {"night", "weekday_morning", "weekday_afternoon",
                    "weekday_evening", "weekend_day", "empty", "occupied"};
                const char* cols[7] = {"productivity", "energy_cost", "carbon", "comfort_dev_K",
                    "out_of_band", "elec_kw", "gas_kw"};
                for (size_t gi = 0; gi < g_scen_names.size() && gi < SB_MAX_BLOBS; ++gi) {
                    for (int b = 0; b < 7; ++b) {
                        const double n = hd[gi][b][0];
                        if (n <= 0.0) continue;
                        char key[160];
                        snprintf(key, sizeof(key), "scenario/%s/%s/count", g_scen_names[gi].c_str(), bins[b]);
                        dict_set(out, key, n);
                        for (int j = 0; j < 7; ++j) {
                            snprintf(key, sizeof(key), "scenario/%s/%s/%s", g_scen_names[gi].c_str(), bins[b], cols[j]);
                            dict_set(out, key, hd[gi][b][1+j] / n);
                        }
                    }
                }
            }
        }
    }
    double c = h[0];
    if (c <= 0.0) return;
    dict_set(out, "live/reward", h[1] / c);
    dict_set(out, "live/productivity", h[2] / c);
    dict_set(out, "live/energy_cost_norm", h[3] / c);
    dict_set(out, "live/carbon_norm", h[4] / c);
    dict_set(out, "live/comfort_dev_K", h[5] / c);
    dict_set(out, "live/worst_dev_K", h[6] / c);
    dict_set(out, "live/out_of_band_frac", h[7] / c);
    dict_set(out, "live/elec_kw", h[8] / c * 3600.0 / dt_sec_);
    dict_set(out, "live/gas_kw", h[9] / c * 3600.0 / dt_sec_);
}
/* Fill the trainer's action mask.  A mask column of 0 keeps that head out of the policy loss and out
 * of ratio_per_dim: the sampled decoder is wider than the building's action vector, and a padding
 * head (a decoy Gaussian / Bernoulli) has no meaning in the environment, so training it only moves
 * the ratio denominator.  puf_reset writes the same row into the env's own mask pointer, but the
 * trainer path owns that buffer and it is null there; this is the entry point it does call, once at
 * setup.  Rows are per env (canonical / generic builds) or per (env, zone slot) (entity build). */
void puf_mask_init(unsigned char* mask, int width, int nagents) {
    if (width < 1 || width > SB_MASK_WIDTH) {
        fprintf(stderr, "smart_buildings: mask init width %d out of range\n", width);
        abort();
    }
#ifdef SB_ENTITY
    if (width < 2) { fprintf(stderr, "smart_buildings: entity mask width %d < 2\n", width); abort(); }
    (void)nagents;
    for (auto& g : g_groups) {
        sb_group_load(g.st);
        unsigned char* base = mask + (size_t)g.off * SB_NENT * width;
        cudaMemset(base, 0, (size_t)g.n * SB_NENT * width);
        entity_mask_k<<<(g.n + 63) / 64, 64>>>(d_cmap, base, g.n, width);
    }
    cudaDeviceSynchronize();
#else
    std::vector<unsigned char> host((size_t)nagents * width, 0);
    for (size_t gi = 0; gi < g_groups.size(); gi++) {
        SbGroup& g = g_groups[gi];
        sb_group_load(g.st);
        if (g.off < 0 || g.n < 0 || g.off + g.n > nagents) {
            fprintf(stderr, "smart_buildings: mask group %zu/%zu rows %d..%d of agents %d\n",
                    gi + 1, g_groups.size(), g.off, g.off + g.n, nagents);
            abort();
        }
        unsigned char* m = host.data() + (size_t)g.off * width;
        if (g_canon) {
            /* the canonical actions this building actually has (same map the reset writer uses) */
            const CanonMap* cm = h_cmap;
            for (int sl = 0; sl < CN_ZS; sl++) {
                if (cm->zAd[sl] >= 0) m[2 * sl] = 1;
                if (cm->zAr[sl] >= 0) m[2 * sl + 1] = 1;
            }
            for (int k = 0; k < cm->K && k < CN_K; k++) {
                if (cm->aT[k] >= 0) m[CN_AA + 4 * k] = 1;
                if (cm->aP[k] >= 0) m[CN_AA + 4 * k + 1] = 1;
                if (cm->aRec[k] >= 0) m[CN_AA + 4 * k + 2] = 1;
                if (cm->aRun[k] >= 0) m[CN_AA + 4 * k + 3] = 1;
            }
            if (cm->hW >= 0) m[CN_AH] = 1;
            if (cm->hDPa >= 0) m[CN_AH + 1] = 1;
            if (cm->hRun >= 0) m[CN_AH + 2] = 1;
            sb_zone_command_mask(m, cm);
            int act = 0;
            for (int h = 0; h < width; h++) act += m[h];
            fprintf(stderr, "smart_buildings: mask %zu/%zu: %d/%d active heads\n",
                    gi + 1, g_groups.size(), act, width);
        } else {
            /* generic layout: the scenario's native actions (all envs of a group share the shape) */
            const SbSection* ao = sb_find(&blob, "act_owner_kind");
            const uint8_t* isd = sb_u8(&blob, "act_is_discrete");
            int n_act = (ao != NULL) ? (int)ao->count : 0;
            int n_cont = 0, n_disc = 0;
            for (int j = 0; j < n_act; j++) {
                if (isd[j]) n_disc++; else n_cont++;
            }
            for (int h = 0; h < SB_CONT && h < width; h++) m[h] = (h < n_cont) ? 1 : 0;
            for (int h = 0; h < SB_DISC && SB_CONT + 2 * h + 1 < width; h++) {
                unsigned char v = (h < n_disc) ? 1 : 0;
                m[SB_CONT + 2 * h] = v;
                m[SB_CONT + 2 * h + 1] = v;
            }
            fprintf(stderr, "smart_buildings: mask %zu/%zu: generic layout %d cont / %d disc\n",
                    gi + 1, g_groups.size(), n_cont, n_disc);
        }
        for (int i = 1; i < g.n; i++) memcpy(m + (size_t)i * width, m, width);
    }
    if (cudaMemcpy(mask, host.data(), (size_t)nagents * width, cudaMemcpyHostToDevice) != cudaSuccess)
        abort();
    cudaDeviceSynchronize();
#endif
}
#include "sb_full_state.cuh"
#endif /* !SB_TEST */

}  /* namespace sbgpu */

#ifndef SB_TEST
/* The trainer's entry points stay at global scope and forward into the
 * namespace: PufferLib compiles every env into one translation unit, so no
 * file-scope name of this env may collide with another env's. */
Env* puf_vec_create(int n, Dict* k, obs_t* o, float* a, float* r, float* t) {
    return sbgpu::puf_vec_create(n, k, o, a, r, t);
}
void puf_bind_stream(cudaStream_t s) { sbgpu::puf_bind_stream(s); }
void puf_init(Env* e, Dict* k) { sbgpu::puf_init(e, k); }
void puf_reset(Env* e) { sbgpu::puf_reset(e); }
void puf_step(Env* e) { sbgpu::puf_step(e); }
void puf_render(Env* e) { (void)e; }
void puf_close(Env* e) { sbgpu::puf_close(e); }
void puf_log(Log* l, Dict* o) { sbgpu::puf_log(l, o); }
void puf_live_log(Env* e, Dict* o) { sbgpu::puf_live_log(e, o); }
void puf_mask_init(unsigned char* m, int w, int n) { sbgpu::puf_mask_init(m, w, n); }
#endif

#ifdef SB_TEST
using namespace sbgpu;   /* only the trace driver wants unqualified access */
#endif

/* ------------------------------------------------------------------ gate --- */
#ifdef SB_TEST
/* Gate only: SB_GRAPH=1 captures one step into a CUDA graph and replays it for every
 * step, as the trainer does with base.cudagraphs, so the parity gate also proves the
 * step is capturable and replays correctly (all per-step state lives on the device). */
static cudaGraphExec_t g_gexec = NULL;
static int sb_gate_step(void) {
    if (getenv("SB_GRAPH") == NULL) return sb_gpu_run_step();
    if (g_gexec == NULL) {
        cudaGraph_t g;
        CU(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        if (sb_gpu_run_step() != 0) return 1;
        CU(cudaStreamEndCapture(stream, &g));
        CU(cudaGraphInstantiate(&g_gexec, g, 0));
        CU(cudaGraphDestroy(g));
    }
    CU(cudaGraphLaunch(g_gexec, stream));
    return 0;
}

/* groups <blobA> <traceA> <blobB> <traceB> <nenv> <steps>: the multi-scenario batch must not let
 * one group's state leak into another.  A and B are each set up twice; copies 1 run alone, copies 2
 * run interleaved (A2, B2, A2, B2, ...) with the same recorded actions.  Every step's observation
 * buffer, reward and temperature field must be bit-identical between solo and interleaved runs. */
static bool read_trace_actions(const char* path, std::vector<std::vector<double>>& nat) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return false;
    int hdr[7];
    if (fread(hdr, 4, 7, f) != 7) return false;
    const int t_nc = hdr[2], t_nd = hdr[3], t_nsteps = hdr[4], t_nfld = hdr[6];
    std::vector<double> obs0(t_nfld), c1(t_nc), d1(t_nd), oc(n_zones), g1(k), T1(k), s1(1), tn(ncv), rw1(1), rq(12), ob(t_nfld);
    if (fread(obs0.data(), 8, t_nfld, f) != (size_t)t_nfld) return false;
    nat.assign(t_nsteps, std::vector<double>(n_action));
    for (int st = 0; st < t_nsteps; st++) {
        if (fread(c1.data(), 8, t_nc, f) != (size_t)t_nc || fread(d1.data(), 8, t_nd, f) != (size_t)t_nd ||
            fread(nat[st].data(), 8, n_action, f) != (size_t)n_action ||
            fread(oc.data(), 8, n_zones, f) != (size_t)n_zones || fread(g1.data(), 8, k, f) != (size_t)k ||
            fread(T1.data(), 8, k, f) != (size_t)k || fread(s1.data(), 8, 1, f) != 1 ||
            fread(tn.data(), 8, ncv, f) != (size_t)ncv || fread(rw1.data(), 8, 1, f) != 1 ||
            fread(rq.data(), 8, 12, f) != 12 || fread(ob.data(), 8, t_nfld, f) != (size_t)t_nfld) return false;
    }
    fclose(f);
    return true;
}
static unsigned long long fnv(unsigned long long h, const void* p, size_t n) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static int sb_groups_test(int argc, char** argv) {
    if (argc < 8) { printf("usage: gate groups <blobA> <traceA> <blobB> <traceB> <nenv> <steps>\n"); return 2; }
    const char* blobs[2] = {argv[2], argv[4]};
    const char* traces[2] = {argv[3], argv[5]};
    const int N = atoi(argv[6]), S = atoi(argv[7]);
    std::vector<unsigned char> pristine;
    sb_group_save(pristine);
    // groups 0,1 = A1,B1 (solo); 2,3 = A2,B2 (interleaved)
    std::vector<std::vector<unsigned char>> gs(4);
    std::vector<std::vector<double>> nat[2];
    for (int gi = 0; gi < 4; gi++) {
        sb_group_load(pristine);
        nenv = N;
        if (sb_gpu_setup(blobs[gi % 2]) != 0) return 1;
        g_obs = d_obs;
        if (sb_gpu_reset() != 0) return 1;
        if (gi < 2 && !read_trace_actions(traces[gi], nat[gi])) { printf("bad trace %s\n", traces[gi]); return 1; }
        sb_group_save(gs[gi]);
    }
    if ((int)nat[0].size() < S || (int)nat[1].size() < S) { printf("traces shorter than %d steps\n", S); return 1; }
    cudaStream_t ts[4];
    for (int gi = 0; gi < 4; gi++) cudaStreamCreateWithFlags(&ts[gi], cudaStreamNonBlocking);
    const cudaStream_t main_stream = stream;
    /* enqueue one step of group gi on stream `st_`, no waiting */
    auto enqueue = [&](int gi, int st, cudaStream_t st_) {
        sb_group_load(gs[gi]);
        stream = st_;
        const auto& a = nat[gi % 2][st];
        static std::vector<std::vector<double>> keep(4);
        static std::vector<std::vector<int>> keepi(4);
        keep[gi].assign((size_t)nenv * n_action, 0.0);
        for (int e = 0; e < nenv; e++) memcpy(&keep[gi][(size_t)e * n_action], a.data(), 8 * (size_t)n_action);
        keepi[gi].assign(nenv, st + 1);
        cudaMemcpyAsync(d_native, keep[gi].data(), 8 * keep[gi].size(), cudaMemcpyHostToDevice, st_);
        cudaMemcpyAsync(d_stepv, keepi[gi].data(), 4 * (size_t)nenv, cudaMemcpyHostToDevice, st_);
        if (sb_gpu_run_step() != 0) abort();
        sb_group_save(gs[gi]);
        stream = main_stream;
    };
    auto collect = [&](int gi, int st) -> unsigned long long {
        sb_group_load(gs[gi]);
        std::vector<float> ob((size_t)nenv * SB_OBS);
        std::vector<double> rw(nenv), tp((size_t)nenv * ncv);
        cudaMemcpy(ob.data(), d_obs, 4 * ob.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(rw.data(), d_rew, 8 * rw.size(), cudaMemcpyDeviceToHost);
        sb_field_get(tp, 0, (int)(tp.size() / ncv));
        unsigned long long h = 1469598103934665603ULL;
        unsigned long long ho = fnv(h, ob.data(), 4 * ob.size()), hr = fnv(h, rw.data(), 8 * rw.size()),
                           ht = fnv(h, tp.data(), 8 * tp.size());
        if (getenv("SB_GT_VERBOSE") && st < 2) printf("g%d st%d obs=%016llx rew=%016llx temp=%016llx\n", gi, st, ho, hr, ht);
        return fnv(fnv(ho, &hr, 8), &ht, 8);
    };
    auto step = [&](int gi, int st) -> unsigned long long {
        enqueue(gi, st, ts[gi]);
        cudaDeviceSynchronize();
        return collect(gi, st);
    };
    std::vector<unsigned long long> solo[2], mix[2];
    for (int w = 0; w < 2; w++)
        for (int st = 0; st < S; st++) solo[w].push_back(step(w, st));
    if (getenv("SB_GT_SEQ")) {          /* diagnostic: copies 2 also run one after the other */
        for (int w = 0; w < 2; w++)
            for (int st = 0; st < S; st++) mix[w].push_back(step(2 + w, st));
    } else if (getenv("SB_GT_PAR")) {   /* both groups' steps in flight at once on their own streams */
        for (int st = 0; st < S; st++) {
            enqueue(2, st, ts[2]);
            enqueue(3, st, ts[3]);
            cudaDeviceSynchronize();
            mix[0].push_back(collect(2, st));
            mix[1].push_back(collect(3, st));
        }
    } else
    for (int st = 0; st < S; st++) {
        mix[0].push_back(step(2, st));
        mix[1].push_back(step(3, st));
    }
    int bad = 0;
    for (int w = 0; w < 2; w++)
        for (int st = 0; st < S; st++)
            if (solo[w][st] != mix[w][st]) { if (!bad) printf("first mismatch: scenario %d step %d\n", w, st); bad++; }
    printf("groups: %d steps x 2 scenarios x %d envs, solo vs interleaved: %d differing step hashes\n", S, N, bad);
    printf(bad ? "SB GPU GROUPS: FAIL\n" : "SB GPU GROUPS: OK\n");
    return bad != 0;
}

/* wmerge <nper> <steps> <blob1> <trace1> [<blob2> <trace2> ...]: several weathers of one building
 * stepped as ONE group (nper envs each, weather-major) must give, for every env, exactly the bits
 * that weather gives on its own: per step, the hash of each weather's observations, rewards and
 * temperature fields in the merged group == the hash of the solo run.  Every observation is also
 * checked against that weather's oracle trace (exact). */
static int sb_wmerge_test(int argc, char** argv) {
    if (argc < 6 || (argc - 4) % 2 != 0) { printf("usage: gate wmerge <nper> <steps> <blob> <trace> [<blob> <trace> ...]\n"); return 2; }
    const int N = atoi(argv[2]), S = atoi(argv[3]), W = (argc - 4) / 2;
    std::vector<const char*> bl(W), tr(W);
    for (int w = 0; w < W; w++) { bl[w] = argv[4 + 2 * w]; tr[w] = argv[5 + 2 * w]; }
    std::vector<unsigned char> pristine;
    sb_group_save(pristine);
    std::vector<std::vector<std::vector<double>>> nat(W);
    std::vector<std::vector<std::vector<double>>> oobs(W);   /* oracle obs per step */
    int nfld = 0, bad_obs = 0;
    auto read_obs = [&](const char* path, std::vector<std::vector<double>>& out) -> bool {
        FILE* f = fopen(path, "rb");
        if (f == NULL) return false;
        int hdr[7];
        if (fread(hdr, 4, 7, f) != 7) return false;
        const int t_nc = hdr[2], t_nd = hdr[3], t_nsteps = hdr[4], t_nfld = hdr[6];
        nfld = t_nfld;
        std::vector<double> buf((size_t)t_nc + t_nd + n_action + n_zones + 2 * k + 1 + ncv + 1 + 12);
        std::vector<double> o0(t_nfld);
        if (fread(o0.data(), 8, t_nfld, f) != (size_t)t_nfld) return false;
        out.assign(t_nsteps, std::vector<double>(t_nfld));
        for (int st = 0; st < t_nsteps; st++) {
            if (fread(buf.data(), 8, buf.size(), f) != buf.size()) return false;
            if (fread(out[st].data(), 8, t_nfld, f) != (size_t)t_nfld) return false;
        }
        fclose(f);
        return true;
    };
    auto hash_block = [&](int e0, int ne) -> unsigned long long {
        std::vector<float> ob((size_t)ne * SB_OBS);
        std::vector<double> rw(ne), tp((size_t)ne * ncv);
        cudaMemcpy(ob.data(), d_obs + (size_t)e0 * SB_OBS, 4 * ob.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(rw.data(), d_rew + e0, 8 * rw.size(), cudaMemcpyDeviceToHost);
        sb_field_get(tp, e0, ne);
        unsigned long long h = 1469598103934665603ULL;
        h = fnv(h, ob.data(), 4 * ob.size()); h = fnv(h, rw.data(), 8 * rw.size());
        return fnv(h, tp.data(), 8 * tp.size());
    };
    auto check_obs = [&](int e0, int ne, const std::vector<double>& ref) {
        std::vector<float> ob((size_t)ne * SB_OBS);
        cudaMemcpy(ob.data(), d_obs + (size_t)e0 * SB_OBS, 4 * ob.size(), cudaMemcpyDeviceToHost);
        for (int e = 0; e < ne; e++)
            for (int i = 0; i < nfld; i++) if ((double)ob[(size_t)e * SB_OBS + i] != ref[i]) bad_obs++;
    };
    auto run = [&](int nenv_, const std::vector<int>& wof, int stp) {
        std::vector<double> natv((size_t)nenv_ * n_action);
        for (int e = 0; e < nenv_; e++)
            memcpy(&natv[(size_t)e * n_action], nat[wof[e]][stp].data(), 8 * (size_t)n_action);
        CU(cudaMemcpy(d_native, natv.data(), 8 * natv.size(), cudaMemcpyHostToDevice));
        std::vector<int> sv(nenv_, stp + 1);
        CU(cudaMemcpy(d_stepv, sv.data(), 4 * (size_t)nenv_, cudaMemcpyHostToDevice));
        if (sb_gpu_run_step() != 0) return 1;
        CU(cudaDeviceSynchronize());
        return 0;
    };
    std::vector<std::vector<unsigned long long>> solo(W);
    for (int w = 0; w < W; w++) {
        sb_group_load(pristine);
        nenv = N;
        if (sb_gpu_setup(bl[w]) != 0) return 1;
        g_obs = d_obs;
        if (sb_gpu_reset() != 0) return 1;
        if (!read_trace_actions(tr[w], nat[w]) || !read_obs(tr[w], oobs[w]) || (int)nat[w].size() < S) {
            printf("bad trace %s\n", tr[w]); return 1;
        }
        std::vector<int> wof(N, w);
        for (int st = 0; st < S; st++) {
            if (run(N, wof, st) != 0) return 1;
            solo[w].push_back(hash_block(0, N));
            check_obs(0, N, oobs[w][st]);
        }
    }
    sb_group_load(pristine);
    nenv = N * W;
    std::vector<int> cnt(W, N);
    if (sb_gpu_setup_w(bl.data(), W, cnt.data()) != 0) return 1;
    g_obs = d_obs;
    if (sb_gpu_reset() != 0) return 1;
    std::vector<int> wof(nenv);
    for (int e = 0; e < nenv; e++) wof[e] = h_envw[e];
    int bad = 0;
    for (int st = 0; st < S; st++) {
        if (run(nenv, wof, st) != 0) return 1;
        for (int w = 0; w < W; w++) {
            if (hash_block(w * N, N) != solo[w][st]) { if (!bad) printf("first mismatch: weather %d step %d\n", w, st); bad++; }
            check_obs(w * N, N, oobs[w][st]);
        }
    }
    printf("wmerge: %d weathers x %d envs x %d steps: %d differing (weather, step) hashes merged vs solo, "
           "%d observation values differ from the oracle traces\n", W, N, S, bad, bad_obs);
    printf(bad || bad_obs ? "SB GPU WMERGE: FAIL\n" : "SB GPU WMERGE: OK\n");
    return bad || bad_obs;
}

/* mgtest <nper> <steps> <blob> <trace> [<blob> <trace> ...]: several groups (one per blob) stepped
 * with the fused multi-group solve (sb_mg_step, the trainer's path) must give every group exactly the
 * bits it gets stepped alone (sb_gpu_run_step).  Also times both: solo groups one after the other on
 * one stream, and the fused phased step (SB_GRAPH=1: each captured once into a CUDA graph). */
static int sb_mgtest(int argc, char** argv) {
    if (argc < 6 || (argc - 4) % 2 != 0) { printf("usage: gate mgtest <nper> <steps> <blob> <trace> ...\n"); return 2; }
    const int N = atoi(argv[2]), S = atoi(argv[3]), G = (argc - 4) / 2;
    std::vector<unsigned char> pristine;
    sb_group_save(pristine);
    std::vector<std::vector<std::vector<double>>> nat(G);
    std::vector<SbGroup> solo(G), fused(G);
    for (int pass = 0; pass < 2; pass++)
        for (int gi = 0; gi < G; gi++) {
            sb_group_load(pristine);
            nenv = N;
            if (sb_gpu_setup(argv[4 + 2 * gi]) != 0) return 1;
            g_obs = d_obs;
            if (sb_gpu_reset() != 0) return 1;
            if (pass == 0 && !read_trace_actions(argv[5 + 2 * gi], nat[gi])) { printf("bad trace\n"); return 1; }
            SbGroup& g = pass == 0 ? solo[gi] : fused[gi];
            sb_group_save(g.st);
            g.off = 0; g.n = N;
            cudaStreamCreateWithFlags(&g.s, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&g.done, cudaEventDisableTiming);
        }
    int Smin = S;
    for (int gi = 0; gi < G; gi++) Smin = std::min(Smin, (int)nat[gi].size());
    const cudaStream_t main_stream = stream;
    auto set_actions = [&](std::vector<SbGroup>& gs, int st) {
        for (int gi = 0; gi < G; gi++) {
            sb_group_load(gs[gi].st);
            std::vector<double> natv((size_t)N * n_action);
            for (int e = 0; e < N; e++) memcpy(&natv[(size_t)e * n_action], nat[gi][st].data(), 8 * (size_t)n_action);
            cudaMemcpy(d_native, natv.data(), 8 * natv.size(), cudaMemcpyHostToDevice);
            std::vector<int> sv(N, st + 1);
            cudaMemcpy(d_stepv, sv.data(), 4 * (size_t)N, cudaMemcpyHostToDevice);
        }
    };
    auto hash_all = [&](std::vector<SbGroup>& gs, std::vector<unsigned long long>& out) {
        for (int gi = 0; gi < G; gi++) {
            sb_group_load(gs[gi].st);
            std::vector<float> ob((size_t)N * SB_OBS);
            std::vector<double> rw(N), tp((size_t)N * ncv);
            cudaMemcpy(ob.data(), d_obs, 4 * ob.size(), cudaMemcpyDeviceToHost);
            cudaMemcpy(rw.data(), d_rew, 8 * rw.size(), cudaMemcpyDeviceToHost);
            sb_field_get(tp, 0, (int)(tp.size() / ncv));
            unsigned long long h = 1469598103934665603ULL;
            h = fnv(h, ob.data(), 4 * ob.size()); h = fnv(h, rw.data(), 8 * rw.size());
            out.push_back(fnv(h, tp.data(), 8 * tp.size()));
        }
    };
    const bool graph = getenv("SB_GRAPH") != NULL;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    std::vector<unsigned long long> hs, hf;
    double ms_solo = 0.0, ms_fused = 0.0;
    /* solo */
    g_groups = solo;
    cudaGraphExec_t gx = NULL;
    for (int st = 0; st < Smin; st++) {
        set_actions(solo, st);
        cudaEventRecord(t0, main_stream);
        if (graph) {
            if (gx == NULL) {
                cudaGraph_t gr;
                CU(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeGlobal));
                for (int gi = 0; gi < G; gi++) { sb_group_load(solo[gi].st); stream = main_stream; if (sb_gpu_run_step() != 0) return 1; }
                CU(cudaStreamEndCapture(main_stream, &gr));
                CU(cudaGraphInstantiate(&gx, gr, 0));
            }
            CU(cudaGraphLaunch(gx, main_stream));
        } else
            for (int gi = 0; gi < G; gi++) { sb_group_load(solo[gi].st); stream = main_stream; if (sb_gpu_run_step() != 0) return 1; }
        cudaEventRecord(t1, main_stream);
        CU(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms, t0, t1); if (st > 0) ms_solo += ms;
        hash_all(solo, hs);
    }
    /* fused */
    g_groups = fused;
    g_mg_force = true;
    sb_mg_setup();
    if (!g_mg_on) { printf("fused solve not enabled\n"); return 1; }
    gx = NULL;
    for (int st = 0; st < Smin; st++) {
        set_actions(fused, st);
        cudaEventRecord(t0, main_stream);
        if (graph) {
            if (gx == NULL) {
                cudaGraph_t gr;
                CU(cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeGlobal));
                sb_mg_step(main_stream, false);
                CU(cudaStreamEndCapture(main_stream, &gr));
                CU(cudaGraphInstantiate(&gx, gr, 0));
            }
            CU(cudaGraphLaunch(gx, main_stream));
        } else
            sb_mg_step(main_stream, false);
        cudaEventRecord(t1, main_stream);
        CU(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms, t0, t1); if (st > 0) ms_fused += ms;
        hash_all(fused, hf);
    }
    int bad = 0;
    for (size_t i = 0; i < hs.size(); i++) if (hs[i] != hf[i]) { if (!bad) printf("first mismatch: group %zu step %zu\n", i % G, i / G); bad++; }
    printf("mgtest: %d groups x %d envs x %d steps: %d differing (group, step) hashes fused vs solo\n", G, N, Smin, bad);
    printf("ms/step (all groups): solo one after the other %.2f, fused %.2f%s\n", ms_solo / (Smin - 1),
           ms_fused / (Smin - 1), graph ? " (CUDA graphs)" : "");
    printf(bad ? "SB GPU MGTEST: FAIL\n" : "SB GPU MGTEST: OK\n");
    return bad != 0;
}

/* resetiso <blob> <trace> <nenv> <steps> <k>: the per-env restart.  Three runs from the same start
 * with the trace's actions: no restart; ONLY env 0 restarts after step k (its flag alone); EVERY env
 * restarts after step k (the group restart the multi-episode gate checks against the CPU env).  In
 * the second run every other env must be bit-identical to the first run and env 0 bit-identical to
 * the third (observations, rewards, field), at every step. */
static int sb_resetiso_test(int argc, char** argv) {
    if (argc < 7) { printf("usage: gate resetiso <blob> <trace> <nenv> <steps> <k>\n"); return 2; }
    const int N = atoi(argv[4]), S = atoi(argv[5]), K = atoi(argv[6]);
    if (N < 2) { printf("needs >= 2 envs\n"); return 2; }
    std::vector<unsigned char> pristine;
    sb_group_save(pristine);
    std::vector<std::vector<double>> nat;
    auto envhash = [&](int e) -> unsigned long long {
        std::vector<float> ob(SB_OBS);
        double rw;
        std::vector<double> tp;
        cudaMemcpy(ob.data(), d_obs + (size_t)e * SB_OBS, 4 * SB_OBS, cudaMemcpyDeviceToHost);
        cudaMemcpy(&rw, d_rew + e, 8, cudaMemcpyDeviceToHost);
        sb_field_get(tp, e, 1);
        unsigned long long h = 1469598103934665603ULL;
        h = fnv(h, ob.data(), 4 * ob.size()); h = fnv(h, &rw, 8);
        return fnv(h, tp.data(), 8 * tp.size());
    };
    std::vector<std::vector<unsigned long long>> H[3];
    for (int run = 0; run < 3; run++) {
        sb_group_load(pristine);
        nenv = N;
        if (sb_gpu_setup(argv[2]) != 0) return 1;
        g_obs = d_obs;
        if (sb_gpu_reset() != 0) return 1;
        if (run == 0) {
            if (!read_trace_actions(argv[3], nat) || (int)nat.size() < S) { printf("bad trace\n"); return 1; }
        }
        std::vector<int> sv(N);
        for (int st = 0; st < S; st++) {
            std::vector<double> natv((size_t)N * n_action);
            for (int e = 0; e < N; e++) memcpy(&natv[(size_t)e * n_action], nat[st].data(), 8 * (size_t)n_action);
            CU(cudaMemcpy(d_native, natv.data(), 8 * natv.size(), cudaMemcpyHostToDevice));
            CU(cudaMemcpy(sv.data(), d_stepv, 4 * (size_t)N, cudaMemcpyDeviceToHost));
            for (int e = 0; e < N; e++) sv[e] += 1;
            CU(cudaMemcpy(d_stepv, sv.data(), 4 * (size_t)N, cudaMemcpyHostToDevice));
            if (sb_gpu_run_step() != 0) return 1;
            if (run > 0 && st == K) {
                std::vector<int> fl(N + 1, run == 2 ? 1 : 0);
                fl[0] = 1; fl[1] = 1;               /* run 1: env 0 only; run 2: every env */
                CU(cudaMemcpy(d_reset, fl.data(), 4 * fl.size(), cudaMemcpyHostToDevice));
                sb_enqueue_reset(d_reset);
            }
            CU(cudaDeviceSynchronize());
            std::vector<unsigned long long> h(N);
            for (int e = 0; e < N; e++) h[e] = envhash(e);
            H[run].push_back(h);
        }
    }
    int bad = 0, bad0 = 0;
    for (int st = 0; st < S; st++) {
        for (int e = 1; e < N; e++) if (H[0][st][e] != H[1][st][e]) { if (!bad) printf("first leak: env %d step %d\n", e, st); bad++; }
        if (H[1][st][0] != H[2][st][0]) { if (!bad0) printf("env 0 differs from the group restart at step %d\n", st); bad0++; }
    }
    printf("resetiso: %d envs x %d steps, env 0 restarted alone after step %d: %d other-env (env, step) hashes "
           "differ from no restart, %d env-0 steps differ from the whole-group restart\n", N, S, K, bad, bad0);
    printf(bad || bad0 ? "SB GPU RESETISO: FAIL\n" : "SB GPU RESETISO: OK\n");
    return bad || bad0;
}

/* basecheck <blob> <steps> <out.bin>: one env, base-only actions (canonical + residual with a zero policy, i.e.
 * the on-device pid_water), full episode.  Writes per step: the oracle observation the base saw (obs_len float32),
 * the base's normalized actions in native order (n_action float32), the step reward (float64).  Header: int32
 * {obs_len, n_action, steps}.  research/tools/basecheck.py compares the actions with the starter-kit pid_water
 * policy on the same observations and the reward with the GPU pid_water run. */
static int sb_basecheck(int argc, char** argv) {
    if (argc < 5) { printf("usage: gate basecheck <blob> <steps> <out.bin>\n"); return 2; }
    const int S = atoi(argv[3]);
    g_canon = 1; g_resid = 1;
    nenv = 1;
    if (sb_gpu_setup(argv[2]) != 0) return 1;
    g_obs = d_obsraw;
    g_obs_pol = d_obs;
    if (sb_gpu_reset() != 0) return 1;
    FILE* f = fopen(argv[4], "wb");
    if (f == NULL) return 1;
    int hdr[3] = {obs_len_, n_action, S};
    fwrite(hdr, 4, 3, f);
    std::vector<float> ob(obs_len_), ba(n_action);
    double tot = 0.0;
    for (int st = 0; st < S; st++) {
        CU(cudaMemcpy(ob.data(), d_obsraw, 4 * (size_t)obs_len_, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(ba.data(), d_base, 4 * (size_t)n_action, cudaMemcpyDeviceToHost));
        decode_res_k<<<(n_action + 127) / 128, 128, 0, stream>>>(d_cmap, d_base, NULL, 0, n_action, 1, d_act_disc,
            d_act_mn, d_act_mx, d_act_mnr, d_act_mxr, 1, g_res_scale, g_res_dscale, NULL, d_native);
        int sv = st + 1;
        CU(cudaMemcpyAsync(d_stepv, &sv, 4, cudaMemcpyHostToDevice, stream));
        if (sb_gpu_run_step() != 0) return 1;
        CU(cudaDeviceSynchronize());
        double r = 0.0;
        CU(cudaMemcpy(&r, d_rew, 8, cudaMemcpyDeviceToHost));
        tot += (double)(float)r;
        fwrite(ob.data(), 4, ob.size(), f);
        fwrite(ba.data(), 4, ba.size(), f);
        fwrite(&r, 8, 1, f);
    }
    fclose(f);
    printf("basecheck: %d steps, mean reward %.10g (score %.10g)\n", S, tot / S, fabs(tot / S));
    return 0;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    if (argc > 1 && strcmp(argv[1], "basecheck") == 0) return sb_basecheck(argc, argv);
    if (argc > 1 && strcmp(argv[1], "resetiso") == 0) return sb_resetiso_test(argc, argv);
    if (argc > 1 && strcmp(argv[1], "mgtest") == 0) return sb_mgtest(argc, argv);
    if (argc > 1 && strcmp(argv[1], "groups") == 0) return sb_groups_test(argc, argv);
    if (const char* p = getenv("SB_GATE_START_OFFSET")) {
        g_test_start = atoi(p);
        if (g_test_start < 0 || g_test_start > 6 * 288 || g_test_start % 288 != 0) {
            fprintf(stderr, "invalid SB_GATE_START_OFFSET\n");
            return 2;
        }
    }
    if (argc > 1 && strcmp(argv[1], "wmerge") == 0) return sb_wmerge_test(argc, argv);
    if (argc < 3) {
        printf("usage: %s <blob.sbb> <trace.step> [nenv]\n", argv[0]);
        return 2;
    }
    nenv = argc > 3 ? atoi(argv[3]) : 1;
    if (nenv < 1 || nenv > SB_NE) return 2;
    if (sb_gpu_setup(argv[1]) != 0) return 1;
    g_obs = d_obs;                  /* the step writes the caller's obs buffer */
    if (sb_gpu_reset() != 0) return 1;
    sb_print_solve_layout("batch", nenv, 1);
    /* trace "-" is allowed for `play mid|random` (no oracle trace needed) */
    const bool no_trace = strcmp(argv[2], "-") == 0 && argc > 4 && strcmp(argv[4], "play") == 0;
    FILE* f = no_trace ? NULL : fopen(argv[2], "rb");
    if (f == NULL && !no_trace) return 1;
    /* header: n_cvs, n_zones, nc, nd, nsteps, k, n_field */
    int hdr[7] = {0, 0, 0, 0, 0, 0, 0};
    if (!no_trace && fread(hdr, 4, 7, f) != 7) return 1;
    int t_nc = hdr[2], t_nd = hdr[3], t_nsteps = hdr[4], t_nfld = hdr[6];
    std::vector<double> obs0(t_nfld);
    int reset_obs_bad = 0;
    if (fread(obs0.data(), 8, t_nfld, f) != (size_t)t_nfld) return 1;   /* reset obs */
    {
        /* the reset observation, against the oracle's */
        std::vector<float> ro((size_t)nenv * SB_OBS);
        CU(cudaMemcpy(ro.data(), d_obs, 4 * (size_t)nenv * SB_OBS, cudaMemcpyDeviceToHost));
        double worst_r = 0.0; int nbad = 0, first_bad = -1; (void)0;
        for (int e2 = 0; e2 < nenv; e2++)
            for (int i = 0; i < t_nfld; i++) {
                double d = fabs((double)ro[(size_t)e2 * SB_OBS + i] - obs0[i]);
                if (d > worst_r) worst_r = d;
                if (d != 0.0 && e2 == 0) { nbad++; if (first_bad < 0) first_bad = i; }
            }
        if (getenv("SB_VERBOSE"))
            for (int i = 0; i < t_nfld; i++)
                if ((double)ro[i] != obs0[i]) printf("  reset obs[%d]: gpu=%.9g oracle=%.9g\n", i, (double)ro[i], obs0[i]);
        printf("reset obs vs oracle: worst=%.3e, %d of %d entries differ (first at %d)\n",
               worst_r, nbad, t_nfld, first_bad);
        reset_obs_bad = (worst_r != 0.0);
    }
    if (argc > 4 && strcmp(argv[4], "play") == 0) {
        /* play <policy> [steps] [fps]: step the batch with a scripted policy and write
         * SB_SNAPSHOT files for the viewer (tools/watch.sh).  Policies act in native units:
         * replay = the trace's recorded actions, looped; mid = the middle of every
         * continuous range (discrete at its minimum); random = uniform per env.  The batch
         * restarts in-step after the scenario's episode length, like the trainer. */
        SnapCfg sc = snap_cfg();
        if (sc.path == NULL) { printf("play needs SB_SNAPSHOT=<file>\n"); return 2; }
        const char* pol = argc > 5 ? argv[5] : "replay";
        const int ep_len = sb_blob_ep_len();
        g_ep_len = ep_len;
        const long total = argc > 6 ? atol(argv[6]) : (long)ep_len;
        const double fps = argc > 7 ? atof(argv[7]) : 0.0;
        std::vector<std::vector<double>> nat_tr;
        if (strcmp(pol, "replay") == 0) {
            if (no_trace) { printf("play replay needs the oracle trace, not \"-\"\n"); return 2; }
            std::vector<double> c1(t_nc), d1(t_nd), oc(n_zones), g1(k), T1(k), s1(1), tn(ncv), rw1(1), rq(12), ob(t_nfld);
            nat_tr.assign(t_nsteps, std::vector<double>(n_action));
            for (int st = 0; st < t_nsteps; st++) {
                if (fread(c1.data(), 8, t_nc, f) != (size_t)t_nc || fread(d1.data(), 8, t_nd, f) != (size_t)t_nd ||
                    fread(nat_tr[st].data(), 8, n_action, f) != (size_t)n_action ||
                    fread(oc.data(), 8, n_zones, f) != (size_t)n_zones || fread(g1.data(), 8, k, f) != (size_t)k ||
                    fread(T1.data(), 8, k, f) != (size_t)k || fread(s1.data(), 8, 1, f) != 1 ||
                    fread(tn.data(), 8, ncv, f) != (size_t)ncv || fread(rw1.data(), 8, 1, f) != 1 ||
                    fread(rq.data(), 8, 12, f) != 12 || fread(ob.data(), 8, t_nfld, f) != (size_t)t_nfld) return 1;
            }
        } else if (strcmp(pol, "mid") != 0 && strcmp(pol, "random") != 0) {
            printf("unknown policy '%s' (replay | mid | random)\n", pol);
            return 2;
        }
        const double* amn = sb_f64(&blob, "act_min_native");
        const double* amx = sb_f64(&blob, "act_max_native");
        const uint8_t* adisc = sb_u8(&blob, "act_is_discrete");
        std::vector<double> natv((size_t)nenv * n_action);
        unsigned long long rs = 88172645463325252ULL;
        auto urand = [&]() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; };
        g_test_ep_len = ep_len;
        double ep_ret = 0.0;
        for (long st = 0; st < total; st++) {
            const int sv = (int)(st % ep_len) + 1;
            for (int e2 = 0; e2 < nenv; e2++)
                for (int a = 0; a < n_action; a++) {
                    double v;
                    if (!nat_tr.empty()) v = nat_tr[st % t_nsteps][a];
                    else if (pol[0] == 'm') v = adisc[a] ? amn[a] : 0.5 * (amn[a] + amx[a]);
                    else v = adisc[a] ? (urand() < 0.5 ? amn[a] : amx[a]) : amn[a] + urand() * (amx[a] - amn[a]);
                    natv[(size_t)e2 * n_action + a] = v;
                }
            CU(cudaMemcpy(d_native, natv.data(), 8 * natv.size(), cudaMemcpyHostToDevice));
            std::vector<int> svv(nenv, sv);
            CU(cudaMemcpyAsync(d_stepv, svv.data(), 4 * (size_t)nenv, cudaMemcpyHostToDevice, stream));
            if (sb_gate_step() != 0) return 1;
            CU(cudaDeviceSynchronize());
            double rw1 = 0.0;
            CU(cudaMemcpy(&rw1, d_rew + sc.env, 8, cudaMemcpyDeviceToHost));
            ep_ret += rw1;
            { float er = (float)ep_ret; CU(cudaMemcpy(d_epret + sc.env, &er, 4, cudaMemcpyHostToDevice)); }
            if ((st + 1) % sc.every == 0 && sb_snapshot_write(sc.env, sc.path) != 0) { printf("snapshot write failed\n"); return 1; }
            if (sv == ep_len) ep_ret = 0.0;
            if (fps > 0.0) usleep((useconds_t)(1e6 / fps));
        }
        printf("play done: %ld steps (%s)\n", total, pol);
        return 0;
    }
    if (argc > 4) {
        /* Multi-episode mode: replay the trace's native actions for every episode
         * of a reference written by csim's test_episodes (the CPU env's
         * reset/step sequence) and diff the reset observation, every observation
         * and every reward.  Episodes after the first go through sb_gpu_reset()
         * exactly as the trainer's restart does. */
        FILE* rf = fopen(argv[4], "rb");
        if (rf == NULL) { printf("cannot open reference %s\n", argv[4]); return 1; }
        int rh[3];
        if (fread(rh, 4, 3, rf) != 3 || rh[0] != t_nfld) { printf("bad reference header\n"); return 1; }
        const int ref_steps = rh[1], E = rh[2];
        std::vector<std::vector<double>> natAll(t_nsteps, std::vector<double>(n_action));
        {
            std::vector<double> c1(t_nc), d1(t_nd), oc(n_zones), g1(k), T1(k), s1(1), tn(ncv), rw(1), rq(12), ob(t_nfld);
            for (int st = 0; st < t_nsteps; st++) {
                if (fread(c1.data(), 8, t_nc, f) != (size_t)t_nc || fread(d1.data(), 8, t_nd, f) != (size_t)t_nd ||
                    fread(natAll[st].data(), 8, n_action, f) != (size_t)n_action ||
                    fread(oc.data(), 8, n_zones, f) != (size_t)n_zones || fread(g1.data(), 8, k, f) != (size_t)k ||
                    fread(T1.data(), 8, k, f) != (size_t)k || fread(s1.data(), 8, 1, f) != 1 ||
                    fread(tn.data(), 8, ncv, f) != (size_t)ncv || fread(rw.data(), 8, 1, f) != 1 ||
                    fread(rq.data(), 8, 12, f) != 12 || fread(ob.data(), 8, t_nfld, f) != (size_t)t_nfld) return 1;
            }
        }
        const int run_steps = ref_steps;
        std::vector<float> ref_o(t_nfld), got((size_t)nenv * SB_OBS);
        int fail = 0;
        /* SB_INSTEP=1: no host reset between episodes - the step itself restarts the
         * batch after `run_steps` steps, as in the trainer.  The observation buffer
         * after an episode's last step then holds the NEXT episode's reset
         * observation (the terminal observation is overwritten, as in the env), so
         * that step is compared against the next reset observation instead. */
        const bool instep = getenv("SB_INSTEP") != NULL;
        if (instep) g_test_ep_len = run_steps;
        for (int ep = 0; ep < E; ep++) {
            if (ep > 0 && !instep && sb_gpu_reset() != 0) return 1;
            if (fread(ref_o.data(), 4, t_nfld, rf) != (size_t)t_nfld) return 1;
            CU(cudaMemcpy(got.data(), d_obs, 4 * (size_t)nenv * SB_OBS, cudaMemcpyDeviceToHost));
            double wR = 0.0, wO = 0.0, wRw2 = 0.0; long nOd = 0;
            for (int e2 = 0; e2 < nenv; e2++)
                for (int i = 0; i < t_nfld; i++)
                    wR = fmax(wR, fabs((double)got[(size_t)e2 * SB_OBS + i] - (double)ref_o[i]));
            for (int st = 0; st < run_steps; st++) {
                std::vector<double> natv((size_t)nenv * n_action);
                for (int e2 = 0; e2 < nenv; e2++)
                    memcpy(&natv[(size_t)e2 * n_action], natAll[st % t_nsteps].data(), 8 * (size_t)n_action);
                CU(cudaMemcpy(d_native, natv.data(), 8 * natv.size(), cudaMemcpyHostToDevice));
                std::vector<int> sv(nenv, st + 1);
                CU(cudaMemcpyAsync(d_stepv, sv.data(), 4 * (size_t)nenv, cudaMemcpyHostToDevice, stream));
                if (sb_gate_step() != 0) return 1;
                CU(cudaDeviceSynchronize());
                CU(cudaMemcpy(got.data(), d_obs, 4 * (size_t)nenv * SB_OBS, cudaMemcpyDeviceToHost));
                std::vector<double> rr(nenv);
                CU(cudaMemcpy(rr.data(), d_rew, 8 * (size_t)nenv, cudaMemcpyDeviceToHost));
                if (fread(ref_o.data(), 4, t_nfld, rf) != (size_t)t_nfld) return 1;
                double ref_r;
                if (fread(&ref_r, 8, 1, rf) != 1) return 1;
                const bool terminal_overwritten = instep && st == run_steps - 1;
                for (int e2 = 0; e2 < nenv; e2++) {
                    for (int i = 0; i < t_nfld && !terminal_overwritten; i++) {
                        double d = fabs((double)got[(size_t)e2 * SB_OBS + i] - (double)ref_o[i]);
                        wO = fmax(wO, d);
                        if (d != 0.0) nOd++;
                    }
                    wRw2 = fmax(wRw2, fabs(rr[e2] - ref_r));
                }
            }
            printf("episode %d: reset obs err=%.3e, step obs err=%.3e (%ld values differ), reward err=%.3e\n",
                   ep, wR, wO, nOd, wRw2);
            if (wR != 0.0 || wO != 0.0 || wRw2 > 1e-6) fail = 1;
        }
        printf(fail ? "SB GPU MULTI-EPISODE: FAIL\n" : "SB GPU MULTI-EPISODE: OK\n");
        return fail;
    }
    std::vector<double> cont(t_nc), disc(t_nd), nat(n_action), occV(n_zones);
    std::vector<double> gg(k), TT(k), swt(1), tnew(ncv), rew(1), rp(12), obsv(t_nfld);
    double wG = 0, wT = 0, wF = 0, wV = 0, wSW = 0, wRw = 0, wObs = 0;
    double tstep = 0.0, eSW = 0.0;
    std::vector<double> boil_saved(nenv, 0.0);
    for (int st = 0; st < t_nsteps; st++) {
        if (fread(cont.data(), 8, t_nc, f) != (size_t)t_nc) return 1;
        if (fread(disc.data(), 8, t_nd, f) != (size_t)t_nd) return 1;
        if (fread(nat.data(), 8, n_action, f) != (size_t)n_action) return 1;
        if (fread(occV.data(), 8, n_zones, f) != (size_t)n_zones) return 1;
        if (fread(gg.data(), 8, k, f) != (size_t)k) return 1;
        if (fread(TT.data(), 8, k, f) != (size_t)k) return 1;
        if (fread(swt.data(), 8, 1, f) != 1) return 1;
        eSW = 0.0;
        GpuSim h0[SB_NE];
        CU(cudaMemcpy(h0, d_sims, sizeof(GpuSim) * nenv, cudaMemcpyDeviceToHost));
        for (int e = 0; e < nenv; e++)
            eSW = fmax(eSW, fabs(boil_saved[e] - swt[0]));
        wSW = fmax(wSW, eSW);
        if (fread(tnew.data(), 8, ncv, f) != (size_t)ncv) return 1;
        if (fread(rew.data(), 8, 1, f) != 1) return 1;
        if (fread(rp.data(), 8, 12, f) != 12) return 1;
        if (fread(obsv.data(), 8, t_nfld, f) != (size_t)t_nfld) return 1;
        double t0 = now_ms();
        std::vector<double> natv((size_t)nenv * n_action);
        for (int e = 0; e < nenv; e++)
            memcpy(&natv[(size_t)e * n_action], nat.data(), 8 * (size_t)n_action);
        CU(cudaMemcpy(d_native, natv.data(), 8 * (size_t)nenv * n_action,
                      cudaMemcpyHostToDevice));
        std::vector<int> sv(nenv, g_test_start + st + 1);
        CU(cudaMemcpyAsync(d_stepv, sv.data(), 4 * (size_t)nenv,
                           cudaMemcpyHostToDevice, stream));
        // d_V before the step is the occupancy used by the physics interval.
        // The post-step d_V has already advanced to the following interval.
        std::vector<double> intervalV((size_t)nenv * n_zones);
        CU(cudaMemcpy(intervalV.data(), d_V, 8 * intervalV.size(), cudaMemcpyDeviceToHost));
        if (sb_gate_step() != 0) return 1;
        CU(cudaDeviceSynchronize());   /* the step no longer syncs itself */
        tstep += now_ms() - t0;
        std::vector<double> myT((size_t)nenv * ncv), myG((size_t)nenv * k),
            myTT((size_t)nenv * k);
        sb_field_get(myT, 0, nenv);
        CU(cudaMemcpy(myG.data(), d_g, 8 * (size_t)nenv * k,
                      cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(myTT.data(), d_t2, 8 * (size_t)nenv * k,
                      cudaMemcpyDeviceToHost));
        double eG = 0, eT = 0, eF = 0;
        for (size_t i = 0; i < (size_t)nenv * k; i++) {
            eG = fmax(eG, fabs(myG[i] - gg[i % k]));
            eT = fmax(eT, fabs(myTT[i] - TT[i % k]));
        }
        for (size_t e = 0; e < (size_t)nenv; e++)
            for (int p = 0; p < ncv; p++)
                eF = fmax(eF, fabs(myT[e * ncv + p] - tnew[p]));
        CU(cudaMemcpy(h0, d_sims, sizeof(GpuSim) * nenv, cudaMemcpyDeviceToHost));
        for (int e = 0; e < nenv; e++) boil_saved[e] = h0[e].boiler_current;
        wG = fmax(wG, eG); wT = fmax(wT, eT); wF = fmax(wF, eF);
        double eV = 0.0;
        for (size_t i = 0; i < intervalV.size(); i++)
            eV = fmax(eV, fabs(intervalV[i] - occV[i % n_zones]));
        wV = fmax(wV, eV);
        std::vector<double> myR((size_t)nenv);
        std::vector<float> myO((size_t)nenv * SB_OBS);
        CU(cudaMemcpy(myR.data(), d_rew, 8 * (size_t)nenv, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(myO.data(), d_obs, 4 * (size_t)nenv * SB_OBS,
                      cudaMemcpyDeviceToHost));
        double eRw = 0.0, eOb = 0.0;
        for (int e = 0; e < nenv; e++) eRw = fmax(eRw, fabs(myR[e] - rew[0]));
        for (size_t i = 0; i < myO.size(); i++) {
            int fi = (int)(i % (size_t)SB_OBS);
            if (fi < t_nfld) eOb = fmax(eOb, fabs((double)myO[i] - obsv[fi]));
        }
        wRw = fmax(wRw, eRw);
        wObs = fmax(wObs, eOb);
        if (st < 12)
            printf("  step %2d: occ=%.1e g=%.1e T=%.1e field=%.3e rew=%.1e obs=%.1e\n",
                   st, eV, eG, eT, eF, eRw, eOb);
    }
    fclose(f);
    printf("phase ms/step: %.2f (%d env, %d rng words)\n", tstep / t_nsteps, nenv,
           0);
    printf("worst: occ=%.3e air_g=%.3e air_T=%.3e field=%.3e rew=%.3e obs=%.3e\n",
           wV, wG, wT, wF, wRw, wObs);
    int bad = reset_obs_bad || (wV != 0.0) || (wG > 1e-9) || (wT > 1e-9) || (wF > 1e-9) ||
              (wObs != 0.0f) || (wRw > 1e-6);
    printf(bad ? "SB GPU ENV STEP: FAIL\n" : "SB GPU ENV STEP: OK\n");
    return bad;
}
#endif
