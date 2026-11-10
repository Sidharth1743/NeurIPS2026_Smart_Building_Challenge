#ifndef SB_SIM_H
#define SB_SIM_H

#include "sb_baked.h"
#include "sb_convect.h"
#include "sb_fdm.h"
#include "sb_occup.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int nr, nc, n_cvs, n_solved, n_ahu, n_vav, n_zones, n_rooms, k, n_action;
    int hws_heating_requests;
    double total_reheat_w;   /* per-step, crosses the FDM phase boundary */
    double coil[256];

    const int32_t *vav_zone_off, *vav_zone_idx, *vav_ahu_idx, *coupled_idx;
    const int32_t *phys_room_idx, *room_off, *room_cells;
    const double *zone_area, *ahu_const, *vav_const;

    const uint8_t *act_kind, *act_act;
    const int32_t *act_idx;

    const double *sched_comfort, *sched_heat_sp, *sched_cool_sp;
    const double *amb;

    int *ahu_run, *ahu_coolreq;
    double *ahu_recirc, *ahu_sat, *ahu_fan, *ahu_flow;

    double *vav_damper, *vav_reheat, *vav_ratio, *vav_heatsp, *vav_coolsp;
    int *vav_mode, *vav_prev_valid, *vav_dcmd_set, *vav_rcmd_set;
    double *vav_dcmd, *vav_rcmd;

    double header_res, flow_factor_sum, pump_head, pump_eff;
    int hws_run, pump_run, is_boiler;
    double boiler_c[10];
    double boiler_setpoint, boiler_return, boiler_current, boiler_step_change;

    double internal_w_m2, standby_frac, occupant_w, cv_area_m2, h, dt_sec;

    SbFdm *fdm; SbConv *conv; SbOccup *occ;
    const int32_t *ap_p, *ap_i; const double *ap_x;   /* air_side_sensed CSR (n_cvs x k) */
    const double *ap_v;

    const uint8_t *f_dev_kind, *f_obs_id;
    const int32_t *f_dev_idx;
    const double *f_norm_mean, *f_norm_var, *aux_time;
    int n_dev, n_aux;
    const double *rw_scalars, *elec_weekday, *elec_weekend, *elec_carbon;
    const double *gas_price; double gas_carbon;
    const int32_t *zone_vav_off, *zone_vav_idx;
    const int64_t *rew_hour, *rew_month;
    const uint8_t *rew_workday;
    double *zo_flow_sp, *zo_flow_act;
    double *avg_temp, *sensing, *vav_area, *ahu_wsum, *ahu_atotal, *ahu_satv;
    double *zone_q, *input_q, *sensed_mean;
    double *air_g, *air_T;
    int step_index;
} SbSim;

int  sb_sim_init(SbSim* s, const SbBlob* b, SbFdm* fdm, SbConv* conv, SbOccup* occ);
void sb_sim_free(SbSim* s);
void sb_sim_reset(SbSim* s, const SbBlob* b);
void sb_sim_apply_native(SbSim* s, const double* native);
void sb_sim_step(SbSim* s, const double* occV);

/* Split step so a group of same-scenario envs can share one batched FDM
 * solve. sb_sim_step == pre + fdm + post, bit-for-bit. */
void sb_sim_step_pre(SbSim* s, const double* occV);
void sb_sim_step_post(SbSim* s);
/* G envs at once; rhs/sol/work are scratch of length n_solved*G each. */
void sb_sim_step_batch(SbSim** s, int G, const double* const* occV,
                       double* rhs, double* sol, double* work);

/* Hybrid -> native decode, mirroring HybridActionEnvironment._format_action +
 * BoundedActionNormalizer.setpoint_value. cont[]/disc[] are the agent actions. */
double ahu_static(SbSim* s, int a);
double vav_unrestricted(SbSim* s, int v);
double vav_flow_demand(SbSim* s, int v);
double total_flow_rate(SbSim* s);
void sb_decode_native(int n, const uint8_t* is_disc,
                      const double* mn, const double* mx,
                      const double* mnr, const double* mxr,
                      const double* cont, const double* disc, double* native);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif
