#include "sb_sim.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RHO_AIR 1.2
#define CP_AIR 1006.0
#define RHO_W 1000.0
#define CP_W 4180.0
#define GRAV 9.8
#define PA_PER_BAR 100000.0
#define MODE_OFF 0
#define MODE_HEAT 1
#define MODE_COOL 2
#define MODE_PASSIVE 3
#define ON 1

static double clampf(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

int sb_sim_init(SbSim* s, const SbBlob* b, SbFdm* fdm, SbConv* conv, SbOccup* occ) {
    memset(s, 0, sizeof(*s));
    s->fdm = fdm; s->conv = conv; s->occ = occ;
    s->n_cvs = fdm->n_cvs; s->n_solved = fdm->n_solved; s->k = fdm->k;
    const int64_t* g = sb_i64(b, "grid");
    s->nr = (int)g[0]; s->nc = (int)g[1];
    s->n_ahu = (int)(sb_find(b, "ahu_const")->count / 9);
    s->n_vav = (int)(sb_find(b, "vav_zone_offsets")->count - 1);
    s->n_zones = (int)sb_find(b, "zone_area")->count;
    s->n_rooms = (int)(sb_find(b, "room_offsets")->count - 1);
    s->n_action = (int)sb_find(b, "act_owner_kind")->count;

    s->vav_zone_off = sb_i32(b, "vav_zone_offsets");
    s->vav_zone_idx = sb_i32(b, "vav_zone_idx");
    s->vav_ahu_idx = sb_i32(b, "vav_ahu_idx");
    s->coupled_idx = sb_i32(b, "coupled_vav_idx");
    s->phys_room_idx = sb_i32(b, "phys_room_idx");
    s->room_off = sb_i32(b, "room_offsets");
    s->room_cells = sb_i32(b, "room_cells");
    s->zone_area = sb_f64(b, "zone_area");
    s->ahu_const = sb_f64(b, "ahu_const");
    s->vav_const = sb_f64(b, "vav_const");
    s->act_kind = sb_u8(b, "act_owner_kind");
    s->act_idx = sb_i32(b, "act_owner_index");
    s->act_act = sb_u8(b, "act_actuator");
    s->sched_comfort = sb_f64(b, "sched_comfort");
    s->sched_heat_sp = sb_f64(b, "sched_heat_sp");
    s->sched_cool_sp = sb_f64(b, "sched_cool_sp");
    s->amb = sb_f64(b, "weather_ambient");
    s->internal_w_m2 = sb_f64_1(b, "internal_gain_w_per_m2", 0.0);
    s->standby_frac = sb_f64_1(b, "internal_gain_standby_fraction", 1.0);
    s->occupant_w = sb_f64_1(b, "occupant_sensible_heat_w", 0.0);
    s->h = sb_f64_1(b, "convection_coefficient", 8.8);
    {
        const double* op = sb_f64(b, "occ_params");   /* [assign, ea, la, ed, ld, step_sec] */
        s->dt_sec = op ? op[5] : 300.0;
    }
    s->cv_area_m2 = 0.04;
    {
        double cv = sb_f64_1(b, "cv_size_cm", 20.0) / 100.0;
        s->cv_area_m2 = cv * cv;
    }
    s->header_res = sb_f64_1(b, "hws_header_resistance", 0.0);
    s->is_boiler = (int)sb_f64_1(b, "hws_is_boiler", 1.0);
    for (int i = 0; i < 10; i++) {
        const double* bc = sb_f64(b, "boiler_const");
        s->boiler_c[i] = bc ? bc[i] : 0.0;
    }
    {
        const double* pc = sb_f64(b, "pump_const");
        s->pump_head = pc ? pc[0] : 0.0;
        s->pump_eff = pc ? pc[1] : 1.0;
    }
    s->zone_vav_off = sb_i32(b, "zone_vav_offsets");
    s->zone_vav_idx = sb_i32(b, "zone_vav_idx");
    s->rew_hour = sb_i64(b, "rew_hour");
    s->rew_month = sb_i64(b, "rew_month");
    s->rew_workday = sb_u8(b, "rew_workday");
    s->f_dev_kind = sb_u8(b, "f_dev_kind");
    s->f_dev_idx = sb_i32(b, "f_dev_idx");
    s->f_obs_id = sb_u8(b, "f_obs_id");
    s->f_norm_mean = sb_f64(b, "f_norm_mean");
    s->f_norm_var = sb_f64(b, "f_norm_var");
    s->aux_time = sb_f64(b, "aux_time");
    s->n_dev = (int)sb_i64_1(b, "n_dev", 0);
    s->n_aux = (int)sb_i64_1(b, "n_aux", 0);
    s->rw_scalars = sb_f64(b, "reward_scalars");
    s->elec_weekday = sb_f64(b, "elec_weekday_price");
    s->elec_weekend = sb_f64(b, "elec_weekend_price");
    s->elec_carbon = sb_f64(b, "elec_carbon_rate");
    s->gas_price = sb_f64(b, "gas_month_price");
    s->gas_carbon = sb_f64_1(b, "gas_carbon_rate", 0.0);
    s->ap_p = sb_i32(b, "air_side_sensed_indptr");
    s->ap_i = sb_i32(b, "air_side_sensed_indices");
    s->ap_x = sb_f64(b, "air_side_sensed_data");

    int na = s->n_ahu, nv = s->n_vav, nz = s->n_zones, kk = s->k;
    s->ahu_run = (int*)calloc(na ? na : 1, sizeof(int));
    s->ahu_coolreq = (int*)calloc(na ? na : 1, sizeof(int));
    s->ahu_recirc = (double*)calloc(na ? na : 1, sizeof(double));
    s->ahu_sat = (double*)calloc(na ? na : 1, sizeof(double));
    s->ahu_fan = (double*)calloc(na ? na : 1, sizeof(double));
    s->ahu_flow = (double*)calloc(na ? na : 1, sizeof(double));
    s->vav_damper = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->vav_reheat = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->vav_ratio = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->vav_heatsp = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->vav_coolsp = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->vav_mode = (int*)calloc(nv ? nv : 1, sizeof(int));
    s->vav_prev_valid = (int*)calloc(nv ? nv : 1, sizeof(int));
    s->vav_dcmd_set = (int*)calloc(nv ? nv : 1, sizeof(int));
    s->vav_rcmd_set = (int*)calloc(nv ? nv : 1, sizeof(int));
    s->vav_dcmd = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->vav_rcmd = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->avg_temp = (double*)calloc(nz ? nz : 1, sizeof(double));
    s->sensing = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->vav_area = (double*)calloc(nv ? nv : 1, sizeof(double));
    s->ahu_wsum = (double*)calloc(na ? na : 1, sizeof(double));
    s->ahu_atotal = (double*)calloc(na ? na : 1, sizeof(double));
    s->ahu_satv = (double*)calloc(na ? na : 1, sizeof(double));
    s->zone_q = (double*)calloc(nz ? nz : 1, sizeof(double));
    s->input_q = (double*)calloc(s->n_cvs, sizeof(double));
    s->sensed_mean = (double*)calloc(kk ? kk : 1, sizeof(double));
    s->air_g = (double*)calloc(kk ? kk : 1, sizeof(double));
    s->air_T = (double*)calloc(kk ? kk : 1, sizeof(double));
    s->zo_flow_sp = (double*)calloc(nz ? nz : 1, sizeof(double));
    s->zo_flow_act = (double*)calloc(nz ? nz : 1, sizeof(double));
    return 0;
}

void sb_sim_free(SbSim* s) {
    free(s->ahu_run); free(s->ahu_coolreq); free(s->ahu_recirc);
    free(s->ahu_sat); free(s->ahu_fan); free(s->ahu_flow);
    free(s->vav_damper); free(s->vav_reheat); free(s->vav_ratio);
    free(s->vav_heatsp); free(s->vav_coolsp); free(s->vav_mode);
    free(s->vav_prev_valid); free(s->vav_dcmd_set); free(s->vav_rcmd_set);
    free(s->vav_dcmd); free(s->vav_rcmd);
    free(s->avg_temp); free(s->sensing); free(s->vav_area);
    free(s->ahu_wsum); free(s->ahu_atotal); free(s->ahu_satv);
    free(s->zone_q); free(s->input_q); free(s->sensed_mean);
    free(s->air_g); free(s->air_T);
    free(s->zo_flow_sp); free(s->zo_flow_act);
    memset(s, 0, sizeof(*s));
}

/* Field-averaged temperature of every physical zone. */
static void sense_avg_temp(SbSim* s) {
    for (int z = 0; z < s->n_zones; z++) {
        int rm = s->phys_room_idx[z];
        int a = s->room_off[rm], e = s->room_off[rm + 1];
        double sum = 0.0;
        for (int t = a; t < e; t++) {
            int p = s->room_cells[2 * t] * s->nc + s->room_cells[2 * t + 1];
            sum += s->fdm->temp[p];
        }
        s->avg_temp[z] = sum / (double)(e - a);
    }
}

/* Area-weighted sensing temperature each VAV sees for its zone. */
static void sense_zone_temps(SbSim* s) {
    for (int v = 0; v < s->n_vav; v++) {
        int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
        double area = 0.0, wt = 0.0;
        for (int t = z0; t < z1; t++) {
            int z = s->vav_zone_idx[t];
            area += s->zone_area[z];
            wt += s->avg_temp[z] * s->zone_area[z];
        }
        s->vav_area[v] = area;
        s->sensing[v] = (area > 0.0) ? wt / area : 0.0;
    }
}

void sb_sim_reset(SbSim* s, const SbBlob* b) {
    for (int a = 0; a < s->n_ahu; a++) {
        const double* c = &s->ahu_const[a * 9];
        s->ahu_run[a] = (int)c[8];
        s->ahu_recirc[a] = c[0];
        s->ahu_sat[a] = c[6];
        s->ahu_fan[a] = c[7];
        s->ahu_flow[a] = 0.0;
        s->ahu_coolreq[a] = 0;
    }
    for (int v = 0; v < s->n_vav; v++) {
        const double* c = &s->vav_const[v * 9];
        s->vav_damper[v] = c[7];
        s->vav_reheat[v] = c[8];
        s->vav_ratio[v] = 1.0;
        s->vav_heatsp[v] = c[5];
        s->vav_coolsp[v] = c[6];
        s->vav_mode[v] = MODE_OFF;
        s->vav_prev_valid[v] = 0;
        s->vav_dcmd_set[v] = 0;
        s->vav_rcmd_set[v] = 0;
        s->vav_dcmd[v] = 0.0;
        s->vav_rcmd[v] = 0.0;
    }
    s->flow_factor_sum = 0.0;
    s->hws_heating_requests = 0;
    s->hws_run = ON;
    s->pump_run = ON;
    s->boiler_setpoint = s->boiler_c[0];
    s->boiler_return = s->boiler_c[1];
    s->boiler_current = s->boiler_c[0];
    s->boiler_step_change = 0.0;
    s->step_index = 0;
    memcpy(s->fdm->temp, sb_f64(b, "temp_init"), sizeof(double) * (size_t)s->n_cvs);
    memcpy(s->fdm->mass_temp, sb_f64(b, "mass_temp_init"),
           sizeof(double) * (size_t)s->n_solved);
    memset(s->input_q, 0, sizeof(double) * (size_t)s->n_cvs);
    /* The observation right after reset reports each VAV's zone temperature,
     * computed from the initial field exactly as the first step's sensing. */
    sense_avg_temp(s);
    sense_zone_temps(s);
}

void sb_sim_apply_native(SbSim* s, const double* native) {
    for (int i = 0; i < s->n_action; i++) {
        double val = native[i];
        int kind = s->act_kind[i], idx = s->act_idx[i], act = s->act_act[i];
        if (kind == 0) {
            const double* c = &s->ahu_const[idx * 9];
            if (act == 0) s->ahu_sat[idx] = val;
            else if (act == 1) s->ahu_fan[idx] = val;
            else if (act == 2) s->ahu_recirc[idx] = clampf(val, 0.0, c[3]);
            else s->ahu_run[idx] = (val == 1.0) ? ON : 0;
        } else if (kind == 1) {
            if (act == 0) {
                s->vav_dcmd[idx] = clampf(val, 0.0, 1.0);
                s->vav_dcmd_set[idx] = 1;
            } else {
                s->vav_rcmd[idx] = clampf(val, 0.0, 1.0);
                s->vav_rcmd_set[idx] = 1;
            }
        } else {
            if (act == 0) s->boiler_setpoint = val;
            else if (act == 1) s->pump_head = val / (GRAV * RHO_W / PA_PER_BAR);
            else { int on = (val == 1.0) ? ON : 0; s->hws_run = on; s->pump_run = on; }
        }
    }
}

double ahu_static(SbSim* s, int a) {
    if (s->ahu_run[a] != ON) return 0.0;
    double sp = s->ahu_fan[a];
    double mn = s->ahu_const[a * 9 + 2];
    return sp > mn ? sp : mn;
}

static double ahu_recirc_temp(SbSim* s, int a) {
    if (s->ahu_atotal[a] > 0.0) return s->ahu_wsum[a] / s->ahu_atotal[a];
    double mean = 0.0;
    for (int p = 0; p < s->n_cvs; p++) mean += s->fdm->temp[p];
    return mean / (double)s->n_cvs;
}

static double get_supply_air_temp(SbSim* s, int a, double recirc, double ambient) {
    double mixed = s->ahu_recirc[a] * recirc
                 + (1.0 - s->ahu_recirc[a]) * ambient;
    if (mixed > s->ahu_sat[a] && s->ahu_run[a] == ON) return s->ahu_sat[a];
    return mixed;
}

double vav_unrestricted(SbSim* s, int v) {
    const double* c = &s->vav_const[v * 9];
    double stat = ahu_static(s, s->vav_ahu_idx[v]);
    double d = s->vav_damper[v] * c[0] * sqrt(stat / c[4]);
    return d > 0.00001 ? d : 0.00001;
}

double vav_flow_demand(SbSim* s, int v) {
    return vav_unrestricted(s, v) * s->vav_ratio[v];
}

static double pump_dpres(SbSim* s) {
    if (s->pump_run != ON) return 0.0;
    return GRAV * RHO_W * s->pump_head / PA_PER_BAR;
}

double total_flow_rate(SbSim* s) {
    double ff = s->flow_factor_sum;
    double dp = pump_dpres(s);
    return ff * sqrt(dp / (1.0 + ff * ff * s->header_res)) / 3600.0;
}

static double reheat_flow_rate(SbSim* s, int v) {
    if (s->flow_factor_sum == 0.0) return 0.0;
    const double* c = &s->vav_const[v * 9];
    double ff = s->vav_reheat[v] * c[1];
    return total_flow_rate(s) * (ff / s->flow_factor_sum);
}

static double reheat_eff(SbSim* s, int v, double sat, double iwt) {
    double air_cap = CP_AIR * RHO_AIR * vav_flow_demand(s, v);
    if (air_cap <= 0.0) return 0.0;
    double water_cap = CP_W * RHO_W * reheat_flow_rate(s, v);
    double e = water_cap / air_cap;
    if (e > 1.0) e = 1.0;
    double gap = iwt - sat;
    if (gap <= 0.0) return e;
    const double* c = &s->vav_const[v * 9];
    double headroom = c[3] - sat;
    if (headroom <= 0.0) return 0.0;
    double hr = headroom / gap;
    return e < hr ? e : hr;
}

static double reheat_energy(SbSim* s, int v, double sat, double iwt) {
    return reheat_eff(s, v, sat, iwt) * CP_AIR * RHO_AIR
         * vav_flow_demand(s, v) * (iwt - sat);
}

static int default_control(SbSim* s, int v, double zt, double hsp, double csp) {
    double mid = 0.5 * (csp - hsp) + hsp;
    if (zt < hsp) s->vav_mode[v] = MODE_HEAT;
    else if (zt > csp) s->vav_mode[v] = MODE_COOL;
    else if (zt < mid && s->vav_mode[v] == MODE_HEAT) s->vav_mode[v] = MODE_HEAT;
    else if (zt > mid && s->vav_mode[v] == MODE_COOL) s->vav_mode[v] = MODE_COOL;
    else s->vav_mode[v] = MODE_OFF;
    return s->vav_mode[v];
}

static int thermostat_update(SbSim* s, int v, double zt, int kk, int comfort) {
    double hsp = s->sched_heat_sp[kk], csp = s->sched_cool_sp[kk];
    if (comfort) {
        default_control(s, v, zt, hsp, csp);
    } else if (s->vav_prev_valid[v] && s->sched_comfort[kk - 1] != 0.0) {
        s->vav_mode[v] = MODE_PASSIVE;
    } else if (s->vav_mode[v] == MODE_PASSIVE && zt > hsp && zt < csp) {
        s->vav_mode[v] = MODE_PASSIVE;
    } else {
        default_control(s, v, zt, hsp, csp);
    }
    s->vav_prev_valid[v] = 1;
    return s->vav_mode[v];
}

static void vav_update_settings(SbSim* s, int v, double sensing, int kk, int comfort) {
    const double* c = &s->vav_const[v * 9];
    s->vav_heatsp[v] = s->sched_heat_sp[kk];
    s->vav_coolsp[v] = s->sched_cool_sp[kk];
    int mode = thermostat_update(s, v, sensing, kk, comfort);
    double damper, reheat;
    if (mode == MODE_HEAT) { damper = 1.0; reheat = 1.0; }
    else if (mode == MODE_COOL) { damper = 1.0; reheat = 0.0; }
    else if (mode == MODE_OFF || mode == MODE_PASSIVE) { damper = c[2]; reheat = 0.0; }
    else { damper = s->vav_damper[v]; reheat = s->vav_reheat[v]; }
    if (s->vav_dcmd_set[v]) damper = s->vav_dcmd[v];
    if (s->vav_rcmd_set[v]) reheat = s->vav_rcmd[v];
    s->vav_damper[v] = damper > c[2] ? damper : c[2];
    s->vav_reheat[v] = reheat;
}

void sb_sim_step_pre(SbSim* s, const double* occV) {
    int kk = s->step_index;
    double ambient = s->amb[kk];
    int comfort = (s->sched_comfort[kk] != 0.0);
    int na = s->n_ahu, nv = s->n_vav, nz = s->n_zones;
    /* add_demand() runs once per VAV during the step. */
    s->hws_heating_requests = nv;

    sense_avg_temp(s);

    for (int a = 0; a < na; a++) { s->ahu_wsum[a] = 0.0; s->ahu_atotal[a] = 0.0; }
    for (int v = 0; v < nv; v++) {
        int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
        double area = 0.0, wt = 0.0;
        for (int t = z0; t < z1; t++) {
            int z = s->vav_zone_idx[t];
            area += s->zone_area[z];
            wt += s->avg_temp[z] * s->zone_area[z];
        }
        s->vav_area[v] = area;
        if (area <= 0.0) { s->sensing[v] = 0.0; continue; }
        double sensing = wt / area;
        s->sensing[v] = sensing;
        vav_update_settings(s, v, sensing, kk, comfort);
        int a = s->vav_ahu_idx[v];
        s->ahu_wsum[a] += sensing * area;
        s->ahu_atotal[a] += area;
    }

    for (int a = 0; a < na; a++)
        s->ahu_satv[a] = get_supply_air_temp(s, a, ahu_recirc_temp(s, a), ambient);

    for (int a = 0; a < na; a++) { s->ahu_flow[a] = 0.0; s->ahu_coolreq[a] = 0; }
    s->flow_factor_sum = 0.0;
    for (int v = 0; v < nv; v++)
        s->flow_factor_sum += s->vav_reheat[v] * s->vav_const[v * 9 + 1];

    double demand[256];
    for (int a = 0; a < na; a++) demand[a] = 0.0;
    for (int v = 0; v < nv; v++) demand[s->vav_ahu_idx[v]] += vav_unrestricted(s, v);
    for (int v = 0; v < nv; v++) {
        int a = s->vav_ahu_idx[v];
        double d = demand[a];
        if (d > 0.0) {
            double r = s->ahu_const[a * 9 + 1] / d;
            s->vav_ratio[v] = r < 1.0 ? r : 1.0;
        } else {
            s->vav_ratio[v] = 1.0;
        }
    }

    for (int z = 0; z < nz; z++) s->zone_q[z] = 0.0;
    s->total_reheat_w = 0.0;
    for (int col = 0; col < s->k; col++) {
        int v = s->coupled_idx[col];
        if (s->vav_area[v] <= 0.0) { s->air_g[col] = 0.0; s->air_T[col] = 0.0; continue; }
        int a = s->vav_ahu_idx[v];
        double sat = s->ahu_satv[a];
        s->air_T[col] = sat;
        double rw = reheat_energy(s, v, sat, s->boiler_current);
        s->total_reheat_w += rw;
        s->coil[col] = sat;
        double fd = vav_flow_demand(s, v);
        if (s->vav_damper[v] > 0.0 && s->vav_const[v * 9] > 0.0) {
            s->air_g[col] = fd * RHO_AIR * CP_AIR;
            int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
            for (int t = z0; t < z1; t++) {
                int z = s->vav_zone_idx[t];
                s->zone_q[z] += rw * s->zone_area[z] / s->vav_area[v];
            }
        } else {
            s->air_g[col] = 0.0;
        }
        if (fd > 0.0) {
            s->ahu_flow[a] += fd;
            double mf = s->ahu_const[a * 9 + 1];
            if (s->ahu_flow[a] > mf) s->ahu_flow[a] = mf;
            s->ahu_coolreq[a] += 1;
        }
    }

    if (s->internal_w_m2 > 0.0) {
        double frac = comfort ? 1.0 : s->standby_frac;
        double wpc = s->internal_w_m2 * s->cv_area_m2 * frac;
        for (int z = 0; z < nz; z++) {
            int rm = s->phys_room_idx[z];
            s->zone_q[z] += wpc * (double)(s->room_off[rm + 1] - s->room_off[rm]);
        }
    }
    if (s->occupant_w > 0.0)
        for (int z = 0; z < nz; z++) s->zone_q[z] += occV[z] * s->occupant_w;

    for (int z = 0; z < nz; z++) {
        int rm = s->phys_room_idx[z];
        int a = s->room_off[rm], e = s->room_off[rm + 1];
        double q = s->zone_q[z] / (double)(e - a);
        for (int t = a; t < e; t++)
            s->input_q[s->room_cells[2 * t] * s->nc + s->room_cells[2 * t + 1]] = q;
    }

}

void sb_sim_step_post(SbSim* s) {
    int nz = s->n_zones;

    if (s->k > 0) {
        for (int c = 0; c < s->k; c++) s->sensed_mean[c] = 0.0;
        for (int p = 0; p < s->n_cvs; p++) {
            double tp = s->fdm->temp[p];
            if (tp == 0.0) continue;
            for (int q = s->ap_p[p]; q < s->ap_p[p + 1]; q++)
                s->sensed_mean[s->ap_i[q]] += s->ap_x[q] * tp;
        }
        for (int col = 0; col < s->k; col++) {
            int v = s->coupled_idx[col];
            if (s->vav_area[v] <= 0.0) continue;
            double q = s->air_g[col] * (s->air_T[col] - s->sensed_mean[col]);
            int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
            for (int t = z0; t < z1; t++) {
                int z = s->vav_zone_idx[t];
                s->zone_q[z] += q * s->zone_area[z] / s->vav_area[v];
            }
        }
        for (int z = 0; z < nz; z++) {
            int rm = s->phys_room_idx[z];
            int a = s->room_off[rm], e = s->room_off[rm + 1];
            double q = s->zone_q[z] / (double)(e - a);
            for (int t = a; t < e; t++)
                s->input_q[s->room_cells[2 * t] * s->nc + s->room_cells[2 * t + 1]] = q;
        }
    }

    sb_conv_apply(s->conv, s->fdm->temp);

    double wcap_rate = CP_W * RHO_W * total_flow_rate(s);
    if (wcap_rate > 0.0 && s->k > 0) {
        double supply_temp = s->boiler_current;
        double rt = supply_temp - s->total_reheat_w / wcap_rate;
        double at = s->coil[0];
        for (int c = 1; c < s->k; c++) {
            if (s->total_reheat_w >= 0.0) { if (s->coil[c] < at) at = s->coil[c]; }
            else { if (s->coil[c] > at) at = s->coil[c]; }
        }
        double lo = supply_temp < at ? supply_temp : at;
        double hi = supply_temp > at ? supply_temp : at;
        if (rt < lo) rt = lo;
        if (rt > hi) rt = hi;
        s->boiler_return = rt;
    }

    double begin = s->boiler_current;
    double sp = s->boiler_setpoint;
    double dt_min = s->dt_sec / 60.0;
    if (sp > begin) {
        double v = begin + s->boiler_c[2] * dt_min;
        s->boiler_current = v < sp ? v : sp;
    } else if (sp < begin) {
        double v = begin - s->boiler_c[3] * dt_min;
        s->boiler_current = v > sp ? v : sp;
    } else {
        s->boiler_current = sp;
    }
    s->boiler_step_change = s->boiler_current - begin;

    s->step_index += 1;}

/* One step = pre (sensing + HVAC + air-side + applied power) -> FDM ->
 * post (sensed means + convection + boiler ramp). Split so a group of envs
 * sharing a scenario can run the FDM once for the whole group. */
void sb_sim_step(SbSim* s, const double* occV) {
    sb_sim_step_pre(s, occV);
    sb_fdm_step(s->fdm, s->input_q, s->amb[s->step_index], s->h, s->air_g,
                s->air_T);
    sb_sim_step_post(s);
}

void sb_sim_step_batch(SbSim** s, int G, const double* const* occV,
                       double* rhs, double* sol, double* work) {
    SbFdm** f = calloc(G, sizeof(SbFdm*));
    const double** iq = calloc(G, sizeof(double*));
    const double** ag = calloc(G, sizeof(double*));
    const double** aT = calloc(G, sizeof(double*));
    double* ambb = calloc(G, sizeof(double));
    double* hb = calloc(G, sizeof(double));
    if (!f || !iq || !ag || !aT || !ambb || !hb) abort();
    for (int g = 0; g < G; g++) {
        sb_sim_step_pre(s[g], occV[g]);
        f[g] = s[g]->fdm;
        iq[g] = s[g]->input_q;
        ag[g] = s[g]->air_g;
        aT[g] = s[g]->air_T;
        ambb[g] = s[g]->amb[s[g]->step_index];
        hb[g] = s[g]->h;
    }
    sb_fdm_step_batch(f, G, iq, ambb, hb, ag, aT, rhs, sol, work);
    for (int g = 0; g < G; g++) sb_sim_step_post(s[g]);
    free(f); free(iq); free(ag); free(aT); free(ambb); free(hb);
}


void sb_decode_native(int n, const uint8_t* is_disc,
                      const double* mn, const double* mx,
                      const double* mnr, const double* mxr,
                      const double* cont, const double* disc, double* native) {
    int ci = 0, di = 0;
    for (int i = 0; i < n; i++) {
        if (is_disc[i]) {
            double dv = disc[di++];
            double merged = (dv == 1.0) ? 1.0 : -1.0;
            native[i] = (merged - mnr[i]) / (mxr[i] - mnr[i]) * (mx[i] - mn[i]) + mn[i];
        } else {
            /* numpy computes the setpoint map in float32 for a float32 action */
            float m = (float)cont[ci++];
            float rng_in = (float)(mxr[i] - mnr[i]);
            float rng_out = (float)(mx[i] - mn[i]);
            float r = (m - (float)mnr[i]) / rng_in * rng_out + (float)mn[i];
            native[i] = (double)r;
        }
    }
}
