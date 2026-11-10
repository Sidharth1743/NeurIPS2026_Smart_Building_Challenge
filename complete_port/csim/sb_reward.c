#include "sb_reward.h"
#include <math.h>
#define SB_PI 3.14159265358979323846
#define RHO_AIR 1.2
#define CP_AIR 1006.0
#define RHO_W 1000.0
#define CP_W 4180.0
#define GRAV 9.8
#define ON 1

/* boiler natural-gas thermal energy rate (W), Boiler.compute_thermal_energy_rate */
static double boiler_thermal(SbSim* s, double rwt, double outside, double tf) {
    const double* c = s->boiler_c;
    double supply = (s->boiler_setpoint > rwt) ? s->boiler_setpoint : rwt;
    double flow = CP_W * RHO_W * tf * (supply - rwt);
    double diss = 0.0;
    if (supply >= outside) {
        double num = c[5] * 2.0 * SB_PI * (supply - outside);
        double r2 = c[6] + c[9];
        double cond = log(r2 / c[6]) / c[8];
        double conv = 1.0 / c[4] / r2;
        diss = num / (cond + conv);
    }
    double tank = 0.0;
    if (s->dt_sec > 0.0)
        tank = CP_W * RHO_W * c[7] * s->boiler_step_change / s->dt_sec;
    double tot = flow + diss + tank;
    return tot > 0.0 ? tot : 0.0;
}

double sb_reward(SbSim* s, const double* occV, int kv, double* parts) {
    int nz = s->n_zones;
    double dt = s->dt_sec;
    double inv_hour = 1.0 / 3600.0;

    for (int z = 0; z < nz; z++) {
        int rm = s->phys_room_idx[z];
        int a = s->room_off[rm], e = s->room_off[rm + 1];
        double sum = 0.0;
        for (int t = a; t < e; t++)
            sum += s->fdm->temp[s->room_cells[2 * t] * s->nc + s->room_cells[2 * t + 1]];
        s->avg_temp[z] = sum / (double)(e - a);
    }
    for (int z = 0; z < nz; z++) {
        double fsp = 0.0, fact = 0.0;
        for (int t = s->zone_vav_off[z]; t < s->zone_vav_off[z + 1]; t++) {
            int v = s->zone_vav_idx[t];
            double va = s->vav_area[v];
            if (va <= 0.0) continue;
            double share = s->zone_area[z] / va;
            fsp += s->vav_const[v * 9] * share;
            fact += vav_flow_demand(s, v) * share;
        }
        s->zo_flow_sp[z] = fsp;
        s->zo_flow_act[z] = fact;
    }

    double ambient = s->amb[kv];
    int hour = (int)s->rew_hour[kv];
    int month = (int)s->rew_month[kv];
    int wd = (int)s->rew_workday[kv];

    /* --- AHU blower + cooling electrical --- */
    double blower = 0.0, cooling = 0.0, intake = 0.0, exhaust = 0.0;
    for (int a2 = 0; a2 < s->n_ahu; a2++) s->ahu_wsum[a2] = 0.0, s->ahu_atotal[a2] = 0.0;
    for (int v = 0; v < s->n_vav; v++) {
        int z0 = s->vav_zone_off[v], z1 = s->vav_zone_off[v + 1];
        double area = 0.0, wt = 0.0;
        for (int t = z0; t < z1; t++) {
            int z = s->vav_zone_idx[t];
            area += s->zone_area[z];
            wt += s->avg_temp[z] * s->zone_area[z];
        }
        if (area > 0.0) {
            int a = s->vav_ahu_idx[v];
            s->ahu_wsum[a] += (wt / area) * area;
            s->ahu_atotal[a] += area;
        }
    }
    double tmean = 0.0;
    for (int p = 0; p < s->n_cvs; p++) tmean += s->fdm->temp[p];
    tmean /= (double)s->n_cvs;
    for (int a = 0; a < s->n_ahu; a++) {
        double recirc = (s->ahu_atotal[a] > 0.0)
                        ? s->ahu_wsum[a] / s->ahu_atotal[a] : tmean;
        double st = ahu_static(s, a);
        double fe = s->ahu_const[a * 9 + 5];
        double flow = (s->ahu_run[a] == ON) ? s->ahu_flow[a] : 0.0;
        intake += flow * st / fe;
        exhaust += flow * (1.0 - s->ahu_recirc[a]) * st / fe;
        double mixed = s->ahu_recirc[a] * recirc
                     + (1.0 - s->ahu_recirc[a]) * ambient;
        double sat = (mixed > s->ahu_sat[a] && s->ahu_run[a] == ON)
                     ? s->ahu_sat[a] : mixed;
        cooling += flow * RHO_AIR * CP_AIR * (sat - mixed) / s->ahu_const[a * 9 + 4];
    }

    blower = intake + exhaust;

    /* --- HWS: gas + pump --- */
    double tf = total_flow_rate(s);
    double gas_rate = boiler_thermal(s, s->boiler_return, ambient, tf);
    double pump_e = tf * RHO_W * GRAV * s->pump_head / s->pump_eff;

    /* --- productivity --- */
    double max_pph = s->rw_scalars[0], min_pph = s->rw_scalars[1];
    double ref = s->rw_scalars[2];
    double total_occ = 0.0, cum_prod = 0.0;
    for (int z = 0; z < nz; z++) {
        double occ = occV[z];
        total_occ += occ;
        double zt = s->avg_temp[z];
        double hsp = s->sched_heat_sp[kv], csp = s->sched_cool_sp[kv];
        double dev = zt - csp;
        if (hsp - zt > dev) dev = hsp - zt;
        if (dev < 0.0) dev = 0.0;
        double short_ = (dev <= ref) ? dev / ref
                        : 1.0 + ((dev - ref) / ref) * ((dev - ref) / ref);
        cum_prod += max_pph * (1.0 - short_) * occ * dt * inv_hour;
    }
    double max_prod = max_pph * total_occ * dt * inv_hour;
    double min_prod = min_pph * total_occ * dt * inv_hour;
    double npr;
    if (total_occ > 0.0 && (max_prod - min_prod) != 0.0)
        npr = (cum_prod - min_prod) / (max_prod - min_prod) - 1.0;
    else
        npr = 0.0;

    double elec_rate = blower + fabs(cooling) + pump_e;
    double price = (wd ? s->elec_weekday[hour] : s->elec_weekend[hour]);
    double crate = s->elec_carbon[hour];
    double elec_cost = price * fabs(elec_rate) * dt;
    double elec_cost_max = price * s->rw_scalars[3] * dt;
    double elec_carb = crate * fabs(elec_rate) * dt;
    double elec_carb_max = crate * s->rw_scalars[3] * dt;
    double gprice = s->gas_price[month - 1];
    double g = gas_rate > 0.0 ? gas_rate : 0.0;
    double gas_cost = gprice * g * dt;
    double gas_cost_max = gprice * s->rw_scalars[4] * dt;
    double gas_carb = s->gas_carbon * g * dt;
    double gas_carb_max = s->gas_carbon * s->rw_scalars[4] * dt;

    double den_e = elec_cost_max + gas_cost_max;
    double nec = (den_e > 0.0) ? (elec_cost + gas_cost) / den_e : 0.0;
    double den_c = elec_carb_max + gas_carb_max;
    double ncar = (den_c > 0.0) ? (elec_carb + gas_carb) / den_c : 0.0;

    if (parts) { parts[0]=npr; parts[1]=nec; parts[2]=ncar;
                 parts[3]=elec_cost; parts[4]=gas_cost;
                 parts[5]=blower; parts[6]=cooling; parts[7]=pump_e;
                 parts[8]=price; parts[9]=dt; parts[10]=(double)hour; parts[11]=(double)wd; }
    double pw = s->rw_scalars[5], ew = s->rw_scalars[6], cw = s->rw_scalars[7];
    double raw = npr * pw - nec * ew - ncar * cw;
    return raw / (pw + ew + cw);
}
