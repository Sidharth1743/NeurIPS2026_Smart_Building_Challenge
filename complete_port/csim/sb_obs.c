#include "sb_obs.h"
#include <math.h>
#define RHO_W_ 1000.0
#define GRAV_ 9.8
#define PA_BAR_ 100000.0

static double pump_dp(SbSim* s) {
    return (s->pump_run == 1) ? GRAV_ * RHO_W_ * s->pump_head / PA_BAR_ : 0.0;
}

static double obs_value(SbSim* s, int kind, int idx, int id, int j) {
    if (kind == 0) {
        int a = idx;
        const double* c = &s->ahu_const[a * 9];
        switch (id) {
            case 0: return (double)s->ahu_coolreq[a];
            case 1: return ahu_static(s, a);
            case 2: return (s->ahu_run[a] == 1) ? s->ahu_flow[a] : 0.0;
            case 3: return s->ahu_sat[a];
            case 4: return (double)s->ahu_run[a];
            case 5: return s->ahu_flow[a] / c[1];
            case 6: return (1.0 - s->ahu_recirc[a]) * s->ahu_flow[a];
            case 7: return (double)s->ahu_run[a];
            case 8: return s->ahu_recirc[a];
            default: return s->amb[j];
        }
    } else if (kind == 1) {
        int v = idx;
        switch (id) {
            case 0: return s->vav_reheat[v];
            case 1: return s->vav_damper[v];
            case 2: return s->vav_const[v * 9];
            case 3: return s->vav_coolsp[v];
            case 4: return s->vav_heatsp[v];
            default: return s->sensing[v];
        }
    }
    switch (id) {
        case 0: return pump_dp(s);
        case 1: return s->boiler_current;
        case 2: return pump_dp(s);
        case 3: return (double)s->hws_run;
        case 4: return (double)s->hws_heating_requests;
        case 5: return (double)s->hws_run;
        default: return s->boiler_setpoint;
    }
}

void sb_obs(SbSim* s, const double* occV, int j, float* out) {
    for (int i = 0; i < s->n_dev; i++) {
        double v = obs_value(s, s->f_dev_kind[i], s->f_dev_idx[i], s->f_obs_id[i], j);
        double var = s->f_norm_var[i];
        /* The oracle packs every measurement through a protobuf float field,
         * so the raw value is float32-quantized before it is normalized. */
        float v32 = (float)v;
        out[i] = (float)((var > 0.0) ? ((double)v32 - s->f_norm_mean[i]) / sqrt(var) : 0.0);
    }
    int na = s->n_aux;
    const double* aux = &s->aux_time[(size_t)j * (na - 1)];
    for (int k = 0; k < na - 1; k++) out[s->n_dev + k] = (float)aux[k];
    double occ = 0.0;
    for (int z = 0; z < s->n_zones; z++) occ += occV[z];
    out[s->n_dev + na - 1] = (float)(int)occ;
}
