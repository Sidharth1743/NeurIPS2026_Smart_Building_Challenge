#include "sb_occup.h"

#include <stdlib.h>
#include <string.h>

int sb_occup_init(SbOccup* o, const SbBlob* b) {
    memset(o, 0, sizeof(*o));
    const double* p = sb_f64(b, "occ_params");
    if (!p) return -1;
    o->n_zones = (int)(sb_find(b, "zone_area")->count);
    o->zone_assignment = (int)p[0];
    o->ea = (int)p[1]; o->la = (int)p[2];
    o->ed = (int)p[3]; o->ld = (int)p[4];
    double step_sec = p[5];
    /* p = 1 / ((end-start)*3600 / step / 2) */
    o->p_arrival = 1.0 / ((o->la - o->ea) * 3600.0 / step_sec / 2.0);
    o->p_departure = 1.0 / ((o->ld - o->ed) * 3600.0 / step_sec / 2.0);
    o->occ_hour = sb_i64(b, "occ_hour");
    o->occ_workday = sb_u8(b, "occ_workday");
    if (!o->occ_hour || !o->occ_workday) return -1;
    o->state = (int*)calloc((size_t)o->n_zones * o->zone_assignment, sizeof(int));
    o->V = (double*)calloc(o->n_zones, sizeof(double));
    if (!o->state || !o->V) return -1;
    o->hour_m1 = (int)sb_i64_1(b, "occ_hour_m1", 0);
    { const uint8_t* wm = sb_u8(b, "occ_workday_m1"); o->workday_m1 = wm ? wm[0] : 0; }
    sb_rng_seed(&o->rng, (uint32_t)sb_i64_1(b, "occ_seed", 17321));
    return 0;
}

void sb_occup_free(SbOccup* o) {
    free(o->state);
    free(o->V);
    memset(o, 0, sizeof(*o));
}

static int peek(SbOccup* o, int* st, int hour, int workday) {
    if (!workday) {
        *st = 0;                       /* AWAY */
    } else if (*st == 0) {
        if (hour >= o->ea && hour <= o->la) {
            if (sb_rng_double(&o->rng) < o->p_arrival) *st = 1;
        }
    } else {
        if (hour < o->ed) {
            /* keep */
        } else if (hour >= o->ld) {
            *st = 0;
        } else if (sb_rng_double(&o->rng) < o->p_departure) {
            *st = 0;
        }
    }
    return *st;
}

static void adv_at(SbOccup* o, int hour, int workday) {
    for (int z = 0; z < o->n_zones; z++) {
        int cnt = 0;
        int* base = &o->state[(size_t)z * o->zone_assignment];
        for (int a = 0; a < o->zone_assignment; a++)
            if (peek(o, &base[a], hour, workday)) cnt++;
        o->V[z] = (double)cnt;   /* interval is exactly one step */
    }
}

void sb_occup_advance(SbOccup* o, int step_index) {
    adv_at(o, (int)o->occ_hour[step_index], (int)o->occ_workday[step_index]);
}

void sb_occup_prime(SbOccup* o) {
    adv_at(o, o->hour_m1, o->workday_m1);
}
