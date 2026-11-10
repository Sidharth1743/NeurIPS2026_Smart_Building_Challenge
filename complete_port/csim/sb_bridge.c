/* Python <-> C-core bridge (ctypes). Drives the verified C simulator with the
 * REAL Python baseline policies, so a score sweep is exact by construction and
 * needs no policy port. Step order matches the PufferLib env exactly:
 *   reset : sim_reset, occup_prime, obs(0), occup_advance(0)
 *   step  : decode, apply, sim_step, obs(k+1), occup_advance(k+1), reward
 * Build: cc -O2 -fPIC -shared sb_bridge.c $(CORE) -o libsb_bridge.so -lm */
#include <stdlib.h>
#include <string.h>

#include "sb_baked.h"
#include "sb_convect.h"
#include "sb_fdm.h"
#include "sb_obs.h"
#include "sb_occup.h"
#include "sb_reward.h"
#include "sb_sim.h"

typedef struct {
    int n_cont, n_disc, n_act, n_obs, n_steps, n_ahu, n_vav, n_zones;
} SbEnvInfo;

typedef struct {
    SbBlob blob;
    SbFdm fdm;
    SbConv conv;
    SbOccup occ;
    SbSim sim;
    double* native;
    double* contd;
    double* discd;
    double* parts;
    const uint8_t* isd;
    const double *mn, *mx, *mnr, *mxr;
    int step;
} SbEnv;

void* sb_load(const char* path) {
    SbEnv* e = (SbEnv*)calloc(1, sizeof(SbEnv));
    if (!e)
        return NULL;
    if (sb_open(&e->blob, path) != 0) {
        free(e);
        return NULL;
    }
    if (sb_fdm_init(&e->fdm, &e->blob) || sb_conv_init(&e->conv, &e->blob) ||
        sb_occup_init(&e->occ, &e->blob) ||
        sb_sim_init(&e->sim, &e->blob, &e->fdm, &e->conv, &e->occ))
        return NULL;
    e->isd = sb_u8(&e->blob, "act_is_discrete");
    e->mn = sb_f64(&e->blob, "act_min_native");
    e->mx = sb_f64(&e->blob, "act_max_native");
    e->mnr = sb_f64(&e->blob, "act_min_norm");
    e->mxr = sb_f64(&e->blob, "act_max_norm");
    int na = e->sim.n_action;
    e->native = (double*)calloc((size_t)na, sizeof(double));
    e->contd = (double*)calloc((size_t)na, sizeof(double));
    e->discd = (double*)calloc((size_t)na, sizeof(double));
    e->parts = (double*)calloc(12, sizeof(double));
    if (!e->native || !e->contd || !e->discd || !e->parts)
        return NULL;
    return e;
}

void sb_free(void* h) {
    SbEnv* e = (SbEnv*)h;
    if (!e)
        return;
    free(e->native);
    free(e->contd);
    free(e->discd);
    free(e->parts);
    sb_close(&e->blob);
    free(e);
}

void sb_info(void* h, int* out, int nout) {
    SbEnv* e = (SbEnv*)h;
    const SbSection* sec = sb_find(&e->blob, "occ_hour");
    int vals[8];
    vals[0] = e->sim.n_action; /* n_act (cont + disc) */
    vals[1] = e->sim.n_dev + e->sim.n_aux;
    vals[2] = sec ? (int)sec->count : 0;
    vals[3] = e->sim.n_ahu;
    vals[4] = e->sim.n_vav;
    vals[5] = e->sim.n_zones;
    vals[6] = e->sim.n_cvs;
    vals[7] = e->sim.k;
    for (int i = 0; i < nout && i < 8; i++)
        out[i] = vals[i];
}

/* Counts of continuous / discrete dims, in the native action order. */
void sb_action_dims(void* h, int* n_cont, int* n_disc) {
    SbEnv* e = (SbEnv*)h;
    int c = 0, d = 0;
    for (int i = 0; i < e->sim.n_action; i++)
        if (e->isd[i])
            d++;
        else
            c++;
    if (n_cont)
        *n_cont = c;
    if (n_disc)
        *n_disc = d;
}

void sb_reset(void* h, float* obs0) {
    SbEnv* e = (SbEnv*)h;
    sb_sim_reset(&e->sim, &e->blob);
    sb_occup_prime(&e->occ);
    e->step = 0;
    if (obs0)
        sb_obs(&e->sim, e->occ.V, 0, obs0);
    sb_occup_advance(&e->occ, 0);
}

void sb_step(void* h, const float* cont, const int* disc, float* obs_out,
             double* rew_out) {
    SbEnv* e = (SbEnv*)h;
    /* sb_decode_native expects the continuous and discrete dims PACKED (that is
     * what the env's hybrid action space hands over), not at their native
     * positions -- only the routing of each dim comes from act_is_discrete. */
    int na = e->sim.n_action, ci = 0, di = 0;
    for (int i = 0; i < na; i++) {
        if (e->isd[i])
            e->discd[di++] = (double)disc[di];
        else
            e->contd[ci++] = (double)cont[ci];
    }
    sb_decode_native(na, e->isd, e->mn, e->mx, e->mnr, e->mxr, e->contd,
                     e->discd, e->native);
    sb_sim_apply_native(&e->sim, e->native);
    sb_sim_step(&e->sim, e->occ.V);
    e->step += 1;
    if (obs_out)
        sb_obs(&e->sim, e->occ.V, e->step, obs_out);
    sb_occup_advance(&e->occ, e->step);
    if (rew_out)
        *rew_out = sb_reward(&e->sim, e->occ.V, e->step, e->parts);
}
