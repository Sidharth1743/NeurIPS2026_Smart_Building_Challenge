/* M5 gate: the whole-sim batched step must equal G sequential sb_sim_step
 * calls bit-for-bit -- temperature field, boiler, air-side, observation and
 * reward. This is the "replicates as it is" check for the batched engine that
 * a PufferLib vector env would drive.
 * usage: test_step_batch <blob.sbb> [G] [steps] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sb_baked.h"
#include "sb_convect.h"
#include "sb_fdm.h"
#include "sb_obs.h"
#include "sb_occup.h"
#include "sb_reward.h"
#include "sb_sim.h"

#define MAXG 16

static SbBlob B;
static SbFdm fdm[MAXG];
static SbConv conv[MAXG];
static SbOccup occ[MAXG];
static SbSim sim[MAXG];
static double* cont[MAXG];
static double* disc[MAXG];
static double* nat[MAXG];
static double* parts[MAXG];
static float* obsbuf;
static double* rhs;
static double* sol;
static double* work;
static int G, STEPS, NC, ND, NA, NZ, NCV, NF, NSOLVED;
static double* ref_temp;
static double* ref_rew;
static float* ref_obs;

static double hval(unsigned long long k) {
    k = k * 6364136223846793005ULL + 1442695040888963407ULL;
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    return (double)((k >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
}

static void gen_actions(int step, int g) {
    for (int i = 0; i < NC; i++)
        cont[g][i] = 2.0 * hval(((unsigned long long)step << 40) ^
                                ((unsigned long long)g << 20) ^ (unsigned)i)
                     - 1.0;
    for (int i = 0; i < ND; i++)
        disc[g][i] = hval(((unsigned long long)step << 40) ^
                          ((unsigned long long)g << 20) ^ (unsigned)i ^ 0x7f)
                         > 0.5;
}

/* The occupant chains and the convection RNG are stateful across a real
 * episode (the oracle's are too -- a building is not re-created per episode),
 * so a *test* that compares two passes must re-seed them, otherwise pass 2
 * starts from pass 1's RNG state and the comparison is meaningless. */
static void reset_env(int g) {
    sb_occup_free(&occ[g]);
    sb_occup_init(&occ[g], &B);
    sb_conv_free(&conv[g]);
    sb_conv_init(&conv[g], &B);
    sb_sim_reset(&sim[g], &B);
    sb_occup_prime(&occ[g]);
    sb_occup_advance(&occ[g], 0);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <blob.sbb> [G] [steps]\n", argv[0]);
        return 2;
    }
    G = argc > 2 ? atoi(argv[2]) : 4;
    STEPS = argc > 3 ? atoi(argv[3]) : 12;
    if (G < 1 || G > MAXG)
        return 2;
    if (sb_open(&B, argv[1]) != 0)
        return 1;
    for (int g = 0; g < G; g++) {
        if (sb_fdm_init(&fdm[g], &B) || sb_conv_init(&conv[g], &B) ||
            sb_occup_init(&occ[g], &B) ||
            sb_sim_init(&sim[g], &B, &fdm[g], &conv[g], &occ[g])) {
            printf("init FAIL\n");
            return 1;
        }
    }
    NA = sim[0].n_action;
    NZ = sim[0].n_zones;
    NCV = sim[0].n_cvs;
    NSOLVED = sim[0].fdm->n_solved;
    NF = sim[0].n_dev + sim[0].n_aux;
    NC = 0; ND = 0;
    const uint8_t* isd = sb_u8(&B, "act_is_discrete");
    for (int i = 0; i < NA; i++) {
        if (isd[i])
            ND++;
        else
            NC++;
    }
    printf("blob %s: G=%d steps=%d n_action=%d (cont %d disc %d) n_field=%d\n",
           argv[1], G, STEPS, NA, NC, ND, NF);

    const double* mn = sb_f64(&B, "act_min_native");
    const double* mx = sb_f64(&B, "act_max_native");
    const double* mnr = sb_f64(&B, "act_min_norm");
    const double* mxr = sb_f64(&B, "act_max_norm");

    for (int g = 0; g < G; g++) {
        cont[g] = calloc(NC, sizeof(double));
        disc[g] = calloc(ND, sizeof(double));
        nat[g] = calloc(NA, sizeof(double));
        parts[g] = calloc(12, sizeof(double));
    }
    obsbuf = calloc(NF, sizeof(float));
    rhs = calloc((size_t)NSOLVED * G * 2, sizeof(double));
    sol = rhs + (size_t)NSOLVED * G;
    work = calloc((size_t)NSOLVED * G, sizeof(double));
    ref_temp = calloc((size_t)NCV * G, sizeof(double));
    ref_rew = calloc(G, sizeof(double));
    ref_obs = calloc((size_t)NF * G, sizeof(float));
    float* cov = calloc((size_t)NF * G, sizeof(float));
    const double** pvp = calloc(G, sizeof(double*));
    SbSim** sp = calloc(G, sizeof(SbSim*));
    int bad = 0;

    /* ---- pass 1: sequential ---- */
    for (int g = 0; g < G; g++) reset_env(g);
    for (int s = 0; s < STEPS; s++) {
        for (int g = 0; g < G; g++) {
            gen_actions(s, g);
            sb_decode_native(NA, isd, mn, mx, mnr, mxr, cont[g], disc[g],
                             nat[g]);
            sb_sim_apply_native(&sim[g], nat[g]);
            sb_sim_step(&sim[g], occ[g].V);
            sb_obs(&sim[g], occ[g].V, s + 1, obsbuf);
            for (int i = 0; i < NF; i++)
                cov[(size_t)g * NF + i] = obsbuf[i];
            sb_occup_advance(&occ[g], s + 1);
            ref_rew[g] += sb_reward(&sim[g], occ[g].V, s + 1, parts[g]);
        }
    }
    for (int g = 0; g < G; g++) {
        memcpy(&ref_temp[(size_t)g * NCV], sim[g].fdm->temp,
               sizeof(double) * (size_t)NCV);
        memcpy(&ref_obs[(size_t)g * NF], &cov[(size_t)g * NF],
               sizeof(float) * (size_t)NF);
    }
    double seq_rew[MAXG];
    memcpy(seq_rew, ref_rew, sizeof(double) * G);

    /* ---- pass 2: batched (one FDM solve per group per step) ---- */
    memset(ref_rew, 0, sizeof(double) * G);
    for (int g = 0; g < G; g++) {
        reset_env(g);
        sp[g] = &sim[g];
    }
    for (int s = 0; s < STEPS; s++) {
        for (int g = 0; g < G; g++) {
            gen_actions(s, g);
            sb_decode_native(NA, isd, mn, mx, mnr, mxr, cont[g], disc[g],
                             nat[g]);
            sb_sim_apply_native(&sim[g], nat[g]);
            pvp[g] = occ[g].V;
        }
        sb_sim_step_batch(sp, G, pvp, rhs, sol, work);
        for (int g = 0; g < G; g++) {
            sb_obs(&sim[g], occ[g].V, s + 1, obsbuf);
            for (int i = 0; i < NF; i++)
                cov[(size_t)g * NF + i] = obsbuf[i];
            sb_occup_advance(&occ[g], s + 1);
            ref_rew[g] += sb_reward(&sim[g], occ[g].V, s + 1, parts[g]);
        }
    }

    long dt = 0, dobs = 0;
    double wt = 0.0, wr = 0.0, wo = 0.0;
    for (size_t i = 0; i < (size_t)NCV * G; i++) {
        double d = fabs(sim[i / NCV].fdm->temp[i % NCV] - ref_temp[i]);
        if (d != 0.0) dt++;
        if (d > wt) wt = d;
    }
    for (size_t i = 0; i < (size_t)NF * G; i++) {
        double d = fabs((double)cov[i] - (double)ref_obs[i]);
        if (d != 0.0) dobs++;
        if (d > wo) wo = d;
    }
    for (int g = 0; g < G; g++)
        wr = fmax(wr, fabs(ref_rew[g] - seq_rew[g]));
    printf("batched vs sequential: temp %ld/%zu differ (max %.3e), obs %ld/%zu "
           "(max %.3e), reward max |d| %.3e\n", dt, (size_t)NCV * G, wt, dobs,
           (size_t)NF * G, wo, wr);
    bad = (dt != 0) || (dobs != 0) || (wr != 0.0);
    printf(bad ? "BATCH STEP: FAIL\n" : "BATCH STEP: OK\n");
    return bad;
}
