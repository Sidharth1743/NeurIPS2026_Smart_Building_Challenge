// CPU regression for ocean/smart_buildings/smart_buildings.h -- no CUDA, no
// trainer. Two modes:
//
//   test_env_cpu <blob> <trace>   env parity: drive the env from the dumped
//                                 oracle action stream and compare the env's
//                                 observation and reward to the oracle's.
//   test_env_cpu <blob>           smoke: random actions, invariants (padding
//                                 zeroed, mask shape, rewards finite), episode
//                                 terminal at max_steps and reset.
//
// The trace format is the one csim/test/dump_step.py writes (see
// csim/test/test_step.c); only cont/disc/rew/obs are used here.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ini.h"
#include "smart_buildings.h"

static unsigned int lcg = 12345u;
static float frand(void) {
    lcg = lcg * 1664525u + 1013904223u;
    return (float)((lcg >> 8) * (1.0 / 16777216.0));
}

static void wire(Env* env, float* obs, float* act, float* rew, float* term,
                 unsigned char* mask) {
    env->agents[0].observations = obs;
    env->agents[0].actions = act;
    env->agents[0].rewards = rew;
    env->agents[0].terminals = term;
    env->agents[0].action_mask = mask;
}

static int check_padding(const float* obs, int n_field, const char* what) {
    for (int i = n_field; i < SB_OBS; i++) {
        if (obs[i] != 0.0f) {
            printf("%s: obs padding not zero at %d (%g)\n", what, i, obs[i]);
            return 1;
        }
    }
    return 0;
}

static int check_mask(const unsigned char* mask, int n_cont, int n_disc) {
    int bad = 0;
    for (int h = 0; h < SB_CONT; h++)
        if (mask[h] != (h < n_cont ? 1 : 0))
            bad = 1;
    for (int h = 0; h < SB_DISC; h++) {
        unsigned char want = (h < n_disc ? 1 : 0);
        if (mask[SB_CONT + 2 * h] != want || mask[SB_CONT + 2 * h + 1] != want)
            bad = 1;
    }
    return bad;
}

/* ---- trace parity ------------------------------------------------------- */

static int run_trace(const char* blob, const char* trace) {
    static Env env;
    Dict kw = {0};
    dict_set_str(&kw, "blob", blob);
    puf_init(&env, &kw);
    static float obs[SB_OBS], act[NUM_ATNS], rew[1], term[1];
    static unsigned char mask[SB_MASK_WIDTH];
    wire(&env, obs, act, rew, term, mask);
    env.rng = 5;
    puf_reset(&env);

    int nc = env.n_cont, nd = env.n_disc, na = env.n_act;
    FILE* f = fopen(trace, "rb");
    if (!f) {
        printf("cannot open trace %s\n", trace);
        return 1;
    }
    int hdr[7];
    if (fread(hdr, sizeof(int), 7, f) != 7)
        return 1;
    int n_cvs = hdr[0], nz = hdr[1], tnc = hdr[2], tnd = hdr[3];
    int nsteps = hdr[4], kk = hdr[5], nfld = hdr[6];
    if (tnc != nc || tnd != nd || nfld != env.obs_len) {
        printf("layout mismatch: trace %d/%d/%d env %d/%d/%d\n", tnc, tnd, nfld,
               nc, nd, env.obs_len);
        return 1;
    }
    printf("trace parity: n_act=%d (cont=%d disc=%d) n_obs=%d steps=%d\n", na, nc,
           nd, nfld, nsteps);
    if (check_mask(mask, nc, nd))
        printf("  mask BAD\n");
    if (check_padding(obs, nfld, "reset"))
        return 1;

    double* cont = malloc(8 * (size_t)tnc);
    double* disc = malloc(8 * (size_t)tnd);
    double* skip_nat = malloc(8 * (size_t)(na ? na : 1));
    double* skip_z = malloc(8 * (size_t)nz);
    double* skip_g = malloc(8 * (size_t)kk);
    double* skip_c = malloc(8 * (size_t)n_cvs);
    double* skip_1 = malloc(8);
    double* skip_rp = malloc(96);
    double* tl_obs0 = malloc(8 * (size_t)nfld);
    double* tl_obs = malloc(8 * (size_t)nfld);
    double* tl_rew = malloc(8);

    if (fread(tl_obs0, 8, (size_t)nfld, f) != (size_t)nfld)
        return 1;
    double wObs = 0.0, wRew = 0.0;
    int iObs = -1;
    for (int i = 0; i < nfld; i++) {
        double d = fabs((double)obs[i] - tl_obs0[i]);
        if (d > wObs) {
            wObs = d;
            iObs = i;
        }
    }
    for (int s = 0; s < nsteps; s++) {
        if (fread(cont, 8, (size_t)tnc, f) != (size_t)tnc)
            return 1;
        if (fread(disc, 8, (size_t)tnd, f) != (size_t)tnd)
            return 1;
        if (fread(skip_nat, 8, (size_t)na, f) != (size_t)na)
            return 1;
        if (fread(skip_z, 8, (size_t)nz, f) != (size_t)nz)
            return 1;
        if (fread(skip_g, 8, (size_t)kk, f) != (size_t)kk)
            return 1;
        if (fread(skip_g, 8, (size_t)kk, f) != (size_t)kk)
            return 1;
        if (fread(skip_1, 8, 1, f) != 1)
            return 1;
        if (fread(skip_c, 8, (size_t)n_cvs, f) != (size_t)n_cvs)
            return 1;
        if (fread(tl_rew, 8, 1, f) != 1)
            return 1;
        if (fread(skip_rp, 8, 12, f) != 12)
            return 1;
        if (fread(tl_obs, 8, (size_t)nfld, f) != (size_t)nfld)
            return 1;

        for (int i = 0; i < nc; i++)
            act[i] = (float)cont[i];
        for (int j = 0; j < nd; j++)
            act[SB_CONT + j] = (float)disc[j];
        puf_step(&env);
        if (check_padding(obs, nfld, "step"))
            return 1;
        if (term[0] != 0.0f) {
            printf("  unexpected terminal at step %d\n", s);
            return 1;
        }
        double dr = fabs((double)rew[0] - tl_rew[0]);
        if (dr > wRew)
            wRew = dr;
        for (int i = 0; i < nfld; i++) {
            double d = fabs((double)obs[i] - tl_obs[i]);
            if (d > wObs) {
                wObs = d;
                iObs = i;
            }
        }
    }
    printf("  obs err=%.3e (idx %d)  reward err=%.3e\n", wObs, iObs, wRew);
    puf_close(&env);
    int bad = (wObs != 0.0) || (wRew > 1e-5);
    printf(bad ? "ENV TRACE PARITY: FAIL\n" : "ENV TRACE PARITY: OK\n");
    return bad;
}

/* ---- smoke ------------------------------------------------------------- */

static int run_smoke(const char* blob, int steps) {
    static Env env;
    Dict kw = {0};
    dict_set_str(&kw, "blob", blob);
    dict_set(&kw, "max_steps", steps);
    puf_init(&env, &kw);
    static float obs[SB_OBS], act[NUM_ATNS], rew[1], term[1];
    static unsigned char mask[SB_MASK_WIDTH];
    wire(&env, obs, act, rew, term, mask);
    env.rng = 100;
    puf_reset(&env);
    if (check_mask(mask, env.n_cont, env.n_disc)) {
        printf("SMOKE: mask BAD\n");
        return 1;
    }
    int bad = 0, terms = 0, episodes = 0;
    double rsum = 0.0;
    /* 2.5 short episodes: exercises the terminal + reset path. */
    int total = steps * 2 + steps / 2;
    for (int i = 0; i < total; i++) {
        for (int h = 0; h < env.n_cont; h++)
            act[h] = frand() * 2.0f - 1.0f;
        for (int h = 0; h < env.n_disc; h++)
            act[SB_CONT + h] = frand() < 0.5f ? 0.0f : 1.0f;
        puf_step(&env);
        if (!(rew[0] > -1e9f && rew[0] < 1e9f)) {
            printf("SMOKE: reward not finite at %d\n", i);
            bad = 1;
            break;
        }
        if (!(term[0] == 0.0f || term[0] == 1.0f)) {
            printf("SMOKE: bad terminal at %d\n", i);
            bad = 1;
            break;
        }
        for (int k = 0; k < env.obs_len; k++) {
            if (!(obs[k] > -1e9f && obs[k] < 1e9f)) {
                printf("SMOKE: obs NaN/inf at step %d idx %d\n", i, k);
                bad = 1;
                break;
            }
        }
        if (check_padding(obs, env.obs_len, "smoke")) {
            bad = 1;
            break;
        }
        rsum += rew[0];
        if (term[0] == 1.0f) {
            terms++;
            episodes++;
        }
    }
    printf("smoke: steps=%d terms=%d episodes=%d rsum=%+.3f perf_mean=%.4f %s\n",
           total, terms, episodes, rsum,
           env.log.n > 0 ? env.log.perf / env.log.n : 0.0, bad ? "BAD" : "ok");
    puf_close(&env);
    return bad;
}

int main(int argc, char** argv) {
    const char* blob = getenv("SB_TEST_BLOB");
    if (argc > 1)
        blob = argv[1];
    if (blob == NULL) {
        printf("usage: test_env_cpu <blob.sbb> [trace]\n");
        return 2;
    }
    int bad = 0;
    if (argc > 2)
        bad |= run_trace(blob, argv[2]);
    else
        bad |= run_smoke(blob, 40);
    printf(bad ? "CPU ENV TEST: FAIL\n" : "CPU ENV TEST: OK\n");
    return bad;
}
