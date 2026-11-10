/* Debug: compare the FDM inputs of the two step paths for one env, then the
 * solver output, to localize a batched-vs-sequential divergence.
 * usage: dbg_batch <blob.sbb> <step_index> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sb_baked.h"
#include "sb_convect.h"
#include "sb_fdm.h"
#include "sb_occup.h"
#include "sb_reward.h"
#include "sb_sim.h"

static SbBlob B;
static SbFdm fa, fb;
static SbConv ca, cb;
static SbOccup oa, ob;
static SbSim sa, sb;

static double hval(unsigned long long k) {
    k = k * 6364136223846793005ULL + 1442695040888963407ULL;
    k ^= k >> 33;
    return (double)((k >> 11) & ((1ULL << 53) - 1)) / (double)(1ULL << 53);
}

static double dmax(const double* a, const double* b, int n) {
    double w = 0.0;
    for (int i = 0; i < n; i++)
        w = fmax(w, fabs(a[i] - b[i]));
    return w;
}

int main(int argc, char** argv) {
    const char* blob = argc > 1 ? argv[1] : "baked/building_train_1__fairbanks_1.sbb";
    int K = argc > 2 ? atoi(argv[2]) : 0;
    if (sb_open(&B, blob) != 0) return 1;
    if (sb_fdm_init(&fa, &B) || sb_conv_init(&ca, &B) || sb_occup_init(&oa, &B) ||
        sb_sim_init(&sa, &B, &fa, &ca, &oa)) return 1;
    if (sb_fdm_init(&fb, &B) || sb_conv_init(&cb, &B) || sb_occup_init(&ob, &B) ||
        sb_sim_init(&sb, &B, &fb, &cb, &ob)) return 1;
    int na = sa.n_action, ncv = sa.n_cvs, n = sa.fdm->n_solved, k = sa.k;
    const uint8_t* isd = sb_u8(&B, "act_is_discrete");
    const double* mn = sb_f64(&B, "act_min_native");
    const double* mx = sb_f64(&B, "act_max_native");
    const double* mnr = sb_f64(&B, "act_min_norm");
    const double* mxr = sb_f64(&B, "act_max_norm");
    double* cont = calloc(na, sizeof(double));
    double* disc = calloc(na, sizeof(double));
    double* nat = calloc(na, sizeof(double));

    for (int s = 0; s <= K; s++) {
        int nc = 0, nd = 0;
        for (int i = 0; i < na; i++) {
            if (isd[i]) disc[nd++] = hval(((unsigned long long)s << 20) ^ i) > 0.5;
            else cont[nc++] = 2.0 * hval(((unsigned long long)s << 20) ^ i) - 1.0;
        }
        sb_decode_native(na, isd, mn, mx, mnr, mxr, cont, disc, nat);
        /* A: pre -> fdm -> post ; B: pre -> batched fdm -> post */
        sb_sim_apply_native(&sa, nat);
        sb_sim_apply_native(&sb, nat);
        sb_sim_step_pre(&sa, oa.V);
        sb_sim_step_pre(&sb, ob.V);
        if (s == K) {
            printf("step %d inputs:\n", s);
            printf("  ambient : %.17g vs %.17g\n", sa.amb[sa.step_index], sb.amb[sb.step_index]);
            printf("  h       : %.17g vs %.17g\n", sa.h, sb.h);
            printf("  input_q : max|d| %.3e\n", dmax(sa.input_q, sb.input_q, ncv));
            printf("  air_g   : max|d| %.3e  (nonzero %d)\n", dmax(sa.air_g, sb.air_g, k), k);
            printf("  air_T   : max|d| %.3e\n", dmax(sa.air_T, sb.air_T, k));
            printf("  temp    : max|d| %.3e\n", dmax(sa.fdm->temp, sb.fdm->temp, ncv));
            printf("  mass    : max|d| %.3e\n", dmax(sa.fdm->mass_temp, sb.fdm->mass_temp, n));
        }
        sb_fdm_step(&fa, sa.input_q, sa.amb[sa.step_index], sa.h,
                    sa.air_g, sa.air_T);
        {
            SbFdm* f[1]; const double* iq[1]; const double* ag[1];
            const double* aT[1]; double amb[1], hh[1];
            double* rhs = calloc((size_t)n * 2, sizeof(double));
            double* sol = rhs + n;
            double* work = calloc(n, sizeof(double));
            f[0] = &fb; iq[0] = sb.input_q; ag[0] = sb.air_g; aT[0] = sb.air_T;
            amb[0] = sb.amb[sb.step_index]; hh[0] = sb.h;
            sb_fdm_step_batch(f, 1, iq, amb, hh, ag, aT, rhs, sol, work);
            free(rhs); free(work);
        }
        if (s == K) {
            printf("  y       : max|d| %.3e\n", dmax(fa.y, fb.y, n));
            printf("  temp'   : max|d| %.3e\n", dmax(fa.temp, fb.temp, ncv));
        }
        sb_sim_step_post(&sa);
        sb_sim_step_post(&sb);
    }
    printf("final temp: max|d| %.3e\n", dmax(fa.temp, fb.temp, ncv));
    return 0;
}
