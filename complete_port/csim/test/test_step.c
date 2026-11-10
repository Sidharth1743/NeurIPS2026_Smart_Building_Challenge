/* M2/M3 gate: run the full step from actions alone; compare air-side, temp,
 * reward and observation to the oracle. usage: test_step <blob> <trace> */
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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <blob> <trace>\n", argv[0]); return 2; }
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 1;
    static SbFdm fdm; static SbConv conv; static SbOccup occ; static SbSim sim;
    if (sb_fdm_init(&fdm, &b) || sb_conv_init(&conv, &b) || sb_occup_init(&occ, &b)
        || sb_sim_init(&sim, &b, &fdm, &conv, &occ)) { printf("init FAIL\n"); return 1; }
    sb_sim_reset(&sim, &b);

    const uint8_t* isd = sb_u8(&b, "act_is_discrete");
    const double* mn = sb_f64(&b, "act_min_native");
    const double* mx = sb_f64(&b, "act_max_native");
    const double* mnr = sb_f64(&b, "act_min_norm");
    const double* mxr = sb_f64(&b, "act_max_norm");
    int na = sim.n_action, kk = sim.k;

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "open %s\n", argv[2]); return 1; }
    int hdr[7];
    if (fread(hdr, sizeof(int), 7, f) != 7) return 1;
    int n_cvs = hdr[0], nz = hdr[1], nc = hdr[2], nd = hdr[3],
        nsteps = hdr[4], ktr = hdr[5], nfld = hdr[6];
    printf("trace n_cvs=%d nz=%d nc=%d nd=%d k=%d steps=%d nfld=%d\n",
           n_cvs, nz, nc, nd, ktr, nsteps, nfld);
    if (n_cvs != sim.n_cvs || nz != sim.n_zones || ktr != kk || nfld != sim.n_dev + sim.n_aux) {
        printf("shape mismatch\n"); return 1;
    }

    double* cont = malloc(8 * nc); double* disc = malloc(8 * nd);
    double* nat = malloc(8 * na); double* occV = malloc(8 * nz);
    double* g = malloc(8 * kk); double* T = malloc(8 * kk);
    double* tnew = malloc(8 * n_cvs); double* dec = malloc(8 * na);
    double* swt = malloc(8); double* rew = malloc(8); double* rp = malloc(96);
    double* rpC = malloc(96);
    double* obs0 = malloc(8 * nfld); double* obs = malloc(8 * nfld);
    float* obsC = malloc(4 * nfld);

    if (fread(obs0, 8, nfld, f) != (size_t)nfld) return 1;
    double wObs = 0.0;
    int wObsIdx = -1;
    sb_occup_prime(&occ);
    sb_obs(&sim, occ.V, 0, obsC);
    for (int i = 0; i < nfld; i++) {
        double d = fabs((double)obsC[i] - obs0[i]);
        if (d > wObs) { wObs = d; wObsIdx = i; }
    }
    if (wObsIdx >= 0)
        printf("  obs@reset err=%.3e idx=%d py=%.6f c=%.6f\n", wObs, wObsIdx,
               obs0[wObsIdx], (double)obsC[wObsIdx]);
    else
        printf("  obs@reset err=%.3e (exact)\n", wObs);
    sb_occup_advance(&occ, 0);

    double wT = 0, wDec = 0, wOcc = 0, wG = 0, wTT = 0, wSW = 0, wRw = 0;
    for (int s = 0; s < nsteps; s++) {
        if (fread(cont, 8, nc, f) != (size_t)nc) return 1;
        if (fread(disc, 8, nd, f) != (size_t)nd) return 1;
        if (fread(nat, 8, na, f) != (size_t)na) return 1;
        if (fread(occV, 8, nz, f) != (size_t)nz) return 1;
        if (fread(g, 8, kk, f) != (size_t)kk) return 1;
        if (fread(T, 8, kk, f) != (size_t)kk) return 1;
        if (fread(swt, 8, 1, f) != 1) return 1;
        if (fread(tnew, 8, n_cvs, f) != (size_t)n_cvs) return 1;
        if (fread(rew, 8, 1, f) != 1) return 1;
        if (fread(rp, 8, 12, f) != 12) return 1;
        if (fread(obs, 8, nfld, f) != (size_t)nfld) return 1;

        sb_decode_native(na, isd, mn, mx, mnr, mxr, cont, disc, dec);
        for (int i = 0; i < na; i++) wDec = fmax(wDec, fabs(dec[i] - nat[i]));
        for (int z = 0; z < nz; z++) wOcc = fmax(wOcc, fabs(occ.V[z] - occV[z]));

        sb_sim_apply_native(&sim, nat);
        wSW = fmax(wSW, fabs(sim.boiler_current - swt[0]));
        sb_sim_step(&sim, occV);
        for (int c = 0; c < kk; c++) {
            wG = fmax(wG, fabs(sim.air_g[c] - g[c]));
            wTT = fmax(wTT, fabs(sim.air_T[c] - T[c]));
        }
        for (int p = 0; p < n_cvs; p++) wT = fmax(wT, fabs(sim.fdm->temp[p] - tnew[p]));

        sb_obs(&sim, occV, s + 1, obsC);
        for (int i = 0; i < nfld; i++) {
            double d = fabs((double)obsC[i] - obs[i]);
            if (d > wObs) { wObs = d; wObsIdx = i; }
        }
        sb_occup_advance(&occ, s + 1);
        double rr = sb_reward(&sim, occ.V, s + 1, rpC);
        wRw = fmax(wRw, fabs(rr - rew[0]));
        if (s < 3)
            printf("  step %2d: T=%.2e g=%.1e Tair=%.1e dec=%.1e occ=%.1e rew=%.1e obs=%.1e\n",
                   s, wT, wG, wTT, wDec, wOcc, wRw, wObs);
    }
    printf("worst: T=%.3e air_g=%.3e air_T=%.3e swt=%.3e dec=%.3e occ=%.3e rew=%.3e obs=%.3e idx=%d\n",
           wT, wG, wTT, wSW, wDec, wOcc, wRw, wObs, wObsIdx);
    if (wDec > 1e-9 || wOcc > 1e-12 || wG > 1e-9 || wTT > 1e-9 || wSW > 1e-9
        || wT > 1e-6 || wRw > 1e-5 || wObs > 1e-4) { printf("STEP GATE: FAIL\n"); return 1; }
    printf("STEP GATE: OK\n");
    return 0;
}
