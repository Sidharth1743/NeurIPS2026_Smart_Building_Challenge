/* Full-episode gate: replay a whole episode driven by a real baseline policy
 * and compare EVERY step -- decoded action, occupancy, air-side, boiler,
 * observation, reward and probe temperatures -- plus the full field at
 * checkpoint steps. Trace format: pysim/dump_episode.py.
 * usage: test_episode <blob.sbb> <trace.ep> */
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

static int rd(FILE* f, double* p, size_t n) { return fread(p, 8, n, f) == n; }

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <blob.sbb> <trace.ep>\n", argv[0]);
        return 2;
    }
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0)
        return 1;
    static SbFdm fdm;
    static SbConv conv;
    static SbOccup occ;
    static SbSim sim;
    if (sb_fdm_init(&fdm, &b) || sb_conv_init(&conv, &b) ||
        sb_occup_init(&occ, &b) || sb_sim_init(&sim, &b, &fdm, &conv, &occ)) {
        printf("init FAIL\n");
        return 1;
    }
    sb_sim_reset(&sim, &b);
    const uint8_t* isd = sb_u8(&b, "act_is_discrete");
    const double* mn = sb_f64(&b, "act_min_native");
    const double* mx = sb_f64(&b, "act_max_native");
    const double* mnr = sb_f64(&b, "act_min_norm");
    const double* mxr = sb_f64(&b, "act_max_norm");

    FILE* f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "open %s\n", argv[2]); return 1; }
    int hdr[8];
    if (fread(hdr, sizeof(int), 8, f) != 8) return 1;
    int nx = hdr[0], nz = hdr[1], nc = hdr[2], nd = hdr[3];
    int nsteps = hdr[4], kk = hdr[5], nfld = hdr[6], nprobe = hdr[7];
    int na = nc + nd;
    printf("trace: n_cvs=%d nz=%d nc=%d nd=%d k=%d steps=%d nfld=%d nprobe=%d\n",
           nx, nz, nc, nd, kk, nsteps, nfld, nprobe);
    if (nx != sim.n_cvs || nz != sim.n_zones || kk != sim.k ||
        nfld != sim.n_dev + sim.n_aux || na != sim.n_action) {
        printf("shape mismatch\n");
        return 1;
    }
    /* The checkpoint list lives in a trailer but its offset is computable, so
     * the field checkpoints can be located before the replay. */
    long rec = (long)(nc + nd + na + nz + 2 * kk + 1 + 1 + nfld + nprobe) * 8;
    long trailer = 32 + 8L * nfld + 4L * nprobe + (long)nsteps * rec;
    if (fseek(f, trailer, SEEK_SET) != 0) return 1;
    int ncheck = 0;
    if (fread(&ncheck, 4, 1, f) != 1) return 1;
    int* csteps = malloc(4 * (size_t)(ncheck ? ncheck : 1));
    if (fread(csteps, 4, (size_t)ncheck, f) != (size_t)ncheck) return 1;
    long field_off = trailer + 4 + 4L * ncheck;

    double* cont = malloc(8 * nc);
    double* disc = malloc(8 * nd);
    double* nat = malloc(8 * na);
    double* dec = malloc(8 * na);
    double* occV = malloc(8 * nz);
    double* g = malloc(8 * kk);
    double* T = malloc(8 * kk);
    double* swt = malloc(8);
    double* rew = malloc(8);
    double* tl_obs = malloc(8 * nfld);
    double* tl_obs0 = malloc(8 * nfld);
    double* probes = malloc(8 * nprobe);
    double* tl_probes = malloc(8 * nprobe);
    double* parts = calloc(12, sizeof(double));
    double* tnew = malloc(8 * nx);
    double* cf = malloc(8 * (size_t)nx * (size_t)(ncheck ? ncheck : 1));
    int* pidx = malloc(4 * (size_t)(nprobe ? nprobe : 1));
    float* obsC = malloc(4 * nfld);
    float* obsC0 = malloc(4 * nfld);
    if (!cont || !disc || !nat || !dec || !occV || !g || !T || !swt || !rew ||
        !tl_obs || !tl_obs0 || !probes || !tl_probes || !tnew || !cf || !pidx ||
        !obsC || !obsC0)
        return 1;

    if (fseek(f, 32, SEEK_SET) != 0) return 1;
    if (!rd(f, tl_obs0, nfld)) return 1;
    if (fread(pidx, 4, (size_t)nprobe, f) != (size_t)nprobe) return 1;

    double wDec = 0, wOcc = 0, wG = 0, wTT = 0, wSW = 0, wObs = 0, wProbe = 0;
    double wField = 0, wRew = 0, rsumC = 0, rsumT = 0;
    long obsDiff = 0, probeDiff = 0, fieldDiff = 0;
    int ci = 0;

    sb_occup_prime(&occ);
    sb_obs(&sim, occ.V, 0, obsC0);
    for (int i = 0; i < nfld; i++)
        wObs = fmax(wObs, fabs((double)obsC0[i] - tl_obs0[i]));
    sb_occup_advance(&occ, 0);

    for (int s = 0; s < nsteps; s++) {
        if (!rd(f, cont, nc) || !rd(f, disc, nd) || !rd(f, nat, na) ||
            !rd(f, occV, nz) || !rd(f, g, kk) || !rd(f, T, kk) || !rd(f, swt, 1) ||
            !rd(f, rew, 1) || !rd(f, tl_obs, nfld) || !rd(f, tl_probes, nprobe))
            return 1;

        sb_decode_native(na, isd, mn, mx, mnr, mxr, cont, disc, dec);
        for (int i = 0; i < na; i++)
            wDec = fmax(wDec, fabs(dec[i] - nat[i]));
        for (int z = 0; z < nz; z++)
            wOcc = fmax(wOcc, fabs(occ.V[z] - occV[z]));

        sb_sim_apply_native(&sim, nat);
        wSW = fmax(wSW, fabs(sim.boiler_current - swt[0]));
        sb_sim_step(&sim, occV);

        for (int c = 0; c < kk; c++) {
            wG = fmax(wG, fabs(sim.air_g[c] - g[c]));
            wTT = fmax(wTT, fabs(sim.air_T[c] - T[c]));
        }
        sb_obs(&sim, occV, s + 1, obsC);
        for (int i = 0; i < nfld; i++) {
            double d = fabs((double)obsC[i] - tl_obs[i]);
            if (d > 0.0) obsDiff++;
            if (d > wObs) wObs = d;
        }
        for (int i = 0; i < nprobe; i++) {
            double d = fabs(sim.fdm->temp[pidx[i]] - tl_probes[i]);
            if (d > 0.0) probeDiff++;
            if (d > wProbe) wProbe = d;
        }
        sb_occup_advance(&occ, s + 1);
        double r = sb_reward(&sim, occ.V, s + 1, parts);
        wRew = fmax(wRew, fabs(r - rew[0]));
        rsumC += r;
        rsumT += rew[0];
        if (ci < ncheck && csteps[ci] == s) {
            memcpy(&cf[(size_t)ci * nx], sim.fdm->temp, 8 * (size_t)nx);
            ci++;
        }
    }

    if (fseek(f, field_off, SEEK_SET) != 0) return 1;
    for (int c = 0; c < ncheck; c++) {
        if (!rd(f, tnew, nx)) return 1;
        for (int i = 0; i < nx; i++) {
            double d = fabs(cf[(size_t)c * nx + i] - tnew[i]);
            if (d > 0.0) fieldDiff++;
            if (d > wField) wField = d;
        }
    }
    printf("steps=%d  decode=%.3e air_g=%.3e air_T=%.3e boiler=%.3e occ=%.3e\n",
           nsteps, wDec, wG, wTT, wSW, wOcc);
    printf("  obs   : max=%.3e  differing=%ld/%ld\n", wObs, obsDiff,
           (long)nfld * nsteps);
    printf("  probes: max=%.3e  differing=%ld/%ld\n", wProbe, probeDiff,
           (long)nprobe * nsteps);
    printf("  reward: max=%.3e  cum mine=%.9f  cum trace=%.9f\n", wRew, rsumC,
           rsumT);
    printf("  field : checkpoints=%d  max=%.3e  differing=%ld/%ld\n", ncheck,
           wField, fieldDiff, (long)nx * ncheck);
    /* air_T carries the same float32 noise (~5e-13) as the 30-step gate. */
    int bad = (wDec != 0.0) || (wOcc != 0.0) || (wG != 0.0) || (wTT > 1e-9) ||
              (wSW != 0.0) || (wObs > 1e-6f) || (wProbe > 1e-6) ||
              (wField > 1e-6) || (wRew > 1e-4) ||
              (fabs(rsumC - rsumT) > 1e-3);
    printf(bad ? "EPISODE GATE: FAIL\n" : "EPISODE GATE: OK\n");
    return bad;
}
