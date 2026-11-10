/* CPU reference for multi-episode parity.  Drives the verified C core exactly the
 * way the CPU PufferLib env does (puf_reset / puf_step: reset, occupant prime,
 * reset observation, interval-0 advance; step = apply native action, sb_sim_step
 * on the env's own occupancy, observation with the step's interval, occupancy
 * advance, reward) for E episodes, replaying the oracle trace's native actions
 * in each episode, and writes every reset observation / observation / reward to
 * a file the GPU gate diffs against (gpu/smart_buildings.cu, SB_TEST).
 *   usage: test_episodes <blob.sbb> <trace.step> <episodes> <out.ref> [steps]
 * `steps` (default: the trace length) runs longer episodes by cycling the trace's
 * native actions.
 * Output (little endian): int32 {n_field, steps, episodes}, then per episode:
 *   float32 reset_obs[n_field]; steps x { float32 obs[n_field]; float64 reward }.
 * The convection / occupancy generators and occupant chains are never reseeded:
 * they carry over from one episode to the next, like the env's. */
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

static int rd(FILE* f, void* p, size_t sz, size_t n) { return fread(p, sz, n, f) == n; }

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <blob> <trace> <episodes> <out.ref>\n", argv[0]);
        return 2;
    }
    const int E = atoi(argv[3]);
    const int run_steps_arg = argc > 5 ? atoi(argv[5]) : 0;
    static SbBlob b;
    if (sb_open(&b, argv[1]) != 0) return 1;
    static SbFdm fdm; static SbConv conv; static SbOccup occ; static SbSim sim;
    if (sb_fdm_init(&fdm, &b) || sb_conv_init(&conv, &b) || sb_occup_init(&occ, &b) ||
        sb_sim_init(&sim, &b, &fdm, &conv, &occ)) { printf("init FAIL\n"); return 1; }
    const int na = sim.n_action;
    FILE* f = fopen(argv[2], "rb");
    if (!f) return 1;
    int hdr[7];
    if (!rd(f, hdr, 4, 7)) return 1;
    const int n_cvs = hdr[0], nz = hdr[1], nc = hdr[2], nd = hdr[3], nsteps = hdr[4],
              kk = hdr[5], nfld = hdr[6];
    double* obs0 = malloc(8 * nfld);
    if (!rd(f, obs0, 8, nfld)) return 1;
    /* per-step native actions from the trace */
    double* nat = malloc((size_t)8 * na * nsteps);
    double* cont = malloc(8 * nc); double* disc = malloc(8 * nd);
    double* occV = malloc(8 * nz); double* g = malloc(8 * kk); double* T = malloc(8 * kk);
    double* sw = malloc(8); double* tnew = malloc(8 * n_cvs); double* rew = malloc(8);
    double* rp = malloc(96); double* ob = malloc(8 * nfld);
    for (int s = 0; s < nsteps; s++) {
        if (!rd(f, cont, 8, nc) || !rd(f, disc, 8, nd) || !rd(f, nat + (size_t)s * na, 8, na) ||
            !rd(f, occV, 8, nz) || !rd(f, g, 8, kk) || !rd(f, T, 8, kk) || !rd(f, sw, 8, 1) ||
            !rd(f, tnew, 8, n_cvs) || !rd(f, rew, 8, 1) || !rd(f, rp, 8, 12) ||
            !rd(f, ob, 8, nfld)) return 1;
    }
    fclose(f);

    FILE* out = fopen(argv[4], "wb");
    if (!out) return 1;
    const int run_steps = run_steps_arg > 0 ? run_steps_arg : nsteps;
    int oh[3] = {nfld, run_steps, E};
    fwrite(oh, 4, 3, out);
    float* o32 = malloc(4 * (size_t)nfld);
    double parts[12];
    for (int ep = 0; ep < E; ep++) {
        /* puf_reset */
        sb_sim_reset(&sim, &b);
        sb_occup_prime(&occ);
        sb_obs(&sim, occ.V, 0, o32);
        fwrite(o32, 4, nfld, out);
        sb_occup_advance(&occ, 0);
        double worst_o = 0.0;
        for (int i = 0; i < nfld; i++)
            if (ep == 0) worst_o = fmax(worst_o, fabs((double)o32[i] - obs0[i]));
        if (ep == 0) printf("episode 0 reset obs vs oracle: %.3e\n", worst_o);
        double ret = 0.0;
        for (int s = 0; s < run_steps; s++) {
            sb_sim_apply_native(&sim, nat + (size_t)(s % nsteps) * na);
            sb_sim_step(&sim, occ.V);
            sb_obs(&sim, occ.V, s + 1, o32);
            sb_occup_advance(&occ, s + 1);
            double r = sb_reward(&sim, occ.V, s + 1, parts);
            fwrite(o32, 4, nfld, out);
            fwrite(&r, 8, 1, out);
            ret += r;
        }
        printf("episode %d: return %.9f\n", ep, ret);
    }
    fclose(out);
    return 0;
}
