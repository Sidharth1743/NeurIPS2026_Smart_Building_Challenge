#include "sb_gpu_kernels.cuh"
#include "sb_rng.h"
#include <cstring>

#include "sb_gpu_host.hpp"

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    if (argc < 3) { printf("usage: %s <blob.sbb> <trace.step> [nenv]\n", argv[0]); return 2; }
    int nenv_arg = argc > 3 ? atoi(argv[3]) : 1;
    if (nenv_arg < 1 || nenv_arg > NE) return 2;
    nenv = nenv_arg;
    if (sb_gpu_setup(argv[1]) != 0) return 1;
    FILE* f = fopen(argv[2], "rb");
    if (!f) { printf("cannot open %s\n", argv[2]); return 1; }
    int hdr[7];
    if (fread(hdr, sizeof(int), 7, f) != 7) return 1;
    int t_ncv = hdr[0], t_nz = hdr[1], t_nc = hdr[2], t_nd = hdr[3];
    int nsteps = hdr[4], t_k = hdr[5], t_nfld = hdr[6];
    printf("trace: steps=%d nfld=%d\n", nsteps, t_nfld);
    std::vector<double> obs0(t_nfld);
    if (fread(obs0.data(), 8, t_nfld, f) != (size_t)t_nfld) return 1;
    std::vector<double> cont(t_nc), disc(t_nd), nat(n_action), occV(n_zones);
    std::vector<double> gg(k), TT(k), swt(1), tnew(ncv), rew(1), rp(12), obsv(t_nfld);
    double wG = 0, wT = 0, wSW = 0, wF = 0, wV = 0;
    const int th = 256;
    const int zth = (nenv * n_zones + th - 1) / th;
    const int cth = (nenv * ncv + th - 1) / th;
    const int nth2 = (nenv * n + th - 1) / th;
    const int kth = (nenv * k + th - 1) / th;
    size_t smem = (size_t)k * k * 8 + (size_t)k * 2 * 8;
    if (smem > 48 * 1024)
        CU(cudaFuncSetAttribute(airside_k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    /* S = sensed A^-1 applied, once (shared) */
    {
        double* d_apc; double* d_yrk;
        CU(cudaMalloc(&d_apc, 8 * (size_t)n * k));
        CU(cudaMalloc(&d_yrk, 8 * (size_t)n * k));
        cudssMatrix_t Bk, Xk;
        CK(cudssMatrixCreateDn(&Bk, n, k, n, d_apc, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CK(cudssMatrixCreateDn(&Xk, n, k, n, d_yrk, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CU(cudaMemset(d_apc, 0, 8 * (size_t)n * k));
        applied_cols_k<<<k, 128, 0, stream>>>(sd_ap, sd_ai, sd_apx, d_apc, k, n);
        CK(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, Xk, Bk));
        CU(cudaStreamSynchronize(stream));
        sensed_resp_k<<<(k * k + 127) / 128, 128, 0, stream>>>(sd_sp, sd_si, sd_spx,
                                                               d_yrk, d_sr, k, n);
        CU(cudaStreamSynchronize(stream));
        g_keep_apc = d_apc; g_keep_yrk = d_yrk;  /* see above: keep alive */
    }
    /* Group descriptors AFTER the analysis/factorization descriptors and the
     * air-side build, so the solver's state is not disturbed. */
    cudssMatrix_t Bg, Xg, B2g, X2g;
    CK(cudssMatrixCreateDn(&Bg, n, nenv, n, d_f, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CK(cudssMatrixCreateDn(&Xg, n, nenv, n, d_y, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CK(cudssMatrixCreateDn(&B2g, n, nenv, n, d_r, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CK(cudssMatrixCreateDn(&X2g, n, nenv, n, d_sol2, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    const double* op = sb_f64(&blob, "occ_params");
    const int nassign = (int)op[0];
    const int occ_ea = (int)op[1], occ_la = (int)op[2], occ_ed = (int)op[3],
              occ_ld = (int)op[4];
    const double occ_step = op[5];
    const double p_arr = 1.0 / ((occ_la - occ_ea) * 3600.0 / occ_step / 2.0);
    const double p_dep = 1.0 / ((occ_ld - occ_ed) * 3600.0 / occ_step / 2.0);
    double tstep = 0.0, t_pre = 0.0, t_sol = 0.0, t_post = 0.0, t_conv = 0.0, eSW = 0.0;
    double wRw = 0.0, wObs = 0.0, eRw = 0.0, eOb = 0.0;
    int wCons = 0;
    std::vector<GpuRng> rng0(nenv);
    CU(cudaMemcpy(rng0.data(), d_rngc, sizeof(GpuRng) * nenv, cudaMemcpyDeviceToHost));
    std::vector<double> boil_saved(nenv, boil_host[0]);
    {   /* the occupancy for [ts_0, ts_1) must exist before the first step */
        std::vector<int> sv(nenv, 0);
        CU(cudaMemcpy(d_stepv, sv.data(), 4 * (size_t)nenv, cudaMemcpyHostToDevice));
        occup_k<<<nenv, 32, 0, stream>>>(sh_hour, sh_wd, d_stepv, n_zones, nassign,
                                         occ_ea, occ_la, occ_ed, occ_ld, p_arr,
                                         p_dep, d_occstate, d_V, d_rngo);
    }
    for (int st = 0; st < nsteps; st++) {
        if (fread(cont.data(), 8, t_nc, f) != (size_t)t_nc) return 1;
        if (fread(disc.data(), 8, t_nd, f) != (size_t)t_nd) return 1;
        if (fread(nat.data(), 8, n_action, f) != (size_t)n_action) return 1;
        if (fread(occV.data(), 8, n_zones, f) != (size_t)n_zones) return 1;
        if (fread(gg.data(), 8, k, f) != (size_t)k) return 1;
        if (fread(TT.data(), 8, k, f) != (size_t)k) return 1;
        if (fread(swt.data(), 8, 1, f) != 1) return 1;
        /* the trace's swt is the boiler BEFORE the step: compare the value our
         * state carried in from the previous step, not the ramped output. */
        eSW = 0.0;
        for (int e = 0; e < nenv; e++)
            eSW = fmax(eSW, fabs(boil_saved[e] - swt[0]));
        wSW = fmax(wSW, eSW);
        if (fread(tnew.data(), 8, ncv, f) != (size_t)ncv) return 1;
        if (fread(rew.data(), 8, 1, f) != 1) return 1;
        if (fread(rp.data(), 8, 12, f) != 12) return 1;
        if (fread(obsv.data(), 8, t_nfld, f) != (size_t)t_nfld) return 1;
        for (int e = 0; e < nenv; e++)
            CU(cudaMemcpy(d_native + (size_t)e * n_action, nat.data(),
                          (size_t)n_action * 8, cudaMemcpyHostToDevice));
        double t0 = now_ms();
        sb_gpu_set_actions(nat.data());
        sb_gpu_step();
        std::vector<double> myT((size_t)nenv * ncv), myG((size_t)nenv * k), myTT((size_t)nenv * k);
        CU(cudaMemcpy(myT.data(), d_temp, 8 * (size_t)nenv * ncv, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(myG.data(), d_air_g, 8 * (size_t)nenv * k, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(myTT.data(), d_air_T, 8 * (size_t)nenv * k, cudaMemcpyDeviceToHost));
        GpuSim h0[NE];
        CU(cudaMemcpy(h0, d_sims, sizeof(GpuSim) * nenv, cudaMemcpyDeviceToHost));
        double eG = 0, eT = 0, eF = 0;
        for (size_t i = 0; i < (size_t)nenv * k; i++) {
            eG = fmax(eG, fabs(myG[i] - gg[i % k]));
            eT = fmax(eT, fabs(myTT[i] - TT[i % k]));
        }
        for (size_t e = 0; e < (size_t)nenv; e++)
            for (int p = 0; p < ncv; p++)
                eF = fmax(eF, fabs(myT[e * ncv + p] - tnew[p]));
        for (int e = 0; e < nenv; e++) boil_saved[e] = h0[e].boiler_current;
        wG = fmax(wG, eG); wT = fmax(wT, eT); wF = fmax(wF, eF);
        std::vector<double> hV((size_t)nenv * n_zones);
        CU(cudaMemcpy(hV.data(), d_V, 8 * (size_t)nenv * n_zones, cudaMemcpyDeviceToHost));
        double eV = 0.0;
        for (size_t i = 0; i < hV.size(); i++)
            eV = fmax(eV, fabs(hV[i] - occV[i % n_zones]));
        wV = fmax(wV, eV);
        /* The observation is taken with the occupancy of the interval this step
         * used; the chain then advances and the reward uses the next interval,
         * which is the order the oracle's env loop uses. */
        sb_gpu_finish(st + 1);
        tstep += now_ms() - t0;
            std::vector<double> myRew(nenv);
            std::vector<float> myObs((size_t)nenv * obs_len_);
            CU(cudaMemcpy(myRew.data(), d_rew, 8 * (size_t)nenv, cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(myObs.data(), d_obs, 4 * (size_t)nenv * obs_len_,
                          cudaMemcpyDeviceToHost));
            for (int e = 0; e < nenv; e++) eRw = fmax(eRw, fabs(myRew[e] - rew[0]));
            for (size_t i = 0; i < myObs.size(); i++) {
                int fi = (int)(i % (size_t)obs_len_);
                if (fi < t_nfld) eOb = fmax(eOb, fabs((double)myObs[i] - obsv[fi]));
            }
            wRw = fmax(wRw, eRw);
            wObs = fmax(wObs, eOb);
        }
        if (st < 12)
            printf("  step %2d: occ=%.3e air_g=%.3e air_T=%.3e field=%.3e boiler=%.3e\n",
                   st, eV, eG, eT, eF, eSW);
    }
    printf("phase ms/step: pre=%.3f solve=%.3f post=%.3f conv=%.3f (total %.3f)\n",
           t_pre / nsteps, t_sol / nsteps, t_post / nsteps, t_conv / nsteps,
           tstep / nsteps);
    printf("worst: occ=%.3e air_g=%.3e air_T=%.3e field=%.3e boiler=%.3e"
           " reward=%.3e obs=%.3e | %.2f ms/step (%d env, %d rng words)\n",
           wV, wG, wT, wF, wSW, wRw, wObs, tstep / nsteps, nenv, wCons);
    int bad = (wV != 0.0) || (wG > 1e-9) || (wT > 1e-9) || (wF > 1e-9)
           || (wObs != 0.0f) || (wRw > 1e-6);
    printf(bad ? "GPU SIM STEP: FAIL\n" : "GPU SIM STEP: OK\n");
    return bad;
}

extern "C" {}
