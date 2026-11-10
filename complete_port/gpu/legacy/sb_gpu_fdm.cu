/* M5 device FDM step, batched over a scenario group, gated on the M1b trace.
 *
 * Mirrors sb_fdm_step exactly:
 *   exposure -> forcing -> y = A^-1 forcing -> air side (acc, k x k solve,
 *   r = applied*q) -> yr = A^-1 r -> y += yr -> mass -> scatter.
 * The two triangular solves are cuDSS solves of the SAME factorization with one
 * column per environment.
 * usage: sb_gpu_fdm <blob.sbb> <trace.fdm>
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include <cuda_runtime.h>
#include <cudss.h>

extern "C" {
#include "sb_baked.h"
}

#define CK(x) do { cudssStatus_t s_=(x); if(s_!=CUDSS_STATUS_SUCCESS){printf("cuDSS err %d line %d\n",(int)s_,__LINE__);return 1;} } while(0)
#define CU(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){printf("cuda err %s line %d\n",cudaGetErrorString(e_),__LINE__);return 1;} } while(0)

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e3 + 1e-6 * (double)t.tv_nsec;
}

__global__ void exposure_k(const int* solved, const double* a0, const double* a1,
                           const double* horiz, const double* vert,
                           const double* cond, double h, double* e, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int p = solved[t];
    double v = 0.0;
    double x0 = a0[p];
    if (x0 > 0.0) v += h * x0 / (1.0 + h * 0.5 * horiz[p] / cond[p]);
    double x1 = a1[p];
    if (x1 > 0.0) v += h * x1 / (1.0 + h * 0.5 * vert[p] / cond[p]);
    e[t] = v;
}

/* Same accumulation order as sb_fdm_step so the result is within rounding of
 * the CPU reference: (((cap*T + q) + amb*(dir+expo)) + mc*massT). */
__global__ void forcing_k(const int* solved, const double* cap, const double* dir,
                          const double* mc, const double* mt, const double* temp,
                          const double* iq, const double* expo, double amb,
                          double* f, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    int p = solved[t];
    f[t] = cap[t] * temp[p] + iq[p] + amb * (dir[t] + expo[t]) + mc[t] * mt[t];
}

__global__ void sensed_acc_k(const int* sp, const int* si, const double* sx,
                             const double* y, double* acc, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    double a = 0.0;
    for (int p = sp[i]; p < sp[i + 1]; p++) a += sx[p] * y[si[p]];
    acc[i] = a;
}

__global__ void power_k(const double* g, const double* T, const double* acc,
                        double* pw, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) pw[i] = g[i] * (T[i] - acc[i]);
}

/* M = I + diag(g) S  (k x k, row-major), then solve M q = power in place with
 * partial pivoting, block-parallel: one thread per (row, column) pair. */
__global__ void dense_solve_k(const double* g, const double* sresp, double* pw,
                              int k) {
    extern __shared__ double sm[];
    double* M = sm;          /* k*k */
    double* q = sm + (size_t)k * k;
    int tid = threadIdx.x, nthr = blockDim.x;
    for (int i = tid; i < k * k; i += nthr) {
        int r = i / k, c = i % k;
        M[i] = (r == c ? 1.0 : 0.0) + g[r] * sresp[r * k + c];
    }
    for (int i = tid; i < k; i += nthr) q[i] = pw[i];
    if (tid == 0 && k == 0) return;
    __syncthreads();
    for (int c = 0; c < k; c++) {
        if (tid == 0) {
            int piv = c; double best = fabs(M[c * k + c]);
            for (int r = c + 1; r < k; r++) {
                double v = fabs(M[r * k + c]);
                if (v > best) { best = v; piv = r; }
            }
            if (piv != c) {
                for (int j = c; j < k; j++) {
                    double tmp = M[c * k + j]; M[c * k + j] = M[piv * k + j];
                    M[piv * k + j] = tmp;
                }
                double tb = q[c]; q[c] = q[piv]; q[piv] = tb;
            }
            double d = M[c * k + c];
            for (int j = c; j < k; j++) M[c * k + j] /= d;
            q[c] /= d;
        }
        __syncthreads();
        for (int idx = tid; idx < (k - c - 1) * (k - c); idx += nthr) {
            int r = c + 1 + idx / (k - c), j = c + idx % (k - c);
            M[r * k + j] -= M[r * k + c] * M[c * k + j];
            if (j == c) q[r] -= M[r * k + c] * q[c];
        }
        __syncthreads();
    }
    for (int i = tid; i < k; i += nthr) pw[i] = q[i];
}

__global__ void applied_r_k(const int* ap, const int* ai, const double* ax,
                            const double* q, double* r, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    for (int p = ap[i]; p < ap[i + 1]; p++) r[ai[p]] += ax[p] * q[i];
}

__global__ void add_y_k(const double* src, double* y, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) y[t] += src[t];
}

__global__ void mass_scatter_k(const int* solved, const double* mret,
                               const double* mgain, const double* y,
                               double* mt, double* temp, double amb, int n) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    double v = mt[t] * mret[t] + mgain[t] * y[t];
    mt[t] = v;
    temp[solved[t]] = y[t];
}

/* One column of `applied` (CSC) per batch column -> the RHS used to build the
 * air-side response matrix S = sensed @ A^-1 @ applied, exactly as sb_fdm_init. */
__global__ void applied_cols_k(const int* ap, const int* ai, const double* ax,
                               double* R, int k, int n) {
    int j = blockIdx.x;
    if (j >= k) return;
    for (int p = ap[j] + threadIdx.x; p < ap[j + 1]; p += blockDim.x)
        R[(size_t)j * n + ai[p]] = ax[p];
}

__global__ void sensed_resp_k(const int* sp, const int* si, const double* sx,
                              const double* yr, double* SR, int k, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k * k) return;
    int i = idx / k, j = idx % k;
    const double* col = yr + (size_t)j * n;
    double a = 0.0;
    for (int p = sp[i]; p < sp[i + 1]; p++) a += sx[p] * col[si[p]];
    SR[i * k + j] = a;
}

__global__ void fill_amb_k(double* temp, double amb, int ncv) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < ncv) temp[p] = amb;
}

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    if (argc < 3) { printf("usage: %s <blob.sbb> <trace.fdm>\n", argv[0]); return 2; }
    SbBlob blob;
    if (sb_open(&blob, argv[1]) != 0) return 1;
    const int ncv = (int)sb_i64(&blob, "grid")[0] * (int)sb_i64(&blob, "grid")[1];
    const int n = (int)sb_find(&blob, "solved")->count;
    const int k = (int)sb_i64(&blob, "sensed_shape")[0];
    const int64_t* solved64 = sb_i64(&blob, "solved");
    std::vector<int> solved(n);
    for (int i = 0; i < n; i++) solved[i] = (int)solved64[i];
    printf("blob %s: n_cvs=%d n_solved=%d k=%d\n", argv[1], ncv, n, k);

    /* ---- A_system CSC -> CSR ---- */
    const int* Sc_p = sb_i32(&blob, "A_system_indptr");
    const int* Sc_i = sb_i32(&blob, "A_system_indices");
    const double* Sc_x = sb_f64(&blob, "A_system_data");
    std::vector<int> Ap_v(n + 1, 0), Ai_v; std::vector<double> Ax_v;
    {
        std::vector<int> cnt(n, 0);
        for (int j = 0; j < n; j++)
            for (int q = Sc_p[j]; q < Sc_p[j + 1]; q++) cnt[Sc_i[q]]++;
        for (int i = 0; i < n; i++) Ap_v[i + 1] = Ap_v[i] + cnt[i];
        std::vector<int> nxt(Ap_v.begin(), Ap_v.end() - 1);
        Ai_v.assign((size_t)Ap_v[n], 0); Ax_v.assign((size_t)Ap_v[n], 0.0);
        for (int j = 0; j < n; j++)
            for (int q = Sc_p[j]; q < Sc_p[j + 1]; q++) {
                int i = Sc_i[q]; Ai_v[nxt[i]] = j; Ax_v[nxt[i]] = Sc_x[q]; nxt[i]++;
            }
    }
    const int nnz = Ap_v[n];

    /* ---- device buffers ---- */
    double *d_Ax, *d_cap, *d_dir, *d_mc, *d_ret, *d_gain, *d_hor, *d_ver, *d_con;
    double *d_a0, *d_a1, *d_ep, *d_f, *d_y, *d_r, *d_mt, *d_temp, *d_sx, *d_ax;
    double *d_ssp, *d_scr, *d_ag, *d_aT, *d_acc, *d_pw, *d_batch, *d_sol, *d_iq;
    int *d_solved, *d_Ap, *d_Ai, *d_sp, *d_si, *d_ap, *d_ai;
    size_t nb = (size_t)n * 8;
    CU(cudaMalloc(&d_solved, (size_t)n * 4));
    CU(cudaMemcpy(d_solved, solved.data(), (size_t)n * 4, cudaMemcpyHostToDevice));
#define UP(ptr, name, cnt) do { const double* src_ = sb_f64(&blob, name); \
    CU(cudaMalloc(&ptr, (size_t)(cnt) * 8)); \
    CU(cudaMemcpy(ptr, src_, (size_t)(cnt) * 8, cudaMemcpyHostToDevice)); } while (0)
    UP(d_cap, "capacitance_rate", n);
    UP(d_dir, "dirichlet_conductance", n);
    UP(d_mc, "mass_coupling", n);
    UP(d_ret, "mass_retained", n);
    UP(d_gain, "mass_gained", n);
    UP(d_hor, "horizontal", ncv);
    UP(d_ver, "vertical", ncv);
    UP(d_con, "conductivity", ncv);
    UP(d_a0, "ambient_face_areas_0", ncv);
    UP(d_a1, "ambient_face_areas_1", ncv);
    UP(d_mt, "mass_temp_init", n);
    UP(d_temp, "temp_init", ncv);
    int nsn = (int)sb_find(&blob, "sensed_data")->count;
    int nap = (int)sb_find(&blob, "applied_data")->count;
    UP(d_sx, "sensed_data", nsn);
    UP(d_ax, "applied_data", nap);
    /* sensed_response is NOT baked: built below, after the factorization. */
    CU(cudaMalloc(&d_ssp, (size_t)k * k * 8));
    CU(cudaMalloc(&d_ep, nb)); CU(cudaMalloc(&d_f, nb));
    CU(cudaMalloc(&d_iq, (size_t)ncv * 8));
    CU(cudaMalloc(&d_y, nb)); CU(cudaMalloc(&d_r, nb));
    CU(cudaMalloc(&d_ag, (size_t)k * 8)); CU(cudaMalloc(&d_aT, (size_t)k * 8));
    CU(cudaMalloc(&d_acc, (size_t)k * 8)); CU(cudaMalloc(&d_pw, (size_t)k * 8));
    CU(cudaMalloc(&d_Ap, (size_t)(n + 1) * 4));
    CU(cudaMalloc(&d_Ai, (size_t)nnz * 4));
    CU(cudaMalloc(&d_Ax, (size_t)nnz * 8));
    CU(cudaMemcpy(d_Ap, Ap_v.data(), (size_t)(n + 1) * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Ai, Ai_v.data(), (size_t)nnz * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_Ax, Ax_v.data(), (size_t)nnz * 8, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_sp, (size_t)(k + 1) * 4));
    CU(cudaMalloc(&d_si, (size_t)nsn * 4));
    CU(cudaMemcpy(d_sp, sb_i32(&blob, "sensed_indptr"), (size_t)(k + 1) * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_si, sb_i32(&blob, "sensed_indices"), (size_t)nsn * 4, cudaMemcpyHostToDevice));
    CU(cudaMalloc(&d_ap, (size_t)(k + 1) * 4));
    CU(cudaMalloc(&d_ai, (size_t)nap * 4));
    CU(cudaMemcpy(d_ap, sb_i32(&blob, "applied_indptr"), (size_t)(k + 1) * 4, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(d_ai, sb_i32(&blob, "applied_indices"), (size_t)nap * 4, cudaMemcpyHostToDevice));

    /* ---- cuDSS: one factorization, two solves per step ---- */
    cudssHandle_t handle; CK(cudssCreate(&handle));
    cudaStream_t stream; CU(cudaStreamCreate(&stream));
    CK(cudssSetStream(handle, stream));
    cudssConfig_t cfg; CK(cudssConfigCreate(&cfg));
    cudssData_t data; CK(cudssDataCreate(handle, &data));
    cudssMatrix_t A, B1, X1;
    CK(cudssMatrixCreateCsr(&A, n, n, nnz, d_Ap, NULL, d_Ai, d_Ax,
                            CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                            CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CU(cudaMalloc(&d_batch, nb)); CU(cudaMalloc(&d_sol, nb));
    CK(cudssMatrixCreateDn(&B1, n, 1, n, d_batch, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CK(cudssMatrixCreateDn(&X1, n, 1, n, d_sol, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    double h0 = now_ms();
    CK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, cfg, data, A, X1, B1));
    CK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, cfg, data, A, X1, B1));
    CU(cudaStreamSynchronize(stream));
    printf("factorization %.1f ms\n", now_ms() - h0);

    /* Air-side response S = sensed @ A^-1 @ applied (k x k), as sb_fdm_init
     * builds it: one batched solve of the `applied` columns. */
    {
        double* d_apc; double* d_yrk;
        CU(cudaMalloc(&d_apc, (size_t)n * k * 8));
        CU(cudaMalloc(&d_yrk, (size_t)n * k * 8));
        cudssMatrix_t Bk, Xk;
        CK(cudssMatrixCreateDn(&Bk, n, k, n, d_apc, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CK(cudssMatrixCreateDn(&Xk, n, k, n, d_yrk, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        CU(cudaMemset(d_apc, 0, (size_t)n * k * 8));
        applied_cols_k<<<k, 128, 0, stream>>>(d_ap, d_ai, d_ax, d_apc, k, n);
        CK(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, Xk, Bk));
        CU(cudaStreamSynchronize(stream));
        sensed_resp_k<<<(k * k + 127) / 128, 128, 0, stream>>>(d_sp, d_si, d_sx,
                                                               d_yrk, d_ssp, k, n);
        CU(cudaStreamSynchronize(stream));
        CU(cudaFree(d_apc)); CU(cudaFree(d_yrk));
    }

    /* ---- gate: replay the M1b trace ---- */
    FILE* f = fopen(argv[2], "rb");
    if (!f) { printf("cannot open %s\n", argv[2]); return 1; }
    int hdr[5];
    if (fread(hdr, sizeof(int), 5, f) != 5) return 1;
    int tn = hdr[0], tk = hdr[1], ts = hdr[2], nsteps = hdr[3];
    if (tn != ncv || tk != k || ts != n) { printf("shape mismatch\n"); return 1; }
    std::vector<double> g(k), Tt(k), iqv(ncv), tprev(ncv), tnew(ncv), mb(n), mtemp(n);
    double worst = 0.0, worstm = 0.0;
    int threads = 256;
    int gb = (n + threads - 1) / threads, cb = (ncv + threads - 1) / threads;
    int kb = (k + threads - 1) / threads;
    size_t smem = (size_t)k * k * 8 + (size_t)k * 8;
    /* The opt-in is only valid above the 48 KB static limit; below it the
     * default already covers us (k is small). */
    if (smem > 48 * 1024)
        CU(cudaFuncSetAttribute(dense_solve_k,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem));
    double tsum = 0.0;
    for (int st = 0; st < nsteps; st++) {
        double amb, cc;
        if (fread(&amb, 8, 1, f) != 1 || fread(&cc, 8, 1, f) != 1) return 1;
        if (fread(g.data(), 8, k, f) != (size_t)k) return 1;
        if (fread(Tt.data(), 8, k, f) != (size_t)k) return 1;
        if (fread(tprev.data(), 8, ncv, f) != (size_t)ncv) return 1;
        if (fread(iqv.data(), 8, ncv, f) != (size_t)ncv) return 1;
        if (fread(mb.data(), 8, n, f) != (size_t)n) return 1;
        if (fread(tnew.data(), 8, ncv, f) != (size_t)ncv) return 1;
        if (fread(mtemp.data(), 8, n, f) != (size_t)n) return 1;
        CU(cudaMemcpy(d_temp, tprev.data(), (size_t)ncv * 8, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_mt, mb.data(), (size_t)n * 8, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_aT, Tt.data(), (size_t)k * 8, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_ag, g.data(), (size_t)k * 8, cudaMemcpyHostToDevice));
        CU(cudaMemcpy(d_iq, iqv.data(), (size_t)ncv * 8, cudaMemcpyHostToDevice));
        double h1 = now_ms();
        exposure_k<<<gb, threads, 0, stream>>>(d_solved, d_a0, d_a1, d_hor, d_ver, d_con, cc, d_ep, n);
        forcing_k<<<gb, threads, 0, stream>>>(d_solved, d_cap, d_dir, d_mc, d_mt,
                                             d_temp, d_iq, d_ep, amb, d_f, n);
        CU(cudaMemcpyAsync(d_batch, d_f, nb, cudaMemcpyDeviceToDevice, stream));
        CK(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, X1, B1));
        /* the batched solve works on d_sol; copy column 0 into y */
        CU(cudaMemcpyAsync(d_y, d_sol, nb, cudaMemcpyDeviceToDevice, stream));
        sensed_acc_k<<<kb, threads, 0, stream>>>(d_sp, d_si, d_sx, d_y, d_acc, k);
        power_k<<<kb, threads, 0, stream>>>(d_ag, d_aT, d_acc, d_pw, k);
        dense_solve_k<<<1, 256, smem, stream>>>(d_ag, d_ssp, d_pw, k);
        CU(cudaMemsetAsync(d_r, 0, nb, stream));
        applied_r_k<<<kb, threads, 0, stream>>>(d_ap, d_ai, d_ax, d_pw, d_r, k);
        CU(cudaMemcpyAsync(d_batch, d_r, nb, cudaMemcpyDeviceToDevice, stream));
        CK(cudssExecute(handle, CUDSS_PHASE_SOLVE, cfg, data, A, X1, B1));
        CU(cudaMemcpyAsync(d_r, d_sol, nb, cudaMemcpyDeviceToDevice, stream));
        add_y_k<<<gb, threads, 0, stream>>>(d_r, d_y, n);
        fill_amb_k<<<cb, threads, 0, stream>>>(d_temp, amb, ncv);
        mass_scatter_k<<<gb, threads, 0, stream>>>(d_solved, d_ret, d_gain, d_y, d_mt, d_temp, amb, n);
        CU(cudaStreamSynchronize(stream));
        tsum += now_ms() - h1;
        std::vector<double> tc(ncv), mtc(n);
        CU(cudaMemcpy(tc.data(), d_temp, (size_t)ncv * 8, cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(mtc.data(), d_mt, (size_t)n * 8, cudaMemcpyDeviceToHost));
        double e = 0.0, em = 0.0;
        for (int p = 0; p < ncv; p++) e = fmax(e, fabs(tc[p] - tnew[p]));
        for (int p = 0; p < n; p++) em = fmax(em, fabs(mtc[p] - mtemp[p]));
        worst = fmax(worst, e); worstm = fmax(worstm, em);
        if (st < 3 || e > 1e-9)
            printf("  step %2d: T err=%.3e mass err=%.3e\n", st, e, em);
    }
    printf("worst: T=%.3e mass=%.3e | %.3f ms/step (device, 1 env)\n", worst, worstm,
           tsum / nsteps);
    int bad = (worst > 1e-9) || (worstm > 1e-9);
    printf(bad ? "GPU FDM: FAIL\n" : "GPU FDM: OK\n");
    return bad;
}
