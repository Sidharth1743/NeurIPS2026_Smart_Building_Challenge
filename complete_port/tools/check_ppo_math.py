#!/usr/bin/env python3
"""Compile the actual trainer loss/cache kernels and check their derivatives.

CUDA runs only in puffertank. The reference is an independent double-precision
scalar objective, differentiated by central finite differences. No simulator,
training data, copied implementation of the backward pass, or host CUDA toolkit
is used. Generated sources and output are retained for reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
PL_ROOT = os.environ.get("SB_PL_ROOT", os.path.expanduser("~/.cache/sb/pufferlib"))

CP = Path(__file__).resolve().parents[1]


def definition(source: str, name: str, kind: str = "function") -> str:
    if kind == "function":
        start = source.index(name + "(")
        start = source.rfind("\n", 0, start) + 1
    else:
        start = source.index(kind + " " + name)
    brace = source.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        if source[end] == "{": depth += 1
        elif source[end] == "}": depth -= 1
        end += 1
    if kind != "function": end = source.index(";", end) + 1
    return source[start:end] + "\n"


HARNESS = r'''
template<class T> T* managed(int n) {
    T* p = nullptr; CUDA_OK(cudaMallocManaged(&p, sizeof(T)*n));
    memset(p, 0, sizeof(T)*n); return p;
}
struct Case {
    const char* name; int mode, per_dim; double advantage, shift;
    double logits[5], logstd[4], actions[3], mask[4];
    double floor;   // policy.beta_floor; 0 in the table means the default 1
};
static double g_floor = 1.0;
static double bounded(double x, double low, double high) {
    return std::max(low, std::min(high, x));
}
// log(sech^2(x)), the log-determinant of tanh at the pre-squash draw.
static double squash_logdet(double x) {
    double ax = std::abs(x);
    return 1.3862943611198906 - 2.0 * ax - 2.0 * std::log1p(std::exp(-2.0 * ax));
}
static double softplus1(double u) { return std::log1p(std::exp(u)) + g_floor; }
static double g_c0 = 0.0;   // policy.binary_run offset in mode 4
// Integral of f(t) h(t) over [lo, hi] (lo = 0 or hi = 1 at one end) for the Beta(a, b) density f,
// midpoint rule after t = s^(1/a) on the half at 0 and 1 - t = s^(1/b) on the half at 1. The
// substitution cancels the t^(a-1) and (1-t)^(b-1) singularities when a shape is below 1.
template <class H>
static double beta_integral(double a, double b, double hi, H h) {
    const int n = 400000;
    const double lnB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    double sum = 0.0;
    const double m = std::min(hi, 0.5), lsm = a * std::log(m);   // log of the s range, m^a
    for (int i = 0; i < n; ++i) {
        double t = m * std::pow((i + 0.5) / n, 1.0 / a);
        sum += std::exp((b - 1) * std::log1p(-t) - lnB + lsm - std::log(a * n)) * h(t);
    }
    if (hi > 0.5) {
        const double sh = std::pow(0.5, b), sl = std::pow(1.0 - hi, b);
        for (int i = 0; i < n; ++i) {
            double s = sl + (sh - sl) * (i + 0.5) / n, r = std::pow(s, 1.0 / b), t = 1.0 - r;
            sum += std::exp((a - 1) * std::log(t) - lnB) / b * h(t) * (sh - sl) / n;
        }
    }
    return sum;
}
// Entropy of Beta(a, b) as the integral of -f log f. Independent of the closed form and of the
// digamma the kernel uses.
static double beta_entropy_numeric(double a, double b) {
    const double lnB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    return beta_integral(a, b, 1.0, [&](double t) {
        return -((a - 1) * std::log(t) + (b - 1) * std::log1p(-t) - lnB); });
}
static void reference_dist(const Case& c, const double* x, const double* ls,
                           double* lp, double* entropy, int* active) {
    *lp = *entropy = 0; *active = 0;
    for (int h=0; h<2; ++h) {
        if (!c.mask[h]) continue;
        if (c.mode==4 && h==1) {
            // policy.binary_run: head 1 is Bernoulli, p = sigmoid(u + c0), action 1 on / 0 off.
            double p = 1.0 / (1.0 + std::exp(-(x[h] + g_c0)));
            *lp += c.actions[h] > 0.5 ? std::log(p) : std::log1p(-p);
            *entropy += -p * std::log(p) - (1 - p) * std::log1p(-p);
            ++*active;
            continue;
        }
        if (c.mode==3 || c.mode==4 || c.mode==5) {
            // Beta head: x[h] is u, ls[h] is v (the second decoder block).
            double a = softplus1(x[h]), b = softplus1(ls[h]), xx = c.actions[h];
            if(c.mode==5){double m=1.0/(1.0+std::exp(-x[h])),k=2.0+10.0/(1.0+std::exp(-ls[h]));a=m*k;b=(1-m)*k;}
            double lnB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
            *lp += (a - 1) * std::log(xx) + (b - 1) * std::log1p(-xx) - lnB;
            *entropy += beta_entropy_numeric(a, b);
            ++*active;
            continue;
        }
        double mu = c.mode==2 ? 3*std::tanh(x[h]/3)
                  : c.mode==1 ? std::tanh(x[h]) : x[h];
        double z = (c.actions[h]-mu)/std::exp(ls[h]);
        *lp += -.5*z*z - .9189385332046727 - ls[h];
        *entropy += 1.4189385332046727 + ls[h];
        // Executed-action entropy. The stored draw fixes the noise; moving the
        // location moves the draw with it, which is what pulls a rail sample back.
        if (c.mode==2) {
            double mu0 = 3*std::tanh(c.logits[h]/3.0);
            double sig0 = std::exp(c.logstd[h]);
            double eps = (c.actions[h] - mu0) / sig0;
            *entropy += squash_logdet(mu + std::exp(ls[h]) * eps);
        }
        ++*active;
    }
    if (c.mask[2] || c.mask[3]) {
        double m=-1e100, sum=0;
        for (int j=0;j<2;++j) if(c.mask[2+j]) m=std::max(m,x[2+j]);
        for (int j=0;j<2;++j) if(c.mask[2+j]) sum+=std::exp(x[2+j]-m);
        double lse=m+std::log(sum);
        *lp += x[2+(int)c.actions[2]]-lse;
        for (int j=0;j<2;++j) if(c.mask[2+j]) {
            double logp=x[2+j]-lse; *entropy-=std::exp(logp)*logp;
        }
        ++*active;
    }
}
static double objective(const Case& c, const double* x, const double* ls,
                        double oldlp, double offset) {
    double lp, ent; int active; reference_dist(c,x,ls,&lp,&ent,&active);
    double denominator=c.per_dim ? std::max(active,1) : 1;
    double ratio=std::exp((lp+offset-oldlp)/denominator);
    double pg=std::max(-c.advantage*ratio,
                       -c.advantage*bounded(ratio,.8,1.2));
    // The existing geometric-ratio variant deliberately scales the PG gradient
    // by active-head count. Check that documented objective, rather than silently
    // interpreting it as ordinary PPO.
    pg*=denominator;
    double clipped=.1+bounded(x[4]-.1,-10,10);
    double vf=.5*std::max((x[4]+.5)*(x[4]+.5),(clipped+.5)*(clipped+.5));
    return pg+.5*vf-.01*ent;
}
// psi and psi' against scipy.special (digamma, polygamma(1, .)), tabulated on the host.
__global__ void beta_special_k(const double* x, double* out, int n) {
    int i = threadIdx.x; if (i >= n) return;
    out[2*i] = beta_digamma(x[i]); out[2*i+1] = beta_trigamma(x[i]);
}
static int beta_special_check(int* checks) {
    const double ref[][3] = {
        {1.0,-0.5772156649015329,1.6449340668482266},{1.0001,-0.5770511835143349,1.6446936879331446},
        {1.3132616875182228,-0.1542532379526638,1.1185682772656016},{1.5,0.03648997397857652,0.9348022005446793},
        {2.0,0.42278433509846713,0.6449340668482266},{3.7,1.1671535393615113,0.31003785767003833},
        {5.999,1.7059363290792255,0.18135575138433863},{6.0,1.7061176684318005,0.18132295573711532},
        {10.0,2.251752589066721,0.10516633568168576},{100.0,4.600161852738088,0.010050166663333573},
        {1e4,9.21029037114285,0.00010000500016666666},{1e6,13.81551005796419,1.0000005000001667e-06}};
    const int n = sizeof(ref) / sizeof(ref[0]);
    double *x = managed<double>(n), *out = managed<double>(2 * n);
    for (int i = 0; i < n; ++i) x[i] = ref[i][0];
    beta_special_k<<<1, 32>>>(x, out, n);
    CUDA_OK(cudaDeviceSynchronize());
    int bad = 0;
    for (int i = 0; i < n; ++i) for (int k = 0; k < 2; ++k) {
        double got = out[2*i+k], want = ref[i][1+k];
        ++*checks;
        if (!(std::abs(got - want) <= 1e-9 * std::max(1.0, std::abs(want)))) {
            printf("MISMATCH %s(%.9g) got %.17g ref %.17g\n", k ? "trigamma" : "digamma", ref[i][0], got, want); ++bad;
        }
    }
    printf("%s beta digamma/trigamma vs scipy (%d points)\n", bad ? "FAIL" : "PASS", n);
    return bad ? 1 : 0;
}
// The float digamma/trigamma the policy head uses, against the same scipy values: a float tolerance.
__global__ void beta_special_f_k(const double* x, double* out, int n) {
    int i = threadIdx.x; if (i >= n) return;
    out[2*i] = beta_digamma_f((float)x[i]); out[2*i+1] = beta_trigamma_f((float)x[i]);
}
static int beta_special_f_check(int* checks) {
    const double ref[][3] = {
        {1.0,-0.5772156649015329,1.6449340668482266},{1.3132616875182228,-0.1542532379526638,1.1185682772656016},
        {2.0,0.42278433509846713,0.6449340668482266},{3.7,1.1671535393615113,0.31003785767003833},
        {6.0,1.7061176684318005,0.18132295573711532},{10.0,2.251752589066721,0.10516633568168576},
        {100.0,4.600161852738088,0.010050166663333573},{1e4,9.21029037114285,0.00010000500016666666}};
    const int n = sizeof(ref) / sizeof(ref[0]);
    double *x = managed<double>(n), *out = managed<double>(2 * n);
    for (int i = 0; i < n; ++i) x[i] = ref[i][0];
    beta_special_f_k<<<1, 32>>>(x, out, n);
    CUDA_OK(cudaDeviceSynchronize());
    int bad = 0;
    for (int i = 0; i < n; ++i) for (int k = 0; k < 2; ++k) {
        double got = out[2*i+k], want = ref[i][1+k];
        ++*checks;
        if (!(std::abs(got - want) <= 1e-6 * std::max(1.0, std::abs(want)))) {
            printf("MISMATCH %s_f(%.9g) got %.17g ref %.17g\n", k ? "trigamma" : "digamma", ref[i][0], got, want); ++bad;
        }
    }
    printf("%s beta float digamma/trigamma vs scipy (%d points, 1e-6)\n", bad ? "FAIL" : "PASS", n);
    return bad ? 1 : 0;
}
// The mixed-precision head against an all-double evaluation of the same formulas (the double
// special functions above are checked against scipy to 1e-9), over concentrations 2 .. ~2e4 and
// draws from the rails to the middle. Bounds: log-prob within 2e-5 + 1e-7 (a + b) + 2e-7 m, where
// m = (a - 1)|log x| + (b - 1)|log(1 - x)| (one float rounding of each log, times its weight; the
// float draw itself already moves the log-prob by ~6e-8 (a + b)); entropy within 1e-5 + 2e-6 (a + b)
// (float digamma, ~1e-7 relative, times weights summing to 2 (a + b - 2); the entropy value is only
// reported, its gradient is the trigamma terms below);
// each derivative within 1e-5 + 1e-4 |ref| + 2e-7 (a + b).
__device__ void beta_head_double(double u, double v, float action, double* lp, double* ent, double* g) {
    const double a = beta_concentration(u), b = beta_concentration(v), s = a + b;
    const double x = beta_clamp_x(action), lx = log(x), l1x = log1p(-x);
    const double lnB = lgamma(a) + lgamma(b) - lgamma(s);
    const double pa = beta_digamma(a), pb = beta_digamma(b), ps = beta_digamma(s);
    *lp = (a - 1) * lx + (b - 1) * l1x - lnB;
    *ent = lnB - (a - 1) * pa - (b - 1) * pb + (s - 2) * ps;
    const double ta = beta_trigamma(a), tb = beta_trigamma(b), ts = beta_trigamma(s);
    const double su = beta_sigmoid(u), sv = beta_sigmoid(v);
    g[0] = (lx - pa + ps) * su; g[1] = (l1x - pb + ps) * sv;
    g[2] = (-(a - 1) * ta + (s - 2) * ts) * su; g[3] = (-(b - 1) * tb + (s - 2) * ts) * sv;
}
__global__ void beta_head_compare_k(const float* uvx, double* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float lp, ent, g[4]; double dlp, dent, dg[4];
    beta_continuous_head(uvx[3*i], uvx[3*i+1], uvx[3*i+2], &lp, &ent, g);
    beta_head_double(uvx[3*i], uvx[3*i+1], uvx[3*i+2], &dlp, &dent, dg);
    double* o = out + 14 * (size_t)i;
    o[0] = lp; o[1] = dlp; o[2] = ent; o[3] = dent;
    for (int k = 0; k < 4; ++k) { o[4+k] = g[k]; o[8+k] = dg[k]; }
    const double a = beta_concentration(uvx[3*i]), b = beta_concentration(uvx[3*i+1]);
    const double x = beta_clamp_x(uvx[3*i+2]);
    o[12] = a + b;
    o[13] = fabs(a - 1) * fabs(log(x)) + fabs(b - 1) * fabs(log1p(-x));
}
static int beta_head_compare_check(int* checks, float floor) {
    CUDA_OK(cudaMemcpyToSymbol(d_beta_floor, &floor, sizeof(floor)));
    const float us[] = {-6, -1, 0, 0.7f, 2, 5, 20, 100, 1000, 10000};
    const float xs[] = {1e-6f, 0.002f, 0.05f, 0.3f, 0.5f, 0.77f, 0.98f, 0.99999f};
    const int nu = sizeof(us) / sizeof(us[0]), nx = sizeof(xs) / sizeof(xs[0]);
    const int n = nu * nu * nx;
    float* uvx = managed<float>(3 * n); double* out = managed<double>(14 * n);
    int i = 0;
    for (float u : us) for (float v : us) for (float x : xs) { uvx[3*i] = u; uvx[3*i+1] = v; uvx[3*i+2] = x; ++i; }
    beta_head_compare_k<<<(n + 127) / 128, 128>>>(uvx, out, n);
    CUDA_OK(cudaDeviceSynchronize());
    int bad = 0; double worst_lp = 0, worst_ent = 0, worst_g = 0;
    for (int j = 0; j < n; ++j) {
        const double* o = out + 14 * (size_t)j; const double c = o[12], lpb = 2e-5 + 1e-7 * c + 2e-7 * o[13];
        double elp = std::abs(o[0] - o[1]), eent = std::abs(o[2] - o[3]);
        bool ok = std::isfinite(o[0]) && elp <= lpb && eent <= 1e-5 + 2e-6 * c;
        for (int k = 0; k < 4; ++k) {
            double e = std::abs(o[4+k] - o[8+k]);
            ok = ok && std::isfinite(o[4+k]) && e <= 1e-5 + 1e-4 * std::abs(o[8+k]) + 2e-7 * c;
            worst_g = std::max(worst_g, e / (1e-5 + 1e-4 * std::abs(o[8+k]) + 2e-7 * c));
        }
        worst_lp = std::max(worst_lp, elp / lpb);
        worst_ent = std::max(worst_ent, eent / (1e-5 + 2e-6 * c));
        *checks += 1;
        if (!ok) {
            if (bad < 10) printf("MISMATCH beta head u=%g v=%g x=%g a+b=%g lp %.9g/%.9g ent %.9g/%.9g g %.6g/%.6g %.6g/%.6g %.6g/%.6g %.6g/%.6g\n",
                uvx[3*j], uvx[3*j+1], uvx[3*j+2], c, o[0], o[1], o[2], o[3], o[4], o[8], o[5], o[9], o[6], o[10], o[7], o[11]);
            ++bad;
        }
    }
    printf("%s beta mixed-precision head vs double, floor %g (%d points; worst error / bound: lp %.3f ent %.3f grad %.3f)\n",
           bad ? "FAIL" : "PASS", (double)floor, n, worst_lp, worst_ent, worst_g);
    const float one = 1.0f;
    CUDA_OK(cudaMemcpyToSymbol(d_beta_floor, &one, sizeof(one)));
    return bad ? 1 : 0;
}
// The sampler's draw: mean, variance and P(x < 0.25) against the analytic mean and variance and
// a numerical CDF, within 5 standard errors, over 2^20 draws per shape pair.
__global__ void beta_sample_k(float a, float b, float* out, int per, unsigned long long seed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t st; curand_init(seed, t, 0, &st);
    for (int k = 0; k < per; ++k) out[(size_t)t * per + k] = beta_draw(&st, a, b);
}
static int beta_sampler_check(int* checks) {
    // The last five have a shape below 1 (policy.beta_floor < 1): the boosted gamma draw.
    const double shapes[][2] = {{1,1},{1.6931471805599454,1.6931471805599454},{5,1.3},{1,50},{300,2},{2e4,3e4},
                                {0.5,0.5},{1.1931471805599454,1.1931471805599454},{0.6,3},{2,0.7},{0.5,40}};
    const int threads = 4096, per = 256; const long n = (long)threads * per;
    float* out = managed<float>(n);
    int bad = 0;
    for (auto& sh : shapes) {
        double a = (float)sh[0], b = (float)sh[1];   // the shapes the float sampler actually draws
        beta_sample_k<<<threads / 256, 256>>>((float)a, (float)b, out, per, 1234ULL);
        CUDA_OK(cudaDeviceSynchronize());
        double m = 0, m2 = 0, below = 0;
        for (long i = 0; i < n; ++i) { double v = out[i]; m += v; m2 += v * v; below += v < 0.25; if (!(v > 0 && v < 1)) ++bad; }
        m /= n; double var = m2 / n - m * m; below /= n;
        double mu = a / (a + b), s2 = a * b / ((a + b) * (a + b) * (a + b + 1));
        // Var((x-mu)^2) = var^2 (2 + excess kurtosis), with the Beta's exact excess kurtosis.
        const double ex = 6 * ((a - b) * (a - b) * (a + b + 1) - a * b * (a + b + 2))
                          / (a * b * (a + b + 2) * (a + b + 3));
        double k4 = s2 * s2 * (2 + ex);
        const double cdf = beta_integral(a, b, 0.25, [](double) { return 1.0; });
        bool ok = std::abs(m - mu) < 5 * std::sqrt(s2 / n) + 1e-7
               && std::abs(var - s2) < 5 * std::sqrt(k4 / n) + 1e-9
               && std::abs(below - cdf) < 5 * std::sqrt(std::max(cdf * (1 - cdf), 1e-12) / n) + 1e-6;
        *checks += 3;
        printf("%s beta sampler a=%g b=%g mean %.6g/%.6g var %.6g/%.6g P(x<.25) %.6g/%.6g\n",
               ok ? "PASS" : "FAIL", a, b, m, mu, var, s2, below, cdf);
        if (!ok) ++bad;
    }
    return bad ? 1 : 0;
}
// policy.beta_cn_tau: the AR(1) update of pufferl.cu sample_logits driving beta_draw_cn. Each thread
// runs `steps` coherent draws from a stationary start; the last draw of every thread is one iid sample
// of the per-step marginal, which must still be Beta(a, b); the last two draws give the lag-1
// correlation the coherence is for.
__global__ void beta_cn_k(float a, float b, float rho, int steps, float* last, float* prev,
                          unsigned long long seed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t st; curand_init(seed, t, 0, &st);
    float z1 = curand_normal(&st), z2 = curand_normal(&st), x = 0.0f, xp = 0.0f;
    const float s = sqrtf(1.0f - rho * rho);
    for (int k = 0; k < steps; ++k) {
        z1 = rho * z1 + s * curand_normal(&st);
        z2 = rho * z2 + s * curand_normal(&st);
        xp = x; x = beta_draw_cn(&st, a, b, z1, z2);
    }
    last[t] = x; prev[t] = xp;
}
static int beta_cn_check(int* checks) {
    const double shapes[][2] = {{1,1},{1.6931471805599454,1.6931471805599454},{5,1.3},{1,50},{300,2},{0.6,3}};
    const int n = 262144, steps = 48;
    const float rho = expf(-1.0f / 12.0f);
    float* last = managed<float>(n); float* prev = managed<float>(n);
    int bad = 0;
    for (auto& sh : shapes) {
        double a = (float)sh[0], b = (float)sh[1];
        beta_cn_k<<<n / 256, 256>>>((float)a, (float)b, rho, steps, last, prev, 4321ULL);
        CUDA_OK(cudaDeviceSynchronize());
        double m = 0, m2 = 0, below = 0, mp = 0, mp2 = 0, cxy = 0;
        for (long i = 0; i < n; ++i) {
            double v = last[i], w = prev[i];
            m += v; m2 += v * v; below += v < 0.25; mp += w; mp2 += w * w; cxy += v * w;
            if (!(v > 0 && v < 1)) ++bad;
        }
        m /= n; double var = m2 / n - m * m; below /= n; mp /= n;
        double corr = (cxy / n - m * mp) / std::sqrt(var * (mp2 / n - mp * mp));
        double mu = a / (a + b), s2 = a * b / ((a + b) * (a + b) * (a + b + 1));
        const double ex = 6 * ((a - b) * (a - b) * (a + b + 1) - a * b * (a + b + 2))
                          / (a * b * (a + b + 2) * (a + b + 3));
        double k4 = s2 * s2 * (2 + ex);
        const double cdf = beta_integral(a, b, 0.25, [](double) { return 1.0; });
        bool ok = std::abs(m - mu) < 5 * std::sqrt(s2 / n) + 1e-7
               && std::abs(var - s2) < 5 * std::sqrt(k4 / n) + 1e-9
               && std::abs(below - cdf) < 5 * std::sqrt(std::max(cdf * (1 - cdf), 1e-12) / n) + 1e-6
               && (std::min(a, b) < 1 || corr > 0.5);  // below 1 the boost's fresh uniform dilutes it
        *checks += 4;
        printf("%s coherent beta sampler a=%g b=%g mean %.6g/%.6g var %.6g/%.6g P(x<.25) %.6g/%.6g lag-1 corr %.3f\n",
               ok ? "PASS" : "FAIL", a, b, m, mu, var, s2, below, cdf, corr);
        if (!ok) ++bad;
    }
    return bad ? 1 : 0;
}
#if HAVE_BINARY_RUN
// policy.binary_run: c0 (continued fraction) against the numerical Beta integral, then the coherent
// draw on iff Phi(z1) < p: its per-step marginal must be p and a switch must hold longer than iid.
static int binary_c0_check(int* checks) {
    int bad = 0;
    for (double fl : {1.0, 0.5, 0.25, 3.0}) {
        const double s = fl + std::log(2.0);
        const double q = 1.0 - beta_integral(s, s, 0.25, [](double) { return 1.0; });
        const double want = std::log(q / (1.0 - q)), got = binary_run_c0(fl);
        ++*checks;
        const bool ok = std::abs(got - want) < 1e-6;
        printf("%s binary_run c0 floor %g: %.9f / %.9f (P(on) at u = 0 %.6f)\n", ok ? "PASS" : "FAIL", fl, got, want, q);
        if (!ok) ++bad;
    }
    return bad;
}
__global__ void binary_cn_k(float p, float rho, int steps, float* last, float* flips, unsigned long long seed) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t st; curand_init(seed, t, 0, &st);
    float z1 = curand_normal(&st), z2 = curand_normal(&st), on = 0.0f, prev = 0.0f, n = 0.0f;
    const float s = sqrtf(1.0f - rho * rho);
    for (int k = 0; k < steps; ++k) {
        z1 = rho * z1 + s * curand_normal(&st);
        z2 = rho * z2 + s * curand_normal(&st);
        prev = on; on = normcdff(z1) < p ? 1.0f : 0.0f;
        if (k > 0) n += on != prev;
    }
    last[t] = on; flips[t] = n / (steps - 1);
}
static int binary_cn_check(int* checks) {
    const int n = 262144, steps = 48;
    const float rho = expf(-1.0f / 12.0f);
    float* last = managed<float>(n); float* flips = managed<float>(n);
    int bad = 0;
    for (double p : {0.8211, 0.5, 0.05, 0.97}) {
        binary_cn_k<<<n / 256, 256>>>((float)p, rho, steps, last, flips, 999ULL);
        CUDA_OK(cudaDeviceSynchronize());
        double m = 0, f = 0;
        for (int i = 0; i < n; ++i) { m += last[i]; f += flips[i]; }
        m /= n; f /= n;
        const bool ok = std::abs(m - p) < 5 * std::sqrt(p * (1 - p) / n) && f < 0.5 * 2 * p * (1 - p);
        *checks += 2;
        printf("%s coherent binary draw p=%g P(on) %.6f switch rate %.4f (iid %.4f)\n",
               ok ? "PASS" : "FAIL", p, m, f, 2 * p * (1 - p));
        if (!ok) ++bad;
    }
    return bad ? 1 : 0;
}
#endif
static double head_beta_lp(double u,double v,double action) {
    double a=softplus1(u),b=softplus1(v);
    return (a-1)*std::log(action)+(b-1)*std::log1p(-action)
        -std::lgamma(a)-std::lgamma(b)+std::lgamma(a+b);
}
static double head_objective(const double* x,const float* old,const float* acts,
                             const float* mask,const float* zone,double building,int local) {
    double v=0;
    for(int h=0;h<2;++h) if(mask[h]) {
        double ratio=std::exp(head_beta_lp(x[h],x[5+h],acts[h])-old[h]);
        double adv=local?zone[0]:building;
        v+=std::max(-adv*ratio,-adv*bounded(ratio,.8,1.2));
    }
    return v;
}
int main() {
    Case cases[] = {
      {"masked_padding",0,0,1.2,0,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,0,0}},
      {"normal_mixed",0,0,1.2,0,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,1,1}},
      {"tanh_mean",1,0,1.2,0,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,1,1}},
      {"squash_interior",2,0,1.2,0,{.4,-.6,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,1,1}},
      {"squash_saturated",2,0,1.2,0,{18,-18,.3,-.2,.15},{-.7,-.9,0,0},{2.6,-2.5,1},{1,1,1,1}},
      {"geometric_ratio",0,1,1.2,.05,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,1,1}},
      {"positive_clipped",0,0,1.2,.45,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,1,1}},
      {"negative_unclipped",0,0,-1.2,.45,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,1,1}},
      {"negative_clipped",0,0,-1.2,-.45,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,1,1,1}},
      {"partial_discrete_mask",0,1,1.2,.05,{.2,-.4,.3,-.2,.15},{-.7,-.9,0,0},{-.1,.2,1},{1,0,0,1}},
      // policy.beta: logits are u, the logstd slots carry v (the second block), actions are x in (0, 1).
      {"beta_interior",3,0,1.2,0,{.4,-.6,.3,-.2,.15},{.2,.5,0,0},{.3,.7,1},{1,1,1,1}},
      {"beta_concentrated",3,0,1.2,0,{4,-2,.3,-.2,.15},{-1,5,0,0},{.9,.05,1},{1,1,1,1}},
      {"beta_near_rail",3,0,-1.2,0,{6,0,.3,-.2,.15},{0,7,0,0},{.999,.002,1},{1,1,1,1}},
      {"beta_geometric_ratio",3,1,1.2,.05,{.4,-.6,.3,-.2,.15},{.2,.5,0,0},{.3,.7,1},{1,1,1,1}},
      {"beta_positive_clipped",3,0,1.2,.45,{.4,-.6,.3,-.2,.15},{.2,.5,0,0},{.3,.7,1},{1,1,1,1}},
      {"beta_masked_head",3,1,-1.2,.05,{.4,-.6,.3,-.2,.15},{.2,.5,0,0},{.3,.7,1},{1,0,1,1}},
      // policy.beta_floor = 0.5: u = -0.43 is the uniform; u, v < -0.43 put a shape below 1.
      {"beta_floor05_uniform",3,0,1.2,0,{-.4328,-.4328,.3,-.2,.15},{-.4328,-.4328,0,0},{.3,.7,1},{1,1,1,1},.5},
      {"beta_floor05_u_shaped",3,0,-1.2,0,{-2,-1,.3,-.2,.15},{-3,-1.5,0,0},{.02,.97,1},{1,1,1,1},.5},
      {"beta_floor05_mixed",3,0,1.2,.05,{1.5,-2.5,.3,-.2,.15},{-2,2,0,0},{.85,.01,1},{1,1,1,1},.5},
      {"beta_floor05_clipped",3,1,1.2,.45,{-1,.8,.3,-.2,.15},{.3,-1.2,0,0},{.4,.6,1},{1,1,1,1},.5},
      {"beta_mean_precision_small_shape",5,0,1.2,0,{-2,.8,.3,-.2,.15},{-1,.2,0,0},{.05,.6,1},{1,1,1,1}},
      {"beta_mean_precision_geometric",5,1,-1.2,.05,{-1.5,.7,.3,-.2,.15},{.1,-.5,0,0},{.2,.7,1},{1,0,1,1}},
      {"beta_mean_precision_clipped",5,1,1.2,.45,{-.7,.6,.3,-.2,.15},{-.4,.3,0,0},{.15,.8,1},{1,1,1,1}},
#if HAVE_BINARY_RUN
      // policy.binary_run: head 0 Beta, head 1 Bernoulli (action 1 on / 0 off); its v (logstd[1]) is unused.
      {"binary_on",4,0,1.2,0,{.4,.3,.3,-.2,.15},{.2,.5,0,0},{.3,1,1},{1,1,1,1}},
      {"binary_off_clipped",4,0,1.2,.45,{.4,-1.5,.3,-.2,.15},{.2,-.3,0,0},{.3,0,1},{1,1,1,1}},
      {"binary_negative_geometric",4,1,-1.2,.05,{-.6,2.5,.3,-.2,.15},{.5,.1,0,0},{.7,0,1},{1,1,1,1}},
      {"binary_saturated_on",4,0,-1.2,0,{.4,9,.3,-.2,.15},{.2,.5,0,0},{.3,1,1},{1,1,1,1}},
      {"binary_masked",4,1,1.2,.05,{.4,.3,.3,-.2,.15},{.2,.5,0,0},{.3,1,1},{1,0,1,1}},
      {"binary_floor05",4,0,1.2,0,{-1,-.8,.3,-.2,.15},{-.5,.4,0,0},{.2,0,1},{1,1,1,1},.5},
#endif
    };
    float *x=managed<float>(10),*ls=managed<float>(4),*acts=managed<float>(3);
    float *mask=managed<float>(4),*oldlp=managed<float>(1),*adv=managed<float>(1);
    float *values=managed<float>(1),*ret=managed<float>(1),*imp=managed<float>(1);
    float *cache=managed<float>(4),*newlp=managed<float>(1),*livev=managed<float>(1);
    float *gg=managed<float>(4),*gs=managed<float>(4),*gv=managed<float>(1);
    float *entcoef=managed<float>(3),*parts=managed<float>(LOSS_N);
    int *sizes=managed<int>(3);sizes[0]=sizes[1]=1;sizes[2]=2;
    entcoef[0]=entcoef[1]=entcoef[2]=.01f;*values=.1f;*ret=-.5f;
    Prec dec={};dec.data=x;dec.shape[0]=dec.shape[1]=1;dec.shape[2]=5;
    Prec logstd={};logstd.data=ls;logstd.shape[0]=4;
    int failed=0, checks=0;
    for (const auto& c: cases) {
        for(int j=0;j<10;++j)x[j]=0.0f;
        for(int j=0;j<5;++j)x[j]=c.logits[j];
        if(c.mode>=3)for(int j=0;j<2;++j)x[5+j]=c.logstd[j];
        for(int j=0;j<4;++j){ls[j]=c.logstd[j];mask[j]=c.mask[j];}
        for(int j=0;j<3;++j)acts[j]=c.actions[j];
        *adv=c.advantage;
        int tm=c.mode==1, ts=c.mode==2, bt=c.mode>=3;
        int mp=c.mode==5; CUDA_OK(cudaMemcpyToSymbol(d_beta_mean_precision,&mp,sizeof(mp)));
        g_floor=c.floor>0?c.floor:1.0; float ff=(float)g_floor;
        CUDA_OK(cudaMemcpyToSymbol(d_beta_floor,&ff,sizeof(ff)));
#if HAVE_BINARY_RUN
        {
            unsigned char slots[BINARY_RUN_SLOTS]={0}; slots[1]=c.mode==4;
            g_c0=c.mode==4?binary_run_c0(g_floor):0.0; float c0=(float)g_c0;
            CUDA_OK(cudaMemcpyToSymbol(d_binary_run_slot,slots,sizeof(slots)));
            CUDA_OK(cudaMemcpyToSymbol(d_binary_c0,&c0,sizeof(c0)));
        }
#endif
        CUDA_OK(cudaMemcpyToSymbol(d_beta,&bt,sizeof(bt)));
        CUDA_OK(cudaMemcpyToSymbol(d_tanh_mean,&tm,sizeof(tm)));
        CUDA_OK(cudaMemcpyToSymbol(d_tanh_squash,&ts,sizeof(ts)));
        cache_imp_and_v<<<1,256>>>(dec,acts,oldlp,mask,logstd,sizes,imp,livev,cache,newlp);
        CUDA_OK(cudaDeviceSynchronize());
        double ref_lp,ref_ent;int active;
        reference_dist(c,c.logits,c.logstd,&ref_lp,&ref_ent,&active);
        double offset=*newlp-ref_lp;
        *oldlp=*newlp-c.shift*(c.per_dim?active:1);
        // At unchanged policy, recomputing the likelihood must reproduce the
        // rollout value, including padding decisions, before any optimizer step.
        cache_imp_and_v<<<1,256>>>(dec,acts,oldlp,mask,logstd,sizes,imp,livev,cache,newlp);
        CUDA_OK(cudaDeviceSynchronize());
        // The production path reuses these buffers: cached class log-probs and
        // new joint log-prob are overwritten in-place by the loss gradients.
        std::copy(cache,cache+4,gg); *gv=*newlp;
        PPOGraphArgs g={imp,acts,oldlp,adv,values,ret};
        PPOKernelArgs a={gg,gs,gv,x,ls,x+4,sizes,mask,3,.2f,10.f,.5f,entcoef,nullptr,1,4,1,true,c.per_dim};
        ppo_loss_compute<<<1,PPO_THREADS>>>(parts,a,g);
        CUDA_OK(cudaDeviceSynchronize());
        bool pass=true;
        for(int j=0;j<5;++j){
            double xp[5],xm[5];std::copy(c.logits,c.logits+5,xp);std::copy(c.logits,c.logits+5,xm);
            double eps=1e-5;xp[j]+=eps;xm[j]-=eps;
            double ref=(objective(c,xp,c.logstd,*oldlp,offset)-objective(c,xm,c.logstd,*oldlp,offset))/(2*eps);
            double got=j<4?gg[j]:gv[0];
            ++checks;
            if(!std::isfinite(got)||std::abs(got-ref)>3e-4+1e-3*std::abs(ref)){
                printf("MISMATCH %s logit[%d] got %.9g ref %.9g\n",c.name,j,got,ref);pass=false;
            }
        }
        for(int j=0;j<2;++j){
            double sp[4],sm[4];std::copy(c.logstd,c.logstd+4,sp);std::copy(c.logstd,c.logstd+4,sm);
            double eps=1e-5;sp[j]+=eps;sm[j]-=eps;
            double ref=(objective(c,c.logits,sp,*oldlp,offset)-objective(c,c.logits,sm,*oldlp,offset))/(2*eps);
            ++checks;
            if(!std::isfinite(gs[j])||std::abs(gs[j]-ref)>3e-4+1e-3*std::abs(ref)){
                printf("MISMATCH %s logstd[%d] got %.9g ref %.9g\n",c.name,j,(double)gs[j],ref);pass=false;
            }
        }
        ++checks;
        if(std::abs(parts[LOSS_ENT]-ref_ent)>3e-4){
            printf("MISMATCH %s active entropy got %.9g ref %.9g\n",c.name,(double)parts[LOSS_ENT],ref_ent);pass=false;
        }
        // The approved zone-credit arm must produce the ordinary building-advantage
        // gradient on both Beta valves even when an opposing zone advantage exists.
        // Check the actual PPO kernel, including masking and geometric-ratio clipping.
        if (strcmp(c.name,"beta_geometric_ratio")==0 || strcmp(c.name,"beta_positive_clipped")==0) {
            float ref_g[4],ref_s[4],ref_v=gv[0];
            std::copy(gg,gg+4,ref_g); std::copy(gs,gs+4,ref_s);
            float* za=managed<float>(1); za[0]=-3.0f;
            a.zone_adv=za; a.zone_slots=1; a.zone_stride=1; a.valve_zone=1; a.valve_global=1;
            std::copy(cache,cache+4,gg); *gv=*newlp;
            ppo_loss_compute<<<1,PPO_THREADS>>>(parts,a,g); CUDA_OK(cudaDeviceSynchronize());
            for(int j=0;j<4;++j) {
                ++checks;
                if(!std::isfinite(gg[j]) || std::abs(gg[j]-ref_g[j])>1e-6f) pass=false;
            }
            for(int j=0;j<2;++j) {
                ++checks;
                if(!std::isfinite(gs[j]) || std::abs(gs[j]-ref_s[j])>1e-6f) pass=false;
            }
            ++checks;
            if(!std::isfinite(gv[0]) || std::abs(gv[0]-ref_v)>1e-6f) pass=false;
            if(strcmp(c.name,"beta_geometric_ratio")==0) {
                a.valve_global=0;
                std::copy(cache,cache+4,gg); *gv=*newlp;
                ppo_loss_compute<<<1,PPO_THREADS>>>(parts,a,g); CUDA_OK(cudaDeviceSynchronize());
                ++checks;
                // Non-vacuous: the old local-credit path must differ with opposite advantages.
                if(std::abs(gg[0]-ref_g[0])<1e-4f && std::abs(gs[0]-ref_s[0])<1e-4f) pass=false;
                ++checks;
                // Non-zone heads must not change when the zone signal changes.
                if(std::abs(gg[2]-ref_g[2])>1e-6f || std::abs(gg[3]-ref_g[3])>1e-6f) pass=false;
            }
            CUDA_OK(cudaFree(za));
        }
        printf("%s %s\n",pass?"PASS":"FAIL",c.name);
        if(!pass)++failed;
    }
    // Actual per-head PPO kernel, checked against independent finite differences.
    // Include cancellation, both advantage signs, masks, and the local valve path.
    {
        int bt=1; CUDA_OK(cudaMemcpyToSymbol(d_beta,&bt,sizeof(bt)));
        int mp=0; CUDA_OK(cudaMemcpyToSymbol(d_beta_mean_precision,&mp,sizeof(mp)));
        float floor=1;CUDA_OK(cudaMemcpyToSymbol(d_beta_floor,&floor,sizeof(floor)));g_floor=1;
#if HAVE_BINARY_RUN
        unsigned char slots[BINARY_RUN_SLOTS]={0}; CUDA_OK(cudaMemcpyToSymbol(d_binary_run_slot,slots,sizeof(slots)));
#endif
        float *old=managed<float>(3),*zone=managed<float>(1);
        double* diag=managed<double>(27);
        for(int test=0;test<8;++test) {
            const double xx[10]={.4,-.6,.3,-.2,.15,.2,.5,0,0,0};
            for(int j=0;j<10;++j)x[j]=xx[j];
            acts[0]=.3f;acts[1]=.7f;acts[2]=0;
            mask[0]=1;mask[1]=test==6?0:1;mask[2]=mask[3]=0;
            *adv=(test%2)?-1.2f:1.2f; zone[0]=(test%2)?1.2f:-1.2f;
            bool local=test>=4; float shifts[2]={0,0};
            if(test==2||test==3||test==4||test==5||test==6){shifts[0]=.4f;shifts[1]=-.4f;}
            if(test==7){shifts[0]=-.4f;shifts[1]=.4f;}
            for(int j=0;j<2;++j)old[j]=(float)head_beta_lp(xx[j],xx[5+j],acts[j])-shifts[j];
            *oldlp=(mask[0]?old[0]:0)+(mask[1]?old[1]:0);
            entcoef[0]=entcoef[1]=entcoef[2]=0;
            cache_imp_and_v<<<1,256>>>(dec,acts,oldlp,mask,logstd,sizes,imp,livev,cache,newlp);
            CUDA_OK(cudaDeviceSynchronize());
            std::copy(cache,cache+4,gg); *gv=*newlp;
            PPOGraphArgs g={imp,acts,oldlp,adv,values,ret,old,diag};
            PPOKernelArgs a={gg,gs,gv,x,ls,x+4,sizes,mask,3,.2f,10.f,0.f,entcoef,nullptr,1,4,1,true,1};
            if(local){a.zone_adv=zone;a.zone_slots=1;a.zone_stride=1;a.valve_zone=1;}
            ppo_loss_compute<<<1,PPO_THREADS>>>(parts,a,g);CUDA_OK(cudaDeviceSynchronize());
            bool pass=true;
            for(int h=0;h<2;++h)for(int v=0;v<2;++v){
                int j=v?5+h:h;double xp[10],xm[10];std::copy(xx,xx+10,xp);std::copy(xx,xx+10,xm);
                double eps=1e-5;xp[j]+=eps;xm[j]-=eps;
                double ref=(head_objective(xp,old,acts,mask,zone,*adv,local)-head_objective(xm,old,acts,mask,zone,*adv,local))/(2*eps);
                double got=v?gs[h]:gg[h];++checks;
                if(!std::isfinite(got)||std::abs(got-ref)>4e-4+2e-3*std::abs(ref)) {
                    printf("MISMATCH headwise test %d head %d v %d got %.9g ref %.9g\n",test,h,v,got,ref);pass=false;
                }
            }
            ++checks;
            double expected=head_objective(xx,old,acts,mask,zone,*adv,local);
            if(std::abs(parts[LOSS_PG]-expected)>5e-4)pass=false;
            ++checks;
            if(diag[0]!=(mask[0]+mask[1]) || diag[6]!=0)pass=false;
            if(test<2 && diag[1]>1e-4)pass=false; // unchanged policy
            if((test==2||test==3||test==4||test==5) && diag[5]<.5)pass=false;
            printf("%s headwise_case_%d\n",pass?"PASS":"FAIL",test);if(!pass)++failed;
        }
        CUDA_OK(cudaFree(old));CUDA_OK(cudaFree(zone));CUDA_OK(cudaFree(diag));
    }
    {
        int bt=0; CUDA_OK(cudaMemcpyToSymbol(d_beta,&bt,sizeof(bt)));
        int mp=0; CUDA_OK(cudaMemcpyToSymbol(d_beta_mean_precision,&mp,sizeof(mp)));
        float one=1.0f; CUDA_OK(cudaMemcpyToSymbol(d_beta_floor,&one,sizeof(one))); g_floor=1.0;
#if HAVE_BINARY_RUN
        unsigned char slots[BINARY_RUN_SLOTS]={0}; float c0=0.0f; g_c0=0.0;
        CUDA_OK(cudaMemcpyToSymbol(d_binary_run_slot,slots,sizeof(slots)));
        CUDA_OK(cudaMemcpyToSymbol(d_binary_c0,&c0,sizeof(c0)));
#endif
    }
#if HAVE_BINARY_RUN
    failed += binary_c0_check(&checks);
    failed += binary_cn_check(&checks);
#endif
    failed += beta_special_check(&checks);
    failed += beta_special_f_check(&checks);
    failed += beta_head_compare_check(&checks, 1.0f);
    failed += beta_head_compare_check(&checks, 0.5f);
    failed += beta_sampler_check(&checks);
    failed += beta_cn_check(&checks);
    printf("PPO MATH CHECK: %d scalar checks, %d failing cases\n",checks,failed);
    return failed?1:0;
}
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(PL_ROOT + "/plverify"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--canonical", action="store_true", help="Read source reconstructed from the canonical patch")
    args = parser.parse_args()
    if args.canonical:
        from puffer_patch_source import patched_source
        algo = patched_source("src/algo.cu")
        puffer = patched_source("src/pufferl.cu")
    else:
        algo = (args.checkout / "src/algo.cu").read_text()
        puffer = (args.checkout / "src/pufferl.cu").read_text()
    generated = """#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <curand_kernel.h>
#include "sb_r110_conditional_head.cuh"
using precision_t = float;
#define to_float(x) (x)
#define from_float(x) (x)
#define NUM_ATNS 3
struct Prec { float* data; long shape[8]; };
constexpr int PPO_THREADS=256, PPO_MAX_HEAD_A=4;
__device__ int d_tanh_mean=0, d_tanh_squash=0;
__device__ float d_squash_chain_floor=0;
__device__ int d_freeze_logstd=0;
__device__ int d_beta=0;
__device__ float d_beta_floor=1.0f;
__device__ float d_beta_conc_max=0.0f;   // r94 cap; referenced by the patched beta_continuous_head; 0 (off) here, so the harness cases stay uncapped
__device__ int d_beta_mean_precision=0;
#define CUDA_OK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"CUDA: %s\\n",cudaGetErrorString(e));exit(2); } } while(0)
constexpr int BINARY_RUN_SLOTS=512;
__device__ unsigned char d_binary_run_slot[BINARY_RUN_SLOTS];
__device__ float d_binary_c0=0.0f;
"""
    generated += "#define HAVE_BINARY_RUN %d\n" % int("binary_run_head(" in algo)
    generated += definition(puffer, "block_reduce_sum")
    generated += definition(algo, "LossIdx", "enum")
    generated += definition(algo, "PPOGraphArgs", "struct")
    generated += definition(algo, "PPOKernelArgs", "struct")
    names = ["finite_or_clamp", "squash_location", "safe_continuous_mean", "safe_continuous_logstd",
             "ent_head_role", "ppo_head_active", "load_logit_masked", "ppo_discrete_logsumexp", "squash_executed_logdet",
             "ppo_continuous_head", "beta_digamma", "beta_trigamma", "beta_concentration",
             "beta_sigmoid", "beta_clamp_x", "beta_digamma_f", "beta_trigamma_f", "beta_concentration_f",
             "beta_sigmoid_f", "beta_continuous_head", "binary_run_slot", "binary_run_prob_f",
             "binary_run_head", "binary_run_ibeta", "binary_run_c0", "beta_gamma_draw", "beta_draw",
             "beta_gamma_draw_cn", "beta_draw_cn",
             "cache_imp_and_v", "plant_temp_head", "entropy_continuous_head", "ppo_loss_compute"]
    for line in algo.splitlines():
        if line.startswith("constexpr float BETA_X_"):
            generated += line + "\n"
    for name in names:
        if name + "(" in algo:
            generated += definition(algo, name)
    generated += HARNESS
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "check.cu").write_text(generated)
    (out / "source_sha256.txt").write_text(hashlib.sha256(algo.encode()).hexdigest() + "\n")
    relative = out.relative_to(CP)
    prefix = ["docker", "run", "--rm", "--gpus", f"device={args.gpu}", "-v", f"{CP}:/cp",
              "-w", "/cp", "pufferai/puffertank:5.0"]
    subprocess.run(prefix + ["nvcc", "-O2", "-std=c++17", "-arch=sm_86", "-I/cp/patches/src",
                             f"/cp/{relative}/check.cu", "-o", f"/cp/{relative}/check"], check=True)
    completed = subprocess.run(prefix + [f"/cp/{relative}/check"], text=True, capture_output=True)
    (out / "result.txt").write_text(completed.stdout + completed.stderr)
    print(completed.stdout, end="")
    print(completed.stderr, end="")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
