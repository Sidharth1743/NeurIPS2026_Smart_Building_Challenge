// SAC/env/sb_env.cu — standalone rollout driver for the Smart Buildings GPU env.
//
// Owned by SAC/. It *includes* the read-only training-path source at
// complete_port/gpu/smart_buildings.cu (the same translation unit the PufferLib
// trainer compiles) and adds only a main() plus a shared-memory / socket
// protocol. Nothing under complete_port is edited.
//
// Protocol (one control UNIX socket, one shm data file):
//   shm layout:  Hdr | actions n*ACT_STRIDE f32 | obs n*OBS_STRIDE f32 | rew n f32 | term n f32
//   opcode 'S': Python has written actions; driver steps; writes obs/rew/term; replies 1 byte 0
//   opcode 'L': live log; driver runs puf_live_log (+ per-env Log means); replies u32 len + text
//   opcode 'P': per-step reward parts of the step just taken (SB_RECORD companion: the
//               same eight doubles the trainer's recorder writes); replies u32 count + f64[]
//   opcode 'R': host reset; replies 1 byte 0
//   opcode 'Q': exit
//
// Build: SAC/env/build.sh (puffertank:5.0, compute_86).

#include "../../complete_port/gpu/smart_buildings.cu"

// The r139 build is compiled with -DSB_SHARED_POLICY, which (a) makes the
// canonical observation carry 0 in the base-controller proposal columns instead
// of a pid_water value and (b) declares sb_set_feat (only called when
// env.zone_memory = 1, which this driver never sets). The stub keeps the link
// clean; it changes no observation value.
// With env.zone_memory = 1 the env allocates its per-zone memory features and
// hands us the device pointer here (sbs_local_feature reads them on the device in
// the trainer). We forward them to the learner over shm instead; the env code and
// the observation vector are untouched.
static float* s_zm_dev = NULL;
extern "C" void sb_set_feat(float* p) { s_zm_dev = p; }

#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctime>

// ---------------------------------------------------------------- protocol --

struct Hdr {
    uint64_t magic;      // 0x534143425245504fULL
    uint32_t version;    // 1
    uint32_t n_agents;
    uint32_t obs_dim;     // CN_OBS values are meaningful
    uint32_t obs_stride;  // SB_OBS row stride in the env buffer
    uint32_t act_dim;     // CN_ACT
    uint32_t act_stride;  // SB_CONT + SB_DISC
    uint32_t mask_width;  // SB_MASK_WIDTH
    uint32_t ep_len;
    uint64_t step;
    uint64_t pad;
};

static const uint64_t HDR_MAGIC = 0x534143425245504fULL;

static bool g_rn_frozen = false;
static double g_rn_mu0[SB_MAX_BLOBS], g_rn_var0[SB_MAX_BLOBS];

static const char* arg_get(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}
static double arg_num(int argc, char** argv, const char* name, double def) {
    const char* s = arg_get(argc, argv, name, NULL);
    return s ? atof(s) : def;
}
static int arg_int(int argc, char** argv, const char* name, int def) {
    const char* s = arg_get(argc, argv, name, NULL);
    return s ? atoi(s) : def;
}

static void die(const char* msg) {
    fprintf(stderr, "sb_env: %s\n", msg);
    exit(1);
}

static void* xmap(const char* path, size_t bytes) {
    int fd = open(path, O_RDWR | O_CREAT, 0644);
    if (fd < 0) die("cannot open shm file");
    if (ftruncate(fd, (off_t)bytes) != 0) die("ftruncate failed");
    if (fchmod(fd, 0666) != 0) die("fchmod shm failed");
    void* p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) die("mmap failed");
    close(fd);
    return p;
}

static void send_all(int fd, const void* buf, size_t n) {
    const char* p = (const char*)buf;
    while (n > 0) {
        ssize_t k = write(fd, p, n);
        if (k <= 0) die("control write failed");
        p += k;
        n -= (size_t)k;
    }
}
static void recv_all(int fd, void* buf, size_t n) {
    char* p = (char*)buf;
    while (n > 0) {
        ssize_t k = read(fd, p, n);
        if (k <= 0) die("control read failed (peer closed)");
        p += k;
        n -= (size_t)k;
    }
}

// ---------------------------------------------------------------- main ------

int main(int argc, char** argv) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);

    const char* blob_dir = arg_get(argc, argv, "--blob_dir", NULL);
    const char* blob = arg_get(argc, argv, "--blob", NULL);
    const int groups = arg_int(argc, argv, "--groups", 4);
    const int agents = arg_int(argc, argv, "--agents", 256);
    const char* shm_path = arg_get(argc, argv, "--shm", NULL);
    const char* ctrl_path = arg_get(argc, argv, "--ctrl", NULL);
    const char* header_path = arg_get(argc, argv, "--header", NULL);
    const int graph = arg_int(argc, argv, "--graph", 1);
    const int desync = arg_int(argc, argv, "--desync", 1);
    const int rew_norm = arg_int(argc, argv, "--rew_norm", 1);
    const int max_steps = arg_int(argc, argv, "--max_steps", 0);
    const char* eval_out = arg_get(argc, argv, "--eval_out", NULL);
    // Freeze the per-scenario reward normalizer after this many vector steps:
    // puf_live_log() updates the host EMA and re-uploads the device mirror, so we
    // snapshot the host state on the first call past the threshold and restore it
    // (and re-push the mirror) on every later call. Metrics keep flowing.
    const int freeze_rn_at = arg_int(argc, argv, "--freeze_rew_norm_at", 0);
    const int zone_memory = arg_int(argc, argv, "--zone_memory", 0);
    if (!shm_path || !ctrl_path || !header_path) die("--shm --ctrl --header are required");

    if (eval_out) setenv("SB_EVAL_OUT", eval_out, 1);

    // ---- env kwargs (the same keys config/smart_buildings.ini exposes) ----
    Dict kw = {0};
    dict_set_str(&kw, "blob", blob ? blob : "");
    dict_set_str(&kw, "blob_dir", blob_dir ? blob_dir : "");
    dict_set(&kw, "groups", groups);
    dict_set(&kw, "merge_weathers", 1);
    dict_set(&kw, "max_steps", max_steps);
    dict_set(&kw, "desync", desync);
    dict_set(&kw, "random_start", 0);
    dict_set(&kw, "random_start_seed", 0);
    dict_set(&kw, "reward_scale", 1.0);
    dict_set(&kw, "rew_norm", rew_norm);
    dict_set(&kw, "rew_center", 0);
    dict_set(&kw, "canonical", 1);
    dict_set(&kw, "residual", 0);
    dict_set(&kw, "plant_pair", 0);
    dict_set(&kw, "comfort_shape", 0);
    dict_set(&kw, "guard_d1", 0);
    dict_set(&kw, "zone_command", 0);
    dict_set(&kw, "probe", 0);
    dict_set(&kw, "prestart_explore", 0);
    dict_set(&kw, "zone_memory", zone_memory);
    dict_set(&kw, "zone_adv", 0);
    dict_set(&kw, "zone_critic", 0);
    dict_set(&kw, "ahu_critic", 0);
    dict_set(&kw, "interleave", 0);
    dict_set(&kw, "zone_signed", 0);
    dict_set(&kw, "zone_night_share", 0);

    // ---- host buffers the env reads/writes every step (pinned) ----
    const size_t obytes = (size_t)agents * SB_OBS * sizeof(float);
    const size_t abytes = (size_t)agents * (SB_CONT + SB_DISC) * sizeof(float);
    const size_t rbytes = (size_t)agents * sizeof(float);
    float *h_obs = NULL, *h_act = NULL, *h_rew = NULL, *h_term = NULL;
    if (cudaHostAlloc((void**)&h_obs, obytes, cudaHostAllocPortable) != cudaSuccess) die("cudaHostAlloc obs");
    if (cudaHostAlloc((void**)&h_act, abytes, cudaHostAllocPortable) != cudaSuccess) die("cudaHostAlloc act");
    if (cudaHostAlloc((void**)&h_rew, rbytes, cudaHostAllocPortable) != cudaSuccess) die("cudaHostAlloc rew");
    if (cudaHostAlloc((void**)&h_term, rbytes, cudaHostAllocPortable) != cudaSuccess) die("cudaHostAlloc term");
    memset(h_act, 0, abytes);
    memset(h_rew, 0, rbytes);
    memset(h_term, 0, rbytes);

    cudaStream_t stream;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) die("cudaStreamCreate");

    Env* env = puf_vec_create(agents, &kw, (obs_t*)h_obs, h_act, h_rew, h_term);
    if (!env) die("puf_vec_create failed");
    puf_bind_stream(stream);

    // ---- mask (per agent; one row per building) ----
    unsigned char* d_mask = NULL;
    if (cudaMalloc(&d_mask, (size_t)agents * SB_MASK_WIDTH) != cudaSuccess) die("cudaMalloc mask");
    puf_mask_init(d_mask, SB_MASK_WIDTH, agents);
    std::vector<unsigned char> h_mask((size_t)agents * SB_MASK_WIDTH);
    if (cudaMemcpy(h_mask.data(), d_mask, h_mask.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        die("mask D2H");
    cudaFree(d_mask);

    // scenario id per agent = cluster of identical mask rows (one row per building)
    std::vector<int> scen(agents, 0), uniq_owner;
    {   // members are contiguous blocks of identical rows; keep the first row of each block
        int s = -1;
        for (int b = 0; b < agents; ++b) {
            const unsigned char* r = h_mask.data() + (size_t)b * SB_MASK_WIDTH;
            if (b == 0 || memcmp(r, r - SB_MASK_WIDTH, SB_MASK_WIDTH) != 0) { s = (int)uniq_owner.size(); uniq_owner.push_back(b); }
            scen[b] = s;
        }
    }

    puf_reset(env);
    if (cudaStreamSynchronize(stream) != cudaSuccess) die("reset sync");

    // ---- graph capture of one full vector step ----
    cudaGraphExec_t exec = NULL;
    if (graph) {
        cudaGraph_t g = NULL;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
            die("cudaStreamBeginCapture");
        puf_step(env);
        if (cudaStreamEndCapture(stream, &g) != cudaSuccess) die("cudaStreamEndCapture");
        if (cudaGraphInstantiate(&exec, g, 0) != cudaSuccess) die("cudaGraphInstantiate");
        cudaGraphDestroy(g);
        fprintf(stderr, "sb_env: one step captured in a CUDA graph\n");
    }

    // ---- shared data region ----
    const uint32_t OBS_DIM = CN_OBS, ACT_DIM = CN_ACT;
    const size_t n_a = (size_t)agents * (SB_CONT + SB_DISC);
    const size_t n_o = (size_t)agents * SB_OBS;
    const size_t n_r = (size_t)agents;
    const size_t n_z = (size_t)agents * CN_ZS * 11;   // env memory features, ZM_FEAT = 11
    const size_t hdr_bytes = (sizeof(Hdr) + 63) & ~(size_t)63;
    const size_t total = hdr_bytes + (n_a + n_o + 2 * n_r) * sizeof(float) + n_z * sizeof(float);
    unsigned char* shm = (unsigned char*)xmap(shm_path, total);
    memset(shm, 0, total);
    Hdr* hdr = (Hdr*)shm;
    hdr->magic = HDR_MAGIC;
    hdr->version = 1;
    hdr->n_agents = (uint32_t)agents;
    hdr->obs_dim = OBS_DIM;
    hdr->obs_stride = SB_OBS;
    hdr->act_dim = ACT_DIM;
    hdr->act_stride = SB_CONT + SB_DISC;
    hdr->mask_width = SB_MASK_WIDTH;
    hdr->ep_len = (uint32_t)(max_steps > 0 ? max_steps : sbgpu::sb_blob_ep_len());
    hdr->step = 0;
    float* s_act = (float*)(shm + hdr_bytes);
    float* s_obs = s_act + n_a;
    float* s_rew = s_obs + n_o;
    float* s_term = s_rew + n_r;
    float* s_zm = s_term + n_r;

    // ---- header file for the client ----
    {
        FILE* f = fopen(header_path, "w");
        if (!f) die("cannot write header");
        fprintf(f, "{\"n_agents\":%d,\"obs_dim\":%u,\"obs_stride\":%u,\"act_dim\":%u,\"act_stride\":%u,"
                   "\"mask_width\":%u,\"ep_len\":%d,\"zm_dim\":%d,\"masks\":[",
                agents, OBS_DIM, (unsigned)SB_OBS, ACT_DIM, (unsigned)(SB_CONT + SB_DISC),
                (unsigned)SB_MASK_WIDTH, (int)hdr->ep_len, zone_memory ? 11 : 0);
        for (size_t u = 0; u < uniq_owner.size(); ++u) {
            fprintf(f, "%s[", u ? "," : "");
            const unsigned char* r = h_mask.data() + (size_t)uniq_owner[u] * SB_MASK_WIDTH;
            for (int c = 0; c < SB_MASK_WIDTH; ++c) fprintf(f, "%s%d", c ? "," : "", (int)r[c]);
            fprintf(f, "]");
        }
        fprintf(f, "],\"scen\":[");
        for (int b = 0; b < agents; ++b) fprintf(f, "%s%d", b ? "," : "", scen[b]);
        fprintf(f, "]}\n");
        fclose(f);
    }
    fprintf(stderr, "sb_env: %d agents, obs %u/%u, act %u/%u, mask %u, ep_len %d, graph %d, rew_norm %d\n",
            agents, OBS_DIM, (unsigned)SB_OBS, ACT_DIM, (unsigned)(SB_CONT + SB_DISC),
            (unsigned)SB_MASK_WIDTH, (int)hdr->ep_len, graph, rew_norm);

    // ---- control socket ----
    int srv = socket(AF_UNIX, SOCK_STREAM, 0);
    if (srv < 0) die("socket");
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", ctrl_path);
    unlink(ctrl_path);
    if (bind(srv, (struct sockaddr*)&addr, sizeof(addr)) != 0) die("bind");
    if (chmod(ctrl_path, 0777) != 0) die("chmod socket failed");
    if (listen(srv, 1) != 0) die("listen");
    fprintf(stderr, "sb_env: ready on %s\n", ctrl_path);
    int fd = accept(srv, NULL, NULL);
    if (fd < 0) die("accept");
    close(srv);

    double t_last = 0.0;
    long n_steps = 0;
    for (;;) {
        unsigned char op = 0;
        recv_all(fd, &op, 1);
        if (op == 'Q') break;
        if (op == 'R') {
            puf_reset(env);
            if (cudaStreamSynchronize(stream) != cudaSuccess) die("reset sync");
            memcpy(s_obs, h_obs, n_o * sizeof(float));
            memcpy(s_rew, h_rew, n_r * sizeof(float));
            memcpy(s_term, h_term, n_r * sizeof(float));
            if (zone_memory && s_zm_dev != NULL)
                if (cudaMemcpy(s_zm, s_zm_dev, n_z * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                    die("zm D2H");
            unsigned char ok = 0;
            send_all(fd, &ok, 1);
            continue;
        }
        if (op == 'L') {
            Dict out = {0};
            char namebuf[64];
            snprintf(namebuf, sizeof(namebuf), "live");
            out.name = namebuf;
            puf_live_log(env, &out);
            if (freeze_rn_at > 0 && (int)hdr->step >= freeze_rn_at) {
                if (!g_rn_frozen) {
                    memcpy(g_rn_mu0, sbgpu::g_rn_mu, sizeof(g_rn_mu0));
                    memcpy(g_rn_var0, sbgpu::g_rn_var, sizeof(g_rn_var0));
                    g_rn_frozen = true;
                    fprintf(stderr, "sb_env: reward normalizer frozen at step %llu\n",
                            (unsigned long long)hdr->step);
                } else {
                    memcpy(sbgpu::g_rn_mu, g_rn_mu0, sizeof(sbgpu::g_rn_mu));
                    memcpy(sbgpu::g_rn_var, g_rn_var0, sizeof(sbgpu::g_rn_var));
                    static float rn_f[2 * SB_MAX_BLOBS];
                    for (int i = 0; i < SB_MAX_BLOBS; ++i) {
                        const double var = sbgpu::g_rn_var[i] < 1e-12 ? 1e-12 : sbgpu::g_rn_var[i];
                        rn_f[2 * i] = (float)sbgpu::g_rn_mu[i];
                        rn_f[2 * i + 1] = (float)(1.0 / sqrt(var));
                    }
                    if (sbgpu::d_rew_norm != NULL)
                        cudaMemcpyAsync(sbgpu::d_rew_norm, rn_f, 4 * 2 * (size_t)SB_MAX_BLOBS,
                                        cudaMemcpyHostToDevice, stream);
                }
            }
            // per-env episode means over agents that finished at least one episode
            double n = 0, ret = 0, len = 0, score = 0, prod = 0, en = 0, ca = 0, dev = 0, worst = 0, oob = 0;
            for (int b = 0; b < agents; ++b) {
                const Log& L = env[b].log;
                if (!(L.n > 0.0f)) continue;
                n += L.n; ret += L.episode_return; len += L.episode_length; score += L.score;
                prod += L.productivity; en += L.energy_cost_norm; ca += L.carbon_norm;
                dev += L.comfort_dev_K; worst += L.worst_dev_K; oob += L.out_of_band_frac;
            }
            std::string txt;
            char line[1024];
            if (n > 0) {
                snprintf(line, sizeof(line), "env/n\t%.17g\nenv/episode_return\t%.17g\nenv/episode_length\t%.17g\n"
                         "env/score\t%.17g\nenv/productivity\t%.17g\nenv/energy_cost\t%.17g\nenv/carbon\t%.17g\n"
                         "env/comfort_dev_K\t%.17g\nenv/worst_dev_K\t%.17g\nenv/out_of_band_frac\t%.17g\n",
                         n, ret / n, len / n, score / n, prod / n, en / n, ca / n, dev / n, worst / n, oob / n);
                txt += line;
            }
            for (int i = 0; i < out.size; ++i) {
                DictItem* it = &out.items[i];
                if (it->str) continue;
                snprintf(line, sizeof(line), "%s\t%.17g\n", it->key, it->value);
                txt += line;
            }
            uint32_t tlen = (uint32_t)txt.size();
            send_all(fd, &tlen, 4);
            if (tlen) send_all(fd, txt.data(), tlen);
            continue;
        }
        if (op == 'P') {
            // Per-step reward parts of the step just taken: one block of SB_NPARTS doubles per
            // env (productivity, energy_cost, carbon, comfort_dev_K, worst_dev_K, out_of_band,
            // elec_kwh, gas_kwh). d_rparts is allocated per env (d_rparts + e*SB_NPARTS), so this
            // is a plain copy; the core's sb_rec_parts() helper is not used because it refuses
            // more than one group and every eval scenario here is its own group.
            static std::vector<double> parts;
            parts.resize((size_t)agents * SB_NPARTS);
            if (cudaStreamSynchronize(stream) != cudaSuccess) die("parts sync");
            if (cudaMemcpy(parts.data(), sbgpu::d_rparts, 8 * parts.size(),
                           cudaMemcpyDeviceToHost) != cudaSuccess) die("parts D2H");
            uint32_t cnt = (uint32_t)parts.size();
            send_all(fd, &cnt, 4);
            send_all(fd, parts.data(), 8 * (size_t)cnt);
            continue;
        }
        if (op != 'S') die("bad opcode");

        // ---- one vector step ----
        memcpy(h_act, s_act, n_a * sizeof(float));
        if (exec) {
            if (cudaGraphLaunch(exec, stream) != cudaSuccess) die("cudaGraphLaunch");
        } else {
            puf_step(env);
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) die("step sync");
        memcpy(s_obs, h_obs, n_o * sizeof(float));
        memcpy(s_rew, h_rew, n_r * sizeof(float));
        memcpy(s_term, h_term, n_r * sizeof(float));
        if (zone_memory && s_zm_dev != NULL)
            if (cudaMemcpy(s_zm, s_zm_dev, n_z * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                die("zm D2H");
        hdr->step = (uint64_t)++n_steps;
        {
            unsigned char ok = 0;
            send_all(fd, &ok, 1);
        }
        if (n_steps % 288 == 0) {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double t = ts.tv_sec + 1e-9 * ts.tv_nsec;
            if (t_last > 0.0)
                fprintf(stderr, "sb_env: %ld steps, %.2f ms/step\n", n_steps, 1e3 * (t - t_last) / 288.0);
            t_last = t;
        }
    }
    fprintf(stderr, "sb_env: %ld steps, exiting\n", n_steps);
    if (exec) cudaGraphExecDestroy(exec);
    return 0;
}
