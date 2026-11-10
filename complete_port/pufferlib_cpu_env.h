/* Smart Buildings (NeurIPS 2026) PufferLib 5.0 env -- CPU backend.
 *
 * Wraps the L2/L3-green C port in csim/ (synced from complete_port/csim; the C
 * core is the verified parity reference -- see complete_port/README.md). One
 * env instance drives one baked scenario (.sbb blob, mmap-backed).
 *
 *   obs:  SB_OBS = 1024 floats. The scenario's real observation (n_field =
 *         n_dev + n_aux, 352..703 for the shipped shapes) in field order, then
 *         zero padding to 1024. Padded dims are inactive and must be masked out
 *         of the loss (the policy patch does this from action_mask/obs_masks).
 *   act:  SB_CONT = 256 continuous setpoint/flow commands in [-1,1] followed by
 *         SB_DISC = 16 discrete run bits {0,1}. The scenario's real action
 *         (n_cont + n_disc, e.g. 92+5) occupies the front of each block; the
 *         tail is inactive and its action_mask entry is 0.
 *   reset: a fresh episode at the scenario's start timestamp. Occupancy chains
 *         are primed once for the pre-episode interval and the reset
 *         observation is emitted, exactly like Gym reset().
 *   step:  hybrid action -> native setpoints -> device/thermostat update ->
 *         finite-difference + convection step -> observation -> reward. Bit-
 *         identical to csim test_step; the CPU regression (test_env_cpu.c)
 *         checks that.
 *   done:  after steps_per_episode (8640) steps. The Python env reports this as
 *         truncation; PufferLib's Agent has only terminals, so it is set here
 *         and the GAE bootstrap at the cap is zero (documented L4 caveat, same
 *         as crashit).
 *
 * Scenario selection: kwargs `blob` (explicit .sbb path) or `blob_dir` (pick
 * `paths[rng % n]`, sorted for determinism). Never a hardcoded building id.
 */
#ifndef SMART_BUILDINGS_H
#define SMART_BUILDINGS_H

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float obs_t;
#include "pufferenv.h"

#include "csim/sb_baked.h"
#include "csim/sb_convect.h"
#include "csim/sb_fdm.h"
#include "csim/sb_obs.h"
#include "csim/sb_occup.h"
#include "csim/sb_reward.h"
#include "csim/sb_sim.h"

#define SB_OBS 1024
#define SB_CONT 256
#define SB_DISC 16
#define SB_NATIVE_MAX (SB_CONT + SB_DISC)
#define SB_MAX_BLOBS 64
#define SB_PATH_MAX 512

/* ACT_SIZES: SB_CONT heads of size 1 (one Gaussian each) then SB_DISC heads of
 * size 2 (Bernoulli). Built by repetition so the initializer stays readable. */
#define SB_ONE_1 1
#define SB_ONE_2 SB_ONE_1, SB_ONE_1
#define SB_ONE_4 SB_ONE_2, SB_ONE_2
#define SB_ONE_8 SB_ONE_4, SB_ONE_4
#define SB_ONE_16 SB_ONE_8, SB_ONE_8
#define SB_ONE_32 SB_ONE_16, SB_ONE_16
#define SB_ONE_64 SB_ONE_32, SB_ONE_32
#define SB_ONE_128 SB_ONE_64, SB_ONE_64
#define SB_ONE_256 SB_ONE_128, SB_ONE_128
#define SB_TWO_1 2
#define SB_TWO_2 SB_TWO_1, SB_TWO_1
#define SB_TWO_4 SB_TWO_2, SB_TWO_2
#define SB_TWO_8 SB_TWO_4, SB_TWO_4
#define SB_TWO_16 SB_TWO_8, SB_TWO_8

#define OBS_SIZE SB_OBS
#define NUM_ATNS (SB_CONT + SB_DISC)
#define ACT_SIZES { SB_ONE_256, SB_TWO_16 }

/* Mask width the trainer derives from ACT_SIZES: one entry per continuous head
 * plus one per discrete category. */
#define SB_MASK_WIDTH (SB_CONT + 2 * SB_DISC)

struct Log {
    float perf;            /* mean per-step reward (the competition metric) */
    float score;           /* same quantity; kept as its own column */
    float episode_return;  /* sum of rewards over the episode */
    float episode_length;  /* steps per episode */
    float reward_mean;     /* running mean of the raw step reward */
    float n;               /* Required as the last field */
};

struct Env {
    Log log;               /* Required field */
    Agent agents[1];
    /* Required by the trainer (env_setup / eval_make read these). */
    int num_agents;
    int tag;
    int boundary_reached;

    /* Loaded scenario. Heap-allocated: the trainer reallocs the Env array
     * right after the init loop, so nothing here may point into the Env. */
    SbBlob* blob;
    SbFdm* fdm;
    SbConv* conv;
    SbOccup* occ;
    SbSim* sim;

    const uint8_t* is_disc;
    int n_act, n_cont, n_disc;
    double* cont;
    double* disc;
    double* native;
    double* parts;
    float* obs;
    int obs_len;

    int step;
    int max_steps;
    int loaded;
    unsigned int rng;
    int seed;
};

/* ---- helpers ------------------------------------------------------------ */

static int smart_buildings_endsWith(const char* s, const char* suf) {
    size_t n = strlen(s), m = strlen(suf);
    return n >= m && strcmp(s + n - m, suf) == 0;
}

/* Collect *.sbb under dir into out (sorted). Returns the count. */
static int smart_buildings_list_blobs(const char* dir, char (*out)[SB_PATH_MAX]) {
    DIR* d = opendir(dir);
    if (!d)
        return 0;
    int n = 0;
    struct dirent* e;
    while ((e = readdir(d)) != NULL && n < SB_MAX_BLOBS) {
        if (!smart_buildings_endsWith(e->d_name, ".sbb"))
            continue;
        snprintf(out[n], SB_PATH_MAX, "%s/%s", dir, e->d_name);
        n++;
    }
    closedir(d);
    /* Deterministic order (readdir order is filesystem-dependent). */
    for (int i = 1; i < n; i++) {
        char key[SB_PATH_MAX];
        memcpy(key, out[i], SB_PATH_MAX);
        int j = i - 1;
        while (j >= 0 && strcmp(out[j], key) > 0) {
            memcpy(out[j + 1], out[j], SB_PATH_MAX);
            j--;
        }
        memcpy(out[j + 1], key, SB_PATH_MAX);
    }
    return n;
}

static void smart_buildings_zero_tail(float* dst, int n, int from) {
    for (int i = from; i < n; i++)
        dst[i] = 0.0f;
}

static void smart_buildings_write_obs(Env* env, int j) {
    /* sb_obs writes obs_len floats; everything past that is padded and inert. */
    sb_obs(env->sim, env->occ->V, j, env->obs);
    smart_buildings_zero_tail(env->obs, SB_OBS, env->obs_len);
    memcpy(env->agents[0].observations, env->obs, SB_OBS * sizeof(float));
}

/* Publish the scenario's action layout into the trainer's mask: active
 * continuous heads and active discrete categories are 1, padding is 0. */
static void smart_buildings_write_mask(Env* env) {
    unsigned char* m = env->agents[0].action_mask;
    if (m == NULL)
        return;
    for (int h = 0; h < SB_CONT; h++)
        m[h] = (h < env->n_cont) ? 1 : 0;
    int off = SB_CONT;
    for (int h = 0; h < SB_DISC; h++) {
        unsigned char v = (h < env->n_disc) ? 1 : 0;
        m[off + 2 * h] = v;
        m[off + 2 * h + 1] = v;
    }
}

/* Required function */
void puf_reset(Env* env) {
    if (!env->loaded) {
        fprintf(stderr, "smart_buildings: reset before load\n");
        abort();
    }
    sb_sim_reset(env->sim, env->blob);
    /* The Gym reset() reads the reset observation, which advances the occupant
     * chains for the pre-episode interval [ts_-1, ts_0). */
    sb_occup_prime(env->occ);
    env->step = 0;
    for (int s = 0; s < env->sim->n_action; s++) {
        env->native[s] = 0.0;
    }
    /* rewards/terminals are NOT touched here: the trainer reads the terminal
     * that ended the episode, and the env auto-starts the next one (the
     * cartpole pattern). */
    smart_buildings_write_mask(env);
    smart_buildings_write_obs(env, 0);
    /* Step 0 consumes [ts_0, ts_1); prime it now so the first step and the
     * reward that follows both see the right interval. */
    sb_occup_advance(env->occ, 0);
}

/* Required function */
void puf_step(Env* env) {
    SbSim* sim = env->sim;
    /* Native order mixes continuous and discrete dims; cont[]/disc[] are the
     * front slices of the two padded agent-action blocks. */
    /* The oracle's BoundedActionNormalizer rejects normalized values outside
     * [min_norm, max_norm] ([-1, 1] here), so an unbounded Gaussian sample has
     * to be clamped before it reaches the setpoint map. */
    for (int i = 0; i < env->n_cont; i++) {
        double a = (double)env->agents[0].actions[i];
        if (a > 1.0)
            a = 1.0;
        else if (a < -1.0)
            a = -1.0;
        env->cont[i] = a;
    }
    for (int i = 0; i < env->n_disc; i++)
        env->disc[i] = (double)env->agents[0].actions[SB_CONT + i];
    sb_decode_native(env->n_act, env->is_disc, sb_f64(env->blob, "act_min_native"),
                     sb_f64(env->blob, "act_max_native"),
                     sb_f64(env->blob, "act_min_norm"),
                     sb_f64(env->blob, "act_max_norm"),
                     env->cont, env->disc, env->native);

    sb_sim_apply_native(sim, env->native);
    sb_sim_step(sim, env->occ->V);
    env->step += 1;
    /* The observation after step k is taken at ts_k+1 and reports the occupancy
     * of the interval that just ran. */
    smart_buildings_write_obs(env, env->step);
    sb_occup_advance(env->occ, env->step);
    double rew = sb_reward(sim, env->occ->V, env->step, env->parts);

    env->agents[0].rewards[0] = (float)rew;
    int done = env->step >= env->max_steps;
    env->agents[0].terminals[0] = done ? 1.0f : 0.0f;

    env->log.episode_return += (float)rew;
    env->log.reward_mean += ((float)rew - env->log.reward_mean) / (float)env->step;
    if (done) {
        float mean = env->log.episode_return / (float)env->step;
        env->log.perf += mean;
        env->log.score += mean;
        env->log.episode_length += (float)env->step;
        env->log.n += 1.0f;
        puf_reset(env);
    }
}

/* Required function. Headless: there is nothing to draw. */
void puf_render(Env* env) {
    (void)env;
}

/* Required function. */
void puf_close(Env* env) {
    free(env->blob);
    free(env->fdm);
    free(env->conv);
    free(env->occ);
    free(env->sim);
    free(env->cont);
    free(env->disc);
    free(env->native);
    free(env->parts);
    free(env->obs);
    env->blob = NULL;
    env->fdm = NULL;
    env->conv = NULL;
    env->occ = NULL;
    env->sim = NULL;
    env->loaded = 0;
}

void puf_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "reward_mean", log->reward_mean);
    dict_set(out, "n", log->n);
}

void puf_init(Env* env, Dict* kwargs) {
    memset(&env->log, 0, sizeof(env->log));
    memset(&env->agents[0], 0, sizeof(Agent));
    env->agents[0].policy = 0;
    env->num_agents = 1;
    env->tag = 0;
    env->boundary_reached = 0;
    env->loaded = 0;

    char paths[SB_MAX_BLOBS][SB_PATH_MAX];
    int n = 0;
    /* Every key is optional here; dict_get* exits on a miss, so probe first. */
    DictItem* one = dict_find(kwargs, "blob");
    if (one != NULL && one->str != NULL && one->str[0] != '\0') {
        snprintf(paths[0], SB_PATH_MAX, "%s", one->str);
        n = 1;
    } else {
        DictItem* di = dict_find(kwargs, "blob_dir");
        const char* dir = (di != NULL && di->str != NULL && di->str[0] != '\0')
                              ? di->str
                              : getenv("SB_BLOB_DIR");
        if (dir != NULL && dir[0] != '\0')
            n = smart_buildings_list_blobs(dir, paths);
        if (n == 0) {
            fprintf(stderr,
                    "smart_buildings: no scenario. Set env.blob or env.blob_dir "
                    "(or SB_BLOB_DIR).\n");
            abort();
        }
    }
    int pick = (n > 1) ? (int)(env->rng % (unsigned int)n) : 0;

    env->blob = (SbBlob*)calloc(1, sizeof(SbBlob));
    env->fdm = (SbFdm*)calloc(1, sizeof(SbFdm));
    env->conv = (SbConv*)calloc(1, sizeof(SbConv));
    env->occ = (SbOccup*)calloc(1, sizeof(SbOccup));
    env->sim = (SbSim*)calloc(1, sizeof(SbSim));
    if (!env->blob || !env->fdm || !env->conv || !env->occ || !env->sim)
        abort();
    if (sb_open(env->blob, paths[pick]) != 0) {
        fprintf(stderr, "smart_buildings: cannot open blob %s\n", paths[pick]);
        abort();
    }
    if (sb_fdm_init(env->fdm, env->blob) || sb_conv_init(env->conv, env->blob) ||
        sb_occup_init(env->occ, env->blob) ||
        sb_sim_init(env->sim, env->blob, env->fdm, env->conv, env->occ)) {
        fprintf(stderr, "smart_buildings: init failed for %s\n", paths[pick]);
        abort();
    }

    env->is_disc = sb_u8(env->blob, "act_is_discrete");
    env->n_act = env->sim->n_action;
    env->n_cont = 0;
    env->n_disc = 0;
    for (int i = 0; i < env->n_act; i++) {
        if (env->is_disc[i])
            env->n_disc++;
        else
            env->n_cont++;
    }
    if (env->n_cont > SB_CONT || env->n_disc > SB_DISC || env->n_act > SB_NATIVE_MAX) {
        fprintf(stderr,
                "smart_buildings: scenario needs %d continuous / %d discrete "
                "actions, env has %d / %d. Raise SB_CONT/SB_DISC.\n",
                env->n_cont, env->n_disc, SB_CONT, SB_DISC);
        abort();
    }
    /* The policy action layout is [all continuous dims][all discrete dims];
     * sb_decode_native merges them with the baked per-dim flags. */
    env->cont = (double*)calloc(SB_CONT, sizeof(double));
    env->disc = (double*)calloc(SB_DISC, sizeof(double));
    env->native = (double*)calloc(env->n_act ? env->n_act : 1, sizeof(double));
    /* sb_reward writes 12 diagnostic parts (see sb_reward.c). */
    env->parts = (double*)calloc(12, sizeof(double));
    env->obs = (float*)calloc(SB_OBS, sizeof(float));
    if (!env->cont || !env->disc || !env->native || !env->parts || !env->obs)
        abort();
    env->obs_len = env->sim->n_dev + env->sim->n_aux;
    if (env->obs_len > SB_OBS) {
        fprintf(stderr, "smart_buildings: scenario obs %d > OBS_SIZE %d\n",
                env->obs_len, SB_OBS);
        abort();
    }
    DictItem* ms = dict_find(kwargs, "max_steps");
    env->max_steps = ms ? (int)ms->value : 0;
    if (env->max_steps <= 0) {
        const SbSection* sec = sb_find(env->blob, "occ_hour");
        env->max_steps = sec ? (int)sec->count : 0;
    }
    if (env->max_steps <= 0)
        env->max_steps = 8640;
    env->loaded = 1;
}

#endif /* SMART_BUILDINGS_H */
