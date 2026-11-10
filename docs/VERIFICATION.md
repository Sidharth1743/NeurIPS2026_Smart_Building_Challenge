# Verification record

The release was run end to end from a clean copy, not from the development workspace.

| | |
|---|---|
| date | 2026-11-10, 15:49 to 16:23 UTC |
| host | Linux, one NVIDIA RTX 3060, 12 GB (the smaller card; the paper's numbers come from a 3090 and an L40S) |
| container | `pufferai/puffertank:5.0` (CUDA 13.0.88) |
| copy | the tracked files of this repository, `rsync`ed to an empty directory, no build output |
| upstreams | starter kit cloned at `23a4560`, PufferLib cloned at `6ffa5b1` and patched, both by `./run.sh setup` |
| commands | `./run.sh setup` then `./run.sh data` then `SB_GPU=1 ./run.sh verify` |
| result | **67 checks passed, 0 failed, `ALL CHECKS PASSED`** |

```
PASS  trainer: ./build.sh smart_buildings from base commit + patch + complete_port
PASS  trainer: actual mixed-head PPO kernels match finite-difference reference
PASS  trainer: random starts, seven oracle clocks, full episodes and CUDA graph resets
PASS  trainer: complete asynchronous checkpoint resumes optimizer, RNG, simulator and pending rollout
PASS  shared policy: derivatives, permutation equivariance, masked outputs, CUDA graph
PASS  building_train_1__fairbanks_1: trace gate (oracle) + reset observation, 8 envs
PASS  building_train_1__fairbanks_1: multi-episode vs CPU env, host reset
PASS  building_train_1__fairbanks_1: multi-episode vs CPU env, in-step restart
PASS  building_train_1__fairbanks_1: trace gate through a CUDA graph
PASS  building_train_1__fairbanks_1: device solve == sb_lu_solve bit for bit (5, 40 envs)
PASS  building_train_1__fairbanks_1: convection == sb_conv_apply bit for bit
   ... one block of six per shipped building shape, then ...
PASS  train_1: trace gate at 1, 64 and 256 envs
PASS  train_1: bit-exact-solve build (SB_LU_EXACT) passes the oracle gate and the multi-episode diff
PASS  train_1: training-only fp32 build (SB_F32) runs the gate, drift bounded
PASS  generated buildings 0502/0511: bake, oracle gate, multi-episode
PASS  generated 4-floor building 0500: SB_CONT_BIG oracle gate
PASS  multi-scenario: sb_group_state.inc matches the sources
PASS  weather merge: building_train_1 and building_train_8, merged == solo bit for bit
PASS  per-env restart: env 0 alone == group restart, other envs untouched
PASS  fused multi-group solve: 6 buildings x 40 envs, fused == solo bit for bit (CUDA graph)
PASS  viewer: play replay writes a snapshot; terminal frame shows step 40/8640
PASS  viewer: pygame window renders the snapshot (headless screenshot > 20 kB)
ALL CHECKS PASSED
```

Wall clock: 33.6 minutes for `verify`, 6.5 minutes for `data` on an empty machine (it bakes 30
scenarios and their oracle traces), and 36 seconds for a cold `setup` once the clones are on the
network. A second `setup` against a provisioned checkout takes 6 seconds.

## Two defects the run found, both fixed

1. `git clone` of PufferLib died with `RPC failed; curl 92 HTTP/2 stream 5 was not closed cleanly`.
   `./run.sh setup` now clones shallow and blob-less over HTTP/1.1, with three retries, and says what
   to do if the network refuses: point `SB_PL_ROOT` at a directory that already holds `plclean`.
2. The gate `trainer: random starts, seven oracle clocks` read TensorBoard through
   `complete_port/viewer/.venv/bin/python`, an environment that is not part of a source checkout.
   `./run.sh setup` now builds it (`pygame`, `numpy`, `tensorboard`), which also gives `tools/watch.sh`
   its interpreter.

A third issue was in this repository's own setup script rather than in the code: a second `setup`
tried to apply the PufferLib patch again and failed with `patch does not apply`. It now tests
`git apply --check` first and only patches an unpatched checkout.

## What this proves, and what it does not

It proves that a fresh checkout of this repository rebuilds both simulator variants, passes every
bit-exactness gate against the unmodified organiser simulator and against the CPU core, trains, and
renders a replay, without any file from the development workspace.

It does not reproduce a score: scoring needs the 90-month scenario set baked (about 20 minutes) and
then `./run.sh score`, and it does not retrain the reported policy (200M steps, 4.6 hours on an L40S).

This document was added after the gates finished, and one build detail was fixed afterwards: the
gzip header of the shipped scalar series carried a timestamp, so two builds of the same release
differed byte for byte. It is written with `mtime=0` now, and two consecutive builds produce the
same `SHA256SUMS`. The uncompressed content of those files is identical to what the gates ran
against, and no gate reads them. Nothing else under `complete_port/`, `data/`, `experiments/` or
`paper/` changed as a result of the run.
