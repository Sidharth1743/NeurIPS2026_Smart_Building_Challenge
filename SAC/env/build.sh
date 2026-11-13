#!/bin/bash
# Build the SAC rollout driver. Reads complete_port sources; writes only SAC/bin.
#   env/build.sh                                     # sm_86 (RTX 3090 / 3060)
#   NVCC_ARCH=compute_89 OUT_NAME=sb_env_sm89 env/build.sh   # sm_89 (L40S, on Modal)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
IMG="${SB_IMAGE:-pufferai/puffertank:5.0}"
PL="${PUFFERLIB:-/data/puffer-work/plverify}"
ARCH="${NVCC_ARCH:-compute_86}"      # compute_86 | compute_89
SM="sm_${ARCH#compute_}"
OUT="${OUT_NAME:-sb_env}"
mkdir -p "$ROOT/SAC/bin"
[ -f "$PL/src/pufferenv.h" ] || { echo "no PufferLib headers at $PL/src"; exit 1; }
[ -f "$PL/raylib-5.5_linux_amd64/include/raylib.h" ] || { echo "no raylib headers under $PL"; exit 1; }
docker run --rm -e OUT="$OUT" -e SM="$SM" -v "$ROOT":/w -v "$PL":/pl:ro -w /w/SAC "$IMG" bash -lc '
set -e
nvcc -O3 -std=c++17 -DSB_SHARED_POLICY -gencode arch=${SM/sm_/compute_},code=$SM \
  -I /w/complete_port/gpu \
  -I /w/complete_port/ocean/smart_buildings \
  -I /w/complete_port/ocean/smart_buildings/csim \
  -I /w/complete_port/patches/src \
  -I /pl/src -I /pl/raylib-5.5_linux_amd64/include \
  -x cu env/sb_env.cu \
  /w/complete_port/ocean/smart_buildings/csim/sb_baked.c \
  /w/complete_port/ocean/smart_buildings/csim/sb_rng.c \
  -o bin/$OUT
'
ls -la "$ROOT/SAC/bin/$OUT"
