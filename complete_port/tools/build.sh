#!/bin/bash
# Build the device-side test programs from complete_port only (no PufferLib checkout):
#   tools/build.sh                 gate (default = fused-multiply-add solve) and convtest
#   SB_LU_EXACT=1 tools/build.sh   gate_exact and lutest (the bit-identical solve)
#   SB_F32=1 tools/build.sh        gate_f32 (training-only: the solve state in fp32, defining
#                                  SB_SOLVE_F32, see gpu/sb_gpu_lusolve.cuh; timing only, never for evaluation)
#   convtest   producer+apply convection vs csim's sb_conv_apply, bit for bit
set -e
. "$(dirname "$0")/common.sh"
[ -n "$SB_F32" ] && [ -n "$SB_LU_EXACT" ] && { echo "SB_F32 and SB_LU_EXACT are mutually exclusive"; exit 2; }
O=/cp/ocean/smart_buildings
FL="-O3 -std=c++17 -gencode arch=compute_86,code=sm_86 -I/cp/gpu -I$O -I$O/csim"
S=""
[ -n "$SB_CONT_BIG" ] && FL="$FL -DSB_CONT_BIG" && S=_big
[ -n "$SB_LU_EXACT" ] && FL="$FL -DSB_LU_EXACT" && S=_exact
[ -n "$SB_F32" ] && FL="$FL -DSB_SOLVE_F32" && S="${S}_f32"
if [ -n "$SB_LU_EXACT" ]; then   # exact-arithmetic builds: the gate and the bit-for-bit solve test
  rm -f "$ART/bin/gate_exact" "$ART/bin/lutest"
  dock "nvcc $FL -DSB_TEST -x cu $O/sb_gpu_env.inc $O/csim/sb_baked.c $O/csim/sb_rng.c -o artifacts/bin/gate_exact 2>&1 | grep -E ' error' -A4; \
        nvcc $FL -x cu gpu/sb_gpu_lutest.cu $O/csim/sb_baked.c $O/csim/sb_lu.c -o artifacts/bin/lutest 2>&1 | grep -E ' error' -A4; true"
  want="gate_exact lutest"
else
  # variants (SB_CONT_BIG, SB_F32) build their own convtest so the default one is never clobbered
  rm -f "$ART/bin/gate$S" "$ART/bin/convtest$S"
  dock "nvcc $FL -DSB_TEST -x cu $O/sb_gpu_env.inc $O/csim/sb_baked.c $O/csim/sb_rng.c -o artifacts/bin/gate$S 2>&1 | grep -E ' error' -A4; \
        nvcc $FL -x cu gpu/sb_gpu_convtest.cu $O/csim/sb_baked.c $O/csim/sb_rng.c $O/csim/sb_convect.c -o artifacts/bin/convtest$S 2>&1 | grep -E ' error' -A4; true"
  want="gate$S convtest$S"
fi
for f in $want; do [ -x "$ART/bin/$f" ] || { echo "BUILD FAILED: $f"; exit 1; }; done
ls -la "$ART/bin"
