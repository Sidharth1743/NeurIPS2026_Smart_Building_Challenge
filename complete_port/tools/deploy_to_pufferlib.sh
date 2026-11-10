#!/bin/bash
# Reproduce the trainer build from complete_port + a PufferLib 5.0 checkout:
#   deploy_to_pufferlib.sh <checkout> [--apply-patch]
# Copies ocean/smart_buildings (symlinks resolved) and config/smart_buildings.ini into the
# checkout; --apply-patch first applies patches/pufferlib_core.diff (mixed action heads,
# PUF_HAS_MASK_INIT, build.sh smart_buildings case) and installs patches/src/tblog.h to a checkout at
# patches/pufferlib_base_commit.txt.
set -e
. "$(dirname "$0")/common.sh"
PL="$(cd "$1" && pwd)"
[ "$2" = "--apply-patch" ] && ( cd "$PL" && git apply "$CP/patches/pufferlib_core.diff" )
cp "$CP/patches/src/beta_math.cuh" "$CP/patches/src/sb_r110_conditional_head.cuh" "$CP/patches/src/sb_r110_conditional_kernels.cuh" "$PL/src/"
cp "$CP/patches/src/tblog.h" "$PL/src/tblog.h"   # untracked in the checkout this port was developed on; pufferl.cu includes it
cp "$CP/patches/src/sb_shared_policy.cuh" "$CP/patches/src/sb_shared_checks.cuh" "$CP/patches/src/sb_zone_reward.cuh" "$CP/patches/src/sb_zone_command.cuh" "$PL/src/"
cp "$CP/patches/src/sb_zone_diagnostics.cuh" "$PL/src/"
cp "$CP/patches/src/sb_full_trainer.cuh" "$PL/src/"
mkdir -p "$PL/ocean/smart_buildings"
rsync -aL --delete "$CP/ocean/smart_buildings/" "$PL/ocean/smart_buildings/"
cp "$CP/config/smart_buildings.ini" "$PL/config/smart_buildings.ini"
echo "deployed to $PL"
