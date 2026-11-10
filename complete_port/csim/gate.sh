#!/bin/bash
SB_ROOT="${SB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# End-to-end gate for one shipped scenario: bake -> dump oracle trace -> compare.
# usage: ./gate.sh <building> <weather> [steps]
set -e
B=$1
W=$2
N=${3:-30}
cd $SB_ROOT/starter-kit
ulimit -v 9000000
.venv/bin/python ../complete_port/pysim/bake_scenario.py \
    --out ../complete_port/csim/baked --name "${B}__${W}" \
    "smart_buildings/training_scenarios/${B}/${W}/config.gin"
.venv/bin/python ../complete_port/pysim/dump_step.py \
    "smart_buildings/training_scenarios/${B}/${W}/config.gin" \
    "/tmp/opencode/${B}_${W}.step" "$N"
cd ../complete_port/csim
ulimit -v 7000000
./test_step "baked/${B}__${W}.sbb" "/tmp/opencode/${B}_${W}.step"
