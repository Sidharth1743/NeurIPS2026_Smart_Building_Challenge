#!/bin/bash
set -e
SRC=/work/plclean/ocean/smart_buildings
nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 -I$SRC -I$SRC/csim -I/work/cudss/nvidia/cu13/include \
  -x cu /work/convtest/sb_gpu_convtest.cu $SRC/csim/sb_baked.c $SRC/csim/sb_rng.c $SRC/csim/sb_convect.c -o /work/opt/convtest \
  -L/work/cudss/nvidia/cu13/lib -lcudss -Xlinker -rpath=/work/cudss/nvidia/cu13/lib 2>&1 | grep -v "warning\|\^\|^$\|static\|declared" | head -20
