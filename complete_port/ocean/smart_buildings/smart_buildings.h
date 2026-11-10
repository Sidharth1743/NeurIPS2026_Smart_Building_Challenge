/* Smart Buildings env: GPU path. The implementation lives in sb_gpu_env.inc (the
 * device kernels in sb_gpu_kernels.cuh; the same pair is the parity gate under
 * complete_port/gpu, built with -DSB_TEST). The CPU reference env that this
 * replaces is preserved at complete_port/pufferlib_cpu_env.h. */
typedef float obs_t;
#define OBS_SIZE 1024
/* NUM_ATNS / ACT_SIZES / SB_CONT are defined in sb_gpu_env.inc (SB_CONT_BIG widens the
 * continuous block from 256 to 320 heads) */
#include "sb_gpu_env.inc"
