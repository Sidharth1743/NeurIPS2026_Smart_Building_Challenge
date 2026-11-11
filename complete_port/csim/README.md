# csim

The CPU core. It is the reference the GPU is diffed against, and the same code the C bridge runs
inside the container, so a difference here is a real difference in the physics.

Every file is C99 with no dependency beyond `libm`. The CUDA kernels in `../gpu` mirror these
functions one for one, which is what makes the bit-for-bit gates meaningful.

| file | what |
|---|---|
| `sb_sim.c` | the finite-difference timestep, the thermal network, the HVAC coupling |
| `sb_baked.c` | reading an `.sbb` blob: geometry, solver matrices, convection, statics |
| `sb_lu.c` | the linear solve; `sb_lu_solve` is the bit-identical one the exact build uses |
| `sb_chol.c`, `sb_convect.c` | the factorisation and the convection operator |
| `sb_obs.c`, `sb_reward.c`, `sb_occup.c`, `sb_rng.c` | observation layout, the reward, occupancy, the RNG |
| `sb_bridge.c` | the ctypes and DLL entry points the Python diagnostics use |
| `test/` | the harnesses below, one `main` each |

## Build and run

```bash
make                 # every harness and libsb_bridge.so
make test_episodes
./test_episodes <blob.sbb> <trace.step> <episodes> <out.ref> [steps]   # the multi-episode reference
./test_step <blob.sbb> <trace.step>                                     # one step against a trace
./test_multi <blob.sbb> <trace.step>                                    # several episodes
```

`gate.sh <building> <weather> [steps]` runs the whole chain for one shipped scenario: bake, dump the
oracle trace, compare the GPU against it. It resolves the repository root from its own path.

## Why it matters

The gates in `../tools/verify_all.sh` compare the GPU against these binaries and against the Python
simulator, on six building shapes, at several environment counts, through CUDA graphs, with
`memcmp`-level equality. `test_episodes` produces the reference those comparisons use, and
`tools/make_artifacts.sh` builds it. Do not edit this directory to make a gate pass: the gate is the
contract.
