# docs

| file | what |
|---|---|
| `HARDWARE.md` | what the paper's numbers were produced on, measured throughput, and the two memory limits we hit (`cudaMalloc failed`, not a slowdown) |
| `VERIFICATION.md` | the two end-to-end runs from clean copies, the gate counts, the reproduced 0.0165, the eval-versus-gateway agreement, and a fresh 50M run scored on both panels |
| `RELEASE_EDITS.md` | what the release changes (nothing in the code) and what it adds |
| `MANIFEST.md` | every file with its size and digest, generated at assembly time |

`../SHA256SUMS` holds the same digests in `sha256sum -c` form and sits at the root on purpose.
