# r15_structured_beta_b2048_s75 — the Beta-head arm at 50 M

The one arm that separates "the off-policy learner cannot climb" from "this action head
cannot move". Everything is r11: structured actor, batch 2048 x utd 2, async, 1 M replay,
running normaliser, compiled features, no lr anneal, seed 75, 50,135,040 steps, the same 30
scenarios. **The only changed factor is the action head**: `beta` (floor 0.5, cap 12 — the
competition's canonical decode) instead of the tanh-Gaussian.

Training panel (30 scenarios, `env/episode_score_scen_mean`): see the TensorBoard run
`r15_structured_beta_b2048_s75` on :6007.

## Held-out 144

| set | score | productivity | energy | carbon |
|---|---|---|---|---|
| 90 (`eval5_heldout18`) | **0.033818** | −0.0437 | 0.0241 | 0.0139 |
| 54 (`eval3_heldout18`) | **0.022917** | — | — | — |
| 144 (union) | **0.029730** | | | |

Per building (90): 1 → 0.0223, 2 → 0.0208, **3 → 0.0814**, 6 → 0.0261, 8 → 0.0185.
Per building (54): 4 → 0.0174, 5 → 0.0250, 7 → 0.0264.

## Against the same-budget Gaussian arm

| at 50 M, same seed, same data, same everything else | 90-set | 144 |
|---|---|---|
| r11 tanh-Gaussian | 0.04570 | — |
| r14 tanh-Gaussian, 49.97 M | — | 0.042717 |
| **r15 Beta** | **0.033818** | **0.029730** |

**−26 % on the 90-set, −30 % on the 144**, and the improvement is not one term: productivity
−0.0535 → −0.0437 (comfort better), energy 0.0267 → 0.0241, carbon 0.0263 → 0.0139 (Halved).
Building 6 is the clearest: 0.0624 → 0.0261. Building 3 stays expensive (0.0814).

Reference points on the same 90 scenarios: r116 (PPO) @50.1 M 0.03010, r139 (PPO) @199 M
0.01651. The Beta arm at 50 M is therefore **within 12 % of r116 at the same budget**,
where the Gaussian arm was 52 % behind it.

## What this does and does not license

- It does license: *representation alone is not sufficient*, and *the tanh-Gaussian head was a
  binding constraint for this off-policy learner* — the head is the only variable, so the
  0.013 of the 90-set gap cannot be attributed to the environment, the data, the budget or the
  seed.
- It does **not** license "the learning setup decides": one factor moved, and SAC is still
  2x behind r139 at the far larger 200 M budget. It says the earlier failure was not evidence
  against off-policy learning; it is not evidence that off-policy learning wins.
- The head is a prior like any other decode, recorded here with its measured effect rather
  than its intent.

## Artifacts

- checkpoints: `results/r15_structured_beta_b2048_s75/ckpt/` (21, 2.5 M … 50,135,040)
- recordings: `record90/50135040/`, `record54/50135040/` (obs, loc, rew, term, mask, parts)
- analysis: `ANALYSIS_144.md`
- Kaggle: `sidharth1743/submission-sac-r15-beta-50m` v1, ref 58276615,
  commit-log `head beta floor 0.5 cap 12.0`, example-scenario mean reward **−0.0138**
  (every Gaussian checkpoint: −0.0218 to −0.0250)
