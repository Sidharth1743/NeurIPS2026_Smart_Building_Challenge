# ablations.md — the definitive ledger (v3, rewritten 2026-11-07 22:10 IST)

One document, one structure, no superseded layers. Everything below was **re-derived from artifacts** —
`args.txt` token diffs, `train.log` panels, `score.json`, `curve.json`, driver logs and the deployed patch —
never from the `.md` docs. Where a claim has no artifact behind it, the table says so.

The headline: **r116 was ablated, not inherited.** Twelve single-flag arms were trained to 50,135,040 steps
on one binary (`adbe6911`), one seed (75), one batch (`r85_four`, 4×64), against the same baseline. Seven of
the recipe's flags are confirmed load-bearing; four are inert or helpful when removed; one replacement (Adam
for the whole Muon/role-Adam optimizer stack) is the **best 50M panel number in the campaign**.

---

## §1 The baseline: r116

**Recipe chain** (verified twice by `args.txt` token diffs — each step is exactly one flag except step 6's
three anneal tokens, which act as one):

| step | run | flag added |
|---|---|---|
| 1 | r84_cn12 | `beta_cn_tau=12` (coherent exploration) |
| 2 | r85_four | batch 1 blob → `r85_four` 4-scenario set |
| 3 | r91 | `beta_cn_tau_ahu=48` |
| 4 | r93 | `role_adam=1` |
| 5 | r94 | `beta_conc_max=12` |
| 6 | r94b | `anneal_lr=1 +linear_lr_anneal=1 +anneal_steps=50135040` |
| 7 | r115 | `headwise_ratio=1` (budget 80.2M → 50.135M) |
| 8 | r116 | `zone_mlp=1` |

Inherited unchanged from the r54–r79 lineage: `beta=1`, `beta_floor=1`, `canonical=1`, `zone_memory=1`,
`zone_critic=1`, `valve_zone=1`, `value_shared=1`, `value_delay=1`, `desync=1`, `adv_norm=2`,
`freeze_return=0`, `reward_scale=1`, `ratio_per_dim=1`, γ 0.99, λ 0.95, clip 0.2, vf_clip 1e4,
grad-clip 0.5, minibatch 2304, horizon 288, replay 4, `ent_coef=0.001`.

**Baseline numbers (what every arm is compared to):**

| | value | source |
|---|---|---|
| final training panel s75 / s76 | **0.0330 / 0.0350** | `train.log`, both 680 epochs, 10 ckpts |
| seed spread | 6 % | the two completed seeds |
| frozen closed-loop, 4 training scenarios (s75) | **0.0305** vs r94b's 0.0376 | 4 × `score.json` |
| Fairbanks frozen (s75) | **0.0755** vs r94b 0.1035 (gas 168.6 → 118.5 MWh) | `score.json` |
| official `pid_water`, 30 scenarios | 0.05496 | `starter-kit/smart_buildings/results` |
| **best measured configuration: r139** = r116 + the five removals (§13.2) | **panel 0.0280**, live `env/score` −0.0189 | `train.log` / TB, 10 ckpts |

---

## §2 THE definitive r116 ablation set — 12 arms, one binary, one seed, one flag

All twelve: `r116`'s stored args with exactly one change, fresh from scratch, seed 75, fp64, 50,135,040 steps,
linear LR anneal to the same budget, 10 checkpoints, rc=0, no scoring. Binary `adbe6911` for every arm **and**
for both r116 baseline runs, so the whole table is one-binary. "saves ≤" = how many of the six matched saves
(10/20/30/40/45/50.135M) the arm beat r116 s75.

| # | run | flag change | 10M | 20M | 30M | 40M | 45M | **50M (final)** | Δ final | saves ≤ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | r129 | `desync=0` | 2.641 | 2.301 | 2.290 | 2.083 | 2.158 | **2.213** | **+6,606 %** | 0/6 | **desync essential** |
| 2 | r125 | `headwise_ratio=0` | 0.725 | 0.096 | 0.176 | 0.160 | 0.165 | **0.152** | **+361 %** | 1/6 | **headwise essential** |
| 3 | r127 | `beta_conc_max=0` | 2.180 | 0.168 | 0.150 | 0.070 | 0.060 | **0.054** | **+64 %** | 0/6 | **cap essential** |
| 4 | r123 | `zone_mlp=0` | 0.492 | 0.082 | 0.065 | 0.055 | 0.053 | **0.052** | **+58 %** | 1/6 | **zone MLP essential** |
| 5 | r128 | `role_adam=0` | 0.803 | 0.137 | 0.079 | 0.055 | 0.049 | **0.049** | **+48 %** | 0/6 | **role split essential** |
| 6 | r134 | `zone_signed=1` | 0.718 | 0.136 | 0.083 | 0.047 | 0.043 | **0.044** | **+33 %** | 0/6 | magnitudes right |
| 7 | r131 | `anneal_lr=0` | 0.355 | 0.074 | 0.070 | 0.052 | 0.038 | **0.038** | **+15 %** | 1/6 | anneal helps |
| 8 | r124 | `beta_cn_tau=0` + `beta_cn_tau_ahu=0` | **0.063** | **0.047** | **0.036** | **0.033** | **0.032** | **0.031** | **−6 %** | **6/6** | **coherence inert** |
| 9 | r126 | `beta_cn_tau_ahu=12` | 0.223 | 0.066 | 0.038 | 0.030 | 0.031 | **0.031** | **−6 %** | **6/6** | **τ_AHU=48 inert** |
| 10 | r133 | `beta_floor=0.5` | 0.376 | 0.077 | 0.049 | 0.033 | 0.031 | **0.031** | **−6 %** | 5/6 | floor claim inverts (noise) |
| 11 | r132 | `value_delay=0` | 0.272 | 0.074 | 0.047 | 0.034 | 0.032 | **0.030** | **−9 %** | **6/6** | **delay numbers inert** |
| 12 | **r130** | **`adam=1` + `role_adam=0`** | **0.106** | **0.044** | **0.035** | **0.032** | **0.029** | **0.029** | **−12 %** | **6/6** | **Muon + role split is a liability — best run ever** |
| — | *baseline* | *r116 s75* | *0.273* | *0.126* | *0.052* | *0.036* | *0.034* | ***0.0330*** | — | — | — |
| — | *baseline* | *r116 s76* | *0.336* | *0.146* | *0.072* | *0.044* | *0.039* | ***0.0350*** | — | — | seed spread ≈ 6 % |

Read order: rows 1–7 are **confirmed load-bearing** (removal is worse); rows 8–12 are **dead weight or a
liability** (removal is equal or better, and rows 8, 9, 11, 12 dominate r116 at **every one of the six saves**).
Wall time 7,992–8,467 s per 50.135M on the 3090 (5,922–6,273 SPS), 16,070–16,269 s on the 3060 (3,082 SPS);
no optimizer-dependent speed difference (Adam 6,018 vs Muon 6,069 SPS).

### §2.1 Round 3 — the four flags the census proves were NEVER varied (launched 2026-11-08)

The flag census (§11) proves these four knobs have never once been set to any other value in **any** of the 335
stored runs, in every launch command (`command.txt`), stored args, study scripts and the Modal configs:

| # | run | flag change | census proof | **final panel (50.135M)** | Δ vs r116 (0.033) | saves ≤ | verdict at r116 | GPU | status |
|---|---|---|---|---|---|---|---|---|---|
| **1** | `r135_clip01_s75` | `clip_coef 0.2 → 0.1` | **0.2 in all 424 launches** — never once changed | **0.0510** | **+55 %** | 1/6 (20M ahead) | **trust region: 0.2 was right** — tightening hurts | 0 | ✅ done 02:33 IST (8,331 s class) |
| **2** | `r136_maxgrad10_s75` | `max_grad_norm 0.5 → 1.0` | **0.5 in all 425 launches** — never once changed | **0.0340** | +3 % | 3/6 | **neutral** — gradient clip not sensitive (30M was better: 0.045 vs 0.052) | 0 | ✅ done 06:16 IST |
| **3** | `r137_gamma995_s75` | `gamma 0.99 → 0.995` | only 0.99 (430), 0.999 (37, **r7–r9 raw-obs era**), 0.9966 (4, r33/r34 repeat-compensation) | **0.0380** | **+15 %** | 3/6 (better at 10/20/30M, worse at 40/45/50M) | **γ=0.99 was right** — the longer horizon learns faster early (10M 0.158 vs 0.273) but converges worse from 40M on. First-ever credit-horizon test at r116 | 0 | ✅ done 08:34 IST |
| **4** | `r138_hidden128_s75` | `hidden_size 256 → 128` | **256 in all 490 launches** — never once changed; depth 2 also never varied | **0.0310** | **−6 %** | 4/6 | **capacity is not the constraint** — half-width is at least as good (worse only at 10M) | 1 | ✅ done 04:45 IST (15,530 s, 3,228 SPS on the 3060) |

Gates verified at launch: each command differs from r116's stored args by the run paths plus exactly the listed
flag; `hidden_size` is a runtime hyper (`hypers->hidden_size`) and the zone-MLP/role shapes are built from it
(`{32, H+SBS_LZ}`, readouts 784/1280/864 at H=128), so no rebuild is needed. The only 256s in the policy source
are CUDA block sizes, not model dims; the one `hidden_size` assert is `SB_PPD`-gated (not set).

These four are **PPO-foundational, provably first-time tests**: the trust region (`clip_coef`), the gradient
clip (`max_grad_norm`) on a trunk where the value tangent dominates 20–400×, the credit horizon (`gamma`), and
network capacity. Deferred to a later round (also never varied, lower priority): `vf_coef=0.25` (varied only
inside the confounded r9-era failures), `num_layers=1` (never varied), `replay_ratio=8` (4 in all 267 runs; the
only outlier is an ARS run at 1), `lambda=0.99` (λ=0.98 in the raw-obs era; λ=1 failed in r36/r59),
`min_lr_ratio=0.05` (r107, incomplete), `ent_coef` levels (swept: 0/0.005/0.01/0.05 — dose-insensitive),
`occ_asinh=1` (never trained; only a parity run). **Deferred, user decision: the r135 "best-of" combination**
(Adam everywhere + the four inert-flag removals) — out of scope for now.

---

## §3 Per-arm analysis — what each score actually says

**#1 r129, desync off — the largest effect in the campaign (0.033 → 2.213).**
Catastrophic at every save, never recovers. Mechanism: with `desync=0` all 256 envs start at the same
Sunday-06:00 phase, so every rollout sees one calendar state and the advantage cannot distinguish seasonal
from behavioural error; the desync cut was also the only source of episode-length diversity. The comparison is
clean: the panel density matches r116 (13.4 vs 13.6 rows per 1M steps). Confirms r35→r37 (5.07 → 0.330) at
the finalized recipe.

**#2 r125, headwise off — the run cannot fix its own zones (0.033 → 0.152).**
It is *ahead* at 20M (0.096 vs 0.126), then plateaus at 0.15–0.18 from 30M while r116 falls to 0.033. That is
exactly the failure r115 predicted: the joint geometric ratio hides opposing per-head changes (a per-head
sign-aware clip would bite 10.11 % of Fairbanks zone-head visits), so the zone heads cannot correct their own
errors. Also the strongest of the two "we already have it" checks: `r116 − headwise` ≠ r108 (0.0706 on the
older `f09f2051` binary). The r116-anchored number is **0.152**, and r108's 0.0706 must not be quoted in the
same table — different binary, and the joint-ratio path is binary-sensitive (r115↔r123 proved the headwise
path is *not*: identical to every digit across two binaries).

**#3 r127, cap off — the hot-month lock returns (0.033 → 0.054).**
Early catastrophe (10M: 2.180 vs 0.273) is the lock itself; the run then recovers but never reaches r116 at any
save. Confirms both the mechanism (the lock) and the panel cost at the finalized recipe. The r93-vs-r94 split
(s75 3/4 saves better, s77 1/4) is resolved: at r116 the cap is uniformly worth having.

**#4 r123, zone MLP off — a late-payoff capacity branch (0.033 → 0.052).**
The arm is *ahead* at 20M (0.082 vs 0.126) and behind at every other save: the 32-unit branch costs early
steps and pays from 30M on. It **exactly reproduces r115** on a different binary (0.492/0.082/0.065/0.055/
0.052 vs 0.492/0.082/0.065/0.055/0.052) — the campaign's cleanest reproducibility evidence.

**#5 r128, role split off with the trunk on Muon (0.033 → 0.049).**
The readouts need Adam when the trunk is Muon: Muon's orthogonalization on 4–12-row readout matrices
oversteps (the r89 finding). This is the control side of the optimizer pair (§3 #12).

**#6 r134, signed deficits (0.033 → 0.044).**
Magnitudes are the right representation: the comfort cost is quadratic in |deficit| and the sign is already
available to the policy through the card columns `(T−Hs)/2` and `(Cs−T)/2`. Worse at 5 of 6 saves. The r99
bundle carried this flag untested; now measured.

**#7 r131, anneal off (0.033 → 0.038).**
Mixed early (better at 20M: 0.074 vs 0.126), worse from 30M. Unlike r94 — where no-anneal ended at 0.338 in a
degenerate corner — this run *held* 0.038–0.052. So at r116 the anneal is a mild, honest win (+15 %), not the
rescue it was in the r94 era. The "anneal selects the basin" caveat stands (r94b s77 held 0.879).

**#8 r124, coherence off — the r84 lever does not transfer (0.031, better at 6/6 saves).**
Removing coherent exploration is **better at every one of the six saves** and 4.3× faster to learn
(10M: 0.063 vs 0.273). The r84 win (0.110 vs 0.199, 3/3 seeds) was measured in the r79-era recipe, where
nothing else persisted an exploratory setting. At r116 two other mechanisms hold the valves: the command echo
(r97 measured the head's own-command weight at +3.545 — the latch lives there) and headwise clipping. The
paper's exploration claim must be re-scoped: **regime-dependent, and gone in the finalized recipe.**

**#9 r126, τ_AHU 48 → 12 — the 4-hour AHU try is a liability (0.031, better at 6/6 saves).**
Small margins, unanimous direction. The 4-hour try was motivated by r87–r89 (AHU credit weaker than gradient
noise); with the cap and headwise in place the short try suffices. Resolves the r93-vs-r92 split in favour of
"no effect or slightly harmful".

**#10 r133, Beta floor 0.5 — the shape claim inverts, within noise (0.031, better at 5/6 saves).**
The r80/r81 verdict (floor 0.5: 0.403 vs 0.199) was measured where the floor decided the cold-basin escape.
At r116 the concentration cap already binds the draw (α+β ≤ 12), so the floor's job is gone. −6 % is inside
the seed spread: report "no measurable cost", not a win.

**#11 r132, critic delay summary off (0.030, better at 6/6 saves).**
The 8 anticipation numbers (minutes-to-comfort, worst deficits, slopes, fan path) do not help. Consistent with
the r52 replay study: the critic ranks *time* well (0.97–1.00) but *rooms* poorly (0.12). Another inherited
default that fails at r116.

**#12 r130, Adam everywhere — Muon is a liability, and this was the best single arm (0.029; superseded by r139's 0.028, §13.2).**
Compared to r128 (one flag: `adam 0→1`) it is better at **all six saves** and by −41 % at the end — far
outside the 6 % seed spread. Compared to r116 it is better at 5/6 saves and −12 % overall (≈2× the seed
spread), and 2.6× faster to learn (10M: 0.106 vs 0.273). Mechanism hypothesis: the value tangent dominates the
shared trunk (measured 20–400×), so Muon's orthogonalization oversteps on value-critical directions while
Adam's per-weight scaling adapts. This **inverts r47** (old binary: Adam lost 5/6 checkpoints) — the Muon
claim was binary- and era-dependent. Wall time unchanged (8,331 s, 6,018 SPS); the dashboard's final-window
"28.3K SPS" is a short-window counter, not the run average.

---

## §4 The recipe verdict, flag by flag

| flag | verdict at r116 | action |
|---|---|---|
| `headwise_ratio=1` | **load-bearing** (+361 % without) | keep |
| `env.desync=1` | **load-bearing** (+6,606 % without) | keep |
| `beta_conc_max=12` | **load-bearing** (+64 % without) | keep |
| `policy.zone_mlp=1` | **load-bearing** (+58 % without) | keep |
| `train.role_adam=1` | **load-bearing with Muon** (+48 % without); obsolete if the trunk goes Adam | keep while Muon |
| `env.zone_signed=0` | **load-bearing** (+33 % signed) | keep |
| `train.anneal_lr=1` | **helps** (+15 % without) | keep |
| `policy.beta_cn_tau=12/48` | **inert to harmful** (−6 % without, 6/6 saves) | candidate for removal |
| `policy.value_delay=1` | **inert to harmful** (−9 % without, 6/6 saves) | candidate for removal |
| `policy.beta_floor=1` | **no measurable cost either way** (−6 %, inside noise) | neutral |
| `train.adam=0` (Muon) | **a liability** — plain Adam everywhere is −12 % vs r116 and −41 % vs Muon | candidate for replacement |
| `policy.beta=1`, canonical, zone_memory, engineered 11, siblings, pool, asinh, adv_norm, value_shared, entropy_current, freeze_return, valve_zone, ratio_per_dim, shuffle | **locked** — see §5 | untested at r116 |

---

## §5 The locked claims — what cannot be tested at r116, and the correct unlock

"Unlock" = a **default-on no-op**: the flag defaults to current behaviour; at 0 the feature is **zero-filled**
(never removed), so widths, parameter counts and every other path stay identical to r116. Guards verified in
`patches/pufferlib_core.diff` (`:5360` headwise; `:4104` diagnostics; `smart_buildings.cu:2650` zone-memory;
`:5078` entropy; `:5144` adam; `:3189` timeout).

| rank | locked claim | lock | correct unlock | value of testing | cost |
|---|---|---|---|---|---|
| 1 | **valve blend** (`valve_zone=0`, 0.70/0.30) | headwise guard requires `valve_zone=1` (`:5362`) | delete **only** the `!g_sbs_valve_zone` term from that abort; the blend formula already exists (`pufferl.cu:1420`) | **the only lever with score upside** — every other credit variant failed, and the r97 probes localise the loss to reheat above the cooling setpoint | 1-line diff + rebuild + gate + 2.4 h |
| 2 | **sibling block (40/zone)** | no flag, compiled in `sbs_local_feature` | `policy.sibling=0` → zero the 40 neighbour slots, keep width | the one input group **measured negative** (r48: 0.290/0.204/0.174 vs 0.134) and never re-isolated | 1 build + 2.4 h |
| 3 | **11 engineered features** | no flag; computed in `zm_update_k` | `env.eng=0` → zero f0–f10, keep the ring (so the next row stays separable) | the largest hand-built input layer; PRIORS P23 "no evidence" | 1 build + 2.4 h |
| 4 | **4-hour memory ring** | `zone_critic` requires `zone_memory=1` (`smart_buildings.cu:2650`) | `env.zm_ring=0` → zero the ring-derived f5–f8 and the f9 slope, keep the block and the zone critic | PRIORS P21 "no evidence" | 1 build + 2.4 h |
| 5 | **building pool (80)** | no flag; `sbs_pool_feature` | `policy.pool=0` → zero the 80 pooled numbers | no prior; hits actor and critic (disclose) | 1 build + 2.4 h |
| 6 | **asinh compression** | no flag (only `occ_asinh`, occupancy-only) | `policy.asinh=0` → pass the same values uncompressed | no A/B; r77's occupancy variant was built, never launched | 1 build + 2.4 h |
| 7 | **adv_norm ≠ 2** | diagnostics `assert(adv_norm==2 && value_shared)` (`:4104`), and headwise needs `update_diagnostics=1` | no code: `adv_norm=0` + `update_diagnostics=0` + `headwise=0` — legal, loses all SB_* telemetry, lands on the 0.152 substrate | A18, never isolated | 2.4 h, 3 flags |
| 8 | **Beta head** (`beta=0`) | headwise requires `beta` | no code: `beta=0` + `headwise=0`, Gaussian through the existing [−1,1] clamp, `SB_ZONE_STATS` off | r79 already answered it (0.1221 vs 0.1784, 1 seed + a 5-replay control) | 2.4 h, 2 flags |
| 9 | **canonical cards** | headwise requires `canonical==1`, and `canonical=0` silently breaks the zone memory (it reads the card) → cascade to `zone_critic` | **no clean unlock** | r1/r2 and ppo_canon numbers are confounded | — |
| 10 | **entropy staleness** (`entropy_current=1`) | requires `tanh_squash=1` (`:5078`) | only with the Gaussian path (rank 8) | r40 audit flagged it; never re-examined for Beta | tie to rank 8 |
| 11 | **freeze_return=1** | headwise guard; the zone critic structurally needs per-visit recompute | none that isolates it | C19/C20 inconclusive | — |
| 12 | **Bernoulli run heads** | headwise guard (`binary_run` in the abort list) | **already run: r95** — 25-scenario Modal batch, truncated ~83M, one save, 15 months, no control | motivation measured (39/43 bad months below the 0.25 threshold); verdict: inconclusive, not adopted | — |
| 13 | **AR(1) latent across episodes** | code, not a flag (dig.md defect) | reset `cn_state` at every terminal | a correctness fix, not an ablation — it changes the recipe and needs its own baseline | 1 h + 2.4 h |
| 14 | **run-bit threshold / mean decode** | decode contract (dig.md Claim 3a/c) | the Bernoulli head is the fix | the deterministic mean executes points the sampler rarely visits | — |

Executing ranks 1–6 in one pass: five default-on zero-fill flags + one guard-term removal, deployed to a new
runtime, one `verify_all.sh --trainer` gate, then five 2.4 h arms on one new binary. Ranks 7–8 are runnable
today with no build (both forced onto the `headwise=0` substrate, whose own cost is now measured at +361 %).

---

## §6 The inherited evidence (pre-r116 campaign) — what the earlier 40 arms established

**Wins with a control (numbers verified):**
| lever | recipe | numbers | caveat |
|---|---|---|---|
| coherent exploration | r79-era | 30–60M 0.199 → 0.113, 3/3 seeds, worst save 0.512 → 0.141 | **superseded: inert at r116 (r124)**; transfer fails out of climate |
| Beta floor 1 | r79-era | 30–60M 0.403 → 0.199, floor 0.5 better on 1/3 | **inverted at r116**: floor 0.5 = 0.031 vs 0.033 (r133) |
| LR anneal | r94-era | 5-save 0.1648 → 0.1071; final 0.338 → 0.040 | **confirmed at r116**: +15 % without (r131), but no degenerate corner this time |
| concentration cap | r93-era | hot-month lock removed; panel 0.256 → 0.177 over 8 cells | **confirmed at r116**: +64 % without (r127), uniformly worse |
| Beta head | r79-era | 0.1784 → 0.1221 (1 seed + a 5-replay control) | **locked at r116** — untestable without losing headwise |
| desync | r37-era | 5.07 → 0.330 (1 seed, 1 month) | **confirmed at r116**: +6,606 % (r129) |

**The graveyard** (each had a control or a gate; all rejected — one line each):

| # | rejected | numbers |
|---|---|---|
| 1 | reward normalization/centering/3-term advantage (r9a, r17, r34) | 3-term 5.2 → 11.6, EV 0.09 |
| 2 | potential-based comfort shaping (r9d) | 1.058 vs 0.327 |
| 3 | comfort guard (r2/r6) | dev gain 0.0446, held-out loss 0.0687 — banned |
| 4 | residual on pid_water (r1/r2) | held-out 0.0687 vs 0.0569 — banned |
| 5 | building rotation / entity agents / shared-v1 (r4/r5/r27) | worse on every split |
| 6 | Gaussian width schedules and pins (r13/r16/r21/r59) | walked the mean off the heat; frozen-σ twin diverged in 10M |
| 7 | squash amplitude 6 (r60) | plateau 0.479, hot-rail re-entry |
| 8 | proportional band + 4K water reset (r61) | gas 417 > 250 MWh gate |
| 9 | λ=1 credit window (r36/r59) | 0.237 not held (0.515/7.618 neighbours) |
| 10 | distillation then 200M (r29b) | gate never passed |
| 11 | ARS weight search (r32) | stopped after two power cuts |
| 12 | full-horizon return (r30) | 9.76 |
| 13 | joint-KL gate (r44) | removed the learning step |
| 14 | PopArt (r42) | month 0.926 vs 0.171; final frozen 7.81 |
| 15 | separate value network (r41) | critic row grew linearly, never scored |
| 16 | value trunk cap (r57) / spike guard (r58) | cap: the ratio stands; guard: bite 0 in 314/325 epochs |
| 17 | Adam on the whole net (r47, old binary) | 0.797 → 9.825 vs Muon 1.525 → 0.561 — **inverted at r116 (r130 0.029)** |
| 18 | IDBD / head_adam (r69/r77) | reducer swapped args, no grid-stride — voided |
| 19 | entropy alternatives (r10/r43/r63/r82/r83) | dose-insensitive collapse |
| 20 | no_cmd_echo (r97) | better at 20M (0.1083 vs 0.1241), worse at 10M/30M — cannot hold |
| 21 | Bernoulli run heads (r95) | 25-scenario Modal, truncated, no control — inconclusive |
| 22 | binary_reheat (r106) | trained, never scored (final panel 0.055) |
| 23 | conditional Beta (r110) | 0.061@20M not held (0.193@30M) |
| 24 | mean/concentration Beta (r111) | 0.0438@30M best, final 0.134, AHU oscillation |
| 25 | per-AHU returns (r90) | comfort worse 8/8 pairs at 30M |
| 26 | per-zone value rows (r98) | 0.0676 best vs 0.0340; regression 0.1296@40M |
| 27 | value Adam (r101) | 1.05 vs 1.17 at 12M |
| 28 | global valve credit (r109) | panel 0.354 vs 0.113, user-stopped 34.65M |
| 29 | night fuel share (r112) | final 0.059 vs 0.0397 |
| 30 | team valve baseline (r114) | residual variance 7.8–9.7× |
| 31 | local value residual (r117) | failed 45M/50M registered gates |
| 32 | 80M budget, refitted anneal (r104) | telemetry only, no frozen months |
| 33 | batch swap train_2→train_7 (r96-light) | stuck, value 2,262 |
| 34 | fp32 solve (r96f) | learns, +21 % SPS — perturbation, not breakeven |
| 35 | minibatch 9216 tile (r96) | KL→0, value 3,521 |
| 36 | zone pass refresh (r118) | final panel 0.0368 vs 0.0331, no frozen score |
| 37 | AHU actor MLP (r119) | user-stopped at 11.6M, panel 1.886 vs 0.273 |
| 38 | random episode start (r120) | invalid measurement (1-in-7 logging); 0.0421 vs 0.0341 |
| 39 | 30-scenario scale-up (r122, Modal) | at 238.9M of 1.5B: Fairbanks 0.0955 vs r116 0.0755 |

---

## §7 Confounded bundles

| bundle | contents | status |
|---|---|---|
| r116 vs r94b | zone_mlp + headwise | **resolved by this ablation set**: each component measured at r116 (r123 =1/6 saves better, r125 =1/6) — the pair is a genuine interaction, not two separate wins |
| r99 "all fixes" | 4 changes at once | rejected as a bundle; never decomposed |
| r8 fix bundle | 6 defects at once | accepted wholesale; never decomposed |
| r17 | squash + reward centering | crashed before attribution |
| r76 | zV fix + valve_zone arm | clean (fix in both arms); only 5 frozen months on the fixed binary |

---

## §8 The paper's claims, updated by this ablation set

| # | claim | what the 12 arms say |
|---|---|---|
| **1** | Coherent exploration was the largest lever | **Re-scope.** True in the r79-era recipe (3 seeds, −43 %); **false at r116** — removing it is better at all six saves (r124, −6 %) and 4× faster early. The lever was regime-dependent: at r116 the command echo and headwise clipping already provide the persistence coherence was added for |
| **2** | The remaining failure is credit-aligned, not exploration-shaped | **Unchanged and strengthened**: r127 confirms the cap's value is exploration-shaped only (+64 % without it); the credit problem (r97 probes, −8.5 % headroom in reheat-above-cooling) is still the open loss, and the blend arm that addresses it is guard-locked (§5 rank 1) |
| **3** | Decode defects are measured but unfixed | **Unchanged**: run-bit threshold, joint-ratio hiding (now *fixed* — headwise is load-bearing, +361 %), mean-decode. The joint-ratio defect is the campaign's first decode fix that survived an ablation |
| **new** | **The optimizer** | **New claim**: plain Adam on everything beats Muon+role-Adam (r130 0.029 vs r116 0.033, and vs r128 0.049 one flag apart). Muon was a liability on the finalized recipe |
| **new** | **Single-binary ablation discipline** | The 12-arm set is one binary, one seed, six matched saves per arm — the measurement regime the earlier campaign lacked |

---

## §9 What to do next

**Round 3 (rows 1–4 of §2.1) is COMPLETE (2026-11-08 08:34 IST):** all four never-varied knobs measured to
50.135M, 10 ckpts each, rc=0. Results: `clip_coef=0.1` **+55 %** (trust region 0.2 was right), `max_grad_norm=1.0`
**+3 % (neutral)**, `gamma=0.995` **+15 %** (learns faster early, converges worse — γ=0.99 right),
`hidden_size=128` **−6 % (capacity is not the constraint; half-width is fine)**. Both GPUs are now free.

| # | work | status |
|---|---|---|
| 1 | **`r135` "best-of"** (Adam everywhere + the four inert-flag removals combined) | **DEFERRED — out of scope now** (user decision 2026-11-08). The five components are each measured individually (§2 rows 8–12) |
| 2 | **valve blend with the guard relaxed** (§5 rank 1) | pending user decision — the only lever with upside; needs the one-line guard term removal + rebuild + gate |
| 3 | **the locked input arms** (§5 ranks 2–6): sibling off, engineered off, ring off, pool off, asinh off | pending — needs one build (five default-on zero-fill flags) + one gate |
| 4 | **scoring** | **in progress elsewhere** — excluded from this plan |
| 5 | **deferred flag arms** (never varied, lower priority): `vf_coef=0.25`, `num_layers=1`, `replay_ratio=8`, `lambda=0.99`, `min_lr_ratio`, `occ_asinh=1`, `ent_coef` levels | queued for a later round after rows 1–4 land |

One caveat to carry: **single-seed wins of −6…−12 % are at or near the seed spread** (r116's own seeds differ by 6 %). The arms that dominate at all six saves (r124, r126, r132: 6/6; r130: 6/6 incl. vs r128) are much stronger evidence than their final delta alone — use the "saves ≤" column, not just the last number.

---

## §10 Provenance

- Every number in §2–§3 parsed from `results/runs/<arm>/train.log` (panel per epoch, steps per box) and the
  driver logs (`logs/queue*.log`: START/END/rc/elapsed/steps-per-s/ckpts per run).
- Every arm's args token-diffed against r116's stored `args.txt` before launch; the diff is the run paths plus
  exactly the listed flag(s). Launch banners grepped to confirm the change took (e.g. "Adam on", "deficits
  signed", no "role_adam on" for r130; no coherence banner for r124).
- Guards cited from `patches/pufferlib_core.diff` and `gpu/*.cuh|cu` with line numbers in §5 and §12.
- Interrupted stubs preserved, never deleted: `r128_…_interrupted_cudagraph` (CUDA-graph instantiate failure,
  transient — the identical relaunch succeeded), `r130_…_interrupted_stopped_20261107` and
  `_interrupted_powercut_20261107`, `r134_…` likewise. Two power cuts (12:05 IST Nov 7) and one user stop
  (06:11 UTC Nov 7) cost ≈2 h and four partial restarts; every restart is a fresh from-scratch run.
- TensorBoard: `results/tb_view` (104 runs, arms r76+) served on port 6006; panel tables:
  `study_r123_r129_final_recipe_ablations/compare_panels.py`.
- Older sections of this file (verification sweeps §6/§8/§10, the M1–M10 plan §5, the §12 plan) are merged into
  the entries above; nothing was dropped except duplicated process notes. The full sweep detail remains in
  git history of this file.

---

## §11 The flag census (2026-11-08) — proof of what ran, what never ran, and what is next

Method: every `--flag=value` token from all **335 stored `args.txt`** plus **every `command.txt`** launch line
and every study `launch.sh`/`run.py`, aggregated per flag. "Never varied" = only r116's default value appears
in any launch anywhere in the repo (including the Modal configs of r95/r122 and the raw-obs-era runs whose
`args.txt` predate the flag — those 69/81 "absent" runs used the old binary's compiled default, which is the
same 0.2/0.5/256/2). One flag per arm, one seed: 12 of the candidate flags were already settled by §2; the
census below covers the rest, so nothing is recommended blind.

### 11.1 Candidates that WERE run before — excluded from the new round

| flag | values ever launched | where | verdict already in hand |
|---|---|---|---|
| `train.gamma` | **0.99** (430) · **0.999 × 37** · **0.9966 × 4** | 0.999 only in the r7–r9 raw-obs era (every one of those runs failed on raw inputs, not on γ); 0.9966 is the r33/r34 action-repeat compensation | modern stack never tested → **r137 launched**; family has an era-confounded prior |
| `train.gae_lambda` | **0.95** (263) · **0.98 × 36** (same raw-obs era) · **1.0 × 9** (r30/r36/r59) · 0.999 × 1 | λ=1: **measured failure** (0.237 not held) | λ=0.99 deferred; negative family prior at λ=1 |
| `train.vf_coef` | **0.5** (267) · 1.0 × 3 (r9a/r9d) · 0.25 × 1 (r9c) | every non-0.5 value sits inside the confounded r9-era failures | no clean verdict exists → **vf_coef=0.25 deferred** |
| `train.ent_coef` | **0.001** (240) · 0 × 21 · 0.005 × 1 · **0.01 × 5** (r11, r63) · **0.05 × 4** (r10/r30/r31/r33) | swept across eras | dose-insensitive collapse (r63/r82/r83) → excluded |
| `train.min_lr_ratio` | 0.05 × 2 (r107) | r107 stopped at 25.07M, 5 of 12 saves | partial → deferred |
| `train.shuffle` | **=1 in 58 runs** — the old trainer's default | now headwise-locked | excluded |
| `train.adv_norm` | **=1 × 17, =0 × 2** (old era) | tested early → rank 7 in §5 | excluded from the new round |
| `env.prestart_explore` | =1 × 2 (old era) | tested → has the same episode-logging bias as r120 | excluded |
| `policy.occ_asinh` | =1 × 1 (r77_parity_occ — a parity run, never trained) | never trained → deferred (dig.md #6) | deferred |
| `train.timeout_bootstrap` | =1 × 2 | **inert** at r116 (`component_values=0` path, `:3189`) | excluded |
| `train.head_adam` / `value_adam` / `joint_kl` / `lastlayer_stepsize` / `room_value` / `zone_adv` / `value_trunk_cap` / `value_spike_guard` / `binary_run` / `binary_reheat` / `beta_conditional` / `beta_mean_precision` / `zone_command` / `zone_night_share` / `valve_global` / `zone_value_residual` / `ahu_mlp` / `zone_pass_refresh` / `random_start` / `comfort_shape` / `rew_norm` / `rew_center` / `residual` / `freeze_logstd` / `logstd_*` / `init_logstd` sweep / `plant_pair` / `zone_signed` | each had at least one dedicated arm; **all measured and rejected** (§3-B and §2) | — |

### 11.2 Flags NEVER varied anywhere — the true first-time tests

| flag | default-only across | now |
|---|---|---|
| `train.clip_coef` | **0.2 in all 424 launches** | ✅ r135 done: **0.051 (+55 %)** |
| `train.max_grad_norm` | **0.5 in all 425 launches** | ✅ r136 done: **0.034 (+3 %, neutral)** |
| `policy.hidden_size` | **256 in all 490 launches** | ✅ r138 done: **0.031 (−6 %; capacity not the constraint)** |
| `train.replay_ratio` | **4 in all 267** (one ARS outlier at 1) | deferred |
| `policy.num_layers` | **2 in all 255** | deferred |
| `policy.value_shared`, `env.canonical`, `env.zone_memory`, `env.reward_scale`, `env.zone_signed(done)`, `policy.beta(done)`, `train.ratio_per_dim`, `train.rew_clamp`, `train.linear_lr_anneal(done)` | never varied **and** guard-locked or contract-level (§5) | locked |

### 11.3 The next-round queue, in order

1. **complete:** r138, r135, r136, r137 (round 3, all four done 2026-11-08 08:34 IST).
2. **deferred flags:** `vf_coef=0.25`, `num_layers=1`, `replay_ratio=8`, `lambda=0.99`, `occ_asinh=1`.
3. **needs a build:** §5 ranks 1–6 (blend guard relax + five zero-fill flags), then their arms.
4. **scoring:** in progress elsewhere — not planned here.

---

## §12 Findings behind the numbers — the mechanisms, the baselines, and the strong claims (2026-11-08)

Everything here is either re-derived from a stored artifact or explicitly marked as *inferred*. The bin numbers
come from the live per-time-of-day table in `review_r116_20261105/analysis.txt` (columns: productivity, energy,
carbon, comfort K, out-of-band fraction, electricity kW, gas kW, samples), re-parsed for this document.

### 12.1 The Beta head — yes, with two caveats (Q1)

| evidence | numbers | caveat |
|---|---|---|
| r79 vs the r76 arm | best month **0.1221 vs 0.1784 (−32 %)** | 1 seed; the control is a 5-replay diagnostic, not a matched run |
| mechanism (r78) | Gaussian+tanh pinned to the rails; plant heads off the rails from 40M. Under Beta, reheat tracks its own temperature at corr −0.94…−0.98 | — |
| modern value | `headwise_ratio` exists **only** on the Beta head; headwise is r116's largest non-desync lever (**+361 %** when removed, r125) | indirect: the head cannot be ablated at r116 (`:5360` requires `beta`) |

**Verdict:** Beta helped where it was introduced, and it is the substrate of the recipe's biggest lever. The
direct head-to-head is single-seed and weak; the modern claim must be stated indirectly.

### 12.2 The morning bin — what the zone MLP actually fixed (Q2)

Fairbanks, live bin medians, **40–50.136M** (r116 = r115 + zone MLP, so the difference is the MLP):

| bin | r115 (no MLP) | **r116 (+ MLP)** | change |
|---|---|---|---|
| weekday morning | prod −0.316 · comfort **1.61 K** · OOB **0.65** · gas 460 kW | prod −0.158 · comfort **0.81 K** · OOB **0.45** · gas **373 kW** | **comfort −50 % with 19 % less morning gas** |
| weekday afternoon | −0.274 · 1.39 K · 0.64 · 435 kW | −0.081 · 0.41 K · 0.33 · 469 kW | comfort −71 %, gas +8 % |
| weekday evening | −0.301 · 1.52 K · 0.65 · 376 kW | −0.145 · 0.74 K · 0.44 · 331 kW | comfort −51 %, gas −12 % |
| night (empty) | prod 0 · 3.72 K (unpunished) · gas 59 kW · elec 2.1 kW | prod 0 · 3.73 K · gas **66 kW** · elec **1.8 kW** | the plant is **not off** at night |

- The morning was the **worst occupied bin**, not the only losing one (all three occupied bins are negative at
  r115); the morning carries the largest gas draw and the highest OOB.
- The gain appears at 30M+ (20–30M bins are unchanged), matching the MLP's measured late payoff (r123 ahead at
  20M, behind after).
- **What did not do it:** the critic's 8 anticipation numbers (r132: removing them is **9 % better**), and the
  engineered morning features (minutes-to-comfort, arrival deficit, ramp) — designed exactly for this, **locked**,
  so their contribution is unmeasured (P23).

**Verdict (strong):** the morning catch-up is per-zone capacity — the zone MLP halves the morning comfort
deviation and the morning out-of-band share while using *less* morning gas. The night runs a small plant
(66 kW gas, 1.8 kW electricity) precisely to buy that morning.

### 12.3 Cold basins and dips — the runs, and what prevents them at r116 (Q3)

| failure | runs | measured signature |
|---|---|---|
| **cold basin** | r79/r80/r81 (all 8 Beta runs) | entropy < −100 by 5–12M; **86–96 %** of bins in the cold basin; gas 156 → 12 kW; escapes only at 20M/50M |
| **fixed by** | **r84** coherent exploration | cold bins **5 vs 23**; 30–60M **0.110 vs 0.199**, 3/3 seeds |
| **20M dip** | r85_s77 | phoenix_6 0.479, atlanta_6 0.216 at 20M |
| **30M recirc collapse** | r90 | recirc 0.84/0.87 → **0.32/0.24**; comfort worse 8/8 pairs |
| **hot-month lock** | r93 (pre-cap) | phoenix_6 0.696, atlanta_6 0.448 at 40M |
| **late degenerate corner** | r94 s75 | final 0.338; pressure 1.9 kPa + recirc 0.96 + water 348 K + run 100 %; 100–360 MWh elec |

At r116 **none of these recur**, and coherence is not the reason (r124 −6 %, better 6/6 saves). The measured
carriers, in order: **desync** (without it a 2.2 plateau all run: r129), **headwise** (without it a 0.15–0.18
plateau: r125), **the cap** (without it a hot lock at 10M: r127), **zone MLP** (r123), **the anneal** (r131).
Three distinct failure basins — 2.2 (desync), 0.15 (headwise), hot-spike (cap) — which is direct support for
the paper's Claim 2: the remaining failures are *credit/estimator-shaped*, not exploration-shaped.

### 12.4 The starting point versus the final recipe (Q4)

| | score | regime |
|---|---|---|
| plain PPO on the scaled canonical obs, absolute actions, residual off (**`ppo_canon`**, 19.9M steps) | **0.651** vs pid_water **0.0549** — **0/30 scenarios better**; gen-val 0.516 vs 0.0569; gen-val-s 0.594 vs 0.0492 | 30 dev scenarios |
| r1_final / r7_final (same week) | 0.273 / 0.568 | 30 scenarios |
| **r116 (final recipe)** | **panel 0.0330 (s75) / 0.0350 (s76)**; frozen 4-scenario **0.0305**; Fairbanks **0.0755** | training panel / 4 scenarios |
| **r130 (best arm)** | **0.029** | training panel |

**The honest framing:** the unmodified PPO is **≈12× worse than the shipped pid_water controller** on 30
scenarios; the campaign's final recipe is at 0.0305 on four training scenarios, and its 30-scenario number is
still being scored (different regime — do not divide 0.651 by 0.0305 and call it a ratio).

### 12.5 Premature convergence, step sizes, the 0.05 ceiling, and r94b's night electricity (Q5)

**(a) Preventing premature convergence.** No single switch. Measured at r116: desync gives every rollout
several calendar phases (without it: **2.213**); the cap keeps the sampled support from collapsing (without it:
**0.054**); headwise lets each head correct itself (without it: **0.152**); the anneal stops the late walk
(without it: **0.038**); entropy is a small constant — every entropy-shaping alternative was refuted
(r10 0.05, r63 anneal, r82 target pinned at the 0.05 cap, r83 per-role).

**(b) Step-size optimization in r116.** Three layers: **role Adam** on the readouts (zone 1296 + AHU 2304 +
plant 1632 + the zone MLP 10,368 + 128 at H=256), **Muon** on the trunk, and a global `max_grad_norm 0.5` plus
the linear LR anneal. Explicit per-feature step sizes (IDBD/SwiftTD, P28) were tried as r69 and **voided by a
reducer bug** — not in r116. The decisive step-size result is **r130**: plain Adam everywhere beats
Muon+role-Adam by **41 %** one flag apart (0.029 vs 0.049) and is also −12 % against r116 — the hand-rolled
optimizer stack was the liability, and Adam is free (6,018 vs 6,069 SPS).

**(c) The 0.05 ceiling — both readings, both true.**
- *Entropy cap:* r82's target arm wound up to 1e-6 and **pinned at the 0.05 cap**, holding entropy with
  near-uniform dampers while pump/water stayed cold — it failed, and the ceiling was abandoned in favour of a
  fixed `ent_coef=0.001`, not broken.
- *Score plateau:* the level sat at ~0.05 for the r85–r115 era (r115 0.052, r94b 0.046, r85 0.063–0.069,
  r110 0.061). It was broken by **r116's pair (0.033)** and then by **r130 (0.029)** — the first runs to hold
  below 0.04 at the final save.

**(d) r94b's night electricity — how it "then solved".** The night operation is a **rational trade**, not a
defect: productivity is 0 when the building is empty, so night fuel pays only its energy+carbon share while it
buys morning comfort (bins: night gas 59–66 kW buys morning comfort 0.81 K instead of 1.61 K). What r94b fixed
is the **degenerate** version of that trade — r94's 50M save walked into pressure 1.9 kPa + recirc 0.96 +
water 348 K + run 100 %, burning 100–360 MWh of electricity in hot months. The sign audit showed that corner
was **credit-aligned** (the applied step agreed with the advantage sign in **70 %** of audited cells): the
reward was not the bug, the *walk* was. The linear anneal stopped the walk; the final checkpoints settled into
a per-weather re-balanced point (warm months water 297/294 K, SAT 281/278; cold months water 310 K, recirc
0.59/0.60; no head pinned at the cap).

### 12.6 Additional findings from the 16-arm set (beyond the Q&A)

1. **The five inert removals cluster at 0.030–0.031** (r124, r126, r132, r133, r138) against r116's 0.033 —
   five independent defaults each cost ~3–9 %, and they do **not** overlap in mechanism (exploration sampler,
   AHU timescale, critic input, Beta shape, trunk width). Strong: the recipe carried ~5 independent dead weights.
2. **The 6/6-saves criterion separates signal from noise.** r124/r126/r132/r130 dominate r116 at *every*
   matched save; r133 (5/6) and r136 (3/6) are inside the 6 % seed spread. A final-delta-only reading would
   have called r133 a win and r136 a null — the dominated-saves count is the better statistic.
3. **Capacity is not the constraint.** `hidden_size=128` (half the trunk) is **−6 % at 50M and better at 4 of 6
   saves** — halving the network costs nothing. With 453.7K parameters at H=256, the policy is not parameter-limited; the loss is credit/estimator-bound.
4. **The trust region is well-tuned and untouchable.** The first-ever `clip_coef` test (0.1) is the third-worst
   arm (+55 %); `max_grad_norm` 1.0 is neutral. The PPO defaults in these two spots were right by inheritance.
5. **γ trades early speed for late quality.** γ=0.995 learns 1.7× faster at 10M (0.158 vs 0.273) and loses from
   40M on (0.042 vs 0.036). A monotone "longer horizon is better" claim is false here.
6. **Adam's advantage is early-learning speed**, not just the plateau: 2.6× at 10M (0.106 vs 0.273) and 6/6
   saves vs both r116 and r128.
7. **The joint-ratio path is binary-sensitive.** r116 − headwise = 0.152 on `adbe6911` but r108 (the same
   recipe) = 0.0706 on `f09f2051`; the headwise path reproduces to every digit across binaries (r115↔r123).
   Cross-version comparisons of headwise=0 are unsafe; the r116-anchored number is the only one to quote.
8. **The r116 recipe is roughly half-measured**: 7 flags load-bearing, 4 inert/neutral, 1 liability (Muon), and
   14 more locked. The largest lever (headwise, +361 %) is also the newest (r115) — the recipe's ceiling was
   raised by the last two changes, not by the long lineage.
9. **The anneal shrank into insurance.** In the r94 era, removing it ended in a degenerate corner (0.338 at
   50M, −88 % with it). At r116, removing it (r131) *held* 0.038–0.052 and cost only 15 %. The fix becomes less
   important once the failure it was added for is gone — a "fix becomes insurance" pattern.
10. **Most improvements act on learning speed, not the floor.** At 10M: r130 0.106, r124 0.063, r126 0.223,
    r132 0.272 vs r116 0.273; only r123 (the zone MLP) is *slower* early (0.492) and pays late. The recipe's
   wins are largely earlier convergence to the same basin — relevant for budget planning, not just for level.
11. **Failure modes differ in recoverability.** The cap's damage is early and partly recoverable (r127: 2.18 at
   10M → 0.054 at 50M); headwise's is a hard plateau (r125: 0.15–0.18 from 30M); desync's never recovers
   (r129: 2.08–2.29 throughout). That ordering — transient / plateau / no-learning — is a diagnostic for
   future arms: a run that lands at ~0.15 is a headwise-class failure, one that stays >2 is a calendar failure.
12. **The frozen month is slightly *better* than the panel**: r116 panel 0.033 vs frozen 4-scenario 0.0305.
    The changing-policy panel is a conservative estimate of the deterministic policy, so panel verdicts do not
    overstate frozen performance.
13. **Hardware caveat, untested by design.** r124, r134 and r138 ran on the 3060; every baseline and the other
    9 arms ran on the 3090 — same binary, same sm_86 architecture, but no cross-card parity run exists (the
    parity rule forbids one). Their verdicts are consistent with their neighbours and r123↔r115 proves
    cross-*version* determinism for the headwise path, but the 3060/3090 equivalence is an assumption, not a
    measurement. Quote it whenever those three arms are cited.

### 12.7 The strong claims that survive (paper-ready)

1. **Coherent exploration is regime-dependent, not a law**: +3 seeds of evidence in the r79 recipe, **zero**
   value in the finalized recipe (removal is better at 6/6 saves). Any paper claim must carry the regime.
2. **Per-zone capacity is the morning fix**: the zone MLP halves the morning comfort deviation (1.61 → 0.81 K)
   and the morning out-of-band share (0.65 → 0.45) with *less* morning gas — measured in the time bins, not just
   the panel.
3. **The optimizer was the liability**: Muon+role-Adam is 41 % worse than plain Adam one flag apart, and Adam is
   also 12 % better than r116 while learning 2.6× faster early.
4. **Capacity and the PPO knobs are not the bottleneck**: half-width is neutral-to-better; the trust region and
   gradient clip are untouchable; γ is already right. The open loss is credit (r97 probes: −8.5 % in
   reheat-above-cooling), and the arm that addresses it (valve blend) is guard-locked.
5. **Failures are shaped by the estimator, not the explorer**: three distinct basins appear when desync,
   headwise, or the cap is removed — and each is a *credit/optimization* shape, consistent with the r94 audit
   (walk follows the advantage sign in 70 % of cells).

---

## §13 The ranked confirmations in r116, and the best-of run (2026-11-08 09:20 IST)

### 13.1 What is confirmed in r116, ranked by measured effect

All numbers are the final training panel at 50.135M against r116 s75 (0.0330), one binary, one seed, one flag
(or the documented pair). "Δ" is the score change caused by the change named in the row.

| rank | confirmed finding | flag | Δ | evidence | **right option** |
|---|---|---|---|---|---|
| 1 | **desync is the calendar of learning** | `env.desync=0` | **+6,606 %** (2.213) | r129, 0/6 saves | **keep 1 — non-negotiable** |
| 2 | **headwise per-head clipping is the correction mechanism** | `headwise_ratio=0` | **+361 %** (0.152) | r125, 1/6 | **keep 1** |
| 3 | **the concentration cap keeps the search alive** | `beta_conc_max=0` | **+64 %** (0.054) | r127, 0/6 | **keep 12** |
| 4 | **per-zone capacity is the morning fix** | `zone_mlp=0` | **+58 %** (0.052) | r123, 1/6 (bins: morning comfort 1.61→0.81 K) | **keep 1** |
| 5 | **the trust region is right at 0.2** | `clip_coef=0.1` | **+55 %** (0.051) | r135, 1/6 — first-ever test | **keep 0.2 — do not tighten** |
| 6 | **the role-Adam split is required while the trunk is Muon** | `role_adam=0` | **+48 %** (0.049) | r128, 0/6 | **keep 1 only with Muon; drops out under Adam (r130)** |
| 7 | **deficit magnitudes are the right representation** | `zone_signed=1` | **+33 %** (0.044) | r134, 0/6 | **keep 0 (magnitudes)** |
| 8 | **the linear anneal still helps (insurance)** | `anneal_lr=0` | **+15 %** (0.038) | r131, 1/6 | **keep 1** |
| 9 | **γ=0.99 is the right horizon** | `gamma=0.995` | **+15 %** (0.038) | r137, 3/6 (faster early, worse late) | **keep 0.99** |
| 10 | **Adam beats Muon for the whole net** | `adam=1`+`role_adam=0` | **−12 %** (0.029) | r130, 6/6 vs r116 **and 6/6 vs r128 (−41 %)** | **adopt — replace the Muon stack** |
| 11 | **the critic delay summary is dead weight** | `value_delay=0` | **−9 %** (0.030) | r132, 6/6 | **remove (set 0)** |
| 12 | **coherent exploration is inert here** | `beta_cn_tau=0`+`beta_cn_tau_ahu=0` | **−6 %** (0.031) | r124, 6/6 (4.3× faster at 10M) | **remove (set 0/0)** |
| 13 | **τ_AHU=48 is inert/harmful** | `beta_cn_tau_ahu=12` | **−6 %** (0.031) | r126, 6/6 | **remove (0, or 12 if coherence stays)** |
| 14 | **the Beta floor is neutral here** | `beta_floor=0.5` | **−6 %** (0.031) | r133, 5/6 (inside noise) | **optional — 0.5 if a change is wanted, 1 otherwise** |
| 15 | **capacity is not the constraint** | `hidden_size=128` | **−6 %** (0.031) | r138, 4/6 | **optional — 128 is free; keep 256 for comparability** |
| 16 | **the gradient clip is not sensitive** | `max_grad_norm=1.0` | **+3 %** (0.034) | r136, 3/6 (neutral) | **keep 0.5 — no reason to move** |

Rows 1–8 (+9) are the recipe's load-bearing settings; rows 10–16 are settings the recipe paid for and did not
need. The **6/6-saves** column is the robust statistic: rows with 6/6 dominate r116 at every matched save, which
is outside the 6 % seed spread; 5/6 and 3/6 are inside it.

### 13.2 `r139_bestof_s75` — the best-of combination (launched 09:20 IST, 3090)

The five mechanisms that beat r116 individually, combined into one arm (r139, since r135 was taken by the clip
test). One mechanism is the optimizer swap; the other four are the inert removals:

| component | measured alone | combined flag |
|---|---|---|
| optimizer | −12 % (r130) | `train.adam=1` + `train.role_adam=0` |
| coherence | −6 % (r124) | `beta_cn_tau=0` + `beta_cn_tau_ahu=0` (the second is forced) |
| critic delay | −9 % (r132) | `policy.value_delay=0` |
| Beta floor | −6 % (r133) | `policy.beta_floor=0.5` |

**Pre-registered prediction:** if the individual effects multiply, 0.0330 × 0.94 × 0.94 × 0.91 × 0.88 =
**≈0.022**. Beating r130's 0.029 would confirm additivity; landing above 0.033 would mean the five share a
mechanism (each measured alone but not independent).

**Result (2026-11-08 11:34 IST, rc=0, 8,110 s, 6,182 SPS, 10 ckpts): the best run in the campaign.**

| metric | r139 | r130 (best single) | r116 (baseline) | Δ vs r116 | Δ vs r130 |
|---|---|---|---|---|---|
| **final panel (50.135M)** | **0.0280** | 0.0290 | 0.0330 | **−15.2 %** | **−3.4 %** |
| **final live `env/score`** (last epoch) | **−0.0189** | −0.0206 | −0.0249 | **−24.1 %** | **−8.3 %** |
| saves beating r116 | 5/6 (only 10M worse) | 6/6 | — | — | — |
| saves beating r130 | 5/6 | — | — | — | — |

Held, not a spike: the last five panel readings are **0.026 / 0.027 / 0.027 / 0.028 / 0.028**. The
combination is **sub-additive** — 0.028 against the additive 0.022 — so the five mechanisms overlap (Adam and
coherence-off both act on early learning speed, which is why the 10M save is the one place r139 trails r116),
but no component hurt. **Verdict: adopted as the best measured configuration** (single seed; the solid claim is
−15 % against r116 at 5/6 saves, not the 0.028 level itself).

Verified at launch: the args token-diff against r116 is exactly the six tokens above; banners confirm
**Adam on**, no `role_adam`/`head_adam`, no coherence banner, headwise and zone critic still on; `beta_conc_max=12`,
`zone_mlp=1`, `headwise_ratio=1` unchanged. Guards all pass (adam forbids role_adam; tau=0 forces tau_ahu=0;
floor 0.5 needs cap > 1). Speed as expected (~6K SPS). ETA ~2.3 h.

### 13.3 What r139 is, and what it is not

**Adopted configuration (all r116 flags except these six tokens):**

`train.adam=1`, `train.role_adam=0`, `policy.beta_cn_tau=0`, `policy.beta_cn_tau_ahu=0`,
`policy.value_delay=0`, `policy.beta_floor=0.5`.

Still on: Beta head, headwise clipping, zone MLP, cap 12, canonical, zone memory, zone critic, valve-zone
credit, desync, shared value trunk, single scalar return, clip 0.2, vf_clip 1e4, grad-clip 0.5, γ 0.99,
λ 0.95, minibatch 2304, horizon 288, replay 4, entropy 0.001, linear anneal to 50.135M.

**Advantages it carries (measured):**
1. **Best final panel and best final live score of the campaign** — 0.0280 / −0.0189 against r116's 0.0330 /
   −0.0249 and r130's 0.0290 / −0.0206.
2. **3× faster to a good level**: 20M panel 0.042 vs r116's 0.126; it reaches at 20M what r116 reaches at ~35M.
3. **Simpler optimizer state**: plain Adam everywhere — no Muon orthogonalisation, no role-Adamsplit, no head
   Adam sidecar, no Muon momentum; same wall-clock speed (6,182 vs 6,069 SPS).
4. **Fewer inputs and a wider legal shape range**: the 8 critic delay numbers are gone and the Beta shapes may
   go below 1 (floor 0.5).
5. **Held**: last five panel readings 0.026/0.027/0.027/0.028/0.028, no spike, and 5/6 saves better than both
   r116 and r130.

**What it is not (do not overclaim):**
- It is **one seed**. Against r116 the gap (−15 %, 5/6 saves) is outside the 6 % spread; against r130 (−3.4 %)
  it is inside it. The honest claim is "best measured, clearly better than r116, not separable from r130".
- It has **no recording** (`SB_RECORD` was not set), so no per-time-of-day bins and no action-level analysis
  exist for it. The morning/night behaviour of §12.2 belongs to r116/r115, not to r139.
- It is **not a frozen score**: the panel is a changing-policy training read. Any leaderboard or 30-scenario
  claim needs the scoring pass (in progress elsewhere).
- The five removals are **sub-additive**: 0.028 against the additive 0.022. If a sixth change is added, expect
  further overlap.

---

## §14 The Modal scale-up: r139 @ 200M (launched 2026-11-08 12:16 IST)

The best recipe (§13.2) at the 30-scenario / 960-agent scale that r122 proved, with the budget
cut to 200M and annealed to zero. Account `sid250581`, one L40S, detached.

| | value | verified how |
|---|---|---|
| args | 80 tokens, **13 changed** from r122 | token diff; worker `startup.json` says `args identical to study dir: True` |
| budget / anneal | 199,895,040 steps = 723 epochs (276,480/epoch), linear anneal to 0 | 723 × 276,480 |
| checkpoints | every 144 epochs → 39.8/79.6/119.4/159.3/199.1M + the final 199.9M | args |
| batch | 30 scenarios, 960 agents, horizon 288, minibatch 2304, replay 4, cudagraphs 1 | identical to r122 |
| binary / data | `f761da99…` / `492c483e…` (1,597,565,970 B) | hashes in `startup.json`; the worker re-verifies both at startup |
| speed | **≈13.8K steps/s** (epochs 12→15 in 60 s at 12:23 IST) | app log |
| ETA / cost | ≈4.0–4.5 h; ≈$11 at $2.42/h | measured rates |

**Three bugs found before this launch, all fixed and verified:**
1. **`full_checkpoint=1` is impossible for r139.** `sb_full_supported()` requires
   `role_adam && beta_cn_tau>0 && !adam` (`sb_full_trainer.cuh:155-165`); the trainer aborted with
   `complete checkpoint unsupported for this configuration` (rc −6). Fixed with
   `full_checkpoint=0` + `portable_resume=0`, **verified by a local run of the same binary (rc 0)**.
2. **`full_checkpoint=0` prints no save banner**, so r122's log-triggered publishing would never
   fire. Fixed with a directory watcher (two stable scans + a final sweep), **verified offline on
   the real smoke saves** (2 snapshots, sha256/bytes matched, idempotent).
3. **`role_adam=0` writes no `.hadam`** (`:5788`); r122 hard-coded four suffixes. The component
   list is now dynamic. A local smoke confirmed the save set is exactly
   `.bin` (1,812,032 B) + `.bin.mb` (3,624,068 B) — no `.hadam`, no `.full` — so ~5.4 MB per
   snapshot instead of r122's 14.5 GB.

**Consequence for the paper's training section:** the flagship run is now *r139*, not r122 — the
final recipe with the optimizer fix, the 200M annealed budget and the leaner checkpoints. r122
(1,500.2M planned, stopped at 507.6M by credit) remains the scale-up evidence, with its best
Kaggle row **0.01818 at 89,579,520** and degradation after.
