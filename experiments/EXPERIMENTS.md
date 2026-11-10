# Experiments

The policy learns only by acting in the environment. Do not put a competition baseline into the action: no residual on `pid_water`, no action head initialized so the policy starts as that controller, no comfort guard, no comfort shaping. Those baselines are scores to beat. The public target is **0.00598** on the Kaggle leaderboard (updated by the user 2026-10-19), and the policy has to generalize past the scenarios it trained on.

The trainer is PufferLib 5.0 PPO (`./puffer train`). Scores below are the leaderboard metric (mean over scenarios of the absolute mean per-step reward; lower is better), from the fp64 binary, full 8640-step episodes.

## r10 — one building, one weather (finished 2026-10-18)

2026-10-18. `building_train_1` with `fairbanks_1` only. 256 environments of that scenario. Absolute canonical actions (`canonical=1`, `residual=0`, `zero_init_actions=0`). The learning reward is the environment reward (`rew_norm=0`, `reward_scale=1`, `comfort_shape=0`, `rew_clamp=0`, `action_repeat=1`).

|                             |                                                                          |
| --------------------------- | ------------------------------------------------------------------------ |
| learning rate               | 0.0004, constant (`anneal_lr=0`)                                         |
| entropy coefficient         | 0.05, constant (`anneal_ent_coef=0`)                                     |
| discount, GAE               | 0.99, 0.95                                                               |
| clip, value coef, grad clip | 0.2, 0.5, 0.5                                                            |
| epochs, horizon, minibatch  | 4, 256, 2048                                                             |
| advantages                  | standardized once over the rollout (`adv_norm=2`), minibatches shuffled  |
| per-head ratio              | on                                                                       |
| steps                       | 50,000,000 requested (the trainer runs whole epochs: 256×256 steps each) |
| solve                       | fp32 on the 3090 (`SB_F32=1`). Scores use the fp64 binary.               |

One update: roll out 256×256 transitions, compute GAE, standardize advantages over that rollout, then four passes of shuffled minibatches of 2048.

The logged entropy is the mean, over the rollout, of the sum of per-head entropies. At the initial log-std of −1 that sum is on the order of a few tens of nats for this building's active heads, so 0.05 times it can outweigh the policy loss. It stays at 0.05 for this run. Watch `loss/entropy` against `loss/policy`. If the bonus still dominates after the score has stopped falling, the next run anneals it.

While this trains, the 3060 scores each new checkpoint on `fairbanks_1` (the scenario it is learning) and on `atlanta_2` of the same building (a weather it has not trained on). After the run, the 3090 scores gen-val and the 3060 scores gen-val-s. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r10`).

A falling score on fairbanks_1, and a score that is not nonsense on atlanta_2, is what this run is for. 0.00603 is the destination after the same policy can be trained past this one scenario.

### Result (2026-10-18)

Finished 49.9M steps in 56 min 1 s. Final weights `results/runs/r10/ckpt/0000000049938432.bin`.

The training score and the training productivity curve are the same number. Reward = 0.6 × productivity − 0.2 × energy − 0.2 × carbon, and 0.6 × productivity is the score to within 0.05 at every checkpoint. Energy's largest move changes the reward by 0.013.

On the training episodes the comfort gain at 30M (mean deviation 31 K → 22 K, score −8.2 → −3.8) arrived with gas use rising from 13 to 69 thousand kWh per episode. By 50M both had given ground (deviation 27 K, gas 44 thousand kWh, score −5.8). The fraction of steps outside the band stayed 0.99 the whole run. The worst zone stayed 52–57 K.

Deterministic fp64 scores: fairbanks_1 8.10 at 1.3M steps, 2.47 at 47M (still 98% comfort). atlanta_2 of the same building 0.238 → 0.095. gen-val 0.261 against pid_water 0.057. gen-val-s 0.247 against pid_water 0.049. On both held-out sets the policy is about five times worse on comfort and about twice the energy of pid_water. Entropy rose from 61 to 342; the TensorBoard decline after 30M is the sampled episodes, while the deterministic fairbanks score stayed near 2.5 from 30M to 47M.

## r11 — same scenario, decaying entropy (finished 2026-10-18)

Same building and weather as r10 (`building_train_1` / `fairbanks_1`, 256 environments, absolute canonical actions, learning rate 0.0004 constant, every other PPO knob unchanged). The only change is the entropy bonus: `ent_coef=0.01`, `anneal_ent_coef=1`, `min_ent_coef_ratio=0.1`, so the coefficient cosines from 0.01 down to 0.001. 30M steps. No gen-val.

`loss/entropy` is the sum over the ~97 active heads, so the healthy band is about 40–120 (mean log-std −1 to −0.2), warning near 180, frozen near 280. The follower stops the trainer if three logged points in a row sit past the warning line with KL under 0.001 and clip fraction under 0.01. The 3060 scores each new checkpoint on fairbanks_1 and on atlanta_2. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r11`).

### Result (2026-10-18)

Finished 29.9M steps in 33 min 42 s. Final weights `results/runs/r11/ckpt/0000000029949952.bin`. Entropy stayed between 52 and 100 for all 456 updates (final 98.3, mean log-std −0.41, σ ≈ 0.67). KL at the end was 0.003 and the clip fraction 0.033. The freeze rule did not fire. Explained variance was 0.998.

The deterministic mean did not learn the heating the samples found. Training episodes went from score −8.0 and 13 thousand kWh of gas at 3M to score −4.0, deviation 23.8 K and 60 thousand kWh at 30M, with 98.7% of steps still outside the band. The deterministic fairbanks episode went from 7.66 at 1.3M to 8.12 at 29.9M, and its energy cost fell from 0.060 to 0.018. Atlanta went from 0.225 to 0.210 (best checkpoint 0.141 at 15.7M). r10's deterministic fairbanks score was 2.47, with the mean itself using the heat. Starting the bonus at 0.01 left the mean in the low-energy action.

## r12 — continue r10's 30.1M weights at ent_coef 0.001 (finished 2026-10-18)

2026-10-18. Loads `r10/ckpt/0000000030146560.bin` (fairbanks 2.58, summed entropy about 233). Same building, weather, learning rate 0.0004, and absolute actions. `ent_coef=0.001` held constant. `checkpoint_optim=0`, so the momentum from the 0.05 run is not loaded. 20M further steps. The 3060 scores fairbanks_1 and atlanta_2. No gen-val.

A cosine from 0.05 to 0.001 over 30M would already be near 0.003 at the step where r10's mean moved, which is the r11 regime. This run instead cuts the bonus on the policy that had just made that move, and watches whether the standard deviation comes back down. Working looks like summed entropy falling from about 230 toward 120 and fairbanks going under 2.5. Failed looks like entropy staying above 230, or fairbanks returning to the 7 range. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r12`).

### Result (2026-10-18)

Finished 20.0M further steps in 22 min 27 s. Final weights `results/runs/r12/ckpt/0000000019988480.bin`.

Deterministic fairbanks went from 2.58 on the loaded weights to 1.25. Over the same extra steps, r10 with the bonus left at 0.05 bottomed at 2.12 and finished at 2.47. Atlanta finished at 0.096, against 0.110 on the loaded weights and 0.095 for r10 at 47M. Sampled training episodes went from mean deviation 19 K and 98% of steps out of band at 3M to 5.2 K and 76% out of band at 20M. The worst zone went from 52 K to 34 K. Gas stayed near 70 thousand kWh. The final fairbanks episode also used less energy than the loaded checkpoint (0.061 against 0.133) while the comfort term fell.

Summed entropy went from 235 to 222 (standard deviation 2.7 to 2.4). It did not return to the healthy band. KL sat near 0 for long stretches while the fairbanks score kept falling, so a KL near 0 did not mean the weights had stopped. The score was still falling at 20M. The training-episode score ended at 0.35 while the deterministic fairbanks score was 1.25: with a standard deviation of 2.4 the samples hit the action rails and evaluation plays the mean.

## r13 — r12's mean, standard deviation pinned at 0.2 (interrupted)

2026-10-18. Loads `r12/ckpt/0000000019988480.bin` (fairbanks 1.25). Same building, weather, learning rate 0.0004, absolute actions, `ent_coef=0.001` constant, `checkpoint_optim=0`. `logstd_anneal=1` with `init_logstd=-1.6` and `logstd_final=-1.6`, so every epoch the log-std is overwritten to −1.6 (standard deviation 0.2) instead of annealing to the config default of −3. 20M further steps. The 3060 scores fairbanks_1 and atlanta_2. No gen-val.

Summed entropy should fall from about 222 to about −6. The continuous heads alone would be about −18 at log-std −1.6, and the discrete heads add about +11, which is the logged sum. Working looks like the deterministic fairbanks score staying near 1.25 and falling, with the training-episode score moving toward it. Failed looks like fairbanks climbing back toward 2.5. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r13`).

The power cut stopped this at 8.0M steps (8 min 51 s of trainer time). The 7.86M checkpoint is empty and was removed. The last intact weights are `r13/ckpt/0000000006553600.bin` (6.55M). Scored before the cut: fairbanks 1.34 at 1.3M, 1.23 at 2.6M, 1.45 at 5.2M. Atlanta 0.133 at 1.3M and 0.179 at 2.6M. The 5.2M atlanta eval did not finish.

## r13b — remainder of r13 (finished 2026-10-18)

2026-10-18. Loads that 6.55M checkpoint and trains the remaining 13.5M steps with the same flags, including log-std pinned at −1.6. Optimizer momentum was never saved, so the resume starts the optimizer cold, which is how r13 started too. The 3060 scores fairbanks and atlanta. No gen-val.

### Result (2026-10-18)

r13b finished 13.4M steps in 15 min 14 s. Together with r13's 6.55M, that is the planned 20M. Final weights `results/runs/r13b/ckpt/0000000013434880.bin`. Entropy stayed between −6.5 and −6.3, so the pin held.

Deterministic fairbanks went from 1.25 on the r12 weights to 1.23 at 2.6M, then 1.45 at 5.2M, and 1.80 at the end. Atlanta went from 0.096 to 0.137. The training-episode score and the deterministic score now agree (1.59 and 1.80). They agreed by both getting worse. Mean deviation on the training episodes went from 5.2 K under the wide policy to 13.2 K, the out-of-band fraction from 0.76 to 0.89, and the worst zone from 34 K to 43 K. Electricity fell from 18 to 4.7 thousand kWh. The narrow samples walked the mean back toward using less electricity and being less comfortable. The 1.25 score did not survive once the actions had to stay near the mean.

## r14 — both weathers, r12's mean, noise left wide (finished 2026-10-18)

2026-10-18. The pin at 0.2 showed the 1.25 fairbanks score was the mean of a policy whose samples sit on the action rails. Shrinking the noise walked that mean backward. r12's deterministic fairbanks score was still falling at the end, so this run continues from `r12/ckpt/0000000019988480.bin` with `ent_coef=0.001` and the learned standard deviation left where it is (about 2.4). `logstd_anneal=0`. `checkpoint_optim=0`.

It trains `building_train_1` on fairbanks_1 and atlanta_2 together (256 environments, the two weathers share one solve). Same learning rate 0.0004, absolute actions. 40M steps. The 3060 scores each checkpoint on fairbanks and on atlanta, the same deterministic fp64 episodes as before. No gen-val.

The number that counts is those two deterministic scores. The training-episode score will look better while the standard deviation stays near 2.4, because the samples hit the rails. Working looks like fairbanks staying under 1.25 and falling, and atlanta moving down with it. Failed looks like fairbanks climbing back toward 1.8, which is what the narrow policy did. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r14`).

### Result (2026-10-18)

Finished 40.0M steps in 47 min 59 s. Final weights `results/runs/r14/ckpt/0000000039976960.bin`.

Deterministic fairbanks 1.25 on the loaded weights, 1.49 at 1.3M, 0.99 at 7.9M, 0.86 at 32.8M, 0.92 at 40M. Atlanta 0.096, then 0.159 at 1.3M, best 0.066 at 15.7M, 0.080 at 40M. The 32.8M checkpoint is the better pair (fairbanks 0.856, atlanta 0.090). Comfort is still 94% of the fairbanks penalty. Productivity on that episode went from −2.02 to −1.44, while energy rose from 0.061 to 0.092 and carbon from 0.126 to 0.184.

Sampled fairbanks episodes ended at 0.38 against the deterministic 0.92. Sampled atlanta episodes ended at 0.04 against the deterministic 0.08. Atlanta's mean learned the action. Fairbanks' mean moved toward the samples and did not arrive. Summed entropy rose from 222 to 254 (standard deviation 2.4 to 3.3). KL at the end was 0.0015 and the clip fraction 0.007. The last 8M of fairbanks sits between 0.86 and 0.97.

## r15 — stop the noise from growing (finished 2026-10-18)

2026-10-18. A longer run at `ent_coef=0.001` keeps the only upward force on log-std. Over r14 that force took the standard deviation from 2.4 to 3.3, and the mean gradient scales as one over variance. The code clamps log-std at 2 (standard deviation about 7.4), so the same bonus can still push it further. Credit assignment is not the lever: gamma 0.999 and comfort shaping were already measured failures, and the learning rule bars shaping.

This run loads the better r14 pair, `r14/ckpt/0000000032768000.bin` (32.8M: fairbanks 0.856, atlanta 0.090), and sets `ent_coef=0`. With the bonus off, the only pressure on log-std is the policy-gradient term, which shrinks it when samples sit near the mean. It does not overwrite log-std. Same two weathers, learning rate 0.0004, absolute actions, 40M steps. The 3060 scores fairbanks and atlanta. No gen-val.

Working looks like summed entropy staying near 250 rather than climbing, and fairbanks dropping under 0.86. Failed looks like fairbanks climbing back toward 1.2, which would mean the bonus was holding the noise in the range that pulls the mean. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r15`).

### Result (2026-10-18)

Finished 40.0M steps in 48 min 11 s. Final weights `results/runs/r15/ckpt/0000000039976960.bin`.

Summed entropy stayed between 246.7 and 249.7 (standard deviation 3.12 the whole way). The bonus was the upward force, and removing it did not make the policy gradient shrink the noise either. On-policy, that term averages near zero.

Deterministic fairbanks never beat the 0.856 it started from. It was 0.886 at 1.3M, 1.19 at 11.8M, and 0.927 at 40M. Atlanta went from 0.090 to 0.072. The pair sum went from 0.946 to 0.999. Sampled fairbanks episodes stayed near 0.36 against a mean of 0.93. KL stayed near 0.002 and the clip fraction near 0.02. The mean wandered and came back to where r14 had already finished.

## r16 — shrink the noise on a schedule (started 2026-10-18)

2026-10-18. r15 left the standard deviation frozen at 3.12. The policy gradient on log-std averages near zero on these samples, and the mean gradient scales as one over variance, so the mean cannot move while the samples sit on the action rails. Pinning the standard deviation at 0.2 (r13) walked the mean backward. This run overwrites log-std every epoch with a straight line from the current width to 1.0, and leaves the entropy bonus at 0 so it cannot fight the line.

Loads `r14/ckpt/0000000032768000.bin` (fairbanks 0.856, atlanta 0.090). `ent_coef=0`, `logstd_anneal=1`, `init_logstd=1.14` (standard deviation 3.12), `logstd_final=0` (standard deviation 1.0) over 40M steps. Same two weathers, learning rate 0.0004, absolute actions, gamma 0.99, GAE 0.95, horizon 256. The 3060 scores fairbanks and atlanta. No gen-val.

Summed entropy should fall from about 248 toward about 137 (97 active heads at log-std 0). Working looks like that fall, and fairbanks dropping under 0.856 without atlanta climbing back toward 0.16. Failed looks like fairbanks climbing toward 1.2 while the noise is still wide, which is the r13 walk happening early, or entropy staying near 248, which means the overwrite did not take. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r16`).

### Result (2026-10-18)

Finished 40.0M steps in 48 min 32 s. Final weights `results/runs/r16/ckpt/0000000039976960.bin`.

The overwrite took. Summed entropy fell from 259 to 149, which is log-std 1.14 down to 0 on the 97 continuous heads. The training-episode score fell from 2.0 to 0.36 as the samples left the rails. KL stayed near 0 until the entropy passed about 200, then the clip fraction rose to about 0.07. The value loss spiked to 3254 in the first minute, from the log-std flatten, and was back under 10 by the end.

Deterministic fairbanks did not get better. It was 0.836 at 1.3M, while the noise was still the loaded width, then 1.23 at 19.7M, 0.846 at 34.1M, and 1.11 at 40M. The loaded weights are 0.856. Atlanta stayed between 0.083 and 0.139 and finished at 0.093, against 0.090 on the loaded weights and 0.066 at r14's best atlanta checkpoint. On the fairbanks episode, productivity went from −1.31 to −1.79 and energy from 0.113 to 0.074. The same walk r13 took by pinning the standard deviation at 0.2, reached here by walking it down to 1.0. The 0.856 mean is the center of a wide policy. Narrowing the samples removes the heat those samples were adding.

## r17 — squash the sample, center the learning reward (started 2026-10-18)

2026-10-18. Two separate failures, two separate fixes, one run. They are confounded; the next comparison is against r14's 0.856, not against a single-change ablation.

The noise schedule is the wrong fix. r13 jumped the standard deviation to 0.2 and r16 walked it from 3.1 to 1. Both moved fairbanks toward less energy and worse comfort. The heat in the 0.856 mean comes from samples the environment clips to the rail. The log-prob is of the unclipped sample, and `tanh_mean` multiplies the mean gradient by (1 − μ²), which is zero at that rail. Explained variance on r16 sat near 0.99 after the first million steps, so the critic is not what stalled the mean.

Reward centering is a different problem. On r16, `norm/adv_mean` sits around zero once the critic has fit (order 0.1 to 1, last value −0.195 at 40M). Stepping the average reward by the advantage, which is what the paper's PPO appendix did (equation 10), would jitter around zero and never remove the constant r̄/(1−γ). The on-policy estimator is the running mean of the reward (equation 4). r7–r9 already showed that dividing by a moving standard deviation makes the critic chase the target, so this path does not divide. μ starts at 0: the loaded critic predicts uncentered returns, and subtracting the true mean on the first update would shift the value target by about 90 on fairbanks.

Loads `r14/ckpt/0000000032768000.bin`. `policy.tanh_squash=1` (noise inside tanh; the stored action is the pre-tanh sample; deterministic eval stays tanh(logit), so step 0 matches the loaded mean). `env.rew_center=1`, `env.center_beta=0.01`, `rew_norm=0`. `ent_coef=0`, `logstd_anneal=0`, `init_logstd=-1` so the checkpoint's learned log-std is kept. Same two weathers, learning rate 0.0004, gamma 0.99, GAE 0.95, horizon 256, 40M steps. The 3060 scores fairbanks and atlanta on the raw reward. No gen-val.

Working looks like entropy staying near the loaded ~248, `norm/reward_mean` creeping away from 0 without a value-loss spike like r16's 3254, and deterministic fairbanks under 0.856. Failed looks like fairbanks climbing toward 1.2, or entropy leaving the loaded band. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r17`).

### Result (2026-10-18)

Stopped at 15.7M steps. The critic died at 0.39M and the weights have been frozen since 0.46M. Explained variance is 0 because the return sums are non-finite: the logger stores 0 when the return variance is not a positive number. `norm/adv_inv_std` is exactly 10000, which is `1/sqrt(1e-8)`, the value you get when every advantage is NaN and `max(NaN, 0)` becomes 0. Policy loss and value loss stop being logged after 0.33M because the numbers are non-finite. Entropy is bit-identical at 148.72845 from 0.46M through 15.7M, and KL is about 1e-11, so no update has landed.

The crash is at 0.26M, while the loaded width was still in place (entropy 248) and the centering mean was still −0.006. KL there is 8.4e12 and the policy loss is 2.7e13. The logit ran until `exp(log-ratio)` overflowed. With a negative advantage the clip does not clear an infinite ratio, the shared value head goes NaN, and every later step is skipped. Training episodes at the stop: fairbanks about 3.4, atlanta about 0.14, comfort deviation about 7 K. The 3060 never wrote a score; `puffer_f64` was not present, so each eval exited before the json.

## r18 — same squash, finite ratio (started 2026-10-18)

2026-10-18. r17 died before squash was measured. The log-ratio is now clamped to ±20 before the exponential, and a non-finite policy or value gradient is dropped, so the gradient clip can hold the step. Same load, same width, same centering. `puffer_f64` is rebuilt before the follower starts.

Loads `r14/ckpt/0000000032768000.bin`. `policy.tanh_squash=1`, `env.rew_center=1`, `env.center_beta=0.01`, `ent_coef=0`, `logstd_anneal=0`, `init_logstd=-1`, learning rate 0.0004, 40M steps, fairbanks and atlanta. The 3060 scores the raw reward. No gen-val.

Stop in the first million steps if explained variance drops under 0.5 or KL goes above about 10. Working looks like explained variance near 0.9, entropy near 248, and a fairbanks score at the first checkpoint. If that score is already climbing toward 1.2 while entropy is still near 248, this width is the wrong operating point for squash. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r18`).

### Result (2026-10-18)

Stopped at 0.72M. The clamp held: explained variance stayed between 0.67 and 0.98, entropy stayed at 248, and the value loss stayed finite (peak 180, against r17's NaN). KL did not. It was 1.3e4 at 0.20M, 1.8e7 at 0.39M, 9 at 0.46M, then 7.3e7 at 0.72M. The ratio is on the ±20 cap every other update, so the loaded width is the wrong operating point for squash. No checkpoint was scored.

## r19 — squash at standard deviation 1 (started 2026-10-18)

2026-10-18. r18 kept the critic alive and showed that σ = 2.8 puts the squashed ratio on the clamp. This run pins log-std at 0 (standard deviation 1) from the first epoch, so the checkpoint cannot restore the wide noise. The log-prob matches the executed action, and tanh of the logit still depends on the mean. Centering is off, so the only change from the r14 weights is that matched width.

Loads `r14/ckpt/0000000032768000.bin`. `policy.tanh_squash=1`, `logstd_anneal=1`, `init_logstd=0`, `logstd_final=0`, `ent_coef=0`, `rew_center=0`, learning rate 0.0004, 40M steps, fairbanks and atlanta. The 3060 scores the raw reward. No gen-val.

Stop after the first million steps if explained variance is under 0.5 or KL is still above 10. Working looks like entropy near 149 and a fairbanks score that does not climb toward 1.2. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r19`).

### Result (2026-10-18)

Stopped at 2.6M. The pin took: entropy sat at 148.7 from the first log. The critic held, explained variance 0.80–0.96. KL did not come down. It was 3 at 0.07M, then between 1e4 and 3e6 through 1.3M. Deterministic fairbanks was 0.916 at 1.3M and 0.866 at 2.6M, against 0.856 on the loaded weights. Atlanta was 0.101 then 0.096, against 0.090.

## r20 — squash inside a ±3 logit box (started 2026-10-19)

2026-10-19. r18 and r19 both left the squashed ratio at 1e4–1e6, at standard deviation 2.8 and at 1. Another width would repeat that. This run changes where the Gaussian sits. The location is the logit clamped to ±3, which is artanh(0.995): past that, tanh is already on the rail, and the mean gradient is zero so a runaway logit is not scored again. `norm/logit_outside` is the fraction of active heads outside the box. If that fraction is near 0 and the ratio is still huge, the loaded logits were not the problem. If the fraction is high and the ratio falls under 10, they were.

Loads `r14/ckpt/0000000032768000.bin`. `policy.tanh_squash=1`, `logstd_anneal=1`, `init_logstd=0`, `logstd_final=0`, `ent_coef=0`, `rew_center=0`, learning rate 0.0004, 40M steps, fairbanks and atlanta. The 3060 scores the raw reward with `tanh_mean` and no squash, so a head outside the box is scored as the loaded action. No gen-val.

Stop after the first million if KL is still above 10. A worse fairbanks score with a sane KL is a result, not a reason to stop. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r20`).

### Result (2026-10-19)

Finished 40M. The bound did what it was for: `norm/logit_outside` stayed near 0.77 and KL stayed near 0.02, with explained variance near 0.99. The score did not. Fairbanks was 0.921 at 9.2M and 1.571 at 40M, against 0.856 on the loaded weights. Atlanta's best was 0.063 at 13.1M, and 0.082 at 40M, against 0.090. Freezing the saturated heads left the inner quarter free to walk fairbanks off the loaded action. Best weights remain `r14/ckpt/0000000032768000.bin`.

## r21 — 300M, explore at the loaded width, then commit (started 2026-10-19)

2026-10-19. r20 showed the ratio is finite once the Gaussian center stays inside ±3, and that a hard zero on the mean gradient does not learn. The location is now `3·tanh(logit/3)`: still inside ±3, derivative `1-(loc/3)²`, so a saturated head can move over a long run. The width is the loaded one for the first 80% (log-std 1.14, standard deviation 3.12, the width that reached 0.856), then a linear move to log-std −1 (standard deviation 0.37) over the last 20%. That commit is not the pin at 0.2, which already failed, and it does not spend the whole run getting narrower. `ent_coef` stays 0 because the schedule owns the width.

Batch, exact. 256 agents and horizon 256 give 65,536 steps per epoch. Minibatch 2048 is 8 rows, so the rollout is 32 blocks; replay 4 is 128 minibatches, four passes of those 32, which is the shuffle the trainer accepts. `total_timesteps=300023808` is 4578 epochs, the multiple of 65,536 at 300M. A checkpoint every 100 epochs is 6,553,600 steps. The 3060 scores each one with `tanh_squash` on, so the number is tanh of the location the run is learning, not tanh of the raw logit. The hold ends at epoch 3662 (240,018,432 steps).

Same two weathers, from `r14/ckpt/0000000032768000.bin`. Learning rate 0.0004, gamma 0.99, GAE 0.95, centering off. No gen-val. A third weather (`building_train_1` / `denver_7`) is added only if, during the hold, one score improves while the other is worse by 0.3 or more for three checkpoints in a row. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r21`).

### Result (2026-10-19)

Stopped at 286M, inside the commit. The first 90M did move. Deterministic fairbanks went from 0.856 to 0.193 at 85.2M (`ckpt/0000000085196800.bin`), and the pair was best at 91.8M: fairbanks 0.210, atlanta 0.061 (`ckpt/0000000091750400.bin`). That fairbanks gain is comfort (productivity −1.31 to −0.23), not a cut in heating. After that the score wandered: fairbanks 0.23–0.57, atlanta 0.05–0.11, and the end was 0.373 / 0.117.

`norm/logit_outside` went 0.71 → 0.97 by 80M → 0.998 by 280M. The location sits at ±3 and its derivative is effectively zero, so the later steps were not learning. Entropy stayed at 259 until the hold ended at 240M, then the schedule took it to 99. KL rose from 0.002 to 0.05 and clipfrac from 0.01 to 0.25. Explained variance stayed near 0.98 the whole way, including while the mean could no longer move. No third weather: atlanta did not fall apart during the improvement.

## r22 — chain floor, five weathers, 40M (started 2026-10-19)

2026-10-19. r21 learned until the logits left ±3, then the chain rule died and the commit only dropped the entropy. The forward location stays `3·tanh(logit/3)`, so this checkpoint still executes the action that scored 0.210 / 0.061. The loss now floors that chain at 0.25, which is the region past about |logit| = 4. Log-std is left at the checkpoint's 1.14. No entropy bonus, no anneal.

The batch is building_train_1's five training weathers together (atlanta, denver, fairbanks, phoenix, singapore), 64 envs each, 320 agents. Horizon 256, minibatch 2048, replay 4: 81,920 steps per epoch, 488 epochs, 39,976,960 steps. A checkpoint every 61 epochs is 4,997,120 steps. Loads `r21/ckpt/0000000091750400.bin`. Learning rate 0.0004, gamma 0.99, GAE 0.95, centering off. The 3060 scores fairbanks, atlanta, and denver with `tanh_squash` on. No gen-val.

The first attempt was stopped at 10M by a threshold. Fairbanks had moved from 0.337 at 5M to 0.232, and `norm/logit_outside` had turned down from 0.981 to 0.944, so the cut fired while the run was improving. Restarted fresh from the same checkpoint, with no automatic stop. TensorBoard: http://127.0.0.1:6006 (`results/runs`, run `r22`).

### Completed result and interpretation (2026-10-19)

Completed 39,976,960 steps, trainer uptime 3054.8 s. The best of the eight saved checkpoints on the **three monitored training scenarios** is 24,985,600 steps (`r22/ckpt/0000000024985600.bin`): fairbanks 0.183536658, atlanta 0.041155705, denver 0.032405089, mean **0.085699151**. The matched pid_water mean is 0.088564368 (the policy improves on it by 3.24%); fixed_setpoint is 0.068156660 and remains better. This is not a Kaggle leaderboard estimate. The original fairbanks/atlanta pair improves from r21's 0.135752826 to 0.112346181 (17.2%).

The final checkpoint is worse than the selected one: fairbanks 0.209621828, atlanta 0.042196116, denver 0.036570037, three-scenario mean 0.096129327. The 35M checkpoint temporarily regresses to 0.178826896. Preserve all checkpoints; do not infer convergence or recommend the final weights by default. At 25M, fairbanks' weighted comfort charge is 0.125961292 and weighted energy plus carbon is 0.057575144, versus 0.154172802 and 0.056084879 on r21's loaded checkpoint: the improvement there is comfort, with slightly higher energy/carbon charges. Complete score decomposition and all checkpoints: `complete_port/results/campaign_r23/r22_review.json`.

The final explained variance is 0.9738 and value loss 0.453; the run remained finite. Latent entropy increases from 259.43 to 277.86 despite no entropy bonus, implying about 21% growth in the geometric mean of the 97 active continuous widths. The width was trainable, not fixed. The earlier assertion that the advantage-weighted log-std gradient must average to zero was too strong. Also, `logit_outside` counts |raw logit| > 3; the exact derivative at 3 is still 0.42, so that diagnostic alone does not establish a frozen actor. The chain floor and the expanded weather mixture changed together, preventing causal attribution to either.

## r23 campaign — verified loss, matched evaluation, then controlled training (2026-10-19)

User authorizes both GPUs, careful experiments, and logging; target updated to 0.00598. No promise of a winning score. No important files or checkpoints are to be deleted. Default fp64 physics only, unchanged official reward, absolute learned actions, no residual controller, action initialization from a controller, comfort guard, or comfort shaping. Sealed seeds 30000–30019 stay untouched.

Before edits, the historical evaluation binary, configs, trainer sources, patch, and selected checkpoint hashes were preserved under `complete_port/results/campaign_r23/`. Reproducible driver: `complete_port/tools/campaign_r23.py`. The 3060 evaluates r21@91.8M, r22@25M, and r22-final on the same five-weather set and on evalcore3 (15 scenarios, unseen buildings). These development results must remain distinct from a final test and the public leaderboard. No evaluation data enters a training batch.

First gate: reproduce masked padding-head gradients and check the CUDA PPO loss against independent double-precision/finite-difference calculations before changing it. Planned controlled experiments compare the corrected loss with the same forward policy, then isolate step size and exploration; architecture or dataset changes get separate run IDs. Every trainer-source change must pass `tools/verify_all.sh --trainer`, plus focused gradient checks. Decisions will use full 8640-step deterministic fp64 scores, reward components, and unseen-building development performance. Failed hypotheses and incomplete evaluations remain recorded.

### Loss audit and matched initial scores

The first test-harness attempt incorrectly separated buffers that production intentionally reuses; it is not evidence of a trainer defect (`math_before/` retained). With that harness setup corrected, the unmodified trainer fails three of ten cases: nonzero gradients and fake entropy on fully masked categorical heads, the deliberate squash chain floor at saturation, and an incorrect geometric-ratio denominator when categorical class 0 alone is masked. Normal Gaussian, tanh mean, interior squash, and clipping-sign cases pass. After canonical masking fixes and making the chain floor explicit (default 0 = exact; 0.25 reproduces r22), all **80 scalar checks pass**. Sources and CUDA output are retained in `results/campaign_r23/math_before_v2/` and `math_after/`. The physics is unchanged.

New diagnostics distinguish joint sampled log-probability change, the geometric-ratio clamp fraction, non-finite ratios, and the actual squash derivative distribution. `loss/kl` keeps its historical geometric-ratio meaning. A cold-optimizer-only freeze switch preserves existing per-head widths without replacing them with a new uniform width.

Historical fp64 binary, fixed five-weather batch: r21@91.8M **0.148698754**, r22@25M **0.079433892**, r22-final **0.085689033**, matched pid_water **0.086294041**. evalcore3 (15 scenarios, three buildings not trained by r21/r22): r21 **0.257756722**, r22@25M **0.183006974**; remaining evaluations pending. Thus r22 does improve transfer in this development test (29.0%), but it remains far from the target. Combined-batch scores differ slightly from prior individual-scenario evaluations; compare identical batch layouts and preserve the distinction rather than silently mixing numbers. `results/campaign_r23/pareto.json` records comfort/energy/carbon fronts separately for identical scenario sets.

GPU scheduling preference: 3090 trains (or verifies), 3060 evaluates; after a training phase, use both for outstanding evaluations. Optimize actual physics transitions per wall second, not the unreliable displayed SPS. A new `tr23_shipped30` set is exactly the declared shipped 30, uses hard links, and passes the deep leakage check against all 45 eval9 scenarios. It does not reuse the historical unfiltered `artifacts/train` inventory.

Pre-registered first training screen, after the full verification gate: an unchanged-policy learning-rate-zero smoke check, then r23a/r23b, each 4,997,120 steps from r22@25M, five weathers/320 agents/horizon 256/minibatch 2048/replay 4, raw reward, gamma 0.99, lambda 0.95, entropy 0, no width anneal, chain floor 0.25, seeds 73. r23a uses learning rate 0.0004; r23b uses 0.0001. Both include the verified masking repair. No causal comparison against an unrelated starting checkpoint. Treat these short runs as screens, not proof of convergence; evaluate full episodes and unseen-building development scores before extending. Multi-building training and a shared local policy are the next generalization candidates, rather than indefinitely tuning one building.

### Verified runtime, pilot execution, and transfer architecture

`results/verify_20261019_102139.log` ends **ALL CHECKS PASSED** (oracle/graph/reset/group gates, trainer smoke, and 80 scalar PPO checks). The corrected fp64 runtime is immutable under `results/campaign_r23/fixed_runtime`. The lr=0 smoke run `r23_identity` reproduces the loaded r22@25M checkpoint byte for byte (SHA256 `5e458eb0426a9c1a4fbf62e5bc00bc3bb2a59413f4180d62648b0c00a12a2f61`); importance 1, clip fraction 0, near-zero sampled KL. Its 100-step episodes are a smoke check, never a competition score.

`r23a_maskfix_lr4` completed 4,997,120 steps with 461.28 s trainer uptime (**10.83K physical environment steps/s**, fp64), 485.08 s process wall time. No non-finite ratios or log-ratio clamp events. Final actual mean squash derivative 0.0202, 94.0% below 0.01; the explicit 0.25 surrogate was retained as specified. Joint sampled log-probability change is 4.62 while geometric-ratio KL is 0.00292: do not confuse these statistics. The lower-lr `r23b_maskfix_lr1` is running, and full-episode matched evaluations are on GPU 1. The corrected evaluator reproduces the historical r22 five-weather output exactly, score 0.07943389249.

Completed unseen-building controls: r22-final 0.1966018673; pid_water **0.05078483742**, which dominates all three learned historical candidates in comfort, energy, and carbon on evalcore3. Thus neither r22's training score nor its 29% improvement over r21 establishes competitive generalization. The gap is a failed transfer test, not a success claim.

Implemented an optional **shared local actor with pooled recurrent context** (`SB_SHARED_POLICY`): shared zone/AHU/hot-water actor matrices, a smaller recurrent context, and symmetric sensor pooling. It excludes every PID proposal/integrator observation column and starts random, with no controller action, safety guard, or shaped reward. All local slots share parameters so outputs for new zones/AHUs are learned even when a particular canonical slot was absent during training. Positive shortfalls are sensor features only, not rewards. The current canonical limit remains 9 AHUs/90 slots; no broader coverage claim. Planned fresh multi-building screen: exactly tr23_shipped30, H64/one recurrent layer, fixed initial log-std -0.7, exact squash derivative, constant reward scale 0.1, no entropy bonus. No architecture training until compiled finite differences, padding masks, zone/AHU permutation tests, CUDA graphs, and the required full verification pass. This is an architectural hypothesis, not a novelty claim.

Isolated prototype setup failures retained in `shared_build.log`: dependency download stalled, so reused existing local raylib; an archive inside the parent Git tree silently skipped a relative patch, so initialized its own disposable Git metadata and redeployed canonical sources. No important files removed. Reviewer checks caught and corrected an absent-slot context-gradient mask before the first compiled gradient test. The new `eval_controlled.py` preserves outputs and records binary/checkpoint hashes, full-episode length, graph mode, and timing. A full-episode graph/no-graph comparison is queued before changing evaluation defaults.

### Stopped at the user's request (2026-10-19 05:35 UTC)

All campaign host process groups, active GPU containers, and queued jobs were stopped. Confirmed no campaign compute remains on either GPU. No training or evaluation will resume without new user instructions. No important files/checkpoints deleted. Automatic approval review blocked an evaluation launch after the stop request; it was not retried.

Completed r23b: 4,997,120 steps, 457.29 s trainer uptime (10.93K physical SPS), 481.13 s process wall time. Final non-finite/clamped ratio fractions 0; exact mean chain 0.03215, fraction below 0.01 0.90476. Full deterministic r23a five-weather score **0.08447111271**, 6.34% worse than parent **0.07943389249**; parent dominates r23a in all three reward charges. Do not promote r23a. Its evalcore3 JSON is incomplete because that job was stopped. r23b has **no valid full evaluation**: its first graph attempt could not export scores, and editing the shell evaluation runner while that invocation was active caused a shell-read error. That was an avoidable orchestration mistake; the finalized runner passes bash syntax checks and preserves full logs, errors, and existing outputs. Failed artifacts/metadata retained; no score inferred from them.

Graph export diagnosis: puf_step host callbacks do not execute during CUDA graph replay. Added an export at puf_log, which reads only the device diagnostic symbol g_first_scen after synchronization, never managed Env. A verification-only immutable fp64 binary (`graph_runtime`) with horizon 32 reproduces the **entire** parent five-weather JSON exactly, including all components. Measured 88.18 s process wall time; the non-graph r23a evaluation logged 86.83 s trainer uptime on a different checkpoint, so there is **no demonstrated speed improvement** and no valid controlled speedup claim. The initial graph experiment's 289.36 s wall time includes queue waiting and must not be used as execution speed. The new evaluator starts timing only after the GPU reservation is granted.

Shared-policy focused checks (`shared_policycheck.log`): **60 finite differences, 0 failures**; independent of PID-derived observation columns (delta 0), zone permutation equivariance (delta 0), AHU/block permutation equivariance (delta 2.98e-8), absent/padded outputs zero, and forward CUDA graph all pass. **No shared-policy training was launched.** The second full gate (`shared_full_gate.log`) was stopped after the six shipped building oracle/multi-episode/graph/LU/convection checks and train_1 at 1/64/256 envs passed. It has **not** reached ALL CHECKS PASSED. Therefore the new shared-policy/log-export changes are not fully certified; complete that gate before training or promoting that runtime. The earlier corrected flat runtime's full gate remains passed and its two completed pilots are preserved.

Measured Pareto artifacts: `results/campaign_r23/pareto.json`, `.csv`, `.png`, `.svg`; finite candidate fronts, separated by identical scenario sets, not a claim of global optimality. Target 0.00598 has **not** been beaten or tested on Kaggle. Strongest established r22 candidate remains the 24,985,600-step checkpoint; its unseen-building deficit remains unresolved.

### User correction: longitudinal research and prior approval (2026-10-19)

The user rejects short-horizon chasing and requires a proposed research protocol followed by explicit permission before execution. Use 20M/30M agreed runs to capture dips, recovery and late patterns. `pid_water` is not the research baseline: use our learned policy as the control and keep the public 0.00598 target separate from local scores. These requirements are also recorded in AGENTS.md. All GPU work remains stopped.

Interpretation correction: r23a's 6.34% regression is an observed endpoint at an additional 5M steps, not a rejection of the learning rate, the masking fix, or their eventual learning curve. No run studied their 20M/30M behavior. Historical controller comparisons do not establish research success or failure. The shared-policy prototype remains untrained and its incomplete full gate remains pending; it will not be introduced into another experiment without approval.

### r24 — training-distribution comparison (approved 2026-10-19)

Hypothesis: broader scenario experience improves transfer. A continues on one building/five weathers; B uses the 30 shipped scenarios/six buildings. Both load r22@24,985,600; training mixture is the changed factor.

Fixed per arm: 29,982,720 additional steps (122 epochs), 960 agents, H256/two layers, LR 1e-4, horizon 256, minibatch 2048, replay 4, seed 73. Same verified fp64 runtime and official reward; no method changes or score-based stopping. Save ~1M; full evaluations ~5/10/15/20/25/30M. Primary: last-three evalcore3 mean; report complete curves, components and worst-building behavior. Final eval9 checks broader development transfer; one seed is exploratory.

Verification: `verify_20261019_112634.log` — **ALL CHECKS PASSED**. Runs `r24a_single30m` then `r24b_multi30m` on 3090; 3060 evaluates, then both clear evaluation. Both trainings finished. The study driver ended `evaluation_incomplete` after the parent eval9 job failed; the arm curves are in `complete_port/results/study_r24/eval/`. Full protocol: `complete_port/results/study_r24/`. Shared-policy prototype is excluded; no Kaggle claim.

B's unseen-building curve on evalcore3: 0.173, 0.185, 0.194, 0.201, 0.153, 0.164. The 25M checkpoint is the best point and the control for r25. A's last-three mean was 0.178; B's was 0.173; the parent was 0.183.

### r25 — acquire six new buildings (approved 2026-10-19)

Hypothesis: acting in floors this policy has never occupied moves the held-out buildings, rather than only the floors in the batch. Control is r24b at 24,576,000 steps (evalcore3 0.153). The batch is `10001`, `10004`, `10012`, `15002`, `15021`, `15025`, five weathers each. The shipped six, sealed seeds 30000–30019, and the 45 development scenarios stay out.

Fixed: about 30M steps, the r24b settings, fp64 runtime, seed 73, cold optimizer, chain floor 0.25, entropy coefficient 0, no width schedule. 32 environments per scenario did not allocate on the 3090. The run is 24 per scenario, 720 agents, 30,044,160 steps. Curves every ~5M: the new buildings, the original five weathers, and evalcore3, with comfort, energy, and carbon. No score-based stop. One seed. Driver: `complete_port/results/study_r25/run_r25.py`.

The first start was cut by a power loss near 2M steps. Those checkpoints are kept under `results/runs/r25_acquire30m_poweroff`. Restarted fresh from the same 24,576,000 parent.

### Result (2026-10-19)

Finished 30,044,160 steps. The new buildings fell from 0.136 to 0.071, and the last point is the best. The original five weathers went from 0.087 to 0.122; fairbanks on that building went from 0.213 to 0.325. Held-out evalcore3 was 0.132, 0.163, 0.174, 0.168, 0.157, 0.127 against the loaded checkpoint's 0.153. The last point is better on all three held-out buildings. The last-three mean is 0.151. Cold held-out weathers did not improve. One seed. Scores: `complete_port/results/study_r25/eval/`.

### r26 — keep the six buildings and put building_train_1 back (approved 2026-10-19)

Hypothesis: the building in the batch is the one that moves. Load r25 at 30,044,160. Train about 30M steps on those six buildings plus `building_train_1`'s five weathers, 35 scenarios. Same settings, entropy coefficient 0, 16 environments per scenario (560 agents). Curves every ~5M on the new buildings, the original five weathers, and evalcore3. No score-based stop.

Bar, on the best checkpoint of the full curve: new buildings at or under 0.071, held-out at or under 0.127 with held-out Fairbanks not worse than 0.174, and original Fairbanks at or under 0.213. Driver: `complete_port/results/study_r26/run_r26.py`.

### Result (2026-10-19)

Finished 29,962,240 steps. New buildings 0.071 → 0.058. Original five weathers 0.122 → 0.077, and Fairbanks on that building 0.325 → 0.185. Held-out evalcore3 0.129, 0.126, 0.132, 0.124, 0.113, 0.113 against the loaded 0.127. All three held-out buildings improved. Held-out Fairbanks ended at 0.185, better than the loaded 0.203 and short of the older 0.174 mark. The final checkpoint is the best point on all three curves. Scores: `complete_port/results/study_r26/eval/`.

The 0.058 mean is 28 scenarios between 0.024 and 0.070, plus `building_10001__fairbanks_8` at 0.306 and `building_10004__denver_2` at 0.153. Those two were in the batch for r25 and r26, about 60M steps, and never descended (`10001` Fairbanks stayed 0.32–0.37 through r25). Held-out score is 90% comfort. Warm held-out ended at 0.070; cold held-out stayed near 0.176 from 20M. Training worst-zone deviation stayed near 12 K and entropy near 252 for the whole run.

### r27 — shared zone actor from random (approved 2026-10-19)

Hypothesis: the flat canonical head is why cold control learned on the buildings in the batch does not move held-out cold. Train the shared zone actor from random initialization for about 30M steps on the same 35 scenarios. It does not load the r26 checkpoint. Hidden size 256 and two layers, matching r26, so the actor is the changed factor. Same reward, learning rate 1e-4, entropy coefficient 0, seed 73. Gate `results/verify_20261019_200556.log` ended **ALL CHECKS PASSED**. The first start was stopped at 2.9M steps: that binary still executed `pid_water` and wrote its proposal into observation slots. The shared actor did not read those slots and the action was not residual. Those checkpoints stay in `results/runs/r27_shared30m_pidkernel` and are not loaded. The restart uses a rebuild where `base_pidw_k` is compiled out (`puffer_shared` SHA256 `b2b1ed04268ae1feec19743d013737a0b258d80a8ad64451c74c1a27d6fb4bc2`). The gate that includes that change, `results/verify_20261019_212146.log`, ended **ALL CHECKS PASSED**. Decision on the mean of the 20M, 25M and 30M checkpoints: held-out cold at or under 0.150 is a transfer win; in-batch still above 0.10 is inconclusive; in-batch at or under 0.10 with held-out cold still at or above 0.169 means this actor is not the lever. Driver: `complete_port/results/study_r27/run_r27.py`.

Stopped at the user's request near 21.8M steps (last checkpoint `0000000021790720.bin`). Explained variance reached 0.98 by 14M and stayed there. From the first finished episodes to the stop, electricity fell from about 79 kW to 18 kW and gas from about 227 kW to 62 kW, while productivity went from about −0.3 to about −1.8 and the training scenario mean stayed near 1.10. Sixteen shoulder scenarios improved by a few hundredths. Hot scenarios and January Fairbanks got worse, by as much as 4 points on `building_train_1__fairbanks_1`. The transfer test was not reached. Checkpoints stay in `results/runs/r27_shared30m`.

### r28 — shared actor, cold windows only (approved 2026-10-19)

Hypothesis: the shared map follows the comfort shortfall when every scenario wants heat. r27's batch mixed heating and cooling, and the energy gradient was the one that agreed everywhere. Control is the r27 curve. These weights are not loaded. The batch is `building_train_1__fairbanks_1` (1 January), `building_10001__fairbanks_8` (22 October), and `building_10004__denver_2` plus `building_15002__denver_2` (26 February). May, June, and September Fairbanks stay out. Same loss, hidden size 256, two layers, seed 73. Training is the fp32 solve with 160 environments per scenario (640 agents; 512 if that does not allocate). Scores are the fp64 binary on these four, every ~5M through 30M. No score-based stop. A 15M point where the fuels fell and productivity did not is reported and the run still finishes. Driver: `complete_port/results/study_r28/run_r28.py`.

### Result (2026-10-20)

Finished 29,982,720 steps. fp64 means of the four: 1.313, 1.545, 1.435, 1.369, 1.414, 1.435. The 15M and 30M points are the same, 1.435. Panel productivity went from −2.06 to −2.32 while energy went from 0.155 to 0.092 and carbon from 0.216 to 0.137. That is the pre-registered failure: the fuels fell and comfort did not.

The mean is January. `building_train_1__fairbanks_1` was 4.08, 5.07, 4.87, 4.80, 5.01, 5.04. Its energy ended at 0.007 and carbon at 0.013, so the 5.04 is comfort. The other three fell the whole way: October Fairbanks 0.793 → 0.464 (best 0.383 at 20M), `building_10004` Denver 0.192 → 0.130, `building_15002` Denver 0.187 → 0.108. Mean of those three at the end is 0.234. Training worst-zone deviation stayed near 20–27 K, explained variance reached 0.99, entropy rose from 44 to 50. One seed. Scores: `complete_port/results/study_r28/eval/`.

### r29 — proposed, not approved (2026-10-20)

A year-shaped run, not more steps on r26. Shared actor. Short action match to the r26 policy, then 200M steps walking January through November four times, with earlier months kept in the batch and a small L2 anchor to the matched weights. Learning rate 3e-5, constant. fp32 training on the 3090. No mid-run eval. After the run, the three checkpoints with the best training productivity are scored on the fp64 binary. Sealed seeds 30000–30019 stay out. There is no December weather window in the data.

### r29b — distill r26 into the shared actor, then 200M (approved 2026-10-20)

Approved as specified by the review. The teacher is r26 at 29,962,240, rolled out with log-std fixed at −1 (σ ≈ 0.37) on the 35 scenarios, two full episodes per environment, 16 environments each, 9,676,800 transitions. Thirty-five factorizations do not fit on the 3090 beside the trainer, so the rollout is five sequential groups of seven scenarios. The transition count is unchanged. Learning rate during the rollout is 0, so the loaded weights do not move. The student then matches the squash locations and the discounted returns for 10 passes at 3e-4. The fp64 gate accepts `building_train_1__fairbanks_1` at or under 0.60, which is the near-0.50 band, and the 35-scenario mean at or under 0.10. A score past 1.0 on that Fairbanks scenario, or a mean past 0.30, is a failed distillation: the action loss is diagnosed, one repair is trained (a lower value weight when that term dominates, otherwise the same loss at learning rate 1e-4, ten more passes from the best pass), and the gate is repeated. The year-shaped run starts only if that repair does not move the action loss. A passing gate starts the 200M, which is the r26 PPO recipe at learning rate 1e-4 on the same 35. The nine unseen buildings stay out of training. Their r26 score, finished on the 3060, is 0.117 (productivity −0.176, energy 0.036, carbon 0.023). The 18 cold scenarios are 0.191 and the 27 warm ones are 0.068. Worst is `building_20011__fairbanks_3` at 0.324.

Stopped at the user's request during pass 6 of 10 (2026-10-20 07:56). Passes 1–5 are saved. Action loss 5.65, 4.19, 3.80, 3.70, 3.72; the value term 20.3, 16.2, 15.5, 14.1, 12.7. The best action match is `student_pass04.bin`. The five rollout groups and `student.passes.json` stay. The gate and the 200M were not started. Trainer check `results/verify_r29b.log` ended **ALL CHECKS PASSED**.

### r30 — one January Fairbanks scenario, full-horizon return (started 2026-10-20)

Hypothesis: the λ = 0.95 advantage drops the quadratic comfort, and standardizing the advantage removes the scale that was meant to outweigh a fuel cut. Train the flat policy from scratch on `building_train_1__fairbanks_1` only. Same settings as r10 except `gae_lambda=1` and `adv_norm=0`. 29,884,416 steps, 256 environments. The 3060 scores that same scenario at about 5M, 10M, 15M, 20M, 25M and the end. No other weather, no other building, no score stop. Control is r10 on this scenario: fp64 2.47 at 47M, worst zone still 52–57 K. A win is a lower fp64 score with the comfort term actually down, not a fuel cut. Driver: `complete_port/results/study_r30/run_r30.py`.

### Result (2026-10-20)

Finished 29,884,416 steps in 35 min, exit 0. fp64 scores: 11.03, 10.21, 10.31, 10.37, 10.21, 9.76. The best point is the last one. Productivity −18.27 → −16.22, energy 0.266 → 0.099, carbon 0.066 → 0.025. The 9.76 is comfort. Fuel fell at every checkpoint after 5M except a small bounce at 15M, and the rooms did not recover. r10 on this scenario is 2.47. Checkpoints: `results/runs/r30_fairbanks30m/ckpt/`. Scores: `results/study_r30/eval/`.

### r31 — paired official return on the plant (started 2026-10-20)

The building does not start cold. Every real air and mass node in `building_train_1__fairbanks_1` is 21 °C. Outdoor at 06:00 on 1 January is −0.6 °C. Left off, the rooms are about 7 °C by day 4. From there, full heat with the dampers open recovers the mean to 18 °C in about 6 hours, and the official return over the next 21 hours is about half as bad as staying off (−244 against −484). A mild setpoint does not recover and scores worse than staying off. Hot water that the dampers never deliver does nothing.

Hypothesis: the supply-air and hot-water setpoints should move by the official-return difference of a persistently hotter plant, while the zone dampers keep r10's advantage. One scenario, the flat policy from scratch, r10's settings (`gae_lambda=0.95`, `adv_norm=2`). 256 learning environments. Each has a probe that repeats its action for the whole 256-step window with those two setpoints raised by 0.25 (+5 K water, +2.5 K supply air). The discounted difference of the official return, divided by 0.25, is the mean's gradient. It is the derivative with respect to a setpoint that stays raised, not the derivative of one decision. Probe rows do not train. 29,884,416 learning steps (59,768,832 counted by the trainer). Seed 73. The 3060 scores the same scenario at the six checkpoints. No score stop. A win is a lower fp64 score because the rooms got warmer. Control is r10 at 2.47 and r30 at 9.76. Driver: `complete_port/results/study_r31/run_r31.py`.

### Result (2026-10-20)

Finished 29,884,416 learning steps in 66 min, exit 0. fp64 scores at 5M, 10M, 15M, 20M, 25M and 30M learning steps: 7.26, 6.65, 1.30, 1.23, 1.57, 2.11. The best point is 20M: productivity −1.96, energy 0.084, carbon 0.172. The first two points were a fuel cut (energy 0.021 then 0.011) with productivity still near −12. The drop to 1.30 is the rooms warming, and the fuel went up with them. After 20M the fuel fell again and comfort worsened. r10 on this scenario is 2.47. r30 ended at 9.76. One scenario, one seed. Checkpoints: `results/runs/r31_fairbanks30m/ckpt/`. Scores: `results/study_r31/eval/`.

### r32 — paired search in the weights (started 2026-10-20)

A small step from off is worse than staying off (−689 against −484 over 21 hours), and full heat with the dampers open is better (−244). The fixed one-sided damper step was rejected: it is the same push in a cold month and a hot one, and it leaves the air handler running. The observation already has the zone temperature and the clock.

Hypothesis: a random change of the whole network, tried in both directions and kept only when the month's official return improves, can cross that ridge. The trial is the weight noise. The action played is the mean. 16 directions, step 0.05, noise scale 0.02, 8 updates, 32 environments, full 8640-step episodes. Same flat policy from scratch, seed 73, January Fairbanks only. The 3060 scores the starting weights and every update. No score stop. A win is a lower fp64 score than 1.23 that is still there at the last update, because the rooms came back. Control is r31. Driver: `complete_port/results/study_r32/run_r32.py`.

32 environments is the fastest measured step for this one building, about 10 ms. At 256 environments the same step is about 19 ms, and every copy plays the same action, so the extra environments add time and no new trial. The squash, the entropy schedule, and the plant-probe changes act on the PPO update. This run does not take that update.

Power cut at 2026-10-20 15:22 local, after the random-start checkpoint only. That score was 9.14 (productivity −15.17, energy 0.137, carbon 0.061), kept under `results/study_r32/eval/poweroff/`. No update had been saved, so the search was started again from the same seed. Verification was already passed and was not repeated.

Second power cut the same day. The container is gone. Checkpoints 276,480 and 9,400,320 stay on disk. The search is held and is not resumed.

### r33 — shared actor, anticipation from the observation (2026-10-20)

Reinforcement learning only, on the 3090. One scenario, `building_train_1__fairbanks_1`, shared actor from scratch, seed 73, 30,081,024 steps. The action is the network. Nine features (time to the comfort band, tomorrow's workday, outside-air change, and a 4-hour per-zone memory) are computed from the observation. Training adds a comfort potential, a comfort-start value, and separate productivity, energy and carbon heads. Decisions are 15 minutes: the per-step discounts are the cube roots of 0.99 and 0.95. Episodes used for learning can start between 02:00 and 06:00. Every agent step appends the five reward parts to `results/study_r33/trajectory.bin`. No evaluation during training. No score stop. Control is r31 on this scenario, 1.23 at 20M and 2.11 at 30M. r32 is not the control and is not loaded. Driver: `complete_port/results/study_r33/run_r33.py`.

### Result (2026-10-20)

Finished 30,081,024 steps in 46 min 42 s, exit 0. Six checkpoints, every 5,013,504 steps, through `results/runs/r33_fairbanks_rl/ckpt/0000000030081024.bin`. The trajectory is 574 MB. No fp64 score yet.

The dashboard figure of 0.05 at 2.36M is not a month. The stored trajectory's 8640-step windows at that point score 6.5, and no full window in the run is near 0.05 (the lowest is about 5.4). All 256 agents sit in that band. The logged return at the 0.05 line is −434, about an hour of ordinary reward reported with length 8640. The last stored month scores about 13. Entropy rose from 43 to 75. Explained variance ended at −0.28.

### r34 — one return, the competition score (2026-10-20)

Started 2026-10-20 13:21 UTC on the 3090. The learned return is the competition score: 0.6 × productivity − 0.2 × energy − 0.2 × carbon, on the raw terms. No potential is added, and the three terms are not standardized. Same actor, scenario, and seed as r33. Entropy coefficient 0.001, constant. Decisions are held for 15 minutes. 30,081,024 steps. The bar on this scenario is r14 at 0.856.

### Result (2026-10-20)

Finished in 46 min 48 s, exit 0. fp64 scores at 5.0M, 10.0M, 15.0M, 20.1M, 25.1M and 30.1M: 5.220, 5.268, 5.329, 9.780, 10.138, 11.619. The best point is the first. At 5.0M, productivity −8.684, energy 0.019, carbon 0.030. At 15.0M the fuels are almost off (energy 0.001, carbon 0.002) and productivity is −8.880. At 30.1M productivity is −19.316, energy 0.086, carbon 0.062. Explained variance ended at 0.09. The advantage was still three separate returns, and the critic did not see the delay features. Scores: `complete_port/results/study_r34/scores/`.

### Next — 50M, one scalar return

The 50M run keeps this one Fairbanks month and the competition return. The advantage is the scalar GAE of that return (`component_values=0`), not a mix of three heads. The value is one head. It reads a delay summary of the same features as the actor: minutes to comfort, the outside-air change, the worst current and 4-hour deficits, the coldest slope, the worst and the mean arrival deficit, and the worst idle-damper path (`value_delay=1`). A decision every 5-minute step (`action_repeat=1`, γ = 0.99, λ = 0.95). `vf_clip_coef` 10000, so the value gradient stays on for a return of several hundred. `tanh_squash` on, `squash_chain_floor` 0.25, `tanh_mean` off: the mean can sit near a legal heating action and still move. Entropy coefficient 0.001, constant. Prestart off. From scratch. No potential, no per-term standardization, no baseline in the action, no score stop. 50,135,040 steps, a checkpoint every 5,013,504, each scored fp64 on this month. Gate `results/verify_20261020_200341.log` ended **ALL CHECKS PASSED**. Started as r35 on the 3090. Bar: r14 at 0.856. At 5M the explained variance has to be high; a value that still explains about a tenth of the return is a numerical failure. Driver: `complete_port/results/study_r35/run_r35.py`.

### Result (2026-10-20)

Finished 50,135,040 steps in 58 min, exit 0. The critic fits: explained variance is mostly 0.7–0.98, and entropy stays near 40. The stored 8640-step months are real, and all 256 agents agree. They score 3.96 at 2.2M (plant still on: energy 0.097, carbon 0.159), then 5.01–5.12 from 4.4M through 31M with energy near 0.001, then 6.3–7.7 from 33M to 42M while electricity returns and the rooms get colder, then 5.05 at 48.7M. fp64 of the 5.0M and 10.0M checkpoints is 5.065 and 5.071 (productivity −8.44, energy 0.0003, carbon 0.0005). The 5.05 is the occupied-hour comfort term; nights score about 0. Weekday mornings score about 13.7 with the worst zone near 32 K. The scenario is unsolved. The best saved point so far is the first checkpoint, and it is above r14's 0.856. fp64 scores at 5M through 50M: 5.065, 5.071, 5.071, 5.071, 5.072, 5.072, 7.368, 7.385, 5.340, 5.072. The best checkpoint is the first. Productivity stays near −8.45 except at 35M and 40M, where it is −12.3 while electricity is on and gas is not. Scores: `complete_port/results/study_r35/scores/`.

### r36 — same month, λ = 1 (started 2026-10-20)

Do not continue r35. After month two those weights are the plant-off policy. The first heated month, 3.96 at 2.2M, was not checkpointed. r36 is from scratch, same scenario, same binaries as r35. One change: λ = 1, so an action keeps the day inside a 288-step rollout. γ stays 0.99. Checkpoint every finished month, 2,211,840 steps. 39,813,120 steps, eighteen months, the 40M length that ends on a month. A win is a later month whose productivity is below the first month while gas is still in use. A month near 5 with the fuels near zero is the r35 outcome again. Driver: `complete_port/results/study_r36/run_r36.py`.

Stopped at the user's request after about 33 minutes, near 19M, so r37 could take the 3090. Checkpoints through 17.7M stay in `results/runs/r36_fairbanks_rl/ckpt/`. Occupied-day `ev_ret` was still falling, about −500 in the 16–20M block, and the empty-day updates were the ones moving the policy.

### r37 — same month, environments spread across the calendar (started 2026-10-20)

r35's flags, from scratch, on the r35 binaries. One change: `env.desync=1`. The first episode of each environment ends at a different step. Later episodes are full months and stay offset, so a 288-step batch holds every hour instead of one shared day. γ = 0.99, λ = 0.95, reward scale 1. 50,135,040 steps, a checkpoint every 5,013,504. Scoring is the real Sunday 06:00 reset with desync off. The value on occupied days has to reach a positive `ev_ret`. A plant-off month near 5 is the r35 outcome again. Driver: `complete_port/results/study_r37/run_r37.py`.

### Result (2026-10-20)

Finished 50,135,040 steps in 62 min, exit 0. fp64 scores at 5.0M through 50.1M: 1.483, 2.373, 1.201, 0.540, 0.429, 0.330, 0.473, 0.452, 0.440, 0.605. The best point is 30.1M: productivity −0.297, energy 0.227, carbon 0.534. That is below r14's 0.856 on this month. After 30M the score rose, and at 50.1M productivity is −0.786 while energy is 0.202 and carbon is 0.465. One scenario, one seed. Scores: `complete_port/results/study_r37/scores/`.

### r38 — entropy of the action the building executes (started 2026-10-20)

r37's flags, from scratch. One change: the entropy bonus is the entropy of tanh of the pre-squash draw, the action the building executes, instead of the Gaussian before the squash. The Gaussian term stays as the entropy of that pre-image, and the log-determinant of tanh is added. The noise is held fixed, so a sample on the rail pulls the location back. Entropy coefficient stays 0.001. Desync stays 1, λ stays 0.95, reward scale 1, squash floor 0.25. 50,135,040 steps, a checkpoint every 5,013,504, seed 73. Scoring is the Sunday 06:00 reset with desync off. A win is an fp64 score below 0.330, with the worst zone under about 35 K and gas at or under about 400 MWh, while `logit_outside` flattens or falls under about 0.7 and entropy stops climbing. Driver: `complete_port/results/study_r38/run_r38.py`.

### Stopped at 43.4M (2026-10-21)

The trainer is gone. Last TensorBoard step is 43,352,064 (epoch 588, 54 min). Checkpoints exist through 40.1M (`0000000040108032.bin`). The 45.1M and 50.1M saves were not written. The driver state is still `prepared`. No fp64 score exists. The log ends mid-line.

On full desync training episodes the best point is 42.1M, and those weights were not saved: score 0.194, productivity −0.158, energy 0.161, carbon 0.331, gas 229 MWh, electricity 14.6 MWh, mean deviation 5.1 K, worst zone 32.4 K, 72% of steps outside the band. The 40.1M checkpoint's episode is 0.265. The stop is 0.271, with gas back to 298 MWh, electricity 25.5 MWh, and the worst zone 43 K. r37's best fp64 month is 0.330 at 30.1M (productivity −0.297, energy 0.227, carbon 0.534). On r37 the desync episode and the fp64 month stayed within about a tenth, so this band is plausible and unscored.

The configuration matches the recommendation. The args differ from r37 only in the run name. `desync=1`, `ent_coef=0.001`, learning rate 1e-4, chain floor 0.25, and the rest of the r37 list are unchanged. Gate `verify_20261020_235257.log` ended **ALL CHECKS PASSED**. The bonus is the Gaussian entropy of the pre-squash draw plus `log|d tanh/dx|` at that draw, with the sampled noise held fixed. That is the entropy of the action the building executes.

`logit_outside` fell to 0.16 and `squash_chain_lt_001` fell to about 0, from r37's 0.85 and 0.44. The 8–18M stretch is a fan spike: score 8.3 at 10.1M, electricity 186 MWh, worst zone 62 K. After 32M the clip fraction is 0.25–0.65 and KL reaches 1.14, because the means can move at the same learning rate. The log-std gradient still contains the Gaussian +1, so a sample sitting on its location still rewards width. Training with desync off is r35, which stayed near 5.07 with the plant off.

### Continued to 50.1M (2026-10-21)

Loaded `r38_fairbanks_rl/ckpt/0000000040108032.bin` and trained 10,027,008 more steps, the remainder of the original budget. Same flags, including `desync=1` and `ent_coef=0.001`. Adam started cold. Desync phases were drawn again. New checkpoints are `results/runs/r38b_fairbanks_rl/ckpt/` at 5,013,504 and 10,027,008 local steps, which are 45.1M and 50.1M of the original run. The earlier r38 binaries were not overwritten. Finished in 12 min 22 s, exit 0.

Desync training episodes, cumulative steps. At 45.1M: score 0.205, productivity −0.233, energy 0.099, carbon 0.225, gas 157 MWh, electricity 5.5 MWh, worst zone 34 K, `logit_outside` 0.12. At 50.1M: score 0.197, productivity −0.230, energy 0.089, carbon 0.204, gas 143 MWh, electricity 5.0 MWh, worst zone 38.5 K, `logit_outside` 0.12, squash derivative 0.78, clip fraction 0.27. The best episode was 0.161 at 42.5M and was not saved. A bump at 48.1M reached 0.328. r37's training episode at 50.1M was 0.576, with gas 350 MWh, electricity 9.6 MWh, productivity −0.723, and `logit_outside` 0.85. No fp64 Sunday score yet.

### fp64 Sunday scores (2026-10-21, GPU 1)

Desync off, deterministic, the boxed action map these weights were trained with. r37's best on this month is 0.330 at 30.1M (productivity −0.297, energy 0.227, carbon 0.534, gas 374 MWh, electricity 9.7 MWh, worst zone 51 K).

| step         | score | productivity | energy | carbon | gas MWh | elec MWh | worst K |
| ------------ | ----- | ------------ | ------ | ------ | ------- | -------- | ------- |
| 5.0M         | 6.096 | −10.061      | 0.111  | 0.187  | 125     | 14.0     | 52.9    |
| 10.0M        | 5.847 | −9.333       | 0.525  | 0.709  | 454     | 95.0     | 52.9    |
| 15.0M        | 7.789 | −12.554      | 0.682  | 0.601  | 338     | 169.7    | 60.3    |
| 20.1M        | 0.790 | −0.867       | 0.639  | 0.709  | 433     | 145.0    | 53.5    |
| 25.1M        | 0.416 | −0.233       | 0.442  | 0.937  | 650     | 33.8     | 44.6    |
| 30.1M        | 0.224 | −0.197       | 0.169  | 0.359  | 248     | 12.7     | 35.4    |
| 35.1M        | 0.273 | −0.348       | 0.097  | 0.224  | 157     | 4.6      | 21.2    |
| 40.1M        | 0.184 | −0.228       | 0.073  | 0.161  | 112     | 4.5      | 22.3    |
| 45.1M splice | 0.197 | −0.201       | 0.111  | 0.272  | 192     | 4.0      | 26.8    |
| 50.1M splice | 0.284 | −0.346       | 0.112  | 0.268  | 190     | 4.9      | 27.5    |

The best saved point is 40.1M, score 0.184. Gas is 112 MWh and the worst zone is 22 K. The 45.1M and 50.1M rows loaded that checkpoint with a cold optimizer; the Sunday month went back up to 0.284. One seed. Scores: `complete_port/results/study_r38/scores/`.

### r39 — raw-logit tanh-Gaussian, from scratch (started 2026-10-21)

Freezing log-std does not move a location stuck inside `3*tanh(logit/3)`. Shrinking the width walked the heat off in r13 and r16. The σ climb on r38 is the ordinary Gaussian score `(diff²/var − 1)`, and the training windows improved while it rose, so width is not the regression. The standard policy is a Gaussian on the raw logit, `tanh` of that draw as the executed action, and the entropy of that executed action. The chain is 1. Log-std stays learned at `ent_coef` 0.001. One change from r38: `policy.squash_box=0`, and the 0.25 chain floor is off because it only stood in for the dying box derivative. From scratch, seed 73, desync 1, 50,135,040 steps, fp32 on GPU 0. The r38 and r38b checkpoints are scored fp64 on GPU 1, desync off, with the boxed action map they were trained under. r38b's 45.1M and 50.1M points are a cold-optimizer splice. Driver: `complete_port/results/study_r39/launch.sh`.

The first start exited in under a second: the ini only accepts keys that already exist, and `squash_box` was not one of them. No steps were trained. The key is now in `config/smart_buildings.ini` with default 1, and this run passes 0. Restarted from scratch at 2026-10-21 04:02 UTC. The trainer log says the Gaussian location is the raw logit.

Stopped at 39.1M on instruction, 2026-10-21. The update had already failed: KL 8.2e6, clip fraction 1, entropy −2843, productivity −8.46, electricity 0, gas 1.3 MWh, score about 5.07. No result is taken from r39.

### r38 40.1M, where the −0.228 productivity sits (2026-10-21)

fp64 Sunday month of `r38_fairbanks_rl/ckpt/0000000040108032.bin`, desync off. Score 0.184, productivity −0.228, energy 0.073, carbon 0.161. Per-step zone temperatures and occupancy, 39 zones. The occupancy-weighted oracle formula rebuilds that productivity. Night 19:00–06:00 and weekend daytime 06:00–19:00 contribute 0. Morning −0.075, afternoon −0.085, evening −0.068. The occupied deviation is 1.7 K at 07:00 and 4.4 K at 18:00. Hours 12:00–18:00 hold about two thirds of the loss. 06:00 is about 8% of it. 84% of the loss is below the heating setpoint. The largest zone is 8.3% of the loss. The top eight are 47%. The 22.3 K worst zone is an empty night against the 16°C setback and is not billed. Dump: `complete_port/results/study_r38/r38_40m_zones.bin`.

### r40 — r38 again, optimizer saved, 70M (started 2026-10-21)

Same r38 binaries and flags, from scratch, seed 73. The patch hash is `dba0af`, the train binary `b2fa3c`, the score binary `cff73b`. One change: `checkpoint_optim=1`, so each checkpoint writes its Adam sidecar. Nothing is loaded. 70,189,056 steps, 952 epochs, a checkpoint every 5,013,504, fourteen of them. fp32 on GPU 0. Sunday scoring is fp64 on GPU 1 after training, desync off. Driver: `complete_port/results/study_r40/run_r40.py`.

`complete_port/results/study_r40/score_10m.sh` waited for the 40.1M checkpoint, then scored 40.1M, 50.1M, 60.2M, and 70.2M on GPU 1. Each point is the Sunday month of `building_train_1__fairbanks_1` and of `building_train_1__atlanta_2`. Training finished at 07:11 UTC, exit 0, 85.9 min.

Fairbanks, lower better. The 40.1M month matches r38's 0.184.

| step  | score | productivity | energy | carbon | gas MWh | elec MWh | worst K |
| ----- | ----- | ------------ | ------ | ------ | ------- | -------- | ------- |
| 40.1M | 0.184 | −0.228       | 0.073  | 0.161  | 112     | 4.5      | 22.3    |
| 50.1M | 1.290 | −2.016       | 0.147  | 0.253  | 170     | 19.5     | 50.6    |
| 60.2M | 0.191 | −0.179       | 0.144  | 0.276  | 188     | 15.5     | 43.0    |
| 70.2M | 0.340 | −0.443       | 0.129  | 0.238  | 162     | 14.7     | 44.5    |

Atlanta on the same weights:

| step  | score | productivity | energy | carbon | gas MWh | elec MWh | worst K |
| ----- | ----- | ------------ | ------ | ------ | ------- | -------- | ------- |
| 40.1M | 0.356 | −0.546       | 0.051  | 0.093  | 63      | 6.2      | 16.6    |
| 50.1M | 0.355 | −0.539       | 0.057  | 0.103  | 69      | 7.4      | 16.6    |
| 60.2M | 0.385 | −0.581       | 0.069  | 0.112  | 74      | 10.6     | 15.7    |
| 70.2M | 0.206 | −0.278       | 0.084  | 0.112  | 71      | 15.5     | 12.0    |

Fairbanks' best of those four was 40.1M. Atlanta's best of those four is 70.2M. One seed. Scores: `complete_port/results/study_r40/scores/`.

The driver's Fairbanks pass of all fourteen checkpoints finished at 07:33 UTC. Through 40.1M the Sunday scores match r38. The best saved point is 65.2M, score 0.171, productivity −0.197, energy 0.092, carbon 0.173, gas 118 MWh, electricity 9.6 MWh, worst zone 28.0 K. 55.1M is 0.179. 45.1M is 0.234. 70.2M is 0.340. All fourteen checkpoints have an Adam sidecar.

### r41 — separate value network, from scratch (2026-10-21)

r40's 49M to 50M step is the measured failure. `loss/ev_ret2` went from 3,875 to 55,256 and `loss/value` from 39 to 958, while advantages stayed normalized. The value residual is not. One global clip of 0.5 then pointed the shared trunk at that residual, and the Sunday month went to 1.290, of which about 1.21 is comfort. The same marker, `norm/logit_outside` rising while the value loss is large, is the 70.2M regression from 0.171 to 0.340.

Andrychowicz et al. 2020 (arXiv:2006.05990, choice C47) found a value network with no learned layer shared with the policy better on four of five continuous-control tasks. Stable-Baselines3's current default is the same split: separate actor and critic MLPs, one summed loss, one global clip. The return is still the mixed score. `ent_coef`, the squash, the chain floor, and `rew_norm` stay as in r40.

The training binary is the shared-policy build. The critic is now a two-layer tanh MLP of width 256. Its input is the fixed building summary the encoder already computes, plus the eight delay numbers when `value_delay=1`. The value gradient is not added into the trunk. The old linear value row stays in the checkpoint and receives a zero gradient, so the finite-difference check still lines up. One global clip of 0.5 remains.

From scratch, seed 73, r40's flags, `checkpoint_optim=1`, 70,189,056 steps, fp32 on GPU 0. Sunday scoring is fp64 on GPU 1 after training, desync off. A win is a Fairbanks month under 0.171 that does not jump when `loss/value` spikes, with productivity, energy, carbon, and both fuels. One seed. Driver: `complete_port/results/study_r41/launch.sh`.

Power cut at 23.1M, 2026-10-21 16:30. The last finished save is 20,054,016. Its weights and Adam sidecar are both 2,020,416 bytes. The continuation loads that pair, keeps the same flags, and trains the remaining 50,135,040 steps. The new counter starts at 0, so its 50,135,040 is the original 70.2M. Desync phases are drawn again. Driver: `complete_port/results/study_r41/continue_r41.py`, run `r41b_fairbanks_rl`. Stopped on purpose at local 37.3M. The critic's output row grew in a straight line at the learning rate, and explained variance was high only while the return scale sat under that bound.

### r42 — PopArt on the separate critic (2026-10-21)

r41's last row is a matrix, so Muon normalizes it and it moves by the learning rate no matter how large the value error is. The value cannot exceed that row's L1 norm, and there is no bias. r40's critic did fit. Its ceiling is the unnormalized return, up to about 75× inside one run, entering the actor through the shared trunk and the one global clip.

PopArt (van Hasselt et al. 2016) keeps r41's critic. The value is σ times the normalized head plus μ. μ and the second moment update once per epoch from the GAE return, β = 1 on the first epoch and 0.1 after. The backward into the head is divided by σ. The last row, the bias, and their Muon momentum are rescaled in that same step so the value is unchanged. The return, the actor, and the single clip of 0.5 stay as in r40.

From scratch, seed 73, r40's flags, 50,135,040 steps, fp32 on GPU 0. Sunday scoring is fp64 on GPU 1 after training, desync off. A win is a Fairbanks month under 0.171. One seed. Driver: `complete_port/results/study_r42/launch.sh`.

Training finished 14:15 UTC, exit 0. The best of the ten Sunday months is 30.1M, score 0.926, productivity −1.339, energy 0.213, carbon 0.400. It then went back to 2.1–2.4. r40 at 30.1M was 0.224 and its best is 0.171. The critic fitted; the month did not. Scores: `complete_port/results/study_r42/scores/`.

### r40 audit — frozen replays and update semantics (2026-10-21)

No new training. On the user-assigned 3060, the existing fp64 diagnostic binary exactly reproduced r40's 65.2M and 70.2M stored score fields in 85.1 s and 84.8 s. Best: 0.170987, productivity −0.196890, energy 0.091458, carbon 0.172806, electricity 9.565 MWh, gas 117.640 MWh. Final: 0.339512, productivity −0.443319, energy 0.129361, carbon 0.238242, electricity 14.669 MWh, gas 161.736 MWh. Occupied deviation rises 2.95→5.78 K; all three scored costs worsen. The best point's comfort loss is 63% cold, 37% hot. The late regression lacks the 50M value/KL explosion, so critic dominance is not established as its cause.

The audit found reused-behavior-sample entropy after policy updates, latent continuous entropy on five thresholded run heads, changing GAE with stale normalization statistics, and policy-loss logging that omits the geometric surrogate's ×97 scale. CPU counterexamples are retained. Optimizer sidecars are Muon momentum, not Adam; the r40 manifest's fp64-training text is wrong (fp32 training, fp64 scores). No source correction or follow-up training was applied; r42/3090 remained untouched. Full curve, time/zone/action decomposition, verified fuel reconstruction, and an unlaunched single-factor proposal: `complete_port/results/audit_r40_20261021/REPORT.md`.

### r43 — current-distribution entropy from r40 (approved 2026-10-21)

User approved the audit's first proposal on the now-free 3090. Two matched restarts from r40's 65,175,552 weights and Muon momentum, 20,054,016 steps each, control then treatment. The only learning change is fresh current-policy entropy samples for continuous commands at each PPO visit; the five active run-head latent regularizers remain as in r40. Preserve its scalar shared critic, reward, features, optimizer, flags and fp32-training/fp64-scoring convention. Parent and four points per arm at 5,013,504-step intervals; GPU1 scores and decomposes the same Fairbanks month. Primary comparison is the four-point mean; practical improvement also requires a best registered point below 0.170987. No score-based stopping. Full gate and archived-binary first-update parity are prerequisites; training has not started at this entry. Detailed protocol, checks, frozen runtime and driver: `complete_port/results/study_r43/`.

Verification `complete_port/results/verify_20261021_200003.log` ended **ALL CHECKS PASSED**. Parent replay matched every stored score field. Archived and restored controls produced byte-identical first-epoch weights and Muon sidecars; the fresh-noise graph check produced 128 finite diagnostic visits. Controller PID 133572 started `r43_control_r40` on GPU0 at 14:51:26 UTC; `r43_entropy_current` is queued automatically after the fixed control budget. Per-arm checkpoints are local continuation steps. Live status: `complete_port/results/study_r43/study_status.json`.

User stopped all experiments on 2026-10-21 and rejected checkpoint-initialized classical RL studies. r43 control was terminated before its fixed budget completed; the treatment never started. The three saved registered checkpoints and diagnostics are preserved, but this is an incomplete, cancelled comparison. No from-scratch replacement of that checkpoint study is authorized. AGENTS.md now requires random initialization and fresh optimizer state for new arms, with checkpoint training resumes allowed only by an explicit user exception. Stop details: `complete_port/results/study_r43/user_stop.json`.

### r44 — joint KL gate, Fairbanks and Atlanta, from scratch (2026-10-21)

r40's logged `loss/kl` stays near 0.03 while `loss/joint_sampled_kl` has median 19 nats. The clip divides the joint log-ratio by 97, so a move of about 18 nats is still inside it. On the r43 continuation, the actor-head step was 0.000315 in L2 on every minibatch visit, whether the gradient was 200 or 590. That is Muon. `ent_coef` stays 0.001: it is about a thousandth of a unit normalized advantage, and it does not set the step.

The gate closes the actor and the trunk once a visit's sampled joint KL exceeds 1 nat. The shared value row keeps stepping. The value is r40's, on the MinGRU, with the delay features. The second weather is `building_train_1__atlanta_2`, 128 envs beside 128 Fairbanks envs, one merged group. From scratch, seed 73, 140,378,112 steps, so each weather gets about the exposure r40 had. A win is a Fairbanks month under 0.171 and Atlanta not worse than its 0.206 transfer. One seed. Driver: `complete_port/results/study_r44/launch.sh`.

Stopped at about 37.6M on 2026-10-21. The gate was the wrong assumption. Logged per-head KL was 0.00014 and the policy loss about −0.04, so the actor was not taking the step r40 learned with. Fairbanks training episodes sat near 5 from 9M (4.86 at the stop). Atlanta eased from 0.24 to 0.09. The oscillation is those two populations alternating. Logit-outside was still 0.38. r40's month fell from 0.42 to 0.17 while joint KL stayed near 20 nats, so a 1 nat cap removes the update that learned. Checkpoints kept. No Sunday fp64 score. `complete_port/results/study_r44/user_stop.json`.

### r46 — zone advantage, five weathers interleaved (2026-10-21)

Approved run. r40's update, from scratch, seed 73. The return stays R. Zone reheat and damper heads use that zone's occupancy-weighted productivity share plus the building's energy and carbon. Plant heads keep the full R. The pieces are not normalized. The five official `building_train_1` weathers are interleaved so every minibatch of 8 environments contains all five. `ent_coef` 0.001, minibatch 2304, no KL stop. 330,891,264 steps, so Fairbanks sees about as many transitions as at r40's 0.171 month. The 3060 runs all five weathers for thirty days at once. The reading is each weather on its own, productivity, energy, and carbon, not the average. The kept checkpoint is the one whose worst weather is best. A falling average with Fairbanks near 5 is a failed run. This is a five-weather fit, not a private-leaderboard result. Driver: `complete_port/results/study_r46/launch.sh`.

Finished 2026-10-22 03:01 UTC. All 66 checkpoints scored. Kept `0000000070189056` at 70.2M, where Fairbanks is the worst city at 0.174 (productivity −0.175, energy 0.122, carbon 0.226). At that same point Atlanta 0.087, Denver 0.080, Phoenix 0.105, Singapore 0.079. Fairbanks then left that level and ended at 6.70, while the other four stayed near 0.04–0.17. Their own bests were earlier and lower (Atlanta 0.033, Denver 0.028, Phoenix 0.052, Singapore 0.032) at checkpoints where Fairbanks was already worse. Scores: `complete_port/results/study_r46/scores/selection.json`.

### r47 — Adam against Muon, Fairbanks only (2026-10-22)

PufferLib's optimizer is Muon, not Adam. `train.adam=1` is Kingma Adam with the PyTorch defaults: β1 0.9, β2 0.999, ε 1e-8, bias correction, weight decay 0. The existing grad clip of 0.5 is applied before the moments. The learning rate stays 1e-4. Default 0 leaves Muon unchanged. r40 is a Muon curve on an older binary, so both arms are this binary, from scratch, seed 73, `building_train_1` Fairbanks, shared value, zone advantage off. 30,081,024 steps, six checkpoints, no evaluation. Adam on the 3090, Muon on the 3060. Driver: `complete_port/results/study_r47/launch.sh`.

Both arms exited 0 at 16:58 UTC. Adam took 37 minutes on the 3090, Muon 73 minutes on the 3060. Six checkpoints each, through 30,081,024. The training panels were desynchronized episodes. The fp64 months below are the comparison.

Every registered checkpoint was replayed deterministic, desync off, one Fairbanks month, on the frozen fp64 diagnostic binary (no fp32 solve). GPU 0 scored Adam, GPU 1 scored Muon. The same month on the r46 shared-policy fp64 binary agrees to 8×10⁻⁷. Score is −0.6×productivity + 0.2×energy + 0.2×carbon. Fuels are reconstructed from those two terms and the blob tariff, and they match each replay's own fuel log. The protocol names no score threshold: the decision is this arm against that arm. Muon is lower at 10M, 15M, 20M, 25M, and 30M. Adam is lower only at 5M. Muon's best registered point is 25.1M at 0.561. Its 30.1M month went back to 1.008. Adam after 5M is a plant that is nearly off and a month between 5.2 and 9.8. All twelve checkpoints and both sidecars are kept. Curve, bins, and replays: `complete_port/results/study_r47/`.

|  Step |  Adam | productivity | energy | carbon | elec MWh | gas MWh |  Muon | productivity | energy | carbon | elec MWh | gas MWh |
| ----: | ----: | -----------: | -----: | -----: | -------: | ------: | ----: | -----------: | -----: | -----: | -------: | ------: |
|  5.0M | 0.797 |       −1.112 |  0.219 |  0.430 |     20.7 |     294 | 1.525 |       −2.425 |  0.134 |  0.216 |     18.7 |     143 |
| 10.0M | 5.160 |       −8.599 |  0.001 |  0.001 |     0.04 |    0.94 | 1.481 |       −2.324 |  0.157 |  0.278 |     18.2 |     187 |
| 15.0M | 7.655 |      −12.753 |  0.011 |  0.005 |      3.3 |     2.0 | 1.152 |       −1.592 |  0.407 |  0.578 |     70.7 |     375 |
| 20.1M | 6.962 |      −11.600 |  0.004 |  0.003 |      1.2 |     1.5 | 0.631 |       −0.309 |  0.648 |  1.583 |     15.4 |    1116 |
| 25.1M | 9.825 |      −16.260 |  0.267 |  0.075 |     90.8 |     5.9 | 0.561 |       −0.321 |  0.541 |  1.298 |     13.4 |     913 |
| 30.1M | 8.578 |      −14.289 |  0.017 |  0.005 |      5.6 |    0.74 | 1.008 |       −1.303 |  0.343 |  0.787 |     14.7 |     550 |

Adam's month after 5M is almost all occupied-hour comfort. At 30.1M the night share is 0.003 and the morning, afternoon, and evening shares are 4.01, 2.61, and 1.96. Muon at 25.1M splits as night 0.224, morning 0.139, afternoon 0.071, evening 0.052, weekend daytime 0.074, with 913 MWh of gas. One seed. Not a leaderboard result. r40 on the older binary was 0.224 at 30.1M and 0.171 at 65.2M; this pair does not replace that month.

### r48 — fresh entropy and the sibling block

Both arms are Fairbanks only, seed 73, Muon, fresh current-policy entropy, shared value, from scratch unless a later paragraph says a checkpoint was loaded. A continuation is a new process: the random streams, the first-episode cut, and the hidden state start again. Those months are a second segment. Score is −0.6×productivity + 0.2×energy + 0.2×carbon, productivity stored signed. One seed is not a leaderboard result. The public target is 0.00598.

The 60-nat joint-KL gate stays dead. On the first entropy run it would have closed 61% of epochs and 98% of epochs in the 15–23M window, which is where that month fell from 2.2 to 0.5. A future gate needs a per-visit open-fraction and a threshold at the operating p99, about 400 nats.

#### Entropy

Three runs. Curves: `curve_a.json`, `curve_a_to70.json`, `curve_a_scratch80.json`.

The first run, 0–30.1M, ends at its best point, **0.512**. Productivity −0.644, energy 0.228, carbon 0.399, electricity 32.2 MWh, gas 269 MWh. The morning ramp is 0.286 of that 0.512, 56% of the month. Afternoon is 0.099 and evening is 0.108. Night is 0.010. Thirty zones carry 90% of the occupied comfort loss. Zone 27 is the coldest single zone, at 41 K, and it is 7% of the loss.

The continuation from that 30.1M checkpoint, a separate process through original step 70.2M, has its best at **0.162** at 65.2M. Productivity −0.205, energy 0.057, carbon 0.137, electricity 3.0 MWh, gas 97 MWh. The morning bin is still 60% of the month (0.097 of 0.162), and the same 25–35 zones share the loss. Zone 27 is again the coldest, at 33 K, and it is 11% of the loss. At 50.1M the same continuation scores 4.917 with productivity −8.19 and gas 7.6 MWh: the plant is nearly off. The 70.2M point is 0.241. `curve_a_to70.json` stores null in the 35.1M slot; that month was 3.223 when it was scored.

The from-scratch 80.2M run is a third process. Its scratch segment is best at **0.134** at 70.2M: productivity −0.115, energy 0.099, carbon 0.225, electricity 6.7 MWh, gas 157 MWh. That is the lowest registered Fairbanks month on this recipe. The next checkpoint is 0.231 at 75.2M. The resumed segment, loaded from 75.2M after the process died in epoch 1028, scores 0.188 at 80.2M (productivity −0.239, energy 0.069, carbon 0.153, electricity 4.0 MWh, gas 106 MWh). The eight epochs after 75.2M were not saved.

The 0.134 replay already held the zone dump. The per-step double reconstructs the stored score to 3×10⁻¹¹. Morning contributes 0.060 of the 0.134, 45%. Afternoon contributes 0.039, evening 0.028, night 0.007, and weekend daytime 0.000. The occupied weekday bins score 0.343, 0.330, and 0.318 per step, so the morning's larger share is the longer bin (1513 steps against 1008 and 756). Night productivity is 0. Gas in those bins is 64, 48, 31, and 16 MWh. Thirty-nine zones are in the building. Zone 8 carries 7.8% of the occupied comfort loss, at a worst deviation of 7.6 K. Twenty-six zones reach 90% of the loss, and twenty-six zones are each at least 1%. The coldest zone is 24, at 18.0 K, and it is 3.9% of the loss. Zone 27 is 4.7% at 17.5 K. Split: `results/study_r48/replays/scratch80a_0000000070189056/decomposition.json`.

Where the entropy months stay high:

- Through 25M the building is still cold. The 15.0M month is 9.77, productivity −16.0. The update has not yet paid for heat before occupancy.
- Empty nights have no comfort penalty, so turning the plant down looks cheap. The bill arrives at the Monday morning ramp. That is the 4.9 plant-off month inside the continuation, and the 1.56 month at 50.1M of the scratch run (gas 59 MWh, productivity −2.55).
- The heated months buy comfort with gas. The 0.512 month burns 269 MWh. The 0.162 month burns 97 MWh and is still mostly the morning. The 0.134 month burns 157 MWh for a better comfort term and a worse carbon term than 0.162. The checkpoint after 0.134 gives the gain back.
- The training dashboard on a continuation stayed near 37–42 K worst-zone while the fp64 month moved. The dashboard mixes episode phases. It is not the month.

#### Sibling block

Same recipe, plus the four sibling fields on each zone slot (temperature, damper, reheat, presence). Three runs. Curves: `curve_b.json`, `curve_b_to70.json`, `curve_b_scratch80.json`. The score binary has no zone dump, so later months have totals only.

The first run is best at **0.290** at 25.1M. Productivity −0.249, energy 0.209, carbon 0.493, electricity 5.6 MWh, gas 346 MWh. Every time-of-day bin is at or under 0.060, and the morning bin that cost the entropy arm 0.286 falls to 0.073. Mean occupied deviation is 1.9 K and the worst zone is 24 K. The 30.1M point rises to 0.402. The training log stops at epoch 257 (19.0M) while the six checkpoints are complete, so the descent has no entropy or KL record.

The continuation from that 30.1M checkpoint never beats 0.290. Its best is **0.204** at 55.1M: productivity −0.199, energy 0.138, carbon 0.284, electricity 11.3 MWh, gas 196 MWh. At 60.2M the month is 0.450 with electricity 97 MWh and gas 536 MWh. The 70.2M point is 0.465.

The from-scratch 80.2M run, resumed from 45.1M after it died in epoch 649, is best at the last checkpoint, **0.174** at 80.2M. Productivity −0.221, energy 0.068, carbon 0.139, electricity 5.8 MWh, gas 96 MWh. The scratch segment's own best is 0.379 at 40.1M, with gas 348 MWh. The resume passes through 756 MWh of gas at 60.2M (score 0.573, carbon 1.108) before the gas falls and the score falls with it. The 37 epochs after 45.1M were not saved. There is no zone dump on this 0.174 month.

Where the sibling months stay high:

- The first 20M is a venting building: electricity 50–59 MWh and gas at or under 20 MWh, scores 7.3 to 11.8. Entropy is already descending through that window. The sibling features delay the first heated month.
- The 0.290 month closes the morning bins by burning 346 MWh of gas. Carbon is 0.099 of the 0.290. The comfort term is 0.149. The continuation and the scratch run do not keep a morning-closed month at that gas bill; they either burn still more gas or give the comfort back.
- The 0.174 point is one checkpoint at the end of a resume, after a gas spike. The checkpoint before it is 0.341. It matches the r46 Fairbanks anchor and it is above the entropy scratch point of 0.134.

#### Per-zone critic

`r50_fairbanks_zone_critic_b`, the sibling actor with one value per zone, from scratch, 50.1M. Curve: `results/study_r50/curve_zone_critic.json`. The only low month is **0.439** at 15.0M: productivity −0.466, energy 0.289, carbon 0.509, electricity 33.6 MWh, gas 343 MWh. The other nine months are 1.88 to 7.18. From 20.1M on, productivity is −7.4 to −11.9 and gas is 28–77 MWh. The 50.1M point is the worst, 7.181. The first attempt died in epoch 41 with no checkpoint; this curve is the fresh rerun. The zone-advantage gradient measured earlier sat at the run-to-run noise floor, and this critic did not beat either arm.

The plant-off direction is in the months and in the log. 15.0M is the last heated month. The seven checkpoints from 20.1M to 50.1M stay plant-off: gas falls 343→77 MWh and productivity −0.47→−7.37 at the first of them, and gas stays 28–77 MWh after that. Against the sibling scratch log, the same actor and seed, the shared-trunk gradient stays about 15k–95k in every 5M block while the sibling falls from about 13k at 15M to about 1–6k. One visit at epoch 346 (25.5M) reaches 738k. The 45M block mean is still about 14.5k. Zone-head KL is about 0.64 in the block that ends at 15M, then about 0.05–0.22 through the freeze; the block that ends at 30M averages 0.22. The actor-head gradient stays a few hundred on both runs. The value-row gradient is several times the sibling's through about 35M (about 1,100 against about 100 at 30M). By the 45M and 50M blocks it is at or below the sibling. That late drop fits targets that became trivial once the plant was off. The sibling comparison at 50M is a partial block: that log ends in epoch 649. The value matrix is 91 linear rows on one pooled vector plus the eight delay summaries. Zone damper and reheat heads use the normalized zone advantage in place of the building advantage. The contract test checks that the zone targets sum to the scored return. It does not show that the learned rows tracked those targets.

The damper trace is the fp64 record of the 20.1M and 50.1M checkpoints, run on the 3090 with the staged score binary. Both scores match the stored months (4.461 and 7.181). A command of −1 is a shut valve and +1 is fully open. Occupancy is the building count in the recorded observation: 2963 occupied steps and 5677 empty steps. At 20.1M, occupied reheat averages 0.6% open and 99.4% of those commands are under 5% open. Occupied dampers have median 0.5% open and 78.8% under 5%, with 14.5% still above 95%, so the mean is 18%. Morning dampers are the open tail: median 0.7%, and 26% on the open rail. Hot water while occupied averages 314 K, against 334 K while the building is empty. The setpoint range is 310–350 K. At 50.1M the occupied dampers average 3.9% open, 91.9% are under 5%, and the open rail is 0.9%. Morning dampers are 95% shut and morning reheat is 100% shut. Occupied reheat stays at 0.6% open. Hot water while occupied averages 319 K. The 50.1M month is more shut than the 20.1M month, and at both checkpoints the occupied hours are more shut than the empty hours. Record: `results/study_r50/plantoff_actions.json`.

#### Decision

The entropy recipe is the control. Its lowest month is 0.134 at 70.2M of the from-scratch run, and that point was not held. The sibling block has one morning-closed month at 0.290, paid for with 346 MWh of gas, and a later 0.174 that has not been split by time of day. The 60-nat KL gate stays dead. A rerun of the 100% room-card replacement is not the next experiment. Rebuilding a mix from the raw zone correction repeats the dead `zone_adv` arm, whose visit-0 effect was about 1e-6 of the control gradient. A new zone-conditioned trunk, and a cap on the trunk gradient, are different experiments and are not approved.

The two eval measurements are in. They ran on the 3090. The 0.134 morning share is 45% of the month, and the afternoon and evening score the same per step, so the loss is the occupied weekday spread across 26 zones. Another architecture is not the next experiment. The critic's zone valves are shut while the building is occupied, and the 50.1M checkpoint is more shut than the 20.1M checkpoint.

#### r51 — 30% room card, scoreboards still attached

Approved 2026-10-23. One change from the failed critic: damper and reheat heads use 0.70 times the whitened building advantage plus 0.30 times the whitened zone advantage. Plant and air-handler heads stay on the building advantage. The 91 value rows stay trained and still update the shared trunk. Saying those rows are harmless until they can shake the trunk was a topology argument. No run has trained them alone, so this arm leaves them fully attached and the pull question waits. Detaching them is a later arm, only if this one still freezes and the trunk gradient is still large. It is not part of this run.

Fairbanks, seed 73, from scratch, 50,135,040 steps, the r50 flags otherwise. GPU 0. The visit-0 probe on that same binary printed ratio 0.556 (delta 54973, control 98884), above the 0.1 floor. `r51_fairbanks_zone_mix` then exited 0 at 16:01 UTC, wall 3929 s. Ten checkpoints, each 1,762,496 bytes with a matching sidecar. The log reaches epoch 679, 87,040 updates, with no NaN. During the 50M the only abort is a numerical failure. The trunk gradient staying at 15k–95k is expected, because the value geometry is unchanged, and it is recorded so the later detach arm can be judged. It is not a failure signal. The question is whether the seven months from 20.1M to 50.1M stay heated: gas above 100 MWh and productivity above −1 on at least five of them, and occupied reheat averaging more than 5% open at both 20.1M and 50.1M. One seed. Driver: `results/study_r51/blend.py`. Verify before the binary: `results/verify_20261023_195704.log`, ALL CHECKS PASSED.

The ten fp64 months, curve `results/study_r51/curve_zone_mix.json`:

|  Step | score | productivity | energy | carbon | elec MWh | gas MWh |
| ----: | ----: | -----------: | -----: | -----: | -------: | ------: |
|  5.0M | 2.001 |       −3.246 |  0.096 |  0.168 |     11.4 |     113 |
| 10.0M | 4.581 |       −7.626 |  0.011 |  0.016 |      1.9 |      10 |
| 15.0M | 0.351 |       −0.396 |  0.201 |  0.368 |     21.0 |     250 |
| 20.1M | 4.344 |       −7.214 |  0.031 |  0.050 |      4.2 |      34 |
| 25.1M | 0.299 |       −0.311 |  0.200 |  0.362 |     22.0 |     245 |
| 30.1M | 0.452 |       −0.610 |  0.157 |  0.274 |     19.0 |     184 |
| 35.1M | 0.304 |       −0.420 |  0.096 |  0.162 |     12.4 |     108 |
| 40.1M | 0.323 |       −0.459 |  0.085 |  0.151 |     10.1 |     102 |
| 45.1M | 0.365 |       −0.538 |  0.077 |  0.136 |      9.4 |      91 |
| 50.1M | 0.274 |       −0.350 |  0.115 |  0.206 |     13.3 |     139 |

Five of the seven months from 20.1M to 50.1M have gas above 100 MWh and productivity above −1: 25.1, 30.1, 35.1, 40.1, and 50.1. The 20.1M month is plant-off (productivity −7.21, gas 34 MWh). The 45.1M month stays heated on comfort (productivity −0.54) with gas at 91 MWh. Occupied reheat averages 28% open at 20.1M and 36% open at 50.1M. The 20.1M median is 0.6% open, so that mean is a fully-open tail (23% of commands) over a mostly shut valve. At 50.1M the median is 17% open and 23% of occupied commands are under 5% open. The failed critic's occupied reheat was 0.6% open at both of those checkpoints. Dampers stay mostly shut: occupied means 4.9% at 20.1M and 4.1% at 50.1M. The lowest month is 0.274 at 50.1M, above the entropy month of 0.134.

The trunk gradient did not stay at 15k–95k. The 10M and 20M blocks, the two plant-off months, average about 26k and 19k. From 25M the block means are about 0.2k–11k, and the 50M block is about 700. The freeze did not hold and the late trunk pull came down, so the queued detach arm is not warranted. One seed. Not a leaderboard result.

Continuation approved 2026-10-23. Same binary, same flags, Muon loaded from `0000000050135040.bin` and its sidecar. New run `r51_fairbanks_zone_mix_to85`, 35,094,528 steps, seven checkpoints. The file steps start at 0, so the original steps are 55.1M through 85.2M. This segment is separate from the first 50.1M: the random streams, the first-episode cut, and the hidden state start again. Driver: `results/study_r51/extend_to85.py`.

The continuation exited 0 at 17:18 UTC. The momentum line was `loaded optimizer momentum from .../0000000050135040.bin.mb`. Curve: `results/study_r51/curve_zone_mix_to85.json`.

| Original step | score | productivity | energy | carbon | elec MWh | gas MWh |
| ------------: | ----: | -----------: | -----: | -----: | -------: | ------: |
|         55.1M | 0.233 |       −0.219 |  0.182 |  0.325 |     20.9 |     220 |
|         60.2M | 0.579 |       −0.836 |  0.160 |  0.228 |     27.4 |     148 |
|         65.2M | 0.337 |       −0.435 |  0.150 |  0.228 |     23.4 |     150 |
|         70.2M | 0.319 |       −0.423 |  0.119 |  0.208 |     15.0 |     140 |
|         75.2M | 0.276 |       −0.278 |  0.184 |  0.363 |     17.4 |     249 |
|         80.2M | 0.201 |       −0.280 |  0.055 |  0.111 |      4.8 |      77 |
|         85.2M | 0.207 |       −0.268 |  0.073 |  0.158 |      4.9 |     110 |

This segment's lowest month is **0.201** at 80.2M. The next checkpoint is 0.207. Both stay heated: productivity is about −0.27, and the gas drop to 77 MWh at 80.2M is not the plant-off signature (that was productivity near −7 and gas near 30 MWh). The 60.2M month rose to 0.579. The first segment's 0.274 is not this curve. The entropy month of 0.134 is still lower. One seed. Not a leaderboard result.

#### r52 — official 30 scenarios, entropy recipe

Started 2026-10-23 22:05 UTC on GPU 0, from scratch. The batch is the 30 shipped scenarios, 32 envs each, six groups of 160. Budget 2,105,671,680 steps, 14 checkpoints, first at 150,405,120. Seed 73. No evaluation. Muon sidecar on every checkpoint. Driver `results/study_r52/train30.py`. Gate `results/verify_20261024_030720.log` ended **ALL CHECKS PASSED**.

The binary is the current shared actor, fp32 (`EEv5LuDevPf` present), 7,183,144 bytes, staged at `results/study_r52/runtime/puffer_shared`. The sibling block is compiled in (banner: temp, damper, reheat, presence; 416.9K parameters). Zone critic and zone advantage stay off. Learning flags match the entropy recipe: reward scale 1, entropy coefficient 0.001 with the fresh current-policy entropy, no learning-rate or entropy anneal, value clip 10000, advantage norm 2, squash floor 0.25, zone memory on. CUDA graphs are off. The zone-feature trace is capped at the 288-step rollout (`SB_ZM_TRAJ=288`); the old 1024-step cap did not fit 960 envs in 24 GB.

Epoch 2 is up: 553.0K steps, 11.5K steps/s, VRAM 17.9/24 GB, policy −0.107, value 24.0, entropy 18.5, KL 0.343. The dashboard's remaining time at that rate is 2d 3h. One seed. Not a leaderboard result. A falling mean with Fairbanks plant-off is a failed reading of the mean. 0.134 is the Fairbanks context only.

The first segment stopped during epoch 965. The log ends mid-dashboard at 266.8M steps, 11.8K steps/s, uptime 6h 21m, with no CUDA error. The only checkpoint is `0000000150405120.bin` and its sidecar, both 1,667,456 bytes, written at 150,405,120. The updates from there to 266.8M were not saved.

Continuation started 2026-10-24 04:45 UTC. Same binary, same flags, new run `r52_shipped30_entropy_rest`. It loaded that checkpoint and printed `loaded optimizer momentum from .../0000000150405120.bin.mb`. Remaining budget 1,955,266,560 steps, 13 checkpoints. File steps start at 0, so those files are original steps 300,810,240 through 2,105,671,680. This segment is separate: the random streams, the first-episode cut, the recurrent state, and the zone-temperature memory start again. Learning rate and entropy coefficient stay constant. Driver: `results/study_r52/resume_rest.py`. Epoch 2 of this process is up at 12.1K steps/s. The dashboard's remaining time is 1d 21h. One seed. Not a leaderboard result.

The power cut stopped `r52_shipped30_entropy_rest` during epoch 111, at 30.7M file steps. That directory has no checkpoint. Those updates are gone.

The next segment loads the same 150,405,120 weights again. Run `r52_shipped30_entropy_50m`. The save interval is 181 epochs, 50,042,880 steps, the epoch boundary next to 50M. Thirty-nine files on that grid, then the last file at the remaining 1,955,266,560 steps. Driver: `results/study_r52/resume_50m.py`.

The 06:21 UTC start had CUDA graphs off and was stopped after a few minutes, before any checkpoint. The live process started 06:26 UTC with graphs on, the same weights, and the same momentum line `loaded optimizer momentum from .../0000000150405120.bin.mb`. From epoch 4 to epoch 8 the clock is 1,105,920 steps in 90 s, **12.3K steps/s**, VRAM 21.1/24 GB. The October 16 figure of 23.4K is the fp32 30-scenario trainer at horizon 256, before this shared actor and before zone memory. One seed. Not a leaderboard result. This segment is separate from the earlier ones.

#### Night value on the four held months

Measured 2026-10-24 on GPU 1. Frozen eval, not a learner. The four months are the entropy scratch at 70.2M (0.134) and the separate 80.2M resume (0.188), the sibling resume at 80.2M (0.174), and the r51 zone-mix resume at 80.2M (0.201). Each replay reproduced its stored fp64 cost. Records: `results/study_r52/night_value/`.

The critic is one scalar. It falls through every weekday night, a median range of 17–27, and inside a night it tracks the remaining discounted return at 0.97–1.00. Across the 21 weekday nights the value at 05:55 ranks that remaining return at 0.54 on the 0.134 month and at 0.80–0.91 on the other three. The same value ranks the next morning's productivity at 0.37, 0.69, 0.81, and 0.82. It does not rank rooms: the median correlation with a single zone's morning shortfall is +0.12 on the best month and −0.24 to −0.46 on the others, while the rooms still differ (cross-room shortfall std 0.31–0.49). r51's zone rows were not in the dumped scalar. The "8–12 times" level in the first pass compared the trained value with `rew.bin`, which stores one tenth of the official reward. After that factor the slopes are 0.64, 0.73, 0.44, and about 0.5–0.6.

#### r53 — one value on the zone's own features

The 90 zone-critic rows were dumped on the same step alignment as the value record, exact fp64. They reproduce r51 at 80.2M (0.2012762604) and the two r50b months (15.04M at 0.439, 50.05M at 7.18). On 39 present Fairbanks zones, the correlation of a row at the probe hour with that zone's Monday–Friday 06:00–12:00 deficit is +0.08, +0.05, +0.08, +0.07, +0.11 on r51 at 16:00, 22:00, 23:55, 02:00, and 05:55. Both r50b months stay between −0.17 and +0.09. A head of 90 rows on the pooled trunk has no room to read.

r53 replaces that head. From scratch, Fairbanks only, 50,135,040 steps, ten checkpoints on the 5,013,504 grid, seed 73, GPU 1. One shared head, twelve parameters, zero at the start: bias plus a weight on asinh of each of that zone's eleven features. An absent slot is zero and stays out of the loss. Damper and reheat train on that zone's whitened advantage. Plant and AHU keep the scalar. The room gradient is the outer product into the twelve-vector. `env.zone_critic=1` writes the per-zone return. `train.zone_critic=0` and `train.zone_adv=0`. Everything else is the entropy recipe, on the current shared actor, so the sibling block is in this binary. The month to beat is 0.134. The sibling-only month on this weather was 0.174. Gate `results/verify_20261024_171509.log` ended **ALL CHECKS PASSED**. Driver `results/study_r53/run_r53.py`.

Started 2026-10-24 12:14 UTC. The banner is the room-value line, the shared scalar, the per-zone return, the sibling block, and Muon, with no loaded weights. The first file `0000000005013504.bin` and its sidecar are both 1,667,504 bytes, 416,876 floats. Epoch 68 on the 3060 reads 2.1K steps/s and 7.8 GB. One seed. Not a leaderboard result.

Stopped 2026-10-24 15:05 UTC at the user's request, during epoch 298, dashboard about 22.0M steps. The driver records exit 137 from that stop. Four checkpoints remain, 5,013,504 through 20,054,016, each 1,667,504 bytes with a sidecar. No fp64 month was scored.

The room credit was on and the return had already left its only better region. The desync dashboard peaked at 2.233 at epoch 159 (11.7M steps) and from epoch 220 on stayed between 5.93 and 8.42; the last box is 7.59, entropy −232, value loss 3060, KL 0.71. At the matched 20.9M step the sibling scratch, same building and actor family, was at 1.22 and still falling, and the entropy scratch that later reached 0.134 was at 5.46. Zone KL at epoch 60 was 1.55 here, against 0.013 and 0.066 on those two, so the valves were moving and the score went the other way. The remaining ~30M would have been more of that update. r54 on GPU 0 was left running.

Decision after the stop (2026-10-24): the room return is the right target for the valves, and this wiring is not the instrument. `valve_zone` drops the building advantage, and `r_z` is 0 all night, so the valves lose the only signal that reaches Monday. The 12-weight baseline does not enter the trunk. Damper and reheat stay two independent heads (`SBS_OUTPUTS` zone role is 2), so a cold room cannot name one heat plan. The open starter-kit PR 2 is the current tank bill (`gas = max(0, flow + dissipation + tank ramp)`); the port matches that oracle and does not adopt the PR ahead of the gateway.

#### r55 — one command per zone

Started 2026-10-24 16:07 UTC on GPU 0, from scratch, name `r55_zone_command`. It keeps `train.freeze_return=1` from r54: one GAE from the stored rollout values, one rollout-wide advantage normalization, every visit on that pair. The new factor is `env.zone_command=1`. The squashed damper-slot sample is the only zone action. Heat (`u > 0`) opens reheat at the 0.1 damper floor, then opens the damper. Cool opens the damper with reheat shut. The middle is that floor and a shut coil. The pump runs when any present zone command is heating. The hot-water temperature stays the one learned plant sample. Reheat, the pump head, and the pump run are masked. The gateway still receives damper, reheat, and the pump fields. Observation width and \(R\) stay as they are. Zone critic, room value, and zone advantage stay off, which is what `freeze_return` requires. Fairbanks, seed 73, 80,216,064 steps, sixteen saves on the 5,013,504 grid, so r54's ten months sit on this curve and it continues through 80.2M. Gate `complete_port/results/verify_20261024_211242.log` ended ALL CHECKS PASSED. The curve header is included from both the env and the policy check, so the guard is the macro `SB_ZONE_COMMAND_CUH`. The banner is `zone command on (one heat curve per zone, pump follows heat)`, with the frozen-return banner, Muon, squash floor 0.250, and 56 of 288 heads active. No loaded weights. The first checkpoint and its Muon sidecar are both 1,667,456 bytes. Epoch 3 is 13.6K steps/s. The sixteen fp64 months run after training, with the same decode. The comparison is r54. Its best is 0.227 at 50.1M, productivity −0.168, energy 0.240, carbon 0.391, gas 260 MWh. A win is a month under that which is still heated at the next two saves, gas above 100 MWh and productivity above −1, with the damper off the floor while the coil is open. This actor also carries the sibling block. One seed.

Finished 2026-10-24 18:12 UTC, exit 0, epoch 1088, sixteen fp64 months. Costs / productivity / energy / carbon / electricity MWh / gas MWh: 5.0M 9.551 / −15.830 / 0.150 / 0.113 / 40.0 / 59.6; 10.0M 5.829 / −9.711 / 0.007 / 0.004 / 2.2 / 1.7; 15.0M 5.389 / −8.978 / 0.006 / 0.003 / 2.0 / 0.8; 20.1M 10.900 / −18.019 / 0.303 / 0.138 / 92.6 / 49.9; 25.1M 9.326 / −15.292 / 0.439 / 0.314 / 118.0 / 161.2; 30.1M 10.630 / −17.478 / 0.482 / 0.232 / 146.1 / 88.9; 35.1M 6.321 / −10.331 / 0.342 / 0.272 / 88.9 / 147.4; 40.1M 2.112 / −2.699 / 1.018 / 1.446 / 172.4 / 938.0; 45.1M 0.685 / −0.329 / 0.940 / 1.496 / 138.4 / 991.9; 50.1M 0.301 / −0.324 / 0.178 / 0.354 / 18.9 / 244.4; 55.1M **0.217** / −0.168 / 0.227 / 0.354 / 36.1 / 234.1; 60.2M 0.239 / −0.261 / 0.165 / 0.248 / 27.6 / 162.8; 65.2M 0.645 / −0.195 / 1.079 / 1.562 / 179.4 / 1016.3; 70.2M 0.600 / −0.253 / 0.911 / 1.333 / 148.5 / 869.3; 75.2M 0.391 / −0.162 / 0.559 / 0.913 / 77.9 / 607.4; 80.2M 0.348 / −0.263 / 0.347 / 0.604 / 41.2 / 406.5. The first heated month with productivity above −1 is 45.1M. The best, 0.217 at 55.1M, is under r54's 0.227, and the next two saves stay heated (gas 163 and 1016 MWh, productivity −0.261 and −0.195). The month log does not record damper against reheat, so that part of the win test is not in these scores. It is not under 0.134. r54 was already under 0.4 and heated at 10.0M; this curve is plant-off through 35.1M. The 65.2M month is the same fuel spike r54 had at 20.1M, and 80.2M is still recovering at 0.348. The last logged dashboard, epoch 1072, is score 0.393, value loss 1.45, KL 0.029, entropy −171, worst zone 23.9 K. One seed. Not a leaderboard result.

The curve's zero is the off state. `u = 0` is the damper floor and a shut coil, `u < 0` opens the damper with the coil shut, and any one zone with `u > 0` slams the pump to full. A zero-mean Gaussian therefore starts on the plant-off side of a step. The canonical layout already forbids that step for the run bit: a threshold at 0 chatters, and the neutral sample has to be a running device (the r8 failure). r55 put the step back on the zone command. That is why 10.0M and 15.0M are 1.7 and 0.8 MWh of gas. The same comfort at 55.1M and 65.2M with gas 234 and 1016 MWh is the hot-water head, still a free sample, charging the tank after the zones are already warm. Squash floor 0.25 is a floor on the tanh derivative so a saturated mean keeps a gradient. It does not floor the standard deviation. `init_logstd = -1` is a standard deviation of about 0.37, so the actions are explored. The actor reads each zone as its own features plus the delay summary and the sibling block, not as a flat 983-vector. r54 used this PPO, this reward, and this floor, and it was heated by 10.0M. The next decode keeps one command and moves the zero: `u = 0` is half reheat at the damper floor with the pump running, `u < 0` closes the coil without opening the damper, and the damper opens for cooling only when that zone's supply is colder than the zone and the zone is above its cooling setpoint. The air handler does not heat, so January supply is often colder than the room; the cooling setpoint is what keeps that from opening the damper. Pump head and the hot-water cap follow the average coil request, so one noisy zone cannot start the plant and a warm building cannot sit at full boiler.

#### r56 — neutral half heat

The first container stopped at nine checkpoints (45.1M). Those files are `results/runs/r56_zone_command_decode_partial`. The 0.70 banner rejection is removed. Restarted from scratch 2026-10-25 00:58 UTC on GPU 0, name `r56_zone_command_decode`. Same recipe as r55, including `train.freeze_return=1` and `env.zone_command=1`. The decode is the one above. Fairbanks, seed 73, 80,216,064 steps, sixteen saves on the 5,013,504 grid. Gate `complete_port/results/verify_20261025_003118.log` ended ALL CHECKS PASSED. Banner `zone command on (neutral half heat, pump follows the average, water capped)`, Muon, squash floor 0.250, 56 of 288 heads active, no loaded weights. The first checkpoint and its Muon sidecar are both 1,667,456 bytes. The comparison is r54's 0.227 at 50.1M and r55's 0.217 at 55.1M. A win is heated by 10M, and no later month with the same comfort at several times the gas. The sixteen fp64 months run after training. One seed.

The 00:58 UTC process died during epoch 580. No exit status was written. The last complete file is 40,108,032, with a Muon sidecar of the same size. The steps after that file were not saved. Continuation `r56_zone_command_decode_cont` loaded that pair at 2026-10-25 02:12 UTC on the same binary and trains the remaining 40,108,032 steps. Its step counter starts at 0. A saved step S is original step 40,108,032+S. The stdout line is `loaded optimizer momentum from /res/runs/r56_zone_command_decode/ckpt/0000000040108032.bin.mb`. Philox, the desync phase, the recurrent state, and the zone-temperature memory start empty. Learning rate and entropy stay at the constants. The two segments are separate. The continuation exited 0 at 2026-10-25 03:04 UTC. fp64 months, desync off, `zone_command=1`. Costs are productivity, energy, carbon, electricity MWh, gas MWh.

Segment 1, `r56_zone_command_decode`: 5.0M 1.581/−2.585/0.050/0.100/4.1/68.8; 10.0M 0.542/−0.309/0.523/1.260/11.7/886; 15.0M 0.537/−0.257/0.559/1.354/12.7/952; 20.1M 0.622/−0.330/0.616/1.502/13.9/1058; 25.1M 0.742/−0.487/0.650/1.599/15.1/1128; 30.1M 0.554/−0.480/0.387/0.942/13.6/665; 35.1M **0.284**/−0.202/0.248/0.567/11.0/397; 40.1M 0.377/−0.380/0.232/0.513/12.6/357.

Segment 2, original steps: 45.1M 0.473/−0.413/0.387/0.739/37.6/505; 50.1M **0.219**/−0.269/0.105/0.185/13.0/125; 55.1M 0.582/−0.649/0.384/0.579/59.2/381; 60.2M 0.293/−0.310/0.170/0.364/12.2/253; 65.2M 0.347/−0.231/0.405/0.636/59.4/421; 70.2M 0.337/−0.218/0.375/0.653/46.6/441; 75.2M 0.288/−0.357/0.113/0.255/7.2/178; 80.2M 0.269/−0.286/0.153/0.336/11.4/234.

Heated by 10.0M. The low point is 0.219 at original 50.1M. That is under r54's 0.227 and above r55's 0.217, and it is not under 0.134. The next month is 0.582 with worse comfort, so the 0.219 does not hold. Curve: `complete_port/results/study_r56/curve.json`. One seed. Not a leaderboard result.

#### r57 — value trunk cap (stopped 2026-10-25, failed)

r56's config with one change, from scratch, Fairbanks, seed 73, 80,216,064 steps, sixteen saves on the 5,013,504 grid, GPU 0: `train.value_trunk_cap=1`. Once per minibatch visit the value path's contribution to the shared trunk gradient is clamped at the policy path's magnitude, `scale = min(1, ||g_pi||/||g_V||)`. The value head's own gradient, the actor heads and the policy gradient are untouched and `max_grad_norm` stays 0.5. Off is the untouched trunk bit for bit, and the cap cannot bite on the separate-critic path, where the value gradient into the trunk is zero by construction. Gate `complete_port/results/verify_20261025_091402.log` ended ALL CHECKS PASSED (57 checks); `policycheck` reports 108+108+60+60 finite differences, 0 failures, so the off path is unchanged. Driver `results/study_r57/run_r57.py`. Telemetry: `SB_VTC` per epoch, `norm/vtc_capped_frac` and `norm/vtc_min_scale` in TensorBoard, `results/study_r57/vtc.jsonl`.

Stopped at the user's request 2026-10-25 05:25 UTC after 860 epochs, 63,406,080 steps. Twelve of sixteen checkpoints remain through 60,162,048, each with its Muon sidecar. No fp64 month was scored; the arm is marked failed and is not extended.

Reason: the constant the arm is built on is refuted by its own telemetry. `||g_V||/||g_pi||` at the trunk is not a spike ratio but a standing one. The cap bit on 849 of 860 epochs, and the tightest scale per epoch was 1.000 once (epoch 1) and then 0.034, 0.019, 0.043, 0.040, 0.028, 0.105, 0.067, 0.072, 0.046 at epochs 101 to 860 — median 0.044, minimum 0.0144. So instead of shaving the trunk during spikes the cap permanently removes the critic from the trunk, keeping 2–10% of its share, while the value head's own gradient is deliberately left uncapped, so the target the clamp was meant to protect is not protected at all. Its throughput cost is not the cause: steady SPS is 13.2K against r55's 13.1K on the same shape.

Scale and curve at 63.4M, r57 against r56's control at the matched 63.1M (desync episode mean / comfort deviation K / productivity): **r57 3.945 / 18.5 / −6.37; r56 0.369 / 6.24 / −0.52**. r57's out-of-band fraction is 0.917. `loss/value` 98.3 and `loss/ev_ret2` 3.36e5, the second moment growing monotonically 1.19 → 1.5e4 → 4.7e4 → 3.4e5, so the pre-registered falsifier ("no ev_ret2 excursion above about 10× its rolling quiet median") fails by orders of magnitude. The run is in the cold basin, about 11× r56's score at the same step.

Verdict: the value path's trunk gradient is a standing multiple of the policy's on this recipe, not a spike, so a per-visit magnitude cap is the wrong instrument — it changes the acting point permanently rather than clamping a spike. Do not extend r57 or score its twelve files as a curve of this recipe, and do not load them. The control stays r56 (best 0.219 at 50.1M). Preserved: `results/runs/r57_decode_value_trunk/` (twelve checkpoints and sidecars, `train.log`, `train.docker.log`, `FAILED.md`) and `results/study_r57/` (`vtc.jsonl`, `sources/`, `run_r57.py`). One seed. Not a leaderboard result.

#### r58 — value spike guard (relative, against its own trailing scale)

r56's config with one change, from scratch, Fairbanks, seed 73, 80,216,064 steps, sixteen saves on the 5,013,504 grid, GPU 0: `train.value_spike_guard=1`. r57's defect was a constant measured against the policy, so this guard is measured against the value tangent's own history. Once per minibatch visit, `s = min(1, q·M/‖g_V‖)` with q = 3, `‖g_V‖ = ‖grad_value_h‖₂`, and M the trailing geometric mean of ‖g_V‖ — an EMA of `ln‖g_V‖` whose halflife is one epoch of visits (128). A short storm barely moves M (a 6-visit storm at 250× drifts `ln M` by about 0.2 nats), so the threshold holds the quiet baseline through the surge. It is applied to `grad_value_h` only: the policy path into the trunk, the critic's own row (`wgrad_value`), the value loss and the actor heads are untouched, and a chronic ratio below q passes at s = 1 by construction — which is exactly what r57 could not do. Off is the untouched trunk, bit for bit, and r57's own flag stays in the binary at its default 0, so this arm's only active factor against r56 is the guard. Gate `complete_port/results/verify_20261025_110318.log` ended ALL CHECKS PASSED; `policycheck` reports 108+108+60+60 finite differences, 0 failures. Driver `results/study_r58/run_r58.py`.

Why q = 3: r56's quiet `ev_ret2` fluctuates by at most 2× epoch to epoch while every measured storm is 14×–250× the quiet level (636 → 9,040; → 178,300; r40 3,875 → 55,256). A 3× ceiling sits above the noise and far below every recorded surge.

Telemetry, fixed for r57's gap: per epoch the bite fraction, the applied scale (mean and minimum), the `‖g_V‖` geometric level, and the histogram of `log2(‖g_V‖/M/q)` — so the typical ratio is visible and not only the worst one. `SB_VSG` per epoch, `norm/vsg_*` in TensorBoard, `results/study_r58/vsg.jsonl`. The first logged epochs read bite 0.000, s_mean 1.000, r_mean about 1.0–1.4, r_max at most 2.6: inert below the threshold, as constructed. Sixteen fp64 months run after training. One seed.

#### r52 stopped — the full months are plant-off

Stopped 2026-10-24 13:16 UTC at the user's request, during the segment past file step 300M. Six checkpoints, 50.0M through 300.3M, each 1,667,456 bytes with a sidecar. The training mean of the thirty desync episodes was 0.566, 0.146, 0.079, 0.270, 0.203, 0.346. That 0.079 at 150.1M is the low point of the dashboard. It is not a month.

fp64, desync off, all 30 full episodes, on GPU 0. The 150.1M file and the 300.3M file score **1.106** and **1.106**. Energy stays under 0.0003 and carbon under 0.0007 on every scenario, so the score is the comfort term and the plant is off. The two files agree to three decimals on every scenario. `building_train_1__fairbanks_1` is 5.072, productivity −8.45. The learned month on that weather remains 0.134. pid_water on these 30 averages 0.054, and this policy is above pid_water on all 30. Scores: `results/study_r52/eval/`. One seed. Not a leaderboard result.

Verdict (2026-10-24, analysis on those months; the user's scored table is the source):

- **Same known failure, saturated.** Plant-off on all 30 — the same basin r35 ended in (its 5.07 on Fairbanks matches this run's 5.072). Identical to three decimals on every scenario across checkpoints 150M apart means the policy's fp64 behavior did not move in 150M steps: the recipe has no descent surface left here, and "more checkpoints" is measured out, not conjecture.
- **pid_water dominates everywhere, 1.5×–24× per scenario.** Per-weather scenario means from the 30 rows: Phoenix 1.810, Fairbanks 1.399, Singapore 1.370, Atlanta 0.732, Denver 0.220; pid_water on the same splits 0.078, 0.079, 0.047, 0.030, 0.038. The top six rows (61.6% of the sum: fairbanks_1 5.072, phoenix_6 5.046, singapore_1 3.229, phoenix_5 2.726, fairbanks_8 2.215, phoenix_4 2.148) are the extreme climates — under plant-off each row degenerates to its building's conditioning load, so the weather ordering is HVAC difficulty, not per-building behavior. No scenario exercises the policy into a learned mode.
- **The 0.079 dashboard low is the desync trap on record.** Months 1.106 while the mixed-phase dashboard read 0.079–0.346. Short/night-heavy episodes dominate the dashboard and cannot move the month; keep this instance against any future "the dashboard improved" claim.
- **Why the batch goes off when Fairbanks-only runs partially heat** (consistent with the night-value and zone-row findings): fuel is billed on the same step as the setpoint, comfort after thermal-mass delay; γλ puts ~0.05 on a one-day-ahead comfort signal and ~3e−4 on a Sunday→Monday one, and the scalar critic ranks time but not rooms, so batch-averaged across 30 buildings the reliable direction is spend-nothing. Fairbanks-only arms escaped partially because one climate keeps heating a recurring gradient; the 30-batch whitens it away.
- **Measurement byproducts.** (a) The run is the all-30 no-control bound: fuel ≈ 0 on every scenario, so any later batch arm has a concrete re-ignition gate — gas > 0 with monthly score below this table's row. (b) The collapse is dated only to "within the first 150M"; scoring the 50.0M and 100.1M files (~2.5 h each) would close the window and set when the schedule and trust-region rungs must engage. (c) The scoring chain reproduced every scenario to 3 decimals across two checkpoints — fp64 eval determinism confirmed.
- Nothing here is a submission candidate; the batch arm must beat pid_water mean 0.054 before it is even west of the shipped baseline, and the Fairbanks target remains 0.134.

#### r54 — return frozen at the start of the epoch

Approved 2026-10-24. The entropy recipe on `building_train_1` / `fairbanks_1`, from scratch, seed 73, 50,135,040 steps, ten checkpoints on the 5,013,504 grid. GPU 0, fp32 solve, CUDA graphs on. The one change is `train.freeze_return=1`: GAE and the rollout-wide advantage normalization run once from the stored rollout values, and all 128 visits train on that pair. Zone critic, room value, and zone advantage stay off. The current shared actor includes the sibling block, which the 0.134 network did not. r53 on GPU 1 is a different run and is left alone.

The comparison curve is the entropy scratch's first ten months (`results/study_r48/curve_a_scratch80.json`). That 50.1M month is 1.558, productivity −2.553, energy 0.046, carbon 0.087, electricity 5.5 MWh, gas 59 MWh. The 0.134 month is at 70.2M, outside this budget. A win inside 50M is a month under 0.134 that is still heated at the next two saves: gas above 100 MWh and productivity above −1. After training, each checkpoint is one deterministic fp64 Fairbanks month. Gate `results/verify_20261024_195122.log` ended **ALL CHECKS PASSED**. Driver `results/study_r54/run_r54.py`.

Started 2026-10-24 14:40 UTC on GPU 0. Banner confirmed: return frozen at epoch start, Muon, Fairbanks, squash floor 0.250, no loaded weights. Epoch 3 is 13.4K steps/s, evaluate 5.5 s, train 2.9 s, copy 0 ms. The first file `0000000005013504.bin` and its sidecar are both 1,667,456 bytes. The same driver scores all ten checkpoints in fp64 when the 50,135,040 steps finish. One seed. Not a leaderboard result.

Finished 2026-10-24 15:59 UTC. Training exit 0 at epoch 680. Ten fp64 months, `results/study_r54/`. Costs: 5.0M 4.068, 10.0M 0.282, 15.0M 0.329, 20.1M 0.927, 25.1M 0.377, 30.1M 0.521, 35.1M 0.406, 40.1M 0.388, 45.1M 0.358, **50.1M 0.227**. The last month is productivity −0.168, energy 0.240, carbon 0.391, electricity 33.8 MWh, gas 260 MWh. Every month after 5.0M has gas above 100 MWh and productivity above −1. The 20.1M month is the fuel spike (gas 1,063 MWh) and the next save is back at 0.377. Value loss is about 4–12 from 10M on, and KL stays about 0.07–0.21. The scratch’s best inside 50M is 0.347 at 40.1M, and its 50.1M month is 1.558 with gas 59 MWh. 0.227 is not under 0.134. The curve ends at its best point, so the “next two saves” check does not exist for it.

#### r58 live read (2026-10-25 11:47 local, epoch 351 of 1088, 25.9M of 80.2M steps)

The guard is inert over the whole run, not only its first epochs. `SB_VSG` logs 325 epochs: bite 0 in 314, and the 11 that bit are the EMA warm-up (epoch 5, bite 0.98). After epoch 10, `r_mean` 1.05, `r_max` at most 2.5, `s_min` 1.0. Over the same epochs the trunk/head gradient ratio (`grad_shared`/`grad_actor_heads` in `SB_UPDATE`) goes 32× at epoch 25 → 52× at 100 → 350× at 315, and the trunk's Muon step stays 5.0× the heads'. So the value's hold on the shared trunk is chronic and grows 7×, and a guard that normalizes `‖g_V‖` by its own trailing mean cannot see it. r57 measured the ratio against the policy and bit every epoch; r58 measures it against itself and never bites. **The arm cannot answer its own question**, and its only difference from r56 is binary noise.

State at 25.9M: cost 2.58, productivity −4.47 (0.6 × 4.47 = 2.68 of 2.76, so 97 % of the cost is comfort), comfort deviation 12.4 K, worst zone 41.7 K, out-of-band 0.81, gas 177 MWh, electricity 17.0 MWh, explained variance 0.991, value loss 312, joint sampled KL 8.0, entropy −92 (min −190 at 21M). That is the under-heated basin, not plant-off (plant-off is gas < 100 MWh with productivity < −2). No month is scored yet; the driver runs the sixteen fp64 months after training.

The same seed, same flags, inert flag: r56 and r58 agree to about 5 % through epoch 110, then r56 falls to comfort deviation 4.2 K and stays at 4–6 K while r58 stays at 14–16 K. Calibrating the dashboard against the 33 measured (dashboard, fp64 month) pairs of r54/r55/r56: comfort deviation ≤ 6 K → month 0.22–0.74; 11 K → 2.1; 16–20 K → 1.6–6.3; 24–33 K → 5.4–10.9. r58 has sat at 8.6–15 K since 21M. **Two runs of one recipe differ 4–5× in the month by 20M steps**, which is larger than every effect the five arms since r50 were chasing, and every one of those arms is a single run.

#### r59 — the credit window, λ = 1 (proposed 2026-10-25, not approved)

Protocol and driver: `complete_port/results/study_r59/` (`PROTOCOL.md`, `run_r59.py`, which refuses to start without an `APPROVED` file). One change from r58: `train.gae_lambda` 0.95 → 1.0, with the value guard off (the ini default). γ stays 0.99, so the effective credit window goes from 16.8 steps (84 min) to 100 steps (8.3 h), and a comfort bill six hours after the heat enters the advantage at 0.485 instead of 0.012 — 40×. That is the only axis never given a fair run: r36 (λ = 1) was stopped at 19M on 2026-10-20 to free the 3090 for r37, with the occupied-day value still falling.

Why this and not another trunk guard: the cost is the morning ramp (45 % of the 0.134 month), fuel is billed on the step the setpoint moves while comfort is billed 2–6 h later, and the critic already fits (r58 explained variance 0.991). Pre-registered: at the 40.1M and 45.1M saves gas > 200 MWh **and** productivity > −0.5; best month < 0.174, the best month ever measured on the current (sibling-block) actor. Plant-off past 40M falsifies the window as the constraint. Explained variance < 0.5 at 5.0M invalidates the run, because λ = 1 makes the value target about 5× larger. 80,216,064 steps, sixteen saves, seed 73, from scratch, GPU 0, about 2 h 5 min including scoring. One seed. Not launched.

Queued behind it, each needing its own approval: horizon 576 with λ = 1 so one rollout holds a night and the following morning; a per-zone value whose night target is that zone's future comfort (r53's `r_z` was 0 all night, which is why the valves had no signal); and per-epoch logging of `‖g_V‖` and `‖g_pi‖` at the trunk separately, since two arms have now been built on a statistic nobody logged. Before another 80M arm, a two-seed 30M control pair should pin the spread that r56 against r58 already shows to be at least 4×.

#### r58 stopped at 42.2M; r59 started (2026-10-25 12:20 IST)

Read at 41.4M (epoch 558) the arm looked finished: productivity −6.42, gas 96 MWh, out-of-band 0.92, comfort deviation 19.8–23.5 K over the 36–41M window, cost −3.9 to −4.9, i.e. drifting into the plant-off basin with the fuel falling. On that read the run was stopped at 12:15 IST (epoch 573, 42.25M steps, exit 137). Eight checkpoints remain, 5,013,504 through 40,108,032, each with its Muon sidecar; no fp64 month was scored and none will be. That was one read too early: the last four epochs before the stop were the heated mode, not a blip — comfort deviation 10.1, 6.16, 9.54, 8.39, 8.06 K at epochs 569–573, cost −1.80, −0.99, −1.73, −1.43, −1.39, gas rising 96 → 195 MWh. Epoch 570 sits in the ≤ 6 K band that maps to months of 0.22–0.74. So this is the third observation of the same bistability (r56 at epoch 120, r55 at 45M, r58 at 41.7M), and the transition landed inside the window between the 40.1M save and the stop, so those weights are not on disk.

Recorded as stopped, not as a curve: the guard was inert for the whole run (bite 0 in 314 of 325 epochs), so no reading of it tests the arm's hypothesis, and its only difference from r56 was binary noise. Preserved: `results/runs/r58_value_spike_guard/` (eight checkpoints and sidecars, `train.log`, `train.docker.log`, `status.json`), `results/study_r58/` (`vsg.jsonl`, `run_r58.py`, the two runtimes).

`r59_credit_window_lambda1` started 06:50 UTC on GPU 0 from the protocol in `results/study_r59/PROTOCOL.md`, approved by the user's instruction of 12:14 IST. Banner confirmed: no value guard, Muon, frozen return, advantage normalization, zone command (neutral half heat), sibling block, 56 of 288 heads, no loaded weights, `--train.gae_lambda=1`, `--train.value_spike_guard=0`, 8.35 GB, 13K SPS. The value row's gradient at visit 0 is 11.5 against the heads' 4.7, where r58 started at 2.2 against 8.0 — the λ = 1 target is the larger one, as expected. The first start failed on a missing `config/default.ini` in the copied runtime (the runner had copied only the binary); the tree is copied now and that empty directory is kept as `results/runs/r59_credit_window_lambda1_badconfig`. Sixteen fp64 months follow training.

The first two starts of r59 were void and are kept, not deleted. The 06:47 one died on a missing `config/default.ini` (the runner had copied only the binary; the runtime tree is copied now) and is `results/runs/r59_credit_window_lambda1_badconfig`. The 06:50 one trained, but its save grid was 20 epochs: `--base.checkpoint_interval` was written without its f-string and arrived as the literal `{INTERVAL}`, so the trainer used its own default and saved at 1,474,560 instead of 5,013,504, which cannot reach the pre-registered grid. It was stopped at 3.0M steps and is `results/runs/r59_credit_window_lambda1_badinterval`; the flag and an arg check are fixed. The live run started 06:55 UTC, and its args read `--base.checkpoint_interval=68` and `--train.gae_lambda=1`.

The aborted 06:50 attempt still carries the λ = 1 training trajectory, and its dashboard is the first signal on the hypothesis: at epoch 36 (2.7M steps) score −0.380 and comfort deviation 2.995 K, where r56 and r58 were at 14.8 K and −2.29 at the same step. It is a desync dashboard, not a month, and the value loss was 23,091 (the λ = 1 target is the large one), so the 5.0M explained-variance check decides whether the run stands.

r59 at 5.9M steps (epoch 80): the finished-episode score fell 2.87 → 2.23 → 0.82 from 4.7M to 5.9M, gas rose 55 → 138 MWh, comfort deviation 7.1 K, worst zone 31.9 K, value loss 52,349 (3.7M) → 3,081 (5.2M) → 228 (5.9M), trunk/head gradient ratio 10.9× at epoch 71 against r58's 32× at epoch 25. The 5.0M explained-variance gate fired (0.006) and is recorded as a waiver in the protocol's addendum with four new binding stop conditions; the value loss, not that diagnostic, is what shows the critic converging, and under `freeze_return=1` the diagnostic goes negative while the loss falls. The policy is heating from the first millions, which is the behaviour the arm exists to test.

#### r59 cut by a host reboot at 14.6M; resumed to a 50.1M total (2026-10-25 13:00 IST)

The first process did not fail in training. `last -x reboot` shows the host rebooting at 12:53 IST (and at 06:25 and 07:25 the same morning), and the docker daemon came back at 12:53:13. The container went with the boot: no `status.json`, `train.log` ends mid-dashboard at 14.6M steps, and the last complete save is 10,027,008 with its sidecar. Nothing in the log indicates a numerical failure — the loss, the KL and the policy were healthy at the cut (score 0.49, comfort deviation 5.1–6.1 K, gas 203 MWh).

#### r59 result — the credit window (λ = 1), 50,135,040 steps

Ten fp64 Fairbanks months on the 5,013,504 grid. Score = 0.6 × comfort + 0.2 × energy + 0.2 × carbon, lower better; comfort is 0.6 × (−productivity).

|  step |     score | productivity | comfort |  fuel | elec MWh | gas MWh |
| ----: | --------: | -----------: | ------: | ----: | -------: | ------: |
|  5.0M |     0.620 |       −0.958 |   0.575 | 0.046 |      8.9 |      99 |
| 10.0M |     0.492 |       −0.505 |   0.303 | 0.189 |     18.7 |     450 |
| 15.0M |     0.446 |       −0.401 |   0.240 | 0.205 |     32.6 |     465 |
| 20.1M |     0.314 |       −0.225 |   0.135 | 0.179 |     25.3 |     410 |
| 25.1M |     0.517 |       −0.504 |   0.302 | 0.214 |     34.2 |     486 |
| 30.1M |     0.522 |       −0.505 |   0.303 | 0.220 |     35.0 |     498 |
| 35.1M |     0.515 |       −0.495 |   0.297 | 0.218 |     34.8 |     494 |
| 40.1M | **0.237** |       −0.202 |   0.121 | 0.116 |      6.7 |     290 |
| 45.1M |     7.618 |      −12.666 |   7.600 | 0.018 |      3.5 |      40 |
| 50.1M |     5.421 |       −8.875 |   5.325 | 0.096 |     21.9 |     204 |

Best 0.237 at 40.1M, a single checkpoint bracketed by 0.515 and 7.618, so it is not held. Median 0.516, mean 1.670.

The hypothesis was right about the mechanism and wrong about the month. λ = 1 does change _when_ the policy heats: at 5.0M it already holds a 0.620 month on 99 MWh, where r56 was 1.581, r54 4.068 and r55 9.551 at the same step, and its 20.1M comfort term (0.135) is the best any arm has measured at that step. The 84-minute window was blocking the timing of heat, and widening it unblocked it.

What it then bought was fuel, not comfort. From 10M to 35M the policy burns 410–498 MWh of gas and still carries a 0.30 comfort term — four times r56's best-month fuel for twice its comfort loss. That is the r55 65.2M signature again: gas burned without being delivered to the zones, the hot-water head and reheat running while the dampers stay shut. At 45.1M it falls into the plant-off attractor (gas 40 MWh, comfort 7.600) and does not climb out by 50.1M.

On the (comfort, fuel) plane no arm since r48 has entered the box comfort < 0.10 and fuel < 0.10. r48's 0.134 sits at (0.069, 0.065) and dominates everything after it; r56's 0.219 is (0.161, 0.058), cheap and uncomfortable; r59's 0.237 is (0.121, 0.116), comfortable and expensive. The arms have been trading along that frontier, not crossing it.

Accounting: this curve is two segments. The first process was cut by the host reboot at 12:53 IST, 14.6M steps in; the continuation ran 10,027,008 → 50,135,040. The launcher was then killed during the scoring pass (a harness restart took its process group), which left the last two months unscored; they were scored afterwards by `score_all.py`, and the continuation names its files by local step, so the first pass had mislabelled six months before that was caught and rebuilt. `curve.json` is the corrected grid. One seed. Not a leaderboard result.

The user set the budget at 50,135,040 total and asked for the run to be continued. `results/study_r59_resume/run_r59_resume.py` continues from that save as `r59_credit_window_lambda1_cont` for 40,108,032 local steps, same binary and flags, `--base.load_model_path=/res/runs/r59_credit_window_lambda1/ckpt/0000000010027008.bin`. The gate line is present and clean: `loaded optimizer momentum from /res/runs/r59_credit_window_lambda1/ckpt/0000000010027008.bin.mb`, no sidecar warning, and `loaded initial policy from …/0000000010027008.bin`. Eight saves on the same 68-epoch grid, so original steps 15,040,512 through 50,135,040.

The two segments are separate and are reported separately. The process counter starts at 0, so a local save S is original 10,027,008 + S. This process re-seeds Philox and the desync phase, and the recurrent state and zone memory start empty; the learning rate and the entropy coefficient are constants, so no anneal is lost. The splice point is original 10,027,008, and the months are scored on the 5,013,504 grid: 5.0M and 10.0M from the first segment, 15.0M through 50.1M from this one — ten months, the same grid as r54, r55 and r56.

The launcher now runs detached (`setsid`), because a reboot or a killed shell takes the docker client and the container with it; the earlier launch did not survive. The 5M save interval bounds a cut to about six minutes of work.

### r59_lambda1_logstd_const — the credit window with a frozen standard deviation (approved and started 2026-10-25)

Protocol and driver: `complete_port/results/study_r59_logstd_const/` (`PROTOCOL.md`, `run_r59_logstd_const.py`, which refuses to start without an `APPROVED` file). Approved by the user's instruction "okay run it for 65m name it as r59_lambda1_logstd_const". One change on the r59 base (`--train.gae_lambda=1`, guard off, zone command, freeze_return, adv_norm 2, seed 73, Fairbanks, from scratch, GPU 0): `--policy.freeze_logstd=1` with the binary's arming requirement `--base.checkpoint_optim=0`. The logstd gradient is zeroed at the source (`algo.cu`: policy-std term and entropy term both), so sigma stays at init −1 (0.37) for the whole run; means keep full learning; deterministic fp64 evaluation is untouched; saves are weights-only, no `.mb` sidecars. No code change — the r59 binaries and gate `results/verify_20261025_110318.log` (ALL CHECKS PASSED) carry it.

Budget: 65,175,552 steps = 13 × 5,013,504, so the first ten checkpoints align with r59's curvename grid plus three extension saves (55.1M, 60.2M, 65.2M), then thirteen deterministic fp64 Fairbanks months.

#### r61_mod_band — the proportional band does not bound the fuel (2026-10-25)

One change from the r59 family: the zone command decodes to a proportional band (2 K, setpoint-relative) with the water reset at 4 K of mean shortfall; amplitude 3, λ = 1, from scratch, seed 73, 65,175,552 steps, 13 saves, GPU 0, exit 0.

**Pre-registered gate 1 failed.** Two sustained saturation windows — 6.8–10.9M (4.2M steps, gas 337 mean / 399 max) and 12.4–15.5M (3.2M steps, 411 / 417) — both above the 250 MWh bound. A monotone decode cannot bound a saturated policy: the rail is the policy, not the decode.

Months: 1.522, 0.296, **0.287**, 0.560, 8.385, 6.422, 5.313, 6.211, 7.505, 6.700, 8.261, 9.666, 7.667. Best 0.2870 at 15.0M — comfort 0.112 (the best any arm has measured) and fuel 0.175 on 412 MWh of gas, three times r48's fuel. Plant-off from 25.1M and still 7.67 at 65.2M. final − best +7.380; max 1M-window pinned share 1.00; entropy range 249 nats.

Measured bars in our own format (preflight `baseline_row.json`, same scenario, same metric): `fixed_setpoint` **0.1491** (comfort 0.033, fuel 0.116, gas 253 MWh) and `pid_water` **0.2139** (comfort 0.070, fuel 0.144, gas 295 MWh). r61's best is 1.9× the better baseline; r48's 0.134 still beats both, on fuel rather than comfort.

Verdict: refuted as a fuel bound. The decode is not where the fuel is lost; the learner is. Next in the register: P28–P30 (per-feature step sizes with a bound and decay; plasticity maintenance), then P31.

Mechanism under test: the r59 collapse anatomy measured from its TB — saturation freeze (KL → ~1e-6, `logit_outside` → 1.0 at orig 36.1–36.5M), then entropy-bonus-only churn under Muon's scale-invariance (entropy −218 → −81 nats over ~3M steps; the code's own r18–r22 comments name the family and note the 0.25 floor "does not floor the standard deviation"), then detonation (KL ×60, ev*ret2 ×20, EV 0.87 → 0.45) into plant-off. Freezing logstd removes σ churn the way r20 removed the ratio explosion with a hard clamp, without r20's cost (a hard clamp zeroed the \_mean* gradient; freeze_logstd leaves means untouched).

Pre-registered: the M-gate (no KL-extinction + entropy-climb recurrence while σ is frozen; a recurrence refutes σ-creep as the detonator), parity gates at 10.0M ≤ 0.492 and 20.1M ≤ 0.314 against the matched λ = 1 control, and the standing hold falsifier (after any sub-0.134 month the next two saves ≤ best + 0.05, heated). Every month reported with productivity signed and both fuels. Known risk: exploration locked at σ 0.37 for the entire run.

Started 09:31 UTC detached; banners confirmed from scratch including `pufferl: checkpoint logstd frozen, cold optimizer`; GPU 0 at 8.4 GB, GPU 1 untouched.

#### r59_lambda1_logstd_const stopped at 56.4M steps (epoch 777) — premise refuted, 2026-10-25

Stopped at 10:44 UTC on the pre-registered M-gate substance and the user's judgment
("i don't think this is a right run"). No fp64 months were scored; the 5.0M–55.1M
checkpoints stand for diagnostics only. The live curve showed the training already in the
extinction regime from ~13–15M onward: `loss/kl` at 2.5e-7–1e-5 (`logit_outside` 0.9999,
`loss/entropy` pinned at −227) while the policy held a **full-hot pin**: productivity
−0.21…−0.30 (the best comfort any arm has measured at 10–20M) against gas 1,234–1,428 kW —
about 900–1,030 MWh/month, 2–5× the worst recorded month of any arm, stationary for 5M+ steps.

The divergence from its same-seed twin (r59, identical except the frozen logstd) is the
fingerprint: bit-identical behavior to 0.9M, then plant-off by 5M (prod −5.5…−6.5, gas
30–56 kW), then the full-hot pin by 10M. One arm visited BOTH extreme pins of the campaign
within 10M steps. σ-creep is therefore refuted as the necessary cause of the extinction
regime; the regime is structural.

#### r59_lambda1_logstd_const corrections (2026-10-25, read from the TB series)

Two clauses of the stop paragraph above were wrong and are corrected here against the
TensorBoard series (`loss/entropy`, `env/gas_kwh`, `norm/logit_outside`,
`norm/squash_chain_mean`):

- The M-gate's entropy-climb branch DID fire: after holding −231 from 10M, entropy rose to
  **−67 at 25.1–25.4M**, sank back to −185…−207 through 27–38M, rose again to −83 at
  44.2–45.4M, and ended at −233 by 56.3M. The stop record's "the entropy tag saturates
  downward instead of climbing" describes only the plateau phases, not the curve.
- The run re-entered the hot rail **without a live logstd channel**: after the 30–45M
  plant-off detour (gas 15–57 MWh at the 30.2/35.2/40.3M saves), gas climbed back through
  870–883 MWh at 50.1–50.3M with `logit_outside` 0.845–0.862, and to 984–999 MWh with
  `logit_outside` 0.955–0.99 by 55.4–56.4M. "r48-style re-entry required a live logstd
  channel" is refuted; the frozen-σ policy still moved between attractors (entropy swings
  −233 → −67, gas 15 → 1,025 → 870 MWh), so freezing σ did not deaden the wobble either.
- Even inside the 10–25M "pin" window the series is not stationary: `logit_outside` dips to
  0.293 and `squash_chain_mean` spike-tops 0.679 before falling back. The pinned-window
  figures quoted above (0.97–0.999, 0.023–0.043) describe the deep-pinned plateaus such as
  ~13–15M, not the whole window.

Consequence unchanged and now on firmer ground: the flag is retired, and the wobble
(pin → extinction → storm → re-pin) is structural to the decode's authority shape, not to
how σ moves. This is the grounding for the r61 decode-band proposal.

### r60_alive_mean — squash amplitude 3 → 6 (approved 2026-10-25, "okay run it with name r60_alive_mean")

One factor on the r59 base (frozen σ reverted): the location map `squash_location` widens from
`3·tanh(logit/3)` to `6·tanh(logit/6)` (patches/pufferlib_core.diff), so the loss chain
`1−(loc/6)²` keeps a pinned head's mean gradient honest where r21's map died (at the measured
pin depths loc 2.9–3.0 the chain is 0.75 vs 0.0067), the dead zone moves from logit ≈7 to ≈19,
and the 0.25 surrogate floor goes inactive in the reachable band. Hypothesis: the
pin→extinction→storm cycle (r59 detonated at 40.8M; the frozen-σ arm pinned by 13M) is the dead
chain's ratchet, not a training-hyperparameter accident. Executed actions, reward, observations,
physics, checkpoint layout unchanged; near logit 0 the maps coincide, so the seed-73 curve must
track r59's twin until the first deep pin would have formed, then diverge downward.

- Correction recorded before launch: the initial pitch (rational sigmoid `3x/(3+|x|)`) was
  withdrawn from analysis of its own algebra — at the same pinned loc its slope is
  19–60× DEADER than tanh's ((3−loc)²/9 vs (9−loc²)/9 quad tails; any R→(−3,3) map must die at
  the asymptote). The amplitude widening is the minimal change that reaches the goal state.
- Verification: check_ppo_math.py reference updated to amplitude 6 (reference cases
  squash_saturated logits ±18 keep the chain-floor path covered): 80/80 finite-difference
  checks pass. verify_all.sh --trainer → results/verify_20261025_164809.log, 58/58,
  ALL CHECKS PASSED (trainer phase built from the final patch).
- Binaries: results/study_r60_alive_mean/{runtime,score_runtime}/puffer_shared —
  train sha256 a29aff81… (SB_F32 + SB_SHARED_POLICY, contains "squash amplitude 6",
  freeze banner not triggered), score sha256 bb9f0ebe… (fp64 solve, shared policy).
- Gates (pre-registered in PROTOCOL.md): parity ¶ vs r59 at 5.0M ≤ 0.620, 10.0M ≤ 0.492,
  20.1M ≤ 0.314; no-extinction (logit_outside > 0.9 sustained ≥1M, or KL < 1e-5 sustained ≥1M
  without convergence); hold test (any month ≤ 0.20 must keep the next save ≤ best+0.05 and
  heated). Kill at a fired gate; verdict to Experiments.md/HISTORY.md after scoring.
- Launch: 2026-10-25 11:49 UTC, GPU 0, 8.4 GB, from-scratch banners confirmed, visit-0 grad
  fingerprint matches r59's twin to 6 digits (heads 4.667/4.661, value row 11.5455, shared
  step 0.00091303). 65,175,552 steps = 13 × 5,013,504, first ten saves aligned with r59's ten.

### r61_mod_band — decode authority band (proposal v2, verified grounding, NOT yet approved, 2026-10-25)

Single factor on the r60 base (λ=1, learned σ, zone command, freeze_return, adv_norm 2,
ent_coef 0.001, seed 73, Fairbanks, from scratch): the zone-command decode opens coil and
damper together, both proportional to the policy's u (the current `sb_zone_curve` opens the
coil fully by u=0.5 with the damper still at its floor and reaches boiler-max water at
u=1), and the water fill is capped by a 4 K demand reset band instead of "the zones are
asking for it, so full boiler". Constants copied from the shipped controller that scores on
this scenario's climate: REHEAT_FULL_K 2.0, zone_damper_full_k 2.0, RESET_FULL_K 4.0
(`starter-kit/smart_buildings/baselines/pid_plus.py`), which its own docstring calls a
"resolution problem rather than a lack of authority". Open-loop only: no deficit feedback
in the per-zone command, so the decode never replaces the policy's knob.

Verified grounding from the frozen-σ series: saturation is what makes a pin expensive —
gas 850–1,025 MWh inside the 10–25M pin while the same arm passed a 15–57 MWh plant-off
detour and re-entered hot rail at 50–56M; with the rail demand-limited, full boiler survives
only when the building is genuinely ≥4 K short (mean-zone deficit), which is the case the
score pays for.

Pre-registered gates (mechanism gate read before any fp64 month):

- mechanism: any sustained saturation window (logit_outside > 0.9 ≥ 1M steps) must hold
  gas ≤ 250 MWh — saturation no longer couples to boiler max. logit_outside falling below 0.5
  is diagnostic, not decisive: a bounded-fuel pin still answers the rail question.
- parity vs r59's months: 5.0M ≤ 0.620, 10.0M ≤ 0.492, 20.1M ≤ 0.314; miss all three → stop.
- too tight: env/worst_dev_K > 5 K with gas < 100 MWh at two consecutive saves → the band
  is under the plant's proven 2 K envelope; the NEXT arm widens it, no mid-run change.
- too loose: gas > 500 MWh with comfort ≤ 0.10 at two consecutive saves → the rail did not bind.
- target: a month < 0.1475 held two saves (gas > 100 MWh, productivity > −1) — under the best
  shipped baseline on this exact scenario (fixed_setpoint 0.1475, pid 0.1513, pid_water 0.2117
  per REFERENCE.md, same |mean step reward| metric since R ≥ 0 per step). r48's 0.134 has
  cleared this bar once; nothing since has.
  Open design question for the PROTOCOL — resolved 2026-10-25: the water reset reads the
  mean zone deficit (pid_water warns that a coldest-zone-driven reset makes satisfied zones
  receive hot water too — the worst criterion for the worst-zone metric); a quantile reset is
  the pre-registered NEXT arm if the too-tight branch fires, not a mid-run switch.
  Corrections accepted into the protocol: the |R| = −R metric equivalence is MEASURED at the
  pre-flight (positive-reward step count included), not argued from term signs; the frozen-σ
  entropy series is a genuine multi-excursion wobble (−67 at 25.1–25.4M, −79…−88 at
  45.0–45.4M during the plant re-light, −160 at 53.2M); r61 states its independence from the
  live r60 (r60 owns rail/chain amplitude 6; r61 runs amplitude 3 and owns the decode band —
  no σ freeze, no amplitude or chain change).
  Protocol written: `complete_port/results/study_r61_mod_band/PROTOCOL.md` (gates in read
  order 1–5, the mechanism gate tests fuel-≤-250 MWh inside saturation windows before any
  fp64 month; oracle pre-flight `pid_water` + `fixed_setpoint` decompositions pin the
  baseline bar; build plan with the amplitude-3 revert, sha-pinned binaries, verify gate).
  Status: NOT launched, awaiting the user's launch decision; nothing starts before
  r60_alive_mean's verdict.

r61_mod_band execution record (2026-10-25, user instruction: "do it and run the
verification in 3060. after r60_alive_mean run completed. shall run r61_mod_band
training run in 3090"): the pre-flight, implementation and gate are done; the launch
is queued behind r60_alive_mean.

Pre-flight (oracle, CPU, `study_r61_mod_band/preflight.py` → `baseline_row.json`,
building_train_1/fairbanks_1, 8640 steps, both policies run with reward_scale 1,
shift 0): pid_water mean −reward 0.21392 (productivity signed −0.11691, energy
0.27507, carbon 0.44382; elec 40.30 MWh, gas 295.0 MWh); fixed_setpoint mean −reward
0.14910 (productivity signed −0.05516, energy 0.20656, carbon 0.37347; elec 23.54 MWh,
gas 252.9 MWh). **|R| = −R is measured, not assumed: zero positive-reward steps in
both baselines**, and the component decomposition reproduces the stored env reward to
2e-9, so this month's bar in our metric is exactly mean −reward. Our own oracle row
reads 0.2139/0.1491 where the shipped REFERENCE.md recorded 0.2117/0.1475 (+1.1%
relative on both — consistent build/RNG delta, now pinned by our own row; the t bar
stands at 0.1491 for this pipeline).

Implementation (canonical sources): `patches/pufferlib_core.diff` reverts the squash
map to amplitude 3 (squash_location 3·tanh(logit/3), loss chain 1−(loc/3)², rail
read at |logit| > 3, pufferl banner "squash amplitude 3 (3\*tanh(logit/3), r61 mod
band)"; check_ppo_math.py reference reverted to match); `patches/src/sb_zone_command.cuh`
(= `gpu/sb_zone_command.cuh`, one file, symlinked) replaces the 0.5-split curve with
the joint proportional warm stroke (u>0: damper = floor + u·(1−floor), reheat = u) and
replaces `sb_water_cap` with `sb_water_reset_cap` — the mean over zones of
max(0, heating setpoint − zone temp) native K, neutral at 2 K, full at 4 K, off when
inside bands; the zone_command kernel computes the deficit from the observation slots
it already resolves (open loop, no zone-error feedback into u); contract test cases
updated in `patches/src/sb_shared_checks.cuh` (incl. cap cases 0→−1, 2→0, 4→1).
The cool decode, plant-from-heat and floor are untouched.

Binaries: fresh worktree /data/puffer-work/plr61 (base 6ffa5b1 + deploy + patch);
fp64 shared → `puffer_score` sha 66b5fd6b…c3f13, SB_F32+SB_SHARED_POLICY → `puffer`
(train) sha 30bb7908…df63; train binary carries the fp32 solve symbol; score binary
does not. Driver `study_r61_mod_band/run_r61_mod_band.py` (r60-driver pattern, + the
pre-flight row and gate-log checks, + "squash amplitude 6" forbidden in both binaries).
Gate `results/verify_20261025_183001.log` (SB_GPU=1, on the 3060). Launch: queued —
`study_r61_mod_band/wait_and_launch.sh` (armed) starts the driver once r60's driver
has exited with `study_r60_alive_mean/curve.json` present AND the gate log ends ALL
CHECKS PASSED; APPROVED is written only after the gate passes.

r60_alive_mean VERDICT (2026-10-25, run complete, 13 fp64 months in
`study_r60_alive_mean/curve.json`): NOT a win. Curve: 5.0M 4.145 (entry extinction,
gas 15.6 MWh, comfort −6.89); 10.2M 0.434; then a five-checkpoint plateau **0.479 with
gas pinned 404–424 MWh while logit_outside held ≥0.99 through the 14–31M saturation
window**; partial descent as the wobble pulled logit_outside to ~0.42 (45.1M 0.353,
50.1M **best 0.3035**, 55.1M 0.304); then **hot-rail re-entry at the end** — 60.2M
0.644 (gas 891 MWh, elec 73.8) and 65.2M 0.734 (gas 988.5 MWh, elec 115.8). Best
0.3035 sits above r59's best 0.237 and far above the 0.1491 bar; nothing held.

What the run establishes: (a) the alive-mean map fixed the dead-chain detonation
mechanism (no mid-run extinction; pin and re-entry happened without it), so the
σ/KL/chain-ratchet family is closed — three arms tested, each refuted by its own
telemetry; (b) the pin equilibrium is fuel-blind: saturation sat at ~424 MWh for
20M+ steps with an honest mean gradient, exactly what the decode-authority
hypothesis predicts; (c) re-entry without a pinned logstd channel reproduces r59's
correction — the wobble is structural to the decode's authority shape, not to the
location map. r61's premise is confirmed by r60's own fuel series; amplitude 6 is
retired with it (r61 runs the reverted r37 map).

r61_mod_band LAUNCHED (2026-10-25 13:29 UTC) on GPU 0 by the armed waiter after the
user's instruction "after r60_alive_mean run completed, shall run r61_mod_band
training run in 3090": from scratch, seed 73, 65,175,552 steps, 13 checkpoints.
From-scratch banners confirmed (pufferl: Muon; squash amplitude 3 (3\*tanh(logit/3),
r61 mod band); zone command on (proportional band, pump follows the average,
water reset 4.0 K); no loaded momentum; checkpoint_optim=0). Gate
verify_20261025_183001.log: ALL CHECKS PASSED (ran on GPU 1, sb GPU assignment by
the user). Pre-flight row pinned in that message above. Gates live per

#### r62 — the learning-rate anneal, two seeds against their own controls (started 2026-10-25 22:06 IST)

First arm of the prior register. Base is **r54's exact recipe** (λ 0.95, `zone_command` off, frozen return, adv_norm 2, Muon, squash amplitude 3, chain floor 0.25, `ent_coef` 0.001 flat, `vf_clip_coef` 10000); the one change is `anneal_lr` — arm 1 (cosine 1e-4 → 0), control 0 (flat 1e-4). Two seeds, 73 and 74, each with its own control; 65,175,552 steps, 13 saves, each scored as one fp64 month. Decision is read on the **final** checkpoint: win = both seeds' final month within 0.05 of that run's own best and the arm's final-month mean below the control's; strong win = final under 0.1491; refuted = either seed more than 0.10 above its own best, or plant-off at the end.

Parity checked before launch: the r61 binary reproduces r54's 50.1M month **exactly** on r54's checkpoint — 0.2267760469 with every component identical to the last digit — so with `zone_command=0` the map is r54's and no rebuild or gate was needed. Restarted once after a scheduling change; the two partial directories are kept as `*_restart1`. Seed 73's pair runs on one card, seed 74's straddles both, so a seed-74 gap is read against seed 73 first.

study_r61_mod_band/PROTOCOL.md: mechanism gate read before any fp64 month
(sustained saturation window (logit_outside > 0.9 ≥ 1M steps) ⇒ gas ≤ 250 MWh),

**Result (2026-10-26 04:19).** Refuted on both criteria.

| run         |        best month | final month | final − best | final gas / productivity |
| ----------- | ----------------: | ----------: | -----------: | ------------------------ |
| arm s73     |       3.119 @5.0M |       8.583 |       +5.464 | 3 MWh / −14.29           |
| control s73 | **0.2259 @50.1M** |       0.363 |       +0.137 | 528 MWh / −0.16          |
| arm s74     |      0.285 @15.0M |       4.653 |       +4.368 | 21 MWh / −7.73           |
| control s74 |      0.243 @10.0M |       7.210 |       +6.966 | 16 MWh / −11.98          |

Arm final mean **6.618** against control **3.786**; arm best mean **1.702** against control **0.235**. No arm beat its control on the best month, and every final month is plant-off or venting.

The control is the useful positive: on seed 73 it reproduces r54's month at 50.1M to 0.001 (0.2259 against r54's 0.2268), so the binary, the flags and the parity claim hold end to end — and it then drifts to 0.363 by 65.2M, which is r54's own recipe failing the hold test at the longer budget.

Why the anneal lost: at 10M the cosine has spent only 6% of the step size, and the two arms had already split — arm s73 at 11.19 (venting) against control s73 at 0.282 (heating), same seed, same flags. A 1–6% step-size difference flips the attractor. The anneal then freezes whatever basin the policy is in: the arm that collapsed at 10M is still at 8.58 at the end, while the constant-lr control keeps moving. A schedule applied after the basin is chosen can only lock it in.

parity 5.0M ≤ 0.620 / 10.0M ≤ 0.492 / 20.1M ≤ 0.314, too-tight (worst_dev_K > 5 K
with gas < 100 MWh at two saves ⇒ widen in the NEXT arm), too-loose (gas > 500 MWh
with comfort ≤ 0.10 at two saves), target a month < 0.1491 held two saves.

#### r63 — the entropy-coefficient anneal, dose probe included (C4, approved 2026-10-26 on the user's recommendation: "hold C14, run C4 first, because r62 showed the damage happens by 15M while the learning rate is still at 85–90% of its start")

Base is r62's control (r54's exact recipe, `anneal_lr=0`; controls reused: same base,
same seeds, same budget). Change: `anneal_ent_coef=1`, cosine `ent_coef → min_ent_coef_ratio × ent_coef`.
s73/s74 run 0.01 → 0.001 (ratio 0.1, both seeds wanted); s75 runs 0.005 → 0.001 (ratio 0.2)
on GPU 1 as a dose probe with no paired control. Budget 65,175,552 steps, 13 saves, every
checkpoint scored as one deterministic fp64 month. Binary = `study_r61_mod_band` runtime
and its score runtime, hash-pinned, flags only.

**Mechanism gate (read before any month), measured on seed 73: FAILED.** The gate wanted
entropy above −150 nats through 10M and `logit_outside` under 0.5 through 15M. Even at the
10× premium, entropy fell to −112.7 by 2M, −179 at 8M and −167.6 at 10M (min −179.2 through
10M); it never recovered, ending near −240. `logit_outside` crossed 0.5 inside the 15M window
(max 0.565). So the coefficient did reach the window and the collapse happened anyway.

**Result — s73 complete (2026-10-26 06:26 IST), REFUTED per the protocol's refutation clause.**

|        step |     score | productivity | energy | carbon | elec MWh |   gas MWh |
| ----------: | --------: | -----------: | -----: | -----: | -------: | --------: |
|        5.0M |     4.860 |       −7.826 |  0.342 |  0.478 |     58.8 |     308.7 |
|       10.0M |     1.305 |       −2.084 |  0.091 |  0.180 |      9.8 |     124.2 |
|       15.0M |     5.188 |       −8.636 |  0.011 |  0.022 |      1.0 |      15.1 |
|       20.1M |     0.567 |       −0.439 |  0.445 |  1.071 |     14.0 |     752.9 |
|       25.1M |     0.379 |       −0.162 |  0.419 |  0.991 |     14.6 |     696.9 |
|       30.1M | **0.376** |       −0.485 |  0.133 |  0.291 |      9.5 |     202.8 |
|       35.1M |     5.834 |       −9.556 |  0.230 |  0.273 |     46.9 |     169.6 |
| 40.1M–65.2M | 5.80–9.86 |   −9.6…−16.4 | ≤ 0.16 | ≤ 0.16 | 9.9–43.8 | **14–98** |

Best 0.3756 @30.1M, final 9.3547 (gas 14.0, prod −15.6 — plant-off), `final − best` +8.979.
The curve is a bounce: extinction at 15M, a heated rail 20–30M (gas 203–753 MWh), then
plant-off for good from 35M. Two things the register should carry: the **pin never appeared**
(logit_outside > 0.9 on 7 of 884 telemetry points, max 0.94) — extinction happened on an
unpinned policy — and the 331-nat entropy swing has the family's shape despite the premium.

**Dose probe s75, partial:** the host killed it at ~46.1M (epoch 625); its nine checkpoints
scored deterministically on GPU 1 at 14:44 IST: 5.0M 3.096 (gas 363), 10.0M 9.022 (gas 457),
15.0M 1.447, then plant-off from 20.1M (gas 5.4 → 70.1, prod −17.3 at 45.1M, final partial
month 10.54). The lower premium died **earlier** than the 10× one. Curve:
`study_r63_ent_anneal/r63_ent_anneal_s75/partial_curve.json`.

**s74 complete (2026-10-26, training exit 0 at 04:11 UTC, 884 epochs; 13 months scored by
04:29 UTC), REFUTED per the protocol's refutation clause.** Curve
(`results/study_r63_ent_anneal/r63_ent_anneal_s74/curve.json`), score / productivity / energy /
carbon / elec MWh / gas MWh: 5.0M 1.397/−2.076/0.242/0.514/17.9/356.2; 10.0M 0.234/−0.147/0.225/
0.503/12.0/350.6; 15.0M 4.505/−7.451/0.074/0.098/13.5/62.4; 20.1M 6.012 (gas 29.9); 25.1M 4.667
(gas 40.9); 30.1M **0.231**/−0.254/0.136/0.255/14.1/173.8; 35.1M 0.245/−0.342 (gas 88.9);
40.1M 0.238/−0.325 (gas 92.9); 45.1M 3.362 (gas 24.2); 50.1M 4.202 (gas 13.7); 55.1M 3.886;
60.2M 2.291 (gas 129.5); 65.2M **5.122**/−8.534 with gas 0.5 — fully off. Best 0.2307 @30.1M,
`final − best` **+4.891**, ends plant-off, so the refuted limb (gas < 100 MWh, productivity
< −2) fires on the second seed too.

Two things s74 adds beyond s73. First, the mechanism signature is identical despite the 10×
premium: entropy fell to −316.6 by 9.5M and `logit_outside` reached 0.80 at 10M (case B at both
reads, as routed), and the pin never appeared at all (0 of 884 points above 0.9; s73 had 7).
Second, **30.1M–40.1M is the longest held row any arm has produced: three consecutive saves at
0.2307/0.2454/0.2384, within +0.015 of best, all heated** (gas 89–174 MWh, productivity
−0.25…−0.34). r62's anneal arm on this seed held two saves (0.285/0.299); nothing has held
three. It is above the 0.1491 bar and above r48's 0.134, so it does not clear the target — but
it shows the recipe can sit still for 10M steps, and the failure is walking off the plateau from
inside it: at 40–45M entropy sat at −150…−185 and `logit_outside` at 0.43–0.55 (unsaturated)
while the month went 0.238 → 3.362.

The seed direction is inconsistent, which settles the arm: on s74 the anneal outlived its own
control (r62 ctrl s74 died at 20M; the arm held to 40M), on s73 it died earlier than its control
(ctrl final 0.363; arm plant-off from 35M). Not an exploration lever in either direction. One
seed each; not a leaderboard result.

What this arm settles: the entropy collapse is **not** the standing `ent_coef` — a 10×
coefficient does not hold exploration open and the endgame dies plant-off anyway, on the same
budget under the same recipe that r62's own control held all the way on seed 73. Per the
protocol: the saturation is not an exploration problem and **C14 moves up**.

**Case routing on the user's 14:55 decision tree (read at 10M and 15M, 2026-10-26).** The tree
uses two telemetry thresholds: entropy −150 nats and `logit_outside` 0.5.

| seed                          | ~10M                                      | ~15M                                                | case                                       |
| ----------------------------- | ----------------------------------------- | --------------------------------------------------- | ------------------------------------------ |
| r63 s74 (live arm)            | entropy −237.8…−315.9, logit 0.56→0.80    | entropy −205.7…−257.5, logit 0.43–0.61 (core > 0.5) | **B**                                      |
| r63 s73 (complete)            | entropy −167.6, logit 0.288 (A limb here) | entropy < −150, logit 0.565                         | **B by 15M**                               |
| r63 s75 (0.005 dose, partial) | —                                         | —                                                   | dead by 20M, never entered a heated window |

At 10M the cosine anneal still pays ~0.0095 effective bonus; entropy is −300 there. That is
case B verbatim: **the update overpowers a squash-corrected bonus at about 0.0095 ⇒ next is
C14, per-feature step sizes.** Case D (λ = 1) is rejected: s73's 5M month already bills gas
308.7 MWh with comfort −7.8, a heated-not-collapsed draw — the gradient was not pointing "off"
at entry; it became deterministic (entropy −303) while heated 20–30M, which is B's signature.

Holding verdict per the new AGENTS rule (2026-10-26): no r-family arm has ever held. Best
single-touch months: r54's grid came closest (final = best), r61/r62/r63 all give +0.13…+8.98
by the end. C14's pre-declared convergence test is a row held two consecutive saves, per `§7`
metric 6 — C14's own gates must include that.

#### r70 powered probe + r71 zone credit (2026-10-26)

**r70 powered probe** (CPU, eval-only). This is the frozen `r62_ctrl_s73` at 50.1M, run for 24 independent months
per noise regime, with σ 0.3 on reheat (`study_r70_gsde_probe/powered/analysis.json`). The table gives t for the
reheat gradient; + means the advantage rewards more heat. Each cell is GAE (λ 0.95) / full return (λ = 1).

| noise    | building credit | own-zone credit | 70/30 blend   |
| -------- | --------------- | --------------- | ------------- |
| white    | −2.35 / −2.68   | +0.81 / +0.72   | −1.52 / −1.79 |
| 1 h hold | +1.34 / +1.57   | +1.69 / +1.46   | +1.61 / +1.76 |
| 4 h hold | −0.36 / −0.53   | +1.61 / +1.42   | +0.16 / +0.07 |

- No cell passes |t| > 3.
- Under white noise, building credit leans toward cutting heat; own-zone credit removes that lean.
- Held noise gave no resolved comfort gradient.
- The first one-month C18 verdict is superseded. It used the clipped applied noise, which is biased.

**r71** (`complete_port/results/study_r71_zone_credit/`) put the zone critic (0.70 building + 0.30 zone advantage
on the valve heads) against a matched control. Both use `freeze_return=0`, because the binary refuses the zone
critic with a frozen return.

- **Runs:** seeds 73 and 74, 65.2M steps each, all four trained to the end. `r71_ctrl_s74` was moved to the 3090
  by user instruction.
- **Scoring:** month scoring was stopped by user order at 20:20 IST. Seed 73's pair has 13 fp64 months; seed 74's
  pair has only the 5.0M month.
- **Seed 73 months:**
  - arm: best 0.2743 @50.1M, final 0.2774 (+0.003); held 35–40M and 60–65M near 0.28–0.32.
  - control: best 0.3222 @40.1M, final 0.7504 (+0.428).
- **Live score, mean of the last four 5M windows:** arm 0.337 / 0.276, control 0.412 / 0.266 (s73 / s74). This is
  the stochastic policy with desync phases mixed, not a month. Its log correlation with the months is 0.62–0.73.

**Verdict.**

- **Zone credit:** not shown. The arm wins on s73, and the control matches it on s74.
- **What all four r71 runs share:** `freeze_return=0`. None pinned (r62 controls: 0.105 / 0.223 of epochs), and none
  went to plant-off in the second half.
- **`r71_ctrl` vs `r62_ctrl`:** they differ by exactly `freeze_return` (same binary; `args.txt` diff), on the same
  seeds.
  - s74: the r62 control was plant-off from 20M to the end; the r71 control recovered to 0.24 live.
  - s73: the r71 control over-fired (400–760 MWh live gas), and the r62 control's month best (0.226) beats it.
  - Seed 73's pair ran on different GPUs.
- **`freeze_return=0` (P19) is a candidate, not a finding.** Nothing reached 0.1491.

#### r72 — seeds 75/76 for the three cells (2026-10-27)

`complete_port/results/study_r72_seeds/`. All six runs exited 0 after 884 epochs. fp64 months were scored on the
last four saves only (50.1 / 55.1 / 60.2 / 65.2M). The tables are in `live_tables.txt` and `live_analysis.json`.

| cell                                 | s73   | s74                  | s75                        | s76                 | runs with a plant-off window after 25M |
| ------------------------------------ | ----- | -------------------- | -------------------------- | ------------------- | -------------------------------------- |
| fr1 (`freeze_return=1`) final month  | 0.363 | 7.210                | 6.635 (plant-off from 30M) | 0.778 (gas 898 MWh) | 2/4                                    |
| fr0 (`freeze_return=0`) final month  | 0.750 | unscored, live 0.265 | **0.249**                  | 1.891 (cold 30–50M) | 1/4                                    |
| zone (fr0 + zone critic) final month | 0.277 | unscored, live 0.276 | 0.324 (gas 485 MWh)        | 0.286               | 0/4                                    |

- **Best month:** fr0_s75 at 60.2M, **0.1691**: productivity −0.187, energy 0.087, carbon 0.197, electricity 5.8 MWh,
  gas 137.7 MWh. That is the lowest month of the current family. It sits inside a four-save row (0.236 / 0.238 / 0.169 /
  0.249), but it is not under 0.1491 and was not held there.
- **`freeze_return`:** not supported as registered. fr0 lost s76 to a cold collapse, and fr1_s75 collapsed with 0 pinned
  epochs, so pinning is not the collapse mechanism.
- **Zone credit:** supported as registered, as stability, not level.
  - No zone run has an off window after 25M.
  - The scored finals cluster at 0.28–0.32 on three seeds.
  - zone_s76 still scored 3.818 at 50.1M (gas 63 MWh) before recovering.
  - zone_s75 over-fires gas.
- **Statistics:** at n = 4 the collapse counts (2, 1, 0) are a direction, not a significant difference.
- **The bar:** no cell reaches 0.1491.

**r72 dynamics** (`study_r72_seeds/dynamics.py`, `dynamics.txt`). These are the training logs of all twelve runs in
1M bins, with block-bootstrap trends over 45–65M. Score here is the training-episode score, not a month.

- **Collapse is reversible on zone, absorbing on fr1.**
  - The two fr1 plant-off runs (r62_ctrl_s74, r72_fr1_s75) are flat for their last 20M (t +0.7 / −0.9), with gas 0–21
    MWh. More steps of the same weights are not evidence they come back.
  - Every zone run that collapsed recovered:
    - r71_zone_s73: four episodes of 1–5M each.
    - zone_s75: 16M, from 39M to 55M.
    - zone_s76: first entered the heated mode at 56M.
  - fr0_s76 is still recovering at 65M (t −2.3).
- **Which runs were still moving at 65M.**
  - Improving: zone_s75 (gas −382 MWh per 10M, t −8.3, comfort held), zone_s76, fr0_s76, r71_ctrl_s73.
  - Worsening: r62_ctrl_s73 and r71_zone_s74. Both show entropy falling and logit_outside rising; r71_zone_s74 also
    shows value loss rising.
  - Two of the four zone finals (s75, s76) are on a descent that started late. The 65M "stability" is partly timing.
- **Leading indicators** of a worse score 5M later, Spearman over 600 run-bins (autocorrelated, so the effective n is
  smaller):
  - value-loss rise +0.27
  - logit_outside rise +0.19
  - entropy fall −0.18
  - clip fraction level −0.23
  - KL level −0.13

  These are weak but consistent: a flip is preceded by the critic's loss jumping and the heads moving toward the rails.

**r73 / E1 (zone cell to 130M, running).** `r72_zone_s75`'s binary, flags and seed, one from-scratch process to
130,351,104 steps (26 saves), on GPU 0 from 04:41 IST 2026-10-27. Replay check: the weights and Muon sidecars at
5.0–25.1M are byte-identical to r72_zone_s75's. By user order the run is train only: no months; `stop_after_train.sh`
stops the runner when training exits and byte-compares the first half with r72 (`replay_check.txt`). The month readings
in `study_r73_zone130/PROTOCOL.md` therefore wait for a scoring pass. Seed 76 did not fit the 4 h budget.

**r74 / E2 (own-zone valve credit, running).** New flag `train.valve_zone=1`: damper and reheat take the
whitened zone advantage alone instead of 0.70 building + 0.30 zone (PRIORS P34). Gate
`verify_20261027_044522.log`: ALL CHECKS PASSED. With the flag off, the new binary reproduces the r61 binary's
checkpoint and Muon sidecar byte for byte over two epochs of the zone recipe, so the four existing zone runs are the
control and only four new runs (seeds 73–76) are needed. Driver: `study_r74_valve_zone/run_r74.py`. By user order (3 h window)
one seed, 75, trains from scratch on GPU 1 to 65.2M, train only; launched 05:23 IST at ~6.9K SPS. No earlier run used
the flag, so there was nothing to extend.

**Planned, not run: continues 2026-10-27 18:00 IST** (user, 2026-10-27 05:32 IST). Each needs its own launch approval as
usual.

- **E2, URGENT:** own-zone valve credit (`train.valve_zone=1`), seeds 73, 74 and 76 against the zone cell. Seed 75 is
  training now (r74_valve_s75, GPU 1, train only).
- **E1: does zone credit keep improving past 65M?**
  - _Design:_ run zone s75 and zone s76 from scratch to 130M on the 3090, on the same binary and seed.
  - _Built-in check:_ no learning-rate, entropy or logstd schedule depends on the budget, and the same seed on the same
    binary has reproduced a run before (r62_ctrl_s73 matched r54 month for month). So the first 65M should replay the
    existing runs exactly, and the second half answers the question in a single process, with no resume.
  - _Cost:_ about 5.7 h on the 3090.
  - _Pass:_ the late descent continues and holds, with no absorbing collapse.
  - _Status at writing:_ the s75 half is training now as r73_zone130_s75 (train only; byte-identical to r72_zone_s75
    through 25.1M). s76 has not run.

**Stopped by the user, 2026-10-27 ~08:50 IST.** r73 / E1 at about 107.9M (last save 105.3M; the weights replay
r72_zone_s75 byte for byte through 65.2M). r74 / E2 at epoch 537, about 39.6M (last save 35.1M). Neither was scored.

**r75 comfort-wall analysis (CPU only, no GPU; `complete_port/results/study_r75_ceiling/`).**

- _Wall:_ over 8,802 full training episodes after 20M from r62, r71, r72, r73 and r74, no episode beats productivity
  −0.148, whatever the fuel (100–2,000 MWh gas). fixed_setpoint reaches −0.055 on 253 MWh. `frontier.py`.
- _Spread, not level:_ in r48's 0.134 month the building mean is in band all day. 56% of the occupied comfort loss is
  overheated zones (25.6–28.6 °C) and 44% is cold zones (17.9–19.9 °C). `zones_r48.py`.
- _One command for every zone:_ in all 47 recorded months, the across-zone std of the applied damper and reheat
  commands is ≤ 0.006, and the policy's own outputs agree. `spread_all.py`, `valves_r48.py`.
- _Not learned:_ the zone head's own-temperature weights stay at their initial ≈ 0.01 through 105M (r73) and 35M (r74,
  valve_zone), and the reheat sign is wrong. `head_weights.py`.
- _Rail:_ in the 0.134 month, 81% of damper and 50% of reheat zone-steps sit on the flat part of tanh.
- _Reading:_ every arm so far controls only building-level heat. The comfort floor is the zone-load spread, which fuel
  cannot cross, and the score hills are movement along that one knob. The open-loop zone valve is the problem to
  solve, before holding.
- _Limits:_ the spread and rail measurements come from the r40–r51 recordings. On r71–r74 only the wall and the weights
  are measured.

**Bug: zone credit used the wrong room (2026-10-27, found on CPU).** `zV[slot]` stored the VAV index, but the per-zone
reward, the zone_adv share and the zone-memory occupancy index per-zone arrays with it.

- _Extent:_ VAV index differs from zone index for 38 of 39 VAVs on building_train_1, and for all but one VAV in every
  shipped building.
- _Effect:_ on r48's month, the temperature each slot was credited with correlates 0.059 with its own room (the
  zone-specific part).
- _What it voids:_ the zone-credit readings of r51 zone_mix, r71, r72 zone, r73 and r74. Those valves were trained on
  other rooms' comfort, which fits the own-temperature weights never moving (≈ 0.01).
- _Fix:_ `zV = vav_zone_idx[vav_zone_offsets[v]]` in `complete_port/gpu/smart_buildings.cu`. `verify_all.sh --trainer`
  is pending (needs the 3090). The comfort wall analysis (r75) stands: building-credit runs never differentiate zones
  either.

**r76: correct per-zone credit (2026-10-27, approved, running).** This run re-tests zone credit on the fixed binary.

- _Fix verified:_ `SB_ZCHECK=1` gives 0 wrong zones (the old map: 97% wrong); `verify_20261027_153152.log` ALL CHECKS
  PASSED; zone-critic-off parity with the r74 binary is byte-identical.
- _Runs:_ zone cell from scratch, 80M, seeds 75/76. Arm: `valve_zone=1`; control: 0.70/0.30.
  - 3090: arm_s75, then ctrl_s75.
  - 3060: arm_s76.
  - ctrl_s76 goes to the first GPU free.
- _Reads and details:_ `complete_port/results/study_r76_zone_fix/PROTOCOL.md`.

- _Stopped 2026-10-27 18:21 IST (user order; the host had already rebooted ~18:14 and killed both processes):_
  arm_s75 complete (80.2M, 16 saves); ctrl_s75 killed at ~43.7M (last save 40.1M); arm_s76 killed at ~50.9M
  (last save 50.1M); ctrl_s76 never started. Train only, no months scored. Read: `study_r76_zone_fix/ANALYSIS.md`.
- _r76 read (CPU, training score, not fp64):_ arm_s75 collapsed 2–34M; its best held stretch is 68–72.5M at
  0.17–0.20 (productivity ≈ −0.2, energy 0.09, carbon 0.20, gas ≈ 145 MWh, electricity ≈ 6 MWh); then gas
  fell 250→150 kW at 72.8–73.7M, productivity fell, and from 74.3M the value loss spiked (77–399, EV −2). Final
  0.258. The own-temperature reheat weight stayed +0.002–0.009 (gate < −0.3, not met). Under Muon 90–95% of the zone
  head's steps cancel. In r48, r62, r72, r73 and r76, corr(log trunk grad, log value loss) is +0.74 to +0.96. The
  comfort term is 60–80% of the score.
- _Claims check (`study_r75_ceiling/claims_check.py`, 48 recorded months):_ the AHU cannot heat (SAT dead on 83%
  of occupied steps); heat is water-limited on 81% of steps; the pump is not masked (97 active heads); hot zones
  keep reheat open (98% > 0.3) and the reheat head sits at its top rail; warm-up is 13% of the comfort loss.

**r77: three single-factor arms on the r76 arm recipe (2026-10-27, approved 18:32 IST, running).** Seed 75,
from scratch, train only. Every run logs ||g_V|| and ||g_pi|| at the trunk (`train.trunk_log=1`, read-only).

- C `r77_headadam`: the zone head is stepped by Adam, and Muon does not step it. 3090, first, 70.2M.
- D `r77_ctrl`: the control plus logging only. 3090, second, 80.2M, so the log covers the 72.8–74.3M dip.
- B `r77_occ`: pool feature 69 (occupancy) goes through asinh. 3060, 70.2M, starts after the verify gate.
- A (separate value network) was dropped: r41/r42 reached 0.926 against 0.171, and r57 (critic cut from the trunk)
  went to the cold basin.
- With all flags off, the run is byte-identical to the r76 binary after two epochs. The parity and smoke runs with
  the flags on were stopped at the user's order.
- Details: `complete_port/results/study_r77_trunk/PROTOCOL.md`.
- _Stopped 2026-10-27 ~20:00 IST (user order) after an external audit (`complete_port/results/audit_astra_20261027/`):_
  C reached 60.2M (12 saves kept). Its Adam clip was broken: the reducer arguments were swapped and the sum of squares
  covered 65,536 of 440K elements, so Adam ran on the unclipped gradient. Not a valid test. r69 (IDBD) had the same
  two bugs. D logged 0 epochs. B never started because the gate failed on my plain-build guard.
- _Learner defects found and fixed (all in the binaries since r48):_
  1. The zone trace had one slot under async. It is a race, not observed in r76: the first-visit KL is about 1e-15 in
     all 1,088 epochs.
  2. The rollout graphs were captured before `g_zv_roll` existed, so rollout zone values were never recorded. Their
     only use is the zone value clip at 10000. The clip is an identity when the stale memory is finite (for example
     zero). r76's zone critic is unverified, not shown to be broken.
- _Fixes:_ two trace slots; zone buffers allocated before the first rollout, one per slot; the reducer; the build guard.
  Gate `verify_20261027_200752.log` ALL CHECKS PASSED. The audit's sanitizer runs on the fixed binary give 0 errors,
  against 8,640, 2,880 and 3 before. Fixed runtimes are in `results/study_r78_fixed/`.
  The zone-credit surrogate (zone return ≠ ∂R) is a design property and is unchanged.

**r78 diagnosis: saturation on r76_arm_s75 (2026-10-27, fp64 months, no training; `complete_port/results/study_r78_diag/`).**

- fp64 months: 70.2M is 0.178 (productivity −0.156, energy 0.130, carbon 0.294, electricity 7.7, gas 205 MWh). 80.2M
  is 0.288 (gas 83 MWh).
- The zone head sends one command to all zones at every save (across-zone std ≤ 0.01).
- Hot water stays at its low rail (310–313 K) in all five months. The other plant heads sit at rails 60–95% of
  occupied steps, and executed exploration moves the command 1–4% of its range.
- The dip is the uniform reheat falling from 0.92 to 0.52 as gas is cut. The best month has 14 hot and 8 cold zones.

**r79: Beta policy (2026-10-27, approved; `complete_port/results/study_r79_beta/`).**

- Single factor on the r76 arm recipe: the tanh-squashed Gaussian is replaced by a Beta policy (`policy.beta=1`,
  Chou et al. 2017, RLlib `TorchBeta`): α, β = softplus + 1 from two outputs per head, env action 2x − 1,
  deterministic = mean, closed-form entropy, `ent_coef` 0.001. Squash, chain floor and fresh-noise entropy are off.
- Checks before launch: PPO kernel math (170 checks, 6 Beta cases against an independent reference, digamma and
  trigamma against scipy, sampler moments and CDF), shared-policy finite differences of the second block, and
  compute-sanitizer memcheck/initcheck at 0 errors.
- Seed 75, from scratch, 80.2M on the 3090, train only. Control: r76_arm_s75 months (study_r78_diag).
- Result (fp64 months on the 3060, deterministic mean): 4.93, 7.02, 4.81 (5–15M, comfort off), then 0.236 (20M),
  0.165, 0.157, 0.142, **0.1221 @40.1M** (best; prod −0.093, energy 0.098, carbon 0.234, elec 4.4 MWh, gas
  164.7 MWh), 0.199 (45M), 0.136, 0.134, 0.131, 0.131, 0.124, 0.150, 0.143 (80.2M; prod −0.069, energy 0.158,
  carbon 0.351, elec 9.1 MWh, gas 245.6 MWh). `final − best` +0.021. Held: 50–70M is five consecutive saves in
  0.124–0.136; six saves within 0.015 of best. Productivity better than −0.148 on 10 of the 11 saves from 30M
  (not 45M). r76 at the same seed: 0.2435 @35M, best 0.1784 @70M, 0.2884 @80M; r79 is lower on every
  comparable save from 25M. Entropy fell to about −240 by epoch 100, then rose and sat at −40 to −75; KL
  finite (old_kl 0.017–0.09), no NaN. Cost: 8.3K SPS against r76's 12.8K (double Beta math in the PPO kernels).
  One seed.
- Mechanism (recorded months): plant heads off the rails from 40M (r76: water railed on 70–95% of occupied
  steps). Across-zone reheat spread rises 0.002 (40M) → 0.006 (45M) → 0.033 (80M), the first run above 0.006,
  and reheat tracks the zone's own temperature (correlation −0.94 to −0.98). The damper spread stays at 0.002–0.003, and
  no zone is in band for 90% of occupied steps. Hot water is higher at night than while occupied (preheat).
  Details are in the study's `PROTOCOL.md`.
- Seed 76: the first process was stopped at epoch 59 (user order) and restarted from scratch on a build whose recorder
  also writes the Beta concentration and per-step reward parts and fuels. Gate `verify_20261028_012921.log`.
- Seed 76 power cut (2026-10-28, about 03:00 IST). The run died near 30.1M with 6 saves. Months 5–25M were all
  comfort-off (4.59–5.27, gas 3–21 MWh), whereas seed 75 had escaped by 20M. Kept as `runs/r79_beta_s76_powercut_30m`.
  Restarted from scratch at 07:59 IST on the same binaries.
- Seed 76 result (restarted run, fp64 months, recorded): comfort-off for 5–45M (4.59–6.51, then 2.90 and 2.69
  at 40–45M), escaped at 50.1M (0.202), then 0.162, 0.157 (60M), **dip** 0.290 (65M) and 0.318 (70M), all 39
  zones cold while occupied, then 0.142, **0.1321 @80.2M** (best = final; prod −0.146, energy 0.066, carbon 0.157,
  elec 3.6 MWh, gas 110.2 MWh). `final − best` 0. Held: two consecutive saves (75–80M) within 0.011 of best; it is a
  hill, not a plateau (55–60M at 0.16, 65–70M at 0.29–0.32). Seed 75 escaped at 20M, seed 76 at 50M: the escape
  time is seed-dependent and is most of the budget. Reheat spread 0.007–0.011 on 75–80M; median α+β on reheat
  fell from 120 (5M) to 2 (30–45M, std 0.56, nearly uniform) and came back to 10–32 after the escape.
- Speed build (2026-10-28, user choice "safe3"): mixed-precision Beta head (lnB and the logp/entropy sums double,
  digamma/trigamma/logs/gradient fp32), fp32 Marsaglia–Tsang sampler, row-parallel diagnostics and zone-stat
  kernels. `check_ppo_math` 986 checks 0 failures (mutant fails 8); gate `verify_20261028_104356.log` ALL CHECKS
  PASSED (57, same list). Measured on the 3090 with r79's flags, seed 76, 2.95M: 8.05K → **11.9K SPS** steady
  (9.16 → 6.18 s/epoch), ×1.48; 80.2M ≈ 1h52m instead of 2h45m. Not bit-identical to the old binary (different
  sampler stream and rounding), so runs on it are a new build, not more seeds of r79's binary.
  `results/speed_beta_20261028/`.
- Speed2 build (2026-10-28, user order: faster without changing the score). Not `base.async` (it trains on a
  one-update-old policy, an algorithm change). Two kernel changes, both exact: `zm_update_k` one block per env /
  thread per zone (building sums in slot order), and `sample_logits` one warp per agent (lane 0 keeps the RNG
  stream order, the Beta log-probs run across the warp, summed in head order). Proof: training is
  bit-deterministic run to run, and the speed2 binary reproduces the speed build's 2.95M run exactly (5,120
  SB_UPDATE lines, weights and Muon momentum byte-identical). 3090, r79 flags: 12.0K → **13.3K SPS** (6.15 →
  5.55 s/epoch); from r79's binary 8.05K → 13.3K (×1.65). A multi-block diagnostics norm was exact but not
  faster (diagnostics run off the critical path) and was reverted. Gate `verify_20261028_124502.log` ALL CHECKS
  PASSED (57). The remaining step is the exact physics (~13 ms of ~15 ms per env step).
  `results/speed2_20261028/`.
- r79 replication on speed2 (2026-10-28): seeds 75 (3090) and 76 (3060), 70.19M (14 saves) each, from scratch (a 15.04M plan was stopped by user order at
  epochs 111 / 54 as too short to tell; kept as `runs/r79_beta_speed2_s7{5,6}_stopped`), every save fp64
  scored + recorded; compared from logs against the stored r79 seeds (`compare_speed2.py`).
- Exact-r79 proof (2026-10-28, user choice `proof_short`): speed2 differs from r79's stream only through the
  speed build's fp32 Beta head/sampler and parallel zone-advantage sum. With those three restored and the speed2
  kernels kept, seed 76 at 2.95M reproduces stored r79_beta_s76 byte for byte (5,121 SB_UPDATE lines; weights
  and momentum equal the old binary's) at **12.2K SPS** (×1.52 over r79's 8.05K). Proof binary only, not in
  `patches/`, no gate. `complete_port/results/exact_r79_20261028/`.
- r79 review (2026-10-28, CPU only, stored logs and recordings; `complete_port/results/study_r79_beta/review_20261028/`):
  - _Beta floor:_ α, β = softplus + 1 has a dead corner. On most saves at least one parameter of most heads sits at 1
    with u ≈ −12…−21 (gradient factor 1e-5…1e-9). On s76's 50–55M months the eval damper is exactly 0.5 on 88–99%
    of occupied steps, pump 83–85%, water 66–70% (both parameters on the floor: uniform in training, midpoint in eval).
    Damper β is on the floor on 74–100% of occupied steps from 60M. `beta_floor.txt`, `floor_depth.txt`.
  - _Where the comfort loss is_ (oracle shortfall, true occupied mask; Jan 2 and Jan 16 are holidays, arrivals from
    05:00): a fixed per-zone offset could fix 50–74% of it, one building-wide shift 23–42%, both 95–98%; which zones
    are hot or cold is constant through the month (half-vs-half r 0.98–1.00). The 05–06 h arrival hour holds 11–23%.
    Comfort is 0.088–0.106 of the 0.13–0.16 s76 scores. `zones.txt`.
  - _Zone head:_ the reheat own-temperature gain (u − v on T − heatSP) has the right sign and was still accelerating
    at the end: s75 −0.010 (30M), −0.099 (50M), −0.201 (70M), −0.363 (80M); s76 about 30M behind. The damper's
    per-zone gain stays ≈ 0. `head_weights_beta.txt`.
  - _Curves:_ both seeds slide into the cold basin identically at 2.5–5.5M (gas 156 → 12 kW, entropy −79 → −190,
    KL 0.002, clip fraction 0; productivity falls only from 4M; value loss then 1e4). Escape 20M / 50M. After escape
    the stochastic training score is flat (s75 0.13–0.20 from 22M to 82M) while the actor step stays 0.00041.
    s76's 65–72M dip comes with value loss ×15, plant KL ×4, clip fraction ×3. `curves.txt`, `collapse_window.txt`.
- r80 Beta floor 0.5 (2026-10-28, approved; `complete_port/results/study_r80_beta_floor/PROTOCOL.md`): α, β = softplus + 0.5
  (`policy.beta_floor`, default 1; Marsaglia–Tsang boost for shapes < 1), else r79 seed 76 on the speed2 build, 80.2M, GPU 0,
  every save scored fp64 + recorded on GPU 0 after training. Checks: ppo math 1833/0 fail; floor-1 parity with speed2 byte
  for byte (first try failed on an FMA-fusion change in the sampler, fixed); gate `results/verify_20261028_161103.log` ALL
  CHECKS PASSED. Started 10:59 UTC. Control r79_beta_speed2_s76 (GPU 1) died in the 16:06 IST power cut at 12 of 14 saves
  (60.2M), unscored; not resumed (needs an exception).
  - **Result (2026-10-28, 16 fp64 months):** best 0.1121 @65.2M (prod −0.041, energy 0.131, carbon 0.305, elec 6.2,
    gas 213.6 MWh); final 0.1626 @80.2M (prod −0.152, energy 0.111, carbon 0.247, elec 7.2, gas 172.2 MWh);
    final − best +0.051; held (≤ 1.1 × best, consecutive) 1 — from 55M the months alternate 0.120 / 0.141 / 0.112 /
    0.142 / 0.118 / 0.163. Control r79_beta_speed2_s76: best 0.1134 @45.1M, held 1. Matched 30.1–60.2M window (7
    saves): r80 mean 0.171 (sd 0.033) vs control 0.140 (sd 0.016).
  - **Verdict: not a fix, not better.** Fails its own score criterion (curve no better than control). Mechanism: the
    both-floor share stays < 0.5 (max 0.46, damper 35M), but the single-parameter floor is unchanged (damper β on the
    0.5 floor 65–97% of occupied steps from 20M; control 62–93% at floor 1): the parameter still runs to a saturating
    floor, now with J/U shapes, and the training policy is noisier (damper median executed std 0.34–0.69 vs control
    0.18–0.44). The control never had the uniform corner on its good months, so the targeted symptom was absent there.
    One possible gain: no month ≥ 1 at any save (first save 0.268); every floor-1 run had one at ≤ 20M (3.7–7.0).
    One seed; old vs speed2 builds of the _same_ seed diverge into different stories (s75 window mean 0.146 vs 0.265),
    so a single-seed arm cannot resolve effects under ~0.1. Review: `study_r80_beta_floor/review_r79_r80/`.
- r81 floor replication (2026-10-28, approved; `study_r80_beta_floor/PROTOCOL_r81.md`): n = 3 per arm. New runs
  `r81_floor05_s75`, `r81_floor05_s77` (GPU 0, in turn) and `r81_floor1_s77` (GPU 1), r80's binaries, 60.2M / 12 saves,
  scored fp64 + recorded on GPU 0; existing members speed2 s75/s76 (floor 1) and r80 s76 (floor 0.5). Decision on the
  30.1–60.2M mean (win ≥ 0.02 lower and lower in ≥ 2 of 3 seed pairs) and the ≤ 20.1M cold-basin count. Started
  13:12 UTC.
  - **Result: floor 0.5 loses; revert to floor 1.** 30–60M arm mean floor 1 0.199 (s75 0.265, s76 0.140, s77 0.192)
    vs floor 0.5 0.403 (s75 0.875, s76 0.171, s77 0.164); floor 0.5 lower in 1 of 3 pairs. Basin ≤ 20M: floor 1 3/3,
    floor 0.5 2/3 (not removed). r81_floor05_s75 escaped at 30M (0.237) and fell back cold 35–50M (1.0–1.76, gas
    65–102 MWh). **New best month: r81_floor1_s77 0.0939 @50.1M** (prod −0.050, energy 0.093, carbon 0.226, gas
    159 MWh), held 1 (0.112 / 0.127 after), after the deepest basin of any run (10.6 at 20M). Every run held 1.
  - Across all 8 Beta runs: every run is cold in its first 2.5M bin; training entropy < −100 → 86–96% cold bins,
    −100…−40 → 12–16%; escape follows re-widening. Entropy narrows before productivity falls.
    `review_r79_r80/entropy_vs_state.txt`, `holding.txt`.
- r82 entropy target (2026-10-28, approved; `complete_port/results/study_r82_ent_target/PROTOCOL.md`): total-entropy
  target −80, SAC temperature rule per epoch, coef in [1e-6, 0.05], floor 1, seeds 75 (3090), 77 (3060); seed 76
  stopped before start by user order. Gate `results/verify_20261028_221109.log` ALL CHECKS PASSED.
  - Controller windup: above −80 the coefficient fell to 1e-6, so the 2.5–5M slide went unopposed (−125 / −158); it
    then sat at the 0.05 cap 22 / 37 epochs with entropy < −100 (the pre-registered cap rule tripped; values finite,
    entropy rising throughout). At the cap the update ran hot (clip fraction 0.10–0.13, g_act 100–160).
  - r82_ent80_s75 (complete): 30–60M mean **0.148** (sd 0.016) vs control speed2 s75 0.265 (sd 0.065); best 0.1226
    @35.1M (prod −0.093, energy 0.103, carbon 0.231, elec 6.8, gas 161 MWh); final 0.1422 (prod −0.127, energy 0.105,
    carbon 0.225, elec 8.8, gas 156 MWh); final − best +0.020; held 1. But cold longer: 10 cold training bins vs 5,
    months 4.6–5.7 to 15M, escape 20M. Mechanism failed.
  - Why: the total is ~90 heads, mostly zone heads. At 15M (total −48, "wide") the dampers were near-uniform (−0.16
    nats/head) while pump (m 0.27, −1.83) and water (m 0.20, −1.95) were narrow and off. Every cold month has the same
    plant signature (reheat open 0.75–0.98, dampers 0.02–0.03, water low, pump off); escapes coincide with the pump
    head widening/turning on (r82 s75 20M, r81_floor1_s77 25M). `study_r82_ent_target/review/role_entropy.txt`.
  - r82_ent80_s77 stopped by user order at ~55.1M (11 checkpoints kept, no month scored) to free GPU 1 for r83.
- r83 per-role entropy target (2026-10-29 00:55 IST, user order, `complete_port/results/study_r83_ent_role/PROTOCOL.md`):
  one SAC-rule coefficient per role (zone / AHU / plant) on that role's mean entropy per active head, target −1.5,
  coef in [0.001, 0.01] (floor = control). Seeds 75 → 76 on GPU 0, 77 on GPU 1; 60.2M, 12 saves, fp64 + SB*RECORD,
  role_entropy per month. New logging on this build: `loss/ent_head*{zone,ahu,plant}`, `SB_ENTROLE`. Gate
`results/verify_20261029_003405.log`57 PASS ALL CHECKS PASSED (first try failed only check_ppo_math: harness
lacked the new helper; fixed in`tools/check_ppo_math.py`). Train binary sha 79f0792f… (SB_F32).
  - Calibration note at 3M: training per-head entropy is −1.1 to −1.3 while the total is already −115 (~95 active
    heads), so −1.5/head ≈ −140 total, deeper than the −100 cold boundary; all coefficients still at the floor.
    User decision: stopped at ~4M (kept as `runs/r83_entrole_s{75,77}_stopped_t15`), relaunched 19:31 UTC with
    target −1.0 per head (≈ −95 total), same binary.
  - Result (all 36 months, `study_r83_ent_role/review/decision_r83.txt`): 30–60M mean s75 0.191 / s76 0.247 /
    s77 0.156 vs controls 0.265 / 0.140 / 0.192; arm 0.198 vs 0.199 → **no difference** (lower in 2/3). Basin 3/3
    (months ≥ 1 to 10–15M); cold training bins 9 / 8 / 7 vs 5 / 8 / 10; plant entropy min −2.01 / −1.53 / −1.70
    with plant and AHU coefficients at the 0.01 cap through the cold phase. Best: s75 0.1343@50.1 (prod −0.105,
    energy 0.106, carbon 0.252, elec 5.0, gas 177 MWh), s76 0.1345@55.1 (−0.128, 0.085, 0.203, 3.7, 143), s77
    0.1270@30.1 (−0.114, 0.088, 0.207, 4.9, 145). Final − best +0.027 / +0.060 / +0.011; held 1 / 1 / 3.
    Mechanism failed: the plant stayed wide and all three still went cold.
  - Cold signatures across all 12 Beta runs (`review/cold_signatures.txt`; occupied-step mean commands): every
    warm month has AHU recirculation ≥ 0.7 (at the oracle's max_recirculation 0.7 cap: least outside air) and
    dampers open. Cold months are one of two: dampers closed (0.01–0.07; no air delivered) or dampers open with
    recirculation 0.01–0.5 (mostly −30 °C outside air; r83 s75 10M had water 0.98, pump 0.97, elec 94 MWh and
    scored 12.3). Neither actuator alone is rewarded: opening a damper under low recirculation chills the room,
    raising recirculation with dampers shut changes nothing. The pump-off reading of r82 was one instance.
  - The in-run role_entropy step failed on a wrong path in the copied script; fixed and run on the stored
    recordings (no month re-scored).
- r84 coherent Beta exploration (2026-10-29 06:38 IST, user approval; `complete_port/results/study_r84_coherent/
PROTOCOL.md`): `policy.beta_cn_tau=12` — the sampler's Gamma proposals start from per-agent, per-head AR(1)
  normals (1 h correlation), so each step is still exactly Beta(α, β) and the log-prob is unchanged; no entropy
  target, fixed ent_coef 0.001. Seeds 75 → 76 on GPU 0, 77 on GPU 1; 60.2M, 12 saves, fp64 + SB_RECORD +
  role_entropy per month. Controls as r83. Gate `results/verify_20261029_061946.log` 57 PASS ALL CHECKS PASSED,
  including the new coherent-sampler test in `check_ppo_math.py` (marginal exact for six shapes, lag-1 corr 0.75–0.84
  at shapes ≥ 1). Train binary sha e200150c… (SB_F32).
  - All three seeds got SIGTERM at 01:21 UTC (machine went down; s75 10.6M, s77 5.1M, s76 epoch 0); partial runs
    kept as `runs/r84_cn12_s{75,76,77}_reboot_0121`. Restarted from scratch 03:16 UTC on user order, same binary.
    Stopped again 03:30 and restarted 03:49 UTC from the user's other session (03:16 partials deleted, logged).
  - r84_cn12_s75 (complete): no month ≥ 1. 5M 0.238, 10M 0.182, then 0.102–0.122 from 15M to 60M. 30–60M mean
    **0.109** (sd 0.007) vs control 0.265 (r83 0.191, r82 0.148); best 0.1024@30.1M (prod −0.097, energy 0.063,
    carbon 0.157, elec 2.1, gas 111 MWh); final 0.1153 (−0.093, 0.087, 0.211, 2.7, 149); final − best +0.013;
    **held 5** (25.1–45.1M). Cold training bins 2 (2.5, 7.5M) vs 5. Recorded commands: dampers 0.4–0.8 open and
    recirculation 0.86–0.94 from the first save; supply air hot (0.94–0.97), water low, static pressure at minimum;
    electricity 2–3 MWh, the lowest of any run. Training entropy −100 to −140: narrow and warm.
  - s76 and s77 got SIGTERM 05:38 UTC (s76 14.7M, s77 42.6M). User order: s76 deleted and restarted from scratch on
    GPU 0; s77 **resumed (approved exception)** on GPU 1 from `ckpt/0000000040108032.bin` + Muon sidecar, same
    binary, new run `runs/r84_cn12_s77_seg2` (20.05M, local saves 5–20M = global 45.1–60.2M). Proof line
    `loaded optimizer momentum from /res/runs/r84_cn12_s77/ckpt/0000000040108032.bin.mb`, no warning. Not carried:
    Philox and coherent-noise state, desync phase, memory. The driver's from-scratch banner check killed its own
    wrapper; the container trained on: 4 saves of the right size, TensorBoard reaches 20,054,016, `train.log`
    (SB_UPDATE/SB_ENTROLE stdout) stops at 1.8M. Scored with `run_r84_resume_s77.py --score-only`.
  - r84_cn12_s76 (complete, from scratch): no month ≥ 1. 5M 0.324, 10M 0.143, then 0.088–0.119. 30–60M mean
    **0.103** vs control 0.140; best 0.0882@40.1M (prod −0.053, energy 0.083, carbon 0.199, elec 3.8, gas 140 MWh);
    final 0.1001 (−0.079, 0.077, 0.186, 3.6, 131); final − best +0.012; held 1 (the neighbours 0.0976 are 1.107 ×
    best). Cold training bins 1 (2.5M).
  - r84_cn12_s77, reported per segment. Segment 1 (0–40.1M): 0.305, 0.231, 0.152, 0.133, 0.118, 0.109, 0.129,
    0.141; best 0.1086@30.1M (−0.093, 0.077, 0.189, 3.1, 133); held 2; cold bins 2 (2.5, 7.5M). Segment 2 (global
    45.1–60.2M): 0.121, 0.115, 0.101, 0.117; best 0.1014@55.1M (−0.089, 0.069, 0.172, 2.6, 122); final 0.1168
    (−0.114, 0.070, 0.174, 2.6, 123); held 1; cold bins 0. The 30–60M mean over both segments is 0.119 vs control
    0.192 (segment 1's 30–40M alone 0.126).
  - **Decision (pre-registered): win.** Score: arm 0.110 vs control 0.199 (−0.089), lower in 3/3 seed pairs
    (−0.156, −0.037, −0.073). Basin: 0/3 runs with a month ≥ 1 (controls 3/3). Mechanism: cold training bins 5 vs
    23; the basin ends before the first save in 3/3. Holding is the open gap: held (≤ 1.1 × best, consecutive) 5 / 1
    / 1; the 30–60M level holds inside 0.088–0.141. Fairbanks only: no evidence on other weathers.
  - Transfer eval (user order 2026-10-29 13:15 IST; evaluation only): the seven 30.1–60.2M saves of each seed on
    phoenix_5 (b1), phoenix_4 (b8), atlanta_5 (b4), denver_6 (b8), denver_3 (b5), fairbanks_2 (b2); 126 fp64
    months, recorded, both GPUs (`study_r84_coherent/eval_gen6.py`, `gen6/summary.txt`). Seed-mean final:
    0.346 / 0.309 / 0.275 / 0.228 / 0.151 / 0.127 (pid_water 0.134 / 0.158 / 0.029 / 0.020 / 0.044 / 0.080;
    bang_bang 0.268 / 0.308 / 0.250 / 0.241 / 0.185 / 0.186). The loss is comfort: final productivity −0.29 to −0.66
    in the hot and mild months, energy and carbon low. Seed order is reversed from Fairbanks (s77 best everywhere,
    s75 worst); s77's lows (0.06–0.09) are single saves, held 1. Training on one cold month does not transfer.
  - Comfort diagnosis of the 21 r84 Fairbanks months (CPU, recordings; `study_r84_coherent/review/
zone_diag_occupied.txt`, `day_diag.txt`, `hw_vs_coldzones.txt`). Days 1 and 15 (Jan 2, Jan 16) are holidays
    with zero occupancy: the policy runs them as weekends at no comfort cost (a weekday-clock mask had wrongly put
    50–84 % of the deficit there; use the occupancy observation). The real comfort cost: (1) the 06:00 start: hour 6
    holds 11–34 % of the occupied squared deficit, the first three weekday hours 25–40 % of productivity, Mondays
    16–33 %; (2) six zones (35, 38, 11, 0, 39, 27, across AHUs 0–3) are 3–5 K under the heating setpoint for
    90–100 % of occupied time and hold ~55 % of the squared deficit. Every zone gets the same command (spread across
    zones: damper 0.01–0.04, reheat 0–0.07); reheat sits at 0.97–0.98 in s75/s77 while the hot-water supply is held
    at 38–45 °C (water command 0.03–0.06). Not a capacity wall: across 98 warm recorded months zone 35's deficit
    falls with hot-water temperature (corr −0.4) and reaches 0.02–0.6 K in some months.
- r85 four weathers (2026-10-29 16:14 IST, user order; `complete_port/results/study_r85_four/PROTOCOL.md`): r84's
  binary and flags from scratch, the batch changed to four scenarios, 64 envs each (same 256 agents and update):
  b1/fairbanks_1 (extreme cold, −21 °C), b4/denver_2 (cold, 5 °C), b5/atlanta_6 (mild heat, 25 °C), b2/phoenix_6
  (extreme heat, 33 °C). 80.2M, 16 saves; seeds 75 → 76 on GPU 0, 77 on GPU 1; ~8K / ~4.5K SPS measured. Scored as
  saves land: training scenarios every 10M, the six gen6 scenarios held out at 60/70/80M (150 fp64 months,
  recorded). Decision: held-out six-scenario mean vs r84's last three saves, win ≥ 0.03 lower in ≥ 2/3 pairs;
  fairbanks_1 regression > 0.03 reported.
  - Layout attempts (user orders): the first 4 × 64 launch was stopped externally at s75 16M / s77 9M (kept,
    `*_4x64_stopped`). 4 × 256 (1024 agents) failed `cudaMalloc` on 24 GB. 4 × 128 (512 agents) measured 9.9K
    SPS on the 3090 alone, below 4 × 64 on both GPUs (~8K + ~4.5K): stopped before a save. Restarted from scratch
    at 4 × 64 on both GPUs at 17:08 IST (s75 → s76 on GPU 0, s77 on GPU 1).
  - 20:54 IST: s76 stopped at ~1M on user order (one finished seed decides; kept as `r85_four_s76_stopped_user`);
    s77 continues. Same seed + layout reproduced the stopped 4 × 64 run's 10M months bit for bit.
  - Read of s75 (complete) and s77 (to 40M), `study_r85_four/review/{train_curves,cold_link}.txt`:
    held-out six-scenario mean at 60/70M 0.067 vs r84 s75 ~0.32 and pid_water 0.077; ten scored scenarios
    0.068–0.070 vs pid_water 0.083, fixed_setpoint 0.099. Wins in heat (phoenix_5 0.05–0.07 vs pid_water 0.134,
    phoenix_4 0.05–0.06 vs 0.158); loses in cold/shoulder (fairbanks_2 0.139 vs 0.080, denver_3 0.07–0.09 vs 0.044,
    denver_6 0.03–0.045 vs 0.020). fairbanks_1 60–80M 0.135 vs r84 0.114 (+0.021, within the 0.03 rule), still
    under every baseline (pid_water 0.212). Warm scenarios collapse at single saves (s75 40M: phoenix_6 0.397,
    atlanta_6 0.244; s77 20M: 0.479, 0.216; electricity falls to ~22 MWh), with value-loss, KL and |adv| spikes in
    the same training bins: not held. Every recorded month of r84 and r85, in every climate, holds the water
    command at 0.02–0.07 (hot-water supply 38–45 °C); the cold losses are zones under setpoint (fairbanks_2 45/85,
    denver_3 46–62/79 zones cold > 50 % of occupied time, dampers 0.18–0.34 in denver_3).
  - Water head read (`study_r85_four/review/water_head.txt`, recordings' loc/conc): head 216 is live (mask 1) and
    maps to a 310–350 K setpoint; 0.02 is ~38 °C. Over training its Beta concentration grows monotonically
    (α+β ~6 at 5–10M to 45–77 at 60–80M) while the occupied mean falls to 0.02–0.05; P(training draw > 0.25, i.e.
    > 47 °C) falls from 0.4–0.97 to 0.0001–0.03. In r85 it falls in all four climates together (phoenix_6 and
    > denver_2 start at 0.6, fairbanks_1 at 0.14–0.75; all at 0.02 by 60–70M). r85's AHU supply-air heads do the
    > same: 0.04–0.10 (≈ 286 K) in fairbanks_1 as in phoenix_6, where r84 held 0.97 (305 K) on fairbanks_1. Zone
    > heads differ by climate (reheat 0.9 in fairbanks_1, 0.3–0.6 in phoenix_6); the building-level setpoints do not.
- r86 counterfactual (2026-10-29, eval only, `complete_port/results/study_r86_counterfactual/`): r85_four_s75 80.2M,
  one fp64 recorded month per arm, `SB_FORCE` holds named heads (bin_cf = r80 score sources + that hook, sha 159449…).
  Hotter water is worse in every scenario: fairbanks_1 +0.014/+0.023/+0.031 (47/57/67 °C), fairbanks_2
  +0.030/+0.052/+0.068, denver_3 −0.001/+0.009/+0.015, phoenix_6 +0.000/+0.001/+0.002. It overheats zones (fairbanks_2
  hot zones 38 → 70) because reheat backs off only 0.80 → 0.60. The water head at its bottom is right for this
  policy; it is not the defect. Warmer supply air: denver_3 −0.015 (22 °C, gas 95 → 69 MWh), fairbanks_1 −0.004
  (27 °C), fairbanks_2 +0.010 (dampers close to 0.37, 36 zones cold), phoenix_6 +0.045 / +0.198.
  Correction of the r85 read: against pid_water, r85 loses on energy and carbon, not comfort. Productivity / energy /
  carbon, r85 vs pid_water: denver_2 −0.012/0.085/0.148 vs −0.081/0.011/0.018 (score tie 0.054); denver_3
  −0.072/0.045/0.072 vs −0.065/0.009/0.015; denver_6 −0.017/0.126/0.080 vs −0.025/0.021/0.005; fairbanks_2
  −0.075/0.094/0.228 vs −0.100/0.038/0.061. Occupied supply air is 12.7–13.7 °C in all seven scenarios read
  (fairbanks to phoenix) with reheat 0.6–0.84: cool-then-reheat. Unoccupied supply air does vary (18.6–29.5 °C).
  Analysis: `review/analysis.txt`.
- r86 joint and recirculation checks (2026-10-29, eval only, same checkpoint, `bin_cf2` adds `col:*s` reheat scaling;
  `review/recirc.txt`, `review/recirc_over_training.txt`). Supply air 19/22 °C × zone reheat scaled 0.25–1 on
  denver_2/3/6 and fairbanks_2 (34 months): no arm below pid_water on denver_3, denver_6 or fairbanks_2; cutting
  reheat collapses comfort except on denver_6 (s19_r50 0.030). Recirculation held at the 0.70 cap (rc70), all ten
  scored scenarios: lower in 10/10, ten-scenario mean 0.0629 → 0.0527 (pid_water 0.0828), comfort unchanged
  (e.g. phoenix_5 0.0647 → 0.0482, elec 46.7 → 27.1 MWh; fairbanks_1 0.1276 → 0.0953, gas 237 → 152 MWh). rc35:
  9/10, mean 0.0609. In cold scenarios rc70 raises electricity (fairbanks_2 12.7 → 25.4 MWh) because the AHU now
  cools warmer mixed air to the ~13 °C setpoint. Over training the policy's recirculation starts at 0.55–0.82
  (10M) in every scenario and falls to 0.03–0.08 (hot scenarios by 30M, then denver_2, fairbanks_1 to 0.30) while its
  α+β grows 4–7 → 15–47; r84 (fairbanks_1 only) held 0.86–0.94. Every climate is better with it high, so this is
  not a climate conflict. Why PPO walked it down is not in the logs (only per-group entropy is recorded).
  r85_four_s77 finished (16 saves); its 80.2M save swung to near-zero energy with poor comfort (atlanta_6 0.168,
  phoenix_6 0.225, fairbanks_2 0.204; recirculation 0.72–0.93). Held-out six-scenario mean 60–80M: s75 0.067, s77 0.094.
- r87 credit replay (2026-10-29/30, `complete_port/results/study_r87_credit/`): r85_four_s75 replayed with
  `SB_CREDIT` (per group and head type, the correlation of the advantage with the draw's offset from the Beta mean).
  It was bit-identical to r85 (31,104 update rows) until a power cut at ~17.9M. The user resumed it from 15.04M on a new
  random stream (`runs/r87_credit_s75_resume`), so that is a separate segment. On the identical segment the
  recirculation credit points **up** in fairbanks_1, denver_2 and phoenix_6 (t up to +5.5), and atlanta_6 is weakly
  negative. It is 2–5× weaker than supply air's, yet r85's recirculation fell over the same steps. The head moved
  against its own credit, so the cause is the shared update path, not the reward or credit. The per-AHU advantage is
  dropped. Table: `study_r87_credit/credit_read.txt`.
- r88 weight swap (2026-10-30, eval only, `complete_port/results/study_r88_swap/`): r85_four_s75's AHU readout
  (role[1], floats 46088..48392) swapped between the 10M and 20M saves, 4 fp64 months plus 2 references.
  Occupied recirculation, phoenix_6 / atlanta_6: 10M 0.548 / 0.615; 20M 0.239 / 0.223; 20M body with 10M readout
  0.638 / 0.604; 10M body with 20M readout 0.337 / 0.332. The readout carries the fall, not the shared body, so a
  separate critic is not the fix (and was dropped before: r41/r42). Every readout row moves 0.08–0.24 per 5M whatever
  its signal strength. The recirculation mean direction (u−v) moves persistently over 5–20M (consecutive cosines
  0.65, 0.35, 0.55), so it is driven, not a random walk, and against its measured advantage push. The pure 10M month
  repeats r85's actions exactly (score 0.1231578470 vs 0.1231578454).
- r89 update-path replay (2026-10-30, `complete_port/results/study_r89_ppd/`): r85_four_s75 replayed to 30.08M with
  SB_PPD. It is bit-identical to r85: 52,864 update rows, and all six checkpoints byte-equal. Per visit it records
  how far each AHU and plant Beta mean moves under the applied step and under three loss gradients (as trained,
  without entropy, without entropy and clip).
  Findings:
  - The optimizer, entropy and clip do not cause the fall. The step follows the gradient's sign at bin level.
    Entropy and clip are small and mostly push against the fall.
  - The clip on the geometric-mean ratio over ~230 heads barely acts: under 5% of samples fall outside it, and it
    never binds on A>0.
  - At 19.0–20.5M supply air falls 0.76 → 0.28 in all four scenarios at once. Recirculation follows from 21M
    (0.68 → 0.33), then water, and concentration locks them (α+β 7 → 13–18).
  - The gradient on recirculation, supply air and water is noise: in every 5M bin and scenario 41–65% of epochs
    point down, |t| ≤ 3.5. That noise sits against a real effect: rc70 on the 80M policy cut phoenix_6
    electricity 44 → 30 MWh.
    Cause: the whole-building advantage, dominated by ~90 zones' comfort, buries the AHU heads' real effect, and an
    optimizer whose step does not scale with signal turns that noise into full-size moves of the shared readout.
    Tables: `ppd_read.txt`, `switch_epochs.txt`, `snr.txt`.

- **r90 per-AHU advantage** (approved 2026-10-30, `complete_port/results/study_r90_ahu_critic/PROTOCOL.md`,
  PRIORS P37). Hypothesis: crediting each AHU with its own return fixes r89's finding that the AHU head gradients are
  noise. That return is its zones' comfort plus the whole building's energy and carbon, and at night the building
  reward. Change vs r85:
  - `env.ahu_critic=1` and `train.ahu_critic=1`: 9 more zone-critic value rows, each with its own GAE and whitening.
  - The sat, sp, recirc and run heads use the whitened AHU advantage.
    Everything else is r85's flags and layout. Seeds 75, 76, 77 run from scratch, one after another, on GPU 0 (3090)
    only, to 30.08M. The SB_CREDIT and SB_PPD diagnostics are in the same pass. fp64 months are recorded at
    10/20/30M on the 4 training and 6 held-out scenarios (90). Gate `results/verify_20261030_123632.log` ALL CHECKS
    PASSED. Launched 2026-10-30 07:31 UTC, 7.0K SPS. Seed 77 was stopped by the user before its first save.
    **Result: rejected** (`study_r90_ahu_critic/review/decision.md`).
  - Gradient consistency is unchanged: 17/96 and 13/96 cells have |t| > 3, against r85's 14/96.
  - Occupied recirc ≥ 0.4 held only in s75. In s76 at 30M it was 0.32 on phoenix_6 and 0.24 on atlanta_6.
  - fp64 training-4 mean, r90 vs r85 (2 seeds each): 10M 0.222 vs 0.178, 20M 0.105 vs 0.177, 30M 0.099 vs 0.067.
  - Productivity at 30M is worse in all 8 pairs, with similar energy and carbon.
  - What moved: the 20M recirculation collapse and dip are gone, but α+β still locks, now at high recirc.
  - Confound: r_k gave each AHU its own zones' comfort but the whole building's energy (3–9 AHUs per building).
    Energy was therefore overweighted 3–9× and Σ r_k ≠ R. The next credit arm must split energy by share (needs
    approval).

- **r92 readout Adam** (approved 2026-10-30, `complete_port/results/study_r92_role_adam/PROTOCOL.md`, PRIORS P39).
  Hypothesis: Muon orthogonalises each 4–12-row readout, so every head direction takes a unit-size step whatever its
  signal; that turns r89's AHU/plant gradient noise into full-size readout moves. Change vs r85: `train.role_adam=1`
  (Adam 0.9/0.999/1e-8, lr 1e-4, after the same 0.5 clip, on role[0..2]; Muon keeps the trunk). Binary = r90 sources +
  role_adam only (r91's `beta_cn_tau_ahu` edit is not bit-identical to r85 when off, so it is excluded). Seeds 75 then
  77 from scratch on GPU 0 only, to 50.14M, scored after each seed's training (70 fp64 recorded months per seed).
  New read-only log `SB_ROLESTEP` (per-role Σ‖g‖, Σ‖ΔW‖, net epoch displacement); SB_CREDIT, SB_PPD, trunk_log,
  update_diagnostics, TensorBoard all on. Parity `checks/parity.txt`: flag off = r85's SB_UPDATE rows bit for bit,
  logging on = flag off, role_adam=1 finite. Measured at start: Adam moves the readouts 2–8× further per epoch than
  Muon (AHU net 0.04–0.21 vs 0.03; plant 0.07–0.28 vs 0.026).

- **r94 Beta concentration cap** (approved 2026-10-31, `complete_port/results/study_r94_conc_cap/PROTOCOL.md`, PRIORS P40).
  Hypothesis: the r91/r93 hot-month failure (SAT lock at α+β → ~30, AHU run heads off, zones 305–311 K) is the
  exploration collapse of 2026-10-29 acting through the Beta concentration: once the heads re-concentrate, a corrective
  setting is not held long enough to be credited. Change vs r93: `policy.beta_conc_max=12` — a cap on α+β that scales
  both shapes to at most 12 in sampler and loss, mean preserved exactly (P40; the cap binds on the sampled support, not
  on the weights, so the raw heads can still read back higher). Seeds 75 and 77 from scratch, r93's flags otherwise,
  to the 50.14M save. Order amended the same day: 50.14M per seed (was 40.11M); each seed trains on GPU 0, its 50
  pre-registered fp64 months (5 checkpoints × 10 scenarios) then score on GPU 1; s77's training was pipelined by user
  order onto the idle 3090 while s75 scored on the 3060, and per user order each s77 checkpoint is scored as it lands
  (driver score-only mode; `waited`/`trainer_up` patch in `run_r94.py`). The first launch died at 02:36 (rc 137,
  externally killed); its data is kept unused as `results/runs/r94_conc_cap_s75_interrupted_20261031` and the arm
  relaunched from scratch. Gate: `study_r94_conc_cap/chain.log` GATE PASSED.
  **Result, s75, all 50 months scored (lower is better; training-4 = fairbanks_1, phoenix_6, denver_2, atlanta_6):**
  - Curve (ten-month mean): 10.03M 0.161 → 20.05M **0.100** → 30.08M 0.124 → 40.11M **0.101** (training-4 **0.057**)
    → 50.14M **0.338** (training-4 0.257). `final − best = +0.237` — the holding rule is not met.
  - Control r93-s75 over the same grid: 0.141 → 0.164 → 0.205 → 0.268 — monotone degradation; r85-s75 (40M) 0.215.
  - At 40M the training-4 mean beats `pid_water` (0.057 vs 0.091) — the first arm to do so — with both fuels down in
    the hot months (phoenix_6: elec 78 / gas 10 MWh vs r93's 108 / 189; comfort deficit 1.4 K vs 10.9 K). Held-out at
    40M is still worse than `pid_water` (0.101 vs 0.077).
  - Mechanism read from `conc.bin`: the raw heads re-concentrated with the failures — α+β ~39 at 30M (hot months
    degraded), ~7.5 at 40M (hot months solved and holding), ~34–40 at 50M (corner back). The cap keeps the sampler's
    support wide while the weights sharpen; that kept-hot support is the plausible carrier of 40M's hot-month solve.
  - The 50M corner, all six warm/hot months, four channels pinned together: static pressure 1.85–1.95 kPa (93–97% of
    range, vs 63–215 Pa in the good cold months), recirculation 0.96, hot water 347–348 K, run bits 100% on; fan
    electricity 100–360 MWh/month, overshoot counted as bad comfort (prod −0.43…−1.78). Cold months hold to the final
    save (fairbanks_1 0.079 is its monotone best, denver_2 0.023 held). Training-log last-10M: episode mean 0.072 →
    0.189 while total entropy fell −181 → −235.
  - **SB_CREDIT sign audit over 40–50M** (`study_r94_conc_cap/review/credit_audit.txt`): the corner is **credit-aligned,
    not a defect**. In the 45–50M window the hot groups' advantage is resolved POSITIVE exactly on the heads that
    walked into the corner (phoenix_6 recirc t +3.8 as its mean walked 0.22 → 0.40, atlanta_6 recirc t +4.0, water
    +2.5, sp +1.8/+2.0, pump +1.8/+1.4), the cold groups' credit points the other way and their heads stayed low, and
    the applied step followed the credit sign on 0.62–0.85 of epochs (no-clip descent 0.70–0.93) with mean agree 0.70
    over all 48 corner-window cells. The per-epoch training credit (desync-1 rollouts, per-step comfort-dominated)
    genuinely rewards the fuel-heavy commands that the full desync-0 month bills for — the walk is a local optimum of
    the estimator, not of the environment. Verdict for the next arm unchanged: **r94b = r94 + linear lr anneal over
    the real 50.14M** (noted in PRIORS §4/P40) — a stopping term, or budget-side change; NOT a credit or
    exploration-prior fix.
  - **Result, s77** (scored as its checkpoints landed, driver score-only mode; all 50 months scored, both GPUs idle at
    the end; lower is better): curve 10.03M 0.380 → 20.05M 0.174 → 30.08M 0.165 → 40.11M 0.208 (training-4 0.190,
    HX6 0.220; warm-month bump phoenix_6 0.396 / phoenix_5 0.385 / phoenix_4 0.345) → 50.14M **0.0926** (training-4
    **0.072**, HX6 0.106) — **the final save is its best point; all 10 months improved from 40.11M**. `final − best = 0`,
    but only one save sits at the level (same single-save shape as s75's best).
  - Control r93-s77: its own scoring leg was interrupted (10M and 20M complete, 30M 1/9, 40M 3/9, 50M never scored —
    left as-is). Clean points: 10M 0.2165, 20M 0.4167 with HX6 1.23 (hot-month lock: phoenix_5 2.79, phoenix_4 2.19,
    atlanta_5 1.69). r94-s77 ran 0.165–0.174 flat through 20–40M where the control locked — the cap beat the control
    on the second seed too.
  - Mechanism read from the s77 records (40.11M vs 50.14M; same decode both): at 40M the warm months carry water 330
    (phoenix_6) / 324 K (atlanta_6), SAT 295/290 K; at 50M water falls to 297/294 K and SAT to 281/278 K while the cold
    months raise water to 310 K and recirc to 0.59/0.60 — a per-weather re-balanced operating point. Contrast s75's 50M
    corner on the same decode: water 335/334 K, recirc 0.76–0.79. s77's 50M is NOT that corner.
  - No head pinned at the cap: at 50.14M median α+β per head is 9.3–12.8 with the water head touching the 12 ceiling on
    20–30% of rows (hot months); at 40M the sat/water heads did the same. Support stays open — the draw keeps moving,
    which matched the all-months-improved move into the final save.
  - Pre-registered tests (`PROTOCOL.md`), both seeds: (1) **lock signature at any save — PASS** (the only flagged cell,
    s75 20M `atlanta_6`, run-on 2.9%, is a 0.048 near-perfect month — productivity −0.033, elec 16.9 MWh; worst single
    month over all 100 scored is 0.485). (2) **recovery — PASS on s77** (the regressed 40.11M save fully moved back at
    50.14M). (3) **score — PASS/conditional**: both seeds beat `pid_water` on training-4 once (0.057 / 0.072 vs
    0.0911); s77's HX6 0.106 remains worse than `pid_water`'s 0.0773, and held-out generalization was never the
    measured claim of this arm.
  - **Consolidated two-seed verdict:** the concentration cap removed the r91/r93 lock class on both seeds and each
    reached training-4 below `pid_water` — it is the arm that got there. **Holding is still unmet on both seeds**
    (s75: best 40.11M 0.101, then 0.338; s77: only the final save holds 0.0926), and the seeds disagree on WHICH save
    is best (s75 peaks at 40.11M then collapses; s77 bumps at 40.11M then peaks at 50.14M). That shape — a good point
    reached, then walked away from, with credit agreeing with the walk both ways — is what a stopping term addresses:
    **r94b = r94 + linear lr anneal over the real 50.14M** (PRIORS §4 item 8), not another credit or exploration
    prior. Not approved yet.
- **r94b linear LR anneal** (approved and launched 2026-11-01, `complete_port/results/study_r94b_lr_anneal/`,
  PRIORS §4 item 8 / P40). r94's recipe unchanged + the stopping term: `--train.anneal_lr=1
--train.linear_lr_anneal=1 --train.anneal_steps=50135040` — linear 1e-4 → 0 (ini `min_lr_ratio`=0) over the
  real 50.14M; `total_timesteps` untouched (the r94 guard stands). Source arm: `train.linear_lr_anneal` +
  `train.anneal_steps` added to `patches/pufferlib_core.diff` (default-off; with the flags off the branch is
  never entered, so the binary is behaviourally r94's), `verify_all.sh --trainer` gate on
  `results/verify_r94b_chain.out`. User order (amended 2026-11-01): seeds 75/76 train+score on GPU 0 (3090), seed 77 train+score on GPU 1
  (3060; that process waits for the r94-s76 arm's containers to clear — the 12 GB card cannot hold
  trainer + eval). Each seed trains to its 50.14M save first, then its 50 fp64 months score on the same
  GPU, one at a time. Pre-registered tests (`PROTOCOL.md`): same lock/recovery/score grid as r94 plus the holding rule
  (`final − best`, ≥ 2 consecutive saves at level) — the arm's target is the holding gap r94 left open.
  The on-binary proof of shape and budget is the epoch-0 stderr line `SB_LINEAR_LR_ANNEAL`.
- **r94 third seed (76)** (approved and launched 2026-11-01, same study folder per user order,
  `study_r94_conc_cap/run_r94_s76.py`). r94's recipe unchanged, seed 76 from scratch, trained on GPU 1 (3060)
  with that seed's 50 months scored as they land on the same GPU; fresh runtime dirs `bin_s76_train`/
  `bin_s76_score` (the study's r94-era `bin_train`/`bin_score` stay frozen as the record of what r94 ran on).
  Pre-registered read: r94's three tests + holding rule; the point is whether the single-save shape r94-s75/s77
  showed is seed-robust.

- **r95 Bernoulli run heads** (implemented and launched 2026-10-31; PRIORS P41). From `review_r91_r93`: 39 of
  43 months above 0.3 after 10M had a run switch below the 0.25 decode threshold for most occupied hours.
  `policy.binary_run=1` makes the 9 AHU run heads and the hot-water run Bernoulli, p = σ(u + c0), c0 = 1.5236
  (p = 0.821 at u = 0, the Beta head's initial on-rate); coherent draw on iff Φ(z1) < p. check_ppo_math 1917/0
  failing. Launched on Modal 2026-10-31 09:12 IST (approved): `binary_run=1`, 25 scenarios (all five weathers of
  buildings 1, 2, 4, 5, 8; `train25_holdout_b7`), 800 agents, seed 75, 576M steps, 10 saves with Muon sidecars,
  L40S at ~13–14K SPS. Gate `study_r95_binary_run/gate_log.txt` ALL CHECKS PASSED. Benchmarks in `PROTOCOL.md`.

### r95 result (recovered from the on-disk scorer artifacts 2026-11-05; `study_r95_binary_run/ANALYSIS.md`)

- **The run is truncated**: `live/epochs.tsv` stops at 360 epochs (~83M of 576M steps); only **save 1
  (57,600,000)** was scored — 15 months with full recordings on 2026-10-31, never written up. Saves 2–10 were
  never scored (they would be on the Modal volume). **r96, the protocol's control, never launched** (the second
  Modal account had no payment method).
- **Held-out correction**: `train25_holdout_b7` contains the six gen6-named scenarios (train_1/phoenix_5,
  train_8/phoenix_4, train_4/atlanta_5, train_5/denver_3, train_8/denver_6, train_2/fairbanks_2) — they
  **trained**; PROTOCOL.md's "plus the existing gen6 held-out set" is wrong as written. The only held-out
  scenarios are building_train_7's five weathers.
- **Scores at save 1, vs pid_water** (REFERENCE.md): in-batch 10 months **0.1117 vs 0.0828** (worse +0.029,
  better 4/10); **held-out building_train_7: 0.0246 vs 0.0427 (better 5/5, −42 %)**; all 15: 0.0827 vs 0.0694
  (worse). The losses are comfort-heavy in the extreme months (productivity −0.23…−0.39 on fairbanks_1/
  fairbanks_2/phoenix_5/phoenix_6/atlanta_6); the wins include phoenix_4 at −0.130 vs pid_water.
- **Mechanism confirmed behaviorally**: the run switches hold commanded states that differentiate by climate
  (hot scenarios: fan subsets + hot-water off, water p ≈ 0.01; cold: more fans + water on, p ≈ 0.73–0.80) and
  genuinely toggle where toggling pays (302–401 transitions/month in fairbanks_1/fairbanks_2/phoenix_6/
  atlanta_6; on-fractions 0.05–0.37). AHU recirculation holds high (mean 0.85), aligned with the rc70
  counterfactual, and empty-period run fraction exceeds occupied-period in fairbanks_2/phoenix_6 (pre-heating).
- **Verdict: inconclusive-positive on mechanism, no attribution to the head; not adopted.** One save, one seed,
  10 % of budget, no control, and the 25-scenario broad batch alone could explain the held-out-building win
  (r25→r26 precedent). The overall mean at save 1 is worse than pid_water. The clean single-factor test remains
  `r94b recipe + binary_run` vs `r94b` on `r85_four` — the original local proposal the Modal plan superseded.
  Follow-up: pull saves 2–10 from the Modal volume if it still exists and score them. This entry also corrects
  the record: earlier lines said "NOT approved or run" — r95 ran.
- **Behavior verdict vs the Beta head (2026-11-05, `study_r95_binary_run/BERNOULLI_VS_BETA.md`)**: across 79
  recorded months (r94b 50, r116 4, r95 15) the decode-threshold pathology the Bernoulli targets — switch
  means inside ±0.05 of the threshold — appears in **0 months** of any run (≤2.5 % of occupied head-steps);
  both parameterizations commit per-AHU states and adapt them to climate. The Bernoulli's one score-relevant
  behavioral difference: at 10 % budget it under-fans the cold training month (fairbanks_1 fans ~11–25 % on vs
  ~40 %, gas 114 MWh without delivery, score 0.2775) — while committing fans fully in the held-out cold month
  (0.041). The Bernoulli head works mechanically; the defect it targets is absent from the r94b lineage's
  frozen months; not adopted, flag kept.

## r96 — SPS, the tile, and the tile's learning-rate compensation (2026-11-01)

User order: stop r94b on both cards, raise the SPS without dropping a log line, replace the batch's largest building,
check the SPS and check scoring. No power-limit change (the 300 W / 140 W caps are user-set).

### What was measured, and what it found

Timing runs, r94b's flags and all diagnostics, only the batch varied (`results/study_r96_speed/` — `measure.py`,
`sps.tsv`). "SPS" is `agents*horizon / (Evaluate + Train)` on the last full board; PufferLib's own `SPS` field is
wrong in every regime. Every number below was run on this box.

| setting                                |  agents | solve    | minibatch |   rollout |   update |                               SPS 3090 |
| -------------------------------------- | ------: | -------- | --------: | --------: | -------: | -------------------------------------: |
| r85_four 4x64 — the r94b shape         |     256 | fp64     |      2304 |    9.95 s |   2.94 s |                                  5,720 |
| r96_light_four 4x128                   |     512 | fp64     |      2304 |     11.08 |     4.32 |                                  9,574 |
| r96_light_four 4x160                   |     640 | fp64     |      2304 |     14.05 |     5.49 |                                  9,432 |
| r96_light_four 4x128                   |     512 | fp32     |      2304 |      8.88 |     4.33 |                                 11,157 |
| r96_light_four 4x128                   |     512 | fp32     |      9216 |      8.77 |     2.41 |                                 13,187 |
| **r96_light_four 4x160 — the default** | **640** | **fp32** |  **9216** | **10.05** | **3.07** |                             **14,044** |
| r96_light_four 4x192                   |     768 | fp32     |      9216 |     12.03 |     3.65 |                                 14,109 |
| r96_light_four 4x192                   |     768 | fp32     |     18432 |         — |        — | `cudaGraphInstantiate failed` (>24 GB) |
| 3060, r96_light_four 4x64              |     256 | fp32     |      9216 |     10.01 |     2.62 |                                  5,837 |

1. **Layout saturates at 4 envs per lane.** 64 -> 128 per group is 1.67x; 128 -> 160 is 9,574 -> 9,432 (nothing);
   160 -> 192 is <1% of the rollout. With four groups concurrent, 512 columns already hide the solve's DAG latency.
2. **The recurrent update was starved by its tile.** `rows = minibatch_size / horizon` was 2304/288 = 8, so the
   288-step scan ran 8 sequences wide. 9216 gives 32 rows and 1.8x on the update for the same samples and the same
   `replay_ratio`.
3. **The diagnostics cost 1.17 s/epoch** (`SB_CREDIT` + `SB_PPD` + `SB_ROLESTEP` + `update_diagnostics=1`):
   2.41 -> 1.24 s off. Kept on the order; the price is now measured.
4. **fp32 training solve is 1.25x on the rollout** (11.08 -> 8.88 s) and halves the solve state, which is what
   buys the wider batch in VRAM.
5. **640 agents is the 24 GB ceiling** (19.9 GB); 768 needs 22.8 GB and dies at minibatch 18432.

### The batch

`building_train_2__phoenix_6` was the largest building in the batch (14.84M solve terms, 126.8 us of state per
agent-step, 1.5x the next largest). It is replaced by `building_train_7__phoenix_3` (4.03M terms): same climate in
the slot, train_7 is not in the held-out eval list, and the set's mean state per agent-step falls 77.6 -> 56.0 MB
(-28%). New set `artifacts/sets/r96_light_four` = train_1**fairbanks_1, train_4**denver_2, train_5**atlanta_6,
train_7**phoenix_3.

### The collapse the tile caused, and the fix on trial

The first r96 arm (`study_r96_light/run_r96.py`) ran the fast tile with r94b's `learning_rate=1e-4` and was stopped
at 25.07M steps on the user's order. Nothing blew up — no NaN, `nonfinite_ratio_frac` 0, clipfrac 0, logits inside
the squash — the optimiser was starved: `steps/sample = replay_ratio/minibatch` fell 1.74e-3 -> 4.34e-4, i.e. 4x
fewer parameter updates for the same experience.

| at ~19-25M steps                      |                                    r96 (mb 9216, lr 1e-4) |                                   r94b (mb 2304, lr 1e-4) |
| ------------------------------------- | --------------------------------------------------------: | --------------------------------------------------------: |
| median `loss/kl`                      |                                     **0.0** (mean 2.1e-3) |                                                   1.1e-02 |
| median `loss/value`                   |                                **3,521** (never descends) |                                                      47.7 |
| score panel                           |                1.40 @6.4M -> 1.81 @10.5M -> **2.22 @25M** |                     2.32 @5M -> 0.26 @11.5M -> 0.287 @20M |
| `energy_cost_norm` / elec             |                        0.115 -> **0.200**, 34k -> **68k** |                                0.128 -> 0.067, 45k -> 35k |
| fp64, 3 shared months @ matched steps | **0.5896** (prod -0.8515, en 0.1593, carb 0.2342) @18.80M | **0.3254** (prod -0.5140, en 0.0319, carb 0.0529) @20.05M |

The invariant that moves the parameters is `steps/sample x lr`. Both candidate arms keep it equal to r94b's, so
the tile can stay wide:

| arm (r96 category)          | GPU  | agents | `minibatch_size` | `learning_rate` | steps/sample x lr |        SPS measured |
| --------------------------- | ---- | -----: | ---------------: | --------------: | ----------------: | ------------------: |
| **r96a_ctrl_tile2304**      | 3090 |    640 |             2304 |            1e-4 |           1.74e-7 |          **11,594** |
| **r96b_fix_tile9216_lr4e4** | 3090 |    640 |             9216 |            4e-4 |           1.74e-7 | (sequenced after a) |
| **r96c_g1_tile2304**        | 3060 |    256 |             2304 |            1e-4 |           1.74e-7 |           **5,465** |

Both 3090 arms run the same seed (76) on the same card, so the only differences are the tile and its LR. Budget
19,906,560 steps = 108 epochs at 640 agents = r94b's 5,013,504-step save grid; the 3060 arm uses seed 77 and
r94b's own 68-epoch grid (270 epochs). From scratch, every r94b flag kept (Beta head, `beta_cn_tau` 12/48,
`beta_conc_max` 12, `role_adam`, zone critic, value delay/shared, `adv_norm=2`, `vf_clip_coef` 1e4, gamma 0.99,
lambda 0.95, the linear LR anneal to 50.14M), `update_diagnostics=1`, TensorBoard on, `SB_CREDIT`/`SB_ROLESTEP`/
`SB_PPD` on, fp32 training solve, fp64 scoring (0 `EEv5LuDevPf` in `bin_score`).

Pre-registered decision: score the 19.9M save of each arm fp64 on the three training months r94b also trained on
(train_1**fairbanks_1, train_4**denver_2, train_5\_\_atlanta_6) and compare with **r94b@20.05M = 0.3254**. Adopt the
fast tile if the fix arm is within 10% of the control; otherwise the tile stays 2304. Safety: the fix arm is
stopped if the first five epochs show `old_kl > 0.1` or `clipfrac > 0.2` (there is no LR warmup knob in this
harness, and at 4e-4 the first update is the only risk). Driver `results/study_r96_light/run_r96_tile.py`.

### The two runs, and what changed from the previous arm

Protocol: `complete_port/results/study_r96_light/PROTOCOL_tile.md`. Driver `run_r96_tile.py`.

| run                         | GPU  | agents | `minibatch_size` | `learning_rate` | steps/sample x lr | measured SPS |
| --------------------------- | ---- | -----: | ---------------: | --------------: | ----------------: | -----------: |
| **r96a_ctrl_tile2304**      | 3090 |    640 |             2304 |            1e-4 |           1.74e-7 |   **11,594** |
| **r96b_fix_tile9216_lr4e4** | 3090 |    640 |             9216 |        **4e-4** |           1.74e-7 |       14,044 |
| r96c_g1_tile2304            | 3060 |    256 |             2304 |            1e-4 |           1.74e-7 |    **5,465** |

**r96a is the control:** r94b's optimiser (mb 2304, lr 1e-4, replay 4) on the new batch and the new solve. If the tile
is the only cause of the collapse, r96a has to reproduce r94b's curve. **r96b is the treatment:** the wide tile's
1.18x speed, with the 4x-in-fewer-steps deficit paid for by 4x the learning rate instead of 4x the samples per step,
so `steps/sample x lr` is identical to r94b's and to r96a's. The two run **on the same card with the same seed (76)**,
one after the other, so the tile and its LR are the only differences between them; that is why they are sequenced
rather than run in parallel. **r96c** is the 3060's arm: 12 GB cannot hold 640 agents, so it keeps 4x64 and the
control tile, i.e. a configuration whose learning is known good, on r94b's own 68-epoch grid.

Against **r94b**, four things changed, each measured on its own (`study_r96_speed/sps.tsv`):

| change                                                            | why                                                                                    | measured                                             |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| batch: `train_2__phoenix_6` -> `train_7__phoenix_3`               | train_2 was the largest building in the batch (14.84M solve terms, 1.5x the next)      | set mean state per agent-step 77.6 -> 56.0 MB (-28%) |
| layout 4x64 -> **4x160** (2 -> 5 envs/lane)                       | below 4 envs/lane the solve is DAG-latency bound                                       | 5,720 -> 11,594 SPS                                  |
| fp64 -> **fp32 training solve** (`SB_F32=1` + `SB_SHARED_POLICY`) | 1.25x on the rollout and half the solve state, which is what makes the wider batch fit | rollout 11.08 -> 8.88 s                              |
| `minibatch_size` 2304 -> **9216** (lr scaled in r96b)             | the recurrent scan ran 8 sequences wide                                                | update 4.33 -> 2.41 s (1.8x)                         |

Against the **stopped r96 arm**, r96b changes exactly one thing: `learning_rate` 1e-4 -> 4e-4. r96a changes only the
tile back. Same blob set, same 640 agents, same fp32 train / fp64 score builds, same seed, same save grid, same
diagnostics, same LR-anneal schedule.

Fixed in both 3090 arms: 19,906,560 steps = 108 epochs = r94b's 5,013,504-step save grid; from scratch; every r94b
flag (Beta head, `beta_cn_tau` 12/48, `beta_conc_max` 12, `role_adam`, zone critic, `valve_zone`, value delay/shared,
`adv_norm=2`, `vf_clip_coef` 1e4, gamma 0.99, lambda 0.95, `anneal_lr=1 linear_lr_anneal=1 anneal_steps=50135040`);
`update_diagnostics=1`, TensorBoard, `SB_CREDIT`/`SB_ROLESTEP`/`SB_PPD`; fp32 train, fp64 score
(0 `EEv5LuDevPf`). Scoring: every save, deterministic fp64, recorded, on the three months r94b also trained on.
Reference: **r94b s76 @ 20,054,016 = 0.3254**; the stopped arm at 18.80M was 0.5896.

Decision: adopt the wide tile if r96b is within 10% of r96a and both are within 10% of 0.3254; mechanism check is
median `loss/kl` back near 1e-2 (r96 read 0.0) and `loss/value` under ~200 by 20M (r96 read 3,521). r96b is stopped
if the first five epochs show `old_kl > 0.1` or `clipfrac > 0.2` — there is no LR warmup knob here, and 4e-4 is the
first update's only risk. If r96b loses, the tile stays 2304 and the speed win is 11,594 SPS (1.95x the r94b shape);
the middle point (mb 4608 + lr 2e-4, same product, ~13,200 SPS) becomes the next arm rather than a mid-run change.

### Status — tile A/B stopped, replaced by the split arms (2026-11-01)

**The tile A/B was stopped on the user's order** before r96b produced a board. Kept: `r96a_ctrl_tile2304` completed
its full 19,906,560 steps at 11,081 SPS wall with four saves (4.98 / 9.95 / 14.93 / 19.91M) — that is a usable
control checkpoint set, and if it is ever scored the reference is `r94b_lr_s75`; `r96c_g1_tile2304` at 10.03M with
two saves; `r96b_fix_tile9216_lr4e4` six minutes in, no save. Nothing deleted.

**Why it was replaced:** it answers "is the wide tile usable", but three things changed at once against r94b — the
batch, the solve, the tile — so an underperformance could not be attributed. The split arms settle that by
elimination first.

### The split arms (running)

Protocol `PROTOCOL_tile.md` Part 2, driver `run_r96_split.py`. Both are **r94b exactly** (256 agents, 4x64,
`minibatch_size` 2304, `learning_rate` 1e-4, replay 4, the linear LR anneal, every r94b flag and diagnostic) except
for **one factor each**, and each is matched to an existing r94b arm on the **same card and the same seed**:

| arm   | run                    | GPU  | batch              | solve    | seed | control       | the one change                      |
| ----- | ---------------------- | ---- | ------------------ | -------- | ---: | ------------- | ----------------------------------- |
| **F** | `r96f_fp32_check_s75`  | 3090 | r85_four           | **fp32** |   75 | `r94b_lr_s75` | the fp32 training solve             |
| **B** | `r96b_batch_check_s77` | 3060 | **r96_light_four** | fp64     |   77 | `r94b_lr_s77` | the batch swap (train_2 -> train_7) |

12,533,760 steps each (170 epochs), 2,506,752-step saves, from scratch, in parallel. **No fp64 months: the read is
the live curve**, because every r94b seed has separated by 10-12.5M:

| control       | 5.0M (value / scen) | 10.0M                                          | 12.5M                                          |
| ------------- | ------------------- | ---------------------------------------------- | ---------------------------------------------- |
| `r94b_lr_s75` | 4,810 / 2.42        | **56.6 / 0.753** (prod -0.806, comfort 5.11 K) | **185 / 0.355** (prod -0.676, comfort 6.08 K)  |
| `r94b_lr_s77` | 4,186 / 2.44        | **43.6 / 1.174** (prod -1.263, comfort 6.46 K) | **86.4 / 0.386** (prod -0.748, comfort 6.32 K) |

A stuck run reads value loss 1,400-6,700 and a score panel of 1.4-2.2 at those steps, which is what the stopped r96
arm did between 10M and 25M. The driver prints each arm beside its control at 5.0 / 10.0 / 12.5M and marks `OK` or
`STUCK` on the user's thresholds (value loss < 1,000, score panel < 4).

**What follows:** Arm F stuck -> fp32 is the cause; it goes off the table and the fp64 ceiling for this layout is
**9,432 SPS** at 4x160/2304 against 11,594 on fp32 (a wider minibatch on fp64 is unmeasured). Arm B stuck -> the batch
swap is the cause; fp32 stays and the slot gets a different building than train_7. **Both learn** -> both changes are
innocent, the tile is the only suspect, and the minibatch test runs next on the setup that reproduces r94b: **9216 at
lr 4e-4**, plus **2e-4 as a middle point**, so 9216 can become the default on a control that actually learns.

### Split-arm results (2026-11-01)

| arm                          |      steps |       SPS | 12.02M value | 12.02M panel | control value | control panel | verdict    |
| ---------------------------- | ---------: | --------: | -----------: | -----------: | ------------: | ------------: | ---------- |
| **F** `r96f_fp32_check_s75`  | 12,533,760 | **7,090** |      **3.2** |    **0.382** |         237.9 |         0.305 | **LEARNS** |
| **B** `r96b_batch_check_s77` | 12,533,760 |     3,977 |    **2,262** |    **1.589** |         140.9 |         0.362 | **STUCK**  |

**Arm F — fp32 is not an issue.** The first four value boards are identical to the third decimal (0.85 / 17.4 / 246.8
/ 1,669-1,672), then a 2e-3 relative rounding difference amplifies through the closed loop. The critic lags about 3M
steps and arrives (value 3.2 at 12.02M, better than the control's 237.9), the policy lands at 0.382 against 0.305, the
gas descends in lockstep, and fp32 is 21% faster (7,090 vs 5,882 SPS). Two consistent offsets: entropy −119 against
−176 and `old_kl` 0.065 against 0.014, i.e. the worse-fitted critic makes larger advantage-driven steps. Recorded as
a _perturbation of a chaotic loop_, not a systematic breakeven.

**Arm B — the batch swap IS the cause, and it is not the building.** Removing `train_2__phoenix_6` and adding
`train_7__phoenix_3` leaves the critic unable to fit: value loss 2,496 (8.0M) → 1,857 (10.0M) → **2,262** (12.0M)
against the control's 2,152 → 43.6 → 140.9, and the panel stalls at 1.52-1.68 against 0.36-0.49. So the r96 collapse
has two independent causes and both are now measured: the tile (via `steps/sample x lr`) and the batch.

**Why the batch, in one table** (per-scenario, last TensorBoard value at 12.5M):

| building                                | Arm B comfort / reward | control comfort / reward |
| --------------------------------------- | ---------------------- | ------------------------ |
| train_1\_\_fairbanks_1                  | **26.8 K / −5.963**    | 12.3 K / −1.584          |
| train_4\_\_denver_2                     | 2.90 K / −0.245        | 1.26 K / −0.097          |
| train_5\_\_atlanta_6                    | 4.71 K / −0.357        | 1.95 K / −0.120          |
| train_7**phoenix_3 / train_2**phoenix_6 | **4.03 K / −0.294**    | 4.60 K / −0.170          |

`train_7__phoenix_3` is **not** the problem: its own comfort and per-step cost sit next to the building it replaced.
What broke is `building_train_1__fairbanks_1`, which fell into the deep cold basin (26.8 K against 12.3 K) at a
per-step cost of −5.96 — 24x the next building in the same batch. With **one shared value head over four climates**
(`value_shared=1`) the critic's squared error is dominated by that tail, so the value loss cannot descend (1,800-2,300
against 33-141) and every building's advantage inherits the mis-fit. Worse critic → worse policy → worse comfort →
bigger cost → harder critic: a positive feedback into the basin.

### Provenance: whose buildings are these

- All four buildings in this batch — train_1, train_4, train_5, train_7 — are **organizer-shipped**. `starter-kit/
smart_buildings/buildings/` contains exactly the six shipped buildings (train_1, 2, 4, 5, 7, 8) and nothing else;
  every generated building this project ever made (building_0500, 10001, 30008, 20001, ...) was produced at run time by
  `run_floor_generator.py` and is not in the repo.
- All four weathers are organizer stations, and `building_train_7__phoenix_3` is literally one of the 30 scenarios in
  `starter-kit/smart_buildings/training_set.yaml` (train_7 is paired with atlanta_8, denver_5, fairbanks_7,
  **phoenix_3**, singapore_6). The swap exchanged one official scenario for another official scenario.
- So "the building is defective" is **not** the explanation, and nothing here is ours to blame.

### Where the score actually stands, per weather

`r94b_lr_s75` at its best save (40.11M), all ten scored scenarios, split into the three terms:

| scenario               |      score |      comfort |       energy |       carbon |
| ---------------------- | ---------: | -----------: | -----------: | -----------: |
| train_1\_\_fairbanks_1 | **0.1022** |       0.0386 |       0.0184 |       0.0452 |
| train_2\_\_fairbanks_2 |     0.0715 |       0.0499 |       0.0061 |       0.0155 |
| train_5\_\_denver_3    |     0.0309 |       0.0246 |       0.0024 |       0.0039 |
| train_1\_\_phoenix_5   |     0.0261 |       0.0059 |       0.0163 |       0.0038 |
| train_4\_\_denver_2    |     0.0256 |       0.0087 |       0.0057 |       0.0111 |
| train_8\_\_phoenix_4   |     0.0231 |       0.0075 |       0.0125 |       0.0030 |
| train_2\_\_phoenix_6   |     0.0196 |       0.0087 |       0.0088 |       0.0021 |
| train_8\_\_denver_6    |     0.0171 |       0.0016 |       0.0095 |       0.0060 |
| train_4\_\_atlanta_5   |     0.0138 |       0.0041 |       0.0076 |       0.0020 |
| train_5\_\_atlanta_6   | **0.0102** |       0.0034 |       0.0053 |       0.0015 |
| **mean**               | **0.0340** | 0.0153 (45%) | 0.0093 (27%) | 0.0094 (28%) |

Three readings, and none of them is "it fails everywhere":

1. **It is scored in every weather.** The range is 0.0102 to 0.1022, a factor of 10, with the warm scenarios
   (atlanta, phoenix) at 0.010-0.026 and the cold ones (both Fairbanks) at 0.07-0.10. The shape is "cold weather is
   5-10x worse", not "nothing works".
2. **No weather reaches the target.** The best single scenario, 0.0102, is still 1.7x the 0.00598 target, and the mean
   is 5.7x it. So the gap is not concentrated in one scenario that could be fixed alone.
3. **No single term carries the gap.** Comfort is 45% of the score, energy 27%, carbon 28% — all three are the same
   order. The target is 0 because it needs comfort, energy _and_ carbon near zero at once; we are at roughly a third of
   0.034 in each, which is why fixing any one term alone cannot reach it.

## r97 — the actor may not read its own command (approved and started 2026-11-01)

**Why.** The stored fp64 months of `r94b_lr_s75` were audited from their recordings (`results/audit_r94b_20261101/REPORT.md`):
the zone reheat valve never closes (0.933 cold / 0.935 in-band / **0.935 hot** / 0.833 night on Fairbanks, one value
across 39–85 zones, spread 0.005), the plant sits at the bottom water rail all day (311.3–312.0 K), and the cause is in
the weights: the observation reports each head's own last command **bit-exactly** (`obs reheat(t+1) == loc.bin(t)`, max
error 0.0 over a month), so the row's largest weight is on that column (zone reheat +3.545 vs −0.100 on `T−Hs`;
damper +3.031; AHU SAT setpoint +1.818; AHU run +1.820; plant water +1.750; plant dp +1.033; plant run +1.250). Under
`beta_cn_tau` the AR(1) draw makes `a_{t-1}` a proxy for the current exploration sample, so PPO profits from copying it.
Decisive control from disk: `r79_beta_s75` (same layout, **per-step** draws) has **+0.009** on that column and the
**correct** temperature sign (−0.363).

**Change (one flag).** `[policy] no_cmd_echo` = 1 / 2 (`patches/src/sb_shared_policy.cuh`, `sbs_local_feature`; reader in
`pufferl.cu` so train and eval build the same policy). Level 1 zeroes the two zone valve command columns; level 2 adds the
AHU SAT-setpoint / recirculation / run columns and the plant water-setpoint / dp-setpoint / run-status columns. Every
measured temperature, other devices' setpoints, static pressure, air and water flows, boiler temperature, the sibling
block, the memory features and the whole pooled trunk stay. No new parameters. Same patch, diagnostics only:
`sbs_beta_params` now applies `d_beta_conc_max`, so SB_PPD and `sbs_diag_*` report the capped shapes the sampler and the
loss actually use. Prior: P42.

**Arms** — fp64, `r85_four` 4x64, seed 75, from scratch, 80,216,064 steps (1088 epochs, 16 saves), linear LR 1e-4 → 0 over
the real budget, every other r94b flag unchanged:

| arm | run | GPU | level |
| --- | --- | --- | --- |
| Z | `r97_noecho_zone_s75` | 0 (3090) | 1 |
| A | `r97_noecho_all_s75` | 1 (3060) | 2 |

**Gate** `results/verify_20261101_232028.log` (stream `results/verify_r97_no_echo.out`): **ALL CHECKS PASSED**, binary
sha256 `35c805d9…`. Pre-launch proof the mask bites: on the r94b 40.11M checkpoint a 64-step episode at levels 0/1/2 gives
different commands (zone reheat 0.035 mean / 0.150 max apart, plant water 0.061), so eval honours the flag.

**Started** 2026-11-01 18:09 UTC, both cards at 100%: zone 6.0K SPS (≈3 h 40 m), all 3.2K SPS (≈7 h). Both print
`SB_LINEAR_LR_ANNEAL base 0.0001 -> 0 over 1088 epochs (80216064 steps)` and their `no command echo N` banner.
Pre-registered tests and the 8-save scoring grid (10.03 … 80.22M × 10 scenarios, recorded): `study_r97_no_echo/PROTOCOL.md`.

## r98 — per-zone value alone (verdict 2026-11-02, run started 06:03 UTC)

`r98_zone_value_s75`, GPU 0, 50.14M, 10 saves, seed 75, from scratch. One change against the control
`r94b_lr_s75`: `train.zone_value=1` (fix 1 of the r94b audit's four). fp64 months recorded under
`results/study_r98_r99/scores/r98/`. Control months: `results/study_r94b_lr_anneal/scores/s75/`.

**Verdict: FAILS its pre-registered tests. Not better than the control at any comparable save, and not held.**
Comparison on the 4 training scenarios (the only panels both runs have at every save):

| save | r98 mean (prod / energy / carbon) | r94b mean (prod / energy / carbon) |
| ---- | --------------------------------- | ---------------------------------- |
| 10M  | **0.9898** (−1.63 / 0.028 / 0.035) | 0.2101 (−0.28 / 0.100 / 0.116) |
| 20M  | 0.0977 (−0.110 / 0.081 / 0.078)    | 0.0979 (−0.117 / 0.064 / 0.074) |
| 30M  | 0.0676 (−0.076 / 0.055 / 0.055)    | 0.0635 (−0.059 / 0.050 / 0.091) |
| 40M  | **0.1296** (−0.136 / 0.117 / 0.124) — regression | 0.0394 (−0.025 / 0.048 / 0.075) |
| 50M  | 0.0726 (−0.085 / 0.052 / 0.057)    | 0.0376 (−0.022 / 0.048 / 0.076) |

- Test 2 (score ≤ 0.0340 held two saves): fail — best 0.0676 at 30M, 0.1296 at 40M, 0.0726 at 50M.
- Test 1 (mechanism: hot-minus-cold reheat differential ≥ 0.1): fail — 0.009–0.02 at every save; the reheat
  latch is still its own echo (R2(lag) 0.986–0.997, coef on zone T −0.006…−0.021) — `verify_r98_*.json`.
- Guard (value loss ≤ control): fail — late dashboard value loss ~1.0–1.5 vs control ~0.30; `grad_value_row`
  still 10–15 at 50M vs control 2–3.

**What did move (keep).** The per-zone pathway opened real state-dependent behavior the control never had:
fairbanks_1 reheat at night 0.83 → 0.13 and night gas share 33% → 6% (30M); corr(T, damper) across zones
+0.88 → −0.82 (dampers now read the zone); latch R2(T) 0.000 → 0.15–0.38; water command unpinned at night
(cmd 0.64, hWS 342–345 K). On fairbanks_1 at 50M this cut energy 0.099 → 0.059 and carbon 0.239 → 0.135 —
but the same months paid more in productivity (−0.060 → −0.106), so the score is unchanged: it bought gas
with comfort. Nothing in the per-zone credit prices the morning comfort cost of a night shut-off.

**Why it failed.** (i) The per-zone reward share is still building-common while occupied, so the targeted
hot→close differential never got a per-zone teacher. (ii) 90 fresh per-zone value rows on one on-policy
sample each never fit (value loss 3–4× control through 30–50M); with `valve_zone=1` that noise injects
straight into the valve gradient — actor-head gradients 53–73 late vs control 29–36. (iii) The mis-fit
critic could not price the plant-wide cool-and-reheat mode the policy fell into at 35–45M (both_frac 0.54–0.63,
elec 90–103 MWh on warm scenarios), and entropy had collapsed to −258 (control −159) by 25M, so it could not
leave; the LR anneal to 0 pulled it partly back. (iv) A fresh random V_z is a noise baseline: the 10M panel
(0.99 vs 0.21) and ~10M of lost ground are the cost.

**r99 (`r99_all_fixes_s75`, bundle of all four fixes) mid-run read at ~43M:** same value-loss signature
(1.3–2.5), comfort worse than r98 at the same stage (prod −0.25 vs −0.13, out-of-band 0.63–0.73), entropy
−165. Not on track to beat either arm; its recorded months land ~2026-11-02 20:30 IST and decide the bundle.

**Next (proposed, needs approval):** attack the measured failure — the noisy per-zone value baseline — with
one init-only change: zero-init the zone-value row weights (bias learnable), so V_z starts as a per-zone
constant instead of random offsets; keep everything else at the r94b recipe. If the value loss still ends
≥ 2× control, the follow-up is per-slot advantage scaling (standard estimator mechanics, contract-allowed).
Pre-register the mechanism gate as the p2 counterfactual plus the fairbanks night-gas-share guard, since the
night shut-off is where this pathway pays first.

## r102–r104 — seeds, precision, batch and budget on the control recipe (five train-only arms, started 2026-11-02)

User order 2026-11-02: five runs of the control recipe, **no scoring for any of them** (`env/score` and the
TensorBoard losses are the read), **no gate** (tree unchanged since the 10:49 UTC pass; `bin_f64` is
byte-identical to that gate's binary, sha `f32a1753…`; `bin_f32` is the same source with `SB_SOLVE_F32`).
Two independent tracks — the 3060 track never waits for the 3090 track, and vice versa.
Driver `complete_port/results/study_r102_seeds/run_seeds.py`, tracks `chain_gpu0.sh` / `chain_gpu1.sh`,
log `results/study_r102_seeds/chain.log`.

Every arm is `r94b_lr_s75`'s flag file with the stated change only: fp64 unless noted, Beta head
(`beta_cn_tau` 12 / `beta_cn_tau_ahu` 48, `beta_conc_max` 12), `role_adam`, `zone_critic`, `valve_zone`,
`value_shared` + `value_delay`, `adv_norm=2`, `vf_clip_coef=1e4`, `replay_ratio` 4, 256 agents (4 groups × 64),
`minibatch` 2304, horizon 288, `desync=1`, `reward_scale=1`, `cudagraphs=1`, saves every 68 epochs (5.0135M).
No `zone_value`, no `binary_*`, no `no_cmd_echo`, no `value_adam` — this is the control, not the r98 stack.

| arm | run dir | GPU | solve | batch | steps | lr anneal | start (IST) | projected end (IST) |
| --- | ------- | --- | ----- | ----- | ----- | --------- | ----------- | ------------------- |
| 1 | `r94b_lr_s76` | 3090 | fp64 | `r85_four` | 50,135,040 | 50,135,040 | 2026-11-02 19:06 | 2026-11-02 ~21:30 |
| 2 | `r94b_lr_s77` | 3090 | fp64 | `r85_four` | 50,135,040 | 50,135,040 | after arm 1 | 2026-11-02 ~23:55 |
| 3 | `r102_fp32_s75` | 3060 | fp32 | `r85_four` | 50,135,040 | 50,135,040 | 2026-11-02 19:12 | 2026-11-02 ~22:15 |
| 4 | `r103_fp64_light_s75` | 3060 | fp64 | `r96_light_four` | 50,135,040 | 50,135,040 | 2026-11-02 22:31 | 2026-11-03 ~02:00 |
| 5 | `r104_fp64_80m_s75` | 3090 | fp64 | `r85_four` | 80,216,064 | **80,216,064** | after arm 2 | 2026-11-03 ~03:45 |

- Arms 1–2 are **restarts from scratch** to 50.14M. Their Nov-1 partials (4 and 3 saves) were moved to
  `r94b_lr_s76_interrupted_20261102_133618` / `r94b_lr_s77_interrupted_20261102_133618` — kept, not deleted,
  and not used as initialisers (a checkpoint resume would restart the lr anneal at the new process's epoch 0).
- Arm 5's decay is **fitted to its own budget** (user order: "adjust the learning rate decay suitable for
  this"): lr reaches 0 at the 80.2M save, so it does not reproduce the control's 30M frozen tail. Its
  question is whether the extra 30M *holds and improves* the level instead of only freezing it.
- Measured startup rate: arm 1 **5,986 steps/s** (3090, fp64), arm 3 **4,483 steps/s** (3060, fp32) — against
  the recorded 5,882 / 4,611.
- Arm 4 was ordered as fp32 + light batch and **changed to fp64 + light batch** (user order 2026-11-02 22:31 IST):
  the same one-building swap on the default solve, same control, same 50.1M. `r103_fp32_light_s75` never
  created a run directory, so nothing was discarded. This makes arm 4 a clean batch comparison against
  `r94b_lr_s75` (fp64, `r85_four`), and leaves precision to arm 3.

**Status (measured, 2026-11-02 22:35 IST).** Arm 1 `r94b_lr_s76` finished **21:31 IST: 8,690 s (2.41 h),
5,769 steps/s, 10/10 saves**, final `0000000050135040.bin` complete. Arm 3 `r102_fp32_s75` (fp32, 3060)
finished **22:29 IST: 11,840 s (3.29 h), 4,234 steps/s, 10/10 saves** — 27 % slower per step than the 3090
at the same batch, as expected. Arm 2 `s77` started 21:31 IST, arm 4 `r103_fp64_light_s75` started 22:31 IST;
arm 5 follows arm 2. A first read of `s76`'s `env/score` curve is the only number available so far, and it is
a training-scenario meter, not a panel score.

**How each pair is read.** (i) s75 vs s76 vs s77: is the control's 0.0340 panel / 0.02588 on all 30 a property
of the recipe or the draw of one seed? Report best and final per save on all three, with `final − best`.
(ii) fp32 vs fp64 at the same seed and batch: does the training-only fp32 solve change what is learned, not
just the speed (21 % on the 3090, and the light-batch arm adds the 4th building's 28 % state reduction)?
(iii) 80M: held level and `final − best` with the anneal spread over the whole run, against the control's
50M-decay curve. No leaderboard claim from any of the five until a 30-scenario fp64 score exists.

**Scoring plan for these arms (armed 2026-11-02 23:40 IST, runs itself when each card goes idle).**
User order: score only `sets/tr23_shipped30` — the 30 competition scenarios in one eval per checkpoint (30 envs,
`desync=0`, one 8640-step month each), which writes the leaderboard metric directly (~8 min/checkpoint, against
~16 min for the 10-month panel). Only saves **at or after 10M** are scored (the 5.01M save is skipped); **`r102_fp32_s75`
is not scored** (the fp32 arm). 43 evals, split equally across the two cards, sequential per card, each card
starting when its own training finishes: card 1 (frees ~02:02) 22 evals = control@10M + `s76` 9 + `s77` 9 +
`r103` @10–20M; card 0 (frees ~03:48) 21 evals = `r103` @25–50M + `r104` 15. Expected finish ≈ 05:00 / 06:36 IST.
The control's 10M checkpoint is included because no 30-scenario number exists for it (only 40.11M = 0.02588).
Driver `results/study_r102_seeds/score30.py` (`plan` prints all 43 jobs), log `score30.log`. Guards before the
first eval on each card: every arm's stored `--policy./--env./--train.` flags must equal the control's (a
mismatched decoder loads the checkpoint into the wrong slots), the binary must be the fp64 build with no fp32
solve symbol, the blob set must hold 30 scenarios, and a card never waits on the other card's arms.

## r106 — the zone reheat valves as Bernoulli gates, on the control recipe (started 2026-11-03 08:05 IST)

`r106_binary_reheat_s75`, GPU 0 (3090), fp64, `r85_four`, seed 75, **50,135,040 steps**, anneal 50,135,040,
saves every 68 epochs, from scratch, **train only**. One change against `r94b_lr_s75`: `policy.binary_reheat=1` —
the 90 zone-reheat heads join the Bernoulli head the AHU run commands already use, so a closed valve is sampled
(`p(u=0) = 0.821070`, deterministic `p > 0.5`) instead of sitting ~7σ outside the Beta support
(`beta_conc_max` 12 at the occupied reheat mean 0.89). Banner confirmed in the pre-launch smoke.

**Why this lever and not the others (2026-11-03 analysis, `results/study_r102_seeds/REPORT.md`).**
- `train.value_adam` is withdrawn. Read from `r101` (two attempts, 13.8M and 9.4M): it does repair the
  `zone_value` rank-1 defect — value loss at matched steps 718/218 against the control arm r98's 1270/722 at
  10M/12M, advantage abs-mean 10.7 vs 14.9 — but it does not lift that arm (cost 1.05 vs 1.17 at 12M), and on
  the r94b recipe the defect does not exist: the value matrix has a full-rank gradient, Muon's step varies
  1.48–4.23 x lr, and the value loss is 10–15x lower at matched steps (124/48.6 at 8M/10M vs 1100/718).
- The five-run seed/precision/batch/budget family: s76 0.0545, s75 0.0550, r103 light 0.0585, r104 80M 0.1003,
  r102 fp32 0.1520, s77 0.8927 (final cost). All six hold their last three saves. Comfort is 57–61 % of the
  cost for the healthy runs; **Fairbanks owns 48–52 %** of the four-building mean.
- The Fairbanks floor is **converged, not truncated**: 0.1112 at 50M on the control, and r104 reached 0.1100
  after 80M with lr still alive (0.1265 → 0.1100 over 50M → 80M). More annealing time converges to the same 0.11.
- The measured defect at that floor: reheat command spread across zones 0.0006–0.01 (one number, no per-zone
  differentiation), plant water 311–312 K (38–39 °C) on all ten scenarios. The action "shut this hot zone's
  reheat" is priced at 0.0314 (p2) / 0.0311 (p3) against 0.0340 — **8 %** on the panel — and has never been
  reachable by the sampler, so no arm has ever tried it.

**Pre-registered gates.** (1) mechanism: the p2 counterfactual on the best save must lose at least half of its
−0.0026; (2) Fairbanks_1 night gas stays under 33 %; (3) cost ≤ 0.0550 at 50.1M, held two saves; (4) critic
guard: value loss ≤ 0.5 at 50M (the control's is 0.22 — `zone_value` is off, so nothing should destabilise it).
Measured rate at launch: ~6,000 steps/s, so ~2 h 20 m of training.

## r107 — is the 50M level a truncation or a convergence? (schedule test, started 2026-11-03 08:56 IST)

`r107_lrhold_s75`, GPU 1 (3060), fp64, `r85_four`, seed 75, **60,162,048 steps**, 12 saves, from scratch,
train only. One change against `r94b_lr_s75`: `train.min_lr_ratio=0.05`, so lr decays 1e-4 → **5e-6** over
50,135,040 steps and then **holds 5e-6 for the last 10.03M** instead of reaching exactly 0 (the default
`min_lr_ratio=0.0` is what froze the control). No source change — the flag already exists
(`SB_LINEAR_LR_ANNEAL base 0.0001 -> 5e-06 over 680 epochs (50135040 steps)`) and **no recorded run had ever
used it**.

**The question it answers.** Every run in the seed/precision/batch family reaches its best cost at its *last*
save, and the Fairbanks floor came out at 0.1112 @50M (control) and 0.1100 @80M (r104). Those two readings
disagree about the cause: "still falling when frozen" (truncation — the level is where the LR happened to stop)
versus "two different schedules converge to the same 0.11" (a real floor). r104 cannot separate them because it
kept lr *higher from step 0*, which changes the whole trajectory; r107 keeps the control's trajectory and only
removes the hard zero.

**Read.** Saves at 50.1M → 60.2M. If the cost keeps falling below 0.0550, the level is a truncation artifact and
the recipe wants more effective steps at small lr. If it flattens at ~0.055 or drifts up like r104's 50→80M tail
(comfort −0.0021, energy+carbon +0.0035 over 30M), then 0.055 is a converged floor and the schedule is not the
lever. Measured rate at launch: 3,218 steps/s on the 3060 (~5 h 10 m); r106 runs in parallel on the 3090.

## r108 — nonlinear shared zone actor (approved 2026-11-03)

From scratch, seed 75, fp64 `r85_four`, GPU 0, 50,135,040 steps; r94b is the stored control. One changed prior: a shared 32-unit tanh zone actor on recurrent context plus existing local features, added to the existing affine Beta outputs with a zero-initialized output matrix (P43). Same critic, reward, action bounds and r94b settings. The original plan was to record ten development months at 10.03/20.05/30.08/40.11/50.14M. Success requires Fairbanks 1 ≤ 0.09218 and panel ≤ 0.03400 at the same two consecutive saves. Protocol and run status: `complete_port/results/study_r108_local_mlp/`.

**Result (2026-11-03):** Training completed at 50.135M; user stopped remaining scoring after the complete 10.027M panel and the 20.054M Fairbanks month. Late 40–50M stochastic training panel median 0.07243 vs r94b 0.05463; 25M was better but regressed and recovered. The 20M Fairbanks month improved 0.20753 → 0.13908 with much better comfort but gas 122.1 → 317.0 MWh and carbon 0.17660 → 0.45256. The 10-month registered success and final deterministic score remain untested. Full curve, tradeoffs, restrictions and proposed credit hypothesis: `complete_port/results/study_r108_local_mlp/POSTMORTEM.md`.

## r109 — building-wide valve credit (approved 2026-11-03)

Fresh seed 75, fp64 `r85_four`, 4×64 on GPU 0, 50,135,040 steps. Change P44 only against the r108 architecture and recipe: zone damper/reheat PPO updates use the building advantage while retaining the zone critic. Train only; one trainer gate chains directly to the main run. Protocol, status and diagnostics: `complete_port/results/study_r109_global_valve/`.

**User-stopped:** r109 stopped at 34.65M logged steps on 2026-11-03; six 5.0135M-grid checkpoints retained. During 30–34.65M its stochastic training panel median was 0.354 vs r108 0.131 and r94b s75 0.113; no deterministic r109 months were scored. Credit replacement on the nonlinear actor did not deliver a control-beating trajectory in this partial run.

## r110 — raw-sample conditional Beta process (approved 2026-11-03)

Fresh seed 75, fp64 r85_four, GPU 0, r94b architecture/settings and 50.135M schedule. One changed prior P45: coherent Beta quantile behavior and conditional raw-sample PPO likelihood. The gate passed and training finished. **Original scoring defect:** despite the deterministic flag, all 10 x 30 episodes sampled actions; the stdout also printed the ten-panel sign incorrectly. The approved 3090 mean-action rescore now records 10 checkpoints x 10 months: 0.13290 at 10M, best 0.06136 at 20M, 0.19284 at 30M, 0.09286 at 40M, 0.06947 final; final-minus-best +0.00812, longest row within 10% of best one save. r94b s75 scores 0.12408/0.10070/0.03400/0.03969 at 20/30/40/50M. At the respective best points P/E/C and electricity/gas MWh are r110 -0.04458/0.10566/0.06739 and 35.71/44.85; r94b -0.02551/0.04636/0.04710 and 12.87/38.04. The early r110 win was not held. Full components, scenario rankings, action and time slices and a proposed unapproved next hypothesis: `complete_port/results/study_r110_conditional_beta/review/MEAN_POSTMORTEM.md`.

## r111 — independent Beta mean and concentration (approved 2026-11-04)

One change to r110: P46 parameterizes the Beta mean and concentration independently, allowing shapes below one. Fresh seed 75 on the 3090, 50.135M fp64 training steps, with ten recorded ten-month mean-action checkpoint evaluations after training. The gate passed and training completed. Initial network weights use the same random initialization, but this map induces a different initial action distribution. Best 0.04377 at 30.08M, final 0.13366; final minus best +0.08989 and only one save within 10% of best. At best, productivity/energy/carbon were -0.01919/0.09606/0.06522 and electricity/gas 32.92/47.22 MWh. The 30–40M recorded regression and AHU oscillation are on the 3090; other scoring checkpoints used the 3060 or mixed devices. This did not beat the held r94b level. Evidence, provenance and an unapproved next hypothesis: `complete_port/results/study_r111_mean_precision/review/POSTMORTEM.md`.

## r112 — empty-building fuel credit (completed 2026-11-04)

P47 alone on the r94b s75 recipe, fresh 3090 fp64 training to 50.135M; trainer gate passed and all 55 registered deterministic months were recorded. Panel at 10/20/30/40/50M: 0.39984/0.18180/0.22732/0.09494/0.05898, against r94b 0.23718/0.12408/0.10070/0.03400/0.03969. Panel final P/E/C −0.06223/0.05896/0.04923 and electricity/gas 17.80/38.62 MWh; r94b −0.03521/0.04758/0.04525 and 13.40/34.85 MWh. Fairbanks final 0.13117 vs 0.10345: reduced gas 142.84 vs 168.63 MWh but worse productivity −0.12295 vs −0.05958. The held objective and preregistered win failed; one of ten final scenarios improves. Full matched curve, time/weather slices, raw credit diagnostics, action analysis and unapproved next design: `complete_port/results/study_r112_night_credit/review/POSTMORTEM.md`.

## r114 — team valve credit with a detached zone baseline (approved 2026-11-04)

P48 on the r94b seed-75 recipe: replace zone-local valve GAE with building GAE minus a shared state-only zone baseline, fitted with separate Adam. Fresh 50.135M fp64 steps on GPU 0, then 55 registered deterministic recorded months; one trainer gate chains directly to training and scoring. Full protocol, diagnostics, restrictions and status: `complete_port/results/study_r114_team_baseline/`.

**Training result:** gate passed and all 10 checkpoints landed; training elapsed 8,059 s, 6,221 env-steps/s. At 35/40/45/50M, four-scenario training episode score 0.1722/0.2458/0.2689/0.2649 vs r94b s75 0.1117/0.0539/0.0651/0.0461. At 50M Fairbanks 0.6797 vs 0.1094; Phoenix 0.1754 vs 0.0265. Baseline residual variance increased in warm groups; details and full training curves in `complete_port/results/review_r94b_r114_training_20261105/analysis.txt`. The user said evaluation was not needed: remaining scoring was stopped mid-month with prior artifacts retained; no frozen-policy claim.

## r115 — action-head PPO clipping (approved 2026-11-05)

P49 on the r94b seed-75 recipe, fresh fp64 50.135M training on GPU 0. One required trainer gate automatically launches training; no scoring. Record per-group and role likelihood changes, sign-aware clipping and aggregate cancellation in addition to the existing detailed training diagnostics. Correlated-sampler marginal ratios remain approximate. Protocol and status: `complete_port/results/study_r115_headwise_ratio/`.

**Training result:** gate passed and all ten policy and optimizer saves landed. 20/30/40/45/50M panel 0.08131/0.06455/0.05695/0.05328/0.05203 versus r94b s75 0.15871/0.06880/0.05394/0.06510/0.04613. r115 reached the plateau earlier and held its last three saves within 10% of best; final is higher than r94b's. At 50M Fairbanks/Phoenix/Denver/Atlanta were 0.11236/0.04341/0.03499/0.01734. Fairbanks is still the largest cost. This is changing-policy, fixed-window training telemetry, not a frozen-policy score. Detailed curves, component and time breakdown, headwise ratios and proposal: `complete_port/results/review_r115_r94b_training_20261105/`.

## r116 — nonlinear zone actor with headwise PPO (approved 2026-11-05)

P50: r115 s75 with only `policy.zone_mlp=1`, fresh fp64 50.135M training on GPU 0, no scoring. An existing trainer gate checks the combined actor and PPO path and chains directly to training. Passive diagnostics add state-conditioned zone actions, terminal-safe jitter, same-time zone spread and zone value target/residual moments by scenario and observed time/occupancy. Fixed protocol and status: `complete_port/results/study_r116_zone_mlp_headwise/`.

**Result and evidence (2026-11-05):** The trainer gate passed; fresh fp64 GPU 0 training completed 50.135M steps in 8,368 s (5,991 steps/s), and all ten checkpoints and optimizer sidecars landed. No scoring was run. The latest-completed-episode equal-group training panel at 20/30/35/40/45/50M is **0.12846/0.05276/0.04337/0.03677/0.03433/0.03308**, versus r94b s75 **0.15871/0.06880/0.11166/0.05394/0.06510/0.04613**, r115 **0.08131/0.06455/0.06302/0.05695/0.05328/0.05203**, and r108 (same zone MLP, geometric ratio) **0.14777/0.13896/0.11896/0.08496/0.07070/0.07063**. r116 is worse than r115 at 20M but better throughout the 30–50M save grid. The best grid point is the final save; final minus best is 0, and the final two saves are within 10% of that best. Both 45M and 50M pass the predeclared Fairbanks ≤0.100, panel ≤0.050, Phoenix ≤0.050 and Atlanta ≤0.025 training gates. These are stochastic episodes collected while the policy changes, with each group's last completed episode carried forward; they are not frozen checkpoint or leaderboard results. Only one r116 seed has run.

**Where the gain came from.** At 50M Fairbanks/Phoenix/Denver/Atlanta are **0.07649/0.02449/0.01809/0.01326** versus r94b s75 **0.10944/0.02652/0.03383/0.01473**. The controlled r115→r116 change is the zero-output-initialized shared nonlinear zone actor; the learned branch's output weight norm grew from approximately 0 to 5.87. Actual late training data show mean same-time across-zone reheat spread of 0.153 in observed-occupied Fairbanks, with mean reheat 0.898/0.729/0.452 for cold/in-band/hot present zones. Building occupancy is observed one interval late; these are present zones, not individually occupied zones. r115 lacks comparable native spread telemetry. On the same already-recorded r108 Fairbanks input sequence, CPU-replayed 50M checkpoints show average daytime across-zone reheat-mean spread 0.13875 for r116 versus 0.00886 for r115 and 0.03322 for r94b; r108 reconstruction error is 8.3e-8 mean. The input trajectory, including previous-action observations, was generated by r108, so it can be out of distribution for the other policies and this is not their closed-loop outcome. These observations support a changed local response, not a causal proof of score improvement. r108's MLP alone regressed late, and r115's headwise clipping alone improved speed/stability but not the final r94b comparison; the combination is the first observed late training gain. This does not isolate why the combination works or establish generalization.

At 40–50M, count-weighted live Fairbanks productivity/energy/carbon and electricity/gas were **−0.04596/0.07335/0.17191 and 5.50/167.72 kW**, versus r115 **−0.10375/0.07882/0.18605 and 5.66/181.70 kW**: better comfort with lower average fuel and carbon, though afternoon gas and night fuel rose. Phoenix improved comfort **−0.04964→−0.02088** while electricity rose **51.22→53.37 kW**; Denver improved comfort and reduced both fuels; Atlanta improved comfort with slightly more gas. These live means are not the carried-episode panel and r94b lacks the same live-bin instrumentation. Fairbanks remains **58%** of r116's final four-window sum; its 0.07649 alone implies a panel floor of 0.01912 even if the other windows were zero. During 40–50M its weekday morning live productivity/energy/carbon was −0.16265/0.16979/0.38357, with 14.33/373.21 kW electricity/gas. Its afternoon comfort improved substantially but gas averaged 468.54 kW versus r115 436.08 kW. The 0.005–0.01 objective is not reached.

**Remaining learning concern and limits.** The pooled building-value explained variance was high (40–50M median 0.980), but first-replay per-zone values versus their own bootstrapped GAE targets had EV around 0.64–0.79 in Fairbanks occupied states and approximately −0.12–0.24 across occupied hot/in-band/cold states in Phoenix and Atlanta. Targets are critic-dependent bootstraps, not independent return labels, and the states/visits are correlated; this points to a candidate representation problem, not proof of incorrect true values or that changing the critic will improve the score. Deep post-mortem, exact window statistics, action/critic slices, previous-run deduplication and candidate design: `complete_port/results/review_r116_20261105/`.

**Research references for r116.** These papers explain components or related ideas; they do not establish that a particular paper caused this run's improvement. The run's distinctive one-factor change from r115 is the zero-output-initialized 32-unit shared zone MLP, while the Beta policy, temporally coherent sampler and headwise PPO were already present. Implementation and measured effects remain the evidence for r116.

- **Headwise likelihood clipping:** Han and Sung, [Dimension-Wise Importance Sampling Weight Clipping for Sample-Efficient Reinforcement Learning](https://proceedings.mlr.press/v97/han19b.html), ICML 2019. This motivates treating action dimensions separately in high-dimensional policies. r116 clips each active Beta head's marginal ratio with its own advantage; it is **not the complete DISC algorithm** and does not include DISC's old-trajectory replay/V-trace method. On r116 data at 20–30M, 3.88% of Fairbanks zone-head replay visits were sign-clipped, 3.51% while the aggregate geometric ratio was inside its clip interval; this establishes that the mechanism was exercised, not that it caused the gain.
- **Bounded stochastic actions:** Chou, Maturana and Scherer, [Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution](https://proceedings.mlr.press/v70/chou17a.html), ICML 2017. The actual Beta implementation in `complete_port/patches/pufferlib_core.diff` names this paper. r116 uses bounded Beta actions, with its own shape floor, concentration cap and correlated sampler; the paper's results do not prove that this modified sampler has an unbiased PPO gradient.
- **Temporally coherent exploration:** Korenkevych et al., [Autoregressive Policies for Continuous Control Deep Reinforcement Learning](https://arxiv.org/abs/1903.11524), IJCAI 2019, and Hollenstein, Martius and Piater, [Colored Noise in PPO](https://arxiv.org/abs/2312.11091), AAAI 2024, support the general study of correlated exploration. r116 uses AR(1) normal proposals inside a Gamma/Beta sampler (`beta_cn_tau=12`, `beta_cn_tau_ahu=48`); it is not either paper's exact policy/noise algorithm. Its PPO ratios score Beta marginals rather than exact history-conditional likelihoods, so standard independent-sampling guarantees do not directly follow.
- **PPO and GAE:** Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) and [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). r116 uses PPO-style replay and GAE with γ=0.99 and λ=0.95, but its headwise clipping and zone-local advantages differ from a standard joint PPO objective. These papers explain the estimator, not proof of the proposed next critic or of performance on this building.
- **Shared zone representation:** Zaheer et al., [Deep Sets](https://arxiv.org/abs/1703.06114), NeurIPS 2017, is relevant background for shared processing of variable-size sets and permutation properties. r116 shares the same 32-unit nonlinear branch across present zones, but its inputs also include structured sibling/slot features; it is not the paper's exact architecture, and this citation alone does not prove arbitrary zone permutation invariance. The r116 telemetry and matched-input analysis show differentiated outputs; they do not isolate an MLP causal effect.
- **Adaptive optimization:** Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015. r116's role readouts and zone MLP use Adam while the trunk uses its existing Muon path; the result is not a test of Adam alone.
- **Building context:** Goldfeder et al., [The Smart Buildings Control Suite](https://arxiv.org/abs/2410.03756), 2024/2025, describes the HVAC simulator and reward tradeoffs motivating this competition. The competition oracle and its actual reward implementation remain authoritative for the specific scored objective; the paper's comfort curve must not be substituted for the competition's.

### 2026-11-05 — r116's first frozen months (scored on GPU 1, `study_r116_zone_mlp_headwise/scored/ANALYSIS.md`)

The user asked whether r116's actions behave correctly; no closed-loop recording existed (scoring had been
stopped), so the 50.135M checkpoint was scored deterministic fp64 on the 3060 — one month each on the four
training scenarios, compared with the recorded r94b_lr_s75 months at the same step:

- **Scores: r116 mean 0.0305 vs r94b 0.0377.** fairbanks_1 **0.0755 vs 0.1035 (−27 %)** — better on all three
  terms, gas 118.5 vs 168.6 MWh; denver_2 0.0148 vs 0.0198 (−25 %, gas −46 %); phoenix_6 0.0210 vs 0.0183 and
  atlanta_6 0.0105 vs 0.0090 are ties. This is the first frozen-policy evidence for the r116 late-training
  gain; the training-panel ordering holds closed-loop.
- **Behavior is correct in 3/4 scenarios**: monotone cold > in-band > hot reheat (fairbanks_1 0.916/0.748/0.396,
  denver_2 0.804/0.376/0.162), reheat-vs-(T−Hs) correlation −0.55/−0.57, cross-zone spread 0.12–0.18 against
  r94b's 0.006–0.044 — the zone differentiation is real on the policy's own inputs, not only on r108's replay.
  Zero rail saturation, zero NaNs anywhere.
- **One anomaly: atlanta_6** — the cold bucket gets less reheat than in-band (0.233 vs 0.735) and the state
  correlation is +0.03: the branch does not track zone temperature there. The month still posts the best
  comfort of the four (prod −0.0067), so nothing is billed, but "reads its own zone temperature" does not hold
  uniformly.
- Limits: four training scenarios, one checkpoint, one seed; held-out months unscored. Analysis script:
  `scored/analyze_closedloop.py` next to the recordings.

## r117 — local nonlinear zone-value residual (approved 2026-11-05)

P51: user approved the candidate in `complete_port/results/review_r116_20261105/REVIEW.md`. Fresh seed 75, fp64 r85_four (4×64), GPU 0, 50.135M steps and ten saves, no scoring. Add a shared 32-unit tanh value residual to r116's pooled per-zone values, with zero output at initialization, detached actor/trunk inputs and separate Adam moment entries. The measured uncertainty remains: low EV is against bootstrapped targets, and omitted shared fuel credit remains unchanged. One existing trainer gate is chained directly to training; registered mechanism and performance criteria, checkpoint state, and telemetry: `complete_port/results/study_r117_local_value_residual/protocol.json`.

**Result (2026-11-05):** Required trainer gate ended `ALL CHECKS PASSED`; fresh fp64 GPU 0 training completed 50.135M steps in 8,708.6 s (5,757 steps/s), with ten policy, Muon and role-Adam saves. No scoring. Four-window training panel at 15/20/25/30/40/45/50M was **0.10495/0.07662/0.06278/0.05561/0.04796/0.04837/0.04415** versus r116 **0.17323/0.12846/0.07106/0.05276/0.03677/0.03433/0.03308**. r117 was better early but worse from 30M onward and fails both registered 45M/50M panel and Fairbanks thresholds. Its last three grid points are within 10% of its grid best; final minus best 0. It holds a worse level, not a late collapse. One seed; changing-policy training telemetry, not frozen-policy or leaderboard scores.

At 50M Fairbanks/Phoenix/Denver/Atlanta were **0.09261/0.03453/0.02992/0.01953**, versus r116 **0.07649/0.02449/0.01809/0.01326**. Late 40–50M live Fairbanks productivity/energy/carbon and electricity/gas were **−0.07887/0.07347/0.16752, 6.09/162.82 kW**, against r116 **−0.04596/0.07335/0.17191, 5.50/167.72 kW**. r117 used slightly less gas but lost comfort; its Fairbanks productivity guard failed. Phoenix night electricity improved but weekday comfort and electricity worsened; Denver and Atlanta also worsened. The new critic's output norm reached 1.208, and some GAE-target explained variances increased, but target distributions changed and Phoenix in-band residual variance grew 0.00720→0.00788. r117's distinct 48-step, terminal-aware bootstrapped value diagnostic has negative EV in Phoenix occupied bins; r116 did not record this diagnostic, so no matched 48-step comparison is possible. Reject P51 as a measured seed-75 training improvement. Full curves, costs, action and critic slices, prior-run comparisons and an unapproved next candidate: `complete_port/results/review_r117_20261105/REVIEW.md`.


**r117 research basis and outcome.** Schulman et al., [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), motivated the existing temporal-difference/GAE interpretation of value prediction; r117 retained r116's GAE and added our own shared local residual critic, not a critic architecture taken from that paper. The branch learned, but the late training outcome was worse than r116, as measured above. The paper does not establish that the branch should help.

## r118 — synchronize zone GAE and normalization per replay pass (approved 2026-11-05)

P52: fresh r116-based seed-75, four fixed weather windows, fp64 RTX 3090, 50.135M steps, ten saves, no scoring. Recompute and cache zone advantages, value targets and present-zone normalization together at each of four replay-pass boundaries. Keep the actor, building advantage, local reward, other settings and simulator fixed. Per-pass same-sample diagnostics measure actual critic drift by scenario, time, occupancy and zone state. One required trainer gate automatically starts training. Protocol and predetermined decision criteria: `complete_port/results/study_r118_zone_pass_refresh/protocol.json`.

**Research paper:** Andrychowicz et al., [What Matters In On-Policy Reinforcement Learning?](https://arxiv.org/html/2006.05990), section 3.5 and Appendix B.1, studied recomputing advantages once per data pass. We adapt the pass-level refresh to the existing zone-local GAE, its targets and normalization; our fixed minibatch order, correlated Beta sampler and headwise PPO differ from their setup. **Did it help?** The completed r118 result is recorded below; the paper does not prove improvement in this environment.


**Result (2026-11-05).** The gate ended `ALL CHECKS PASSED` and r118 completed 50.135M fp64 steps on GPU 0 in 8,680.1 s (5,775.8 SPS); all ten weight/Muon/role-Adam saves landed. No scoring. At 40/45/50M its four-window training panel was **0.04261/0.04170/0.03681**, against r116 **0.03677/0.03433/0.03308** and r117 **0.04796/0.04837/0.04415**. Its 45M and 50M panel and Fairbanks gates failed; the 40–50M Fairbanks live productivity and E+C guards passed. Final Fairbanks/Phoenix/Denver/Atlanta were 0.07640/0.03020/0.01723/0.02340: Denver improved over r116 and Fairbanks was essentially tied, while both warm groups worsened. Final minus grid-best 0; one grid save within 10% of best, so no two-save hold. These are changing-policy training readings, not frozen-policy scores.

40–50M live Fairbanks P/E/C and electric/gas were −0.04373/0.07906/0.18556 and 5.97/181.20 kW versus r116 −0.04596/0.07335/0.17191 and 5.50/167.72 kW. Denver better comfort came with higher gas; Phoenix and Atlanta worsened in comfort and both fuel means. In occupied Atlanta at 20–30M, the first-pass mean absolute cached-versus-current normalized zone-GAE drift was 0.145 with 6.1% sign changes; at 40–50M it was 0.050 and 1.6%. It therefore measured a real mechanism, not evidence that the drift caused the r116 ceiling. The [Andrychowicz et al.](https://arxiv.org/html/2006.05990) recommendation combines per-pass recomputation and shuffled transitions; r118 retains fixed trajectory blocks and applies the change to zone credit only. The local adaptation accelerated early training but failed the late overall improvement. Reject P52 as the next control. Full curves, time, action, critic and drift diagnostics, common-input AHU reconstruction and an unapproved AHU nonlinear-actor candidate: `complete_port/results/review_r118_20261105/REVIEW.md`.


## r119 — shared nonlinear AHU actor (approved 2026-11-05)

P53: user approved the single-factor candidate from `complete_port/results/review_r118_20261105/REVIEW.md`. Fresh r116-based seed 75, fp64 r85_four (4×64), RTX 3090, 50.135M steps and ten saves; no scoring. Add a shared 32-unit tanh AHU actor residual over existing context/local inputs with a zero-initialized eight-logit output matrix; preserve the plant/zone actors, reward, critic, sampling and optimizer schedules. One required trainer gate chains directly to training. Capture per-group/time/observed-occupancy AHU and plant actions, action boundaries, transition changes and branch weight/gradient norms. Protocol and registered criteria: `complete_port/results/study_r119_ahu_mlp/protocol.json`.

**Research basis:** [Deep Sets](https://arxiv.org/abs/1703.06114) motivates a shared entity transform; our structured AHU head is not the paper's complete architecture. [Deep Residual Learning](https://arxiv.org/abs/1512.03385) motivates a learned additive branch; its zero-output initialization is our own choice. The related r116 zone MLP improved late training on one seed, but the AHU adaptation was user-stopped early after a poor training trajectory. Neither paper proves an HVAC benefit.

**r119 partial result:** The required gate passed. The user stopped r119 at a last logged 11,649,024 steps (epoch 157), retaining two saves. Latest-completed equal-group training panel at 5.014/10.027M was 1.77508/1.88588, versus r116 s75 1.87963/0.27344; Fairbanks at 10.027M was 6.82356 vs 0.31081. At the last logged step the panel was 1.48120 and Fairbanks 5.19762. Building value loss at 10.027M was 3,043.45 vs 14.58, while explained variance was 0.902 vs 0.884; that does not establish whether the critic caused the poor trajectory. This was a poor early result, not a completed 50M comparison; recovery is unknown. No scoring or resume. The [Deep Sets](https://arxiv.org/abs/1703.06114)/[ResNet](https://arxiv.org/abs/1512.03385)-inspired shared AHU branch did not help at the observed stage. See `complete_port/results/study_r119_ahu_mlp/FAILURE.md`.


## r116 finalized research baseline — second seed (2026-11-05)

The user selected r116 as the research baseline. Fresh r116 seed 76 completed the same 50.135M fp64 3090 protocol as seed 75, on the same frozen binary and flags aside from seed and paths, with all ten weight/Muon sidecars; no scoring. Its 40/45/50M latest-completed four-window training panel was **0.04410/0.03832/0.03522**, versus seed 75 **0.03677/0.03433/0.03308** and matched r94b s76 **0.06391/0.05194/0.05404**. Both r116 seeds held within 10% of their own final grid best for two consecutive saves, and each final-minus-best is zero. At 50M seed-76 Fairbanks/Phoenix/Denver/Atlanta were **0.08048/0.02677/0.01924/0.01441**, all below r94b s76 **0.10374/0.04451/0.04441/0.02352**.

The 40–50M seed-76 Fairbanks live P/E/C and electricity/gas were **−0.05047/0.08161/0.18633 and 6.70/181.11 kW**; seed 75 was **−0.04596/0.07335/0.17191 and 5.50/167.72 kW**. Both learned differentiated zone responses, though their warm-weather actions and fuel trades differ. The two-seed result strengthens the combined [Deep Sets](https://arxiv.org/abs/1703.06114)-inspired shared zone MLP and headwise-clipping recipe; it does not isolate those components or prove the paper's method caused the improvement. Four changing-policy training windows and two seeds are not a frozen-policy, unseen-weather or leaderboard result; the 0.005–0.01 objective remains unmet. r116's `.mb` sidecars do not include its separate role-Adam moments. Full curves, time/comfort/energy/fuels, actions and caveats: `complete_port/results/review_r116_two_seeds_20261105/REVIEW.md`.

## r120 — r116 with random episode starts (2026-11-05)

User-requested fresh seed-75 RTX 3090 fp64 run, 50.135M steps, four r85 buildings × 64 environments, no evaluation. Each full episode begins at one of seven random day offsets, including its first; desync truncation is off and actual weather follows the shifted date. Same r116 policy and PPO settings; calendar seed 75. Gate `complete_port/results/verify_20261105_210656.log` ended **ALL CHECKS PASSED**. Scenario, component, fuel, time, action, critic and start-day diagnostics are retained. The changed start, weather window and desync jointly change the training distribution, so raw training panels are not fixed-window comparisons. Paper background and protocol: `complete_port/results/study_r120_random_start/protocol.json`. Result pending.

## r116 seed 77 — stopped 2026-11-05 20:37 IST (seed-linked critic failure)

`r116_zone_mlp_headwise_s77` (one change against the selected baseline: `--base.seed=77`) stopped at **epoch 234
(17.3M, 3 saves)** on user order, because it was repeating `r94b_lr_s77`'s failure: its value target's second
moment grew 21.6k -> **106.5k** while s75's and s76's fell to 477 / 1990, so the normalised-step value head
froze at a value loss of ~3,000 with explained variance 0.92 (right ordering, wrong scale), `adv_abs_mean`
stayed **20-26** against 1.8-2.3, entropy narrowed to -244 and Fairbanks froze at 6.5-6.9 (cost 1.7444 at 10.03M
vs 0.2776 / 0.3511 for s75 / s76). Up to 4M seed 77 was the *best* of the three, so this is a trajectory
divergence, not an initialisation defect; and the same seed failed the same way in the r94b family, which makes
it seed-linked rather than arm-linked. Files kept; no resume, no rescore. Write-up:
`complete_port/results/study_r116_zone_mlp_headwise/STOPPED_s77.md`.

## r120 — random start ("start any day"), FAILED as an arm and as a measurement (2026-11-06)

`r120_random_start_s75` = r116 + `env.random_start=1` (`env.random_start_seed=75`, `desync=0`, blob set
`r85_four_random_start`), 50.14M finished to epoch 680. Marked failed on user order: its curve reads straight and
below r116. The reason is mechanical, not learning.

- **The curve is 2-3 points per save.** 22 of 680 epochs logged an episode against r116's 650 (38-68 per 5M
  window). `episode_log_k` only accepts `st >= 8640`, and the random start is `day*288` with `day = mix(rng) % 7`
  (one of the first seven days), with the episode ending when the month clock hits 8640 — so 6 of 7 episodes are
  partial months and are dropped, and only the `day == 0` draw (1 in 7) is ever logged.
- **Train/eval mismatch by construction**: mean training episode 7,776 steps, 6/7 truncated at the month end,
  while every scored month is a full 8,640 steps from Sunday 06:00.
- **Level**: r120 leads early (5M 0.3532 vs 1.9608; 25.1M 0.0509 vs 0.0732), r116 overtakes at ~35M and finishes
  better (50.1M **0.0341 vs 0.0421**). Critic healthy (value loss 0.14 vs 0.17, adv_abs 0.27 both), entropy wider
  (-167 vs -214) — not a divergence.
- A correct version must log every episode with its length, keep the scored length in training (full 8640 from a
  random offset, wrapping), and report the `g_ep_start` day histogram alongside the curve. Write-up:
  `complete_port/results/study_r120_random_start/FAILURE.md`.

## r122 — 30-scenario scale-up (running)

Fresh seed 75 from the r116 recipe, 6 buildings × 5 weather windows, 32 environments each, fp64 physics, 1,500,180,480 total steps (50,006,016 per scenario). Linear learning-rate anneal spans the full budget; checkpoints are every 29,859,840 steps. Larger batch and scenario mixture are changes from r116 and have not been shown to preserve its learning curve. Protocol: `complete_port/results/study_r122_train30_portable_s75/protocol.json`.

**Research basis:** r116 adapts the shared entity-transform idea from [Deep Sets](https://arxiv.org/abs/1703.06114) to its zone head and uses headwise clipping inspired by [PPO](https://arxiv.org/abs/1707.06347); neither paper establishes that this 30-scenario scaling will improve results. **Did this scale-up help?** Pending.

The local trainer gate `complete_port/results/verify_20261106_125046.log` ended **ALL CHECKS PASSED**. A small complete-state continuation restored from the 3090 to the 3060, though subsequent weights differed across GPUs. The L40S and 960-agent run were not part of that gate.

**Started 2026-11-06:** Fresh seed 75 on the first Modal account, L40S, detached app `ap-2rWcJKU2cmtBBQfw4GZ9pl`. Startup metadata confirms the frozen executable, 30-scenario archive and requested args; the first `SB_UPDATE` at epoch 0 landed. Full 960-agent throughput and learning are not yet established. Saved source and launch record: `complete_port/results/study_r122_train30_portable_s75/`.

**Submission prepared 2026-11-06 (step 238,878,720):** `submission/238878720/` holds the numpy policy and the checkpoint (`sha256 37a89647…9dcc5`, identical to the snapshot manifest). The architecture was pinned from the trainer's registration order (453,736 floats = the exact file size; the unused log-std row is still `init_logstd` −1.0), the numpy actor reproduces r116's recorded deterministic month to max 1.07e-6, the canonical observation and zone-memory block rebuilt from the official environment match the port exactly at step 0, and r116's own checkpoint re-scores 0.075465 through the official simulator against the port's recorded 0.075480. All six shipped buildings fit the canonical slots with every action mapped. Single-month diagnostics taken while checking the code (fairbanks_1 and atlanta_2 for 238,878,720; one held-out weather month for 507,617,280) are kept in `complete_port/results/study_r122_submission/` and are deliberately **not** listed as scores here. The finished official 30-scenario `pid_water` pass gives a mean score of **0.05496** (30 full 8640-step months, range 0.01689–0.21392, `complete_port/results/study_pid30/`); it is recorded with the other full 30-scenario evaluations at the end of this file.

**r122 ended 2026-11-07 01:36 IST at $29.5304 of the $30 credit** (credit watchdog on the workspace meter, stop issued at `meter $29.5304 >= $29.50`; $2.429/h all-in, 12.16 h). Final committed snapshot **507,617,280 steps = 33.8 % of 1,500,180,480** (epoch 1,836 of 5,426; 16.9M per scenario = 1,958 months), final training meter `env/score` −0.02608. 17 snapshots. Archive `complete_port/results/r122_archive/` holds every small file (weights, `.mb`, `.hadam`, manifests, `train.log`, TB prefix, segment records, `/inputs`; 11 GB) with all 51 components verified against the run's own sha256/bytes; the `.bin.full` complete-state files (242.67 GB) stay on the volume. **Kaggle from checkpoint 238,878,720: ref 58061991 `COMPLETE`, public score 0.02392** (r46's loaded submission 0.11788, neutral 0.41304). Four more checkpoints of the same run were then scored; the full five-point leaderboard record, the best checkpoint (89,579,520 at **0.01818**) and what the training curve did and did not predict are in the next section.

### r122 Kaggle scores — five checkpoints, one run (all `COMPLETE`, 2026-11-07)

Five checkpoints of r122 were scored on the leaderboard. All five loaded their weights (`floats 453736` in the kernel log) and all five are `SubmissionStatus.COMPLETE`. Public score is the mean of −R over the hidden evaluation scenarios; lower is better.

| checkpoint | curve mean (30 training scenarios) | worst scenario | scenario wins | kernel | submission ref | **public score** | curve − score |
| ---------: | ---------------------------------: | -------------: | ------------: | ------ | -------------: | ---------------: | ------------: |
| 59,719,680 | 0.02453 | 0.09099 | 10 / 30 | `submission-r122-59719680` | 58077975 | **0.01842** | +0.0061 |
| **89,579,520** | 0.02524 | **0.08497** | 5 / 30 | `submission-r122-89579520` | 58077977 | **0.01818** | +0.0071 |
| 119,439,360 | 0.02453 | 0.08622 | 5 / 30 | `submission-r122-119439360` | 58077978 | **0.02104** | +0.0035 |
| 209,018,880 | 0.02570 | 0.10992 | 5 / 30 | `submission-r122-209018880` | 58077981 | **0.02262** | +0.0031 |
| 238,878,720 | 0.02779 | 0.11279 | 0 / 30 | `submission-r122-238878720` | 58061991 | **0.02392** | +0.0039 |

Ordering: **89,579,520 (0.01818) < 59,719,680 (0.01842) < 119,439,360 (0.02104) < 209,018,880 (0.02262) < 238,878,720 (0.02392).** The first three are the same level (0.0182–0.0210, 0.0029 apart); the last two are worse by 0.003–0.006.

What the curve said, and what the leaderboard said:
- **Direction: right.** The curve put the 59.7M/89.6M/119.4M band ahead of 209.0M and 238.9M, and the leaderboard agrees: the three early saves beat the two late ones.
- **Resolution: not enough for the top.** The curve tied 59,719,680 and 119,439,360 at 0.02453 and put 89,579,520 third (0.02524); the leaderboard order among those three is 89.58M, 59.72M, 119.44M, and the margin between its top two is 0.00024 — below the curve's own noise. Any of the three is a defensible submission; the 30-scenario curve cannot separate them.
- **Level: pessimistic by 0.003–0.007.** Every checkpoint scores better than its curve window mean, so the training meter is a relative proxy only, not a leaderboard estimate.
- **The worst scenario in the curve is the best predictor of the leaderboard rank.** 89,579,520 has the lowest curve worst case (0.08497) and the best Kaggle score; 238,878,720 has the highest (0.11279) and the worst Kaggle score among the five. With a mean that is dominated by a few cold scenarios, the tail statistic ordered the five exactly.

Anchors, same leaderboard: r46 shared-MinGRU loaded submission **0.11788**; our r2 residual **0.08148**; `bang_bang` **0.21092**; neutral action **0.41304**. The best r122 row is therefore **6.5× better than r46** and **4.5× better than our previous best**, and **3.0× the 0.00598 public target**.

Caveats: one public score per checkpoint (one evaluation, one seed), the public split is a subset of the hidden scenarios, and these are not the local 30 shipped scenarios — the two cannot be subtracted. Checkpoint choice at the 0.018 level is unresolved between the top two. Curve data: `complete_port/results/study_r122_curves/score_matrix.csv`; archive of all five weights: `complete_port/results/r122_archive/`.

## Full 30-scenario fp64 evaluations on the shipped 30 (SUPERSEDED 2026-11-07 — kept as a record)

**Superseded as a measurement on 2026-11-07.** These 30 are the training scenarios on training weathers, so the set is not a held-out evaluation; the numbers stay here because they were run and because the `pid_water` row pins the baseline. The measurement set is now 5 buildings × 18 held-out windows = 90 scenarios (next section up).

The ported fp64 evaluator and the official CPU pipeline both produce the same quantity: **one deterministic 8,640-step month on each of the 30 shipped scenarios, then the mean of −mean per-step reward**. Every row below is a complete 30/30 evaluation that exists on disk today; nothing in this table comes from a subset, a panel or a single month.

| policy / checkpoint | score | productivity | energy | carbon | worst scenario | file |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **`pid_water` (official CPU pass, 30/30, 2026-11-06)** | **0.05496** | 0.0581 | 0.0575 | 0.0428 | 0.2139 | `results/study_pid30/` |
| zero-residual baseline on the fp64 evaluator | 0.05491 | 0.0582 | 0.0574 | 0.0427 | 0.2138 | `results/eval/zero_resid_dev.json` |
| **r94b s75 @ 40,108,032** | **0.02588** | 0.0169 | 0.0390 | 0.0397 | 0.1022 | `results/study_r98_r99/r30_r94b_40M.json` |
| r110 @ 20,054,016 (its best save) | 0.07094 | 0.0738 | 0.0832 | 0.0500 | 0.1747 | `study_r110_conditional_beta/scores30/…20054016` |
| r110 @ 50,135,040 (final) | 0.08741 | 0.0982 | 0.0892 | 0.0533 | 0.1995 | `study_r110_conditional_beta/scores30/…50135040` |
| r52 @ 150,128,640 | 1.10604 | 1.8433 | 0.0000 | 0.0001 | 5.0719 | `results/study_r52/eval/r52_150128640.json` |
| r52 @ 300,257,280 | 1.10617 | 1.8430 | 0.0000 | 0.0001 | 5.0719 | `results/study_r52/eval/r52_300257280.json` |

**The baseline is pinned, and the two pipelines agree.** The fresh `pid_water` pass (official `evaluate.py`, numba, CPU) gives 0.05496 and the port's fp64 evaluator on the same controller (`zero_resid_dev`) gives 0.05491 — means within 5e-5, per-scenario max difference 3.96e-3 (`train_1/fairbanks_1` 0.21392 vs 0.21375). So the bar against which every local arm should be read is **0.0550**, not an inherited number.

**What that says about the best learned month we have measured on all 30.** r94b s75 at its best save is **0.02588**, i.e. **2.1× better than `pid_water`**, and it wins on every term: comfort cost 0.0169 against 0.0581 (3.4×), energy 0.0390 against 0.0575 (1.5×), carbon 0.0397 against 0.0428, and its worst month is half the baseline's (0.1022 against 0.2139). That is the honest local control; it is not a leaderboard number.

The rest of the table is the register's spread on the same 30 scenarios: r110's ten-save 30-scenario curve runs 0.27901, 0.10828, 0.07512, **0.07094**, 0.07434, 0.16000, 0.11352, 0.09623, 0.09079, 0.08741 from 5.0M to 50.1M (`scores30/`), so its 10-scenario panel best (0.06136) does not survive on the full set; and r52's two plant-off saves are 1.106 on both, identical to three decimals, which is the "no descent surface left" signature described in its own entry. The pre-r10 era, all on the same 30: `r2_s010` 0.04375 and `r2_final` 0.04457 were **residual controllers on top of the baseline action**, a construction the learning rule now bars, so they are recorded as history and not as a comparison; `r1_final` 0.27298, `r7_final` 0.56839 and `ppo_canon` 0.65113 are the flat-canonical attempts of that week.

**Excluded on purpose.** No single-month or four-scenario number is listed here, because one month is not a score: r116's four frozen months (fairbanks_1 0.07548, phoenix_6 0.02101, denver_2 0.01481, atlanta_6 0.01053) and r122's one-month checks (fairbanks_1 0.095508, atlanta_2 0.010009, held-out fairbanks_3 0.04998) stay in their own sections as diagnostics. **r122 has no local 30-scenario evaluation at all** — its protocol ran no scoring — so its only scores are the Kaggle hidden-set rows above (0.01818–0.02392), which are a different scenario set and cannot be subtracted from this table.

## Dedicated evaluation set — 5 buildings × 18 held-out windows = 90 scenarios (started 2026-11-07)

The 30-scenario table at the end of this file is superseded as a *measurement*: it re-uses the training scenarios and the training weather windows. This is the set the next comparison is read on, and it is the same set for every policy.

**Buildings — five, chosen for spread, not convenience.**

| building | zones | AHUs | zones/AHU | obs fields | cont / disc actions | floors | origin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `building_train_8` | 30 | 3 | 10.0 | 283 | 71 / 4 | 2 | shipped |
| `building_train_1` | 39 | 4 | 9.8 | 352 | 92 / 5 | 3 | shipped (most climate-sensitive: `pid_water` spread 0.189 across stations) |
| **`building_train_6`** | 43 | 6 | 7.2 | 406 | 106 / 7 | 2 | **new**, official floor generator seed 1012, exterior CV 0.260 |
| **`building_train_3`** | 70 | 7 | 10.0 | 583 | 163 / 8 | 3 | **new**, official floor generator seed 1018, exterior CV 0.520 |
| `building_train_2` | 85 | 9 | 9.4 | 703 | 199 / 10 | 2 | shipped (largest) |

That is a 2.8× span in zones, 3× in AHUs and 2.5× in observation width, and two of the five buildings have never been trained on. Both new buildings pass the canonical-slot fit (`_layout` reports 70 zones / 7 AHU and 43 zones / 6 AHU, every native action mapped, no fallback).

**Windows — the 18 that are not in the 30 training pairs:** `atlanta_4`, `atlanta_7`, `denver_1`, `denver_4`, `fairbanks_3`, `fairbanks_6`, `phoenix_2`, `phoenix_7`, `singapore_2`, `singapore_5`, plus **all eight `minneapolis_1..8`** (a station never seen in training). 90 scenario configs under `complete_port/artifacts/gen/heldout18/<building>/<weather>/`, 90 baked blobs in `complete_port/artifacts/sets/eval5_heldout18/`.

**Policies — five, each on all 90 months.**

| # | policy | checkpoint | how chosen | engine |
| --- | --- | --- | --- | --- |
| 1 | `pid_water` | — | the competition baseline | official CPU `evaluate.py` code path, 6+ parallel workers |
| 2 | r122 s75 | **119,439,360** | best `env/score` window mean (0.02404) | 3090, fp64, `SB_RECORD` |
| 3 | r122 s75 | **89,579,520** | lowest worst scenario (0.08497) | 3090, fp64, `SB_RECORD` |
| 4 | r116 s75 | **50,135,040** | best `env/score` (0.03375 window, 0.02493 last point) | 3090, fp64, `SB_RECORD` |
| 5 | r116 s75 | **45,121,536** | lowest worst scenario (0.07627) | 3090, fp64, `SB_RECORD` |

**Logging — what every learned month already writes, and what is produced from it.** The recorder (`SB_RECORD`, with `--base.cudagraphs=-1`) writes, per step and per scenario: `obs.bin` (the 1024-wide canonical observation: every zone's temperature, heating and cooling setpoints, damper, reheat, presence; every AHU field; the plant; the clock; occupancy), `loc.bin` (the 219 canonical actions actually executed), `parts.bin` (float64 productivity, energy cost, carbon, comfort deviation, worst-zone deviation, out-of-band fraction, electricity kWh, gas kWh), `rew.bin`, `done.bin`, `mask.bin`, `conc.bin` (Beta α+β per head) and `zone_feat.bin`. Month totals, 30 daily rows, seven day-parts (night / weekday morning / afternoon / evening / weekend day / empty / occupied), per-zone temperature and valve traces, and per-action distribution/rails/latching are all recovered offline from those files — no trainer change, so no new gate is needed for the logging.

**Gate and driver.** `tools/verify_all.sh --trainer` ran on the 3090 itself and ended **ALL CHECKS PASSED** (67 PASS, 0 FAIL, `results/verify_20261107_221257.log`). The scorer is `tools/score_eval90.py`: one part **per building** — the 18 weathers of a building are merged into one env group because `SB_RECORD` refuses more than one group — fp64, deterministic, `--base.cudagraphs=-1`, merged by `tools/merge_eval_json.py`. Records under `complete_port/results/study_r122_eval90/`.

**Results — five policies on the same 90 months (all verified, 2026-11-08 07:55 IST).**

| # | policy | all 90 | `train_8` | `train_1` | `train_6` (new) | `train_2` | `train_3` (new) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `pid_water` | 0.04609 | 0.03918 | 0.03538 | 0.04908 | 0.03792 | 0.06889 |
| 2 | r122 s75 @119,439,360 | 0.027229 | 0.014583 | 0.016841 | 0.020538 | 0.030384 | 0.053797 |
| 3 | r122 s75 @89,579,520 | **0.025315** | 0.01323 | 0.01603 | 0.01621 | 0.03141 | 0.04970 |
| 4 | r116 s75 @50,135,040 | 0.030123 | 0.029150 | 0.022381 | 0.032187 | 0.020539 | 0.046356 |
| 5 | r116 s75 @45,121,536 | 0.030705 | 0.02884 | 0.02212 | 0.03363 | 0.02027 | 0.04866 |

Every cell is 18 held-out months. The three terms for the same rows (90-month means):

| policy | productivity | energy | carbon |
| --- | ---: | ---: | ---: |
| `pid_water` | −0.05429 | 0.03908 | 0.02851 |
| r122 s75 @119,439,360 | −0.02717 | 0.02751 | 0.02713 |
| r122 s75 @89,579,520 | −0.02285 | 0.02971 | 0.02833 |
| r116 s75 @50,135,040 | −0.03119 | 0.03210 | 0.02494 |
| r116 s75 @45,121,536 | −0.03207 | 0.03263 | 0.02468 |

What the set says:

- **The learned policy beats `pid_water` on every building**: 0.0253–0.0272 against 0.0461. The gain is comfort and electricity, not a fuel swap: productivity improves from −0.0543 to −0.0229 and energy from 0.0391 to 0.0297, while carbon stays at 0.0283 against 0.0285.
- **r122 @89,579,520 is the best row**, and it is the checkpoint Kaggle also ranked best (0.01818). It wins `train_8`, `train_1`, `train_6` and `train_3`; it loses `train_2` to both r116 saves (0.03141 vs 0.02054). For the two r122 saves the 90-set order (0.0253 before 0.0272) agrees with the leaderboard order (0.01818 before 0.02104).
- **The new 70-zone `train_3` is the tail**: every policy's worst scenarios live there (best learned 0.0497, `pid_water` 0.0689). The 43-zone `train_6` is not a hard building for r122 (0.0162), so "new building" is not what makes a building hard; the tail is control coupling on the bigger plant.
- The two r116 saves are 0.0006 apart overall and trade `train_2`/`train_3` against `train_6`/`train_8`; on this set the r122 pair is 0.003–0.005 ahead.

Verification performed on every row:

- every merged JSON reports `complete = true`, 90 scenarios whose names are exactly the set directory, and a score equal to the mean of its five 18-scenario parts (<10⁻⁹);
- every checkpoint was re-hashed from the source file and matches the sha256 recorded at launch, and each eval log prints the zone-MLP + Beta + concentration-cap banner of the architecture it was trained with;
- 20 recordings (4 checkpoints × 5 buildings) are complete at 18 agents × 8640 steps (`obs` 637,009,920 B, `loc`/`conc` 179,159,040 B, `parts` 9,953,280 B, `rew`/`done` 622,080 B), and `reward = −(0.6·productivity + 0.2·energy + 0.2·carbon)` rebuilt from `parts.bin` matches `rew.bin` on every step to 4.6×10⁻⁷;
- all 90 `pid_water` months have 8640 steps, an 8640-row `reward_components.csv`, and a score recomputed from that CSV equal to the stored summary to <10⁻⁶;
- the **official Python simulator** re-scored `train_1/fairbanks_6` with r122 @119,439,360's weights: 0.004812 against the port's 0.00481093.


## r139 — Adam, no temporal coherence (30-scenario batch, 200M annealed) — 2026-11-09

Recipe: r116 + `adam=1, role_adam=0, beta_cn_tau=0, beta_cn_tau_ahu=0, value_delay=0,
beta_floor=0.5`, 960 agents (30×32), **199,895,040 steps annealed to zero**, seed 75, one Modal
L40S (profile `sid250581`). Six checkpoints: 39.8 / 79.6 / 119.4 / 159.3 / 199.1 / 199.9M.

**Held-out 90** (our set, `results/eval90_analysis/r139_199065600/`): r139 @199,065,600
**0.016514** against r122 @89,579,520 0.025315 (−34.8%) and `pid_water` 0.046092 (−64.2%). All
three terms and all five buildings improve; both fuels fall (CO₂ 20.6 vs 24.6 t). The 54-scenario
extension to the three missing shipped buildings (`results/study_eval457/COMPARE.md`) is also
r139-favoured, 0.009793 vs 0.019217 (−49%), 54/54 scenarios.

**Leaderboard (public):**

| checkpoint | training panel mean / worst | held-out 90 | public score |
| --- | --- | --- | --- |
| **159,252,480** | 0.0129 / 0.0702 | 0.018530 | **0.01750** |
| 199,065,600 | 0.0109 / 0.0545 | 0.016514 | 0.02944 |
| r122 @89,579,520 (reference) | 0.0210 / 0.135 | 0.025315 | 0.01818 |
| r122 @119,439,360 (reference) | — | 0.027229 | 0.02104 |

The 90-set ranks 199 > 159 > r122@89.6 > r122@119.4; Kaggle ranks **159 > r122@89.6 >
r122@119.4 > 199**. The two r122 saves order the same way on both, the two r139 saves invert, and
the 159 save is the only checkpoint that beats every r122 save on **both** measures (90: 0.0185 vs
0.0253; Kaggle: 0.01750 vs 0.01818).

**Extension to the other three shipped buildings** (`results/study_eval457/`): the same 18 windows on
`train_4/5/7`, 54 more fp64 months. r139 @199,065,600 **0.009793**, r139 @159,252,480 0.011194,
r122 @89,579,520 0.019217, r122 @119,439,360 0.020749, `pid_water` **0.043733**. The 54-set orders
the four checkpoints exactly as the 90-set does (199.1 < 159.3 < r122@89.6 < r122@119.4), so the
missing buildings do not explain the leaderboard inversion either: 199.1M is 12.5% better than
159.3M there too. On all 144 months: r139@199.1 0.013994, r139@159.3 0.015779, r122@89.6 0.023029,
r122@119.4 0.024799, `pid_water` 0.045207 — the learned policy is 69% below the baseline.

The baseline was submitted to the leaderboard as a reference (kernel
`sidharth1743/submission-pid-water`, ref 58184591). The packaged policy is a verbatim copy of
`pid.py` + `pid_plus.py` and reproduces the local month on `train_4/atlanta_4` (0.026300) and the
public `train_1/atlanta_2` (−0.0249 kernel vs 0.02491 local).

Completeness of the table: it carries each run's best held-out and best Kaggle save — r139
@199.1M (best 90) and @159.3M (best Kaggle, and best overall), r122 @89.6M (best of both). The
caveat is on r122's side: it has five Kaggle rows (59,719,680 0.01842; **89,579,520 0.01818**;
119,439,360 0.02104; 209,018,880 0.02262; 238,878,720 0.02392) but only two were scored on the
90-set, so r122's best held-out is the best *of the evaluated saves*, not of all five.

**Both local selectors inverted.** 199.1M is better on the training panel (lowest mean *and*
lowest worst) and on the held-out 90 (0.016514 vs 0.018530; better on 88/90 scenarios and on all
five buildings), yet 159.3M is better on the hidden 48 (0.01750 vs 0.02944) and is the **best
public entry**, ahead of r122 @89,579,520 (0.01818). "Lowest worst" ordered r122's five saves
exactly, but neither the panel nor our 90 transfers across the r139 saves. On a 48-scenario mean
the 0.0119 gap is one hidden scenario at ≈0.57: the hidden set contains a condition absent from
all 144 local months, where the 199.1M save collapses and the mid-anneal save does not.

The 199.1M row was not a submission error: the CPU Kaggle policy reproduces the 3090 recordings on
`train_4/minneapolis_1` (0.02557 vs 0.02552) and `train_7/minneapolis_1` (0.05681 vs 0.05710), and
the commit's `atlanta_2` matches the local month (−0.0041 vs −0.00406). The gap was the save.
