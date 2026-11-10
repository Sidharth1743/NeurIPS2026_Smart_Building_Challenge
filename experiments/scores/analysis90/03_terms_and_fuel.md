> The CSV tables this page names are rebuilt by the scripts in this folder from the
> checkpoint's per-step recording, which is 7.6 GB and is not shipped. The scores themselves
> are in `../panel90_r139/` and `../panel54/`.

# 03 — Terms and fuel

Terms are the competition's normalised quantities; `score = −0.6·productivity + 0.2·energy + 0.2·carbon`. Two extra physical units are derived so the terms can be read as a building result, not only as a number:

- **Electricity and gas in MWh per month**, straight from the recorder (`elec_kwh`, `gas_kwh`).
- **CO₂ in kg per month**, from the per-step kWh times the competition's own factors: electricity 87.8–104.7 g/kWh depending on the hour (`CARBON_EMISSION_BY_HOUR`), gas 0.1813 kg/kWh. The baseline stores money rather than kWh, so its kWh are recovered from the recorded cost and the price schedule (`WEEKDAY_PRICE_BY_HOUR`, `WEEKEND_PRICE_BY_HOUR`, `GAS_PRICE_BY_MONTH`); that reconstruction reproduces its `carbon_emitted` to <10⁻⁶ kg, so both policies are measured with the same factors.

## Comfort is where the difference is

| building | zones | productivity | pid | less regret | in-band (ours) | in-band (pid) | comfort dev K | worst-zone p99 K | out of band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 30 | -0.0044 | -0.0395 | +89% | 40.4% | 2.7% | 0.799 | 16.6 | 26.1% |
| building_train_1 | 39 | -0.0090 | -0.0368 | +76% | 15.1% | 3.0% | 0.414 | 12.5 | 23.3% |
| building_train_6 | 43 | -0.0111 | -0.0660 | +83% | 16.7% | 1.1% | 0.407 | 12.5 | 23.9% |
| building_train_3 | 70 | -0.0528 | -0.0804 | +34% | 0.3% | 0.1% | 0.953 | 15.8 | 44.6% |
| building_train_2 | 85 | -0.0369 | -0.0487 | +24% | 2.2% | 0.5% | 0.821 | 18.3 | 47.0% |

`in-band` is the share of **occupied** steps on which *every* occupied zone is inside the comfort band (the productivity regret is exactly zero). It is a stricter and more legible measure than the mean regret: the learned policy takes `train_8` from 2.7% to 40.4% and `train_1` from 3.0% to 15.1% of occupied steps, and `train_6` from 1.1% to 16.7%. The 70-zone building stays effectively never-in-band for every policy (0.3% vs 0.1%): one zone out of band spoils the whole step.

## Fuel: both kWh and kg, for both policies

| building | elec MWh ours | elec MWh pid | gas MWh ours | gas MWh pid | elec share ours | elec share pid | kg CO₂ ours | kg CO₂ pid | kg change | kg/MWh ours | kg/MWh pid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 3.4 | 6.2 | 13.7 | 17.6 | 20% | 26% | 2830 | 3802 | -26% | 165 | 160 |
| building_train_1 | 5.2 | 7.5 | 17.2 | 19.3 | 23% | 28% | 3635 | 4238 | -14% | 162 | 158 |
| building_train_6 | 9.0 | 11.9 | 18.9 | 12.6 | 32% | 49% | 4310 | 3453 | +25% | 155 | 141 |
| building_train_3 | 23.4 | 33.4 | 33.9 | 24.6 | 41% | 58% | 8484 | 7766 | +9% | 148 | 134 |
| building_train_2 | 9.9 | 14.3 | 34.6 | 21.5 | 22% | 40% | 7224 | 5304 | +36% | 162 | 148 |

Monthly totals per scenario (mean over the 18 windows). The two small buildings cut **both** fuels (electricity -45% and -31%, gas -22% and -11%): efficiency. The three large plants cut electricity (-31% to -24%) and **raise gas (+38% to +61%)**: the plant is re-balanced from the electric side to the boiler.

### Why the carbon term is flat overall but worse on three buildings

- In this model gas emits **0.1813 kg/kWh** and the grid **0.0878–0.1047 kg/kWh**, so a shift from electricity to gas raises kg per unit of heat. That is exactly what happens on `train_6`, `train_3` and `train_2`: kg rises 25%, 9% and 36%, and the carbon term moves with it (+25%, +9%, +36%).
- On `train_8` and `train_1` both fuels fall, so kg falls 26% and 14% and the carbon term improves. The aggregate carbon term is flat (0.02851 → 0.02833) because the two small buildings offset the three large ones; it is a cancellation, not an absence of fuel shifting.
- The clean way to see the mix is `kg per MWh`: the learned policy is 148–165 kg/MWh on every building, and the baseline is a few points lower on each, because its mix is more electric.
- This does not reverse the score: carbon is 22% of the score and moves by 0.0002, while comfort moves by 0.019 and energy by 0.002. It is the one term where the learned policy is worse, and it belongs in the paper with the same weight as the wins.
- **The aggregation matters, and both numbers should be quoted.** The competition score is the mean of the per-scenario normalised carbon, which is 0.02833 against 0.02851 (−0.6%). Summing the physical kg over the five buildings instead gives 26.5 t against 24.6 t (**+8%**), because the mean weights every scenario equally while the total is dominated by the two large plants, and those are exactly where the shift to gas raises emissions. The score is the competition number; the tonne total is the physical footprint.

## Do the three terms move together?

| building | Δ productivity (+ = better) | Δ energy (− = better) | Δ carbon (− = better) | all three improve? |
| --- | --- | --- | --- | --- |
| building_train_8 | 0.0351 | -0.0144 | -0.0101 | yes |
| building_train_1 | 0.0278 | -0.0085 | -0.0048 | yes |
| building_train_6 | 0.0548 | -0.0043 | 0.0044 | no |
| building_train_3 | 0.0277 | -0.0160 | 0.0031 | no |
| building_train_2 | 0.0118 | -0.0037 | 0.0066 | no |

The comfort and energy terms improve on all five buildings. The carbon term improves on two. No building loses comfort or energy, so the win is not a reallocation of harm between buildings.

## What this file adds beyond the score table

- **In-band share of occupied steps**, the legible comfort measure the paper can quote.
- **MWh, fuel shares and kg CO₂ per month**, so the terms are physical, not only normalised.
- **The per-building carbon reversal**: the aggregate flat carbon is two buildings improving and three getting worse, which the mean alone hides.
- **The exact fuel split for both policies** (the baseline in kWh recovered from its own cost, validated against its recorded kg), which is what turns "carbon flat" into "gas for electricity on the big plants".
- **The aggregation trap**: the mean normalised carbon is −0.6% while the physical total is +8%, and both belong in the paper.

