# 02 — Terms and fuel: r139 @199,065,600

Normalised terms, then the physical units derived from the same recording: electricity and gas MWh per month from `elec_kwh`/`gas_kwh`, and CO₂ kg from the competition factors (electricity 87.8–104.7 g/kWh by hour, gas 0.1813 kg/kWh). The baseline is converted from its recorded cost with the price schedule, which reproduces its `carbon_emitted` exactly.

| building | productivity | energy | carbon | comfort dev K | worst p99 K | out of band | in band (all) | in band (occupied) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | -0.0008 | 0.0186 | 0.0213 | 0.833 | 18.3 | 21.1% | 78.9% | 98.4% |
| building_train_1 | -0.0009 | 0.0217 | 0.0225 | 0.331 | 11.9 | 14.3% | 85.7% | 98.3% |
| building_train_6 | -0.0071 | 0.0230 | 0.0189 | 0.399 | 11.8 | 23.3% | 76.7% | 84.4% |
| building_train_3 | -0.0472 | 0.0473 | 0.0325 | 0.997 | 15.6 | 51.4% | 48.6% | 47.1% |
| building_train_2 | -0.0022 | 0.0165 | 0.0159 | 0.674 | 18.6 | 20.1% | 79.9% | 97.4% |
| **all** | -0.0116 | 0.0254 | 0.0222 | 0.647 |  |  |  |  |

`in band` counts zone-steps inside the band (`max(Hs−T,0)+max(T−Cs,0)=0`), decoded from the canonical observation; the occupied column uses the building occupancy mask. **The all-steps column is not scored**: it includes the empty-building free float, which the reward does not bill. The occupied column is the one to compare with the productivity term. The order matches the score: `train_3` has the largest deviation and the lowest in-band share.

| building | elec r139 | elec pid | gas r139 | gas pid | kg r139 | kg pid | kg/MWh r139 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| building_train_8 | 2.8 | 6.2 | 9.8 | 17.6 | 2064 | 3802 | 163 |
| building_train_1 | 4.7 | 7.5 | 13.5 | 19.3 | 2916 | 4238 | 160 |
| building_train_6 | 8.3 | 11.9 | 15.7 | 12.6 | 3668 | 3453 | 153 |
| building_train_3 | 21.1 | 33.4 | 29.1 | 24.6 | 7367 | 7766 | 147 |
| building_train_2 | 8.3 | 14.3 | 21.0 | 21.5 | 4622 | 5304 | 158 |

Electricity is lower on every building (45.2 vs 73.3 MWh total). Gas is lower on 3 of 5 (89.1 vs 95.6 total); it is higher on building_train_6, building_train_3. CO₂ is lower on 4 of 5 (higher on building_train_6), 20636 vs 24563 kg in total. The net effect is a real efficiency gain, not the electricity→gas trade the r122 analysis found (there gas rose on the large plants and carbon flattened).

## What this file adds

- The checkpoint's comfort distribution (deviation, worst zone, in-band share) and its fuel and CO₂ in physical units, per building and against the baseline.
- The fuel comparison is derived from the baseline's own cost and price schedule, so both sides are measured with the same factors.

