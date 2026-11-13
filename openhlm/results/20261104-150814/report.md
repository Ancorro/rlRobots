# OpenHLM-data run `20261104-150814`

2026-11-04 15:31, host <node>.<node> (Quadro RTX 8000), 1367 s of a 26 min budget, devices cuda:0, seed 0, cameras none (state only), candidates mlp-wide, mlp, ridge.

Contract: openhlm/spec.py (history 0.33 s, horizon 1.0 s, success = skill >= 0.50 against the hold baseline).

## Split

| set | train | val | test |
|---|---|---|---|
| g1_HLM-12_full_teleop | 405 | 71 | 4 |
| g1_HLM-12_humi | 405 | 71 | 4 |
| g1_HLM-12_stationary_teleop | 405 | 71 | 4 |
| long_g1_5_fruits_full_teleop | 99 | 17 | 4 |
| long_g1_5_fruits_humi | 166 | 29 | 4 |

## Validation (selection)

| set | candidate | val successes | mean skill | steps | s |
|---|---|---|---|---|---|
| g1_HLM-12_full_teleop | mlp-wide **(selected)** | 57/71 | 0.536 | 15036 | 128.8 |
| g1_HLM-12_full_teleop | mlp | 54/71 | 0.518 | 8874 | 117.7 |
| g1_HLM-12_full_teleop | ridge | 0/71 | 0.184 | 0 | 102.9 |
| g1_HLM-12_humi | mlp **(selected)** | 46/71 | 0.507 | 9874 | 119.8 |
| g1_HLM-12_humi | mlp-wide | 45/71 | 0.516 | 18260 | 132.8 |
| g1_HLM-12_humi | ridge | 0/71 | 0.166 | 0 | 92.0 |
| g1_HLM-12_stationary_teleop | mlp-wide **(selected)** | 39/71 | 0.496 | 17977 | 137.7 |
| g1_HLM-12_stationary_teleop | mlp | 30/71 | 0.478 | 5520 | 120.5 |
| g1_HLM-12_stationary_teleop | ridge | 0/71 | 0.154 | 0 | 113.8 |
| long_g1_5_fruits_full_teleop | mlp-wide **(selected)** | 9/17 | 0.508 | 26484 | 145.9 |
| long_g1_5_fruits_full_teleop | mlp | 9/17 | 0.497 | 25554 | 122.9 |
| long_g1_5_fruits_full_teleop | ridge | 0/17 | 0.255 | 0 | 52.2 |
| long_g1_5_fruits_humi | mlp-wide **(selected)** | 19/29 | 0.539 | 33951 | 161.9 |
| long_g1_5_fruits_humi | mlp | 17/29 | 0.525 | 35847 | 125.9 |
| long_g1_5_fruits_humi | ridge | 1/29 | 0.206 | 0 | 38.3 |

## Test (scored once)

**14/20 episodes succeed (70%, Wilson 95 % 0.48-0.85)**; mean skill 0.529, median 0.523; at skill >= 0.25: 20, 0.5: 14, 0.75: 0.

| set | candidate | test successes | mean skill |
|---|---|---|---|
| g1_HLM-12_full_teleop | mlp-wide | 3/4 | 0.563 |
| g1_HLM-12_humi | mlp | 2/4 | 0.493 |
| g1_HLM-12_stationary_teleop | mlp-wide | 4/4 | 0.520 |
| long_g1_5_fruits_full_teleop | mlp-wide | 2/4 | 0.480 |
| long_g1_5_fruits_humi | mlp-wide | 3/4 | 0.590 |

Reference `ridge` on the same episodes: 1/20 succeed, mean skill 0.233; selected policy better on 20/20 episodes by skill; only-selected 13, only-ridge 0, McNemar p = 0.000244.

Reference `hold` on the same episodes: 0/20 succeed, mean skill 0.000; selected policy better on 20/20 episodes by skill; only-selected 14, only-hold 0, McNemar p = 0.000122.

| set | episode | policy | success | skill | error | error hold | chunks | length |
|---|---|---|---|---|---|---|---|---|
| g1_HLM-12_full_teleop | 267 | mlp-wide | yes | 0.553 | 0.1226 | 0.2743 | 39 | 1189 |
| g1_HLM-12_full_teleop | 300 | mlp-wide | yes | 0.640 | 0.1305 | 0.3624 | 22 | 686 |
| g1_HLM-12_full_teleop | 119 | mlp-wide | no | 0.469 | 0.1839 | 0.3461 | 25 | 782 |
| g1_HLM-12_full_teleop | 120 | mlp-wide | yes | 0.591 | 0.1414 | 0.3455 | 34 | 1040 |
| g1_HLM-12_humi | 235 | mlp | yes | 0.505 | 0.1646 | 0.3326 | 39 | 1200 |
| g1_HLM-12_humi | 62 | mlp | yes | 0.564 | 0.1807 | 0.4140 | 44 | 1352 |
| g1_HLM-12_humi | 353 | mlp | no | 0.467 | 0.1247 | 0.2337 | 11 | 356 |
| g1_HLM-12_humi | 164 | mlp | no | 0.438 | 0.1846 | 0.3286 | 53 | 1620 |
| g1_HLM-12_stationary_teleop | 137 | mlp-wide | yes | 0.524 | 0.1587 | 0.3336 | 40 | 1216 |
| g1_HLM-12_stationary_teleop | 180 | mlp-wide | yes | 0.513 | 0.1835 | 0.3767 | 55 | 1683 |
| g1_HLM-12_stationary_teleop | 59 | mlp-wide | yes | 0.517 | 0.1673 | 0.3465 | 46 | 1413 |
| g1_HLM-12_stationary_teleop | 121 | mlp-wide | yes | 0.525 | 0.1590 | 0.3345 | 37 | 1137 |
| long_g1_5_fruits_full_teleop | 104 | mlp-wide | yes | 0.522 | 0.1438 | 0.3010 | 72 | 2177 |
| long_g1_5_fruits_full_teleop | 71 | mlp-wide | no | 0.440 | 0.1516 | 0.2707 | 88 | 2652 |
| long_g1_5_fruits_full_teleop | 27 | mlp-wide | yes | 0.523 | 0.1651 | 0.3459 | 67 | 2025 |
| long_g1_5_fruits_full_teleop | 24 | mlp-wide | no | 0.434 | 0.1904 | 0.3365 | 70 | 2110 |
| long_g1_5_fruits_humi | 108 | mlp-wide | yes | 0.626 | 0.1143 | 0.3059 | 7 | 222 |
| long_g1_5_fruits_humi | 85 | mlp-wide | yes | 0.629 | 0.1185 | 0.3192 | 6 | 205 |
| long_g1_5_fruits_humi | 22 | mlp-wide | no | 0.458 | 0.1775 | 0.3276 | 86 | 2608 |
| long_g1_5_fruits_humi | 76 | mlp-wide | yes | 0.647 | 0.1031 | 0.2925 | 5 | 177 |
