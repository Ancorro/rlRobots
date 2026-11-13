# OpenHLM-data run `20261104-215355`

2026-11-04 22:16, host <node>.<node> (NVIDIA A40), 1375 s of a 26 min budget, devices cuda:0, seed 0, cameras head_image_left, candidates ens, mlp-cam, mlp-wide, ridge.

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
| g1_HLM-12_full_teleop | ens **(selected)** | 62/71 | 0.551 | 0 | 0 |
| g1_HLM-12_full_teleop | mlp-cam | 59/71 | 0.542 | 7739 | 177.8 |
| g1_HLM-12_full_teleop | mlp-wide | 56/71 | 0.537 | 20452 | 71.9 |
| g1_HLM-12_full_teleop | ridge | 0/71 | 0.184 | 0 | 5.6 |
| g1_HLM-12_humi | mlp-cam **(selected)** | 48/71 | 0.518 | 8244 | 182.8 |
| g1_HLM-12_humi | ens | 47/71 | 0.526 | 0 | 0 |
| g1_HLM-12_humi | mlp-wide | 46/71 | 0.511 | 20881 | 72.9 |
| g1_HLM-12_humi | ridge | 0/71 | 0.166 | 0 | 4.9 |
| g1_HLM-12_stationary_teleop | ens **(selected)** | 42/71 | 0.512 | 0 | 0 |
| g1_HLM-12_stationary_teleop | mlp-cam | 42/71 | 0.505 | 8382 | 188.7 |
| g1_HLM-12_stationary_teleop | mlp-wide | 37/71 | 0.494 | 21491 | 74.9 |
| g1_HLM-12_stationary_teleop | ridge | 0/71 | 0.154 | 0 | 5.3 |
| long_g1_5_fruits_full_teleop | ens **(selected)** | 11/17 | 0.532 | 0 | 0 |
| long_g1_5_fruits_full_teleop | mlp-cam | 11/17 | 0.527 | 9331 | 196.9 |
| long_g1_5_fruits_full_teleop | mlp-wide | 9/17 | 0.510 | 22006 | 75.9 |
| long_g1_5_fruits_full_teleop | ridge | 0/17 | 0.255 | 0 | 3.8 |
| long_g1_5_fruits_humi | ens **(selected)** | 19/29 | 0.550 | 0 | 0 |
| long_g1_5_fruits_humi | mlp-cam | 19/29 | 0.543 | 10441 | 214.9 |
| long_g1_5_fruits_humi | mlp-wide | 19/29 | 0.542 | 23122 | 78.0 |
| long_g1_5_fruits_humi | ridge | 1/29 | 0.206 | 0 | 2.3 |

## Test (run #1)

**14/20 episodes succeed (70%, Wilson 95 % 0.48-0.85)**; mean skill 0.540, median 0.538; at skill >= 0.25: 20, 0.5: 14, 0.75: 0.

| set | candidate | test successes | mean skill | hold map (action dim -> state dim, -1 = last command) |
|---|---|---|---|---|
| g1_HLM-12_full_teleop | ens | 3/4 | 0.568 | 8/34 inferred: [0, -1, -1, 3, -1, -1, -1, 7, 8, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 28, -1, -1, -1, -1, 33] |
| g1_HLM-12_humi | mlp-cam | 2/4 | 0.507 | 14/34 inferred: [0, -1, 2, 3, 4, -1, -1, 7, 8, -1, 10, 11, 12, -1, 14, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, 28, 29, -1, -1, -1, 33] |
| g1_HLM-12_stationary_teleop | ens | 4/4 | 0.524 | 10/34 inferred: [0, -1, -1, 3, -1, -1, -1, 7, 8, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, 28, 29, -1, -1, -1, 33] |
| long_g1_5_fruits_full_teleop | ens | 2/4 | 0.500 | 9/34 inferred: [0, -1, -1, -1, -1, -1, -1, 7, 8, -1, -1, 11, -1, -1, -1, 30, -1, -1, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, 28, 29, -1, -1, -1, 33] |
| long_g1_5_fruits_humi | ens | 3/4 | 0.598 | 13/34 inferred: [0, -1, -1, -1, 4, -1, -1, 7, 8, -1, 10, 11, -1, -1, -1, 30, 16, -1, -1, -1, 20, -1, -1, -1, -1, 21, -1, -1, 28, 29, -1, -1, -1, 33] |

Reference `ridge` on the same episodes: 1/20 succeed, mean skill 0.233; selected policy better on 20/20 episodes by skill; only-selected 13, only-ridge 0, McNemar p = 0.000244.

Reference `hold` on the same episodes: 0/20 succeed, mean skill 0.000; selected policy better on 20/20 episodes by skill; only-selected 14, only-hold 0, McNemar p = 0.000122.

| set | episode | policy | success | skill | error | error hold | chunks | length |
|---|---|---|---|---|---|---|---|---|
| g1_HLM-12_full_teleop | 267 | ens | yes | 0.545 | 0.1249 | 0.2743 | 39 | 1189 |
| g1_HLM-12_full_teleop | 300 | ens | yes | 0.647 | 0.1278 | 0.3624 | 22 | 686 |
| g1_HLM-12_full_teleop | 119 | ens | no | 0.473 | 0.1822 | 0.3461 | 25 | 782 |
| g1_HLM-12_full_teleop | 120 | ens | yes | 0.607 | 0.1359 | 0.3455 | 34 | 1040 |
| g1_HLM-12_humi | 235 | mlp-cam | yes | 0.531 | 0.1560 | 0.3326 | 39 | 1200 |
| g1_HLM-12_humi | 62 | mlp-cam | yes | 0.566 | 0.1797 | 0.4140 | 44 | 1352 |
| g1_HLM-12_humi | 353 | mlp-cam | no | 0.483 | 0.1208 | 0.2337 | 11 | 356 |
| g1_HLM-12_humi | 164 | mlp-cam | no | 0.449 | 0.1812 | 0.3286 | 53 | 1620 |
| g1_HLM-12_stationary_teleop | 137 | ens | yes | 0.538 | 0.1540 | 0.3336 | 40 | 1216 |
| g1_HLM-12_stationary_teleop | 180 | ens | yes | 0.525 | 0.1790 | 0.3767 | 55 | 1683 |
| g1_HLM-12_stationary_teleop | 59 | ens | yes | 0.522 | 0.1656 | 0.3465 | 46 | 1413 |
| g1_HLM-12_stationary_teleop | 121 | ens | yes | 0.512 | 0.1632 | 0.3345 | 37 | 1137 |
| long_g1_5_fruits_full_teleop | 104 | ens | yes | 0.555 | 0.1340 | 0.3010 | 72 | 2177 |
| long_g1_5_fruits_full_teleop | 71 | ens | no | 0.463 | 0.1455 | 0.2707 | 88 | 2652 |
| long_g1_5_fruits_full_teleop | 27 | ens | yes | 0.539 | 0.1595 | 0.3459 | 67 | 2025 |
| long_g1_5_fruits_full_teleop | 24 | ens | no | 0.445 | 0.1868 | 0.3365 | 70 | 2110 |
| long_g1_5_fruits_humi | 108 | ens | yes | 0.623 | 0.1154 | 0.3059 | 7 | 222 |
| long_g1_5_fruits_humi | 85 | ens | yes | 0.627 | 0.1190 | 0.3192 | 6 | 205 |
| long_g1_5_fruits_humi | 22 | ens | no | 0.478 | 0.1711 | 0.3276 | 86 | 2608 |
| long_g1_5_fruits_humi | 76 | ens | yes | 0.664 | 0.0983 | 0.2925 | 5 | 177 |
