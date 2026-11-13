# OpenHLM-data run `20261104-225700`

2026-11-04 23:20, host <node>.<node> (NVIDIA A40), 1387 s of a 26 min budget, devices cuda:0, seed 0, cameras head_image_left,left_wrist_image,right_wrist_image, candidates ens, mlp-wide, mlp-cam, ridge.

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

| set | candidate | val successes | mean skill | steps (best) | s | cameras |
|---|---|---|---|---|---|---|
| g1_HLM-12_full_teleop | ens **(selected)** | 60/71 | 0.538 | 0 (0) | 0 | - |
| g1_HLM-12_full_teleop | mlp-wide | 56/71 | 0.535 | 20619 (20619) still improving | 73.9 | - |
| g1_HLM-12_full_teleop | mlp-cam | 33/71 | 0.489 | 1270 (1270) still improving | 109.7 | head_image_left,left_wrist_image,right_wrist_image |
| g1_HLM-12_full_teleop | ridge | 0/71 | 0.184 | 0 (0) | 6.0 | - |
| g1_HLM-12_humi | mlp-wide **(selected)** | 46/71 | 0.511 | 21167 (18274) | 74.9 | - |
| g1_HLM-12_humi | ens | 43/71 | 0.504 | 0 (0) | 0 | - |
| g1_HLM-12_humi | mlp-cam | 23/71 | 0.432 | 684 (684) still improving | 107.7 | head_image_left,left_wrist_image,right_wrist_image |
| g1_HLM-12_humi | ridge | 0/71 | 0.166 | 0 (0) | 5.3 | - |
| g1_HLM-12_stationary_teleop | ens **(selected)** | 41/71 | 0.511 | 0 (0) | 0 | - |
| g1_HLM-12_stationary_teleop | mlp-wide | 38/71 | 0.494 | 21663 (18244) | 76.9 | - |
| g1_HLM-12_stationary_teleop | mlp-cam | 33/71 | 0.484 | 2295 (2295) still improving | 148.6 | head_image_left,left_wrist_image,right_wrist_image |
| g1_HLM-12_stationary_teleop | ridge | 0/71 | 0.154 | 0 (0) | 5.7 | - |
| long_g1_5_fruits_full_teleop | ens **(selected)** | 12/17 | 0.534 | 0 (0) | 0 | - |
| long_g1_5_fruits_full_teleop | mlp-cam | 12/17 | 0.523 | 7737 (4729) | 250.8 | head_image_left,left_wrist_image,right_wrist_image |
| long_g1_5_fruits_full_teleop | mlp-wide | 9/17 | 0.510 | 22279 (18211) | 77.9 | - |
| long_g1_5_fruits_full_teleop | ridge | 0/17 | 0.255 | 0 (0) | 3.9 | - |
| long_g1_5_fruits_humi | ens **(selected)** | 19/29 | 0.553 | 0 (0) | 0 | - |
| long_g1_5_fruits_humi | mlp-wide | 19/29 | 0.542 | 23394 (18453) | 80.0 | - |
| long_g1_5_fruits_humi | mlp-cam | 19/29 | 0.541 | 12049 (9471) | 337.8 | head_image_left,left_wrist_image,right_wrist_image |
| long_g1_5_fruits_humi | ridge | 1/29 | 0.206 | 0 (0) | 2.4 | - |

## Where the error is (selected policy on validation, normalised absolute error, hold in brackets)

| set | step 1 | 5 | 10 | 15 | 20 | 25 | last | dims with the largest error (dim: policy [hold], * = held from state) |
|---|---|---|---|---|---|---|---|---|
| g1_HLM-12_full_teleop | 0.043 [0.086] | 0.093 [0.180] | 0.133 [0.274] | 0.159 [0.346] | 0.182 [0.401] | 0.201 [0.445] | 0.222 [0.477] | 31: 0.25 [0.54], 17: 0.22 [0.50], 29: 0.21 [0.34], 27: 0.21 [0.54], 21: 0.21 [0.51], 20: 0.19 [0.46], 16: 0.19 [0.47], 19: 0.19 [0.40] |
| g1_HLM-12_humi | 0.065 [0.146] | 0.106 [0.221] | 0.142 [0.294] | 0.169 [0.355] | 0.192 [0.406] | 0.212 [0.450] | 0.231 [0.483] | 29*: 0.26 [0.49], 31: 0.23 [0.44], 19*: 0.22 [0.83], 17: 0.21 [0.44], 10*: 0.21 [0.52], 27: 0.20 [0.45], 21: 0.20 [0.44], 8*: 0.19 [0.43] |
| g1_HLM-12_stationary_teleop | 0.050 [0.110] | 0.092 [0.184] | 0.129 [0.257] | 0.156 [0.319] | 0.179 [0.371] | 0.199 [0.411] | 0.218 [0.445] | 29*: 0.26 [0.51], 31: 0.25 [0.47], 17: 0.21 [0.42], 27: 0.21 [0.46], 21: 0.21 [0.42], 19*: 0.21 [0.70], 20: 0.18 [0.37], 16: 0.18 [0.38] |
| long_g1_5_fruits_full_teleop | 0.045 [0.127] | 0.086 [0.193] | 0.121 [0.259] | 0.148 [0.316] | 0.171 [0.363] | 0.192 [0.401] | 0.211 [0.434] | 31: 0.28 [0.54], 17: 0.23 [0.44], 33*: 0.23 [0.50], 28*: 0.23 [0.43], 29*: 0.22 [0.41], 18: 0.22 [0.36], 23: 0.20 [0.41], 0*: 0.19 [0.38] |
| long_g1_5_fruits_humi | 0.051 [0.134] | 0.088 [0.184] | 0.122 [0.238] | 0.147 [0.284] | 0.167 [0.323] | 0.186 [0.358] | 0.203 [0.390] | 10*: 0.24 [0.39], 31: 0.23 [0.38], 29*: 0.23 [0.33], 33*: 0.23 [0.38], 17: 0.20 [0.29], 0*: 0.20 [0.34], 23: 0.19 [0.29], 18: 0.18 [0.24] |

## Test (run #1)

**15/20 episodes succeed (75%, Wilson 95 % 0.53-0.89)**; mean skill 0.539, median 0.537; at skill >= 0.25: 20, 0.5: 15, 0.75: 0.

| set | candidate | test successes | mean skill | hold map (action dim -> state dim, -1 = last command) |
|---|---|---|---|---|
| g1_HLM-12_full_teleop | ens | 3/4 | 0.550 | 8/34 inferred: [0, -1, -1, 3, -1, -1, -1, 7, 8, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 28, -1, -1, -1, -1, 33] |
| g1_HLM-12_humi | mlp-wide | 3/4 | 0.510 | 14/34 inferred: [0, -1, 2, 3, 4, -1, -1, 7, 8, -1, 10, 11, 12, -1, 14, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, 28, 29, -1, -1, -1, 33] |
| g1_HLM-12_stationary_teleop | ens | 4/4 | 0.531 | 10/34 inferred: [0, -1, -1, 3, -1, -1, -1, 7, 8, -1, 10, 11, -1, -1, -1, -1, -1, -1, -1, 23, -1, -1, -1, -1, -1, -1, -1, -1, 28, 29, -1, -1, -1, 33] |
| long_g1_5_fruits_full_teleop | ens | 2/4 | 0.505 | 9/34 inferred: [0, -1, -1, -1, -1, -1, -1, 7, 8, -1, -1, 11, -1, -1, -1, 30, -1, -1, -1, 21, -1, -1, -1, -1, -1, -1, -1, -1, 28, 29, -1, -1, -1, 33] |
| long_g1_5_fruits_humi | ens | 3/4 | 0.599 | 13/34 inferred: [0, -1, -1, -1, 4, -1, -1, 7, 8, -1, 10, 11, -1, -1, -1, 30, 16, -1, -1, -1, 20, -1, -1, -1, -1, 21, -1, -1, 28, 29, -1, -1, -1, 33] |

Reference `ridge` on the same episodes: 1/20 succeed, mean skill 0.233; selected policy better on 20/20 episodes by skill; only-selected 14, only-ridge 0, McNemar p = 0.000122.

Reference `hold` on the same episodes: 0/20 succeed, mean skill 0.000; selected policy better on 20/20 episodes by skill; only-selected 15, only-hold 0, McNemar p = 6.1e-05.

| set | episode | policy | success | skill | error | error hold | chunks | length |
|---|---|---|---|---|---|---|---|---|
| g1_HLM-12_full_teleop | 267 | ens | yes | 0.544 | 0.1250 | 0.2743 | 39 | 1189 |
| g1_HLM-12_full_teleop | 300 | ens | yes | 0.612 | 0.1407 | 0.3624 | 22 | 686 |
| g1_HLM-12_full_teleop | 119 | ens | no | 0.440 | 0.1939 | 0.3461 | 25 | 782 |
| g1_HLM-12_full_teleop | 120 | ens | yes | 0.604 | 0.1369 | 0.3455 | 34 | 1040 |
| g1_HLM-12_humi | 235 | mlp-wide | yes | 0.508 | 0.1635 | 0.3326 | 39 | 1200 |
| g1_HLM-12_humi | 62 | mlp-wide | yes | 0.570 | 0.1782 | 0.4140 | 44 | 1352 |
| g1_HLM-12_humi | 353 | mlp-wide | yes | 0.513 | 0.1138 | 0.2337 | 11 | 356 |
| g1_HLM-12_humi | 164 | mlp-wide | no | 0.451 | 0.1803 | 0.3286 | 53 | 1620 |
| g1_HLM-12_stationary_teleop | 137 | ens | yes | 0.558 | 0.1473 | 0.3336 | 40 | 1216 |
| g1_HLM-12_stationary_teleop | 180 | ens | yes | 0.526 | 0.1786 | 0.3767 | 55 | 1683 |
| g1_HLM-12_stationary_teleop | 59 | ens | yes | 0.522 | 0.1655 | 0.3465 | 46 | 1413 |
| g1_HLM-12_stationary_teleop | 121 | ens | yes | 0.517 | 0.1616 | 0.3345 | 37 | 1137 |
| long_g1_5_fruits_full_teleop | 104 | ens | yes | 0.567 | 0.1304 | 0.3010 | 72 | 2177 |
| long_g1_5_fruits_full_teleop | 71 | ens | no | 0.467 | 0.1443 | 0.2707 | 88 | 2652 |
| long_g1_5_fruits_full_teleop | 27 | ens | yes | 0.537 | 0.1602 | 0.3459 | 67 | 2025 |
| long_g1_5_fruits_full_teleop | 24 | ens | no | 0.449 | 0.1853 | 0.3365 | 70 | 2110 |
| long_g1_5_fruits_humi | 108 | ens | yes | 0.636 | 0.1113 | 0.3059 | 7 | 222 |
| long_g1_5_fruits_humi | 85 | ens | yes | 0.633 | 0.1171 | 0.3192 | 6 | 205 |
| long_g1_5_fruits_humi | 22 | ens | no | 0.489 | 0.1674 | 0.3276 | 86 | 2608 |
| long_g1_5_fruits_humi | 76 | ens | yes | 0.637 | 0.1062 | 0.2925 | 5 | 177 |
