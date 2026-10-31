# Third-Party Models, Data, Software and Documents

The repository-level MIT License covers only original project code and
documentation written for this project. It does not relicense the robot models,
object datasets, simulators, libraries, pretrained models or third-party
documents listed below. Licences were read from each project's official
repository or page on 2026-10-22; check the current terms before redistributing
anything.

## Robot Models

### SO-101 arm (vendored)

`so101/assets/robotstudio_so101/` is copied from MuJoCo Menagerie
(https://github.com/google-deepmind/mujoco_menagerie, folder `robotstudio_so101`,
commit `c96a32d28fb5da84da38c1da4d749e7a13212855`), including its mesh files.
It is licensed under the Apache License 2.0; the upstream `LICENSE` is kept in
that folder.

### Unitree G1 with Dex3 hands (fetched, not redistributed)

`so101/assets/fetch_g1.py` copies MuJoCo Menagerie's `unitree_g1` folder (same
commit) into `so101/assets/unitree_g1/`, which is gitignored. That model is
licensed under the BSD 3-Clause License, copyright HangZhou YuShu Technology
Co., Ltd. ("Unitree Robotics"); its `LICENSE` is copied along with it.
`so101/assets/rebuild_g1.py` rebuilds the same model without GitHub access: the
MJCF from the committed `scene_g1_cull.xml` (generated from that Menagerie
model) and the STL meshes from Unitree's `g1_description` as packaged in the
`example-robot-data` (or `holosoma`) PyPI wheel, under the same BSD 3-Clause
licence; the result is gitignored the same way.

The Isaac Lab part uses the G1 assets that Isaac Lab and Unitree's
`unitree_sim_isaaclab` (https://github.com/unitreerobotics/unitree_sim_isaaclab;
its README code headers state Apache License 2.0) download at setup. They are
not redistributed here.

### Unitree G1 + Dex3 teleoperation recordings

`so101/unitree_data.py` reads tables from Unitree's `unitreerobotics/G1_Dex3_*`
Hugging Face datasets at runtime (not redistributed). `so101/assets/dex3_synergy.json`
is a small hand-closing direction derived from them. The dataset page checked
(`G1_Dex3_PickBottle_Dataset`) declares Apache License 2.0; the other
`G1_Dex3_*` datasets were not checked one by one.

## Object Data

### Google Scanned Objects

Google Scanned Objects (Downs et al., "Google Scanned Objects: A High-Quality
Dataset of 3D Scanned Household Items," IEEE International Conference on Robotics and Automation (ICRA) 2022,
arXiv:2204.11918) is
licensed under the Creative Commons Attribution 4.0 International License
(CC-BY 4.0), as stated on the official Google Research blog post:
https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/

The meshes and textures are not stored in this repository.
`so101/assets/objects/fetch_gso.py` downloads them from
https://github.com/kevinzakka/mujoco_scanned_objects (pinned commit
`6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`; its MuJoCo model files are under
the MIT License, copyright Kevin Zakka; its 3D assets carry GSO's CC-BY 4.0) into
the gitignored `gso/` folder, and `gso_g1.py` builds the gitignored `gso_g1/`
folder. What the repository does store is derived metadata:
`gso_index.json` (50 objects) and `gso_g1_index.json` (114 objects), with
object names, sizes, masses, inertia, text phrases and drop-test results.
Attribution: Google Scanned Objects by Google Research, CC-BY 4.0.

### DexGraspNet and ShapeNetSem (not redistributed)

`so101/assets/objects/pool_g1.py` can build an object pool from DexGraspNet
meshes and ShapeNetSem, which its docstring describes as CC BY-NC 4.0 and
non-commercial research use (gated) respectively. Neither is included here, and
those licences were not re-checked for this release. Results obtained with that
pool rely on non-commercial data.

## Simulators and Frameworks (dependencies, not redistributed)

| Project | Licence (from the official repository or package page) |
|---|---|
| MuJoCo (https://github.com/google-deepmind/mujoco) | Apache License 2.0 |
| NVIDIA Isaac Sim (https://github.com/isaac-sim/IsaacSim) | Repository code: Apache License 2.0. The repository licence states that building or using it requires additional components (for example the Omniverse Kit SDK and 3D models and textures) under NVIDIA's separate terms: https://www.nvidia.com/en-us/agreements/enterprise-software/isaac-sim-additional-software-and-materials-license/. The `isaacsim` 5.1.0 package on the Python Package Index (PyPI) is labelled "Other/Proprietary License (NVIDIA Proprietary Software)". |
| NVIDIA Isaac Lab (https://github.com/isaac-sim/IsaacLab) | BSD 3-Clause License |
| AGILE (https://github.com/nvidia-isaac/WBC-AGILE), the pretrained G1 leg policy used through Isaac Lab's locomanipulation task | Apache License 2.0 for most code, with a BSD 3-Clause rsl_rl patch, as stated in its README (the `LICENCE` file itself was not read). That is the training code; the leg-policy checkpoint used here comes with Isaac Lab's locomanipulation task as an NVIDIA asset, is not redistributed, and its terms were not checked separately. |
| rsl_rl (https://github.com/leggedrobotics/rsl_rl) | BSD 3-Clause License |
| robomimic (https://github.com/ARISE-Initiative/robomimic) | MIT License |
| LeRobot (https://github.com/huggingface/lerobot) | Apache License 2.0 |
| Stable-Baselines3 (https://github.com/DLR-RM/stable-baselines3) | MIT License |
| Gymnasium (https://github.com/Farama-Foundation/Gymnasium) | MIT License |

Other Python dependencies (PyTorch, transformers, vLLM and those in
`uv.lock`, `vla/requirements.txt` and `containers/stack/requirements*.txt`)
keep the licences declared by their projects.

## Pretrained Models

Perception, depth, planner and decision models used by the stack (for example
Grounding DINO, SAM 2.1, SAM 3, SigLIP 2, DAM-3B, Depth Anything, SmolVLA,
Gemma 4, Qwen and JEV-9B) are downloaded at runtime from their publishers and
are not redistributed. Each has its own model licence or terms of use, some
gated; read them before use. Their licences were not checked for this release.

## Saved Documents in `docs/sources/`

`docs/sources/` holds two third-party documents saved verbatim so that numbers
in `docs/baselines.md` could be checked against them:

- `LeRobot_hilserl.mdx`: LeRobot's guide to Human-in-the-Loop Sample-Efficient
  Reinforcement Learning (HIL-SERL), from the LeRobot repository
  (Apache License 2.0).
- `SmolVLA-RL_README.md`: the README of https://github.com/Mervin-James/SmolVLA-RL.
  That repository shows no licence (no LICENSE file and no licence in its
  README as of 2026-10-22), so no reuse right is granted. **Replace this file
  with a link, or get the author's permission, before publishing.**

## Cited Papers

Papers referred to in `docs/` and in code comments are cited by title or link
only. No paper text, figures or PDFs are redistributed.
