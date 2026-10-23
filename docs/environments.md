# Object sets and indoor environments (notes for later)

Compiled 2026-10-08 from web searches and dataset pages; not yet downloaded or tried unless stated. Notes on household objects for a separate grasp benchmark now, indoor environments for testing the full stack later ("we will work on it later").

## Household object sets (grasp benchmark)

| Set | What | Physics data | Licence / access | Our status |
|---|---|---|---|---|
| Google Scanned Objects (GSO), via kevinzakka/mujoco_scanned_objects | 1,030 scanned household items, real scale; MuJoCo files with convex hulls (V-HACD) | none (we set mass) | CC-BY 4.0, open | 50 in use (SO-101 bench, `so101/assets/objects/`); the G1 household benchmark is being built from it |
| ShapeNetSem (Savva, Chang, Hanrahan; ShapeNet, Chang et al. 2015) | ~12k CAD models, subset of ShapeNet | real-world dimensions, category-level material, volume and weight estimates | non-commercial research; Hugging Face gated, asks name, PI/advisor, affiliation | access granted (2026-10-09); ShapeNetSem.zip (12.2 GB) downloading to cluster `$SHARE/datasets/ShapeNetSem` |
| ACRONYM (NVIDIA 2021) | 17.7M simulated parallel-jaw grasps on 8,872 ShapeNetSem objects, 262 categories | uniform density 150 kg/m3, friction 1.0 | CC BY-NC 4.0; needs ShapeNetSem meshes | reference for scale/mass choices |
| HSSD-200 models (Khanna, ..., Chang; CVPR 2024) | 18,656 object models from 211 human-authored scenes (much is furniture/decor) | semantic metadata; mass not seen | CC BY-NC 4.0; Hugging Face gated (one-click accept) | no access (403); needs the gated-access accept |
| PartNet-Mobility / SAPIEN (Xiang, ..., Chang, Guibas, Su 2020) | articulated objects (drawers, lids, doors) | joints, parts | registration | later: for opening things |

## Indoor environments (full-stack test, later)

Ones that already run in Isaac Sim (our simulator) first.

| Name | Kind | Scale | Simulator | Licence / access |
|---|---|---|---|---|
| BEHAVIOR-1K / OmniGibson (Stanford) | hand-built interactive houses, 1,000 activities | 50 scenes, ~9-10k objects with physical and semantic properties | Isaac Sim 4.5 | data bundle licence + NVIDIA EULA; >30 GB |
| GRUtopia / GRScenes (Shanghai AI Lab, 2024) | interactive annotated scenes, homes and commercial; has a Unitree H1 house demo | 99-100 released of a 100k set | Isaac Sim 4.2 | user agreement (Hugging Face) |
| InteriorAgent (SpatialVerse / Kujiale) | USD interiors with physics-ready geometry | not checked | Isaac Sim 4.2 / 4.5 | custom terms of use, gated |
| Infinigen Indoors (Princeton, CVPR 2024) | procedural photoreal houses, unlimited | generator | exports USD for Isaac Sim (articulated furniture exported separately) | code open; licence not checked |
| InternScenes (NeurIPS 2025) | scans made simulatable + procedural + designer scenes | ~40k scenes, 1.96M objects, ~186 GB | Isaac Sim navigation benchmark in the paper; USD not confirmed | CC BY-NC-SA 4.0, gated |
| HSSD-200 (Chang co-author) | human-authored realistic houses | 211 scenes | Habitat (glb; would need conversion) | CC BY-NC 4.0, gated |
| ProcTHOR-10K / Holodeck (AI2) | procedural (ProcTHOR) and LLM-written (Holodeck, Objaverse assets) houses | 10k+ / generator | AI2-THOR (Unity) | open; conversion needed |
| SceneSmith (2026 preprint) | agents write simulation-ready houses: layout, furniture, dense small clutter, articulated furniture, physical properties; used for robot policy evaluation | generator | sim-ready (format not checked) | not checked |
| SceneCode (2026 preprint) | language -> Blender programs -> editable rooms with articulated parts | generator | MuJoCo export | not checked |
| RoboCasa365 (UT Austin + NVIDIA, ICLR 2026) | 2,500 kitchens, 365 tasks, 3,200+ objects, articulated appliances, 2,200 h demos | dataset + generator | MuJoCo (robosuite); LeRobot integration | CC BY 4.0 paper; code open |

Pick for later (to check first): BEHAVIOR-1K or GRScenes for a ready Isaac Sim house with the G1 walking in; Infinigen Indoors or SceneSmith for unlimited new houses (test sets the stack never saw); RoboCasa365 if we test in MuJoCo.

Sources: [ShapeNet report](https://arxiv.org/abs/1512.03012), [ShapeNetSem on Hugging Face](https://huggingface.co/datasets/ShapeNet/ShapeNetSem-archive), [ACRONYM](https://arxiv.org/abs/2011.09584), [HSSD-200](https://arxiv.org/abs/2306.11290), [hssd-models](https://huggingface.co/datasets/hssd/hssd-models), [BEHAVIOR-1K](https://behavior.stanford.edu), [GRUtopia](https://github.com/OpenRobotLab/GRUtopia), [GRScenes](https://huggingface.co/datasets/InternRobotics/GRScenes), [InteriorAgent](https://huggingface.co/datasets/spatialverse/InteriorAgent), [Infinigen export](https://infinigen.cs.princeton.edu/docs/latest/ExportingToSimulators.html), [InternScenes](https://arxiv.org/abs/2509.10813), [SceneSmith](https://arxiv.org/abs/2602.09153), [SceneCode](https://arxiv.org/abs/2605.19587), [Holodeck](https://arxiv.org/abs/2312.09067), [ProcTHOR](https://arxiv.org/abs/2206.06994), [RoboCasa365](https://arxiv.org/abs/2603.04356).
