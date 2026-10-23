# Runbook

How to set up and run the parts of this project. Results are in `docs/results.md`; the project overview is in `docs/project.md`.

## Placeholders used below

| Placeholder | Meaning |
|---|---|
| `$REPO` | your local checkout of this repository |
| `$SHARE` | your large shared directory on the cluster (the scripts were written for a path of the form `/nfs/hpc/share/<user>`) |
| `$CLUSTER` | an ssh host alias for the cluster's login node, defined in your own ssh config |
| `$SLURM_ACCOUNT` | your Slurm account name |
| `HF_HOME` | Hugging Face cache; on the cluster keep it at `$SHARE/hf_cache` (home quotas are usually small) |
| `<port>` | the real arm's serial port (for example `COM5` on Windows, `/dev/ttyACM0` on Linux) |

### What is site-specific

The local parts (MuJoCo, Gymnasium, the SO-101 tasks, the viewer) run anywhere Python 3.12 runs. The cluster parts were written for one university cluster and will need editing before they run elsewhere:

- Every `*.sbatch` file in `vla/`, `isaac/` and `containers/stack/` hard-codes `#SBATCH --account`, `--partition` (names such as `gpu`, `ampere`, `dgxh`, `preempt`, `share`), `--constraint` GPU feature tags, `--exclude` node lists and `--output` log paths, plus a share path near the top (`S=` or `I=`). Replace these with your own account, partitions, GPU features and `$SHARE`.
- Cluster scripts assume **Slurm** for scheduling and **Apptainer** for containers. On the original cluster `module load slurm` was needed before `sbatch` was on `PATH`; yours may differ.
- The Isaac scripts assume an **Isaac Sim 5.1 container** (`isaac-sim:5.1.0`), unpacked into a writable sandbox at `$SHARE/isaac/isaac-sb` with Isaac Lab v2.3.2 installed into it (`isaac/setup_cluster.sbatch` does this), and an Apptainer home at `$SHARE/isaac-home`. This was needed because the cluster nodes run an older glibc (2.34) than the Isaac Sim pip wheels require (2.35).
- The planner benchmark uses a vLLM container (`$SHARE/llmserve/vllm.sif`, vLLM 0.24.0); the stack uses a separate image (see "Stack image").
- Compute nodes had no internet access: models were downloaded on the login node first (`huggingface_hub.snapshot_download`) and jobs ran with `HF_HUB_OFFLINE=1`.

## Local environment

`uv sync` recreates `.venv` from `uv.lock`. PyTorch comes from the CPU-only index pinned in `pyproject.toml`; other packages come from PyPI (the Python Package Index).

```bash
cd $REPO
uv sync
uv run python -c "import gymnasium, stable_baselines3, torch, mujoco"
```

Versions at the time of writing: Python 3.12, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, MuJoCo 3.14.0.

## Train and watch (toy reinforcement learning)

- `train.py [ENV_ID] [TIMESTEPS] [--no-gui]` trains with PPO (Proximal Policy Optimization) and shows training live in MuJoCo's viewer; writes the model, learning curve and summary to `runs/<ENV_ID>/`.
- `watch.py [ENV_ID]` replays the trained agent in real time. It uses `mujoco.viewer` because Gymnasium's viewer crashes on camera moves with MuJoCo 3.14.

```bash
uv run python train.py
uv run python watch.py
```

## MuJoCo viewer and models (optional, outside this repo)

- MuJoCo's standalone `simulate` app (from the MuJoCo release archive) opens any `scene.xml` by drag and drop.
- MuJoCo Menagerie models used here: `unitree_g1`, `franka_emika_panda`, `shadow_hand`, `unitree_go2`. A sparse clone of Menagerie is enough.

## SO-101 arm tasks (local, CPU)

`SO101Reach-v0`: gripper tip to a random target; five joint-target nudges per step at about 29 Hz; success within 2 cm. Model vendored from MuJoCo Menagerie (commit c96a32d, Apache-2.0).

```bash
uv run python train.py SO101Reach-v0 500000
uv run python watch.py SO101Reach-v0
```

## Decision models in the viewer

`decide.py [rule|random|laya|jev|<Laya checkpoint>] [EPISODES] [--task=...]` shows each choice and its probabilities live in the MuJoCo viewer and writes a chart to `runs/decide/comparison.png`. Laya needs `.venv-vla`. `jev` calls a paid API and needs `TYPESAFE_API_KEY`.

```bash
uv run python decide.py rule 10
.venv-vla/bin/python decide.py laya 5        # Windows: .venv-vla\Scripts\python
```

## VLA (vision-language-action) stack

A separate environment, `.venv-vla`, built from `vla/requirements.txt` (LeRobot 0.6.1 with SmolVLA). The main `uv.lock` stays CPU-only.

```bash
uv venv -p 3.12 .venv-vla
uv pip install -p .venv-vla -r vla/requirements.txt torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip uninstall -p .venv-vla torchcodec      # on Windows torchcodec needs a separate FFmpeg install; LeRobot falls back to PyAV
.venv-vla/bin/python vla/rollout.py lerobot/smolvla_base 2
```

- `vla/rollout.py [POLICY] [EPISODES]` runs a LeRobot policy on `so101/pixel_env.py` (front + wrist cameras). Headless machines need `MUJOCO_GL=egl`.
- `lerobot/smolvla_base` is fine-tune-only: its normalization statistics use legacy SO-100 dataset keys, so un-normalization is silently skipped.
- Pick demos: `vla/record_picks.py [--episodes=200] [--worlds=256] [--no-gui]` runs the scripted pick in many GPU worlds and renders cameras on CPU afterwards; writes `runs/datasets/so101_pick` (repo id `local/so101_pick`). `--oracle` adds the target's start x, y to the state (`so101_pick_oracle`); `--oracle-noise=CM` offsets that hint per episode.
- SmolVLA expects `observation.images.camera1..3`; map front to `camera1` and wrist to `camera2` with `--rename_map`.
- Fine-tune (pass the cached snapshot folder as `--policy.path`; on Windows a hub id fails path validation). Batch 16 used 4.5 GB and ran at 1.6-2 steps/s on a 16 GB card:

```bash
.venv-vla/bin/lerobot-train --policy.path=$HF_HOME/hub/models--lerobot--smolvla_base/snapshots/<snap> \
  --policy.device=cuda --policy.push_to_hub=false --dataset.repo_id=local/so101_pick --dataset.root=runs/datasets/so101_pick \
  --dataset.video_backend=pyav '--rename_map={"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=16 --steps=2000 --save_freq=1000 --log_freq=50 --num_workers=2 --wandb.enable=false --output_dir=runs/smolvla_pick_test
```

- Pick evaluation: `vla/rollout.py CKPT 20 --task=pick [--layouts=train] [--no-distractor] [--oracle [--oracle-noise=CM]] [--noise-scale=0] [--n-action-steps=N]` (held-out seeds 900000+). Each episode's policy noise is seeded so runs can be paired. `--noise-scale=0` (noise-free sampling) scored better and was used for later comparisons.
- Tabletop task (shapes to trays and a shelf): `python decide.py rule 5 --task=tabletop[-detect] [--fast]`.
- Off-the-shelf detector check: `vla/detect_check.py [--episodes=40] [--size=640] [--methods=mask,gdino,gdino+colour,owlv2] [--picks]`. SAM 3 (`facebook/sam3`) is gated on Hugging Face: accept its terms first.

## General pick-and-place (camera only)

- Pipeline: RGB-D cameras, then `so101/see.py` (Grounding DINO + SigLIP 2 + SAM 2.1), then `so101/grasp.py` (grasp from points), then `bench_general.py` (tasks, retries, scoring).
- Watch it: `.venv-vla/bin/python bench_general.py --gui --scenes=3 --workers=1 [--set=scanned --objects=3]`.
- Benchmark: `bench_general.py --scenes=N --workers=W [--set=primitives|scanned] [--seed=8] [--cams=cam_front] [--depth-noise] [--fit-table] [--tag=x]`. Videos: `--video=DIR --only=SEEDS`.
- Scanned objects: `so101/assets/objects/fetch_gso.py` downloads them (Google Scanned Objects, CC-BY 4.0; not committed).
- Mask model: `--masker=sam3|sam2-tiny` (default SAM 2.1 large). SAM 3's box-prompted head is loaded by key mapping (`see.sam3_tracker`); a plain `from_pretrained` leaves it random. Mask quality alone: `PYTHONPATH=. python vla/mask_check.py --set=scanned`.
- Single-camera depth instead of a depth camera: `--mono=small|base|large|metric-small` (Depth Anything; `base` was the one kept). Camera realism: `--cams=cam_front` (one camera), `--depth-noise`, `--pose-err=CM,DEG`.

## Real SO-101 arm (code written, not yet tested on hardware)

- `so101/robot.py`: `SimArm` (simulator with optional servo sloppiness), `RealArm` (LeRobot), `Guard` (safety layer in front of either) and `KinematicArm` (dry runs). The planner only sees joint readings.
- `Guard` checks joint ranges, caps arm joints at 4 rad/s, keeps the jaw tips above -3 mm and the tip within 42 cm and below 40 cm. `mode="stop"` freezes in place on a breach (it keeps holding); `"warn"` only counts. In the viewer, Space stops and Space again carries on. The power plug is the hard stop.
- Dry run: `bench_general.py --gui --dry-run [--ask]` plays each move as a see-through ghost arm first (red if a safety rule would break) and refuses plans that break one.
- First setup (find the port with `lerobot-find-port`):

```bash
lerobot-setup-motors --robot.type=so101_follower --robot.port=<port>
lerobot-calibrate --robot.type=so101_follower --robot.port=<port> --robot.id=so101
python -m so101.robot zero --port=<port>     # match LeRobot's angles to the simulator's -> runs/calib/arm.json
python -m so101.robot check --port=<port>    # motors off: the viewer copy must follow the arm by hand
python -m so101.robot selftest               # checks the zero maths against a pretend arm
```

- Camera calibration (`so101/calib.py`, ChArUco board): `python -m so101.calib board` (print at 100 %), then `lens` and `pose`.
- Camera-based arm calibration (`so101/armcal.py`), after `robot zero` and the board steps:

```bash
python -m so101.armcal marker              # print runs/calib/marker.png at 100 %; the black square is 3.0 cm
# tape it flat on the outside of the fixed jaw, centred about 6 cm up from the tips, "TIP" edge toward the tips
python -m so101.armcal run --port=<port>   # about 20 poses above the table, then updates arm.json and cam0.json
python -m so101.armcal selftest [--gui]    # simulated check
```

- The fit cannot measure the shoulder pan offset (it looks the same as moving the camera around the base, so the board decides it) or the wrist roll offset (it looks the same as a crooked marker). Benchmark with `bench_general.py --cams=cam_front --arm-calib [--tape=MM,DEG]`.

## Planner benchmark (open requests to moves)

- `so101/planner.py`: scene text, request families with exact goals, prompt, answer parsing and checking. Objects are named by id (o1, ...).
- Plan-only benchmark, no physics. Models run in a vLLM container (`uv pip install vllm` failed to build llguidance on the original login node):

```bash
python plan_bench.py make --scenes=30 && python plan_bench.py baselines   # login node, .venv-vla
sbatch vla/plan.sbatch --model=qwen3.5-9b --think=on                       # model keys: plan_bench.MODELS
sbatch --constraint="h100|h200" vla/plan.sbatch --model=qwen3.8-27b --think=on
.venv-vla/bin/python plan_bench.py report                                  # table + runs/plan/comparison.png
```

- `nvidia/Cosmos-Reason2-8B` is gated: accept its terms on your Hugging Face account first.

## Cluster setup (Slurm)

- Copy the code to `$SHARE/rlRobots` (the original setup used tar over ssh). Keep caches on `$SHARE`:

```bash
export UV_CACHE_DIR=$SHARE/uv_cache UV_PYTHON_INSTALL_DIR=$SHARE/uv_python HF_HOME=$SHARE/hf_cache
uv venv -p 3.12 --python-preference only-managed .venv-vla && uv pip install -p .venv-vla -r vla/requirements.txt
sbatch vla/rollout.sbatch lerobot/smolvla_base 5
```

- Build venvs on uv-managed Python on shared storage: system Python may exist only on submit nodes.
- Sync code: `tar czf - so101 bench_general.py vla | ssh $CLUSTER "cd $SHARE/rlRobots && tar xzf -"`. Do not re-sync while jobs run: workers loading half-written mesh files crashed.
- Any one-off script on a cluster GPU: `sbatch vla/run.sbatch -m so101.armcal selftest` (arguments go to Python; log in `runs/slurm/`). CPU jobs: `vla/cpu.sbatch`.
- General benchmark: `sbatch vla/bench.sbatch --set=scanned --objects=3 --scenes=150 --tag=x` (8 worker processes on one GPU, needs 48 GB+; results in `runs/general/`, summary with `python vla/bench_summary.py`).
- Frozen code for long benchmarks (so edits can continue meanwhile): copy `so101/*.py`, `so101/assets/*.xml`, `bench_general.py` and `vla/` into `runs/snap-<tag>/`, symlink `.venv-vla` and the large asset folders there, and submit from that folder.
- Depth models (`.venv-depth`): a copy of `.venv-vla` plus `depth_anything_3` and `moge` installed `--no-deps`, then only the modules they import (omegaconf, utils3d-moge, moviepy==1.0.3, addict, trimesh, evo, scipy). Login-node Python may need `OPENBLAS_NUM_THREADS=1`.
- Some partitions offer GPUs PyTorch 2.11 cannot use (M60, GTX 1080, V100 for the Isaac container); the scripts pick GPUs by feature tag.

## Unitree G1 humanoid in MuJoCo (fixed base, right arm and Dex3 hand)

- Model: `python so101/assets/fetch_g1.py` copies MuJoCo Menagerie `unitree_g1` (commit c96a32d, BSD-3) into `so101/assets/unitree_g1/` (not committed). Scene `so101/assets/scene_g1.xml`, code `so101/g1.py`.
- Checks (CPU job): `sbatch vla/cpu.sbatch -m so101.g1 hand` (opening table), `... reach` (IK (inverse kinematics) reach with contact check), `... claw` (scripted picks of 8 shapes, about 6 min).
- Benchmark: `sbatch --time=03:00:00 vla/bench.sbatch --robot=g1 --set=primitives --seed=8 --scenes=100 --cams=cam_front --depth-noise --fit-table --tag=x`; results `runs/general/g1-<set>-s<seed>-n<scenes>-<tag>.json`. G1 grasp options: `--g1-grasp=chooser`, `--g1-hold=touch`.
- Grasp sampler: `sbatch --cpus-per-task=4 --mem=12G vla/cpu.sbatch -m so101.g1_grasps sample "--shape=lying cylinder" --n=60 --seed=1 --procs=4 --retest=3` writes `runs/grasps/*.jsonl`. Held-out lift test: `... -m so101.g1_grasps test --policy=scripted --shape=box --n=60 --procs=4` (seeds 900000+).
- Grasp chooser: `set -f; sbatch ... vla/cpu.sbatch -m so101.g1_chooser train 'runs/grasps/sample-*.jsonl' --model=runs/grasps/chooser-vN.pt`. Quote the glob (and `set -f`): otherwise the submitting shell expands it at submit time and later files are missed. Test with `-m so101.g1_grasps test --policy=chooser --model=... --tag=vN`; compare with `python -m so101.g1_grasps paired scripted chooser:vN` and `figure scripted chooser:vN`.
- Residual refinement: `sbatch --cpus-per-task=16 --mem=48G --time=06:00:00 vla/cpu.sbatch -m so101.g1_residual train --model=runs/grasps/chooser-v3.pt --iters=15 --batch=96 --procs=16`; test with `--policy=residual`.
- Grasp videos (GPU, headless): `sbatch vla/run.sbatch -m so101.g1_grasps video --from=runs/grasps/<file>.jsonl --rows=1,2 --out=runs/grasps/x.mp4`.
- Unitree G1 + Dex3 teleoperation recordings (`so101/unitree_data.py`): tables and metadata only (13 `unitreerobotics/G1_Dex3_*` datasets, about 0.6 GB without videos), downloaded into `HF_HOME` on the login node. Needs pyarrow: `python -m so101.unitree_data check|synergy`.

## GPU physics (MuJoCo Warp, Newton)

- MuJoCo Warp in `.venv-vla` adds only `mujoco-warp` and `warp-lang`. GPU sort env: `so101/gpu_sort.py`. Per-world buffers `nconmax=24, njmax=96` work; sizing them as totals asks for far too much memory.
- Newton (NVIDIA GPU physics on MuJoCo Warp) in its own venv so it cannot break `.venv-vla`: `uv venv -p 3.12 .venv-newton && uv pip install -p .venv-newton "newton[examples]"`.
- G1 feasibility script: `so101/g1_warp.py` with `scene_g1_simple.xml`.
- On an older Pascal GPU (sm_61) MuJoCo Warp crashed compiling Warp's libmathdx tile Cholesky; `warp.config.enable_mathdx_solver = False` is an untested workaround.

## Isaac Lab (G1 + Dex3)

Setup scripts:
- `isaac/setup_ubuntu.sh`: an Ubuntu 22.04+ machine with a GPU (written for a cloud instance): Isaac Sim 5.1 via pip, Isaac Lab v2.3.2 core, Unitree's `unitree_sim_isaaclab`. Paths inside it are for that instance; edit them.
- `isaac/setup_cluster.sbatch`: unpacks the Isaac Sim 5.1 container into a writable sandbox and installs Isaac Lab v2.3.2 into it.

Fixes that cost time:
- Use Isaac Lab **v2.3.2**, not main (main is 3.0 and has no `isaaclab.sh`).
- `setuptools<81`; `flatdict` without build isolation; skip `isaaclab_mimic` in the base install (egl-probe fails to build); `libglu1-mesa libxt6`.
- `PROJECT_ROOT=<unitree_sim_isaaclab>` for its USD (Universal Scene Description) paths; Unitree's `tasks` package needs that repo on `sys.path`; Unitree's reward and termination terms assume one robot.
- RL: `rsl-rl-lib==3.1.2` (newer fails with `KeyError: 'actor'`); `scripts/reinforcement_learning/rsl_rl/train.py --task ... --headless`.
- Isaac imports (`pxr`) work only after `SimulationApp` / `AppLauncher` starts.
- Locomanipulation G1: import pinocchio before `AppLauncher`; `qpsolvers==4.4.0` (4.13 breaks pink 3.1.0); IK world targets are env-local (subtract env origins).
- Isaac Sim 5.1's RTX renderer crashed at startup on drivers newer than its tested 580 branch; Isaac Sim 6.1 rendered walking replays. Lift and walk clips are redrawn in MuJoCo by `isaac/render_mj.py`.
- On the cluster, use GPUs with RT cores (A40, L40S, RTX 8000, RTX 6000), not H100 / H200; V100s fail with "no kernel image is available".

Cluster job scripts (all run in the Isaac container; edit the Slurm header and the `I=` share path first):

| Script | Runs |
|---|---|
| `isaac/grasp_cluster.sbatch` | grasp-parameter RL (`isaac/g1_grasp_rl.py`): `N=60 REPS=8 ITERS=60 OUT=runs/isaac/rl-v4 ARGS="--redraw --params=8 --snap=30"`; evaluate only with `POLICIES=a.pt,b.pt [EVAL_SEED=700000]` |
| `isaac/dexsuite_cluster.sbatch` | Dexsuite PPO (`isaac/g1_dexsuite.py`): `MODE=train NE=2048 ITERS=5000 OUT=... [ARGS=--resume]`; `MODE=eval CK=a.pt OUT=... [EVAL_ARGS="--objects=household --n=3"]`; `MODE=diag` |
| `isaac/walk_cluster.sbatch` | any walking script, e.g. `SCRIPT=isaac/walk_grasp.py ARGS="--shape=box,ball --n=64 --carry --ik batch --record"` |
| `isaac/sdg_cluster.sbatch` | the locomanipulation imitation pipeline and the box task, chosen by `STEP=` (list below) |
| `isaac/render_cluster.sbatch` | MuJoCo redraws of recorded runs: `REC=.../rec-<tag>.npz OUT=.../clips [KIND=lift\|walkgrasp]` |
| `isaac/py_cluster.sbatch` | a CPU-only Python script with the container's Python |

`isaac/sdg_cluster.sbatch` steps (see the comment block at its top for every option):
- `gen`, `bc`, `play`: Isaac Lab Mimic demo generation from NVIDIA's annotated G1 demos, robomimic BC (behaviour cloning), rollouts.
- `nav`: walking-between-tables data (`locomanipulation_sdg`). `KIT= SHIM=1 CAMS=` loads only the mobility_gen path planner with cameras off; this avoids the RTX renderer crash.
- `navbc`, `navplay`, `navmod`, `placebc`: navigation BC, its evaluation, the modular pick / walk / place evaluation, place-only BC.
- `boxscript` (`BOX_GOAL=x,y [RECORD=1] [ENVS=N] [IK=batch DEV=cuda:0]`), `boxbc`, `boxbcplay` (`[HANDOFF=steps]`), `boxrl`, `boxwalk` (`[PICK=bc]`): the two-hand box task. Box settings come from `BOX_SIZE=x,y,z`, `BOX_MASS`, `BOX_POS`, `BOX_SHAPE=cylinder`, `BOX_SIZED=1` (size-aware BC).

Walking pick-and-place (`isaac/walk_grasp.py`) options worth knowing: `--shape` takes a comma list so one job mixes shapes; `--ik batch` uses `isaac/batch_ik.py` (one damped least-squares step for all robots on the GPU) instead of Pink (one robot at a time on CPU); `--roll_damp` (default 20) damps round and lying objects because PhysX has no rolling friction; `--slow_place`, `--settle_place` slow the put-down. Batch IK settings come from `IK_WAIST` and `IK_STEP`.

## Stack (perception, planner, decision model, skills)

- `stack_v1.py`, `stack_v2.py`, `ground_bench.py`, `reid_bench.py`; design and pre-registered tests in `docs/stack_v1.md` and `docs/stack_v2.md`.
- `vla/stack.sbatch` runs the stack on two GPUs; `vla/stack_serve.sh` starts the model servers (JEV-9B or Gemma on vLLM, Gemma on Ollama for the planner, DAM-3B). Smoke test: `MODE=smoke sbatch vla/stack.sbatch`; a run: `STACK=v2 MODE=run ARGS="--episodes=..." sbatch vla/stack.sbatch`. `GEMMA_VLLM=1 [GEMMA_LORA=dir]` serves Gemma as the decision model.
- Decision LoRA (low-rank adapter): `vla/decision_data.py` (examples from simulated runs), `vla/train_decide_lora.py`, `vla/decide_lora.sbatch`. Training runs in the vLLM container because older transformers cannot read Gemma 4's checkpoint.
- MuJoCo rendering needs EGL access to `/dev/dri` on the node; some nodes lack it.

### Stack image (`containers/stack`)

- One Apptainer image holds the stack's software: `vllm/vllm-openai:v0.31.0` (torch, transformers, vLLM) plus `containers/stack/requirements.txt` (MuJoCo, gymnasium, av, peft, ...) and DAM-3B's own environment at `/opt/dam` (`requirements-dam.txt`). No code and no weights: jobs bind the checkout and `hf_cache`.
- Rebuild only after changing `requirements*.txt` or `stack.def`: `sbatch containers/stack/build.sbatch` writes `$SHARE/images/stack-<date>.sif` and points `stack.sif` at the newest. Builds use `--fakeroot`. Old images stay, so an old run can be repeated with the image it used.
- Use: `IMAGE=$SHARE/images/stack.sif` on `vla/stack.sbatch` runs the stack, the vLLM servers and DAM-3B from the image (Ollama still runs from its own binary).
- Notebook services without Apptainer or Docker can install the same `requirements.txt` into their Python.

## Hardware needs (measured on an RTX 4060 Ti 16 GB)

- Simulator + rule or random decider: about one CPU core, 0.8 GB RAM, under 1 GB GPU (viewer only).
- Laya inference: about 2.4 GB GPU, 41-50 ms per decision. SmolVLA inference: about 1 GB GPU.
- Laya full fine-tune, bf16 autocast + AdamW: 8.4 GB at batch 1, 9.5 GB at batch 8. Batch 32 needs 20 GB and spills to system RAM (about 100x slower); on 16 GB keep batch at or below 16 and use gradient accumulation.
- SmolVLA fine-tune (frozen vision-language backbone, defaults): 4.5 GB at batch 16.
- Perception (DINO + SigLIP 2 + SAM 2.1): 2.4 GB weights, 3.3 GB peak, about 1.2 s per two-camera look.
- The full stack needs about 44 GiB of GPU memory and was run on two GPUs.
