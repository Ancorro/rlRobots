"""Run any LeRobot VLA (vision-language-action) policy as the "System 1" controller of the simulated SO-101.

Usage:  python vla/rollout.py [POLICY] [EPISODES] [--task=reach|pick] [--layouts=heldout|train] [--no-distractor] [--oracle] [--oracle-noise=CM] [--n-action-steps=N] [--noise-scale=S]
(default POLICY: lerobot/smolvla_base)
POLICY is a Hugging Face Hub id or a local checkpoint folder (e.g. one written by lerobot-train).
Pick also reports, at the tick the gripper is first told to close, the tip's xy distance to the true cube and to the
oracle hint (near the hint but far from the cube = followed the state; near the cube = the cameras corrected it), and its
height above the cube centre (the scripted pick closes at 0).

Follows lerobot-eval's loop: sim observation -> preprocess_observation -> add task sentence ->
policy preprocessor -> select_action -> postprocessor -> sim step. Writes success rate, speed,
GPU (graphics processing unit) memory and a video to runs/vla/<task>[-train]/<policy name>/.
This is the evaluation half of RL (reinforcement learning) tuning: the same loop, plus rewards
and gradient updates, is what an RL trainer runs.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

from so101.arm import GRIP_CLOSED, GRIP_OPEN
from so101.pixel_env import SO101PixelReach
from so101.pixel_pick import SO101PixelPick

POS = [a for a in sys.argv[1:] if not a.startswith("--")]
TASK = next((a[7:] for a in sys.argv if a.startswith("--task=")), "reach")
LAYOUTS = next((a[10:] for a in sys.argv if a.startswith("--layouts=")), "heldout")
NOISE = float(next((a[15:] for a in sys.argv if a.startswith("--oracle-noise=")), 0))
STEPS = int(next((a[17:] for a in sys.argv if a.startswith("--n-action-steps=")), 0))  # 0: as trained; fewer = replan more often
SCALE = float(next((a[14:] for a in sys.argv if a.startswith("--noise-scale=")), 1))  # flow-matching start noise; 0 = deterministic
POLICY = POS[0] if POS else "lerobot/smolvla_base"
EPISODES = int(POS[1]) if len(POS) > 1 else 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN = Path(POLICY.rstrip("/\\"))  # a lerobot-train checkpoint .../<run>/checkpoints/<step>/pretrained_model -> "<run>-<step>"
NAME = f"{RUN.parents[2].name}-{RUN.parents[0].name}" if RUN.name == "pretrained_model" else RUN.name
OUT = Path("runs/vla") / (TASK + ("-train" if LAYOUTS == "train" else "")) / (NAME + (f"-noise{NOISE:g}" if NOISE else "") + (f"-act{STEPS}" if STEPS else "") + (f"-ns{SCALE:g}" if SCALE != 1 else ""))
OUT.mkdir(parents=True, exist_ok=True)

cfg = PreTrainedConfig.from_pretrained(POLICY)
cfg.device = DEVICE
cfg.n_action_steps = STEPS or cfg.n_action_steps
policy = get_policy_class(cfg.type).from_pretrained(POLICY, config=cfg).to(DEVICE).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=POLICY, preprocessor_overrides={"device_processor": {"device": DEVICE}})
if SCALE != 1:  # flow-matching policies (SmolVLA) start each chunk from Gaussian noise; scale it down
    policy.model.sample_noise = lambda shape, device: SCALE * torch.randn(shape, device=device)

env = SO101PixelPick(layouts=LAYOUTS, distractor="--no-distractor" not in sys.argv, oracle="--oracle" in sys.argv,
                     oracle_noise=NOISE) if TASK == "pick" else SO101PixelReach()
assert cfg.output_features["action"].shape[0] == 6, "policy must output 6 SO-101 joint targets; fine-tune with reinit_modules first"
# Our two cameras fill the policy's camera slots in order. Reach (base model): any extra slot reuses
# the front view. Pick (fine-tuned with front->camera1, wrist->camera2): extra slots stay empty, as in training.
slots = [k for k in cfg.input_features if k.startswith("observation.images.")]
cams = ["front", "wrist"] + (["front"] * max(0, len(slots) - 2) if TASK == "reach" else [])


def policy_obs(obs):
    o = preprocess_observation(obs)
    for slot, cam in zip(slots, cams):
        o[slot] = o[f"observation.images.{cam}"]
    o = {k: v for k, v in o.items() if k in cfg.input_features}
    o["task"] = [env.task_description]
    return pre(o)


results, grasps, frames, t0, n = [], [], [], time.perf_counter(), 0
for ep in range(EPISODES):
    obs, _ = env.reset(seed=ep)
    torch.manual_seed(ep)  # same policy noise per episode, so runs compare episode by episode
    policy.reset()  # clears the queued action chunk
    done, grasp = False, None
    while not done:
        with torch.inference_mode():
            action = post(policy.select_action(policy_obs(obs)))
        obs, _, term, trunc, info = env.step(action.squeeze(0).float().cpu().numpy())
        done, n = term or trunc, n + 1
        if TASK == "pick" and grasp is None and env.data.ctrl[5] < (GRIP_OPEN + GRIP_CLOSED) / 2:
            tip = env.base.arm.tip_pos() * 100  # cm
            grasp = (np.linalg.norm(tip[:2] - env.true_cm), np.linalg.norm(tip[:2] - env.target_cm), tip[2] - env.base.pos(env.cube)[2] * 100)
        if ep == 0:
            frames.append(obs["pixels"]["front"])
    results.append(bool(info["is_success"]))
    line = f"episode {ep + 1}: success={info['is_success']} final distance {info['distance']:.3f} m"
    if TASK == "pick":
        grasps.append(grasp or (np.nan,) * 3)
        line += f"; at close: tip to cube {grasp[0]:.1f} cm, to hint {grasp[1]:.1f} cm, above cube centre {grasp[2]:.1f} cm" if grasp else "; never closed"
    print(line, flush=True)

summary = {
    "policy": POLICY,
    "task": TASK,
    "layouts": LAYOUTS,
    "device": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
    "success_rate": sum(results) / len(results),
    "episodes": EPISODES,
    **({"oracle_noise_cm": NOISE} if NOISE else {}),
    "n_action_steps": cfg.n_action_steps,
    **({"noise_scale": SCALE} if SCALE != 1 else {}),
    **({"closed_frac": float(np.mean([not np.isnan(g[0]) for g in grasps])),
        "close_tip_to_cube_cm": round(float(np.nanmedian([g[0] for g in grasps])), 2),
        "close_tip_to_hint_cm": round(float(np.nanmedian([g[1] for g in grasps])), 2),
        "close_tip_above_cube_cm": round(float(np.nanmedian([g[2] for g in grasps])), 2)} if TASK == "pick" and any(g[0] == g[0] for g in grasps) else {}),
    "env_steps_per_s": round(n / (time.perf_counter() - t0), 1),
    "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if DEVICE == "cuda" else None,
}
imageio.mimsave(OUT / "episode0.mp4", frames, fps=env.metadata["render_fps"])
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
