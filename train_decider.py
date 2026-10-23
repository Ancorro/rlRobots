"""Tune the Laya decision model on the SO-101 sort or tabletop task, with the MuJoCo viewer open.

Stage 1 (warm start): log loss (a proper scoring rule, as in Laya's RLCD training) on the rule
baseline's choices, from demos on training seeds. RL from ~0% success learns nothing
(zero-shot Laya gives the correct 4-skill sequence ~4e-6 probability), so this comes first.
Stage 2 (RL): policy gradient (REINFORCE, group-mean baseline) on episode reward, sampling skills
from the model itself. Sort: the gripper slips on 30% of carries, so recovery is trained too.

Usage:  python train_decider.py [--task=sort|tabletop] [--demos=100] [--updates=20] [--group=8] [--workers=8] [--no-gui]
Episodes run side by side in --workers processes (so101/env_worker.py); the viewer shows the first.
The tabletop task trains on exact positions (fast); evaluate with `decide.py <ckpt> --task=tabletop-detect`.
Seeds 0-999 train; evaluate on held-out seeds with decide.py. Checkpoints:
runs/laya/laya-<task>-warm and runs/laya/laya-<task>-rl (load with `python decide.py runs/laya/laya-sort-rl --task=sort`).
Fine-tuning invalidates Laya's shipped temperatures: confidences are no longer calibrated.
"""
import json
import sys
import time
from pathlib import Path

import laya
import matplotlib
import mujoco
import mujoco.viewer
import numpy as np
import torch
from huggingface_hub import snapshot_download

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from so101.laya_policy import LayaPolicy
from so101.env_worker import Envs, make_env
from so101.skill_env import SKILLS
from so101.usage import Usage

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
GUI = "--no-gui" not in sys.argv
DEMOS, UPDATES, GROUP = int(ARGS.get("demos", 100)), int(ARGS.get("updates", 20)), int(ARGS.get("group", 8))
WORKERS = int(ARGS.get("workers", 8))
MICRO = 8  # states per forward/backward; 8 fits in ~10 GB, 32 overflows 16 GB
EPOCHS, LR_WARM, LR_RL = 3, 1e-5, 2e-6
OUT = Path("runs/laya")
BASE = "convaiinnovations/laya"
DONE = SKILLS.index("done")
TASK = ARGS.get("task", "sort")

torch.cuda.set_per_process_memory_fraction(0.85)  # fail loudly instead of spilling into RAM (~100x slower)
env = make_env(TASK)
envs = Envs(TASK, WORKERS, env)
policy = LayaPolicy(laya.load(BASE), env.questions)
opt = torch.optim.AdamW(policy.model.parameters(), lr=LR_WARM)
usage = Usage()
log = {"warm_loss": [], "rl_success": [], "rl_return": [], "update_s": [], "torch_gpu_gb": []}
viewer = mujoco.viewer.launch_passive(env.model, env.data) if GUI else None
if viewer:
    env.on_tick = viewer.sync


def show(*lines):
    if viewer:
        viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           "\n".join(["stage", "progress", "result", "computer"]),
                           "\n".join([*lines, usage.now()]))])
        viewer.sync()


def log_prob(rows, action):
    """Log-probability of a (skill, object) action; the object does not matter for done."""
    lp = torch.log_softmax(rows[0], -1)[action[0]]
    return lp if action[0] == DONE else lp + torch.log_softmax(rows[1], -1)[action[1]]


def episodes(seeds, choose=None):
    """Run episodes side by side, WORKERS at a time; choose(states) -> actions (None: the rule)."""
    out = []
    for k in range(0, len(seeds), WORKERS):
        out += envs.run(seeds[k:k + WORKERS], choose)
    return out


def sample(states):
    with torch.no_grad():
        rows = policy.logits(states)
    return [[int(torch.distributions.Categorical(logits=z).sample()) for z in r] for r in rows]


# --- stage 1: warm start on the rule's choices ---------------------------------------------------
data = []
for i in range(0, DEMOS, WORKERS):
    show("warm start: collecting rule demos", f"episodes {i + 1}-{min(i + WORKERS, DEMOS)} / {DEMOS}",
         f"{len(data)} decisions")
    if usage.low_ram():
        sys.exit("stopping: under 2 GB of RAM free on this PC")
    for ss, aa, _, _ in episodes(list(range(i, min(i + WORKERS, DEMOS)))):
        data += zip(ss, aa)
rng = np.random.default_rng(0)
for epoch in range(EPOCHS):
    order = rng.permutation(len(data))
    for k in range(0, len(order), MICRO):
        batch = [data[j] for j in order[k:k + MICRO]]
        rows = policy.logits([s for s, _ in batch])
        loss = -torch.stack([log_prob(r, a) for r, (_, a) in zip(rows, batch)]).mean()
        loss.backward()
        opt.step(), opt.zero_grad(set_to_none=True)
        log["warm_loss"].append(loss.item())
        show("warm start: log loss on rule labels", f"epoch {epoch + 1} / {EPOCHS}, batch {k // MICRO + 1}",
             f"loss {np.mean(log['warm_loss'][-20:]):.3f}")
policy.save(OUT / f"laya-{TASK}-warm", snapshot_download(BASE))

# --- stage 2: policy gradient on episode return --------------------------------------------------
for g in opt.param_groups:
    g["lr"] = LR_RL
for u in range(UPDATES):
    t = time.time()
    torch.cuda.reset_peak_memory_stats()
    show("RL: sampling skills from the model", f"update {u + 1} / {UPDATES}, {GROUP} episodes",
         f"last update success {log['rl_success'][-1] if log['rl_success'] else float('nan'):.0%}")
    eps = episodes([1 + DEMOS + u * GROUP + e for e in range(GROUP)], sample)
    if usage.low_ram():
        print("stopping early: under 2 GB of RAM free on this PC", flush=True)
        break
    returns = np.array([ret for _, _, ret, _ in eps])
    adv = returns - returns.mean()  # group-mean baseline (as in GRPO, without the std scaling)
    steps = [(s, a, adv[i]) for i, (ss, aa, _, _) in enumerate(eps) for s, a in zip(ss, aa)]
    if np.abs(adv).max() > 1e-6:
        for k in range(0, len(steps), MICRO):
            chunk = steps[k:k + MICRO]
            rows = policy.logits([s for s, _, _ in chunk])
            loss = -sum(float(w) * log_prob(r, a) for r, (_, a, w) in zip(rows, chunk)) / len(steps)
            loss.backward()
        opt.step(), opt.zero_grad(set_to_none=True)
    log["rl_success"].append(float(np.mean([ok for *_, ok in eps])))
    log["rl_return"].append(float(returns.mean()))
    log["update_s"].append(time.time() - t)
    log["torch_gpu_gb"].append(torch.cuda.max_memory_allocated() / 1e9)
    print(f"update {u + 1}: success {log['rl_success'][-1]:.0%}, return {returns.mean():.2f}, "
          f"{log['update_s'][-1]:.0f} s, GPU {log['torch_gpu_gb'][-1]:.1f} GB", flush=True)
policy.save(OUT / f"laya-{TASK}-rl", snapshot_download(BASE))
envs.close()
if viewer:
    viewer.close()

summary = {"task": TASK, "demos": DEMOS, "warm_decisions": len(data), "updates": UPDATES, "group": GROUP,
           "warm_final_loss": float(np.mean(log["warm_loss"][-20:])),
           "rl_success_first": log["rl_success"][0], "rl_success_last5": float(np.mean(log["rl_success"][-5:])),
           "mean_update_s": float(np.mean(log["update_s"])), "torch_gpu_gb_peak": max(log["torch_gpu_gb"])}
summary |= usage.summary()
(OUT / f"train_summary_{TASK}.json").write_text(json.dumps(summary | {"log": log}, indent=2))
print(json.dumps(summary))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
ax[0].plot(log["warm_loss"])
ax[0].set(title="warm start: log loss", xlabel="batch")
ax[1].plot(100 * np.array(log["rl_success"]))
ax[1].set(title="RL: training-episode success", xlabel="update", ylabel="%", ylim=(0, 105))
plt.tight_layout()
plt.savefig(OUT / f"train_curves_{TASK}.png", dpi=120)
