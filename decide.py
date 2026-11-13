"""Watch a decision model ("System One", Jev-style) control the SO-101 arm, live in MuJoCo's viewer.

Usage:  python decide.py [MODEL] [EPISODES] [--task=TASK] [--fast] [--no-gui]
MODEL: rule | random | laya | jev | <Laya checkpoint id or folder>      (default: rule)
TASK:  reach  move the tip to the red ball, one 15 mm step per decision (default)
       pick   put the green cube in the tray, one 15 mm step + gripper per decision
       sort   two cubes into the trays of their colour, one skill (pick / place / done) per decision
       sort-drop   sort, but the gripper slips on 30% of carries (tests recovery)
       tabletop    four shapes (cube, ball, cylinder, bar) to trays or the shelf, two goals per episode, one skill per decision
       tabletop-detect   tabletop, with positions from the Grounding DINO detector instead of the simulator

On screen: the model's answer and probabilities for each question, progress, and the running
success count, and CPU / RAM / GPU memory use (also in summary.json). Reach/pick: a green arrow shows the move just chosen. Sort: a yellow ball marks
the object chosen, and the skill runs smoothly in the viewer. --fast skips the slow-down that
makes each decision watchable. Results go to runs/decide/<task>/<model>/summary.json, and
runs/decide/<task>/comparison.png charts every model run so far on that task.
Laya needs the GPU environment (.venv-vla); rule and random run in the main one.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from so101.decide_env import SO101DecideReach
from so101.decide_pick import SO101DecidePick
from so101.deciders import make, options
from so101.skill_env import SO101Sort
from so101.tabletop import SO101Tabletop
from so101.usage import Usage

TASKS = {"reach": SO101DecideReach, "pick": SO101DecidePick, "sort": SO101Sort,
         "sort-drop": lambda: SO101Sort(drop_prob=0.3), "tabletop": SO101Tabletop,
         "tabletop-detect": lambda: SO101Tabletop(perception="detector")}
GUI = "--no-gui" not in sys.argv
FAST = "--fast" in sys.argv
TASK = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--task=")), "reach")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
MODEL = args[0] if args else "rule"
EPISODES = int(args[1]) if len(args) > 1 else 10
PAUSE = 0.0 if FAST else (1.0 if TASK.startswith(("sort", "tabletop")) else 0.15)  # seconds per decision, to follow it
OUT = Path("runs/decide") / TASK

env = TASKS[TASK]()
decider = make(MODEL)
usage = Usage()  # CPU, RAM and GPU memory, sampled in the background


def progress(info):
    return info["status"] if "status" in info else f"{info['distance_mm']:.0f} mm"


def overlay(viewer, ep, answers, info, wins, latency_ms):
    rows = []
    for q, a in answers.items():
        top = sorted(a["probabilities"].items(), key=lambda kv: -kv[1])[:3]
        rows.append(f"{q}: {a['choice']:<11} " + "  ".join(f"{o[:10]} {p:.2f}" for o, p in top))
    left = "\n".join(["model", "task", "episode", "progress", "successes", "decision time", "computer", "",
                      *[""] * len(rows)])
    right = "\n".join([decider.name, TASK, f"{ep + 1} / {EPISODES}  (step {env.steps})", progress(info),
                       f"{wins} / {ep}", f"{latency_ms:.0f} ms", usage.now(), "", *rows])
    viewer.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, left, right)])


def marker(viewer, answers):
    """Reach/pick: green arrow from the tip along the chosen move. Sort/tabletop: yellow ball over the chosen object."""
    viewer.user_scn.ngeom = 0
    g = viewer.user_scn.geoms[0]
    if "object" in answers:
        p = env.pos(answers["object"]["choice"]) + [0, 0, 0.05]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.01, 0, 0]), p, np.eye(3).ravel(),
                            np.array([1, 0.85, 0.1, 1], np.float32))
    else:
        move = np.array([options(env)[ax].index(answers[ax]["choice"]) - 1 for ax in "xyz"], float)
        if not move.any():
            return
        tip = env.tip_pos()
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array([0.1, 0.9, 0.1, 1], np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, tip, tip + 0.06 * move / np.linalg.norm(move))
    viewer.user_scn.ngeom = 1


def run(viewer=None):
    wins, steps, latencies = 0, [], []
    skills = {"ok": 0, "n": 0}
    if viewer is not None and hasattr(env, "on_tick"):  # watch skills move, at about real speed
        def tick():
            viewer.sync()
            time.sleep(0 if FAST else 0.03)
        env.on_tick = tick
    for ep in range(EPISODES):
        _, info = env.reset(seed=1000 + ep)  # held-out seeds, never used for training
        done = False
        while not done and (viewer is None or viewer.is_running()):
            t = time.perf_counter()
            answers = decider.decide(env)
            latencies.append((time.perf_counter() - t) * 1000)
            if viewer is not None:
                marker(viewer, answers)
                overlay(viewer, ep, answers, info, wins, latencies[-1])
                viewer.sync()
            action = [opts.index(answers[q]["choice"]) for q, opts in options(env).items()]
            _, _, term, trunc, info = env.step(action)
            done = term or trunc
            if "skill_ok" in info:
                skills["ok"] += info["skill_ok"]
                skills["n"] += 1
            if viewer is not None:
                overlay(viewer, ep, answers, info, wins, latencies[-1])
                viewer.sync()
                time.sleep(PAUSE)
        wins += bool(info["is_success"])
        steps.append(env.steps)
        print(f"episode {ep + 1}: {'success' if info['is_success'] else 'miss'} "
              f"after {env.steps} decisions, {progress(info)}", flush=True)
        if viewer is not None:
            overlay(viewer, ep + 1, answers, info, wins, latencies[-1])
            viewer.sync()
            time.sleep(0 if FAST else 1.0)
            if not viewer.is_running():
                break
    summary = {"model": decider.name, "task": TASK, "episodes": len(steps), "success_rate": wins / len(steps),
               "mean_decisions": float(np.mean(steps)), "median_decision_ms": float(np.median(latencies))}
    if skills["n"]:
        summary["skill_success_rate"] = skills["ok"] / skills["n"]
    return summary | usage.summary()


if GUI:
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        summary = run(viewer)
else:
    summary = run()
print(json.dumps(summary))
(OUT / decider.name).mkdir(parents=True, exist_ok=True)
(OUT / decider.name / "summary.json").write_text(json.dumps(summary, indent=2))

# Chart every model tried so far on this task.
runs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/summary.json"))]
plt.figure(figsize=(7, 3.5))
bars = plt.bar([r["model"] for r in runs], [100 * r["success_rate"] for r in runs], color="tab:blue")
plt.bar_label(bars, [f"{100 * r['success_rate']:.0f}%\n({r['episodes']} ep)" for r in runs])
plt.ylabel("success rate (%)")
plt.ylim(0, 115)
plt.title(f"SO-101 {TASK}: decision models compared")
plt.tight_layout()
plt.savefig(OUT / "comparison.png", dpi=120)
