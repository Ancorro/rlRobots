"""Replays of the decision layer (docs/plan.md layer 2): a decision model runs the SO-101 sort task, headless, and every
decision is kept (the state text it read, its probability for every option, its choice, the skill's outcome) next to a
video of the arm carrying it out. For the stack report page.

  python vla/decide_replay.py MODEL [--task=sort|sort-drop|tabletop] [--episodes=2] [--videos=2] [--out=runs/replay/decide]
  MODEL: rule | random | laya | jev-local | <Laya checkpoint folder> (e.g. runs/laya/laya-sort-warm)
  Videos for the first --videos episodes only; every episode's decisions and outcome go to <task>-<model>-summary.json.
  -> <out>/<task>-<model>-ep<seed>.mp4 and .json      (MUJOCO_GL=egl on a headless GPU box)
"""
import json
import sys
from pathlib import Path

import av
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from so101.deciders import make, options  # noqa: E402
from so101.skill_env import SO101Sort  # noqa: E402
from so101.tabletop import SO101Tabletop  # noqa: E402

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
MODEL = next(a for a in sys.argv[1:] if not a.startswith("--"))
TASK = ARGS.get("task", "sort")
OUT = Path(ARGS.get("out", "runs/replay/decide"))
W, H, FPS = 640, 480, 30

env = {"sort": SO101Sort, "sort-drop": lambda: SO101Sort(drop_prob=0.3), "tabletop": SO101Tabletop}[TASK]()
decider = make(MODEL)
name = Path(MODEL).name
OUT.mkdir(parents=True, exist_ok=True)

summary = []
for ep in range(int(ARGS.get("episodes", 2))):
    seed = 1000 + ep  # decide.py's held-out seeds
    env.reset(seed=seed)
    video = ep < int(ARGS.get("videos", 2))
    r = mujoco.Renderer(env.model, H, W)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (0.2, 0.0, 0.05), 0.75, 150, -30
    path = OUT / f"{TASK}-{name}-ep{seed}"
    box = av.open(str(path) + ".mp4", "w")
    st = box.add_stream("libx264", rate=FPS)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    tick = [0]

    def frame():
        if not video:
            return
        r.update_scene(env.data, cam)
        for p in st.encode(av.VideoFrame.from_ndarray(r.render(), format="rgb24")):
            box.mux(p)

    def on_tick():
        tick[0] += 1
        if tick[0] % 2 == 0:
            frame()
    env.on_tick = on_tick
    trace, done, info = [], False, {}
    while not done:
        state = env.state_text()
        answers = decider.decide(env)
        t_video = tick[0] / 2 / FPS
        action = [opts.index(answers[q]["choice"]) for q, opts in options(env).items()]
        for _ in range(15):  # hold the frame a moment at each decision
            frame()
            tick[0] += 2
        _, _, term, trunc, info = env.step(action)
        done = term or trunc
        trace.append(dict(t=round(t_video, 2), state=state,
                          answers={q: dict(choice=a["choice"], probabilities={o: round(p, 3) for o, p in
                                                                             a["probabilities"].items()})
                                   for q, a in answers.items()},
                          skill_ok=info.get("skill_ok"), status=info.get("status")))
    for p in st.encode():
        box.mux(p)
    box.close()
    if not video:
        Path(str(path) + ".mp4").unlink(missing_ok=True)
    rec = dict(model=name, task=TASK, seed=seed, success=bool(info.get("is_success")), decisions=trace)
    summary.append(rec)
    if video:
        path.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(f"{path}: {len(trace)} decisions, success {rec['success']}", flush=True)
ok = sum(r["success"] for r in summary)
(OUT / f"{TASK}-{name}-summary.json").write_text(json.dumps(dict(model=name, task=TASK, episodes=len(summary), success=ok,
                                                               runs=summary), indent=1))
print(f"{TASK} {name}: {ok}/{len(summary)} episodes succeeded")
