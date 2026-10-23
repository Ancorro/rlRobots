"""Record scripted pick demos as a LeRobot dataset, for fine-tuning SmolVLA (plan step 3).

The scripted pick skill is the teacher. Physics and skills run on the GPU (so101/gpu_sort.py, many
worlds at once); camera images are rendered afterwards on the CPU from the saved states, since
MuJoCo Warp does not render. Only successful picks are kept.

Each episode: the command "pick up the green cube" (or red), front + wrist camera images, the 6 joint
readings and the 6 joint targets, in LeRobot SO-101 units (as so101/pixel_env.py).
Batches alternate: even batches start from the home pose, odd ones after the other cube was already sorted,
so the arm starts above a tray, as in the second pick of the sort task. With --episodes <= ~240 at 256
worlds, all episodes come from batch 0 and start from home.

Usage:  python vla/record_picks.py [--episodes=200] [--worlds=256] [--no-gui] [--oracle [--oracle-noise=CM]]
Writes runs/datasets/so101_pick (repo id local/so101_pick).
--oracle (diagnostic): state gets the target cube's start x, y in cm appended (8 values); writes so101_pick_oracle. Layout seeds 100000+ (sort evals use 1000+).
--oracle-noise=CM: that position gets one random offset per episode (normal, CM per axis), like a detector's error; the
actions still go to the true cube, so the policy has to correct with the cameras. Writes so101_pick_oracle_n<CM> (0.5 -> n0p5).
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco.viewer
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101.gpu_sort import PICK, PLACE, SortBatch
from so101.pixel_env import IMAGE_SIZE
from so101.skill_env import CUBES

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
EPISODES, WORLDS = int(ARGS.get("episodes", 200)), int(ARGS.get("worlds", 256))
GUI = "--no-gui" not in sys.argv
ORACLE = "--oracle" in sys.argv
NOISE = float(ARGS.get("oracle-noise", 0))
NAME = ("so101_pick_oracle" + (f"_n{NOISE:g}".replace(".", "p") if NOISE else "")) if ORACLE else "so101_pick"
ROOT = Path("runs/datasets") / NAME
FPS = 33  # one control tick = 30 ms
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def batch(seed0, after_sort, viewer):
    """Run one pick per world. Returns kept episodes [(states, mocap, actions, cube)] (radians)."""
    env = SortBatch(WORLDS)
    env.reset(range(seed0, seed0 + WORLDS))
    cube = np.random.default_rng(seed0).integers(0, len(CUBES), WORLDS)  # which cube to pick, per world
    if after_sort:  # sort the other cube first, with the rule's skills
        other = 1 - cube
        env.step(np.column_stack([np.full(WORLDS, PICK), other]))
        env.step(np.column_stack([np.full(WORLDS, PLACE), len(CUBES) + other]))
    states, actions, active = [env.qpos.cpu().numpy()], [], []

    def tick():
        states.append(env.qpos.cpu().numpy())
        actions.append(env.ctrl.cpu().numpy())
        active.append((env.phase > 0).cpu().numpy())
        if viewer and len(actions) % 2 == 0:  # world 0 live in the viewer
            with viewer.lock():
                env.mirror(viewer.vdata)
            viewer.sync()

    env.on_tick = tick
    _, _, _, ok = env.step(np.column_stack([np.full(WORLDS, PICK), cube]))
    mocap = env.mocap.cpu().numpy()
    states, actions, active = np.array(states), np.array(actions), np.array(active)
    kept = []
    for w in np.flatnonzero(ok):
        # a tick counts if the skill was running when it began; the last tick of the skill is included
        t = int(active[:, w].sum()) + 1
        kept.append((states[:t, w], mocap[w], actions[:t, w], int(cube[w])))
    return kept


def main():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    m = mujoco.MjModel.from_xml_path(str(Path("so101/assets/scene_sort.xml")))
    lo, hi = m.actuator_ctrlrange.T

    def units(q):  # radians -> LeRobot SO-101 units (as SO101PixelReach.to_lerobot)
        u = (q - lo) / (hi - lo)
        return np.concatenate([u[:5] * 200 - 100, u[5:] * 100]).astype(np.float32)

    img = {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": (6,), "names": JOINTS}
    state = {"dtype": "float32", "shape": (8,), "names": JOINTS + ["target_x_cm", "target_y_cm"]} if ORACLE else vec
    ds = LeRobotDataset.create(f"local/{NAME}", FPS, root=ROOT, robot_type="so101_follower",
                               streaming_encoding=True, features={"observation.images.front": img, "observation.images.wrist": img,
                                         "observation.state": state, "action": vec})
    data, renderer = mujoco.MjData(m), mujoco.Renderer(m, IMAGE_SIZE, IMAGE_SIZE)
    viewer = None
    if GUI:
        vdata = mujoco.MjData(m)
        viewer = mujoco.viewer.launch_passive(m, vdata)
        viewer.vdata = vdata
    t0, saved, tried, frames, seed, kept_total = time.time(), 0, 0, 0, 100_000, 0
    noise = np.random.default_rng(0)  # hint offsets, one per saved episode
    while saved < EPISODES:
        t = time.time()
        eps = batch(seed, after_sort=(seed // WORLDS) % 2 == 1, viewer=viewer)
        tried, seed, kept_total = tried + WORLDS, seed + WORLDS, kept_total + len(eps)
        print(f"GPU: {len(eps)}/{WORLDS} picks succeeded in {time.time() - t:.0f} s", flush=True)
        t = time.time()
        for states, mocap, actions, cube in eps[:EPISODES - saved]:
            data.mocap_pos[:] = mocap
            adr = m.jnt_qposadr[m.body(CUBES[cube]).jntadr[0]]
            extra = states[0][adr:adr + 2] * 100 if ORACLE else np.zeros(0)  # target cube start, cm
            if NOISE:
                extra = extra + noise.normal(0, NOISE, 2)
            for s, a in zip(states[:-1], actions):  # observation before the tick, target sent at the tick
                data.qpos[:] = s
                mujoco.mj_forward(m, data)
                cams = {}
                for name in ("front", "wrist_cam"):
                    renderer.update_scene(data, camera=name)
                    cams[name] = renderer.render().copy()
                ds.add_frame({"observation.images.front": cams["front"], "observation.images.wrist": cams["wrist_cam"],
                              "observation.state": np.concatenate([units(s[:6]), extra]).astype(np.float32), "action": units(a),
                              "task": f"pick up the {CUBES[cube].replace('_', ' ')}"})
                frames += 1
            ds.save_episode()
            saved += 1
        print(f"CPU: rendered and saved {saved} episodes so far ({time.time() - t:.0f} s)", flush=True)
    ds.finalize()
    print(f"done: {saved} episodes, {frames} frames, {kept_total}/{tried} scripted picks succeeded, {time.time() - t0:.0f} s -> {ROOT}")


if __name__ == "__main__":
    main()
