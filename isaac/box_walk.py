"""The box task with walking (owner 2026-10-21: script the box into the walk scene): the G1 picks the box up with the
scripted two-hand squeeze (isaac/box_scripted.py), walks to the second table along the planned path, sets the box
down there, lets go and backs its hands off. --pick bc (owner 2026-10-22: "do what works"): the standing box BC
(box_bcplay.py, 17/20 with the script's set-down) grabs and lifts instead, then hands its last wrist poses and fingers
to the walk and set-down below.

Scene and walking as isaac/nav_modular_eval.py: Isaac Lab's locomanipulation_sdg scene (second table and obstacles
placed at random each trial, path planned), its pure-pursuit path follower, the goal pose = the start pose relative to
the start table, moved to the end table. The steering wheel is swapped for the box (isaac/box_cfgs.py: BOX_SIZE,
BOX_MASS, BOX_POS). Wrist targets are kept in the robot's base (pelvis) frame, so while walking the box moves with it.

Success: the box at rest on the end table's footprint, at the height it started at (the tables are alike), both wrists
5 cm clear of its sides, for --hold steps. Drops: the box below 0.5 m. Falls: pelvis below 0.4 m.
Runs one robot at a time (the scene set-up and planner handle one env); run several jobs side by side.

  /isaac-sim/python.sh box_walk.py --trials 5 --out r.json [generate_data.py args: --device cpu --headless --task ...
      --dataset ... --enable_pinocchio]
"""
import argparse
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
p = argparse.ArgumentParser()
p.add_argument("--trials", type=int, default=5)
p.add_argument("--horizon", type=int, default=3000)
p.add_argument("--hold", type=int, default=10)
p.add_argument("--hip", type=float, default=0.72, help="pelvis height command while picking and placing (box_scripted.py: 0.72)")
p.add_argument("--walk_hip", type=float, default=0.8, help="pelvis height while walking (generate_data.py and nav_modular_eval.py: 0.8)")
p.add_argument("--carry_in", type=float, default=0.0, help="after the lift, pull both wrists this far toward the body (m)")
p.add_argument("--speed", type=float, default=1.0, help="scale on the path follower's forward speed")
p.add_argument("--settle", type=int, default=0, help="steps standing still, arms at rest, before the pick")
p.add_argument("--place_aim", type=int, default=1, help="1: lower the hands by the box's measured height above its start, not a fixed --lift")
p.add_argument("--min_turn", type=float, default=0.5, help="least turning rate command while turning in place (rad/s)")
p.add_argument("--turn_steps", type=int, default=400, help="> 0: place after this many turning steps even if not within the angle")
p.add_argument("--hip_steps", type=int, default=50, help="steps to change the pelvis height between the two")
p.add_argument("--reach", type=float, default=0.2, help="waypoint and goal tolerance, m (as nav_modular_eval.py)")
p.add_argument("--palm", type=float, default=0.012)
p.add_argument("--thumb", type=float, default=0.09)  # walk15: 0.09 31/60 vs 0.075 22/60
p.add_argument("--squeeze", type=float, default=0.035)
p.add_argument("--close", type=float, default=0.4)
p.add_argument("--grip_below", type=float, default=0.05)
p.add_argument("--lift", type=float, default=0.12)
p.add_argument("--slow", type=float, default=2.0)
p.add_argument("--jitter", type=float, default=0.03)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--pick", choices=["script", "bc"], default="script")
p.add_argument("--pick_ckpt", help="--pick bc: the box BC checkpoint (box_bcplay.py's)")
p.add_argument("--pick_min", type=float)
p.add_argument("--pick_max", type=float)
p.add_argument("--handoff", type=int, default=100, help="--pick bc: walk once the box has been 5 cm up this many steps")
p.add_argument("--pick_steps", type=int, default=800, help="--pick bc: give up if not handed off by then")
p.add_argument("--clean_reset", type=int, default=20, help="--pick bc: hold the arms this many steps, then put the box back at rest (box_bcplay.py)")
p.add_argument("--sized", action="store_true", help="--pick bc: the BC also takes the box size (BoxSized task's nets)")
p.add_argument("--carry_rise", type=float, default=0.12, help="--pick bc: after the handoff lower the hands until the box is this high (m; the script carries at ~0.12, the BC lifts 0.2-0.45 and the robot fell)")
p.add_argument("--bc_arms", default="-0.148,0.196,0.847,0.149,0.196,0.847", help="--pick bc: move the wrists here (env frame) before the BC starts: box_bcplay.py's reset pose")
p.add_argument("--centre", type=float, default=0.0,  # off: walk11/12 side-stepping 0/60 (base still moving at the grab)
               help="> 0: side-step until the box is within this many m of straight ahead before the pick (walk10: lopsided picks 0/17)")
p.add_argument("--closed_loop", type=int, default=1, help="re-aim beside and squeeze at the box every step")
p.add_argument("--record", action="store_true", help="save each trial's motion next to --out (.npz): pelvis pose, joints, box pose per step")
ours, rest = p.parse_known_args()
script = os.path.join(os.environ.get("ISAACLAB", os.path.expandvars("/nfs/hpc/share/$USER/isaac/IsaacLab")),
                      "scripts/imitation_learning/locomanipulation_sdg/generate_data.py")
sys.argv = [script, "--output_file", "/tmp/unused.hdf5"] + rest

import nav_planner_shim as S  # noqa: E402  (hooks AppLauncher: planner import, camera drop)

head = open(script).read().split(S.MARK, 1)[0]
g = {"__name__": "nav_gen", "__file__": script}
exec(compile(head, script, "exec"), g)  # starts the app, defines setup_navigation_scene, the controller etc.

import random  # noqa: E402

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import nav_bc_feats as F  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
from box_cfgs import BELOW, SIZE, box_object  # noqa: E402

args = g["args_cli"]
CFG = g["LocomanipulationSDGControlConfig"]()
CFG.distance_threshold = ours.reach
L_IDX, R_IDX = [0, 1, 2, 6, 7, 8, 12], [3, 4, 5, 9, 10, 11, 13]
PICK_KEYS = ["left_eef_pos", "left_eef_quat", "right_eef_pos", "right_eef_quat", "hand_joint_state", "object"]


def np1(t):
    return t.detach().cpu().numpy().astype(np.float64).reshape(-1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def held(robot, box, box_c, z0):
    """The box is really held: 5 cm up, upright (its z axis within ~25 deg of vertical) and between the two wrists.
    A box knocked onto its side reads 0.064 m 'up' (walk6 and walk7: 30 trials walked off with it lying on the table)."""
    c = box_c()
    q = np1(box.data.root_quat_w)
    up_z = 1 - 2 * (q[1] ** 2 + q[2] ** 2)  # z component of the box's z axis
    names = robot.data.body_names
    lw, rw = (np1(robot.data.body_pos_w[:, names.index(f"{s}_wrist_yaw_link")]) for s in ("left", "right"))
    # sideways between the wrists: the box centre projects inside the wrist-to-wrist segment (the grip puts the wrists
    # ~0.15 m in front of the centre, thumbs on the near face, so a plain distance test rejected real grips: walk8 0/60)
    d = (rw - lw)[:2]
    t = float(np.dot((c - lw)[:2], d) / max(np.dot(d, d), 1e-9))
    between = 0.0 < t < 1.0 and np.linalg.norm(d) < SIZE[0] + 0.15
    return bool(c[2] - z0 > 0.05 and up_z > 0.5 and between)  # within 60 deg: on its side is ~90 (walk9: 0.9 rejected
    # boxes still in the hands, tilted, 0.1-0.24 m up)


def bc_pick(env, robot, box, start, st, r, step, mark, box_c, z0, device, OPEN):
    """The box BC grabs and lifts (box_bcplay.py's loop); on handoff its last wrist poses (base frame) and fingers go
    into st for the walk. Returns True if the trial ended."""
    o = env.scene.env_origins[0]

    def pose(name):
        i = robot.data.body_names.index(name)
        return np.concatenate([np1(robot.data.body_pos_w[0, i] - o), np1(robot.data.body_quat_w[0, i])])

    if ours.clean_reset:  # arms move to the BC's training reset pose and settle, then the box goes back to its start at rest
        L0, R0 = pose("left_wrist_yaw_link"), pose("right_wrist_yaw_link")
        tgt = np.array([float(v) for v in ours.bc_arms.split(",")])
        n = 60
        for k in range(1, n + ours.clean_reset + 1):
            f = min(1.0, k / n)
            L, R = L0.copy(), R0.copy()
            L[:3], R[:3] = L0[:3] + f * (tgt[:3] - L0[:3]), R0[:3] + f * (tgt[3:] - R0[:3])
            env.step(torch.tensor(np.concatenate([L, R, OPEN, [0, 0, 0, ours.hip]]), dtype=torch.float32).view(1, -1))
        box.write_root_pose_to_sim(start[:, :7])
        box.write_root_velocity_to_sim(torch.zeros(1, 6))
        env.sim.forward()
        env.scene.update(dt=0.0)
    st["po"] = env.observation_manager.compute()["policy"]
    pick, _ = FileUtils.policy_from_checkpoint(ckpt_path=ours.pick_ckpt, device=device)
    pick.start_episode()
    mark("bc")
    r["bc_start"] = dict(box=np1(st["po"]["object"])[:3].round(3).tolist(), left=np1(st["po"]["left_eef_pos"]).round(3).tolist(),
                         right=np1(st["po"]["right_eef_pos"]).round(3).tolist())
    up = 0
    for i in range(ours.pick_steps):
        obs = {k: torch.tensor(np1(st["po"][k]), dtype=torch.float32) for k in PICK_KEYS}
        if ours.sized:  # the walk scene's env has no object_size term
            obs["object_size"] = torch.tensor(list(SIZE) + [float(os.environ.get("BOX_SHAPE", "box") == "cylinder")], dtype=torch.float32)
        a = (np.asarray(pick(obs), dtype=np.float64) + 1) * (ours.pick_max - ours.pick_min) / 2 + ours.pick_min
        a[28:31] = 0.0  # stand still while picking
        if step(a):
            return True
        up = up + 1 if held(robot, box, box_c, z0) else 0
        if up >= ours.handoff:
            a_n = a.copy()
            for q in (slice(3, 7), slice(10, 14)):
                a_n[q] /= np.linalg.norm(a_n[q])  # the net's quaternions are not unit
            a_b = F.actions_to_base(np1(env.get_base().get_pose()), a_n)
            st["L"], st["Lq"], st["R"], st["Rq"] = a_b[0:3], a_b[3:7], a_b[7:10], a_b[10:14]
            st["hand"] = a[14:28].copy()
            r["handoff"] = r["steps"]
            r["handoff_rise"] = round(float(box_c()[2] - z0), 3)
            drop = max(0.0, box_c()[2] - z0 - ours.carry_rise)  # carry low, as the script does
            L1, R1 = st["L"].copy(), st["R"].copy()
            for k in range(1, 41):
                st["L"], st["R"] = L1 - [0, 0, drop * k / 40], R1 - [0, 0, drop * k / 40]
                if step():
                    return True
            return False
    r["handoff"] = None
    return False


def main():
    env_cfg = g["parse_env_cfg"](args.task.split(":")[-1], device=args.device, num_envs=1)
    env_cfg.sim.device = "cpu"
    env_cfg.recorders = None
    env_cfg.scene.object = box_object()
    env_cfg.terminations.success = None  # the wheel's check, at the start table: not this task's
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    robot, box = env.scene["robot"], env.scene["object"]
    data = g["HDF5DatasetFileHandler"]()
    data.open(args.dataset)
    names = list(data.get_episode_names())
    with h5py.File(args.dataset, "r") as f:  # finger angles, as box_scripted.py
        act = f["data"]["demo_0"]["actions"][:]
    OPEN = act[0, 14:28].copy()
    SHUT = OPEN.copy()
    SHUT[L_IDX], SHUT[R_IDX] = act[105, 14:28][L_IDX], act[150, 14:28][R_IDX]
    HAND = OPEN + ours.close * (SHUT - OPEN)
    random.seed(ours.seed)
    np.random.seed(ours.seed)
    rng = np.random.default_rng(ours.seed)
    nav_v = g["compute_navigation_velocity"]
    side = SIZE[0] / 2 + ours.palm
    rows, recs = [], []
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    for t in range(ours.trials):
        env.reset()
        ep = data.load_episode(random.choice(names), args.device)  # only its start pose and table pose are used
        start = box.data.default_root_state.clone()
        start[:, :3] += env.scene.env_origins
        start[:, :2] += torch.tensor(rng.uniform(-ours.jitter, ours.jitter, 2), dtype=torch.float32)
        box.write_root_pose_to_sim(start[:, :7])
        box.write_root_velocity_to_sim(torch.zeros(1, 6))
        _, path, goal, approach = g["setup_navigation_scene"](env, ep, CFG.approach_distance, True)
        goal_p, appr_p = np1(goal.get_pose()), np1(approach.get_pose())
        table = np1(env.get_end_fixture().get_pose())
        r = dict(trial=t, success=False, dropped=False, fell=False, steps=0, phase_steps={}, max_rise=0.0,
                 end_table=table.round(3).tolist(), start_table=np1(env.get_start_fixture().get_pose()).round(3).tolist())

        def base():
            return np1(env.get_base().get_pose())

        def box_c():  # box centre, world frame
            return np1(box.data.root_pos_w) + np.array([0, 0, SIZE[2] / 2 - BELOW])

        def wrist(name):
            return np1(robot.data.body_pos_w[:, robot.data.body_names.index(name)])

        for _ in range(ours.settle):  # let the robot and the box come to rest after the reset
            env.step(torch.tensor(np.concatenate([np1(env.action_manager.action)[:28], [0.0, 0.0, 0.0, ours.hip]]),
                                  dtype=torch.float32).view(1, -1))
        b0 = base()
        z0 = box_c()[2]
        # rest wrist poses, base frame; their orientations are kept for the whole trial
        Lb = F.to_base(b0, np.concatenate([wrist("left_wrist_yaw_link"), np1(robot.data.body_quat_w[:, robot.data.body_names.index("left_wrist_yaw_link")])]))
        Rb = F.to_base(b0, np.concatenate([wrist("right_wrist_yaw_link"), np1(robot.data.body_quat_w[:, robot.data.body_names.index("right_wrist_yaw_link")])]))
        rest_L, rest_R = Lb[:3].copy(), Rb[:3].copy()
        # the box's sideways axis (its x, env frame) and up, in the base frame at the start
        ex = F.qrot(F.qconj(b0[3:]), np.array([1.0, 0, 0]))
        ex = ex / np.linalg.norm(ex)
        ey = F.qrot(F.qconj(b0[3:]), np.array([0, 1.0, 0]))
        st = dict(L=Lb[:3].copy(), R=Rb[:3].copy(), Lq=Lb[3:].copy(), Rq=Rb[3:].copy(), hand=OPEN.copy(), v=0.0, vy=0.0, w=0.0,
                  phase="pick", held=0, hip=ours.hip, po=None)
        frames = []

        def step(a=None):  # a: a whole action (the BC's); otherwise built from st's base-frame wrist targets
            if a is None:
                b = base()
                L = F.from_base(b, np.concatenate([st["L"], st["Lq"]]))
                R = F.from_base(b, np.concatenate([st["R"], st["Rq"]]))
                a = np.concatenate([L, R, st["hand"], [st["v"], st["vy"], st["w"], st["hip"]]])
            obs, _, term, trunc, _ = env.step(torch.tensor(a, dtype=torch.float32).view(1, -1))
            st["po"] = obs["policy"]
            r["steps"] += 1
            if ours.record:
                frames.append(torch.cat([robot.data.root_pos_w[0] - env.scene.env_origins[0], robot.data.root_quat_w[0],
                                         robot.data.joint_pos[0], box.data.root_pos_w[0] - env.scene.env_origins[0],
                                         box.data.root_quat_w[0]]).numpy().astype(np.float32))
            c = box_c()
            r["max_rise"] = max(r["max_rise"], c[2] - z0)
            if bool(term[0] or trunc[0]):
                tm = env.termination_manager
                r["ended_by"] = [n for n in tm.active_terms if bool(tm.get_term(n)[0])] or ["truncated"]
                r["dropped"] = "object_dropping" in r["ended_by"]
                return True
            if base()[2] < 0.4:
                r["fell"] = True
                return True
            if st["phase"] in ("place", "wait"):
                vel = np.abs(np1(box.data.root_vel_w)[:3]).max()
                lw, rw = wrist("left_wrist_yaw_link"), wrist("right_wrist_yaw_link")
                clear = min(np.linalg.norm((lw - c)[:2]), np.linalg.norm((rw - c)[:2])) > SIZE[0] / 2 + 0.05
                ok = F.on_table(np.concatenate([c, [1, 0, 0, 0]]), table) and abs(c[2] - z0) < 0.02 and vel < 0.2 and clear
                st["held"] = st["held"] + 1 if ok else 0
                if st["held"] >= ours.hold:
                    r["success"] = True
                    return True
            return r["steps"] >= ours.horizon

        def go(n, L=None, R=None, hand=None, aim=None):  # straight-line wrist moves in the base frame
            # aim: (dx, dz) re-aims the target at the box every step (closed loop, as box_scripted.py): the box shifts
            # while the hands close, and an open-loop squeeze tipped it (walk10-13: tipped picks had one wrist 6 cm
            # further from the box centre than the other, successes within 1 cm)
            aL, aR, ah = st["L"].copy(), st["R"].copy(), st["hand"].copy()
            n = int(n * ours.slow)
            for k in range(1, n + 1):
                f = k / n
                if aim is not None and ours.closed_loop:
                    L, R = at(*aim)
                if L is not None:
                    st["L"] = aL + f * (L - aL)
                if R is not None:
                    st["R"] = aR + f * (R - aR)
                if hand is not None:
                    st["hand"] = ah + f * (hand - ah)
                if step():
                    return True
            return False

        def at(dx, dz=0.0):  # wrist targets beside the box where it is now, base frame (as box_scripted.at)
            cb = F.to_base(base(), np.concatenate([box_c(), [1, 0, 0, 0]]))[:3]
            near = cb - ey * (SIZE[1] / 2 + ours.thumb)
            up = np.array([0, 0, SIZE[2] / 2 - ours.grip_below + dz])
            return near - ex * (side + dx) + up, near + ex * (side + dx) + up

        def mark(name):
            st["phase"] = name
            r["phase_steps"].setdefault(name, r["steps"])

        if t == 0:  # frames once: pelvis pose, box, and the box axes in the pelvis frame (the script assumes the robot faces +y)
            print(f"[box_walk] frames: base {b0.round(3).tolist()} box {box_c().round(3).tolist()} ex_b {ex.round(3).tolist()} "
                  f"ey_b {ey.round(3).tolist()} rest L {rest_L.round(3).tolist()} R {rest_R.round(3).tolist()}", flush=True)
        done = False
        vb = np1(robot.data.root_lin_vel_w)[:2]
        r["pre_pick"] = dict(base=base()[:3].round(3).tolist(), speed=round(float(np.linalg.norm(vb)), 3))
        if ours.centre > 0:  # side-step until the box is straight ahead (base-frame y = left)
            mark("centre")
            for k in range(400):
                off = F.to_base(base(), np.concatenate([box_c(), [1, 0, 0, 0]]))[1]  # not `side`: that is the grip half-width
                if abs(off) < ours.centre:
                    break
                st["vy"] = float(np.clip(1.5 * off, -0.3, 0.3))
                if abs(st["vy"]) < 0.1:
                    st["vy"] = 0.1 * np.sign(off)  # AGILE ignores small commands
                if step():
                    done = True
                    break
            st["vy"] = 0.0
            # no settle after: standing still before the pick ruined it (walk11, and --settle 100 2026-10-21: 0/12)
            r["pre_pick"]["after_centre_speed"] = round(float(np.linalg.norm(np1(robot.data.root_lin_vel_w)[:2])), 3)
            r["centre_steps"], r["centre_left"] = k, round(float(F.to_base(base(), np.concatenate([box_c(), [1, 0, 0, 0]]))[1]), 3)
        if ours.pick == "bc" and not done:
            done = bc_pick(env, robot, box, start, st, r, step, mark, box_c, z0, device, OPEN)
        for name, n, aim, hand in [] if (ours.pick == "bc" or done) else [("above", 40, (0.07, 0.10), None), ("beside", 30, (0.07, 0.0), None),
                                   ("squeeze", 30, (-ours.squeeze, 0.0), None), ("fingers", 20, None, HAND),
                                   ("lift", 40, (-ours.squeeze, ours.lift), None)]:
            mark(name)
            L, R = at(*aim) if aim else (None, None)
            if go(n, L, R, None if hand is None else hand, aim=aim if name in ("beside", "squeeze") else None):
                done = True
            r.setdefault("pick_track", []).append([name, (box_c() - np.concatenate([b0[:2], [0]]) * [1, 1, 0]).round(3).tolist(),
                                                    (wrist("left_wrist_yaw_link") - box_c()).round(3).tolist(),
                                                    (wrist("right_wrist_yaw_link") - box_c()).round(3).tolist()])
            if done:
                break
        r["lifted"] = bool(r["max_rise"] > 0.05)
        r["held"] = held(robot, box, box_c, z0) if not done else False
        q = np1(box.data.root_quat_w)
        r["tilt_deg"] = round(float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))), 1)  # walk only with the box really in the hands
        if not done and r["held"]:
            mark("walk")  # wrists stay where the lift left them (less --carry_in toward the body), in the base frame
            if ours.carry_in > 0 and go(40, st["L"] - ey * ours.carry_in, st["R"] - ey * ours.carry_in):
                done = True
            for k in range(1, ours.hip_steps + 1):  # stand up to the walking height first
                if done:
                    break
                st["hip"] = ours.hip + k / ours.hip_steps * (ours.walk_hip - ours.hip)
                if step():
                    done = True
                    break
            phase = "path"
            while not done:
                b = base()
                pose2 = env.get_base().get_pose_2d()[0]
                if phase == "path":
                    _, s_near, _, _ = path.find_nearest(pose2[:2])
                    target = (torch.tensor(appr_p[:2], dtype=pose2.dtype) if s_near is None
                              else path.get_point_by_distance(distance=s_near + CFG.following_offset))
                    if s_near is None or np.linalg.norm(b[:2] - appr_p[:2]) < CFG.distance_threshold:
                        phase = "approach"
                if phase == "approach":
                    target = torch.tensor(goal_p[:2], dtype=pose2.dtype)
                    if np.linalg.norm(b[:2] - goal_p[:2]) < CFG.distance_threshold:
                        phase = "turn"
                if phase == "turn":
                    t_turn = r["phase_steps"].setdefault("turn", r["steps"])
                    dyaw = wrap(float(np1(goal.get_pose_2d()[0])[2]) - float(pose2[2]))
                    w = CFG.angular_gain * dyaw
                    st["v"], st["w"] = 0.0, float(np.sign(w) * max(abs(w), ours.min_turn))
                    r["last_dyaw"] = round(dyaw, 3)
                    if abs(dyaw) < CFG.angle_threshold or (ours.turn_steps > 0 and r["steps"] - t_turn >= ours.turn_steps):
                        break
                else:
                    st["v"], st["w"] = (float(x) for x in nav_v(pose2, target, CFG))
                    st["v"] *= ours.speed
                if r["steps"] % 250 == 0:
                    r.setdefault("track", []).append([r["steps"], phase, round(float(np.linalg.norm(b[:2] - appr_p[:2])), 2),
                                                      round(float(np.linalg.norm(b[:2] - goal_p[:2])), 2), round(st["v"], 2),
                                                      round(float(box_c()[2] - z0), 3)])
                done = step()
            st["v"], st["w"] = 0.0, 0.0
            for k in range(1, ours.hip_steps + 1):  # back down to the picking height
                if done:
                    break
                st["hip"] = ours.walk_hip + k / ours.hip_steps * (ours.hip - ours.walk_hip)
                done = step()
            r["reached"] = not done
            if not done:
                mark("place")
                done = go(60)  # stand still, the box settles in the hands
                for _ in range(2 if ours.place_aim else 0):  # lower by how high the box really is (it rides 0.06-0.3 m up)
                    if done:
                        break
                    drop = np.array([0, 0, box_c()[2] - z0 - 0.01])
                    done = go(40, st["L"] - drop, st["R"] - drop)
                r["place_rel_height"] = round(float(box_c()[2] - z0), 3)
                drop = 0.0 if ours.place_aim else ours.lift - 0.01
                for n, L, R, hand in [] if done else [
                                      (40, st["L"] - [0, 0, drop], st["R"] - [0, 0, drop], None),
                                      (15, None, None, OPEN),
                                      (30, st["L"] - ex * 0.10 - [0, 0, drop], st["R"] + ex * 0.10 - [0, 0, drop], None),
                                      (40, rest_L, rest_R, None)]:
                    if go(n, L, R, hand):
                        done = True
                        break
                if not done:
                    mark("wait")
                    go(60)
        c = box_c()
        r["walk_m"] = round(float(np.linalg.norm(goal_p[:2] - b0[:2])), 2)
        r["end_on_table"] = bool(F.on_table(np.concatenate([c, [1, 0, 0, 0]]), table))
        r["end_rel_height"] = round(float(c[2] - z0), 3)
        r["last_phase"] = st["phase"]
        rows.append(r)
        if ours.record:
            recs.append(np.stack(frames) if frames else np.zeros((0, 1), np.float32))
        print(f"[box_walk] {r}", flush=True)
    summary = dict(n=len(rows), hip=ours.hip, success=sum(r["success"] for r in rows),
                   lifted=sum(r.get("lifted", False) for r in rows), held=sum(r.get("held", False) for r in rows), reached=sum(r.get("reached", False) for r in rows),
                   on_table=sum(r["end_on_table"] for r in rows), dropped=sum(r["dropped"] for r in rows),
                   fell=sum(r["fell"] for r in rows))
    print(f"[box_walk] summary {summary}", flush=True)
    json.dump(dict(summary=summary, args=vars(ours), rows=rows), open(ours.out, "w"), indent=1,
              default=lambda x: x.item() if hasattr(x, "item") else str(x))  # numpy bools and floats
    if ours.record:  # trials differ in length: one array each
        np.savez_compressed(ours.out.replace(".json", ".npz"), joint_names=np.array(robot.joint_names), dt=env.step_dt,
                            size=np.array(SIZE), below=BELOW, **{f"r{i}": f for i, f in enumerate(recs)})
    env.close()


with torch.no_grad():
    main()
g["simulation_app"].close()
