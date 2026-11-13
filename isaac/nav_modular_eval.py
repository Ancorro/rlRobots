"""Walking-between-tables from parts that already work: the manipulation BC picks, the generator's path follower walks,
then the object is placed. No nav BC.

Each trial sets the scene up as generate_data.py does (random manipulation demo start state, end table and obstacles
placed at random, path planned). Phases:
  pick      manipulation BC (Plan #2, 45/50 on its own task) until the object is --lift m up;
  walk      wrist and finger targets frozen relative to the base, hip height kept; generate_data.py's pure-pursuit
            controller follows the planned path to the approach pose, then goes straight to the goal pose and turns
            to the goal heading (the goal is the start pose relative to the start table, moved to the end table);
  place     --place=bc: the manipulation BC again, its observations moved from the end table's frame into the start
            table's (where it was trained) and its wrist targets moved back;
            --place=replay: generate_data.py's drop-off, the recorded demo's release replayed relative to the end table.
Success: the stock task's own check (task_done_pick_place: object in its target area, below 1.1 m, slower than
0.2 m/s on each axis, right wrist pulled back) evaluated in the start table's frame, for --hold steps. Also recorded:
the first step the object is over the end table's footprint above 0.5 m (nav_bc_feats.on_table; true while still in
the hand, so not a success). Main score, released: the object at rest (< 0.05 m/s) on the end table's footprint below
0.85 m with the right wrist pulled back (start-table frame x < 0.26), for 100 steps or at the trial's end. Drops: the env's object_dropping term (object below 0.5 m).

  /isaac-sim/python.sh nav_modular_eval.py --pick_ckpt m.pth --pick_min a --pick_max b --trials 20 --out r.json \
      [generate_data.py args: --device cpu --headless --task ... --dataset ... --enable_pinocchio]
"""
import argparse
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
p = argparse.ArgumentParser()
p.add_argument("--pick_ckpt", required=True)
p.add_argument("--pick_min", type=float, required=True)
p.add_argument("--pick_max", type=float, required=True)
p.add_argument("--place_ckpt", help="separate BC for the place phase (nav_to_place.py data); default the pick BC")
p.add_argument("--place_min", type=float)
p.add_argument("--place_max", type=float)
p.add_argument("--place", choices=["bc", "replay"], default="bc")
p.add_argument("--trials", type=int, default=20)
p.add_argument("--horizon", type=int, default=2500)
p.add_argument("--hold", type=int, default=10)
p.add_argument("--lift", type=float, default=0.06, help="hand-off height above the start, m")
p.add_argument("--pick_max_steps", type=int, default=400)
p.add_argument("--replay_from", type=int, default=131, help="recording step the drop-off replay starts at")
p.add_argument("--hip", type=float, default=0.8, help="hip height while walking (generate_data.py: 0.8); <0 keeps BC's")
p.add_argument("--carry", choices=["frozen", "demo"], default="frozen",
               help="frozen: wrists stay where the lift ended; demo: blend to the recorded carry pose (relative to base)")
p.add_argument("--carry_step", type=int, default=130, help="recording step of the demo carry pose")
p.add_argument("--blend", type=int, default=50, help="steps to blend into the demo carry pose")
p.add_argument("--reach", type=float, default=0.2,
               help="waypoint and goal tolerance, m (generate_data.py 0.1: AGILE stalls at 0.10-0.14 m under its ~0.1 m/s command)")
p.add_argument("--settle", type=int, default=0, help="steps standing still, arms frozen, before the BC place")
p.add_argument("--ramp", type=int, default=0, help="steps blending the wrist targets from the frozen pose into the BC's")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
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
import numpy as np  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
import torch  # noqa: E402

import nav_bc_feats as F  # noqa: E402

args = g["args_cli"]
CFG_REACH = ours.reach
PICK_KEYS = ["left_eef_pos", "left_eef_quat", "right_eef_pos", "right_eef_quat", "hand_joint_state", "object"]
CFG = g["LocomanipulationSDGControlConfig"]()  # generate_data.py's defaults (gains, 0.2 rad, 1.0 m approach)
CFG.distance_threshold = CFG_REACH


def np1(t):
    return t.detach().cpu().numpy().astype(np.float64).reshape(-1)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def end_state(env, m):
    """Object and right wrist in the start table's frame, object speed: what the stock success check looks at
    (target x 0.40-0.85, y 0.35-0.60, z < 1.10, wrist x < 0.26, each speed axis < 0.20)."""
    ob, rb = env.scene["object"].data, env.scene["robot"].data
    o = env.scene.env_origins
    x = m.pose(np.concatenate([np1(ob.root_pos_w - o), np1(ob.root_quat_w)]))
    w = m.pose(np.concatenate([np1(rb.body_pos_w[:, rb.body_names.index("right_wrist_yaw_link")] - o), [1.0, 0, 0, 0]]))
    return dict(obj=[round(float(v), 3) for v in x[:3]], wrist_x=round(float(w[0]), 3),
                speed=round(float(np.abs(m.vec(np1(ob.root_vel_w)[:3])).max()), 3))


def released(env, m, obj, table):
    e = end_state(env, m)
    return bool(F.on_table(obj, table) and obj[2] < 0.85 and e["speed"] < 0.05 and e["wrist_x"] < 0.26)


def placed(env, m):
    ob, rb = env.scene["object"].data, env.scene["robot"].data
    o = env.scene.env_origins
    return F.placed(m, np.concatenate([np1(ob.root_pos_w - o), np1(ob.root_quat_w)]), np1(ob.root_vel_w)[:3],
                    np1(rb.body_pos_w[:, rb.body_names.index("right_wrist_yaw_link")] - o))


def bc_obs(po, m=None):
    o = {k: np1(po[k]) for k in PICK_KEYS}
    if m is not None:
        for s in ("left", "right"):
            x = m.pose(np.concatenate([o[f"{s}_eef_pos"], o[f"{s}_eef_quat"]]))
            o[f"{s}_eef_pos"], o[f"{s}_eef_quat"] = x[:3], x[3:]
        ob = o["object"]
        o["object"] = np.concatenate([m.pose(ob[:7]), m.vec(ob[7:10]), m.vec(ob[10:13])])
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in o.items()}


def bc_act(pick, po, m=None, lo=None, hi=None):
    lo, hi = (ours.pick_min, ours.pick_max) if lo is None else (lo, hi)
    a = (np.asarray(pick(bc_obs(po, m)), dtype=np.float64) + 1) * (hi - lo) / 2 + lo
    if m is not None:
        for s in (slice(0, 7), slice(7, 14)):
            x = a[s].copy()
            x[3:] /= np.linalg.norm(x[3:])
            a[s] = m.back(x)
    return a


def main():
    env_cfg = g["parse_env_cfg"](args.task.split(":")[-1], device=args.device, num_envs=1)
    env_cfg.sim.device = "cpu"
    env_cfg.recorders = None
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    data = g["HDF5DatasetFileHandler"]()
    data.open(args.dataset)
    names = list(data.get_episode_names())
    random.seed(ours.seed)
    np.random.seed(ours.seed)
    torch.manual_seed(ours.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    nav_v = g["compute_navigation_velocity"]
    rows = []
    for t in range(ours.trials):
        pick, _ = FileUtils.policy_from_checkpoint(ckpt_path=ours.pick_ckpt, device=device)
        pick.start_episode()
        place = pick
        if ours.place_ckpt:
            place, _ = FileUtils.policy_from_checkpoint(ckpt_path=ours.place_ckpt, device=device)
        ep = data.load_episode(random.choice(names), args.device)
        env.reset_to(state=ep.get_initial_state(), env_ids=torch.tensor([0]), is_relative=True)
        _, path, goal, approach = g["setup_navigation_scene"](env, ep, CFG.approach_distance, True)
        goal_p, appr_p = np1(goal.get_pose()), np1(approach.get_pose())
        table = np1(env.get_end_fixture().get_pose())
        m = F.TableMap(np1(env.get_start_fixture().get_pose()), table)
        po = env.observation_manager.compute()["policy"]
        z0 = float(np1(po["object_pos"])[2])
        phase, rec, out = "pick", ours.replay_from, g["LocomanipulationSDGOutputData"]()
        rest, held, r = 0, 0, dict(trial=t, success=False, steps=0, dropped=False, fell=False, ended_by=[], phase_steps={})
        for i in range(ours.horizon):
            base = np1(env.get_base().get_pose())
            pose2 = env.get_base().get_pose_2d()[0]
            r["phase_steps"].setdefault(phase, i)
            if phase == "pick":
                a = bc_act(pick, po)
                if float(np1(po["object_pos"])[2]) - z0 > ours.lift:
                    phase = "path"
                    held_b, hip = F.actions_to_base(base, a)[:28], (a[31] if ours.hip < 0 else ours.hip)
                    start_b, t_lift = held_b.copy(), i  # wrists in the base frame, fingers as sent
                    if ours.carry == "demo":
                        rec_c = env.load_input_data(ep, ours.carry_step)
                        rb_ = np1(rec_c.base_pose)
                        demo_b = np.concatenate([F.to_base(rb_, np1(rec_c.left_hand_pose_target)),
                                                 F.to_base(rb_, np1(rec_c.right_hand_pose_target))])
                        for q in (slice(3, 7), slice(10, 14)):  # same hemisphere as the start, for the nlerp
                            demo_b[q] *= np.sign(np.dot(demo_b[q], start_b[q])) or 1.0
                elif i >= ours.pick_max_steps:
                    break
            if phase in ("path", "approach", "turn", "settle"):
                if phase == "path":
                    _, s_near, _, _ = path.find_nearest(pose2[:2])  # None for a one-point path
                    target = (torch.tensor(appr_p[:2], dtype=pose2.dtype) if s_near is None
                              else path.get_point_by_distance(distance=s_near + CFG.following_offset))
                    if s_near is None or np.linalg.norm(base[:2] - appr_p[:2]) < CFG.distance_threshold:
                        phase = "approach"
                if phase == "approach":
                    target = torch.tensor(goal_p[:2], dtype=pose2.dtype)
                    if np.linalg.norm(base[:2] - goal_p[:2]) < CFG.distance_threshold:
                        phase = "turn"
                if phase == "turn":
                    dyaw = wrap(float(np1(goal.get_pose_2d()[0])[2]) - float(pose2[2]))
                    v, w = 0.0, CFG.angular_gain * dyaw
                    if abs(dyaw) < CFG.angle_threshold:
                        phase, t_set = "settle", i
                if phase == "settle":
                    v, w = 0.0, 0.0
                    if i - t_set >= ours.settle:
                        phase, t_place = "place", i
                        place.start_episode()
                if phase in ("path", "approach"):
                    v, w = (float(x) for x in nav_v(pose2, target, CFG))
                if ours.carry == "demo":  # nlerp from the lift pose to the demo carry pose
                    f = min(1.0, (i - t_lift) / ours.blend)
                    held_b = start_b.copy()
                    held_b[:14] = (1 - f) * start_b[:14] + f * demo_b
                if i % 250 == 0:
                    r.setdefault("track", []).append([i, phase, round(float(np.linalg.norm(base[:2] - appr_p[:2])), 2),
                                                      round(float(np.linalg.norm(base[:2] - goal_p[:2])), 2), round(v, 2)])
                if phase != "place":
                    wr = F.actions_from_base(base, np.concatenate([held_b, np.zeros(4)]))[:28]
                    a = np.concatenate([wr, [v, 0.0, w, hip]])
            if phase == "place":
                if ours.place == "bc":
                    a = (bc_act(place, po, m, ours.place_min, ours.place_max) if ours.place_ckpt
                         else bc_act(pick, po, m))
                    f = 1.0 if ours.ramp <= 0 else min(1.0, (i - t_place) / ours.ramp)
                    if f < 1.0:  # wrists from the frozen carry pose into the BC's targets, base frame
                        a_b = F.actions_to_base(base, a)
                        for q in (slice(3, 7), slice(10, 14)):
                            a_b[q] *= np.sign(np.dot(a_b[q], held_b[q])) or 1.0
                        a_b[:14] = (1 - f) * held_b[:14] + f * a_b[:14]
                        a[:14] = F.actions_from_base(base, a_b)[:14]
                else:
                    rec, _ = g["handle_drop_off_state"](env, ep, rec, goal, CFG, out)  # past the recording's end: last targets
                    a = np1(env.build_action_vector(
                        base_velocity_target=out.base_velocity_target,
                        left_hand_joint_positions_target=out.left_hand_joint_positions_target,
                        right_hand_joint_positions_target=out.right_hand_joint_positions_target,
                        left_hand_pose_target=out.left_hand_pose_target,
                        right_hand_pose_target=out.right_hand_pose_target))
            if phase == "place" and not (i - r["phase_steps"].setdefault("place", i)) % 100:
                r.setdefault("place_track", []).append(end_state(env, m))  # every 100 steps of the place phase
            obs_dict, _, term, trunc, _ = env.step(torch.tensor(a, dtype=torch.float32).view(1, -1))
            po = obs_dict["policy"]
            obj = np.concatenate([np1(po["object_pos"]), np1(po["object_rot"])])
            r["steps"] = i + 1
            if bool(term[0] or trunc[0]):  # obs here are already post-reset
                tm = env.termination_manager
                r["ended_by"] = [n for n in tm.active_terms if bool(tm.get_term(n)[0])] or ["truncated"]
                r["dropped"] = "object_dropping" in r["ended_by"]
                break
            if F.on_table(obj, table):
                r.setdefault("over_table_step", i)
            held = held + 1 if placed(env, m) else 0
            rest = rest + 1 if phase == "place" and released(env, m, obj, table) else 0
            if rest >= 100:
                break
            if held >= ours.hold:
                r["success"] = True
                break
            if base[2] < 0.4:
                r["fell"] = True
                break
        r["last_phase"] = phase
        r["released"] = r["success"] or (phase == "place" and not r["dropped"] and not r["fell"]
                                          and released(env, m, obj, table))
        if phase == "place" and not r["dropped"]:
            r["end"] = end_state(env, m)
        rows.append(r)
        print(f"[nav_modular] {r}", flush=True)
    summary = dict(n=len(rows), place=ours.place, place_ckpt=ours.place_ckpt, settle=ours.settle, ramp=ours.ramp,
                   released=sum(r["released"] for r in rows), success=sum(r["success"] for r in rows),
                   over_table=sum("over_table_step" in r for r in rows), dropped=sum(r["dropped"] for r in rows), fell=sum(r["fell"] for r in rows),
                   lifted=sum("path" in r["phase_steps"] for r in rows),
                   reached=sum("place" in r["phase_steps"] for r in rows))
    print(f"[nav_modular] summary {summary}", flush=True)
    json.dump(dict(summary=summary, rows=rows), open(ours.out, "w"), indent=1)
    env.close()


with torch.no_grad():
    main()
g["simulation_app"].close()
