"""Roll out a nav BC policy (trained on nav_to_bc.py data) in the walking-between-tables scene and judge it.

Each trial sets the scene up as generate_data.py does (a random manipulation demo's start state, end table and
obstacles placed at random, path planned), then the policy alone drives the robot: observations and actions in the
robot frame (nav_bc_feats.py), wrists turned back to the env frame. Success: the stock task's
success check in the start table's frame (nav_bc_feats.placed: object in its target area at rest, right wrist pulled
back) for HOLD consecutive steps. Before 2026-10-20 success was the end table's footprint above 0.5 m, which also
passes while the object is still in the hand; that is now only recorded (over_table_step). Uses nav_planner_shim.py's planner import and camera drop.

  /isaac-sim/python.sh nav_bc_eval.py --checkpoint m.pth --norm x.norm.json --trials 20 --out r.json \
      [generate_data.py args: --device cpu --headless --task ... --dataset ... --enable_pinocchio]
"""
import argparse
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--norm", required=True)
p.add_argument("--trials", type=int, default=20)
p.add_argument("--horizon", type=int, default=2500)
p.add_argument("--hold", type=int, default=10)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--trace", default="", help="npz of per-step object, wrists, hands and actions for every trial")
ours, rest = p.parse_known_args()
script = os.path.join(os.environ.get("ISAACLAB", os.path.expandvars("/nfs/hpc/share/$USER/isaac/IsaacLab")),
                      "scripts/imitation_learning/locomanipulation_sdg/generate_data.py")
sys.argv = [script, "--output_file", "/tmp/unused.hdf5"] + rest

import nav_planner_shim as S  # noqa: E402  (hooks AppLauncher: planner import, camera drop)

head = open(script).read().split(S.MARK, 1)[0]
g = {"__name__": "nav_gen", "__file__": script}
exec(compile(head, script, "exec"), g)  # starts the app, defines setup_navigation_scene etc.

import random  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
import torch  # noqa: E402

import nav_bc_feats as F  # noqa: E402

args = g["args_cli"]
norm = json.load(open(ours.norm))
lo, hi = np.array(norm["min"]), np.array(norm["max"])
span = np.where(hi - lo > 1e-6, hi - lo, 1.0)


def np1(t):
    return t.detach().cpu().numpy().astype(np.float64).reshape(-1)


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
    rows, traces = [], []
    for t in range(ours.trials):
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ours.checkpoint, device=device)
        policy.start_episode()
        ep = data.load_episode(random.choice(names), args.device)
        env.reset_to(state=ep.get_initial_state(), env_ids=torch.tensor([0]), is_relative=True)
        _, path, goal, approach = g["setup_navigation_scene"](env, ep, 1.0, True)
        path_k = F.resample_path(path.points.detach().cpu().numpy() if torch.is_tensor(path.points) else path.points)
        goal_p, appr_p = np1(goal.get_pose()), np1(approach.get_pose())
        table = np1(env.get_end_fixture().get_pose())
        po = env.observation_manager.compute()["policy"]
        m = F.TableMap(np1(env.get_start_fixture().get_pose()), table)
        tr = []
        held, r = 0, dict(trial=t, success=False, steps=0, dropped=False, fell=False, min_goal_dist=9.0, ended_by=[])
        for i in range(ours.horizon):
            base = np1(env.get_base().get_pose())
            o = F.obs(base, goal_p, appr_p, path_k,
                      np.concatenate([np1(po["left_eef_pos"]), np1(po["left_eef_quat"])]),
                      np.concatenate([np1(po["right_eef_pos"]), np1(po["right_eef_quat"])]),
                      np.concatenate([np1(po["object_pos"]), np1(po["object_rot"])]), np1(po["hand_joint_state"]))
            a = (np.asarray(policy(o), dtype=np.float64) + 1) * span / 2 + lo
            a = F.actions_from_base(base, a)
            if ours.trace:  # object/wrists in the base frame, hand joints, the action sent
                tr.append(np.concatenate([o["object_b"][:3], o["left_eef_b"][:3], o["right_eef_b"][:3], o["hand_joint_state"],
                                          F.actions_to_base(base, a), base[:3]]))
            obs_dict, _, term, trunc, _ = env.step(torch.tensor(a, dtype=torch.float32).view(1, -1))
            po = obs_dict["policy"]
            obj = np.concatenate([np1(po["object_pos"]), np1(po["object_rot"])])
            r["steps"] = i + 1
            r["min_goal_dist"] = min(r["min_goal_dist"], float(np.linalg.norm(base[:2] - goal_p[:2])))
            ob, rb = env.scene["object"].data, env.scene["robot"].data
            if F.on_table(obj, table):
                r.setdefault("over_table_step", i)
            ok = F.placed(m, obj, np1(ob.root_vel_w)[:3], np1(rb.body_pos_w[:, rb.body_names.index("right_wrist_yaw_link")]))
            held = held + 1 if ok else 0
            if held >= ours.hold:
                r["success"] = True
                break
            if obj[2] < 0.3:
                r["dropped"] = True
                break
            if base[2] < 0.4:
                r["fell"] = True
                break
            if bool(term[0] or trunc[0]):  # which env termination term fired (they stay set after the auto-reset)
                tm = env.termination_manager
                r["ended_by"] = [n for n in tm.active_terms if bool(tm.get_term(n)[0])] or ["truncated"]
                r["dropped"] = "object_dropping" in r["ended_by"]  # object below 0.5 m; obs here are already post-reset
                r["end_goal_dist"] = round(float(np.linalg.norm(base[:2] - goal_p[:2])), 3)
                break
        rows.append(r)
        traces.append(np.array(tr, dtype=np.float32))
        print(f"[nav_bc_eval] {r}", flush=True)
    summary = dict(n=len(rows), success=sum(r["success"] for r in rows),
                   over_table=sum("over_table_step" in r for r in rows), dropped=sum(r["dropped"] for r in rows),
                   fell=sum(r["fell"] for r in rows), checkpoint=ours.checkpoint,
                   ended_by={n: sum(n in r["ended_by"] for r in rows) for n in {n for r in rows for n in r["ended_by"]}})
    print(f"[nav_bc_eval] summary {summary}", flush=True)
    json.dump(dict(summary=summary, rows=rows), open(ours.out, "w"), indent=1)
    if ours.trace:
        np.savez(ours.trace, **{f"t{i}": x for i, x in enumerate(traces)})
    env.close()


with torch.no_grad():
    main()
g["simulation_app"].close()
