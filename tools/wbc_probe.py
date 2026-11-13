"""Does NVIDIA's decoupled whole-body controller keep the G1 standing through dex-v6b's arm motions? (path 2, 2026-11-11)

Background (docs/results.md 6.2): the fixed-base grasp policy dex-v6b lifts 65/72 on a welded base and 0/72 on the
walking base, because its arm and waist motions pitch AGILE's standing base 22 degrees. This probe replays those recorded
joint targets, open loop, on the G1 in MuJoCo with the leg policy from NVlabs/GR00T-WholeBodyControl (`decoupled_wbc`,
GR00T-WholeBodyControl-Balance.onnx: 12 leg + 3 waist joints at 50 Hz, commands velocity, height, torso roll/pitch/yaw)
and measures how far the pelvis moves. No table and no object: only the base disturbance is measured.

Sources (isaac/obs_compare.py's recordings, 72 validation layouts, 150 steps of 20 ms, 16 joints: waist yaw, waist pitch,
7 right-arm joints, 7 fingers; this MuJoCo model has no finger joints, so those are left out):
  fixed    the targets dex-v6b sent on the welded base (the motion that grasps: what a base has to tolerate)
  walking  the targets it sent on AGILE's base in Isaac Lab (the same commands that kicked AGILE; AGILE's own numbers
           are read from the recording and printed beside)
Waist: the leg policy owns the three waist joints. replay = waist yaw and pitch targets overridden by the recording;
wbc = left to the leg policy (arm joints only replayed); cmd = the recorded waist angles sent as the policy's torso
pitch and yaw commands. Speed 0.5 = the same motion at half speed.

  .venv/Scripts/python tools/wbc_probe.py --wbc ../others/GR00T-WholeBodyControl [--layouts 72]
"""
import argparse
import collections
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

p = argparse.ArgumentParser()
p.add_argument("--wbc", default="../others/GR00T-WholeBodyControl", help="checkout of NVlabs/GR00T-WholeBodyControl")
p.add_argument("--probe", default="runs/wbc_probe", help="folder with probe-fixed-small.npz and probe-walking-small.npz")
p.add_argument("--out", default="runs/wbc_probe/result.json")
p.add_argument("--layouts", type=int, default=72)
p.add_argument("--arm_gains", choices=["ours", "wbc"], default="ours",
               help="ours: g1_dexsuite.py's walking robot (shoulder 300/3, elbow and wrist 400/2.5, Menagerie torque limits); "
                    "wbc: the reference script's 100/0.5 on every arm joint")
p.add_argument("--height", type=float, default=0.74, help="the leg policy's height command (its config default)")
args = p.parse_args()

G = Path(args.wbc) / "decoupled_wbc/sim2mujoco/resources/robots/g1"
CFG = yaml.safe_load((G / "g1_gear_wbc.yaml").read_text())
KP, KD, DEF = (np.array(CFG[k], float) for k in ("kps", "kds", "default_angles"))
DT, DECIM, HIST = CFG["simulation_dt"], CFG["control_decimation"], CFG["obs_history_len"]
assert abs(DT * DECIM - 0.02) < 1e-9, "the recordings are 20 ms steps"
balance = ort.InferenceSession(str(G / "policy/GR00T-WholeBodyControl-Balance.onnx"), providers=["CPUExecutionProvider"])

m = mujoco.MjModel.from_xml_path(str(G / "g1_gear_wbc.xml"))
m.opt.timestep = DT
d = mujoco.MjData(m)
JOINTS = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]  # 29, in qpos order after the base
NJ = len(JOINTS)
assert NJ == 29 and m.nu == 29, (NJ, m.nu)
J = {n: i for i, n in enumerate(JOINTS)}
PELVIS, WRIST = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("pelvis", "right_wrist_yaw_link"))
ARM = JOINTS[15:]  # 7 left, 7 right
if args.arm_gains == "ours":
    AKP = np.array([300.0 if "shoulder" in n else 400.0 for n in ARM])
    AKD = np.array([3.0 if "shoulder" in n else 2.5 for n in ARM])
else:
    AKP, AKD = np.full(14, 100.0), np.full(14, 0.5)
ALIM = np.array([5.0 if ("wrist_pitch" in n or "wrist_yaw" in n) else 25.0 for n in ARM])  # Menagerie G1 torque limits


def tilt_deg(q):  # body z axis from vertical
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1))))


def rot(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


class Robot:
    def __init__(self):
        self.action = np.zeros(15, np.float32)
        self.hist = collections.deque([np.zeros(86, np.float32)] * HIST, maxlen=HIST)

    def obs(self, rpy):
        o = np.zeros(86, np.float32)  # the reference script's layout (run_mujoco_gear_wbc.py compute_observation)
        o[3], o[4:7] = args.height, rpy  # velocity command 0 (o[0:3])
        o[7:10] = d.qvel[3:6] * CFG["ang_vel_scale"]
        w, x, y, z = d.qpos[3:7]
        o[10:13] = [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))]  # gravity in the base frame
        pad = np.zeros(NJ)
        pad[:15] = DEF
        o[13:13 + NJ] = (d.qpos[7:] - pad) * CFG["dof_pos_scale"]
        o[13 + NJ:13 + 2 * NJ] = d.qvel[6:] * CFG["dof_vel_scale"]
        o[13 + 2 * NJ:] = self.action
        return o

    def step(self, arm_tgt, waist=None, rpy=(0.0, 0.0, 0.0)):
        """One 20 ms control step: leg policy, then DECIM physics steps of PD torques. waist: (yaw, pitch) targets that
        replace the policy's own, or None."""
        self.hist.append(self.obs(rpy))
        self.action = balance.run(None, {"input": np.concatenate(self.hist)[None]})[0][0]
        tgt = self.action * CFG["action_scale"] + DEF
        if waist is not None:
            tgt[J["waist_yaw_joint"]], tgt[J["waist_pitch_joint"]] = waist
        for _ in range(DECIM):
            q, dq = d.qpos[7:], d.qvel[6:]
            d.ctrl[:15] = KP * (tgt - q[:15]) - KD * dq[:15]
            d.ctrl[15:] = np.clip(AKP * (arm_tgt - q[15:]) - AKD * dq[15:], -ALIM, ALIM)
            mujoco.mj_step(m, d)

    def save(self):
        return d.qpos.copy(), d.qvel.copy(), self.action.copy(), [h.copy() for h in self.hist], d.time

    def load(self, s):
        mujoco.mj_resetData(m, d)
        d.qpos[:], d.qvel[:], self.action, d.time = s[0], s[1], s[2].copy(), s[4]
        self.hist = collections.deque([h.copy() for h in s[3]], maxlen=HIST)
        mujoco.mj_forward(m, d)


REC = ["waist_yaw_joint", "waist_pitch_joint"] + [f"right_{n}_joint" for n in
                                                  ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
MIRROR = {"shoulder_roll": -1, "shoulder_yaw": -1, "wrist_roll": -1, "wrist_yaw": -1}  # left arm = mirrored right arm


def arm_targets(right7):
    a = np.zeros(14)
    a[7:] = right7
    for k, n in enumerate(ARM[:7]):
        key = n.replace("left_", "").replace("_joint", "")
        a[k] = START[2 + k] * MIRROR.get(key, 1)  # the left arm holds the mirrored start pose
    return a


def replay(rob, src, layout, waist_mode, speed, still=False):
    q_tgt = src["q_tgt"][1:, layout, :9].astype(float)  # step 0 of the recording holds no target yet
    K = int((len(q_tgt) - 1) / speed)
    P0, R0 = d.xpos[PELVIS].copy(), rot(d.xquat[PELVIS])
    tilt, drift, height, werr, track, wtrack = [], [], [], [], [], []
    fell = False
    for k in range(K):
        t = k * speed
        i = int(t)
        tg = START[:9].copy() if still else q_tgt[i] + (t - i) * (q_tgt[min(i + 1, len(q_tgt) - 1)] - q_tgt[i])
        waist, rpy = None, (0.0, 0.0, 0.0)
        if waist_mode == "replay":
            waist = (tg[0], tg[1])
        elif waist_mode == "cmd":
            rpy = (0.0, tg[1], tg[0])
        rob.step(arm_targets(tg[2:9]), waist, rpy)
        P, R = d.xpos[PELVIS], rot(d.xquat[PELVIS])
        rel = R.T @ (d.xpos[WRIST] - P)  # where the wrist would be had the pelvis stayed where it stood
        werr.append(float(np.linalg.norm(d.xpos[WRIST] - (P0 + R0 @ rel))))
        tilt.append(tilt_deg(d.qpos[3:7]))
        drift.append(float(np.linalg.norm(P[:2] - P0[:2])))
        height.append(float(P[2]))
        track.append(float(np.abs(tg[2:9] - d.qpos[7 + 22:7 + 29]).mean()))
        wtrack.append(float(np.abs(tg[:2] - d.qpos[[7 + J["waist_yaw_joint"], 7 + J["waist_pitch_joint"]]]).mean()))
        fell = fell or P[2] < 0.5 or tilt[-1] > 45
    n05, n1 = int(0.5 / 0.02 / speed), int(1.0 / 0.02 / speed)  # the recording's first 0.5 s and 1 s
    return dict(tilt_max=max(tilt), tilt_05=max(tilt[:n05]), drift_1s=max(drift[:n1]), drift_max=max(drift), height_min=min(height),
                wrist_max=max(werr), wrist_grasp=float(np.mean(werr[n05:n1])), arm_track=float(np.mean(track)),
                waist_track=float(np.mean(wtrack)), fell=bool(fell))


src = {b: np.load(Path(args.probe) / f"probe-{b}-small.npz", allow_pickle=True) for b in ("fixed", "walking")}
assert list(src["fixed"]["joints"][:9]) == REC, "the recording's first 9 joints are waist yaw, waist pitch and the right arm"
START = np.median(src["fixed"]["q"][0], 0).astype(float)  # the arm pose every layout starts from

rob = Robot()
mujoco.mj_resetData(m, d)
d.qpos[7:7 + 15] = DEF
mujoco.mj_forward(m, d)
for k in range(150):  # 3 s: arms from zero to the start pose over 1 s, then stand
    s = min(1.0, (k + 1) / 50)
    rob.step(arm_targets(s * START[2:9]), (s * START[0], s * START[1]))
settled = rob.save()
print(f"settled: pelvis height {d.qpos[2]:.3f} m, tilt {tilt_deg(d.qpos[3:7]):.1f} deg, mass {m.body_subtreemass[1]:.1f} kg", flush=True)

N = min(args.layouts, src["fixed"]["q_tgt"].shape[1])
CONDS = [("still", "fixed", "replay", 1.0, True)] + [
    (f"{s}/{w}/x{v:g}", s, w, v, False) for s in ("fixed", "walking") for w in ("replay", "wbc", "cmd") for v in (1.0, 0.5)]
KEYS = ["tilt_max", "tilt_05", "drift_1s", "drift_max", "height_min", "wrist_max", "wrist_grasp", "arm_track", "waist_track"]
out, t0 = {}, time.time()
for name, s, w, v, still in CONDS:
    rows = []
    for i in range(1 if still else N):
        rob.load(settled)
        rows.append(replay(rob, src[s], i, w, v, still))
    agg = {k: dict(median=float(np.median([r[k] for r in rows])), p90=float(np.percentile([r[k] for r in rows], 90))) for k in KEYS}
    out[name] = dict(source=s, waist=w, speed=v, n=len(rows), fell=int(sum(r["fell"] for r in rows)), **agg, rows=rows)
    a = out[name]
    print(f"{name:20s} n {a['n']:3d} fell {a['fell']:2d} | tilt deg max {a['tilt_max']['median']:5.1f} (p90 {a['tilt_max']['p90']:5.1f}), first 0.5 s "
          f"{a['tilt_05']['median']:5.1f} | drift 1 s {a['drift_1s']['median'] * 100:5.1f} cm, max {a['drift_max']['median'] * 100:5.1f} | "
          f"wrist off by base motion: grasp window {a['wrist_grasp']['median'] * 100:5.1f} cm (p90 {a['wrist_grasp']['p90'] * 100:5.1f}), max "
          f"{a['wrist_max']['median'] * 100:5.1f} | arm track {a['arm_track']['median']:.3f} rad, waist {a['waist_track']['median']:.3f}", flush=True)

# AGILE's base under the same commands, from the Isaac Lab recording (closed loop there: the policy saw the base move)
root = src["walking"]["root"][:, :N]
tl = np.array([[tilt_deg(root[t, i, 3:7]) for i in range(N)] for t in range(root.shape[0])])
dr = np.linalg.norm(root[:, :, :2] - root[0:1, :, :2], axis=2)
agile = dict(tilt_max=float(np.median(tl.max(0))), tilt_05=float(np.median(tl[:25].max(0))), drift_1s=float(np.median(dr[:50].max(0))),
             drift_max=float(np.median(dr.max(0))), n=N)
print(f"AGILE in Isaac Lab (recording): tilt deg max {agile['tilt_max']:.1f}, first 0.5 s {agile['tilt_05']:.1f}; "
      f"drift 1 s {agile['drift_1s'] * 100:.1f} cm, max {agile['drift_max'] * 100:.1f}")
settings = dict(vars(args), sim_dt=DT, decimation=DECIM, leg_kp=KP.tolist(), leg_kd=KD.tolist(), arm_kp=AKP.tolist(), arm_kd=AKD.tolist(),
                arm_torque_limit=ALIM.tolist(), policy="GR00T-WholeBodyControl-Balance.onnx", velocity_cmd=[0, 0, 0],
                mujoco=mujoco.__version__, seconds=round(time.time() - t0, 1))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(dict(settings=settings, agile_isaac=agile, conditions=out), indent=1))
print(f"wrote {args.out} ({settings['seconds']} s)")
