"""Unitree's real G1 + Dex3 teleoperation recordings (Hugging Face `unitreerobotics/G1_Dex3_*_Dataset`, LeRobot v3,
30 frames/s) as a motion prior for the simulated G1 (so101/g1.py): which of their joints are ours, how the real right
hand closes (its main joint synergy, the 'eigengrasp' of Ciocarlie and Allen 2009 / postural synergies of Santello et
al. 1998) and how the hand meets the object (tilt at the lowest point).

Data: tables and metadata only (no videos), downloaded on the cluster login node into HF_HOME. Reading parquet needs
pyarrow, which .venv-vla lacks: a wheel unpacked at $SHARE/pylibs/pyarrow25 (put it on PYTHONPATH).

Their state/action: 28 joints, left arm 7, right arm 7, left hand 7, right hand 7, radians; the names map one-to-one to
the Menagerie model's (kRightHandThumb0 -> right_hand_thumb_0_joint). No waist joints (theirs stay still).

Commands (cluster, CPU): python -m so101.unitree_data check [DATASET ...]    names, ranges, replay of the right hand
                         python -m so101.unitree_data synergy                 right-hand synergy -> runs/unitree/
                         python -m so101.unitree_data replay PickBottle:0,3   kinematic replay -> runs/unitree/replay.mp4
"""
import glob
import os
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

from so101 import g1

HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
PICKS = ("PickApple", "PickBottle", "PickCharger", "PickGum", "PickSnack", "PickTissue", "PickDoll", "GraspSquare")
OUT = Path("runs/unitree")


def ours(name):
    """kRightHandThumb0 -> right_hand_thumb_0_joint, kRightShoulderPitch -> right_shoulder_pitch_joint."""
    side, hand, part, num = re.match(r"k(Left|Right)(Hand)?([A-Za-z]+?)(\d)?$", name).groups()
    words = "_".join(w.lower() for w in re.findall(r"[A-Z][a-z]+", part))
    return f"{side.lower()}_{'hand_' if hand else ''}{words}{'_' + num if num else ''}_joint"


def load(name):
    """(joint names (ours), state frames x 28, action frames x 28, episode index per frame, fps) of one dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = glob.glob(f"{HUB}/datasets--unitreerobotics--G1_Dex3_{name}_Dataset/snapshots/*")[0]
    info = json.load(open(f"{d}/meta/info.json"))
    t = pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{d}/data/*/*.parquet"))])
    vec = lambda c: t.column(c).combine_chunks().flatten().to_numpy().reshape(t.num_rows, -1)
    names = [ours(c) for c in info["features"]["observation.state"]["names"][0]]
    return names, vec("observation.state"), vec("action"), t.column("episode_index").to_numpy(), info["fps"]


def right_hand(names, S, ep, moving=0.15):
    """Right-hand joint frames (g1.HAND order) of the episodes where the right hand works (its joints span more than
    `moving` radians; in many recordings only the left hand picks), with their episode index."""
    cols = [names.index(j) for j in g1.HAND]
    H = S[:, cols]
    keep = np.zeros(len(S), bool)
    for e in np.unique(ep):
        r = ep == e
        keep[r] = np.ptp(H[r], 0).max() > moving
    return H[keep], ep[keep]


def cmd_check():
    """Joint ranges against ours (a sign flip shows as values outside them), and per episode the right hand's path:
    pinch-site height, hand tilt from straight down at the lowest point, thumb-to-index tip distance."""
    model = g1.build()
    data = mujoco.MjData(model)
    g1.start(model, data)
    pinch, hb = model.site("pinch").id, model.body(g1.HAND_BODY).id
    tips = [model.geom(f"tip_{b}").id for b in g1.TIPS]
    for name in sys.argv[2:] or PICKS:
        names, S, A, ep, fps = load(name)
        print(f"\n== {name}: {len(np.unique(ep))} episodes, {len(S) / len(np.unique(ep)) / fps:.0f} s each")
        for i, j in enumerate(names):
            lo, hi = model.jnt_range[model.joint(j).id]
            out = np.mean((S[:, i] < lo - 0.02) | (S[:, i] > hi + 0.02))
            if "right" in j:
                print(f"  {j:28s} {np.percentile(S[:, i], 1):6.2f} {np.percentile(S[:, i], 99):6.2f}  ours [{lo:5.2f},"
                      f" {hi:5.2f}]  outside {out:5.1%}  |action - state| p99 {np.percentile(np.abs(A[:, i] - S[:, i]), 99):.2f}")
        H, hep = right_hand(names, S, ep)
        print(f"  right hand works in {len(np.unique(hep))} of {len(np.unique(ep))} episodes")
        right = [i for i, j in enumerate(names) if j.startswith("right")]
        adr = [model.jnt_qposadr[model.joint(names[i]).id] for i in right]
        tilts, gaps = [], []
        for e in np.unique(hep):
            rows = np.flatnonzero(ep == e)[::3]
            h, tilt, gap = [], [], []
            for r in rows:
                data.qpos[adr] = S[r, right]
                mujoco.mj_kinematics(model, data)
                h.append(data.site_xpos[pinch][2])
                tilt.append(np.degrees(np.arccos(np.clip(-(data.xmat[hb].reshape(3, 3) @ g1.DOWN)[2], -1, 1))))
                gap.append(np.linalg.norm(data.geom_xpos[tips[0]] - data.geom_xpos[tips[1]]))
            k = int(np.argmin(h))
            tilts.append(tilt[k])
            gaps.append(min(gap[k:]))
        print(f"  hand tilt from down at the lowest point: median {np.median(tilts):.0f} deg, quartiles"
              f" {np.percentile(tilts, 25):.0f}-{np.percentile(tilts, 75):.0f}; thumb-index tips after it median"
              f" {np.median(gaps) * 100:.1f} cm")


def cmd_synergy():
    """How the real right hand closes on things: in every pick recording where it works, the hand pose 1 s before and
    1 s after the hand's lowest point (where it takes the object); the mean of those closing moves is the synergy
    (open pose -> closed pose), saved for the simulated hand (runs/unitree/synergy.json). Also the main directions
    (principal components) of the closing moves, to see whether one synergy covers them."""
    model = g1.build()
    data = mujoco.MjData(model)
    g1.start(model, data)
    pinch = model.site("pinch").id
    tips = [model.geom(f"tip_{b}").id for b in g1.TIPS]
    hand_adr = [model.jnt_qposadr[model.joint(j).id] for j in g1.HAND]
    opens, closeds = [], []
    for name in PICKS:
        names, S, _, ep, fps = load(name)
        right = [i for i, j in enumerate(names) if j.startswith("right") and "hand" not in j]
        arm_adr = [model.jnt_qposadr[model.joint(names[i]).id] for i in right]
        hand = [names.index(j) for j in g1.HAND]
        _, works = right_hand(names, S, ep)
        for e in np.unique(works):
            rows = np.flatnonzero(ep == e)
            h = []
            for r in rows[::3]:
                data.qpos[arm_adr] = S[r, right]
                mujoco.mj_kinematics(model, data)
                h.append(data.site_xpos[pinch][2])
            k = rows[::3][int(np.argmin(h))]
            sec = int(fps)
            if k - sec < rows[0] or k + sec > rows[-1]:
                continue
            opens.append(S[k - sec, hand])
            closeds.append(S[k + sec, hand])
    O, C = np.array(opens), np.array(closeds)
    D = C - O
    _, sv, Vt = np.linalg.svd(D - D.mean(0), full_matrices=False)

    def tip_gap(q):
        data.qpos[hand_adr] = q
        mujoco.mj_kinematics(model, data)
        return float(np.linalg.norm(data.geom_xpos[tips[0]] - data.geom_xpos[tips[1]]))
    o, c = O.mean(0), C.mean(0)
    print(f"{len(D)} grasps; open pose {dict(zip(g1.HAND, np.round(o, 2)))}")
    print(f"closing move (mean) {dict(zip(g1.HAND, np.round(c - o, 2)))}")
    print(f"closing moves' spread per joint {np.round(D.std(0), 2)}; variance share of their components"
          f" {np.round(sv[:4] ** 2 / np.sum(sv ** 2), 2)}")
    print(f"thumb-index tip gap: open pose {tip_gap(o) * 100:.1f} cm, closed pose {tip_gap(c) * 100:.1f} cm")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synergy.json").write_text(json.dumps(dict(joints=list(g1.HAND), open=o.round(4).tolist(),
                                                      closed=c.round(4).tolist(), grasps=len(D))))


def cmd_replay():
    """Kinematic replay of recorded episodes on the fixed-base G1 (both arms and hands set from the recording, frame by
    frame, no physics) into an mp4: the whole robot and a close-up that follows the right hand, 15 frames/s.
    python -m so101.unitree_data replay PickBottle:0,3 GraspSquare:0 (GPU job: headless rendering)"""
    import av
    model = g1.build()
    data = mujoco.MjData(model)
    g1.start(model, data)
    r = mujoco.Renderer(model, 360, 480)
    pinch = model.site("pinch").id
    OUT.mkdir(parents=True, exist_ok=True)
    out = av.open(str(OUT / "replay.mp4"), "w")
    stream = None
    for spec in sys.argv[2:]:
        name, eps = spec.split(":")
        names, S, _, ep, _ = load(name)
        adr = [model.jnt_qposadr[model.joint(j).id] for j in names]
        for e in map(int, eps.split(",")):
            for f in np.flatnonzero(ep == e)[::2]:
                data.qpos[adr] = S[f]
                mujoco.mj_kinematics(model, data)
                frames = []
                for look, dist, az in (((0.05, 0, 0.15), 1.4, 160), (data.site_xpos[pinch], 0.35, 150)):
                    v = mujoco.MjvCamera()
                    v.lookat[:], v.distance, v.azimuth, v.elevation = look, dist, az, -20
                    r.update_scene(data, v)
                    frames.append(r.render())
                frame = np.concatenate(frames, 1)
                if stream is None:
                    stream = out.add_stream("libx264", rate=15)
                    stream.width, stream.height, stream.pix_fmt = frame.shape[1], frame.shape[0], "yuv420p"
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                    out.mux(packet)
            print(name, e, flush=True)
    for packet in stream.encode():
        out.mux(packet)
    out.close()


if __name__ == "__main__":
    {"check": cmd_check, "synergy": cmd_synergy, "replay": cmd_replay}[sys.argv[1]]()
