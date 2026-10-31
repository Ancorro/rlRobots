"""Why does the fixed-base grasp policy miss on the walking base? Compare two probes of the same policy on the same eval
layouts (g1_dexsuite.py eval --probe=FILE, one with --base=fixed and one with --base=walking): what the policy sees, what
it commands, what the arm does, where the hand goes. Numpy only: runs on a laptop.

  python isaac/obs_compare.py fixed.npz walking.npz [--steps=50] [--top=8]

Reads, per layout (matched by seed and shape), the first --steps policy steps:
  1. observation vector, term by term (the policy's input; rsl_rl normalizes it, so differences are in raw units):
     mean |difference| at step 0 and over the steps, against the term's own spread across layouts on the fixed base
     (a difference larger than that spread is something the policy never saw)
  2. the object in the root (pelvis) frame and the command: the frame the policy aims in
  3. the root's roll and pitch on the walking base (the fixed base is level)
  4. actions: how far the two runs' commands diverge, step by step (the same input must give the same output)
  5. joint tracking: target minus achieved, per joint, per base (drives and gains)
  6. the hand's approach: pinch-to-object distance over time, per base, and the outcome counts
and ends with a verdict: which of observation, frame, tracking or policy explains the gap first in time.
"""
import json
import sys

import numpy as np


def load(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["terms"] = json.loads(str(d["terms"]))
    d["groups"] = json.loads(str(d["groups"]))
    return d


def align(a, b):
    """Indices into a and b of the layouts both have, in a's order."""
    kb = {(str(s), int(k)): i for i, (s, k) in enumerate(zip(b["shapes"], b["seeds"]))}
    ia, ib = [], []
    for i, (s, k) in enumerate(zip(a["shapes"], a["seeds"])):
        j = kb.get((str(s), int(k)))
        if j is not None:
            ia.append(i)
            ib.append(j)
    return np.array(ia), np.array(ib)


def slices(terms):
    out, k = [], 0
    for g, n, d in terms:
        out.append((f"{g}/{n}", slice(k, k + d)))
        k += d
    return out


def roll_pitch(q):
    """Degrees from (w, x, y, z) quaternions, (..., 4)."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    return roll, pitch


def compare(a, b, steps=50, top=8, out=print):
    ia, ib = align(a, b)
    if not len(ia):
        out("no common layouts (seeds and shapes differ)")
        return {}
    T = min(steps, a["obs"].shape[0], b["obs"].shape[0])
    out(f"{len(ia)} common layouts; {T} steps ({T * float(a['dt']):.2f} s) compared; bases {a['base']} vs {b['base']}; "
        f"checkpoints {'same' if str(a['checkpoint']) == str(b['checkpoint']) else 'DIFFERENT'}")
    out(f"outcome: {a['base']} lifted {int(a['success'][ia].sum())}/{len(ia)}, {b['base']} {int(b['success'][ib].sum())}/{len(ib)}; "
        f"{b['base']} why: {dict(zip(*np.unique(b['why'][ib], return_counts=True)))}")
    verdict = {}
    # 1. observations term by term
    A, B = a["obs"][:T, ia], b["obs"][:T, ib]
    out("\n1. observation terms: mean |diff| at step 0 / over steps, vs the term's spread across layouts (fixed base, step 0)")
    rows = []
    for name, sl in slices(a["terms"]):
        d0 = np.abs(A[0, :, sl] - B[0, :, sl]).mean()
        dt = np.abs(A[:, :, sl] - B[:, :, sl]).mean()
        spread = max(A[0, :, sl].std(axis=0).mean(), 0.01)  # floor: terms that are the same in every layout at reset (joints at
        rows.append((name, sl.stop - sl.start, d0, dt, spread, d0 / spread))  # the start pose) would give any difference a huge ratio
    rows.sort(key=lambda r: -r[5])
    for name, dim, d0, dt, spread, ratio in rows[:top]:
        out(f"   {name:40s} dim {dim:4d}  step0 {d0:8.4f}  steps {dt:8.4f}  spread {spread:8.4f}  ratio {ratio:6.2f}")
    worst = rows[0]
    verdict["obs_worst"] = (worst[0], float(worst[5]))
    # 2. the frame: object in the root frame, command
    oa, ob = a["obj_b"][0, ia], b["obj_b"][0, ib]
    ca, cb = a["cmd"][0, ia], b["cmd"][0, ib]
    out(f"\n2. object in the root frame at step 0: mean diff {np.abs(oa - ob).mean(0).round(4).tolist()} (x fwd, y left, z up), "
        f"{a['base']} mean {oa.mean(0).round(3).tolist()}, {b['base']} mean {ob.mean(0).round(3).tolist()}")
    out(f"   command (target in the root frame): mean diff {np.abs(ca - cb).mean(0).round(4).tolist()}")
    verdict["frame_diff_m"] = float(np.abs(oa - ob).mean())
    # 3. root attitude
    ra, pa = roll_pitch(a["root"][:T, ia, 3:7])
    rb, pb = roll_pitch(b["root"][:T, ib, 3:7])
    out(f"\n3. root roll / pitch (deg): {a['base']} {np.abs(ra).mean():.2f} / {np.abs(pa).mean():.2f}; "
        f"{b['base']} {np.abs(rb).mean():.2f} / {np.abs(pb).mean():.2f}; {b['base']} root height above its env origin "
        f"{b['root'][0, ib, 2].mean():.3f} (nominal root {a['root_nominal'].round(3).tolist()}), sway over the steps "
        f"{np.linalg.norm(b['root'][:T, ib, :2] - b['root'][0, ib, :2], axis=-1).max():.3f} m")
    verdict["pitch_deg"] = float(np.abs(pb).mean())
    # the base during the first steps: does the policy's own motion knock it over? (dex-v6b probe 2026-10-27: pitch 0 -> 22 deg
    # and the root 5 cm lower within 0.1 s of the first action, the hand never left the start height in the root frame)
    k5 = min(5, T - 1)
    dp, dz = float(np.abs(pb[k5] - pb[0]).mean()), float((b["root"][k5, ib, 2] - b["root"][0, ib, 2]).mean())
    hz_a, hz_b = a["hand"][:T, ia, 2].mean(1), b["hand"][:T, ib, 2].mean(1)
    out(f"   base in the first {k5} steps ({k5 * float(a['dt']):.2f} s): {b['base']} pitch change {dp:.1f} deg, height change {dz * 100:+.1f} cm; "
        f"pitch by step {np.round(pb.mean(1)[[0, 1, 2, k5, min(10, T - 1), min(20, T - 1), T - 1]], 1).tolist()}")
    out(f"   hand height in the root frame by step (m): {a['base']} {np.round(hz_a[[0, k5, min(10, T - 1), min(20, T - 1), T - 1]], 3).tolist()}, "
        f"{b['base']} {np.round(hz_b[[0, k5, min(10, T - 1), min(20, T - 1), T - 1]], 3).tolist()}")
    verdict["base_kick_deg"] = dp
    # 4. actions
    da = np.abs(a["act"][1:T, ia] - b["act"][1:T, ib]).mean(axis=(1, 2))
    aa = np.abs(a["act"][1:T, ia]).mean()
    out(f"\n4. actions: mean |diff| step 1 {da[0]:.3f}, step 5 {da[min(4, len(da) - 1)]:.3f}, mean over steps {da.mean():.3f} "
        f"(typical |action| on {a['base']}: {aa:.3f}); the two runs give the same commands to the same input only if this starts near 0")
    verdict["act_diff_step1"] = float(da[0]) / max(aa, 1e-6)
    # 5. joint tracking
    ea = np.abs(a["q_tgt"][:T, ia] - a["q"][:T, ia]).mean(axis=(0, 1))
    eb = np.abs(b["q_tgt"][:T, ib] - b["q"][:T, ib]).mean(axis=(0, 1))
    joints = [str(j).replace("_joint", "") for j in a["joints"]]
    out(f"\n5. joint tracking |target - position| (rad), mean over steps and layouts, {a['base']} / {b['base']}:")
    for j, x, y in sorted(zip(joints, ea, eb), key=lambda r: -(r[2] - r[1]))[:top]:
        out(f"   {j:28s} {x:.3f} / {y:.3f}")
    qa0, qb0 = a["q"][0, ia], b["q"][0, ib]
    out(f"   start joint positions: mean |diff| {np.abs(qa0 - qb0).mean():.3f} rad, worst "
        f"{joints[int(np.abs(qa0 - qb0).mean(0).argmax())]} {np.abs(qa0 - qb0).mean(0).max():.3f}")
    verdict["track_ratio"] = float(eb.mean() / max(ea.mean(), 1e-6))
    # 6. the approach
    na, nb = a["near"][:T, ia], b["near"][:T, ib]
    marks = [0, min(9, T - 1), min(24, T - 1), min(49, T - 1), T - 1]
    out(f"\n6. pinch-to-object distance (m), median over layouts at steps {marks}: "
        f"{a['base']} {[round(float(np.median(na[k])), 3) for k in marks]}, {b['base']} {[round(float(np.median(nb[k])), 3) for k in marks]}")
    ha, hb = a["hand"][0, ia], b["hand"][0, ib]
    out(f"   hand (pinch) in the root frame at step 0: {a['base']} {ha.mean(0).round(3).tolist()}, {b['base']} {hb.mean(0).round(3).tolist()}")
    verdict["near_end"] = (float(np.median(na[-1])), float(np.median(nb[-1])))
    # the verdict
    out("\nverdict:")
    if worst[5] <= 3 and verdict["base_kick_deg"] > 5:
        out(f"   BASE DYNAMICS: the observations match at the start, but the base pitches {verdict['base_kick_deg']:.0f} deg within "
            f"{k5 * float(a['dt']):.2f} s of the first actions (|action| {aa:.2f}: motions learned on a welded base). The policy's own arm "
            f"and waist motion knocks the standing base, the root frame tilts, and the hand never arrives. Try: no waist actions on the "
            f"walking base, a stronger action filter (--ema), then fine-tune with the base free and a smoothness term.")
    elif worst[5] > 3:
        out(f"   OBSERVATION: {worst[0]} differs by {worst[5]:.1f}x its spread across layouts at step 0; the policy is off-distribution "
            f"before it acts. Fix the frame or the term on the walking base before any fine-tune.")
    elif verdict["act_diff_step1"] > 0.5:
        out("   POLICY OUTPUT: inputs nearly match but the first actions differ; check observation normalization / history padding.")
    elif verdict["track_ratio"] > 2:
        out(f"   TRACKING: the arm follows its targets {verdict['track_ratio']:.1f}x worse on the walking base (drives, gains, AGILE fighting the waist).")
    else:
        out("   inputs, outputs and tracking agree at the start: the gap opens later (see 6.); look at sway and the contact phase.")
    return verdict


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    kw = dict(x[2:].split("=", 1) for x in sys.argv[1:] if x.startswith("--") and "=" in x)
    if len(args) != 2:
        raise SystemExit(__doc__)
    compare(load(args[0]), load(args[1]), steps=int(kw.get("steps", 50)), top=int(kw.get("top", 8)))
