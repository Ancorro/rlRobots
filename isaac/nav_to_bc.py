"""Locomanipulation SDG output files -> one robomimic BC dataset with robot-frame features (nav_bc_feats.py).

Keeps the demos whose last frame has the object on the end table's footprint (generate_data.py marks every demo
success=False). Actions are normalized per dimension to [-1, 1] over their 0.5-99.5 percentiles and clipped (the generator's turn
command spikes to +-2 pi); the bounds go to <out>.norm.json.

  python nav_to_bc.py out.hdf5 in1.hdf5 [in2.hdf5 ...]
"""
import json
import sys

import h5py
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import nav_bc_feats as F  # noqa: E402

out, ins = sys.argv[1], sys.argv[2:]
demos, kept, seen, partial = [], 0, 0, 0
for path in ins:
    try:
        f = h5py.File(path, "r")
    except OSError as err:  # a generation job cut off by its time limit mid-write
        print(f"skip {path}: {err}")
        continue
    with f:
        env_args = f["data"].attrs["env_args"]
        for name in f["data"]:
            e = f["data"][name]
            seen += 1
            if "locomanipulation_sdg_output_data" not in e or "actions" not in e:  # stuck or cut-off runs
                partial += 1
                continue
            sd = e["locomanipulation_sdg_output_data"]
            obj = np.concatenate([e["obs/object_pos"][:], e["obs/object_rot"][:]], -1)
            if not F.on_table(obj[-1], sd["end_fixture_pose"][-1]):
                continue
            base = sd["base_pose"][:].astype(np.float64)
            path_k = F.resample_path(sd["base_path"][0])
            o = F.obs(base, sd["base_goal_pose"][:], sd["base_goal_approach_pose"][:],
                      np.broadcast_to(path_k, (len(base),) + path_k.shape),
                      np.concatenate([e["obs/left_eef_pos"][:], e["obs/left_eef_quat"][:]], -1),
                      np.concatenate([e["obs/right_eef_pos"][:], e["obs/right_eef_quat"][:]], -1),
                      obj, e["obs/hand_joint_state"][:])
            demos.append((o, F.actions_to_base(base, e["actions"][:])))
            kept += 1
print(f"kept {kept} of {seen} demos (object on the end table; {partial} without nav data)")

A = np.concatenate([a for _, a in demos])
lo, hi = np.percentile(A, 0.5, 0), np.percentile(A, 99.5, 0)  # the turn command spikes to +-2 pi
span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
json.dump(dict(min=lo.tolist(), max=hi.tolist(), obs_keys=F.OBS_KEYS, path_k=F.PATH_K, sources=ins),
          open(out.replace(".hdf5", ".norm.json"), "w"), indent=1)
with h5py.File(out, "w") as g:
    d = g.create_group("data")
    total = 0
    for i, (o, a) in enumerate(demos):
        e = d.create_group(f"demo_{i}")
        e.create_dataset("actions", data=np.clip(2 * (a - lo) / span - 1, -1, 1).astype(np.float32))
        for k, v in o.items():
            e.create_dataset(f"obs/{k}", data=v)
        e.attrs["num_samples"], e.attrs["success"] = len(a), True
        total += len(a)
    d.attrs["env_args"], d.attrs["total"] = env_args, total
print(f"wrote {out}: {len(demos)} demos, {total} steps, action dims {A.shape[1]}")
