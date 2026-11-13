"""Training set for a place BC: the manipulation BC's own observations and actions, with the walking-between-tables
drop-offs added in the frame the BC was trained in.

Sources:
  stock demos (generated_dataset_g1_locomanip.hdf5): --stock full keeps whole demos (pick and place in one net),
      --stock tail keeps each demo from the step the object is --lift m above its start (where nav_modular_eval.py
      hands over to the walk), so episodes start where our place phase starts;
  nav demos (nav300_*.hdf5, object on the end table at the end): the drop-off, from the first step past the
      generator's --navigate_step (recording_step > 130), observations and wrist targets moved from the end table's
      frame to the start table's (nav_bc_feats.TableMap, as nav_modular_eval.py does at run time).

  python nav_to_place.py out.hdf5 --stock_file stock.hdf5 --stock tail nav1.hdf5 [nav2.hdf5 ...]
"""
import argparse
import sys

import h5py
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import nav_bc_feats as F  # noqa: E402

KEYS = ["left_eef_pos", "left_eef_quat", "right_eef_pos", "right_eef_quat", "hand_joint_state", "object"]
p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("nav", nargs="*")
p.add_argument("--stock_file", required=True)
p.add_argument("--stock", choices=["full", "tail", "none"], default="tail")
p.add_argument("--lift", type=float, default=0.06)
p.add_argument("--navigate_step", type=int, default=130)
a = p.parse_intermixed_args()  # out, options, then the nav files

demos = []  # (obs dict, actions)
with h5py.File(a.stock_file, "r") as f:
    env_args = f["data"].attrs["env_args"]
    if a.stock != "none":
        for name in f["data"]:
            e = f["data"][name]
            o = {k: e[f"obs/{k}"][:] for k in KEYS}
            z = o["object"][:, 2]
            s = 0 if a.stock == "full" else int(np.argmax(z - z[0] > a.lift))
            if a.stock == "tail" and z.max() - z[0] <= a.lift:
                continue
            demos.append(({k: v[s:] for k, v in o.items()}, e["actions"][s:]))
n_stock = len(demos)

seen = kept = 0
for path in a.nav:
    try:
        f = h5py.File(path, "r")
    except OSError as err:
        print(f"skip {path}: {err}")
        continue
    with f:
        for name in f["data"]:
            e = f["data"][name]
            seen += 1
            if "locomanipulation_sdg_output_data" not in e:
                continue
            sd = e["locomanipulation_sdg_output_data"]
            obj = np.concatenate([e["obs/object_pos"][:], e["obs/object_rot"][:]], -1)
            if not F.on_table(obj[-1], sd["end_fixture_pose"][-1]):
                continue
            s = int(np.argmax(sd["recording_step"][:, 0] > a.navigate_step))
            if s == 0:
                continue
            m = F.TableMap(sd["start_fixture_pose"][0].astype(np.float64), sd["end_fixture_pose"][0].astype(np.float64))
            o = {k: e[f"obs/{k}"][s:].astype(np.float64) for k in KEYS}
            for side in ("left", "right"):
                x = m.pose(np.concatenate([o[f"{side}_eef_pos"], o[f"{side}_eef_quat"]], -1))
                o[f"{side}_eef_pos"], o[f"{side}_eef_quat"] = x[:, :3], x[:, 3:]
            ob = o["object"]
            o["object"] = np.concatenate([m.pose(ob[:, :7]), m.vec(ob[:, 7:10]), m.vec(ob[:, 10:13])], -1)
            act = e["actions"][s:].astype(np.float64)
            for q in (slice(0, 7), slice(7, 14)):
                act[:, q] = m.pose(act[:, q])
            demos.append(({k: v.astype(np.float32) for k, v in o.items()}, act.astype(np.float32)))
            kept += 1
print(f"stock {a.stock}: {n_stock} demos; nav drop-offs kept {kept} of {seen}")

with h5py.File(a.out, "w") as g:
    d = g.create_group("data")
    total = 0
    for i, (o, act) in enumerate(demos):
        e = d.create_group(f"demo_{i}")
        e.create_dataset("actions", data=act)
        for k, v in o.items():
            e.create_dataset(f"obs/{k}", data=v)
        e.attrs["num_samples"], e.attrs["success"] = len(act), True
        total += len(act)
    d.attrs["env_args"], d.attrs["total"] = env_args, total
print(f"wrote {a.out}: {len(demos)} demos, {total} steps, mean length {total / len(demos):.0f}")
