"""Before swapping the steering wheel for a box (owner 2026-10-21: one box via Mimic demos from the wheel demos): measure
the wheel as the scene builds it (bounding box in world and in its own frame, root position, mass) and where the two
hands hold it in the generated demos (wrist positions in the wheel's frame at the step it first rises 1 cm). A box of the
wheel's bounding size, centred where the wheel's root is, puts its sides where the recorded hands already go.

  /isaac-sim/python.sh box_probe.py --dataset sdg/generated_dataset_g1_locomanip.hdf5 --headless --enable_pinocchio
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Locomanipulation-G1-Abs-v0")
parser.add_argument("--dataset", required=True)
parser.add_argument("--demos", type=int, default=100)
parser.add_argument("--enable_pinocchio", default=False, action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.enable_pinocchio:
    import pinocchio  # noqa: F401
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def qrot_inv(q, v):  # rotate v by the inverse of unit quaternion q (w, x, y, z)
    w, x, y, z = q
    u = np.array([-x, -y, -z])
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
stage = env.scene.stage
prim = stage.GetPrimAtPath("/World/envs/env_0/Object")
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
wb = cache.ComputeWorldBound(prim).ComputeAlignedRange()
lb = cache.ComputeLocalBound(prim).GetRange()
ob = env.scene["object"].data
origin = env.scene.env_origins[0].cpu().numpy()
print("BOX_PROBE object root (env frame):", (ob.root_pos_w[0].cpu().numpy() - origin).round(4).tolist(),
      "quat", ob.root_quat_w[0].cpu().numpy().round(4).tolist())
print("BOX_PROBE world bbox min", (np.array(wb.GetMin()) - origin).round(4).tolist(), "max", (np.array(wb.GetMax()) - origin).round(4).tolist(),
      "size", (np.array(wb.GetMax()) - np.array(wb.GetMin())).round(4).tolist())
print("BOX_PROBE local bbox (own frame, before scale on the prim)", np.array(lb.GetMin()).round(4).tolist(), np.array(lb.GetMax()).round(4).tolist())
print("BOX_PROBE mass kg", env.scene["object"].root_physx_view.get_masses().cpu().numpy().round(4).tolist())

L, R = [], []
with h5py.File(args.dataset, "r") as f:
    for name in list(f["data"])[: args.demos]:
        o = f["data"][name]["obs"]
        obj = o["object"][:]
        z = obj[:, 2]
        k = int(np.argmax(z - z[0] > 0.01))
        if k == 0:
            continue
        p, q = obj[k, :3], obj[k, 3:7]
        L.append(qrot_inv(q, o["left_eef_pos"][k] - p))
        R.append(qrot_inv(q, o["right_eef_pos"][k] - p))
L, R = np.array(L), np.array(R)
print(f"BOX_PROBE wrists in the wheel frame at first rise, {len(L)} demos: left mean {L.mean(0).round(3).tolist()} sd {L.std(0).round(3).tolist()}; "
      f"right mean {R.mean(0).round(3).tolist()} sd {R.std(0).round(3).tolist()}; gap {np.linalg.norm(L - R, axis=1).mean():.3f} m")
env.close()
simulation_app.close()
