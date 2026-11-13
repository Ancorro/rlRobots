"""Why the box gets no Mimic demos (owner 2026-10-21: one box via Mimic demos from the wheel demos): replay recorded wheel
demos open loop, starting from each demo's own first state, with the wheel (control) or the box (isaac/box_cfgs.py), and
print where the object goes. If the wheel replays succeed and the box ones don't, the line shows which part breaks:
the box never rises (grasp), rises then falls (hold), or ends in place but fails the success check.

  /isaac-sim/python.sh box_replay.py --object box --dataset sdg/generated_dataset_g1_locomanip.hdf5 --headless --enable_pinocchio
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--object", choices=["box", "wheel"], default="box")
parser.add_argument("--dataset", required=True)
parser.add_argument("--demos", type=int, default=5)
parser.add_argument("--enable_pinocchio", default=False, action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.enable_pinocchio:
    import pinocchio  # noqa: F401
simulation_app = AppLauncher(args).app

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401
from box_cfgs import BoxEnvCfg  # noqa: E402
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg  # noqa: E402

cfg = BoxEnvCfg() if args.object == "box" else LocomanipulationG1EnvCfg()
cfg.scene.num_envs = 1
cfg.sim.device = "cpu"
env = ManagerBasedRLEnv(cfg=cfg)
env.reset()
ids = torch.tensor([0])
obj = env.scene["object"]
origin = env.scene.env_origins[0]


def tensors(g):  # h5py group -> nested dict of 1-row tensors
    return {k: tensors(v) if isinstance(v, h5py.Group) else torch.tensor(np.array(v)) for k, v in g.items()}


with h5py.File(args.dataset, "r") as f:
    for name in list(f["data"])[: args.demos]:
        d = f["data"][name]
        env.reset_to(tensors(d["initial_state"]), ids, is_relative=True)
        z0 = (obj.data.root_pos_w[0, 2] - origin[2]).item()
        rec_z = d["obs"]["object"][:, 2]
        zmax, end, why = z0, None, "ran out"
        for t, a in enumerate(d["actions"][:]):
            _, _, term, trunc, _ = env.step(torch.tensor(a, dtype=torch.float32)[None])
            p = (obj.data.root_pos_w[0] - origin).numpy()
            zmax = max(zmax, p[2])
            if t % 40 == 0:
                print(f"BOX_REPLAY {name} t={t} obj {p.round(3).tolist()} recorded wheel z {rec_z[min(t, len(rec_z) - 1)]:.3f}")
            if term[0] or trunc[0]:
                why = [n for n in env.termination_manager.active_terms if env.termination_manager.get_term(n)[0]]
                end = t
                break
        p = (obj.data.root_pos_w[0] - origin).numpy()
        print(f"BOX_REPLAY {args.object} {name}: start z {z0:.3f}, max rise {zmax - z0:.3f} m (recorded {rec_z.max() - rec_z[0]:.3f}), "
              f"end {p.round(3).tolist()} at step {end} of {len(d['actions'])}, ended by {why}")
env.close()
simulation_app.close()
