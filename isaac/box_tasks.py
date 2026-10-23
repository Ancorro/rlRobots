"""Gym ids for the box versions of the G1 pick-and-place task (isaac/box_cfgs.py), and a runner that registers them
before running one of Isaac Lab's scripts unchanged:

  /isaac-sim/python.sh box_tasks.py scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-Locomanipulation-G1-Box-Abs-Mimic-v0 ...
  /isaac-sim/python.sh box_tasks.py scripts/imitation_learning/robomimic/train.py --task Isaac-PickPlace-Locomanipulation-G1-Box-Abs-v0 ...
The config entry points are strings, so nothing Isaac-specific is imported until the script has started the app.
"""
import os
import runpy
import sys

import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ISAACLAB = os.environ.get("ISAACLAB", os.path.expandvars("/nfs/hpc/share/$USER/isaac/IsaacLab"))
BC_JSON = os.path.join(ISAACLAB, "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/agents/robomimic/bc_rnn_low_dim.json")

gym.register(
    id="Isaac-PickPlace-Locomanipulation-G1-Box-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "box_cfgs:BoxEnvCfg", "robomimic_bc_cfg_entry_point": BC_JSON},
    disable_env_checker=True,
)



def _sized_bc_json():
    """The stock BC-RNN config with object_size added to the low-dim observations."""
    import json
    import tempfile

    c = json.load(open(BC_JSON))
    c["observation"]["modalities"]["obs"]["low_dim"].append("object_size")
    path = os.path.join(tempfile.gettempdir(), "bc_rnn_low_dim_sized.json")
    json.dump(c, open(path, "w"), indent=1)
    return path


gym.register(
    id="Isaac-PickPlace-Locomanipulation-G1-BoxSized-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point": "box_cfgs:BoxSizedEnvCfg", "robomimic_bc_cfg_entry_point": _sized_bc_json()},
    disable_env_checker=True,
)
gym.register(
    id="Isaac-Locomanipulation-G1-Box-Abs-Mimic-v0",
    entry_point="isaaclab_mimic.envs.pinocchio_envs.locomanipulation_g1_mimic_env:LocomanipulationG1MimicEnv",
    kwargs={"env_cfg_entry_point": "box_cfgs:BoxMimicEnvCfg"},
    disable_env_checker=True,
)

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
