"""Run Isaac Lab's locomanipulation_sdg/generate_data.py with only mobility_gen's path planner, not the extension.

Enabling isaacsim.replicator.mobility_gen pulls in omni.replicator.core and the viewport, which start the RTX renderer;
on the cluster container that segfaults (enabled at startup) or hangs (enabled after app start, 2026-10-19) in
rtx.scenedb. generate_data.py only needs impl/path_planner.py (numpy plus one compiled binding), so register the
package folders by hand, skipping their __init__ files, and import that one module. A hang dumps every thread's stack.

  /isaac-sim/python.sh .../nav_planner_shim.py <generate_data.py> [its args...]
"""
import faulthandler
import importlib
import os
import sys
import types

faulthandler.dump_traceback_later(int(os.environ.get("HANG_S", "600")), repeat=True)  # stacks every HANG_S seconds

ROOT = "/isaac-sim/exts/isaacsim.replicator.mobility_gen/isaacsim/replicator/mobility_gen"


def _pkg(name, path):
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m
    return m


def _install():
    import isaacsim

    try:
        import isaacsim.replicator
    except ModuleNotFoundError:  # only on the path when a replicator extension is enabled
        isaacsim.replicator = _pkg("isaacsim.replicator", ROOT.rsplit("/", 1)[0])
    mg = _pkg("isaacsim.replicator.mobility_gen", ROOT)
    mg.impl = _pkg("isaacsim.replicator.mobility_gen.impl", ROOT + "/impl")
    mg.bindings = importlib.import_module("isaacsim.replicator.mobility_gen.bindings")
    mg.impl.path_planner = importlib.import_module("isaacsim.replicator.mobility_gen.impl.path_planner")
    isaacsim.replicator.mobility_gen = mg
    print("[nav_planner_shim] path_planner loaded:", mg.impl.path_planner.generate_paths.__name__, flush=True)
    _drop_camera()


def _drop_camera():
    """Remove the nav task's robot-view RGB camera: any camera needs RTX, which Isaac Sim 5.1 crashes on with
    drivers past 580 (cluster 615). The recorded data loses only that image; the low-dim observations stay."""
    from isaaclab_mimic.locomanipulation_sdg.envs import g1_locomanipulation_sdg_env as g

    post = g.G1LocomanipulationSDGEnvCfg.__post_init__

    def post_no_cam(self):
        post(self)
        self.scene.robot_pov_cam = None
        self.observations.policy.robot_pov_cam = None

    g.G1LocomanipulationSDGEnvCfg.__post_init__ = post_no_cam
    print("[nav_planner_shim] robot_pov_cam dropped", flush=True)


# Install after the app starts (isaacsim.* modules resolve only then), before the script's imports.
from isaaclab.app import AppLauncher  # noqa: E402

_init = AppLauncher.__init__


def _init_then_install(self, *a, **k):
    _init(self, *a, **k)
    print("[nav_planner_shim] app started", flush=True)
    _install()


AppLauncher.__init__ = _init_then_install

# generate_data.py scores nothing and has no limit on NAVIGATE (one run circled 57k steps, 2026-10-19). Run its
# definitions, wrap replay() with a judge and a NAVIGATE cap, then run its main block.
NAV_MAX = 3000  # control steps (60 s)
MARK = 'if __name__ == "__main__":'


class Stuck(Exception):
    pass


def _patch(g):
    nav, replay = g["handle_navigate_state"], g["replay"]
    count = [0]

    def nav_capped(*a, **k):
        count[0] += 1
        if count[0] > NAV_MAX:
            raise Stuck
        return nav(*a, **k)

    def replay_judged(*a, **k):
        env = k.get("env", a[0] if a else None)
        count[0], stuck = 0, False
        try:
            replay(*a, **k)
        except Stuck:
            stuck = True
        o = env.scene["object"].data.root_pos_w[0] - env.scene.env_origins[0]
        p = env.scene["robot"].data.root_pos_w[0] - env.scene.env_origins[0]
        on, local = _on_table2(env, o)
        print(f"[nav_judge] stuck={stuck} nav_steps={min(count[0], NAV_MAX)} obj={[round(float(x), 3) for x in o]} "
              f"obj_in_table2={local} robot_z={float(p[2]):.2f} placed={not stuck and on}",
              flush=True)
        if stuck:
            env.reset()

    g["handle_navigate_state"], g["replay"] = nav_capped, replay_judged


import torch  # noqa: E402


def _on_table2(env, o):
    """Is the object on the drop-off table (end fixture, re-placed at random each run) and off the floor?
    Footprint +-1.45 x +-0.45 m, as its occupancy-map boundary in g1_locomanipulation_sdg_env.py."""
    x, y, yaw = (float(v) for v in env.get_end_fixture().get_pose_2d()[0][:3])
    dx, dy = float(o[0]) - x, float(o[1]) - y
    c, s_ = torch.cos(torch.tensor(-yaw)), torch.sin(torch.tensor(-yaw))
    lx, ly = float(c * dx - s_ * dy), float(s_ * dx + c * dy)
    return abs(lx) <= 1.45 and abs(ly) <= 0.45 and float(o[2]) > 0.5, (round(lx, 2), round(ly, 2))


if __name__ == "__main__":  # nav_bc_eval.py imports this module for the hooks only
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    head, main = open(script).read().split(MARK, 1)
    g = {"__name__": "nav_gen", "__file__": script}
    exec(compile(head, script, "exec"), g)
    _patch(g)
    exec(compile("if True:" + main, script, "exec"), g)
