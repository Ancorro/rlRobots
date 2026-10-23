"""Run Isaac Lab's locomanipulation_sdg/generate_data.py with the mobility_gen extension enabled after the app starts.

Enabled at startup (--kit_args='--enable isaacsim.replicator.mobility_gen') Isaac segfaults in librtx.scenedb while
creating the first stage (2026-10-18, any GPU, cameras on or off); without it the script fails importing mobility_gen's
path planner. So: start the app plainly, then enable the extension, then let the script import it.

  /isaac-sim/python.sh .../nav_late_ext.py <generate_data.py> [its args...]
"""
import runpy
import sys

from isaaclab.app import AppLauncher

EXT = "isaacsim.replicator.mobility_gen"
_init = AppLauncher.__init__


def _init_then_enable(self, *a, **k):
    _init(self, *a, **k)
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    manager.set_extension_enabled_immediate(EXT, True)
    print(f"[nav_late_ext] {EXT} enabled: {manager.is_extension_enabled(EXT)}", flush=True)


AppLauncher.__init__ = _init_then_enable
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
