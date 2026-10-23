"""Dry run: rehearse a skill on a ghost arm before the real one moves.

The ghost is a weightless copy (so101/robot.py KinematicArm) posed from the arm's present joint readings. The
skill runs on it to the end, assuming every grasp works (a ghost has nothing to grip), and every target is checked
against the safety rules. With a viewer, the ghost is drawn see-through over the scene (blue, or red if a rule
breaks) and plays the moves at twice the speed. ask=True then waits for Enter before the arm moves.
"""
import copy
import time

import mujoco
import numpy as np

from so101.robot import TICK, KinematicArm


def rehearse(arm, make, guard=None, viewer=None, speed=2, ask=False):
    """make(arm) -> skill generator. Returns the safety rules the plan breaks (empty list: fine to run)."""
    ghost = copy.copy(arm)  # same model and settings; own joints and pose
    ghost.robot = KinematicArm(arm.q)
    ghost.state = mujoco.MjData(arm.model)
    ghost.q = ghost.prev = None
    ghost.assume_holding = True
    ghost.sync()
    path, skill = [], make(ghost)
    while len(path) < 5000:
        try:
            q = next(skill)
        except StopIteration:
            break
        path.append(np.asarray(q, float))
        ghost.step(q)
    broken = {}
    if guard is not None:
        for q in path:
            for b in guard.check(q)[0]:
                broken.setdefault(b.split(":")[0], b)  # first breach of each rule
    broken = list(broken.values())
    if viewer is not None and viewer.is_running():
        show(viewer, arm.model, path, speed, bad=bool(broken), root=arm.root, qadr=getattr(arm, "qadr", None))
    if ask and not broken:
        if input("Dry run done. Run it on the arm? [Enter = yes, n = no] ").strip().lower().startswith("n"):
            return ["refused by you"]
    return broken


def add_ghost(model, d, scn, rgba, opt=None, pert=None, root="base"):
    """Draw the robot posed in `d` into scene `scn`, see-through in colour rgba (other bodies left out)."""
    robot = model.body_rootid == model.body(root).id
    n0 = scn.ngeom
    mujoco.mjv_addGeoms(model, d, opt or mujoco.MjvOption(), pert or mujoco.MjvPerturb(),
                        int(mujoco.mjtCatBit.mjCAT_DYNAMIC), scn)
    for i in range(n0, scn.ngeom):
        g = scn.geoms[i]
        mine = g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM) and robot[model.geom_bodyid[g.objid]]
        g.rgba[:] = rgba if mine else (0, 0, 0, 0)


def show(viewer, model, path, speed=2, bad=False, root="base", qadr=None):
    """Play joint path `path` as a see-through ghost arm in a passive MuJoCo viewer."""
    d = mujoco.MjData(model)
    rgba = (1.0, 0.25, 0.25, 0.35) if bad else (0.3, 0.75, 1.0, 0.35)
    for q in path[::speed] + [path[-1]] if path else []:
        d.qpos[slice(0, len(q)) if qadr is None else qadr] = q
        mujoco.mj_kinematics(model, d)
        with viewer.lock():
            viewer.user_scn.ngeom = 0
            add_ghost(model, d, viewer.user_scn, rgba, viewer.opt, viewer.perturb, root)
        viewer.sync()
        time.sleep(TICK)
    time.sleep(0.3)
    with viewer.lock():
        viewer.user_scn.ngeom = 0
    viewer.sync()
