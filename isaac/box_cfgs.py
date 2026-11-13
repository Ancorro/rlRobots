"""The stock G1 pick-and-place task and its Mimic generator with the steering wheel swapped for a box (owner 2026-10-21:
one box via Mimic demos from the wheel demos). Everything else is NVIDIA's: robot, table, actions, observations, the
success check (task_done_pick_place) and the three Mimic subtasks, whose recorded hand poses are relative to the
object's frame. So the box is sized and placed like the wheel (isaac/box_probe.py): its sides sit where the recorded
hands closed on the wheel's rim, and its frame is where the wheel's was. The wheel's frame is at its bottom (0.5 cm above
its lowest point), not its centre, so the box is a small USD file whose cube sits above its frame the same way; a plain
CuboidCfg is centred on its frame, and centred at the wheel's root it sank half into the table (0 of ~2,400 Mimic tries).

Size, mass and start position from the environment (set by sdg_cluster.sbatch): BOX_SIZE="x,y,z" m, BOX_MASS kg,
BOX_POS="x,y,z" (env frame, the box's frame, i.e. 0.5 cm above its bottom like the wheel's). Imported lazily by the gym entries in isaac/box_tasks.py, after the app starts.
"""
import os
import tempfile

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab_mimic.envs.pinocchio_envs.locomanipulation_g1_mimic_env_cfg import LocomanipulationG1MimicEnvCfg
from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg


def _vec(name, default):
    return tuple(float(v) for v in os.environ.get(name, default).split(","))


SIZE = _vec("BOX_SIZE", "0.28,0.28,0.05")
MASS = float(os.environ.get("BOX_MASS", "0.5"))
POS = _vec("BOX_POS", "-0.35,0.45,0.6996")
BELOW = 0.0048  # the wheel's lowest point is this far below its frame

WALL = float(os.environ.get("BOX_WALL", "0"))  # > 0: an open-top crate with walls this thick (the hands hook its rim
# like the wheel's; a solid box only gets pushed, isaac/box_replay.py); 0: a solid box
SHAPE = os.environ.get("BOX_SHAPE", "box")  # box | cylinder (upright, diameter SIZE[0] = SIZE[1], height SIZE[2]; owner
# 2026-10-22 goal: objects of varied shape. The scripted squeeze uses SIZE[0] / 2 as the half-width, so it fits as is)
YAW = float(os.environ.get("BOX_YAW", "0"))  # degrees the box shape is turned inside its frame: the recorded left hand
# comes at the wheel from (+0.145, -0.199), about 54 deg off the x axis, so 36 puts a wall square to it

HEAD = """#usda 1.0
(
    defaultPrim = "Box"
    upAxis = "Z"
    metersPerUnit = 1
)
def Xform "Box" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"])
{{
    float physics:mass = {m}
    def Material "friction" (prepend apiSchemas = ["PhysicsMaterialAPI"])
    {{
        float physics:staticFriction = 1
        float physics:dynamicFriction = 1
    }}
"""
PART = """    def Cube "{name}" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"])
    {{
        double size = 1
        double3 xformOp:translate = ({x}, {y}, {z})
        float3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.55, 0.4, 0.25)]
        rel material:binding:physics = </Box/friction> (bindMaterialAs = "weakerThanDescendants")
    }}
"""
TURN = """    def Xform "shape"
    {{
        float xformOp:rotateZ = {yaw}
        uniform token[] xformOpOrder = ["xformOp:rotateZ"]
{parts}    }}
"""


def parts():
    """(name, centre, size) of each cube, in the box frame (BELOW above the bottom)."""
    sx, sy, sz = SIZE
    if WALL <= 0:
        return [("solid", (0, 0, sz / 2 - BELOW), (sx, sy, sz))]
    w, zc = WALL, sz / 2 - BELOW
    return [("floor", (0, 0, w / 2 - BELOW), (sx, sy, w)),
            ("wall_xn", (-(sx - w) / 2, 0, zc), (w, sy, sz)), ("wall_xp", ((sx - w) / 2, 0, zc), (w, sy, sz)),
            ("wall_yn", (0, -(sy - w) / 2, zc), (sx - 2 * w, w, sz)), ("wall_yp", (0, (sy - w) / 2, zc), (sx - 2 * w, w, sz))]


CYL = """    def Cylinder "solid" (prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"])
    {{
        double radius = {r}
        double height = {h}
        uniform token axis = "Z"
        double3 xformOp:translate = (0, 0, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]
        color3f[] primvars:displayColor = [(0.3, 0.45, 0.6)]
        rel material:binding:physics = </Box/friction> (bindMaterialAs = "weakerThanDescendants")
    }}
"""


def box_usd():
    sx, sy, sz = SIZE
    if SHAPE == "cylinder":
        path = os.path.join(tempfile.gettempdir(), f"cyl-{sx}-{sz}-{MASS}.usda")
        body = CYL.format(r=sx / 2, h=sz, z=sz / 2 - BELOW)
        body = "".join("    " + line if line.strip() else line for line in body.splitlines(True))
        with open(path, "w") as f:
            f.write(HEAD.format(m=MASS) + TURN.format(yaw=0, parts=body) + "}" + chr(10))
        return path
    path = os.path.join(tempfile.gettempdir(), f"box-{sx}-{sy}-{sz}-{MASS}-{WALL}-{YAW}.usda")
    cubes = "".join(PART.format(name=n, x=c[0], y=c[1], z=c[2], sx=d[0], sy=d[1], sz=d[2]) for n, c, d in parts())
    cubes = "".join("    " + line if line.strip() else line for line in cubes.splitlines(True))
    text = HEAD.format(m=MASS) + TURN.format(yaw=YAW, parts=cubes) + "}\n"
    with open(path, "w") as f:
        f.write(text)
    return path


def box_object():
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=box_usd(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


GOAL = os.environ.get("BOX_GOAL", "")  # "x,y": the box's goal centre for the scripted two-hand task
# (isaac/box_scripted.py); the stock check then wants it within 8 cm of there. Empty: the stock goal region.


@configclass
class BoxEnvCfg(LocomanipulationG1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = box_object()
        if GOAL:
            gx, gy = (float(v) for v in GOAL.split(","))
            self.box_goal_centre = (gx, gy, POS[2] + SIZE[2] / 2 - BELOW)
            self.terminations.success.func = box_set_down
            self.terminations.success.params.update(min_x=gx - 0.08, max_x=gx + 0.08, min_y=gy - 0.08, max_y=gy + 0.08)


def box_set_down(env, task_link_name="", min_x=0.0, max_x=0.0, min_y=0.0, max_y=0.0, **_):
    """The scripted task's success: the box rests in the goal region (centre within 2 cm of its start height, slower than
    0.2 m/s) and both wrists are 5 cm clear of its sides. The stock check also passed a box still held in the air."""
    import torch

    box, robot, o = env.scene["object"], env.scene["robot"], env.scene.env_origins
    p = box.data.root_pos_w - o
    names = robot.data.body_names
    lx = robot.data.body_pos_w[:, names.index("left_wrist_yaw_link"), 0] - o[:, 0]
    rx = robot.data.body_pos_w[:, names.index("right_wrist_yaw_link"), 0] - o[:, 0]
    clear = SIZE[0] / 2 + 0.05
    done = (p[:, 0] > min_x) & (p[:, 0] < max_x) & (p[:, 1] > min_y) & (p[:, 1] < max_y)
    done &= torch.abs(p[:, 2] - POS[2]) < 0.02
    done &= torch.all(torch.abs(box.data.root_vel_w[:, :3]) < 0.2, dim=1)
    return done & (p[:, 0] - lx > clear) & (rx - p[:, 0] > clear)


def object_size(env):
    """The object's size (x, y, z m) and shape flag (1 = cylinder) for every env: tells a size-aware BC which box it holds (owner 2026-10-22 goal: objects
    of varied shape and size). One size per job (the USD is made per job), so mixed-size data comes from merging jobs."""
    import torch

    v = list(SIZE) + [1.0 if SHAPE == "cylinder" else 0.0]  # size and a cylinder flag
    return torch.tensor(v, dtype=torch.float32, device=env.device).repeat(env.num_envs, 1)


@configclass
class BoxSizedEnvCfg(BoxEnvCfg):
    """BoxEnvCfg plus the object_size observation."""

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        self.observations.policy.object_size = ObsTerm(func=object_size)


@configclass
class BoxMimicEnvCfg(LocomanipulationG1MimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.object = box_object()
        self.datagen_config.name = "demo_src_g1_locomanip_box"
