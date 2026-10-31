"""A stand-in for Isaac Lab, enough to run isaac/walk_grasp.py end to end on the CPU with real tensor shapes: the
script's imports (isaaclab.*, isaaclab_tasks.*, gymnasium, pinocchio), a scene with N robots and objects whose tensors
have the shapes the real one has, and an env.step that moves each robot root by its walking command and keeps the
right wrist where the action asks (a kinematic wrist: the object follows it when the hand is closed near it). No
physics: what this checks is that every tensor operation in the script has the right shape and dtype, on every code
path (scripted, learned grab, primitives, scanned objects, the carry and set-down, the BENCH_RESULT record).

    install(n_bodies=60)  -> registers the fake modules in sys.modules; then run the script with runpy
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "isaac"))
import grab_policy as G  # noqa: E402  (the quaternion helpers, same (w, x, y, z) convention as Isaac Lab)

HAND_JOINTS = [f"{s}_hand_{n}_joint" for s in ("left", "right") for n in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
LEG_JOINTS = [f"{s}_{n}_joint" for s in ("left", "right") for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
ARM_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [f"{s}_{n}_joint" for s in ("left", "right")
                                                                                for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
JOINTS = LEG_JOINTS + ARM_JOINTS + HAND_JOINTS  # 12 + 17 + 14 = 43
BODIES = ["pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "right_hand_palm_link", "right_hand_thumb_2_link",
          "right_hand_index_1_link", "right_hand_middle_1_link", "left_hand_palm_link"]


class NS(types.SimpleNamespace):
    """A config namespace that grows attributes on demand (parse_env_cfg's result is edited freely by the script)."""

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        v = NS()
        setattr(self, k, v)
        return v

    def replace(self, **kw):
        n = NS(**vars(self))
        for k, v in kw.items():
            setattr(n, k, v)
        return n


def _cfg(**kw):
    return NS(**kw)


class Data:
    pass


class Articulation:
    def __init__(self, n, dev, origins):
        self.body_names, self.joint_names = list(BODIES), list(JOINTS)
        self.data = Data()
        self.data.root_pos_w = origins + torch.tensor([0.0, 0.0, 0.75], device=dev)
        self.data.root_quat_w = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], device=dev).repeat(n, 1)  # facing +y
        self.data.body_pos_w = torch.zeros(n, len(BODIES), 3, device=dev)
        self.data.body_quat_w = torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(n, len(BODIES), 1)
        self.data.joint_pos = torch.zeros(n, len(JOINTS), device=dev)
        self.data.joint_pos_target = torch.zeros(n, len(JOINTS), device=dev)
        self.data.root_lin_vel_w = torch.zeros(n, 3, device=dev)
        self.refresh()

    def refresh(self):
        rp, rq = self.data.root_pos_w, self.data.root_quat_w
        for k, b in enumerate(BODIES):
            off = {"pelvis": (0, 0, 0), "torso_link": (0, 0, 0.25), "left_wrist_yaw_link": (0.25, 0.15, 0.0),
                   "right_wrist_yaw_link": (0.25, -0.15, 0.0)}.get(b, (0.30, -0.15, -0.05))
            self.data.body_pos_w[:, k] = rp + G.matrix_from_quat(rq) @ torch.tensor(off, dtype=torch.float32, device=rp.device)
            self.data.body_quat_w[:, k] = rq


class RigidObject:
    def __init__(self, n, dev):
        self.data = Data()
        self.data.root_pos_w = torch.zeros(n, 3, device=dev)
        self.data.root_quat_w = torch.tensor([[1.0, 0, 0, 0]], device=dev).repeat(n, 1)
        self.data.root_lin_vel_w = torch.zeros(n, 3, device=dev)

    def write_root_pose_to_sim(self, pose):
        self.data.root_pos_w[:] = pose[:, :3]
        self.data.root_quat_w[:] = pose[:, 3:7]

    def write_root_velocity_to_sim(self, v):
        self.data.root_lin_vel_w[:] = v[:, :3]


class Scene:
    def __init__(self, n, dev):
        self.env_origins = torch.stack([torch.tensor([6.0 * i, 0.0, 0.0], device=dev) for i in range(n)])
        self._items = {"robot": Articulation(n, dev, self.env_origins), "object": RigidObject(n, dev)}

    def __getitem__(self, k):
        if k not in self._items:
            self._items[k] = RigidObject(len(self.env_origins), self.env_origins.device)
        return self._items[k]


class Env:
    """step(action): action = [left wrist pos 3, quat 4, right wrist pos 3, quat 4, fingers 14, vx vy wz hip] per env."""

    def __init__(self, n, dev):
        self.scene, self.device, self.step_dt = Scene(n, dev), dev, 0.02
        self.cfg = NS(actions=NS(upper_body_ik=NS(hand_joint_names=list(HAND_JOINTS))))
        self.unwrapped = self
        self.n = n
        self.held = torch.zeros(n, dtype=torch.bool, device=dev)

    def reset(self):
        return {}, {}

    def step(self, a):
        assert a.shape == (self.n, 32) and a.dtype == torch.float32, (a.shape, a.dtype)
        assert torch.isfinite(a).all()
        r, o, org = self.scene["robot"], self.scene["object"], self.scene.env_origins
        vx, vy, wz = a[:, 28], a[:, 29], a[:, 30]
        yaw = torch.atan2(2 * (r.data.root_quat_w[:, 0] * r.data.root_quat_w[:, 3]), 1 - 2 * r.data.root_quat_w[:, 3] ** 2)
        yaw = yaw + wz * self.step_dt
        fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], 1)
        left = torch.stack([-torch.sin(yaw), torch.cos(yaw)], 1)
        r.data.root_pos_w[:, :2] += (vx[:, None] * fwd + vy[:, None] * left) * self.step_dt
        r.data.root_quat_w[:] = torch.stack([torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)], 1)
        r.refresh()
        # the right wrist and hand go where the action asks (kinematic arm); the hand bodies ride 5 cm below the wrist
        hb = r.body_names.index("right_wrist_yaw_link")
        r.data.body_pos_w[:, hb] = a[:, 7:10] + org
        r.data.body_quat_w[:, hb] = a[:, 10:14] / a[:, 10:14].norm(dim=1, keepdim=True)
        for b in ("right_hand_palm_link", "right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"):
            r.data.body_pos_w[:, r.body_names.index(b)] = r.data.body_pos_w[:, hb] + torch.tensor([0.0, 0.0, -0.05], device=self.device)
        r.data.joint_pos[:, [r.joint_names.index(j) for j in HAND_JOINTS]] = a[:, 14:28]
        # a crude grasp: the thumb closed (right thumb_1 target < 0) with the pinch near the object -> the object follows the hand
        pinch = r.data.body_pos_w[:, hb] + torch.tensor([0.1269, 0.0753, 0.0213], device=self.device)
        near = (o.data.root_pos_w - pinch).norm(dim=1) < 0.06
        closed = a[:, 14 + 8] < -0.5
        self.held = (self.held | (near & closed)) & closed
        o.data.root_pos_w[self.held] = pinch[self.held]
        o.data.root_lin_vel_w[:] = 0.0
        return {}, torch.zeros(self.n), torch.zeros(self.n, dtype=torch.bool), torch.zeros(self.n, dtype=torch.bool), {}


def make_env_cfg(task, device, num_envs):
    cfg = NS()
    cfg.scene = NS(env_spacing=2.5, replicate_physics=True, robot=NS(actuators={"hands": NS(stiffness=20.0)}, spawn=NS(articulation_props=NS())))
    cfg.terminations = NS()
    return cfg


def install(n_envs_holder=None):
    """Register the fake modules. The env size is taken from parse_env_cfg's num_envs."""
    m = lambda name, **kw: sys.modules.setdefault(name, types.ModuleType(name)) and sys.modules[name].__dict__.update(kw) or sys.modules[name]  # noqa: E731
    m("pinocchio")
    app = m("isaaclab.app")
    app.AppLauncher = lambda *a, **k: NS(app=NS(close=lambda: None))
    m("isaaclab")
    sim = m("isaaclab.sim")
    for c in ("CollisionPropertiesCfg", "RigidBodyMaterialCfg", "PreviewSurfaceCfg", "CuboidCfg", "CylinderCfg", "CapsuleCfg",
              "SphereCfg", "RigidBodyPropertiesCfg", "MassPropertiesCfg", "MultiAssetSpawnerCfg", "SpawnerCfg"):
        setattr(sim, c, type(c, (NS,), {"__init__": lambda self, *a, **kw: NS.__init__(self, **kw)}))
    m("isaaclab_tasks")
    m("isaaclab_tasks.manager_based")
    m("isaaclab_tasks.manager_based.locomanipulation")
    m("isaaclab_tasks.manager_based.locomanipulation.pick_place")
    assets = m("isaaclab.assets")
    assets.AssetBaseCfg = type("AssetBaseCfg", (NS,), {"__init__": lambda self, **kw: NS.__init__(self, **kw), "InitialStateCfg": NS})
    assets.RigidObjectCfg = type("RigidObjectCfg", (NS,), {"__init__": lambda self, **kw: NS.__init__(self, **kw), "InitialStateCfg": NS})
    u = m("isaaclab.utils")
    u.configclass = lambda cls: cls
    mm = m("isaaclab.utils.math")
    mm.quat_apply = lambda q, v: (G.matrix_from_quat(q) @ v[..., None])[..., 0]
    mm.quat_inv = lambda q: q * torch.tensor([1.0, -1, -1, -1], device=q.device)
    mm.quat_mul, mm.quat_from_angle_axis, mm.quat_from_matrix, mm.matrix_from_quat = G.quat_mul, G.quat_from_angle_axis, G.quat_from_matrix, G.matrix_from_quat
    mm.combine_frame_transforms = lambda p1, q1, p2, q2: (p1 + mm.quat_apply(q1, p2), G.quat_mul(q1, q2))
    mm.subtract_frame_transforms = lambda p1, q1, p2, q2: (mm.quat_apply(mm.quat_inv(q1), p2 - p1), G.quat_mul(mm.quat_inv(q1), q2))
    tu = m("isaaclab_tasks.utils")
    holder = {}

    def parse_env_cfg(task, device="cpu", num_envs=1):
        holder["n"] = num_envs
        return make_env_cfg(task, device, num_envs)
    tu.parse_env_cfg = parse_env_cfg
    import gymnasium as gym  # the real one (so101 registers its tasks with it); only make() is replaced
    gym.make = lambda task, cfg=None: Env(holder["n"], "cpu")
    return holder
