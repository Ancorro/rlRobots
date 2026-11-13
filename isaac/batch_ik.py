"""Arm IK (inverse kinematics) for every robot at once, in torch, in place of the stock task's Pink IK (owner 2026-10-21:
"why cant we do 1000 parallel on isaac anymore"). Pink solves one robot at a time on the CPU, so 32 robots per job ran
only 3x faster than 1. This term takes the same action (left wrist pos + quat, right wrist pos + quat, 14 hand joints;
env frame) and does one damped least-squares step per physics step for all robots, from PhysX's Jacobians, so the sim
can run on the GPU with thousands of copies.

Like Pink's config: wrist position weighted 8, orientation 2, gain 0.5 per step, a weak pull of the shoulders and waist
toward their default angles, gravity compensation on the arm joints. Use: BOX_IK=batch (isaac/box_cfgs.py).
"""
from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

ARM_JOINTS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint",
              ".*_wrist_pitch_joint", ".*_wrist_roll_joint", ".*_wrist_yaw_joint", "waist_.*_joint"]
POSTURE_JOINTS = [".*_shoulder_.*_joint", "waist_.*_joint"]  # Pink's NullSpacePostureTask joints


class BatchIKAction(ActionTerm):
    cfg: BatchIKActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        a = self._asset
        self._arm_ids, _ = a.find_joints(ARM_JOINTS)
        self._hand_ids, _ = a.find_joints(cfg.hand_joint_names)  # asset order, as Pink's term maps the 14 hand columns
        post, _ = a.find_joints(POSTURE_JOINTS)
        waist, _ = a.find_joints("waist_.*_joint")
        self._post_mask = torch.tensor([cfg.waist_hold if j in waist else float(j in post) for j in self._arm_ids], device=self.device)
        self._wrists = [a.body_names.index(n) for n in cfg.wrist_links]
        self._jac_cols = [j + 6 for j in self._arm_ids]  # floating base: PhysX puts the root's 6 dofs first
        self._raw = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._q_arm = a.data.default_joint_pos[:, self._arm_ids].clone()
        lim = a.data.soft_joint_pos_limits[:, self._arm_ids]
        self._lo, self._hi = lim[..., 0], lim[..., 1]
        w = torch.tensor([cfg.position_cost] * 3 + [cfg.orientation_cost] * 3, device=self.device) ** 2
        self._w = torch.cat([w, w])  # both wrists, (12,)

    @property
    def action_dim(self):
        return 2 * 7 + len(self.cfg.hand_joint_names)

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions

    def apply_actions(self):
        a, n = self._asset, len(self._arm_ids)
        o = self._env.scene.env_origins
        J = a.root_physx_view.get_jacobians()[:, self._wrists][..., self._jac_cols].reshape(self.num_envs, 12, n)
        err = []
        for k, b in enumerate(self._wrists):
            tp, tq = self._raw[:, 7 * k: 7 * k + 3] + o, self._raw[:, 7 * k + 3: 7 * k + 7]
            ep, er = math_utils.compute_pose_error(a.data.body_pos_w[:, b], a.data.body_quat_w[:, b], tp, tq, "axis_angle")
            err += [ep, er]
        e = torch.cat(err, 1) * self.cfg.gain  # (N, 12)
        q = a.data.joint_pos[:, self._arm_ids]
        JtW = J.transpose(1, 2) * self._w
        H = JtW @ J + self.cfg.damping * torch.eye(n, device=self.device) + torch.diag_embed(self.cfg.posture * self._post_mask)
        g = (JtW @ e.unsqueeze(-1)).squeeze(-1)
        g = g + self.cfg.posture * self._post_mask * self.cfg.posture_gain * (a.data.default_joint_pos[:, self._arm_ids] - q)
        dq = torch.linalg.solve(H, g).clamp(-self.cfg.max_step, self.cfg.max_step)
        self._q_arm = torch.clamp(q + dq, self._lo, self._hi)
        a.set_joint_position_target(self._q_arm, joint_ids=self._arm_ids)
        a.set_joint_position_target(self._raw[:, 14:], joint_ids=self._hand_ids)
        if not a.cfg.spawn.rigid_props.disable_gravity:  # as Pink's term: hold the arms' weight
            gc = a.root_physx_view.get_gravity_compensation_forces()[:, self._jac_cols]
            a.set_joint_effort_target(gc, joint_ids=self._arm_ids)

    def reset(self, env_ids=None):
        self._raw[env_ids] = 0.0


@configclass
class BatchIKActionCfg(ActionTermCfg):
    class_type: type = BatchIKAction
    hand_joint_names: list = None
    wrist_links: tuple = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    position_cost: float = 8.0
    orientation_cost: float = 2.0
    gain: float = 0.5
    damping: float = 0.05
    posture: float = 0.02
    posture_gain: float = 0.3
    waist_hold: float = float(__import__("os").environ.get("IK_WAIST", "5000"))  # posture weight on the waist joints, x posture: the
    # AGILE legs balance for a near-upright waist. 1024 robots (LOG 2026-10-21): waist 1, step 0.1: 429 fell, 384 succeeded;
    # 5000, 0.02: 10 fell, 520 succeeded (Pink: 58%)
    max_step: float = float(__import__("os").environ.get("IK_STEP", "0.02"))  # rad per physics step


def use_batch_ik(cfg):
    """Swap the env cfg's Pink arm term for this one; the action layout stays the same."""
    pink = cfg.actions.upper_body_ik
    cfg.actions.upper_body_ik = BatchIKActionCfg(asset_name="robot", hand_joint_names=list(pink.hand_joint_names))
