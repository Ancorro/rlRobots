"""PPO (proximal policy optimization) for the two-hand box task on the walking G1 (owner 2026-10-21: "lets just RL ? why
cant we rl walking here", "we can warm up with the nvidia existing walking network?").

The stock locomanipulation G1 (isaac/box_cfgs.py's box scene) with NVIDIA's pretrained AGILE walking policy kept frozen
as the legs, and isaac/batch_ik.py's arm IK (inverse kinematics) for all robots at once on the GPU, so thousands of
robots train side by side. The policy acts every 20 ms:
  - 6 numbers: each wrist's target moves up to --step m along x, y, z of the pelvis frame (orientations stay at rest),
  - 1 number: both hands' grip, open .. the scripted demos' closed grip (finger angles from the wheel demos),
  - with --walk, 3 more: AGILE's forward, sideways and turning speed commands (hip height stays 0.72).
Without --walk the legs get a zero-speed command (standing; the goal is then on the same table, BOX_GOAL).

Reward in stages: wrists to their grip points beside the box; box raised (while both hands are near it); box carried
over the goal (while raised); the box set down on the goal with both hands clear (box_cfgs.box_set_down, which also
ends the episode). An episode also ends when the robot falls, the box tips past 45 degrees, or the box falls.

  python isaac/box_rl.py train --num_envs 4096 --iters 300 --out runs/boxrl-v1 --dataset generated_dataset_g1_locomanip.hdf5
"""
import argparse
import os
import sys
import time

import pinocchio  # noqa: F401  (before AppLauncher; the locomanipulation package imports pink)
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["train"])
p.add_argument("--num_envs", type=int, default=4096)
p.add_argument("--iters", type=int, default=300)
p.add_argument("--out", required=True)
p.add_argument("--dataset", required=True, help="the wheel demos, for the open and closed finger angles")
p.add_argument("--step", type=float, default=0.01, help="largest wrist target move per policy step (m)")
p.add_argument("--jitter", type=float, default=0.06, help="box start shift at reset, +- m in x and y")
p.add_argument("--walk", action="store_true", help="the policy also commands the legs (vx, vy, yaw rate)")
p.add_argument("--episode_s", type=float, default=12.0)
p.add_argument("--resume", default="", help="a model_*.pt to start from (its reward terms may differ)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
app = AppLauncher(args).app


def _fail(*exc):  # Kit's python.sh exits 0 after an uncaught exception; exit 1 so --dependency=afterok chains stop
    import traceback
    traceback.print_exception(*exc)
    sys.stdout.flush()
    os._exit(1)


sys.excepthook = _fail

from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import torch  # noqa: E402

import isaaclab.envs.mdp as base_mdp  # noqa: E402
import isaaclab.utils.math as mu  # noqa: E402
import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: E402,F401
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.action_cfg import AgileBasedLowerBodyActionCfg  # noqa: E402
from isaaclab_tasks.manager_based.locomanipulation.pick_place.mdp.actions import AgileBasedLowerBodyAction  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from batch_ik import BatchIKAction, BatchIKActionCfg  # noqa: E402
from box_cfgs import BELOW, GOAL, POS, SIZE, BoxEnvCfg, box_set_down  # noqa: E402

assert GOAL, "set BOX_GOAL"
L_IDX, R_IDX = [0, 1, 2, 6, 7, 8, 12], [3, 4, 5, 9, 10, 11, 13]  # each hand's joints in the 14 hand values
with h5py.File(args.dataset, "r") as f:
    _act = f["data"]["demo_0"]["actions"][:]
OPEN = _act[0, 14:28].copy()
SHUT = OPEN.copy()
SHUT[L_IDX], SHUT[R_IDX] = _act[105, 14:28][L_IDX], _act[150, 14:28][R_IDX]
SHUT = OPEN + 0.4 * (SHUT - OPEN)  # box_scripted.py's --close 0.4
SETTLE = 10  # policy steps the arms hold still after a reset before the box is put back
REST_Z = POS[2] + SIZE[2] / 2 - BELOW  # box centre height at rest on the table
SIDE = SIZE[0] / 2 + 0.012  # wrist distance from the box centre at contact (box_scripted.py's --palm)


class BoxArmAction(BatchIKAction):
    """7 policy numbers -> the 28-number wrist pose + hand action batch_ik.py tracks. Wrist targets live in the pelvis
    frame, so they move with the robot when it walks."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n = self.num_envs
        self._open = torch.tensor(OPEN, dtype=torch.float32, device=self.device)
        self._shut = torch.tensor(SHUT, dtype=torch.float32, device=self.device)
        self._act = torch.zeros(n, 7, device=self.device)
        self._tb = torch.zeros(n, 2, 3, device=self.device)
        self._raw = torch.zeros(n, 28, device=self.device)  # the base class sized it from action_dim (7)
        self._box0 = None  # each env's box pose (world) right after its reset
        self._rest = None  # (2, 3) wrist positions and (2, 4) orientations in the pelvis frame at the default pose

    @property
    def action_dim(self):
        return 7

    @property
    def raw_actions(self):
        return self._act

    def _pelvis(self):  # noqa: D401
        return self._asset.data.root_pos_w, self._asset.data.root_quat_w

    def process_actions(self, actions):
        a, n = self._asset, self.num_envs
        pp, pq = self._pelvis()
        if self._rest is None:  # first call: every robot is at its default pose
            wp, wq = a.data.body_pos_w[0, self._wrists], a.data.body_quat_w[0, self._wrists]
            self._rest = (mu.quat_apply_inverse(pq[0].expand(2, 4), wp - pp[0]),
                          mu.quat_mul(mu.quat_conjugate(pq[0]).expand(2, 4), wq))
            self._tb[:] = self._rest[0]
            print(f"BOXRL wrist rest, pelvis frame: {self._rest[0].cpu().numpy().round(3).tolist()}", flush=True)
        self._act[:] = actions.clamp(-1.0, 1.0)
        t = self._env.episode_length_buf
        self._act[t < SETTLE] = 0.0  # arms hold their rest pose while they settle after a reset
        back = (t == SETTLE).nonzero().flatten()
        if len(back) and self._box0 is not None:  # then the box goes back to its reset pose, at rest (box_bcplay.py's
            # clean reset: 17% of episodes tipped the box in the first 5 steps without it, 21901146)
            box = self._env.scene["object"]
            box.write_root_pose_to_sim(self._box0[back], env_ids=back)
            box.write_root_velocity_to_sim(torch.zeros(len(back), 6, device=self.device), env_ids=back)
        self._tb += self._act[:, :6].view(n, 2, 3) * args.step
        self._tb[:] = torch.max(torch.min(self._tb, self._rest[0] + 0.35), self._rest[0] - 0.35)
        grip = (self._act[:, 6:7] + 1) / 2
        hand = self._open + grip * (self._shut - self._open)
        o = self._env.scene.env_origins
        pose = []
        for k in range(2):
            pose += [mu.quat_apply(pq, self._tb[:, k]) + pp - o, mu.quat_mul(pq, self._rest[1][k].expand(n, 4))]
        self._raw[:] = torch.cat(pose + [hand], 1)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._act[env_ids] = 0.0
        box = self._env.scene["object"]
        if self._box0 is None:
            self._box0 = box.data.root_pose_w.clone()
        self._box0[env_ids] = box.data.root_pose_w[env_ids]  # the reset event has just placed it
        if self._rest is not None:
            self._tb[env_ids] = self._rest[0]


@configclass
class BoxArmActionCfg(BatchIKActionCfg):
    class_type: type = BoxArmAction


class LegsAction(AgileBasedLowerBodyAction):
    """AGILE legs. Standing: no policy numbers, command zero speed. --walk: 3 policy numbers -> vx, vy, yaw rate."""
    SCALE = (0.5, 0.3, 0.5)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.command = torch.tensor([0.0, 0.0, 0.0, 0.72], device=self.device).repeat(self.num_envs, 1)
        self._scale = torch.tensor(self.SCALE, device=self.device)

    @property
    def action_dim(self):
        return 3 if args.walk else 0

    def process_actions(self, actions):
        if args.walk:
            self.command[:, :3] = actions.clamp(-1.0, 1.0) * self._scale
        super().process_actions(self.command)


@configclass
class LegsActionCfg(AgileBasedLowerBodyActionCfg):
    class_type: type = LegsAction


# --- observations, rewards, terminations ---------------------------------------------------------------------------------
def _box_centre(env):  # (N, 3), env frame
    b = env.scene["object"]
    return b.data.root_pos_w - env.scene.env_origins + torch.tensor([0.0, 0.0, SIZE[2] / 2 - BELOW], device=env.device)


def _in_pelvis(env, v_env):  # env-frame points -> pelvis frame
    r = env.scene["robot"]
    return mu.quat_apply_inverse(r.data.root_quat_w, v_env + env.scene.env_origins - r.data.root_pos_w)


def _wrists(env):  # (N, 2, 3), env frame
    r = env.scene["robot"]
    ids = [r.body_names.index("left_wrist_yaw_link"), r.body_names.index("right_wrist_yaw_link")]
    return r.data.body_pos_w[:, ids] - env.scene.env_origins[:, None]


def _grip_points(env):  # where each wrist should be to squeeze the box (box_scripted.py's at(c, -squeeze)), (N, 2, 3)
    c = _box_centre(env)
    y = c[:, 1] - SIZE[1] / 2 - 0.075
    z = c[:, 2] + SIZE[2] / 2 - 0.05
    return torch.stack([torch.stack([c[:, 0] - SIDE + 0.035, y, z], 1), torch.stack([c[:, 0] + SIDE - 0.035, y, z], 1)], 1)


def _goal(env):
    return torch.tensor(env.cfg.box_goal_centre, device=env.device)


def obs_task(env):
    """Box centre, goal, both wrists and both grip points in the pelvis frame; box orientation and velocity."""
    b = env.scene["object"]
    c = _box_centre(env)
    w, g = _wrists(env), _grip_points(env)
    parts = [_in_pelvis(env, c), _in_pelvis(env, _goal(env).expand_as(c)), b.data.root_quat_w, b.data.root_lin_vel_w,
             b.data.root_ang_vel_w]
    parts += [_in_pelvis(env, w[:, k]) for k in range(2)] + [_in_pelvis(env, g[:, k]) - _in_pelvis(env, w[:, k]) for k in range(2)]
    return torch.cat(parts, 1)


def obs_targets(env):  # the wrist targets the policy has set, pelvis frame
    return env.action_manager.get_term("upper_body_ik")._tb.reshape(env.num_envs, 6)


def _grip_dist(env):  # (N,) mean wrist distance to its grip point
    return (_wrists(env) - _grip_points(env)).norm(dim=-1).mean(1)


def r_reach(env, std):
    return 1 - torch.tanh(_grip_dist(env) / std)


def _rise(env):
    return (_box_centre(env)[:, 2] - REST_Z).clamp(0.0, 0.12)


def r_lift(env):  # not over the goal: there lift and lower (weighted alike) cancelled, so lowering gained nothing (stand-v2)
    return _rise(env) / 0.12 * (_grip_dist(env) < 0.08) * ~_over_goal(env)


def r_carry(env, std):
    d = (_box_centre(env)[:, :2] - _goal(env)[:2]).norm(dim=1)
    return (1 - torch.tanh(d / std)) * was_lifted(env) * (_grip_dist(env) < 0.08)  # not gated on height: lowering keeps it


def was_lifted(env):
    """(N,) the box has been 5 cm up, hands near it, at some step of this episode (an untrained policy shoved it onto the
    goal in 7% of episodes: success must come from a carry)."""
    if not hasattr(env, "_box_lifted"):
        env._box_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    t = env.episode_length_buf
    env._box_lifted &= t > SETTLE
    env._box_lifted |= (_rise(env) > 0.05) & (_grip_dist(env) < 0.08) & (t > SETTLE)
    return env._box_lifted


def placed(env, min_x, max_x, min_y, max_y):  # box_cfgs.box_set_down, after a lift
    return box_set_down(env, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y) & was_lifted(env)


def tipped_early(env):  # tipped within 5 steps of being put back: should be ~0 if the settle works
    return box_tipped(env) & (env.episode_length_buf <= SETTLE + 5)


def _over_goal(env):
    return was_lifted(env) & ((_box_centre(env)[:, :2] - _goal(env)[:2]).norm(dim=1) < 0.05)


def r_lower(env):  # over the goal after a lift: the lower the better
    return _over_goal(env) * (1 - _rise(env) / 0.12)


def r_release(env):  # resting on the goal after a lift: open the hands and move the wrists out past the sides
    down = _over_goal(env) & (_rise(env) < 0.015)
    grip = (env.action_manager.get_term("upper_body_ik")._act[:, 6] + 1) / 2
    c, w = _box_centre(env), _wrists(env)
    out = torch.stack([c[:, 0] - w[:, 0, 0], w[:, 1, 0] - c[:, 0]], 1) - SIZE[0] / 2  # each wrist's gap past its side
    return down * ((1 - grip) + (out.clamp(0.0, 0.06) / 0.06).mean(1))


def r_place(env):
    return placed(env, **_goal_region()).float()


def _goal_region():
    gx, gy = (float(v) for v in GOAL.split(","))
    return dict(min_x=gx - 0.08, max_x=gx + 0.08, min_y=gy - 0.08, max_y=gy + 0.08)


def box_tipped(env):  # the box's up axis more than 45 degrees off vertical, after the settle (the box goes back then)
    up = mu.quat_apply(env.scene["object"].data.root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    return (up[:, 2] < 0.707) & (env.episode_length_buf > SETTLE)


def robot_fell(env):
    r = env.scene["robot"]
    return r.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] < 0.5


@configclass
class PolicyObs(ObsGroup):
    task = ObsTerm(func=obs_task)
    targets = ObsTerm(func=obs_targets)
    arm_q = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
        ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint", "waist_.*_joint"])})
    hand_q = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hand_.*"])})
    base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)
    gravity = ObsTerm(func=base_mdp.projected_gravity)
    last = ObsTerm(func=base_mdp.last_action)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Rewards:
    reach = RewTerm(func=r_reach, weight=1.0, params={"std": 0.1})
    lift = RewTerm(func=r_lift, weight=4.0)
    carry = RewTerm(func=r_carry, weight=4.0, params={"std": 0.1})
    lower = RewTerm(func=r_lower, weight=8.0)
    release = RewTerm(func=r_release, weight=4.0)
    place = RewTerm(func=r_place, weight=50.0)
    rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.01)
    fell = RewTerm(func=robot_fell, weight=-20.0)
    tipped = RewTerm(func=box_tipped, weight=-10.0)


@configclass
class Terminations:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    fell = DoneTerm(func=robot_fell)
    tipped = DoneTerm(func=box_tipped)
    tipped_early = DoneTerm(func=tipped_early)
    dropped = DoneTerm(func=base_mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")})
    success = DoneTerm(func=placed, params=_goal_region())


@configclass
class Events:
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    box = EventTerm(func=base_mdp.reset_root_state_uniform, mode="reset", params={
        "pose_range": {"x": (-args.jitter, args.jitter), "y": (-args.jitter, args.jitter)}, "velocity_range": {},
        "asset_cfg": SceneEntityCfg("object")})


@configclass
class BoxRLEnvCfg(BoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        pink = self.actions.upper_body_ik
        self.actions.upper_body_ik = BoxArmActionCfg(asset_name="robot", hand_joint_names=list(pink.hand_joint_names))
        legs = self.actions.lower_body_joint_pos
        self.actions.lower_body_joint_pos = LegsActionCfg(asset_name="robot", joint_names=legs.joint_names,
                                                          policy_output_scale=legs.policy_output_scale,
                                                          obs_group_name=legs.obs_group_name, policy_path=legs.policy_path)
        self.observations.policy = PolicyObs()
        self.rewards = Rewards()
        self.terminations = Terminations()
        self.events = Events()
        self.episode_length_s = args.episode_s
        self.scene.num_envs = args.num_envs
        self.sim.device = "cuda:0"
        self.teleop_devices = None
        self.recorders = None


@configclass
class RunnerCfg(RslRlOnPolicyRunnerCfg):  # as isaac/g1_dexsuite.py's (Dexsuite's)
    num_steps_per_env = 24
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    save_interval = 50
    experiment_name = "box_rl"
    policy = RslRlPpoActorCriticCfg(init_noise_std=0.5, actor_obs_normalization=True, critic_obs_normalization=True,
                                    actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu")
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
                                     num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
                                     gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0)


torch.set_float32_matmul_precision("high")
env = ManagerBasedRLEnv(cfg=BoxRLEnvCfg())
agent = RunnerCfg()
agent.max_iterations = args.iters
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
venv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
runner = OnPolicyRunner(venv, agent.to_dict(), log_dir=str(out), device="cuda:0")
if args.resume:
    print(f"starting from {args.resume}", flush=True)
    runner.load(args.resume, load_optimizer=False)
t0 = time.time()
runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
print(f"TRAIN-DONE {time.time() - t0:.0f} s", flush=True)
sys.stdout.flush()
os._exit(0)
