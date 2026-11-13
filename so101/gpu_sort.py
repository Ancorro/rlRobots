"""GPU batch version of the sort task (so101/skill_env.py): N copies of the scene on MuJoCo Warp.

Same task, questions, rewards and scripted skills as SO101Sort, but every copy ("world") runs
at once on the GPU:
- physics: mujoco_warp.step on all worlds;
- IK (inverse kinematics): batched damped least squares with finite differences, using
  mujoco_warp.kinematics on a scratch batch of 6 x N worlds (each world + 5 joint nudges);
- skills: pick / place rewritten as step-by-step programs (phases) applied to all worlds by mask.
Decisions happen in rounds: every unfinished world gets one skill, the batch runs until all
skills finish, then the decision model answers for every world at once.
Layouts use the same seeds and sampler as SO101Sort; physics is float32, so results differ slightly.
"""
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from so101.arm import GRIP_CLOSED, GRIP_OPEN
from so101.envs import ARM_JOINTS
from so101.skill_env import (CUBES, GOAL_TEXT, MAX_SKILLS, OBJECTS, QUESTIONS, SCENE, SKILLS, TRAY_HALF, TRAYS,
                             SO101Sort)
from so101.skills import CONTROL_STEPS, GRIP_TICKS, HOVER_Z, SPEED

PICK, PLACE, DONE = (SKILLS.index(s) for s in ("pick", "place", "done"))
SETTLE = 5  # ticks to let the servos settle after a move (as skills.move_to)
IK_ITERS, FD = 6, 1e-4  # IK iterations per tick (warm-started), finite-difference step (float32)
REASONS = ["holding {obj}", "already holding something; place it first", "missed {obj}",
           "dropped {obj} while lifting", "not holding anything to place", "dropped {obj} while carrying",
           "released {obj} over {tray}", "pick needs a cube", "place needs a tray", "stopped"]
# Skill programs: phase ids. Pick 1-8, place 11-15, 0 = no skill running.
P_START, P_ABOVE, P_SLOW, P_CLOSE, P_CHECK, P_MISS_UP, P_LIFTED, P_DROP_END = 1, 2, 3, 4, 5, 6, 7, 8
Q_START, Q_OVER, Q_LOWERED, Q_OPENED, Q_END = 11, 12, 13, 14, 15


class SortBatch:
    questions = QUESTIONS

    def __init__(self, n, drop_prob=0.0, nconmax=64, njmax=512):
        self.n, self.drop_prob = n, drop_prob
        self.mjm = mujoco.MjModel.from_xml_path(str(SCENE))
        self.mjm.opt.iterations, self.mjm.opt.ls_iterations = 30, 50  # float32 needs more than the CPU 10 / 20
        home = SO101Sort()
        home.reset(seed=0)  # the CPU env's settled start pose, reused for every world
        self.home_qpos = torch.tensor(home.data.qpos[:ARM_JOINTS + 1], device="cuda")
        self.home_ctrl = torch.tensor(home.data.ctrl, device="cuda", dtype=torch.float32)
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.put_data(self.mjm, home.data, nworld=n, nconmax=nconmax, njmax=njmax)
        self.s = mjw.put_data(self.mjm, home.data, nworld=6 * n, nconmax=16, njmax=64)  # IK scratch (kinematics only)
        t = wp.to_torch
        self.qpos, self.qvel, self.ctrl = t(self.d.qpos), t(self.d.qvel), t(self.d.ctrl)
        self.mocap, self.xpos, self.site = t(self.d.mocap_pos), t(self.d.xpos), t(self.d.site_xpos)
        self.warm = t(self.d.qacc_warmstart)
        self.s_qpos, self.s_xpos, self.s_site = t(self.s.qpos), t(self.s.xpos), t(self.s.site_xpos)
        self.tip, self.wrist = self.mjm.site("gripperframe").id, self.mjm.body("gripper").id
        self.body = [self.mjm.body(o).id for o in OBJECTS]
        self.cube_adr = [self.mjm.jnt_qposadr[self.mjm.body(c).jntadr[0]] for c in CUBES]
        self.mocap_id = [self.mjm.body_mocapid[self.mjm.body(t).id] for t in TRAYS]
        lo, hi = self.mjm.actuator_ctrlrange.T
        self.lo = torch.tensor(lo[:ARM_JOINTS], device="cuda", dtype=torch.float32)
        self.hi = torch.tensor(hi[:ARM_JOINTS], device="cuda", dtype=torch.float32)
        z = lambda *s, dt=torch.float32: torch.zeros(n, *s, device="cuda", dtype=dt)  # noqa: E731
        self.goal, self.seg_a, self.seg_b, self.q = z(3), z(3), z(3), z(ARM_JOINTS)
        self.grip, self.seg_n, self.seg_t, self.hold = z(), z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
        self.phase, self.obj, self.tray = z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
        self.ok, self.reason, self.tick, self.slip_at = z(dt=torch.bool), z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
        with wp.ScopedCapture() as cap:  # one CUDA graph for a control tick's physics steps
            for _ in range(CONTROL_STEPS):
                mjw.step(self.m, self.d)
        self.physics = cap.graph
        self.steps = np.zeros(n, int)
        self.done = np.zeros(n, bool)
        self.last = ["none yet"] * n
        self.rng = [np.random.default_rng(0) for _ in range(n)]
        self.on_tick = None  # called every tick, e.g. to mirror world 0 into the viewer

    # --- reading the scene ---------------------------------------------------------------------
    def tip_pos(self):
        return self.site[:, self.tip]

    def pos(self, k):
        """Positions of object k (index into OBJECTS) in every world, (n, 3)."""
        return self.xpos[:, self.body[k]]

    def holding_each(self):
        """(n, 2) bool: is each cube held. Grasp = jaw told to close but blocked, cube at the tip."""
        blocked = (self.grip == GRIP_CLOSED) & (self.qpos[:, ARM_JOINTS] > GRIP_CLOSED + 0.15)
        near = torch.stack([(self.pos(c) - self.tip_pos()).norm(dim=1) < 0.03 for c in range(len(CUBES))], 1)
        return near & blocked[:, None]

    def sorted_each(self, held=None):
        held = self.holding_each() if held is None else held
        out = []
        for c in range(len(CUBES)):
            cp, tp = self.pos(c), self.pos(len(CUBES) + c)
            out.append(((cp[:, :2] - tp[:, :2]).abs() < TRAY_HALF).all(1) & (cp[:, 2] < 0.03) & ~held[:, c])
        return torch.stack(out, 1)

    def state_texts(self, worlds=None):
        """The CPU env's state text for each world (positions copied to the CPU once)."""
        worlds = range(self.n) if worlds is None else worlds
        P = torch.stack([self.pos(k) for k in range(len(OBJECTS))], 1).cpu().numpy()
        held, srt = self.holding_each().cpu().numpy(), self.sorted_each().cpu().numpy()
        texts = []
        for w in worlds:
            def mm(p):
                return f"x={p[0] * 1000:.0f} y={p[1] * 1000:.0f} z={p[2] * 1000:.0f} mm"
            h = next((CUBES[c] for c in range(len(CUBES)) if held[w, c]), None)
            lines = [f"Task: {GOAL_TEXT}", f"Gripper: {'holding the ' + h.replace('_', ' ') if h else 'empty'}."]
            for c, name in enumerate(CUBES):
                where = ("in the gripper" if held[w, c] else
                         f"in the {TRAYS[c].replace('_', ' ')}" if srt[w, c] else
                         next((f"in the {t.replace('_', ' ')}" for k, t in enumerate(TRAYS)
                               if np.all(np.abs(P[w, c, :2] - P[w, len(CUBES) + k, :2]) < TRAY_HALF)
                               and P[w, c, 2] < 0.03), "on the table"))
                lines.append(f"{name.replace('_', ' ')}: {mm(P[w, c])}, {where}.")
            lines += [f"{t.replace('_', ' ')}: {mm(P[w, len(CUBES) + k])}." for k, t in enumerate(TRAYS)]
            lines.append(f"Last skill: {self.last[w]}. Skills used: {self.steps[w]} of {MAX_SKILLS}.")
            texts.append(" ".join(lines))
        return texts

    # --- reset ------------------------------------------------------------------------------------
    def reset(self, seeds, worlds=None):
        """Start new episodes in `worlds` (default: all), layouts sampled as SO101Sort.reset does."""
        worlds = list(range(self.n)) if worlds is None else list(worlds)
        lay = []
        for w, seed in zip(worlds, seeds):
            self.rng[w] = rng = np.random.default_rng(seed)  # same generator gymnasium's seeding makes
            while True:
                xy = np.column_stack([rng.uniform(0.12, 0.27, 4), rng.uniform(-0.16, 0.16, 4)])
                if min(np.linalg.norm(xy[i] - xy[j]) for i in range(4) for j in range(i + 1, 4)) > 0.09:
                    break
            lay.append(xy)
        idx = torch.tensor(worlds, device="cuda")
        xy = torch.tensor(np.array(lay), device="cuda", dtype=torch.float32)
        q = self.qpos[idx]
        q[:, :ARM_JOINTS + 1] = self.home_qpos
        for c, adr in enumerate(self.cube_adr):
            q[:, adr:adr + 2] = xy[:, c]
            q[:, adr + 2] = 0.015
            q[:, adr + 3:adr + 7] = torch.tensor([1.0, 0, 0, 0], device="cuda")
        self.qpos[idx] = q
        mp = self.mocap[idx]
        for k, mid in enumerate(self.mocap_id):
            mp[:, mid, :2], mp[:, mid, 2] = xy[:, len(CUBES) + k], 0.002
        self.mocap[idx] = mp
        self.qvel[idx], self.warm[idx] = 0.0, 0.0
        self.ctrl[idx] = self.home_ctrl
        self.q[idx], self.grip[idx] = self.home_ctrl[:ARM_JOINTS], GRIP_OPEN
        self.goal[idx] = torch.tensor([0.18, 0.0, HOVER_Z], device="cuda")
        self.phase[idx] = 0
        torch.cuda.synchronize()
        mjw.kinematics(self.m, self.d)
        wp.synchronize()
        for w in worlds:
            self.steps[w], self.done[w], self.last[w] = 0, False, "none yet"

    # --- motion -----------------------------------------------------------------------------------
    def _ik(self):
        """One batched IK solve for every world's goal, warm-started from the last solution."""
        n, eye = self.n, torch.eye(ARM_JOINTS, device="cuda")
        w = torch.where(self.goal[:, 2] < 0.05, 0.03, 0.002)[:, None, None]
        down = torch.tensor([0.0, 0.0, -1.0], device="cuda")
        q = self.q
        for _ in range(IK_ITERS):
            Q = q[:, None, :] + FD * torch.cat([torch.zeros(1, ARM_JOINTS, device="cuda"), eye])[None]
            self.s_qpos[:, :ARM_JOINTS] = Q.reshape(-1, ARM_JOINTS)
            torch.cuda.synchronize()
            mjw.kinematics(self.m, self.s)
            wp.synchronize()
            tip = self.s_site[:, self.tip].reshape(n, 6, 3)
            a = tip - self.s_xpos[:, self.wrist].reshape(n, 6, 3)
            r = torch.cat([self.goal[:, None] - tip, w * (down - a / a.norm(dim=2, keepdim=True))], 2)
            J = ((r[:, 1:] - r[:, :1]) / FD).transpose(1, 2)  # (n, 6 residuals, 5 joints)
            g = (J.transpose(1, 2) @ r[:, 0, :, None])[..., 0]  # joints pinned at a limit and pushed past it
            J = J * ~(((q >= self.hi) & (g < 0)) | ((q <= self.lo) & (g > 0)))[:, None, :]  # drop out of the solve
            dq = torch.linalg.solve(J.transpose(1, 2) @ J + 1e-4 * eye, (J.transpose(1, 2) @ r[:, 0, :, None]))[..., 0]
            q = torch.minimum(torch.maximum(q - dq, self.lo), self.hi)
        self.q = q

    def _move(self, m, dest, speed=SPEED):
        """Start a straight-line move to dest for worlds in mask m (as skills.move_to)."""
        self.seg_a[m], self.seg_b[m] = self.goal[m], dest[m]
        dist = (dest[m] - self.goal[m]).norm(dim=1)
        self.seg_n[m] = torch.clamp(torch.ceil(dist / (speed * CONTROL_STEPS * 0.005)), min=1).long()
        self.seg_t[m], self.hold[m] = 0, SETTLE

    def _grip(self, m, value):
        self.grip[m] = value
        self.seg_a[m] = self.seg_b[m] = self.goal[m]
        self.seg_n[m], self.seg_t[m], self.hold[m] = 1, 0, GRIP_TICKS - 1

    def _end(self, m, ok, reason):
        """Finish the skill for worlds in mask m; ok / reason may be scalars or per-world tensors."""
        pick = lambda v: v[m] if torch.is_tensor(v) else v  # noqa: E731
        self.phase[m], self.ok[m], self.reason[m] = 0, pick(ok), pick(reason)

    def _advance(self):
        """Run the skill programs for worlds whose current move or grip has finished."""
        ready = (self.phase > 0) & (self.seg_t >= self.seg_n + self.hold)
        if not ready.any():
            return
        held = self.holding_each()
        rows = torch.arange(self.n, device="cuda")
        obj_pos = torch.stack([self.pos(k) for k in range(len(OBJECTS))], 1)[rows, self.obj]
        tray_pos = torch.stack([self.pos(k) for k in range(len(OBJECTS))], 1)[rows, self.tray]
        holds_obj = held[rows, self.obj.clamp(max=len(CUBES) - 1)] & (self.obj < len(CUBES))
        ph = self.phase.clone()
        at = lambda p: ready & (ph == p)  # noqa: E731
        hover = lambda p: torch.cat([p[:, :2], torch.full_like(p[:, 2:], HOVER_Z)], 1)  # noqa: E731
        # pick
        m = at(P_START)
        self._end(m & (self.grip == GRIP_CLOSED), False, 1)
        m = m & (self.grip != GRIP_CLOSED)
        self._move(m, hover(obj_pos))
        self.phase[m] = P_ABOVE
        m = at(P_ABOVE)
        self._move(m, obj_pos + torch.tensor([0, 0, 0.04], device="cuda"))
        self.phase[m] = P_SLOW
        m = at(P_SLOW)
        self._move(m, self.goal - torch.tensor([0, 0, 0.04], device="cuda"), SPEED / 4)
        self.phase[m] = P_CLOSE
        m = at(P_CLOSE)
        self._grip(m, GRIP_CLOSED)
        self.phase[m] = P_CHECK
        m = at(P_CHECK)
        miss = m & ~holds_obj
        self._grip(miss, GRIP_OPEN)
        self.phase[miss] = P_MISS_UP
        got = m & holds_obj
        self._move(got, hover(self.goal))
        self.phase[got] = P_LIFTED
        m = at(P_MISS_UP)
        self._move(m, hover(self.goal))
        self.phase[m] = 100 + 2  # 100 + reason: end once this move finishes
        m = at(P_LIFTED)
        self._end(m & holds_obj, True, 0)
        drop = m & ~holds_obj
        self._grip(drop, GRIP_OPEN)
        self.phase[drop] = 100 + 3
        # place (self.obj = held cube, self.tray = tray)
        m = at(Q_START)
        self._end(m & ~holds_obj, False, 4)
        m = m & holds_obj
        self._move(m, hover(tray_pos))
        self.phase[m] = Q_OVER
        m = at(Q_OVER)
        drop = m & ~holds_obj
        self._grip(drop, GRIP_OPEN)
        self.phase[drop] = 100 + 5
        m = m & holds_obj
        self._move(m, torch.cat([tray_pos[:, :2], torch.full_like(tray_pos[:, 2:], 0.035)], 1))
        self.phase[m] = Q_LOWERED
        m = at(Q_LOWERED)
        self._grip(m, GRIP_OPEN)
        self.phase[m] = Q_OPENED
        m = at(Q_OPENED)
        self._move(m, hover(self.goal))
        self.phase[m] = 100 + 6
        # deferred endings: phase 100 + reason, after the last move or grip finished
        m = ready & (ph >= 100)
        self._end(m, ph == 106, ph - 100)

    def _tick(self):
        if self.drop_prob:  # simulated slip: the jaw lets go mid-carry (as SO101Sort._run)
            slip = (self.tick == self.slip_at) & self.holding_each().any(1)
            self.grip[slip] = GRIP_OPEN
        self._advance()
        self.seg_t += 1
        frac = (self.seg_t.clamp(max=self.seg_n).float() / self.seg_n.float())[:, None]
        self.goal = self.seg_a + (self.seg_b - self.seg_a) * frac
        self._ik()
        self.ctrl[:, :ARM_JOINTS], self.ctrl[:, ARM_JOINTS] = self.q, self.grip
        torch.cuda.synchronize()
        wp.capture_launch(self.physics)
        wp.synchronize()
        self.tick += 1
        if self.on_tick:
            self.on_tick()

    # --- one decision round -----------------------------------------------------------------------
    def step(self, actions):
        """actions (n, 2) ints for every world (ignored for finished worlds). Runs one skill per
        unfinished world to completion. Returns rewards, done, success, skill_ok (numpy, per world)."""
        actions = np.asarray(actions)
        live = ~self.done
        before = self.sorted_each().sum(1).cpu().numpy()
        held = self.holding_each().cpu().numpy()
        held_cube = np.where(held.any(1), held.argmax(1), -1)
        sk, ob = actions[:, 0], actions[:, 1]
        phase = np.zeros(self.n, int)
        reason = np.full(self.n, -1)
        obj, tray = ob.copy(), np.zeros(self.n, int)
        is_cube, is_tray = ob < len(CUBES), ob >= len(CUBES)
        phase[live & (sk == PICK) & is_cube] = P_START
        reason[live & (sk == PICK) & ~is_cube] = 7
        reason[live & (sk == PLACE) & ~is_tray] = 8
        reason[live & (sk == PLACE) & is_tray & (held_cube < 0)] = 4
        go_place = live & (sk == PLACE) & is_tray & (held_cube >= 0)
        phase[go_place], obj[go_place], tray[go_place] = Q_START, held_cube[go_place], ob[go_place]
        reason[live & (sk == DONE)] = 9
        slip = np.array([r.integers(10, 40) if self.drop_prob and r.random() < self.drop_prob else -1
                         for r in self.rng])
        cuda = lambda x: torch.tensor(x, device="cuda")  # noqa: E731
        self.phase, self.obj, self.tray = cuda(phase), cuda(obj), cuda(tray)
        self.slip_at, self.tick = cuda(slip), torch.zeros_like(self.tick)
        self.seg_t[:], self.seg_n[:], self.hold[:] = 1, 1, 0  # start programs on the first tick
        self.seg_a[:], self.seg_b[:] = self.goal, self.goal
        while bool((self.phase > 0).any()):
            self._tick()
        ok, why = self.ok.cpu().numpy(), self.reason.cpu().numpy()
        n_sorted = self.sorted_each().sum(1).cpu().numpy()
        success = n_sorted == len(CUBES)
        immediate = reason >= 0
        why = np.where(immediate, reason, why)
        ok = np.where(immediate, (reason == 9) & success, ok)
        rewards = n_sorted - before - 0.05 + success * 1.0 - ((sk == DONE) & ~success) * 1.0
        for w in np.flatnonzero(live):
            names = dict(obj=OBJECTS[obj[w]], tray=OBJECTS[tray[w]])
            self.last[w] = (f"{SKILLS[sk[w]]} {OBJECTS[ob[w]]} -> {'done' if ok[w] else 'failed'}: "
                            f"{REASONS[why[w]].format(**names)}")
            self.steps[w] += 1
        finished = live & (success | (sk == DONE) | (self.steps >= MAX_SKILLS))
        self.done |= finished
        return np.where(live, rewards, 0.0), finished, success, ok

    def oracle_actions(self):
        """Rule baseline for every world (as SO101Sort.oracle_action)."""
        held = self.holding_each().cpu().numpy()
        srt = self.sorted_each().cpu().numpy()
        out = np.zeros((self.n, 2), int)
        for w in range(self.n):
            if held[w].any():
                c = int(held[w].argmax())
                out[w] = [PLACE, len(CUBES) + c]
            else:
                todo = [c for c in range(len(CUBES)) if not srt[w, c]]
                out[w] = [PICK, todo[0]] if todo else [DONE, 0]
        return out

    def mirror(self, display, world=0):
        """Copy one world's state into a CPU MjData (for the viewer)."""
        display.qpos[:] = self.qpos[world].cpu().numpy()
        display.mocap_pos[:] = self.mocap[world].cpu().numpy()
        mujoco.mj_forward(self.mjm, display)
