"""General grasping from camera points: plan a top-down grasp for any object, then pick and place it.

`plan(obj, scene, support)` takes the object's 3D points (so101/see.py), all points above the table
(to keep the jaws off neighbours) and the height of the surface it rests on. It tries jaw headings
every 5 degrees and keeps the one where the object is narrowest between the jaws and the jaws'
path is clear; the grasp height is 2 cm below the top (half-way down for short objects).
Nothing is object-specific: no shape names, sizes or colours.

SO-101 jaw geometry (measured 2026-10-03): the fixed jaw sits 2 cm from the tip site, the moving one
opens away from it; gap 0.5 cm closed, 6 cm at 0.6 rad, 9.7 cm at 1.2 rad. `pick_at` aims the fixed
jaw just clear of the object, so closing does not shove it far. A grasp counts only from the jaw
position (blocked part-way), as a real servo gripper reports it.
Another robot's hand (so101/g1.py Hand) gives the same constants; `hand=None` means the SO-101's below.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

from so101.skills import SPEED, move_to, set_grip

GAP_Q = np.array([-0.17, -0.01, 0.15, 0.31, 0.47, 0.63, 0.80, 0.96, 1.12, 1.28])  # gripper joint, radians
GAP_M = np.array([0.0056, 0.0151, 0.0270, 0.0391, 0.0510, 0.0627, 0.0739, 0.0847, 0.0950, 0.1046])  # mean jaw gap
FIXED_OFFSET = 0.02  # fixed jaw tip to the tip site, metres
FIXED_CLEAR, MOVING_CLEAR = 0.006, 0.015  # jaw clearance from the object before closing
MAX_GAP = 0.095
FINGER_HALF = 0.012  # half the jaw pad width across the closing direction
HOVER_MIN = 0.12
RELEASE_ABOVE = 0.004  # object bottom this far above the surface when the jaw opens (12 mm: tall things tipped over)
SMOOTH = os.environ.get("GRASP_SMOOTH", "0") == "1"  # ease in/out when lifting and carrying: tested 2026-10-03, 120/200 vs 169/200 without (slower = more slip)
THIN_TALL = True  # grip thin, tall objects near the middle
# stability variants (2026-10-09, bench_general seed 7): neighbour clearance cost and margin, ease the first lift
BLOCK_COST = float(os.environ.get("GRASP_BLOCK_COST", 0.002))  # cost per neighbour point beside the pads' path
BLOCK_MARGIN = float(os.environ.get("GRASP_BLOCK_MARGIN", 0.012))  # free space wanted beyond the open pads, metres
LIFT_EASE = float(os.environ.get("GRASP_LIFT_EASE", 0.02))  # metres lifted at quarter speed right after closing (default 2026-10-09: primitives seed 7 drops 17 -> 6)
Z_FRAC, Z_BELOW_TOP = 0.5, float(os.environ.get("GRASP_Z_BELOW_TOP", 0.035))  # 2 cm below the top: tall things hung far below the jaw and tipped when placed  # grasp height: the higher of (support + Z_FRAC x height) and (top - Z_BELOW_TOP)


def geometry(hand=None):
    """The hand's grasp constants (this module's, read now: the benchmark sets some from its flags)."""
    h = hand or sys.modules[__name__]
    return h.GAP_Q, h.GAP_M, h.FIXED_OFFSET, h.FIXED_CLEAR, h.MOVING_CLEAR, h.MAX_GAP, h.FINGER_HALF, \
        getattr(h, "SIDE_HALF", h.FINGER_HALF), getattr(h, "MIN_Z", 0.006)


def gap_to_q(gap, hand=None):
    q, m = geometry(hand)[:2]
    return float(np.interp(gap, m, q))


@dataclass
class Grasp:
    xy: np.ndarray  # object centre between the jaws (world)
    z: float  # tip height for the grasp
    yaw: float  # heading the jaws close along
    width: float  # object width between the jaws
    gap: float  # jaw opening to use
    top: float
    support: float  # height of the surface under the object
    blocked: int  # neighbour points in the jaws' path (0 is clear)
    radius: float = 0.02  # object footprint radius around xy (for choosing where to put it down)


def support_height(obj, everything):
    """Surface under the object: low percentile of points in a ring 1-3 cm around its footprint."""
    c = np.median(obj[:, :2], 0)
    r = np.percentile(np.linalg.norm(obj[:, :2] - c, axis=1), 95)
    d = np.linalg.norm(everything[:, :2] - c, axis=1)
    ring = everything[(d > r + 0.01) & (d < r + 0.03), 2]
    return float(np.percentile(ring, 20)) if len(ring) > 20 else 0.0


def plan(obj, scene, support, avoid=(), hand=None, check=None):
    """Best top-down grasp for object points `obj` among `scene` points (both world, metres).
    `avoid`: headings that already failed; headings within 30 degrees of them cost extra, so a retry tries another way.
    `hand`: another robot's grasp geometry (None = the SO-101 jaw).
    `check(grasp)`: whether the robot can do a grasp (arm.can_grasp); the cheapest grasp that passes is kept, the
    cheapest of all if none does."""
    _, _, _, FIXED_CLEAR, MOVING_CLEAR, MAX_GAP, FINGER_HALF, SIDE_HALF, MIN_Z = geometry(hand)
    body = obj[obj[:, 2] > support + 0.003]
    if len(body) < 5:  # nothing above the surface it seems to rest on (bad points): no grasp
        return None
    top = float(np.percentile(body[:, 2], 97))
    z = max(top - Z_BELOW_TOP, support + Z_FRAC * (top - support), support + MIN_Z)
    xy = body[:, :2]
    sub = obj[np.random.default_rng(0).choice(len(obj), min(len(obj), 400), replace=False)]
    c0 = np.median(xy, 0)
    near = scene[(np.linalg.norm(scene[:, :2] - c0, axis=1) < 0.12) & (scene[:, 2] > support + 0.003)]
    if len(near):  # neighbours = scene points not on the object
        d = np.min(np.linalg.norm(near[:, None, :] - sub[None, :, :], axis=2), axis=1)
        near = near[d > 0.006]
    cands = []
    for yaw in np.radians(np.arange(0, 180, 5)):
        u, v = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
        a, b = xy @ u, xy @ v
        b_mid = (b.min() + b.max()) / 2
        slab = np.abs(b - b_mid) < FINGER_HALF  # the part the pads will touch
        if slab.sum() < 5:
            continue
        a_lo, a_hi = a[slab].min(), a[slab].max()
        width = a_hi - a_lo
        gap = width + FIXED_CLEAR + MOVING_CLEAR
        if gap > MAX_GAP:
            continue
        centre = u * (a_lo + a_hi) / 2 + v * b_mid
        blocked = 0
        if len(near):  # pads go down on both sides: keep that space free of other things
            na, nb = (near[:, :2] - centre) @ u, (near[:, :2] - centre) @ v
            reach = width / 2 + max(FIXED_CLEAR, MOVING_CLEAR) + BLOCK_MARGIN
            blocked = int(np.sum((np.abs(na) > width / 2) & (np.abs(na) < reach) & (np.abs(nb) < SIDE_HALF + 0.008)))
        cost = width + BLOCK_COST * blocked
        for y in avoid:
            d = abs((yaw - y + np.pi / 2) % np.pi - np.pi / 2)
            cost += 0.03 * max(0.0, 1 - d / np.radians(30))
        radius = float(np.percentile(np.linalg.norm(xy - centre, axis=1), 98))
        gz = z
        if THIN_TALL and top - support > 2.5 * width:  # thin and tall (a card on edge): grip near the middle so closing does not tip it
            gz = max(support + 0.45 * (top - support), support + MIN_Z)
        cands.append((cost, Grasp(centre, gz, float(yaw), float(width), float(gap), top, support, blocked, radius)))
    cands.sort(key=lambda c: c[0])  # stable: equal costs keep heading order
    for _, g in cands if check else ():
        if check(g):
            return g
    return cands[0][1] if cands else None


def aim(arm, g):
    """Tip goal for grasp g: wrist turned to the heading, the fixed jaw FIXED_CLEAR from the object."""
    _, _, FIXED_OFFSET, FIXED_CLEAR = geometry(arm.hand)[:4]
    goal = np.array([*g.xy, g.z])
    arm.turn_jaw(goal, [g.yaw])
    for _ in range(2):
        arm.pose(goal)
        n = arm.jaw_direction()
        goal = np.array([*(g.xy + n * (FIXED_OFFSET - g.width / 2 - FIXED_CLEAR)), g.z])
    return goal


def pick_at(arm, g, hover=HOVER_MIN):
    """Generator: grasp with plan g and lift to `hover`. Returns (ok, reason)."""
    if arm.jaw_closed():
        return False, "already holding something"
    arm.open = gap_to_q(g.gap, arm.hand)
    goal = aim(arm, g)
    arm.held_offset = (goal[:2] - g.xy, np.arctan2(g.xy[1], g.xy[0]))  # tip relative to the object centre, base heading
    yield from move_to(arm, [*goal[:2], hover])
    yield from set_grip(arm, arm.open)
    yield from move_to(arm, [*goal[:2], g.z + 0.04])
    yield from move_to(arm, goal, speed=SPEED / 4)
    yield from set_grip(arm, arm.grip_closed)
    if not arm.holding(None):
        yield from set_grip(arm, arm.open)
        yield from move_to(arm, [*arm.goal[:2], hover])
        arm.roll = None
        return False, "jaw closed on nothing"
    if LIFT_EASE:  # take the weight slowly first, so the pads settle before the fast lift
        yield from move_to(arm, [*arm.goal[:2], arm.goal[2] + LIFT_EASE], speed=SPEED / 4)
    yield from move_to(arm, [*arm.goal[:2], hover], smooth=SMOOTH)
    if not arm.holding(None):
        yield from set_grip(arm, arm.open)
        arm.roll = None
        return False, "dropped while lifting"
    return True, "holding"


def place_on(arm, xy, surface, hang, hover=HOVER_MIN):
    """Generator: carry the held object over `xy`, lower it until its bottom is RELEASE_ABOVE over `surface`, release.
    `hang` = how far the object reaches below the tip (grasp height minus its support)."""
    if not arm.holding(None):
        return False, "not holding anything"
    off, heading = getattr(arm, "held_offset", (np.zeros(2), 0.0))
    release_z = surface + hang + RELEASE_ABOVE
    turn = arm.carry_turn(np.array([*xy, release_z]), heading)  # how far the held offset turns on the way (SO-101: with the base)
    c, s_ = np.cos(turn), np.sin(turn)
    target = np.asarray(xy, float)
    xy = target + np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])  # object centre lands on xy
    for _ in range(3):  # a tilted jaw puts the object's bottom off to one side of the tip: aim the bottom instead
        a = arm.jaw_axis(np.array([*xy, release_z]))
        drift = a[:2] / max(-a[2], 0.3) * hang  # where the bottom ends up, relative to straight below the tip
        xy = target + np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]]) - drift
    yield from move_to(arm, [*xy, hover], smooth=SMOOTH)
    if not arm.holding(None):
        yield from set_grip(arm, arm.open)
        arm.roll = None
        return False, "dropped while carrying"
    yield from move_to(arm, [*xy, release_z], speed=SPEED / 2, smooth=SMOOTH)
    yield from set_grip(arm, arm.open)
    yield from move_to(arm, [*xy, hover])
    arm.roll = None
    return True, "released"
