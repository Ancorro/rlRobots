"""The learned G1 grasp chooser (so101/g1_chooser.py) on camera points, for bench_general.py --robot=g1 --g1-grasp=chooser.

The chooser was trained on what the grasp sampler knew from sim truth (g1_grasps.scene: the object's centre, half sizes
along x, y, z and the heading of its narrow side; g1_grasps.poses: its top and its width along the closing direction;
g1_grasps.scripted: the planner's grasp on ray-cast true points). Here all of it comes from the object's camera points
(so101/see.py Eyes.find), nothing from the simulator:
  centre      middle of the 1-99 percentile range of the points above the support (x, y; CENTRE); z half-way
              between the support and the top
  top         97th percentile of their heights (as grasp.plan)
  half sizes  half the 1-99 percentile spread along x and y; half the height above the support
  narrow      heading (5-degree steps) where the points' spread is least, as g1_grasps.scene does with the true shape
              (the principal axis turned 90 degrees, for long things); 0 for round footprints (spread ratio < 1.15),
              as the true-shape rule gives for balls and upright cylinders (all headings tie, the first wins)
  width       1-99 percentile spread of the points along the grasp's closing direction (tilted grasps included); the
              grasp is centred on that spread, as grasp.plan centres the jaws
  scripted    candidate: the benchmark's own grasp.plan result on the same points (heading, height), as g1_grasps.scripted
The pick runs the sampler's motion (above the pre-grasp, pre-grasp, open, approach at a quarter speed, close, lift) with
the hand held at the grasp's orientation (G1Arm.frame) and grasp.pick_at's failure reasons. The place stays
grasp.place_on, with the hand keeping the grasp orientation: the held object hangs as it stood, so no tilt drift; if the
arm cannot reach the release point cleanly, the orientation turns about the vertical in 30-degree steps, least first
(as G1Arm.carry_turn turns the grip heading).
"""
import numpy as np

from so101 import g1
from so101.g1_grasps import Grasp, LIFT, check, frame, near, random_grasp

ROUND = 1.15  # footprint spread ratio below which the narrow side is undefined (set to 0, as in training)
CENTRE = "mid"  # object centre (x, y): "mid" of the points' 1-99 percentile range, or their "median". The median leans
# toward the face the camera sees (box 1.1 cm); ray-cast cam_front points, 6 layouts x 6 classes, lifted: median 15,
# mid 18, true centre / sizes / narrow side 22 of 36 (2026-10-06, runs/scratch/smoke2.py)
CHOOSER = {}  # model path -> g1_chooser.Chooser, one per process


def observe(pts, support):
    """(object description as g1_grasps.scene gives it, the points above the support) from camera points, or (None, None)."""
    body = pts[pts[:, 2] > support + 0.003]
    if len(body) < 5:
        return None, None
    top = float(np.percentile(body[:, 2], 97))
    xy = body[:, :2]
    lo, hi = np.percentile(xy, [1, 99], axis=0)
    c = np.median(xy, 0) if CENTRE == "median" else (lo + hi) / 2
    yaws = np.radians(np.arange(0, 180, 5))
    spread = np.array([np.subtract(*np.percentile(xy @ [np.cos(y), np.sin(y)], [99, 1])) for y in yaws])
    narrow = float(yaws[np.argmin(spread)]) if spread.max() > ROUND * max(spread.min(), 1e-4) else 0.0
    obj = dict(centre=[float(c[0]), float(c[1]), (top + support) / 2], half=[*((hi - lo) / 2).tolist(), (top - support) / 2],
               narrow=narrow, top=top, support=float(support))
    return obj, body


def poses(body, obj, g):
    """g1_grasps.poses from points: pinch-site goals (pre-grasp, grasp), hand orientation, opening and width."""
    from so101.grasp import gap_to_q
    F = frame(g)
    a, c = F[:, 0], F[:, 1]
    lo, hi = np.percentile(body @ c, [1, 99])
    width = float(hi - lo)
    centre = np.array(obj["centre"])
    mid = centre[:2] + c[:2] * ((lo + hi) / 2 - centre @ c)  # centred on the spread along the closing direction
    target = np.array([*mid, g.height * obj["top"]])  # height: share of the top above the table, as in training
    grasp = target + c * (g1.Hand.FIXED_OFFSET - width / 2 - g.clear)  # the index pad `clear` from the object
    opening = float(np.clip(gap_to_q(width + g.clear + g.margin, g1.Hand), 0, 1))
    return grasp - a * g.back, grasp, F, opening, width


def from_plan(arm, p, top):
    """The benchmark's scripted grasp (grasp.plan result p) as a g1_grasps.Grasp, as g1_grasps.scripted makes it."""
    _, heading, _ = arm.choose(np.array([*p.xy, p.z]), [p.yaw, p.yaw + np.pi])
    return Grasp(tilt=0.0, azimuth=0.0, heading=float(heading), height=float(p.z / top), clear=g1.Hand.FIXED_CLEAR,
                 margin=g1.Hand.MOVING_CLEAR, back=0.04)


def choose(arm, pts, support, p, seed, model="runs/grasps/chooser-v3.pt", n=256):
    """The chooser's grasp for camera points `pts` (support height, the benchmark's plan result p as the scripted
    candidate): g1_chooser.Chooser.choose's recipe (n/2 wide-prior draws, the scripted grasp and n/2 draws around it,
    ranked by predicted success; the best of the top 80 the arm reaches cleanly). Returns a dict for `pick`, or None
    (bad points, or none of the 80 reachable: the caller falls back to the scripted pick)."""
    import torch
    from dataclasses import asdict
    from so101.g1_chooser import Chooser, features
    if model not in CHOOSER:
        CHOOSER[model] = Chooser(model, n=n)
    ch = CHOOSER[model]
    obj, body = observe(pts, support)
    if obj is None:
        return None
    rng = np.random.default_rng(seed)
    g0 = from_plan(arm, p, obj["top"])
    cands = [random_grasp(rng, obj["narrow"]) for _ in range(ch.n // 2)] + [g0] + [near(rng, g0) for _ in range(ch.n // 2)]
    X = torch.tensor(np.array([features(obj, asdict(g)) for g in cands]))
    with torch.no_grad():
        pr = torch.sigmoid(ch.model((X - ch.mu) / ch.sd).squeeze(1)).numpy()
    for rank, i in enumerate(np.argsort(-pr)[:80]):
        pre, grasp, F, opening, width = poses(body, obj, cands[i])
        posture = check(arm, [pre, grasp], F)[0]
        if posture is not None:
            g = cands[i]
            return dict(g=g, pre=pre, grasp=grasp, F=F, opening=opening, posture=posture, centre=np.array(obj["centre"]),
                        info=dict(predicted=round(float(pr[i]), 3), rank=rank, scripted=bool(i == ch.n // 2),
                                  scripted_predicted=round(float(pr[ch.n // 2]), 3), width=round(width, 4),
                                  features=np.round(X[i].numpy(), 5).tolist(),  # the network input, from the points
                                  obs=dict(centre=np.round(obj["centre"], 4).tolist(), half=np.round(obj["half"], 4).tolist(),
                                           narrow=round(obj["narrow"], 3)),
                                  **{k: round(float(v), 4) for k, v in asdict(g).items()}))
    return None


def reset(arm):
    """Back to the benchmark's top-down grip (before each attempt): no held orientation, no place overrides."""
    arm.frame = None
    for k in ("jaw_axis", "carry_turn"):
        arm.__dict__.pop(k, None)


def pick(arm, c, hover):
    """Generator: the chosen grasp c (from `choose`) and lift to `hover`. Returns (ok, reason) like grasp.pick_at."""
    from so101.skills import SPEED, move_to, set_grip
    if arm.jaw_closed():
        return False, "already holding something"
    arm.frame, arm.posture, arm.roll = c["F"], c["posture"], None
    arm.open = c["opening"]
    pre, grasp = c["pre"], c["grasp"]
    arm.held_offset = (grasp[:2] - c["centre"][:2], float(np.arctan2(c["centre"][1], c["centre"][0])))
    yield from move_to(arm, [*pre[:2], max(pre[2], hover)])
    yield from move_to(arm, pre)
    yield from set_grip(arm, arm.open)
    yield from move_to(arm, grasp, speed=SPEED / 4)
    yield from set_grip(arm, arm.grip_closed)
    if not arm.holding(None):
        yield from set_grip(arm, arm.open)
        yield from move_to(arm, pre)  # back out along the approach before rising
        yield from move_to(arm, [*arm.goal[:2], hover])
        arm.frame = None
        return False, "jaw closed on nothing"
    yield from move_to(arm, [*grasp[:2], max(hover, grasp[2] + LIFT)])
    if not arm.holding(None):
        yield from set_grip(arm, arm.open)
        arm.frame = None
        return False, "dropped while lifting"
    return True, "holding"


def place(arm, xy, surface, hang, hover):
    """Generator: grasp.place_on with the hand kept at the grasp orientation (see the module notes)."""
    from so101.grasp import place_on
    down = np.array([0.0, 0.0, -1.0])

    def turn(goal, heading):
        F = arm.frame
        for t in sorted(np.radians(np.arange(-150, 181, 30)), key=abs):
            ct, st = np.cos(t), np.sin(t)
            Ft = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]]) @ F
            posture = check(arm, [np.asarray(goal, float)], Ft)[0]
            if posture is not None:
                arm.frame, arm.posture = Ft, posture
                return float(t)
        return 0.0  # none clean: keep the grasp orientation (best effort, as G1Arm.carry_turn)
    arm.jaw_axis = lambda goal: down  # the object keeps the orientation it stood in: it hangs straight below as picked
    arm.carry_turn = turn
    out = yield from place_on(arm, xy, surface, hang, hover)
    reset(arm)
    return out


def touch_sensing(arm):
    """Opt-in grasp detector for the G1 (bench_general.py --g1-hold=touch), for either grasp: holding = hand told to
    close and both the thumb tip link and the index tip link touching something that is neither the robot nor fixed to
    the world (table, shelf). A stand-in for the Dex3-1's fingertip tactile pads, read from the simulator's contacts.
    Why: the default rule (skills.Arm.holding, thumb_1 stopped 0.06 rad short of closed) cannot see small things: a
    1.5-2 cm box or cylinder held between the tips leaves the closed hand's joints where they are with nothing in it
    (2026-10-06, runs/scratch/debug_hold.py); and the ramped close can end at 5.6e-17 instead of 0, which
    Arm.jaw_closed's == reads as open (about 14 % of planner openings)."""
    m = arm.model
    links = [m.body(f"right_hand_{n}_link").id for n in ("thumb_2", "index_1")]
    own = arm.moving | arm.rest
    other = ~own & ~arm.fixed

    def holding(name):
        if abs(arm.grip - arm.grip_closed) > 1e-6:
            return False
        d = arm.data
        b1, b2 = m.geom_bodyid[d.contact.geom1[:d.ncon]], m.geom_bodyid[d.contact.geom2[:d.ncon]]
        return all(bool(np.any(((b1 == k) & other[b2]) | ((b2 == k) & other[b1]))) for k in links)
    arm.holding = holding
