"""Random graspable objects for the general pick-and-place benchmark.

Each object is a primitive with random shape, size, colour, mass and friction, sized for the SO-101
gripper (jaw gap up to ~9.5 cm at the opening we use; narrowest side 1.2-5 cm). `random_objects(rng, n)`
gives specs; `build(specs, layout)` adds them to scene_general.xml with free joints and compiles.
Names are unique "<colour> <noun>" phrases, which is how the task refers to them (and how the
detector is asked for them).
"""
import json
from pathlib import Path

import mujoco
import numpy as np

SCENE = Path(__file__).parent / "assets" / "scene_general.xml"
OBJECTS_DIR = Path(__file__).parent / "assets" / "objects"  # scanned objects: fetch_gso.py
COLOURS = {"red": (0.85, 0.12, 0.1), "green": (0.15, 0.7, 0.2), "blue": (0.15, 0.3, 0.9),
           "yellow": (0.95, 0.85, 0.1), "orange": (1.0, 0.5, 0.05), "purple": (0.55, 0.2, 0.75),
           "pink": (1.0, 0.5, 0.7), "white": (0.92, 0.92, 0.9), "black": (0.08, 0.08, 0.08),
           "brown": (0.5, 0.3, 0.12), "cyan": (0.1, 0.8, 0.85)}
SHAPES = ("box", "cylinder", "lying cylinder", "ball", "capsule", "egg")
NOUN = {"box": "block", "cylinder": "cylinder", "lying cylinder": "cylinder", "ball": "ball",
        "capsule": "capsule", "egg": "egg"}


def u(rng, a, b):
    return float(rng.uniform(a, b))


def random_object(rng, shape=None):
    """One object spec: shape, MuJoCo geom type and half-sizes, rest height of its centre, footprint radius."""
    shape = shape or SHAPES[rng.integers(len(SHAPES))]
    if shape == "box":
        narrow = u(rng, 0.006, 0.025)
        s = [narrow, u(rng, narrow, min(0.05, 2.5 * narrow + 0.01)), u(rng, 0.008, 0.03)]
        g = dict(type="box", size=s, z=s[2], quat_axis=None)
        foot = np.hypot(s[0], s[1])
    elif shape == "cylinder":
        r = u(rng, 0.008, 0.025)
        h = u(rng, 0.01, min(0.04, 3.2 * r))
        g = dict(type="cylinder", size=[r, h, 0], z=h, quat_axis=None)
        foot = r
    elif shape == "lying cylinder":
        r = u(rng, 0.007, 0.02)
        h = u(rng, 0.02, 0.05)
        g = dict(type="cylinder", size=[r, h, 0], z=r, quat_axis="y")
        foot = h
    elif shape == "ball":
        r = u(rng, 0.01, 0.024)
        g = dict(type="sphere", size=[r, 0, 0], z=r, quat_axis=None)
        foot = r
    elif shape == "capsule":
        r = u(rng, 0.007, 0.018)
        h = u(rng, 0.012, 0.035)
        g = dict(type="capsule", size=[r, h, 0], z=r, quat_axis="y")
        foot = h + r
    else:  # egg: ellipsoid lying on its side
        a = u(rng, 0.008, 0.022)
        s = [u(rng, a, 2 * a), a, u(rng, 0.7 * a, a)]
        g = dict(type="ellipsoid", size=s, z=s[2], quat_axis=None)
        foot = s[0]
    vol = {"box": 8 * np.prod(g["size"]), "cylinder": 2 * np.pi * g["size"][0] ** 2 * g["size"][1],
           "sphere": 4 / 3 * np.pi * g["size"][0] ** 3,
           "capsule": np.pi * g["size"][0] ** 2 * (2 * g["size"][1] + 4 / 3 * g["size"][0]),
           "ellipsoid": 4 / 3 * np.pi * np.prod(g["size"])}[g["type"]]
    mass = float(np.clip(vol * u(rng, 300, 900), 0.008, 0.2))
    return dict(shape=shape, noun=NOUN[shape], mass=mass, friction=u(rng, 0.7, 1.5), foot=float(foot), **g)


def random_objects(rng, n, shapes=None):
    specs, used = [], set()
    while len(specs) < n:
        spec = random_object(rng, None if shapes is None else shapes[len(specs) % len(shapes)])
        colour = list(COLOURS)[rng.integers(len(COLOURS))]
        if (colour, spec["noun"]) in used:
            continue
        used.add((colour, spec["noun"]))
        spec["colour"] = colour
        spec["name"] = f"{colour} {spec['noun']}"
        rgb = np.clip(np.array(COLOURS[colour]) + rng.uniform(-0.05, 0.05, 3), 0, 1)
        spec["rgba"] = [*rgb, 1.0]
        specs.append(spec)
    return specs


def layout(rng, specs, tries=2000, max_r=0.23):
    """Positions and headings in front of the arm, clear of each other (room for the jaws), the trays and the shelf."""
    for _ in range(tries):
        pts = []
        for s in specs:
            for _ in range(200):
                x, y = u(rng, 0.12, max_r - 0.02), u(rng, -0.14, 0.14)
                r = np.hypot(x, y)
                if r < 0.13 + 0.5 * s["foot"] or r > max_r - 0.25 * s["foot"] or abs(y) > 0.135 - 0.5 * s["foot"]:
                    continue
                if all(np.hypot(x - px, y - py) > s["foot"] + ps["foot"] + 0.035 for (px, py, _), ps in zip(pts, specs)):
                    pts.append((x, y, u(rng, -np.pi, np.pi)))
                    break
            else:
                break
        if len(pts) == len(specs):
            return pts
    raise RuntimeError("no layout found")


def scanned_objects(rng, n, strict=True):
    """n Google Scanned Objects (CC-BY 4.0; so101/assets/objects/gso_index.json) with distinct phrases.
    Objects that look alike share a phrase (e.g. the N64 cartridges), so they never appear together."""
    pool = [o for o in json.loads((OBJECTS_DIR / "gso_index.json").read_text()) if o["fits_strict"] or not strict]
    specs, used = [], set()
    for i in rng.permutation(len(pool)):
        o = pool[i]
        if any(o["phrase"] in u or u in o["phrase"] for u in used):  # same or overlapping names: could not be told apart
            continue
        used.add(o["phrase"])
        specs.append(dict(shape="scanned", noun=o["phrase"], name=o["phrase"], scanned=o["name"],
                          xml=str(OBJECTS_DIR / o["xml"]), foot=float(o["longest_horiz_m"]) / 2, z=0.0, quat_axis=None))
        if len(specs) == n:
            return specs
    raise ValueError("not enough distinct scanned objects")


def body_name(spec):
    if spec["shape"] == "scanned":
        return f"{spec['scanned']}_object"
    return spec["name"].replace(" ", "_")


def build(specs, poses, scene=SCENE, prepare=None):
    """Compile scene_general.xml (or `scene`, edited by `prepare(spec)` first: so101/g1.py) plus the objects at `poses`
    [(x, y, heading)]."""
    spec = mujoco.MjSpec.from_file(str(scene))
    if prepare:
        prepare(spec)
    for s, (x, y, a) in zip(specs, poses):
        q = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)])  # heading about z
        if s["quat_axis"] == "y":  # lying: long axis (local z) along the heading
            lie = np.array([np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0])
            q2 = np.zeros(4)
            mujoco.mju_mulQuat(q2, q, lie)
            q = q2
        if s["shape"] == "scanned":  # attach the object's own MJCF under a prefix (its names clash otherwise)
            child = mujoco.MjSpec.from_file(s["xml"])
            frame = spec.worldbody.add_frame(pos=[x, y, 0.001], quat=q)
            frame.attach_body(child.body("object"), f"{s['scanned']}_", "")
            continue
        b = spec.worldbody.add_body(name=body_name(s), pos=[x, y, s["z"] + 0.001], quat=q)
        b.add_freejoint()
        gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                 "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
                 "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID}[s["type"]]
        b.add_geom(name=body_name(s), type=gtype, size=s["size"], rgba=s["rgba"], mass=s["mass"],
                   friction=[s["friction"], 0.03, 0.003], condim=6, solref=[0.01, 1])
    # no-slip solver passes: without them soft-contact friction lets held objects creep out of the jaws
    # (2026-10-03: 171 -> 191 of 200 placed). Real friction does not creep like that.
    spec.option.noslip_iterations = 3
    return spec.compile()
