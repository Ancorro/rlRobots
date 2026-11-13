"""Scanned and pooled objects for the walking-G1 scripts: which objects a split holds, each layout's object, the
primitive class that stands in for an object in v8's inputs, the points an object is judged at, and the spawner.

One copy, used by the benchmark (isaac/walk_grasp.py --objects gso|pool) and the trainer (isaac/grab_rl.py), so the
trainer's objects are the benchmark's. Moved here from walk_grasp.py on 2026-11-11 without a change of behaviour.
numpy and the standard library at import; Isaac is imported only inside `hull_cfg` and `spawn_hull`, which run after
the app has started.

An object's frame (hulls.npz, so101/assets/objects/gso_g1.py and pool_g1.py): the root sits bottom_z below the lowest
point and up to 7 cm from the footprint centre (pool), so the pinch closes at `grasp_local` and "on the table" is
judged at `centre_local`, never at the root.
"""
import json
import zlib
from pathlib import Path
from typing import Callable

import numpy as np

OBJ_DIR = Path(__file__).resolve().parent.parent / "so101" / "assets" / "objects"
FINGER_DEPTH = 0.045  # the pinch closes no deeper than this below an object's top (g1_grasp_rl.py)


def entries(objects, split, pool_dir=None):
    """(index entries sorted by name, the hulls.npz that holds their convex parts) of an object set on a split.
    gso: the kept scanned household objects, split by crc32('gso/<name>') % 10 (0 test, 1 val, else train).
    pool: the pool's kept one-hand objects (scanned plus DexGraspNet shapes), split by the index's own field, which
    follows the same hash for the scanned objects."""
    if objects == "pool":
        es = [e for e in json.loads((Path(pool_dir) / "pool_g1_index.json").read_text())["objects"]
              if e["kept"] and e["hands"] == 1 and e["split"] == split]
        return sorted(es, key=lambda e: e["name"]), Path(pool_dir) / "hulls.npz"
    want = {"test": 0, "val": 1}.get(split)
    h = lambda e: zlib.crc32(f"gso/{e['name']}".encode()) % 10  # noqa: E731
    es = [e for e in json.loads((OBJ_DIR / "gso_g1_index.json").read_text())["objects"]
          if e["kept"] and (h(e) == want if want is not None else h(e) >= 2)]
    return sorted(es, key=lambda e: e["name"]), OBJ_DIR / "gso_g1" / "hulls.npz"


def spec_of(e, rng, shape):
    """One layout's object from its index entry. rng draws the friction: the first draw from a layout's seed."""
    h = e["height_m"]
    return dict(shape=shape, type="hull", name=e["name"], entry=e, mass=e["mass_kg"], friction=float(rng.uniform(0.7, 1.5)),
                foot=0.5 * float(np.hypot(e["width_m"], e["length_m"])), size=[e["width_m"] / 2, e["length_m"] / 2, h / 2],
                z=-e["bottom_z_m"], quat_axis=None, rgba=[0.8, 0.6, 0.3, 1.0],
                # the pinch: mid-height, but no deeper than FINGER_DEPTH below the top; above the root
                pinch_dz=float(max(h / 2, h - FINGER_DEPTH) + e["bottom_z_m"]))


def layout_of(e, seed, shape):
    """One layout from its seed: the object (friction drawn first), d m past the table edge, y m to the robot's left,
    heading rad."""
    from so101.objects import layout
    rng = np.random.default_rng(seed)
    spec = spec_of(e, rng, shape)
    x, y, a = layout(rng, [spec])[0]
    return dict(shape=shape, seed=seed, spec=spec, d=float(x) - 0.03, y=float(min(y, 0.04)), heading=float(a))


def layouts(es, n, seed_base, shape):
    """The benchmark's layouts of an object set: n poses per object, seeds seed_base + 100 * object + pose (objects in
    name order)."""
    return [layout_of(e, seed_base + 100 * j + i, shape) for j, e in enumerate(es) for i in range(n)]


def stand_in(e):
    """The primitive class and half sizes v8 is given for a scanned object (g1_grasp_rl.py fit_household's rule)."""
    w, ln, hh, fill = e["width_m"], e["length_m"], e["height_m"], e["fill"]
    if fill > 0.9:
        return "box", [w / 2, ln / 2, hh / 2]
    if w / ln > 0.8:
        return ("ball", [(w + ln + hh) / 6, 0, 0]) if 0.75 < hh / w < 1.33 else ("cylinder", [(w + ln) / 4, hh / 2, 0])
    if 0.7 < hh / w < 1.3:
        return "lying cylinder", [(w + hh) / 4, ln / 2, 0]
    return "box", [w / 2, ln / 2, hh / 2]


def grasp_local(spec):
    """In the object's own frame: the point the pinch closes on (primitives: the root, their centre)."""
    return [*spec["entry"]["centre_xy_m"], spec["pinch_dz"]] if spec["type"] == "hull" else [0.0, 0.0, 0.0]


def centre_local(spec):
    """In the object's own frame: its centre (primitives: the root)."""
    if spec["type"] != "hull":
        return [0.0, 0.0, 0.0]
    e = spec["entry"]
    return [*e["centre_xy_m"], e["bottom_z_m"] + e["height_m"] / 2]


def root_xy(spec, x, y, heading):
    """Where the root goes so that the object's footprint centre lies at (x, y) when it is turned by heading."""
    if spec["type"] != "hull":
        return x, y
    cx, cy = spec["entry"]["centre_xy_m"]
    ca, sa = np.cos(heading), np.sin(heading)
    return x - (ca * cx - sa * cy), y - (sa * cx + ca * cy)


_HULLS = {}  # hulls.npz files, loaded when the first object from each is spawned
_CFG = None


def spawn_hull(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A scanned object from its convex parts (hulls.npz), as g1_grasp_rl.spawn_household: one Xform, a mesh per
    part with a convex-hull collider, rigid body, mass, friction, contact reports. MultiAssetSpawnerCfg calls this for
    its prototype and copies it into every env. **kwargs: Isaac Lab 2.3.2's wrapper also passes clone_in_fabric and
    replicate_physics (both --objects=gso runs of 2026-10-28 stopped here with a TypeError before any robot moved)."""
    import isaaclab.sim as sim_utils
    import omni.usd
    from pxr import UsdGeom, UsdPhysics, Vt
    stage = omni.usd.get_context().get_stage()
    if cfg.hull_file not in _HULLS:
        _HULLS[cfg.hull_file] = np.load(cfg.hull_file)
    hulls = _HULLS[cfg.hull_file]
    sim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    V, F, fs = hulls[f"{cfg.name}/v"], hulls[f"{cfg.name}/f"], hulls[f"{cfg.name}/fstart"]
    for k in range(len(fs) - 1):
        f = F[fs[k]:fs[k + 1]]
        ids, f2 = np.unique(f, return_inverse=True)
        m = UsdGeom.Mesh.Define(stage, f"{prim_path}/part_{k}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(V[ids])))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f2.reshape(-1).astype(np.int32)))
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr().Set("convexHull")
        sim_utils.define_collision_properties(f"{prim_path}/part_{k}", cfg.collision_props)
    sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props)
    sim_utils.define_mass_properties(prim_path, cfg.mass_props)
    sim_utils.spawn_rigid_body_material(f"{prim_path}/material", cfg.physics_material)
    sim_utils.bind_physics_material(prim_path, f"{prim_path}/material")
    if cfg.activate_contact_sensors:
        sim_utils.activate_contact_sensors(prim_path)
    return stage.GetPrimAtPath(prim_path)


def hull_cfg(name, hull_file, **common):
    """The spawner config of one object. The config class is built on first use: Isaac's modules import only after the
    app has started."""
    global _CFG
    if _CFG is None:
        import isaaclab.sim as sim_utils
        from isaaclab.utils import configclass

        @configclass
        class HullCfg(sim_utils.SpawnerCfg):
            func: Callable = spawn_hull
            name: str = ""
            hull_file: str = ""
            rigid_props: sim_utils.RigidBodyPropertiesCfg | None = None
            mass_props: sim_utils.MassPropertiesCfg | None = None
            collision_props: sim_utils.CollisionPropertiesCfg | None = None
            physics_material: sim_utils.RigidBodyMaterialCfg | None = None
            activate_contact_sensors: bool = True
            deformable_props: object | None = None  # MultiAssetSpawnerCfg reads every override attribute

        _CFG = HullCfg
    return _CFG(name=name, hull_file=str(hull_file), **common)
