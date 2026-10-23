"""One object pool for G1 + Dex3 grasp training and testing (owner 2026-10-09: many objects, two hands for large ones,
"train test split NOT by set": one random split over the pooled objects, whatever dataset they came from).

Sources (all non-commercial research use at most; licences below):
  gso   Google Scanned Objects, the 114 kept by gso_g1.py (one hand; CC-BY 4.0). Real scale, scanned resting pose.
  sns   ShapeNetSem (Savva, Chang, Hanrahan 2015; non-commercial research, owner's Hugging Face access): metadata.csv
        unit -> metres, up -> +z; tabletop categories only (CATEGORIES). Real-world size estimates (noisy).
        Convex parts by CoACD (threshold 0.08, at most 24 parts).
  dgn   DexGraspNet objects (Wang et al. 2023; CC BY-NC 4.0): 5,355 meshes, normalised scale, with their CoACD parts.
        No real size: each object gets one scale drawn from its name's hash: 80 % one-hand (narrowest width 1.5-4.5 cm),
        20 % two-hand (longest side 25-45 cm).
Every object rests in its most probable stable pose (trimesh), except gso (scanned pose). Kept if it passes either rule:
  one hand  (gso_g1.py): narrowest width 1-5 cm, height 1.5-15 cm, longest <= 20 cm, mass <= 0.5 kg
  two hands: longest side 20-50 cm, narrowest width >= 8 cm, height 4-40 cm, mass <= 3 kg
Mass = convex-part volume x density drawn per object (200-600 kg/m3 one hand, 100-300 two hands; hollow to solid).
Split, fixed in advance by crc32("<source>/<name>") % 10: 0 test, 1 validation, 2-9 train.

  python so101/assets/objects/pool_g1.py --sns=ShapeNetSem.zip --dgn=meshdata/ --gso=gso_g1/hulls.npz --out=DIR --workers=32
  -> DIR/pool_g1_index.json, DIR/hulls.npz (same layout as gso_g1: <id>/v, <id>/f, <id>/fstart)
"""
import argparse
import csv
import io
import json
import zipfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from gso_g1 import footprint, hull2d, cross2, volume_centroid

ONE = dict(width=(0.01, 0.05), height=(0.015, 0.15), longest=0.20, mass=0.5, density=(200.0, 600.0))
TWO = dict(longest=(0.20, 0.50), width_min=0.08, height=(0.04, 0.40), mass=3.0, density=(100.0, 300.0))
CATEGORIES = {  # ShapeNetSem tabletop categories (one and two hands)
    "FoodItem", "CellPhone", "DrinkingUtensil", "ToyFigure", "Bottle", "Book", "Books", "Pencil", "Pen", "VideoGameController",
    "Camera", "Ipod", "Mug", "Cup", "Candle", "SodaCan", "Bowl", "Vase", "USBStick", "Calculator", "Stapler", "ToiletPaper",
    "Shoes", "CerealBox", "Headphones", "TissueBox", "WineBottle", "BeerBottle", "DrinkBottle", "ComputerMouse", "NintendoDS",
    "Battery", "Hammer", "Ball", "PillBottle", "Plate", "FoodPlate", "MilkCarton", "Fruit", "Banana", "Eraser", "Donut",
    "Cake", "Sandwich", "Cap", "Notepad", "RubiksCube", "Fork", "Spoon", "Knife", "FruitBowl", "TapeMeasure", "WineGlass",
    "DiscCase", "MediaDiscs", "Purse", "Laptop", "Keyboard", "TableClock", "Telephone", "VideoGameConsole", "Toy",
    "Box", "Basket", "Bag", "Backpack", "Pillow", "Speaker", "Toaster", "Tray", "Pot", "Pan", "Bucket", "Teapot",
    "Kettle", "Jar", "Can", "Canister", "Helmet", "Briefcase", "PersonalCareProduct", "Soap", "Toothbrush", "Remote",
}


def split_of(key):
    r = zlib.crc32(key.encode()) % 10
    return "test" if r == 0 else "val" if r == 1 else "train"


def rot_to_z(up):
    """Rotation taking unit vector `up` to +z."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def stable(parts):
    """Most probable resting transform of the union of the convex parts (trimesh.poses), as a 4x4."""
    import trimesh
    hull = trimesh.Trimesh(np.concatenate([v for v, _ in parts])).convex_hull
    try:
        T, p = hull.compute_stable_poses(n_samples=1, threshold=0.0)
        return T[int(np.argmax(p))] if len(T) else np.eye(4)
    except Exception:  # noqa: BLE001
        return np.eye(4)


def measure(key, parts, rng, rule_pref=None):
    """Entry for an object given its convex parts (metres, z up, resting): sizes, mass, the rule it passes."""
    V = np.concatenate([v for v, _ in parts])
    vols = [volume_centroid(v, f) for v, f in parts]
    vol = sum(x for x, _ in vols)
    com = sum(x * c for x, c in vols) / vol if vol > 0 else V.mean(0)
    lo, hi = V.min(0), V.max(0)
    w, narrow, length, area, centre = footprint(V[:, :2])
    height = hi[2] - lo[2]
    longest = max(length, height, w)
    u = rng.uniform()
    one = (ONE["width"][0] <= w <= ONE["width"][1] and ONE["height"][0] <= height <= ONE["height"][1] and longest <= ONE["longest"])
    two = (TWO["longest"][0] <= max(length, w) <= TWO["longest"][1] and w >= TWO["width_min"]
           and TWO["height"][0] <= height <= TWO["height"][1])
    hands, why = 0, []
    if one and rule_pref != 2:
        density = ONE["density"][0] + u * (ONE["density"][1] - ONE["density"][0])
        hands = 1 if density * vol <= ONE["mass"] else 0
        why += [] if hands else [f"mass {density * vol:.2f} kg"]
    elif two and rule_pref != 1:
        density = TWO["density"][0] + u * (TWO["density"][1] - TWO["density"][0])
        hands = 2 if density * vol <= TWO["mass"] else 0
        why += [] if hands else [f"mass {density * vol:.2f} kg"]
    else:
        density = 0.0
        why.append(f"size w {w * 100:.1f} l {length * 100:.1f} h {height * 100:.1f} cm")
    base = V[V[:, 2] < lo[2] + 0.003, :2]
    bh = hull2d(base) if len(base) >= 3 else base
    inside = len(bh) >= 3 and all(cross2(bh[(i + 1) % len(bh)] - bh[i], com[:2] - bh[i]) >= 0 for i in range(len(bh)))
    src, name = key.split("/", 1)
    return dict(name=key, source=src, split=split_of(key), hands=hands, kept=hands > 0, why=why, parts=len(parts),
                width_m=round(float(w), 4), length_m=round(float(length), 4), height_m=round(float(height), 4),
                narrow_rad=round(float(narrow), 4), centre_xy_m=[round(float(x), 4) for x in centre],
                bottom_z_m=round(float(lo[2]), 4), fill=round(float(area / max(w * length, 1e-9)), 3),
                volume_cm3=round(vol * 1e6, 1), density=round(density), mass_kg=round(float(density * vol), 4),
                com_m=[round(float(x), 4) for x in com], com_over_base=bool(inside))


def split_obj(text):
    """Parts of a multi-object OBJ ('o'/'g' groups), each (vertices, triangles) with its own vertex indices."""
    V, groups, cur = [], [], None
    for line in text.splitlines():
        if line.startswith("v "):
            V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith(("o ", "g ")):
            cur = []
            groups.append(cur)
        elif line.startswith("f "):
            if cur is None:
                cur = []
                groups.append(cur)
            idx = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            cur += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    V = np.array(V, float)
    out = []
    for g in groups:
        if not g:
            continue
        f = np.array(g, int)
        ids, f2 = np.unique(f, return_inverse=True)
        out.append((V[ids], f2.reshape(-1, 3)))
    return out


def place(parts, R=None, T=None, s=1.0):
    """Scale, rotate (R 3x3, then the 4x4 T), and set the lowest point at z = 0."""
    out = []
    for v, f in parts:
        v = v * s
        if R is not None:
            v = v @ R.T
        if T is not None:
            v = v @ T[:3, :3].T + T[:3, 3]
        out.append((v, f))
    z0 = min(v[:, 2].min() for v, _ in out)
    return [(v - [0, 0, z0], f) for v, f in out]


def do_sns(job):
    key, raw, unit, up = job
    import coacd
    import trimesh
    try:
        m = trimesh.load(io.BytesIO(raw), file_type="obj", force="mesh", process=True)
        v = np.asarray(m.vertices) * unit @ rot_to_z(up).T
        if len(m.faces) < 4 or np.ptp(v, 0).max() > 0.6:
            return None
        coacd.set_log_level("error")
        parts = coacd.run_coacd(coacd.Mesh(v, np.asarray(m.faces)), threshold=0.08, max_convex_hull=24)
        parts = [(np.asarray(a, float), np.asarray(b, int)) for a, b in parts]
        parts = place(parts, T=stable(parts))
        return measure(key, parts, np.random.default_rng(zlib.crc32(key.encode()))), parts
    except Exception as e:  # noqa: BLE001
        return dict(name=key, source="sns", kept=False, why=[f"error {type(e).__name__}"]), None


def do_dgn(path):
    key = f"dgn/{path.parent.parent.name}"
    try:
        parts = split_obj(path.read_text())
        if not parts:
            return None
        parts = place(parts)
        parts = place(parts, T=stable(parts))
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        V = np.concatenate([v for v, _ in parts])
        w, _, length, _, _ = footprint(V[:, :2])
        height = np.ptp(V[:, 2])
        two = rng.uniform() < 0.2
        if two:
            s = rng.uniform(0.25, 0.45) / max(length, w, height)
        else:
            s = rng.uniform(0.015, 0.045) / max(w, 1e-6)
            s = min(s, rng.uniform(0.06, 0.20) / max(length, height, w))  # keep it hand-sized
        parts = place(parts, s=s)
        return measure(key, parts, rng, rule_pref=2 if two else 1), parts
    except Exception as e:  # noqa: BLE001
        return dict(name=key, source="dgn", kept=False, why=[f"error {type(e).__name__}"]), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sns", default="")
    ap.add_argument("--dgn", default="")
    ap.add_argument("--gso", default="")
    ap.add_argument("--gso_index", default=str(Path(__file__).resolve().parent / "gso_g1_index.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="per source, for a quick check")
    ap.add_argument("--dgn_max", type=int, default=0, help="DexGraspNet objects kept, picked by crc32(name) (0 = all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index, arrays = [], {}

    def keep(res):
        if res is None:
            return
        e, parts = res
        index.append(e)
        if e["kept"]:
            arrays[f"{e['name']}/v"] = np.concatenate([v for v, _ in parts]).astype(np.float32)
            off = np.cumsum([0] + [len(v) for v, _ in parts])[:-1]
            arrays[f"{e['name']}/f"] = np.concatenate([f + o for (_, f), o in zip(parts, off)]).astype(np.int32)
            arrays[f"{e['name']}/fstart"] = np.cumsum([0] + [len(f) for _, f in parts]).astype(np.int32)

    if args.gso:  # already measured and kept by gso_g1.py (scanned pose, one hand)
        H = np.load(args.gso)
        for e in json.loads(Path(args.gso_index).read_text())["objects"]:
            if not e["kept"]:
                continue
            V, F, fs = H[f"{e['name']}/v"], H[f"{e['name']}/f"], H[f"{e['name']}/fstart"]
            parts = []
            for k in range(len(fs) - 1):
                f = F[fs[k]:fs[k + 1]]
                ids, f2 = np.unique(f, return_inverse=True)
                parts.append((V[ids].astype(float), f2.reshape(-1, 3)))
            key = f"gso/{e['name']}"
            keep((measure(key, parts, np.random.default_rng(zlib.crc32(key.encode())), rule_pref=1), parts))
        print(f"gso: {sum(e['source'] == 'gso' for e in index)}", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        if args.dgn:
            paths = sorted(Path(args.dgn).glob("*/coacd/decomposed.obj"))
            if args.dgn_max:  # fixed subset, independent of the split (which uses crc32 % 10)
                paths = sorted(sorted(paths, key=lambda q: zlib.crc32(f"pick/{q.parent.parent.name}".encode()))[: args.dgn_max])
            paths = paths[: args.limit or None]
            for res in ex.map(do_dgn, paths, chunksize=8):
                keep(res)
            print(f"dgn: {len(paths)} read", flush=True)
        if args.sns:
            z = zipfile.ZipFile(args.sns)
            root = "ShapeNetSem-backup/"
            rows = list(csv.DictReader(io.TextIOWrapper(z.open(root + "metadata.csv"), "utf-8")))
            rows = [r for r in rows if CATEGORIES & set(r["category"].split(",")) and r["unit"]]
            rows = rows[: args.limit or None]

            def jobs():
                for r in rows:
                    sid = r["fullId"].split(".", 1)[1]
                    try:
                        raw = z.read(f"{root}models-OBJ/models/{sid}.obj")
                    except KeyError:
                        continue
                    up = [float(x) for x in r["up"].replace("\\", "").split(",")] if r["up"] else [0.0, 0.0, 1.0]
                    yield f"sns/{sid}", raw, float(r["unit"]), up
            for res in ex.map(do_sns, jobs(), chunksize=4):
                keep(res)
            print(f"sns: {len(rows)} candidates", flush=True)
    np.savez_compressed(out / "hulls.npz", **arrays)
    (out / "pool_g1_index.json").write_text(json.dumps(dict(
        sources=dict(gso="Google Scanned Objects (CC-BY 4.0) via gso_g1.py", sns="ShapeNetSem (non-commercial research)",
                     dgn="DexGraspNet meshdata (CC BY-NC 4.0), mirrors.pku.edu.cn/dl-release/DexGraspNet-ICRA2023"),
        rules=dict(one_hand=ONE, two_hands=TWO), split="crc32(name) % 10: 0 test, 1 val, else train",
        objects=index), indent=1))
    kept = [e for e in index if e["kept"]]
    for s in ("gso", "sns", "dgn"):
        k = [e for e in kept if e["source"] == s]
        print(f"{s}: kept {len(k)} of {sum(e['source'] == s for e in index)}; one hand {sum(e['hands'] == 1 for e in k)}, "
              f"two {sum(e['hands'] == 2 for e in k)}; train/val/test {[sum(e['split'] == p for e in k) for p in ('train', 'val', 'test')]}")
    print(f"hulls.npz {(out / 'hulls.npz').stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
