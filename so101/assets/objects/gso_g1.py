"""Household objects for the G1 + Dex3 grasp benchmark (owner 2026-10-08: "a larger distribution of items including some
household objects for a separate benchmark").

Source: Google Scanned Objects (CC-BY 4.0, Downs et al. 2022) via github.com/kevinzakka/mujoco_scanned_objects (MIT,
pinned commit below): 1,030 scanned household items at real scale, each with up to 32 convex collision parts (V-HACD).
Only the collision parts are used (the benchmark runs headless). Each object keeps its scanned resting pose.

Kept if the Dex3 pinch could hold it in that pose (g1.Hand): narrowest horizontal width 1-5 cm (the hand opens 7.3 cm and
needs ~1.8 cm pad clearance; the thumb cannot close below ~0.8 cm), height 1.5-15 cm, longest side <= 20 cm (fits the
layout area), mass <= 0.5 kg. Mass = hull volume x a density drawn per object, uniform 200-600 kg/m3 (hollow packaging
to solid plastic; GSO has no masses). Writes gso_g1_index.json (kept and rejected, with why) and gso_g1/hulls.npz
(kept objects' parts: vertices in metres, triangles), which isaac/g1_grasp_rl.py --objects=household reads.

  python so101/assets/objects/gso_g1.py [--src=DIR]   (DIR: the repo's models/ folder; default: sparse-clone it here)
"""
import argparse
import json
import subprocess
import zlib
from pathlib import Path

import numpy as np

REPO = "https://github.com/kevinzakka/mujoco_scanned_objects.git"
COMMIT = "6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c"
HERE = Path(__file__).resolve().parent
OUT = HERE / "gso_g1"
WIDTH = (0.01, 0.05)
HEIGHT = (0.015, 0.15)
LONGEST = 0.20
MASS_MAX = 0.5
DENSITY = (200.0, 600.0)


def read_obj(path):
    v, f = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(x.split("/")[0]) - 1 for x in line.split()[1:]]
            f += [[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)]
    return np.array(v, float), np.array(f, int)


def volume_centroid(v, f):
    """Signed volume and centroid of a closed triangle mesh (divergence theorem)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = vol6.sum() / 6
    cen = ((a + b + c) * vol6[:, None]).sum(0) / (24 * vol) if abs(vol) > 1e-12 else v.mean(0)
    return abs(vol), cen


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def hull2d(p):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    p = np.unique(np.round(p, 6), axis=0)
    if len(p) < 3:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(pts):
        h = []
        for q in pts:
            while len(h) >= 2 and cross2(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, up = half(p), half(p[::-1])
    return np.array(lo[:-1] + up[:-1])


def footprint(xy):
    """Narrowest width and its direction (rotating calipers over the hull's edges), the length across it, the
    footprint's area and the centre of its oriented box."""
    h = hull2d(xy)
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        e = e / np.linalg.norm(e)
        n = np.array([-e[1], e[0]])
        pn, pe = h @ n, h @ e
        w = pn.max() - pn.min()
        if best is None or w < best[0]:
            centre = n * (pn.max() + pn.min()) / 2 + e * (pe.max() + pe.min()) / 2
            best = (w, float(np.arctan2(n[1], n[0])), pe.max() - pe.min(), centre)
    area = 0.5 * abs(np.dot(h[:, 0], np.roll(h[:, 1], -1)) - np.dot(h[:, 1], np.roll(h[:, 0], -1)))
    return best[0], best[1], best[2], area, best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="")
    args = ap.parse_args()
    src = Path(args.src) if args.src else HERE / "gso_g1" / "repo" / "models"
    if not src.exists():
        repo = src.parent
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout", REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "sparse-checkout", "init", "--no-cone"], check=True)
        (repo / ".git" / "info" / "sparse-checkout").write_text("/models/*/model_collision_*.obj\n/LICENSE\n")
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", COMMIT], check=True)
    index, arrays = [], {}
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        parts = sorted(d.glob("model_collision_*.obj"), key=lambda p: int(p.stem.split("_")[-1]))
        if not parts:
            continue
        meshes = [read_obj(p) for p in parts]
        V = np.concatenate([v for v, _ in meshes])
        vols = [volume_centroid(v, f) for v, f in meshes]
        vol = sum(x for x, _ in vols)
        com = sum(x * c for x, c in vols) / vol
        lo, hi = V.min(0), V.max(0)
        w, narrow, length, area, centre = footprint(V[:, :2])
        rng = np.random.default_rng(zlib.crc32(d.name.encode()))
        density = float(rng.uniform(*DENSITY))
        mass = density * vol
        height = hi[2] - lo[2]
        why = []
        if not WIDTH[0] <= w <= WIDTH[1]:
            why.append(f"width {w * 100:.1f} cm")
        if not HEIGHT[0] <= height <= HEIGHT[1]:
            why.append(f"height {height * 100:.1f} cm")
        if length > LONGEST:
            why.append(f"length {length * 100:.1f} cm")
        if mass > MASS_MAX:
            why.append(f"mass {mass:.2f} kg")
        # resting stability in the scanned pose: the centre of mass over the base (points within 3 mm of the bottom)
        base = V[V[:, 2] < lo[2] + 0.003, :2]
        bh = hull2d(base) if len(base) >= 3 else base
        inside = len(bh) >= 3 and all(cross2(bh[(i + 1) % len(bh)] - bh[i], com[:2] - bh[i]) >= 0 for i in range(len(bh)))
        e = dict(name=d.name, parts=len(parts), width_m=round(float(w), 4), length_m=round(float(length), 4),
                 height_m=round(float(height), 4), narrow_rad=round(narrow, 4), centre_xy_m=[round(float(x), 4) for x in centre],
                 bottom_z_m=round(float(lo[2]), 4), fill=round(float(area / max(w * length, 1e-9)), 3),
                 volume_cm3=round(vol * 1e6, 1), density=round(density), mass_kg=round(float(mass), 4),
                 com_m=[round(float(x), 4) for x in com], com_over_base=bool(inside), kept=not why, why=why)
        index.append(e)
        if not why:
            arrays[f"{d.name}/v"] = np.concatenate([v for v, _ in meshes]).astype(np.float32)
            off = np.cumsum([0] + [len(v) for v, _ in meshes])[:-1]
            arrays[f"{d.name}/f"] = np.concatenate([f + o for (_, f), o in zip(meshes, off)]).astype(np.int32)
            arrays[f"{d.name}/fstart"] = np.cumsum([0] + [len(f) for _, f in meshes]).astype(np.int32)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "hulls.npz", **arrays)
    (HERE / "gso_g1_index.json").write_text(json.dumps(dict(
        source=f"{REPO} @ {COMMIT} (Google Scanned Objects, CC-BY 4.0; MJCF MIT)",
        rules=dict(width_m=WIDTH, height_m=HEIGHT, longest_m=LONGEST, mass_max_kg=MASS_MAX, density_kg_m3=DENSITY),
        objects=index), indent=1))
    kept = [e for e in index if e["kept"]]
    print(f"{len(kept)} of {len(index)} kept; {sum(e['com_over_base'] for e in kept)} with the centre of mass over the base; "
          f"hulls.npz {(OUT / 'hulls.npz').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
