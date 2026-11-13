"""Draw a random one-hand and two-hand object set from Google Scanned Objects (real scale, CC-BY 4.0).

Reads gso_g1_index.json (all 1,030 objects measured by gso_g1.py) and applies the pool_g1.py rules:
  one hand : narrowest width 1-5 cm, height 1.5-15 cm, longest <= 20 cm, mass <= 0.5 kg (the index's `kept` flag)
  two hands: longest side 20-50 cm, narrowest width >= 8 cm, height 4-40 cm, mass <= 3 kg (density 100-300 kg/m3,
             drawn per object from crc32(name) like pool_g1.py)
Writes random_set.json: {"seed", "one_hand": [...], "two_hands": [...]} with name, dims (cm), mass (kg) per object.

  python so101/assets/objects/random_set.py [--n=40] [--seed=0]
Then fetch the meshes with fetch_gso.py / gso_g1.py --src=DIR and build hulls on the cluster (pool_cluster.sbatch).
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
TWO = dict(longest=(0.20, 0.50), width_min=0.08, height=(0.04, 0.40), mass=3.0, density=(100.0, 300.0))


def two_hands(o):
    if not (TWO["longest"][0] <= o["length_m"] <= TWO["longest"][1] and o["width_m"] >= TWO["width_min"]
            and TWO["height"][0] <= o["height_m"] <= TWO["height"][1]):
        return None
    u = np.random.default_rng(zlib.crc32(f"gso/{o['name']}".encode())).random()
    density = TWO["density"][0] + u * (TWO["density"][1] - TWO["density"][0])
    mass = density * o["volume_cm3"] / 1e6
    return mass if mass <= TWO["mass"] else None


def row(o, mass):
    return dict(name=o["name"], width_cm=round(o["width_m"] * 100, 1), length_cm=round(o["length_m"] * 100, 1),
                height_cm=round(o["height_m"] * 100, 1), mass_kg=round(float(mass), 3))


def main():
    args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    n, seed = int(args.get("n", 40)), int(args.get("seed", 0))
    objs = json.loads((HERE / "gso_g1_index.json").read_text())["objects"]
    one = [row(o, o["mass_kg"]) for o in objs if o["kept"]]
    two = [row(o, m) for o in objs if (m := two_hands(o)) is not None]
    rng = np.random.default_rng(seed)
    pick = lambda xs: sorted((xs[i] for i in rng.choice(len(xs), min(n, len(xs)), replace=False)), key=lambda r: r["name"])
    out = dict(seed=seed, source="Google Scanned Objects via kevinzakka/mujoco_scanned_objects (CC-BY 4.0)",
               pool=dict(one_hand=len(one), two_hands=len(two)), one_hand=pick(one), two_hands=pick(two))
    (HERE / "random_set.json").write_text(json.dumps(out, indent=1) + "\n")
    for hands in ("one_hand", "two_hands"):
        print(f"{hands}: {len(out[hands])} of {out['pool'][hands]}")
        for r in out[hands]:
            print(f"  {r['name'][:40]:40s} {r['width_cm']:5.1f} x {r['length_cm']:5.1f} x {r['height_cm']:5.1f} cm  {r['mass_kg']:.2f} kg")


if __name__ == "__main__":
    main()
