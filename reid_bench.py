"""Re-identification benchmark for stack v2's scene memory (so101/memory.py; owner 2026-10-09: "this needs to be able
to re-id well, we'll need to check that").

Each episode: an SO-101 table scene (3-5 objects plus look-alike twins: same colour and kind, size within 10 %; scanned
scenes get one twin of a scanned object), the stack's perception (stack_v1.look: DINO + SAM 2.1, depth points; no
namer), then 6 events, a look after each, then 2 quiet looks. The simulator moves objects directly (no arm motion; the arm stays parked):
  move     the robot puts object A on a place (the memory is told: picked, placed at the place)
  drop     told as a move, but A lands on a random free table spot (the memory should flag "drop?")
  knock    B slides 3-8 cm on the table, untold
  hide     C is out of view for one look (as if behind the arm), then back within 1 cm
  swap     the robot swaps two twins (told, both moves)
  remove   D leaves for good (its track should go missing within 2 looks; nothing else should)
Detections per look (centre, dims, place, SigLIP 2 and DINOv2 crop embeddings, the true object for scoring) are cached,
then every memory method replays the same looks (--score).

Scores per method: identity accuracy (a detection is on its object's first track, the id the planner knows; a fresh
track for a known object counts wrong), ID switches, extra tracks for known objects, drops flagged, false "missing", removed objects found
missing. Pre-registered: settings tuned on dev seeds 600000+ (40 episodes), then one run on test seeds 620000+
(100 episodes): "both" must beat "position" on identity accuracy among twins and moved objects, paired (McNemar).

  python reid_bench.py collect --seed=600000 --episodes=40 --tag=dev      (GPU: perception)  -> runs/reid/<tag>.pkl
  python reid_bench.py score --tag=dev [--grid]                           (CPU)              -> runs/reid/<tag>-score.json
"""
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
FLAGS = {a[2:] for a in sys.argv[2:] if a.startswith("--") and "=" not in a}
OUT = Path("runs/reid")
SAME = 0.04  # m: a detection within this of a true object is that object (scoring only)
EVENTS = ["move", "drop", "knock", "hide", "swap", "remove"]


# --- collection (GPU) ----------------------------------------------------------------------------------------------------
class Embedder:
    """SigLIP 2 (the Eyes' own model) and DINOv2-base on a masked crop (background grey), mean over views."""

    def __init__(self, eyes):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch, self.eyes = torch, eyes
        self.dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(eyes.dev).eval()

    @staticmethod
    def crop(img, mask, box):
        x0, y0, x1, y1 = np.round(box).astype(int)
        pad = int(0.1 * max(x1 - x0, y1 - y0))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
        c = img[y0:y1, x0:x1].copy()
        c[~mask[y0:y1, x0:x1]] = 128
        return c

    def __call__(self, views):
        crops = [self.crop(*v) for v in views]
        with self.torch.inference_mode():
            s = self.eyes.sig.get_image_features(**self.eyes.sig_proc(images=crops, return_tensors="pt").to(self.eyes.dev))
            s = getattr(s, "pooler_output", s)  # newer transformers return an output object
            d = self.dino(**self.dino_proc(images=crops, return_tensors="pt").to(self.eyes.dev)).last_hidden_state[:, 0]
        unit = lambda x: (x / x.norm(dim=-1, keepdim=True)).mean(0).float().cpu().numpy()  # noqa: E731
        return unit(s), unit(d)


def twin(spec, rng, k):
    t = copy.deepcopy(spec)
    if spec["shape"] == "scanned":
        t["scanned"] = f"{spec['scanned']}_twin{k}"
    else:
        f = rng.uniform(0.9, 1.1)
        t["size"] = [v * f for v in spec["size"]]
        t["z"] = spec["z"] * f
        t["foot"] = spec["foot"] * f
        t["name"] = f"{spec['name']} {k + 2}"
    return t


def collect():
    import mujoco

    import stack_v1
    from so101 import planner
    from so101.objects import body_name, build, layout, random_objects, scanned_objects
    from so101.robot import SO101, Guard, SimArm
    from so101.see import Eyes
    from so101.skills import Arm

    seed, n_ep, tag = int(ARGS.get("seed", 600000)), int(ARGS.get("episodes", 40)), ARGS.get("tag", "dev")
    OUT.mkdir(parents=True, exist_ok=True)
    eyes = emb = None
    episodes = []
    for e in range(n_ep):
        rng = np.random.default_rng([seed, e])
        scanned = e % 2 == 1
        for _ in range(50):
            n = int(rng.integers(3, 5))
            base = scanned_objects(rng, n, strict=False) if scanned else random_objects(rng, n)
            specs = base + [twin(base[i], rng, 0) for i in range(1 if scanned else 2)]
            try:
                poses = layout(rng, specs, tries=200, max_r=0.23)
                break
            except RuntimeError:
                continue
        model = build(specs, poses)
        data, scratch = mujoco.MjData(model), mujoco.MjData(model)
        if eyes is None:
            eyes = Eyes(model, data, cams=("cam_left", "cam_right"))
            emb = Embedder(eyes)
        eyes.attach(model, data, seed=e)
        arm = Arm(model, data, scratch, robot=Guard(SimArm(model, data, seed=e, robot=SO101), model, mode="warn", robot=SO101))
        arm.goal = stack_v1.PARK.copy()

        def settle(k=120):
            for _ in range(k):
                arm.step(arm.target(arm.goal))
        settle(67)
        names = [body_name(s) for s in specs]
        jadr = [model.jnt_qposadr[model.body(b).jntadr[0]] for b in names]
        true_pos = lambda i: data.xpos[model.body(names[i]).id].copy()  # noqa: E731

        def put(i, xy, z=None, yaw=None):
            a = jadr[i]
            data.qpos[a:a + 2] = xy
            data.qpos[a + 2] = (z if z is not None else data.qpos[a + 2]) + 0.002
            if yaw is not None:
                data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            data.qvel[model.body(names[i]).dofadr[0]:model.body(names[i]).dofadr[0] + 6] = 0
            mujoco.mj_forward(model, data)

        def free_spot():
            for _ in range(200):
                xy = np.array([rng.uniform(0.12, 0.26), rng.uniform(-0.15, 0.15)])
                if all(np.linalg.norm(true_pos(j)[:2] - xy) > 0.06 for j in range(len(specs))) and \
                        all(np.linalg.norm(np.array(c) / 100 - xy) > 0.08 for _, c in planner.PLACES.values()):
                    return xy
            return None

        def snap_look():
            objs = stack_v1.look(eyes, None, keep_views=True)
            dets = []
            for o in objs:
                d = [np.linalg.norm(true_pos(i)[:2] - o["centre"][:2]) for i in range(len(specs))]
                s, dn = emb(o["views"])
                dets.append(dict(centre=o["centre"], dims=o["dims"], where=o["where"], emb_siglip=s, emb_dino=dn,
                                 true=int(np.argmin(d)) if min(d) < SAME else None))
            return dets
        twins = [(i, len(base) + k) for k, i in enumerate(range(1 if scanned else 2))]
        looks = [dict(event="start", dets=snap_look(), told=[])]
        gone, hidden = set(), None
        order = list(rng.permutation(EVENTS))
        for ev in order:
            told, info = [], {}
            if hidden is not None:  # back within 1 cm
                i, xy, z = hidden
                put(i, xy + rng.uniform(-0.01, 0.01, 2), z)
                hidden = None
            present = [i for i in range(len(specs)) if i not in gone]
            if ev in ("move", "drop"):
                i = int(rng.choice(present))
                place = list(planner.PLACES)[int(rng.integers(len(planner.PLACES)))]
                pxy = np.array(planner.PLACES[place][1]) / 100
                told = [("picked", i), ("placed", i, pxy.tolist(), place)]
                if ev == "move":
                    put(i, pxy + rng.uniform(-0.01, 0.01, 2), z=true_pos(i)[2] + 0.07)
                else:
                    xy = free_spot()
                    if xy is not None:
                        put(i, xy, yaw=rng.uniform(-np.pi, np.pi))
                info = dict(obj=i, place=place)
            elif ev == "knock":
                i = int(rng.choice(present))
                a = rng.uniform(-np.pi, np.pi)
                put(i, true_pos(i)[:2] + rng.uniform(0.03, 0.08) * np.array([np.cos(a), np.sin(a)]))
                info = dict(obj=i)
            elif ev == "hide":
                i = int(rng.choice(present))
                hidden = (i, true_pos(i)[:2].copy(), float(data.qpos[jadr[i] + 2]))
                put(i, [-1.0, 1.0 + i])
                info = dict(obj=i)
            elif ev == "swap":
                a, b = twins[int(rng.integers(len(twins)))]
                if a in gone or b in gone:
                    continue
                pa, pb = true_pos(a)[:2].copy(), true_pos(b)[:2].copy()
                put(a, [-1.0, 3.0])
                put(b, pa)
                put(a, pb)
                told = [("picked", a), ("placed", a, pb.tolist(), "table"), ("picked", b), ("placed", b, pa.tolist(), "table")]
                info = dict(a=a, b=b)
            elif ev == "remove":
                cand = [i for i in present if hidden is None or i != hidden[0]]
                i = int(rng.choice(cand))
                put(i, [-1.0, -1.0 - i])
                gone.add(i)
                info = dict(obj=i)
            settle()
            looks.append(dict(event=ev, info=info, told=told, dets=snap_look(), gone=sorted(gone),
                              hidden=None if hidden is None else hidden[0]))
        for _ in range(2):  # two quiet looks: a removed object's track can go missing, a dropped one re-link
            settle()
            looks.append(dict(event="idle", info={}, told=[], dets=snap_look(), gone=sorted(gone), hidden=None))
        episodes.append(dict(episode=e, scanned=scanned, objects=[s["name"] for s in specs], twins=twins, looks=looks))
        print(f"EPISODE {e}: {len(specs)} objects, events {order}, detections per look "
              f"{[len(lk['dets']) for lk in looks]}", flush=True)
    path = OUT / f"{tag}.pkl"
    path.write_bytes(pickle.dumps(episodes))
    print(f"COLLECTED {len(episodes)} episodes -> {path}", flush=True)


# --- scoring (CPU) ------------------------------------------------------------------------------------------------------
def run_method(ep, method, emb="siglip", **kw):
    from so101.memory import SceneMemory
    mem = SceneMemory(method=method, **kw)
    owner, first, prev, out = {}, {}, {}, []  # first: each true object's first track (the id the planner knows)
    for k, lk in enumerate(ep["looks"]):
        tid_of = {}
        for act in lk["told"]:  # the actions before this look; detections map to tracks by the previous look's owners
            obj = act[1]
            t = next((t for t, o in owner.items() if o == obj and mem.tracks[t].status != "missing"), None)
            if t is None:
                continue
            if act[0] == "picked":
                mem.picked(t)
            else:
                mem.placed(t, act[2], act[3])
        dets = [dict(centre=d["centre"], dims=d["dims"], where=d["where"], emb=d[f"emb_{emb}"]) for d in lk["dets"]]
        assign, events = mem.update(dets)
        for i, d in enumerate(lk["dets"]):
            t = assign[i]
            if t not in owner:
                owner[t] = d["true"]
                if d["true"] is not None:
                    first.setdefault(d["true"], t)
            tid_of[i] = t
        rows = []
        for i, d in enumerate(lk["dets"]):
            if d["true"] is None:
                continue
            t = tid_of[i]
            rows.append(dict(true=d["true"], ok=first.get(d["true"]) == t, new=any(e == ("new", t) for e in events) and k > 0
                             and d["true"] in prev, switch=d["true"] in prev and prev[d["true"]] != t,
                             twin=any(d["true"] in p for p in ep["twins"]),
                             moved=lk["event"] in ("move", "drop", "knock", "swap", "hide") and d["true"] in
                             {lk["info"].get(x) for x in ("obj", "a", "b")}))
            prev[d["true"]] = t
        missing = [owner[t] for _, t in events if _ == "missing"]
        out.append(dict(event=lk["event"], info=lk.get("info", {}), rows=rows,
                        drop_flag=any(e == "drop?" and owner.get(t) == lk.get("info", {}).get("obj") for e, t in events),
                        missing=missing, relinked=[owner.get(t) for e, t in events if e == "relinked"],
                        gone=lk.get("gone", []), hidden=lk.get("hidden")))
    return out


def summarize(results):
    rows = [r for ep in results for lk in ep for r in lk["rows"]]
    rate = lambda xs: round(float(np.mean(xs)), 4) if xs else None  # noqa: E731
    drops = [lk["drop_flag"] or any(lk["info"].get("obj") in lk2["missing"] + lk2["relinked"] for lk2 in ep[k:k + 3])  # flagged, or its track
             for ep in results for k, lk in enumerate(ep) if lk["event"] == "drop"]  # went missing (a new one started)
    false_drop = [lk["drop_flag"] for ep in results for lk in ep if lk["event"] == "move"]
    false_missing = sum(sum(m not in lk["gone"] and m is not None for m in lk["missing"]) for ep in results for lk in ep)
    removed = [any(o in lk2["missing"] for lk2 in ep[k:k + 3]) for ep in results for k, lk in enumerate(ep) if lk["event"] == "remove"
               for o in [lk["info"]["obj"]]]
    return dict(id_acc=rate([r["ok"] for r in rows]), id_acc_twins=rate([r["ok"] for r in rows if r["twin"]]),
                id_acc_moved=rate([r["ok"] for r in rows if r["moved"]]), switches=sum(r["switch"] for r in rows),
                extra_tracks=sum(r["new"] for r in rows), drops_flagged=rate(drops), moves_flagged_as_drops=rate(false_drop),
                false_missing=false_missing, removed_found=rate(removed), detections=len(rows))


def pair_table(eps, emb):
    """Every (previous-look track candidate, detection) pair on consecutive looks, from the true labels: position
    distance (to the previous look's position of that object, or where it was told to go), appearance cosine, same?"""
    X, y = [], []
    for ep in eps:
        for prev, lk in zip(ep["looks"], ep["looks"][1:]):
            told = {a[1]: np.array(a[2]) for a in lk["told"] if a[0] == "placed"}
            for p in prev["dets"]:
                if p["true"] is None:
                    continue
                at = told.get(p["true"], np.asarray(p["centre"])[:2])
                for d in lk["dets"]:
                    if d["true"] is None:
                        continue
                    e1, e2 = p[f"emb_{emb}"], d[f"emb_{emb}"]
                    X.append([np.linalg.norm(at[:2] - np.asarray(d["centre"])[:2]),
                              float(e1 @ e2 / np.linalg.norm(e1) / np.linalg.norm(e2))])
                    y.append(p["true"] == d["true"])
    return np.array(X), np.array(y)


def calibrate(eps):
    """Same-object probability from (distance, cosine): logistic regression on dev pairs; the cost a gate of P = 0.5
    implies; and the error curve over gates (wrong merges vs extra tracks), per embedding."""
    out = {}
    for emb in ("siglip", "dino"):
        X, y = pair_table(eps, emb)
        F = np.c_[X[:, 0] / 0.05, (1 - X[:, 1]) / 0.1]
        w = np.zeros(3)
        A = np.c_[F, np.ones(len(F))]
        for _ in range(3000):  # plain gradient descent, balanced classes
            p = 1 / (1 + np.exp(-(A @ w)))
            g = A.T @ ((p - y) * np.where(y, 0.5 / y.mean(), 0.5 / (1 - y.mean()))) / len(y)
            w -= 0.5 * g
        p = 1 / (1 + np.exp(-(A @ w)))
        curve = [dict(p_gate=t, wrong_merge=round(float(((p >= t) & ~y).sum() / max((~y).sum(), 1)), 4),
                      missed_same=round(float(((p < t) & y).sum() / max(y.sum(), 1)), 4)) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
        cos_same, cos_diff = X[y, 1], X[~y, 1]
        out[emb] = dict(weights=w.round(4).tolist(), pairs=int(len(y)), same=int(y.sum()), curve=curve,
                        cos_same_p5_p50=np.percentile(cos_same, [5, 50]).round(3).tolist() if len(cos_same) else None,
                        cos_diff_p50_p95=np.percentile(cos_diff, [50, 95]).round(3).tolist() if len(cos_diff) else None)
        print(f"CALIB {emb}: {out[emb]}", flush=True)
    return out


def check_assignment():
    """The memory's assignment (scipy, or its own Hungarian without scipy) against brute force on random matrices."""
    import itertools

    from so101.memory import linear_sum_assignment
    rng = np.random.default_rng(0)
    for _ in range(200):
        n, m = (int(v) for v in rng.integers(1, 6, 2))
        C = rng.uniform(0, 10, (n, m))
        r, c = linear_sum_assignment(C)
        best = min(sum(C[i, j] for i, j in zip(range(n), p)) for p in itertools.permutations(range(m), n)) if n <= m else \
            min(sum(C[i, j] for i, j in zip(p, range(m))) for p in itertools.permutations(range(n), m))
        assert len(r) == min(n, m) and abs(C[r, c].sum() - best) < 1e-9, (C, r, c, best)
    print("assignment check: 200 random matrices match brute force", flush=True)


def score():
    check_assignment()
    tag = ARGS.get("tag", "dev")
    eps = pickle.loads((OUT / f"{tag}.pkl").read_bytes())
    methods = {"v1": dict(method="v1"), "position": dict(method="position"),
               "appearance-siglip": dict(method="appearance", emb="siglip"), "appearance-dino": dict(method="appearance", emb="dino"),
               "both-siglip": dict(method="both", emb="siglip"), "both-dino": dict(method="both", emb="dino")}
    if "grid" in FLAGS:  # dev only: the settings for "both"
        for emb in ("siglip", "dino"):
            for ps in (0.03, 0.05, 0.08):
                for asc in (0.05, 0.1, 0.2):
                    for gate in (2.0, 3.0, 5.0):
                        for rg in (0.5, 1.0, 2.0):
                            methods[f"both-{emb}-p{ps}-a{asc}-g{gate}-r{rg}"] = dict(method="both", emb=emb, pos_scale=ps, app_scale=asc,
                                                                                    gate=gate, relink_gate=rg)
    if "pick" in ARGS:  # the dev pick, for the test run: emb,pos_scale,app_scale,gate,relink_gate
        e, ps, asc, g, rg = ARGS["pick"].split(",")
        methods["picked"] = dict(method="both", emb=e, pos_scale=float(ps), app_scale=float(asc), gate=float(g), relink_gate=float(rg))
    res = {}
    for name, kw in methods.items():
        results = [run_method(ep, **kw) for ep in eps]
        res[name] = dict(settings=kw, **summarize(results),
                         per_det=[r["ok"] for ep in results for lk in ep for r in lk["rows"]],
                         per_kind=[(r["twin"], r["moved"]) for ep in results for lk in ep for r in lk["rows"]])
        s = {k: v for k, v in res[name].items() if k not in ("per_det", "settings")}
        print(f"{name:40s} {s}", flush=True)
    if "picked" in res:  # pre-registered test: picked vs position on twins and moved objects, paired (exact McNemar)
        from math import comb
        a, b, kind = res["picked"]["per_det"], res["position"]["per_det"], res["picked"]["per_kind"]
        for label, sel in (("twins", 0), ("moved", 1)):
            idx = [i for i, k in enumerate(kind) if k[sel]]
            n01 = sum(a[i] and not b[i] for i in idx)
            n10 = sum(b[i] and not a[i] for i in idx)
            n = n01 + n10
            p = min(1.0, 2 * sum(comb(n, k) for k in range(min(n01, n10) + 1)) / 2 ** n) if n else 1.0
            res[f"mcnemar_{label}"] = dict(picked_only=n01, position_only=n10, p=p, detections=len(idx))
            print(f"MCNEMAR {label}: picked right where position wrong {n01}, the reverse {n10}, p = {p:.2g} ({len(idx)} detections)", flush=True)
    res["calibration"] = calibrate(eps)
    (OUT / f"{tag}-score.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    {"collect": collect, "score": score}[sys.argv[1]]()
