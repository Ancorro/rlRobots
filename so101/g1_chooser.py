"""Learned grasp chooser for the G1 (warm start step a, LOG.md 2026-10-06): a small network that predicts whether a grasp
lifts the object, trained on the grasp sampler's executed grasps (so101/g1_grasps.py). To choose, draw candidate grasps
from the sampler's prior, rank them by predicted success, and take the best one the arm can reach cleanly (the same
'sample, rank with a learned quality model, execute the best' recipe as Dex-Net 2.0, Mahler et al. 2017).

It sees only what perception can give: the object's centre and rough size (with a perception-like error added in
training, g1_grasps.perceived) and the heading of its narrow side; plus the grasp's own parameters.

Commands (cluster, CPU): python -m so101.g1_chooser train 'runs/grasps/sample-*.jsonl' [--model=PATH]  (default
                         runs/grasps/chooser.pt); python -m so101.g1_grasps test --policy=chooser [--model=PATH] ...
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

MODEL = Path(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), "runs/grasps/chooser.pt"))


def features(obj, g):
    """Network input for observation `obj` (centre, half sizes, narrow heading; perceived) and grasp g (a dict)."""
    c, h, narrow = obj["centre"], obj["half"], obj["narrow"]
    rel = g["heading"] - narrow  # closing direction against the object's narrow side
    az = g["azimuth"] - np.arctan2(c[1] + 0.1, c[0] + 0.08)  # lean side against the line from the right shoulder
    return np.array([c[0], c[1], *h, max(h[0], h[1]) / max(min(h[0], h[1]), 1e-3), np.cos(2 * narrow), np.sin(2 * narrow),
                     g["tilt"], np.cos(az), np.sin(az), np.cos(rel), np.sin(rel), np.cos(2 * rel), np.sin(2 * rel),
                     np.cos(g["heading"]), np.sin(g["heading"]), g["height"], g["clear"] * 100, g["margin"] * 100,
                     g["back"] * 100, g.get("hand", 0)], np.float32)


def noisy(obj, rng, cm=0.4):
    """obj as perception might report it: centre off by `cm` cm per axis, sizes by 5 %, narrow heading by 5 degrees."""
    return dict(obj, centre=list(np.add(obj["centre"], rng.normal(0, cm / 100, 3))),
                half=list(np.multiply(obj["half"], 1 + rng.normal(0, 0.05))), narrow=obj["narrow"] + rng.normal(0, 0.09))


def net(n_in):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(n_in, 128), torch.nn.ReLU(), torch.nn.Linear(128, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 1))


def train(paths, epochs=300, copies=8, seed=0):
    """Fit on every executed grasp (lifted or not; retested ones weigh in their robust share), `copies` noisy views of
    each observation. Holds out a fifth of the layouts to report accuracy and the success rate of the top-ranked fifth."""
    import torch
    rows = [json.loads(l) for p in paths for l in open(p)]
    rows = [r for r in rows if r.get("grasp") and r["result"] in ("lifted", "missed", "dropped")]
    rng = np.random.default_rng(seed)
    seeds = np.array([r["obj"]["seed"] for r in rows])
    val = np.isin(seeds, rng.choice(np.unique(seeds), len(np.unique(seeds)) // 5, replace=False))
    y = np.array([r.get("robust", 1.0) * r["success"] for r in rows], np.float32)
    X = lambda idx, k: np.array([features(noisy(rows[i]["obj"], rng), rows[i]["grasp"]) for i in idx for _ in range(k)])
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    Xtr, ytr = torch.tensor(X(tr, copies)), torch.tensor(np.repeat(y[tr], copies))
    Xva, yva = torch.tensor(X(va, 1)), y[va]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    torch.manual_seed(seed)
    model = net(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-4)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best = (np.inf, None, 0)  # early stopping on the held-out layouts: the 300-epoch v2 model was overconfident
    for epoch in range(epochs):
        for b in torch.randperm(len(Xtr)).split(256):
            loss = bce(model((Xtr[b] - mu) / sd).squeeze(1), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(bce(model((Xva - mu) / sd).squeeze(1), torch.tensor(yva)))
        if v < best[0]:
            best = (v, {k: t.clone() for k, t in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1])
    with torch.no_grad():
        p = torch.sigmoid(model((Xva - mu) / sd).squeeze(1)).numpy()
    top = p >= np.quantile(p, 0.8)
    pos, neg = p[yva > 0.5], p[yva <= 0.5]
    auc = float(np.mean(pos[:, None] > neg[None, :])) if len(pos) and len(neg) else float("nan")
    print(f"{len(rows)} executed grasps ({int(y.sum())} lifted, robust-weighted), {len(va)} held out by layout; best"
          f" epoch {best[2]} of {epochs}, held-out loss {best[0]:.3f}, ranking AUC {auc:.2f}, lift rate overall"
          f" {yva.mean():.2f}, in the top-ranked fifth {yva[top].mean():.2f}, predicted there {p[top].mean():.2f}")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, n_in=Xtr.shape[1]), MODEL)


class Chooser:
    """choose(arm, geom, obj, rng): the grasp with the best predicted success that the arm can do cleanly
    (g1_grasps.check), with check's answer and the prediction, among `n` candidates: half from the sampler's wide prior,
    half around today's scripted grasp (and that grasp itself); the scripted grasp if none of the best 80 can be done."""

    def __init__(self, path=MODEL, n=None):
        import torch
        n = n or int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--candidates=")), 256))
        d = torch.load(path, weights_only=False)
        self.model = net(d["n_in"])
        self.model.load_state_dict(d["state"])
        self.mu, self.sd, self.n = d["mu"], d["sd"], n

    def choose(self, arm, geom, obj, rng):
        import torch
        from so101.g1_grasps import check, near, poses, random_grasp, scripted
        from dataclasses import asdict
        g0 = scripted(arm, geom, obj)  # today's grasp: a candidate itself and the centre of half the others
        cands = [random_grasp(rng, obj["narrow"]) for _ in range(self.n // 2)]
        cands += [g0] + [near(rng, g0) for _ in range(self.n // 2)] if g0 is not None else []
        X = torch.tensor(np.array([features(obj, asdict(g)) for g in cands]))
        with torch.no_grad():
            p = torch.sigmoid(self.model((X - self.mu) / self.sd).squeeze(1)).numpy()
        for i in np.argsort(-p)[:80]:  # best first; stop at the first the arm can do
            pre, grasp, F, _, _ = poses(arm, geom, cands[i])
            checked = check(arm, [pre, grasp], F)
            if checked[0] is not None:
                return cands[i], checked, float(p[i])
        return (g0, None, None) if g0 is not None else (None, None, None)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train([p for a in sys.argv[2:] if not a.startswith("--") for p in glob.glob(a)])
