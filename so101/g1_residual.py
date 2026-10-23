"""Residual refinement of the learned G1 grasp (step b, LOG.md 2026-10-06): a Gaussian policy over small corrections to
the learned chooser's grasp (heading, height, index-pad clearance, opening margin, approach tilt), trained by REINFORCE
with a per-shape baseline on one-step episodes run by the grasp sampler's executor (reward: lifted 10 cm and held 1 s,
sim truth). The motion stays the open-loop IK executor: v2's failures were misses, so the grasp choice is what to learn.

Training layouts use seeds 500000+ (the held-out test uses 900000+). The policy's mean is linear in the chooser's
features of (observation, chosen grasp); its spread is learned.

Commands (cluster, CPU):
  python -m so101.g1_residual train --model=runs/grasps/chooser-v3.pt --iters=15 --batch=96 --procs=16
      -> runs/grasps/residual.npz and the reward curve runs/grasps/residual_curve.png
  python -m so101.g1_grasps test --policy=residual --model=... --shape=box --n=60 --procs=4   (held-out lift test)
"""
import json
import sys
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np

OUT = Path("runs/grasps")
PATH = OUT / "residual.npz"
DIMS = ("heading", "height", "clear", "margin", "tilt")
SCALE = np.array([0.15, 0.08, 0.003, 0.006, 0.12])  # one unit of correction: radians, height share, metres, metres, radians
TRAIN_SEED = 500000


def correct(g, delta):
    """Grasp g moved by `delta` (in SCALE units), kept inside the sampler's ranges."""
    d = dict(zip(DIMS, np.asarray(delta) * SCALE))
    return replace(g, heading=g.heading + d["heading"], height=float(np.clip(g.height + d["height"], 0.15, 0.9)),
                   clear=float(np.clip(g.clear + d["clear"], 0.0, 0.015)),
                   margin=float(np.clip(g.margin + d["margin"], 0.002, 0.035)),
                   tilt=float(np.clip(g.tilt + d["tilt"], 0.0, 1.4)))


class Residual:
    """mean(phi) = W @ (phi - mu) / sd; spread exp(log_std) per dimension."""

    def __init__(self, n_in, path=None):
        if path is not None and Path(path).exists():
            d = np.load(path)
            self.W, self.log_std, self.mu, self.sd = d["W"], d["log_std"], d["mu"], d["sd"]
        else:
            self.W, self.log_std = np.zeros((len(DIMS), n_in)), np.full(len(DIMS), np.log(0.7))
            self.mu, self.sd = np.zeros(n_in), np.ones(n_in)

    def mean(self, phi):
        return self.W @ ((phi - self.mu) / self.sd)

    def save(self, path=PATH):
        np.savez(path, W=self.W, log_std=self.log_std, mu=self.mu, sd=self.sd)


def episode(job):
    """One training grasp: (seed, shape, W, log_std, mu, sd) -> (features, noise, reward, shape)."""
    from so101.g1_chooser import Chooser, features
    from so101.g1_grasps import check, poses, run, scene
    import so101.g1_grasps as gg
    seed, shape, W, log_std, mu, sd = job
    gg.CHOOSER = gg.CHOOSER or Chooser()
    arm, guard, geom, obj = scene(seed, shape)
    g, _, _ = gg.CHOOSER.choose(arm, geom, obj, np.random.default_rng([seed, 17]))
    if g is None:
        return None
    phi = features(obj, asdict(g))
    eps = np.random.default_rng([seed, 23]).normal(size=len(DIMS))
    g2 = correct(g, W @ ((phi - mu) / sd) + np.exp(log_std) * eps)
    pre, grasp, F, _, _ = poses(arm, geom, g2)
    checked = check(arm, [pre, grasp], F)
    if checked[0] is None:  # a correction the arm cannot do earns nothing
        return phi, eps, 0.0, shape
    out, _ = run(arm, guard, geom, g2, checked=checked)
    return phi, eps, float(out["success"]), shape


def train(iters=15, batch=96, procs=16, lr=0.05, shapes=None):
    """REINFORCE with a per-shape mean-reward baseline and Adam; batch layouts per iteration, spread over shapes."""
    from so101.g1_grasps import SHAPES
    shapes = shapes or SHAPES
    pol = Residual(len(_n_features()))
    m, v = [np.zeros_like(pol.W), np.zeros_like(pol.log_std)], [np.zeros_like(pol.W), np.zeros_like(pol.log_std)]
    curve, seed = [], TRAIN_SEED
    with Pool(procs) as pool:
        for it in range(iters):
            t = time.time()
            jobs = [(seed + i, shapes[i % len(shapes)], pol.W, pol.log_std, pol.mu, pol.sd) for i in range(batch)]
            seed += batch
            res = [r for r in pool.map(episode, jobs) if r is not None]
            phi = np.array([r[0] for r in res])
            if it == 0:  # standardise the features once, from the first batch
                pol.mu, pol.sd = phi.mean(0), phi.std(0) + 1e-6
            x, eps, rew = (phi - pol.mu) / pol.sd, np.array([r[1] for r in res]), np.array([r[2] for r in res])
            sh = np.array([r[3] for r in res])
            adv = rew - np.array([rew[sh == s].mean() for s in sh])
            std = np.exp(pol.log_std)
            gW = (adv[:, None] * eps / std)[:, :, None] * x[:, None, :]
            gs = adv[:, None] * (eps ** 2 - 1)
            for k, (p, g) in enumerate(((pol.W, gW.mean(0)), (pol.log_std, gs.mean(0)))):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                p += lr * m[k] / (1 - 0.9 ** (it + 1)) / (np.sqrt(v[k] / (1 - 0.999 ** (it + 1))) + 1e-8)  # ascent
            pol.log_std = np.clip(pol.log_std, np.log(0.1), np.log(1.5))
            curve.append(dict(iter=it, lifted=float(rew.mean()), n=len(rew), spread=np.exp(pol.log_std).round(2).tolist(),
                              by_shape={s: round(float(rew[sh == s].mean()), 2) for s in shapes if (sh == s).any()},
                              minutes=round((time.time() - t) / 60, 1)))
            print(json.dumps(curve[-1]), flush=True)
            pol.save()
    (OUT / "residual_curve.json").write_text(json.dumps(curve))
    plot(curve)


def _n_features():
    from so101.g1_chooser import features
    obj = dict(centre=[0.2, 0, 0.02], half=[0.02, 0.02, 0.02], narrow=0.0)
    return features(obj, dict(tilt=0, azimuth=0, heading=0, height=0.5, clear=0.006, margin=0.012, back=0.04))


def plot(curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot([c["iter"] for c in curve], [c["lifted"] for c in curve], "k-o", label="all shapes")
    for s in curve[0]["by_shape"]:
        ax.plot([c["iter"] for c in curve], [c["by_shape"].get(s, np.nan) for c in curve], alpha=0.6, label=s)
    ax.set_xlabel("iteration (one batch of training layouts)")
    ax.set_ylabel("lifted share (exploring)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Residual grasp refinement: REINFORCE on the chooser's picks")
    fig.tight_layout()
    fig.savefig(OUT / "residual_curve.png", dpi=130)


if __name__ == "__main__":
    args = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
    if sys.argv[1] == "train":
        train(int(args.get("iters", 15)), int(args.get("batch", 96)), int(args.get("procs", 16)), float(args.get("lr", 0.05)))
