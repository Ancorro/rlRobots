"""Policies scored on OpenHLM-data: the hold baseline (the contract's reference), a ridge regression, and the learned
chunk policy (an MLP on the normalised state history, optionally with a small convolutional encoder on the current
camera frames, that writes the next `horizon` actions as a residual on the hold baseline).

Every policy predicts in the same way: `predict(sd, episode, starts)` -> (chunks x horizon x action_dim) raw actions,
given the SetData and the chunk start frames; `score()` turns that into the contract's per-episode records.

Every policy is conditioned on the set's task index (one-hot): the instruction is known to a policy at run time.

Checkpoints (torch.save): {"kind": "mlp" | "act", "config", "state_dict", "stats", "hmap", "state_names", "action_names",
"fps", "horizon", "history", "cameras", "image_size", "n_tasks", "tasks", "set"}; ridge: {"kind": "ridge", "W", ...}.
`load()` rebuilds any of them.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from openhlm import data, spec

CANDIDATES = {  # name -> training settings (openhlm/train.py); the pipeline trains each and selects on validation.
    # The history and horizon are the contract's (spec.HISTORY_S, HORIZON_S), not a candidate's choice.
    # noise: Gaussian noise on the normalised state history during training (a standard BC regulariser).
    # vision: the candidate gets the camera frames (the pipeline passes --camera only to these; without a frame cache
    # they are skipped); weight: its share of a worker's training budget relative to the others (vision trains slower).
    "mlp": dict(kind="mlp", hidden=512, layers=3, dropout=0.1, lr=1e-3, batch=1024, weight_decay=1e-4, noise=0.02, vision=False, weight=1),
    "mlp-wide": dict(kind="mlp", hidden=1024, layers=4, dropout=0.2, lr=5e-4, batch=1024, weight_decay=1e-3, noise=0.05, vision=False, weight=1),
    # the same MLP plus the convolutional encoder (Encoder) on the current frame of each camera
    # amp: bf16 autocast for the forward pass (the convolutions; the loss stays fp32) where the GPU supports it
    "mlp-cam": dict(kind="mlp", hidden=1024, layers=4, dropout=0.2, lr=5e-4, batch=512, weight_decay=1e-3, noise=0.05, vision=True, weight=2, amp=True),
    # ACT from the LeRobot stack (openhlm/act_policy.py; needs lerobot installed, e.g. .venv-vla): a transformer with a
    # CVAE and a ResNet18 per camera; small settings so a few thousand steps fit the budget
    "act": dict(kind="act", lr=1e-4, batch=128, weight_decay=1e-4, noise=0.02, dim_model=256, n_heads=8, dim_feedforward=1024,
                n_encoder_layers=2, n_decoder_layers=1, use_vae=True, kl_weight=10.0, latent_dim=32, dropout=0.1, pretrained=True,
                vision=True, weight=2),
    "ridge": dict(kind="ridge", alpha=1.0, vision=False, weight=0),
}


def build(cfg, history, state_dim, horizon, action_dim, cameras=0, image_size=96, ctx_dim=1):
    """The candidate's model: an nn.Module with forward(hist, base, frames=None, target=None, ctx=None) -> B x K x Da,
    ctx the task one-hot (ctx_dim = the set's number of tasks)."""
    if cfg["kind"] == "mlp":
        return ChunkMLP(history, state_dim, horizon, action_dim, hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg.get("dropout", 0.0),
                        cameras=cameras, ctx_dim=ctx_dim)
    if cfg["kind"] == "act":
        from openhlm.act_policy import ACTChunk
        return ACTChunk(history, state_dim, horizon, action_dim, cameras=cameras, image_size=image_size, ctx_dim=ctx_dim,
                        **{k: v for k, v in cfg.items() if k not in ("kind", "vision", "weight")})
    raise ValueError(cfg["kind"])


# ----------------------------------------------------------------------------------------------- windows on a device
class Windows:
    """All frames of some episodes of a set as flat tensors on one device, with the valid chunk starts, so a batch is
    two gathers: the state history ending at t and the actions from t on."""

    def __init__(self, sd, episodes, stats, hmap, device, stride=1, frames=False, n_tasks=None):
        self.sd, self.device = sd, device
        self.H, self.K = sd.history, sd.horizon
        S = np.concatenate([sd.episodes[e]["state"] for e in episodes])
        A = np.concatenate([sd.episodes[e]["action"] for e in episodes])
        self.S = torch.as_tensor((S - stats["state_mean"]) / np.maximum(stats["state_std"], spec.STD_FLOOR), dtype=torch.float32, device=device)
        self.A_raw = torch.as_tensor(A, dtype=torch.float32, device=device)
        self.S_raw = torch.as_tensor(S, dtype=torch.float32, device=device)
        self.a_mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=device)
        self.a_std = torch.as_tensor(np.maximum(stats["action_std"], spec.STD_FLOOR), dtype=torch.float32, device=device)
        self.A = (self.A_raw - self.a_mean) / self.a_std
        self.loss_w = torch.as_tensor(spec.scored_dims(stats["action_std"]), dtype=torch.float32, device=device)  # the metric's dims
        self.n_tasks = int(n_tasks or sd.n_tasks)  # a checkpoint's count wins over the (partial) set's
        T = np.concatenate([np.full(len(sd.episodes[e]["state"]), sd.episodes[e]["task_index"]) for e in episodes])
        self.T = torch.as_tensor(np.clip(T, 0, self.n_tasks - 1), dtype=torch.int64, device=device)
        self.hmap = torch.as_tensor(hmap, device=device)
        self.mapped = self.hmap >= 0
        starts, off = [], 0
        for e in episodes:
            starts += [off + t for t in sd.train_starts(e, stride)]
            off += len(sd.episodes[e]["state"])
        self.starts = torch.as_tensor(starts, device=device)
        self.hist_idx = torch.arange(-self.H + 1, 1, device=device)
        self.chunk_idx = torch.arange(self.K, device=device)
        self.frames = None
        if frames and sd.frames:
            n = sum(len(sd.frames[e]) for e in episodes)
            self.frames = torch.empty((n,) + tuple(sd.frames[episodes[0]].shape[1:]), dtype=torch.uint8, device=device)
            fidx, foff = [], 0  # table row (flat) -> cached frame row: the latest cached frame at or before it
            for e in episodes:  # episode by episode onto the device: no second host copy of the whole set
                m = len(sd.frames[e])
                self.frames[foff:foff + m] = torch.from_numpy(np.ascontiguousarray(sd.frames[e]))
                fidx.append(foff + np.arange(len(sd.episodes[e]["state"])) // sd.frame_stride)
                foff += m
            self.fidx = torch.as_tensor(np.concatenate(fidx), device=device)

    def __len__(self):
        return len(self.starts)

    def base_norm(self, t):
        """The policies' residual base in normalised action units for starts t, the same as the hold reference
        (data.hold_chunk): the current state s_t for the action dimensions held from a state dimension, the last
        recorded action a_{t-1} for the dimensions no state dimension tracks (a controller knows its last command)."""
        held = self.S_raw[t][:, self.hmap.clamp(min=0)]
        last = self.A[(t - 1).clamp(min=0)]
        return torch.where(self.mapped[None], (held - self.a_mean) / self.a_std, last)

    def batch(self, t):
        """t: start frames (flat). -> history (B x H x Ds normalised), base (B x Da normalised), actions (B x K x Da
        normalised), frames (B x cams x 3 x h x w float in [0, 1]) or None, task one-hot (B x n_tasks)."""
        hist = self.S[t[:, None] + self.hist_idx[None]]
        act = self.A[t[:, None] + self.chunk_idx[None]]
        fr = None
        if self.frames is not None:
            fr = self.frames[self.fidx[t]].permute(0, 1, 4, 2, 3).float() / 255.0
        ctx = F.one_hot(self.T[t], self.n_tasks).float()
        return hist, self.base_norm(t), act, fr, ctx

    def sample(self, n, gen):
        i = torch.randint(len(self.starts), (n,), device=self.device, generator=gen)
        return self.starts[i]


# ----------------------------------------------------------------------------------------------------------- models
class SpatialSoftmax(nn.Module):
    """Expected image coordinates of each feature channel (Levine et al. 2016): 2 numbers per channel."""

    def forward(self, x):
        b, c, h, w = x.shape
        p = F.softmax(x.flatten(2), -1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        ey = (p.sum(3) * ys).sum(2)
        ex = (p.sum(2) * xs).sum(2)
        return torch.cat([ex, ey], 1)


class Encoder(nn.Module):
    """A small convolutional encoder per camera (shared weights): 4 strided convs, GroupNorm, spatial softmax."""

    def __init__(self, out=64):
        super().__init__()
        ch = [3, 32, 64, 64, out]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 3 if i else 5, stride=2, padding=2 if not i else 1), nn.GroupNorm(8, ch[i + 1]), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.pool = SpatialSoftmax()
        self.dim = 2 * out

    def forward(self, x):  # (B, 3, h, w) -> (B, 2*out)
        return self.pool(self.net(x))


def shift(x, max_px, gen):
    """Random translation of up to max_px pixels (per sample), the usual augmentation for small-data visuomotor BC."""
    b = x.shape[0]
    if max_px <= 0:
        return x
    d = (torch.rand(b, 2, device=x.device, generator=gen) * 2 - 1) * (2.0 * max_px / x.shape[-1])
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1.0
    theta[:, :, 2] = d
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="border", align_corners=False)


class ChunkMLP(nn.Module):
    """Normalised state history (+ camera features) -> K x Da residual on the hold baseline."""

    def __init__(self, history, state_dim, horizon, action_dim, hidden=512, layers=3, dropout=0.1, cameras=0, enc_out=64, ctx_dim=1):
        super().__init__()
        self.H, self.K, self.Da = history, horizon, action_dim
        self.cameras, self.ctx_dim = cameras, ctx_dim
        self.encoder = Encoder(enc_out) if cameras else None
        inp = history * state_dim + ctx_dim + (cameras * self.encoder.dim if cameras else 0)
        blocks, d = [], inp
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(d, horizon * action_dim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # starts as the hold baseline

    def forward(self, hist, hold, frames=None, target=None, ctx=None):
        x = hist.flatten(1)
        if ctx is None:
            ctx = torch.zeros(x.shape[0], self.ctx_dim, device=x.device)
            ctx[:, 0] = 1.0
        x = torch.cat([x, ctx], 1)
        if self.encoder is not None:
            b, c = frames.shape[:2]
            f = self.encoder(frames.flatten(0, 1)).view(b, -1)
            x = torch.cat([x, f], 1)
        delta = self.head(self.body(x)).view(-1, self.K, self.Da)
        return hold[:, None, :] + delta


# --------------------------------------------------------------------------------------------------------- policies
class Hold:
    name = "hold"

    def __init__(self, hmap):
        self.hmap = hmap

    def predict(self, sd, e, starts):
        return np.stack([data.hold_chunk(sd, e, t, self.hmap) for t in starts])


class Ridge:
    """Closed-form ridge regression from the flattened normalised history to the normalised residual chunk."""
    name = "ridge"

    def __init__(self, stats, hmap, history, horizon, alpha=1.0, W=None, n_tasks=None):
        self.stats, self.hmap, self.H, self.K, self.alpha, self.W, self.n_tasks = stats, hmap, history, horizon, alpha, W, n_tasks

    def features(self, w, t):
        hist, hold, act, _, ctx = w.batch(t)
        X = torch.cat([hist.flatten(1), ctx, torch.ones(len(t), 1, device=hist.device)], 1)
        return X, hold, act

    def fit(self, w, max_rows=400_000, batch=8192):
        self.n_tasks = w.n_tasks
        n = len(w)
        sel = w.starts if n <= max_rows else w.starts[torch.linspace(0, n - 1, max_rows, device=w.device).long()]
        d = self.H * w.S.shape[1] + w.n_tasks + 1
        XtX = torch.zeros(d, d, device=w.device, dtype=torch.float64)
        XtY = torch.zeros(d, self.K * w.A.shape[1], device=w.device, dtype=torch.float64)
        for i in range(0, len(sel), batch):
            X, hold, act = self.features(w, sel[i:i + batch])
            Y = (act - hold[:, None]).flatten(1)
            XtX += X.double().T @ X.double()
            XtY += X.double().T @ Y.double()
        reg = self.alpha * len(sel) / 1000.0 * torch.eye(d, device=w.device, dtype=torch.float64)
        reg[-1, -1] = 0.0
        self.W = torch.linalg.solve(XtX + reg, XtY).float().cpu()
        return self

    def predict(self, sd, e, starts):
        w = Windows(sd, [e], self.stats, self.hmap, "cpu", n_tasks=self.n_tasks)
        t = torch.as_tensor(starts)
        X, hold, _ = self.features(w, t)
        out = hold[:, None] + (X @ self.W).view(len(t), self.K, -1)
        return (out * w.a_std + w.a_mean).numpy()

    def state(self):
        return dict(kind="ridge", W=self.W, alpha=self.alpha)


class Learned:
    """A trained chunk model (ChunkMLP or ACTChunk) as a predictor (any device; frames from the set's cache)."""
    name = "mlp"

    def __init__(self, model, stats, hmap, device="cpu", name="mlp"):
        self.model, self.stats, self.hmap, self.device, self.name = model.to(device).eval(), stats, hmap, device, name

    @torch.no_grad()
    def predict(self, sd, e, starts):
        w = Windows(sd, [e], self.stats, self.hmap, self.device, frames=self.model.cameras > 0, n_tasks=self.model.ctx_dim)
        t = torch.as_tensor(starts, device=self.device)
        hist, hold, _, fr, ctx = w.batch(t)
        out = self.model(hist, hold, fr, ctx=ctx)
        return ((out * w.a_std + w.a_mean).float().cpu().numpy())


class Ensemble:
    """The mean of several trained policies' chunks (selected on validation like any candidate; no training of its
    own). Members must share the set's statistics and hold map (they do: both come from the training split)."""
    name = "ens"

    def __init__(self, members):
        self.members = list(members)

    def predict(self, sd, e, starts):
        return np.mean([m.predict(sd, e, starts) for m in self.members], 0)


def score(sd, policy, episodes, std, hmap, split, policy_name=None, set_name=None, diagnostics=None):
    """The contract's per-episode records of a policy on episodes of a set. diagnostics (a dict) collects, over all
    chunks, the normalised absolute error of the policy and of hold by chunk step (K,) and by action dimension (Da,)
    on the scored dimensions, for the report's "where the error is" table."""
    recs = []
    name = policy_name or getattr(policy, "name", "policy")
    hold = Hold(hmap)
    w = np.asarray(spec.scored_dims(std), np.float64)
    sd_std = np.maximum(std, spec.STD_FLOOR)
    for e in episodes:
        starts = sd.chunk_starts(e)
        if not starts:
            continue
        true = np.stack([sd.episodes[e]["action"][t:t + sd.horizon] for t in starts])
        pred = policy.predict(sd, e, starts)
        held = hold.predict(sd, e, starts)
        err = np.mean([data.nmae(p, tr, std) for p, tr in zip(pred, true)])
        err_h = np.mean([data.nmae(h, tr, std) for h, tr in zip(held, true)])
        recs.append(spec.record(set_name or sd.name, e, split, name, err, err_h, len(starts), len(sd.episodes[e]["state"])))
        if diagnostics is not None:
            ep = np.abs(pred - true) / sd_std * w  # (chunks, K, Da)
            eh = np.abs(held - true) / sd_std * w
            diagnostics["step"] = diagnostics.get("step", 0) + ep.sum((0, 2)) / max(w.sum(), 1)
            diagnostics["step_hold"] = diagnostics.get("step_hold", 0) + eh.sum((0, 2)) / max(w.sum(), 1)
            diagnostics["dim"] = diagnostics.get("dim", 0) + ep.sum((0, 1))
            diagnostics["dim_hold"] = diagnostics.get("dim_hold", 0) + eh.sum((0, 1))
            diagnostics["chunks"] = diagnostics.get("chunks", 0) + len(starts)
    return recs


# ------------------------------------------------------------------------------------------------------ checkpoints
def save(path, kind, sd, stats, hmap, config, **kw):
    payload = dict(kind=kind, set=sd.name, stats={k: np.asarray(v).tolist() for k, v in stats.items()}, hmap=np.asarray(hmap).tolist(),
                   state_names=sd.state_names, action_names=sd.action_names, fps=sd.fps, horizon=sd.horizon, history=sd.history,
                   config=config, cameras=list(sd.cameras), n_tasks=sd.n_tasks, tasks=sd.tasks, state_key=sd.state_key,
                   action_key=sd.action_key, hold_source=getattr(sd, "hold_source", "names"), **kw)
    torch.save(payload, path)
    return path


def load(path, device="cpu"):
    """-> (policy, payload). The set's history/horizon in the payload must match the data it is scored on."""
    p = torch.load(path, map_location="cpu", weights_only=False)
    stats = {k: np.asarray(v, np.float32) for k, v in p["stats"].items()}
    hmap = np.asarray(p["hmap"], np.int64)
    if p["kind"] == "ridge":
        return Ridge(stats, hmap, p["history"], p["horizon"], alpha=p["config"].get("alpha", 1.0), W=p["W"], n_tasks=p.get("n_tasks", 1)), p
    c = p["config"]
    model = build(c, p["history"], len(stats["state_mean"]), p["horizon"], len(stats["action_mean"]), cameras=len(p.get("cameras") or []),
                  image_size=p.get("image_size", 96), ctx_dim=p.get("n_tasks", 1))
    model.load_state_dict(p["state_dict"])
    return Learned(model, stats, hmap, device, name=p.get("name", c["kind"])), p
