"""Train one candidate on one OpenHLM set inside a time budget and score it on the set's validation episodes.

  python -m openhlm.train --set g1_HLM-12_humi --candidate mlp --out runs/openhlm/<tag> [--budget-s 200] [--data DIR]
                          [--camera KEY,KEY] [--frames DIR] [--device cuda|cpu] [--seed 0] [--split-file split.json]

Writes <out>/<set>/<candidate>.pt (the checkpoint; openhlm/policy.py load()), <out>/<set>/<candidate>-val.jsonl (the
contract's per-episode records on validation) and <out>/<set>/<candidate>.json (the run: steps, best step, timing,
the hold and ridge references on validation, the candidate's settings). Ends with one TRAIN-DONE line.

Training: every step samples a batch of chunk starts from the training episodes (all starts, every frame), gathers the
state history and the action chunk on the device, and takes an L1 step on the normalised residual (the metric is an L1).
Validation every --eval-every seconds; the checkpoint kept is the one with the best validation success rate (ties: mean
skill); training stops at the budget or after --patience evaluations without improvement. Cosine learning-rate decay
over the budget, AdamW, gradient clipping at 1.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openhlm import data, policy, spec  # noqa: E402


def log(*a):
    print(*a, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True)
    p.add_argument("--candidate", default="mlp", choices=sorted(policy.CANDIDATES))
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="")
    p.add_argument("--split-file", default="", help="pipeline's split.json; default: the contract's five-set split (4 test episodes per set)")
    p.add_argument("--budget-s", type=float, default=300)
    p.add_argument("--eval-every", type=float, default=20, help="seconds between validation passes")
    p.add_argument("--patience", type=int, default=8, help="validation passes without improvement before stopping")
    p.add_argument("--max-steps", type=int, default=200_000)
    p.add_argument("--camera", default="", help="comma-separated camera keys (observation.images.<key>) from the frame cache")
    p.add_argument("--frames", default="", help="frame cache folder (openhlm/frames.py)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shift-px", type=int, default=4, help="random image shift augmentation (pixels)")
    p.add_argument("--max-frames-gb", type=float, default=24.0, help="vision: training frames kept on the device at most (episodes dropped at random beyond it)")
    p.add_argument("--name", default="", help="policy name in the records (default: the candidate)")
    p.add_argument("--cache", default="", help="folder of per-set .npz caches of the tables (data.SetData; built by the pipeline)")
    p.add_argument("--state-key", default="", help="table column of the state (default: the first of spec.STATE_KEYS present)")
    p.add_argument("--action-key", default="", help="table column of the action (default: the first of spec.ACTION_KEYS present)")
    a = p.parse_args(argv)

    t0 = time.time()
    deadline = t0 + a.budget_s
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu")
    cfg = dict(policy.CANDIDATES[a.candidate])
    cams = [c for c in a.camera.split(",") if c] if cfg.get("vision") else []
    if cfg.get("vision") and not cams:
        raise SystemExit(f"{a.candidate} is a vision candidate: give --camera and --frames (openhlm/frames.py)")
    name = a.name or a.candidate
    out = pathlib.Path(a.out) / a.set
    out.mkdir(parents=True, exist_ok=True)

    if a.split_file:
        roles = json.load(open(a.split_file))[a.set]
        roles = {int(k): v for k, v in roles.items()}
    else:
        roles = data.split_sets(a.data, [a.set], canonical=True)[a.set]  # the five-set shares: 4 test episodes
    sd = data.SetData(a.set, data.set_dir(a.data, a.set), episodes=[e for e, r in roles.items() if r != "test"],
                      cameras=cams, frames_dir=a.frames or None, state_key=a.state_key or None, action_key=a.action_key or None,
                      cache_dir=a.cache or None)
    log(f"{a.set}: data ready in {time.time() - t0:.0f} s")
    train_eps = sd.usable([e for e, r in roles.items() if r == "train"])
    val_eps = sd.usable([e for e, r in roles.items() if r == "val"])
    if not train_eps or not val_eps:
        raise SystemExit(f"{a.set}: {len(train_eps)} training and {len(val_eps)} validation episodes long enough for a chunk")
    stats = sd.stats(train_eps)  # from the whole training split, so every candidate's metric is normalised the same way
    if cams:  # the frames live on the device: keep a random subset of training episodes that fits
        rng = np.random.default_rng(a.seed)
        order = list(rng.permutation(train_eps))
        kept, total = [], 0
        for e in order:
            nb = sd.frames[e].nbytes
            if total + nb > a.max_frames_gb * 2 ** 30:
                continue
            kept.append(int(e))
            total += nb
        if len(kept) < len(train_eps):
            log(f"{a.set}: frames of {len(train_eps)} training episodes exceed {a.max_frames_gb} GB; training on {len(kept)} ({total / 2 ** 30:.1f} GB)")
            train_eps = sorted(kept)
    hmap = sd.hold_map(train_eps)  # by names, else inferred from the training episodes
    std = np.maximum(stats["action_std"], spec.STD_FLOOR)
    log(f"{a.set}: fps {sd.fps:g}, history {sd.history}, horizon {sd.horizon}, state {sd.state_key!r} {sd.state_dim}, action {sd.action_key!r} "
        f"{sd.action_dim}, tasks {sd.n_tasks}, train {len(train_eps)} episodes / {sum(len(sd.episodes[e]['state']) for e in train_eps)} frames, "
        f"val {len(val_eps)}, {data.describe_hold_map(sd, hmap)}, cameras {cams or 'none'}, device {dev}")

    # references on validation: hold (skill 0 by definition) and, for the learned candidates, nothing else (ridge is its own candidate)
    refs = {"hold": spec.summary(policy.score(sd, policy.Hold(hmap), val_eps, std, hmap, "val", "hold"))}
    run = dict(set=a.set, candidate=a.candidate, name=name, config=cfg, seed=a.seed, device=str(dev), cameras=cams,
               state_key=sd.state_key, action_key=sd.action_key, hold_map=hmap.tolist(), hold_source=sd.hold_source,
               train_episodes=len(train_eps), val_episodes=len(val_eps), fps=sd.fps, history=sd.history, horizon=sd.horizon,
               state_dim=sd.state_dim, action_dim=sd.action_dim, refs=refs, budget_s=a.budget_s)

    if cfg["kind"] == "ridge":
        w = policy.Windows(sd, train_eps, stats, hmap, dev)  # Ridge.fit subsamples to max_rows itself
        pol = policy.Ridge(stats, hmap, sd.history, sd.horizon, alpha=cfg["alpha"]).fit(w)
        recs = policy.score(sd, pol, val_eps, std, hmap, "val", name)
        summ = spec.summary(recs)
        policy.save(out / f"{a.candidate}.pt", "ridge", sd, stats, hmap, cfg, W=pol.W, name=name)
        spec.write_jsonl(out / f"{a.candidate}-val.jsonl", recs)
        run.update(val=summ, steps=0, best_step=0, seconds=round(time.time() - t0, 1), checkpoint=str(out / f"{a.candidate}.pt"))
        (out / f"{a.candidate}.json").write_text(json.dumps(run, indent=1))
        log(f"TRAIN-DONE set={a.set} candidate={a.candidate} val={summ['successes']}/{summ['episodes']} mean_skill={summ['mean_skill']:.3f} "
            f"hold_mean_skill={refs['hold']['mean_skill']:.3f} seconds={run['seconds']}")
        return 0

    w = policy.Windows(sd, train_eps, stats, hmap, dev, frames=bool(cams))
    image_size = int(w.frames.shape[-2]) if w.frames is not None else 96
    model = policy.build(cfg, sd.history, sd.state_dim, sd.horizon, sd.action_dim, cameras=len(cams), image_size=image_size, ctx_dim=sd.n_tasks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    amp = bool(cfg.get("amp")) and dev.type == "cuda" and torch.cuda.is_bf16_supported()
    torch.backends.cudnn.benchmark = True
    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{a.candidate}: {n_params / 1e6:.2f}M parameters, {len(w)} training windows, batch {cfg['batch']}, budget {a.budget_s:.0f} s"
        + (", bf16 autocast" if amp else ""))

    def evaluate():
        pol = policy.Learned(model, stats, hmap, dev, name=name)
        recs = policy.score(sd, pol, val_eps, std, hmap, "val", name)
        model.train()
        return recs, spec.summary(recs)

    best, best_recs, best_step, since_best, step, curve = None, None, 0, 0, 0, []
    eval_cost, last_eval, last_eval_step, stopped = 0.0, time.time(), 0, ""
    model.train()

    def consider(recs, summ):
        """Keep the checkpoint when validation improved; True when it did."""
        nonlocal best, best_recs, best_step, since_best
        key = (summ["successes"], summ["mean_skill"])
        if best is None or key > best:
            best, best_recs, best_step, since_best = key, recs, step, 0
            policy.save(out / f"{a.candidate}.pt", cfg["kind"], sd, stats, hmap, cfg, state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                        name=name, step=step, image_size=image_size)
            return True
        since_best += 1
        return False

    while step < a.max_steps:
        now = time.time()
        if now + eval_cost >= deadline:
            stopped = "budget"
            break
        frac = min(1.0, (now - t0) / a.budget_s)
        for g in opt.param_groups:
            g["lr"] = cfg["lr"] * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac)))
        t = w.sample(cfg["batch"], gen)
        hist, hold, act, fr, ctx = w.batch(t)
        if cfg.get("noise"):
            hist = hist + cfg["noise"] * torch.randn(hist.shape, device=dev, generator=gen)
        if fr is not None and a.shift_px:
            b, c = fr.shape[:2]
            fr = policy.shift(fr.flatten(0, 1), a.shift_px, gen).view(b, c, *fr.shape[2:])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            pred = model(hist, hold, fr, act, ctx=ctx)
        loss = ((pred.float() - act).abs() * w.loss_w).sum(-1).mean() / w.loss_w.sum().clamp(min=1)  # the metric's L1, on its dims
        if getattr(model, "aux_loss", None) is not None:
            loss = loss + model.aux_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if time.time() - last_eval >= a.eval_every or step == 1:
            te = time.time()
            recs, summ = evaluate()
            eval_cost = max(eval_cost, (time.time() - te) * 1.5)
            last_eval, last_eval_step = time.time(), step
            curve.append(dict(step=step, t=round(last_eval - t0, 1), loss=round(loss.item(), 4), val_successes=summ["successes"],
                              val_mean_skill=round(summ["mean_skill"], 4)))
            log(f"LEARN step={step} t={last_eval - t0:.0f}s loss={loss.item():.4f} val={summ['successes']}/{summ['episodes']} "
                f"mean_skill={summ['mean_skill']:.3f} lr={opt.param_groups[0]['lr']:.2e}")
            if not consider(recs, summ) and since_best >= a.patience:
                stopped = "patience"
                break
    else:
        stopped = "max_steps"
    if step > last_eval_step or best is None:  # the steps since the last evaluation count too
        recs, summ = evaluate()
        consider(recs, summ)
        log(f"LEARN step={step} t={time.time() - t0:.0f}s val={summ['successes']}/{summ['episodes']} mean_skill={summ['mean_skill']:.3f} (final)")
        curve.append(dict(step=step, t=round(time.time() - t0, 1), loss=(round(loss.item(), 4) if step else None), val_successes=summ["successes"],
                          val_mean_skill=round(summ["mean_skill"], 4)))
    log(f"TRAIN-STOP reason={stopped} step={step}")
    summ = spec.summary(best_recs)
    spec.write_jsonl(out / f"{a.candidate}-val.jsonl", best_recs)
    run.update(val=summ, steps=step, best_step=best_step, seconds=round(time.time() - t0, 1), curve=curve, params=n_params,
               checkpoint=str(out / f"{a.candidate}.pt"))
    (out / f"{a.candidate}.json").write_text(json.dumps(run, indent=1))
    log(f"TRAIN-DONE set={a.set} candidate={a.candidate} val={summ['successes']}/{summ['episodes']} mean_skill={summ['mean_skill']:.3f} "
        f"hold_mean_skill={refs['hold']['mean_skill']:.3f} steps={step} best_step={best_step} seconds={run['seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
