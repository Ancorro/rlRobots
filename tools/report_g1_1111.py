"""Builds docs/reports/g1-2026-11-11.html from the saved records: the walk_carry validation runs on three object sets,
the leg-controller probe, and the residual-grab pilots. Numbers on the page are computed here from the record files and
logs under bench/results/batch-ik/, runs/wbc_probe/ and runs/grabrl_local/.

  .venv/Scripts/python tools/report_g1_1111.py [--pilots p2,p3] [--local runs/grabrl_local] [--embed-clips 1]

Three figures in the text are not read from a record and are marked where they appear: the pinch offset that summed
exploration noise produces (a calculation), and the two controllers' licence names.
"""
import argparse
import base64
import collections
import glob
import json
import re
import sys
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "isaac"))
import bench_spec as B  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--local", default=str(REPO / "runs/grabrl_local"),
               help="copies of the grab_rl.py outputs: <pilot>/{learn,hyper}.json and eval-*.json, zero-v8/, zero-scripted/, zero2-v8/, logs/")
p.add_argument("--pilots", default="p2,p3", help="pilots to report, in order; each needs <local>/<name>/learn.json")
p.add_argument("--round2", default="p4,p4c", help="second-round pilots (continued from the first round's best), in order")
p.add_argument("--pool-index", default="C:/isaac/share/datasets/pool_g1/pool_g1_index.json", help="the pool's index (object sizes and sources); optional")
p.add_argument("--embed-clips", type=int, default=0, help="1: the replay clips inside the page (6.7 MB); 0: a link to the replay page")
p.add_argument("--replays-url", default="https://claude.ai/artifact/7dpZH27MDJiS5SD5a794cG")
p.add_argument("--clips", default=str(REPO / "runs/clips_1111"), help="folder with replay mp4 files, clips.json and rec.json (optional)")
p.add_argument("--out", default=str(REPO / "docs/reports/g1-2026-11-11.html"))
args = p.parse_args()
BI = REPO / "bench/results/batch-ik"
SHAPES = list(B.SHAPES)


def load(f):
    return json.loads(Path(f).read_text())


def exempt(r):
    """Success with balls exempt from the lift tilt test (the contract's `lifted` applies it to balls too)."""
    lifted = r["rise"] >= B.LIFT_M and r["near"] < 0.05 and not r["fell_grasp"] and (r["shape"] == "ball" or r["tilt_hold"] < B.LIFTED_TILT_DEG)
    return bool(lifted and r["placed"] and r["upright"] and r["released"] and r["standing"])


def mcnemar(a, b):
    oa, ob = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    n = oa + ob
    return oa, ob, (min(1.0, 2 * sum(comb(n, i) for i in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0)


def pfmt(v):
    return "&lt; 0.0001" if v < 1e-4 else f"{v:.4f}" if v < 0.01 else f"{v:.3f}"


def stage(r, ball_exempt=True):
    """The first stage a failed run stopped at."""
    lifted = r["rise"] >= B.LIFT_M and r["near"] < 0.05 and (r["tilt_hold"] < B.LIFTED_TILT_DEG or (ball_exempt and r["shape"] == "ball"))
    if r["fell_grasp"]:
        return "Robot fell during the grasp"
    if r["rise"] < 0.05:
        return "Hand moved the object before closing" if r["moved_in"] in ("raise", "approach", "descend") else "Fingers closed, object did not rise"
    if not lifted:
        return "Object rose but tipped or slipped in the hand"
    if not r["carried"]:
        return "Dropped while carrying"
    return "Not set down upright on table B"


R = {k: load(BI / f) for k, f in {
    "prim-s": "val-walk-prim-scripted-n64-batch-3090.json", "prim-v": "val-walk-prim-v8-n64-batch-3090.json",
    "gso-s": "val-walk-gso-scripted-batch-3090.json", "gso-v": "val-walk-gso-v8-batch-3090.json",
    "pool-s": "val-walk-pool-scripted-batch-3090.json", "pool-v": "val-walk-pool-v8-batch-3090.json"}.items()}
PINK = {"s": load(REPO / "bench/results/day-walk-scripted-n32-3090.json"), "v": load(REPO / "bench/results/day-walk-v8-n32-3090.json")}
TEST = {"s": load(REPO / "bench/results/day3-walk-scripted-21970682.json"), "v": load(REPO / "bench/results/day3-walk-v8-21970683.json")}
WBC = load(REPO / "runs/wbc_probe/result.json")
LOCAL = Path(args.local)
_rep = BI / "val-walk-prim-v8-repeat-n64-batch-3090.json"
V8_REPEAT = load(_rep) if _rep.exists() else None


def lift_ok(r):
    """Lifted, with balls exempt from the tilt test (what isaac/grab_rl.py's eval counts)."""
    return bool(r["rise"] >= B.LIFT_M and r["near"] < 0.05 and not r["fell_grasp"] and (r["shape"] == "ball" or r["tilt_hold"] < B.LIFTED_TILT_DEG))


def zero_eval(folder):
    fs = glob.glob(str(LOCAL / folder / "eval-zero-val-n*.json"))
    if not fs:
        return None
    rec = load(fs[0])
    rows = rec["rows"]
    return dict(rows={(r["shape"], r["seed"]): r["lifted"] for r in rows}, n=len(rows), lifted=sum(r["lifted"] for r in rows),
                by={s_: sum(r["lifted"] for r in rows if r["shape"] == s_) for s_ in SHAPES}, settings=rec["settings"])


def goal_read(log):
    """The GOAL-READ line a script printed: where the robots stood when the grab's goal was read."""
    f_ = LOCAL / "logs" / log
    if not f_.exists():
        return None
    m = re.search(r"GOAL-READ[^\n]*", f_.read_text(errors="ignore"))
    if not m:
        return None
    t = m.group(0)
    g = lambda pat: float(re.search(pat, t).group(1))  # noqa: E731
    return dict(gap=100 * g(r"pelvis to edge m median ([\d.]+)"), tilt=g(r"tilt deg ([\d.]+)"), speed=100 * g(r"base speed m/s ([\d.]+)"))


def value_loss(log):
    """Value-function loss per iteration from an rsl_rl training log."""
    f_ = LOCAL / "logs" / log
    if not f_.exists():
        return []
    return [float(x) for x in re.findall(r"Mean value_function loss:\s*([-\d.eE+]+)", f_.read_text(errors="ignore"))]


PILOTS = []
for name in args.pilots.split(","):
    d = LOCAL / name
    if not (d / "learn.json").exists():
        continue
    P = dict(name=name, learn=load(d / "learn.json"), hyper=load(d / "hyper.json"), evals=[], walk=None)
    for f_ in sorted((f_ for f_ in glob.glob(str(d / "eval-model_*-val-n*.json")) if re.fullmatch(r"eval-model_\d+-val-n\d+\.json", Path(f_).name)),
                     key=lambda f_: int(re.search(r"model_(\d+)", f_).group(1))):
        P["evals"].append((int(re.search(r"model_(\d+)", f_).group(1)), {(r["shape"], r["seed"]): r["lifted"] for r in load(f_)["rows"]}))
    w = sorted(glob.glob(str(BI / f"val-walk-prim-v8+{name}-*-n64-batch-3090.json")))
    if w:
        P["walk"] = load(w[-1])
    PILOTS.append(P)
ZERO = {k: zero_eval(k) for k in ("zero-v8", "zero-scripted", "zero-v8-repeat", "zero2-v8")}
BEST = max((P for P in PILOTS if P["walk"]), key=lambda P: sum(r["success"] for r in P["walk"]["rows"]), default=None)
RES = BEST["walk"] if BEST else None
for a, b in (("prim-s", "prim-v"), ("gso-s", "gso-v"), ("pool-s", "pool-v")):
    assert [(r["shape"], r.get("name"), r["seed"]) for r in R[a]["rows"]] == [(r["shape"], r.get("name"), r["seed"]) for r in R[b]["rows"]], "paired runs share layouts"

RES_SET = {"prim": RES}
if BEST:
    _tag = BEST["walk"]["policy"].split("+")[1]
    for _k in ("gso", "pool"):
        _f = BI / f"val-walk-{_k}-v8+{_tag}-batch-3090.json"
        RES_SET[_k] = load(_f) if _f.exists() else None
        if RES_SET[_k]:
            assert [(r["name"], r["seed"]) for r in RES_SET[_k]["rows"]] == [(r["name"], r["seed"]) for r in R[_k + "-v"]["rows"]]

# second round: pilots continued from the first round's kept checkpoint
ROUND2 = []
for name in args.round2.split(","):
    d = LOCAL / name
    if not name or not (d / "learn.json").exists():
        continue
    P = dict(name=name, learn=[r for r in load(d / "learn.json") if r["episodes"] > 0], hyper=load(d / "hyper.json"), evals=[], walk={}, pool_eval=None, tag=None)
    for f_ in sorted(glob.glob(str(d / "eval-model_*-val-n384.json")), key=lambda f_: int(re.search(r"model_(\d+)", f_).group(1))):
        P["evals"].append((int(re.search(r"model_(\d+)", f_).group(1)), {(r["shape"], r["seed"]): r["lifted"] for r in load(f_)["rows"]}))
    w = sorted(glob.glob(str(BI / f"val-walk-prim-v8+{name}-*-n64-batch-3090.json")))
    tags = sorted({re.search(r"v8\+(.+?)-n64", Path(f_).name).group(1) for f_ in w}, key=lambda t: int(t.split("-")[-1]))

    def _walks(tag):
        out = {}
        for _k, _pat in (("prim", f"val-walk-prim-v8+{tag}-n64-batch-3090.json"), ("gso", f"val-walk-gso-v8+{tag}-batch-3090.json"), ("pool", f"val-walk-pool-v8+{tag}-batch-3090.json")):
            if (BI / _pat).exists():
                out[_k] = load(BI / _pat)
        return out
    P["other_tags"] = []
    if tags:
        sel = max(P["evals"], key=lambda e: (sum(v for k_, v in e[1].items() if k_[1] - 760000 < 32), e[0]))[0] if P["evals"] else None
        P["tag"] = f"{name}-{sel}" if f"{name}-{sel}" in tags else tags[0]  # the rule's choice: most lifted on primitive layouts 0-31
        P["walk"] = _walks(P["tag"])
        P["other_tags"] = [(t, _walks(t)) for t in tags if t != P["tag"]]
        pe = glob.glob(str(d / f"eval-model_{P['tag'].split('-')[-1]}-pool-val-n*.json"))
        if pe:
            P["pool_eval"] = {(r["name"], r["seed"]): r["lifted"] for r in load(pe[0])["rows"]}
    ROUND2.append(P)
POOL_EV = {}  # the training scene on the pool's validation layouts, per checkpoint: {(pilot, iteration): {(name, seed): lifted}}
for _pn in ("p3",) + tuple(n_ for n_ in args.round2.split(",") if n_):
    for f_ in glob.glob(str(LOCAL / _pn / "eval-model_*-pool-val-n*.json")):
        POOL_EV[(_pn, int(re.search(r"model_(\d+)", f_).group(1)))] = {(r["name"], r["seed"]): r["lifted"] for r in load(f_)["rows"]}
GATE = {}
for _k, _f in (("pool-zero", "eval-zero-pool-val-n232.json"), ("pool-p3", "eval-residual-p3-250-pool-val-n232.json"),
               ("gso-zero", "eval-zero-gso-val-n96.json"), ("gso-p3", "eval-residual-p3-250-gso-val-n96.json")):
    if (LOCAL / "gate" / _f).exists():
        GATE[_k] = {(r["name"], r["seed"]): r["lifted"] for r in load(LOCAL / "gate" / _f)["rows"]}
if GATE.get("pool-p3"):
    POOL_EV[("p3", 250)] = GATE["pool-p3"]

# --- small SVG helpers (colors from the page's tokens) ------------------------------------------------------------------------
FONT = 'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"'


def bars(groups, series, ymax, ylabel, w=640, h=300, unit="%", decimals=0, note=None):
    """Grouped vertical bars. groups: [(label, [value per series], [text above each bar])]; series: [(name, style)]."""
    L, Rm, T, Bm = 52, 14, 30, 58
    pw, ph = w - L - Rm, h - T - Bm
    gw = pw / len(groups)
    bw = min(46, gw * 0.8 / len(series))
    y = lambda v: T + ph * (1 - v / ymax)  # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{ylabel}">']
    ticks = [ymax * i / 4 for i in range(5)]
    for t in ticks:
        o.append(f'<line x1="{L}" y1="{y(t):.1f}" x2="{w - Rm}" y2="{y(t):.1f}" stroke="currentColor" stroke-width=".6" opacity="{0.9 if t == 0 else 0.14}"/>')
        o.append(f'<text x="{L - 7}" y="{y(t) + 4:.1f}" text-anchor="end" {FONT} font-size="11" fill="currentColor" opacity=".7">{t:.{decimals}f}{unit}</text>')
    o.append(f'<text x="14" y="{T + ph / 2:.0f}" transform="rotate(-90 14 {T + ph / 2:.0f})" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{ylabel}</text>')
    for gi, (label, vals, texts) in enumerate(groups):
        x0 = L + gi * gw + (gw - bw * len(series)) / 2
        for si, v in enumerate(vals):
            if v is None:
                continue
            x = x0 + si * bw
            o.append(f'<rect x="{x + 1.5:.1f}" y="{y(v):.1f}" width="{bw - 3:.1f}" height="{max(0.5, T + ph - y(v)):.1f}" {series[si][1]}/>')
            o.append(f'<text x="{x + bw / 2:.1f}" y="{y(v) - 5:.1f}" text-anchor="middle" {FONT} font-size="10.5" font-weight="600" fill="currentColor">{texts[si]}</text>')
        for li, line in enumerate(label.split("\n")):
            o.append(f'<text x="{L + gi * gw + gw / 2:.1f}" y="{T + ph + 17 + 13 * li}" text-anchor="middle" {FONT} font-size="11.5" fill="currentColor">{line}</text>')
    lx = L
    for name, style in series:
        o.append(f'<rect x="{lx}" y="8" width="11" height="11" {style}/><text x="{lx + 16}" y="18" {FONT} font-size="11.5" fill="currentColor">{name}</text>')
        lx += 22 + 6.6 * len(name)
    if note:
        o.append(f'<text x="{w - Rm}" y="18" text-anchor="end" {FONT} font-size="10.5" fill="currentColor" opacity=".7">{note}</text>')
    o.append("</svg>")
    return "".join(o)


def hbars(items, xmax, unit, w=640, label_w=250, row=30):
    """Horizontal bars: items [(label, value, text, style)]."""
    h = 14 + row * len(items) + 26
    pw = w - label_w - 70
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="horizontal bar chart">']
    for i in range(5):
        x = label_w + pw * i / 4
        o.append(f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{h - 24}" stroke="currentColor" stroke-width=".6" opacity="{0.9 if i == 0 else 0.14}"/>')
        o.append(f'<text x="{x:.1f}" y="{h - 8}" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{xmax * i / 4:g}{unit}</text>')
    for k, (label, v, text, style) in enumerate(items):
        yy = 14 + row * k
        o.append(f'<text x="{label_w - 8}" y="{yy + 15}" text-anchor="end" {FONT} font-size="11.5" fill="currentColor">{label}</text>')
        o.append(f'<rect x="{label_w}" y="{yy + 3}" width="{max(0.8, pw * v / xmax):.1f}" height="{row - 11}" {style}/>')
        o.append(f'<text x="{label_w + pw * v / xmax + 6:.1f}" y="{yy + 15}" {FONT} font-size="11" font-weight="600" fill="currentColor">{text}</text>')
    o.append("</svg>")
    return "".join(o)


def line_chart(xs, series, ymax, ylabel, xlabel, refs=(), w=640, h=280):
    L, Rm, T, Bm = 52, 16, 30, 44
    pw, ph = w - L - Rm, h - T - Bm
    x = lambda v: L + pw * (v - min(xs)) / max(1e-9, max(xs) - min(xs))  # noqa: E731
    y = lambda v: T + ph * (1 - v / ymax)  # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{ylabel}">']
    for i in range(5):
        t = ymax * i / 4
        o.append(f'<line x1="{L}" y1="{y(t):.1f}" x2="{w - Rm}" y2="{y(t):.1f}" stroke="currentColor" stroke-width=".6" opacity="{0.9 if i == 0 else 0.14}"/>')
        o.append(f'<text x="{L - 7}" y="{y(t) + 4:.1f}" text-anchor="end" {FONT} font-size="11" fill="currentColor" opacity=".7">{t:.0%}</text>')
    for v in xs[:: max(1, len(xs) // 8)]:
        o.append(f'<text x="{x(v):.1f}" y="{T + ph + 16}" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{v + 1}</text>')
    o.append(f'<text x="{L + pw / 2:.0f}" y="{h - 6}" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{xlabel}</text>')
    o.append(f'<text x="14" y="{T + ph / 2:.0f}" transform="rotate(-90 14 {T + ph / 2:.0f})" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{ylabel}</text>')
    for val, label in refs:
        o.append(f'<line x1="{L}" y1="{y(val):.1f}" x2="{w - Rm}" y2="{y(val):.1f}" stroke="currentColor" stroke-width="1" stroke-dasharray="5 4" opacity=".55"/>')
        o.append(f'<text x="{w - Rm - 4}" y="{y(val) - 5:.1f}" text-anchor="end" {FONT} font-size="10.5" fill="currentColor" opacity=".8">{label}</text>')
    lx = L
    for name, vals, style in series:
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, vals))
        o.append(f'<polyline points="{pts}" fill="none" {style} stroke-width="2"/>')
        for a, b in zip(xs, vals):
            o.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="2.6" {style.replace("stroke=", "fill=")} />')
        o.append(f'<line x1="{lx}" y1="14" x2="{lx + 18}" y2="14" {style} stroke-width="2"/><text x="{lx + 23}" y="18" {FONT} font-size="11.5" fill="currentColor">{name}</text>')
        lx += 34 + 6.6 * len(name)
    o.append("</svg>")
    return "".join(o)


def multi_line(series, ymax, ylabel, xlabel, w=640, h=290, xmin=0):
    """Lines with their own x values: series [(name, [(x, y)], style)]. xmin: where the x axis starts."""
    L, Rm, T, Bm = 52, 16, 30, 44
    pw, ph = w - L - Rm, h - T - Bm
    xmax = max(x for _, pts, _ in series for x, _ in pts)
    X = lambda v: L + pw * (v - xmin) / (xmax - xmin)  # noqa: E731
    Y = lambda v: T + ph * (1 - v / ymax)  # noqa: E731
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{ylabel}">']
    for i in range(5):
        t = ymax * i / 4
        o.append(f'<line x1="{L}" y1="{Y(t):.1f}" x2="{w - Rm}" y2="{Y(t):.1f}" stroke="currentColor" stroke-width=".6" opacity="{0.9 if i == 0 else 0.14}"/>')
        o.append(f'<text x="{L - 7}" y="{Y(t) + 4:.1f}" text-anchor="end" {FONT} font-size="11" fill="currentColor" opacity=".7">{t:.0%}</text>')
    step = 50 if xmax > 150 else 30
    for v in range(xmin, int(xmax) + 1, step):
        o.append(f'<text x="{X(v):.1f}" y="{T + ph + 16}" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{v}</text>')
    o.append(f'<text x="{L + pw / 2:.0f}" y="{h - 6}" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{xlabel}</text>')
    o.append(f'<text x="14" y="{T + ph / 2:.0f}" transform="rotate(-90 14 {T + ph / 2:.0f})" text-anchor="middle" {FONT} font-size="11" fill="currentColor" opacity=".7">{ylabel}</text>')
    lx = L
    for name, pts, style in series:
        o.append(f'<polyline points="{" ".join(f"{X(a_):.1f},{Y(b_):.1f}" for a_, b_ in pts)}" fill="none" {style} stroke-width="2"/>')
        fill = style.replace("stroke=", "fill=")
        for a_, b_ in pts:
            o.append(f'<circle cx="{X(a_):.1f}" cy="{Y(b_):.1f}" r="2.6" {fill}/>')
        o.append(f'<line x1="{lx}" y1="14" x2="{lx + 18}" y2="14" {style} stroke-width="2"/><text x="{lx + 23}" y="18" {FONT} font-size="11.5" fill="currentColor">{name}</text>')
        lx += 40 + 6.6 * len(name)
    o.append("</svg>")
    return "".join(o)


S_BASE, S_ACC, S_THIRD = 'fill="currentColor" opacity=".3"', 'fill="var(--accent)"', 'fill="var(--accent2)"'

# --- Section: three object sets -------------------------------------------------------------------------------------------------
sets = [("Primitives\n6 shapes x 64 layouts", "prim"), ("Household scans\n6 objects x 16 poses", "gso"), ("Object pool\n58 objects x 4 poses", "pool")]
groups, set_rows = [], []
for label, k in sets:
    s, v = R[k + "-s"]["rows"], R[k + "-v"]["rows"]
    n = len(s)
    cs, cv = sum(r["success"] for r in s), sum(r["success"] for r in v)
    oa, ob, pv = mcnemar([r["success"] for r in v], [r["success"] for r in s])
    vals, texts = [100 * cs / n, 100 * cv / n], [f"{cs}/{n}", f"{cv}/{n}"]
    if RES_SET.get(k):
        cr = sum(r["success"] for r in RES_SET[k]["rows"])
        vals.append(100 * cr / len(RES_SET[k]["rows"]))
        texts.append(f"{cr}/{len(RES_SET[k]['rows'])}")
    elif RES:
        vals.append(None)
        texts.append("")
    groups.append((label, vals, texts))
    set_rows.append((label.split("\n")[0], n, cs, cv, oa, ob, pv, sum(r["fell"] for r in s), sum(r["fell"] for r in v)))
series = [("Scripted pinch", S_BASE), ("v8 learned grab", S_ACC)] + ([("v8 + per-step correction", S_THIRD)] if RES else [])
FIG_SETS = bars(groups, series, 80, "Walk-carry success", unit="%")

# by shape, primitives (contract and ball-exempt)
ps, pv_ = R["prim-s"]["rows"], R["prim-v"]["rows"]
shape_groups = []
for sh in SHAPES:
    a = [r for r in ps if r["shape"] == sh]
    b = [r for r in pv_ if r["shape"] == sh]
    vals = [100 * sum(exempt(r) for r in a) / len(a), 100 * sum(exempt(r) for r in b) / len(b)]
    texts = [str(sum(exempt(r) for r in a)), str(sum(exempt(r) for r in b))]
    if RES:
        c = [r for r in RES["rows"] if r["shape"] == sh]
        vals.append(100 * sum(exempt(r) for r in c) / len(c))
        texts.append(str(sum(exempt(r) for r in c)))
    shape_groups.append((sh.replace("lying cylinder", "lying\ncylinder"), vals, texts))
FIG_SHAPES = bars(shape_groups, series, 100, "Success of 64 layouts", unit="%", note="balls not tilt-tested at the lift")

# failure stages, v8, primitives + pool
order = ["Fingers closed, object did not rise", "Object rose but tipped or slipped in the hand", "Hand moved the object before closing",
         "Not set down upright on table B", "Dropped while carrying", "Robot fell during the grasp"]
fail = {}
for k, rows, ex in (("prim", pv_, True), ("pool", R["pool-v"]["rows"], False)):
    c = collections.Counter(stage(r, ex) for r in rows if not (exempt(r) if ex else r["success"]))
    fail[k] = (c, sum(c.values()))
FIG_FAIL = hbars([(o, fail["prim"][0][o], f"{fail['prim'][0][o]} of {fail['prim'][1]}", S_ACC) for o in order], -(-max(fail["prim"][0].values()) // 40) * 40, "")

# ball scoring
ball = {}
for nm, rec in (("Validation, Pink IK, RTX 3090", PINK), ("Test, Pink IK, cluster", TEST), ("Validation, batch IK, RTX 3090", {"s": R["prim-s"], "v": R["prim-v"]})):
    s, v = rec["s"]["rows"], rec["v"]["rows"]
    n = len(s)
    c = (sum(r["success"] for r in s), sum(r["success"] for r in v), sum(exempt(r) for r in s), sum(exempt(r) for r in v))
    p1 = mcnemar([r["success"] for r in v], [r["success"] for r in s])
    p2 = mcnemar([exempt(r) for r in v], [exempt(r) for r in s])
    ball[nm] = (n, *c, p1, p2)
nb = sum(1 for r in PINK["v"]["rows"] if r["shape"] == "ball" and not r["lifted"] and r["carried"] and r["placed"])

# IK comparison on the same 192 layouts
pk = {(r["shape"], r["seed"]): r for r in PINK["s"]["rows"]}
pkv = {(r["shape"], r["seed"]): r for r in PINK["v"]["rows"]}
same_s = [r for r in ps if (r["shape"], r["seed"]) in pk]
same_v = [r for r in pv_ if (r["shape"], r["seed"]) in pkv]
med = lambda xs: sorted(xs)[len(xs) // 2]  # noqa: E731
IK = dict(n=len(same_s), pink_s=sum(pk[(r["shape"], r["seed"])]["success"] for r in same_s), batch_s=sum(r["success"] for r in same_s),
          pink_v=sum(pkv[(r["shape"], r["seed"])]["success"] for r in same_v), batch_v=sum(r["success"] for r in same_v),
          pink_fell=sum(pk[(r["shape"], r["seed"])]["fell_grasp"] for r in same_s), batch_fell=sum(r["fell_grasp"] for r in same_s),
          pink_err=100 * med([pk[(r["shape"], r["seed"])]["reach_err"] for r in same_s]), batch_err=100 * med([r["reach_err"] for r in same_s]),
          pink_gap=100 * med([pk[(r["shape"], r["seed"])]["stop_gap"] for r in same_s]), batch_gap=100 * med([r["stop_gap"] for r in same_s]))
new_s = [r for r in ps if (r["shape"], r["seed"]) not in pk]
new_v = [r for r in pv_ if (r["shape"], r["seed"]) not in pkv]

def nice(name):
    """A scanned object's catalog name as words."""
    words = name.replace("_", " ").split()
    fixed = {"GRANDFATHER DOLL": "Grandfather doll", "Schleich Lion Action Figure": "Schleich lion figure"}
    full = " ".join(words)
    if full in fixed:
        return fixed[full]
    if full.startswith("Office Depot Dell Series"):
        return f"Dell series {words[4]} ink cartridge"
    if full.startswith("Perricoen"):
        return "Perricone concealer"
    if full.startswith("Thomas Friends"):
        return "Thomas wooden train"
    return full[:30]


# household per object and pool distribution
gso = collections.OrderedDict()
for s, v in zip(R["gso-s"]["rows"], R["gso-v"]["rows"]):
    g = gso.setdefault(s["name"], [0, 0, 0])
    g[0] += s["success"]
    g[1] += v["success"]
    g[2] += 1
pool_per = collections.OrderedDict()
for r in R["pool-v"]["rows"]:
    pool_per.setdefault(r["name"], []).append(r["success"])
pool_hist = collections.Counter(sum(x) for x in pool_per.values())

# --- Section: leg controller probe ----------------------------------------------------------------------------------------------
C = WBC["conditions"]
wbc_items = [("AGILE in Isaac Lab, policy in the loop", WBC["agile_isaac"]["tilt_max"], S_BASE),
             ("NVIDIA controller, waist and arm replayed", C["fixed/replay/x1"]["tilt_max"]["median"], S_ACC),
             ("NVIDIA controller, waist sent as torso command", C["fixed/cmd/x1"]["tilt_max"]["median"], S_ACC),
             ("NVIDIA controller, arm only", C["fixed/wbc/x1"]["tilt_max"]["median"], S_ACC),
             ("NVIDIA controller, arms held still", C["still"]["tilt_max"]["median"], S_ACC)]
FIG_WBC = hbars([(a, b, f"{b:.1f} deg", c) for a, b, c in wbc_items], 40, " deg", label_w=290)

# --- Section: residual ----------------------------------------------------------------------------------------------------------
S_P = ['stroke="currentColor" opacity=".55"', 'stroke="var(--accent)"', 'stroke="var(--accent2)"']
FIG_LEARN = ""
if PILOTS:
    FIG_LEARN = multi_line([(f"Pilot {P['name']}", [(r["it"] + 1, r["lifted"]) for r in P["learn"]], S_P[i % 3]) for i, P in enumerate(PILOTS)],
                           0.8, "Training episodes ending lifted", "Training iteration")

clips = []
REC_N = None
if args.clips and (Path(args.clips) / "rec.json").exists():
    _rr = load(Path(args.clips) / "rec.json")["rows"]
    REC_N = (sum(r["success"] for r in _rr), len(_rr))
if args.embed_clips and args.clips and (Path(args.clips) / "clips.json").exists():
    rec_rows = {(r["shape"], r["seed"]): r for r in load(Path(args.clips) / "rec.json")["rows"]} if (Path(args.clips) / "rec.json").exists() else {}
    for c in load(Path(args.clips) / "clips.json"):
        f = Path(args.clips) / c["file"]
        r = rec_rows.get((c["shape"], c["seed"]))
        if r is None:
            what = c["outcome"].replace("-", " ")
        elif r["success"]:
            what = "lifted, carried, and set down upright on the second table"
        elif r["carried"] and r["placed"] and r["shape"] == "ball":
            what = f"carried and set down on the second table, and scored as not lifted because the ball turned {r['tilt_hold']:.0f} degrees in the fingers"
        elif r["rise"] < 0.05:
            what = "the fingers closed and the object stayed on the table"
        else:
            what = stage(r).lower()
        if f.exists():
            clips.append((dict(c, what=what), base64.b64encode(f.read_bytes()).decode()))

import grab_residual as _GR  # noqa: E402  (torch only; the correction's limits as trained)
BEST_DESC = (BEST or (PILOTS[-1] if PILOTS else {"hyper": {"residual": _GR.describe()}}))["hyper"]["residual"]
_p1 = LOCAL / "p1" / "learn.json"
P1_RATE = load(_p1)[0]["lifted"] if _p1.exists() else float("nan")

# --- page -----------------------------------------------------------------------------------------------------------------------
st = R["prim-v"]["settings"]
SETTINGS = (f"random layouts, validation seeds {B.BASE['walk_carry']['val']}+, {st['ik']} IK, walk command {st['vx']} m/s, stop target {st['stand']} m from the table edge, "
            f"perception noise {st['noise']} cm on v8's inputs, angular damping {st['roll_damp']:g} on rolling shapes, {st['gpu']}")
css = (Path.home() / ".claude/skills/reports/templates/paper-html/skeleton.html").read_text(encoding="utf-8")
style = re.search(r"<style>(.*?)</style>", css, re.S).group(1)
script = re.search(r"<script>(.*?)</script>", css, re.S).group(1)
style = style.replace("--accent: #7a1f1f;", "--accent: #1f5f8b;\n    --accent2: #b0621a;").replace("--accent: #e08c8c;", "--accent: #7fb6dc;\n      --accent2: #e0a868;")
style += """
  .page { padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
  .ai-note { text-align: center; color: var(--muted); font: 500 .74rem/1.3 ui-sans-serif, system-ui, sans-serif; margin: 0 0 1rem; }
  figure svg { color: var(--ink); }
  .chart { overflow-x: auto; }
  .clips { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 1rem; margin: 1.5rem 0; }
  .clips figure { margin: 0; counter-increment: none; }
  .clips video { width: 100%; height: auto; border: 1px solid var(--rule); background: var(--panel); }
  .clips figcaption::before { content: none; }
  td.hl { font-weight: 600; }
  td.num { white-space: nowrap; }
  .nw { white-space: nowrap; }
  thead th { vertical-align: bottom; line-height: 1.25; }
  table.wide { font-size: .84rem; }
  table.wide thead th, table.wide tbody td { padding-left: .4rem; padding-right: .4rem; }
  @media print {
    .table-wrap { overflow: visible; }
    table { font-size: .8rem; }
    .clips { display: none; }
    .ai-note { margin-bottom: .4rem; }
  }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; } }
  :root[data-theme="dark"] { color-scheme: dark; }
"""
# open in light, dark by the switch (the page's own default: light)
script = script.replace("var saved = 'auto';", "var saved = 'light';").replace("|| 'auto'", "|| 'light'")


def tr(cells, hl=()):
    return "<tr>" + "".join(f'<td class="{"num" if isinstance(c, (int, float)) or re.fullmatch(r"(&lt;)?[\d./ %(),-]+", str(c)) else ""}{" hl" if i in hl else ""}">{c}</td>' for i, c in enumerate(cells)) + "</tr>"


n_prim = len(ps)
cs_p, cv_p = sum(r["success"] for r in ps), sum(r["success"] for r in pv_)
ev_p, es_p = sum(exempt(r) for r in pv_), sum(exempt(r) for r in ps)
abstract_res = ""
if RES:
    cr, er = sum(r["success"] for r in RES["rows"]), sum(exempt(r) for r in RES["rows"])
    oa, ob, pvr = mcnemar([r["success"] for r in RES["rows"]], [r["success"] for r in pv_])
    _h = [(a["success"], b["success"]) for a, b in zip(RES["rows"], pv_) if a["seed"] - 760000 >= 32]
    abstract_res = (f" A per-step correction trained with PPO on top of v8 ({BEST['learn'][-1]['it'] + 1} iterations, {BEST['hyper']['n_envs']} robots) scored {cr} of {len(RES['rows'])} "
                    f"on the same layouts against {cv_p} for v8 alone, with {oa} layouts gained and {ob} lost (exact McNemar p {'= ' if pvr >= 1e-4 else ''}{pfmt(pvr)}). "
                    f"On the half of the layouts not used to choose its checkpoint it scored {sum(a for a, _ in _h)} of {len(_h)} against {sum(b for _, b in _h)}. "
                    f"With balls exempt from the lift tilt test the counts on all layouts are {er} and {ev_p}.")
    if RES_SET.get("gso") and RES_SET.get("pool"):
        abstract_res += (f" On objects outside its training set it scored {sum(r['success'] for r in RES_SET['gso']['rows'])} of {len(RES_SET['gso']['rows'])} on the household objects and "
                         f"{sum(r['success'] for r in RES_SET['pool']['rows'])} of {len(RES_SET['pool']['rows'])} on the pool.")
if ROUND2 and ROUND2[0]["walk"].get("prim") and RES_SET.get("pool") and RES_SET.get("gso"):
    _m = ROUND2[0]
    _n = lambda rec: sum(r["success"] for r in rec["rows"])  # noqa: E731
    _l = lambda rec: sum(r["lifted"] for r in rec["rows"])  # noqa: E731
    _p3i = int(BEST["walk"]["policy"].split("-")[-1])
    _last = next((w_ for t_, w_ in _m["other_tags"] if int(t_.split("-")[-1]) == _p3i + _m["hyper"]["iters"] - 1 and w_.get("pool") and w_.get("gso")), None)
    abstract_res += (f" A second round continued that policy for {_m['hyper']['iters']} iterations with objects from the pool's training split on "
                     f"{_m['hyper'].get('n_scanned', 0)} of the {_m['hyper']['n_envs']} robots. It did not raise the benchmark scores. The checkpoint chosen by the same rule, "
                     f"{int(_m['tag'].split('-')[-1]) - _p3i} iterations into the round, scored {_n(_m['walk']['prim'])} on primitives, {_n(_m['walk']['pool'])} on the pool and "
                     f"{_n(_m['walk']['gso'])} on the household objects, against {_n(RES)}, {_n(RES_SET['pool'])} and {_n(RES_SET['gso'])} before the round"
                     + (f"; the last checkpoint scored {_n(_last['prim'])}, {_n(_last['pool'])} and {_n(_last['gso'])}." if _last else "."))
html = f"""<meta name="claude:viewport-fit-sides" content="cover">
<title>G1 Walk-Carry Results</title>
<style>{style}</style>
<div class="page">
  <div class="theme-switch" role="group" aria-label="Colour theme">
    <button type="button" data-set-theme="light" aria-pressed="true">Light</button>
    <button type="button" data-set-theme="dark" aria-pressed="false">Dark</button>
    <button type="button" data-set-theme="auto" aria-pressed="false">Auto</button>
  </div>
  <header class="titleblock">
    <h1>Walking G1 Pick, Carry and Place on Scanned Objects, a Leg-Controller Probe, and a Per-Step Grab Correction</h1>
    <p class="byline">Steven Cleasby-Mayeda</p>
    <p class="affil">Oregon State University</p>
    <p class="date">11 November 2026</p>
  </header>

  <section class="abstract">
    <h2>Summary</h2>
    <p>Ran the walking Unitree G1 pick, carry and place benchmark on three object sets for the first time, with the scripted pinch and
    the v8 learned grab on identical layouts. v8 set the object down upright on the second table in {cv_p} of {n_prim} runs on primitive
    shapes against {cs_p} for the scripted pinch, in {set_rows[1][3]} of {set_rows[1][1]} against {set_rows[1][2]} on six scanned household
    objects, and in {set_rows[2][3]} of {set_rows[2][1]} against {set_rows[2][2]} on a 58-object pool (validation split, batch IK, one RTX 3090).
    In all three sets most failed runs end with the fingers closing and the object not rising.{abstract_res}</p>
    <p>Three measurement faults were found and are reported with their effect: scanned objects were judged at their base instead of where
    the hand holds them, balls that turn in the fingers are scored as not lifted, and the batch arm controller scores below the Pink controller
    on the same layouts. A replay of the fixed-base grasp policy's joint targets on NVIDIA's decoupled leg controller in MuJoCo tilted the
    pelvis {C['fixed/wbc/x1']['tilt_max']['median']:.1f} degrees with the arm alone and {C['fixed/replay/x1']['tilt_max']['median']:.1f} degrees
    with the policy's waist motion included.</p>
  </section>

  <article>

    <h2>Task and Settings</h2>
    <p>The robot is Isaac Lab's locomanipulation G1: the 29-joint Unitree G1 with Dex3 hands, NVIDIA's pretrained AGILE policy on the
    legs, and inverse kinematics on the arms. In each run the robot walks to a table, pinches one object with the right hand, lifts it 10 cm,
    backs away, turns, walks to a second table, and sets the object down. A run succeeds when the object was lifted, rests on the second table,
    is upright, the hand is clear of it, and the robot is standing. Each robot has its own scene: the first table is 0.8 to 1.8 m ahead and
    up to 0.3 m to either side, and the second table is at a 75 to 105 degree turn.</p>
    <p>Two grabs are compared on identical layouts. The scripted pinch closes the thumb onto the index finger across the object's narrow
    side. v8 is a small network that reads the object's class, size and position once and returns eight adjustments to that pinch (offsets,
    heading, hand tilt, closing depth and speed). It was trained on a fixed-base robot with six primitive shapes.</p>
    <div class="table-wrap"><table>
      <caption>Settings shared by every walk-carry run in this report. Source: the <code>settings</code> field of each record in <code>bench/results/batch-ik/</code>.</caption>
      <thead><tr><th>Setting</th><th>Value</th></tr></thead>
      <tbody>
        {tr(["Split and seeds", f"validation, layout seeds {B.BASE['walk_carry']['val']} upward"])}
        {tr(["Arm controller", f"batch IK: one damped least-squares step per 5 ms for all robots, waist posture weight 5000, step limit 0.02 rad"])}
        {tr(["Walking", f"command {st['vx']} m/s, stop target {st['stand']} m from the table edge, hip height {st['hip_height']} m, turn rate {st['yaw_rate']} rad/s"])}
        {tr(["v8 inputs", f"perception noise {st['noise']} cm per axis and 5 % on sizes; checkpoint sha256 {R['prim-v']['checkpoint'][:8]}"])}
        {tr(["Set-down", f"hand levels a standing object, lowers to {st['place_drop'] * 1000:.0f} mm above rest, holds {st['settle_place']} s, opens"])}
        {tr(["Physics", f"5 ms step, 20 ms control, angular damping {st['roll_damp']:g} on rolling shapes, object friction 0.7 to 1.5"])}
        {tr(["Machine", f"{st['gpu']}, host {st['host']}"])}
      </tbody></table></div>

    <h2>Three Object Sets</h2>
    <p>v8 placed more objects than the scripted pinch in every set. Layouts where only v8 succeeded outnumber layouts where only the scripted pinch succeeded by
    {min(r[4] / r[5] for r in set_rows):.1f} to {max(r[4] / r[5] for r in set_rows):.1f} times. Primitives are six shapes with random size, mass and friction. The household set is the validation
    share of 114 Google Scanned Objects that fit a one-hand pinch. The pool adds DexGraspNet shapes to the scanned objects and holds 58
    one-hand validation objects. Neither grab was trained on scanned or pool objects.</p>
    <figure><div class="chart">{FIG_SETS}</div>
      <figcaption>Walk-carry success on the validation split for each object set. The label above each bar is successes over runs. Every grab ran on identical
      layouts within a set.{" The correction was trained on primitive shapes only." if RES else ""} Source: <code>bench/results/batch-ik/val-walk-*-batch-3090.json</code>.</figcaption></figure>
    <div class="table-wrap"><table class="wide">
      <caption>Paired comparison per object set. "Only v8" counts layouts where v8 succeeded and the scripted pinch did not. The p-value is the exact McNemar test. The last column counts robots that fell. Settings: {SETTINGS}.</caption>
      <thead><tr><th>Object set</th><th class="num">Runs</th><th class="num">Scripted</th><th class="num">v8</th><th class="num">Only v8</th><th class="num">Only scripted</th><th class="num">p</th><th class="num">Fell: scripted, v8</th></tr></thead>
      <tbody>{"".join(tr([a, n, cs, cv, oa, ob, pfmt(pv), f"{fs}, {fv}"], hl=(3,)) for a, n, cs, cv, oa, ob, pv, fs, fv in set_rows)}</tbody></table></div>
    <p>The household set has six objects, which is too few to rank objects. v8 placed the {", the ".join(f"{nice(k).lower() if not nice(k).startswith(('Dell', 'Schleich', 'Perricone', 'Thomas')) else nice(k)} in {v[1]} of {v[2]} poses" for k, v in gso.items())}.
    On the pool, v8 placed {pool_hist.get(4, 0)} objects in all four poses and {pool_hist.get(0, 0)} objects in none
    ({", ".join(f"{pool_hist.get(k, 0)} objects in {k} of 4" for k in (1, 2, 3))}).</p>

    <h3>Where the runs fail</h3>
    <p>The largest group of failed v8 runs on primitives is the grab: the hand reaches the object, the fingers close, and the object stays on
    the table. Of {fail['prim'][1]} failed runs, {fail['prim'][0][order[0]]} stopped there. The carry and the set-down together account for
    {fail['prim'][0][order[3]] + fail['prim'][0][order[4]]}. On the pool the same stage leads with {fail['pool'][0][order[0]]} of {fail['pool'][1]} failed runs,
    followed by {fail['pool'][0][order[1]]} where the object rose and then tipped or slipped, and {fail['pool'][0][order[3]]} that were not set down upright.</p>
    <figure><div class="chart">{FIG_FAIL}</div>
      <figcaption>First stage at which each failed v8 run on primitives stopped ({fail['prim'][1]} failed runs of {n_prim}; balls are not tilt-tested at the lift here).
      Source: per-row stage flags in <code>val-walk-prim-v8-n64-batch-3090.json</code>.</figcaption></figure>
    <figure><div class="chart">{FIG_SHAPES}</div>
      <figcaption>Success by shape on primitives, 64 layouts per shape, with balls exempt from the lift tilt test. Lying cylinders and capsules are the two shapes
      where v8 places fewer than half. Source: the same records as Figure 1.</figcaption></figure>

    <h2>Measurement Faults Found</h2>
    <h3>Scanned objects were judged at their base</h3>
    <p>The first household run scored 0 of 96 for both grabs. The cause was the scoring. A scanned object's reference frame is at its lowest point, and the
    lift check measured the distance from that point to the pinch point. A doll 12.8 cm tall rose 9.9 cm in the hand (median of 16 poses) and was read as 9.5 cm away from it, which
    is over the 5 cm limit. The checks now use the point on the object where the pinch closes and the object's center. Primitive shapes are unaffected
    because their frame is their center. An earlier attempt on 28 October had stopped before any robot moved, because the object spawn function did not accept two arguments
    that Isaac Lab 2.3.2 passes, so no household walk record existed before this report.</p>
    <h3>Balls that turn in the fingers are scored as not lifted</h3>
    <p>The lift check requires the object's tilt to stay under 45 degrees, which catches a box hanging from one corner. It applies the same limit to balls, which have
    no upright direction. In the earlier Pink IK validation record, {nb} of 32 balls rose 10 cm, stayed within 1.5 cm of the pinch point, were carried and set down, and
    were scored as not lifted. The benchmark contract is unchanged. Table 3 reports both counts.</p>
    <div class="table-wrap"><table class="wide">
      <caption>Success under the contract and with balls exempt from the lift tilt test, recomputed from saved rows without rerunning. The p-values are exact McNemar tests of v8 against scripted.</caption>
      <thead><tr><th>Record</th><th class="num">Runs</th><th class="num">Scripted</th><th class="num">v8</th><th class="num">p</th><th class="num">Scripted,<br>balls exempt</th><th class="num">v8,<br>balls exempt</th><th class="num">p</th></tr></thead>
      <tbody>{"".join(tr([k, v[0], v[1], v[2], pfmt(v[5][2]), v[3], v[4], pfmt(v[6][2])]) for k, v in ball.items())}</tbody></table></div>
    <h3>Batch IK scores below Pink IK</h3>
    <p>Training thousands of robots at once requires the batch arm controller, so the comparisons in this report use it. On the {IK['n']} layouts scored with both,
    the scripted pinch succeeded {IK['pink_s']} times with Pink IK and {IK['batch_s']} with batch IK, and v8 {IK['pink_v']} and {IK['batch_v']}. With batch IK the pinch
    point ends the descent {IK['batch_err']:.1f} cm from its target at the median against {IK['pink_err']:.1f} cm, {IK['batch_fell']} robots fell during the grasp against
    {IK['pink_fell']}, and robots stop {IK['batch_gap']:.1f} cm from the table edge against {IK['pink_gap']:.1f} cm. The cause was not isolated. Layouts 32 to 63 of each
    shape had never been scored before: scripted {sum(r['success'] for r in new_s)} of {len(new_s)}, v8 {sum(r['success'] for r in new_v)} of {len(new_v)}.</p>

    <h2>Leg-Controller Probe</h2>
    <p>The fixed-base grasp policy dex-v6b lifts 65 of 72 validation objects on a welded base and 0 of 72 on the walking base, where its motion pitches AGILE's pelvis.
    The probe asked whether NVIDIA's decoupled whole-body controller tolerates the same motion. It replays the joint targets dex-v6b sent on the welded base, open loop, on the G1 in MuJoCo
    under the controller's balance policy, and measures the pelvis. With the arm alone the pelvis tilted {C['fixed/wbc/x1']['tilt_max']['median']:.1f} degrees at the median and moved the wrist
    {100 * C['fixed/wbc/x1']['wrist_grasp']['median']:.1f} cm. With the policy's waist motion also replayed it tilted {C['fixed/replay/x1']['tilt_max']['median']:.1f} degrees and moved the wrist
    {100 * C['fixed/replay/x1']['wrist_grasp']['median']:.1f} cm. No robot fell in either case. dex-v6b turns the waist 1.4 rad in yaw and 0.5 rad in pitch at the median to reach, so it is not usable unchanged on this controller either.</p>
    <figure><div class="chart">{FIG_WBC}</div>
      <figcaption>Largest pelvis tilt during the 3 s grasp motion, median over {C['fixed/replay/x1']['n']} validation layouts. The AGILE bar comes from the Isaac Lab recording with the policy
      acting on what it saw. The other bars are open-loop replays in MuJoCo. Source: <code>runs/wbc_probe/result.json</code>.</figcaption></figure>
    <div class="table-wrap"><table class="wide">
      <caption>Probe conditions, medians over {C['fixed/replay/x1']['n']} layouts with the 90th percentile in parentheses. Settings: MuJoCo {WBC['settings']['mujoco']}, {WBC['settings']['sim_dt'] * 1000:.0f} ms step,
      20 ms control, height command {WBC['settings']['height']} m, zero velocity command, arm gains 300 and 400 N m/rad with 25 and 5 N m limits, no table or object, model without finger joints.</caption>
      <thead><tr><th>Waist joints</th><th class="num">Tilt,<br>deg</th><th class="num">Drift in<br>1 s, cm</th><th class="num">Wrist moved by<br>the base, cm</th><th class="num">Waist<br>error, rad</th><th class="num">Fell</th></tr></thead>
      <tbody>{"".join(tr([lab, f"{C[k]['tilt_max']['median']:.1f} ({C[k]['tilt_max']['p90']:.1f})", f"{100 * C[k]['drift_1s']['median']:.1f}", f"{100 * C[k]['wrist_grasp']['median']:.1f} ({100 * C[k]['wrist_grasp']['p90']:.1f})", f"{C[k]['waist_track']['median']:.2f}", C[k]['fell']])
                         for lab, k in (("Replayed from the recording", "fixed/replay/x1"), ("Left to the leg policy (arm only)", "fixed/wbc/x1"), ("Sent as torso pitch and yaw commands", "fixed/cmd/x1"), ("Replayed, half speed", "fixed/replay/x0.5")))}</tbody></table></div>
    <p>Two limits apply. The replayed-waist condition overrides joints the leg policy controls, so its tilt mixes the waist's momentum with a policy whose outputs are ignored.
    The comparison with AGILE crosses simulators and compares open-loop replay with a policy in the loop. AGILE with the waist frozen and the policy's actions filtered still pitched 11.2 and 7.2 degrees
    in the first 0.5 s in Isaac Lab runs of 28 October, so for AGILE the waist is not the whole cause.</p>
"""
def hyper_rows(P):
    hy, ru = P["hyper"], P["hyper"].get("runner", {})
    alg, pol = ru.get("algorithm", {}), ru.get("policy", {})
    return [("Length", f"{hy['n_envs']} robots, {P['learn'][-1]['it'] + 1} iterations of {ru.get('num_steps_per_env')} steps, {hy['steps_per_episode']} steps per episode"),
            ("Reward", f"hand near the object {hy['w_reach']}, object rising {hy['w_rise']}, lifted at the end {hy['w_lifted']}, action size -{hy['w_rate']}"),
            ("PPO", f"learning rate {alg.get('learning_rate')} ({alg.get('schedule')}), gamma {alg.get('gamma')}, lambda {alg.get('lam')}, clip {alg.get('clip_param')}, "
                    f"value loss {'clipped' if alg.get('use_clipped_value_loss') else 'not clipped'}, entropy {alg.get('entropy_coef')}, {alg.get('num_learning_epochs')} epochs x {alg.get('num_mini_batches')} minibatches"),
            ("Start", f"reset {100 * hy['stand_mean']:.1f} cm from the edge (std {100 * hy['stand_std']:.1f}), {hy['settle_s']} s before the goal is read"
                      + (f", hands raised over {hy['raise_s']} s" if "raise_s" in hy else ", hands raised in one step")
                      + (", robots spread over the episode" if hy.get("desync") else ", all robots in step")),
            ("Networks", f"actor and critic {pol.get('actor_hidden_dims')}, ELU, observation normalization, starting noise {pol.get('init_noise_std')}, last actor layer zeroed"),
            ("Objects", f"six primitive shapes, training seeds {B.BASE['walk_carry']['train']} upward, pose redrawn each episode; base grab {hy['base']} with noise {hy['noise']} cm"),
            ("Machine", f"{hy.get('gpu')}")]


if PILOTS:
    names = [f"Pilot {P['name']}" for P in PILOTS]
    rows_h = [hyper_rows(P) for P in PILOTS]
    html += f"""
    <h2>Per-Step Grab Correction</h2>
    <p>v8 chooses its grasp once and cannot react when the closing fingers push a thin object away. The correction is a second policy that acts every 20 ms during the grab.
    It reads the object's position and velocity relative to the pinch point, the finger angles and how far they lag their targets, and the base tilt, and returns five targets:
    a pinch offset of up to {100 * BEST_DESC['max_off_m']:.0f} cm along three axes, a hand turn of up to {BEST_DESC['max_yaw']} rad, and a change of the finger closing of up to {BEST_DESC['max_grip']}.
    The applied correction moves {100 * BEST_DESC['alpha']:.0f} % of the way to its target per step. A zero output is exactly the v8 grab.</p>
    <p>Training places the robot where the walk stops, standing on AGILE's legs, and runs one {BEST_DESC['seconds']} s grab per episode on six primitive shapes. This differs from the plan in three ways:
    the correction sits on v8 instead of the scripted pinch, the robot is placed at the table instead of walking to it, and episodes do not start from saved successful grasps.</p>
    <p>A first version added each action to the correction directly. Exploration noise then summed over the episode into a pinch offset of about 2 cm (a calculation from the noise level,
    not a measurement). Its first 2,048 training episodes ended lifted at a rate of {P1_RATE:.2f}, against {ZERO['zero-v8']['lifted'] / ZERO['zero-v8']['n']:.2f} for the uncorrected grab on
    validation layouts, so that run was stopped after 15 iterations and the filtered form above replaced it.</p>
    <div class="table-wrap"><table class="wide">
      <caption>Training settings of each pilot. Source: <code>hyper.json</code> written by <code>isaac/grab_rl.py</code> for each run.</caption>
      <thead><tr><th>Setting</th>{"".join(f"<th>{n}</th>" for n in names)}</tr></thead>
      <tbody>{"".join("<tr><td>" + rows_h[0][k][0] + "</td>" + "".join(f"<td>{rh[k][1]}</td>" for rh in rows_h) + "</tr>" for k in range(len(rows_h[0])))}</tbody></table></div>
"""
    # the training scene against the benchmark
    z1, z2, zs = ZERO["zero-v8"], ZERO["zero2-v8"], ZERO["zero-scripted"]
    lift_v, lift_s = sum(lift_ok(r) for r in pv_), sum(lift_ok(r) for r in ps)
    g_old, g_new, g_bench = goal_read("g1111c-diag-s0.5.log"), goal_read("g1111c-zero2-v8.log"), goal_read("g1111c-walk-v8-repeat.log")
    if z1 and zs and g_old and g_bench:
        html += f"""
    <h3>The training scene against the benchmark</h3>
    <p>With no correction, the first training scene lifted {z1['lifted']} of {z1['n']} objects with v8 and {zs['lifted']} with the scripted pinch on the benchmark's own object layouts. The benchmark,
    after its walk, lifted {lift_v} and {lift_s} of the same objects. The first scene did not start the grab from the benchmark's state. It sent the hands 20 cm up in one step and read the goal
    {PILOTS[0]['hyper']['settle_s']} s later, with the robot tilted {g_old['tilt']:.1f} degrees, moving {g_old['speed']:.1f} cm/s, {g_old['gap']:.1f} cm from the table edge. The benchmark's robots at the
    same moment are tilted {g_bench['tilt']:.1f} degrees, moving {g_bench['speed']:.1f} cm/s, {g_bench['gap']:.1f} cm from the edge."""
        if z2 and g_new:
            html += f""" The second scene raises the hands over {z2['settings'].get('raise_s', 1.0)} s and reads the goal after {z2['settings']['settle_s']} s: {g_new['tilt']:.1f} degrees,
    {g_new['speed']:.1f} cm/s, {g_new['gap']:.1f} cm, and {z2['lifted']} of {z2['n']} lifted with v8."""
        html += "</p>\n"
        html += f"""
    <div class="table-wrap"><table class="wide">
      <caption>Objects lifted with no correction, by shape, 64 layouts each. The benchmark column counts lifts after its walk, with balls exempt from the tilt test as in the training scenes.
      Sources: <code>eval-zero-val-n384.json</code> of each scene and <code>val-walk-prim-v8-n64-batch-3090.json</code>.</caption>
      <thead><tr><th>Scene, v8 grab</th>{"".join(f'<th class="num">{s_.capitalize()}</th>' for s_ in SHAPES)}<th class="num">All</th></tr></thead>
      <tbody>
        {tr(["Benchmark, after the walk"] + [sum(lift_ok(r) for r in pv_ if r["shape"] == s_) for s_ in SHAPES] + [lift_v])}
        {tr(["First training scene (pilot p2)"] + [z1["by"][s_] for s_ in SHAPES] + [z1["lifted"]])}
        {tr(["Second training scene (pilot p3)"] + [z2["by"][s_] for s_ in SHAPES] + [z2["lifted"]]) if z2 else ""}
      </tbody></table></div>
"""
        bl = {(r["shape"], r["seed"]): lift_ok(r) for r in pv_}
        agree = lambda z: sum(z["rows"][k] == bl[k] for k in bl)  # noqa: E731
        html += f"""
    <p>Layout by layout, the first scene gives the benchmark's lift outcome on {agree(z1)} of {len(bl)} layouts"""
        if z2:
            html += f""" and the second on {agree(z2)}. The second scene matches the benchmark's count ({z2['lifted']} against {lift_v}) and still disagrees with it on {len(bl) - agree(z2)} individual layouts,
    so a correction that helps in the training scene is checked in the benchmark before it is counted."""
        html += "</p>\n"
    html += f"""
    <h3>Training</h3>
    <figure><div class="chart">{FIG_LEARN}</div>
      <figcaption>Share of training episodes that ended with the object lifted, one point per 15 iterations. Training episodes carry exploration noise and use training-split objects.
      The pilots use different start states (Table 5), so their levels are not directly comparable; the slope of each line is the learning signal. Source: <code>learn.json</code> of each run.</figcaption></figure>
"""
    vl = value_loss("g1111c-train-p2.log")
    big = [v for v in vl if v > 50]
    if big:
        html += f"""
    <p>Pilot p2's lift rate stayed between {min(r['lifted'] for r in PILOTS[0]['learn']):.2f} and {max(r['lifted'] for r in PILOTS[0]['learn']):.2f} for its whole run. Its critic did not learn the reward
    given at the end of the grab: the value loss was above 50 on {len(big)} of {len(vl)} logged updates (median of those {sorted(big)[len(big) // 2]:.0f}) from the first tenth of training to the last,
    and below 1 on {sum(v < 1 for v in vl)}. The end reward was {PILOTS[0]['hyper']['w_lifted']:g} with a clipped value loss, every robot ran in step so the end reward arrived in one update of about 7.5,
    and the grab's end was marked as a time-out. Pilot p3 changes those settings and the start state together (Table 5), so its result does not separate their effects.</p>
"""
        vl3 = value_loss("g1111c-train-p3.log")
        if vl3 and len(PILOTS) > 1:
            lr = PILOTS[-1]["learn"]
            html += f"""
    <p>In pilot p3 the value loss stayed at or below {max(vl3[10:]):.2f} after the first ten updates (median {sorted(vl3[10:])[len(vl3[10:]) // 2]:.3f}, {len(vl3)} updates), and the share of training
    episodes ending lifted went from {lr[0]['lifted']:.2f} in the first 15 iterations to {lr[-1]['lifted']:.2f} in the last 15, with exploration noise falling from {lr[0]['noise_std']:.3f} to {lr[-1]['noise_std']:.3f}.</p>
"""
    # checkpoints, grab alone, paired against no correction in the same scene
    rows_c = []
    for P in PILOTS:
        zk = "zero-v8" if P is PILOTS[0] else "zero2-v8"
        if not ZERO.get(zk) or not P["evals"]:
            continue
        Z = ZERO[zk]["rows"]
        for it, E in P["evals"]:
            keys = list(E)
            oa, ob, pv = mcnemar([E[k] for k in keys], [Z[k] for k in keys])
            new = [k for k in keys if k[1] - 760000 >= 32]
            rows_c.append([f"Pilot {P['name']}, iteration {it + 1}", sum(E.values()), sum(Z[k] for k in keys), oa, ob, pfmt(pv), f"{sum(E[k] for k in new)} / {sum(Z[k] for k in new)}"])
    if rows_c:
        same = ""
        if ZERO.get("zero-v8-repeat"):
            d = sum(ZERO["zero-v8"]["rows"][k] != ZERO["zero-v8-repeat"]["rows"][k] for k in ZERO["zero-v8"]["rows"])
            same = f" A second run of the uncorrected grab in the first scene changed {d} of {ZERO['zero-v8']['n']} outcomes."
        html += f"""
    <div class="table-wrap"><table class="wide">
      <caption>Saved checkpoints on the grab alone, 384 validation layouts, no exploration noise, each against the uncorrected v8 grab in the same training scene. "Gained" counts layouts lifted only with the
      correction. The last column is layouts 32 to 63 of each shape, with and without the correction.{same} Source: <code>eval-model_*-val-n384.json</code>.</caption>
      <thead><tr><th>Checkpoint</th><th class="num">Lifted</th><th class="num">No correction</th><th class="num">Gained</th><th class="num">Lost</th><th class="num">p</th><th class="num">Layouts 32 to 63</th></tr></thead>
      <tbody>{"".join(tr(r) for r in rows_c)}</tbody></table></div>
"""
    # in the full benchmark
    walks = [P for P in PILOTS if P["walk"]]
    if walks:
        rep_txt = ""
        if V8_REPEAT:
            d = sum(a["success"] != b["success"] for a, b in zip(pv_, V8_REPEAT["rows"]))
            rep_txt = (f" A second run of v8 alone on this machine scored {sum(r['success'] for r in V8_REPEAT['rows'])} with {d} of {len(pv_)} outcomes changed, "
                       f"so on one machine the paired counts measure the policies and not run-to-run variation. Across machines the same code has differed on about 9 of 96 layouts.")
        rows_w = []
        for P in walks:
            W = P["walk"]["rows"]
            assert [(r["shape"], r["seed"]) for r in W] == [(r["shape"], r["seed"]) for r in pv_]
            for lab, fn in (("contract", lambda r: r["success"]), ("balls exempt", exempt)):
                oa, ob, pv = mcnemar([fn(r) for r in W], [fn(r) for r in pv_])
                nw = [(fn(a), fn(b)) for a, b in zip(W, pv_) if a["seed"] - 760000 >= 32]
                rows_w.append([f"{P['walk']['policy']}, {lab}", sum(fn(r) for r in W), sum(fn(r) for r in pv_), oa, ob, pfmt(pv), f"{sum(a for a, _ in nw)} / {sum(b for _, b in nw)}",
                               f"{sum(r['fell'] for r in W)}, {sum(r['fell'] for r in pv_)}"])
        Wb = BEST["walk"]["rows"]
        import statistics
        used = [statistics.median(r["residual"][k] for r in Wb) for k in range(5)]
        html += f"""
    <h3>In the full benchmark</h3>
    <p>For each pilot, the checkpoint with the most lifts on layouts 0 to 31 of each shape ran inside the full walk-carry benchmark on the 384 validation layouts of Table 2.{rep_txt}</p>
    <div class="table-wrap"><table class="wide">
      <caption>v8 with the correction against v8 alone on identical layouts, under the contract and with balls exempt from the lift tilt test. Layouts 32 to 63 were not used to choose the checkpoint.
      The last column counts robots that fell. Settings as in Table 1. Sources: <code>val-walk-prim-v8+*-n64-batch-3090.json</code> and <code>val-walk-prim-v8-n64-batch-3090.json</code>.</caption>
      <thead><tr><th>Policy and rule</th><th class="num">With<br>correction</th><th class="num">v8<br>alone</th><th class="num">Gained</th><th class="num">Lost</th><th class="num">p</th><th class="num">Layouts<br>32 to 63</th><th class="num">Fell</th></tr></thead>
      <tbody>{"".join(tr(r) for r in rows_w)}</tbody></table></div>
    <p>By shape, with balls exempt, {BEST['walk']['policy']} against v8 alone: {", ".join(f"{s_} {sum(exempt(r) for r in Wb if r['shape'] == s_)} against {sum(exempt(r) for r in pv_ if r['shape'] == s_)}" for s_ in SHAPES)}.
    The largest correction it applies in a run has a median pinch offset of {1000 * used[0]:.1f}, {1000 * used[1]:.1f} and {1000 * used[2]:.1f} mm along the forward, left and up axes, a hand turn of
    {used[3]:.2f} rad, and a change of closing of {used[4]:.2f}, against limits of {1000 * BEST_DESC['max_off_m']:.0f} mm, {BEST_DESC['max_yaw']} rad and {BEST_DESC['max_grip']}.</p>
"""
        others = [(lab, RES_SET[k], R[k + "-v"], R[k + "-s"]) for lab, k in (("Household scans", "gso"), ("Object pool", "pool")) if RES_SET.get(k)]
        if others:
            rows_o = []
            for lab, rr, vv, ss in others:
                oa, ob, pv = mcnemar([r["success"] for r in rr["rows"]], [r["success"] for r in vv["rows"]])
                rows_o.append([lab, len(rr["rows"]), sum(r["success"] for r in ss["rows"]), sum(r["success"] for r in vv["rows"]), sum(r["success"] for r in rr["rows"]), oa, ob, pfmt(pv),
                               f"{sum(r['fell'] for r in rr['rows'])}, {sum(r['fell'] for r in vv['rows'])}"])
            html += f"""
    <p>The same checkpoint was then run on the two object sets it was not trained on. It placed more objects than v8 alone in both, by a smaller margin than on primitives.</p>
    <div class="table-wrap"><table class="wide">
      <caption>{BEST['walk']['policy']} on objects outside its training set, against v8 alone and the scripted pinch on identical layouts. "Gained" and "Lost" are against v8 alone, with the exact
      McNemar p-value. The last column counts robots that fell, with the correction and for v8 alone. Settings as in Table 1. Source: <code>val-walk-{{gso,pool}}-v8+*-batch-3090.json</code>.</caption>
      <thead><tr><th>Object set</th><th class="num">Runs</th><th class="num">Scripted</th><th class="num">v8<br>alone</th><th class="num">With<br>correction</th><th class="num">Gained</th><th class="num">Lost</th><th class="num">p</th><th class="num">Fell</th></tr></thead>
      <tbody>{"".join(tr(r, hl=(4,)) for r in rows_o)}</tbody></table></div>
"""
        fb, fv = collections.Counter(stage(r) for r in Wb if not exempt(r)), collections.Counter(stage(r) for r in pv_ if not exempt(r))
        html += f"""
    <p>The remaining failed runs of {BEST['walk']['policy']} stop at the same stage as before: the fingers closed and the object did not rise in {fb[order[0]]} runs ({fv[order[0]]} for v8 alone),
    the object rose and then tipped or slipped in {fb[order[1]]} ({fv[order[1]]}), and the hand moved the object before closing in {fb[order[2]]} ({fv[order[2]]}). Dropped while carrying:
    {fb[order[4]]} ({fv[order[4]]}). Not set down upright: {fb[order[3]]} ({fv[order[3]]}).</p>
"""
# --- Section: second round (pool objects added to training) ---------------------------------------------------------------------
R2 = []  # one entry per scored checkpoint: the rule's choice of each pilot first, then its other scored checkpoints
for _P in ROUND2:
    if _P["walk"].get("prim"):
        R2.append(dict(tag=_P["tag"], walk=_P["walk"], pilot=_P))
        R2 += [dict(tag=t, walk=w_, pilot=_P) for t, w_ in _P["other_tags"] if w_.get("prim")]
if ROUND2:
    P3 = PILOTS[-1]
    p3_tag = BEST["walk"]["policy"].split("+")[1]
    p3_it = int(p3_tag.split("-")[-1])
    p3_grab = dict(P3["evals"]).get(p3_it)
    main = ROUND2[0]
    hy = main["hyper"]
    n_pool_envs, n_pool_objs = hy.get("n_scanned", 0), hy.get("n_scanned_objects", 0)
    ctrl = next((P for P in ROUND2 if P["hyper"].get("objects") == "primitives"), None)
    S_FOURTH = 'fill="var(--accent2)" opacity=".45"'
    SETS3 = [("Primitives", "prim", 64), ("Object pool", "pool", 4), ("Household scans", "gso", 16)]

    def r2_hyper(P):
        h, ru = P["hyper"], P["hyper"].get("runner", {})
        alg = ru.get("algorithm", {})
        objs = (f"{h['n_envs'] - h.get('n_scanned', 0)} robots on the six primitive shapes and {h.get('n_scanned', 0)} robots on {h.get('n_scanned_objects', 0)} pool training objects"
                if h.get("objects") == "mix" else f"{h['n_envs']} robots on the six primitive shapes")
        return [("Start", f"{p3_tag} (first round), optimizer state not loaded"),
                ("Length", f"{h['iters']} iterations ({p3_it + 1} to {p3_it + h['iters']}), {ru.get('num_steps_per_env')} steps each, {h['n_envs']} robots"),
                ("Objects", objs + f", training seeds {B.BASE['walk_carry']['train']} upward"),
                ("Reward", f"hand near the object {h['w_reach']}, object rising {h['w_rise']}, lifted at the end {h['w_lifted']}, action size -{h['w_rate']}"),
                ("PPO", f"learning rate {alg.get('learning_rate')} ({alg.get('schedule')}), gamma {alg.get('gamma')}, lambda {alg.get('lam')}, clip {alg.get('clip_param')}, "
                        f"value loss {'clipped' if alg.get('use_clipped_value_loss') else 'not clipped'}, entropy {alg.get('entropy_coef')}"),
                ("Scene", f"reset {100 * h['stand_mean']:.1f} cm from the edge, hands raised over {h['raise_s']} s, goal read after {h['settle_s']} s, base grab {h['base']} with noise {h['noise']} cm"),
                ("Machine", f"{h.get('gpu')}")]

    rh = [r2_hyper(P) for P in ROUND2]
    html += f"""
    <h2>Second Round: Pool Objects Added to Training</h2>
    <p>The first round's correction was trained on primitive shapes only and its lift rate was still rising when training stopped. The second round continues that policy ({p3_tag}) for
    {hy['iters']} more iterations with {n_pool_envs} of the {hy['n_envs']} robots holding an object from the pool's training split ({n_pool_objs} objects) and the rest holding
    primitives. This is pilot {main['name']}.{f" Pilot {ctrl['name']} is the same continuation on primitives only. It separates the effect of longer training from the effect of the added objects." if ctrl else ""}</p>
    <p>No pool training object shares a name or a size signature with any validation or test object of the pool or the household set. The six household validation objects are six of the pool's 58
    validation objects. The measure of interest is how much of the object space a policy handles, so the pool results below are also broken down per object, by size and by source.</p>
    <div class="table-wrap"><table class="wide">
      <caption>Training settings of the second-round pilots. Settings not listed are those of pilot p3 in Table 5. Source: <code>hyper.json</code> of each run.</caption>
      <thead><tr><th>Setting</th>{"".join(f"<th>Pilot {P['name']}</th>" for P in ROUND2)}</tr></thead>
      <tbody>{"".join("<tr><td>" + rh[0][k][0] + "</td>" + "".join(f"<td>{x[k][1]}</td>" for x in rh) + "</tr>" for k in range(len(rh[0])))}</tbody></table></div>
"""
    if GATE.get("pool-zero") and GATE.get("pool-p3"):
        bz = {(r["name"], r["seed"]): lift_ok(r) for r in R["pool-v"]["rows"]}
        bp = {(r["name"], r["seed"]): lift_ok(r) for r in RES_SET["pool"]["rows"]}
        gz, gp = GATE["pool-zero"], GATE["pool-p3"]
        ag = lambda a, b: sum(a[k] == b[k] for k in a)  # noqa: E731
        html += f"""
    <h3>The training scene on pool objects</h3>
    <p>Before training, the training scene was run on the pool's {len(gz)} validation layouts and compared with the benchmark's lifts on the same layouts. With no correction it lifted {sum(gz.values())}
    objects against {sum(bz.values())} in the benchmark, with the same outcome on {ag(gz, bz)} layouts. With {p3_tag} it lifted {sum(gp.values())} against {sum(bp.values())}, with the same outcome on
    {ag(gp, bp)} layouts. The scene judges a scanned object at the point where the pinch closes and places it by its footprint center, as the benchmark does. Moving that code into one shared module
    left the benchmark's household result unchanged on all 96 layouts.</p>
"""
    series_l = [("p3, primitives", [(r["it"] + 1, r["lifted"]) for r in P3["learn"]], 'stroke="currentColor" opacity=".55"')]
    if hy.get("objects") == "mix":
        series_l += [(f"{main['name']}, primitives", [(r["it"] + 1, r["lifted_prim"]) for r in main["learn"]], 'stroke="var(--accent)"'),
                     (f"{main['name']}, pool objects", [(r["it"] + 1, r["lifted_pool"]) for r in main["learn"]], 'stroke="var(--accent2)"')]
    if ctrl:
        series_l.append((f"{ctrl['name']}, primitives", [(r["it"] + 1, r["lifted"]) for r in ctrl["learn"]], 'stroke="var(--accent)" stroke-dasharray="5 4"'))
    first, last = main["learn"][0], main["learn"][-1]
    html += f"""
    <h3>Training</h3>
    <figure><div class="chart">{multi_line(series_l, 0.8, "Training episodes ending lifted", "Training iteration")}</div>
      <figcaption>Share of training episodes that ended with the object lifted, one point per 15 iterations, for pilot p3 and the pilots continued from its iteration {p3_it}. Training episodes carry
      exploration noise and use training-split objects. Source: <code>learn.json</code> of each run.</figcaption></figure>
    <p>In pilot {main['name']} the lift rate of training episodes went from {first['lifted_prim']:.2f} to {last['lifted_prim']:.2f} on primitives and from {first['lifted_pool']:.2f} to
    {last['lifted_pool']:.2f} on pool objects between its first and last logged 15 iterations{f"; in pilot {ctrl['name']} it went from {ctrl['learn'][0]['lifted']:.2f} to {ctrl['learn'][-1]['lifted']:.2f} on primitives" if ctrl else ""}.
    Exploration noise ended at {last['noise_std']:.3f}.</p>
"""
    rows_c2 = []
    if p3_grab:
        for P in ROUND2:
            for it, E in P["evals"]:
                keys = list(E)
                oa, ob, pv = mcnemar([E[k] for k in keys], [p3_grab[k] for k in keys])
                new = [k for k in keys if k[1] - 760000 >= 32]
                rows_c2.append([f"Pilot {P['name']}, iteration {it + 1}", sum(E.values()), sum(p3_grab[k] for k in keys), oa, ob, pfmt(pv), f"{sum(E[k] for k in new)} / {sum(p3_grab[k] for k in new)}"])
    if rows_c2:
        counts = [r[1] for r in rows_c2]
        halves = [int(r[6].split(" / ")[0]) for r in rows_c2]
        firsts = [c_ - h_ for c_, h_ in zip(counts, halves)]
        html += f"""
    <p>Checkpoints saved 50 iterations apart differ by more than the two rounds do. On the 384 primitive validation layouts the six second-round checkpoints lifted between {min(counts)} and
    {max(counts)} objects, and {p3_tag} lifted {rows_c2[0][2]}. The selection rule counts lifts on layouts 0 to 31 of each shape: the six checkpoints scored {", ".join(str(v) for v in firsts)} there, so
    the rule's choice rests on a margin of {sorted(firsts)[-1] - sorted(firsts)[-2]} of 192 layouts.</p>
"""
        pe = ""
        html += f"""
    <div class="table-wrap"><table class="wide">
      <caption>Second-round checkpoints on the grab alone, 384 primitive validation layouts, no exploration noise, each against {p3_tag} in the same training scene. The last column is layouts
      32 to 63 of each shape, for the checkpoint and for {p3_tag}. Source: <code>eval-model_*-val-n384.json</code>.</caption>
      <thead><tr><th>Checkpoint</th><th class="num">Lifted</th><th class="num">{p3_tag}</th><th class="num">Gained</th><th class="num">Lost</th><th class="num">p</th><th class="num">Layouts 32 to 63</th></tr></thead>
      <tbody>{"".join(tr(r) for r in rows_c2)}</tbody></table></div>
"""
    if R2:
        key_of = lambda r: (r.get("name", r["shape"]), r["seed"])  # noqa: E731
        base_of = {"prim": R["prim-v"], "pool": R["pool-v"], "gso": R["gso-v"]}
        p3_of = {"prim": RES, "pool": RES_SET.get("pool"), "gso": RES_SET.get("gso")}
        rows_b, groups2 = [], []
        for lab, k, per in SETS3:
            v8r, p3r = base_of[k]["rows"], p3_of[k]["rows"]
            n = len(v8r)
            vals, texts = [100 * sum(r["success"] for r in v8r) / n, 100 * sum(r["success"] for r in p3r) / n], [f"{sum(r['success'] for r in v8r)}", f"{sum(r['success'] for r in p3r)}"]
            rows_b.append([lab, f'<span class="nw">v8+{p3_tag}</span>', n, sum(r["success"] for r in p3r), *mcnemar([r["success"] for r in p3r], [r["success"] for r in v8r])[:2],
                           pfmt(mcnemar([r["success"] for r in p3r], [r["success"] for r in v8r])[2]), "", "", "", sum(r["fell"] for r in p3r)])
            for P in R2:
                W = P["walk"].get(k)
                if not W:
                    vals.append(None)
                    texts.append("")
                    continue
                w = W["rows"]
                assert [key_of(r) for r in w] == [key_of(r) for r in v8r] == [key_of(r) for r in p3r], "paired runs share layouts"
                a1, b1, q1 = mcnemar([r["success"] for r in w], [r["success"] for r in v8r])
                a2, b2, q2 = mcnemar([r["success"] for r in w], [r["success"] for r in p3r])
                rows_b.append([lab, f'<span class="nw">{W["policy"]}</span>', n, sum(r["success"] for r in w), a1, b1, pfmt(q1), a2, b2, pfmt(q2), sum(r["fell"] for r in w)])
                vals.append(100 * sum(r["success"] for r in w) / n)
                texts.append(f"{sum(r['success'] for r in w)}")
            groups2.append((f"{lab}\n{n} runs", vals, texts))
        series2 = [("v8 alone", S_BASE), (f"+ {p3_tag}", S_ACC)] + [(f"+ {P['tag']}", S_THIRD if i == 0 else S_FOURTH) for i, P in enumerate(R2)]
        Wm = main["walk"]["prim"]["rows"] if main["walk"].get("prim") else None
        html += f"""
    <h3>In the full benchmark</h3>
    <p>The second round did not raise the benchmark score on any object set. Against v8+{p3_tag} on identical layouts, each second-round checkpoint gained and lost about the same number of layouts
    (table below).</p>
    <p>For each second-round pilot, the checkpoint with the most lifts on primitive layouts 0 to 31 ran inside the full walk-carry benchmark on the validation layouts of all three object sets. This is
    the rule used for pilot p3, so pool and household layouts took no part in choosing a checkpoint.{"".join(f" Checkpoint {E['tag']}, the last of the round and not the rule's choice, was scored on the same layouts as well, because the chosen checkpoint is only {int(E['pilot']['tag'].split('-')[-1]) - p3_it} iterations into the round." for E in R2 if E["tag"] != E["pilot"]["tag"])}</p>
    <figure><div class="chart">{bars(groups2, series2, 80, "Walk-carry success", unit="%")}</div>
      <figcaption>Walk-carry success on the validation split for v8 alone, v8 with the first-round correction, and v8 with each second-round correction. The label above each bar is the number of
      successful runs. All policies ran on identical layouts within a set. Source: <code>bench/results/batch-ik/val-walk-*-batch-3090.json</code>.</figcaption></figure>
    <div class="table-wrap"><table class="wide">
      <caption>Paired comparisons on identical layouts. "Gained" and "Lost" count layouts where only the row's policy or only the reference succeeded, with the exact McNemar p-value. The last column
      counts robots that fell. Settings as in Table 1. Source: the same records as the figure above.</caption>
      <thead><tr><th rowspan="2">Object set</th><th rowspan="2">Policy</th><th class="num" rowspan="2">Runs</th><th class="num" rowspan="2">Success</th>
      <th class="num" colspan="3">Against v8 alone</th><th class="num" colspan="3">Against v8+{p3_tag}</th><th class="num" rowspan="2">Fell</th></tr>
      <tr><th class="num">Gained</th><th class="num">Lost</th><th class="num">p</th><th class="num">Gained</th><th class="num">Lost</th><th class="num">p</th></tr></thead>
      <tbody>{"".join(tr(r, hl=(3,)) for r in rows_b)}</tbody></table></div>
"""
        if Wm:
            p3w = RES["rows"]
            h_new = [(a["success"], b["success"], c["success"]) for a, b, c in zip(Wm, p3w, pv_) if a["seed"] - 760000 >= 32]
            fb, f3 = collections.Counter(stage(r) for r in Wm if not exempt(r)), collections.Counter(stage(r) for r in p3w if not exempt(r))
            html += f"""
    <p>On primitive layouts 32 to 63, which were not used to choose any checkpoint, {main['walk']['prim']['policy']} scored {sum(a for a, _, _ in h_new)} of {len(h_new)}, v8+{p3_tag} {sum(b for _, b, _ in h_new)},
    and v8 alone {sum(c for _, _, c in h_new)}. With balls exempt from the lift tilt test the counts on all primitive layouts are {sum(exempt(r) for r in Wm)}, {sum(exempt(r) for r in p3w)} and {ev_p}.
    By shape, with balls exempt, {main['walk']['prim']['policy']} against v8+{p3_tag}: {", ".join(f"{s_} {sum(exempt(r) for r in Wm if r['shape'] == s_)} against {sum(exempt(r) for r in p3w if r['shape'] == s_)}" for s_ in SHAPES)}.
    Failed primitive runs in which the fingers closed and the object did not rise: {fb[order[0]]} ({f3[order[0]]} for v8+{p3_tag}). Runs in which the object rose and then tipped or slipped:
    {fb[order[1]]} ({f3[order[1]]}).</p>
"""
        if main["walk"].get("pool"):
            def per_object(rows):
                c = collections.Counter()
                for r in rows:
                    c[r["name"]] += r["success"]
                names = {r["name"] for r in rows}
                h = collections.Counter(c[n_] for n_ in names)
                return h
            pol3 = [("v8 alone", R["pool-v"]["rows"], S_BASE), (f"+ {p3_tag}", RES_SET["pool"]["rows"], S_ACC), (f"+ {main['tag']}", main["walk"]["pool"]["rows"], S_THIRD)]
            pol3 += [(f"+ {E['tag']}", E["walk"]["pool"]["rows"], S_FOURTH) for E in R2 if E["pilot"] is main and E["tag"] != main["tag"] and E["walk"].get("pool")]
            hists = [per_object(rows) for _, rows, _ in pol3]
            n_obj = len({r["name"] for r in pol3[0][1]})
            cov = bars([(f"{k} of 4 poses", [100 * h.get(k, 0) / n_obj for h in hists], [str(h.get(k, 0)) for h in hists]) for k in range(5)],
                       [(nm, st_) for nm, _, st_ in pol3], 60, "Share of pool objects", unit="%")
            html += f"""
    <h3>Coverage of the object space</h3>
    <figure><div class="chart">{cov}</div>
      <figcaption>The pool's {n_obj} validation objects grouped by the number of their four poses in which each policy completed the task. The label above each bar is the number of objects.
      Source: <code>val-walk-pool-*-batch-3090.json</code>.</figcaption></figure>
    <p>Objects placed in at least one pose: {", ".join(f"{nm.replace('+ ', 'v8+')} {n_obj - h.get(0, 0)} of {n_obj}" for (nm, _, _), h in zip(pol3, hists))}. Objects placed in all four poses:
    {", ".join(f"{nm.replace('+ ', 'v8+')} {h.get(4, 0)}" for (nm, _, _), h in zip(pol3, hists))}.</p>
"""
            if Path(args.pool_index).exists():
                idx = {e["name"]: e for e in load(args.pool_index)["objects"]}
                bins = [("Narrow side under 2 cm", lambda e: e["width_m"] < 0.02), ("Narrow side 2 to 3 cm", lambda e: 0.02 <= e["width_m"] < 0.03),
                        ("Narrow side 3 to 4 cm", lambda e: 0.03 <= e["width_m"] < 0.04), ("Narrow side 4 to 5 cm", lambda e: e["width_m"] >= 0.04),
                        ("Height under 3 cm", lambda e: e["height_m"] < 0.03), ("Height 3 to 6 cm", lambda e: 0.03 <= e["height_m"] < 0.06), ("Height 6 cm and over", lambda e: e["height_m"] >= 0.06),
                        ("Scanned household objects", lambda e: e["source"] == "gso"), ("DexGraspNet shapes", lambda e: e["source"] == "dgn")]
                rows_s = []
                for lab, fn in bins:
                    sel = [fn(idx[r["name"]]) for r in pol3[0][1]]
                    if not any(sel):
                        continue
                    rows_s.append([lab, len({r["name"] for r, k in zip(pol3[0][1], sel) if k}), sum(sel)] + [sum(r["success"] for r, k in zip(rows, sel) if k) for _, rows, _ in pol3])
                html += f"""
    <div class="table-wrap"><table class="wide">
      <caption>Walk-carry successes on the pool's validation objects by object size and source, four poses per object. Sizes are from the pool index. Source: <code>val-walk-pool-*-batch-3090.json</code>
      and <code>pool_g1_index.json</code>.</caption>
      <thead><tr><th>Group</th><th class="num">Objects</th><th class="num">Runs</th>{"".join(f'<th class="num">{nm.replace("+ ", "v8+")}</th>' for nm, _, _ in pol3)}</tr></thead>
      <tbody>{"".join(tr(r) for r in rows_s)}</tbody></table></div>
"""
        if main["walk"].get("pool"):
            at_rest = lambda r: bool(r["lifted"] and r["carried"] and r["placed"] and r["released"] and r["standing"])  # noqa: E731
            pidx = {e["name"]: e for e in load(args.pool_index)["objects"]} if Path(args.pool_index).exists() else {}
            top_z = 0.75 + R["pool-v"]["settings"]["table_z"]
            rows_u = []
            hh = [(E["tag"], [r for r in E["walk"]["gso"]["rows"] if at_rest(r) and not r["upright"]]) for E in R2 if E["pilot"] is main and E["walk"].get("gso")]
            _moved = lambda r: abs(r["end_obj"][2] - top_z - pidx[f"gso/{r['name']}"]["height_m"] / 2) >= 0.006  # noqa: E731
            _hn = [r for _, nu_ in hh for r in nu_]
            hh_txt = (f"; on the household objects, {'all' if all(_moved(r) for r in _hn) else str(sum(_moved(r) for r in _hn)) + ' of the'} {len(_hn)} not-upright objects of the second-round checkpoints ended at another height than they started"
                      if pidx and _hn else "")
            for nm, rows, _ in pol3:
                nu = [r for r in rows if at_rest(r) and not r["upright"]]
                same = sum(abs(r["end_obj"][2] - top_z - pidx[r["name"]]["height_m"] / 2) < 0.006 for r in nu) if pidx else ""
                rows_u.append([f'<span class="nw">{nm.replace("+ ", "v8+")}</span>', sum(r["success"] for r in rows), sum(at_rest(r) for r in rows), len(nu), same, sum(r["tilt_hold"] >= B.UPRIGHT_DEG for r in nu)])
            html += f"""
    <h3>Set-down orientation on pool objects</h3>
    <p>On the pool, more runs end with the object at rest on the second table and the hand clear than the contract counts as successes. The contract's upright test for a scanned object measures how
    far the object's vertical axis has turned from its starting attitude and allows {B.UPRIGHT_DEG:g} degrees. An object that lies on its side, such as a bottle, fails that test when it rolls about its
    long axis after the hand opens, although it rests as it did on the first table. Primitive lying shapes are tested differently, by the angle of their long axis from horizontal. The contract was not
    changed, and both counts are given. The second count is not a corrected score: it also includes objects that tipped onto another face{hh_txt}. About half of the not-upright objects
    were already turned {B.UPRIGHT_DEG:g} degrees or more in the hand at the end of the lift, where the lift test allows {B.LIFTED_TILT_DEG:g} degrees.</p>
    <div class="table-wrap"><table class="wide">
      <caption>Pool validation runs ({len(pol3[0][1])} per policy) under the contract and with the orientation test left out. The fifth column counts not-upright objects whose center ended within 6 mm of
      its starting height above the table, and the last column those already turned in the hand at the end of the lift. Source: <code>val-walk-pool-*-batch-3090.json</code> and <code>pool_g1_index.json</code>.</caption>
      <thead><tr><th>Policy</th><th class="num">Success under<br>the contract</th><th class="num">At rest on table B,<br>any orientation</th><th class="num">Of those,<br>not upright</th><th class="num">Not upright, center at<br>its starting height</th><th class="num">Not upright, turned {B.UPRIGHT_DEG:g}&deg;<br>or more in the hand</th></tr></thead>
      <tbody>{"".join(tr(r) for r in rows_u)}</tbody></table></div>
"""
        ev_rows = []
        for (pn, it), E_ in sorted(POOL_EV.items()):
            wf = BI / f"val-walk-pool-v8+{pn}-{it}-batch-3090.json"
            bl = sum(bool(r["lifted"]) for r in load(wf)["rows"]) if wf.exists() else ""
            ev_rows.append([f"Pilot {pn}, iteration {it + 1}", sum(E_.values()), bl])
        if len(ev_rows) >= 3:
            p3c = [r[1] for r in ev_rows if r[0].startswith("Pilot p3")]
            p4c = [r[1] for r in ev_rows if not r[0].startswith("Pilot p3")]
            html += f"""
    <h3>Pool objects in the training scene, by checkpoint</h3>
    <p>In the training scene, without the walk, the second-round checkpoints lifted {min(p4c)} to {max(p4c)} of the pool's {len(next(iter(POOL_EV.values())))} validation objects and the first-round
    checkpoints {min(p3c)} to {max(p3c)}. Where a checkpoint also ran in the benchmark, the count of objects lifted after the walk is given beside it.</p>
    <div class="table-wrap"><table>
      <caption>Objects lifted on the pool's {len(next(iter(POOL_EV.values())))} validation layouts in the training scene, per checkpoint, and in the benchmark after the walk for the checkpoints that ran
      there. Sources: <code>eval-model_*-pool-val-n232.json</code> and <code>val-walk-pool-*-batch-3090.json</code>.</caption>
      <thead><tr><th>Checkpoint</th><th class="num">Lifted in the training scene</th><th class="num">Lifted in the benchmark</th></tr></thead>
      <tbody>{"".join(tr(r) for r in ev_rows)}</tbody></table></div>
"""
        ctrl_done = bool(ctrl and ctrl["walk"].get("prim"))
        if not ctrl_done:
            html += f"""
    <p>The primitives-only continuation was not run. With no gain from the second round, there is no effect to assign to either longer training or the added objects.</p>
"""
        if ctrl and ctrl["walk"].get("prim") and main["walk"].get("prim"):
            parts = []
            for lab, k, per in SETS3:
                a, b = main["walk"].get(k), ctrl["walk"].get(k)
                if a and b:
                    oa, ob, pv = mcnemar([r["success"] for r in a["rows"]], [r["success"] for r in b["rows"]])
                    parts.append(f"{lab.lower()} {sum(r['success'] for r in a['rows'])} against {sum(r['success'] for r in b['rows'])} ({oa} gained, {ob} lost, p = {pfmt(pv)})")
            html += f"""
    <p>Pool objects in training against the same extra training without them, {main['walk']['prim']['policy']} against {ctrl['walk']['prim']['policy']} on identical layouts: {"; ".join(parts)}.</p>
"""
meas_f = LOCAL / "measured.json"
if ROUND2 and meas_f.exists():
    M = load(meas_f)
    thr = []
    f4 = LOCAL / "logs" / "g1111d-train-p4.log"
    if f4.exists():
        v = sorted(int(x) for x in re.findall(r"Computation: (\d+) steps/s", f4.read_text(errors="ignore"))[:6])  # the first 6 iterations, as the probes ran
        thr.append((1024, v[len(v) // 2], M["alone_1024"]["gpu_mib"], M["alone_1024"]["free_ram_mb"]))
    fq = LOCAL / "logs" / "g1111d-queue.log"
    if fq.exists():
        for m_ in re.finditer(r"PROBE (\d+) robots: steps/s per iteration ([\d ]+); GPU memory peak (\d+) MiB; lowest free RAM (\d+) MB", fq.read_text(errors="ignore")):
            v = sorted(int(x) for x in m_.group(2).split())
            thr.append((int(m_.group(1)), v[len(v) // 2], int(m_.group(3)), int(m_.group(4))))
    okr = [r for r in (R2[0]["walk"]["prim"]["rows"] if R2 else RES["rows"]) if r["success"]]
    med_ = lambda k_: sorted(r[k_] for r in okr)[len(okr) // 2]  # noqa: E731
    back_s = st["back"] / 0.25
    place_s = (1.5 + 1.0 + 0.8) * st["slow_place"] + st["settle_place"]
    task_s = med_("arrive_s") + 1.5 + _GR.T_TOTAL + 1.0 + back_s + med_("turn_s") + 0.5 + med_("walk_b_s") + 1.0 + place_s
    two = M["two_at_once"]
    html += f"""
    <h2>Task Time and Simulation Throughput</h2>
    <p>A successful run takes about {task_s:.0f} s from the first step to the open hand on the second table, and about {task_s + 3.5:.0f} s with the hand cleared and the step back that end the
    recorded run. This is a sum of the scripted phase lengths and the medians of the three recorded
    walking times over {len(okr)} successful primitive runs, not a timed run: walk to the first table {med_('arrive_s'):.1f} s, settle 1.5 s, grab {_GR.T_TOTAL} s, raise 1.0 s, back off {back_s:.1f} s,
    turn {med_('turn_s'):.1f} s, walk to the second table {med_('walk_b_s'):.1f} s, settle 1.5 s, set-down {place_s:.1f} s. The grab and its settle are {100 * (1.5 + _GR.T_TOTAL) / task_s:.0f} % of the
    total and run on a fixed schedule.</p>
    <p>Training time is simulation time: collecting 48 steps from every robot took {thr[0][0] * 48 / thr[0][1]:.1f} s of each iteration and the PPO update about 0.15 s. The training runs in this report
    used {thr[0][0]:,} robots, where about {thr[0][0] * 48 // int(R2[0]['pilot']['hyper']['steps_per_episode']) if R2 else 0} grabs end per update. With {thr[-1][0]:,} robots in one process the same
    scene produced {thr[-1][1] / thr[0][1]:.1f} times the robot-steps per second, so a larger batch gives more samples per iteration for nearly the same time. Each row of the table is the median of the
    first six iterations of a run. Two {thr[0][0]:,}-robot trainings on the card at once took {two['iter_s_a']} s and {two['iter_s_b']} s per iteration against {two['iter_s_alone']} s for one alone,
    a gain of {100 * ((1 / two['iter_s_a'] + 1 / two['iter_s_b']) * two['iter_s_alone'] - 1):.0f} % in total throughput.</p>
    <div class="table-wrap"><table>
      <caption>Simulation throughput of the mixed training scene (half primitives, half pool objects) by batch size on one RTX 3090. Card memory is the whole card's memory in use. Sources: the
      <code>Computation</code> lines of each run's log and <code>measured.json</code>.</caption>
      <thead><tr><th class="num">Robots in one process</th><th class="num">Robot-steps per second</th><th class="num">Seconds per iteration of 48 steps</th><th class="num">Card memory, MiB</th><th class="num">Lowest free RAM, MB</th></tr></thead>
      <tbody>{"".join(tr([f"{n_:,}", f"{v_:,}", f"{n_ * 48 / v_:.1f}", f"{g_:,}", f"{r_:,}"]) for n_, v_, g_, r_ in thr)}</tbody></table></div>
"""
REPLAY2 = ""
_c2 = Path(args.clips).parent / "clips_1111b"
if ROUND2 and (_c2 / "clips.json").exists():
    _side = next(iter(_c2.glob("*.json")), None)
    _rr = load(next(p_ for p_ in _c2.glob("walk-carry-*.json")))["rows"]
    REPLAY2 = (f" Four more clips at the top of that page show pool objects with v8+{ROUND2[0]['tag']}, from a {len(_rr)}-robot recording with one pose per object ({sum(r['success'] for r in _rr)} of "
               f"{len(_rr)} succeeded): a jar and a scanned toy lion picked, carried and set down, a sauce bottle that is set down and then rolls about its long axis, and a watch that stays on the "
               f"first table when the fingers close.")
html += f"""
    <h2>What Was Not Done</h2>
    <p>The test split was not run: it was scored once on 29 October and stays reserved. {"The first-round correction was trained on primitive shapes only and the second round added pool objects; neither was trained with the walk or by starting" if ROUND2 else "The correction was trained on primitive shapes only, without the walk, and without starting"}
    episodes from saved successful grasps{", and it was not run on the household or pool objects" if not (RES_SET.get("gso") and RES_SET.get("pool")) else ". Each run on the household and pool objects is a single run"}.
    No scripted-pinch run with the correction was made.{" The second round is one run from one seed, and no run used the larger batch that the throughput table supports." if ROUND2 else ""} The leg-controller probe has no matching
    arm-only replay on AGILE in Isaac Lab, and the NVIDIA controller was not put under the walk-carry task. The cause of the batch IK gap was not isolated. All runs used one desktop RTX 3090;
    none ran on the cluster.</p>

    <h2>Replays</h2>
    <p>Four runs of the v8 grab{f" from a 24-robot recording on validation layouts ({REC_N[0]} of {REC_N[1]} succeeded)" if REC_N else ""}, drawn in MuJoCo from the recorded Isaac Lab motion with a camera that
    follows the robot, are on the project's replay page: <a href="{args.replays_url}">G1 Grasp Replays</a>. They show a box and a lying cylinder picked, carried and set down, a lying cylinder
    that stays on the first table when the fingers close, and a ball that is carried and set down and scored as not lifted.{REPLAY2}</p>
"""
if clips:
    html += """    <div class="clips">""" + "".join(
        f'<figure><video controls muted loop playsinline preload="metadata" src="data:video/mp4;base64,{b}"></video><figcaption>{c["shape"].capitalize()}, layout seed {c["seed"]}: {c["what"]}.</figcaption></figure>'
        for c, b in clips) + "</div>"
html += f"""
    <h2 class="unnumbered">Sources</h2>
    <section class="references"><ol>
      <li>Isaac Lab 2.3.2, task <code>Isaac-PickPlace-Locomanipulation-G1-Abs-v0</code>, with NVIDIA's pretrained AGILE locomotion policy. <a href="https://github.com/isaac-sim/IsaacLab">github.com/isaac-sim/IsaacLab</a>.</li>
      <li>NVlabs, GR00T Whole-Body Control, <code>decoupled_wbc</code>, <code>GR00T-WholeBodyControl-Balance.onnx</code> (code Apache 2.0, weights NVIDIA Open Model License). <a href="https://github.com/NVlabs/GR00T-WholeBodyControl">github.com/NVlabs/GR00T-WholeBodyControl</a>.</li>
      <li>Google Scanned Objects (CC BY 4.0), through <code>kevinzakka/mujoco_scanned_objects</code>; DexGraspNet object meshes (CC BY-NC).</li>
      <li>MuJoCo Menagerie, <code>unitree_g1</code> (BSD-3-Clause), used to draw the replays.</li>
      <li>rsl_rl 3.1.2 (PPO), through Isaac Lab's <code>isaaclab_rl</code> wrapper.</li>
      <li>Records and code: <code>bench/results/batch-ik/</code>, <code>runs/wbc_probe/result.json</code>, <code>isaac/walk_grasp.py</code>, <code>isaac/grab_rl.py</code>, <code>isaac/grab_residual.py</code>, <code>isaac/hull_objects.py</code>, <code>tools/wbc_probe.py</code>, <code>tools/report_g1_1111.py</code>.</li>
    </ol></section>
  </article>
</div>
<script>{script}</script>
"""
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(html, encoding="utf-8", newline="\n")
print(f"wrote {out} ({len(html) / 1e6:.2f} MB); pilots {[P['name'] for P in PILOTS]}, benchmark runs {[P['walk']['policy'] for P in PILOTS if P['walk']]}, clips embedded {len(clips)}")
