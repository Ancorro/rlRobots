"""Plan-only planner benchmark: can a language model small enough for the 4060 Ti turn open requests into the right
pick-and-place moves? No physics: each plan is checked against the request's exact goal (so101/planner.py).

  python plan_bench.py make [--scenes=30] [--seed=0]       requests -> runs/plan/requests.jsonl (both object sets)
      [--bench=big --per-family=300 --max-start=3 --both-trays=0.3]   a separate, balanced set in runs/plan/big/
      (every command takes --bench=NAME to work on that set)
  python plan_bench.py run --model=KEY [--think=on|off] [--max-tokens=8192]
                                                           one model (vLLM, cluster: sbatch vla/plan.sbatch ...)
  python plan_bench.py ollama --model=gemma4:12b --tag=gemma4-12b-q4 --think=on   same, quantized, via Ollama
                                                           (cluster: sbatch vla/plan_ollama.sbatch ...)
  python plan_bench.py make-horizon --bench=horizon [--horizons=1,2,4,8,12,16,24,32 --per=50]   long move lists
  python plan_bench.py horizon-report --bench=horizon    success and time to the first move vs list length
  python plan_bench.py baselines                           oracle (checks the checker: must be 100%) and random
  python plan_bench.py report                              table + runs/plan/comparison.png
  python plan_bench.py examples [--runs=a-on,b-off]        a few requests with each model's plan -> runs/plan/examples.json

Scene text comes from the simulator's true state (docs/plan.md: exact positions first; perception later).
Results: runs/plan/<KEY>-<think>.jsonl, one line per request (answer, parsed moves, ok, output tokens).
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from so101 import planner

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
OUT = Path("runs/plan") / ARGS.get("bench", "")  # --bench=NAME: a separate request set and its results in its own folder
REQS = OUT / "requests.jsonl"
# key: (Hugging Face id, input kinds besides text (switched off: text-only here), how thinking is switched,
#       temperature (model card's advice))
MODELS = {"qwen3.5-9b": ("Qwen/Qwen3.5-9B", ("image", "video"), "enable_thinking", 0.6),
          "qwen3.5-4b": ("Qwen/Qwen3.5-4B", ("image", "video"), "enable_thinking", 0.6),
          "gemma4-12b": ("google/gemma-4-12B-it", ("image", "video", "audio"), "enable_thinking", 1.0),
          "gemma4-e4b": ("google/gemma-4-E4B-it", ("image", "video", "audio"), "enable_thinking", 1.0),
          "gpt-oss-20b": ("openai/gpt-oss-20b", (), "reasoning_effort", 1.0),
          "cosmos-reason2-8b": ("nvidia/Cosmos-Reason2-8B", ("image", "video"), "system", 0.6),
          "qwen3.8-27b": ("Qwen/Qwen3.8-27B", ("image", "video"), "enable_thinking", 0.6),  # does not fit the 4060 Ti: the ceiling
          "qwen3.8-27b-fp8": ("Qwen/Qwen3.8-27B-FP8", ("image", "video"), "enable_thinking", 0.6)}  # 8-bit: fits a 40 GB A100


def cmd_make():
    from so101.objects import OBJECTS_DIR, layout, random_objects, scanned_objects
    rng = np.random.default_rng(int(ARGS.get("seed", 0)))
    index = json.loads((OBJECTS_DIR / "gso_index.json").read_text())
    rows = []
    for scanned in (False, True):
        made = 0
        while made < int(ARGS.get("scenes", 30)):
            n = 4 if scanned else 5
            specs = scanned_objects(rng, n, strict=False) if scanned else random_objects(rng, n)
            try:
                poses = layout(rng, specs, tries=200, max_r=0.23)
            except RuntimeError:
                continue
            start = {}
            if made % 2:  # half the scenes start with 1 to --max-start objects already on a place
                k = int(rng.integers(1, int(ARGS.get("max-start", 2)) + 1))
                places = list(rng.choice(list(planner.PLACES), k, replace=k > 3))
                if k >= 2 and rng.random() < float(ARGS.get("both-trays", 0)):  # something on both trays: swap requests
                    places[:2] = ["red_tray", "blue_tray"]
                for i, p in zip(rng.choice(n, k, replace=False), places):
                    start[int(i)] = str(p)
            objs = planner.make_scene(specs, poses, index, start)
            for family, text, goal in planner.requests(objs, rng, scanned):
                rows.append(dict(id=len(rows), set="scanned" if scanned else "primitives", scene=made, family=family,
                                 request=text, goal=goal, objects=objs))
            made += 1
    if "per-family" in ARGS:  # balanced set: at most this many of each request type, drawn at random
        cap, keep = int(ARGS["per-family"]), []
        for f in dict.fromkeys(r["family"] for r in rows):
            fr = [r for r in rows if r["family"] == f]
            keep += [fr[i] for i in sorted(rng.choice(len(fr), min(cap, len(fr)), replace=False))]
        rows = sorted(keep, key=lambda r: r["id"])
        for i, r in enumerate(rows):
            r["id"] = i
    OUT.mkdir(parents=True, exist_ok=True)
    REQS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fam = {}
    for r in rows:
        fam[r["family"]] = fam.get(r["family"], 0) + 1
    print(f"{len(rows)} requests -> {REQS}: {fam}")


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def cmd_run():
    from vllm import LLM, SamplingParams
    key, think = ARGS["model"], ARGS.get("think", "on")
    hf, vision, switch, temp = MODELS[key]
    reqs = load(REQS)
    if "limit" in ARGS:
        reqs = reqs[:int(ARGS["limit"])]
    kw = dict(model=hf, max_model_len=int(ARGS.get("max-len", 16384)), gpu_memory_utilization=0.88, seed=0,
              max_num_seqs=int(ARGS.get("max-seqs", 256)))  # Qwen3.8-27B's Mamba-style cache fits 304, not vLLM's 1024
    if vision:  # (Gemma 4 12B: audio must be off too, or vLLM 0.24's dummy audio input fails: no fft_length)
        kw["limit_mm_per_prompt"] = {m: 0 for m in vision}
    llm = LLM(**kw)
    msgs = [planner.prompt(r["objects"], r["request"]) for r in reqs]
    tmpl = {}
    if switch == "enable_thinking":
        tmpl["enable_thinking"] = think == "on"
    elif switch == "reasoning_effort":
        tmpl["reasoning_effort"] = {"on": "medium", "off": "low"}.get(think, think)
    elif switch == "system" and think == "on":  # Cosmos-Reason: reasoning is asked for in the system prompt (model card)
        cue = ("\nAnswer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\n"
               "your answer\n</answer>.")
        msgs = [[{**m[0], "content": m[0]["content"] + cue}, m[1]] for m in msgs]
    if think == "off" and temp < 1.0:
        temp = 0.7  # Qwen's advice without thinking
    sp = SamplingParams(temperature=temp, top_p=0.95, max_tokens=int(ARGS.get("max-tokens", 8192)), seed=0)
    t = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs=tmpl, use_tqdm=False)
    wall = time.time() - t
    rows = []
    for r, o in zip(reqs, outs):
        text, cut = o.outputs[0].text, o.outputs[0].finish_reason == "length"
        ans = None if cut else planner.parse(text)
        rows.append(dict(id=r["id"], set=r["set"], family=r["family"], ok=planner.check(r["objects"], ans, r["goal"]),
                         parsed=ans is not None, truncated=cut,
                         tokens=len(o.outputs[0].token_ids), answer=ans, text=text[-3000:]))
    path = OUT / f"{key}-{think}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = np.mean([r["ok"] for r in rows])
    print(f"{key} think={think}: {ok * 100:.1f}% of {len(rows)} right, parsed {np.mean([r['parsed'] for r in rows]) * 100:.0f}%, "
          f"median {np.median([r['tokens'] for r in rows]):.0f} tokens, {wall / 60:.1f} min -> {path}")


def cmd_ollama():
    """The same benchmark through an Ollama server (OLLAMA_HOST, default 127.0.0.1:11434): the quantized GGUF a PC would
    run. --model=gemma4:12b --tag=gemma4-12b-q4 --think=on|off [--workers=8] (server's OLLAMA_NUM_PARALLEL >= workers)."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    import os
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else "http://" + host
    think = ARGS.get("think", "on") == "on"
    reqs = load(REQS)
    if "limit" in ARGS:
        reqs = reqs[:int(ARGS["limit"])]

    first_move = re.compile(r'\{\s*"object"\s*:\s*"[^"]*"\s*,\s*"place"\s*:\s*"[^"]*"\s*\}')

    def one(r):
        """Streamed, so the time and token count up to the first complete move are known (when a robot could start)."""
        body = dict(model=ARGS["model"], messages=planner.prompt(r["objects"], r["request"]), stream=True, think=think,
                    options=dict(temperature=float(ARGS.get("temp", 1.0)), top_p=0.95, seed=0, num_ctx=16384,
                                 num_predict=int(ARGS.get("max-tokens", 8192))))
        if "--schema" in sys.argv:  # constrained decoding: the answer must match the schema (thinking stays free)
            body["format"] = planner.schema(r["objects"])
        q = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0, thinking, text, chunks, first, o = time.time(), "", "", 0, None, {}
        with urllib.request.urlopen(q, timeout=3600) as f:
            for line in f:
                o = json.loads(line)
                m = o.get("message", {})
                thinking += m.get("thinking", "")
                text += m.get("content", "")
                chunks += bool(m.get("thinking") or m.get("content"))  # ~one token per chunk
                if first is None and first_move.search(text):
                    first = (time.time() - t0, chunks)
        cut = o.get("done_reason") == "length"
        ans = None if cut else planner.parse(text)
        moves = (ans or {}).get("moves") or []
        goal = r["goal"] or {}
        right_moves = sum(isinstance(m, dict) and goal.get(m.get("object")) == m.get("place") for m in moves)
        return dict(id=r["id"], set=r["set"], family=r["family"], ok=planner.check(r["objects"], ans, r["goal"]),
                    parsed=ans is not None, truncated=cut, tokens=o.get("eval_count", chunks), answer=ans,
                    prompt_tokens=o.get("prompt_eval_count", 0), wall_s=time.time() - t0,
                    first_move_s=first and first[0], first_move_tokens=first and first[1],
                    goal_moves=len(goal), right_moves=right_moves, extra_moves=len(moves) - right_moves,
                    text=(thinking[-2500:] + "</think>" + text)[-3000:])
    t = time.time()
    with ThreadPoolExecutor(int(ARGS.get("workers", 8))) as pool:
        rows = list(pool.map(one, reqs))
    path = OUT / f"{ARGS['tag']}-{'on' if think else 'off'}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{ARGS['tag']} think={'on' if think else 'off'}: {np.mean([r['ok'] for r in rows]) * 100:.1f}% of {len(rows)} right, "
          f"parsed {np.mean([r['parsed'] for r in rows]) * 100:.0f}%, median {np.median([r['tokens'] for r in rows]):.0f} tokens, "
          f"{(time.time() - t) / 60:.1f} min -> {path}")


def cmd_make_horizon():
    """Long lists of plain moves: "Put the A in the red tray, the B on the shelf, ... and the Z in the blue tray.",
    --per requests for each list length in --horizons, with --extra objects that must stay. Plan-only, so the table
    is just bigger (positions spread out, no physics). -> runs/plan/<bench>/requests.jsonl, family "list-N"."""
    from so101.objects import random_objects
    rng = np.random.default_rng(int(ARGS.get("seed", 2)))
    rows = []
    for n in map(int, ARGS.get("horizons", "1,2,4,8,12,16,24,32").split(",")):
        for _ in range(int(ARGS.get("per", 50))):
            specs = random_objects(rng, n + int(ARGS.get("extra", 3)))
            poses = [(rng.uniform(0.08, 0.36), rng.uniform(-0.22, 0.22), 0.0) for _ in specs]
            objs = planner.make_scene(specs, poses)
            pick = [objs[i] for i in rng.choice(len(objs), n, replace=False)]
            places = [str(p) for p in rng.choice(list(planner.PLACES), n)]
            parts = [f"the {o['name']} {planner.WHERE[p]}" for o, p in zip(pick, places)]
            text = "Put " + (parts[0] if n == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]) + "."
            rows.append(dict(id=len(rows), set="primitives", scene=len(rows), family=f"list-{n}", request=text,
                             goal={o["id"]: p for o, p in zip(pick, places)}, objects=objs))
    OUT.mkdir(parents=True, exist_ok=True)
    REQS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} requests -> {REQS}")


def cmd_horizon_report():
    """Success, moves right and time to the first move vs list length, for the Ollama runs in this bench; times on the
    4060 Ti estimated from its measured speeds (--prefill=, --decode= tokens/s; 2026-10-05: ~1000 and ~65)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pre, dec = float(ARGS.get("prefill", 1000)), float(ARGS.get("decode", 65))
    runs = {p.stem: load(p) for p in sorted(OUT.glob("*-o*.jsonl"))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    for k, rs in runs.items():
        ns = sorted({int(r["family"].split("-")[1]) for r in rs})
        print(f"\n{k}\n{'moves':>5s} {'right':>6s} {'95% range':>13s} {'moves right':>11s} {'extra':>5s} {'tokens':>6s} "
              f"{'1st-move tok':>12s} {'4060Ti 1st move':>15s} {'4060Ti whole':>12s}")
        acc, first_t = [], []
        for n in ns:
            g = [r for r in rs if r["family"] == f"list-{n}"]
            ok = sum(r["ok"] for r in g)
            lo, hi = wilson(ok, len(g))
            mr = sum(r["right_moves"] for r in g) / sum(r["goal_moves"] for r in g)
            ft = [r["first_move_tokens"] for r in g if r["first_move_tokens"]]
            est_first = np.median([r["prompt_tokens"] / pre + r["first_move_tokens"] / dec for r in g if r["first_move_tokens"]]) if ft else np.nan
            est_all = np.median([r["prompt_tokens"] / pre + r["tokens"] / dec for r in g])
            print(f"{n:5d} {ok / len(g) * 100:5.1f}% {lo * 100:5.1f}-{hi * 100:5.1f}% {mr * 100:10.1f}% "
                  f"{np.mean([r['extra_moves'] for r in g]):5.2f} {np.median([r['tokens'] for r in g]):6.0f} "
                  f"{np.median(ft) if ft else float('nan'):12.0f} {est_first:13.1f} s {est_all:10.1f} s")
            acc.append((n, ok / len(g) * 100, lo * 100, hi * 100))
            first_t.append((n, est_first, est_all))
        a = np.array(acc)
        label = "thinking" if k.endswith("-on") else "no thinking"
        line = axes[0].plot(a[:, 0], a[:, 1], "o-", lw=2, ms=6, label=label)[0]
        axes[0].fill_between(a[:, 0], a[:, 2], a[:, 3], alpha=0.15, color=line.get_color(), lw=0)
        f = np.array(first_t)
        axes[1].plot(f[:, 0], f[:, 2], "o-", lw=2, ms=6, color=line.get_color(), label=f"{label}: whole plan")
        axes[1].plot(f[:, 0], f[:, 1], "o--", lw=1.5, ms=5, color=line.get_color(), label=f"{label}: first move")
    axes[0].set(title="Plans exactly right (shaded: 95% range)", xlabel="moves in the request", ylabel="% right", ylim=(0, 102))
    axes[1].set(title="Time on the 4060 Ti (estimated from its measured speeds)", xlabel="moves in the request",
                ylabel="seconds", yscale="log")
    for ax in axes:
        ax.grid(color="#e6e6e6", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "horizon.png", dpi=130)
    print(f"figure -> {OUT / 'horizon.png'}")


def cmd_baselines():
    reqs = load(REQS)
    rng = np.random.default_rng(0)
    for name, answer in (("oracle", lambda r: planner.oracle(r["objects"], r["goal"])),
                         ("random", lambda r: {"moves": [{"object": str(rng.choice([o["id"] for o in r["objects"]])),
                                                          "place": str(rng.choice(list(planner.PLACES)))}
                                                         for _ in range(int(rng.integers(0, 3)))],
                                               "cannot": bool(rng.random() < 0.15)})):
        rows = []
        for r in reqs:
            ans = answer(r)
            rows.append(dict(id=r["id"], set=r["set"], family=r["family"], ok=planner.check(r["objects"], ans, r["goal"]),
                             parsed=True, truncated=False, tokens=0, answer=ans))
        (OUT / f"{name}-na.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name}: {np.mean([r['ok'] for r in rows]) * 100:.1f}% of {len(rows)}")


def wilson(k, n, z=1.96):
    """95% range for a success rate k/n (Wilson score interval)."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cmd_report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reqs = load(REQS)
    families = sorted({r["family"] for r in reqs}, key=lambda f: [r["family"] for r in reqs].index(f))
    runs = {}
    for p in sorted(OUT.glob("*-*.jsonl")):
        if p.name != REQS.name:
            runs[p.stem] = load(p)
    order = sorted(runs, key=lambda k: -np.mean([r["ok"] for r in runs[k]]))
    best = order[1] if order and order[0].startswith("oracle") and len(order) > 1 else order[0]
    print(f"{'run':28s} {'right':>6s} {'95% range':>13s} {'prims':>6s} {'scan':>6s} {'parsed':>6s} {'cut':>4s} {'tokens':>7s}"
          f"  vs {best}: only-this-right / only-it-right")
    for k in order:
        rs = runs[k]
        by = lambda s: np.mean([r["ok"] for r in rs if r["set"] == s]) * 100
        lo, hi = wilson(sum(r["ok"] for r in rs), len(rs))
        a = {r["id"]: r["ok"] for r in rs}
        b = {r["id"]: r["ok"] for r in runs[best]}
        mine = sum(a[i] and not b[i] for i in a if i in b)  # paired: the same requests, so count disagreements
        theirs = sum(b[i] and not a[i] for i in a if i in b)
        print(f"{k:28s} {np.mean([r['ok'] for r in rs]) * 100:5.1f}% {lo * 100:5.1f}-{hi * 100:5.1f}% {by('primitives'):5.1f}% "
              f"{by('scanned'):5.1f}% {np.mean([r['parsed'] for r in rs]) * 100:5.0f}% {sum(r['truncated'] for r in rs):4d} "
              f"{np.median([r['tokens'] for r in rs]):7.0f}  {mine:4d} / {theirs:4d}")
    grid = np.array([[np.mean([r["ok"] for r in runs[k] if r["family"] == f]) * 100 for f in families] for k in order])
    blues = matplotlib.colors.LinearSegmentedColormap.from_list(  # one-hue sequential ramp (dataviz reference blue)
        "blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    ink, muted = "#2b2b2b", "#6b6b6b"
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(4.5 + 0.7 * len(families), 1.6 + 0.42 * len(order)),
                                  gridspec_kw={"width_ratios": [len(families), 4]})
    im = ax.imshow(grid, cmap=blues, vmin=0, vmax=100, aspect="auto")
    n = {f: sum(r["family"] == f for r in reqs) for f in families}
    ax.set_xticks(range(len(families)), [f"{f} (n={n[f]})" for f in families], rotation=40, ha="right", fontsize=8, color=ink)
    ax.set_yticks(range(len(order)), order, fontsize=9, color=ink)
    for i in range(len(order)):
        for j in range(len(families)):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] > 55 else ink)
    ax.set_title(f"% of requests planned right, by request type ({len(reqs)} requests)", fontsize=10, color=ink, loc="left")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # accuracy vs answer length (output tokens: the time cost on the 4060 Ti), model runs only
    for k in order:
        toks = np.median([r["tokens"] for r in runs[k]])
        if not toks:
            continue
        acc = np.mean([r["ok"] for r in runs[k]]) * 100
        ax2.plot(toks, acc, "o", ms=8, color="#3987e5", mec="white", mew=2)
        ax2.annotate(k, (toks, acc), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=ink)
    ax2.set_xscale("log")
    ax2.set_xlabel("median output tokens per plan (log scale)", fontsize=9, color=muted)
    ax2.set_ylabel("% right, all requests", fontsize=9, color=muted)
    ax2.set_title("Accuracy vs answer length", fontsize=10, color=ink, loc="left")
    ax2.grid(color="#e6e6e6", lw=0.8)
    ax2.tick_params(colors=muted, labelsize=8)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=130)
    print(f"figure -> {OUT / 'comparison.png'}")


def cmd_examples():
    """A few requests with each listed model's plan, for a page: where the top models disagree, plus hard ones they all
    get right. -> runs/plan/examples.json"""
    reqs = {r["id"]: r for r in load(REQS)}
    keys = ARGS.get("runs", "gemma4-12b-on,gemma4-12b-off,gpt-oss-20b-on,cosmos-reason2-8b-on,qwen3.5-9b-off,"
                    "qwen3.8-27b-on").split(",")
    runs = {k: {r["id"]: r for r in load(OUT / f"{k}.jsonl")} for k in keys if (OUT / f"{k}.jsonl").exists()}
    rng = np.random.default_rng(int(ARGS.get("seed", 1)))
    split = [i for i in reqs if len({runs[k][i]["ok"] for k in runs}) > 1]
    hard = [i for i in reqs if reqs[i]["family"] in ("not-category", "sort", "superlative", "swap", "clear", "impossible")
            and all(runs[k][i]["ok"] for k in runs)]
    pick = []
    for pool, n in ((split, int(ARGS.get("split", 6))), (hard, int(ARGS.get("hard", 4)))):
        fams = {}
        for i in rng.permutation(pool):
            fams.setdefault(reqs[int(i)]["family"], []).append(int(i))
        while len([i for i in pick if i in pool]) < n and any(fams.values()):
            for f in list(fams):
                if fams[f] and len([i for i in pick if i in pool]) < n:
                    pick.append(fams[f].pop(0))
    name = lambda r, i: next((o["name"] for o in r["objects"] if o["id"] == i), i)
    out = []
    for i in pick:
        r = reqs[i]
        plans = {}
        for k in runs:
            a = runs[k][i]
            ans = a.get("answer") or {}
            think = a.get("text", "")
            for end in ("</think>", "<channel|>", "assistantfinal"):
                if end in think:
                    think = think.rsplit(end, 1)[0]
                    break
            else:
                think = ""
            plans[k] = dict(ok=a["ok"], cut=a["truncated"], cannot=bool(ans.get("cannot")), reason=ans.get("reason", ""),
                            moves=[[name(r, m.get("object")), m.get("place")] for m in ans.get("moves") or []
                                   if isinstance(m, dict)], tokens=a["tokens"], thinking=think.strip()[-700:])
        out.append(dict(id=i, set=r["set"], family=r["family"], request=r["request"], scene=planner.scene_text(r["objects"]),
                        goal=None if r["goal"] is None else [[name(r, o), p] for o, p in r["goal"].items()],
                        plans=plans))
    (OUT / "examples.json").write_text(json.dumps(out, indent=1))
    print(f"{len(out)} examples -> {OUT / 'examples.json'}")


if __name__ == "__main__":
    {"make": cmd_make, "run": cmd_run, "ollama": cmd_ollama, "baselines": cmd_baselines, "report": cmd_report,
     "examples": cmd_examples, "make-horizon": cmd_make_horizon, "horizon-report": cmd_horizon_report}[sys.argv[1]]()
