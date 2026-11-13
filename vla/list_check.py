"""Can a vision-language model list what is on the table from one camera picture? (The planner's missing
"what's on the table?" step: today its object list comes from the simulator.)

  render  N scenes per set (simple shapes, household objects), arm parked out of view, front camera RGB 640 px
          -> runs/listing/<set>-<i>.png + truth.json (names, and how many pixels of each object the camera sees)
  ask     each picture to an Ollama vision model (OLLAMA_HOST), two ways:
            open    "list every object you see" in its own words
            closed  "which of these known names are on the table?" (the whole name list of that set)
          -> runs/listing/<tag>-<mode>-<think>.jsonl
  judge   open answers matched to the true objects by a text model in vLLM (synonyms need judgment)
  report  found (recall), invented (false names), count right; objects under --min-px pixels counted separately

python vla/list_check.py render --scenes=200 ;  ask --model=gemma4:12b --tag=gemma4-12b-q4 --mode=open|closed --think=off
cluster: sbatch vla/list_ollama.sbatch ... ; sbatch vla/list_judge.sbatch ... (vLLM container)
"""
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
OUT = Path("runs/listing")
FIXTURES = "Ignore the robot arm, the red and blue trays (flat pads), the wooden shelf block and the floor."
OPEN = ("This picture is from a camera looking down at a robot's work table. List every loose object on the table that "
        f"a robot hand could pick up. {FIXTURES} Give each object a short name with its colour, the way you would ask "
        "someone to fetch it. If two objects look alike, list both. Answer with JSON only: "
        '{"objects": ["name", ...]}')
CLOSED = ("This picture is from a camera looking down at a robot's work table. Which of these objects are on the table? "
          f"{FIXTURES} Candidates: {{names}}. Answer with JSON only, using the candidate names exactly as written: "
          '{{"objects": ["name", ...]}}')  # (doubled braces: this string goes through str.format)


def cmd_render():
    import mujoco
    from PIL import Image
    from so101.camera import SimCameras
    from so101.objects import COLOURS, NOUN, body_name, build, layout, random_objects, scanned_objects
    from so101.skills import Arm
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(ARGS.get("seed", 5)))
    truth = []
    for s in ("primitives", "scanned"):
        made = 0
        while made < int(ARGS.get("scenes", 200)):
            specs = random_objects(rng, 5) if s == "primitives" else scanned_objects(rng, 4)
            try:
                poses = layout(rng, specs, tries=200, max_r=0.23)
            except RuntimeError:
                continue
            model = build(specs, poses)
            data, scratch = mujoco.MjData(model), mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            arm = Arm(model, data, scratch)
            data.qpos[:6] = arm.target(np.array([0.10, 0.0, 0.20]))  # parked above the base, as before every look
            mujoco.mj_forward(model, data)
            cams = SimCameras(model, data, ["cam_front"], 640)
            name = f"{s}-{made:03d}"
            Image.fromarray(cams.render_rgb(0)).save(OUT / f"{name}.png")
            roots = cams.true_bodies(0)
            objs = [dict(name=sp["name"], px=int((roots == model.body(body_name(sp)).id).sum())) for sp in specs]
            truth.append(dict(image=name, set=s, objects=objs))
            cams.close()
            made += 1
    vocab = {"primitives": sorted({f"{c} {n}" for c in COLOURS for n in set(NOUN.values())}),
             "scanned": sorted({o["phrase"] for o in json.loads((Path("so101/assets/objects") / "gso_index.json").read_text())})}
    (OUT / "truth.json").write_text(json.dumps(dict(scenes=truth, vocab=vocab), indent=1))
    print(f"{len(truth)} scenes -> {OUT}")


def parse_list(text):
    for end in ("</think>", "<channel|>", "assistantfinal"):
        if end in text:
            text = text.rsplit(end, 1)[1]
    m = list(re.finditer(r"\{[^{}]*\"objects\"\s*:\s*\[.*?\]\s*\}", text, re.S))
    if not m:
        return None
    try:
        return [str(x).strip() for x in json.loads(m[-1].group(0))["objects"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def cmd_ask():
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else "http://" + host
    t = json.loads((OUT / "truth.json").read_text())
    mode, think = ARGS.get("mode", "open"), ARGS.get("think", "off") == "on"

    def one(sc):
        prompt = OPEN if mode == "open" else CLOSED.format(names="; ".join(t["vocab"][sc["set"]]))
        img = base64.b64encode((OUT / f"{sc['image']}.png").read_bytes()).decode()
        body = dict(model=ARGS["model"], stream=False, think=think,
                    messages=[{"role": "user", "content": prompt, "images": [img]}],
                    options=dict(temperature=float(ARGS.get("temp", 1.0)), top_p=0.95, seed=0, num_ctx=8192,
                                 num_predict=int(ARGS.get("max-tokens", 4096))))
        q = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(q, timeout=1800) as f:
            o = json.load(f)
        text = o["message"].get("content", "")
        return dict(image=sc["image"], set=sc["set"], listed=parse_list(text), text=text[-1500:],
                    tokens=o.get("eval_count", 0), prompt_tokens=o.get("prompt_eval_count", 0), wall_s=time.time() - t0)
    with ThreadPoolExecutor(int(ARGS.get("workers", 8))) as pool:
        rows = list(pool.map(one, t["scenes"]))
    path = OUT / f"{ARGS['tag']}-{mode}-{'on' if think else 'off'}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{path}: parsed {np.mean([r['listed'] is not None for r in rows]) * 100:.0f}%, "
          f"median {np.median([r['tokens'] for r in rows]):.0f} output / {np.median([r['prompt_tokens'] for r in rows]):.0f} prompt tokens")


JUDGE = """Match a robot's guessed object names to the true objects on a table.
True objects (index: name): {truth}
Guessed names (index: name): {guess}
Pair a guess with a true object only if the guess plausibly describes that object: compatible colour and kind of thing
(synonyms are fine: ball/sphere, block/cube/box, egg/oval, capsule/pill shape, figure/figurine/toy, bottle/container).
Each true object and each guess can be used at most once. Answer with JSON only: {{"pairs": [[guess_index, true_index], ...]}}"""


def cmd_judge():
    """Open answers only: a text model pairs guesses with true objects. Runs in the vLLM container (no MuJoCo)."""
    from vllm import LLM, SamplingParams
    t = {sc["image"]: sc for sc in json.loads((OUT / "truth.json").read_text())["scenes"]}
    path = OUT / f"{ARGS['run']}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    msgs, idx = [], []
    for i, r in enumerate(rows):
        if r["listed"]:
            truth = "; ".join(f"{j}: {o['name']}" for j, o in enumerate(t[r["image"]]["objects"]))
            guess = "; ".join(f"{j}: {g}" for j, g in enumerate(r["listed"]))
            msgs.append([{"role": "user", "content": JUDGE.format(truth=truth, guess=guess)}])
            idx.append(i)
    # 8-bit judge on a whole 46 GB card: the bf16 one ran out of memory on the 40 GB H100 slices (2026-10-06); on the
    # 3,600-request planner set the 8-bit and bf16 Qwen3.8-27B disagreed on 2 requests.
    llm = LLM(model=ARGS.get("judge", "Qwen/Qwen3.8-27B-FP8"), max_model_len=8192, gpu_memory_utilization=0.9, seed=0,
              max_num_seqs=int(ARGS.get("max-seqs", 48)), limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=2048), chat_template_kwargs={"enable_thinking": False},
                    use_tqdm=False)
    for i, o in zip(idx, outs):
        m = re.findall(r"\{.*\}", o.outputs[0].text, re.S)
        pairs = []
        try:
            pairs = json.loads(m[-1])["pairs"] if m else []
        except (json.JSONDecodeError, KeyError):
            pass
        n_t, n_g = len(t[rows[i]["image"]]["objects"]), len(rows[i]["listed"])
        seen_t, seen_g, keep = set(), set(), []
        for p in pairs:  # enforce one-to-one and valid indices
            if isinstance(p, list) and len(p) == 2 and all(isinstance(v, int) for v in p) and p[0] < n_g and p[1] < n_t \
                    and p[0] not in seen_g and p[1] not in seen_t:
                keep.append(p)
                seen_g.add(p[0])
                seen_t.add(p[1])
        rows[i]["pairs"] = keep
    path.with_name(path.stem + "-judged.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"judged {len(idx)} -> {path.stem}-judged.jsonl")


def matches(r, sc):
    """(found true-object indices, number of guesses that matched)."""
    if r.get("listed") is None:
        return set(), 0
    if "pairs" in r:
        return {p[1] for p in r["pairs"]}, len(r["pairs"])
    names = [o["name"] for o in sc["objects"]]  # closed answers: exact names (look-alike phrases match once each)
    found, used = set(), 0
    for g in r["listed"]:
        j = next((j for j, n in enumerate(names) if n == g and j not in found), None)
        if j is not None:
            found.add(j)
            used += 1
    return found, used


def cmd_report():
    t = {sc["image"]: sc for sc in json.loads((OUT / "truth.json").read_text())["scenes"]}
    min_px = int(ARGS.get("min-px", 300))
    for path in sorted(OUT.glob("*.jsonl")):
        if path.stem.endswith("-open-off") or path.stem.endswith("-open-on"):
            if (OUT / f"{path.stem}-judged.jsonl").exists():
                continue  # report the judged copy
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f"\n{path.stem}")
        for s in ("primitives", "scanned"):
            rs = [r for r in rows if r["set"] == s]
            vis = hid = vis_found = hid_found = guesses = good = count_ok = parsed = 0
            for r in rs:
                sc = t[r["image"]]
                found, used = matches(r, sc)
                parsed += r["listed"] is not None
                guesses += len(r["listed"] or [])
                good += used
                for j, o in enumerate(sc["objects"]):
                    if o["px"] >= min_px:
                        vis += 1
                        vis_found += j in found
                    else:
                        hid += 1
                        hid_found += j in found
                count_ok += len(r["listed"] or []) == sum(o["px"] >= min_px for o in sc["objects"])
            print(f"  {s:10s} found {vis_found}/{vis} visible ({vis_found / max(vis, 1) * 100:.1f}%), "
                  f"{hid_found}/{hid} barely visible (<{min_px} px); invented {guesses - good} of {guesses} names "
                  f"({(guesses - good) / max(guesses, 1) * 100:.1f}%); count right {count_ok}/{len(rs)}; parsed {parsed}/{len(rs)}; "
                  f"median {np.median([r['wall_s'] for r in rs]):.1f} s")


if __name__ == "__main__":
    {"render": cmd_render, "ask": cmd_ask, "judge": cmd_judge, "report": cmd_report}[sys.argv[1]]()
