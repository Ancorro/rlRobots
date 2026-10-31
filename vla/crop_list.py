"""Detector-first scene listing: Grounding DINO boxes with a generic prompt, then a vision-language model names each crop.
Follow-up to vla/list_check.py (whole-picture listing by 4-bit Gemma 4 12B was not good enough). Same 400 scenes,
same truth, same scoring (list_check's found / invented / count-right, imported, not copied).

  truthboxes  replay list_check's render loop (seed 5) to get what truth.json lacks: each object's visible-pixel box
              (segmentation render) and the robot/fixture pixel mask per scene. Asserts the same object names and pixel
              counts within max(10 px, 2 %) of runs/listing/truth.json's (another GPU type rasterizes a few edge pixels
              differently: first scene off by 0-2 px on an H100), keeps truth.json's counts, and counts images that are
              pixel-identical to the saved PNGs.
              -> runs/croplist/truth.json (list_check's schema + "box" per object), runs/croplist/known.npz
  detect      Grounding DINO tiny (see.DINO), prompt "object.", every box with score >= 0.10 kept raw
              -> runs/croplist/dino-object.json (boxes, scores, detector seconds per image)
  name        each kept box (see keep_boxes) cropped (see.py's padding), upscaled to >= 224 px, sent to an Ollama
              vision model, alone (--context=off) or after the whole picture with the box drawn in red (--context=on,
              "set-of-mark" prompting): closed = "which of these names is it?" (the set's whole name list, as
              list_check), open = short name with colour; both may answer null (no loose object: a reflection or shadow
              on the shiny table, the robot arm, a tray, the shelf, bare table). The scene's list (listing()) = the
              non-null answers, where a box with the same name as a higher-scoring box it touches counts once (one
              object boxed twice). --boxes=truth names the true boxes instead (naming-only diagnostic).
              -> runs/croplist/<tag>[-ctx][-truthbox]-<mode>-off.jsonl (list_check's schema, plus per-crop details)
  siglip      the same boxes named by SigLIP 2 (see.py's crop-phrase matcher) against the set's names plus a few
              not-an-object labels (closed only; a cheap non-generative comparison)
              -> runs/croplist/siglip2[-truthbox]-closed-off.jsonl
  judge       list_check.judge (same 8-bit Qwen3.8-27B judge, same prompt) on the open rows      (vLLM container)
  report      list_check.report on runs/croplist (identical metrics), then detector recall, naming accuracy on
              detected objects, threshold sweep, time per scene                                   (.venv-vla)
  figure      found % and wrong-name % by method -> runs/croplist/methods.png

Kept boxes (keep_boxes): score >= --thr (0.12, see.py's value, chosen before looking at results), each side under 40 %
of the image (see.py: not the whole table), greedy non-maximum suppression at IoU 0.5, and dropped if more than 30 % of
the box shows the robot or a fixture (trays, shelf). The fixture mask is known geometry drawn into the image, the same
assumption see.py's known_reject makes (on a real robot: the robot and fixture model rendered into the camera).
History: first set at 75 %; the smoke test showed the parked arm's boxes passing (robot pixels diluted by background),
and no true object's box over all 400 scenes is more than 13 % robot/fixture, so 30 % (junk boxes 3.4 -> 1.2 per scene,
recall unchanged). That choice looked at the evaluation scenes (truth boxes and junk counts, not naming results).

Detector recall: a truth object with at least --min-px (300) visible pixels counts as detected when a kept box has
IoU >= 0.5 with the object's true box (tight box around its visible pixels in the segmentation render); boxes and
objects are paired one-to-one, greedily by descending IoU. Also reported at IoU >= 0.3.

cluster: sbatch vla/crop_name.sbatch ; sbatch --dependency=afterok:<job> vla/crop_judge.sbatch
"""
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import list_check  # noqa: E402  (its ARGS parse the same "<cmd> --k=v" command line)

ARGS = list_check.ARGS
SRC = Path("runs/listing")  # read only: list_check's scenes and truth
OUT = Path("runs/croplist")
MIN_PX = int(ARGS.get("min-px", 300))
NULL = ("If there is no loose object there that a robot hand could pick up (only a reflection or shadow on the shiny "
        "table top, part of the yellow robot arm, a flat red or blue tray, the wooden shelf block, bare table or floor), "
        "answer null. If several objects, answer for the one in the centre. Answer with JSON only: "
        '{{"object": {fmt}}}')
CROP = "This is a crop from a camera picture of a robot's work table. {task} " + NULL
CTX = ("The first picture is from a camera looking down at a robot's work table; one area is marked with a red "
       "rectangle. The second picture is a close-up of that area (without the rectangle). {task} " + NULL)
TASK = dict(closed="Which one of these objects is it? Candidates: {names}. Use the candidate name exactly as written.",
            open="Name the object with a short name and its colour, the way you would ask someone to fetch it.")
FMT = dict(closed='"candidate name" or null', open='"name" or null')
# JEV-9B (--api=jev): the same question as a choice; the options are the set's names plus NOT_OBJECT (null)
JEV_CROP = "A crop from a camera picture of a robot's work table:"
JEV_CTX = "A camera picture looking down at a robot's work table, one area marked with a red rectangle:"
JEV_Q = ("What is in the centre of the close-up? Pick the loose object a robot hand could pick up, or, if there is none "
         "(a reflection or shadow, part of the robot arm, a flat tray, the shelf, bare table), what is there instead.")


def truth():
    return json.loads((OUT / "truth.json").read_text())


# --- truthboxes ---------------------------------------------------------------------------------------------------
def cmd_truthboxes():
    """list_check.cmd_render's loop, unchanged in what it draws, but keeping the segmentation instead of the PNG."""
    import mujoco
    from PIL import Image
    from so101.camera import SimCameras
    from so101.objects import body_name, build, layout, random_objects, scanned_objects
    from so101.skills import Arm
    OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((SRC / "truth.json").read_text())
    old_by = {sc["image"]: sc for sc in old["scenes"]}
    rng = np.random.default_rng(int(ARGS.get("seed", 5)))
    scenes, known, same_png, max_dev = [], {}, 0, 0
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
            data.qpos[:6] = arm.target(np.array([0.10, 0.0, 0.20]))
            mujoco.mj_forward(model, data)
            cams = SimCameras(model, data, ["cam_front"], 640)
            name = f"{s}-{made:03d}"
            rgb = cams.render_rgb(0)
            same_png += bool(np.array_equal(rgb, np.asarray(Image.open(SRC / f"{name}.png").convert("RGB"))))
            roots = cams.true_bodies(0)
            objs = []
            for sp in specs:
                vs, us = np.nonzero(roots == model.body(body_name(sp)).id)
                box = [int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1] if len(us) else None
                objs.append(dict(name=sp["name"], px=int(len(us)), box=box))
            want = old_by[name]["objects"]  # same names; pixel counts may differ by a few px (other GPU's rasterizer)
            dev = [abs(o["px"] - w["px"]) for o, w in zip(objs, want)]
            if [o["name"] for o in objs] != [w["name"] for w in want] or                     any(d > max(10, 0.02 * w["px"]) for d, w in zip(dev, want)):
                raise SystemExit(f"replay differs from runs/listing at {name}: {want} vs {objs}")
            max_dev = max([max_dev] + dev)
            for o, w in zip(objs, want):
                o["px"] = w["px"]  # keep list_check's counts so visible / barely-visible splits are identical
            # robot (bodies under "base") and fixtures, as see.Eyes.attach marks them
            base = model.body("base").id
            robot = {b for b in range(model.nbody) if _under(model, b, base)}
            fix = robot | {model.body(n).id for n in ("shelf", "red_tray", "blue_tray")}
            kn = np.array([model.geom_bodyid[g] in fix for g in range(model.ngeom)] + [False])
            known[name] = np.packbits(cams.known_px(0, kn))
            scenes.append(dict(image=name, set=s, objects=objs))
            cams.close()
            made += 1
    (OUT / "truth.json").write_text(json.dumps(dict(scenes=scenes, vocab=old["vocab"]), indent=1))
    np.savez_compressed(OUT / "known.npz", **known)
    print(f"{len(scenes)} scenes replayed, same objects; pixel counts within {max_dev} px of runs/listing/truth.json; "
          f"{same_png} images pixel-identical "
          f"to the saved PNGs -> {OUT}/truth.json, known.npz")


def _under(model, b, root):
    while b > 0:
        if b == root:
            return True
        b = model.body_parentid[b]
    return False


def known_mask(kn, image):
    return np.unpackbits(kn[image])[:640 * 640].reshape(640, 640).astype(bool)


# --- detect --------------------------------------------------------------------------------------------------------
def cmd_detect():
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from so101.see import DINO
    prompt = ARGS.get("prompt", "object.")
    proc = AutoProcessor.from_pretrained(DINO)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to("cuda").eval()
    out = {}
    for sc in truth()["scenes"]:
        img = np.asarray(Image.open(SRC / f"{sc['image']}.png").convert("RGB"))
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            inputs = proc(images=img, text=prompt, return_tensors="pt").to("cuda")
            r = proc.post_process_grounded_object_detection(dino(**inputs), inputs.input_ids, threshold=0.10,
                                                            text_threshold=0.10, target_sizes=[img.shape[:2]])[0]
        torch.cuda.synchronize()
        out[sc["image"]] = dict(boxes=r["boxes"].cpu().numpy().round(1).tolist(),
                                scores=r["scores"].cpu().numpy().round(4).tolist(), det_s=time.time() - t0)
    (OUT / "dino-object.json").write_text(json.dumps(dict(prompt=prompt, images=out)))
    n = [len(v["boxes"]) for v in out.values()]
    print(f"detect: {len(out)} images, raw boxes per image median {np.median(n):.0f}, "
          f"median {np.median([v['det_s'] for v in list(out.values())[5:]]) * 1000:.0f} ms per image")


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


FIXTURE = 0.3  # see the module docstring (history: 0.75 first)


def keep_boxes(det, known, thr=0.12, fixture=FIXTURE, nms=0.5):
    """Indices into det["boxes"] that survive (see module docstring). Greedy NMS by score, so the result at a higher
    threshold is exactly the subset of this one with score >= that threshold."""
    keep = []
    for i in np.argsort(-np.asarray(det["scores"]), kind="stable"):
        b, s = det["boxes"][i], det["scores"][i]
        if s < thr or b[2] - b[0] >= 640 * 0.4 or b[3] - b[1] >= 640 * 0.4:
            continue
        if any(iou(b, det["boxes"][j]) > nms for j in keep):
            continue
        x0, y0, x1, y1 = np.clip(np.round(b).astype(int), 0, 640)
        if known is not None and x1 > x0 and y1 > y0 and known[y0:y1, x0:x1].mean() > fixture:
            continue
        keep.append(int(i))
    return keep


def pair(boxes, objs, min_iou):
    """Greedy one-to-one pairing of boxes with visible truth objects by descending IoU: {object index: box index}."""
    cand = sorted(((iou(b, o["box"]), j, i) for i, b in enumerate(boxes) for j, o in enumerate(objs)
                   if o["px"] >= MIN_PX and o["box"]), reverse=True)
    got, used = {}, set()
    for v, j, i in cand:
        if v < min_iou:
            break
        if j not in got and i not in used:
            got[j] = i
            used.add(i)
    return got


# --- name ----------------------------------------------------------------------------------------------------------
def png(im):
    f = io.BytesIO()
    im.save(f, "PNG")
    return base64.b64encode(f.getvalue()).decode()


def marked_png(img, b):
    """The whole picture with box b drawn in red (3 px)."""
    from PIL import Image, ImageDraw
    im = Image.fromarray(img)
    ImageDraw.Draw(im).rectangle([round(v) for v in b], outline=(255, 0, 0), width=3)
    return png(im)


def listing_idx(crops):
    """Indices of the crops that make the scene's object list: non-null names; a crop whose name equals that of a
    higher-scoring crop whose box touches its own is the same object boxed twice and is dropped."""
    kept = []
    for i in sorted(range(len(crops)), key=lambda i: -crops[i]["score"]):
        c = crops[i]
        if c["name"] and not any(crops[k]["name"].lower() == c["name"].lower() and _touch(crops[k]["box"], c["box"])
                                 for k in kept):
            kept.append(i)
    return kept


def listing(crops):
    return [crops[i]["name"] for i in listing_idx(crops)]


def _touch(a, b):
    return min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) >= max(a[1], b[1])


def crop_png(img, b):
    """see.py's crop padding (15 % of the long side + 4 px), upscaled so the long side is at least 224 px."""
    from PIL import Image
    x0, y0, x1, y1 = b
    pad = 0.15 * max(x1 - x0, y1 - y0) + 4
    x0, y0, x1, y1 = np.clip(np.round([x0 - pad, y0 - pad, x1 + pad, y1 + pad]).astype(int), 0, 640)
    c = Image.fromarray(img[y0:y1, x0:x1])
    k = 224 / max(c.size)
    if k > 1:
        c = c.resize((round(c.size[0] * k), round(c.size[1] * k)), Image.BICUBIC)
    return png(c)


def parse_one(text):
    """(answered, name or None); answered False = no parseable answer (counted, treated as null)."""
    for end in ("</think>", "<channel|>", "assistantfinal"):
        if end in text:
            text = text.rsplit(end, 1)[1]
    for m in reversed(list(re.finditer(r"\{[^{}]*\"object\"[^{}]*\}", text, re.S))):
        try:  # extra keys are fine (Qwen3.5-9B adds "colour": its colour joins the name if the name lacks it)
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        v = d.get("object")
        if not isinstance(v, str) or not v.strip():
            return True, None
        c = d.get("colour") or d.get("color")
        v = v.strip()
        return True, f"{c.strip()} {v}" if isinstance(c, str) and c.strip() and c.lower() not in v.lower() else v
    return False, None


def cmd_reparse():
    """Re-read the stored answers of a `name` run with the current parse_one (no model calls): --run=<stem>."""
    path = OUT / f"{ARGS['run']}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if any("p" in c for r in rows for c in r["crops"]):
        raise SystemExit(f"{path}: a choice run (--api=jev, label + probability), not text answers: left unchanged")
    for r in rows:
        for c in r["crops"]:
            c["parsed"], c["name"] = parse_one(c["text"])
        r["listed"] = listing(r["crops"])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{path}: unparsed {sum(not c['parsed'] for r in rows for c in r['crops'])}")


def cmd_name():
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else "http://" + host
    t = truth()
    mode, which = ARGS.get("mode", "closed"), ARGS.get("boxes", "dino")
    thr = float(ARGS.get("thr", 0.12))
    if which == "dino":
        det = json.loads((OUT / "dino-object.json").read_text())["images"]
        kn = dict(np.load(OUT / "known.npz"))  # read once: NpzFile is not safe across threads
    ctx = ARGS.get("context", "off") == "on"
    scenes = t["scenes"]
    if "limit" in ARGS:  # smoke test: scenes spread over both sets
        scenes = scenes[::len(scenes) // int(ARGS["limit"])][:int(ARGS["limit"])]

    api = ARGS.get("api", "ollama")  # ollama | openai (a vLLM server: Qwen3.8-27B) | jev (JEV-9B's /v1/decide)
    if api == "jev" and mode != "closed":
        raise SystemExit("--api=jev answers choice questions: closed mode only")

    def post(path, body):
        q = urllib.request.Request(host + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(q, timeout=1800) as f:
            return json.load(f)

    def ask(prompt, imgs):
        if api == "openai":
            o = post("/v1/chat/completions", dict(
                model=ARGS.get("model"), temperature=float(ARGS.get("temp", 1.0)), top_p=0.95, seed=0,
                max_tokens=int(ARGS.get("max-tokens", 256)), chat_template_kwargs={"enable_thinking": False},
                messages=[{"role": "user", "content": [
                    *({"type": "image_url", "image_url": {"url": "data:image/png;base64," + i}} for i in imgs),
                    {"type": "text", "text": prompt}]}]))
            u = o.get("usage") or {}
            return o["choices"][0]["message"].get("content") or "", u.get("completion_tokens", 0), u.get("prompt_tokens", 0)
        body = dict(model=ARGS.get("model", "gemma4:12b"), stream=False, think=False,
                    messages=[{"role": "user", "content": prompt, "images": imgs}],
                    options=dict(temperature=float(ARGS.get("temp", 1.0)), top_p=0.95, seed=0, num_ctx=8192,
                                 num_predict=int(ARGS.get("max-tokens", 256))))
        q = urllib.request.Request(host + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(q, timeout=1800) as f:
            o = json.load(f)
        return o["message"].get("content", ""), o.get("eval_count", 0), o.get("prompt_eval_count", 0)

    def one(sc):
        img = np.asarray(Image.open(SRC / f"{sc['image']}.png").convert("RGB"))
        if which == "dino":
            d = det[sc["image"]]
            idx = keep_boxes(d, known_mask(kn, sc["image"]), thr)
            boxes, scores, det_s = [d["boxes"][i] for i in idx], [d["scores"][i] for i in idx], d["det_s"]
        else:  # true boxes of the visible objects (naming-only diagnostic)
            boxes = [o["box"] for o in sc["objects"] if o["px"] >= MIN_PX]
            scores, det_s = [1.0] * len(boxes), 0.0
        task = TASK[mode].format(names="; ".join(t["vocab"][sc["set"]]))
        prompt = (CTX if ctx else CROP).format(task=task, fmt=FMT[mode])
        t0, crops, tok, ptok = time.time(), [], 0, 0
        names = t["vocab"][sc["set"]]
        for b, s in zip(boxes, scores):  # one scene's crops one after another; scenes run in parallel
            if api == "jev":  # a choice over the set's names plus NOT_OBJECT labels (as siglip), calibrated probabilities
                labels = names + NOT_OBJECT
                r = post("/v1/decide", dict(kind="choice", question=JEV_Q, options=labels, state=[
                    *([JEV_CTX, {"image": "data:image/png;base64," + marked_png(img, b)}, "\nClose-up:"] if ctx
                      else [JEV_CROP]),
                    {"image": "data:image/png;base64," + crop_png(img, b)}]))
                pr = dict(zip(r["options"], r["probabilities"]))
                top = max(pr, key=pr.get)
                crops.append(dict(box=b, score=s, name=top if top in names else None, parsed=True,
                                  text=f"{top} {pr[top]:.3f}", p=round(pr[top], 4),
                                  p_none=round(sum(pr[x] for x in NOT_OBJECT), 4)))
                continue
            text, n, p = ask(prompt, [marked_png(img, b), crop_png(img, b)] if ctx else [crop_png(img, b)])
            ok, name = parse_one(text)
            crops.append(dict(box=b, score=s, name=name, parsed=ok, text=text[-300:]))
            tok, ptok = tok + n, ptok + p
        listed = listing(crops)
        return dict(image=sc["image"], set=sc["set"], listed=listed, text="", tokens=tok, prompt_tokens=ptok,
                    wall_s=det_s + time.time() - t0, det_s=det_s, name_s=time.time() - t0, crops=crops)
    t0 = time.time()
    with ThreadPoolExecutor(int(ARGS.get("workers", 8))) as pool:
        rows = list(pool.map(one, scenes))
    total = time.time() - t0
    tag = ARGS.get("tag", "gemma4-12b-q4") + ("-ctx" if ctx else "") + ("-truthbox" if which == "truth" else "") \
        + ARGS.get("suffix", "")
    path = OUT / f"{tag}-{mode}-off.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    nc = sum(len(r["crops"]) for r in rows)
    print(f"{path}: {nc} crops in {len(rows)} scenes, {total:.0f} s wall ({total / len(rows):.2f} s per scene "
          f"throughput), unparsed {sum(not c['parsed'] for r in rows for c in r['crops'])}, "
          f"null {sum(c['parsed'] and not c['name'] for r in rows for c in r['crops'])}")
    (OUT / f"{path.stem}.time").write_text(json.dumps(dict(total_s=total, scenes=len(rows), crops=nc)))


NOT_OBJECT = ["reflection on a shiny table", "shadow on a table", "yellow robot arm", "empty blue table", "flat red tray",
              "flat blue tray", "wooden shelf"]


def cmd_siglip():
    """Closed naming by a CLIP-style model instead of a VLM: each crop scored against "a photo of a <name>." for the
    set's names plus NOT_OBJECT labels; best label wins, a NOT_OBJECT label = null. Same rows as `name`.
      --model=<HF id> (default see.SIGLIP, SigLIP 2 Base 224; any SigLIP / SigLIP 2 checkpoint loads the same way)
      --model=openclip:<hub id> (open_clip, e.g. openclip:timm/PE-Core-L-14-336 for Meta's Perception Encoder)
      --tag=<name> (default siglip2). Each crop also keeps its best name's and best NOT_OBJECT label's sigmoid
      (SigLIP) or softmax (open_clip) score, for a "none" cutoff measured afterwards. Tag <tag>[-truthbox]-closed-off."""
    import torch
    from PIL import Image
    from so101.see import SIGLIP
    t = truth()
    which = ARGS.get("boxes", "dino")
    name = ARGS.get("model", SIGLIP)
    if name.startswith("moevie:"):  # Meta MoE-ViE: its own open_clip (repo src on PYTHONPATH), packed image batches
        import open_clip
        mname, half = name[7:], (0.5, 0.5, 0.5)
        clip, _, pre = open_clip.create_model_and_transforms(
            mname, pretrained=True, image_mean=half, image_std=half,
            force_preprocess_cfg=dict(patch_size=14, size_range=(448, 448), center_crop=True, window_size=1))
        clip, tok = clip.to("cuda").eval(), open_clip.get_tokenizer(mname)

        def score(ims, texts):
            packed, _ = pre.collate_fn([(pre(Image.fromarray(i)), 0) for i in ims])
            packed = open_clip.image_to_device(packed, "cuda", torch.float32, mean=half, std=half)
            with torch.autocast("cuda"):
                f = clip.encode_image(packed, normalize=True)
                g = clip.encode_text(tok(texts).cuda(), normalize=True)
                return (clip.logit_scale.exp() * f @ g.T).float().softmax(-1)
    elif name.startswith("fgclip2:"):  # FG-CLIP 2 (Qihoo 360): remote code, one image at a time (variable patch count)
        from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
        fg = AutoModelForCausalLM.from_pretrained(name[8:], trust_remote_code=True).to("cuda").eval()
        ftok, fproc = AutoTokenizer.from_pretrained(name[8:]), AutoImageProcessor.from_pretrained(name[8:])

        def patches(im):  # its model card's determine_max_value
            n = (im.size[0] // 16) * (im.size[1] // 16)
            return 1024 if n > 784 else 784 if n > 576 else 576 if n > 256 else 256 if n > 128 else 128

        def score(ims, texts):
            fs = []
            for i in ims:
                im = Image.fromarray(i)
                fs.append(fg.get_image_features(**fproc(images=im, max_num_patches=patches(im),
                                                        return_tensors="pt").to("cuda")))
            f = torch.cat(fs)
            g = fg.get_text_features(**ftok([x.lower() for x in texts], padding="max_length", max_length=64,
                                            truncation=True, return_tensors="pt").to("cuda"), walk_type="short")
            f, g = f / f.norm(dim=-1, keepdim=True), g / g.norm(dim=-1, keepdim=True)
            return (100 * f @ g.T).float().softmax(-1)
    elif name.startswith(("openclip:", "tulip:")):
        import open_clip
        if name.startswith("tulip:"):  # TULIP's open_clip fork (on PYTHONPATH): tulip:<model name>:<checkpoint path>
            _, mname, ckpt = name.split(":", 2)
            clip, _, pre = open_clip.create_model_and_transforms(mname, pretrained=ckpt)
            tok = open_clip.get_tokenizer(mname)
        else:
            clip, _, pre = open_clip.create_model_and_transforms("hf-hub:" + name[9:])
            tok = open_clip.get_tokenizer("hf-hub:" + name[9:])
        clip = clip.to("cuda").eval()

        def score(ims, texts):  # (crops x labels) probabilities
            x = torch.stack([pre(Image.fromarray(i)) for i in ims]).cuda()
            f, g = clip.encode_image(x), clip.encode_text(tok(texts).cuda())
            f, g = f / f.norm(dim=-1, keepdim=True), g / g.norm(dim=-1, keepdim=True)
            return (clip.logit_scale.exp() * f @ g.T).softmax(-1)
    else:
        from transformers import AutoModel, AutoProcessor
        proc = AutoProcessor.from_pretrained(name)
        sig = AutoModel.from_pretrained(name, torch_dtype=torch.float16 if "giant" in name else None).to("cuda").eval()

        def score(ims, texts):
            inputs = proc(text=texts, images=ims, padding="max_length", max_length=64, return_tensors="pt").to("cuda")
            if "giant" in name:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            return torch.sigmoid(sig(**inputs).logits_per_image.float())
    det = json.loads((OUT / "dino-object.json").read_text())["images"]
    kn = dict(np.load(OUT / "known.npz"))
    rows = []
    for sc in t["scenes"]:
        img = np.asarray(Image.open(SRC / f"{sc['image']}.png").convert("RGB"))
        if which == "dino":
            d = det[sc["image"]]
            idx = keep_boxes(d, known_mask(kn, sc["image"]), float(ARGS.get("thr", 0.12)))
            boxes, scores, det_s = [d["boxes"][i] for i in idx], [d["scores"][i] for i in idx], d["det_s"]
        else:
            boxes = [o["box"] for o in sc["objects"] if o["px"] >= MIN_PX]
            scores, det_s = [1.0] * len(boxes), 0.0
        torch.cuda.synchronize()
        t0 = time.time()
        crops = []
        if boxes:
            ims = []
            for x0, y0, x1, y1 in boxes:
                pad = 0.15 * max(x1 - x0, y1 - y0) + 4
                a, b, c, e = np.clip(np.round([x0 - pad, y0 - pad, x1 + pad, y1 + pad]).astype(int), 0, 640)
                ims.append(img[b:e, a:c])
            vocab = t["vocab"][sc["set"]]
            labels = vocab + NOT_OBJECT
            with torch.inference_mode():  # as see.Eyes.match, all labels at once
                pr = score(ims, [f"a photo of a {n}." for n in labels]).cpu().numpy()
            for bx, s_, row in zip(boxes, scores, pr):
                k = int(row.argmax())
                crops.append(dict(box=bx, score=s_, name=labels[k] if k < len(vocab) else None, parsed=True,
                                  text=labels[k], p=round(float(row[:len(vocab)].max()), 5),
                                  p_none=round(float(row[len(vocab):].max()), 5)))
        torch.cuda.synchronize()
        name_s = time.time() - t0
        rows.append(dict(image=sc["image"], set=sc["set"], listed=listing(crops), text="", tokens=0, prompt_tokens=0,
                         wall_s=det_s + name_s, det_s=det_s, name_s=name_s, crops=crops))
    path = OUT / f"{ARGS.get('tag', 'siglip2')}{'-truthbox' if which == 'truth' else ''}-closed-off.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{path}: {sum(len(r['crops']) for r in rows)} crops, null "
          f"{sum(not c['name'] for r in rows for c in r['crops'])}, "
          f"median {np.median([r['name_s'] for r in rows]) * 1000:.0f} ms per scene")


# --- region models (AWS L40S, 2026-10-06; each in its own environment) ---------------------------------------------
# DAM and PAM describe whatever a box holds, junk included (no null answer): a short answer naming only junk is null.
JUNK = re.compile(r"\b(reflection|reflective|shadow|glare|robot|arm|tray|shelf|table|floor|surface|background)s?\b", re.I)


DAM_Q = "<image>\nDescribe the masked region in a few words."


def short_name(text):
    """A region model's answer as a short name: the first sentence, at most 12 words, no article or full stop; null
    when what is left names only junk (JUNK) and no other noun-like word."""
    s = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0].strip().rstrip(".")
    s = re.sub(r"^(the|a|an|this is|it is|label:|category:)\s+", "", s, flags=re.I).strip()
    s = " ".join(s.split()[:12])
    rest = JUNK.sub("", s)
    return None if JUNK.search(s) and not re.search(r"[a-z]{3,}", re.sub(r"\b(shiny|dark|light|grey|gray|white|black|"
                                                                        r"yellow|blue|red|wooden|flat|part|of|on|the|a|an|piece|edge|corner|its|"
                                                                        r"with|and|small|large|metal)\b", "", rest,
                                                                        flags=re.I), re.I) else (s or None)


def cmd_region():
    """Naming with region models; rows in `name`'s schema.
      --backend=dam  DAM-3B-Self-Contained (NVIDIA Describe Anything): SAM ViT-H mask from each box, then
                     "Describe the masked region in a few words." -> short_name                      (open)
      --backend=pam  PAM-3B (Perceive Anything), the box as its visual prompt, recognize mode
                     "Classify this subject and provide its label." -> short_name                    (open)
      --backend=rex  Rex-Omni on the whole picture, detection with the set's names as categories; its boxes go
                     through keep_boxes (size, NMS, fixture filter; no score threshold) and each kept box is one
                     listed name                                                                    (closed)
    --boxes=truth (dam/pam) names the true boxes instead. Tags: <backend>[-truthbox]-<open|closed>-off."""
    import torch
    from PIL import Image
    be, which = ARGS["backend"], ARGS.get("boxes", "dino")
    t = truth()
    det = json.loads((OUT / "dino-object.json").read_text())["images"]
    kn = dict(np.load(OUT / "known.npz"))
    scenes = t["scenes"]
    if "limit" in ARGS:
        scenes = scenes[::len(scenes) // int(ARGS["limit"])][:int(ARGS["limit"])]
    if "shard" in ARGS:  # --shard=i/k: every k-th scene from i, for parallel processes (merge: cat the -shard files)
        i, k = map(int, ARGS["shard"].split("/"))
        scenes = scenes[i::k]
        ARGS.setdefault("suffix", "")
        ARGS["suffix"] += f"-shard{i}of{k}"
    if be == "dam":
        from transformers import AutoModel, SamModel, SamProcessor
        sam = SamModel.from_pretrained("facebook/sam-vit-huge").to("cuda").eval()
        sproc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        dam = AutoModel.from_pretrained("nvidia/DAM-3B-Self-Contained", trust_remote_code=True,
                                        torch_dtype="torch.float16").to("cuda").init_dam(conv_mode="v1",
                                                                                         prompt_mode="full+focal_crop")

        def describe(im, b):
            inp = sproc(im, input_boxes=[[list(map(float, b))]], return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = sam(**inp)
            m = sproc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                         inp["reshaped_input_sizes"].cpu())[0][0]
            mask = Image.fromarray((m[o.iou_scores[0, 0].argmax()].numpy() * 255).astype(np.uint8))
            return dam.get_description(im, mask, DAM_Q, temperature=0.2, top_p=0.5, num_beams=1, max_new_tokens=48)

        def describe_all(im, boxes):
            """--batch: SAM once for all of a scene's boxes, then one DAM generate over them (the same prompt per box,
            each with its own region; greedy, at most 16 new tokens). DAM's own get_description takes one region."""
            inp = sproc(im, input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = sam(**inp)
            ms = sproc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                          inp["reshaped_input_sizes"].cpu())[0]
            best = o.iou_scores[0].argmax(-1).cpu()
            masks = [Image.fromarray((ms[i, best[i]].numpy() * 255).astype(np.uint8)) for i in range(len(boxes))]
            crop1, crop2 = dam.prompt_mode.split("+")
            ims = [dam.get_image_tensor(im, m, crop_mode=crop1, crop_mode2=crop2) for m in masks]
            prompt, _ = dam.get_prompt(DAM_Q)
            tok = sys.modules[type(dam).__module__].tokenizer_image_token
            ids = tok(prompt, dam.tokenizer, -200, return_tensors="pt").unsqueeze(0).repeat(len(boxes), 1).cuda()
            with torch.inference_mode():
                out = dam.model.generate(input_ids=ids, images=ims, do_sample=False, use_cache=True, max_new_tokens=16)
            return [t.strip() for t in dam.tokenizer.batch_decode(out, skip_special_tokens=True)]
    elif be == "pam":
        from huggingface_hub import snapshot_download
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        path = snapshot_download("Perceive-Anything/PAM-3B") + "/PAM-3B"  # the repo keeps the model in a subfolder
        tok, model, _, _ = load_pretrained_model(path, None, "PAM-3B", multimodal=True, torch_dtype="bfloat16",
                                                 device_map="cuda", attn_implementation="sdpa")
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], "<image>\nClassify this subject and provide its label.")
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        def describe(path_png, b):
            with torch.inference_mode():
                out, _, _ = model.generate(ids, images=[[path_png]], visual_prompts=[[round(v) for v in b]],
                                           vp_labels=[None], types=["image"], do_sample=False, num_beams=1,
                                           max_new_tokens=48, use_cache=True)
            return tok.batch_decode(out, skip_special_tokens=True)[0].strip()
    elif be == "rex":
        from rex_omni import RexOmniWrapper
        rex = RexOmniWrapper(model_path="IDEA-Research/Rex-Omni", backend="transformers", max_tokens=2048, attn_implementation="sdpa",
                             temperature=0.0, top_p=0.05, top_k=1, repetition_penalty=1.05)
    rows = []
    t_all = time.time()
    for sc in scenes:
        path_png = str(SRC / f"{sc['image']}.png")
        im = Image.open(path_png).convert("RGB")
        known = known_mask(kn, sc["image"])
        torch.cuda.synchronize()
        t0, crops = time.time(), []
        if be == "rex":
            pred = rex.inference(images=im, task="detection", categories=t["vocab"][sc["set"]])[0]
            flat = [(c, p["coords"]) for c, ps in (pred.get("extracted_predictions") or {}).items() for p in ps
                    if p.get("type") == "box"]
            d = dict(boxes=[list(map(float, b)) for _, b in flat], scores=[1.0 - 1e-3 * i for i in range(len(flat))])
            for i in keep_boxes(d, known, thr=0.0):
                crops.append(dict(box=d["boxes"][i], score=d["scores"][i], name=flat[i][0], parsed=True,
                                  text=flat[i][0]))
            det_s = 0.0
        else:
            if which == "dino":
                dd = det[sc["image"]]
                idx = keep_boxes(dd, known, float(ARGS.get("thr", 0.12)))
                boxes, scores, det_s = [dd["boxes"][i] for i in idx], [dd["scores"][i] for i in idx], dd["det_s"]
            else:
                boxes = [o["box"] for o in sc["objects"] if o["px"] >= MIN_PX]
                scores, det_s = [1.0] * len(boxes), 0.0
            if be == "dam" and "batch" in ARGS and boxes:
                texts = describe_all(im, boxes)
            else:
                texts = [describe(im if be == "dam" else path_png, b) for b in boxes]
            for b, s, text in zip(boxes, scores, texts):
                crops.append(dict(box=b, score=s, name=short_name(text), parsed=True, text=text[-300:]))
        torch.cuda.synchronize()
        name_s = time.time() - t0
        rows.append(dict(image=sc["image"], set=sc["set"], listed=listing(crops), text="", tokens=0, prompt_tokens=0,
                         wall_s=det_s + name_s, det_s=det_s, name_s=name_s, crops=crops))
    mode = "closed" if be == "rex" else "open"
    path = OUT / f"{be}{'-batch' if 'batch' in ARGS else ''}{'-truthbox' if which == 'truth' else ''}{ARGS.get('suffix', '')}-{mode}-off.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    total = time.time() - t_all
    print(f"{path}: {sum(len(r['crops']) for r in rows)} crops in {len(rows)} scenes, {total:.0f} s "
          f"({total / len(rows):.2f} s per scene), null {sum(not c['name'] for r in rows for c in r['crops'])}")
    (OUT / f"{path.stem}.time").write_text(json.dumps(dict(total_s=total, scenes=len(rows),
                                                           crops=sum(len(r["crops"]) for r in rows))))


# --- judge / report ------------------------------------------------------------------------------------------------
def cmd_judge():
    list_check.OUT = OUT  # same judge model, prompt and pairing rules, on runs/croplist
    list_check.cmd_judge()


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def scores(rows, t):
    """list_check.cmd_report's numbers, per set, as a dict (same logic; checked against its printout)."""
    res = {}
    for s in ("primitives", "scanned"):
        rs = [r for r in rows if r["set"] == s]
        vis = vis_found = guesses = good = count_ok = 0
        for r in rs:
            sc = t[r["image"]]
            found, used = list_check.matches(r, sc)
            guesses += len(r["listed"] or [])
            good += used
            for j, o in enumerate(sc["objects"]):
                if o["px"] >= MIN_PX:
                    vis += 1
                    vis_found += j in found
            count_ok += len(r["listed"] or []) == sum(o["px"] >= MIN_PX for o in sc["objects"])
        res[s] = dict(found=100 * vis_found / max(vis, 1), wrong=100 * (guesses - good) / max(guesses, 1),
                      count=100 * count_ok / max(len(rs), 1), n=len(rs), wall=float(np.median([r["wall_s"] for r in rs])))
    return res


def subset(rows, thr):
    """Rows as if the detector threshold were thr (exact: see keep_boxes). Closed answers only (no re-judging)."""
    out = []
    for r in rows:
        cr = [c for c in r["crops"] if c["score"] >= thr]
        out.append(dict(r, crops=cr, listed=listing(cr)))
    return out


def cmd_report():
    list_check.OUT = OUT
    list_check.cmd_report()  # the reference numbers, printed exactly as for the whole-picture runs
    t = {sc["image"]: sc for sc in truth()["scenes"]}
    det = json.loads((OUT / "dino-object.json").read_text())["images"]
    kn = np.load(OUT / "known.npz")
    print(f"\nDetector alone (Grounding DINO tiny, prompt 'object.', objects with >= {MIN_PX} visible px; recall = "
          f"a kept box with IoU >= 0.5 (0.3) to the object's true box, one-to-one)")
    for thr in (0.12, 0.2, 0.3, 0.4):
        for fx in ((FIXTURE, 0.75, None) if thr == 0.12 else (FIXTURE,)):
            for s in ("primitives", "scanned"):
                vis = r5 = r3 = nbox = junk = cnt = n = 0
                for sc in (x for x in t.values() if x["set"] == s):
                    d = det[sc["image"]]
                    idx = keep_boxes(d, known_mask(kn, sc["image"]) if fx else None, thr, fx or 1.0)
                    boxes = [d["boxes"][i] for i in idx]
                    v = sum(o["px"] >= MIN_PX for o in sc["objects"])
                    p5, p3 = pair(boxes, sc["objects"], 0.5), pair(boxes, sc["objects"], 0.3)
                    vis, r5, r3, nbox, junk = vis + v, r5 + len(p5), r3 + len(p3), nbox + len(boxes), junk + len(boxes) - len(p3)
                    cnt, n = cnt + (len(boxes) == v), n + 1
                print(f"  thr {thr:.2f} {f'fixture filter {fx:.2f}' if fx else 'no fixture filter   '} {s:10s} recall "
                      f"{r5 / vis * 100:.1f}% ({r3 / vis * 100:.1f}% at IoU 0.3) of {vis}; {nbox / n:.1f} boxes per scene, "
                      f"{junk / n:.1f} matching no object (IoU < 0.3); box count = visible count {cnt}/{n}")
    print(f"  detector time: median {np.median([d['det_s'] for d in list(det.values())[5:]]) * 1000:.0f} ms per image")
    for path in sorted(OUT.glob("*-off.jsonl")):
        rows = load(path)
        if "crops" not in rows[0]:
            continue
        print(f"\n{path.stem}: naming accuracy on detected objects (closed: exact name; open: judged pair)")
        judged = OUT / f"{path.stem}-judged.jsonl"
        jrows = {r["image"]: r for r in load(judged)} if judged.exists() else {}
        for s in ("primitives", "scanned"):
            ok = tot = null = 0
            for r in (x for x in rows if x["set"] == s):
                sc = t[r["image"]]
                boxes = [c["box"] for c in r["crops"]]
                for j, i in pair(boxes, sc["objects"], 0.5).items():
                    tot += 1
                    nm = r["crops"][i]["name"]
                    null += nm is None
                    if "-open-" in path.stem:  # the judge paired listed names; find this crop's place in the list
                        listed_i = listing_idx(r["crops"])
                        pairs = {tuple(p) for p in jrows.get(r["image"], {}).get("pairs", [])}
                        ok += i in listed_i and (listed_i.index(i), j) in pairs
                    else:
                        ok += nm == sc["objects"][j]["name"]
            if tot:
                print(f"  {s:10s} {ok}/{tot} named right ({ok / tot * 100:.1f}%), {null} said null")
        if "-closed-" in path.stem and "truthbox" not in path.stem:
            for thr in (0.2, 0.3, 0.4):
                sc_ = scores(subset(rows, thr), t)
                print(f"  if thr {thr:.2f}: " + "; ".join(f"{s} found {v['found']:.1f}% wrong {v['wrong']:.1f}% count "
                                                          f"{v['count']:.1f}%" for s, v in sc_.items()))
        tf = OUT / f"{path.stem}.time"
        if tf.exists():
            tm = json.loads(tf.read_text())
            print(f"  time: median {np.median([r['wall_s'] for r in rows]):.1f} s per scene (detector + its crops one "
                  f"after another, 8 scenes in parallel); throughput {tm['total_s'] / tm['scenes']:.2f} s per scene; "
                  f"{tm['crops'] / tm['scenes']:.1f} crops per scene")


# --- figure --------------------------------------------------------------------------------------------------------
def cmd_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = {sc["image"]: sc for sc in truth()["scenes"]}
    runs = [("whole picture, open", SRC / "gemma4-12b-q4-open-off-judged.jsonl"),
            ("whole picture, closed", SRC / "gemma4-12b-q4-closed-off.jsonl"),
            ("DINO crops, open", OUT / "gemma4-12b-q4-open-off-judged.jsonl"),
            ("DINO crops, closed", OUT / "gemma4-12b-q4-closed-off.jsonl"),
            ("DINO boxes marked\n+ crop, open", OUT / "gemma4-12b-q4-ctx-open-off-judged.jsonl"),
            ("DINO boxes marked\n+ crop, closed", OUT / "gemma4-12b-q4-ctx-closed-off.jsonl"),
            ("true boxes, closed\n(naming only)", OUT / "gemma4-12b-q4-truthbox-closed-off.jsonl"),
            ("true boxes marked,\nclosed (naming only)", OUT / "gemma4-12b-q4-ctx-truthbox-closed-off.jsonl"),
            ("DINO crops,\nSigLIP 2 closed", OUT / "siglip2-closed-off.jsonl"),
            ("true boxes,\nSigLIP 2 closed", OUT / "siglip2-truthbox-closed-off.jsonl")]
    runs = [(n, p) for n, p in runs if p.exists()]
    res = [(n, scores(load(p), t)) for n, p in runs]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5), sharey=True)
    x = np.arange(len(res))
    for ax, s, title in zip(axes, ("primitives", "scanned"), ("simple shapes", "household objects")):
        f = [r[s]["found"] for _, r in res]
        w = [r[s]["wrong"] for _, r in res]
        b1 = ax.bar(x - 0.2, f, 0.4, label="found (% of visible objects)", color="tab:green")
        b2 = ax.bar(x + 0.2, w, 0.4, label="wrong names (% of listed)", color="tab:red")
        ax.bar_label(b1, fmt="%.0f", fontsize=8)
        ax.bar_label(b2, fmt="%.0f", fontsize=8)
        ax.set_xticks(x, [n for n, _ in res], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"{title} (200 scenes)")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("%")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Scene listing: whole picture (4-bit Gemma 4 12B) vs Grounding DINO boxes + naming each crop (Gemma or SigLIP 2)")
    fig.tight_layout()
    fig.savefig(OUT / "methods.png", dpi=130)
    for n, r in res:
        print(n.replace("\n", " "), {s: {k: round(v, 1) for k, v in d.items()} for s, d in r.items()})
    print(f"-> {OUT}/methods.png")


if __name__ == "__main__":
    {"truthboxes": cmd_truthboxes, "detect": cmd_detect, "siglip": cmd_siglip, "name": cmd_name, "judge": cmd_judge,
     "region": cmd_region, "reparse": cmd_reparse, "report": cmd_report, "figure": cmd_figure}[sys.argv[1]]()
