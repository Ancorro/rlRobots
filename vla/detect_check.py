"""Can an off-the-shelf detector give the pick skill the target cube's position? (docs/plan.md row 1: perception.)

For each held-out pick layout (same as so101/pixel_pick.py: two cubes plus two pale trays of the same colours):
render the front camera at --size px, find "the green cube" (or red) in the image, and turn the pixel into a
table position by intersecting the camera ray with the plane at the cube centre's height (z = 1.5 cm).
Reports per method: right cube found (within 2 cm of the named cube), xy error in cm, ms per image.

Methods:
  mask  - the simulator's own segmentation mask of the cube (perfect detector): the floor set by the geometry
  gdino - Grounding DINO tiny, prompt "a green cube."
  gdino+colour - Grounding DINO tiny, prompt "a cube.", then the box whose pixels are greenest / reddest
  owlv2 - OWLv2 base, prompts "a green cube" and "a red cube"; best box for the named one
--picks: the scripted pick (so101/skills.py) is sent to each method's position instead of the true one,
and to the true position moved 0 / 0.3 / 0.6 / 1.0 cm in a random direction (how precise perception must be).

Usage:  python vla/detect_check.py [--episodes=40] [--size=640] [--methods=mask,gdino,gdino+colour,owlv2] [--picks]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio
import mujoco
import numpy as np
import torch

from so101.pixel_pick import SO101PixelPick
from so101.skill_env import CUBES
from so101.skills import CONTROL_STEPS, HOVER_Z, move_to, set_grip
from so101.arm import GRIP_CLOSED

ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
EPISODES, SIZE = int(ARGS.get("episodes", 40)), int(ARGS.get("size", 640))
METHODS = ARGS.get("methods", "mask,gdino,gdino+colour,owlv2").split(",")
CUBE_Z = 0.015  # cube centre height when resting, metres
OFFSETS = [0, 0.3, 0.6, 1.0]  # cm, for the precision curve

env = SO101PixelPick()
m, d = env.model, env.data
m.vis.global_.offwidth = m.vis.global_.offheight = max(SIZE, m.vis.global_.offwidth)
renderer = mujoco.Renderer(m, SIZE, SIZE)
cam = m.camera("front").id
f = SIZE / 2 / np.tan(np.radians(m.cam_fovy[cam]) / 2)  # focal length, pixels (fovy is vertical; square image)


def render(segmentation=False):
    if segmentation:
        renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera="front")
    img = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    return img


def to_table(u, v, z=CUBE_Z):
    """Pixel (u right, v down) -> world xy where the camera ray meets the plane at height z."""
    R = d.cam_xmat[cam].reshape(3, 3)  # columns: camera x (right), y (up), z (backwards)
    ray = R @ np.array([(u - SIZE / 2) / f, -(v - SIZE / 2) / f, -1.0])
    p0 = d.cam_xpos[cam]
    return (p0 + ray * (z - p0[2]) / ray[2])[:2]


def to_pixel(p):
    R = d.cam_xmat[cam].reshape(3, 3)
    c = R.T @ (np.asarray(p) - d.cam_xpos[cam])
    return SIZE / 2 + f * c[0] / -c[2], SIZE / 2 - f * c[1] / -c[2]


def box_centre(box):
    x0, y0, x1, y1 = [float(b) for b in box]
    return (x0 + x1) / 2, (y0 + y1) / 2


class GDino:
    def __init__(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        name = "IDEA-Research/grounding-dino-tiny"
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(name).to("cuda").eval()

    def boxes(self, img, text):
        inputs = self.proc(images=img, text=text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model(**inputs)
        r = self.proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.2, text_threshold=0.2,
                                                             target_sizes=[(SIZE, SIZE)])[0]
        return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy()


class Owl:
    def __init__(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        name = "google/owlv2-base-patch16-ensemble"
        self.proc = Owlv2Processor.from_pretrained(name)
        self.model = Owlv2ForObjectDetection.from_pretrained(name).to("cuda").eval()

    def boxes(self, img, texts):
        inputs = self.proc(text=[texts], images=img, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.model(**inputs)
        r = self.proc.post_process_grounded_object_detection(out, threshold=0.05, target_sizes=[(SIZE, SIZE)])[0]
        return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), r["labels"].cpu().numpy()


def colour_score(img, box, colour):
    x0, y0, x1, y1 = [int(round(float(b))) for b in box]
    patch = img[max(y0, 0):y1, max(x0, 0):x1].reshape(-1, 3).astype(float)
    if not len(patch):
        return -1e9
    r, g, b = patch.T
    return float(np.mean(g - np.maximum(r, b)) if colour == "green" else np.mean(r - np.maximum(g, b)))


def detect(method, img, colour):
    """-> pixel (u, v) of the named cube, or None."""
    if method == "mask":
        seg = render(segmentation=True)
        ys, xs = np.nonzero((seg[..., 0] == m.geom(f"{colour}_cube").id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
        return (xs.mean(), ys.mean()) if len(xs) else None
    if method == "gdino":
        boxes, scores = models["gdino"].boxes(img, f"a {colour} cube.")
        return box_centre(boxes[scores.argmax()]) if len(boxes) else None
    if method == "gdino+colour":
        boxes, scores = models["gdino"].boxes(img, "a cube.")
        boxes = [b for b in boxes if (b[2] - b[0]) < SIZE / 4]  # drop whole-table boxes
        return box_centre(max(boxes, key=lambda b: colour_score(img, b, colour))) if boxes else None
    if method == "owlv2":
        boxes, scores, labels = models["owlv2"].boxes(img, ["a green cube", "a red cube"])
        keep = labels == (0 if colour == "green" else 1)
        return box_centre(boxes[keep][scores[keep].argmax()]) if keep.any() else None


def pick_at(arm, obj, xy):
    """so101.skills.pick, but sent to xy (from perception) instead of the cube's true position."""
    z = arm.body_pos(obj)[2]
    yield from move_to(arm, [xy[0], xy[1], HOVER_Z])
    yield from move_to(arm, [xy[0], xy[1], z + 0.04])
    yield from move_to(arm, [xy[0], xy[1], z], speed=0.15 / 4)
    yield from set_grip(arm, GRIP_CLOSED)
    yield from move_to(arm, [xy[0], xy[1], HOVER_Z])
    return arm.holding(obj)


def run_pick(seed, xy):
    env.reset(seed=seed)
    skill = pick_at(env.base.arm, env.cube, xy)
    while True:
        try:
            d.ctrl[:] = next(skill)
        except StopIteration as stop:
            return bool(stop.value)
        mujoco.mj_step(m, d, nstep=CONTROL_STEPS)


models = {}
if any(k.startswith("gdino") for k in METHODS):
    models["gdino"] = GDino()
if "owlv2" in METHODS:
    models["owlv2"] = Owl()

res = {k: [] for k in METHODS}  # (xy error cm or nan, right cube, ms, picked)
geom_check, sample = [], None
for ep in range(EPISODES):
    env.reset(seed=ep)
    colour = env.cube.split("_")[0]
    true, other = env.true_cm / 100, env.other_start
    geom_check.append(np.linalg.norm(to_table(*to_pixel([*true, CUBE_Z])) - true) * 100)
    img = render()
    dets = {}
    for k in METHODS:
        t = time.perf_counter()
        uv = detect(k, img, colour)
        ms = (time.perf_counter() - t) * 1000
        xy = to_table(*uv) if uv is not None else None
        dets[k] = xy
        err = np.linalg.norm(xy - true) * 100 if xy is not None else np.nan
        right = xy is not None and err < 2 and np.linalg.norm(xy - true) < np.linalg.norm(xy - other)
        res[k].append([err, right, ms, np.nan])
    if ep == 0:
        sample = img.copy()
        for k, xy in dets.items():
            if xy is not None:
                u, v = (int(round(c)) for c in to_pixel([*xy, CUBE_Z]))
                sample[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3] = [255, 255, 0]
        imageio.imwrite("runs/detect_sample.png", sample)
    if "--picks" in sys.argv:
        for k in METHODS:
            res[k][-1][3] = run_pick(ep, dets[k]) if dets[k] is not None else False

print(f"{EPISODES} held-out layouts, front camera {SIZE} px; geometry self-check (true centre -> pixel -> table): {np.max(geom_check):.3f} cm max")
print(f"{'method':14s} right cube   xy error cm median / p90 (right cube only)   ms/image" + ("   scripted pick" if "--picks" in sys.argv else ""))
for k, rows in res.items():
    r = np.array(rows, dtype=float)
    ok = r[:, 1] > 0
    e = r[ok, 0]
    line = f"{k:14s} {ok.mean():6.0%}       {np.median(e) if len(e) else np.nan:5.2f} / {np.percentile(e, 90) if len(e) else np.nan:5.2f}                     {np.median(r[1:, 2]) if len(r) > 1 else r[0, 2]:6.0f}"
    if "--picks" in sys.argv:
        line += f"          {int(np.nansum(r[:, 3]))}/{EPISODES}"
    print(line)
if "--picks" in sys.argv:
    rng = np.random.default_rng(0)
    curve = []
    for off in OFFSETS:
        n = 0
        for ep in range(EPISODES):
            env.reset(seed=ep)
            a = rng.uniform(0, 2 * np.pi)
            n += run_pick(ep, env.true_cm / 100 + off / 100 * np.array([np.cos(a), np.sin(a)]))
        curve.append(f"{off:g} cm: {n}/{EPISODES}")
    print("scripted pick at the true position moved by  " + ", ".join(curve))
