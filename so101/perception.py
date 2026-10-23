"""Off-the-shelf perception for the tabletop task (docs/plan.md row 1): front camera -> where each object is.

Grounding DINO tiny (zero-shot, from transformers) finds boxes for the shape words ("a cube. a ball. ...");
each object takes the box of its shape whose pixels best match its colour (LangGrasp's detector + colour check).
Inside that box, the object's colour pixels are cast onto the table plane at the object's centre height
(the shelf top added when it sits on the shelf); their mean is the position and their main direction (PCA)
the heading of long objects. Returns {object: (x, y, grasp height, heading)}, metres and radians, like
SO101Tabletop.exact(); objects not found are left out.
Measured 2026-10-03 with vla/detect_check.py style tests (see LOG.md).
"""
import mujoco
import numpy as np
import torch

SIZE = 640
SHAPE_WORD = {"red_cube": "cube", "blue_ball": "ball", "yellow_cylinder": "cylinder", "green_bar": "bar"}
HALF_HEIGHT = {"red_cube": 0.012, "blue_ball": 0.015, "yellow_cylinder": 0.02, "green_bar": 0.01}


def colour_mask(img, colour):
    r, g, b = (img[..., i].astype(int) for i in range(3))
    return {"red": r - np.maximum(g, b) > 80, "blue": b - np.maximum(r, g) > 60,
            "yellow": np.minimum(r, g) - b > 80, "green": g - np.maximum(r, b) > 50}[colour]


class Detector:
    def __init__(self, model, data, camera="front"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.model, self.data = model, data
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, SIZE)
        model.vis.global_.offheight = max(model.vis.global_.offheight, SIZE)
        self.renderer = mujoco.Renderer(model, SIZE, SIZE)
        self.cam = model.camera(camera).id
        self.camera = camera
        shelf = model.geom("shelf")  # a fixture: read from the scene, not perceived
        self.shelf_xy, self.shelf_half = model.body("shelf").pos[:2].copy(), shelf.size[:2].copy()
        self.shelf_top = model.body("shelf").pos[2] + shelf.size[2]
        self.f = SIZE / 2 / np.tan(np.radians(model.cam_fovy[self.cam]) / 2)
        name = "IDEA-Research/grounding-dino-tiny"
        self.proc = AutoProcessor.from_pretrained(name)
        self.net = AutoModelForZeroShotObjectDetection.from_pretrained(name).to("cuda").eval()

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def to_plane(self, u, v, z):
        """Pixels (arrays, u right, v down) -> world xy where camera rays meet the plane at height z."""
        R = self.data.cam_xmat[self.cam].reshape(3, 3)
        rays = np.stack([(u - SIZE / 2) / self.f, -(v - SIZE / 2) / self.f, -np.ones_like(u, dtype=float)], -1) @ R.T
        p0 = self.data.cam_xpos[self.cam]
        return p0[:2] + rays[:, :2] * ((z - p0[2]) / rays[:, 2])[:, None]

    def boxes(self, img, words):
        inputs = self.proc(images=img, text=". ".join(f"a {w}" for w in words) + ".", return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = self.net(**inputs)
        r = self.proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.2, text_threshold=0.2,
                                                             target_sizes=[(SIZE, SIZE)])[0]
        return r["boxes"].cpu().numpy(), r["text_labels"]

    def locate(self, objects, img=None):
        img = self.render() if img is None else img
        boxes, labels = self.boxes(img, sorted({SHAPE_WORD[o] for o in objects}))
        found = {}
        for o in objects:
            colour = o.split("_")[0]
            mask = colour_mask(img, colour)
            best, best_n = None, 0
            for box, label in zip(boxes, labels):
                if SHAPE_WORD[o] not in label:
                    continue
                x0, y0, x1, y1 = np.clip(np.round(box).astype(int), 0, SIZE)
                n = mask[y0:y1, x0:x1].sum()
                if n > best_n:
                    best, best_n = (x0, y0, x1, y1), n
            if best is None:
                continue
            x0, y0, x1, y1 = best
            vs, us = np.nonzero(mask[y0:y1, x0:x1])
            us, vs = us + x0, vs + y0
            z = HALF_HEIGHT[o]
            xy = self.to_plane(us, vs, z).mean(0)
            if np.all(np.abs(xy - self.shelf_xy) < self.shelf_half):  # on the shelf: cast again at the shelf height
                z = self.shelf_top + HALF_HEIGHT[o]
            pts = self.to_plane(us, vs, z)
            xy = pts.mean(0)
            evals, evecs = np.linalg.eigh(np.cov((pts - xy).T))
            heading = float(np.arctan2(evecs[1, -1], evecs[0, -1]))  # main direction (only meaningful when long)
            found[o] = (xy[0], xy[1], z, heading)
        return found
