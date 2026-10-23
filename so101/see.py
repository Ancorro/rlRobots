"""Camera perception for general pick-and-place: find an object by name and return its 3D points.

Two RGB-D cameras (scene_general.xml: cam_left, cam_right). For a phrase like "red cylinder":
1. Grounding DINO tiny (open-vocabulary detector) proposes boxes in each camera image.
2. SigLIP 2 (image-text matching) scores each box's crop against the phrase; the best box wins.
   DINO alone often mixes up colours; SigLIP reads them well.
3. SAM 2.1 tiny turns the winning box into a pixel mask; the depth image turns mask pixels into
   world points. The other camera's mask for the same object (the box containing the projected
   centre) adds the points that camera sees, so the far side of the object is covered too.
Table points (z < 7 mm) are dropped. All models are off the shelf, ~1.5 GB of GPU memory in total.
`scene_points()` gives every point above the table (for checking the jaws' path is clear).
Images come from a camera backend (so101/camera.py): the simulator by default, or real cameras.
fit_table: fit the table plane in each camera's points and measure heights from it (corrects a slightly
wrong camera height or tilt from calibration).
"""
import numpy as np
import torch

from so101.camera import SimCameras

SIZE = 640  # simulated camera images, pixels
TABLE_Z = 0.007  # metres; points lower than this are the table (and the 4 mm trays: masks bled onto them)
DINO, SIGLIP = "IDEA-Research/grounding-dino-tiny", "google/siglip2-base-patch16-224"
# Outline model. 2026-10-05, seed 8 (and 9), scanned placed of 400, one camera: webcam-only (estimated depth) tiny 287 (289)
# vs large 297 (305), wrong-object looks 42 (42) vs 34 (23); depth camera tiny / large / SAM 3: primitives 374 / 375 / 376,
# scanned 341 / 336 / 341. Large: +0.7 GB, ~80 vs 27 ms per mask (SAM 3: 173 ms, +1.8 GB, no gain). Earlier runs: tiny.
SAM = "facebook/sam2.1-hiera-large"
SAM3 = "facebook/sam3"  # gated: accept the licence on Hugging Face and log in


def sam3_tracker(cls):
    """SAM 3's box-prompted segmenter (the SAM 2-style tracker head) with its real weights. from_pretrained on
    facebook/sam3 leaves the mask decoder and prompt encoder randomly initialised (2026-10-05: IoU 0.007 with the
    true mask): the checkpoint stores them under tracker_model.*, the shared image encoder under
    detector_model.vision_encoder.backbone.* and the tracker's own feature pyramid under tracker_neck.*."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Sam3TrackerConfig
    ckpt = load_file(hf_hub_download(SAM3, "model.safetensors"))
    sd = {}
    for k, v in ckpt.items():
        if k.startswith("tracker_model."):
            sd[k[len("tracker_model."):]] = v
        elif k.startswith("detector_model.vision_encoder.backbone."):
            sd["vision_encoder." + k[len("detector_model.vision_encoder."):]] = v
        elif k.startswith("tracker_neck."):
            sd["vision_encoder.neck." + k[len("tracker_neck."):]] = v
    model = cls(Sam3TrackerConfig.from_pretrained(SAM3))
    missing, _ = model.load_state_dict(sd, strict=False)  # unused: the video memory parts
    if missing:
        raise RuntimeError(f"SAM 3 tracker weights missing: {missing[:5]}")
    return model


class Eyes:
    def __init__(self, model, data, cams=("cam_left", "cam_right"), device="cuda", finder="dino", sim=None, source=None,
                 fit_table=False, masker=SAM, root="base"):
        """finder: "dino" = Grounding DINO boxes + SigLIP choice + SAM 2.1 mask; "sam3" = SAM 3 alone (text -> masks).
        masker: the model that outlines the chosen box (finder "dino"): a SAM 2.1 checkpoint, or "facebook/sam3"
        (SAM 3's box-prompted tracker head, same interface).
        sim: options for SimCameras (depth_noise, pose_err, light). source: a real camera backend instead.
        root: the robot's root body (its pixels are known geometry, never an object)."""
        from transformers import (AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                                  Sam2Processor, Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor)
        self.dev, self.finder, self.cams, self.root = device, finder, cams, root
        self.sim, self.src, self.real, self.fit_table = sim or {}, source, source is not None, fit_table
        self.erode, self.depth_gate = 0, 0.0  # mask edge pixels to drop; depth spread allowed (x object size), 0 = off
        self.known_reject = 0.5  # drop a mask more than this share robot/fixture pixels (0 = off). 2026-10-04: with
        # camera pose error, ~40 extra looks per 400 tasks picked a sliver of the shelf (6 cm tall, so not "flat")
        if self.sim.get("mono"):  # estimated depth smears object edges onto the background: objects looked 2-3 cm too big.
            self.erode = 1  # seed 7, placed of 200 (primitives / scanned): no trim 168 / 126, 1 px 164 / 146, 2 px 161 / 125
        if finder == "sam3":
            self.sam3_proc = Sam3Processor.from_pretrained(SAM3)
            self.sam3 = Sam3Model.from_pretrained(SAM3).to(device).eval()
            self.attach(model, data)
            return
        self.dino_proc = AutoProcessor.from_pretrained(DINO)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to(device).eval()
        if masker == SAM3:
            self.sam_proc = Sam3TrackerProcessor.from_pretrained(SAM3)
            self.sam = sam3_tracker(Sam3TrackerModel).to(device).eval()
        else:
            self.sam_proc = Sam2Processor.from_pretrained(masker)
            self.sam = Sam2Model.from_pretrained(masker).to(device).eval()
        self.sig_proc = AutoProcessor.from_pretrained(SIGLIP)
        self.sig = AutoModel.from_pretrained(SIGLIP).to(device).eval()
        self.attach(model, data)

    def attach(self, model, data, seed=0):
        """Use a new scene (the benchmark rebuilds the model each episode). seed: the scene's realism draws."""
        self.model, self.data = model, data
        if not self.real:
            if self.src is not None:
                self.src.close()  # (MonoDepth closes its inner sim cameras; the depth model is kept)
            opts = dict(self.sim)
            mono = opts.pop("mono", None)
            self.src = SimCameras(model, data, self.cams, SIZE, seed=seed, **opts)
            if mono:  # depth from the colour image (Depth Anything) instead of the depth camera
                if getattr(self, "mono_net", None) is None:
                    from so101.camera import MonoDepth
                    self.mono_net = MonoDepth(self.src, mono, self.dev)
                self.mono_net.inner, self.mono_net.names = self.src, self.src.names
                self.src = self.mono_net
        if model is None:
            self.known = None
            return
        # known geometry: the robot (its own model at its joint angles) and the calibrated fixtures. On a real
        # robot this is the robot/fixture model rendered into each camera; here the segmentation render does it.
        robot = {b for b in range(model.nbody) if self._under(model, b, model.body(self.root).id)}
        fixtures = {model.body(n).id for n in ("shelf", "red_tray", "blue_tray")}
        self.known = np.array([model.geom_bodyid[g] in robot | fixtures for g in range(model.ngeom)] + [False])
        self.src.known = self.known  # (MonoDepth leaves the robot out of its table fit)

    # --- images and geometry -------------------------------------------------------------------
    def snap(self):
        """RGB and depth (metres along the view axis, 0 = no reading) from every camera."""
        self.frames = [self.src.grab(k) for k in range(len(self.src.names))]
        self.views = [(f.rgb, f.depth) for f in self.frames]
        self.known_px = []  # pixels showing the robot or a fixture (never part of an object)
        for k, f in enumerate(self.frames):
            m = self.src.known_px(k, self.known)
            self.known_px.append(np.zeros(f.rgb.shape[:2], bool) if m is None else m)
        self.corr = [None] * len(self.frames)
        if self.fit_table:
            self.corr = [self._table_correction(k) for k in range(len(self.frames))]
        return self.views

    def _table_correction(self, k):
        """Rigid correction (R, t) for camera k's points that puts the table plane at z = 0, level.
        RANSAC plane through the points within 4 cm of the believed table height, then a least-squares refit."""
        f = self.frames[k]
        h, w = f.depth.shape
        vs, us = np.mgrid[0:h:4, 0:w:4]
        vs, us = vs.ravel(), us.ravel()
        ok = (f.depth[vs, us] > 0) & ~self.known_px[k][vs, us]
        p = self._raw_world(k, us[ok], vs[ok])
        p = p[np.abs(p[:, 2]) < 0.04]
        if len(p) < 500:
            return None
        rng = np.random.default_rng(0)
        A = np.c_[p[:, :2], np.ones(len(p))]
        best = None
        for _ in range(100):
            i = rng.choice(len(p), 3, replace=False)
            try:
                coef = np.linalg.solve(A[i], p[i, 2])
            except np.linalg.LinAlgError:
                continue
            n_in = np.sum(np.abs(A @ coef - p[:, 2]) < 0.003)
            if best is None or n_in > best[0]:
                best = (n_in, coef)
        inl = np.abs(A @ best[1] - p[:, 2]) < 0.003
        a, b, c = np.linalg.lstsq(A[inl], p[inl, 2], rcond=None)[0]
        n = np.array([-a, -b, 1.0]) / np.linalg.norm([-a, -b, 1.0])
        axis = np.cross(n, [0, 0, 1])
        R = np.eye(3) if np.linalg.norm(axis) < 1e-9 else _rot(axis, np.arccos(np.clip(n[2], -1, 1)))
        p0 = np.median(p[inl], 0)
        p0[2] = a * p0[0] + b * p0[1] + c
        t = p0 - R @ p0 - np.array([0, 0, p0[2]])
        return R, t

    @staticmethod
    def _under(model, b, root):
        while b > 0:
            if b == root:
                return True
            b = model.body_parentid[b]
        return False

    def _raw_world(self, k, us, vs):
        f = self.frames[k]
        d = f.depth[vs, us]
        fx, fy, cx, cy = f.K
        p = np.stack([(us - cx) / fx * d, -(vs - cy) / fy * d, -d], -1)
        return p @ f.rot.T + f.pos

    def to_world(self, k, us, vs):
        """Pixels of camera k (with depth > 0) -> world points (table-corrected when fit_table)."""
        p = self._raw_world(k, us, vs)
        if self.corr[k] is not None:
            R, t = self.corr[k]
            p = p @ R.T + t
        return p

    def to_pixel(self, k, p):
        p = np.asarray(p, float)
        if getattr(self, "corr", None) and self.corr[k] is not None:
            R, t = self.corr[k]
            p = (p - t) @ R
        f = self.frames[k] if getattr(self, "frames", None) else self.src.grab(k)
        fx, fy, cx, cy = f.K
        c = (p - f.pos) @ f.rot
        return cx + fx * c[0] / -c[2], cy - fy * c[1] / -c[2]

    def scene_points(self, step=2, min_z=TABLE_Z):
        """World points from every camera (every `step`-th pixel), above `min_z` and below 30 cm."""
        pts = []
        for k, f in enumerate(self.frames):
            if f.depth is None:
                continue
            h, w = f.depth.shape
            vs, us = np.mgrid[0:h:step, 0:w:step]
            vs, us = vs.ravel(), us.ravel()
            ok = f.depth[vs, us] > 0
            p = self.to_world(k, us[ok], vs[ok])
            pts.append(p[(p[:, 2] > min_z) & (p[:, 2] < 0.3)])
        return np.concatenate(pts) if pts else np.zeros((0, 3))

    # --- models -----------------------------------------------------------------------------------
    @torch.inference_mode()
    def boxes(self, img, phrase):
        inputs = self.dino_proc(images=img, text=f"{phrase}. object.", return_tensors="pt").to(self.dev)
        out = self.dino(**inputs)
        r = self.dino_proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=0.12, text_threshold=0.12, target_sizes=[img.shape[:2]])[0]
        b = r["boxes"].cpu().numpy()
        keep = ((b[:, 2] - b[:, 0]) < img.shape[1] * 0.4) & ((b[:, 3] - b[:, 1]) < img.shape[0] * 0.4)  # not the whole table/robot
        return b[keep], r["scores"].cpu().numpy()[keep]

    @torch.inference_mode()
    def match(self, crops, phrase):
        inputs = self.sig_proc(text=[f"a photo of a {phrase}."], images=crops, padding="max_length", max_length=64,
                               return_tensors="pt").to(self.dev)
        return self.sig(**inputs).logits_per_image[:, 0].float().cpu().numpy()

    @torch.inference_mode()
    def mask(self, img, box):
        inputs = self.sam_proc(images=img, input_boxes=[[list(map(float, box))]], return_tensors="pt").to(self.dev)
        out = self.sam(**inputs, multimask_output=False)
        return self.sam_proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0, 0].numpy() > 0

    # --- finding an object ------------------------------------------------------------------------
    def find(self, phrase, snap=True):
        """World points of the object best matching `phrase`, or None. Also sets self.last (for debugging)."""
        if snap:
            self.snap()
        if self.finder == "sam3":
            return self.find_sam3(phrase)
        cands = []  # (camera, box, dino score)
        for k, (img, _) in enumerate(self.views):
            for b, s in zip(*self.boxes(img, phrase)):
                cands.append((k, b, s))
        self.seen = dict(phrase=phrase, cands=[(k, b) for k, b, _ in cands], masks=[])  # for the camera view
        if not cands:
            self.last = None
            return None
        crops = []
        for k, b, _ in cands:
            x0, y0, x1, y1 = b
            pad = 0.15 * max(x1 - x0, y1 - y0) + 4
            h, w = self.views[k][0].shape[:2]
            x0, y0, x1, y1 = np.clip(np.round([x0 - pad, y0 - pad, x1 + pad, y1 + pad]).astype(int), 0, [w, h, w, h])
            crops.append(self.views[k][0][y0:y1, x0:x1])
        sig = self.match(crops, phrase)
        self.seen["sig"] = sig
        for best in np.argsort(-sig)[:4]:  # best match first; skip flat things (a tray) and huge masks
            k, box, _ = cands[best]
            pts = self.mask_points(k, box)
            if pts is not None and pts[:, 2].max() > 0.009 and np.ptp(pts[:, :2], 0).max() < 0.18:
                break
        else:
            self.last = None
            self.seen["masks"] = []
            return None
        self.seen["masks"] = [self.seen["masks"][-1]]  # the winner (rejected tries dropped)
        centre = np.median(pts, 0)
        for j in range(len(self.cams)):  # the same object in the other camera(s)
            if j == k:
                continue
            u, v = self.to_pixel(j, centre)
            inside = [(c, b) for c, (cj, b, _) in enumerate(cands) if cj == j and b[0] <= u <= b[2] and b[1] <= v <= b[3]]
            if inside:
                c, b = max(inside, key=lambda cb: sig[cb[0]])
                more = self.mask_points(j, b)
                if more is not None and np.linalg.norm(np.median(more, 0)[:2] - centre[:2]) < 0.03:
                    pts = np.concatenate([pts, more])
                else:
                    self.seen["masks"].pop()
        self.last = dict(cam=k, box=box, siglip=float(sig[best]), n_cands=len(cands))
        self.seen["best"] = (k, box, float(sig[best]))
        return pts

    def find_2d(self, phrase, snap=True):
        """Image-only find for colour cameras without depth (live webcam test): best box and its mask, no 3D.
        Fills self.seen for the camera view; returns (camera, box, siglip score) or None."""
        if snap:
            self.snap()
        cands = [(k, b) for k, (img, _) in enumerate(self.views) for b in self.boxes(img, phrase)[0]]
        self.seen = dict(phrase=phrase, cands=cands, masks=[])
        if not cands:
            return None
        crops = []
        for k, (x0, y0, x1, y1) in cands:
            h, w = self.views[k][0].shape[:2]
            pad = 0.15 * max(x1 - x0, y1 - y0) + 4
            x0, y0, x1, y1 = np.clip(np.round([x0 - pad, y0 - pad, x1 + pad, y1 + pad]).astype(int), 0, [w, h, w, h])
            crops.append(self.views[k][0][y0:y1, x0:x1])
        sig = self.match(crops, phrase)
        best = int(np.argmax(sig))
        k, box = cands[best]
        self.seen.update(sig=sig, best=(k, box, float(sig[best])), masks=[(k, self.mask(self.views[k][0], box))])
        return self.seen["best"]

    @torch.inference_mode()
    def sam3_masks(self, img, phrase):
        inputs = self.sam3_proc(images=img, text=phrase, return_tensors="pt").to(self.dev)
        out = self.sam3(**inputs)
        r = self.sam3_proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5,
                                                              target_sizes=inputs.get("original_sizes").tolist())[0]
        return r["masks"].cpu().numpy().astype(bool), r["scores"].float().cpu().numpy()

    def find_sam3(self, phrase):
        found = []  # (score, camera, points)
        for k, (img, _) in enumerate(self.views):
            masks, scores = self.sam3_masks(img, phrase)
            for m, sc in zip(masks, scores):
                if m.sum() < m.size * 0.1:  # not the table or the robot
                    p = self.points_from_mask(k, m)
                    if p is not None:
                        found.append((float(sc), k, p))
        if not found:
            self.last = None
            return None
        sc, k, pts = max(found, key=lambda f: f[0])
        centre = np.median(pts, 0)
        for sc2, k2, p2 in sorted(found, key=lambda f: -f[0]):  # the same object seen by the other camera
            if k2 != k and np.linalg.norm(np.median(p2, 0)[:2] - centre[:2]) < 0.03:
                pts = np.concatenate([pts, p2])
                break
        self.last = dict(cam=k, score=sc, n_cands=len(found))
        return pts

    def mask_points(self, k, box):
        m = self.mask(self.views[k][0], box)
        if hasattr(self, "seen"):
            self.seen["masks"].append((k, m))
        return self.points_from_mask(k, m)

    def points_from_mask(self, k, m):
        if self.views[k][1] is None:
            return None
        if self.erode:  # edge pixels: estimated depth blends the object with what is behind it
            import cv2
            m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=self.erode).astype(bool)
        if self.known_reject and m.any() and np.mean(self.known_px[k][m]) > self.known_reject:
            return None  # mostly robot or fixture: a slightly wrong camera pose leaves a sliver of it unmasked
        vs, us = np.nonzero(m & ~self.known_px[k] & (self.views[k][1] > 0))
        if len(us) < 10:
            return None
        if self.depth_gate:  # one visible surface: depths within about the object's size of the median
            d = self.views[k][1][vs, us]
            dm = np.median(d)
            size = np.hypot(np.ptp(us), np.ptp(vs)) * dm / self.frames[k].K[0]
            keep = np.abs(d - dm) < max(self.depth_gate * size, 0.015)
            vs, us = vs[keep], us[keep]
            if len(us) < 10:
                return None
        p = self.to_world(k, us, vs)
        p = p[p[:, 2] > TABLE_Z]
        if len(p) < 10:
            return None
        med = np.median(p, 0)
        dist = np.linalg.norm(p - med, axis=1)
        return p[dist < max(3 * np.median(dist), 0.01)]  # drop stray pixels (mask bleeding onto other things)


def _rot(axis, angle):
    from so101.camera import rot_about
    return rot_about(axis, angle)
