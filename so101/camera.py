"""Camera backends for so101/see.py: the same frames from the simulator or from real cameras.

A backend holds one or more cameras and gives, per camera, a `Frame`: colour image, depth image
(metres along the view axis; 0 = no reading; None for a plain webcam), the lens intrinsics
(fx, fy, cx, cy in pixels; pixel (0, 0) is the centre of the top-left pixel) and the pose the robot
*believes* the camera has (from calibration): position and rotation in the robot's frame, MuJoCo
convention (the camera looks along its -z, +y is image up). OpenCV-convention poses (z forward,
y down) convert with `from_opencv`.

SimCameras  renders scene_general.xml's cameras. Realism options (all off by default, for testing
            what a real camera does to the stack): depth noise and holes, a believed pose that is off
            by a calibration error, random lighting and image noise.
Webcam      plain USB camera through OpenCV (colour only). Works with the headless OpenCV build.
RealSense   Intel RealSense D4xx (colour + aligned depth). Untested: no camera here yet; needs
            `pip install pyrealsense2`.
"""
from dataclasses import dataclass

import mujoco
import numpy as np

CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> MuJoCo camera axes (and back)


@dataclass
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None  # (H, W) metres, 0 = no reading
    K: tuple  # fx, fy, cx, cy
    pos: np.ndarray  # believed camera position (robot frame)
    rot: np.ndarray  # believed camera rotation, MuJoCo convention (columns: camera x, y, z in robot frame)


def from_opencv(R_cv, t):
    """Camera pose from OpenCV convention (e.g. cv2.solvePnP, inverted to camera-to-world) to MuJoCo convention."""
    return np.asarray(t, float), np.asarray(R_cv, float) @ CV_TO_MJ


def rot_about(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


class SimCameras:
    """MuJoCo cameras by name. Options:
    depth_noise: realistic RGB-D depth (stereo noise growing with distance, holes at edges, on dark surfaces
                 and at random; 1 mm steps).
    pose_err:    (cm, degrees): the believed pose is off by this much (random direction per scene), like a
                 slightly wrong calibration. Images still come from the true camera.
    light:       random light strength and direction, table tint and image noise per scene."""

    def __init__(self, model, data, names, size=640, depth_noise=False, pose_err=None, light=False, seed=0):
        self.model, self.data, self.names, self.size = model, data, list(names), size
        self.depth_noise, self.rng = depth_noise, np.random.default_rng(seed)
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, size)
        model.vis.global_.offheight = max(model.vis.global_.offheight, size)
        self.rgb_r = mujoco.Renderer(model, size, size)
        self.depth_r = mujoco.Renderer(model, size, size)
        self.depth_r.enable_depth_rendering()
        self.seg_r = mujoco.Renderer(model, size, size)
        self.seg_r.enable_segmentation_rendering()
        self.ids = [model.camera(c).id for c in self.names]
        self.f = [size / 2 / np.tan(np.radians(model.cam_fovy[i]) / 2) for i in self.ids]
        self.err = []  # per camera: (translation, rotation) of the believed pose relative to the true one
        for _ in self.ids:
            if pose_err:
                d = self.rng.normal(size=3)
                t = d / np.linalg.norm(d) * pose_err[0] / 100
                self.err.append((t, rot_about(self.rng.normal(size=3), np.radians(pose_err[1]))))
            else:
                self.err.append((np.zeros(3), np.eye(3)))
        if light:
            self._random_light()

    def _random_light(self):
        m, r = self.model, self.rng
        for i in range(m.nlight):
            m.light_diffuse[i] = r.uniform(0.3, 1.0) * np.ones(3) * r.uniform(0.9, 1.1, 3)
            d = np.array([r.uniform(-0.6, 0.6), r.uniform(-0.6, 0.6), -1.0])
            m.light_dir[i] = d / np.linalg.norm(d)
        m.vis.headlight.ambient[:] = r.uniform(0.1, 0.45)
        m.vis.headlight.diffuse[:] = r.uniform(0.2, 0.8)
        floor = m.geom("floor").matid
        if floor >= 0:
            m.mat_rgba[floor, :3] = r.uniform(0.6, 1.0, 3)
        self.image_noise = r.uniform(1, 5)

    def close(self):
        self.rgb_r.close(), self.depth_r.close(), self.seg_r.close()

    def render_rgb(self, k):
        self.rgb_r.update_scene(self.data, camera=self.names[k])
        img = self.rgb_r.render().copy()
        if getattr(self, "image_noise", 0):
            img = np.clip(img + self.rng.normal(0, self.image_noise, img.shape), 0, 255).astype(np.uint8)
        return img

    def pose(self, k):
        """Believed pose of camera k: the true one moved by the calibration error."""
        i = self.ids[k]
        pos, rot = self.data.cam_xpos[i].copy(), self.data.cam_xmat[i].reshape(3, 3).copy()
        t, R = self.err[k]
        return pos + t, R @ rot

    def grab(self, k):
        self.depth_r.update_scene(self.data, camera=self.names[k])
        rgb, depth = self.render_rgb(k), self.depth_r.render().copy()
        if self.depth_noise:
            depth = self._noisy(depth, rgb)
        c = self.size / 2 - 0.5
        return Frame(rgb, depth, (self.f[k], self.f[k], c, c), *self.pose(k))

    def _noisy(self, d, rgb):
        """D4xx-like depth: error grows with distance squared (about 2 mm at 40 cm), smooth and pixel noise,
        holes where depth jumps (edges), on very dark surfaces and at random; 1 mm steps."""
        r, n = self.rng, self.size
        smooth = r.normal(size=(n // 32 + 2, n // 32 + 2))
        smooth = np.kron(smooth, np.ones((32, 32)))[:n, :n]  # blotchy low-frequency error, like real stereo
        d = d + d ** 2 * (0.008 * r.normal(size=d.shape) + 0.008 * smooth)
        gy, gx = np.abs(np.diff(d, axis=0, prepend=d[:1])), np.abs(np.diff(d, axis=1, prepend=d[:, :1]))
        edge = (gx > 0.01) | (gy > 0.01)
        edge |= np.roll(edge, 1, 0) | np.roll(edge, 1, 1)
        dark = rgb.mean(-1) < 25
        hole = (edge & (r.random(d.shape) < 0.7)) | (dark & (r.random(d.shape) < 0.5)) | (r.random(d.shape) < 0.02)
        d = np.round(d, 3)
        d[hole] = 0
        return d

    def true_bodies(self, k):
        """Per pixel of camera k, the top-level body (child of the world) it shows, from the true pose; -1 = none.
        Benchmark scoring only: which object a mask really covers."""
        self.seg_r.update_scene(self.data, camera=self.names[k])
        seg = self.seg_r.render()
        geom = np.where((seg[..., 0] >= 0) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        root = self.model.body_rootid[self.model.geom_bodyid]
        return np.where(geom >= 0, root[np.maximum(geom, 0)], -1)

    def known_px(self, k, known):
        """Pixels of camera k showing geoms marked in `known` (the robot and fixtures), as rendered from the
        *believed* pose: on a real robot this is its model drawn into the image, so a calibration error
        misplaces it the same way."""
        i = self.ids[k]
        pos, rot = self.data.cam_xpos[i].copy(), self.data.cam_xmat[i].copy()
        self.data.cam_xpos[i], self.data.cam_xmat[i] = self.pose(k)[0], self.pose(k)[1].ravel()
        self.seg_r.update_scene(self.data, camera=self.names[k])
        seg = self.seg_r.render()
        self.data.cam_xpos[i], self.data.cam_xmat[i] = pos, rot
        ids = seg[..., 0]
        return known[np.where(ids >= 0, ids, -1)] & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)


class Webcam:
    """Plain USB camera (colour only, no depth). Lens and pose come from runs/calib/cam<index>.json
    (so101/calib.py) when it exists: images are then undistorted. Otherwise K is a guess from the field of view."""

    def __init__(self, index=0, width=640, height=480, K=None, fov_deg=70, pose=None, calib_dir="runs/calib"):
        import cv2
        self.cv2, self.names = cv2, [f"webcam{index}"]
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError(f"webcam {index} gave no image")
        h, w = img.shape[:2]
        f = w / 2 / np.tan(np.radians(fov_deg) / 2)
        self.K = K or (f, f, w / 2 - 0.5, h / 2 - 0.5)
        self.pos, self.rot = pose or (np.zeros(3), np.eye(3))
        self.undistort = None
        from pathlib import Path
        import json
        path = Path(calib_dir) / f"cam{index}.json"
        if path.exists():
            c = json.loads(path.read_text())
            Km = np.array([[c["K"][0], 0, c["K"][2]], [0, c["K"][1], c["K"][3]], [0, 0, 1]])
            self.undistort = (Km, np.array(c["dist"]))  # undistorted images keep the same K
            self.K = tuple(c["K"])
            if "pos" in c:
                self.pos, self.rot = np.array(c["pos"]), np.array(c["rot"])

    def render_rgb(self, k=0):
        ok, img = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        if self.undistort is not None:
            img = self.cv2.undistort(img, *self.undistort)
        return self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)

    def grab(self, k=0):
        return Frame(self.render_rgb(), None, self.K, self.pos, self.rot)

    def known_px(self, k, known=None):
        return None  # no robot model drawn yet

    def close(self):
        self.cap.release()


class RealSense:
    """Intel RealSense D4xx: colour plus depth aligned to the colour image, hole filling off (holes stay 0).
    UNTESTED (no camera yet). pose: believed (pos, rot) from calibration, MuJoCo convention."""

    def __init__(self, width=640, height=480, fps=30, pose=None, serial=None):
        import pyrealsense2 as rs
        self.rs, self.names = rs, [f"realsense{'' if serial is None else '-' + serial}"]
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipe = rs.pipeline()
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.temporal = rs.temporal_filter()  # averages a few frames: less noise on a still scene
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = (i.fx, i.fy, i.ppx, i.ppy)
        self.pos, self.rot = pose or (np.zeros(3), np.eye(3))
        for _ in range(15):  # auto exposure settles
            self.pipe.wait_for_frames()

    def _frames(self):
        fs = self.align.process(self.pipe.wait_for_frames())
        return fs.get_color_frame(), fs.get_depth_frame()

    def render_rgb(self, k=0):
        return np.asanyarray(self._frames()[0].get_data()).copy()

    def grab(self, k=0):
        for _ in range(4):  # temporal filter needs a few frames
            c, d = self._frames()
            d = self.temporal.process(d)
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.scale
        return Frame(np.asanyarray(c.get_data()).copy(), depth, self.K, self.pos, self.rot)

    def known_px(self, k, known=None):
        return None  # robot model not drawn yet (needs the arm's joint angles and calibration)

    def close(self):
        self.pipe.stop()


MONO_MODELS = {"small": "depth-anything/Depth-Anything-V2-Small-hf", "base": "depth-anything/Depth-Anything-V2-Base-hf",
               "large": "depth-anything/Depth-Anything-V2-Large-hf",
               "metric-small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
               "metric-base": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
               "depthpro": "apple/DepthPro-hf",  # metric; transformers
               # need their own packages (cluster .venv-depth, RUNBOOK): depth_anything_3 and moge
               "da3-mono": "depth-anything/DA3MONO-LARGE", "da3-metric": "depth-anything/DA3METRIC-LARGE",
               "moge2-l": "Ruicheng/moge-2-vitl-normal", "moge2-b": "Ruicheng/moge-2-vitb-normal"}
# what each model's output is, for the table fit: "inverse" (disparity-like) or "depth" (distance-like)
MONO_KIND = {"metric-small": "depth", "metric-base": "depth", "da3-mono": "depth", "da3-metric": "depth",
             "moge2-l": "depth", "moge2-b": "depth"}


class MonoDepth:
    """Depth from a plain colour camera: a depth model guesses depth from the image; the known table plane
    (z = 0 in the robot frame, seen through the calibrated camera pose) fixes its scale and offset each frame.

    Models (MONO_MODELS): Depth Anything V2 and Depth Pro through transformers; Depth Anything 3 and MoGe-2 through
    their own packages (MoGe-2 is told the camera's field of view). Inverse-depth outputs (V2 relative, Depth Pro's
    canonical inverse depth): fit 1/z_table = a * out + b on pixels whose ray hits the table. Depth outputs (metric
    V2, DA3, MoGe-2): fit z_table = a * out + b. The fit drops the 30% worst-fitting pixels three times (objects,
    the robot), so mostly real table pixels decide it.
    Wraps another backend (SimCameras with its depth thrown away, or a Webcam)."""

    def __init__(self, inner, model="small", device="cuda", workspace=((0.0, 0.45), (-0.3, 0.3))):
        import torch
        self.inner, self.names, self.dev, self.ws = inner, inner.names, device, workspace
        self.torch, self.family = torch, model.split("-")[0] if model.startswith(("da3", "moge2")) else "hf"
        self.metric = MONO_KIND.get(model) == "depth"
        name = MONO_MODELS.get(model, model)
        if self.family == "da3":
            from depth_anything_3.api import DepthAnything3
            self.net = DepthAnything3.from_pretrained(name).to(device).eval()
        elif self.family == "moge2":
            from moge.model.v2 import MoGeModel
            self.net = MoGeModel.from_pretrained(name).to(device).eval()
        else:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self.proc = AutoImageProcessor.from_pretrained(name)
            self.net = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()

    def render_rgb(self, k):
        return self.inner.render_rgb(k)

    def known_px(self, k, known=None):
        return self.inner.known_px(k, known)

    def close(self):
        self.inner.close()

    def predict(self, rgb, K=None):
        """The model's raw output at the image's size (NaN where it gives none). K: lens (fx, fy, cx, cy)."""
        h, w = rgb.shape[:2]
        F = self.torch.nn.functional
        with self.torch.inference_mode():
            if self.family == "da3":
                out = self.torch.from_numpy(self.net.inference([rgb]).depth[:1].astype(np.float32))
            elif self.family == "moge2":
                img = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.dev)
                fov = None if K is None else float(np.degrees(2 * np.arctan(w / 2 / K[0])))
                out = self.net.infer(img, fov_x=fov)["depth"][None].float()
            else:
                inputs = self.proc(images=rgb, return_tensors="pt").to(self.dev)
                out = self.net(**inputs).predicted_depth.float()
            if out.shape[-2:] != (h, w):
                out = F.interpolate(out[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
        out = out[0].cpu().numpy()
        return np.where(np.isfinite(out), out, np.nan)

    def table_depth(self, f):
        """Depth each pixel would have if its ray hit the table plane z = 0 (NaN if not, or outside the workspace)."""
        h, w = f.rgb.shape[:2]
        vs, us = np.mgrid[0:h, 0:w]
        fx, fy, cx, cy = f.K
        ray = np.stack([(us - cx) / fx, -(vs - cy) / fy, -np.ones_like(us, float)], -1) @ f.rot.T  # per unit depth
        d = -f.pos[2] / ray[..., 2]
        hit = f.pos + ray * d[..., None]
        ok = (d > 0) & (hit[..., 0] > self.ws[0][0]) & (hit[..., 0] < self.ws[0][1]) & \
             (hit[..., 1] > self.ws[1][0]) & (hit[..., 1] < self.ws[1][1])
        return np.where(ok, d, np.nan)

    def grab(self, k):
        f = self.inner.grab(k)
        out = self.predict(f.rgb, f.K)
        zt = self.table_depth(f)
        m = self.inner.known_px(k, getattr(self, "known", None))
        use = ~np.isnan(zt) & ~np.isnan(out) if m is None else ~np.isnan(zt) & ~np.isnan(out) & ~m
        x, y = out[use], (zt[use] if self.metric else 1 / zt[use])
        keep = np.ones(len(x), bool)
        for _ in range(3):
            a, b = np.polyfit(x[keep], y[keep], 1)
            r = np.abs(a * x + b - y)
            keep = r <= np.percentile(r, 70)
        v = a * out + b
        depth = v if self.metric else np.where(v > 1e-3, 1 / np.maximum(v, 1e-3), 0)
        depth = np.where((depth > 0.02) & (depth < 3), depth, 0).astype(np.float32)
        self.fit = (a, b, float(np.median(r[keep])))
        return Frame(f.rgb, depth, f.K, f.pos, f.rot)
