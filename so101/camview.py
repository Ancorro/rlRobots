"""Live window of what the cameras see, with what perception found drawn on top (bench_general.py --camview), and
`Recorder`, which writes the same picture beside a view of the whole scene to a video (bench_general.py --video=DIR).

Each frame: both camera images side by side. After a look: every box Grounding DINO proposed (grey),
the one SigLIP picked (green, with the phrase and its score), SAM's mask tinted green, and the planned
grasp (the line the jaws close along). A status line says the task and step.
Display only: nothing here feeds back into the robot. OpenCV draws; a Tk window shows it
(the installed OpenCV is the headless build, without windows).
"""
import tkinter

import cv2
import numpy as np



class CamView:
    def __init__(self, eyes, every=4, scale=0.75, window=True):
        """window=False: no window, show() only builds self.frame (for Recorder; the cluster has no display)."""
        self.eyes, self.every, self.scale, self.tick, self.window = eyes, every, scale, 0, window
        self.status, self.overlay, self.grasp = "", False, None
        if window:
            from PIL import ImageTk
            self._tk = ImageTk
            self.root = tkinter.Tk()
            self.root.title("cameras")
            self.label = tkinter.Label(self.root)
            self.label.pack()

    def found(self, grasp=None):
        """Show the latest look (eyes.seen) and its planned grasp until clear()."""
        self.overlay, self.grasp = True, grasp
        self.show(force=True)

    def clear(self):
        self.overlay, self.grasp = False, None

    def show(self, force=False):
        self.tick += 1
        if not force and self.tick % self.every:
            return
        e, seen = self.eyes, getattr(self.eyes, "seen", None)
        imgs = []
        for k, c in enumerate(e.src.names):
            img = cv2.cvtColor(e.src.render_rgb(k), cv2.COLOR_RGB2BGR)
            if self.overlay and seen:
                for kk, m in seen.get("masks", []):
                    if kk == k:
                        img[m] = (0.45 * img[m] + 0.55 * np.array([60, 220, 60])).astype(np.uint8)
                        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img, cnts, -1, (40, 255, 40), 2)
                sig = seen.get("sig")
                for i, (kk, b) in enumerate(seen.get("cands", [])):
                    if kk == k:
                        x0, y0, x1, y1 = map(int, b)
                        cv2.rectangle(img, (x0, y0), (x1, y1), (150, 150, 150), 1)
                        if sig is not None:
                            cv2.putText(img, f"{sig[i]:.1f}", (x0, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                if seen.get("best") and seen["best"][0] == k:
                    x0, y0, x1, y1 = map(int, seen["best"][1])
                    cv2.rectangle(img, (x0, y0), (x1, y1), (40, 255, 40), 2)
                    label = seen["phrase"]
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(img, (x0, y0 - h - 8), (x0 + w + 6, y0), (40, 255, 40), -1)
                    cv2.putText(img, label, (x0 + 3, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                if self.grasp is not None:  # the jaws' closing line across the object, at grasp height
                    g = self.grasp
                    u = np.array([np.cos(g.yaw), np.sin(g.yaw)]) * (g.gap / 2)
                    ends = [e.to_pixel(k, [*(g.xy + s * u), g.z]) for s in (-1, 1)]
                    p0, p1 = [tuple(int(round(v)) for v in p) for p in ends]
                    cv2.line(img, p0, p1, (0, 200, 255), 2)
                    for p in (p0, p1):
                        cv2.circle(img, p, 4, (0, 200, 255), -1)
            cv2.putText(img, c, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            imgs.append(img)
        frame = np.concatenate(imgs, 1)
        bar = np.zeros((34, frame.shape[1], 3), np.uint8)
        cv2.putText(bar, self.status, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        frame = np.concatenate([bar, frame], 0)
        if self.scale != 1:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.frame = frame
        if not self.window:
            return
        from PIL import Image
        self.photo = self._tk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.label.configure(image=self.photo)
        self.root.update()

    def close(self):
        if self.window:
            self.root.destroy()


class Recorder:
    """Video of a run: a view of the whole scene (left) beside CamView's camera picture (right, with what perception
    found), under its status line. One frame per CamView frame; H.264 mp4, plays in a phone browser."""

    def __init__(self, path, model, data, cam, fps=30, size=480, wide=False):
        """wide: a view that takes in a humanoid standing behind the table (G1)."""
        import av
        import mujoco
        self.mujoco, self.data, self.cam, self.size = mujoco, data, cam, size
        self.r = mujoco.Renderer(model, size, size)
        self.view = mujoco.MjvCamera()
        self.view.lookat[:] = (0.05, 0.0, 0.15) if wide else (0.17, 0.0, 0.04)
        self.view.distance, self.view.azimuth, self.view.elevation = (1.5, 150.0, -25.0) if wide else (0.85, 145.0, -30.0)
        self.out = av.open(str(path), "w")
        self.stream = None
        self.fps, self.last = fps, -1

    def add(self):
        if self.cam.tick == self.last or getattr(self.cam, "frame", None) is None:
            return  # CamView builds a frame every `every` ticks: record those
        self.last = self.cam.tick
        self.r.update_scene(self.data, self.view)
        scene = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)
        right = self.cam.frame
        h = right.shape[0]
        scene = cv2.resize(scene, (h, h), interpolation=cv2.INTER_AREA)
        frame = np.concatenate([scene, right], 1)
        frame = frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]  # even size for yuv420p
        if self.stream is None:
            self.stream = self.out.add_stream("libx264", rate=self.fps)
            self.stream.width, self.stream.height = frame.shape[1], frame.shape[0]
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "26", "preset": "medium", "movflags": "+faststart"}
        import av
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format="bgr24")):
            self.out.mux(packet)

    def close(self):
        if self.stream is not None:
            for packet in self.stream.encode():
                self.out.mux(packet)
        self.out.close()
        self.r.close()
