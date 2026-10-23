"""Arm calibration with the camera: a printed marker taped to the gripper, the arm moved through ~20 poses, and one
fit for the joint offsets, the camera pose and where the marker sits on the gripper.

Why: a joint off by 2 degrees puts the gripper ~1 cm from where the software thinks (seed 8 benchmark: placed 306 /
307 of 400 vs 372 / 345), and the hand-measured board puts the camera ~1 cm / 2 deg off (234 / 245). Both errors are
between the camera and the arm; here the camera watches the arm and measures them together.

What the fit can and cannot tell apart (true angle = reading + offset):
  shoulder_pan  turning the whole arm about the base looks the same as moving the camera around the base the other
                way. The board calibration (so101/calib.py) decides it, so pan is only as good as the board's
                placement. Harmless for picking things the camera sees: camera and arm still agree.
  wrist_roll    turning the marker about the roll axis looks the same as a roll offset: decided by how straight
                the marker is taped (marker_deg). A printed clip that holds it square makes this exact.
  the rest      lift, elbow, wrist flex, the camera pose and the marker's place: measured.

  python -m so101.armcal marker            writes runs/calib/marker.png: print at 100%, the black square must
                                           measure 3.0 cm; tape it flat on the outside of the fixed jaw, centred,
                                           its "TIP" edge toward the jaw tips (see MOUNT).
  python -m so101.armcal run --port=COM5 [--cam=0]    real arm + webcam (UNTESTED: no arm yet). Needs the board
                                           calibration (calib lens, calib pose) and the arm zero (robot zero).
                                           Moves the arm through the poses (every leg path-checked), fits, and updates
                                           runs/calib/arm.json (offsets) and runs/calib/cam<N>.json (camera pose).
  python -m so101.armcal selftest [--joint-err=2] [--pose-err=1,2] [--tape=2,3] [--seeds=10] [--gui]
                                           simulator: random joint offsets, camera pose and marker taping errors;
                                           prints what the fit recovers and the gripper-tip error left.
"""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

from so101.camera import rot_about
from so101.robot import CALIB, TICK

MARKER_ID, MARKER = 42, 0.03  # ArUco 4x4 id 42 (the ChArUco board uses ids 0-16); black square 3 cm
BORDER = 0.006  # white border around it, metres
OUT = Path("runs/calib")
# Nominal mount, gripper frame (z = roll axis, jaw tips at -z; the fixed jaw's outside face is at x = -3.6 cm):
# marker centre 6 cm down the jaw, flat on the outside face, facing -x; its image top ("TIP") toward the tips.
MOUNT_POS = np.array([-0.0375, 0.0, -0.06])
MOUNT_ROT = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], float)  # columns: marker x, y, normal (gripper frame)
# Marker corners in the marker frame (x right, y up in the printed image, normal toward the viewer), in ArUco's
# order: top-left, top-right, bottom-right, bottom-left of the printed image.
CORNERS = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * MARKER / 2
DICT = cv2.aruco.DICT_4X4_50
ARM = 5  # arm joints (offsets fitted); the gripper's own joint is not involved


def marker_image(px_per_m=300 / 0.0254, label=True):
    side = int(round(MARKER * px_per_m))
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), MARKER_ID, side)
    b = int(round(BORDER * px_per_m))
    img = cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
    if label:
        cv2.putText(img, "TIP", (b, int(b * 0.8)), cv2.FONT_HERSHEY_SIMPLEX, b / 40, 0, max(1, b // 12))
    return img


def add_marker(spec, pos=MOUNT_POS, rot=MOUNT_ROT, tex_dir=OUT):
    """The marker as a thin see-only plate on the gripper (geom 'cal_marker'). pos/rot: where it really sits."""
    tex_dir.mkdir(parents=True, exist_ok=True)
    path = (tex_dir / "marker_tex.png").resolve()
    if not path.exists():  # write then rename: parallel benchmark workers must not read a half-written file
        tmp = path.with_suffix(f".{np.random.randint(1 << 30)}.png")
        cv2.imwrite(str(tmp), marker_image(4000, label=False))
        tmp.replace(path)
    spec.add_texture(name="cal_marker", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(path))
    mat = spec.add_material(name="cal_marker")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cal_marker"
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, float).ravel())
    half = MARKER / 2 + BORDER
    gripper = spec.body("gripper")
    gripper.add_geom(name="cal_marker", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.0005],
                     pos=np.asarray(pos) - 0.0005 * np.asarray(rot)[:, 2], quat=q, material="cal_marker",
                     contype=0, conaffinity=0, group=1)


def detect(rgb):
    """The marker's 4 corners in the image (ArUco order), sub-pixel, or None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params).detectMarkers(gray)
    if ids is None:
        return None
    hit = [c for c, i in zip(corners, ids.ravel()) if i == MARKER_ID]
    return hit[0].reshape(4, 2) if len(hit) == 1 else None


def _exp(w):
    a = np.linalg.norm(w)
    return np.eye(3) if a < 1e-12 else rot_about(w, a)


def _log(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class Fit:
    """Parameters (23): joint offsets (5), camera (translation 3, rotation 3, from the believed pose), marker mount
    (3 + 3, from nominal). Pixel residuals plus priors: camera ~ board calibration (cam_cm, cam_deg), marker ~
    nominal mount (tape_mm, tape_deg), offsets ~ 0 (joint_deg, loose)."""

    def __init__(self, model, K, cam_pos, cam_rot, cam_cm=1.0, cam_deg=2.0, tape_mm=3.0, tape_deg=3.0, joint_deg=10.0,
                 px=0.5, mount=(MOUNT_POS, MOUNT_ROT)):
        self.model, self.K = model, K
        self.d = mujoco.MjData(model)
        self.g = model.body("gripper").id
        self.cam0 = (np.asarray(cam_pos, float), np.asarray(cam_rot, float))
        self.mount0 = (np.asarray(mount[0], float), np.asarray(mount[1], float))
        self.sig = np.r_[np.full(ARM, np.radians(joint_deg)), np.full(3, cam_cm / 100), np.full(3, np.radians(cam_deg)),
                         np.full(3, tape_mm / 1000), np.full(3, np.radians(tape_deg))]
        self.px = px

    def unpack(self, x):
        e = x[:ARM]
        cam = (self.cam0[0] + x[5:8], _exp(x[8:11]) @ self.cam0[1])
        mount = (self.mount0[0] + x[11:14], _exp(x[14:17]) @ self.mount0[1])
        return e, cam, mount

    def project(self, x, q):
        """Pixel corners of the marker with the arm reading q (6 angles)."""
        e, (cp, cR), (mp, mR) = self.unpack(x)
        self.d.qpos[:6] = q
        self.d.qpos[:ARM] += e
        mujoco.mj_kinematics(self.model, self.d)
        gp, gR = self.d.xpos[self.g], self.d.xmat[self.g].reshape(3, 3)
        w = gp + (mp + CORNERS @ mR.T) @ gR.T
        c = (w - cp) @ cR
        fx, fy, cx, cy = self.K
        return np.stack([cx + fx * c[:, 0] / -c[:, 2], cy - fy * c[:, 1] / -c[:, 2]], 1)

    def residual(self, x, obs):
        r = [((self.project(x, q) - uv) / self.px).ravel() for q, uv in obs]
        return np.concatenate(r + [x / self.sig])

    def solve(self, obs, iters=50):
        """Levenberg-Marquardt from the believed calibration. Returns (x, rms pixel error)."""
        x, lam = np.zeros(17), 1e-3
        r = self.residual(x, obs)
        for _ in range(iters):
            J = np.empty((len(r), len(x)))
            for j in range(len(x)):
                h = 1e-6
                J[:, j] = (self.residual(x + h * np.eye(len(x))[j], obs) - r) / h
            A, b = J.T @ J, J.T @ r
            while True:
                step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), b)
                r2 = self.residual(x + step, obs)
                if r2 @ r2 < r @ r:
                    x, r, lam = x + step, r2, lam / 3
                    break
                lam *= 10
                if lam > 1e8:
                    break
            if lam > 1e8 or np.abs(step).max() < 1e-9:
                break
        n = 8 * len(obs)
        return x, float(np.sqrt(np.mean((r[:n] * self.px) ** 2)))


# --- poses ----------------------------------------------------------------------------------------------
HOME = np.array([0.0, -1.0, 1.0, 0.6, 0.0, 0.0])  # arm up and back, clear of the table


def _robot_geoms(model):
    root = model.body("base").id

    def under(b):
        while b > 0:
            if b == root:
                return True
            b = model.body_parentid[b]
        return False
    return [g for g in range(model.ngeom) if under(model.geom_bodyid[g]) and model.geom_contype[g] | model.geom_conaffinity[g]]


def pose_ok(model, d, q, cam, K, size, low=0.10, facing=0.5, margin=40):
    """Believed pose q is safe (gripper high, no self-contact) and shows the nominal marker to the camera."""
    d.qpos[:6] = q
    mujoco.mj_forward(model, d)
    g = model.body("gripper").id
    for b in ("wrist", "gripper", "moving_jaw_so101_v1"):
        if d.xpos[model.body(b).id][2] < low:
            return False
    tip = d.site_xpos[model.site("gripperframe").id]
    if tip[2] < low or tip[2] > 0.36 or np.linalg.norm(tip[:2]) > 0.36:
        return False
    robot = set(_robot_geoms(model))
    for c in d.contact[:d.ncon]:
        if c.geom1 in robot and c.geom2 in robot:
            return False
    gR = d.xmat[g].reshape(3, 3)
    centre = d.xpos[g] + gR @ MOUNT_POS
    normal = gR @ MOUNT_ROT[:, 2]
    to_cam = cam[0] - centre
    if normal @ to_cam / np.linalg.norm(to_cam) < facing:
        return False
    c = (d.xpos[g] + (MOUNT_POS + CORNERS @ MOUNT_ROT.T) @ gR.T - cam[0]) @ cam[1]
    fx, fy, cx, cy = K
    u, v = cx + fx * c[:, 0] / -c[:, 2], cy - fy * c[:, 1] / -c[:, 2]
    return bool(np.all((u > margin) & (u < size[0] - margin) & (v > margin) & (v < size[1] - margin)))


def plan_poses(model, cam, K, size, n=20, seed=0, tries=20000):
    """A route from HOME back to HOME: [(q, shoot)], n believed joint poses that show the marker (shoot=True),
    spread over the joints, plus HOME stops where no straight joint path is safe. Every leg passes path_ok."""
    rng, d = np.random.default_rng(seed), mujoco.MjData(model)
    lo, hi = model.jnt_range[:6].T
    pool = []
    for _ in range(tries):
        q = np.r_[rng.uniform(lo[:ARM] * 0.8, hi[:ARM] * 0.8), lo[5]]
        if pose_ok(model, d, q, cam, K, size) and path_ok(model, d, HOME, q):
            pool.append(q)
            if len(pool) >= 10 * n:
                break
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} usable poses found: is the marker in view of the camera?")
    pool = np.array(pool)
    pick = [int(rng.integers(len(pool)))]  # farthest-point spread over the arm joints
    dist = np.linalg.norm(pool[:, :ARM] - pool[pick[0], :ARM], axis=1)
    while len(pick) < n:
        pick.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(pool[:, :ARM] - pool[pick[-1], :ARM], axis=1))
    route, left, cur = [], list(pick), HOME
    while left:  # nearest safe next; back to HOME when none (every pose is safe from HOME)
        ok = [i for i in left if path_ok(model, d, cur, pool[i])]
        if ok:
            j = min(ok, key=lambda i: np.abs(pool[i, :ARM] - cur[:ARM]).max())
            route.append((pool[j], True))
            cur = pool[j]
            left.remove(j)
        elif cur is not HOME and path_ok(model, d, cur, HOME):
            route.append((HOME, False))
            cur = HOME
        else:
            break
    while route and not path_ok(model, d, route[-1][0], HOME):
        route.pop()
    return route + [(HOME, False)]


def path_ok(model, d, a, b, low=0.08, reach=0.38, steps=12):
    """The straight joint path a -> b keeps the gripper high and inside the guard's reach (with a margin)."""
    tip = model.site("gripperframe").id
    for t in np.linspace(0, 1, steps):
        d.qpos[:6] = a + t * (b - a)
        mujoco.mj_kinematics(model, d)
        if min(d.xpos[model.body(n).id][2] for n in ("wrist", "gripper", "moving_jaw_so101_v1")) < low:
            return False
        if np.linalg.norm(d.site_xpos[tip][:2]) > reach or d.site_xpos[tip][2] > reach:
            return False
    return True


def go(arm, q, speed=2.0, ticks=100, settle=10):
    """Drive the arm to q along the straight joint path (the one path_ok checked) at `speed` rad/s, under the
    guard's 4 rad/s cap, and let it settle. Returns the averaged reading."""
    a = arm.read()
    n = max(1, int(np.ceil(np.abs(q[:ARM] - a[:ARM]).max() / (speed * TICK))))
    for t in np.linspace(0, 1, n + 1)[1:]:
        arm.send(a + t * (q - a))
        arm.tick()
    for _ in range(ticks):
        if np.abs(arm.read()[:ARM] - q[:ARM]).max() < 0.01:
            break
        arm.send(q)
        arm.tick()
    reads = []
    for _ in range(settle):
        arm.send(q)
        arm.tick()
        reads.append(arm.read())
    return np.mean(reads[-5:], 0)


def collect(arm, model, route, grab, viewer=None):
    """Go to HOME (start with the arm parked near it), then follow the route, reading the joints and finding the
    marker at each pose. grab() -> RGB image. Returns [(reading, corners)] for the poses where the marker was found."""
    obs = []
    go(arm, HOME)
    for q, shoot in route:
        reading = go(arm, q)
        if viewer is not None:
            viewer.sync()
        if shoot:
            uv = detect(grab())
            if uv is not None:
                obs.append((reading, uv))
    return obs


def calibrate(arm, model, grab, K, size, cam, n=20, seed=0, viewer=None, **priors):
    """Plan, move, fit. cam: believed camera (pos, rot). Returns dict with offsets (5, radians), cam (pos, rot),
    mount (pos, rot), rms_px, used poses."""
    route = plan_poses(model, cam, K, size, n, seed)
    obs = collect(arm, model, route, grab, viewer)
    if len(obs) < 8:
        raise RuntimeError(f"marker found in only {len(obs)} of {sum(s for _, s in route)} poses")
    fit = Fit(model, K, *cam, **priors)
    x, rms = fit.solve(obs)
    e, cam_est, mount = fit.unpack(x)
    return {"offsets": e, "cam": cam_est, "mount": mount, "rms_px": rms, "poses": len(obs)}


class Offset:
    """An arm plug-in with fitted offsets applied: reads true = reading + offset, sends target - offset."""

    def __init__(self, arm, offsets):
        self.arm = arm
        self.e = np.r_[offsets, 0.0]

    def read(self):
        return self.arm.read() + self.e

    def send(self, q):
        self.arm.send(np.asarray(q, float) - self.e)

    def __getattr__(self, name):  # tick, stop, relax, close
        return getattr(self.arm, name)


# --- simulator check --------------------------------------------------------------------------------------
def sim_model(mount=(MOUNT_POS, MOUNT_ROT)):
    from so101.objects import SCENE
    spec = mujoco.MjSpec.from_file(str(SCENE))
    add_marker(spec, *mount)
    return spec.compile()


def taped_mount(rng, tape):
    """Marker pose as taped by hand: off by tape[0] mm and tilted/turned by tape[1] deg (random directions), never
    sunk into the jaw: it lifts off the face by its tilt (one edge rests on the jaw) plus up to 1 mm of tape."""
    d = rng.normal(size=3)
    t = d / np.linalg.norm(d) * tape[0] / 1000
    R = rot_about(rng.normal(size=3), np.radians(tape[1])) @ MOUNT_ROT
    n = MOUNT_ROT[:, 2]
    tilt = np.arccos(np.clip(R[:, 2] @ n, -1, 1))
    t = t - (t @ n) * n + (np.sin(tilt) * (MARKER / 2 + BORDER) + rng.uniform(0, 0.001)) * n
    return MOUNT_POS + t, R


def sim_calibrate(sim_arm, cams, k=0, seed=0, tape=(2.0, 3.0), cam_prior=(1.0, 2.0)):
    """Benchmark hook: calibrate a SimArm's joint offsets and camera k of a SimCameras (both with their realism
    errors) as on the real arm, in an empty copy of the scene with a hand-taped marker (tape: mm, deg off nominal).
    Fixes camera k's believed pose in place; returns the joint offsets (wrap the arm in Offset)."""
    from so101.camera import SimCameras
    from so101.robot import Guard, SimArm
    rng = np.random.default_rng(seed + 1000)
    mount = taped_mount(rng, tape)
    model = sim_model(mount)
    data = mujoco.MjData(model)
    data.qpos[:6] = HOME
    mujoco.mj_forward(model, data)
    arm = SimArm(model, data)
    arm.err, arm.play, arm.quantise = sim_arm.err.copy(), sim_arm.play, sim_arm.quantise
    view = SimCameras(model, data, [cams.names[k]], cams.size)
    f = view.grab(0)
    t, R = cams.err[k]
    pos, rot = f.pos + t, R @ f.rot  # what the board calibration believes
    r = calibrate(Guard(arm, model), model, lambda: view.render_rgb(0), f.K, (cams.size, cams.size), (pos, rot),
                  seed=seed, cam_cm=cam_prior[0], cam_deg=cam_prior[1], tape_mm=max(tape[0], 0.5),
                  tape_deg=max(tape[1], 0.5))
    view.close()
    true_pos, true_rot = f.pos, f.rot  # the calibration scene's camera has no error
    cams.err[k] = (r["cam"][0] - true_pos, r["cam"][1] @ true_rot.T)
    return r["offsets"], r


def tip_error(model, err_true, e_est, rng, n=200):
    """Gripper-tip distance, true vs believed, over random reachable poses, after applying e_est (mm)."""
    d1, d2 = mujoco.MjData(model), mujoco.MjData(model)
    lo, hi = model.jnt_range[:6].T
    tip, out = model.site("gripperframe").id, []
    for _ in range(n):
        q = rng.uniform(lo * 0.7, hi * 0.7)
        d1.qpos[:6] = q  # believed (after correction)
        d2.qpos[:6] = q
        d2.qpos[:ARM] += err_true[:ARM] - e_est  # true angle = command - e_est + err
        mujoco.mj_kinematics(model, d1)
        mujoco.mj_kinematics(model, d2)
        out.append(np.linalg.norm(d1.site_xpos[tip] - d2.site_xpos[tip]) * 1000)
    return np.array(out)


def selftest_one(seed, joint_err, pose_err, tape, cams=("cam_front",), n=20, verbose=True):
    from so101.camera import SimCameras
    from so101.robot import Guard, SimArm
    rng = np.random.default_rng(seed + 1000)
    mount = taped_mount(rng, tape)
    model = sim_model(mount)
    data = mujoco.MjData(model)
    sim = SimArm(model, data, joint_err=joint_err, seed=seed)
    data.qpos[:6] = HOME
    mujoco.mj_forward(model, data)
    arm = Guard(sim, model)
    if "--gui" in sys.argv:  # watch it: the viewer follows every tick, in real time
        import time
        from mujoco import viewer as mjviewer
        viewer = mjviewer.launch_passive(model, data)
        tick = sim.tick
        sim.tick = lambda: (tick(), viewer.sync(), time.sleep(TICK))
    eyes = SimCameras(model, data, cams, 640, pose_err=pose_err, seed=seed)
    f = eyes.grab(0)
    true_cam = (data.cam_xpos[eyes.ids[0]].copy(), data.cam_xmat[eyes.ids[0]].reshape(3, 3).copy())
    r = calibrate(arm, model, lambda: eyes.render_rgb(0), f.K, (640, 640), (f.pos, f.rot), n=n, seed=seed,
                  cam_cm=max(pose_err[0], 0.2) if pose_err else 0.2, cam_deg=max(pose_err[1], 0.2) if pose_err else 0.2,
                  tape_mm=max(tape[0], 0.5), tape_deg=max(tape[1], 0.5))
    e = r["offsets"]
    out = {"seed": seed, "poses": r["poses"], "rms_px": r["rms_px"],
           "joint_err_deg_before": np.degrees(sim.err[:ARM]), "joint_err_deg_after": np.degrees(sim.err[:ARM] - e),
           "cam_mm_before": np.linalg.norm(f.pos - true_cam[0]) * 1000,
           "cam_mm_after": np.linalg.norm(r["cam"][0] - true_cam[0]) * 1000,
           "cam_deg_before": np.degrees(np.linalg.norm(_log(f.rot @ true_cam[1].T))),
           "cam_deg_after": np.degrees(np.linalg.norm(_log(r["cam"][1] @ true_cam[1].T))),
           "tip_mm_before": np.median(tip_error(model, sim.err, np.zeros(ARM), np.random.default_rng(seed))),
           "tip_mm_after": np.median(tip_error(model, sim.err, e, np.random.default_rng(seed)))}
    # what matters for picking: a point the camera locates vs where the arm goes. Camera error and pan error both
    # shift that; compare the believed-camera view of the tip with the corrected arm's tip (true frame).
    out["hand_eye_mm_after"] = hand_eye(model, sim.err, e, true_cam, r["cam"], np.random.default_rng(seed))
    out["hand_eye_mm_before"] = hand_eye(model, sim.err, np.zeros(ARM), true_cam, (f.pos, f.rot), np.random.default_rng(seed))
    if verbose:
        print(f"seed {seed}: {r['poses']} poses, {r['rms_px']:.2f} px rms | joints off (deg) before "
              f"{np.round(out['joint_err_deg_before'], 2)} after {np.round(out['joint_err_deg_after'], 2)} | "
              f"camera {out['cam_mm_before']:.1f} mm / {out['cam_deg_before']:.2f} deg -> "
              f"{out['cam_mm_after']:.1f} / {out['cam_deg_after']:.2f} | tip {out['tip_mm_before']:.1f} -> "
              f"{out['tip_mm_after']:.1f} mm | hand-eye {out['hand_eye_mm_before']:.1f} -> {out['hand_eye_mm_after']:.1f} mm")
    eyes.close()
    return out


def hand_eye(model, err_true, e_est, true_cam, est_cam, rng, n=200):
    """Median distance (mm) between where the arm's tip really goes and where the camera says the target is, for
    targets on the table: the camera sees a point p (true frame), reports it in its believed frame; the arm is sent
    there with the corrected joints. Uses table points at 0-10 cm height, reached jaw-down by the arm's IK."""
    from so101.arm import ik_jaw_down
    d, scratch, real = mujoco.MjData(model), mujoco.MjData(model), mujoco.MjData(model)
    lo, hi = model.jnt_range[:6].T
    tip, wrist = model.site("gripperframe").id, model.body("gripper").id
    out = []
    for _ in range(n):
        r, a = rng.uniform(0.15, 0.25), rng.uniform(-0.8, 0.8)
        p = np.array([r * np.cos(a), r * np.sin(a), rng.uniform(0.0, 0.1)])
        seen = est_cam[1] @ (true_cam[1].T @ (p - true_cam[0])) + est_cam[0]  # camera-frame point, believed pose
        d.qpos[:6] = HOME
        q = ik_jaw_down(model, d, scratch, tip, wrist, seen, lo, hi)
        real.qpos[:6] = HOME
        real.qpos[:ARM] = q[:ARM] - e_est + err_true[:ARM]
        mujoco.mj_kinematics(model, real)
        out.append(np.linalg.norm(real.site_xpos[tip] - p) * 1000)
    return float(np.median(out))


def _arg(name, default):
    return next((a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")), default)


def cmd_selftest():
    je = float(_arg("joint-err", 2))
    pe = tuple(map(float, _arg("pose-err", "1,2").split(",")))
    tape = tuple(map(float, _arg("tape", "2,3").split(",")))
    rows = [selftest_one(s, je, pe, tape) for s in range(int(_arg("seeds", 10)))]
    for k in ("tip_mm", "hand_eye_mm", "cam_mm", "cam_deg"):
        b, a = np.median([r[k + "_before"] for r in rows]), np.median([r[k + "_after"] for r in rows])
        print(f"median {k}: {b:.2f} -> {a:.2f}")
    after = np.abs([r["joint_err_deg_after"] for r in rows])
    print("median |joint error| after, deg (pan lift elbow flex roll):", np.round(np.median(after, 0), 2))


def cmd_marker():
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "marker.png"), marker_image())
    print(f"wrote {OUT / 'marker.png'}: print at 100% (300 dpi); the black square must measure 3.0 cm")


def cmd_run():
    """Real arm + webcam. UNTESTED (no arm yet)."""
    from so101.camera import Webcam
    from so101.objects import SCENE
    from so101.robot import Guard, RealArm
    cam_i = int(_arg("cam", 0))
    path = OUT / f"cam{cam_i}.json"
    if not path.exists() or "pos" not in json.loads(path.read_text()):
        sys.exit("run python -m so101.calib lens and pose first")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    real = RealArm(_arg("port", "COM5"))
    arm = Guard(real, model)
    web = Webcam(cam_i)
    f = web.grab()
    size = f.rgb.shape[1], f.rgb.shape[0]
    input("The arm will move through ~20 poses above the table. Clear the table, keep a hand on the power plug. Enter.")
    try:
        r = calibrate(arm, model, web.render_rgb, f.K, size, (f.pos, f.rot))
    finally:
        arm.stop()
    print(f"{r['poses']} poses, {r['rms_px']:.2f} px rms (under 1 is good)")
    print("joint offsets, deg:", np.round(np.degrees(r["offsets"]), 2))
    cal = json.loads(CALIB.read_text())
    cal["offset"] = (np.array(cal["offset"]) + r["offsets"]).tolist()  # q = sign*rad(raw) + offset: add the fix
    CALIB.write_text(json.dumps(cal, indent=1))
    c = json.loads(path.read_text())
    c.update(pos=r["cam"][0].tolist(), rot=r["cam"][1].tolist(), arm_calibrated=True)
    path.write_text(json.dumps(c, indent=1))
    print(f"updated {CALIB} and {path}")
    real.close()
    web.close()


if __name__ == "__main__":
    {"marker": cmd_marker, "run": cmd_run, "selftest": cmd_selftest}[sys.argv[1]]()
