"""Camera calibration with a printed ChArUco board (a chessboard with ArUco markers in the white squares).

Three steps, each a command:
  python -m so101.calib board              writes runs/calib/board.png: print it at 100% (A4/Letter, landscape),
                                           check a square measures 3.0 cm, tape it flat to something stiff.
  python -m so101.calib lens [--cam=0]     lens calibration: hold the board in front of the camera at different
                                           angles; it keeps a view every ~1.5 s when the board is seen, 15 views.
                                           Saves focal length, centre and distortion to runs/calib/cam<N>.json.
  python -m so101.calib pose [--cam=0] [--board-at=0.20,0]
                                           camera position: lay the board flat on the table, its centre at
                                           board-at (metres from the robot base: x forward, y left), the
                                           "TOWARD ROBOT" edge (top of the image file) facing the robot.
                                           Adds the camera pose (robot frame) to runs/calib/cam<N>.json.
  python -m so101.calib selftest           checks the maths in simulation: the board on the sim table, camera
                                           pose recovered from cam_front's image vs the true pose.

Limits: the board position is measured by hand (a few mm). When the arm arrives, touching known points
with the gripper will refine the board-to-robot offset.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SQUARES, SQUARE, MARKER = (7, 5), 0.03, 0.022  # board: 7 x 5 squares of 3 cm (21 x 15 cm), 2.2 cm markers
OUT = Path("runs/calib")
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])


def board():
    return cv2.aruco.CharucoBoard(SQUARES, SQUARE, MARKER, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def board_image(px_per_m=300 / 0.0254):
    """Board picture at print resolution (300 dpi by default), with a margin and the 'toward robot' label on top."""
    b = board()
    w, h = int(SQUARES[0] * SQUARE * px_per_m), int(SQUARES[1] * SQUARE * px_per_m)
    img = b.generateImage((w, h))
    m = int(0.01 * px_per_m)
    img = cv2.copyMakeBorder(img, 4 * m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(img, "TOWARD ROBOT  (squares must measure 3.0 cm)", (m, 3 * m), cv2.FONT_HERSHEY_SIMPLEX,
                m / 40, 0, max(1, m // 15))
    return img, m


def detect(img):
    """Board corners in an image: (object points (N,3) in the board frame, image points (N,2)) or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board()).detectBoard(gray)
    if ids is None or len(ids) < 6:
        return None
    obj, pts = board().matchImagePoints(corners, ids)
    return obj.reshape(-1, 3), pts.reshape(-1, 2)


def camera_in_board(img, K, dist=None):
    """Camera pose in the board frame (OpenCV convention: x right, y down along the board, z into it)."""
    found = detect(img)
    if found is None:
        return None
    obj, pts = found
    Km = np.array([[K[0], 0, K[2]], [0, K[1], K[3]], [0, 0, 1]])
    ok, rvec, tvec = cv2.solvePnP(obj, pts, Km, None if dist is None else np.asarray(dist))
    if not ok:
        return None
    R = cv2.Rodrigues(rvec)[0]  # board -> camera
    err = cv2.projectPoints(obj, rvec, tvec, Km, None if dist is None else np.asarray(dist))[0].reshape(-1, 2) - pts
    return R.T, (-R.T @ tvec).ravel(), float(np.sqrt((err ** 2).sum(1).mean()))  # camera -> board, position, px error


def board_to_robot(centre_xy, yaw=0.0):
    """Board frame -> robot frame for a board lying flat, centre at centre_xy, its image top facing the robot
    (yaw turns it about the vertical). Board x (image right) -> robot +y, board y (image down) -> robot +x,
    board z -> down."""
    R0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R0
    centre_b = np.array([SQUARES[0] * SQUARE / 2, SQUARES[1] * SQUARE / 2, 0])
    t = np.array([*centre_xy, 0.0]) - R @ centre_b
    return R, t


def camera_in_robot(img, K, dist=None, centre_xy=(0.20, 0.0), yaw=0.0, thickness=0.0):
    """Believed camera pose (robot frame, MuJoCo convention: looks along -z) from one image of the board on the
    table. thickness: board height above the table (stiff backing), metres. Returns (pos, rot, px error) or None."""
    r = camera_in_board(img, K, dist)
    if r is None:
        return None
    Rcb, pcb, err = r
    Rbr, tbr = board_to_robot(centre_xy, yaw)
    tbr = tbr + [0, 0, thickness]
    return Rbr @ pcb + tbr, Rbr @ Rcb @ CV_TO_MJ, err


# --- commands -------------------------------------------------------------------------------------------
def _arg(name, default):
    for a in sys.argv:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


def cmd_board():
    OUT.mkdir(parents=True, exist_ok=True)
    img, _ = board_image()
    cv2.imwrite(str(OUT / "board.png"), img)
    print(f"wrote {OUT / 'board.png'}: print at 100% scale (300 dpi), landscape; one square must be 3.0 cm")


def _window(title):
    import tkinter
    from PIL import Image, ImageTk
    root = tkinter.Tk()
    root.title(title)
    label = tkinter.Label(root)
    label.pack()
    text = tkinter.Label(root, font=("Segoe UI", 12))
    text.pack()

    def show(img, msg):
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        label.configure(image=photo)
        label.image = photo
        text.configure(text=msg)
        root.update()
    return root, show


def cmd_lens():
    cam = int(_arg("cam", 0))
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    root, show = _window("lens calibration")
    objs, pts, last, size = [], [], 0.0, None
    while len(objs) < 15:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("no camera image")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        size = rgb.shape[1], rgb.shape[0]
        found = detect(rgb)
        view = rgb.copy()
        if found is not None:
            for p in found[1]:
                cv2.circle(view, tuple(int(v) for v in p), 3, (0, 255, 0), -1)
            if time.time() - last > 1.5 and len(found[0]) >= 12:
                objs.append(found[0].astype(np.float32))
                pts.append(found[1].astype(np.float32))
                last = time.time()
        show(view, f"views kept: {len(objs)}/15  -  tilt and move the board between views")
    root.destroy()
    err, Km, dist, _, _ = cv2.calibrateCamera(objs, pts, size, None, None)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"cam{cam}.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(size=size, K=[Km[0, 0], Km[1, 1], Km[0, 2], Km[1, 2]], dist=dist.ravel().tolist(), lens_error_px=err)
    path.write_text(json.dumps(data, indent=1))
    print(f"lens error {err:.2f} px (under 1 is good); saved {path}")


def cmd_pose():
    cam = int(_arg("cam", 0))
    path = OUT / f"cam{cam}.json"
    if not path.exists():
        sys.exit("run the lens step first")
    data = json.loads(path.read_text())
    centre = tuple(map(float, _arg("board-at", "0.20,0").split(",")))
    cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
    poses = []
    for _ in range(60):  # average a few good views
        ok, frame = cap.read()
        if not ok:
            continue
        r = camera_in_robot(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), data["K"], data["dist"], centre)
        if r is not None and r[2] < 2.0:
            poses.append(r)
        if len(poses) >= 10:
            break
    if not poses:
        sys.exit("board not seen clearly: check lighting and that the whole board is in view")
    pos = np.median([p[0] for p in poses], 0)
    rot = poses[len(poses) // 2][1]
    data.update(pos=pos.tolist(), rot=rot.tolist(), board_at=centre, pose_error_px=float(np.median([p[2] for p in poses])))
    path.write_text(json.dumps(data, indent=1))
    print(f"camera at {np.round(pos * 100, 1)} cm from the robot base; saved {path}")


def cmd_selftest():
    """Board texture on the sim table at (0.20, 0); recover cam_front's pose from its image."""
    import mujoco
    from so101.objects import SCENE
    img, m = board_image(px_per_m=4000)
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = (OUT / "board_tex.png").resolve()
    cv2.imwrite(str(tex_path), img)
    h, w = img.shape
    sx, sy = w / 4000 / 2, h / 4000 / 2  # half sizes of the printed sheet (with margins), metres
    spec = mujoco.MjSpec.from_file(str(SCENE))
    spec.add_texture(name="board", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(tex_path))
    mat = spec.add_material(name="board")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "board"
    # sheet centre: the board centre shifted by the uneven margin (more on top = toward the robot = -x)
    centre = np.array([0.20, 0.0])
    off_top = (4 * m - m) / 4000 / 2
    body = spec.worldbody.add_body(pos=[centre[0] - off_top, centre[1], 0.0005], quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.0005], material="board", contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    from so101.camera import SimCameras
    cams = SimCameras(model, data, ["cam_front", "cam_left"])
    for k, name in enumerate(cams.names):
        f = cams.grab(k)
        r = camera_in_robot(f.rgb, f.K, None, centre)
        if r is None:
            cv2.imwrite(str(OUT / f"selftest_{name}.png"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR))
            print(name, "board not detected; image saved")
            continue
        pos, rot, err = r
        ang = np.degrees(np.arccos(np.clip((np.trace(rot.T @ f.rot) - 1) / 2, -1, 1)))
        print(f"{name}: position error {np.linalg.norm(pos - f.pos) * 1000:.1f} mm, angle error {ang:.2f} deg, "
              f"reprojection {err:.2f} px")


if __name__ == "__main__":
    {"board": cmd_board, "lens": cmd_lens, "pose": cmd_pose, "selftest": cmd_selftest}[sys.argv[1]]()
