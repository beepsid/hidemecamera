"""
Hide Me Cam - a virtual webcam that teleports you in and out of frame with a salute.

Reads your real webcam, watches for a salute (hand to brow, elbow raised), and
streams the result to the OBS Virtual Camera, so any app (Discord, WhatsApp,
Zoom, Meet, Teams, browsers, streaming software) can use it.

  1. Click "Capture background" and step out of frame (3 s countdown).
  2. Salute while visible  -> you dissolve out (the empty-room photo is shown)
     Salute while hidden   -> you beam back in

Preview window keys:
  B  capture background     S  toggle without saluting
  D  debug overlay          M  mirror preview          Q / Esc  quit
"""

import argparse
import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision

APP_NAME = "Hide Me Cam"
MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

HOLD_SECONDS = 0.35        # salute must be held this long to trigger
COOLDOWN_SECONDS = 1.5     # minimum gap between triggers
TRANSITION_SECONDS = 1.2   # length of the spawn / despawn effect
CAPTURE_COUNTDOWN = 3.0    # time to step out of frame after clicking "Capture background"
DETECT_WIDTH = 480         # pose detection runs on a small copy of the frame
DETECT_EVERY = 2           # ...and only on every 2nd frame (15 checks/s is plenty)
GLOW_BGR = (255, 220, 80)  # cyan teleport glow

# Pose landmark indices
L_EYE, R_EYE, L_EAR, R_EAR = 2, 5, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_IDX, R_IDX = 11, 12, 13, 14, 15, 16, 19, 20
SIDES = {
    "right": dict(sh=R_SH, el=R_EL, wr=R_WR, idx=R_IDX, eye=R_EYE, ear=R_EAR),
    "left": dict(sh=L_SH, el=L_EL, wr=L_WR, idx=L_IDX, eye=L_EYE, ear=L_EAR),
}


def ensure_model(kind: str) -> Path:
    path = Path(__file__).parent / "models" / f"pose_landmarker_{kind}.task"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print(f"Downloading pose model ({kind})...")
        urllib.request.urlretrieve(MODEL_URLS[kind], path)
    return path


# --------------------------------------------------------------------------- salute

def check_salute(landmarks, w, h):
    """Return (is_saluting, debug_info). Works with either hand."""
    pts = [(p.x * w, p.y * h, p.visibility) for p in landmarks]
    shoulder_w = math.dist(pts[L_SH][:2], pts[R_SH][:2])
    if shoulder_w < 0.05 * w:
        return False, {}

    best = {}
    for name, s in SIDES.items():
        sh, el, wr, idx, eye, ear = (pts[s[k]] for k in ("sh", "el", "wr", "idx", "eye", "ear"))
        if min(sh[2], el[2], wr[2], idx[2]) < 0.4:
            continue
        # The brow/temple point: between eye and ear, a little above.
        brow = ((eye[0] + ear[0]) / 2, (eye[1] + ear[1]) / 2 - 0.12 * shoulder_w)
        hand = ((idx[0] + wr[0] * 0.3) / 1.3, (idx[1] + wr[1] * 0.3) / 1.3)
        conds = {
            "hand@brow": math.dist(hand, brow) < 0.65 * shoulder_w,
            "hand_up": idx[1] < sh[1] - 0.35 * shoulder_w and wr[1] < sh[1],
            "elbow_up": el[1] < sh[1] + 0.35 * shoulder_w,
            "elbow_out": abs(el[0] - sh[0]) > 0.15 * shoulder_w,
        }
        score = sum(conds.values())
        if score > best.get("score", -1):
            best = dict(side=name, score=score, conds=conds)
    return best.get("score") == 4, best


# --------------------------------------------------------------------------- teleport effect

def person_mask(frame, background, size):
    """Rough 'you' mask: wherever the live frame differs from the empty-room photo.
    Small and blurry on purpose - it only shapes the glow of the effect."""
    a = cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(background, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    # Webcams re-adjust brightness when you walk in/out; match the photo's brightness
    # to the live frame first. The median per-pixel ratio ignores you unless you
    # fill more than half the frame.
    b *= np.median(((a + 1) / (b + 1)).reshape(-1, 3), axis=0)
    diff = np.abs(a - b).max(axis=2)
    m = (diff > 35).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32), (9, 9), 0)


class Teleporter:
    PRESENT, ABSENT, SPAWNING, DESPAWNING = "present", "absent", "spawning", "despawning"

    def __init__(self, shape):
        self.h, self.w = shape[:2]
        self.sw, self.sh = self.w // 4, self.h // 4  # effect maths runs at quarter resolution
        self.state = self.PRESENT
        self.t0 = 0.0
        self.field = None

    @property
    def busy(self):
        return self.state in (self.SPAWNING, self.DESPAWNING)

    def toggle(self, now):
        if self.busy:
            return
        self.state = self.DESPAWNING if self.state == self.PRESENT else self.SPAWNING
        self.t0 = now
        # Fresh noise each time: blotchy dissolve that sweeps top -> bottom like a beam.
        noise = np.random.default_rng().random((self.sh // 6 + 2, self.sw // 6 + 2)).astype(np.float32)
        noise = cv2.resize(noise, (self.sw, self.sh), interpolation=cv2.INTER_CUBIC)
        noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
        ramp = np.linspace(0, 1, self.sh, dtype=np.float32)[:, None]
        self.field = 0.55 * noise + 0.45 * ramp

    def progress(self, now):
        """Visibility 0..1 of the person (eased)."""
        if self.state == self.PRESENT:
            return 1.0
        if self.state == self.ABSENT:
            return 0.0
        t = min((now - self.t0) / TRANSITION_SECONDS, 1.0)
        if t >= 1.0:
            self.state = self.PRESENT if self.state == self.SPAWNING else self.ABSENT
            return self.progress(now)
        e = t * t * (3 - 2 * t)
        return e if self.state == self.SPAWNING else 1 - e

    def render(self, frame, background, now):
        p = self.progress(now)
        if p >= 1.0:
            return frame        # visible: camera passes straight through
        if p <= 0.0:
            return background   # hidden: the empty-room photo, nothing live

        person = person_mask(frame, background, (self.sw, self.sh))
        soft = 0.08
        thresh = p * (1 + soft)
        vis = np.clip((thresh - self.field) / soft, 0, 1)
        edge = np.clip(1 - np.abs(self.field - (thresh - soft / 2)) / (soft * 1.5), 0, 1)
        glow = person * (edge * math.sin(math.pi * p) ** 0.5 * 1.6 + vis * (1 - p) * 0.35)
        # Show live where the dissolve has reached, the photo elsewhere.
        a = cv2.resize(vis, (self.w, self.h))
        glow = cv2.resize(glow, (self.w, self.h))
        out = cv2.blendLinear(frame, background, a, 1 - a)
        glow_img = cv2.merge([cv2.convertScaleAbs(glow, alpha=float(c)) for c in GLOW_BGR])
        return cv2.add(out, glow_img)


# --------------------------------------------------------------------------- preview UI

class Preview:
    WINDOW = f"{APP_NAME} (preview - not sent)"
    SCALE = 0.5  # half-size preview window keeps drawing cheap

    def __init__(self):
        cv2.namedWindow(self.WINDOW)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        self.button = (0, 0, 0, 0)
        self.clicked = False

    def _on_mouse(self, event, x, y, *_):
        x0, y0, x1, y1 = self.button
        if event == cv2.EVENT_LBUTTONUP and x0 <= x <= x1 and y0 <= y <= y1:
            self.clicked = True

    def show(self, out, mirror, lines, button_label):
        img = cv2.resize(out, None, fx=self.SCALE, fy=self.SCALE, interpolation=cv2.INTER_AREA)
        if mirror:
            img = cv2.flip(img, 1)
        for i, (s, color) in enumerate(lines):
            y = 22 + i * 20
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if button_label:
            (tw, th), _ = cv2.getTextSize(button_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            x0, y1 = 10, img.shape[0] - 10
            x1, y0 = x0 + tw + 20, y1 - th - 16
            self.button = (x0, y0, x1, y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 140, 255), -1)
            cv2.putText(img, button_label, (x0 + 10, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self.WINDOW, img)
        key = cv2.waitKey(1) & 0xFF
        closed = cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1
        clicked, self.clicked = self.clicked, False
        return key, clicked, closed


# --------------------------------------------------------------------------- output

def open_virtual_camera(w, h, fps):
    """Send to whichever virtual camera driver is installed:
    OBS Virtual Camera (comes with OBS Studio), or a Unity Capture camera,
    preferably one installed under the name "Hide Me Cam"."""
    attempts = [("obs", None), ("unitycapture", APP_NAME), ("unitycapture", None)]
    errors = []
    for backend, device in attempts:
        try:
            return pyvirtualcam.Camera(w, h, fps, fmt=pyvirtualcam.PixelFormat.BGR,
                                       backend=backend, device=device)
        except Exception as e:
            errors.append(f"  {backend}{f' ({device})' if device else ''}: {e}")
    raise SystemExit("No virtual camera driver found. Install one of:\n"
                     "  - OBS Studio (https://obsproject.com), keep its own 'Start Virtual Camera' off\n"
                     "  - Unity Capture (https://github.com/schellingb/UnityCapture), named \"Hide Me Cam\"\n"
                     "Details:\n" + "\n".join(errors))


# --------------------------------------------------------------------------- main loop

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="real webcam index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--model", choices=MODEL_URLS, default="lite",
                    help="pose model: 'lite' (default, light on CPU) or 'full' (more accurate)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from camera {args.camera}. Try --camera 1 (and close apps using it).")
    if frame.mean() < 1:
        print("Warning: camera frame is black - you may have picked the OBS Virtual Camera itself as input.")
    h, w = frame.shape[:2]
    detect_size = (DETECT_WIDTH, round(DETECT_WIDTH * h / w))

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model(args.model))),
        running_mode=vision.RunningMode.VIDEO,
    ))

    tele = Teleporter(frame.shape)
    preview = Preview()
    background = None      # captured fresh every run - an old photo would no longer match the room
    capture_at = None
    hold_start, last_trigger, armed = None, 0.0, True
    saluting, info = False, {}
    debug, mirror = False, True
    fps, last_t, n = 0.0, time.perf_counter(), 0
    t_start = last_t

    with open_virtual_camera(w, h, args.fps) as vcam:
        print(f"{APP_NAME}: streaming {w}x{h} to '{vcam.device}'. Pick it as the camera in your apps.")
        print("Click 'Capture background' in the preview window, then step out of frame.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames.")
                break
            now = time.perf_counter()
            n += 1

            if n % DETECT_EVERY == 0:
                small = cv2.cvtColor(cv2.resize(frame, detect_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=small), int((now - t_start) * 1000))
                saluting, info = check_salute(result.pose_landmarks[0], w, h) if result.pose_landmarks else (False, {})

            # Salute -> toggle (hold briefly, then re-arm after lowering the hand).
            # Disabled until there's a background photo to show.
            if saluting and armed and background is not None:
                hold_start = hold_start or now
                if now - hold_start >= HOLD_SECONDS:
                    tele.toggle(now)
                    armed, last_trigger, hold_start = False, now, None
            elif not saluting:
                hold_start = None
                if not armed and now - last_trigger > COOLDOWN_SECONDS:
                    armed = True

            if capture_at is not None and now >= capture_at:
                background, capture_at = frame.copy(), None
                print("Background captured.")

            out = frame if background is None else tele.render(frame, background, now)
            vcam.send(out)

            dt, last_t = now - last_t, now
            fps = fps * 0.9 + (1 / dt if dt > 0 else 0) * 0.1

            if capture_at is not None:
                status = (f"Step out of frame! Capturing in {capture_at - now:.1f}s", (0, 200, 255))
            elif background is None:
                status = ("Click 'Capture background', then step out of frame", (0, 200, 255))
            else:
                status = (f"{tele.state.upper()}  -  salute to toggle", (120, 255, 120) if tele.state == tele.PRESENT else (80, 200, 255))
            lines = [status, (f"{fps:4.1f} fps  ->  {vcam.device}", (220, 220, 220))]
            if debug and info:
                lines.append((f"{info['side']} hand  score {info['score']}/4", (255, 255, 255)))
                lines += [(f"  {k}: {'OK' if v else '--'}", (120, 255, 120) if v else (90, 90, 255))
                          for k, v in info["conds"].items()]
            label = None if capture_at is not None else ("Capture background" if background is None else "Retake background")

            key, clicked, closed = preview.show(out, mirror, lines, label)
            if closed or key in (ord("q"), 27):
                break
            if (clicked or key == ord("b")) and capture_at is None and not tele.busy:
                if tele.state == tele.ABSENT:
                    tele.state = tele.PRESENT  # retaking while hidden: show live again
                capture_at = now + CAPTURE_COUNTDOWN
            elif key == ord("s") and background is not None:
                tele.toggle(now)
            elif key == ord("d"):
                debug = not debug
            elif key == ord("m"):
                mirror = not mirror

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
