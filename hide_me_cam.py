"""
Hide Me Cam - a virtual webcam that hides you when you wiggle your fingers.

Reads your real webcam, watches for a finger wiggle (hand up, fingers bending
down and up quickly, like a "bye-bye" wave), and streams the result to a virtual
camera, so any app (Discord, WhatsApp, Zoom, Meet, Teams, browsers, streaming
software) can use it.

  1. Click "Capture background" and step out of frame (3 s countdown).
  2. Wiggle your fingers while visible -> you dissolve out (the empty-room photo is shown)
     Wiggle your fingers while hidden  -> you beam back in

Preview window keys:
  B  capture background     S  toggle without the gesture
  D  debug overlay          M  mirror preview          F  full screen
  Q / Esc  quit
"""

import argparse
import math
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from mediapipe.tasks.python import BaseOptions, vision

APP_NAME = "Hide Me Cam"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

COOLDOWN_SECONDS = 1.5     # minimum gap between triggers
TRANSITION_SECONDS = 1.2   # length of the spawn / despawn effect
CAPTURE_COUNTDOWN = 3.0    # time to step out of frame after clicking "Capture background"
DETECT_WIDTH = 640         # hand tracking runs on a smaller copy of the frame
GLOW_BGR = (255, 220, 80)  # cyan teleport glow

# Hand landmark indices: wrist, then knuckle (MCP) and tip of index, middle, ring, pinky
WRIST = 0
KNUCKLES = (5, 9, 13, 17)
TIPS = (8, 12, 16, 20)
FINGER_NAMES = ("index", "middle", "ring", "pinky")


def ensure_model() -> Path:
    path = Path(__file__).parent / "models" / "hand_landmarker.task"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print("Downloading hand tracking model...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


# --------------------------------------------------------------------------- finger wiggle

def finger_openness(result, aspect):
    """How straight each finger looks (tip-to-wrist distance / palm size), or None if
    there's no raised hand. ~2.0 = straight, ~1.0 = fully bent.

    Measured flat, as seen on screen: with the palm towards the camera, fingers fold
    towards it and look much shorter - a far clearer signal than the model's depth
    estimate. Dividing by palm size cancels out hand distance and tilt."""
    if not result.hand_landmarks:
        return None
    img = result.hand_landmarks[0]
    if img[KNUCKLES[1]].y > img[WRIST].y:
        return None  # fingers pointing down: hand resting, not raised
    p = [np.array((q.x * aspect, q.y)) for q in img]
    palm = np.linalg.norm(p[KNUCKLES[1]] - p[WRIST]) + 1e-6
    return [float(np.linalg.norm(p[t] - p[WRIST]) / palm) for t in TIPS]


class WiggleDetector:
    """Detects a finger wave: each finger dips (bends, then straightens again) one
    after another - index, middle, ring, pinky, or the reverse. Closing or opening
    a fist moves all fingers at the same moment, so it doesn't count."""

    AMP = 0.18         # how far a finger must bend (and come back) to count as a dip
    MAX_DIP = 0.8      # a finger bent longer than this is a held fist, not a dip
    MIN_GAP = 0.025    # each next finger must dip at least this much later (~1 frame)...
    MIN_SPREAD = 0.12  # ...and first-to-last must take at least this long (a fist is quicker)
    WINDOW = 1.5       # all four dips must happen within this time
    REF_SECONDS = 0.6  # a finger's "straight" level = its highest openness over this time
    ORDERS = ([0, 1, 2, 3], [3, 2, 1, 0])

    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = deque()     # (time, openness) for the last REF_SECONDS
        self.down = [None] * 4  # per finger while bent: [time it bent, lowest value]
        self.dips = [None] * 4  # per finger: time of its last completed dip

    def order(self):
        """Fingers in the order they last dipped, e.g. 'I M R P' (for the debug view)."""
        done = sorted((t, f) for f, t in enumerate(self.dips) if t is not None)
        return " ".join("IMRP"[f] for _, f in done)

    def update(self, now, openness):
        if openness is None:
            self.reset()
            return False
        self.hist.append((now, openness))
        while now - self.hist[0][0] > self.REF_SECONDS:
            self.hist.popleft()

        for f, v in enumerate(openness):
            if self.dips[f] is not None and now - self.dips[f] > self.WINDOW:
                self.dips[f] = None
            if self.down[f] is None:
                straight = max(o[f] for _, o in self.hist)
                if v < straight - self.AMP:
                    self.down[f] = [now, straight]            # finger started bending
            elif now - self.down[f][0] > self.MAX_DIP:
                self.down[f] = self.dips[f] = None            # held bent: a fist, not a wave
            elif v > self.down[f][1] - self.AMP / 2:
                # Back near straight: one dip, timed from when it started bending.
                self.dips[f], self.down[f] = self.down[f][0], None

        if None in self.dips:
            return False
        order = sorted(range(4), key=lambda f: self.dips[f])
        times = [self.dips[f] for f in order]
        return (order in self.ORDERS
                and all(b - a >= self.MIN_GAP for a, b in zip(times, times[1:]))
                and times[-1] - times[0] >= self.MIN_SPREAD)


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
    """Resizable preview window (F = full screen). Text is drawn at the camera's full
    resolution and scales with the window, so it stays readable at any size."""

    WINDOW = f"{APP_NAME} (preview - not sent)"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, w, h):
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.WINDOW, w, h)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        self.size = w / 1280  # text size relative to a 720p frame
        self.button = (0, 0, 0, 0)
        self.clicked = False
        self.fullscreen = False

    def _on_mouse(self, event, x, y, *_):
        x0, y0, x1, y1 = self.button
        if event == cv2.EVENT_LBUTTONUP and x0 <= x <= x1 and y0 <= y <= y1:
            self.clicked = True

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)

    def show(self, out, mirror, lines, button_label):
        img = cv2.flip(out, 1) if mirror else out.copy()
        s = self.size
        scale, thick, line_h, pad = 0.9 * s, max(1, round(2 * s)), round(40 * s), round(14 * s)
        if lines:
            # Dark backing panel behind the text so it's readable on any background.
            width = max(cv2.getTextSize(t, self.FONT, scale, thick)[0][0] for t, _ in lines)
            x1, y1 = min(img.shape[1], width + 3 * pad), len(lines) * line_h + pad
            img[:y1, :x1] = img[:y1, :x1] // 3
            for i, (text, color) in enumerate(lines):
                cv2.putText(img, text, (pad, (i + 1) * line_h - round(8 * s)), self.FONT, scale, color, thick, cv2.LINE_AA)
        if button_label:
            (tw, th), _ = cv2.getTextSize(button_label, self.FONT, scale, thick)
            x0, y1 = pad, img.shape[0] - pad
            x1, y0 = x0 + tw + 2 * pad, y1 - th - 2 * pad
            self.button = (x0, y0, x1, y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (60, 140, 255), -1)
            cv2.putText(img, button_label, (x0 + pad, y1 - pad), self.FONT, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
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
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read from camera {args.camera}. Try --camera 1 (and close apps using it).")
    if frame.mean() < 1:
        print("Warning: camera frame is black - you may have picked a virtual camera as input.")
    h, w = frame.shape[:2]
    detect_size = (DETECT_WIDTH, round(DETECT_WIDTH * h / w))

    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    ))

    tele = Teleporter(frame.shape)
    wiggle = WiggleDetector()
    preview = Preview(w, h)
    background = None      # captured fresh every run - an old photo would no longer match the room
    capture_at = None
    last_trigger = 0.0
    debug, mirror = False, True
    fps, last_t = 0.0, time.perf_counter()
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

            # Hand tracking every frame: a finger wiggle is fast (3-5 bends a second).
            small = cv2.cvtColor(cv2.resize(frame, detect_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=small), int((now - t_start) * 1000))
            openness = finger_openness(result, w / h)

            # Wiggle -> toggle. Disabled until there's a background photo to show.
            wiggled = wiggle.update(now, openness)
            if tele.busy or now - last_trigger < COOLDOWN_SECONDS:
                wiggle.reset()  # forget the rest of the gesture so it can't toggle straight back
            elif wiggled and background is not None:
                tele.toggle(now)
                last_trigger = now
                wiggle.reset()

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
                status = (f"{tele.state.upper()}  -  wiggle your fingers to toggle",
                          (120, 255, 120) if tele.state == tele.PRESENT else (80, 200, 255))
            lines = [status, (f"{fps:4.1f} fps  ->  {vcam.device}", (220, 220, 220))]
            if debug:
                if openness is None:
                    lines.append(("no raised hand", (90, 90, 255)))
                else:
                    for n, o, down, dip in zip(FINGER_NAMES, openness, wiggle.down, wiggle.dips):
                        state = "bent" if down else ("dipped" if dip else "")
                        lines.append((f"  {n}: {o:.2f}  {state}", (120, 255, 120) if dip else (255, 255, 255)))
                    lines.append((f"  order: {wiggle.order() or '-'}   (need I M R P or P R M I)", (255, 255, 255)))
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
            elif key == ord("f"):
                preview.toggle_fullscreen()

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
