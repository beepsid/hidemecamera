# Hide Me Cam

A virtual webcam that hides you when you wiggle your fingers. Wiggle them and you dissolve out of the picture, leaving an empty room. Wiggle again and you beam back in.

It works in any app that lets you pick a camera: Discord, WhatsApp, Facebook/Messenger, Zoom, Google Meet, Teams, browsers, and streaming software.

```
your webcam ──► hide_me_cam.py (finger-wiggle detection + teleport effect) ──► virtual camera ──► any app
```

It's light: one small hand-tracking model, roughly 5-10% CPU and under 250 MB RAM on a laptop at 720p/30 fps.

## Requirements

- Windows 10 or 11 and a webcam
- [Python 3.13](https://www.python.org/downloads/) (the version it's tested with; `run.bat` uses it)
- A virtual camera driver. Pick **one**:
  - **OBS Studio** ([obsproject.com](https://obsproject.com/)). Installing it adds the "OBS Virtual Camera" device. OBS doesn't need to be open. Keep OBS's own "Start Virtual Camera" button off while Hide Me Cam runs, because only one program can feed that camera.
  - **No OBS? Unity Capture**, a small free driver. It shows up as a camera called **"Hide Me Cam"**:
    1. Download [UnityCapture](https://github.com/schellingb/UnityCapture/archive/refs/heads/master.zip) and unzip it.
    2. Copy [`driver/install_hide_me_cam_driver.bat`](driver/install_hide_me_cam_driver.bat) into its `Install` folder.
    3. Double-click it and allow the admin prompt.

Hide Me Cam uses OBS Virtual Camera if it's installed, and otherwise the Unity Capture "Hide Me Cam" camera.

## Install and run

```
git clone https://github.com/beepsid/hidemecamera.git
cd hidemecamera
run.bat
```

The first run sets up a virtual environment, installs the packages in `requirements.txt` and downloads the hand-tracking model (about 8 MB). After that, just double-click `run.bat`.

## How to use

1. Start Hide Me Cam. A preview window opens. Only you see its text and button; apps get a clean picture.
2. Click **Capture background**, then step out of the frame. After 3 seconds it takes a photo of the empty room. You do this **every time you start it**, so the photo always matches your room right now. Until then, the gesture is ignored.
3. In your app's camera settings, pick **OBS Virtual Camera** (or **Hide Me Cam** if you installed Unity Capture).
4. Hold one hand up, palm towards the camera, and do a **finger wave**: bend and straighten your fingers one after another, index → middle → ring → pinky (or pinky → index), like a wave rolling across your hand. Closing or opening a fist doesn't count. While hidden, the app shows the room photo. Wave again to come back.

If the lighting changes or you move the camera, click **Retake background**.

## Keys (preview window)

| Key | Action |
| --- | --- |
| B | Capture or retake the background (same as the button) |
| S | Toggle without the gesture |
| D | Debug overlay: shows how bent each finger is and how many wiggles it counted |
| M | Mirror the preview on or off |
| F | Full screen preview on or off (the window can also be resized or maximized) |
| Q / Esc | Quit |

## Options

```
--camera 0        webcam index; if the preview is black you've picked a virtual camera, try 1 or 2
--width 1280 --height 720 --fps 30
```

## Tuning

If the wiggle isn't detected (or triggers by accident), press **D**. It shows each finger's openness (about 1.8 straight, 0.7 fully bent), whether it's `bent` or has `dipped` (bent and straightened again), and the order the fingers dipped in. A toggle needs all four to dip in order (`I M R P` or `P R M I`), each a little after the last, within 1.5 seconds. A finger held bent for more than 0.8 s cancels it (that's a fist). Make sure the hand is clearly in view with fingers pointing up. If a finger never shows `dipped`, lower `AMP`. If the order comes out jumbled, wave a bit slower. To tune further, change `AMP`, `MAX_DIP`, `MIN_SPREAD` or `WINDOW` in `WiggleDetector`, or `COOLDOWN_SECONDS` / `TRANSITION_SECONDS` at the top of `hide_me_cam.py`.
