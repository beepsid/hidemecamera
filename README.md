# Hide Me Cam

A virtual webcam that hides you with a salute. Salute and you dissolve out of the picture, leaving an empty room. Salute again and you beam back in.

It works in any app that lets you pick a camera: Discord, WhatsApp, Facebook/Messenger, Zoom, Google Meet, Teams, browsers, and streaming software.

```
your webcam ──► hide_me_cam.py (salute detection + teleport effect) ──► virtual camera ──► any app
```

It's light: about 8% CPU and 200 MB RAM on a laptop at 720p/30 fps, using a small pose model checked 15 times a second.

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

The first run sets up a virtual environment, installs the packages in `requirements.txt` and downloads the pose model (about 6 MB). After that, just double-click `run.bat`.

## How to use

1. Start Hide Me Cam. A preview window opens. Only you see its text and button; apps get a clean picture.
2. Click **Capture background**, then step out of the frame. After 3 seconds it takes a photo of the empty room. You do this **every time you start it**, so the photo always matches your room right now. Until then, salutes are ignored.
3. In your app's camera settings, pick **OBS Virtual Camera** (or **Hide Me Cam** if you installed Unity Capture).
4. Salute (hand flat at your brow, elbow out to the side) and hold it for about 0.3 seconds. While hidden, the app shows the room photo.

If the lighting changes or you move the camera, click **Retake background**.

## Keys (preview window)

| Key | Action |
| --- | --- |
| B | Capture or retake the background (same as the button) |
| S | Toggle without saluting |
| D | Debug overlay: shows which salute checks pass |
| M | Mirror the preview on or off |
| Q / Esc | Quit |

## Options

```
--camera 0        webcam index; if the preview is black you've picked a virtual camera, try 1 or 2
--model full      more accurate pose model (uses more CPU; default: lite)
--width 1280 --height 720 --fps 30
```

## Tuning

If salutes aren't detected (or trigger by accident), press **D** to see the four checks (`hand@brow`, `hand_up`, `elbow_up`, `elbow_out`). Then adjust the thresholds in `check_salute()`, or `HOLD_SECONDS` / `COOLDOWN_SECONDS` / `TRANSITION_SECONDS` at the top of `hide_me_cam.py`.
