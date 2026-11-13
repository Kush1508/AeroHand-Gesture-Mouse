# Gesture Mouse Controller

## dev/creator=tubakhxn

Control your real laptop mouse with your hand, using only the webcam. Python + OpenCV +
MediaPipe Hand Landmarker + PyAutoGUI + NumPy. No web dashboard, no fake in-app cursor:
it moves the actual operating-system cursor.

## Features
- 21-landmark hand tracking (MediaPipe Hand Landmarker, Tasks API), left or right hand
- Move, left click, double click, click-and-drag, right click, vertical scroll
- One-Euro cursor smoothing (steady when still, low lag when moving), pixel dead-zone
- Works at any screen resolution (maps to the real screen size, DPI-aware on Windows)
- Dark HUD preview: landmarks, highlighted thumb/index, gesture, mouse mode, hand status, FPS, click counters
- Safety: starts **disabled**, ESC emergency stop (global), F8/G toggle, on-screen pause/resume button,
  buttons auto-released when the hand is lost / paused / disabled / app exits
- Calibration of the active hand area + live sensitivity adjustment, saved to `settings.json`
- Model auto-download on first run, with a bundled offline fallback model

## Gesture -> action
| Hand gesture | Mouse action |
|---|---|
| Index finger up (others curled), move it | Move cursor |
| Pinch thumb + index (quick) | Left click |
| Two quick pinches | Double click |
| Pinch and **hold** ~0.35 s, then move | Click-and-drag / drag-and-drop (release pinch to drop) |
| Index + middle **together**, hold ~0.35 s | Right click (fires once per gesture) |
| Index + middle **apart (V)**, move hand up/down | Scroll vertically |
| Open palm (all 4 fingers out) | Pause: cursor frozen, buttons released |
| Fist | Stop cursor movement |
| Hand leaves the camera | Buttons released automatically |

The cursor follows your index finger's middle joint (steadier than the tip while pinching) and freezes
during a quick pinch so clicks land where you aimed.

## Requirements
- Windows 10/11 (Linux/macOS may work but the Windows build is the target), a webcam
- Python **3.9 - 3.12** (64-bit) with the "tcl/tk" option ticked in the installer
- Internet once, to `pip install` and (optionally) download the model

## Install
```
cd gesture_mouse_controller
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
(or just double-click `run.bat`, which does this and launches the app)

## Run
```
python main.py            # options: --camera 1   --start-enabled
```
1. The preview window opens with control **OFF** (safe mode).
2. Raise your hand, check the skeleton tracks it, then press **G** or **F8** (or click ENABLE).
3. Press **C** once to calibrate: sweep your fingertip over the area you find comfortable for 5 s.

## Keyboard shortcuts
| Key | Action |
|---|---|
| **ESC** | Emergency stop: release buttons, disable control, exit (works system-wide) |
| **F8** (global) / **G** / Space (window) | Enable / disable gesture control |
| **C** | Calibrate hand area |
| **+ / -** | Sensitivity up / down |
| **Q** | Quit |

## Tuning (config.py or settings.json)
Everything lives in `config.py` (`Config`). Edit the defaults, or put overrides in `settings.json`.
Key knobs: `pinch_on` / `pinch_off`, `click_hold_s`, `double_click_window_s`, `click_cooldown_s`,
`right_click_dwell_s`, `right_click_cooldown_s`, `stable_frames`, `smooth_min_cutoff`, `smooth_beta`,
`deadzone_px`, `scroll_notches_per_palm`, `scroll_invert`, `preferred_hand`, `camera_index`.
Distances are relative to your palm size, so they work at any distance from the camera.

## Tests
`python tests/test_logic.py` runs the hardware-free logic tests (synthetic hand landmarks, fake mouse backend).

## Troubleshooting
- **Webcam won't open**: close Zoom/Teams/etc., check Settings > Privacy > Camera, try `--camera 1`.
- **ESC does nothing / F8 does nothing**: `pynput` missing or blocked by security software; the keys then only
  work while the preview window is focused. Moving the real mouse to the preview window is always possible.
- **Cursor jitters**: lower `smooth_min_cutoff` (e.g. 0.6), raise `deadzone_px`, improve front lighting.
- **Cursor lags**: raise `smooth_beta`.
- **Hard to reach screen edges / too twitchy**: press **C** to recalibrate, or use +/-.
- **Accidental clicks**: lower `pinch_on`, raise `stable_frames` to 3, raise `click_cooldown_s`.
- **Double click not registered by the OS**: speed up *Settings > Bluetooth & devices > Mouse > Additional mouse settings > double-click speed*, or increase `double_click_window_s` modestly.
- **Right click triggers while starting to scroll**: raise `right_click_dwell_s`.
- **Cursor can't reach edges on a scaled display**: the app sets DPI awareness automatically; run it from a normal Python install, not a sandbox.
- **`ImportError` for mediapipe**: use Python 3.9-3.12 64-bit.
- **Model download blocked**: the bundled `models/hand_landmarker_bundled.task` is used automatically, or put the official
  `hand_landmarker.task` into `models/`.
- On Linux, `pip install` may also need `python3-tk` (PyAutoGUI imports it).
