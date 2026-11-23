# dev/creator=tubakhxn & web integration
import argparse
import io
import json
import logging
import os
import sys
import threading
import time
import webbrowser

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import config as C
import ui
from gesture_recognition import Gesture, GestureInterpreter
from hand_tracker import HandTracker
from main import Calibration, GlobalKeys, open_camera
from mouse_controller import MouseController

# Suppress flask werkzeug log spam for polling endpoints
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(__name__, template_folder="templates")

# Global Shared State
state_lock = threading.Lock()
app_state = {
    "enabled": True,
    "camera_ready": False,
    "hand_detected": False,
    "gesture": "READY",
    "mouse_mode": "ACTIVE",
    "fps": 0.0,
    "stats": {"left": 0, "right": 0, "double": 0, "scroll": 0},
    "calibrating": False,
    "calib_remaining": 0.0,
    "sensitivity": 1.0,
    "smooth_beta": 8.0,
    "camera_index": 0,
}

latest_frame_lock = threading.Lock()
latest_jpeg = None
running = True

cfg = C.Config.load()
controller = None
tracker = None
interp = None
calib = Calibration()
keys = None


def background_vision_loop():
    global latest_jpeg, running, app_state, cfg, controller, tracker, interp, keys, calib

    controller = MouseController(cfg)
    keys = GlobalKeys(controller)

    while running:
        cap = open_camera(cfg)
        if cap is None:
            with state_lock:
                app_state["camera_ready"] = False
            time.sleep(1.0)
            continue

        with state_lock:
            app_state["camera_ready"] = True

        tracker = HandTracker(cfg)
        interp = GestureInterpreter(cfg, aspect=cfg.cam_width / cfg.cam_height)

        fps, last = 0.0, time.monotonic()
        print("\n[Gesture Engine Connected]")
        print(f"Web Dashboard ready at: http://localhost:5000")
        print("Press F8 or Spacebar to toggle control system-wide, ESC to emergency stop.\n")

        while running:
            ok, frame = cap.read()
            if not ok:
                with state_lock:
                    app_state["camera_ready"] = False
                time.sleep(0.05)
                break

            if cfg.mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now

            # Global hotkey triggers
            if keys.emergency.is_set():
                keys.emergency.clear()
                with state_lock:
                    app_state["enabled"] = False
                controller.emergency_stop()
                print("[HOTKEY] Emergency stop triggered via ESC.")

            if keys.toggle.is_set():
                keys.toggle.clear()
                with state_lock:
                    app_state["enabled"] = not app_state["enabled"]
                if not app_state["enabled"]:
                    controller.release_all()
                print(f"[HOTKEY] Control toggled: {'ON' if app_state['enabled'] else 'OFF'}")

            # Process hand landmarks
            hand = tracker.process(frame)
            lm = hand.landmarks if hand else None
            res = interp.update(lm, now)

            with state_lock:
                is_enabled = app_state["enabled"]
                is_calib = calib.active

            if is_calib:
                new_reg = calib.feed(tuple(lm[cfg.cursor_landmark, :2]) if lm is not None else None, now)
                controller.apply(type(res)(), now, False)
                if new_reg:
                    cfg.region = new_reg
                    cfg.save()
            else:
                controller.apply(res, now, is_enabled)

            # Update telemetry state
            with state_lock:
                app_state["camera_ready"] = True
                app_state["hand_detected"] = (lm is not None)
                app_state["gesture"] = res.gesture.value if lm is not None else "READY"
                app_state["mouse_mode"] = "CALIBRATING" if is_calib else (res.mode if is_enabled else "PAUSED")
                app_state["fps"] = float(fps)
                app_state["stats"] = dict(controller.stats)
                app_state["calibrating"] = is_calib
                app_state["calib_remaining"] = float(calib.remaining(now)) if is_calib else 0.0
                app_state["sensitivity"] = float(cfg.sensitivity)
                app_state["smooth_beta"] = float(cfg.smooth_beta)
                app_state["camera_index"] = int(cfg.camera_index)

            # Draw sleek overlays for preview stream
            ui.draw_region(frame, cfg.region)
            if lm is not None:
                ui.draw_hand(frame, lm)

            if is_calib:
                ui.banner(frame, f"CALIBRATING ({calib.remaining(now):.1f}s)", ui.CYAN)
            elif not is_enabled:
                ui.banner(frame, "STANDBY - Press F8 or Click to Enable", ui.AMBER)

            # Compress to JPEG for stream
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with latest_frame_lock:
                latest_jpeg = jpeg.tobytes()

            # Small sleep to cap CPU utilization ~ 60 FPS
            time.sleep(0.005)

        tracker.close()
        cap.release()

    if controller:
        controller.release_all()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with state_lock:
        return jsonify(app_state)


@app.route("/api/toggle", methods=["POST"])
def api_toggle():
    global app_state, controller
    with state_lock:
        app_state["enabled"] = not app_state["enabled"]
        is_on = app_state["enabled"]
    if controller and not is_on:
        controller.release_all()
    return jsonify({"enabled": is_on})


@app.route("/api/set_enabled", methods=["POST"])
def api_set_enabled():
    global app_state, controller
    data = request.get_json() or {}
    new_val = bool(data.get("enabled", False))
    with state_lock:
        app_state["enabled"] = new_val
    if controller and not new_val:
        controller.release_all()
    return jsonify({"enabled": new_val})


@app.route("/api/calibrate", methods=["POST"])
def api_calibrate():
    global calib
    calib.start(time.monotonic())
    return jsonify({"status": "calibrating"})


@app.route("/api/settings", methods=["POST"])
def api_settings():
    global cfg, controller
    data = request.get_json() or {}
    if "sensitivity" in data:
        cfg.sensitivity = max(0.2, min(float(data["sensitivity"]), 3.5))
    if "smooth_beta" in data:
        cfg.smooth_beta = max(1.0, min(float(data["smooth_beta"]), 30.0))
        if controller:
            controller.fx.beta = cfg.smooth_beta
            controller.fy.beta = cfg.smooth_beta
    cfg.save()
    return jsonify({"status": "ok", "sensitivity": cfg.sensitivity, "smooth_beta": cfg.smooth_beta})


def generate_frames():
    global latest_jpeg, running
    while running:
        frame_bytes = None
        with latest_frame_lock:
            frame_bytes = latest_jpeg
        if frame_bytes is None:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def main():
    ap = argparse.ArgumentParser(description="Hand Mouse Minimalist Web Engine")
    ap.add_argument("--port", type=int, default=5000, help="Web server port (default 5000)")
    ap.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = ap.parse_args()

    # Start vision thread
    t = threading.Thread(target=background_vision_loop, daemon=True)
    t.start()

    url = f"http://localhost:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"Starting server at {url}...")
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
