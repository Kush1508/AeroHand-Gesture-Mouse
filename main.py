# dev/creator=tubakhxn
import argparse
import sys
import threading
import time

import cv2
import numpy as np

import config as C
import ui
from gesture_recognition import Gesture, GestureInterpreter
from hand_tracker import HandTracker
from mouse_controller import MouseController


class GlobalKeys:
    def __init__(self, controller):
        self.emergency = threading.Event()
        self.toggle = threading.Event()
        self.ok = False
        try:
            from pynput import keyboard

            def on_press(key):
                if key == keyboard.Key.esc:
                    controller.emergency_stop()
                    self.emergency.set()
                elif key == keyboard.Key.f8:
                    self.toggle.set()

            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.daemon = True
            self.listener.start()
            self.ok = True
        except Exception as exc:
            print(f"[keys] global hotkeys unavailable ({exc}); using window keys only "
                  "(click the preview window for ESC / G to work).")


class Calibration:
    DURATION = 5.0

    def __init__(self):
        self.t0 = None
        self.pts = []

    @property
    def active(self):
        return self.t0 is not None

    def start(self, now):
        self.t0, self.pts = now, []

    def feed(self, p, now):
        if self.t0 is None:
            return None
        if p is not None:
            self.pts.append(p)
        if now - self.t0 >= self.DURATION:
            self.t0 = None
            if len(self.pts) > 20:
                a = np.array(self.pts)
                x0, y0 = a.min(0) - 0.02
                x1, y1 = a.max(0) + 0.02
                if x1 - x0 > 0.15 and y1 - y0 > 0.12:
                    return (float(max(x0, 0)), float(max(y0, 0)), float(min(x1, 1)), float(min(y1, 1)))
        return None

    def remaining(self, now):
        return max(0.0, self.DURATION - (now - self.t0))


def open_camera(cfg):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(cfg.camera_index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    return None


def main():
    ap = argparse.ArgumentParser(description="Hand gesture mouse controller")
    ap.add_argument("--camera", type=int, help="camera index (default from config, usually 0)")
    ap.add_argument("--start-enabled", action="store_true",
                    help="skip safe-start mode (NOT recommended)")
    args = ap.parse_args()

    cfg = C.Config.load()
    if args.camera is not None:
        cfg.camera_index = args.camera

    cap = open_camera(cfg)
    if cap is None:
        sys.exit(f"Could not open webcam index {cfg.camera_index}. Close other apps using the "
                 "camera, check Windows camera privacy settings, or try --camera 1.")
    tracker = HandTracker(cfg)
    controller = MouseController(cfg)
    interp = GestureInterpreter(cfg, aspect=cfg.cam_width / cfg.cam_height)
    keys = GlobalKeys(controller)
    calib = Calibration()

    state = {"enabled": args.start_enabled, "started": args.start_enabled, "click": False}
    cv2.namedWindow(C.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(C.WINDOW_NAME, on_mouse)

    def set_enabled(on):
        state["enabled"] = on
        state["started"] = state["started"] or on
        interp.reset()
        if not on:
            controller.release_all()

    fps, last, running = 0.0, time.monotonic(), True
    print("Ready. Control is DISABLED. Press G or F8 to enable, ESC for emergency stop.")
    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                print("Webcam read failed - stopping safely.")
                break
            if cfg.mirror:
                frame = cv2.flip(frame, 1)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now

            hand = tracker.process(frame)
            lm = hand.landmarks if hand else None
            res = interp.update(lm, now)

            if keys.emergency.is_set():
                set_enabled(False)
                print("EMERGENCY STOP (ESC) - mouse released, exiting.")
                break
            if keys.toggle.is_set():
                keys.toggle.clear()
                set_enabled(not state["enabled"])

            if calib.active:
                new = calib.feed(tuple(lm[cfg.cursor_landmark, :2]) if lm is not None else None, now)
                controller.apply(type(res)(), now, False)
                if new:
                    cfg.region = new
                    cfg.save()
            else:
                controller.apply(res, now, state["enabled"])

            ui.draw_region(frame, cfg.region)
            if lm is not None:
                ui.draw_hand(frame, lm)
            mode = res.mode if state["enabled"] else "DISABLED (SAFE)"
            if calib.active:
                mode = "CALIBRATING"
            on_col = ui.GREEN if state["enabled"] else ui.AMBER
            ui.draw_status(frame, [
                ("HAND DETECTED", "YES" if res.hand else "NO", ui.GREEN if res.hand else ui.RED),
                ("CURRENT GESTURE", res.gesture.value, ui.CYAN),
                ("MOUSE MODE", mode, on_col),
                ("LEFT CLICKS", controller.stats["left"], ui.WHITE),
                ("RIGHT CLICKS", controller.stats["right"], ui.WHITE),
                ("FPS", f"{fps:4.1f}", ui.WHITE),
                ("SENSITIVITY", f"{cfg.sensitivity:.2f}", ui.GREY),
            ])
            ui.draw_button(frame, state["enabled"], state["started"])
            ui.footer(frame)
            if calib.active:
                ui.banner(frame, f"CALIBRATION: sweep fingertip over comfortable range  {calib.remaining(now):.1f}s", ui.CYAN)
            elif not state["enabled"]:
                ui.banner(frame, "CONTROL OFF - press G / F8 or click ENABLE", ui.AMBER)
            if state["click"]:
                if ui.hit_button(frame, *state["click"]):
                    set_enabled(not state["enabled"])
                state["click"] = False

            cv2.imshow(C.WINDOW_NAME, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                controller.emergency_stop(); set_enabled(False); break
            elif k in (ord("g"), ord("G"), ord(" ")):
                set_enabled(not state["enabled"])
            elif k in (ord("q"), ord("Q")):
                break
            elif k in (ord("+"), ord("=")):
                cfg.sensitivity = min(cfg.sensitivity + 0.1, 3.0); cfg.save()
            elif k in (ord("-"), ord("_")):
                cfg.sensitivity = max(cfg.sensitivity - 0.1, 0.3); cfg.save()
            elif k in (ord("c"), ord("C")):
                calib.start(now)
            if cv2.getWindowProperty(C.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        controller.release_all()
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
