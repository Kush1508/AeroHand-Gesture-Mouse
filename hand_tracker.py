# dev/creator=tubakhxn
import os
import shutil
import time
import urllib.request
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

import config as C


@dataclass
class Hand:
    landmarks: np.ndarray
    label: str
    score: float


def _valid_task(path):
    try:
        return os.path.getsize(path) > 1_000_000 and zipfile.is_zipfile(path)
    except OSError:
        return False


def ensure_model():
    if _valid_task(C.MODEL_PATH):
        return C.MODEL_PATH
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    tmp = C.MODEL_PATH + ".part"
    try:
        print("[model] downloading hand_landmarker.task ...")
        with urllib.request.urlopen(C.MODEL_URL, timeout=20) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if not _valid_task(tmp):
            raise RuntimeError("downloaded file is not a valid model bundle")
        os.replace(tmp, C.MODEL_PATH)
        print("[model] saved to", C.MODEL_PATH)
        return C.MODEL_PATH
    except Exception as exc:
        print(f"[model] download failed ({exc}).")
        if os.path.exists(tmp):
            os.remove(tmp)
    if _valid_task(C.BUNDLED_MODEL_PATH):
        print("[model] using bundled offline model.")
        return C.BUNDLED_MODEL_PATH
    raise RuntimeError("No hand model available: connect to the internet once so it can "
                       "download, or place hand_landmarker.task in the models/ folder.")


class HandTracker:
    def __init__(self, cfg):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.cfg = cfg
        opts = vision.HandLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=ensure_model()),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=cfg.num_hands,
            min_hand_detection_confidence=cfg.min_detection_conf,
            min_hand_presence_confidence=cfg.min_presence_conf,
            min_tracking_confidence=cfg.min_tracking_conf)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ts = 0

    def process(self, frame_bgr):
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        res = self.landmarker.detect_for_video(img, ts)
        best = None
        for lms, handed in zip(res.hand_landmarks, res.handedness):
            label, score = handed[0].category_name, handed[0].score
            if self.cfg.preferred_hand != "Any" and label != self.cfg.preferred_hand:
                continue
            if best is None or score > best.score:
                arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
                best = Hand(arr, label, float(score))
        return best

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass
