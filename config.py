# dev/creator=tubakhxn
import json
import os
from dataclasses import dataclass, asdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
BUNDLED_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker_bundled.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

WINDOW_NAME = "Gesture Mouse Controller  |  Dev/Creator = tubakhxn"


@dataclass
class Config:
    camera_index: int = 0
    cam_width: int = 640
    cam_height: int = 480
    mirror: bool = True

    num_hands: int = 1
    preferred_hand: str = "Any"
    min_detection_conf: float = 0.6
    min_presence_conf: float = 0.5
    min_tracking_conf: float = 0.5

    cursor_landmark: int = 6
    sensitivity: float = 1.0
    region: tuple = (0.15, 0.12, 0.85, 0.72)
    smooth_min_cutoff: float = 1.0
    smooth_beta: float = 8.0
    deadzone_px: float = 2.0

    finger_ext_ratio: float = 1.45
    pinch_on: float = 0.30
    pinch_off: float = 0.45
    pinch_index_guard: float = 1.10
    two_together_max: float = 0.34
    two_apart_min: float = 0.46
    stable_frames: int = 2

    click_hold_s: float = 0.35
    click_cooldown_s: float = 0.12
    double_click_window_s: float = 0.45
    right_click_dwell_s: float = 0.35
    right_click_cooldown_s: float = 1.0
    lost_frames_release: int = 3
    scroll_notches_per_palm: float = 10.0
    scroll_deadzone: float = 0.003
    scroll_invert: bool = False

    def save(self):
        data = asdict(self)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls):
        cfg = cls()
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, tuple(v) if k == "region" else v)
            except Exception as exc:
                print(f"[config] ignoring unreadable settings.json: {exc}")
        return cfg
