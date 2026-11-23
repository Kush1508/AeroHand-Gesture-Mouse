# dev/creator=tubakhxn
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

WRIST, THUMB_TIP = 0, 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20
FINGERS = {"index": (INDEX_MCP, INDEX_TIP), "middle": (MIDDLE_MCP, MIDDLE_TIP),
           "ring": (RING_MCP, RING_TIP), "pinky": (PINKY_MCP, PINKY_TIP)}


class Gesture(str, Enum):
    NONE = "NO HAND"
    POINT = "POINT"
    PINCH = "PINCH"
    TWO_FINGER = "TWO FINGERS"
    SCROLL = "V / SCROLL"
    OPEN_PALM = "OPEN PALM"
    FIST = "FIST"
    OTHER = "OTHER"


@dataclass
class Action:
    kind: str
    double: bool = False
    amount: int = 0


@dataclass
class FrameResult:
    gesture: Gesture = Gesture.NONE
    mode: str = "IDLE"
    actions: list = field(default_factory=list)
    cursor: tuple = None
    hand: bool = False


class GestureClassifier:
    def __init__(self, cfg, aspect=4 / 3):
        self.cfg, self.aspect = cfg, aspect
        self.reset()

    def reset(self):
        self._pinching = False
        self._together = False

    def features(self, lm):
        p = lm[:, :2].astype(np.float64) * np.array([self.aspect, 1.0])
        d = lambda a, b: float(np.linalg.norm(p[a] - p[b]))
        palm = max(d(WRIST, MIDDLE_MCP), 1e-6)
        ratios = {n: d(WRIST, tip) / max(d(WRIST, mcp), 1e-6) for n, (mcp, tip) in FINGERS.items()}
        return {
            "palm": palm,
            "ratios": ratios,
            "ext": {n: r > self.cfg.finger_ext_ratio for n, r in ratios.items()},
            "pinch": d(THUMB_TIP, INDEX_TIP) / palm,
            "spread": d(INDEX_TIP, MIDDLE_TIP) / palm,
        }

    def classify(self, lm):
        c, f = self.cfg, self.features(lm)
        ext = f["ext"]
        if self._pinching:
            self._pinching = f["pinch"] < c.pinch_off
        else:
            self._pinching = (f["pinch"] < c.pinch_on and
                              f["ratios"]["index"] > c.pinch_index_guard)
        if self._pinching:
            return Gesture.PINCH, f
        n_ext = sum(ext.values())
        if n_ext == 4:
            return Gesture.OPEN_PALM, f
        if n_ext == 0:
            return Gesture.FIST, f
        if ext["index"] and ext["middle"] and not ext["ring"] and not ext["pinky"]:
            if self._together:
                self._together = f["spread"] < c.two_apart_min
            else:
                self._together = f["spread"] < c.two_together_max
            return (Gesture.TWO_FINGER if self._together else Gesture.SCROLL), f
        if ext["index"] and not ext["middle"]:
            if (not ext["ring"] and not ext["pinky"]) or (f["ratios"]["index"] > max(f["ratios"]["middle"], f["ratios"]["ring"]) + 0.20):
                return Gesture.POINT, f
        return Gesture.OTHER, f


class GestureInterpreter:
    def __init__(self, cfg, aspect=4 / 3):
        self.cfg = cfg
        self.clf = GestureClassifier(cfg, aspect)
        self.reset()

    def reset(self):
        self.clf.reset()
        self.stable = Gesture.NONE
        self._cand, self._cand_n = Gesture.NONE, 0
        self.pinch_state = "IDLE"
        self.pinch_t = 0.0
        self.last_click_t = -1e9
        self.last_right_t = -1e9
        self.two_since, self.right_fired = None, False
        self.scroll_prev, self.scroll_acc = None, 0.0
        self.lost = 0

    def update(self, lm, t):
        c, res = self.cfg, FrameResult()
        if lm is None:
            self.lost += 1
            res.mode = "HAND LOST - HOLDING"
            if self.lost == c.lost_frames_release:
                res.actions.append(Action("RELEASE_ALL"))
                self.reset(); self.lost = c.lost_frames_release
            if self.lost >= c.lost_frames_release:
                res.mode = "TRACKING LOST - RELEASED"
            return res
        self.lost, res.hand = 0, True

        raw, feat = self.clf.classify(lm)
        prev = self.stable
        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = raw, 1
        if self._cand_n >= c.stable_frames:
            self.stable = raw
        g = self.stable
        res.gesture = g

        if prev == Gesture.PINCH and g != Gesture.PINCH:
            self._end_pinch(t, res)
        if prev == Gesture.TWO_FINGER and g != Gesture.TWO_FINGER:
            self.two_since, self.right_fired = None, False
        if prev == Gesture.SCROLL and g != Gesture.SCROLL:
            self.scroll_prev, self.scroll_acc = None, 0.0
        if g == Gesture.PINCH and prev != Gesture.PINCH:
            self.pinch_state, self.pinch_t = "PENDING", t

        if g == Gesture.POINT:
            res.cursor, res.mode = tuple(lm[c.cursor_landmark, :2]), "CURSOR MOVE"
        elif g == Gesture.PINCH:
            if self.pinch_state == "PENDING":
                res.mode = "PINCH (CLICK)"
                if t - self.pinch_t >= c.click_hold_s:
                    self.pinch_state = "DRAGGING"
                    res.actions.append(Action("MOUSE_DOWN"))
            if self.pinch_state == "DRAGGING":
                res.mode = "DRAG"
                res.cursor = tuple(lm[c.cursor_landmark, :2])
        elif g == Gesture.TWO_FINGER:
            res.mode = "RIGHT CLICK ARMED"
            if self.two_since is None:
                self.two_since = t
            if (not self.right_fired and t - self.two_since >= c.right_click_dwell_s
                    and t - self.last_right_t >= c.right_click_cooldown_s):
                self.right_fired, self.last_right_t = True, t
                res.actions.append(Action("RIGHT_CLICK"))
                res.mode = "RIGHT CLICK"
        elif g == Gesture.SCROLL:
            res.mode = "SCROLL"
            y = float((lm[INDEX_TIP, 1] + lm[MIDDLE_TIP, 1]) / 2)
            if self.scroll_prev is not None:
                dy = (self.scroll_prev - y) / feat["palm"]
                if abs(dy) > c.scroll_deadzone:
                    self.scroll_acc += (-dy if c.scroll_invert else dy) * c.scroll_notches_per_palm
                n = int(self.scroll_acc)
                if n:
                    self.scroll_acc -= n
                    res.actions.append(Action("SCROLL", amount=n))
            self.scroll_prev = y
        elif g == Gesture.OPEN_PALM:
            res.mode = "PAUSED (OPEN PALM)"
        elif g == Gesture.FIST:
            res.mode = "CURSOR STOPPED (FIST)"
        else:
            res.mode = "HOLD"
        return res

    def _end_pinch(self, t, res):
        if self.pinch_state == "PENDING":
            if t - self.last_click_t >= self.cfg.click_cooldown_s:
                dbl = (t - self.last_click_t) <= self.cfg.double_click_window_s
                res.actions.append(Action("LEFT_CLICK", double=dbl))
                self.last_click_t = t
        elif self.pinch_state == "DRAGGING":
            res.actions.append(Action("MOUSE_UP"))
        self.pinch_state = "IDLE"
