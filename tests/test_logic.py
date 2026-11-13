# dev/creator=tubakhxn
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from gesture_recognition import Gesture, GestureInterpreter
from mouse_controller import MouseController, OneEuro

ASPECT = 4 / 3


def hand(index="up", middle="curl", ring="curl", pinky="curl", pinch=False,
         spread=0.0, dx=0.0, dy=0.0):
    lm = np.zeros((21, 2), np.float32)
    lm[0] = (0.5, 0.8)
    mcp = {"index": (0.47, 0.66), "middle": (0.5, 0.65), "ring": (0.53, 0.66), "pinky": (0.56, 0.69)}
    ids = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
    state = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    xoff = {"index": -spread, "middle": spread, "ring": 0, "pinky": 0}
    for n, i in ids.items():
        x, y = mcp[n]
        if state[n] == "up":
            lm[i], lm[i + 1], lm[i + 2], lm[i + 3] = (x, y), (x, y - 0.05), (x, y - 0.085), (x, y - 0.115)
            lm[i + 3, 0] += xoff[n] * 0.6
            lm[i + 2, 0] += xoff[n] * 0.4
        else:
            lm[i], lm[i + 1], lm[i + 2], lm[i + 3] = (x, y), (x, y - 0.04), (x, y - 0.01), (x, y + 0.02)
    lm[1], lm[2], lm[3], lm[4] = (0.46, 0.77), (0.43, 0.72), (0.40, 0.68), (0.38, 0.64)
    if pinch:
        lm[6], lm[7], lm[8] = (0.467, 0.61), (0.462, 0.585), (0.46, 0.58)
        lm[4] = (0.465, 0.60)
    lm[:, 0] += dx
    lm[:, 1] += dy
    return np.hstack([lm, np.zeros((21, 1), np.float32)])


POINT = lambda **k: hand(index="up", **k)
PINCH = lambda **k: hand(index="up", pinch=True, **k)
FIST = lambda **k: hand(index="curl", **k)
PALM = lambda **k: hand("up", "up", "up", "up", spread=0.05, **k)
TWO_TOGETHER = lambda **k: hand("up", "up", spread=0.0, **k)
TWO_APART = lambda **k: hand("up", "up", spread=0.07, **k)

FPS = 30.0


class Sim:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.it = GestureInterpreter(self.cfg, ASPECT)
        self.t = 0.0
        self.actions, self.results = [], []

    def run(self, lm_fn, seconds):
        n = max(1, int(seconds * FPS))
        for i in range(n):
            lm = lm_fn(i) if callable(lm_fn) and not isinstance(lm_fn, np.ndarray) else lm_fn
            r = self.it.update(lm, self.t)
            self.results.append(r)
            self.actions += r.actions
            self.t += 1 / FPS
        return self.results[-1]

    def kinds(self):
        return [a.kind for a in self.actions]


passed = failed = 0


def check(name, cond, extra=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {extra}")


class FakeBackend:
    def __init__(self, w=1920, h=1080):
        self.w, self.h, self.log, self.pos, self.down_state = w, h, [], (0, 0), False

    def size(self): return (self.w, self.h)
    def move(self, x, y): self.pos = (x, y); self.log.append(("move", x, y))
    def down(self): self.down_state = True; self.log.append(("down",))
    def up(self): self.down_state = False; self.log.append(("up",))
    def click(self, b="left"): self.log.append(("click", b))
    def scroll(self, n): self.log.append(("scroll", n))


def kinds_of(backend, k): return [e for e in backend.log if e[0] == k]


def main():
    print("\n[1] Single-frame gesture classification")
    s = Sim()
    for name, fn, want in [("point", POINT, Gesture.POINT), ("pinch", PINCH, Gesture.PINCH),
                           ("fist", FIST, Gesture.FIST), ("open palm", PALM, Gesture.OPEN_PALM),
                           ("two fingers together", TWO_TOGETHER, Gesture.TWO_FINGER),
                           ("V apart", TWO_APART, Gesture.SCROLL)]:
        s = Sim()
        r = s.run(fn(), 0.3)
        check(f"{name} -> {want.value}", r.gesture == want, f"got {r.gesture}")

    print("\n[2] Left click / double click / drag")
    s = Sim(); s.run(POINT(), 0.3); s.run(PINCH(), 0.15); s.run(POINT(), 0.3)
    check("quick pinch = exactly one LEFT_CLICK", s.kinds() == ["LEFT_CLICK"], str(s.kinds()))
    check("cursor frozen while pinching", all(r.cursor is None for r in s.results[10:13]))

    s = Sim(); s.run(POINT(), 0.3)
    s.run(PINCH(), 0.12); s.run(POINT(), 0.12); s.run(PINCH(), 0.12); s.run(POINT(), 0.3)
    acts = [a for a in s.actions if a.kind == "LEFT_CLICK"]
    check("double pinch = 2 clicks, 2nd flagged double", len(acts) == 2 and acts[1].double and not acts[0].double,
          str([(a.kind, a.double) for a in s.actions]))

    s = Sim(); s.run(POINT(), 0.3); s.run(PINCH(), 0.8)
    check("hold pinch -> MOUSE_DOWN, no click yet", s.kinds() == ["MOUSE_DOWN"], str(s.kinds()))
    s.run(lambda i: PINCH(dx=0.002 * i), 1.0)
    check("cursor follows hand while dragging", any(r.cursor is not None for r in s.results[-20:]))
    s.run(POINT(), 0.3)
    check("release pinch -> MOUSE_UP (drop)", s.kinds() == ["MOUSE_DOWN", "MOUSE_UP"], str(s.kinds()))

    s = Sim(); s.run(POINT(), 0.3)
    s.run(lambda i: PINCH(dy=0.0005 * ((-1) ** i)), 0.15)
    s.run(POINT(), 0.3)
    check("tremor during pinch gives a single click (no repeats)", s.kinds().count("LEFT_CLICK") == 1, str(s.kinds()))

    s = Sim(); s.run(POINT(), 0.3)
    s.run(lambda i: PINCH() if i % 6 not in (2,) else POINT(), 1.0)
    check("single-frame flicker does not spam clicks", s.kinds().count("LEFT_CLICK") <= 1, str(s.kinds()))

    print("\n[3] Right click / scroll / palm / fist")
    s = Sim(); s.run(POINT(), 0.3); s.run(TWO_TOGETHER(), 3.0)
    check("two fingers held 3 s = exactly one RIGHT_CLICK", s.kinds() == ["RIGHT_CLICK"], str(s.kinds()))
    s = Sim(); s.run(POINT(), 0.3); s.run(TWO_TOGETHER(), 0.15)
    check("brief two-finger pass does not right-click", "RIGHT_CLICK" not in s.kinds())

    s = Sim(); s.run(TWO_APART(), 0.3)
    s.run(lambda i: TWO_APART(dy=-0.004 * i), 1.0)
    up = sum(a.amount for a in s.actions if a.kind == "SCROLL")
    s.run(lambda i: TWO_APART(dy=-0.12 + 0.004 * i), 1.0)
    down = sum(a.amount for a in s.actions if a.kind == "SCROLL") - up
    check("moving V upward scrolls up (+)", up > 0, f"up={up}")
    check("moving V downward scrolls down (-)", down < 0, f"down={down}")
    check("cursor does not move while scrolling", all(r.cursor is None for r in s.results if r.gesture == Gesture.SCROLL))

    s = Sim(); s.run(POINT(), 0.3); r = s.run(FIST(), 0.3)
    check("fist stops cursor", r.cursor is None and r.gesture == Gesture.FIST)
    r = s.run(PALM(), 0.3)
    check("open palm pauses (no cursor)", r.cursor is None and r.gesture == Gesture.OPEN_PALM)

    s = Sim(); s.run(POINT(), 0.3); s.run(PINCH(), 0.8); s.run(PALM(), 0.3)
    check("open palm during drag releases button", s.kinds() == ["MOUSE_DOWN", "MOUSE_UP"], str(s.kinds()))

    print("\n[4] Tracking loss while dragging (safety)")
    s = Sim(); s.run(POINT(), 0.3); s.run(PINCH(), 0.8)
    s.run(None, 0.2)
    check("hand lost mid-drag -> release emitted", "RELEASE_ALL" in s.kinds(), str(s.kinds()))
    be = FakeBackend(); mc = MouseController(s.cfg, be)
    s2 = Sim(); s2.run(POINT(), 0.3)
    for lm_fn, secs in [(PINCH(), 0.8), (None, 0.2)]:
        for _ in range(int(secs * FPS)):
            r = s2.it.update(lm_fn, s2.t); mc.apply(r, s2.t, True); s2.t += 1 / FPS
    check("backend button is UP after tracking loss", not be.down_state and not mc.left_down,
          str(be.log[-3:]))
    s = Sim(); s.run(POINT(), 0.3); s.run(None, 0.05)
    check("1-2 frame dropout does not release/click", "RELEASE_ALL" not in s.kinds())

    print("\n[5] Mouse controller against a fake OS backend")
    cfg = Config()
    for (w, h) in [(1366, 768), (1920, 1080), (2560, 1600), (3840, 2160)]:
        be = FakeBackend(w, h); mc = MouseController(cfg, be)
        x0, y0, x1, y1 = cfg.region
        res = type(Sim().it.update(POINT(), 0))()
        for t_, (nx, ny) in enumerate([(x0 - .1, y0 - .1), (x1 + .1, y1 + .1)]):
            res.cursor = (nx, ny)
            mc.fx.reset(); mc.fy.reset(); mc._last_move_t = None
            mc.apply(res, 10 + t_, True)
        ends = [e[1:] for e in kinds_of(be, "move")]
        check(f"{w}x{h}: corners reachable & clamped", ends == [(0, 0), (w - 1, h - 1)], str(ends))

    be = FakeBackend(); mc = MouseController(cfg, be)
    s = Sim(); s.run(POINT(), 0.2)
    for _ in range(5):
        r = s.it.update(PINCH(), s.t); mc.apply(r, s.t, False); s.t += 1 / FPS
    s.run(PINCH(), 0.8)
    r = type(s.results[0])(); r.cursor = (0.5, 0.4)
    mc.apply(r, 1.0, False)
    check("disabled (safe start): zero mouse events", not [e for e in be.log if e[0] != "up"], str(be.log))

    be = FakeBackend(); mc = MouseController(cfg, be)
    from gesture_recognition import Action, FrameResult
    mc.apply(FrameResult(actions=[Action("MOUSE_DOWN")]), 0, True)
    check("button down while enabled", be.down_state)
    mc.apply(FrameResult(), 0.1, False)
    check("disabling releases held button", not be.down_state)
    mc.apply(FrameResult(actions=[Action("MOUSE_DOWN")]), 0.2, True)
    mc.emergency_stop()
    check("emergency_stop releases held button", not be.down_state)
    n = len(be.log); mc.release_all(); mc.release_all()
    check("release_all idempotent (no spurious mouse-up)", len(be.log) == n)

    be = FakeBackend(); mc = MouseController(cfg, be)
    mc.apply(FrameResult(actions=[Action("SCROLL", amount=3), Action("RIGHT_CLICK")]), 0, True)
    check("scroll + right click reach backend", be.log == [("scroll", 3), ("click", "right")], str(be.log))

    print("\n[6] Smoothing")
    rng = np.random.default_rng(1)
    raw = 0.5 + rng.normal(0, 0.004, 300)
    f = OneEuro(cfg.smooth_min_cutoff, cfg.smooth_beta)
    sm = np.array([f(x, i / FPS) for i, x in enumerate(raw)])
    check("jitter std reduced by >2x when hand is still", sm[60:].std() < raw[60:].std() / 2,
          f"{raw[60:].std():.5f} -> {sm[60:].std():.5f}")
    f = OneEuro(cfg.smooth_min_cutoff, cfg.smooth_beta)
    ramp = np.linspace(0.2, 0.8, 30)
    out = [f(x, i / FPS) for i, x in enumerate(ramp)]
    check("fast sweep lag stays small (<0.05 of screen)", abs(out[-1] - ramp[-1]) < 0.05, f"lag={ramp[-1]-out[-1]:.3f}")

    print(f"\n{passed} passed, {failed} failed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
