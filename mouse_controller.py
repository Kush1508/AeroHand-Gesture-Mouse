# dev/creator=tubakhxn
import atexit
import math
import sys
import threading

if sys.platform == "win32":
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

SCROLL_UNIT = 120 if sys.platform == "win32" else 1


class PyAutoBackend:
    def __init__(self):
        import pyautogui
        pyautogui.PAUSE = 0
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.FAILSAFE = False
        self.pg = pyautogui

    def size(self):
        return tuple(self.pg.size())

    def position(self):
        return tuple(self.pg.position())

    def move(self, x, y):
        self.pg.moveTo(x, y, _pause=False)

    def down(self):
        self.pg.mouseDown(button="left", _pause=False)

    def up(self):
        self.pg.mouseUp(button="left", _pause=False)

    def click(self, button="left"):
        self.pg.click(button=button, _pause=False)

    def scroll(self, notches):
        self.pg.scroll(int(notches) * SCROLL_UNIT, _pause=False)


class OneEuro:
    def __init__(self, min_cutoff, beta, dcutoff=1.0):
        self.min_cutoff, self.beta, self.dcutoff = min_cutoff, beta, dcutoff
        self.reset()

    def reset(self):
        self.t = self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t is None:
            self.t, self.x, self.dx = t, x, 0.0
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


class MouseController:
    def __init__(self, cfg, backend=None):
        self.cfg = cfg
        self.backend = backend or PyAutoBackend()
        self.screen_w, self.screen_h = self.backend.size()
        self.fx = OneEuro(cfg.smooth_min_cutoff, cfg.smooth_beta)
        self.fy = OneEuro(cfg.smooth_min_cutoff, cfg.smooth_beta)
        self.left_down = False
        self.lock = threading.RLock()
        self._last_move_t = None
        self._last_xy = None
        self.stats = {"left": 0, "right": 0, "double": 0, "scroll": 0}
        atexit.register(self.release_all)

    def map_to_screen(self, nx, ny):
        x0, y0, x1, y1 = self.cfg.region
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw = (x1 - x0) / 2 / max(self.cfg.sensitivity, 0.05)
        hh = (y1 - y0) / 2 / max(self.cfg.sensitivity, 0.05)
        u = min(max((nx - cx) / (2 * hw) + 0.5, 0.0), 1.0)
        v = min(max((ny - cy) / (2 * hh) + 0.5, 0.0), 1.0)
        return u, v

    def _to_px(self, u, v):
        return (min(max(round(u * (self.screen_w - 1)), 0), self.screen_w - 1),
                min(max(round(v * (self.screen_h - 1)), 0), self.screen_h - 1))

    def release_all(self):
        with self.lock:
            if self.left_down:
                try:
                    self.backend.up()
                finally:
                    self.left_down = False

    def emergency_stop(self):
        self.release_all()

    def apply(self, result, t, enabled):
        with self.lock:
            if not enabled:
                self.release_all()
                self.fx.reset(); self.fy.reset(); self._last_move_t = None
                return
            for a in result.actions:
                self._do(a)
            if result.cursor is not None:
                self._move(*result.cursor, t)

    def _move(self, nx, ny, t):
        u, v = self.map_to_screen(nx, ny)
        if self._last_move_t is not None and t - self._last_move_t > 0.25:
            self.fx.reset(); self.fy.reset()
        self._last_move_t = t
        sx, sy = self.fx(u, t), self.fy(v, t)
        px, py = self._to_px(sx, sy)
        if self._last_xy is None or math.hypot(px - self._last_xy[0], py - self._last_xy[1]) >= self.cfg.deadzone_px:
            self.backend.move(px, py)
            self._last_xy = (px, py)

    def _do(self, a):
        k = a.kind
        if k == "LEFT_CLICK":
            self.backend.click("left")
            self.stats["left"] += 1
            self.stats["double"] += int(a.double)
        elif k == "RIGHT_CLICK":
            self.backend.click("right")
            self.stats["right"] += 1
        elif k == "MOUSE_DOWN":
            if not self.left_down:
                self.backend.down()
                self.left_down = True
        elif k == "MOUSE_UP":
            self.release_all()
        elif k == "SCROLL":
            self.backend.scroll(a.amount)
            self.stats["scroll"] += abs(a.amount)
        elif k == "RELEASE_ALL":
            self.release_all()
