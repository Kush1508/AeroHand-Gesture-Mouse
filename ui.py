# dev/creator=tubakhxn
import cv2
import numpy as np

BG = (22, 22, 20)
CYAN = (230, 220, 0)
GREEN = (110, 255, 120)
AMBER = (0, 180, 255)
RED = (60, 60, 240)
GREY = (150, 150, 145)
WHITE = (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
               (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
               (18, 19), (19, 20), (0, 17)]
BUTTON = (10, 0, 160, 34)


def draw_hand(img, lm):
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in lm]
    for a, b in CONNECTIONS:
        cv2.line(img, pts[a], pts[b], CYAN, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, p, 3, GREEN if i not in (4, 8) else WHITE, -1, cv2.LINE_AA)
    for i in (4, 8):
        cv2.circle(img, pts[i], 10, GREEN, 2, cv2.LINE_AA)
    cv2.line(img, pts[4], pts[8], GREEN, 1, cv2.LINE_AA)


def draw_region(img, region):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = region
    cv2.rectangle(img, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)), (90, 90, 80), 1)


def _panel(img, x, y, w, h):
    sub = img[y:y + h, x:x + w]
    img[y:y + h, x:x + w] = (sub * 0.25 + 0.75 * np.array(BG)).astype(sub.dtype)
    cv2.rectangle(img, (x, y), (x + w, y + h), CYAN, 1)


def draw_status(img, rows):
    x, y, w = 10, 10, 270
    h = 14 + 22 * len(rows)
    _panel(img, x, y, w, h)
    for i, (label, value, col) in enumerate(rows):
        yy = y + 24 + 22 * i
        cv2.putText(img, label, (x + 10, yy), FONT, 0.42, GREY, 1, cv2.LINE_AA)
        cv2.putText(img, str(value), (x + 130, yy), FONT, 0.46, col, 1, cv2.LINE_AA)


def button_rect(img):
    h, w = img.shape[:2]
    return (10, h - 44, 170, 34)


def draw_button(img, enabled, started):
    x, y, w, h = button_rect(img)
    label = "PAUSE" if enabled else ("RESUME" if started else "ENABLE")
    col = AMBER if enabled else GREEN
    cv2.rectangle(img, (x, y), (x + w, y + h), BG, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
    cv2.putText(img, f"{label}  [G / F8]", (x + 12, y + 23), FONT, 0.55, col, 1, cv2.LINE_AA)


def hit_button(img, mx, my):
    x, y, w, h = button_rect(img)
    return x <= mx <= x + w and y <= my <= y + h


def banner(img, text, col):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    x, y = (w - tw) // 2, h - 20
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 10), BG, -1)
    cv2.putText(img, text, (x, y), FONT, 0.6, col, 2, cv2.LINE_AA)


def footer(img):
    h, w = img.shape[:2]
    cv2.putText(img, "ESC emergency stop | G/F8 on-off | C calibrate | +/- sensitivity | Q quit",
                (190, h - 18), FONT, 0.38, GREY, 1, cv2.LINE_AA)
