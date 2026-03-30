from __future__ import annotations


import cv2
import numpy as np


def _quick_mask(g_u8: np.ndarray) -> np.ndarray:
    # otsu trick and a little cleanup.
    _, mk1 = cv2.threshold(g_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mk2 = cv2.threshold(g_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # pick whichever has a more coin-ish fill ratio.
    f1 = float(np.mean(mk1 > 0))
    f2 = float(np.mean(mk2 > 0))
    if abs(f2 - 0.35) < abs(f1 - 0.35):
        mk = mk2
    else:
        mk = mk1
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
    return mk


def _rough_dedupe(circs: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for x, y, r in circs:
        dup = False
        for x2, y2, r2 in out:
            d = float(np.hypot(x - x2, y - y2))
            if d < max(8.0, 0.55 * max(r, r2)):
                dup = True
                break
        if not dup:
            out.append((x, y, r))
    return out


def do_coin_count(rgb_img_u8: np.ndarray) -> tuple[int, np.ndarray]:
    bgr = cv2.cvtColor(rgb_img_u8, cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (9, 9), 2.0)

    mk = _quick_mask(g)
    min_r = 9
    max_r = max(18, int(min(g.shape[:2]) * 0.19))
    min_d = 14
    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1.20,
        minDist=min_d,
        param1=130,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    raw: list[tuple[int, int, int]] = []
    if circles is not None:
        for x, y, r in circles[0]:
            x_i, y_i, r_i = int(round(x)), int(round(y)), int(round(r))
            y0, y1 = max(0, y_i - 2), min(mk.shape[0], y_i + 3)
            x0, x1 = max(0, x_i - 2), min(mk.shape[1], x_i + 3)
            local = mk[y0:y1, x0:x1]
            if local.size and float(np.mean(local > 0)) > 0.05:
                raw.append((x_i, y_i, r_i))

    kept = _rough_dedupe(raw)
    overlay = rgb_img_u8.copy()
    for x, y, r in kept:
        cv2.circle(overlay, (x, y), r, (72, 236, 142), 2)
        cv2.circle(overlay, (x, y), max(2, r // 12), (255, 90, 90), -1)
    return len(kept), overlay

