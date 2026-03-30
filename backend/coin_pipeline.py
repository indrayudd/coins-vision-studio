from __future__ import annotations

import cv2
import numpy as np


def _deglare_quick(rgb_u8: np.ndarray, gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    spec = ((val > np.quantile(val, 0.94)) & (sat < np.quantile(sat, 0.42)) & (gray_u8 > np.quantile(gray_u8, 0.975))).astype(np.uint8) * 255
    spec = cv2.morphologyEx(spec, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    if float(np.mean(spec > 0)) < 0.001:
        return gray_u8, spec
    fixed = cv2.inpaint(gray_u8, spec, 5, cv2.INPAINT_TELEA)
    return cv2.addWeighted(gray_u8, 0.36, fixed, 0.64, 0.0), spec


def _line_killer(gray_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = cv2.Canny(gray_u8, 42, 128)
    lines = cv2.HoughLinesP(e, 1.0, np.pi / 180.0, threshold=40, minLineLength=60, maxLineGap=8)
    lm = np.zeros_like(gray_u8, dtype=np.uint8)
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = (int(v) for v in line)
            cv2.line(lm, (x1, y1), (x2, y2), 255, thickness=2)
    if np.any(lm):
        lm = cv2.morphologyEx(lm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        lm = cv2.dilate(lm, np.ones((3, 3), np.uint8), iterations=1)
    if float(np.mean(lm > 0)) < 0.002:
        return gray_u8, lm
    fixed = cv2.inpaint(gray_u8, lm, 4, cv2.INPAINT_TELEA)
    return cv2.addWeighted(gray_u8, 0.34, fixed, 0.66, 0.0), lm


def _hough_candidates(work_u8: np.ndarray, min_r: int, max_r: int, min_d: int) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    passes = [
        (1.20, min_d, 140, 56, min_r, max_r, 1.00),
        (1.25, max(12, int(min_d * 0.82)), 120, 47, max(7, int(min_r * 0.86)), max(min_r + 10, int(max_r * 1.28)), 0.85),
        (1.30, max(12, int(min_d * 0.74)), 150, 60, max(8, int(min_r * 0.90)), max(min_r + 10, int(max_r * 1.20)), 0.70),
    ]
    for dp, md, p1, p2, r0, r1, wt in passes:
        c = cv2.HoughCircles(work_u8, cv2.HOUGH_GRADIENT, dp=dp, minDist=md, param1=p1, param2=p2, minRadius=r0, maxRadius=r1)
        if c is None:
            continue
        for x, y, r in c[0]:
            out.append((float(x), float(y), float(r), float(wt)))
    return out


def _consensus(raw: list[tuple[float, float, float, float]]) -> list[tuple[int, int, int]]:
    clusters: list[dict[str, float]] = []
    for x, y, r, wt in sorted(raw, key=lambda t: t[3], reverse=True):
        hit = None
        for cl in clusters:
            d = float(np.hypot(x - cl["x"], y - cl["y"]))
            dr = abs(r - cl["r"])
            if d < max(8.0, 0.46 * max(r, cl["r"])) and dr < 0.48 * max(r, cl["r"]):
                hit = cl
                break
        if hit is None:
            clusters.append({"x": x, "y": y, "r": r, "w": wt, "v": 1.0})
            continue
        tw = hit["w"] + wt
        hit["x"] = (hit["x"] * hit["w"] + x * wt) / tw
        hit["y"] = (hit["y"] * hit["w"] + y * wt) / tw
        hit["r"] = (hit["r"] * hit["w"] + r * wt) / tw
        hit["w"] = tw
        hit["v"] += 1.0

    out: list[tuple[int, int, int]] = []
    for cl in sorted(clusters, key=lambda c: c["v"] + 0.2 * c["w"], reverse=True):
        if cl["v"] < 1.5:
            continue
        xi, yi, ri = int(round(cl["x"])), int(round(cl["y"])), int(round(cl["r"]))
        dup = False
        for x2, y2, r2 in out:
            d = float(np.hypot(xi - x2, yi - y2))
            if d < max(9.0, 0.55 * max(ri, r2)):
                dup = True
                break
        if not dup:
            out.append((xi, yi, ri))
    return out


def coin_count(rgb_img_u8: np.ndarray) -> tuple[int, np.ndarray]:
    bgr = cv2.cvtColor(rgb_img_u8, cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    deglared, _glare = _deglare_quick(rgb_img_u8, g)
    illum = cv2.normalize(cv2.divide(deglared, cv2.GaussianBlur(deglared, (0, 0), sigmaX=24.0, sigmaY=24.0) + 1, scale=186.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(illum)
    lowp = cv2.GaussianBlur(cv2.addWeighted(cv2.bilateralFilter(clahe, d=9, sigmaColor=48, sigmaSpace=48), 0.6, clahe, 0.4, 0), (9, 9), 2.2)
    lowp, _linemask = _line_killer(cv2.medianBlur(lowp, 7))

    min_dim = min(lowp.shape[:2])
    min_r = max(8, int(0.020 * min_dim))
    max_r = max(min_r + 10, int(0.20 * min_dim))
    min_d = max(12, int(0.58 * ((min_r + max_r) / 2.0)))

    raw = _hough_candidates(lowp, min_r=min_r, max_r=max_r, min_d=min_d)
    kept = _consensus(raw)

    overlay = rgb_img_u8.copy()
    for x, y, r in kept:
        cv2.circle(overlay, (x, y), r, (72, 236, 142), 2)
        cv2.circle(overlay, (x, y), max(2, r // 12), (255, 90, 90), -1)
    return len(kept), overlay

