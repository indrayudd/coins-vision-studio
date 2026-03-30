from __future__ import annotations


import cv2
import numpy as np


def _coin_count(rgb_img_u8: np.ndarray) -> tuple[int, np.ndarray]:
    bgr = cv2.cvtColor(rgb_img_u8, cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (9, 9), 2.0)

    e = cv2.Canny(g, 60, 140)
    circles = cv2.HoughCircles(
        g,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=22,
        param1=130,
        param2=28,
        minRadius=10,
        maxRadius=75,
    )

    overlay = rgb_img_u8.copy()
    count = 0
    if circles is not None:
        for x, y, r in circles[0]:
            x_i, y_i, r_i = int(round(x)), int(round(y)), int(round(r))
            cv2.circle(overlay, (x_i, y_i), r_i, (72, 236, 142), 2)
            cv2.circle(overlay, (x_i, y_i), max(2, r_i // 12), (255, 90, 90), -1)
            count += 1
    return count, overlay

