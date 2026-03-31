from __future__ import annotations

import base64
import io
import os
import random
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import blob_dog, peak_local_max
from skimage.segmentation import watershed

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# def _one_threshold(gray_img):
#     # this was bad on anything not plain white.
#     # _, m = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)
#     # return m
#     pass
#
# def _hough_only(gray_img):
#     # circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1.2, 20)
#     # return [] if circles is None else circles[0]
#     return []


def get_image_paths(dataset_root: Path, split: str = "all") -> list[Path]:
    """Return image paths for one split (or all), sorted for repeatable behavior."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    # grab all image files we can find and pray sorting keeps stuff stable
    all_image_candidates = [p for p in dataset_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if split == "all":
        return sorted(all_image_candidates)

    split_key = split.lower()
    return sorted([p for p in all_image_candidates if p.parent.name.lower() == split_key])


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert float-ish image arrays into uint8 range expected by OpenCV/JPEG."""
    # make sure stuff is u8 or frontend gets odd colors
    if image.dtype == np.uint8:
        return image

    image = np.clip(image, 0.0, 1.0)  # yup clip then scale, old classic
    return (image * 255.0).astype(np.uint8)


def _encode_image(image: np.ndarray) -> str:
    """Encode ndarray image as browser-safe base64 JPEG data URL."""
    frame_u8 = _to_uint8(image)
    if frame_u8.ndim == 2:
        frame_u8 = np.stack([frame_u8] * 3, axis=-1)
    if frame_u8.shape[2] == 4:
        frame_u8 = frame_u8[:, :, :3]

    pil_frame = Image.fromarray(frame_u8, mode="RGB")
    jpeg_buffer = io.BytesIO()
    pil_frame.save(jpeg_buffer, format="JPEG", quality=92)
    encoded_payload = base64.b64encode(jpeg_buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_payload}"


def _load_image(path: Path, max_dim: int = 900) -> np.ndarray:
    """Load RGB image and downscale huge photos so pipeline remains stable-ish."""
    # loading + resize guard for giant phone photos
    with Image.open(path) as img:
        img = img.convert("RGB")
        src_w, src_h = img.size
        resize_gain = min(1.0, max_dim / float(max(src_w, src_h)))
        if resize_gain < 1.0:
            img = img.resize((int(src_w * resize_gain), int(src_h * resize_gain)), Image.Resampling.LANCZOS)
        return np.asarray(img)


def _remove_small_components(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Remove tiny connected components from binary mask using min-area cutoff."""
    # drop tiny junk blobs that are definitely not coins
    if min_area <= 1:
        return mask

    num_labs, labels_map, stats_blob, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_blob_mask = np.zeros_like(mask)
    for label in range(1, num_labs):
        area = float(stats_blob[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned_blob_mask[labels_map == label] = 255
    return cleaned_blob_mask


def _mask_from_intensity(low_pass: np.ndarray) -> tuple[np.ndarray, list[float], float]:
    """Build foreground mask from intensity with two Otsu variants and quality scoring."""
    kernel_open = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((7, 7), np.uint8)
    image_area = float(low_pass.shape[0] * low_pass.shape[1])

    def _prepare(binary_mask: np.ndarray) -> tuple[np.ndarray, list[float], float, float]:
        mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rough_areas = [float(cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 80]
        if len(rough_areas) >= 4:
            rough_radii = np.sqrt(np.array(rough_areas) / np.pi)
            rough_radius = float(np.quantile(rough_radii, 0.70))
        else:
            rough_radius = max(10.0, min(mask.shape[:2]) * 0.035)

        min_component_area = max(110.0, np.pi * (0.42 * rough_radius) ** 2)
        mask = _remove_small_components(mask, min_component_area)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = [float(cv2.contourArea(c)) for c in contours if cv2.contourArea(c) > 120]

        fg_fraction = float(np.mean(mask > 0))
        comp_count = len(areas)
        largest_frac = (max(areas) / image_area) if areas else 0.0
        median_area = float(np.median(areas)) if areas else 0.0

        score = 1.4 * min(comp_count, 50) / 50.0
        score -= 2.2 * abs(fg_fraction - 0.34)
        score -= 2.0 * max(0.0, largest_frac - 0.18)
        if median_area > 0:
            score += 0.25 * (1.0 - min(abs(np.log((median_area + 1.0) / 850.0)), 1.0))
        return mask, areas, rough_radius, score

    _, mask_inv = cv2.threshold(low_pass, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_norm = cv2.threshold(low_pass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    inv_candidate = _prepare(mask_inv)
    norm_candidate = _prepare(mask_norm)

    if norm_candidate[3] > inv_candidate[3]:
        return norm_candidate[0], norm_candidate[1], norm_candidate[2]
    return inv_candidate[0], inv_candidate[1], inv_candidate[2]


def _fft_low_pass(gray: np.ndarray, cutoff_ratio: float = 0.11) -> np.ndarray:
    """Do a simple FFT circular low-pass filter to mute high-frequency texture noise."""
    spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    radius = max(6, int(min(rows, cols) * cutoff_ratio))

    mask = np.zeros((rows, cols), dtype=np.float32)
    cv2.circle(mask, (ccol, crow), radius, 1.0, -1)
    filtered = spectrum * mask
    recon = np.fft.ifft2(np.fft.ifftshift(filtered))
    recon = np.abs(recon)
    return cv2.normalize(recon, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _illumination_normalize(gray: np.ndarray) -> np.ndarray:
    """Flatten uneven lighting by dividing by a blurred illumination field."""
    sigma = max(12.0, min(gray.shape[:2]) * 0.05)
    lighting = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray, lighting + 1, scale=186.0)
    return cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _deglare_gray(rgb_uint8: np.ndarray, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Detect bright low-saturation glare zones and inpaint them before edge work."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    v_hi = float(np.quantile(val, 0.94))
    s_lo = float(np.quantile(sat, 0.42))
    bright = gray >= np.quantile(gray, 0.975)
    specular = ((val >= v_hi) & (sat <= s_lo) & bright).astype(np.uint8) * 255
    specular = cv2.morphologyEx(specular, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    specular = cv2.morphologyEx(specular, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    glare_fraction = float(np.mean(specular > 0))
    if glare_fraction < 0.001:
        return gray, specular, glare_fraction

    repaired = cv2.inpaint(gray, specular, 5, cv2.INPAINT_TELEA)
    blended = cv2.addWeighted(gray, 0.36, repaired, 0.64, 0.0)
    return blended, specular, glare_fraction


def _suppress_long_lines(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Suppress long straight background textures (cloth/weave/wood grain vibes)."""
    min_dim = min(image.shape[:2])
    line_edges = cv2.Canny(image, 42, 128)
    min_line_length = max(36, int(0.17 * min_dim))
    line_threshold = max(34, int(0.09 * min_dim))
    lines = cv2.HoughLinesP(
        line_edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=line_threshold,
        minLineLength=min_line_length,
        maxLineGap=8,
    )
    line_mask = np.zeros_like(image, dtype=np.uint8)
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = (int(v) for v in line)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_line_length:
                continue
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=2)

    if np.any(line_mask):
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=1)

    line_fraction = float(np.mean(line_mask > 0))
    if line_fraction < 0.002:
        return image, line_mask, line_fraction

    repaired = cv2.inpaint(image, line_mask, 4, cv2.INPAINT_TELEA)
    suppressed = cv2.addWeighted(image, 0.34, repaired, 0.66, 0.0)
    return suppressed, line_mask, line_fraction


def _estimate_radius_band(mask: np.ndarray, contour_areas: list[float], min_dim: int) -> tuple[int, int, float]:
    """Estimate plausible circle radius search band from mask geometry."""
    radius_samples: list[float] = []
    if contour_areas:
        radius_samples.extend(np.sqrt(np.array(contour_areas) / np.pi).tolist())

    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    valid_dist = dist[dist > 0]
    if valid_dist.size >= 24:
        high = valid_dist[valid_dist >= np.quantile(valid_dist, 0.76)]
        if high.size:
            radius_samples.extend((1.35 * high).tolist())

    if len(radius_samples) >= 4:
        radii = np.array(radius_samples, dtype=np.float32)
        median_radius = float(np.median(radii))
        q80 = float(np.quantile(radii, 0.80))
    else:
        median_radius = max(12.0, 0.052 * min_dim)
        q80 = 1.24 * median_radius

    median_radius = float(np.clip(median_radius, 10.0, 0.16 * min_dim))
    min_radius = max(8, int(0.42 * median_radius))
    max_radius = max(min_radius + 10, int(max(2.85 * median_radius, 1.70 * q80)))
    min_radius = min(min_radius, int(0.13 * min_dim))
    max_radius = min(max_radius, int(0.20 * min_dim))
    return min_radius, max_radius, median_radius


def _watershed_regions(mask: np.ndarray, min_radius: int) -> np.ndarray:
    """Split foreground into region labels using watershed over distance transform."""
    binary = mask > 0
    if not np.any(binary):
        return np.zeros_like(mask, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)
    min_distance = max(5, int(0.72 * min_radius))
    threshold_abs = max(1.1, 0.44 * min_radius)
    coords = peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        labels=binary,
        exclude_border=False,
    )

    if len(coords) == 0:
        _, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return labels.astype(np.int32)

    markers = np.zeros_like(mask, dtype=np.int32)
    for idx, (y, x) in enumerate(coords, start=1):
        markers[int(y), int(x)] = idx
    markers = ndi.label(markers > 0)[0].astype(np.int32)
    labels = watershed(-distance, markers, mask=binary)
    return labels.astype(np.int32)


def _aux_hsv_coin_mask(rgb_uint8: np.ndarray) -> np.ndarray:
    """Build a backup HSV-driven coin-ish mask for low-contrast rescue cases."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 1.2)
    val = cv2.GaussianBlur(hsv[:, :, 2], (5, 5), 1.2)

    sat_thresh = float(np.quantile(sat, 0.52))
    val_low = float(np.quantile(val, 0.12))
    val_high = float(np.quantile(val, 0.985))
    aux = ((sat <= sat_thresh) & (val >= val_low) & (val <= val_high)).astype(np.uint8) * 255
    aux = cv2.morphologyEx(aux, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    aux = cv2.morphologyEx(aux, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    aux = _remove_small_components(aux, min_area=90.0)
    return aux


def _region_hough_candidates(
    image: np.ndarray,
    labels: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> list[tuple[float, float, float, float]]:
    """Run local Hough inside region labels to recover missed circles."""
    if labels.size == 0 or int(labels.max()) == 0:
        return []

    area_min = np.pi * (0.32 * min_radius) ** 2
    area_max = np.pi * (2.40 * max_radius) ** 2
    candidates: list[tuple[float, float, float, float]] = []

    for label_id, slc in enumerate(ndi.find_objects(labels), start=1):
        if slc is None:
            continue
        region = labels[slc] == label_id
        area = float(np.sum(region))
        if area < area_min or area > area_max:
            continue

        local_r = max(0.70 * min_radius, np.sqrt(area / np.pi))
        local_min = max(7, int(0.58 * local_r))
        local_max = min(max_radius, int(1.45 * local_r))
        if local_max <= local_min + 2:
            continue

        patch = image[slc].copy()
        fill = int(np.median(patch))
        patch[~region] = fill
        patch = cv2.GaussianBlur(patch, (7, 7), 1.4)
        circles = _run_hough_pass(
            patch,
            dp=1.15,
            min_dist=max(8, int(0.72 * local_r)),
            param1=112,
            param2=22,
            min_radius=local_min,
            max_radius=local_max,
        )
        for x, y, r in circles:
            candidates.append((float(x + slc[1].start), float(y + slc[0].start), float(r), 0.76))

    return candidates


def _dog_blob_candidates(
    gray: np.ndarray,
    mask: np.ndarray,
    min_radius: int,
    max_radius: int,
) -> list[tuple[float, float, float, float]]:
    """Use DoG blobs as fallback small-coin proposals when Hough is sparse."""
    if max_radius <= min_radius:
        return []

    normalized = gray.astype(np.float32) / 255.0
    min_sigma = max(1.4, (0.58 * min_radius) / np.sqrt(2.0))
    max_sigma = max(min_sigma + 1.2, (1.08 * max_radius) / np.sqrt(2.0))
    blobs = blob_dog(
        normalized,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=1.42,
        threshold=0.05,
        overlap=0.56,
    )

    candidates: list[tuple[float, float, float, float]] = []
    h, w = gray.shape
    for y, x, sigma in blobs:
        r = float(np.clip(np.sqrt(2.0) * sigma, 0.62 * min_radius, 1.30 * max_radius))
        x_i, y_i = int(round(x)), int(round(y))
        if x_i < 0 or x_i >= w or y_i < 0 or y_i >= h:
            continue
        y0, y1 = max(0, y_i - 2), min(h, y_i + 3)
        x0, x1 = max(0, x_i - 2), min(w, x_i + 3)
        local = mask[y0:y1, x0:x1]
        if local.size == 0:
            continue
        if float(np.mean(local > 0)) < 0.08:
            continue
        candidates.append((float(x), float(y), r, 0.52))
    return candidates


def _run_hough_pass(
    working_image: np.ndarray,
    *,
    dp: float,
    min_dist: int,
    param1: int,
    param2: int,
    min_radius: int,
    max_radius: int,
) -> np.ndarray:
    """Wrapper for one OpenCV Hough pass with specific parameters."""
    circles = cv2.HoughCircles(
        working_image,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return np.empty((0, 3), dtype=np.float32)
    return circles[0].astype(np.float32)


def _circle_edge_support(edges: np.ndarray, x: float, y: float, r: float, samples: int = 180) -> float:
    """Measure how much Canny edge energy lies on a proposed circle rim."""
    if r <= 0:
        return 0.0
    theta = np.linspace(0.0, 2.0 * np.pi, num=samples, endpoint=False)
    xs = np.clip(np.round(x + r * np.cos(theta)).astype(int), 0, edges.shape[1] - 1)
    ys = np.clip(np.round(y + r * np.sin(theta)).astype(int), 0, edges.shape[0] - 1)
    return float(np.mean(edges[ys, xs] > 0))


def _circle_quality(gray: np.ndarray, edges: np.ndarray, mask: np.ndarray, x: float, y: float, r: float) -> tuple[float, float, float]:
    """Compute local circle quality: edge support, contrast, and mask fill."""
    if r <= 4:
        return 0.0, 0.0, 0.0

    edge_support = _circle_edge_support(edges, x, y, r)

    yy, xx = np.ogrid[: gray.shape[0], : gray.shape[1]]
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    inner = dist_sq <= (0.72 * r) ** 2
    ring = (dist_sq >= (0.93 * r) ** 2) & (dist_sq <= (1.18 * r) ** 2)
    if not np.any(inner) or not np.any(ring):
        return edge_support, 0.0, 0.0

    inner_mean = float(np.mean(gray[inner]))
    ring_mean = float(np.mean(gray[ring]))
    contrast = abs(ring_mean - inner_mean) / 255.0

    mask_fill = float(np.mean(mask[inner] > 0))
    return edge_support, contrast, mask_fill


def _radial_gradient_profile(grad: np.ndarray, x: float, y: float, r: float) -> float:
    """Score radial gradient behavior (stronger ring than center is good)."""
    yy, xx = np.ogrid[: grad.shape[0], : grad.shape[1]]
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    inner = dist_sq <= (0.65 * r) ** 2
    ring = (dist_sq >= (0.92 * r) ** 2) & (dist_sq <= (1.18 * r) ** 2)
    if not np.any(inner) or not np.any(ring):
        return 0.0

    inner_grad = float(np.mean(grad[inner]))
    ring_grad = float(np.mean(grad[ring]))
    ratio = ring_grad / (inner_grad + 1e-6)
    return float(np.clip((ratio - 1.0) / 1.4, 0.0, 1.0))


def _consensus_circles(
    candidates: list[tuple[float, float, float, float]],
    gray: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    closeup_mode: bool,
    min_valid_radius: float,
    max_valid_radius: float,
    relaxed_mode: bool,
) -> list[tuple[int, int, int]]:
    """Cluster candidate circles and keep the most consistent/high-quality ones."""
    if not candidates:
        return []

    clusters: list[dict[str, float]] = []
    for x, y, r, weight in sorted(candidates, key=lambda c: c[3], reverse=True):
        matched = None
        for cluster in clusters:
            distance = float(np.hypot(x - cluster["x"], y - cluster["y"]))
            radius_delta = abs(r - cluster["r"])
            if distance < max(8.0, 0.46 * max(r, cluster["r"])) and radius_delta < 0.48 * max(r, cluster["r"]):
                matched = cluster
                break

        if matched is None:
            clusters.append({"x": x, "y": y, "r": r, "weight": weight, "votes": 1.0})
            continue

        total = matched["weight"] + weight
        matched["x"] = (matched["x"] * matched["weight"] + x * weight) / total
        matched["y"] = (matched["y"] * matched["weight"] + y * weight) / total
        matched["r"] = (matched["r"] * matched["weight"] + r * weight) / total
        matched["weight"] = total
        matched["votes"] += 1.0

    if not clusters:
        return []

    radii = np.array([cluster["r"] for cluster in clusters], dtype=float)
    median_radius = float(np.median(radii))
    soft_min = max(min_valid_radius, 0.30 * median_radius)
    soft_max = min(max_valid_radius, 3.35 * median_radius)

    filtered = [cluster for cluster in clusters if soft_min <= cluster["r"] <= soft_max]
    filtered = [cluster for cluster in filtered if min_valid_radius <= cluster["r"] <= max_valid_radius]
    if not filtered:
        filtered = clusters

    scored: list[tuple[float, dict[str, float]]] = []
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    for cluster in filtered:
        support, contrast, mask_fill = _circle_quality(gray, edges, mask, cluster["x"], cluster["y"], cluster["r"])
        profile = _radial_gradient_profile(grad_mag, cluster["x"], cluster["y"], cluster["r"])
        score = 0.72 * cluster["votes"] + 1.7 * support + 1.2 * contrast + 0.75 * mask_fill + 0.95 * profile + 0.14 * cluster["weight"]
        if relaxed_mode:
            has_consensus = (
                cluster["votes"] >= 1.0
                and profile >= 0.03
                and support >= 0.085
                and (mask_fill >= 0.06 or contrast >= 0.03 or (support >= 0.14 and profile >= 0.05))
            )
            has_strong_single = support >= 0.13 and profile >= 0.06 and (contrast >= 0.035 or mask_fill >= 0.10)
        else:
            has_consensus = (
                cluster["votes"] >= 1.5
                and profile >= 0.06
                and support >= 0.095
                and (mask_fill >= 0.10 or contrast >= 0.045)
                and (contrast >= 0.03 or cluster["votes"] >= 2.0)
            )
            has_strong_single = support >= 0.16 and contrast >= 0.035 and profile >= 0.10 and mask_fill >= 0.16
        if has_consensus or has_strong_single:
            scored.append((score, cluster))

    if not scored:
        scored = [(0.0, cluster) for cluster in filtered]

    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[int, int, int]] = []
    for _, cluster in scored:
        x_i, y_i, r_i = int(round(cluster["x"])), int(round(cluster["y"])), int(round(cluster["r"]))
        duplicate = False
        for x2, y2, r2 in kept:
            distance = float(np.hypot(x_i - x2, y_i - y2))
            if distance < max(9.0, 0.55 * max(r_i, r2)):
                duplicate = True
                break
        if not duplicate:
            kept.append((x_i, y_i, r_i))

    return kept


def _dedupe_circles(circles: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Drop near-duplicate circles after all rescue passes."""
    if not circles:
        return []

    kept: list[tuple[int, int, int]] = []
    for x_i, y_i, r_i in sorted(circles, key=lambda c: c[2], reverse=True):
        duplicate = False
        for x2, y2, r2 in kept:
            distance = float(np.hypot(x_i - x2, y_i - y2))
            if distance < max(9.0, 0.58 * max(r_i, r2)):
                duplicate = True
                break
        if not duplicate:
            kept.append((x_i, y_i, r_i))
    return kept


def _component_rescue_circles(
    mask: np.ndarray,
    *,
    edges: np.ndarray,
    min_radius: int,
    max_radius: int,
    existing: list[tuple[int, int, int]],
    max_new: int,
) -> list[tuple[int, int, int]]:
    """Rescue circles from mask components when consensus undercounts."""
    if max_new <= 0:
        return []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    area_min = float(np.pi * (0.50 * min_radius) ** 2)
    area_max = float(np.pi * (1.52 * max_radius) ** 2)
    proposals: list[tuple[float, tuple[int, int, int]]] = []
    occupied = list(existing)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < area_min or area > area_max:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = float((4.0 * np.pi * area) / (perimeter * perimeter + 1e-6))
        if circularity < 0.54:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = float(max(w, h) / (min(w, h) + 1e-6))
        if aspect > 1.55:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        if radius < 0.68 * min_radius or radius > 1.24 * max_radius:
            continue

        enclosure_fill = float(area / (np.pi * radius * radius + 1e-6))
        if enclosure_fill < 0.57:
            continue

        support = _circle_edge_support(edges, float(cx), float(cy), float(radius), samples=108)
        if support < 0.09:
            continue

        duplicate = False
        for x2, y2, r2 in occupied:
            distance = float(np.hypot(cx - x2, cy - y2))
            if distance < max(8.0, 0.60 * max(radius, r2)):
                duplicate = True
                break
        if duplicate:
            continue

        score = 0.62 * circularity + 0.22 * enclosure_fill + 0.16 * support
        proposals.append((score, (int(round(cx)), int(round(cy)), int(round(radius)))))

    proposals.sort(key=lambda item: item[0], reverse=True)
    rescued: list[tuple[int, int, int]] = []
    for _, circle in proposals:
        if len(rescued) >= max_new:
            break
        x_i, y_i, r_i = circle
        duplicate = False
        for x2, y2, r2 in occupied:
            distance = float(np.hypot(x_i - x2, y_i - y2))
            if distance < max(8.0, 0.58 * max(r_i, r2)):
                duplicate = True
                break
        if duplicate:
            continue
        rescued.append(circle)
        occupied.append(circle)
    return rescued


def _detect_circles(
    rgb_uint8: np.ndarray,
) -> tuple[
    list[tuple[int, int, int]],
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Main detection stack returning final circles, diagnostics, and stage images."""
    # the monster function: everything plus the kitchen sink.
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    deglared_gray, glare_mask, glare_ratio = _deglare_gray(rgb_uint8, gray)
    illum_gray = _illumination_normalize(deglared_gray)
    clahe_frame = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(illum_gray)
    fft_smooth = _fft_low_pass(clahe_frame, cutoff_ratio=0.11)
    bilateral_smooth = cv2.bilateralFilter(clahe_frame, d=9, sigmaColor=48, sigmaSpace=48)
    low_pass_mix = cv2.addWeighted(bilateral_smooth, 0.60, fft_smooth, 0.40, 0)
    low_pass_mix = cv2.GaussianBlur(low_pass_mix, (9, 9), 2.2)
    denoise_gray = cv2.medianBlur(low_pass_mix, 7)
    line_clean_gray, line_mask, line_ratio = _suppress_long_lines(denoise_gray)

    # adaptive canny thresholds from grad quantiles.
    grad_x = cv2.Sobel(line_clean_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(line_clean_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    p58 = float(np.quantile(grad_mag, 0.58))
    p86 = float(np.quantile(grad_mag, 0.86))
    canny_low = max(16, int(0.82 * p58))
    canny_high = max(canny_low + 18, int(1.08 * p86))
    edge_map = cv2.Canny(line_clean_gray, threshold1=canny_low, threshold2=canny_high)
    edge_map[line_mask > 0] = 0

    mask_seed = cv2.addWeighted(low_pass_mix, 0.56, line_clean_gray, 0.44, 0.0)
    fg_mask, contour_areas, _rough_radius_est = _mask_from_intensity(mask_seed)
    consensus_mask = fg_mask.copy()
    initial_fg_ratio = float(np.mean(fg_mask > 0))
    if initial_fg_ratio < 0.34:
        aux_hsv_mask = _aux_hsv_coin_mask(rgb_uint8)
        merged_fg_mask = cv2.bitwise_or(fg_mask, aux_hsv_mask)
        merged_fg_mask = cv2.morphologyEx(merged_fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        merged_fg_mask = cv2.morphologyEx(merged_fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        merged_fg_ratio = float(np.mean(merged_fg_mask > 0))
        if merged_fg_ratio <= 0.66 and merged_fg_ratio > initial_fg_ratio + 0.03:
            consensus_mask = merged_fg_mask
        if merged_fg_ratio <= 0.62 and merged_fg_ratio > initial_fg_ratio + 0.04:
            fg_mask = merged_fg_mask
            contours_local, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_areas = [float(cv2.contourArea(c)) for c in contours_local if cv2.contourArea(c) > 120]
    closeup_mode = False

    frame_min_dim = min(fg_mask.shape[0], fg_mask.shape[1])
    radius_min, radius_max, radius_mid = _estimate_radius_band(fg_mask, contour_areas, frame_min_dim)
    region_labels = _watershed_regions(fg_mask, min_radius=radius_min)
    minimum_dis = max(12, int(0.58 * radius_mid))
    hough_edge_gate = 140
    hough_vote_gate = 56
    fg_ratio = float(np.mean(fg_mask > 0))
    if fg_ratio > 0.45 and line_ratio < 0.02:
        minimum_dis = max(minimum_dis, int(0.72 * radius_mid))
        hough_vote_gate = min(88, hough_vote_gate + 6)
        hough_input = cv2.GaussianBlur(line_clean_gray, (13, 13), 3.6)
    else:
        hough_input = line_clean_gray

    hough_pass_plan = [
        (hough_input, 1.20, minimum_dis, hough_edge_gate, hough_vote_gate, radius_min, radius_max, 1.00),
        (
            low_pass_mix,
            1.25,
            max(12, int(minimum_dis * 0.82)),
            max(90, int(hough_edge_gate * 0.88)),
            max(24, int(hough_vote_gate * 0.84)),
            max(7, int(radius_min * 0.86)),
            max(radius_min + 10, int(radius_max * 1.28)),
            0.85,
        ),
        (
            line_clean_gray,
            1.30,
            max(12, int(minimum_dis * 0.74)),
            min(180, int(hough_edge_gate * 1.06)),
            min(96, int(hough_vote_gate * 1.08)),
            max(8, int(radius_min * 0.90)),
            max(radius_min + 10, int(radius_max * 1.20)),
            0.70,
        ),
    ]

    circle_proposals: list[tuple[float, float, float, float]] = []
    for work_img, dp_cfg, dis_cfg, p1_cfg, p2_cfg, r_min_cfg, r_max_cfg, pass_weight in hough_pass_plan:
        circles = _run_hough_pass(
            work_img,
            dp=dp_cfg,
            min_dist=dis_cfg,
            param1=p1_cfg,
            param2=p2_cfg,
            min_radius=r_min_cfg,
            max_radius=r_max_cfg,
        )
        for x, y, r in circles:
            circle_proposals.append((float(x), float(y), float(r), float(pass_weight)))

    # tiny-coin rescue pass at bigger scale (similar spirit to scale-space ideas).
    rescue_scale = 1.34
    upscaled_input = cv2.resize(hough_input, None, fx=rescue_scale, fy=rescue_scale, interpolation=cv2.INTER_CUBIC)
    upscaled_circles = _run_hough_pass(
        upscaled_input,
        dp=1.25,
        min_dist=max(12, int(minimum_dis * rescue_scale * 0.78)),
        param1=max(92, int(hough_edge_gate * 0.84)),
        param2=max(20, int(hough_vote_gate * 0.72)),
        min_radius=max(8, int(radius_min * rescue_scale * 0.90)),
        max_radius=max(10, int(radius_max * rescue_scale * 1.18)),
    )
    for x, y, r in upscaled_circles:
        circle_proposals.append((float(x / rescue_scale), float(y / rescue_scale), float(r / rescue_scale), 0.68))

    region_proposals = _region_hough_candidates(
        line_clean_gray,
        labels=region_labels,
        min_radius=radius_min,
        max_radius=radius_max,
    )
    circle_proposals.extend(region_proposals)

    if not closeup_mode and len(circle_proposals) < 12:
        rescue_pass_plan = [
            (
                low_pass_mix,
                1.25,
                max(10, int(minimum_dis * 0.76)),
                max(90, int(hough_edge_gate * 0.84)),
                max(24, int(hough_vote_gate * 0.62)),
                max(7, int(radius_min * 0.82)),
                max(radius_min + 10, int(radius_max * 1.35)),
                0.55,
            ),
            (
                hough_input,
                1.30,
                max(10, int(minimum_dis * 0.64)),
                max(86, int(hough_edge_gate * 0.80)),
                max(20, int(hough_vote_gate * 0.54)),
                max(7, int(radius_min * 0.78)),
                max(radius_min + 10, int(radius_max * 1.42)),
                0.45,
            ),
        ]

        for work_img, dp_cfg, dis_cfg, p1_cfg, p2_cfg, r_min_cfg, r_max_cfg, pass_weight in rescue_pass_plan:
            circles = _run_hough_pass(
                work_img,
                dp=dp_cfg,
                min_dist=dis_cfg,
                param1=p1_cfg,
                param2=p2_cfg,
                min_radius=r_min_cfg,
                max_radius=r_max_cfg,
            )
            for x, y, r in circles:
                circle_proposals.append((float(x), float(y), float(r), float(pass_weight)))

    # quick gate so nonsense circles don't flood consensus.
    filtered_proposals: list[tuple[float, float, float, float]] = []
    support_floor = 0.07 if glare_ratio > 0.010 else 0.08
    if fg_ratio < 0.30:
        support_floor = min(support_floor, 0.06)
    for x, y, r, weight in circle_proposals:
        if r < 0.78 * radius_min or r > 1.38 * radius_max:
            continue
        support = _circle_edge_support(edge_map, x, y, r, samples=132)
        if support < support_floor:
            continue
        filtered_proposals.append((x, y, r, float(weight + 0.16 * support)))

    circle_proposals = filtered_proposals
    texture_score = float(np.mean(np.abs(cv2.Laplacian(line_clean_gray, cv2.CV_32F))) / 255.0)
    dog_added_count = 0
    if len(circle_proposals) < 10 and texture_score < 0.24:
        dog_rescue = _dog_blob_candidates(line_clean_gray, fg_mask, min_radius=radius_min, max_radius=radius_max)
        for x, y, r, weight in dog_rescue:
            support = _circle_edge_support(edge_map, x, y, r, samples=120)
            if support < 0.11:
                continue
            circle_proposals.append((x, y, r, float(weight + 0.14 * support)))
            dog_added_count += 1

    valid_radius_min = float(max(5, int(0.66 * radius_min)))
    valid_radius_max = float(radius_max if not closeup_mode else int(0.56 * frame_min_dim))
    relaxed_mode = (
        (fg_ratio > 0.68 and len(circle_proposals) > 36 and line_ratio < 0.02)
        or line_ratio > 0.015
        or glare_ratio > 0.010
        or (fg_ratio < 0.32 and len(circle_proposals) >= 60)
        or (radius_min <= 10 and len(circle_proposals) >= 70)
    )
    final_circles = _consensus_circles(
        circle_proposals,
        gray=line_clean_gray,
        edges=edge_map,
        mask=consensus_mask,
        closeup_mode=closeup_mode,
        min_valid_radius=valid_radius_min,
        max_valid_radius=valid_radius_max,
        relaxed_mode=relaxed_mode,
    )

    expected_from_proposals = int(round(len(circle_proposals) / 3.35))
    rescue_gap = max(0, expected_from_proposals - len(final_circles))
    rescue_risk_mode = (
        line_ratio > 0.012
        or glare_ratio > 0.010
        or fg_ratio < 0.30
        or radius_min <= 10
    )
    rescue_limit = 0
    if rescue_gap >= 2 and rescue_risk_mode:
        rescue_limit = min(6, max(2, rescue_gap // 2))
    rescued_components = _component_rescue_circles(
        fg_mask,
        edges=edge_map,
        min_radius=radius_min,
        max_radius=radius_max,
        existing=final_circles,
        max_new=rescue_limit,
    )
    if rescued_components:
        final_circles = _dedupe_circles([*final_circles, *rescued_components])

    debug_params = {
        "canny_low_threshold": float(canny_low),
        "canny_high_threshold": float(canny_high),
        "hough_min_radius": float(radius_min),
        "hough_max_radius": float(radius_max),
        "hough_min_dist": float(minimum_dis),
        "hough_param2": float(hough_vote_gate),
        "closeup_mode": float(1.0 if closeup_mode else 0.0),
        "hough_pass_candidates": float(len(circle_proposals)),
        "foreground_fraction": fg_ratio,
        "line_artifact_fraction": line_ratio,
        "glare_fraction": glare_ratio,
        "texture_energy": texture_score,
        "dog_fallback_candidates": float(dog_added_count),
        "region_proposals": float(len(region_proposals)),
        "component_rescued": float(len(rescued_components)),
    }
    return final_circles, debug_params, gray, clahe_frame, low_pass_mix, line_clean_gray, edge_map, fg_mask, glare_mask


def _mask_overlay(rgb_uint8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render mask boundaries on RGB image for explainability panel."""
    # draw contour borders for preview card.
    overlay = rgb_uint8.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (245, 176, 62), 2)
    return overlay


def run_coin_pipeline(image_path: Path) -> dict[str, Any]:
    """Run complete pipeline on one image and package all intermediate stages."""
    # top-level runner the API calls for each image.
    rgb_uint8 = _load_image(image_path)
    final_circles, debug_params, gray_frame, clahe_frame, low_pass_frame, line_clean_frame, edge_frame, fg_mask, glare_mask = _detect_circles(rgb_uint8)

    overlay_frame = rgb_uint8.copy()
    for x, y, r in final_circles:
        cv2.circle(overlay_frame, (x, y), r, (72, 236, 142), 2)
        cv2.circle(overlay_frame, (x, y), max(2, r // 12), (255, 90, 90), -1)

    mask_outline_frame = _mask_overlay(rgb_uint8, fg_mask)

    step_images = [
        {
            "id": "original",
            "title": "Original Frame",
            "description": "Randomly sampled image from the selected dataset split.",
            "image": _encode_image(rgb_uint8),
        },
        {
            "id": "grayscale",
            "title": "Grayscale Projection",
            "description": "RGB image converted to grayscale for classical edge and circle operations.",
            "image": _encode_image(gray_frame.astype(np.float32) / 255.0),
        },
        {
            "id": "contrast",
            "title": "CLAHE Contrast Boost",
            "description": "Local contrast enhancement to make coin boundaries stand out in uneven lighting.",
            "image": _encode_image(clahe_frame.astype(np.float32) / 255.0),
        },
        {
            "id": "deglare",
            "title": "Specular Suppression",
            "description": "Bright low-saturation glare highlights are inpainted to recover missing coin rim edges.",
            "image": _encode_image(glare_mask.astype(np.float32) / 255.0),
        },
        {
            "id": "lowpass",
            "title": "Low-Pass Filtering",
            "description": "Bilateral + Gaussian smoothing suppresses high-frequency background artifacts before edge extraction.",
            "image": _encode_image(low_pass_frame.astype(np.float32) / 255.0),
        },
        {
            "id": "line_suppress",
            "title": "Line-Artifact Suppression",
            "description": "Long linear textures are inpainted to avoid stripe and weave backgrounds dominating circle votes.",
            "image": _encode_image(line_clean_frame.astype(np.float32) / 255.0),
        },
        {
            "id": "edges",
            "title": "Canny Edge Map",
            "description": "Adaptive Canny edge extraction from image gradient statistics.",
            "image": _encode_image(edge_frame.astype(np.float32) / 255.0),
        },
        {
            "id": "mask",
            "title": "Foreground Segmentation",
            "description": "Otsu-based foreground mask isolates coin-like regions before circle search.",
            "image": _encode_image(fg_mask.astype(np.float32) / 255.0),
        },
        {
            "id": "watershed",
            "title": "Region Boundary View",
            "description": "Foreground component boundaries used to set stable circle search scale.",
            "image": _encode_image(mask_outline_frame),
        },
        {
            "id": "hough",
            "title": "Final Circle Detections",
            "description": "Hough circles with geometric dedupe to avoid repeated detections on the same coin.",
            "image": _encode_image(overlay_frame),
        },
    ]

    return {
        "coin_count": len(final_circles),
        "steps": step_images,
        "parameters": {
            **debug_params,
            "candidate_counts": {
                "watershed": 0,
                "hough": len(final_circles),
                "total": len(final_circles),
                "final": len(final_circles),
            },
        },
    }


def estimate_coin_count(image_path: Path) -> int:
    """Fast count-only wrapper used by evaluation jobs."""
    # minimal wrapper for scripts.
    rgb_uint8 = _load_image(image_path)
    final_circles, *_ = _detect_circles(rgb_uint8)
    return len(final_circles)


def draw_and_process(dataset_root: Path, split: str = "all", seed: int | None = None) -> dict[str, Any]:
    """Sample one image from split, process it, and attach metadata fields."""
    # draw exactly one random sample and process it.
    image_pool = get_image_paths(dataset_root=dataset_root, split=split)
    if not image_pool:
        raise ValueError(f"No images found in split '{split}' under {dataset_root}")

    rng_box = random.Random(seed)
    chosen_image = rng_box.choice(image_pool)

    draw_result = run_coin_pipeline(chosen_image)
    draw_result["image_path"] = str(chosen_image.relative_to(dataset_root.parent))
    draw_result["split"] = split
    draw_result["seed"] = seed
    return draw_result


def draw_many_and_process(
    dataset_root: Path,
    *,
    split: str = "all",
    seed: int | None = None,
    count: int = 10,
) -> list[dict[str, Any]]:
    """Sample many images from split and run pipeline for each one."""
    # same thing but for a batch.
    image_pool = get_image_paths(dataset_root=dataset_root, split=split)
    if not image_pool:
        raise ValueError(f"No images found in split '{split}' under {dataset_root}")
    if count <= 0:
        raise ValueError("Count must be positive.")

    rng_box = random.Random(seed)
    if count >= len(image_pool):
        chosen_batch = image_pool.copy()
        rng_box.shuffle(chosen_batch)
        chosen_batch = chosen_batch[:count]
    else:
        chosen_batch = rng_box.sample(image_pool, count)

    batch_results: list[dict[str, Any]] = []
    for sample_idx, image_path in enumerate(chosen_batch):
        draw_result = run_coin_pipeline(image_path)
        draw_result["image_path"] = str(image_path.relative_to(dataset_root.parent))
        draw_result["split"] = split
        draw_result["seed"] = None if seed is None else seed + sample_idx
        batch_results.append(draw_result)
    return batch_results
