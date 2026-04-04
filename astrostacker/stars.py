from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class StarField:
    points: np.ndarray
    response: np.ndarray
    enhanced: np.ndarray


def _sample_masked_values(
    image: np.ndarray, mask: np.ndarray, max_samples: int = 250_000
) -> np.ndarray:
    values = image[mask]
    if values.size <= max_samples:
        return values
    stride = max(1, values.size // max_samples)
    return values[::stride]


def detect_stars(
    image: np.ndarray,
    sky_mask: np.ndarray,
    *,
    max_stars: int = 4096,
) -> StarField:
    if not np.any(sky_mask):
        empty = np.zeros((0, 2), np.float32)
        return StarField(
            points=empty,
            response=np.zeros(0, dtype=np.float32),
            enhanced=np.zeros(sky_mask.shape, dtype=np.float32),
        )

    sky_mask = sky_mask.astype(bool)
    ys, xs = np.nonzero(sky_mask)
    margin = 16
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(sky_mask.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(sky_mask.shape[1], int(xs.max()) + margin + 1)

    gray = cv2.cvtColor(
        np.clip(image[y0:y1, x0:x1], 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2GRAY
    )
    mask_crop = sky_mask[y0:y1, x0:x1]
    background = cv2.GaussianBlur(gray, (0, 0), 10.0)
    enhanced_crop = np.clip(gray - background, 0.0, None)
    enhanced_crop *= mask_crop.astype(np.float32)

    sampled = _sample_masked_values(enhanced_crop, mask_crop)
    if sampled.size == 0:
        enhanced = np.zeros(sky_mask.shape, dtype=np.float32)
        enhanced[y0:y1, x0:x1] = enhanced_crop
        empty = np.zeros((0, 2), np.float32)
        return StarField(points=empty, response=np.zeros(0, dtype=np.float32), enhanced=enhanced)

    median = float(np.median(sampled))
    noise = float(np.median(np.abs(sampled - median)) * 1.4826 + 1e-5)
    threshold = median + noise * 5.0

    local_max = cv2.dilate(enhanced_crop, np.ones((5, 5), dtype=np.uint8))
    peaks = (enhanced_crop >= (local_max - 1e-6)) & (enhanced_crop > threshold) & mask_crop
    coords = np.argwhere(peaks)
    if len(coords) == 0:
        enhanced = np.zeros(sky_mask.shape, dtype=np.float32)
        enhanced[y0:y1, x0:x1] = enhanced_crop
        return StarField(
            points=np.zeros((0, 2), np.float32), response=np.zeros(0), enhanced=enhanced
        )

    responses = enhanced_crop[coords[:, 0], coords[:, 1]]
    order = np.argsort(responses)[::-1][:max_stars]
    coords = coords[order]
    responses = responses[order].astype(np.float32)

    refined = []
    yy, xx = np.mgrid[0:5, 0:5].astype(np.float32)
    for y, x in coords:
        py0 = max(y - 2, 0)
        py1 = min(y + 3, enhanced_crop.shape[0])
        px0 = max(x - 2, 0)
        px1 = min(x + 3, enhanced_crop.shape[1])
        patch = enhanced_crop[py0:py1, px0:px1]
        if patch.size == 0 or float(np.sum(patch)) <= 0:
            refined.append((x0 + float(x), y0 + float(y)))
            continue
        patch_h, patch_w = patch.shape
        weight_sum = float(np.sum(patch))
        cx = float(np.sum(patch * xx[:patch_h, :patch_w]) / weight_sum)
        cy = float(np.sum(patch * yy[:patch_h, :patch_w]) / weight_sum)
        refined.append((x0 + px0 + cx, y0 + py0 + cy))

    enhanced = np.zeros(sky_mask.shape, dtype=np.float32)
    enhanced[y0:y1, x0:x1] = enhanced_crop
    return StarField(points=np.asarray(refined, np.float32), response=responses, enhanced=enhanced)
