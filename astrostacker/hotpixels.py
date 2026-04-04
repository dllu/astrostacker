from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class HotPixelSummary:
    persistent_mask: np.ndarray
    detection_counts: np.ndarray
    threshold_count: int
    total_detections: list[int]


def _plane_index(dy: int, dx: int) -> int:
    return dy * 2 + dx


def _iter_bayer_planes(image: np.ndarray, pattern: str) -> tuple[int, int, str, np.ndarray]:
    for dy in range(2):
        for dx in range(2):
            yield dy, dx, pattern[_plane_index(dy, dx)], image[dy::2, dx::2]


def detect_hot_pixels(
    raw_visible: np.ndarray,
    bayer_pattern: str,
    *,
    kernel_size: int = 5,
    threshold_sigma: float = 16.0,
    min_excess: float = 0.02,
    min_value: float = 0.02,
    max_bright_support: int = 2,
) -> np.ndarray:
    hot_mask = np.zeros(raw_visible.shape, dtype=bool)
    support_kernel = np.ones((3, 3), dtype=np.uint8)
    for dy, dx, _color, plane in _iter_bayer_planes(raw_visible.astype(np.float32), bayer_pattern):
        if min(plane.shape) < kernel_size:
            continue
        local_median = cv2.medianBlur(plane, kernel_size)
        residual = np.clip(plane - local_median, 0.0, None)
        local_noise = cv2.medianBlur(np.abs(plane - local_median), kernel_size) * 1.4826 + 1e-4
        bright_support = cv2.filter2D(
            (plane > (local_median + np.maximum(min_excess * 0.5, local_noise * 3.0))).astype(np.uint8),
            cv2.CV_16U,
            support_kernel,
            borderType=cv2.BORDER_REFLECT,
        )
        candidates = residual > np.maximum(min_excess, local_noise * threshold_sigma)
        candidates &= plane > np.maximum(min_value, local_median + min_excess)
        candidates &= bright_support <= max_bright_support
        hot_mask[dy::2, dx::2] = candidates
    return hot_mask


def build_persistent_hot_pixel_map(
    detections: list[np.ndarray],
    *,
    min_fraction: float = 0.15,
    min_count: int = 2,
) -> HotPixelSummary:
    if not detections:
        empty = np.zeros((0, 0), dtype=bool)
        return HotPixelSummary(
            persistent_mask=empty,
            detection_counts=np.zeros((0, 0), dtype=np.uint16),
            threshold_count=min_count,
            total_detections=[],
        )
    counts = np.sum(
        np.stack([mask.astype(np.uint16) for mask in detections], axis=0), axis=0, dtype=np.uint16
    )
    threshold = max(min_count, int(np.ceil(len(detections) * min_fraction)))
    persistent_mask = counts >= threshold
    total_detections = [int(np.count_nonzero(mask)) for mask in detections]
    return HotPixelSummary(
        persistent_mask=persistent_mask,
        detection_counts=counts,
        threshold_count=threshold,
        total_detections=total_detections,
    )


def repair_hot_pixels(
    raw_visible: np.ndarray,
    hot_mask: np.ndarray,
    bayer_pattern: str,
    *,
    kernel_size: int = 5,
    iterations: int = 2,
) -> np.ndarray:
    if not np.any(hot_mask):
        return raw_visible.astype(np.float32, copy=True)
    corrected = raw_visible.astype(np.float32, copy=True)
    for _ in range(max(1, iterations)):
        changed = False
        for dy, dx, _color, plane in _iter_bayer_planes(corrected, bayer_pattern):
            plane_mask = hot_mask[dy::2, dx::2]
            if not np.any(plane_mask) or min(plane.shape) < kernel_size:
                continue
            local_median = cv2.medianBlur(plane, kernel_size)
            plane[plane_mask] = local_median[plane_mask]
            corrected[dy::2, dx::2] = plane
            changed = True
        if not changed:
            break
    return corrected
