from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_png(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    data = np.asarray(image)
    if data.dtype.kind == "f":
        data = np.clip(data, 0.0, 1.0)
        data = np.round(data * 255.0).astype(np.uint8)
    elif data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    if data.ndim == 3 and data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), data)


def overlay_points(
    image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    canvas = np.copy(image)
    height, width = canvas.shape[:2]
    for x, y in np.asarray(points):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        cv2.circle(canvas, (int(round(x)), int(round(y))), 5, color, 1, cv2.LINE_AA)
    return canvas


def overlay_vectors(
    image: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    color: tuple[int, int, int],
) -> np.ndarray:
    canvas = np.copy(image)
    height, width = canvas.shape[:2]
    for (x0, y0), (x1, y1) in zip(np.asarray(starts), np.asarray(ends), strict=True):
        if not np.all(np.isfinite((x0, y0, x1, y1))):
            continue
        if (
            (x0 < 0 and x1 < 0)
            or (x0 >= width and x1 >= width)
            or (y0 < 0 and y1 < 0)
            or (y0 >= height and y1 >= height)
        ):
            continue
        start = (int(round(x0)), int(round(y0)))
        end = (int(round(x1)), int(round(y1)))
        cv2.line(canvas, start, end, color, 1, cv2.LINE_8)
    return canvas


def draw_square_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    size: int = 3,
) -> np.ndarray:
    canvas = np.copy(image)
    radius = max(0, size // 2)
    height, width = canvas.shape[:2]
    for x, y in np.asarray(points):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        xi = int(round(x))
        yi = int(round(y))
        if xi < -radius or xi >= width + radius or yi < -radius or yi >= height + radius:
            continue
        x0 = max(0, xi - radius)
        x1 = min(width, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(height, yi + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        canvas[y0:y1, x0:x1] = np.asarray(color, dtype=canvas.dtype)
    return canvas
