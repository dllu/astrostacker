from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from astrostacker.metadata import LensCorrectionProfile


@dataclass(slots=True)
class RemapGrid:
    map_x: np.ndarray
    map_y: np.ndarray


def build_radial_remap(
    width: int,
    height: int,
    profile: LensCorrectionProfile,
) -> RemapGrid:
    yy, xx = np.indices((height, width), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radial_scale = float(np.hypot(cx, cy))
    radial_scale = max(radial_scale, 1.0)
    x = (xx - cx) / radial_scale
    y = (yy - cy) / radial_scale
    radius = np.sqrt(x * x + y * y)
    corrected_radius = radius + profile.radial_distortion_delta(radius)
    scale = np.ones_like(radius, dtype=np.float32)
    mask = radius > 1e-6
    scale[mask] = corrected_radius[mask] / radius[mask]
    map_x = (x * scale * radial_scale + cx).astype(np.float32)
    map_y = (y * scale * radial_scale + cy).astype(np.float32)
    return RemapGrid(map_x=map_x, map_y=map_y)


def undistort_image(image: np.ndarray, grid: RemapGrid) -> np.ndarray:
    return cv2.remap(
        image,
        grid.map_x,
        grid.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def undistort_valid_mask(shape: tuple[int, int], grid: RemapGrid) -> np.ndarray:
    mask = np.ones(shape, dtype=np.float32)
    remapped = cv2.remap(
        mask,
        grid.map_x,
        grid.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.clip(remapped, 0.0, 1.0)


def undistort_points(
    points: np.ndarray, width: int, height: int, profile: LensCorrectionProfile
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radial_scale = float(np.hypot(cx, cy))
    radial_scale = max(radial_scale, 1.0)
    x = (points[:, 0] - cx) / radial_scale
    y = (points[:, 1] - cy) / radial_scale
    radius = np.sqrt(x * x + y * y)
    corrected_radius = radius + profile.radial_distortion_delta(radius)
    scale = np.ones_like(radius, dtype=np.float32)
    mask = radius > 1e-6
    scale[mask] = corrected_radius[mask] / radius[mask]
    corrected_x = x * scale * radial_scale + cx
    corrected_y = y * scale * radial_scale + cy
    return np.column_stack([corrected_x, corrected_y]).astype(np.float32)
