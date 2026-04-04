from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class FusionResult:
    fused: np.ndarray
    weight_sum: np.ndarray


class StreamingFusion:
    def __init__(self, shape: tuple[int, int, int], trim: int) -> None:
        height, width, channels = shape
        self.trim = trim
        self.sky_num = np.zeros((height, width, channels), dtype=np.float32)
        self.sky_den = np.zeros((height, width), dtype=np.float32)
        self.fg_num = np.zeros((height, width, channels), dtype=np.float32)
        self.fg_den = np.zeros((height, width, channels), dtype=np.float32)
        self.fg_min = np.full((height, width, channels), np.inf, dtype=np.float32)
        self.fg_max = np.full((height, width, channels), -np.inf, dtype=np.float32)
        self.fg_min_weight = np.zeros((height, width, channels), dtype=np.float32)
        self.fg_max_weight = np.zeros((height, width, channels), dtype=np.float32)
        self.fg_count = np.zeros((height, width), dtype=np.float32)

    def add_sky(
        self,
        image: np.ndarray,
        homography: np.ndarray,
        sky_weight: np.ndarray,
    ) -> None:
        height, width = self.sky_den.shape
        # `homography` maps source-frame coordinates into the reference frame, which
        # is exactly the transform `warpPerspective` expects when `WARP_INVERSE_MAP`
        # is not set.
        warped_image = cv2.warpPerspective(
            image,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_weight = cv2.warpPerspective(
            sky_weight.astype(np.float32),
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        self.sky_num += warped_image * warped_weight[..., None]
        self.sky_den += warped_weight

    def add_foreground(self, image: np.ndarray, foreground_weight: np.ndarray) -> None:
        weight = foreground_weight.astype(np.float32)
        self.fg_num += image * weight[..., None]
        self.fg_den += weight[..., None]
        valid = weight > 1e-3
        if np.any(valid):
            expanded = valid[..., None]
            min_update = expanded & (image < self.fg_min)
            max_update = expanded & (image > self.fg_max)
            self.fg_min = np.where(min_update, image, self.fg_min)
            self.fg_max = np.where(max_update, image, self.fg_max)
            self.fg_min_weight = np.where(min_update, weight[..., None], self.fg_min_weight)
            self.fg_max_weight = np.where(max_update, weight[..., None], self.fg_max_weight)
            self.fg_count += valid.astype(np.float32)

    def finalize(self) -> FusionResult:
        fg_num = self.fg_num.copy()
        fg_den = self.fg_den.copy()
        if self.trim > 0:
            trim_mask = self.fg_count > self.trim * 2
            if np.any(trim_mask):
                trim_mask_3d = trim_mask[..., None]
                fg_num = np.where(
                    trim_mask_3d,
                    fg_num - self.fg_min * self.fg_min_weight - self.fg_max * self.fg_max_weight,
                    fg_num,
                )
                fg_den = np.where(
                    trim_mask_3d,
                    fg_den - self.fg_min_weight - self.fg_max_weight,
                    fg_den,
                )
        fg_num = np.clip(fg_num, 0.0, None)
        fg_den = np.clip(fg_den, 0.0, None)
        total_num = self.sky_num + fg_num
        total_den = self.sky_den[..., None] + fg_den
        fused = total_num / np.clip(total_den, 1e-6, None)
        weight_sum = np.mean(total_den, axis=2)
        return FusionResult(fused=np.clip(fused, 0.0, 16.0), weight_sum=weight_sum)


def create_streaming_fusion(shape: tuple[int, int, int], trim: int) -> StreamingFusion:
    return StreamingFusion(shape=shape, trim=trim)
