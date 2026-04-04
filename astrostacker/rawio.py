from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rawpy
from rawpy import LibRawError, NotSupportedError

from astrostacker.hotpixels import detect_hot_pixels, repair_hot_pixels
from astrostacker.metadata import LensCorrectionProfile, load_lens_profile


@dataclass(slots=True)
class RawSensorFrame:
    path: Path
    raw_visible: np.ndarray
    black_level: float
    white_level: float
    bayer_pattern: str
    cfa_rgb: np.ndarray
    rgb_preview: np.ndarray
    debug_preview: np.ndarray


@dataclass(slots=True)
class RawFrame:
    path: Path
    raw_visible: np.ndarray
    black_level: float
    white_level: float
    bayer_pattern: str
    rgb_preview: np.ndarray
    linear_rgb: np.ndarray | None
    cfa_rgb: np.ndarray
    profile: LensCorrectionProfile
    hot_pixel_mask: np.ndarray | None
    transient_hot_pixel_count: int
    persistent_hot_pixel_count: int
    corrected_hot_pixel_count: int


def _normalize_linear(image: np.ndarray, percentile: float = 99.7) -> np.ndarray:
    scale = np.percentile(image, percentile)
    if scale <= 0:
        scale = 1.0
    return np.clip(image / scale, 0.0, 1.0)


def _simple_cfa_to_rgb(raw: np.ndarray, pattern: str) -> np.ndarray:
    pattern_map = {
        "RGGB": cv2.COLOR_BayerRG2RGB,
        "BGGR": cv2.COLOR_BayerBG2RGB,
        "GRBG": cv2.COLOR_BayerGR2RGB,
        "GBRG": cv2.COLOR_BayerGB2RGB,
    }
    code = pattern_map.get(pattern, cv2.COLOR_BayerRG2RGB)
    clipped = np.clip(raw, 0.0, 1.0)
    return cv2.cvtColor((clipped * 65535.0).astype(np.uint16), code).astype(np.float32) / 65535.0


def _color_desc_to_string(color_desc: str | bytes | np.ndarray, raw_pattern: np.ndarray) -> str:
    if isinstance(color_desc, bytes):
        lookup = [chr(value) for value in color_desc]
    else:
        lookup = [
            chr(int(value)) if isinstance(value, (np.integer, int)) else str(value)
            for value in color_desc
        ]
    return "".join(lookup[int(index)] for index in raw_pattern.flatten().tolist())


def _postprocess_linear_rgb(raw: rawpy.RawPy) -> np.ndarray:
    algorithms = [
        rawpy.DemosaicAlgorithm.DCB,
        rawpy.DemosaicAlgorithm.LMMSE,
        rawpy.DemosaicAlgorithm.AHD,
        rawpy.DemosaicAlgorithm.LINEAR,
    ]
    last_error: Exception | None = None
    for algorithm in algorithms:
        try:
            post = raw.postprocess(
                gamma=(1.0, 1.0),
                no_auto_bright=True,
                output_bps=16,
                user_flip=0,
                demosaic_algorithm=algorithm,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.raw,
                highlight_mode=rawpy.HighlightMode.Blend,
            )
            return post.astype(np.float32) / 65535.0
        except (NotSupportedError, LibRawError) as exc:
            last_error = exc
    raise RuntimeError("Unable to demosaic RAW with available rawpy algorithms.") from last_error


def _load_raw_visible(
    raw: rawpy.RawPy,
) -> tuple[np.ndarray, float, float, str]:
    raw_visible = raw.raw_image_visible.astype(np.float32)
    black_level = float(np.mean(np.atleast_1d(raw.black_level_per_channel)))
    white_level = float(raw.white_level or np.max(raw_visible))
    bayer_pattern = _color_desc_to_string(raw.color_desc, raw.raw_pattern)
    linear_raw = np.clip(
        (raw_visible - black_level) / max(white_level - black_level, 1.0), 0.0, 1.0
    )
    return linear_raw, black_level, white_level, bayer_pattern


def read_raw_sensor(raw_path: Path, *, preview_scale: int = 8) -> RawSensorFrame:
    with rawpy.imread(str(raw_path)) as raw:
        linear_raw, black_level, white_level, bayer_pattern = _load_raw_visible(raw)
        cfa_rgb = _simple_cfa_to_rgb(linear_raw, bayer_pattern)
        preview = cv2.resize(
            cfa_rgb,
            (cfa_rgb.shape[1] // preview_scale, cfa_rgb.shape[0] // preview_scale),
            interpolation=cv2.INTER_AREA,
        )
        rgb_preview = _normalize_linear(preview)
        debug_scale = max(1, preview_scale // 4)
        debug_preview = cv2.resize(
            cfa_rgb,
            (cfa_rgb.shape[1] // debug_scale, cfa_rgb.shape[0] // debug_scale),
            interpolation=cv2.INTER_AREA,
        )
        debug_preview = _normalize_linear(debug_preview)
    return RawSensorFrame(
        path=raw_path,
        raw_visible=linear_raw,
        black_level=black_level,
        white_level=white_level,
        bayer_pattern=bayer_pattern,
        cfa_rgb=cfa_rgb,
        rgb_preview=rgb_preview,
        debug_preview=debug_preview,
    )


def read_raw_frame(
    raw_path: Path,
    *,
    preview_scale: int = 4,
    full_demosaic: bool = True,
    persistent_hot_pixel_mask: np.ndarray | None = None,
) -> RawFrame:
    with rawpy.imread(str(raw_path)) as raw:
        linear_raw, black_level, white_level, bayer_pattern = _load_raw_visible(raw)
        transient_hot_pixels = detect_hot_pixels(linear_raw, bayer_pattern)
        transient_hot_pixel_count = int(np.count_nonzero(transient_hot_pixels))
        combined_hot_pixels = transient_hot_pixels
        persistent_hot_pixel_count = 0
        if persistent_hot_pixel_mask is not None and persistent_hot_pixel_mask.shape == linear_raw.shape:
            persistent_hot_pixel_count = int(np.count_nonzero(persistent_hot_pixel_mask))
            combined_hot_pixels = combined_hot_pixels | persistent_hot_pixel_mask
        corrected_hot_pixel_count = int(np.count_nonzero(combined_hot_pixels))
        corrected_linear_raw = repair_hot_pixels(linear_raw, combined_hot_pixels, bayer_pattern)
        corrected_sensor = corrected_linear_raw * max(white_level - black_level, 1.0) + black_level
        raw.raw_image_visible[:] = np.clip(
            corrected_sensor,
            0.0,
            float(raw.white_level or np.max(raw.raw_image_visible)),
        ).astype(raw.raw_image_visible.dtype)

        cfa_rgb = _simple_cfa_to_rgb(corrected_linear_raw, bayer_pattern)
        preview = cv2.resize(
            cfa_rgb,
            (cfa_rgb.shape[1] // preview_scale, cfa_rgb.shape[0] // preview_scale),
            interpolation=cv2.INTER_AREA,
        )
        rgb_preview = _normalize_linear(preview)

        linear_rgb = None
        if full_demosaic:
            linear_rgb = _postprocess_linear_rgb(raw)

    return RawFrame(
        path=raw_path,
        raw_visible=corrected_linear_raw,
        black_level=black_level,
        white_level=white_level,
        bayer_pattern=bayer_pattern,
        rgb_preview=rgb_preview,
        linear_rgb=linear_rgb,
        cfa_rgb=cfa_rgb,
        profile=load_lens_profile(raw_path),
        hot_pixel_mask=combined_hot_pixels.astype(bool),
        transient_hot_pixel_count=transient_hot_pixel_count,
        persistent_hot_pixel_count=persistent_hot_pixel_count,
        corrected_hot_pixel_count=corrected_hot_pixel_count,
    )
