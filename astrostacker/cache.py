from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from astrostacker.segmentation import SegmentationResult
from astrostacker.stars import StarField


def _hash_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:24]


def _hash_array(array: np.ndarray) -> str:
    packed = np.ascontiguousarray(array)
    return hashlib.sha256(packed.view(np.uint8)).hexdigest()[:24]


def _path_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def segmentation_cache_path(
    cache_dir: Path,
    raw_paths: list[Path],
    *,
    segmentation_downsample: int,
    dilation_radius: int,
    blur_radius: int,
    sam3_checkpoint: str | None,
) -> Path:
    payload = {
        "inputs": [_path_fingerprint(path) for path in raw_paths],
        "segmentation_downsample": segmentation_downsample,
        "dilation_radius": dilation_radius,
        "blur_radius": blur_radius,
        "sam3_checkpoint": sam3_checkpoint,
        "version": 1,
    }
    return cache_dir / "segmentation" / f"{_hash_payload(payload)}.npz"


def load_cached_segmentation(path: Path) -> SegmentationResult | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return SegmentationResult(
            sky_mask=data["sky_mask"].astype(bool),
            foreground_mask=data["foreground_mask"].astype(bool),
            soft_sky_weight=data["soft_sky_weight"].astype(np.float32),
            backend=str(data["backend"].item()),
        )


def save_cached_segmentation(path: Path, segmentation: SegmentationResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sky_mask=segmentation.sky_mask.astype(np.uint8),
        foreground_mask=segmentation.foreground_mask.astype(np.uint8),
        soft_sky_weight=segmentation.soft_sky_weight.astype(np.float16),
        backend=np.array(segmentation.backend),
    )


def star_field_cache_path(
    cache_dir: Path,
    raw_path: Path,
    *,
    sky_mask: np.ndarray,
) -> Path:
    payload = {
        "input": _path_fingerprint(raw_path),
        "sky_mask": _hash_array(sky_mask.astype(np.uint8)),
        "version": 1,
    }
    return cache_dir / "stars" / f"{_hash_payload(payload)}.npz"


def load_cached_star_field(path: Path) -> StarField | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        return StarField(
            points=data["points"].astype(np.float32),
            response=data["response"].astype(np.float32),
            enhanced=data["enhanced"].astype(np.float32),
        )


def save_cached_star_field(path: Path, star_field: StarField) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=star_field.points.astype(np.float32),
        response=star_field.response.astype(np.float32),
        enhanced=star_field.enhanced.astype(np.float16),
    )
