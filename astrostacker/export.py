from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def resolve_output_path(path: Path) -> Path:
    if path.suffix == "":
        return path.with_suffix(".tiff")
    if path.suffix.lower() == ".dng":
        return path.with_suffix(".tiff")
    return path


def write_linear_tiff(
    path: Path,
    image: np.ndarray,
    *,
    software: str = "astrostacker",
) -> Path:
    resolved = resolve_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    data = np.round(clipped * 65535.0).astype(np.uint16)
    tifffile.imwrite(
        str(resolved),
        data,
        photometric="rgb",
        metadata=None,
        compression=None,
        software=software,
    )
    return resolved
