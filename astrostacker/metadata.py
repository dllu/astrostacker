from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_RATIONAL_RE = re.compile(r"(-?\d+)(?:/(\d+))?")


@dataclass(slots=True)
class LensCorrectionProfile:
    make: str | None
    model: str | None
    lens_model: str | None
    vignette_nodes: np.ndarray | None
    vignette_values: np.ndarray | None
    distortion_nodes: np.ndarray | None
    distortion_values: np.ndarray | None

    def radial_gain(self, radius: np.ndarray) -> np.ndarray:
        if self.vignette_nodes is None or self.vignette_values is None:
            return np.ones_like(radius, dtype=np.float32)
        nodes = np.concatenate(
            [np.array([0.0], dtype=np.float32), self.vignette_nodes.astype(np.float32)]
        )
        values = np.concatenate(
            [np.array([1.0], dtype=np.float32), self.vignette_values.astype(np.float32)]
        )
        gains = np.interp(radius, nodes, values)
        gains = np.where(radius <= nodes[-1], gains, values[-1])
        return np.clip(gains.astype(np.float32), 0.5, 4.0)

    def radial_distortion_delta(self, radius: np.ndarray) -> np.ndarray:
        if self.distortion_nodes is None or self.distortion_values is None:
            return np.zeros_like(radius, dtype=np.float32)
        nodes = np.concatenate(
            [np.array([0.0], dtype=np.float32), self.distortion_nodes.astype(np.float32)]
        )
        values = np.concatenate(
            [np.array([0.0], dtype=np.float32), self.distortion_values.astype(np.float32)]
        )
        delta = np.interp(radius, nodes, values)
        delta = np.where(radius <= nodes[-1], delta, values[-1])
        return delta.astype(np.float32)


def _parse_rationals(text: str) -> list[float]:
    values: list[float] = []
    for token in text.split():
        match = _RATIONAL_RE.fullmatch(token)
        if not match:
            continue
        numerator = float(match.group(1))
        denominator = float(match.group(2) or "1")
        values.append(numerator / denominator)
    return values


def _parse_fuji_curve(
    values: list[float],
) -> tuple[float, np.ndarray, np.ndarray] | tuple[None, None, None]:
    if len(values) < 19:
        return None, None, None
    scale = float(values[0])
    nodes = np.asarray(values[1:10], dtype=np.float32)
    samples = np.asarray(values[10:19], dtype=np.float32)
    return scale, nodes, samples


def load_lens_profile(raw_path: Path) -> LensCorrectionProfile:
    cmd = ["exiv2", "-pt", str(raw_path)]
    output = subprocess.check_output(cmd, text=True)
    tags: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[0].startswith("Exif."):
            tags[parts[0]] = parts[3].strip()

    distortion_scale, distortion_nodes, distortion_values = _parse_fuji_curve(
        _parse_rationals(tags.get("Exif.Fujifilm.GeometricDistortionParams", ""))
    )
    _vignette_scale, vignette_nodes, vignette_values = _parse_fuji_curve(
        _parse_rationals(tags.get("Exif.Fujifilm.VignettingParams", ""))
    )
    if distortion_values is not None and distortion_scale not in (None, 0.0):
        distortion_values = distortion_values / float(distortion_scale)
    if vignette_values is not None:
        vignette_values = vignette_values[0] / np.clip(vignette_values, 1e-6, None)

    return LensCorrectionProfile(
        make=tags.get("Exif.Image.Make"),
        model=tags.get("Exif.Image.Model"),
        lens_model=tags.get("Exif.Photo.LensModel"),
        vignette_nodes=vignette_nodes,
        vignette_values=vignette_values,
        distortion_nodes=distortion_nodes,
        distortion_values=distortion_values,
    )
