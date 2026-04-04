from __future__ import annotations

import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_RATIONAL_RE = re.compile(r"(-?\d+)(?:/(\d+))?")
_EXIV2_WARNED = False


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


@dataclass(slots=True, frozen=True)
class ExifTag:
    key: str
    type_name: str
    value: str


@dataclass(slots=True, frozen=True)
class OutputExifMetadata:
    tags: tuple[ExifTag, ...]


_OUTPUT_EXIF_TAGS = (
    "Exif.Image.Make",
    "Exif.Image.Model",
    "Exif.Image.DateTime",
    "Exif.Photo.DateTimeOriginal",
    "Exif.Photo.DateTimeDigitized",
    "Exif.Photo.OffsetTime",
    "Exif.Photo.OffsetTimeOriginal",
    "Exif.Photo.OffsetTimeDigitized",
    "Exif.Photo.ExposureTime",
    "Exif.Photo.FNumber",
    "Exif.Photo.ExposureBiasValue",
    "Exif.Photo.ISOSpeedRatings",
    "Exif.Photo.PhotographicSensitivity",
    "Exif.Photo.RecommendedExposureIndex",
    "Exif.Photo.FocalLength",
    "Exif.Photo.FocalLengthIn35mmFilm",
    "Exif.Photo.LensMake",
    "Exif.Photo.LensModel",
    "Exif.Photo.LensSpecification",
    "Exif.GPSInfo.GPSLatitudeRef",
    "Exif.GPSInfo.GPSLatitude",
    "Exif.GPSInfo.GPSLongitudeRef",
    "Exif.GPSInfo.GPSLongitude",
    "Exif.GPSInfo.GPSAltitudeRef",
    "Exif.GPSInfo.GPSAltitude",
    "Exif.GPSInfo.GPSDateStamp",
    "Exif.GPSInfo.GPSTimeStamp",
)


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


def _read_exif_tags(
    image_path: Path,
    *,
    keys: tuple[str, ...] | None = None,
) -> dict[str, ExifTag]:
    if shutil.which("exiv2") is None:
        _warn_missing_exiv2("reading EXIF metadata")
        return {}

    cmd = ["exiv2", "-PEkycv"]
    if keys is not None:
        for key in keys:
            cmd.extend(["-K", key])
    cmd.append(str(image_path))
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        _warn_missing_exiv2("reading EXIF metadata")
        return {}

    tags: dict[str, ExifTag] = {}
    for line in output.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3 or not parts[0].startswith("Exif."):
            continue
        key = parts[0]
        type_name = parts[1]
        value = parts[3].strip() if len(parts) == 4 else ""
        tags[key] = ExifTag(key=key, type_name=type_name, value=value)
    return tags


def _warn_missing_exiv2(action: str) -> None:
    global _EXIV2_WARNED
    if _EXIV2_WARNED:
        return
    warnings.warn(
        f"exiv2 is not available; {action} will be skipped.",
        RuntimeWarning,
        stacklevel=2,
    )
    _EXIV2_WARNED = True


def load_lens_profile(raw_path: Path) -> LensCorrectionProfile:
    tags = {tag.key: tag.value for tag in _read_exif_tags(raw_path).values()}

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


def load_output_exif_metadata(raw_path: Path) -> OutputExifMetadata:
    tags = _read_exif_tags(raw_path, keys=_OUTPUT_EXIF_TAGS)
    ordered_tags = tuple(tags[key] for key in _OUTPUT_EXIF_TAGS if key in tags and tags[key].value)
    return OutputExifMetadata(tags=ordered_tags)


def apply_output_exif_metadata(path: Path, metadata: OutputExifMetadata | None) -> None:
    if metadata is None or not metadata.tags:
        return
    if shutil.which("exiv2") is None:
        _warn_missing_exiv2("writing output EXIF metadata")
        return
    cmd = ["exiv2"]
    for tag in metadata.tags:
        cmd.extend(["-M", f"set {tag.key} {tag.type_name} {tag.value}"])
    cmd.append(str(path))
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        _warn_missing_exiv2("writing output EXIF metadata")
