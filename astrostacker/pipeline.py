from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from astrostacker.alignment import AlignmentResult, FrameAlignment, solve_frame_homography
from astrostacker.cache import (
    load_cached_segmentation,
    load_cached_star_field,
    save_cached_segmentation,
    save_cached_star_field,
    segmentation_cache_path,
    star_field_cache_path,
)
from astrostacker.debug import ensure_dir, overlay_points, overlay_vectors, write_png
from astrostacker.export import write_linear_tiff
from astrostacker.fusion import FusionResult, create_streaming_fusion
from astrostacker.hotpixels import build_persistent_hot_pixel_map, detect_hot_pixels
from astrostacker.lens import build_radial_remap, undistort_image, undistort_valid_mask
from astrostacker.metadata import LensCorrectionProfile
from astrostacker.rawio import read_raw_frame, read_raw_sensor
from astrostacker.segmentation import SegmentationResult, segment_sky
from astrostacker.stars import StarField, detect_stars


@dataclass(slots=True)
class FrameContext:
    path: Path
    rgb_preview: np.ndarray
    full_shape: tuple[int, int]
    profile: LensCorrectionProfile


@dataclass(slots=True)
class PipelineOutputs:
    frames: list[FrameContext]
    segmentation: SegmentationResult
    star_fields: list[StarField]
    alignment: AlignmentResult
    fusion: FusionResult
    output_path: Path


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    width = shape[1]
    height = shape[0]
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)


def _log_timing(label: str, start_time: float) -> None:
    elapsed = perf_counter() - start_time
    print(f"[astrostacker] {label}: {elapsed:.2f}s")


def _log_hot_pixels(label: str, *, transient: int, persistent: int, corrected: int) -> None:
    print(
        "[astrostacker] "
        f"{label}: transient={transient} persistent_map={persistent} corrected_total={corrected}"
    )


def _phase_image(image: np.ndarray) -> np.ndarray:
    phase = cv2.cvtColor(np.clip(image, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(phase, (0, 0), 1.2)


def _empty_points() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


def _detect_star_field(
    *,
    cache_dir: Path | None,
    path: Path,
    corrected: np.ndarray,
    sky_mask: np.ndarray,
) -> StarField:
    stars = None
    if cache_dir is not None:
        stars = load_cached_star_field(
            star_field_cache_path(
                cache_dir,
                path,
                sky_mask=sky_mask,
            )
        )
    if stars is None:
        stars = detect_stars(corrected, sky_mask)
        if cache_dir is not None:
            save_cached_star_field(
                star_field_cache_path(
                    cache_dir,
                    path,
                    sky_mask=sky_mask,
                ),
                stars,
            )
    return stars


def _write_star_debug(debug_dir: Path, index: int, corrected: np.ndarray, stars: StarField) -> None:
    overlay = overlay_points(
        np.clip(corrected, 0.0, 1.0),
        stars.points if len(stars.points) > 0 else _empty_points(),
        color=(255, 192, 0),
    )
    write_png(debug_dir / f"stars_{index:02d}.png", overlay)
    write_png(
        debug_dir / f"enhanced_stars_{index:02d}.png", np.clip(stars.enhanced * 8.0, 0.0, 1.0)
    )


def _write_alignment_debug(
    debug_dir: Path,
    index: int,
    corrected: np.ndarray,
    valid_mask: np.ndarray,
    stars: StarField,
    frame_alignment: FrameAlignment,
) -> None:
    transformed_points = (
        cv2.perspectiveTransform(
            stars.points.reshape(-1, 1, 2),
            frame_alignment.homography.astype(np.float32),
        ).reshape(-1, 2)
        if len(stars.points) > 0
        else _empty_points()
    )
    masked = np.clip(corrected * valid_mask[..., None], 0.0, 1.0)
    vector_overlay = overlay_vectors(
        masked,
        stars.points if len(stars.points) > 0 else _empty_points(),
        transformed_points,
        color=(0, 255, 255),
    )
    write_png(debug_dir / f"alignment_vectors_{index:02d}.png", vector_overlay)
    residual_overlay = overlay_vectors(
        masked,
        frame_alignment.aligned_match_source[frame_alignment.inlier_mask]
        if len(frame_alignment.aligned_match_source) > 0
        else _empty_points(),
        frame_alignment.match_target[frame_alignment.inlier_mask]
        if len(frame_alignment.match_target) > 0
        else _empty_points(),
        color=(255, 64, 64),
    )
    write_png(debug_dir / f"alignment_error_vectors_{index:02d}.png", residual_overlay)


def _build_alignment_result(
    reference_index: int,
    frame_alignments: list[FrameAlignment],
) -> AlignmentResult:
    return AlignmentResult(
        homographies=[item.homography for item in frame_alignments],
        reference_index=reference_index,
        matched_pairs=[len(item.match_source) for item in frame_alignments],
        inlier_counts=[int(np.count_nonzero(item.inlier_mask)) for item in frame_alignments],
        reprojection_errors=[item.reprojection_error for item in frame_alignments],
        match_sources=[item.match_source for item in frame_alignments],
        match_targets=[item.match_target for item in frame_alignments],
        inlier_masks=[item.inlier_mask for item in frame_alignments],
        aligned_match_sources=[item.aligned_match_source for item in frame_alignments],
    )


def _hot_pixel_preview(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _hot_pixel_preview_points(mask: np.ndarray, scale: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return _empty_points()
    points = np.column_stack(
        [
            coords[:, 1].astype(np.float32) / float(scale),
            coords[:, 0].astype(np.float32) / float(scale),
        ]
    )
    return points.astype(np.float32)


def run_pipeline(
    raw_paths: list[Path],
    *,
    output_path: Path,
    debug_dir: Path,
    cache_dir: Path | None = Path(".astrostacker-cache"),
    preview_scale: int = 4,
    segmentation_downsample: int = 4,
    dilation_radius: int = 15,
    blur_radius: int = 7,
    trim: int = 1,
    sam3_checkpoint: str | None = None,
) -> PipelineOutputs:
    if not raw_paths:
        raise ValueError("No RAW input paths were provided.")

    pipeline_start = perf_counter()
    ensure_dir(debug_dir)
    reference_index = len(raw_paths) // 2

    frames: list[FrameContext | None] = [None] * len(raw_paths)
    star_fields: list[StarField | None] = [None] * len(raw_paths)
    frame_alignments: list[FrameAlignment | None] = [None] * len(raw_paths)

    hot_pixel_start = perf_counter()
    hot_pixel_detections: list[np.ndarray] = []
    reference_sensor_preview_shape: tuple[int, int] | None = None
    for index, path in enumerate(raw_paths):
        sensor_frame = read_raw_sensor(path, preview_scale=preview_scale)
        transient_candidates = detect_hot_pixels(sensor_frame.raw_visible, sensor_frame.bayer_pattern)
        hot_pixel_detections.append(transient_candidates)
        debug_scale = max(1, preview_scale // 4)
        transient_overlay = overlay_points(
            sensor_frame.debug_preview,
            _hot_pixel_preview_points(transient_candidates, debug_scale),
            color=(255, 64, 64),
        )
        write_png(debug_dir / f"hot_pixels_transient_{index:02d}.png", transient_overlay)
        if index == reference_index:
            reference_sensor_preview_shape = (
                max(1, sensor_frame.raw_visible.shape[0] // preview_scale),
                max(1, sensor_frame.raw_visible.shape[1] // preview_scale),
            )
    hot_pixel_summary = build_persistent_hot_pixel_map(hot_pixel_detections)
    if reference_sensor_preview_shape is not None and hot_pixel_summary.persistent_mask.size > 0:
        write_png(
            debug_dir / "hot_pixels_persistent_preview.png",
            _hot_pixel_preview(hot_pixel_summary.persistent_mask, reference_sensor_preview_shape),
        )
    print(
        "[astrostacker] hot pixel map: "
        f"persistent={int(np.count_nonzero(hot_pixel_summary.persistent_mask))} "
        f"threshold={hot_pixel_summary.threshold_count}/{len(raw_paths)}"
    )
    for path, count in zip(raw_paths, hot_pixel_summary.total_detections, strict=True):
        print(f"[astrostacker] hot pixel scan {path.name}: transient_candidates={count}")
    _log_timing("hot pixel scan", hot_pixel_start)

    reference_start = perf_counter()
    reference_frame = read_raw_frame(
        raw_paths[reference_index],
        preview_scale=preview_scale,
        full_demosaic=True,
        persistent_hot_pixel_mask=hot_pixel_summary.persistent_mask,
    )
    if reference_frame.linear_rgb is None:
        raise RuntimeError(f"Full demosaic failed for {reference_frame.path}")
    _log_hot_pixels(
        f"frame {reference_frame.path.name} hot pixels",
        transient=reference_frame.transient_hot_pixel_count,
        persistent=reference_frame.persistent_hot_pixel_count,
        corrected=reference_frame.corrected_hot_pixel_count,
    )
    frames[reference_index] = FrameContext(
        path=reference_frame.path,
        rgb_preview=reference_frame.rgb_preview,
        full_shape=reference_frame.cfa_rgb.shape[:2],
        profile=reference_frame.profile,
    )
    reference_grid = build_radial_remap(
        reference_frame.linear_rgb.shape[1],
        reference_frame.linear_rgb.shape[0],
        reference_frame.profile,
    )
    reference_corrected = undistort_image(reference_frame.linear_rgb, reference_grid).astype(
        np.float32
    )
    reference_valid_mask = (
        undistort_valid_mask(reference_corrected.shape[:2], reference_grid) > 0.999
    )
    _log_timing("reference decode", reference_start)

    reference_segmentation_preview = cv2.resize(
        reference_frame.cfa_rgb,
        (
            max(1, reference_frame.cfa_rgb.shape[1] // segmentation_downsample),
            max(1, reference_frame.cfa_rgb.shape[0] // segmentation_downsample),
        ),
        interpolation=cv2.INTER_AREA,
    )
    segmentation = None
    segmentation_inputs = [raw_paths[reference_index]]
    if cache_dir is not None:
        segmentation = load_cached_segmentation(
            segmentation_cache_path(
                cache_dir,
                segmentation_inputs,
                segmentation_downsample=segmentation_downsample,
                dilation_radius=max(1, dilation_radius // segmentation_downsample),
                blur_radius=max(1, blur_radius // segmentation_downsample),
                sam3_checkpoint=sam3_checkpoint,
            )
        )
    if segmentation is None:
        segmentation_start = perf_counter()
        segmentation = segment_sky(
            [reference_segmentation_preview],
            dilation_radius=max(1, dilation_radius // segmentation_downsample),
            blur_radius=max(1, blur_radius // segmentation_downsample),
            sam3_checkpoint=sam3_checkpoint,
        )
        _log_timing("segmentation", segmentation_start)
        if cache_dir is not None:
            save_cached_segmentation(
                segmentation_cache_path(
                    cache_dir,
                    segmentation_inputs,
                    segmentation_downsample=segmentation_downsample,
                    dilation_radius=max(1, dilation_radius // segmentation_downsample),
                    blur_radius=max(1, blur_radius // segmentation_downsample),
                    sam3_checkpoint=sam3_checkpoint,
                ),
                segmentation,
            )
    else:
        print("[astrostacker] segmentation: cache hit")
    write_png(debug_dir / "sky_mask_preview.png", segmentation.sky_mask.astype(np.float32))
    write_png(
        debug_dir / "foreground_mask_preview.png", segmentation.foreground_mask.astype(np.float32)
    )
    write_png(debug_dir / "sky_weight_preview.png", segmentation.soft_sky_weight)

    reference_sky_mask = (
        cv2.resize(
            segmentation.sky_mask.astype(np.uint8),
            (reference_corrected.shape[1], reference_corrected.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        & reference_valid_mask
    )
    reference_sky_weight = _resize_mask(segmentation.soft_sky_weight, reference_corrected.shape[:2])
    reference_sky_weight = (
        np.clip(reference_sky_weight, 0.0, 1.0).astype(np.float32) * reference_valid_mask
    )
    reference_fg_weight = (1.0 - reference_sky_weight) * reference_valid_mask
    reference_stars = _detect_star_field(
        cache_dir=cache_dir,
        path=reference_frame.path,
        corrected=reference_corrected,
        sky_mask=reference_sky_mask,
    )
    star_fields[reference_index] = reference_stars
    _write_star_debug(debug_dir, reference_index, reference_corrected, reference_stars)

    reference_alignment = FrameAlignment(
        homography=np.eye(3, dtype=np.float32),
        match_source=reference_stars.points.astype(np.float32),
        match_target=reference_stars.points.astype(np.float32),
        aligned_match_source=reference_stars.points.astype(np.float32),
        inlier_mask=np.ones(len(reference_stars.points), dtype=bool),
        reprojection_error=0.0,
    )
    frame_alignments[reference_index] = reference_alignment
    _write_alignment_debug(
        debug_dir,
        reference_index,
        reference_corrected,
        reference_valid_mask.astype(np.float32),
        reference_stars,
        reference_alignment,
    )

    fusion_state = create_streaming_fusion(shape=reference_corrected.shape, trim=trim)
    fusion_state.add_sky(reference_corrected, reference_alignment.homography, reference_sky_weight)
    fusion_state.add_foreground(reference_corrected, reference_fg_weight)
    reference_phase = _phase_image(reference_corrected)

    streaming_start = perf_counter()
    for index, path in enumerate(raw_paths):
        if index == reference_index:
            continue

        frame_start = perf_counter()
        frame = read_raw_frame(
            path,
            preview_scale=preview_scale,
            full_demosaic=True,
            persistent_hot_pixel_mask=hot_pixel_summary.persistent_mask,
        )
        if frame.linear_rgb is None:
            raise RuntimeError(f"Full demosaic failed for {path}")
        _log_hot_pixels(
            f"frame {frame.path.name} hot pixels",
            transient=frame.transient_hot_pixel_count,
            persistent=frame.persistent_hot_pixel_count,
            corrected=frame.corrected_hot_pixel_count,
        )
        frames[index] = FrameContext(
            path=frame.path,
            rgb_preview=frame.rgb_preview,
            full_shape=frame.cfa_rgb.shape[:2],
            profile=frame.profile,
        )

        grid = build_radial_remap(
            frame.linear_rgb.shape[1],
            frame.linear_rgb.shape[0],
            frame.profile,
        )
        corrected = undistort_image(frame.linear_rgb, grid).astype(np.float32)
        valid_mask = undistort_valid_mask(corrected.shape[:2], grid) > 0.999
        sky_mask = (
            cv2.resize(
                segmentation.sky_mask.astype(np.uint8),
                (corrected.shape[1], corrected.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            & valid_mask
        )
        sky_weight = _resize_mask(segmentation.soft_sky_weight, corrected.shape[:2])
        sky_weight = np.clip(sky_weight, 0.0, 1.0).astype(np.float32) * valid_mask
        fg_weight = (1.0 - sky_weight) * valid_mask

        stars = _detect_star_field(
            cache_dir=cache_dir,
            path=frame.path,
            corrected=corrected,
            sky_mask=sky_mask,
        )
        star_fields[index] = stars
        _write_star_debug(debug_dir, index, corrected, stars)

        phase_image = _phase_image(corrected)
        window = cv2.createHanningWindow((phase_image.shape[1], phase_image.shape[0]), cv2.CV_32F)
        (shift_x, shift_y), _ = cv2.phaseCorrelate(reference_phase, phase_image, window=window)
        initial_homography = np.array(
            [[1.0, 0.0, shift_x], [0.0, 1.0, shift_y], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        frame_alignment = solve_frame_homography(reference_stars, stars, initial_homography)
        frame_alignments[index] = frame_alignment
        _write_alignment_debug(
            debug_dir, index, corrected, valid_mask.astype(np.float32), stars, frame_alignment
        )

        fusion_state.add_sky(corrected, frame_alignment.homography, sky_weight)
        fusion_state.add_foreground(corrected, fg_weight)
        _log_timing(f"frame {frame.path.name}", frame_start)
    _log_timing("streaming stage", streaming_start)

    if any(frame is None for frame in frames):
        raise RuntimeError("Failed to collect frame context for all frames.")
    if any(field is None for field in star_fields):
        raise RuntimeError("Failed to collect star fields for all frames.")
    if any(item is None for item in frame_alignments):
        raise RuntimeError("Failed to solve alignment for all frames.")

    alignment = _build_alignment_result(
        reference_index,
        [item for item in frame_alignments if item is not None],
    )

    finalize_start = perf_counter()
    fusion = fusion_state.finalize()
    _log_timing("fusion finalize", finalize_start)

    export_start = perf_counter()
    write_png(debug_dir / "stacked_preview.png", np.clip(fusion.fused ** (1.0 / 2.2), 0.0, 1.0))
    resolved_output_path = write_linear_tiff(
        output_path,
        np.clip(fusion.fused / max(np.percentile(fusion.fused, 99.8), 1e-6), 0.0, 1.0),
    )
    _log_timing("export", export_start)
    _log_timing("pipeline total", pipeline_start)
    return PipelineOutputs(
        frames=[frame for frame in frames if frame is not None],
        segmentation=segmentation,
        star_fields=[field for field in star_fields if field is not None],
        alignment=alignment,
        fusion=fusion,
        output_path=resolved_output_path,
    )
