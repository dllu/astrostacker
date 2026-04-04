from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from astrostacker.debug import ensure_dir, overlay_points, write_png
from astrostacker.hotpixels import detect_hot_pixels
from astrostacker.rawio import RawSensorFrame, read_raw_sensor
from astrostacker.segmentation import segment_sky


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0.5


def _preview_points(mask: np.ndarray, scale: int) -> np.ndarray:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.column_stack(
        [
            coords[:, 1].astype(np.float32) / float(scale),
            coords[:, 0].astype(np.float32) / float(scale),
        ]
    ).astype(np.float32)


def _detect(
    sensor_frame: RawSensorFrame,
    *,
    kernel_size: int,
    threshold_sigma: float,
    min_excess: float,
    min_value: float,
    max_bright_support: int,
) -> np.ndarray:
    return detect_hot_pixels(
        sensor_frame.raw_visible,
        sensor_frame.bayer_pattern,
        kernel_size=kernel_size,
        threshold_sigma=threshold_sigma,
        min_excess=min_excess,
        min_value=min_value,
        max_bright_support=max_bright_support,
    )


def _fitness(
    candidate_mask: np.ndarray,
    baseline_mask: np.ndarray,
    foreground_mask: np.ndarray,
    sky_mask: np.ndarray,
) -> tuple[float, float, float, int, int]:
    baseline_fg = int(np.count_nonzero(baseline_mask & foreground_mask))
    candidate_fg = int(np.count_nonzero(candidate_mask & foreground_mask))
    candidate_sky = int(np.count_nonzero(candidate_mask & sky_mask))
    foreground_area = int(np.count_nonzero(foreground_mask))
    sky_area = int(np.count_nonzero(sky_mask))

    term1 = candidate_fg / max(baseline_fg, 1)
    fg_density = candidate_fg / max(foreground_area, 1)
    sky_density = candidate_sky / max(sky_area, 1)
    term2 = fg_density / max(sky_density, 1e-9)
    fitness = term1 * term2
    return fitness, term1, term2, candidate_fg, candidate_sky


def _parse_list(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune hot-pixel detection on a single RAW frame.")
    parser.add_argument("input", type=Path, help="Input RAW file.")
    parser.add_argument("--debug-dir", type=Path, default=Path("/tmp/astrostacker-hotpixel-tune"))
    parser.add_argument("--preview-scale", type=int, default=24)
    parser.add_argument("--segmentation-downsample", type=int, default=4)
    parser.add_argument("--dilation-radius", type=int, default=21)
    parser.add_argument("--blur-radius", type=int, default=9)
    parser.add_argument("--sam3-checkpoint", type=str, default=None)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--threshold-sigma", type=float, default=14.0)
    parser.add_argument("--min-excess", type=float, default=0.04)
    parser.add_argument("--min-value", type=float, default=0.02)
    parser.add_argument("--max-bright-support", type=int, default=1)
    parser.add_argument("--sweep", action="store_true", help="Run a parameter sweep in-process.")
    parser.add_argument("--sweep-threshold-sigma", type=str, default="8,10,12,14")
    parser.add_argument("--sweep-min-excess", type=str, default="0.02,0.03,0.04")
    parser.add_argument("--sweep-max-bright-support", type=str, default="1,2")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.debug_dir)

    sensor_frame = read_raw_sensor(args.input, preview_scale=args.preview_scale)
    segmentation_preview = cv2.resize(
        sensor_frame.cfa_rgb,
        (
            max(1, sensor_frame.cfa_rgb.shape[1] // args.segmentation_downsample),
            max(1, sensor_frame.cfa_rgb.shape[0] // args.segmentation_downsample),
        ),
        interpolation=cv2.INTER_AREA,
    )
    segmentation = segment_sky(
        [segmentation_preview],
        dilation_radius=max(1, args.dilation_radius // args.segmentation_downsample),
        blur_radius=max(1, args.blur_radius // args.segmentation_downsample),
        sam3_checkpoint=args.sam3_checkpoint,
    )
    foreground_mask = _resize_mask(segmentation.foreground_mask, sensor_frame.raw_visible.shape)
    sky_mask = _resize_mask(segmentation.sky_mask, sensor_frame.raw_visible.shape)

    baseline_mask = _detect(
        sensor_frame,
        kernel_size=args.kernel_size,
        threshold_sigma=8.0,
        min_excess=0.02,
        min_value=args.min_value,
        max_bright_support=2,
    )
    candidate_mask = _detect(
        sensor_frame,
        kernel_size=args.kernel_size,
        threshold_sigma=args.threshold_sigma,
        min_excess=args.min_excess,
        min_value=args.min_value,
        max_bright_support=args.max_bright_support,
    )

    baseline_fg = int(np.count_nonzero(baseline_mask & foreground_mask))
    foreground_area = int(np.count_nonzero(foreground_mask))
    sky_area = int(np.count_nonzero(sky_mask))

    fitness, term1, term2, candidate_fg, candidate_sky = _fitness(
        candidate_mask, baseline_mask, foreground_mask, sky_mask
    )

    debug_scale = max(1, args.preview_scale // 4)
    baseline_overlay = overlay_points(
        sensor_frame.debug_preview,
        _preview_points(baseline_mask, debug_scale),
        color=(255, 192, 0),
    )
    candidate_overlay = overlay_points(
        sensor_frame.debug_preview,
        _preview_points(candidate_mask, debug_scale),
        color=(255, 64, 64),
    )
    write_png(args.debug_dir / "baseline_hot_pixels.png", baseline_overlay)
    write_png(args.debug_dir / "candidate_hot_pixels.png", candidate_overlay)
    write_png(args.debug_dir / "foreground_mask.png", segmentation.foreground_mask.astype(np.float32))
    write_png(args.debug_dir / "sky_mask.png", segmentation.sky_mask.astype(np.float32))

    if args.sweep:
        sweep_threshold_sigma = _parse_list(args.sweep_threshold_sigma, float)
        sweep_min_excess = _parse_list(args.sweep_min_excess, float)
        sweep_max_bright_support = _parse_list(args.sweep_max_bright_support, int)
        results = []
        for threshold_sigma in sweep_threshold_sigma:
            for min_excess in sweep_min_excess:
                for max_bright_support in sweep_max_bright_support:
                    sweep_mask = _detect(
                        sensor_frame,
                        kernel_size=args.kernel_size,
                        threshold_sigma=threshold_sigma,
                        min_excess=min_excess,
                        min_value=args.min_value,
                        max_bright_support=max_bright_support,
                    )
                    sweep_fitness, sweep_term1, sweep_term2, sweep_fg, sweep_sky = _fitness(
                        sweep_mask, baseline_mask, foreground_mask, sky_mask
                    )
                    results.append(
                        (
                            sweep_fitness,
                            sweep_term1,
                            sweep_term2,
                            threshold_sigma,
                            min_excess,
                            max_bright_support,
                            sweep_fg,
                            sweep_sky,
                            sweep_mask,
                        )
                    )
        results.sort(key=lambda item: item[0], reverse=True)
        print("sweep_results_top10:")
        for item in results[:10]:
            print(
                "  "
                f"fitness={item[0]:.6f} "
                f"term1={item[1]:.6f} "
                f"term2={item[2]:.6f} "
                f"threshold_sigma={item[3]:.2f} "
                f"min_excess={item[4]:.3f} "
                f"max_bright_support={item[5]} "
                f"foreground={item[6]} "
                f"sky={item[7]}"
            )
        best = results[0]
        best_overlay = overlay_points(
            sensor_frame.debug_preview,
            _preview_points(best[8], debug_scale),
            color=(0, 255, 255),
        )
        write_png(args.debug_dir / "best_sweep_hot_pixels.png", best_overlay)

    print(f"input={args.input}")
    print(f"segmentation_backend={segmentation.backend}")
    print(
        "candidate_params="
        f"kernel_size={args.kernel_size} "
        f"threshold_sigma={args.threshold_sigma} "
        f"min_excess={args.min_excess} "
        f"min_value={args.min_value} "
        f"max_bright_support={args.max_bright_support}"
    )
    print(f"baseline_foreground_hot_pixels={baseline_fg}")
    print(f"candidate_foreground_hot_pixels={candidate_fg}")
    print(f"candidate_sky_hot_pixels={candidate_sky}")
    print(f"foreground_area={foreground_area}")
    print(f"sky_area={sky_area}")
    print(f"term1_fg_recall={term1:.6f}")
    print(f"term2_density_ratio={term2:.6f}")
    print(f"fitness={fitness:.6f}")
    print(f"debug_dir={args.debug_dir}")


if __name__ == "__main__":
    main()
