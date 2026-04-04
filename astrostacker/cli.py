from __future__ import annotations

import argparse
from pathlib import Path

from astrostacker.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stack tripod astrophotography RAW sequences.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input RAW files.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TIFF path. If you pass a .dng path, astrostacker will write a .tiff next to it.",
    )
    parser.add_argument(
        "--debug-dir", type=Path, default=Path("debug"), help="Debug PNG output dir."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".astrostacker-cache"),
        help="Optional preprocessing cache dir for masks and star detections.",
    )
    parser.add_argument("--preview-scale", type=int, default=8, help="Preview downsample factor.")
    parser.add_argument(
        "--segmentation-scale",
        type=int,
        default=2,
        help="Extra downsample factor for segmentation previews.",
    )
    parser.add_argument(
        "--dilation-radius", type=int, default=21, help="Foreground dilation radius."
    )
    parser.add_argument("--blur-radius", type=int, default=9, help="Sky mask blur sigma.")
    parser.add_argument(
        "--trim", type=int, default=1, help="Outlier trim count for foreground fusion."
    )
    parser.add_argument(
        "--sam3-checkpoint",
        type=str,
        default=None,
        help="Optional local SAM3 checkpoint path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(
        raw_paths=args.inputs,
        output_path=args.output,
        debug_dir=args.debug_dir,
        cache_dir=args.cache_dir,
        preview_scale=args.preview_scale,
        segmentation_scale=args.segmentation_scale,
        dilation_radius=args.dilation_radius,
        blur_radius=args.blur_radius,
        trim=args.trim,
        sam3_checkpoint=args.sam3_checkpoint,
    )


if __name__ == "__main__":
    main()
