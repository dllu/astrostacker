from __future__ import annotations

from pathlib import Path

import numpy as np

from astrostacker.alignment import solve_homographies, _transform_points
from astrostacker.cache import load_cached_star_field
from astrostacker.debug import draw_square_points, overlay_vectors, write_png


def transform(points: np.ndarray, homography: np.ndarray):
    n = points.shape[0]
    points_homo = np.hstack((points, np.ones((n, 1))))
    transformed = points_homo @ homography.T
    return transformed[:, :2] / transformed[:, 2:]


def main() -> None:
    moving = load_cached_star_field(Path(".astrostacker-cache/stars/9bfc368271572d1485276c5d.npz"))
    static = load_cached_star_field(Path(".astrostacker-cache/stars/48290314eee954ed7b8f2d0f.npz"))
    if moving is None or static is None:
        raise RuntimeError("cached stars missing")

    result = solve_homographies([moving, static], reference_index=1)
    residual_starts = result.aligned_match_sources[0][result.inlier_masks[0]]
    residual_ends = result.match_targets[0][result.inlier_masks[0]]
    motion_starts = result.match_sources[0][result.inlier_masks[0]]
    motion_ends = residual_starts

    print("matched_pairs", result.matched_pairs)
    print("inlier_counts", result.inlier_counts)
    print("reprojection_errors", result.reprojection_errors)
    print("H0")
    print(result.homographies[0])

    canvas = np.zeros((static.enhanced.shape[0], static.enhanced.shape[1], 3), dtype=np.uint8)
    canvas = draw_square_points(canvas, static.points, color=(255, 0, 0), size=3)
    canvas = draw_square_points(canvas, moving.points, color=(0, 255, 0), size=3)
    canvas = draw_square_points(
        canvas, _transform_points(moving.points, result.homographies[0]), color=(0, 0, 255), size=3
    )
    canvas = overlay_vectors(canvas, motion_starts, motion_ends, color=(0, 255, 255))
    # canvas = overlay_vectors(canvas, residual_starts, residual_ends, color=(255, 255, 255))

    out_path = Path("/tmp/astrostacker-rerun/first_last_icp_debug_rerun2.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_png(out_path, canvas)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
