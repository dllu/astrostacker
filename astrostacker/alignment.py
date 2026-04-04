from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import cKDTree

from astrostacker.stars import StarField


@dataclass(slots=True)
class AlignmentResult:
    homographies: list[np.ndarray]
    reference_index: int
    matched_pairs: list[int]
    inlier_counts: list[int]
    reprojection_errors: list[float]
    match_sources: list[np.ndarray]
    match_targets: list[np.ndarray]
    inlier_masks: list[np.ndarray]
    aligned_match_sources: list[np.ndarray]


@dataclass(slots=True)
class FrameAlignment:
    homography: np.ndarray
    match_source: np.ndarray
    match_target: np.ndarray
    aligned_match_source: np.ndarray
    inlier_mask: np.ndarray
    reprojection_error: float


def _transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    transformed = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography.astype(np.float32))
    return transformed.reshape(-1, 2)


def _match_star_fields(
    reference_points: np.ndarray,
    points: np.ndarray,
    initial_homography: np.ndarray,
    *,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(reference_points) < 4 or len(points) < 4:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    warped_points = _transform_points(points, initial_homography)
    ref_tree = cKDTree(reference_points)
    distances, ref_indices = ref_tree.query(warped_points, distance_upper_bound=max_distance)
    try:
        inverse_homography = np.linalg.inv(initial_homography.astype(np.float64)).astype(np.float32)
    except np.linalg.LinAlgError:
        inverse_homography = np.eye(3, dtype=np.float32)
    back_projected_refs = _transform_points(reference_points, inverse_homography)
    src_tree = cKDTree(points)
    _, reverse_src_indices = src_tree.query(back_projected_refs, distance_upper_bound=max_distance)

    candidates: list[tuple[float, int, int]] = []
    for point_index, (distance, ref_index) in enumerate(zip(distances, ref_indices, strict=True)):
        if not np.isfinite(distance) or ref_index >= len(reference_points):
            continue
        if reverse_src_indices[ref_index] != point_index:
            continue
        candidates.append((float(distance), point_index, int(ref_index)))
    candidates.sort()

    used_points: set[int] = set()
    used_refs: set[int] = set()
    src_matches: list[np.ndarray] = []
    dst_matches: list[np.ndarray] = []
    for _, point_index, ref_index in candidates:
        if point_index in used_points or ref_index in used_refs:
            continue
        used_points.add(point_index)
        used_refs.add(ref_index)
        src_matches.append(points[point_index])
        dst_matches.append(reference_points[ref_index])

    if not src_matches:
        return np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32)
    src = np.asarray(src_matches, dtype=np.float32)
    dst = np.asarray(dst_matches, dtype=np.float32)
    return src, dst


def _fit_homography(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    reproj_threshold: float = 10.0,
) -> tuple[np.ndarray | None, np.ndarray]:
    if len(src) < 4:
        return None, np.zeros(0, dtype=bool)
    homography, inliers = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=reproj_threshold,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None:
        return None, np.zeros(len(src), dtype=bool)
    if inliers is None:
        inlier_mask = np.zeros(len(src), dtype=bool)
    else:
        inlier_mask = inliers.ravel().astype(bool)
    if np.count_nonzero(inlier_mask) >= 4:
        refined, _ = cv2.findHomography(src[inlier_mask], dst[inlier_mask], method=0)
        if refined is not None:
            homography = refined
    return homography.astype(np.float32), inlier_mask


def _build_pair_index(points: np.ndarray) -> dict[int, list[tuple[int, int, float]]]:
    bins: dict[int, list[tuple[int, int, float]]] = {}
    for i in range(len(points)):
        vectors = points[i + 1 :] - points[i]
        distances = np.linalg.norm(vectors, axis=1)
        for offset, distance in enumerate(distances, start=i + 1):
            if distance < 32.0 or distance > 512.0:
                continue
            key = int(round(float(distance)))
            bins.setdefault(key, []).append((i, offset, float(distance)))
    return bins


def _rigid_homography_from_pairs(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    src_pair: tuple[int, int, float],
    dst_pair: tuple[int, int, float],
    *,
    reverse: bool,
) -> np.ndarray:
    src_i, src_j, _ = src_pair
    dst_i, dst_j, _ = dst_pair
    a0 = src_points[src_i]
    a1 = src_points[src_j]
    if reverse:
        b0 = dst_points[dst_j]
        b1 = dst_points[dst_i]
    else:
        b0 = dst_points[dst_i]
        b1 = dst_points[dst_j]
    src_vec = a1 - a0
    dst_vec = b1 - b0
    theta = float(np.arctan2(dst_vec[1], dst_vec[0]) - np.arctan2(src_vec[1], src_vec[0]))
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float32)
    translation = b0 - rotation @ a0
    return np.array(
        [
            [rotation[0, 0], rotation[0, 1], translation[0]],
            [rotation[1, 0], rotation[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _score_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    homography: np.ndarray,
    *,
    max_distance: float,
) -> tuple[int, float]:
    warped = _transform_points(src_points, homography)
    tree = cKDTree(dst_points)
    distances, _ = tree.query(warped, distance_upper_bound=max_distance)
    valid = np.isfinite(distances)
    if not np.any(valid):
        return 0, float("inf")
    return int(np.count_nonzero(valid)), float(np.median(distances[valid]))


def _estimate_coarse_homography(
    reference_field: StarField,
    field: StarField,
    initial_homography: np.ndarray,
) -> np.ndarray:
    reference_points = reference_field.points.astype(np.float32)
    points = field.points.astype(np.float32)
    best_homography = initial_homography.astype(np.float32)
    best_count, best_error = _score_homography(
        points,
        reference_points,
        best_homography,
        max_distance=10.0,
    )

    subset_size = min(len(points), len(reference_points), 128)
    if subset_size < 2:
        return best_homography
    src_subset = points[:subset_size]
    dst_subset = reference_points[:subset_size]
    ref_pairs_by_bin = _build_pair_index(dst_subset)
    src_pairs_by_bin = _build_pair_index(src_subset)
    if not ref_pairs_by_bin or not src_pairs_by_bin:
        return best_homography

    max_candidates = 4096
    tried = 0
    for key, src_pairs in sorted(
        src_pairs_by_bin.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        dst_pairs = [
            pair
            for neighbor_key in (key - 1, key, key + 1)
            for pair in ref_pairs_by_bin.get(neighbor_key, [])
        ]
        if not dst_pairs:
            continue
        for src_pair in src_pairs:
            for dst_pair in dst_pairs:
                for reverse in (False, True):
                    candidate = _rigid_homography_from_pairs(
                        src_subset, dst_subset, src_pair, dst_pair, reverse=reverse
                    )
                    count, error = _score_homography(
                        points,
                        reference_points,
                        candidate,
                        max_distance=10.0,
                    )
                    tried += 1
                    if count > best_count or (count == best_count and error < best_error):
                        best_homography = candidate
                        best_count = count
                        best_error = error
                    if tried >= max_candidates:
                        return best_homography
    return best_homography


def _weighted_homography_fit(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray
) -> np.ndarray | None:
    if len(src) < 4:
        return None
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(weights) & (weights > 1e-6)
    if np.count_nonzero(valid) < 4:
        return None
    src = np.asarray(src[valid], dtype=np.float64)
    dst = np.asarray(dst[valid], dtype=np.float64)
    sqrt_w = np.sqrt(weights[valid])
    a_rows: list[np.ndarray] = []
    b_rows: list[float] = []
    for (x, y), (u, v), ws in zip(src, dst, sqrt_w, strict=True):
        a_rows.append(
            np.array(
                [ws * x, ws * y, ws, 0.0, 0.0, 0.0, -ws * u * x, -ws * u * y], dtype=np.float64
            )
        )
        b_rows.append(ws * u)
        a_rows.append(
            np.array(
                [0.0, 0.0, 0.0, ws * x, ws * y, ws, -ws * v * x, -ws * v * y], dtype=np.float64
            )
        )
        b_rows.append(ws * v)
    a = np.vstack(a_rows)
    b = np.asarray(b_rows, dtype=np.float64)
    try:
        h, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    homography = np.array(
        [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(homography)):
        return None
    return homography.astype(np.float32)


def _refine_homography_icp(
    src_inliers: np.ndarray,
    dst_inliers: np.ndarray,
    initial_homography: np.ndarray,
    *,
    sigma: float = 5.0,
    max_distance: float = 5.0,
    iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(src_inliers) < 4 or len(dst_inliers) < 4:
        empty = np.zeros((0, 2), dtype=np.float32)
        return initial_homography.astype(np.float32), empty, empty, np.zeros(0, dtype=np.float32)
    tree_dst = cKDTree(dst_inliers)
    homography = initial_homography.astype(np.float32)
    src_corr = np.zeros((0, 2), dtype=np.float32)
    dst_corr = np.zeros((0, 2), dtype=np.float32)
    weights = np.zeros(0, dtype=np.float32)
    for _ in range(iterations):
        aligned = _transform_points(src_inliers, homography)
        distances, dst_indices = tree_dst.query(aligned, distance_upper_bound=max_distance)

        pair_indices: list[tuple[int, int]] = []
        pair_weights: list[float] = []
        used_dst: set[int] = set()
        order = np.argsort(distances)
        for point_index in order:
            distance = distances[point_index]
            dst_index = dst_indices[point_index]
            if not np.isfinite(distance) or dst_index >= len(dst_inliers):
                continue
            if dst_index in used_dst:
                continue
            weight = float(np.exp(-0.5 * np.square(distance / sigma)))
            if weight < 1e-3:
                continue
            used_dst.add(int(dst_index))
            pair_indices.append((int(point_index), int(dst_index)))
            pair_weights.append(weight)
        if len(pair_indices) < 4:
            break
        src_corr = src_inliers[[point_index for point_index, _ in pair_indices]].astype(np.float32)
        dst_corr = dst_inliers[[dst_index for _, dst_index in pair_indices]].astype(np.float32)
        weights = np.asarray(pair_weights, dtype=np.float32)
        refined = _weighted_homography_fit(src_corr, dst_corr, weights)
        if refined is None:
            break
        delta = np.max(np.abs(refined - homography))
        homography = refined
        if delta < 1e-4:
            break
    return homography.astype(np.float32), src_corr, dst_corr, weights


def _solve_single_homography(
    reference_field: StarField,
    field: StarField,
    initial_homography: np.ndarray,
) -> FrameAlignment:
    reference_points = reference_field.points
    points = field.points
    coarse_homography = _estimate_coarse_homography(reference_field, field, initial_homography)
    candidate_src, candidate_dst = _match_star_fields(
        reference_points, points, coarse_homography, max_distance=20.0
    )
    if len(candidate_src) < 4:
        return FrameAlignment(
            homography=coarse_homography.astype(np.float32),
            match_source=candidate_src,
            match_target=candidate_dst,
            aligned_match_source=np.zeros((0, 2), dtype=np.float32),
            inlier_mask=np.zeros(len(candidate_src), dtype=bool),
            reprojection_error=float("inf"),
        )
    homography, inlier_mask = _fit_homography(candidate_src, candidate_dst, reproj_threshold=5.0)
    if homography is None:
        return FrameAlignment(
            homography=coarse_homography.astype(np.float32),
            match_source=candidate_src,
            match_target=candidate_dst,
            aligned_match_source=np.zeros((0, 2), dtype=np.float32),
            inlier_mask=np.zeros(len(candidate_src), dtype=bool),
            reprojection_error=float("inf"),
        )

    ransac_src = candidate_src[inlier_mask]
    ransac_dst = candidate_dst[inlier_mask]
    aligned_src = _transform_points(ransac_src, homography)
    icp_homography, icp_src, icp_dst, icp_weights = _refine_homography_icp(
        ransac_src, ransac_dst, homography
    )
    if len(icp_src) >= 4:
        homography = icp_homography
        src = icp_src
        dst = icp_dst
        aligned_src = _transform_points(src, homography)
        inlier_mask = np.ones(len(src), dtype=bool)
    else:
        src = ransac_src
        dst = ransac_dst
        inlier_mask = np.ones(len(src), dtype=bool)

    errors = (
        np.linalg.norm(aligned_src - dst, axis=1) if len(src) > 0 else np.zeros(0, dtype=np.float32)
    )
    if len(errors) > 0 and np.any(inlier_mask):
        reprojection_error = float(np.median(errors[inlier_mask]))
    elif len(errors) > 0:
        reprojection_error = float(np.median(errors))
    else:
        reprojection_error = float("inf")
    return FrameAlignment(
        homography=homography.astype(np.float32),
        match_source=src,
        match_target=dst,
        aligned_match_source=aligned_src,
        inlier_mask=inlier_mask,
        reprojection_error=reprojection_error,
    )


def solve_frame_homography(
    reference_field: StarField,
    field: StarField,
    initial_homography: np.ndarray,
) -> FrameAlignment:
    return _solve_single_homography(reference_field, field, initial_homography)


def solve_homographies(
    star_fields: list[StarField],
    initial_homographies: list[np.ndarray] | None = None,
    reference_index: int | None = None,
) -> AlignmentResult:
    if not star_fields:
        raise ValueError("No star fields were provided.")
    if reference_index is None:
        reference_index = len(star_fields) // 2
    ref = star_fields[reference_index]
    homographies: list[np.ndarray] = []
    matched_pairs: list[int] = []
    inlier_counts: list[int] = []
    reprojection_errors: list[float] = []
    match_sources: list[np.ndarray] = []
    match_targets: list[np.ndarray] = []
    inlier_masks: list[np.ndarray] = []
    aligned_match_sources: list[np.ndarray] = []
    if initial_homographies is None:
        initial_homographies = [np.eye(3, dtype=np.float32) for _ in star_fields]
    for index, (field, initial_homography) in enumerate(
        zip(star_fields, initial_homographies, strict=True)
    ):
        if index == reference_index:
            homographies.append(np.eye(3, dtype=np.float32))
            matched_pairs.append(len(ref.points))
            inlier_counts.append(len(ref.points))
            reprojection_errors.append(0.0)
            match_sources.append(ref.points.astype(np.float32))
            match_targets.append(ref.points.astype(np.float32))
            inlier_masks.append(np.ones(len(ref.points), dtype=bool))
            aligned_match_sources.append(ref.points.astype(np.float32))
            continue
        frame_alignment = _solve_single_homography(
            ref,
            field,
            initial_homography,
        )
        homographies.append(frame_alignment.homography)
        matched_pairs.append(len(frame_alignment.match_source))
        inlier_counts.append(int(np.count_nonzero(frame_alignment.inlier_mask)))
        reprojection_errors.append(frame_alignment.reprojection_error)
        match_sources.append(frame_alignment.match_source)
        match_targets.append(frame_alignment.match_target)
        inlier_masks.append(frame_alignment.inlier_mask)
        aligned_match_sources.append(frame_alignment.aligned_match_source)
    return AlignmentResult(
        homographies=homographies,
        reference_index=reference_index,
        matched_pairs=matched_pairs,
        inlier_counts=inlier_counts,
        reprojection_errors=reprojection_errors,
        match_sources=match_sources,
        match_targets=match_targets,
        inlier_masks=inlier_masks,
        aligned_match_sources=aligned_match_sources,
    )
