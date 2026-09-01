from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from adapters.common.fimd_io import get_pair, load_pair_in_query_space
from adapters.common.geometry import (
    fit_quadratic,
    point_errors,
    transform_points_homography,
    transform_points_quadratic,
    warp_quadratic_inverse,
)
from adapters.common.visualization import create_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="SIFT FIMD smoke-test adapter")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pair-id", default="02")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ratio-threshold", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--nfeatures", type=int, default=5000)
    parser.add_argument("--contrast-threshold", type=float, default=0.01)
    parser.add_argument(
        "--preprocessing", choices=("gray", "green", "green_clahe"),
        default="green_clahe",
    )
    parser.add_argument(
        "--geometry", choices=("homography", "quadratic"), default="quadratic"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = get_pair(args.data_root, args.pair_id)
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_pair{pair.pair_id}_{args.preprocessing}_{args.geometry}_smoke"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    started = perf_counter()
    query, reference, query_gt, reference_gt, scale = load_pair_in_query_space(pair)
    preprocessing_seconds = perf_counter() - started

    if args.preprocessing == "gray":
        gray_query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    else:
        gray_query = query[:, :, 1]
        gray_reference = reference[:, :, 1]
        if args.preprocessing == "green_clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_query = clahe.apply(gray_query)
            gray_reference = clahe.apply(gray_reference)
    sift = cv2.SIFT_create(
        nfeatures=args.nfeatures,
        contrastThreshold=args.contrast_threshold,
    )

    matching_started = perf_counter()
    query_kp, query_desc = sift.detectAndCompute(gray_query, None)
    ref_kp, ref_desc = sift.detectAndCompute(gray_reference, None)
    if query_desc is None or ref_desc is None:
        raise RuntimeError("SIFT produced no descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(query_desc, ref_desc, k=2)
    matches = [m for m, n in pairs if m.distance < args.ratio_threshold * n.distance]
    matching_seconds = perf_counter() - matching_started
    if len(matches) < 4:
        raise RuntimeError(f"Only {len(matches)} ratio-test matches; need at least 4")

    query_points = np.float64([query_kp[m.queryIdx].pt for m in matches])
    reference_points = np.float64([ref_kp[m.trainIdx].pt for m in matches])
    geometry_started = perf_counter()
    matrix, inlier_mask = cv2.findHomography(
        query_points, reference_points, cv2.RANSAC, args.ransac_threshold
    )
    geometry_seconds = perf_counter() - geometry_started
    if matrix is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed")

    inliers = inlier_mask.reshape(-1).astype(bool)
    if args.geometry == "homography":
        aligned = cv2.warpPerspective(
            query, matrix, (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        predicted_gt = transform_points_homography(query_gt, matrix)
        transform_payload = {"homography": matrix.tolist()}
    else:
        if int(inliers.sum()) < 6:
            raise RuntimeError(
                f"Only {int(inliers.sum())} homography inliers; need 6 for quadratic"
            )
        forward_coefficients = fit_quadratic(
            query_points[inliers], reference_points[inliers]
        )
        inverse_coefficients = fit_quadratic(
            reference_points[inliers], query_points[inliers]
        )
        aligned = warp_quadratic_inverse(
            query,
            inverse_coefficients,
            (reference.shape[1], reference.shape[0]),
        )
        predicted_gt = transform_points_quadratic(query_gt, forward_coefficients)
        transform_payload = {
            "forward_query_to_reference": forward_coefficients.tolist(),
            "inverse_reference_to_query": inverse_coefficients.tolist(),
        }
    errors = point_errors(predicted_gt, reference_gt)

    aligned_dir = run_dir / "aligned_images"
    aligned_dir.mkdir()
    cv2.imwrite(str(aligned_dir / f"FIMD{pair.pair_id}_query_aligned.png"), aligned)
    create_overlay(
        aligned, reference, reference_gt, predicted_gt,
        aligned_dir / f"overlay_result_SIFT_FIMD{pair.pair_id}.png",
    )
    create_overlay(
        query, reference, reference_gt, query_gt,
        aligned_dir / f"target_and_source_FIMD{pair.pair_id}.png",
    )

    np.savez_compressed(
        run_dir / "matches.npz",
        query_points=query_points,
        reference_points=reference_points,
        inlier_mask=inlier_mask.reshape(-1).astype(bool),
    )
    (run_dir / "transform.json").write_text(
        json.dumps(transform_payload, indent=2), encoding="utf-8"
    )
    result = {
        "pair_id": pair.pair_id,
        "status": "success",
        "direction": "query_t_to_resized_reference_r",
        "geometry": args.geometry,
        "preprocessing": args.preprocessing,
        "sift_nfeatures": args.nfeatures,
        "sift_contrast_threshold": args.contrast_threshold,
        "reference_scale_x": scale[0],
        "reference_scale_y": scale[1],
        "query_keypoints": len(query_kp),
        "reference_keypoints": len(ref_kp),
        "ratio_test_matches": len(matches),
        "ransac_inliers": int(inlier_mask.sum()),
        "ransac_threshold_px": args.ransac_threshold,
        "control_point_errors_px": errors.tolist(),
        "mle_px": float(errors.mean()),
        "timing_seconds": {
            "preprocessing": preprocessing_seconds,
            "feature_and_matching": matching_seconds,
            "geometry": geometry_seconds,
            "end_to_end_with_writes": perf_counter() - started,
        },
    }
    (run_dir / "pair_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
