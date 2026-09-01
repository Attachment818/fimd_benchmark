from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from adapters.common.fimd_io import get_pair, load_pair_in_query_space
from adapters.common.geometry import (
    fit_quadratic,
    point_errors,
    transform_points_quadratic,
    warp_quadratic_inverse,
)
from adapters.common.visualization import create_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Official SuperPoint FIMD adapter")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pair-id", default="02")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--official-script", type=Path,
        default=Path("third_party/SuperPoint/demo_superpoint.py"),
    )
    parser.add_argument(
        "--weights", type=Path,
        default=Path("third_party/SuperPoint/superpoint_v1.pth"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-width", type=int, default=1024)
    parser.add_argument("--model-height", type=int, default=1024)
    parser.add_argument("--nms-distance", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.015)
    parser.add_argument("--nn-threshold", type=float, default=0.7)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument(
        "--preprocessing", choices=("gray", "green", "green_clahe"),
        default="green",
    )
    return parser.parse_args()


def load_official_module(script_path: Path):
    if not script_path.is_file():
        raise FileNotFoundError(script_path)
    spec = importlib.util.spec_from_file_location("official_superpoint", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official SuperPoint script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess(image: np.ndarray, mode: str, width: int, height: int) -> np.ndarray:
    if mode == "gray":
        channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        channel = image[:, :, 1]
        if mode == "green_clahe":
            channel = cv2.createCLAHE(2.0, (8, 8)).apply(channel)
    resized = cv2.resize(channel, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.model_width % 8 or args.model_height % 8:
        raise ValueError("SuperPoint model width and height must be divisible by 8")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    use_cuda = args.device.startswith("cuda")
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(torch.device(args.device))

    pair = get_pair(args.data_root, args.pair_id)
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_pair{pair.pair_id}_superpoint_quadratic_smoke"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    total_started = perf_counter()
    query, reference, query_gt, reference_gt, scale = load_pair_in_query_space(pair)
    query_input = preprocess(
        query, args.preprocessing, args.model_width, args.model_height
    )
    reference_input = preprocess(
        reference, args.preprocessing, args.model_width, args.model_height
    )
    preprocessing_seconds = perf_counter() - total_started

    official = load_official_module(args.official_script)
    load_started = perf_counter()
    frontend = official.SuperPointFrontend(
        weights_path=str(args.weights),
        nms_dist=args.nms_distance,
        conf_thresh=args.confidence_threshold,
        nn_thresh=args.nn_threshold,
        cuda=use_cuda,
    )
    synchronize(args.device)
    model_load_seconds = perf_counter() - load_started

    inference_started = perf_counter()
    with torch.inference_mode():
        query_points_model, query_desc, _ = frontend.run(query_input)
        reference_points_model, reference_desc, _ = frontend.run(reference_input)
    synchronize(args.device)
    inference_seconds = perf_counter() - inference_started
    if query_desc is None or reference_desc is None:
        raise RuntimeError("SuperPoint produced no descriptors")

    matching_started = perf_counter()
    tracker = official.PointTracker(max_length=2, nn_thresh=args.nn_threshold)
    matches = tracker.nn_match_two_way(
        query_desc, reference_desc, args.nn_threshold
    )
    if matches.shape[1] < 6:
        raise RuntimeError(f"Only {matches.shape[1]} mutual matches; need at least 6")
    query_matches = query_points_model[:2, matches[0].astype(int)].T.astype(np.float64)
    reference_matches = reference_points_model[:2, matches[1].astype(int)].T.astype(np.float64)

    query_h, query_w = query.shape[:2]
    query_matches[:, 0] *= query_w / args.model_width
    query_matches[:, 1] *= query_h / args.model_height
    reference_matches[:, 0] *= reference.shape[1] / args.model_width
    reference_matches[:, 1] *= reference.shape[0] / args.model_height
    matching_seconds = perf_counter() - matching_started

    geometry_started = perf_counter()
    homography, inlier_mask = cv2.findHomography(
        query_matches,
        reference_matches,
        cv2.RANSAC,
        args.ransac_threshold,
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("SuperPoint homography prefilter failed")
    inliers = inlier_mask.reshape(-1).astype(bool)
    if int(inliers.sum()) < 6:
        raise RuntimeError(f"Only {int(inliers.sum())} RANSAC inliers; need 6")
    forward = fit_quadratic(query_matches[inliers], reference_matches[inliers])
    inverse = fit_quadratic(reference_matches[inliers], query_matches[inliers])
    predicted_gt = transform_points_quadratic(query_gt, forward)
    errors = point_errors(predicted_gt, reference_gt)
    aligned = warp_quadratic_inverse(
        query, inverse, (reference.shape[1], reference.shape[0])
    )
    geometry_seconds = perf_counter() - geometry_started

    aligned_dir = run_dir / "aligned_images"
    aligned_dir.mkdir()
    cv2.imwrite(str(aligned_dir / f"FIMD{pair.pair_id}_query_aligned.png"), aligned)
    create_overlay(
        aligned,
        reference,
        reference_gt,
        predicted_gt,
        aligned_dir / f"overlay_result_SuperPoint_FIMD{pair.pair_id}.png",
    )
    np.savez_compressed(
        run_dir / "matches.npz",
        query_points=query_matches,
        reference_points=reference_matches,
        match_scores=matches[2],
        inlier_mask=inliers,
    )
    (run_dir / "transform.json").write_text(
        json.dumps({
            "forward_query_to_reference": forward.tolist(),
            "inverse_reference_to_query": inverse.tolist(),
        }, indent=2),
        encoding="utf-8",
    )
    result = {
        "pair_id": pair.pair_id,
        "status": "success",
        "direction": "query_t_to_resized_reference_r",
        "geometry": "quadratic",
        "device": args.device,
        "preprocessing": args.preprocessing,
        "model_input_size": [args.model_width, args.model_height],
        "reference_scale_x": scale[0],
        "reference_scale_y": scale[1],
        "query_keypoints": int(query_points_model.shape[1]),
        "reference_keypoints": int(reference_points_model.shape[1]),
        "mutual_matches": int(matches.shape[1]),
        "ransac_inliers": int(inliers.sum()),
        "control_point_errors_px": errors.tolist(),
        "mle_px": float(errors.mean()),
        "timing_seconds": {
            "preprocessing": preprocessing_seconds,
            "model_load": model_load_seconds,
            "two_image_inference": inference_seconds,
            "matching": matching_seconds,
            "geometry_and_warp": geometry_seconds,
            "end_to_end_with_load_and_writes": perf_counter() - total_started,
        },
        "versions": {
            "python_torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "opencv": cv2.__version__,
        },
    }
    (run_dir / "pair_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

