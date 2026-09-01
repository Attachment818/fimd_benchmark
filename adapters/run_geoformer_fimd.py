from __future__ import annotations

import argparse
import json
import sys
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
    parser = argparse.ArgumentParser(description="Official GeoFormer FIMD adapter")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pair-id", default="02")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--official-root", type=Path, default=Path("third_party/GeoFormer")
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("third_party/GeoFormer/saved_ckpt/geoformer.ckpt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--match-threshold", type=float, default=0.2)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    return parser.parse_args()


def resize_gray(image: np.ndarray, image_size: int):
    """Match the official loader: cap the shorter side and use /8 dimensions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = image_size / min(width, height) if min(width, height) > image_size else 1.0
    resized_width = max(8, int(round(width * scale)) // 8 * 8)
    resized_height = max(8, int(round(height * scale)) // 8 * 8)
    resized = cv2.resize(gray, (resized_width, resized_height))
    tensor = torch.from_numpy(resized).float()[None, None] / 255.0
    return tensor, (width / resized_width, height / resized_height)


def load_official_model(official_root: Path, weights: Path, threshold: float, device: str):
    if not official_root.is_dir():
        raise FileNotFoundError(official_root)
    if not weights.is_file():
        raise FileNotFoundError(weights)
    official_path = str(official_root.resolve())
    if official_path not in sys.path:
        sys.path.insert(0, official_path)

    from model.full_model import GeoFormer
    from model.geo_config import default_cfg as geoformer_cfg
    from model.loftr_src.loftr.utils.cvpr_ds_config import default_cfg

    configuration = dict(default_cfg)
    configuration["match_coarse"] = dict(configuration["match_coarse"])
    configuration["match_coarse"]["thr"] = threshold
    geoformer_cfg["coarse_thr"] = threshold
    model = GeoFormer(configuration)
    checkpoint = torch.load(weights, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatibility = model.load_state_dict(state_dict, strict=False)
    model = model.eval().to(device)
    return model, list(incompatibility.missing_keys), list(incompatibility.unexpected_keys)


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 8:
        raise ValueError("GeoFormer image size must be positive and divisible by 8")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(torch.device(args.device))

    pair = get_pair(args.data_root, args.pair_id)
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_pair{pair.pair_id}_geoformer_quadratic_smoke"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    total_started = perf_counter()
    query, reference, query_gt, reference_gt, reference_scale = load_pair_in_query_space(pair)
    query_tensor, query_model_scale = resize_gray(query, args.image_size)
    reference_tensor, reference_model_scale = resize_gray(reference, args.image_size)
    query_tensor = query_tensor.to(args.device)
    reference_tensor = reference_tensor.to(args.device)
    preprocessing_seconds = perf_counter() - total_started

    load_started = perf_counter()
    model, missing_keys, unexpected_keys = load_official_model(
        args.official_root, args.weights, args.match_threshold, args.device
    )
    synchronize(args.device)
    model_load_seconds = perf_counter() - load_started

    inference_started = perf_counter()
    batch = {"image0": query_tensor, "image1": reference_tensor}
    with torch.inference_mode():
        batch = model(batch)
    synchronize(args.device)
    inference_seconds = perf_counter() - inference_started
    query_matches = batch["mkpts0_f"].detach().cpu().numpy().astype(np.float64)
    reference_matches = batch["mkpts1_f"].detach().cpu().numpy().astype(np.float64)
    scores = batch["mconf"].detach().cpu().numpy()
    if len(query_matches) < 6:
        raise RuntimeError(f"Only {len(query_matches)} GeoFormer matches; need at least 6")
    query_matches *= np.asarray(query_model_scale)
    reference_matches *= np.asarray(reference_model_scale)

    geometry_started = perf_counter()
    homography, inlier_mask = cv2.findHomography(
        query_matches, reference_matches, cv2.RANSAC, args.ransac_threshold
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("GeoFormer homography prefilter failed")
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
        aligned_dir / f"overlay_result_GeoFormer_FIMD{pair.pair_id}.png",
    )
    np.savez_compressed(
        run_dir / "matches.npz",
        query_points=query_matches,
        reference_points=reference_matches,
        match_scores=scores,
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
        "preprocessing": "official_grayscale",
        "model_short_side": args.image_size,
        "match_threshold": args.match_threshold,
        "reference_scale_x": reference_scale[0],
        "reference_scale_y": reference_scale[1],
        "matches": int(len(query_matches)),
        "ransac_inliers": int(inliers.sum()),
        "control_point_errors_px": errors.tolist(),
        "mle_px": float(errors.mean()),
        "checkpoint_loading": {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
        },
        "timing_seconds": {
            "preprocessing": preprocessing_seconds,
            "model_load": model_load_seconds,
            "inference": inference_seconds,
            "geometry_and_warp": geometry_seconds,
            "end_to_end_with_load_and_writes": perf_counter() - total_started,
        },
        "versions": {
            "pytorch": torch.__version__,
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
