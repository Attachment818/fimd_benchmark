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
import torchvision.models as tv_models

from adapters.common.fimd_io import get_pair, load_pair_in_query_space
from adapters.common.geometry import (
    fit_quadratic,
    point_errors,
    transform_points_quadratic,
    warp_quadratic_inverse,
)
from adapters.common.visualization import create_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Official NCNet FIMD adapter")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pair-id", default="02")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, default=Path("third_party/NCNet"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-variant", choices=("pfpascal", "ivd"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=400)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    return parser.parse_args()


def preprocess(image: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return ((tensor - mean) / std).unsqueeze(0).to(device)


def load_official_model(official_root: Path, checkpoint: Path, device: str):
    if not official_root.is_dir():
        raise FileNotFoundError(official_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    official_path = str(official_root.resolve())
    if official_path not in sys.path:
        sys.path.insert(0, official_path)

    # The official checkpoint contains the complete ResNet feature extractor.
    # Prevent torchvision's legacy pretrained=True constructor from downloading
    # a second, unrelated copy before those checkpoint weights are restored.
    original_resnet101 = tv_models.resnet101

    def resnet101_without_download(*args, **kwargs):
        kwargs.pop("pretrained", None)
        kwargs["weights"] = None
        return original_resnet101(*args, **kwargs)

    tv_models.resnet101 = resnet101_without_download
    try:
        from lib.model import ImMatchNet

        model = ImMatchNet(use_cuda=device.startswith("cuda"), checkpoint=str(checkpoint))
    finally:
        tv_models.resnet101 = original_resnet101
    return model.eval()


def extract_bidirectional_matches(correlation: torch.Tensor):
    from lib.point_tnf import corr_to_matches

    forward = corr_to_matches(
        correlation, scale="positive", do_softmax=True,
        invert_matching_direction=False,
    )
    reverse = corr_to_matches(
        correlation, scale="positive", do_softmax=True,
        invert_matching_direction=True,
    )
    values = [torch.cat((a, b), dim=1).squeeze(0) for a, b in zip(forward, reverse)]
    x_query, y_query, x_reference, y_reference, scores = values

    _, _, height_query, width_query, height_reference, width_reference = correlation.shape
    x_query = x_query * (width_query - 1) / width_query + 0.5 / width_query
    y_query = y_query * (height_query - 1) / height_query + 0.5 / height_query
    x_reference = (
        x_reference * (width_reference - 1) / width_reference + 0.5 / width_reference
    )
    y_reference = (
        y_reference * (height_reference - 1) / height_reference + 0.5 / height_reference
    )

    stacked = torch.stack((x_query, y_query, x_reference, y_reference), dim=1)
    order = torch.argsort(scores, descending=True)
    stacked = stacked[order].detach().cpu().numpy()
    scores = scores[order].detach().cpu().numpy()
    _, unique_indices = np.unique(stacked, axis=0, return_index=True)
    unique_indices.sort()
    return stacked[unique_indices], scores[unique_indices], list(correlation.shape[2:])


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.image_size <= 0:
        raise ValueError("NCNet image size must be positive")
    if args.device != "cuda:0" and args.device.startswith("cuda"):
        raise ValueError(
            "Official NCNet uses unqualified .cuda() calls; select the physical GPU "
            "with CUDA_VISIBLE_DEVICES and pass --device cuda:0"
        )
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        torch.cuda.set_device(0)

    pair = get_pair(args.data_root, args.pair_id)
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        + f"_pair{pair.pair_id}_ncnet_{args.checkpoint_variant}_quadratic_smoke"
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    total_started = perf_counter()
    query, reference, query_gt, reference_gt, reference_scale = load_pair_in_query_space(pair)
    query_tensor = preprocess(query, args.image_size, args.device)
    reference_tensor = preprocess(reference, args.image_size, args.device)
    preprocessing_seconds = perf_counter() - total_started

    load_started = perf_counter()
    model = load_official_model(args.official_root, args.checkpoint, args.device)
    synchronize(args.device)
    model_load_seconds = perf_counter() - load_started

    inference_started = perf_counter()
    with torch.inference_mode():
        correlation = model({
            "source_image": query_tensor,
            "target_image": reference_tensor,
        })
    synchronize(args.device)
    inference_seconds = perf_counter() - inference_started

    matching_started = perf_counter()
    normalized_matches, scores, correlation_shape = extract_bidirectional_matches(correlation)
    query_matches = normalized_matches[:, :2] * np.asarray([query.shape[1], query.shape[0]])
    reference_matches = normalized_matches[:, 2:] * np.asarray([
        reference.shape[1], reference.shape[0]
    ])
    matching_seconds = perf_counter() - matching_started
    if len(query_matches) < 6:
        raise RuntimeError(f"Only {len(query_matches)} NCNet matches; need at least 6")

    geometry_started = perf_counter()
    homography, inlier_mask = cv2.findHomography(
        query_matches, reference_matches, cv2.RANSAC, args.ransac_threshold
    )
    if homography is None or inlier_mask is None:
        raise RuntimeError("NCNet homography prefilter failed")
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
        aligned_dir / f"overlay_result_NCNet_{args.checkpoint_variant}_FIMD{pair.pair_id}.png",
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
        "checkpoint_variant": args.checkpoint_variant,
        "preprocessing": "official_imagenet_rgb_normalization",
        "model_input_size": [args.image_size, args.image_size],
        "correlation_shape": correlation_shape,
        "reference_scale_x": reference_scale[0],
        "reference_scale_y": reference_scale[1],
        "bidirectional_unique_matches": int(len(query_matches)),
        "ransac_inliers": int(inliers.sum()),
        "control_point_errors_px": errors.tolist(),
        "mle_px": float(errors.mean()),
        "timing_seconds": {
            "preprocessing": preprocessing_seconds,
            "model_load": model_load_seconds,
            "inference": inference_seconds,
            "matching": matching_seconds,
            "geometry_and_warp": geometry_seconds,
            "end_to_end_with_load_and_writes": perf_counter() - total_started,
        },
        "versions": {
            "pytorch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "torchvision": __import__("torchvision").__version__,
            "opencv": cv2.__version__,
        },
        "compatibility": {
            "skip_redundant_torchvision_resnet_download": True,
            "official_core_modified": False,
        },
    }
    (run_dir / "pair_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
