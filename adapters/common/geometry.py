from __future__ import annotations

import cv2
import numpy as np


def transform_points_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    transformed = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        np.asarray(matrix, dtype=np.float64),
    )
    return transformed.reshape(-1, 2)


def point_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(f"Point shape mismatch: {predicted.shape} vs {target.shape}")
    return np.linalg.norm(predicted - target, axis=1)


def _quadratic_design(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    return np.column_stack((np.ones_like(x), x, y, x * y, x * x, y * y))


def fit_quadratic(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit source->target second-order polynomial coefficients (6 x 2)."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"Invalid quadratic point shapes: {source.shape}, {target.shape}")
    if len(source) < 6:
        raise ValueError("At least 6 correspondences are required for a quadratic transform")
    coefficients, _, rank, _ = np.linalg.lstsq(_quadratic_design(source), target, rcond=None)
    if rank < 6 or not np.isfinite(coefficients).all():
        raise ValueError(f"Degenerate quadratic fit (rank={rank})")
    return coefficients


def transform_points_quadratic(points: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (6, 2):
        raise ValueError(f"Expected 6x2 quadratic coefficients, got {coefficients.shape}")
    return _quadratic_design(points) @ coefficients


def warp_quadratic_inverse(
    source_image: np.ndarray,
    inverse_coefficients: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Warp source to target using target->source inverse coefficients."""
    width, height = output_size
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    target_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    source_points = transform_points_quadratic(target_points, inverse_coefficients)
    map_x = source_points[:, 0].reshape(height, width).astype(np.float32)
    map_y = source_points[:, 1].reshape(height, width).astype(np.float32)
    return cv2.remap(
        source_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
