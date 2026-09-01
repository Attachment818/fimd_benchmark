from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_overlay(
    aligned_query: np.ndarray,
    reference: np.ndarray,
    reference_points: np.ndarray,
    predicted_points: np.ndarray,
    output_path: Path,
    alpha: float = 0.5,
    radius: int | None = None,
    thickness: int | None = None,
) -> None:
    if aligned_query.shape != reference.shape:
        raise ValueError(
            f"Overlay image shapes differ: {aligned_query.shape} vs {reference.shape}"
        )
    height, width = reference.shape[:2]
    radius = radius or max(5, round(min(height, width) / 300))
    thickness = thickness or max(2, round(radius / 3))
    overlay = cv2.addWeighted(reference, alpha, aligned_query, 1.0 - alpha, 0)
    for x, y in np.asarray(reference_points):
        cv2.circle(overlay, (round(x), round(y)), radius, (0, 255, 0), thickness)
    for x, y in np.asarray(predicted_points):
        cv2.circle(overlay, (round(x), round(y)), radius, (0, 0, 255), thickness)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise OSError(f"Could not write overlay: {output_path}")

