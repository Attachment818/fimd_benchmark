from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FimdPair:
    pair_id: str
    query_path: Path
    reference_path: Path
    control_points_path: Path


def get_pair(data_root: Path, pair_id: str) -> FimdPair:
    pair_id = str(pair_id).zfill(2)
    pair_dir = data_root / f"{pair_id}_r_t"
    pair = FimdPair(
        pair_id=pair_id,
        query_path=pair_dir / f"{pair_id}_t.jpg",
        reference_path=pair_dir / f"{pair_id}_r.jpg",
        control_points_path=pair_dir / f"control_points_{pair_id}_r_t.txt",
    )
    missing = [str(path) for path in (
        pair.query_path, pair.reference_path, pair.control_points_path
    ) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"FIMD pair {pair_id} is incomplete: {missing}")
    return pair


def load_pair_in_query_space(pair: FimdPair):
    query = cv2.imread(str(pair.query_path), cv2.IMREAD_COLOR)
    reference_original = cv2.imread(str(pair.reference_path), cv2.IMREAD_COLOR)
    if query is None or reference_original is None:
        raise ValueError(f"Could not read images for FIMD pair {pair.pair_id}")

    points = np.loadtxt(pair.control_points_path, dtype=np.float64)
    if points.shape != (12, 4):
        raise ValueError(
            f"Expected 12x4 control points for {pair.pair_id}, got {points.shape}"
        )
    reference_gt = points[:, :2].copy()
    query_gt = points[:, 2:4].copy()

    query_h, query_w = query.shape[:2]
    ref_h, ref_w = reference_original.shape[:2]
    scale_x = query_w / ref_w
    scale_y = query_h / ref_h
    reference_gt[:, 0] *= scale_x
    reference_gt[:, 1] *= scale_y
    reference = cv2.resize(
        reference_original, (query_w, query_h), interpolation=cv2.INTER_LINEAR
    )
    return query, reference, query_gt, reference_gt, (scale_x, scale_y)

