from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Audit the 70-pair FIMD dataset")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for number in tqdm(range(1, 71), desc="Auditing FIMD pairs"):
        pair_id = f"{number:02d}"
        pair_dir = args.data_root / f"{pair_id}_r_t"
        query_path = pair_dir / f"{pair_id}_t.jpg"
        reference_path = pair_dir / f"{pair_id}_r.jpg"
        gt_path = pair_dir / f"control_points_{pair_id}_r_t.txt"
        for path in (query_path, reference_path, gt_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        query = cv2.imread(str(query_path), cv2.IMREAD_UNCHANGED)
        reference = cv2.imread(str(reference_path), cv2.IMREAD_UNCHANGED)
        gt = np.loadtxt(gt_path, dtype=np.float64)
        if query is None or reference is None:
            raise ValueError(f"Unreadable image in pair {pair_id}")
        if gt.shape != (12, 4) or not np.isfinite(gt).all():
            raise ValueError(f"Invalid control points in {gt_path}: {gt.shape}")

        qh, qw = query.shape[:2]
        rh, rw = reference.shape[:2]
        ref_in_bounds = (
            (gt[:, 0] >= 0) & (gt[:, 0] < rw) &
            (gt[:, 1] >= 0) & (gt[:, 1] < rh)
        ).all()
        query_in_bounds = (
            (gt[:, 2] >= 0) & (gt[:, 2] < qw) &
            (gt[:, 3] >= 0) & (gt[:, 3] < qh)
        ).all()
        if not ref_in_bounds or not query_in_bounds:
            raise ValueError(f"Out-of-bounds control point in pair {pair_id}")

        rows.append({
            "pair_id": pair_id,
            "query_path": query_path.relative_to(args.data_root.parent.parent).as_posix(),
            "reference_path": reference_path.relative_to(args.data_root.parent.parent).as_posix(),
            "control_points_path": gt_path.relative_to(args.data_root.parent.parent).as_posix(),
            "query_width": qw,
            "query_height": qh,
            "reference_width": rw,
            "reference_height": rh,
            "reference_scale_x": f"{qw / rw:.12f}",
            "reference_scale_y": f"{qh / rh:.12f}",
            "control_point_count": len(gt),
            "query_sha256": sha256(query_path),
            "reference_sha256": sha256(reference_path),
            "control_points_sha256": sha256(gt_path),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(rows)} FIMD pairs; manifest: {args.output}")


if __name__ == "__main__":
    main()

