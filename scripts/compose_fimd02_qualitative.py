from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANELS = (
    ("(a) Target and Source", "supplementary_experiments/fimd/target_and_source_FIMD02.jpg"),
    ("(b) SIFT", "paper_figures/FIMD02/SIFT/overlay_result_SIFT_FIMD02.png"),
    ("(c) NCNet", "paper_figures/FIMD02/NCNet/overlay_result_NCNet_FIMD02.png"),
    ("(d) SuperPoint", "paper_figures/FIMD02/SuperPoint/overlay_result_SuperPoint_FIMD02.png"),
    ("(e) GeoFormer", "paper_figures/FIMD02/GeoFormer/overlay_result_GeoFormer_FIMD02.png"),
    ("(f) SuperRetina", "supplementary_experiments/fimd/overlay_result_SuperRetina_FIMD02.jpg"),
    ("(g) RetinaRegNet", "supplementary_experiments/fimd/overlay_result_RRN_FIMD02.jpg"),
    ("(h) Ours", "supplementary_experiments/fimd/overlay_result_Ours_FIMD02.jpg"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compose the FIMD02 4x2 paper figure")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_figures/FIMD02/FIMD02_qualitative_4x2.png"),
    )
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument("--panel-height", type=int, default=600)
    parser.add_argument("--header-height", type=int, default=46)
    parser.add_argument("--font-size", type=int, default=25)
    parser.add_argument("--marker-radius", type=int, default=7)
    parser.add_argument("--marker-width", type=int, default=2)
    return parser.parse_args()


def load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def dilate(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant")
        result = np.logical_or.reduce([
            padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
            for dy in range(3) for dx in range(3)
        ])
    return result


def component_centers(mask: np.ndarray, minimum_pixels: int = 30):
    height, width = mask.shape
    # Red and green rings overlap in well-registered cases, so one color can
    # split the other into several arcs. Join nearby arcs before labeling.
    join_iterations = max(2, round(min(height, width) / 500))
    connected_mask = dilate(mask, iterations=join_iterations)
    remaining = {y * width + x for y, x in np.argwhere(connected_mask)}
    centers = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        component = [seed]
        while stack:
            value = stack.pop()
            y, x = divmod(value, width)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    ny, nx = y + dy, x + dx
                    neighbor = ny * width + nx
                    if 0 <= ny < height and 0 <= nx < width and neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
        if len(component) >= minimum_pixels:
            ys, xs = zip(*(divmod(value, width) for value in component))
            centers.append((len(component), float(np.mean(xs)), float(np.mean(ys))))
    centers.sort(reverse=True)
    return [(x, y) for _, x, y in centers[:12]]


def remove_markers_and_find_red_centers(image: Image.Image, source_name: str):
    pixels = np.asarray(image.convert("RGB"))
    red = (
        (pixels[:, :, 0] > 220)
        & (pixels[:, :, 1] < 70)
        & (pixels[:, :, 2] < 70)
    )
    green = (
        (pixels[:, :, 1] > 180)
        & (pixels[:, :, 0] < 120)
        & (pixels[:, :, 2] < 120)
    )
    red_centers = component_centers(red)
    if len(red_centers) < 12:
        raise ValueError(
            f"{source_name}: could not find 12 red control-point rings; got "
            f"{len(red_centers)}"
        )

    marker_mask = dilate(red | green, iterations=3)
    cleaned = pixels.astype(np.float32).copy()
    known = ~marker_mask
    remaining = marker_mask.copy()
    while remaining.any():
        padded_values = np.pad(cleaned, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded_known = np.pad(known, 1, mode="constant")
        sums = np.zeros_like(cleaned)
        counts = np.zeros(marker_mask.shape, dtype=np.float32)
        for dy in range(3):
            for dx in range(3):
                if dx == 1 and dy == 1:
                    continue
                neighbor_known = padded_known[dy:dy + known.shape[0], dx:dx + known.shape[1]]
                sums += padded_values[dy:dy + known.shape[0], dx:dx + known.shape[1]] * neighbor_known[:, :, None]
                counts += neighbor_known
        fillable = remaining & (counts > 0)
        if not fillable.any():
            raise RuntimeError("Could not repair marker pixels")
        cleaned[fillable] = sums[fillable] / counts[fillable, None]
        known[fillable] = True
        remaining[fillable] = False
    return Image.fromarray(np.clip(cleaned, 0, 255).astype(np.uint8)), red_centers


def main() -> None:
    args = parse_args()
    if args.panel_width <= 0 or args.panel_height <= 0 or args.header_height <= 0:
        raise ValueError("Panel and header dimensions must be positive")
    project_root = args.project_root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path

    font = load_font(args.font_size)
    pair_root = project_root / "data/FIMD/02_r_t"
    control_points = np.loadtxt(pair_root / "control_points_02_r_t.txt")
    reference_width, reference_height = Image.open(pair_root / "02_r.jpg").size
    query_width, query_height = Image.open(pair_root / "02_t.jpg").size
    reference_points_normalized = control_points[:, :2] / np.asarray([
        reference_width, reference_height
    ])
    query_points_normalized = control_points[:, 2:] / np.asarray([
        query_width, query_height
    ])
    tile_height = args.header_height + args.panel_height
    canvas = Image.new(
        "RGB", (4 * args.panel_width, 2 * tile_height), color="white"
    )
    draw = ImageDraw.Draw(canvas)

    for index, (label, relative_path) in enumerate(PANELS):
        input_path = project_root / relative_path
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        image = Image.open(input_path).convert("RGB")
        original_width, original_height = image.size
        image, red_centers = remove_markers_and_find_red_centers(
            image, relative_path
        )
        green_centers = reference_points_normalized * np.asarray([
            original_width, original_height
        ])
        if index == 0:
            red_centers = query_points_normalized * np.asarray([
                original_width, original_height
            ])
        # All FIMD02 originals are 3:2. Historical 3888x3888 exports were
        # vertically stretched; resizing them back to a 3:2 tile restores the
        # original retinal-image aspect ratio for qualitative presentation.
        image = image.resize(
            (args.panel_width, args.panel_height), resample=Image.Resampling.LANCZOS
        )
        marker_draw = ImageDraw.Draw(image)
        scale_x = args.panel_width / original_width
        scale_y = args.panel_height / original_height
        for centers, color in ((green_centers, "#00ff00"), (red_centers, "#ff0000")):
            for center_x, center_y in centers:
                x_center = center_x * scale_x
                y_center = center_y * scale_y
                radius = args.marker_radius
                marker_draw.ellipse(
                    (
                        x_center - radius,
                        y_center - radius,
                        x_center + radius,
                        y_center + radius,
                    ),
                    outline=color,
                    width=args.marker_width,
                )
        column = index % 4
        row = index // 4
        x = column * args.panel_width
        y = row * tile_height
        canvas.paste(image, (x, y + args.header_height))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        draw.text(
            (x + (args.panel_width - text_width) / 2, y + (args.header_height - text_height) / 2 - box[1]),
            label,
            fill="black",
            font=font,
        )
        if column:
            draw.line((x, y, x, y + tile_height), fill=(220, 220, 220), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)
    print(f"Wrote {output_path} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
