from __future__ import annotations

import argparse
from pathlib import Path

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
    return parser.parse_args()


def load_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def main() -> None:
    args = parse_args()
    if args.panel_width <= 0 or args.panel_height <= 0 or args.header_height <= 0:
        raise ValueError("Panel and header dimensions must be positive")
    project_root = args.project_root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = project_root / output_path

    font = load_font(args.font_size)
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
        # All FIMD02 originals are 3:2. Historical 3888x3888 exports were
        # vertically stretched; resizing them back to a 3:2 tile restores the
        # original retinal-image aspect ratio for qualitative presentation.
        image = image.resize(
            (args.panel_width, args.panel_height), resample=Image.Resampling.LANCZOS
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
