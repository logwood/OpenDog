#!/usr/bin/env python3
"""Render high/low-scoring Phase B pairs for qualitative inspection only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    fitted = ImageOps.contain(image, size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", size, "#eeeeee")
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    panel.paste(fitted, offset)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("upstream/Pet-ReID-IMAG"))
    parser.add_argument(
        "--scores", default="logs/fusion_submit/submit_modern.csv"
    )
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("results/phase_b_pair_examples.png")
    )
    args = parser.parse_args()
    if args.per_group < 1:
        raise ValueError("--per-group must be positive")

    repo = args.repo.resolve()
    frame = pd.read_csv(repo / args.scores)
    required = {"imageA", "imageB", "prediction"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Score CSV must contain {sorted(required)}")
    selected = pd.concat(
        [
            frame.nlargest(args.per_group, "prediction").assign(group="highest"),
            frame.nsmallest(args.per_group, "prediction").assign(group="lowest"),
        ],
        ignore_index=True,
    )

    with (repo / "data/test/filename_map.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        local_by_original = {
            row["original_name"]: row["local_name"] for row in csv.DictReader(handle)
        }
    image_root = repo / "data/test/test"

    margin = 24
    label_width = 190
    image_size = (300, 220)
    row_height = image_size[1] + 54
    title_height = 82
    width = margin * 3 + label_width + image_size[0] * 2
    height = title_height + margin + row_height * len(selected)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25)
    text_font = load_font(18)
    small_font = load_font(14)
    draw.text(
        (margin, 16),
        "Phase B qualitative pair examples",
        font=title_font,
        fill="#111111",
    )
    draw.text(
        (margin, 50),
        "No hidden labels: high/low cosine scores are observations, not correctness claims.",
        font=small_font,
        fill="#9b2c2c",
    )

    for index, row in selected.iterrows():
        top = title_height + margin + index * row_height
        color = "#1f7a4d" if row["group"] == "highest" else "#b33a3a"
        draw.rounded_rectangle(
            (margin, top, margin + label_width - 12, top + image_size[1]),
            radius=12,
            fill="#f6f6f6",
            outline=color,
            width=3,
        )
        draw.text((margin + 14, top + 24), row["group"].upper(), font=text_font, fill=color)
        draw.text(
            (margin + 14, top + 62),
            f"score {float(row['prediction']):.3f}",
            font=text_font,
            fill="#222222",
        )
        draw.text((margin + 14, top + 108), "imageA", font=small_font, fill="#555555")
        draw.text((margin + 100, top + 108), "imageB", font=small_font, fill="#555555")

        image_a = image_root / local_by_original[row["imageA"]]
        image_b = image_root / local_by_original[row["imageB"]]
        x_a = margin + label_width
        x_b = x_a + image_size[0] + margin
        canvas.paste(fit_image(image_a, image_size), (x_a, top))
        canvas.paste(fit_image(image_b, image_size), (x_b, top))
        draw.text((x_a, top + image_size[1] + 5), "A", font=small_font, fill="#444444")
        draw.text((x_b, top + image_size[1] + 5), "B", font=small_font, fill="#444444")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    print(
        {
            "output": str(output),
            "pairs": len(selected),
            "highest": selected[selected.group == "highest"].prediction.tolist(),
            "lowest": selected[selected.group == "lowest"].prediction.tolist(),
        }
    )


if __name__ == "__main__":
    main()
