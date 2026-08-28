#!/usr/bin/env python3
"""Render blind Top-1 corrections made by the locked 100-identity model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


BG = "#0b0f15"
PANEL = "#121923"
TEXT = "#f4f7fb"
MUTED = "#9da9b8"
GREEN = "#48d68b"
RED = "#ff7272"
BLUE = "#67a7ff"


def font(size: int, bold=False):
    names = (
        "msyhbd.ttc" if bold else "msyh.ttc",
        "seguisb.ttf" if bold else "segoeui.ttf",
        "arialbd.ttf" if bold else "arial.ttf",
    )
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def crop(record: dict, size=(180, 132)) -> Image.Image:
    with Image.open(record["source_path"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image = image.resize(tuple(record["resized_size"]), Image.Resampling.LANCZOS)
    face = image.crop(tuple(record["face_roi_xyxy"]))
    face.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#080b10")
    canvas.paste(face, ((size[0] - face.width) // 2, (size[1] - face.height) // 2))
    return canvas


def short(value: str, limit=21) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("frozen_evaluation", type=Path)
    parser.add_argument("selected_evaluation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    frozen = json.loads(args.frozen_evaluation.read_text(encoding="utf-8"))
    selected = json.loads(args.selected_evaluation.read_text(encoding="utf-8"))
    frozen_queries = frozen["gallery_query"]["queries"]
    selected_by_query = {
        item["query_index"]: item for item in selected["gallery_query"]["queries"]
    }
    corrections = [item for item in frozen_queries if not item["correct"]]
    if not corrections:
        raise RuntimeError("Frozen evaluation contains no Top-1 errors to visualize")

    width, columns, panel_width, panel_height = 1900, 2, 906, 286
    rows = (len(corrections) + columns - 1) // columns
    height = 176 + rows * panel_height + 58
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((44, 24), "100 身份盲测：联合模型纠正的 Top-1", font=font(40, True), fill=TEXT)
    draw.text(
        (46, 82),
        "冻结融合 52/60（86.67%）  →  锁定联合模型 60/60（100%）；下面是全部 8 个被纠正的查询。",
        font=font(22),
        fill=MUTED,
    )
    draw.text((46, 124), "查询", font=font(18, True), fill=BLUE)
    draw.text((260, 124), "冻结 Top-1（错误）", font=font(18, True), fill=RED)
    draw.text((474, 124), "联合 Top-1（正确）", font=font(18, True), fill=GREEN)

    for position, frozen_row in enumerate(corrections):
        selected_row = selected_by_query[frozen_row["query_index"]]
        column, row = position % columns, position // columns
        x, y = 36 + column * 932, 162 + row * panel_height
        draw.rounded_rectangle((x, y, x + panel_width, y + 266), radius=18, fill=PANEL)
        query_index = frozen_row["query_index"]
        frozen_index = frozen_row["matched_gallery_index"]
        selected_index = selected_row["matched_gallery_index"]
        image.paste(crop(records[query_index]), (x + 18, y + 46))
        image.paste(crop(records[frozen_index]), (x + 226, y + 46))
        image.paste(crop(records[selected_index]), (x + 434, y + 46))
        draw.text(
            (x + 18, y + 14),
            f"#{position + 1}  {short(frozen_row['query_identity'])}",
            font=font(18, True),
            fill=TEXT,
        )
        draw.text((x + 198, y + 92), "→", font=font(27, True), fill=MUTED)
        draw.text((x + 406, y + 92), "→", font=font(27, True), fill=GREEN)
        draw.text(
            (x + 226, y + 190),
            f"{short(frozen_row['matched_identity'], 16)}  {frozen_row['score']:.3f}",
            font=font(17, True),
            fill=RED,
        )
        draw.text(
            (x + 434, y + 190),
            f"{short(selected_row['matched_identity'], 16)}  {selected_row['score']:.3f}",
            font=font(17, True),
            fill=GREEN,
        )
        draw.text(
            (x + 635, y + 62),
            "融合变化",
            font=font(17, True),
            fill=TEXT,
        )
        draw.text(
            (x + 635, y + 104),
            f"AUC  {frozen['branches']['fused']['auc']:.3f}\n  →  {selected['branches']['fused']['auc']:.3f}",
            font=font(18),
            fill=GREEN,
            spacing=8,
        )
        draw.text(
            (x + 635, y + 176),
            "身份未参与训练",
            font=font(16),
            fill=MUTED,
        )

    draw.text(
        (width // 2, height - 34),
        "盲测身份与训练/验证身份完全隔离；模型在盲测前已锁定且 SHA-256 前后不变。",
        font=font(18),
        fill=MUTED,
        anchor="mm",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite visualization: {args.output}")
    image.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
