#!/usr/bin/env python3
"""Render representative corrections and remaining errors from 200-way retrieval."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


BG, PANEL = "#0b0f15", "#121923"
TEXT, MUTED = "#f4f7fb", "#9da9b8"
GREEN, RED, BLUE = "#48d68b", "#ff7272", "#67a7ff"


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


def short(value: str, limit=18) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("frozen_evaluation", type=Path)
    parser.add_argument("selected_evaluation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corrections", type=int, default=6)
    parser.add_argument("--errors", type=int, default=6)
    args = parser.parse_args()

    records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    by_identity = defaultdict(list)
    for record in records:
        by_identity[record["identity"].casefold()].append(record)
    frozen = json.loads(args.frozen_evaluation.read_text(encoding="utf-8"))
    selected = json.loads(args.selected_evaluation.read_text(encoding="utf-8"))
    frozen_rows = frozen["branches"]["fused"]["queries"]
    selected_rows = selected["branches"]["fused"]["queries"]
    selected_by_query = {row["query_index"]: row for row in selected_rows}
    corrections = [
        row for row in frozen_rows
        if not row["correct"] and selected_by_query[row["query_index"]]["correct"]
    ]
    corrections.sort(key=lambda row: selected_by_query[row["query_index"]]["top5"][0]["score"] - row["top5"][0]["score"], reverse=True)
    remaining = sorted(
        (row for row in selected_rows if not row["correct"]),
        key=lambda row: row["true_identity_rank"],
        reverse=True,
    )
    shown = [
        ("CORRECTED", row, selected_by_query[row["query_index"]])
        for row in corrections[: args.corrections]
    ] + [("REMAINING ERROR", row, row) for row in remaining[: args.errors]]

    width, columns, panel_width, panel_height = 1900, 2, 906, 286
    rows_count = (len(shown) + columns - 1) // columns
    height = 176 + rows_count * panel_height + 58
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((44, 24), "800→200 严格盲测：Top-1 案例", font=font(40, True), fill=TEXT)
    draw.text(
        (46, 82),
        "冻结融合 336/400（84.0%） → 联合模型 380/400（95.0%）；展示 6 个纠错和 6 个剩余难例。",
        font=font(22), fill=MUTED,
    )
    draw.text((46, 124), "查询", font=font(18, True), fill=BLUE)
    draw.text((260, 124), "旧/预测身份", font=font(18, True), fill=RED)
    draw.text((474, 124), "正确身份图库", font=font(18, True), fill=GREEN)

    for position, (kind, left_row, selected_row) in enumerate(shown):
        column, row_index = position % columns, position // columns
        x, y = 36 + column * 932, 162 + row_index * panel_height
        color = GREEN if kind == "CORRECTED" else RED
        draw.rounded_rectangle((x, y, x + panel_width, y + 266), radius=18, fill=PANEL, outline=color, width=2)
        query_record = records[left_row["query_index"]]
        predicted_identity = left_row["top5"][0]["identity"]
        true_identity = left_row["query_identity"]
        sheet.paste(crop(query_record), (x + 18, y + 46))
        sheet.paste(crop(by_identity[predicted_identity][0]), (x + 226, y + 46))
        sheet.paste(crop(by_identity[true_identity][0]), (x + 434, y + 46))
        draw.text((x + 18, y + 14), f"{kind}  {short(true_identity)}", font=font(17, True), fill=color)
        draw.text((x + 226, y + 190), f"{short(predicted_identity)}  {left_row['top5'][0]['score']:.3f}", font=font(16, True), fill=RED)
        draw.text((x + 434, y + 190), short(true_identity), font=font(16, True), fill=GREEN)
        if kind == "CORRECTED":
            draw.text((x + 635, y + 70), "训练后", font=font(17, True), fill=TEXT)
            draw.text((x + 635, y + 112), f"Top-1 正确\nscore {selected_row['top5'][0]['score']:.3f}", font=font(18), fill=GREEN, spacing=8)
        else:
            draw.text((x + 635, y + 70), "正确身份排名", font=font(17, True), fill=TEXT)
            draw.text((x + 635, y + 112), f"Rank {left_row['true_identity_rank']}\nTop-5 {'是' if left_row['true_identity_rank'] <= 5 else '否'}", font=font(18), fill=RED, spacing=8)

    draw.text(
        (width // 2, height - 34),
        "200 个测试身份均未参与训练；每个查询同时比较全部 200 个身份原型。",
        font=font(18), fill=MUTED, anchor="mm",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite visualization: {args.output}")
    sheet.save(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
