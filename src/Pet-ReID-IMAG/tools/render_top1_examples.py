#!/usr/bin/env python3
"""Render selected Top-1 retrieval examples from the locked multimodal model."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from pet_id.workspace_paths import (
    EVALUATIONS_ROOT,
    LEGACY_RUNS_ROOT,
    resolve_legacy_path,
)

from render_multimodal_showcase import (
    BG,
    BLUE,
    GREEN,
    MUTED,
    ORANGE,
    RED,
    TEXT,
    YELLOW,
    draw_centered,
    face_crop,
    font,
    load_json,
    panel,
    text_fit,
)


def render_examples(runs_root: Path, output: Path) -> None:
    comparison = load_json(runs_root / "dogfacenet_expanded20/comparison.json")
    manifest = load_json(runs_root / "dogfacenet_expanded20/manifest.json")
    records = manifest["records"]
    details = {row["identity"]: row for row in comparison["selected"]["details"]}
    examples = [
        ("wanda", "困难"),
        ("boris", "困难"),
        ("george", "困难"),
        ("charlie", "清晰"),
        ("gorda", "清晰"),
        ("71029-radar-energievoller-froehliche", "清晰"),
    ]

    image = Image.new("RGB", (1800, 1180), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 32), "Top-1 检索长什么样？", font=font(42, bold=True), fill=TEXT)
    draw.text(
        (50, 92),
        "左边是查询图；模型在图库中逐一比较，右边只展示相似度最高的那一张。",
        font=font(21),
        fill=MUTED,
    )

    for position, (identity, difficulty) in enumerate(examples):
        row = details[identity]
        grid_x, grid_y = position % 2, position // 2
        x, y = 38 + grid_x * 880, 142 + grid_y * 330
        panel(draw, (x, y, x + 844, y + 304), radius=18)
        name = text_fit(identity, 25)
        draw.text((x + 20, y + 14), name, font=font(21, bold=True), fill=TEXT)
        draw.text(
            (x + 824, y + 16),
            f"{difficulty} · Top-1 正确",
            font=font(17, bold=True),
            fill=GREEN,
            anchor="ra",
        )

        query_index = row["query_index"]
        top1_index = row["correct_index"] if row["correct"] else row["impostor_index"]
        image.paste(face_crop(records[query_index], (220, 180)), (x + 20, y + 58))
        image.paste(face_crop(records[top1_index], (220, 180)), (x + 300, y + 58))
        draw_centered(draw, (x + 245, y + 115, x + 295, y + 175), "→", font(34, bold=True), YELLOW)
        draw.text((x + 130, y + 246), "查询", font=font(16, bold=True), fill=BLUE, anchor="ma")
        draw.text((x + 410, y + 246), "图库 Top-1", font=font(16, bold=True), fill=GREEN, anchor="ma")

        value_x = x + 555
        draw.text((value_x, y + 72), f"Top-1：{name}", font=font(18, bold=True), fill=GREEN)
        draw.text((value_x, y + 112), f"余弦相似度  {row['correct_score']:.3f}", font=font(18), fill=TEXT)
        draw.text(
            (value_x, y + 154),
            f"第二名：{text_fit(row['impostor_identity'], 15)}",
            font=font(16),
            fill=MUTED,
        )
        draw.text((value_x, y + 191), f"领先余量  {row['margin']:+.3f}", font=font(19, bold=True), fill=GREEN)
        draw.text((value_x, y + 235), "✓ 身份一致", font=font(18, bold=True), fill=GREEN)

    draw.text(
        (42, 1140),
        "注意：Top-1 正确只表示第一名身份正确；余量越小，说明它越接近被第二名反超。",
        font=font(19),
        fill=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_wanda_before_after(runs_root: Path, output: Path) -> None:
    comparison = load_json(runs_root / "dogfacenet_expanded20/comparison.json")
    manifest = load_json(runs_root / "dogfacenet_expanded20/manifest.json")
    records = manifest["records"]
    frozen = next(row for row in comparison["frozen"]["details"] if row["identity"] == "wanda")
    selected = next(row for row in comparison["selected"]["details"] if row["identity"] == "wanda")

    image = Image.new("RGB", (1640, 720), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 32), "同一个查询，Top-1 可以怎样改变", font=font(40, bold=True), fill=TEXT)
    draw.text((50, 88), "Wanda 是扩展集合中唯一被联合模型纠正的 Top-1 案例。", font=font(21), fill=MUTED)
    panel(draw, (40, 138, 1600, 660), radius=22)

    items = [
        (selected["query_index"], "查询图", "Wanda", BLUE),
        (frozen["impostor_index"], "冻结模型 Top-1", "Boris · 错误", RED),
        (selected["correct_index"], "联合模型 Top-1", "Wanda · 正确", GREEN),
    ]
    starts = [85, 600, 1115]
    for x, (idx, title, label, color) in zip(starts, items):
        draw.text((x + 180, 170), title, font=font(22, bold=True), fill=color, anchor="ma")
        image.paste(face_crop(records[idx], (360, 320)), (x, 220))
        draw.text((x + 180, 560), label, font=font(21, bold=True), fill=color, anchor="ma")

    draw_centered(draw, (470, 330, 585, 430), "→", font(44, bold=True), ORANGE)
    draw_centered(draw, (985, 330, 1100, 430), "→", font(44, bold=True), GREEN)
    draw.text((780, 600), "冻结：Boris 0.533（错误）", font=font(20, bold=True), fill=RED, anchor="ma")
    draw.text((1330, 600), "联合：Wanda 0.549（正确）", font=font(20, bold=True), fill=GREEN, anchor="ma")
    draw.text((820, 680), "Top-1 不是分类概率，而是图库中余弦相似度最高的候选。", font=font(19), fill=MUTED, anchor="ma")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=LEGACY_RUNS_ROOT)
    parser.add_argument("--repo", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATIONS_ROOT / "multimodal-top1-examples",
    )
    args = parser.parse_args()
    runs_root = (
        resolve_legacy_path(args.repo / "logs")
        if args.repo is not None
        else resolve_legacy_path(args.runs_root)
    )
    output_dir = resolve_legacy_path(args.output_dir)
    render_examples(runs_root, output_dir / "pet-reid-top1-examples.png")
    render_wanda_before_after(runs_root, output_dir / "pet-reid-top1-wanda-before-after.png")
    print(output_dir / "pet-reid-top1-examples.png")
    print(output_dir / "pet-reid-top1-wanda-before-after.png")


if __name__ == "__main__":
    main()
