#!/usr/bin/env python3
"""Render presentation-ready summary boards for the multimodal pet-ID experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from pet_id.workspace_paths import (
    EVALUATIONS_ROOT,
    LEGACY_RUNS_ROOT,
    resolve_legacy_path,
)


BG = "#090d14"
PANEL = "#111923"
PANEL_2 = "#172231"
TEXT = "#f5f7fa"
MUTED = "#aab6c5"
BLUE = "#53a7ff"
GREEN = "#49d17d"
ORANGE = "#ff9b54"
RED = "#ff6b6b"
YELLOW = "#ffd166"
BORDER = "#2a394b"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "msyhbd.ttc" if bold else "msyh.ttc"
    return ImageFont.truetype(str(Path(r"C:\Windows\Fonts") / name), size=size)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def panel(draw: ImageDraw.ImageDraw, box, *, fill=PANEL, radius=22) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=BORDER, width=2)


def text_fit(value: str, maximum: int) -> str:
    return value if len(value) <= maximum else value[: maximum - 1] + "…"


def face_crop(record: dict, size: tuple[int, int], *, show_nose=True) -> Image.Image:
    with Image.open(resolve_legacy_path(record["source_path"])) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image = image.resize(tuple(record["resized_size"]), Image.Resampling.LANCZOS)
    fx1, fy1, fx2, fy2 = (int(round(v)) for v in record["face_roi_xyxy"])
    nx1, ny1, nx2, ny2 = (int(round(v)) for v in record["nose_roi_xyxy"])
    face = image.crop((fx1, fy1, fx2, fy2))
    if show_nose:
        crop_draw = ImageDraw.Draw(face)
        crop_draw.rectangle(
            (nx1 - fx1, ny1 - fy1, nx2 - fx1, ny2 - fy1),
            outline=YELLOW,
            width=max(3, face.width // 90),
        )
    face.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#0c121a")
    canvas.paste(face, ((size[0] - face.width) // 2, (size[1] - face.height) // 2))
    return canvas


def draw_centered(draw, box, value, face, fill=TEXT) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), value, font=face)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        value,
        font=face,
        fill=fill,
    )


def comparison_bar(draw, x, y, width, label, base, selected, *, maximum=1.0) -> None:
    draw.text((x, y), label, font=font(19, bold=True), fill=TEXT)
    draw.text((x + width, y), f"{base:.4f} → {selected:.4f}", font=font(17), fill=MUTED, anchor="ra")
    y += 34
    draw.rounded_rectangle((x, y, x + width, y + 16), radius=8, fill="#263344")
    base_width = max(2, width * base / maximum)
    selected_width = max(2, width * selected / maximum)
    draw.rounded_rectangle((x, y, x + base_width, y + 7), radius=4, fill=ORANGE)
    draw.rounded_rectangle((x, y + 9, x + selected_width, y + 16), radius=4, fill=GREEN)


def render_overview(runs_root: Path, output: Path) -> None:
    protocol = load_json(runs_root / "dogfacenet_protocol32/protocol_completion.json")
    validation = load_json(runs_root / "dogfacenet_protocol32/validation_selection.json")
    blind = load_json(runs_root / "dogfacenet_protocol32/blind_test_result.json")
    expanded = load_json(runs_root / "dogfacenet_expanded20/comparison.json")
    expanded_manifest = load_json(runs_root / "dogfacenet_expanded20/manifest.json")
    records = expanded_manifest["records"]
    detail = {row["identity"]: row for row in expanded["selected"]["details"]}
    wanda = detail["wanda"]

    image = Image.new("RGB", (1800, 1240), BG)
    draw = ImageDraw.Draw(image)
    draw.text((54, 38), "犬只身份识别：鼻纹 × ArcFace × AnyFace", font=font(46, bold=True), fill=TEXT)
    draw.text((56, 102), "局部端到端联合系统 · 阶段成果总览", font=font(24), fill=MUTED)

    # System flow.
    panel(draw, (44, 154, 1756, 352))
    steps = [
        ("输入图像", "高清犬脸"),
        ("AnyFace", "脸框 + 5关键点"),
        ("SAM 2", "鼻部 mask / ROI"),
        ("双分支编码", "IMAG鼻纹 + dog.pt脸部"),
        ("视角感知联合颈", "残差适配 + 门控"),
        ("身份检索", "余弦相似度 + gallery"),
    ]
    start_x, box_w, gap = 72, 248, 27
    for index, (title, subtitle) in enumerate(steps):
        x = start_x + index * (box_w + gap)
        draw.rounded_rectangle((x, 202, x + box_w, 310), radius=16, fill=PANEL_2)
        draw_centered(draw, (x, 213, x + box_w, 255), title, font(22, bold=True), BLUE if index < 3 else GREEN)
        draw_centered(draw, (x + 8, 260, x + box_w - 8, 301), subtitle, font(16), MUTED)
        if index < len(steps) - 1:
            draw.line((x + box_w + 7, 256, x + box_w + gap - 7, 256), fill=YELLOW, width=4)
            draw.polygon(
                ((x + box_w + gap - 9, 249), (x + box_w + gap - 9, 263), (x + box_w + gap - 1, 256)),
                fill=YELLOW,
            )

    # Protocol and model selection.
    panel(draw, (44, 382, 560, 730))
    draw.text((72, 410), "严格数据协议", font=font(28, bold=True), fill=TEXT)
    rows = [
        ("训练", "20 身份 · 40张进入优化器"),
        ("验证", "6 身份 · 18张 · 仅选模型"),
        ("盲测", "6 身份 · 18张 · 锁定后打开"),
        ("扩展评估", "20 身份 · 60张 · 展示补充"),
    ]
    for index, (name, value) in enumerate(rows):
        y = 466 + index * 55
        draw.text((74, y), name, font=font(19, bold=True), fill=BLUE)
        draw.text((182, y), value, font=font(18), fill=TEXT)
    draw.line((72, 690, 532, 690), fill=BORDER, width=2)
    draw.text((74, 700), "身份 / 路径 / SHA‑256 重合均为 0", font=font(18, bold=True), fill=GREEN)

    panel(draw, (590, 382, 1110, 730))
    draw.text((618, 410), "验证集选择：联合残差必须克制", font=font(28, bold=True), fill=TEXT)
    candidates = {row["name"]: row for row in validation["candidates_ranked"]}
    chosen = validation["selected"]
    comparison_bar(draw, 620, 475, 450, "AUC（冻结 → 选中）", candidates["frozen"]["auc"], chosen["auc"])
    comparison_bar(
        draw,
        620,
        560,
        450,
        "最小查询领先分差",
        candidates["frozen"]["min_margin"],
        chosen["min_margin"],
        maximum=0.10,
    )
    draw.text((620, 652), "锁定：step 60 · joint mix 0.25%", font=font(22, bold=True), fill=YELLOW)
    draw.text((620, 688), "10% 无额外收益；20% 明显退化", font=font(17), fill=MUTED)

    panel(draw, (1140, 382, 1756, 730))
    draw.text((1168, 410), "跨集合结果", font=font(28, bold=True), fill=TEXT)
    result_rows = [
        ("验证 6身份", candidates["frozen"]["auc"], chosen["auc"], "AUC"),
        ("盲测 6身份", blind["frozen"]["auc"], blind["selected"]["auc"], "AUC"),
        ("扩展 20身份", expanded["frozen"]["auc"], expanded["selected"]["auc"], "AUC"),
    ]
    for index, (label, base, selected, metric) in enumerate(result_rows):
        y = 474 + index * 70
        delta = selected - base
        color = GREEN if delta > 0 else RED
        draw.text((1168, y), label, font=font(19, bold=True), fill=TEXT)
        draw.text((1345, y), f"{base:.4f} → {selected:.4f}", font=font(18), fill=MUTED)
        draw.text((1692, y), f"{delta:+.4f}", font=font(18, bold=True), fill=color, anchor="ra")
    draw.text((1168, 682), "扩展 Top‑1：19/20 → 20/20", font=font(22, bold=True), fill=GREEN)

    # Corrected Wanda case.
    panel(draw, (44, 760, 1756, 1176))
    draw.text((72, 786), "代表性成果：扩展评估中纠正 Wanda 的错误匹配", font=font(30, bold=True), fill=TEXT)
    indices = [wanda["query_index"], wanda["correct_index"], wanda["impostor_index"]]
    labels = ["查询：Wanda", "正确图库：Wanda", "原最佳错误：Boris"]
    colors = [BLUE, GREEN, RED]
    for col, (idx, label, color) in enumerate(zip(indices, labels, colors)):
        x = 78 + col * 350
        image.paste(face_crop(records[idx], (310, 230)), (x, 850))
        draw.text((x, 1093), label, font=font(18, bold=True), fill=color)
    draw.text((1150, 864), "冻结基线", font=font(22, bold=True), fill=ORANGE)
    draw.text((1150, 908), "正确 0.513  <  错误 0.533", font=font(24), fill=RED)
    draw.text((1150, 960), "领先分差  −0.019 · 识别失败", font=font(22, bold=True), fill=RED)
    draw.text((1150, 1025), "锁定模型", font=font(22, bold=True), fill=GREEN)
    draw.text((1150, 1069), "正确 0.549  >  错误 0.509", font=font(24), fill=TEXT)
    draw.text((1150, 1121), "领先分差  +0.040 · 识别纠正", font=font(22, bold=True), fill=GREEN)

    draw.text(
        (50, 1200),
        "结论：新联合颈部改善了新身份检索稳定性，但 Peanut 外观突变仍是明确失败模式；AUC 与 Top‑1 必须同时报告。",
        font=font(18),
        fill=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_query_board(runs_root: Path, output: Path) -> None:
    comparison = load_json(runs_root / "dogfacenet_expanded20/comparison.json")
    manifest = load_json(runs_root / "dogfacenet_expanded20/manifest.json")
    records = manifest["records"]
    baseline = {row["identity"]: row for row in comparison["frozen"]["details"]}
    selected = comparison["selected"]["details"]
    width, height = 2000, 1720
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.text((44, 30), "补充评估：20个全新身份 · 20个查询结果", font=font(40, bold=True), fill=TEXT)
    draw.text((46, 88), "每格：查询 / 正确图库 / 最强错误图库；该集合未参与训练、验证或盲测模型选择", font=font(20), fill=MUTED)
    cols, rows = 4, 5
    tile_w, tile_h = 480, 300
    for position, row in enumerate(selected):
        grid_x, grid_y = position % cols, position // cols
        x, y = 30 + grid_x * 490, 135 + grid_y * 310
        panel(draw, (x, y, x + tile_w, y + tile_h), radius=16)
        was_correct = baseline[row["identity"]]["correct"]
        status_color = GREEN if row["correct"] else RED
        title = text_fit(row["identity"], 24)
        draw.text((x + 16, y + 12), title, font=font(18, bold=True), fill=TEXT)
        if not was_correct and row["correct"]:
            draw.text((x + tile_w - 16, y + 13), "已纠正", font=font(16, bold=True), fill=YELLOW, anchor="ra")
        indices = (row["query_index"], row["correct_index"], row["impostor_index"])
        labels = ("query", "same", text_fit(row["impostor_identity"], 12))
        colors = (BLUE, GREEN, RED)
        for col, (index, label, color) in enumerate(zip(indices, labels, colors)):
            ix = x + 15 + col * 152
            image.paste(face_crop(records[index], (140, 155)), (ix, y + 48))
            draw.text((ix + 70, y + 208), label, font=font(14, bold=True), fill=color, anchor="ma")
        draw.text(
            (x + 16, y + 244),
            f"正确 {row['correct_score']:.3f}   错误 {row['impostor_score']:.3f}",
            font=font(16),
            fill=TEXT,
        )
        draw.text(
            (x + tile_w - 16, y + 244),
            f"margin {row['margin']:+.3f}",
            font=font(16, bold=True),
            fill=status_color,
            anchor="ra",
        )
        base_margin = baseline[row["identity"]]["margin"]
        draw.text((x + 16, y + 273), f"冻结 {base_margin:+.3f} → 联合 {row['margin']:+.3f}", font=font(14), fill=MUTED)
    draw.text((36, 1690), "锁定模型：20/20 Top‑1；冻结基线：19/20。黄色鼻框仅用于展示 AnyFace/SAM 定位区域。", font=font(18), fill=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def render_evidence(runs_root: Path, output: Path) -> None:
    blind = load_json(runs_root / "dogfacenet_protocol32/blind_test_result.json")
    blind_manifest = load_json(runs_root / "dogfacenet_protocol32/blind_test_manifest.json")
    expanded = load_json(runs_root / "dogfacenet_expanded20/comparison.json")
    expanded_manifest = load_json(runs_root / "dogfacenet_expanded20/manifest.json")
    b_records = blind_manifest["records"]
    e_records = expanded_manifest["records"]
    wanda = next(row for row in expanded["selected"]["details"] if row["identity"] == "wanda")
    wanda_base = next(row for row in expanded["frozen"]["details"] if row["identity"] == "wanda")

    image = Image.new("RGB", (1800, 1120), BG)
    draw = ImageDraw.Draw(image)
    draw.text((48, 34), "证据页：一个被纠正的案例 + 一个仍未解决的失败模式", font=font(38, bold=True), fill=TEXT)

    # Wanda success.
    panel(draw, (40, 112, 1760, 535))
    draw.text((68, 136), "A. 扩展评估：Wanda 从误认 Boris 到正确匹配", font=font(27, bold=True), fill=GREEN)
    wanda_indices = [wanda["query_index"], wanda["correct_index"], wanda["impostor_index"]]
    wanda_labels = ["Wanda 查询", "Wanda 正确图库", "Boris 最强错误"]
    for col, (idx, label) in enumerate(zip(wanda_indices, wanda_labels)):
        x = 70 + col * 360
        image.paste(face_crop(e_records[idx], (325, 245)), (x, 205))
        draw.text((x, 463), label, font=font(18, bold=True), fill=(BLUE, GREEN, RED)[col])
    draw.text((1195, 230), "冻结基线", font=font(23, bold=True), fill=ORANGE)
    draw.text((1195, 274), f"正确 {wanda_base['correct_score']:.3f}", font=font(21), fill=TEXT)
    draw.text((1195, 310), f"错误 {wanda_base['impostor_score']:.3f}", font=font(21), fill=TEXT)
    draw.text((1195, 351), f"margin {wanda_base['margin']:+.3f}", font=font(24, bold=True), fill=RED)
    draw.text((1448, 230), "锁定模型", font=font(23, bold=True), fill=GREEN)
    draw.text((1448, 274), f"正确 {wanda['correct_score']:.3f}", font=font(21), fill=TEXT)
    draw.text((1448, 310), f"错误 {wanda['impostor_score']:.3f}", font=font(21), fill=TEXT)
    draw.text((1448, 351), f"margin {wanda['margin']:+.3f}", font=font(24, bold=True), fill=GREEN)
    draw.text((1195, 420), "扩展 Top‑1：19/20 → 20/20", font=font(22, bold=True), fill=YELLOW)

    # Peanut limitation.
    panel(draw, (40, 565, 1760, 1045))
    draw.text((68, 588), "B. 盲测边界：Peanut 的光照 / 距离 / 视角变化仍会破坏同犬一致性", font=font(27, bold=True), fill=RED)
    indices = [12, 13, 14, 9]
    labels = ["Peanut A", "Peanut B", "Peanut C · query", "Elza · different"]
    for col, (idx, label) in enumerate(zip(indices, labels)):
        x = 68 + col * 360
        image.paste(face_crop(b_records[idx], (325, 245)), (x, 650))
        draw.text((x, 908), label, font=font(18, bold=True), fill=(BLUE, BLUE, GREEN, RED)[col])
    notes = [
        "A↔B 同犬  0.111 → 0.094  ↓",
        "A↔C 同犬  0.158 → 0.138  ↓",
        "B↔C 同犬  0.584 → 0.625  ↑",
        "C↔Elza 异犬  0.381 → 0.387  ↑",
    ]
    for col, note in enumerate(notes):
        draw.text((68 + col * 360, 955), note, font=font(16, bold=True), fill=(RED, RED, GREEN, RED)[col])
    draw.text((68, 1005), "盲测 6/6 查询均正确且所有查询 margin 增大，但该困难配对使全配对 AUC 轻微下降 0.00082。", font=font(19), fill=MUTED)
    draw.text((46, 1080), "展示原则：成功案例、整体指标和失败边界放在同一页，避免只挑漂亮结果。", font=font(18), fill=MUTED)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=LEGACY_RUNS_ROOT)
    parser.add_argument("--repo", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALUATIONS_ROOT / "multimodal-showcase",
    )
    args = parser.parse_args()
    runs_root = (
        resolve_legacy_path(args.repo / "logs")
        if args.repo is not None
        else resolve_legacy_path(args.runs_root)
    )
    output = resolve_legacy_path(args.output_dir)
    render_overview(runs_root, output / "pet-reid-showcase-overview.png")
    render_query_board(runs_root, output / "pet-reid-expanded20-query-board.png")
    render_evidence(runs_root, output / "pet-reid-results-evidence.png")
    summary = {
        "overview": str((output / "pet-reid-showcase-overview.png").resolve()),
        "expanded_query_board": str((output / "pet-reid-expanded20-query-board.png").resolve()),
        "evidence": str((output / "pet-reid-results-evidence.png").resolve()),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "showcase_outputs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
