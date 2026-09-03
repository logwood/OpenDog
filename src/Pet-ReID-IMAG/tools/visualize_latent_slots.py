# encoding: utf-8
"""Render C2-C5 latent read maps for MHA or SA-MESH checkpoints."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pet_reid_matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastreid.config import get_cfg
from fastreid.data.transforms import build_transforms
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from pet_id import add_retri_config
from pet_id.workspace_paths import normalize_runtime_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("images", nargs="+")
    return parser.parse_args()


def build_capture_model(config_file, weights):
    cfg = get_cfg()
    add_retri_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    cfg.MODEL.WEIGHTS = weights
    normalize_runtime_config(cfg)
    cfg.freeze()

    model = build_model(cfg).to(torch.device(cfg.MODEL.DEVICE)).eval()
    Checkpointer(model).load(weights)
    if not hasattr(model, "workspace"):
        raise TypeError("The selected model does not contain a latent workspace")
    model.workspace.set_attention_capture(True)
    return cfg, model


def render_image(image_path, image, attention_maps, output_path):
    stages = ("c2", "c3", "c4", "c5")
    figure, axes = plt.subplots(len(stages), 9, figsize=(25, 11), squeeze=False)
    display_width, display_height = image.size

    for row, stage_name in enumerate(stages):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"{stage_name.upper()} input")
        axes[row, 0].axis("off")
        stage = attention_maps[stage_name]
        stage_height, stage_width = stage["spatial_size"]
        weights = stage["attention"][0]
        if weights.shape[0] != 8:
            raise ValueError(f"Expected 8 slots, found {weights.shape[0]}")
        maps = weights.reshape(8, 1, stage_height, stage_width)
        maps = F.interpolate(
            maps,
            size=(display_height, display_width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        for slot_index in range(8):
            heatmap = maps[slot_index]
            heatmap = heatmap - heatmap.min()
            heatmap = heatmap / heatmap.max().clamp_min(1e-12)
            axes[row, slot_index + 1].imshow(image)
            axes[row, slot_index + 1].imshow(
                heatmap.numpy(), cmap="turbo", alpha=0.55, vmin=0.0, vmax=1.0
            )
            axes[row, slot_index + 1].set_title(f"slot {slot_index}")
            axes[row, slot_index + 1].axis("off")

    figure.suptitle(str(image_path), fontsize=11)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg, model = build_capture_model(args.config_file, args.weights)
    transform = build_transforms(cfg, is_train=False)

    for image_index, raw_path in enumerate(args.images):
        image_path = Path(raw_path)
        image = Image.open(image_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(model.device)
        model.workspace.set_attention_capture(True)
        with torch.no_grad():
            model({"images": tensor})
        output_path = output_dir / f"{image_index:02d}_{image_path.stem}_slots.png"
        render_image(
            image_path,
            image,
            model.workspace.attention_maps(),
            output_path,
        )
        print(output_path.resolve())


if __name__ == "__main__":
    main()
