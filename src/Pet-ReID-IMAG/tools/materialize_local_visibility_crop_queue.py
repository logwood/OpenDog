"""Materialize real model-local crops for human visibility annotation.

This is a data-acquisition step, not an auto-labeler. Every label remains
unknown until an annotator inspects the saved crop. Development/blind manifests
are never used; the input records must come from the previously audited queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pet_id.unified_local_path import load_local_path_checkpoint  # noqa: E402

PARTS = ("face", "nose")
UNKNOWN = {"usable": None, "provenance": "unknown", "evidence": None}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_rgb(record):
    path = Path(record["source_path"])
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)


def save_rgb(value, path):
    array = value.detach().float().clamp(0, 255).round().byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=95, subsampling=0)


def select_records(train_template, validation_template, *, max_identities, seed):
    train = json.loads(Path(train_template).read_text(encoding="utf-8-sig"))
    validation = json.loads(Path(validation_template).read_text(encoding="utf-8-sig"))
    if (
        train.get("schema_name") != "model_local_part_visibility"
        or validation.get("schema_name") != "model_local_part_visibility"
        or train.get("split") != "train"
        or validation.get("split") != "validation"
    ):
        raise ValueError("Only unmodified annotation queue templates are accepted")
    train_records, validation_records = train["records"], validation["records"]
    train_ids = {r["identity"].strip().casefold() for r in train_records}
    validation_ids = {r["identity"].strip().casefold() for r in validation_records}
    if train_ids & validation_ids:
        raise ValueError("Annotation templates are not identity-disjoint")
    if max_identities <= 0:
        chosen_train, chosen_validation = train_ids, validation_ids
    else:
        train_limit = max(1, round(max_identities * 0.8))
        validation_limit = max(1, max_identities - train_limit)
        chosen_train = set(sorted(train_ids, key=lambda value: hashlib.sha256(f"{seed}:train:{value}".encode()).hexdigest())[:train_limit])
        chosen_validation = set(sorted(validation_ids, key=lambda value: hashlib.sha256(f"{seed}:validation:{value}".encode()).hexdigest())[:validation_limit])
    chosen = chosen_train | chosen_validation
    result = []
    for split, records in (("train", train_records), ("validation", validation_records)):
        for record in records:
            identity = record["identity"].strip().casefold()
            if identity not in chosen:
                continue
            if any(label != UNKNOWN for label in record["local_visibility"].values()):
                raise ValueError("Refusing to overwrite an already annotated record")
            result.append((split, record))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-template", type=Path, required=True)
    parser.add_argument("--validation-template", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-identities", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    records = select_records(args.train_template, args.validation_template, max_identities=args.max_identities, seed=args.seed)
    if not records:
        raise ValueError("No records selected for crop annotation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    device = torch.device(args.device)
    model, payload = load_local_path_checkpoint(args.checkpoint, device=device)
    model.eval().requires_grad_(False)
    entries = []
    with torch.inference_mode():
        for index, (split, record) in enumerate(records):
            source = Path(record["source_path"])
            if digest(source) != record["source_sha256"]:
                raise ValueError(f"Source hash mismatch: {source}")
            output_value = model(read_rgb(record).to(device), return_aux=True)
            prefix = f"{index:06d}_{record['source_sha256'][:12]}"
            face_paths = []
            for scale_index, crop in enumerate(output_value["face_crops"][0]):
                path = output / "crops" / f"{prefix}_face_{scale_index}.jpg"
                save_rgb(crop, path)
                face_paths.append(str(path.relative_to(output)))
            nose_path = output / "crops" / f"{prefix}_nose.jpg"
            save_rgb(output_value["nose_crop"][0], nose_path)
            entries.append({
                "index": index,
                "split": split,
                "identity": record["identity"],
                "source_path": record["source_path"],
                "source_sha256": record["source_sha256"],
                "face_crop_paths": face_paths,
                "nose_crop_path": str(nose_path.relative_to(output)),
                "attention_concentration": float(output_value["attention_concentration"][0]),
                "nose_valid_fraction": float(output_value["nose_valid_fraction"][0]),
                "face_nose_cosine": float(output_value["face_nose_cosine"][0]),
                "local_visibility": {part: dict(UNKNOWN) for part in PARTS},
                "teacher_weak": {
                    "available": False,
                    "reason": "not_auto_labeled; detector/geometry proxies are not usability truth",
                },
            })
            if index == 0 or (index + 1) % 32 == 0 or index + 1 == len(records):
                print(json.dumps({"phase": "materialize_crops", "records": index + 1, "total": len(records)}), flush=True)
    manifest = {
        "schema_version": 1,
        "status": "awaiting_human_model_crop_labels",
        "target_graph_sha256": "1d8f961d836027ca4e52d4dac63b9f1e44824d779199f373de630b7a27b276ba",
        "source_queue_train": str(Path(args.train_template).resolve()),
        "source_queue_validation": str(Path(args.validation_template).resolve()),
        "source_queue_checkpoint_sha256": digest(args.checkpoint),
        "model_config": model.configuration(),
        "selection": {"seed": args.seed, "max_identities": args.max_identities, "records": len(entries), "identities": len({e["identity"].casefold() for e in entries})},
        "instructions": [
            "Inspect the saved face and nose crops, not the full source image.",
            "Mark usable only when the crop contains identity-bearing, semantically correct evidence for that part.",
            "Keep occluded, ambiguous, wrong-part, or uncertain crops unknown.",
            "Record human provenance and annotation_id for every known label.",
        ],
        "blind_images_read": False,
        "records": entries,
    }
    (output / "crop_queue.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "queue": str(output / "crop_queue.json"), "records": len(entries), "identities": manifest["selection"]["identities"]}), flush=True)


if __name__ == "__main__":
    main()
