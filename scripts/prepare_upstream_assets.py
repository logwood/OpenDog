#!/usr/bin/env python3
"""Prepare the released Pet-ReID assets for native Windows or Linux use.

The archives contain ``*`` in image names.  Every extracted image receives a
portable ``_`` name and an explicit CSV mapping back to the original name used
by the competition pair file.  The script also creates a deterministic,
identity-disjoint validation split with balanced verification pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = WORKSPACE_ROOT / "src" / "Pet-ReID-IMAG"
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data" / "processed" / "pet-reid-imag"
DEFAULT_RUNS_ROOT = WORKSPACE_ROOT / "artifacts" / "runs" / "legacy"
EXPECTED = {
    "train_images": 38636,
    "test_images": 4000,
    "weights": 4,
    "features": 8,
}


def validate_destination(repo: Path) -> None:
    if not (repo / "pet_id" / "train_net.py").is_file():
        raise SystemExit(f"Not a Pet-ReID-IMAG checkout: {repo}")


def safe_target(root: Path, relative: PurePosixPath) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {relative}")
    target = root.joinpath(*relative.parts)
    target.resolve().relative_to(root.resolve())
    return target


def portable_relative(relative: PurePosixPath) -> PurePosixPath:
    """Return the lossless-on-disk representation used on every platform."""
    return PurePosixPath(*(part.replace("*", "_") for part in relative.parts))


def ensure_no_collision(
    seen: dict[str, PurePosixPath], original: PurePosixPath, portable: PurePosixPath
) -> None:
    # Windows paths are case-insensitive, so audit all platforms with the same
    # stricter rule.  This keeps a prepared bundle movable between systems.
    key = portable.as_posix().casefold()
    previous = seen.get(key)
    if previous is not None and previous != original:
        raise ValueError(
            f"Portable filename collision: {previous} and {original} -> {portable}"
        )
    seen[key] = original


def copy_member(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path, force: bool
) -> bool:
    if info.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        return False
    if target.exists() and not force:
        if target.is_file() and target.stat().st_size == info.file_size:
            return False
        raise FileExistsError(f"Refusing to overwrite {target}; use --force if intentional")
    target.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(info) as source, target.open("wb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
    return True


def write_generated(path: Path, content: str, force: bool) -> bool:
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return False
        if not force:
            raise FileExistsError(f"Refusing to overwrite {path}; use --force if intentional")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="")
    return True


def extract_train(bundle: Path, data_root: Path, force: bool) -> int:
    written = 0
    seen: dict[str, PurePosixPath] = {}
    with zipfile.ZipFile(bundle / "dir_train.zip") as zf:
        for info in zf.infolist():
            source = PurePosixPath(info.filename)
            if not source.parts or source.parts[0] != "dir_train" or len(source.parts) == 1:
                continue
            original = PurePosixPath(*source.parts[1:])
            portable = portable_relative(original)
            ensure_no_collision(seen, original, portable)
            target = safe_target(data_root / "dir_train_fusai", portable)
            written += copy_member(zf, info, target, force)
    return written


def extract_test(bundle: Path, data_root: Path, force: bool) -> tuple[int, int]:
    written = 0
    seen: dict[str, PurePosixPath] = {}
    mappings: list[tuple[str, str]] = []
    with zipfile.ZipFile(bundle / "test.zip") as zf:
        for info in zf.infolist():
            source = PurePosixPath(info.filename)
            if not source.parts or source.parts[0] != "test" or len(source.parts) == 1:
                continue
            if source.name == ".DS_Store":
                continue
            original = PurePosixPath(*source.parts[1:])
            portable = portable_relative(original)
            ensure_no_collision(seen, original, portable)
            target = safe_target(data_root / "test", portable)
            written += copy_member(zf, info, target, force)
            if original.suffix.lower() == ".jpg":
                mappings.append((portable.name, original.name))

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["local_name", "original_name"])
    writer.writerows(sorted(mappings))
    mapping_written = write_generated(
        data_root / "test" / "filename_map.csv", output.getvalue(), force
    )
    return written, int(mapping_written)


def extract_logs(bundle: Path, runs_root: Path, force: bool) -> int:
    written = 0
    with zipfile.ZipFile(bundle / "logs.zip") as zf:
        for info in zf.infolist():
            source = PurePosixPath(info.filename)
            if not source.parts or source.parts[0] != "logs" or len(source.parts) == 1:
                continue
            parts = list(source.parts[1:])
            if parts[0] == "S200_224":
                parts[0] = "s200_224"
            target = safe_target(runs_root, portable_relative(PurePosixPath(*parts)))
            written += copy_member(zf, info, target, force)
    return written


def stable_key(seed: int, *parts: str) -> bytes:
    payload = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(payload).digest()


def generate_validation_split(
    data_root: Path, ratio: float, seed: int, pairs_per_class: int, force: bool
) -> dict:
    if not 0 < ratio < 1:
        raise ValueError("--validation-ratio must be between 0 and 1")
    if pairs_per_class < 1:
        raise ValueError("--validation-pairs-per-class must be positive")

    train_root = data_root / "dir_train_fusai"
    identities = sorted(
        (p.name for p in train_root.iterdir() if p.is_dir()), key=lambda x: int(x)
    )
    if len(identities) < 2:
        raise RuntimeError(f"No identity directories found under {train_root}")

    holdout_count = max(2, min(len(identities) - 1, round(len(identities) * ratio)))
    validation_ids = sorted(
        sorted(identities, key=lambda x: stable_key(seed, x))[:holdout_count], key=int
    )
    validation_set = set(validation_ids)
    train_ids = [identity for identity in identities if identity not in validation_set]

    images_by_id: dict[str, list[Path]] = {}
    for identity in validation_ids:
        images = sorted((train_root / identity).glob("*.jpg"))
        if len(images) >= 2:
            images_by_id[identity] = images
    validation_ids = [identity for identity in validation_ids if identity in images_by_id]

    positives: list[tuple[Path, Path, int]] = []
    for identity in validation_ids:
        for image_a, image_b in itertools.combinations(images_by_id[identity], 2):
            positives.append((image_a, image_b, 1))
    positives.sort(
        key=lambda pair: stable_key(
            seed, pair[0].relative_to(train_root).as_posix(), pair[1].relative_to(train_root).as_posix()
        )
    )
    if len(positives) < pairs_per_class:
        raise RuntimeError(
            f"Only {len(positives)} positive pairs are available; requested {pairs_per_class}"
        )
    positives = positives[:pairs_per_class]

    rng = random.Random(seed ^ 0x50455449)
    negatives: list[tuple[Path, Path, int]] = []
    negative_keys: set[tuple[str, str]] = set()
    while len(negatives) < pairs_per_class:
        identity_a, identity_b = rng.sample(validation_ids, 2)
        image_a = rng.choice(images_by_id[identity_a])
        image_b = rng.choice(images_by_id[identity_b])
        relative_a = image_a.relative_to(train_root).as_posix()
        relative_b = image_b.relative_to(train_root).as_posix()
        key = tuple(sorted((relative_a, relative_b)))
        if key in negative_keys:
            continue
        negative_keys.add(key)
        negatives.append((image_a, image_b, 0))

    pairs = positives + negatives
    rng.shuffle(pairs)

    split_root = data_root / "splits"
    train_text = "\n".join(train_ids) + "\n"
    validation_text = "\n".join(validation_ids) + "\n"
    pairs_output = io.StringIO(newline="")
    pairs_writer = csv.writer(pairs_output, lineterminator="\n")
    pairs_writer.writerow(["imageA", "imageB", "label"])
    for image_a, image_b, label in pairs:
        pairs_writer.writerow(
            [
                image_a.relative_to(train_root).as_posix(),
                image_b.relative_to(train_root).as_posix(),
                label,
            ]
        )

    manifest = {
        "version": 1,
        "strategy": "identity_disjoint_sha256_holdout_balanced_pairs",
        "seed": seed,
        "validation_ratio": ratio,
        "all_identities": len(identities),
        "train_identities": len(train_ids),
        "validation_identities": len(validation_ids),
        "positive_pairs": pairs_per_class,
        "negative_pairs": pairs_per_class,
    }
    generated = 0
    generated += write_generated(split_root / "train_ids.txt", train_text, force)
    generated += write_generated(split_root / "validation_ids.txt", validation_text, force)
    generated += write_generated(
        split_root / "validation_pairs.csv", pairs_output.getvalue(), force
    )
    generated += write_generated(
        split_root / "split_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        force,
    )
    manifest["files_written"] = int(generated)
    return manifest


def verify(data_root: Path, runs_root: Path) -> dict:
    train_root = data_root / "dir_train_fusai"
    test_root = data_root / "test" / "test"
    train_images = sum(1 for p in train_root.glob("*/*.jpg") if p.is_file())
    test_images = sum(1 for p in test_root.glob("*.jpg") if p.is_file())
    models = ("s101_224", "s101_256", "s101_288", "s200_224")
    weights = sum((runs_root / model / "model_final.pth").is_file() for model in models)
    features = sum(
        (runs_root / model / f"{kind}_f.npy").is_file()
        for model in models
        for kind in ("query", "gallery")
    )

    split_root = data_root / "splits"
    train_ids = (
        set((split_root / "train_ids.txt").read_text(encoding="utf-8").split())
        if (split_root / "train_ids.txt").is_file()
        else set()
    )
    validation_ids = (
        set((split_root / "validation_ids.txt").read_text(encoding="utf-8").split())
        if (split_root / "validation_ids.txt").is_file()
        else set()
    )
    labels: list[int] = []
    validation_pairs = split_root / "validation_pairs.csv"
    if validation_pairs.is_file():
        with validation_pairs.open("r", encoding="utf-8", newline="") as handle:
            labels = [int(row["label"]) for row in csv.DictReader(handle)]

    report = {
        "data_root": str(data_root.resolve()),
        "runs_root": str(runs_root.resolve()),
        "train_images": train_images,
        "test_images": test_images,
        "test_csv": (data_root / "test" / "test_data.csv").is_file(),
        "filename_map": (data_root / "test" / "filename_map.csv").is_file(),
        "weights": weights,
        "features": features,
        "train_identities": len(train_ids),
        "validation_identities": len(validation_ids),
        "split_disjoint": bool(train_ids and validation_ids and train_ids.isdisjoint(validation_ids)),
        "validation_pairs": len(labels),
        "positive_pairs": sum(label == 1 for label in labels),
        "negative_pairs": sum(label == 0 for label in labels),
    }
    assets_ready = all(report[key] == value for key, value in EXPECTED.items())
    validation_ready = (
        report["split_disjoint"]
        and report["validation_pairs"] > 0
        and report["positive_pairs"] == report["negative_pairs"]
    )
    report["ready"] = bool(
        assets_ready
        and report["test_csv"]
        and report["filename_map"]
        and validation_ready
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument(
        "--source-root", "--repo", dest="source_root", type=Path,
        default=DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--validation-seed", type=int, default=2022)
    parser.add_argument("--validation-pairs-per-class", type=int, default=1000)
    args = parser.parse_args()

    bundle = args.bundle_root.resolve()
    source_root = args.source_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    validate_destination(source_root)
    written = {"train": 0, "test": 0, "mapping": 0, "logs": 0, "split": 0}
    split_report: dict = {}
    if not args.verify_only:
        for archive in ("dir_train.zip", "test.zip", "logs.zip"):
            if not (bundle / archive).is_file():
                raise SystemExit(f"Missing archive: {bundle / archive}")
        written["train"] = extract_train(bundle, data_root, args.force)
        written["test"], written["mapping"] = extract_test(bundle, data_root, args.force)
        written["logs"] = extract_logs(bundle, runs_root, args.force)
        split_report = generate_validation_split(
            data_root,
            args.validation_ratio,
            args.validation_seed,
            args.validation_pairs_per_class,
            args.force,
        )
        written["split"] = split_report.pop("files_written")

    report = verify(data_root, runs_root)
    report["source_root"] = str(source_root)
    report["files_written"] = written
    if split_report:
        report["split"] = split_report
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
