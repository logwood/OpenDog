# encoding: utf-8
"""Portable PetID datasets with an identity-disjoint verification split."""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

from .workspace_paths import PROCESSED_DATA_ROOT

__all__ = [
    "PetID",
    "PetIDFull",
    "PetIDSmoke",
    "PetIDValidation",
    "PetIDValidationSmoke",
    "PetIDTest",
    "PetIDTestPseudo",
]


TRAIN_ROOT = PROCESSED_DATA_ROOT / "dir_train_fusai"
SPLIT_ROOT = PROCESSED_DATA_ROOT / "splits"
TEST_ROOT = PROCESSED_DATA_ROOT / "test" / "test"


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(
            f'Required data manifest "{path}" is missing. '
            "Run scripts/prepare_upstream_assets.py first."
        )
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _identity_dirs(train_root: Path) -> list[str]:
    if not train_root.is_dir():
        raise RuntimeError(
            f'Training data directory "{train_root}" is missing. '
            "Run scripts/prepare_upstream_assets.py first."
        )
    return sorted((path.name for path in train_root.iterdir() if path.is_dir()), key=int)


def _training_items(identity_names: list[str], dataset_name: str) -> list[list[str]]:
    items: list[list[str]] = []
    for identity in identity_names:
        directory = TRAIN_ROOT / identity
        files = sorted(directory.glob("*.jpg"))
        if len(files) < 2:
            continue
        pid = f"{dataset_name}_{int(identity)}"
        camid = f"{dataset_name}_0"
        items.extend([[str(path), pid, camid] for path in files])
    if not items:
        raise RuntimeError("The selected PetID training split contains no images")
    return items


def _verification_items(max_pairs_per_label: int | None = None):
    pairs_path = SPLIT_ROOT / "validation_pairs.csv"
    if not pairs_path.is_file():
        raise RuntimeError(
            f'Required validation pair file "{pairs_path}" is missing. '
            "Run scripts/prepare_upstream_assets.py first."
        )

    query: list[list[object]] = []
    gallery: list[list[object]] = []
    counts = {0: 0, 1: 0}
    with pairs_path.open("r", encoding="utf-8", newline="") as handle:
        for source_index, row in enumerate(csv.DictReader(handle)):
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid verification label at row {source_index + 2}: {label}")
            if max_pairs_per_label is not None and counts[label] >= max_pairs_per_label:
                continue
            image_a = TRAIN_ROOT / Path(row["imageA"])
            image_b = TRAIN_ROOT / Path(row["imageB"])
            if not image_a.is_file() or not image_b.is_file():
                raise RuntimeError(
                    f"Validation pair references a missing image: {image_a}, {image_b}"
                )
            pair_index = len(query)
            query.append([str(image_a), label, pair_index])
            gallery.append([str(image_b), label, pair_index])
            counts[label] += 1
            if max_pairs_per_label is not None and all(
                count >= max_pairs_per_label for count in counts.values()
            ):
                break

    if not query or len(query) != len(gallery):
        raise RuntimeError("Validation pairs are empty or incomplete")
    if max_pairs_per_label is not None and any(
        count != max_pairs_per_label for count in counts.values()
    ):
        raise RuntimeError(
            f"Requested {max_pairs_per_label} smoke pairs per class, found {counts}"
        )
    return query, gallery


@DATASET_REGISTRY.register()
class PetID(ImageDataset):
    """Default training set: all identities except the fixed validation holdout."""

    dataset_name = "pet_id"

    def __init__(self, root="datasets", **kwargs):
        del root
        train_ids = _read_nonempty_lines(SPLIT_ROOT / "train_ids.txt")
        available = set(_identity_dirs(TRAIN_ROOT))
        missing = sorted(set(train_ids) - available, key=int)
        if missing:
            raise RuntimeError(f"Training split references missing identities: {missing[:5]}")
        train = _training_items(train_ids, self.dataset_name)
        super().__init__(train, [], [], **kwargs)


@DATASET_REGISTRY.register()
class PetIDFull(ImageDataset):
    """All 6,000 identities, intended for a final train-on-all-data run."""

    dataset_name = "pet_id_full"

    def __init__(self, root="datasets", **kwargs):
        del root
        train = _training_items(_identity_dirs(TRAIN_ROOT), self.dataset_name)
        super().__init__(train, [], [], **kwargs)


@DATASET_REGISTRY.register()
class PetIDSmoke(ImageDataset):
    """Eight identities for a fast end-to-end trainer/resume GPU check."""

    dataset_name = "pet_id_smoke"

    def __init__(self, root="datasets", **kwargs):
        del root
        train_ids = _read_nonempty_lines(SPLIT_ROOT / "train_ids.txt")[:8]
        train = _training_items(train_ids, self.dataset_name)
        super().__init__(train, [], [], **kwargs)


@DATASET_REGISTRY.register()
class PetIDValidation(ImageDataset):
    """Balanced positive/negative pairs from identities excluded from ``PetID``."""

    dataset_name = "pet_id_validation"

    def __init__(self, root="datasets", **kwargs):
        del root, kwargs
        query, gallery = _verification_items()
        super().__init__([], query, gallery)


@DATASET_REGISTRY.register()
class PetIDValidationSmoke(ImageDataset):
    """Small balanced pair set used only by the end-to-end smoke config."""

    dataset_name = "pet_id_validation_smoke"

    def __init__(self, root="datasets", **kwargs):
        del root, kwargs
        query, gallery = _verification_items(max_pairs_per_label=32)
        super().__init__([], query, gallery)


@DATASET_REGISTRY.register()
class PetIDTest(ImageDataset):
    """Phase B test images using original competition names as feature IDs."""

    dataset_name = "pet_id_test"

    def __init__(self, root="datasets", **kwargs):
        del root, kwargs
        query = self.process_test(TEST_ROOT)
        # Query/gallery both contain every image because the pair CSV may place
        # any image on either side.  The explicit name map keeps the portable
        # on-disk name separate from the competition name.
        gallery = list(query)
        super().__init__([], query, gallery)

    @staticmethod
    def _filename_map() -> dict[str, str]:
        path = PROCESSED_DATA_ROOT / "test" / "filename_map.csv"
        if not path.is_file():
            raise RuntimeError(
                f'Required filename map "{path}" is missing. '
                "Run scripts/prepare_upstream_assets.py first."
            )
        with path.open("r", encoding="utf-8", newline="") as handle:
            mapping = {
                row["local_name"]: row["original_name"] for row in csv.DictReader(handle)
            }
        if len(mapping) != 4000:
            raise RuntimeError(f"Expected 4,000 test filename mappings, found {len(mapping)}")
        return mapping

    def process_test(self, test_path: Path) -> list[list[str]]:
        if not test_path.is_dir():
            raise RuntimeError(
                f'Test image directory "{test_path}" is missing. '
                "Run scripts/prepare_upstream_assets.py first."
            )
        mapping = self._filename_map()
        image_paths = sorted(test_path.glob("*.jpg"))
        unknown = [path.name for path in image_paths if path.name not in mapping]
        if unknown:
            raise RuntimeError(f"Test filename map is incomplete: {unknown[:5]}")
        return [[str(path), mapping[path.name], "pet_0"] for path in image_paths]


@DATASET_REGISTRY.register()
class PetIDTestPseudo(ImageDataset):
    dataset_name = "pet_id_test_pseudo"
    dataset_dir = "pet_id"

    def __init__(self, root="datasets", **kwargs):
        image_dir = Path(root) / self.dataset_dir / "train" / "images"
        query = self.process_test(image_dir)
        super().__init__([], query, list(query), **kwargs)

    @staticmethod
    def process_test(image_dir: Path) -> list[list[str]]:
        image_paths = sorted(glob.glob(os.path.join(str(image_dir), "*.jpg")))
        return [[path, os.path.basename(path), "pet_0"] for path in image_paths]
