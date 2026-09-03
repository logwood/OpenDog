# encoding: utf-8
"""DogFaceNet alignment indexing and locally end-to-end training data.

The released alignment archive stores identity labels in filenames and three
target landmarks in ``labels.csv``.  Geometry is computed once with frozen
AnyFace/SAM 2, while training still starts from resized full-image pixels and
uses differentiable ROI extraction inside :class:`LocalEndToEndPetIDModel`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from .localization import (
    AnyFaceDetector,
    FaceDetection,
    SAM2NoseSegmenter,
    crop_aligned_face,
    face_quality,
    laplacian_sharpness_quality,
    nose_quality,
    nose_roi_box,
    viewpoint_signals,
)
from .workspace_paths import resolve_legacy_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def dogfacenet_identity_from_filename(filename: str) -> str:
    """Recover the implicit individual ID from an alignment image filename.

    Most files use ``identity.original_name.jpg``.  The source identity names
    ``B.Atis``, ``B.Dömpi`` and ``B.Lukas`` contain the separator themselves,
    so their first two tokens form the label.
    """

    name = Path(filename).name
    parts = name.split(".")
    if len(parts) < 2 or not parts[0]:
        raise ValueError(f"DogFaceNet filename has no identity prefix: {filename}")
    if parts[0].casefold() == "b" and len(parts) >= 3 and parts[1]:
        return ".".join(parts[:2])
    return parts[0]


def _read_bgr(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return image


def apply_exif_orientation_to_points(
    points: np.ndarray,
    *,
    encoded_size: tuple[int, int],
    orientation: int,
) -> np.ndarray:
    """Map encoded JPEG coordinates into the EXIF-oriented display image."""

    points = np.asarray(points, dtype=np.float32)
    width, height = encoded_size
    x, y = points[..., 0], points[..., 1]
    orientation = int(orientation or 1)
    if orientation == 1:
        transformed = (x, y)
    elif orientation == 2:
        transformed = (width - 1 - x, y)
    elif orientation == 3:
        transformed = (width - 1 - x, height - 1 - y)
    elif orientation == 4:
        transformed = (x, height - 1 - y)
    elif orientation == 5:
        transformed = (y, x)
    elif orientation == 6:
        transformed = (height - 1 - y, x)
    elif orientation == 7:
        transformed = (height - 1 - y, width - 1 - x)
    elif orientation == 8:
        transformed = (y, width - 1 - x)
    else:
        transformed = (x, y)
    return np.stack(transformed, axis=-1).astype(np.float32)


def _file_crc32(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


@dataclass(frozen=True)
class AlignmentIndexRecord:
    source_path: Path
    canonical_filename: str
    identity: str
    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: tuple[float, float]

    @property
    def eye_distance(self) -> float:
        return math.dist(self.left_eye, self.right_eye)

    @property
    def annotation_points(self) -> np.ndarray:
        return np.asarray((self.left_eye, self.right_eye, self.nose), dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path.resolve()),
            "canonical_filename": self.canonical_filename,
            "identity": self.identity,
            "left_eye": list(self.left_eye),
            "right_eye": list(self.right_eye),
            "nose": list(self.nose),
            "eye_distance": self.eye_distance,
        }


def build_alignment_index(
    dataset_root,
    *,
    archive_path=None,
    repair_suffixes=True,
    repair_crc=True,
) -> tuple[list[AlignmentIndexRecord], dict]:
    """Resolve CSV annotations to extracted images without renaming user data."""

    dataset_root = Path(dataset_root)
    image_root = dataset_root / "images"
    csv_path = dataset_root / "labels.csv"
    if not image_root.is_dir() or not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected images/ and labels.csv under DogFaceNet root: {dataset_root}"
        )
    image_paths = [
        path for path in image_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    actual_by_name = {path.name.casefold(): path for path in image_paths}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))

    resolved: dict[int, tuple[Path, str]] = {}
    claimed_paths: set[Path] = set()
    exact_count = 0
    for index, row in enumerate(annotations):
        path = actual_by_name.get(row["filename"].casefold())
        if path is not None:
            resolved[index] = (path, "exact")
            claimed_paths.add(path)
            exact_count += 1

    suffix_count = 0
    if repair_suffixes:
        suffix_candidates: dict[str, list[Path]] = defaultdict(list)
        for path in image_paths:
            if path in claimed_paths:
                continue
            suffix = path.name.split(".", 1)[-1].casefold()
            suffix_candidates[suffix].append(path)
        for index, row in enumerate(annotations):
            if index in resolved:
                continue
            suffix = row["filename"].split(".", 1)[-1].casefold()
            candidates = [
                path for path in suffix_candidates.get(suffix, ()) if path not in claimed_paths
            ]
            if len(candidates) == 1:
                resolved[index] = (candidates[0], "suffix")
                claimed_paths.add(candidates[0])
                suffix_count += 1

    crc_count = 0
    if archive_path and repair_crc:
        archive_path = Path(archive_path)
        if not archive_path.is_file():
            raise FileNotFoundError(f"DogFaceNet archive not found: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            archive_infos = {
                Path(info.filename).name.casefold(): info
                for info in archive.infolist()
                if Path(info.filename).suffix.lower() in IMAGE_SUFFIXES
            }
        desired_signatures = {}
        for index, row in enumerate(annotations):
            if index in resolved:
                continue
            info = archive_infos.get(row["filename"].casefold())
            if info is not None:
                desired_signatures[index] = (int(info.file_size), int(info.CRC))
        needed_sizes = {signature[0] for signature in desired_signatures.values()}
        actual_signatures: dict[tuple[int, int], list[Path]] = defaultdict(list)
        for path in image_paths:
            if path in claimed_paths or path.stat().st_size not in needed_sizes:
                continue
            actual_signatures[(path.stat().st_size, _file_crc32(path))].append(path)
        for index, signature in desired_signatures.items():
            candidates = [
                path
                for path in actual_signatures.get(signature, ())
                if path not in claimed_paths
            ]
            if len(candidates) == 1:
                resolved[index] = (candidates[0], "crc")
                claimed_paths.add(candidates[0])
                crc_count += 1

    records = []
    resolution_counts = Counter()
    unresolved = []
    for index, row in enumerate(annotations):
        if index not in resolved:
            unresolved.append(row["filename"])
            continue
        source_path, method = resolved[index]
        resolution_counts[method] += 1
        records.append(
            AlignmentIndexRecord(
                source_path=source_path,
                canonical_filename=row["filename"],
                identity=dogfacenet_identity_from_filename(row["filename"]),
                left_eye=(float(row["lex"]), float(row["ley"])),
                right_eye=(float(row["rex"]), float(row["rey"])),
                nose=(float(row["nox"]), float(row["noy"])),
            )
        )
    identities = Counter(record.identity.casefold() for record in records)
    report = {
        "images": len(image_paths),
        "annotations": len(annotations),
        "resolved": len(records),
        "resolution_methods": dict(resolution_counts),
        "unresolved": len(unresolved),
        "unresolved_examples": unresolved[:20],
        "identities": len(identities),
        "identities_with_two_or_more": sum(count >= 2 for count in identities.values()),
        "exact_matches": exact_count,
        "suffix_repairs": suffix_count,
        "crc_repairs": crc_count,
    }
    return records, report


@dataclass(frozen=True)
class TargetDetectionMatch:
    detection: FaceDetection
    score: float
    normalized_nose_distance: float
    normalized_eye_distance: float


def match_annotated_target(
    detections: Sequence[FaceDetection],
    annotation_points: np.ndarray,
    *,
    max_score=1.25,
    max_nose_distance=1.25,
) -> TargetDetectionMatch | None:
    """Select the AnyFace result belonging to the CSV-annotated animal."""

    points = np.asarray(annotation_points, dtype=np.float32)
    if points.shape != (3, 2):
        raise ValueError(f"Expected left-eye/right-eye/nose points, got {points.shape}")
    eye_scale = max(float(np.linalg.norm(points[1] - points[0])), 8.0)
    matches = []
    for detection in detections:
        detected = np.asarray(detection.landmarks_xy[:3], dtype=np.float32)
        direct = np.linalg.norm(detected[:2] - points[:2], axis=1).mean()
        swapped = np.linalg.norm(detected[[1, 0]] - points[:2], axis=1).mean()
        eye_distance = float(min(direct, swapped) / eye_scale)
        nose_distance = float(np.linalg.norm(detected[2] - points[2]) / eye_scale)
        score = 0.65 * nose_distance + 0.35 * eye_distance
        score += 0.03 * (1.0 - detection.confidence)
        matches.append(
            TargetDetectionMatch(
                detection=detection,
                score=score,
                normalized_nose_distance=nose_distance,
                normalized_eye_distance=eye_distance,
            )
        )
    if not matches:
        return None
    best = min(matches, key=lambda match: match.score)
    if best.score > float(max_score) or best.normalized_nose_distance > float(
        max_nose_distance
    ):
        return None
    return best


def _expanded_face_box(
    detection: FaceDetection, image_shape: Sequence[int], padding=0.12
) -> tuple[float, float, float, float]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = detection.bbox_xyxy
    pad_x, pad_y = padding * detection.width, padding * detection.height
    return (
        max(x1 - pad_x, 0.0),
        max(y1 - pad_y, 0.0),
        min(x2 + pad_x, float(width)),
        min(y2 + pad_y, float(height)),
    )


def _roll_angle(detection: FaceDetection) -> float:
    left, right = detection.left_eye, detection.right_eye
    return math.atan2(right[1] - left[1], right[0] - left[0])


def geometry_cache_namespace(paths: Iterable[Path | str], settings: dict) -> str:
    records = []
    for value in paths:
        path = Path(value).resolve()
        stat = path.stat()
        records.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps({"files": records, "settings": settings}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def prepare_alignment_record(
    item: AlignmentIndexRecord,
    detector: AnyFaceDetector,
    segmenter: SAM2NoseSegmenter,
    *,
    output_root,
    namespace: str,
    max_long_side=1280,
    allow_raw_nose_fallback=True,
) -> dict:
    """Cache frozen geometry for one labeled target dog."""

    output_root = Path(output_root)
    source_stat = item.source_path.stat()
    key_payload = (
        f"{item.source_path.resolve()}|{source_stat.st_size}|{source_stat.st_mtime_ns}|"
        f"{namespace}|{max_long_side}"
    )
    key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:24]
    record_path = output_root / "records" / f"{key}.json"
    if record_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))

    with Image.open(item.source_path) as encoded_image:
        encoded_size = encoded_image.size
        exif_orientation = int(encoded_image.getexif().get(274, 1) or 1)
    image = _read_bgr(item.source_path)
    original_height, original_width = image.shape[:2]
    scale = min(1.0, float(max_long_side) / max(original_height, original_width))
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    if (resized_width, resized_height) != (original_width, original_height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    oriented_points = apply_exif_orientation_to_points(
        item.annotation_points,
        encoded_size=encoded_size,
        orientation=exif_orientation,
    )
    annotation_points = oriented_points * np.asarray((scale, scale), dtype=np.float32)
    detections = detector.detect(image)
    match = match_annotated_target(detections, annotation_points)
    if match is None:
        raise RuntimeError(
            f"No AnyFace detection matches annotated target ({len(detections)} detections)"
        )
    detection = match.detection
    aligned_face = crop_aligned_face(image, detection)
    face_q = face_quality(detection, aligned_face)
    segmentation = None
    try:
        segmentation = segmenter.segment(image, detection)
        nx1, ny1, nx2, ny2 = segmentation.roi_box_xyxy
        raw_nose = image[ny1:ny2, nx1:nx2]
        nose_q = nose_quality(segmentation, raw_nose)
        mask_crop = segmentation.mask[ny1:ny2, nx1:nx2]
        predicted_iou = segmentation.predicted_iou
        selected_candidate = segmentation.selected_candidate
        nose_available = True
    except Exception:
        if not allow_raw_nose_fallback:
            raise
        nx1, ny1, nx2, ny2 = nose_roi_box(detection, image.shape)
        raw_nose = image[ny1:ny2, nx1:nx2]
        mask_crop = np.ones((ny2 - ny1, nx2 - nx1), dtype=bool)
        nose_q = 0.15 * laplacian_sharpness_quality(raw_nose)
        predicted_iou = 0.0
        selected_candidate = -1
        nose_available = True

    mask_root = output_root / "masks"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    mask_path = mask_root / f"{key}.png"
    if not cv2.imwrite(str(mask_path), mask_crop.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to save nose mask: {mask_path}")
    face_box = _expanded_face_box(detection, image.shape)
    nose_resolution = float(np.clip(min(nx2 - nx1, ny2 - ny1) / 96.0, 0.0, 1.0))
    face_resolution = float(
        np.clip(min(detection.width, detection.height) / 160.0, 0.0, 1.0)
    )
    record = {
        "schema_version": 4,
        "cache_key": key,
        "source_path": str(item.source_path.resolve()),
        "canonical_filename": item.canonical_filename,
        "identity": item.identity,
        "original_size": [original_width, original_height],
        "encoded_size": list(encoded_size),
        "exif_orientation": exif_orientation,
        "resized_size": [resized_width, resized_height],
        "scale": scale,
        "annotation_points": annotation_points.tolist(),
        "detection": detection.to_dict(),
        "target_match": {
            "score": match.score,
            "normalized_nose_distance": match.normalized_nose_distance,
            "normalized_eye_distance": match.normalized_eye_distance,
            "detections_in_image": len(detections),
        },
        "face_roi_xyxy": list(face_box),
        "nose_roi_xyxy": [nx1, ny1, nx2, ny2],
        "roll_angle_radians": _roll_angle(detection),
        "viewpoint_signals": viewpoint_signals(detection).tolist(),
        "quality_signals": [
            nose_q,
            face_q,
            detection.confidence,
            predicted_iou,
            nose_resolution,
            face_resolution,
        ],
        "branch_available": [nose_available, True],
        "segmentation": {
            "predicted_iou": predicted_iou,
            "selected_candidate": selected_candidate,
            "roi_mask_fraction": float(mask_crop.mean()),
        },
        "mask_path": str(mask_path.resolve()),
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


class PreparedDogFaceNetDataset(Dataset):
    """Load full resized images plus cached target geometry for local E2E training."""

    def __init__(
        self,
        manifest_path,
        *,
        training=False,
        horizontal_flip_probability=0.0,
        color_jitter=0.0,
        min_images_per_identity=1,
        max_images_per_identity=0,
    ):
        self.manifest_path = resolve_legacy_path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        records = list(manifest["records"])
        counts = Counter(record["identity"].casefold() for record in records)
        allowed = {
            identity for identity, count in counts.items() if count >= min_images_per_identity
        }
        eligible_records = [
            record for record in records if record["identity"].casefold() in allowed
        ]
        maximum = int(max_images_per_identity)
        if maximum > 0:
            selected_counts = Counter()
            self.records = []
            for record in eligible_records:
                identity = record["identity"].casefold()
                if selected_counts[identity] >= maximum:
                    continue
                self.records.append(record)
                selected_counts[identity] += 1
        else:
            self.records = eligible_records
        identities = sorted({record["identity"].casefold() for record in self.records})
        self.identity_to_label = {identity: index for index, identity in enumerate(identities)}
        self.targets = [
            self.identity_to_label[record["identity"].casefold()] for record in self.records
        ]
        self.training = bool(training)
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        self.color_jitter = float(color_jitter)
        if not self.records:
            raise RuntimeError("Prepared DogFaceNet manifest has no eligible records")

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = _read_bgr(resolve_legacy_path(record["source_path"]))
        width, height = (int(value) for value in record["resized_size"])
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_path = resolve_legacy_path(record["mask_path"])
        compact_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if compact_mask is None:
            raise RuntimeError(f"Failed to read cached nose mask: {record['mask_path']}")
        nose_box = np.asarray(record["nose_roi_xyxy"], dtype=np.float32)
        face_box = np.asarray(record["face_roi_xyxy"], dtype=np.float32)
        nx1, ny1, nx2, ny2 = (int(round(value)) for value in nose_box)
        target_mask_size = (max(nx2 - nx1, 1), max(ny2 - ny1, 1))
        if compact_mask.shape[::-1] != target_mask_size:
            compact_mask = cv2.resize(
                compact_mask, target_mask_size, interpolation=cv2.INTER_NEAREST
            )
        full_mask = np.zeros((height, width), dtype=np.float32)
        full_mask[ny1:ny2, nx1:nx2] = compact_mask[: ny2 - ny1, : nx2 - nx1] / 255.0
        angle = float(record["roll_angle_radians"])
        if "viewpoint_signals" in record:
            view_signals = np.asarray(record["viewpoint_signals"], dtype=np.float32)
        elif "detection" in record:
            detection_data = record["detection"]
            view_signals = viewpoint_signals(
                FaceDetection(
                    bbox_xyxy=tuple(detection_data["bbox_xyxy"]),
                    confidence=float(detection_data["confidence"]),
                    landmarks_xy=tuple(
                        tuple(point) for point in detection_data["landmarks_xy"]
                    ),
                    class_id=int(detection_data.get("class_id", 0)),
                )
            )
        else:
            # Synthetic/legacy manifests may predate cached AnyFace metadata.
            view_signals = np.zeros(4, dtype=np.float32)

        if self.training and random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            full_mask = np.ascontiguousarray(full_mask[:, ::-1])
            face_box = np.asarray(
                (width - face_box[2], face_box[1], width - face_box[0], face_box[3]),
                dtype=np.float32,
            )
            nose_box = np.asarray(
                (width - nose_box[2], nose_box[1], width - nose_box[0], nose_box[3]),
                dtype=np.float32,
            )
            angle = -angle
            view_signals = view_signals.copy()
            view_signals[:3] *= -1.0
        if self.training and self.color_jitter > 0:
            alpha = random.uniform(1.0 - self.color_jitter, 1.0 + self.color_jitter)
            beta = random.uniform(-32.0, 32.0) * self.color_jitter
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(
                np.uint8
            )

        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float(),
            "nose_mask": torch.from_numpy(full_mask[None]),
            "face_box": torch.from_numpy(face_box),
            "nose_box": torch.from_numpy(nose_box),
            "roll_angle_radians": torch.tensor(angle, dtype=torch.float32),
            "quality_signals": torch.tensor(
                record["quality_signals"], dtype=torch.float32
            ),
            "viewpoint_signals": torch.from_numpy(view_signals),
            "branch_available": torch.tensor(
                record["branch_available"], dtype=torch.bool
            ),
            "target": torch.tensor(self.targets[index], dtype=torch.long),
            "identity": record["identity"],
            "source_path": record["source_path"],
        }


def collate_prepared_dogfacenet(samples: Sequence[dict], *, size_divisibility=32) -> dict:
    if not samples:
        raise ValueError("Cannot collate an empty DogFaceNet batch")
    divisor = max(int(size_divisibility), 1)
    max_height = max(sample["image"].shape[1] for sample in samples)
    max_width = max(sample["image"].shape[2] for sample in samples)
    padded_height = int(math.ceil(max_height / divisor) * divisor)
    padded_width = int(math.ceil(max_width / divisor) * divisor)
    images = torch.zeros((len(samples), 3, padded_height, padded_width), dtype=torch.float32)
    masks = torch.zeros((len(samples), 1, padded_height, padded_width), dtype=torch.float32)
    face_rois, nose_rois = [], []
    for index, sample in enumerate(samples):
        height, width = sample["image"].shape[-2:]
        images[index, :, :height, :width] = sample["image"]
        masks[index, :, :height, :width] = sample["nose_mask"]
        face_rois.append(torch.cat((torch.tensor([index]), sample["face_box"])))
        nose_rois.append(torch.cat((torch.tensor([index]), sample["nose_box"])))
    return {
        "images_0_255": images,
        "face_rois": torch.stack(face_rois).float(),
        "nose_rois": torch.stack(nose_rois).float(),
        "roll_angles_radians": torch.stack(
            [sample["roll_angle_radians"] for sample in samples]
        ),
        "nose_masks": masks,
        "quality_signals": torch.stack(
            [sample["quality_signals"] for sample in samples]
        ),
        "viewpoint_signals": torch.stack(
            [sample["viewpoint_signals"] for sample in samples]
        ),
        "branch_available": torch.stack(
            [sample["branch_available"] for sample in samples]
        ),
        "targets": torch.stack([sample["target"] for sample in samples]),
        "identities": [sample["identity"] for sample in samples],
        "source_paths": [sample["source_path"] for sample in samples],
    }


class PKBatchSampler(Sampler[list[int]]):
    """Sample P identities and K images per identity for metric learning."""

    def __init__(
        self,
        targets: Sequence[int],
        *,
        identities_per_batch: int,
        images_per_identity: int,
        steps: int,
        seed=42,
    ):
        self.groups: dict[int, list[int]] = defaultdict(list)
        for index, target in enumerate(targets):
            self.groups[int(target)].append(index)
        self.identities_per_batch = int(identities_per_batch)
        self.images_per_identity = int(images_per_identity)
        self.steps = int(steps)
        self.seed = int(seed)
        eligible = [
            identity
            for identity, indices in self.groups.items()
            if len(indices) >= self.images_per_identity
        ]
        if len(eligible) < self.identities_per_batch:
            raise ValueError(
                f"Need {self.identities_per_batch} eligible identities, got {len(eligible)}"
            )
        self.eligible = eligible

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.steps):
            identities = rng.sample(self.eligible, self.identities_per_batch)
            batch = []
            for identity in identities:
                batch.extend(rng.sample(self.groups[identity], self.images_per_identity))
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.steps
