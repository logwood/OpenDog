# encoding: utf-8
"""Tests for DogFaceNet filename labels and prepared local-E2E batches."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from pet_id.dogfacenet_alignment import (
    PKBatchSampler,
    PreparedDogFaceNetDataset,
    apply_exif_orientation_to_points,
    build_alignment_index,
    collate_prepared_dogfacenet,
    dogfacenet_identity_from_filename,
    match_annotated_target,
)
from pet_id.localization import FaceDetection


def _detection(nose, confidence=0.9):
    nose_x, nose_y = nose
    return FaceDetection(
        bbox_xyxy=(nose_x - 20, nose_y - 20, nose_x + 20, nose_y + 25),
        confidence=confidence,
        landmarks_xy=(
            (nose_x - 10, nose_y - 8),
            (nose_x + 10, nose_y - 8),
            (nose_x, nose_y),
            (nose_x - 8, nose_y + 10),
            (nose_x + 8, nose_y + 10),
        ),
    )


class DogFaceNetAlignmentTest(unittest.TestCase):
    def test_exif_rotation_maps_csv_points_to_display_pixels(self):
        points = np.asarray(((10, 20), (30, 40)), dtype=np.float32)
        transformed = apply_exif_orientation_to_points(
            points,
            encoded_size=(100, 60),
            orientation=6,
        )
        np.testing.assert_allclose(transformed, np.asarray(((39, 10), (19, 30))))

    def test_identity_is_encoded_in_filename(self):
        self.assertEqual(dogfacenet_identity_from_filename("272100.272100_0.jpg"), "272100")
        self.assertEqual(dogfacenet_identity_from_filename("dog5.DSCF1315.JPG"), "dog5")
        self.assertEqual(dogfacenet_identity_from_filename("B.Atis.B.Atis1.jpg"), "B.Atis")
        self.assertEqual(
            dogfacenet_identity_from_filename("B.Dömpi.B.Doempi1.jpg"), "B.Dömpi"
        )

    def test_index_reads_filename_labels_and_landmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            names = ("dog5.frame1.jpg", "B.Atis.B.Atis1.jpg")
            for name in names:
                cv2.imwrite(str(image_root / name), np.zeros((40, 50, 3), dtype=np.uint8))
            with (root / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("", "filename", "lex", "ley", "rex", "rey", "nox", "noy"))
                writer.writerow((0, names[0], 10, 12, 30, 12, 20, 24))
                writer.writerow((1, names[1], 11, 12, 31, 12, 21, 24))
            records, report = build_alignment_index(root)
            self.assertEqual(report["resolved"], 2)
            self.assertEqual({record.identity for record in records}, {"dog5", "B.Atis"})
            self.assertEqual(records[0].nose, (20.0, 24.0))

    def test_landmarks_select_the_annotated_dog(self):
        annotation = np.asarray(((20, 20), (40, 20), (30, 31)), dtype=np.float32)
        wrong = _detection((90, 80), confidence=0.99)
        target = _detection((30, 31), confidence=0.75)
        match = match_annotated_target((wrong, target), annotation)
        self.assertIsNotNone(match)
        self.assertIs(match.detection, target)
        self.assertLess(match.normalized_nose_distance, 1e-6)

    def test_prepared_dataset_collates_variable_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for index, (width, height, identity) in enumerate(
                ((50, 40, "dog-a"), (32, 48, "dog-b"))
            ):
                image_path = root / f"image_{index}.jpg"
                mask_path = root / f"mask_{index}.png"
                cv2.imwrite(str(image_path), np.full((height, width, 3), 127, dtype=np.uint8))
                cv2.imwrite(str(mask_path), np.full((10, 12), 255, dtype=np.uint8))
                records.append(
                    {
                        "source_path": str(image_path),
                        "identity": identity,
                        "resized_size": [width, height],
                        "mask_path": str(mask_path),
                        "face_roi_xyxy": [2, 3, width - 2, height - 2],
                        "nose_roi_xyxy": [5, 6, 17, 16],
                        "roll_angle_radians": 0.0,
                        "quality_signals": [0.8, 0.7, 0.9, 0.8, 0.6, 0.7],
                        "branch_available": [True, True],
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            dataset = PreparedDogFaceNetDataset(manifest_path)
            batch = collate_prepared_dogfacenet([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["images_0_255"].shape), (2, 3, 64, 64))
            self.assertEqual(tuple(batch["nose_masks"].shape), (2, 1, 64, 64))
            torch.testing.assert_close(batch["face_rois"][:, 0], torch.tensor([0.0, 1.0]))
            self.assertEqual(batch["targets"].unique().numel(), 2)

    def test_pk_sampler_contains_positives_for_each_identity(self):
        targets = [0, 0, 0, 1, 1, 2, 2]
        sampler = PKBatchSampler(
            targets,
            identities_per_batch=2,
            images_per_identity=2,
            steps=3,
            seed=7,
        )
        for batch in sampler:
            selected = [targets[index] for index in batch]
            counts = {target: selected.count(target) for target in set(selected)}
            self.assertEqual(sorted(counts.values()), [2, 2])


if __name__ == "__main__":
    unittest.main()
    apply_exif_orientation_to_points,
