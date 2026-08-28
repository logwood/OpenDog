"""Unit tests for gallery packaging helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pet_id.gallery import (
    load_exif_oriented_bgr,
    load_gallery_model,
    normalized_prototypes,
    sha256_file,
)


class GalleryTest(unittest.TestCase):
    def test_prototypes_are_identity_means_and_l2_normalized(self):
        references = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        prototypes = normalized_prototypes(references, labels, 2)
        np.testing.assert_allclose(np.linalg.norm(prototypes, axis=1), 1.0)
        np.testing.assert_allclose(prototypes[0], np.asarray([2**-0.5, 2**-0.5]))
        np.testing.assert_allclose(prototypes[1], np.asarray([-1.0, 0.0]))

    def test_phone_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rotated.jpg"
            image = Image.new("RGB", (40, 20), "red")
            exif = Image.Exif()
            exif[274] = 6
            image.save(path, exif=exif)
            loaded = load_exif_oriented_bgr(path)
            self.assertEqual(loaded.shape[:2], (40, 20))

    def test_gallery_archive_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.npz"
            np.savez_compressed(features, values=np.ones((1, 2), dtype=np.float32))
            metadata = {
                "features_file": features.name,
                "features_sha256": sha256_file(features),
            }
            model = root / "model.json"
            model.write_text(json.dumps(metadata), encoding="utf-8")
            loaded_metadata, arrays = load_gallery_model(model)
            self.assertEqual(loaded_metadata, metadata)
            np.testing.assert_array_equal(arrays["values"], np.ones((1, 2)))
            features.write_bytes(features.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                load_gallery_model(model)


if __name__ == "__main__":
    unittest.main()
