import importlib.util
from pathlib import Path
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_pretrained_body_semantics.py"
)
SPEC = importlib.util.spec_from_file_location("_pretrained_body_semantics", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PretrainedBodySemanticsTest(unittest.TestCase):
    def test_selects_dog_box_containing_target_face(self):
        boxes = torch.tensor(
            [
                [0.0, 0.0, 40.0, 40.0],
                [45.0, 45.0, 100.0, 100.0],
                [40.0, 40.0, 105.0, 105.0],
            ]
        )
        labels = torch.tensor([18, 18, 1])
        scores = torch.tensor([0.99, 0.80, 0.999])

        box, score = MODULE.select_target_dog_box(
            boxes,
            labels,
            scores,
            [60.0, 60.0, 80.0, 80.0],
            dog_label=18,
            score_threshold=0.5,
        )

        self.assertEqual(box, [45.0, 45.0, 100.0, 100.0])
        self.assertAlmostEqual(score, 0.8, places=5)

    def test_body_retrieval_uses_two_gallery_records_per_identity(self):
        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [1.0, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.1, 1.0],
            ]
        )
        identities = ["a"] * 4 + ["b"] * 4
        paths = [f"image-{index}.jpg" for index in range(8)]

        result = MODULE.evaluate_branch(features, identities, paths, 2)

        self.assertEqual(result["gallery_identities"], 2)
        self.assertEqual(result["query_records"], 4)
        self.assertEqual(result["top1_correct"], 4)
        self.assertEqual(result["top1_accuracy"], 1.0)

    def test_imagenet_dog_range_is_118_classes(self):
        categories = [f"class-{index}" for index in range(151)]
        categories += ["Chihuahua"]
        categories += [f"breed-{index}" for index in range(116)]
        categories += ["Mexican hairless"]

        start, end = MODULE.dog_breed_class_range(categories)

        self.assertEqual(start, 151)
        self.assertEqual(end, 269)


if __name__ == "__main__":
    unittest.main()
