"""Small contract tests for the strict unified ONNX runtime."""

import unittest

import numpy as np

from pet_id.unified_runtime import resolve_unified_provider, validate_raw_rgb_input


class UnifiedRuntimeTest(unittest.TestCase):
    def test_auto_prefers_cuda_only_when_both_layers_are_ready(self):
        self.assertEqual(
            resolve_unified_provider(
                "auto",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                torch_cuda_available=True,
            ),
            "cuda",
        )
        self.assertEqual(
            resolve_unified_provider(
                "auto",
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                torch_cuda_available=False,
            ),
            "cpu",
        )

    def test_explicit_cuda_never_silently_falls_back(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            resolve_unified_provider(
                "cuda",
                ["CPUExecutionProvider"],
                torch_cuda_available=True,
            )

    def test_explicit_cpu_requires_cpu_provider(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            resolve_unified_provider(
                "cpu",
                ["CUDAExecutionProvider"],
                torch_cuda_available=True,
            )

    def test_invalid_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "auto, cuda, cpu"):
            resolve_unified_provider(
                "tensorrt",
                ["CPUExecutionProvider"],
                torch_cuda_available=False,
            )

    def test_raw_rgb_contract_checks_without_transforming_pixels(self):
        source = np.arange(2 * 3 * 4 * 5, dtype=np.uint8).reshape(2, 3, 4, 5)
        actual = validate_raw_rgb_input(source)
        self.assertEqual(actual.dtype, np.float32)
        self.assertTrue(actual.flags.c_contiguous)
        np.testing.assert_array_equal(actual, source.astype(np.float32))

        with self.assertRaisesRegex(ValueError, "at least one"):
            validate_raw_rgb_input(np.empty((0, 3, 4, 5), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_raw_rgb_input(np.empty((1, 3, 0, 5), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "finite"):
            invalid = np.zeros((1, 3, 4, 5), dtype=np.float32)
            invalid[0, 0, 0, 0] = np.nan
            validate_raw_rgb_input(invalid)
        for invalid_value in (-1.0, 256.0):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, r"0\.\.255"):
                    invalid = np.zeros((1, 3, 4, 5), dtype=np.float32)
                    invalid[0, 0, 0, 0] = invalid_value
                    validate_raw_rgb_input(invalid)


if __name__ == "__main__":
    unittest.main()
