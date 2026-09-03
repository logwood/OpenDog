# DogFaceNet joint800 ONNX embedding

This package exports the identity network only. AnyFace detection, SAM2 nose masking, EXIF handling, and rotated crop extraction remain application preprocessing.

Inputs are RGB float32 crops in the 0-255 range plus the quality, viewpoint, and branch-availability signals documented in `metadata.json`. The main `embedding` output is an L2-normalized 512-D descriptor intended for cosine gallery retrieval. The 800-class training classifier is intentionally not part of this deployment graph, so new identities can be registered without re-exporting the model.

Fusion mode: `shared_space_v2`.

See `validation.json` for ONNX checker and PyTorch/ONNX Runtime parity results.
