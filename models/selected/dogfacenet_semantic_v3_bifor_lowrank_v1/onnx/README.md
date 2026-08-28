# Semantic V3 + BIFOR low-rank ONNX

The main output is a 512-D L2-normalized descriptor combining 92% semantic-v3 nose/face and 8% frozen headless BIFOR body information. The BIFOR flip-TTA and locked rank-500 projection are inside ONNX.

AnyFace, SAM2, and the frozen dog-body detector remain preprocessing. Use `pet_id.bifor_onnx_runtime` for the raw-image pipeline. Existing semantic-v3 galleries must be re-encoded before retrieval. The projector was selected on a historical BF16 feature cache; this package intentionally uses production FP32 ONNX, so secondary metrics can differ slightly.
