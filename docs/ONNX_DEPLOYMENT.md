# ONNX deployment

The production default is the raw-RGB end-to-end V3 graph
`models/selected/unified_pet_reid_v3_v1/onnx/e2e/unified_pet_reid.onnx`.
It has one dynamic input `[N,3,H,W]` and one `[N,512]` output. Centered
letterboxing, pixel normalization, learned geometry/crops, fusion, and L2
normalization are graph nodes; only image decoding/EXIF handling and the
transport BGR-to-RGB conversion remain outside ONNX. The older fixed-square
artifact in the parent `onnx/` directory is retained only for rollback.
The current development generation is the validated high-resolution V4
candidate in `models/selected/unified_pet_reid_v4_v1`; it has not yet replaced
the production pointer. See [`VERSIONING_CN.md`](VERSIONING_CN.md).

The commands below that mention the Semantic V3 multibranch pipeline are an
explicit compatibility workflow; they are not the default deployment.

The accepted semantic-residual v3 identity network is available at
`models/selected/dogfacenet_semantic_v3_v1/onnx/pet_embedding.onnx`. It emits a
512-dimensional joint embedding and keeps the legacy package beside it for
rollback. AnyFace detection, SAM 2 nose segmentation, EXIF handling, quality
signals, and rotated ROI extraction remain in the application pipeline. Only
the jointly trained identity network is replaced by ONNX Runtime.

## Environment

Install exactly one ONNX Runtime variant in an environment. This repository's
PyTorch build uses CUDA 12, so use the pinned CUDA 12-compatible package:

```powershell
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements-onnx-gpu.txt
```

`onnxruntime-gpu` 1.27 and newer PyPI wheels use CUDA 13 and do not match the
current PyTorch `cu126` environment. The runtime adapter verifies that the
requested provider is really active and refuses an entire-session silent
fallback to CPU.

## Extract descriptors from images

Run the following commands from `src/Pet-ReID-IMAG`.

```powershell
python tools/multimodal_inference.py D:\images `
  --config-file ..\..\models\selected\dogfacenet_semantic_v3_v1\config.yaml `
  --backend onnx `
  --onnx-model ..\..\models\selected\dogfacenet_semantic_v3_v1\onnx\pet_embedding.onnx `
  --onnx-provider cuda `
  --onnx-warmup-batches 1,4,8 `
  --output-dir ..\..\artifacts\evaluations\onnx_inference
```

The output summary records the active provider, model path, ONNX Runtime
version, and preprocessing device. Descriptor caches include the ONNX model in
their namespace, so PyTorch and ONNX cache entries cannot be mixed.

## Build a production gallery

```powershell
python tools/build_pet_gallery_model.py `
  ..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\dataset_manifest.json `
  ..\..\models\selected\dogfacenet_semantic_v3_v1\model_final.pth `
  --config-file ..\..\models\selected\dogfacenet_semantic_v3_v1\config.yaml `
  --backend onnx `
  --onnx-model ..\..\models\selected\dogfacenet_semantic_v3_v1\onnx\pet_embedding.onnx `
  --onnx-provider cuda `
  --production-only `
  --output-dir ..\..\models\selected\local_pet_gallery_semantic_v3_onnx_v1
```

`--production-only` stores the joint, nose, and face descriptors from the
locked ONNX model and omits the research-only frozen PyTorch ablations. The
builder binds the ONNX hash and source-checkpoint hash into the gallery model.

## Identify images against the gallery

```powershell
python tools/validate_pet_gallery.py ..\..\data\queries\inbox `
  --gallery-model ..\..\models\selected\local_pet_gallery_semantic_v3_onnx_v1\gallery_model.json `
  --backend onnx `
  --onnx-provider cuda `
  --production-only `
  --output-dir ..\..\artifacts\evaluations\new_image_run
```

When the gallery was built with ONNX, the query command automatically selects
the recorded ONNX backend and model unless explicitly overridden. An override
whose SHA-256 differs from the gallery's model is rejected before inference.

## Benchmark the complete pipeline

```powershell
python tools/benchmark_multimodal_runtime.py D:\images\sample.jpg `
  --backend onnx `
  --onnx-provider cuda `
  --warmup-runs 2 `
  --iterations 10 `
  --output ..\..\artifacts\evaluations\onnx_end_to_end_benchmark.json
```

The report separates AnyFace detection, SAM 2 segmentation, ROI plus identity
inference, and remaining preprocessing. Startup/model loading is reported
separately from warm latency.

This remains closed-set retrieval: every query is ranked against an existing
gallery identity. Unknown-dog rejection requires a separately calibrated score
and margin threshold.

For identities enrolled with several viewpoints, the API can retain
view-specific evidence at scoring time without changing the ONNX graph:

```text
/v1/identify?scoring_mode=reference_set&reference_top_k=3&reference_score_weight=0.4
```

The reference-set score is blended with the identity centroid and uses the mean
of the strongest references. Existing galleries already store one descriptor
per image, so enabling this mode does not require re-exporting or retraining a
model.
