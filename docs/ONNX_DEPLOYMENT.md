# ONNX CUDA deployment

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
