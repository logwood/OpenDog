# Semantic V3 + BIFOR low-rank v1

This versioned package integrates the locked 8% BIFOR body residual into the
current Semantic V3 ONNX descriptor. The public image/API input remains one dog
image and the main output remains a 512-D L2-normalized descriptor.

The ONNX graph has seven pre-cropped inputs because body localization stays
outside the graph, alongside AnyFace and SAM2. Use the provided runtime adapter
instead of feeding the graph directly when starting from raw images.

Raw-image inference:

```powershell
python src/Pet-ReID-IMAG/tools/bifor_multimodal_inference.py path/to/dog.jpg
```

Existing API server:

```powershell
python src/Pet-ReID-IMAG/tools/serve_pet_api.py `
  --backend onnx-bifor `
  --config-file models/selected/dogfacenet_semantic_v3_bifor_lowrank_v1/config.yaml `
  --onnx-model models/selected/dogfacenet_semantic_v3_bifor_lowrank_v1/onnx/pet_embedding.onnx `
  --storage-dir artifacts/runtime/pet_api_gallery_semantic_v3_bifor_lowrank_v1
```

Do not reuse a Semantic V3 gallery: the low-rank projection defines a new
coordinate space even though both descriptors are 512-D. Re-enroll or re-encode
every gallery reference under this model fingerprint.

Validation details are in `onnx/validation.json`,
`artifacts/runs/bifor/onnx_protocol_validation_v1/evaluation.json`, and
`artifacts/runs/bifor/onnx_api_smoke_v1/report.json`. The stricter raw-image
online-preprocessing protocol is recorded in
`artifacts/runs/bifor/onnx_raw_runtime_validation_v1/evaluation.json`.
