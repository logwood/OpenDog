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

Migrate the current persistent Semantic V3 Gallery without overwriting it:

```powershell
D:\CondaData\envs\torch312\python.exe `
  src/Pet-ReID-IMAG/tools/migrate_pet_gallery.py `
  --onnx-provider cuda --device cuda
```

The migration verifies every source and target image hash, identity, display
name, reference count, 512-D prototype, and model fingerprint in a sibling
staging directory. It publishes the target directory only after every check
passes. The current completed report is
`artifacts/runs/bifor/gallery_migration_v1/report.json`.

One-click full application startup with the migrated Gallery:

```powershell
.\start-pet-reid-bifor.cmd
# CPU fallback: .\start-pet-reid-bifor-cpu.cmd
```

Semantic V3 remains available through the original launchers for immediate
rollback. The shared stack launcher also accepts
`-Model semantic-v3-bifor` or `-Model semantic-v3` explicitly.

Do not reuse a Semantic V3 gallery: the low-rank projection defines a new
coordinate space even though both descriptors are 512-D. Re-enroll or re-encode
every gallery reference under this model fingerprint.

Validation details are in `onnx/validation.json`,
`artifacts/runs/bifor/onnx_protocol_validation_v1/evaluation.json`, and
`artifacts/runs/bifor/onnx_api_smoke_v1/report.json`. The stricter raw-image
online-preprocessing protocol is recorded in
`artifacts/runs/bifor/onnx_raw_runtime_validation_v1/evaluation.json`.
The migrated persistent Gallery achieved 4/4 Top-1 on held-out images in
`artifacts/runs/bifor/migrated_gallery_api_v1/report.json`; the isolated
CUDA Java → Python → Web acceptance run is
`artifacts/runs/live-stack-e2e/20260829-bifor-full-03/live-stack-smoke.json`.
The held-out report also records a detected body, confidence and body box for
all four queries.
