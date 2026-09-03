# DogFaceNet semantic fusion v3 v1

This package is the accepted 512-dimensional semantic-residual fusion model.
It is installed beside, and does not overwrite, `dogfacenet_joint800_v1`.

## Locked evidence

- Fresh-blind protocol: 64 unseen identities,
  128 held-out queries.
- Legacy fused Top-1/Top-5: 96.8750% /
  100.0000%.
- Semantic-v3 fused Top-1/Top-5: 96.8750% /
  100.0000%.
- Development cross-identity nose-conflict Top-1: 91.5% legacy versus
  96.5% semantic v3.
- Checkpoint SHA-256: `abe38784b93655936156c86eabf4d89dd09e9d05fa6fa20d321110e3ecd878d6`.

`model_lock.json`, `blind_completion.json`, and `deployment_record.json`
contain the complete hashes and decision record.

## ONNX deployment

- Model: `onnx/pet_embedding.onnx`
- Dynamic input batch: 1 through 8
- Output embedding: `[N, 512]`
- ONNX SHA-256:
  `21a2e2543b34ebd09426cba6bb1ff2c9dc0651b15d5186453e91ecbcf55d532a`
- PyTorch-wrapper maximum error: 0
- ONNX Runtime maximum error: `2.98e-7`
- Batch-8 nearest-neighbor Top-1 parity: exact

The old 3072-dimensional v1 gallery is not compatible with this model. Build
or enroll a separate v3 gallery; model fingerprints prevent accidental mixing.

## Integration validation

- Python API CUDA/ONNX dual-branch smoke: passed; query branches [true, true],
  weights 0.1177 / 0.8823, expected Top-1 pet-a.
- Java Spring gateway unit/integration tests at package lock time: 10/10 passed.
- Current workspace Java regression suite: 13/13 passed.
- Current workspace Python regression suite: 97/97 passed.
- Real Java -> Python -> CUDA ONNX multipart path: passed with 2 identities and
  4 reference images in the separate
  data/gallery_store/pet_api_gallery_semantic_v3_v1 gallery.
- Audit reports: artifacts/runs/legacy/pet_api_semantic_v3_dual_branch_smoke.json and
  artifacts/runs/legacy/java_gateway_semantic_v3_e2e/e2e_result.json.
