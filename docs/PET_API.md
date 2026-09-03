# Pet enrollment and identification API

This service defaults to the production-baseline `unified_pet_reid_v3_v1`
embedding network. The current development generation is the validated V4
candidate, but it has not been promoted to the default deployment. See
[`VERSIONING_CN.md`](VERSIONING_CN.md) for the distinction between model
generation and deployment role. The V3 production-baseline ONNX graph maps raw
RGB `[N,3,H,W]` pixels directly to
L2-normalized 512-dimensional descriptors; it accepts dynamic spatial sizes with
dynamic spatial dimensions and performs letterboxing, normalization, geometry,
fusion, and L2 normalization internally. Runtime inference does not load
AnyFace, SAM2, a body detector, or another identity model. The locked development result is
`157/211` versus the fixed parent's `156/210`; the single blind attempt
passed at `158/204` against fixed thresholds `69/104`. Training identities
are not a fixed production label set: newly enrolled pets are represented by
the normalized mean of their reference descriptors.

The default storage path is `data/gallery_store/pet_api_gallery_unified_v3_v1`.
It is created on first start and is intentionally separate from the older
Semantic V3, BIFOR, Agent, and temporary galleries. If the directory is absent,
that means no unified V3 identities have been enrolled yet; it is not a model
failure. For the current-development dynamic high-resolution V4 backend, use
[`UNIFIED_HIGHRES_V4.md`](UNIFIED_HIGHRES_V4.md) and its separate
`pet_api_gallery_unified_v4_v1` directory.

## Install

Use the same Python environment that contains PyTorch and ONNX Runtime CUDA:

```powershell
Set-Location .\src\Pet-ReID-IMAG
$Python = if ($env:PET_REID_PYTHON) { $env:PET_REID_PYTHON } else { "python" }
& $Python -m pip install -r requirements-api.txt
```

All commands below run from `src/Pet-ReID-IMAG`. The root
`start-pet-reid.cmd` and `start-pet-reid-cpu.cmd` launch the complete
Python, Java and web stack automatically. In every standalone PowerShell
session, define `$Python` once before copying later command blocks:

```powershell
$Python = if ($env:PET_REID_PYTHON) { $env:PET_REID_PYTHON } else { "python" }
```

## Start the service

The following command creates the independent UnifiedPetReID SQLite gallery.
It deliberately does not import a gallery encoded by an older model.

```powershell
$env:PET_REID_API_KEY = "replace-with-a-long-random-secret"

& $Python tools\serve_pet_api.py `
  --backend unified-onnx `
  --onnx-provider cuda `
  --onnx-model ..\..\models\selected\unified_pet_reid_v3_v1\onnx\e2e\unified_pet_reid.onnx `
  --storage-dir ..\..\data\gallery_store\pet_api_gallery_unified_v3_v1
```

The ONNX file is the only model artifact required at runtime. The packaged
`model_final.pth` is retained solely for training provenance and reproducible
export; neither the API nor the quick-start scripts read it during unified
inference.

The server listens on `127.0.0.1:8000` by default. Interactive OpenAPI testing
is available at <http://127.0.0.1:8000/docs>. A non-loopback bind such as
`--host 0.0.0.0` requires an API key unless the unsafe override is explicitly
provided. Run a single worker because the CUDA pipeline is loaded once and GPU
inference is serialized safely inside the process.

## Enroll images

Use a stable application-level identifier. Two or more clear images with one
pet per image are recommended. JPEG, PNG, WebP and BMP are accepted.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/pets/dog-001/images" `
  -H "X-API-Key: $env:PET_REID_API_KEY" `
  -F "display_name=豆豆" `
  -F "files=@D:\pet-images\front.jpg" `
  -F "files=@D:\pet-images\left.jpg" `
  -F "files=@D:\pet-images\right.jpg"
```

Exact duplicate uploads to the same pet are skipped. The same image cannot be
assigned to two different pets. Each image should visibly contain one primary
pet: the unified graph consumes the complete letterboxed image and does not run
an independent detector that could reliably reject group photos.

## Identify an image

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/identify?top_k=3" `
  -H "X-API-Key: $env:PET_REID_API_KEY" `
  -F "file=@D:\queries\query.jpg"
```

When an identity has references from several viewpoints, the default
`scoring_mode=centroid` compares the query with the normalized mean descriptor.
To retain view-specific evidence, opt into the reference-set scorer:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/identify?top_k=3&scoring_mode=reference_set&reference_top_k=3&reference_score_weight=0.4" `
  -H "X-API-Key: $env:PET_REID_API_KEY" `
  -F "file=@D:\queries\query.jpg"
```

`reference_set` blends the centroid cosine with the mean of the strongest
`reference_top_k` per-image cosines. It uses a top-k mean instead of a raw
maximum, so one accidental high-similarity reference cannot decide the result
alone. The same parameters are accepted by `POST /v1/batches`; the selected
mode and diagnostics are saved in the batch/history result. This is an
inference-time policy change: it does not alter model weights, the ONNX graph,
or the stored descriptor schema, and existing galleries can use it immediately.

For a learned query-conditioned matcher, start the service with a trained
reference-set checkpoint (PyTorch `.pth` or exported `.onnx`):

```powershell
python src/Pet-ReID-IMAG/tools/serve_pet_api.py `
  --reference-matcher-checkpoint artifacts/runs/reference_set_matcher/experiment/model_best.pth
```

Then select it explicitly for a request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/identify?top_k=3&scoring_mode=learned_reference_set" `
  -H "X-API-Key: $env:PET_REID_API_KEY" `
  -F "file=@D:\queries\query.jpg"
```

`learned_reference_set` receives the query and every enrolled reference
descriptor for each identity, then chooses the useful views conditionally. The
service keeps `centroid` as its default, so loading a matcher does not change
existing calls; the learned mode is also available to batch requests. The
matcher is recorded in `/health` and history diagnostics, while the gallery
remains bound to the image-encoder fingerprint.

If an identity has more references than the matcher checkpoint accepts in one
forward pass, the runtime scores deterministic chunks and aggregates the two
strongest chunk scores. The response diagnostics expose `chunk_count` and
`chunk_scores`; all enrolled views remain available to the matcher.

Without a threshold the result is explicitly marked `closed_set_top1` and
always ranks the closest enrolled pet. Unknown-pet rejection must be calibrated
on representative known and unknown images before using, for example:

```text
/v1/identify?top_k=3&match_threshold=0.35&minimum_margin=0.08
```

Do not treat the example numbers as production thresholds.

## Other endpoints

- `GET /health` — backend/provider and gallery counts; no API key required.
- `GET /v1/pets` — list enrolled pets.
- `GET /v1/pets/{pet_id}` — pet details and reference image IDs.
- `PATCH /v1/pets/{pet_id}` — change the display name without changing the stable ID.
- `GET /v1/pets/{pet_id}/images/{image_id}` — download a stored reference.
- `DELETE /v1/pets/{pet_id}/images/{image_id}` — remove one reference; deleting
  the final reference also removes the pet.
- `DELETE /v1/pets/{pet_id}` — remove a pet and all references.
- `GET /v1/history` — paginated comparison history and filters.
- `GET /v1/history/{history_id}` — saved result, model/gallery snapshot and diagnostics.
- `PATCH /v1/history/{history_id}/review` — mark a result correct, incorrect or uncertain.
- `POST /v1/batches` — enqueue up to 1000 queries for serialized background inference.
- `GET /v1/batches/{batch_id}` — progress, metrics and per-image results.
- `GET /v1/batches/{batch_id}/results.csv` — export the complete batch result set.
- `GET /v1/hard-cases` — rejected, low-margin, incorrect or manually flagged
  comparisons. Branch-only reasons apply only to explicit legacy backends.
- `GET /v1/gallery/backup` and `POST /v1/gallery/restore` — model-bound ZIP backup and
  non-destructive merge restore.

The Java gateway exposes history and gallery operations under `/v1/**`. Batch,
hard-case and backup endpoints are exposed under `/v1/admin/**` and require the
Java-side `X-Admin-Key`; the quick-start script writes its per-run random value to
`artifacts/workspace_logs/quick_start/admin-key.txt` and removes it on stop.

## Test

Fast, model-free API and storage tests:

```powershell
& $Python -m unittest tests.test_gallery_api -v
```

Reusable real ONNX/CUDA enrollment-and-identification smoke test. It uses a
temporary gallery and does not modify production state:

```powershell
& $Python tools\smoke_test_pet_api.py `
  --enroll local-1=..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\images\gallery\local-1\001_mmexport1787622883216.jpg `
  --enroll local-1=..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\images\gallery\local-1\002_mmexport1787622883370.jpg `
  --enroll local-2=..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\images\gallery\local-2\001_mmexport1787622179349.jpg `
  --enroll local-2=..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\images\gallery\local-2\002_mmexport1787622181189.jpg `
  --query ..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\images\validation\local-1\003_mmexport1787622883567.jpg `
  --expected-pet-id local-1 `
  --output ..\..\artifacts\evaluations\pet_api_unified_v3_cuda_smoke.json
```

For `--backend unified-onnx`, the smoke report must contain
`unified_single_graph_observed=true`; `--require-query-dual-branch` is only
for explicit Semantic V3/BIFOR compatibility tests.

Full regression suite:

```powershell
& $Python -m unittest discover -s tests -v
```

### Real Java -> Python -> web stack test

From the workspace root, the following command creates a new isolated gallery
under `artifacts/runs/live-stack-e2e/<run-id>/`, starts the CPU ONNX service,
Java gateway and frontend, exercises enrollment/identification/history/review,
administrator batches/CSV/hard cases and gallery backup/merge-restore, then
always stops the three services in `finally`:

```powershell
.\scripts\test-live-stack.ps1 -Provider cpu
```

Use `-Provider cuda` for the CUDA execution provider. The command refuses to
reuse an existing run directory or an already-running stack, so it cannot
silently test or modify the production gallery. The JSON report, batch CSV
and gallery ZIP remain in that run directory for inspection. The default test
images come from the prepared DogFaceNet alignment dataset.

Check the real serving contract after starting the server:

```powershell
curl.exe http://127.0.0.1:8000/health
```

The health response must report `backend=onnxruntime-unified`,
`single_graph=true`, `raw_spatial_input=true`, `external_models=[]`, and model
SHA-256 `2db41b25d770eb285cd313f4e81a1f77c2017e70d827c0b9a1e48cf74edaf8a5`.

The persistent state lives under `--storage-dir`: SQLite stores identities and
descriptors transactionally, while original images are stored under
content-addressed SHA-256 paths. The database is bound to the embedding model
hash and refuses to mix descriptors created by another checkpoint or ONNX file.

