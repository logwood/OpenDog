# Pet enrollment and identification API

This service defaults to the accepted `dogfacenet_semantic_v3_v1` embedding
network. It was retrained on all 800 development identities and accepted only
after a locked, fresh-blind test on 64 unseen identities (128 queries). Training
identities are not a fixed production label set: newly enrolled pets are
represented by mean, L2-normalized 512-dimensional reference descriptors.

## Install

Use the same Python environment that contains PyTorch and ONNX Runtime CUDA:

```powershell
D:\CondaData\envs\torch312\python.exe -m pip install -r requirements-api.txt
```

## Start the service

The following command creates an incremental SQLite gallery and imports the
already validated two-pet prototype gallery. Repeating the command is safe; the
seed import is idempotent.

```powershell
$env:PET_REID_API_KEY = "replace-with-a-long-random-secret"

D:\CondaData\envs\torch312\python.exe tools\serve_pet_api.py `
  --backend onnx `
  --onnx-provider cuda `
  --storage-dir models\pet_api_gallery_semantic_v3_v1 `
  --seed-gallery-model models\local_pet_gallery_semantic_v3_onnx_v1\gallery_model.json
```

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
assigned to two different pets. Enrollment requires exactly one detected pet
by default so a group photo cannot silently contaminate an identity prototype.

## Identify an image

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/identify?top_k=3" `
  -H "X-API-Key: $env:PET_REID_API_KEY" `
  -F "file=@D:\queries\query.jpg"
```

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
- `GET /v1/hard-cases` — rejected, low-margin, low-quality, branch-conflict or manually
  flagged comparisons.
- `GET /v1/gallery/backup` and `POST /v1/gallery/restore` — model-bound ZIP backup and
  non-destructive merge restore.

The Java gateway exposes history and gallery operations under `/v1/**`. Batch,
hard-case and backup endpoints are exposed under `/v1/admin/**` and require the
Java-side `X-Admin-Key`; the quick-start script writes its per-run random value to
`logs/quick_start/admin-key.txt` and removes it on stop.

## Test

Fast, model-free API and storage tests:

```powershell
D:\CondaData\envs\torch312\python.exe -m unittest tests.test_gallery_api -v
```

Reusable real ONNX/CUDA enrollment-and-identification smoke test. It uses a
temporary gallery and does not modify production state:

```powershell
D:\CondaData\envs\torch312\python.exe tools\smoke_test_pet_api.py `
  --enroll local-1=data\local_pet_gallery_v1\images\gallery\local-1\001_mmexport1787622883216.jpg `
  --enroll local-1=data\local_pet_gallery_v1\images\gallery\local-1\002_mmexport1787622883370.jpg `
  --enroll local-2=data\local_pet_gallery_v1\images\gallery\local-2\001_mmexport1787622179349.jpg `
  --enroll local-2=data\local_pet_gallery_v1\images\gallery\local-2\002_mmexport1787622181189.jpg `
  --query data\local_pet_gallery_v1\images\validation\local-1\003_mmexport1787622883567.jpg `
  --expected-pet-id local-1 `
  --output logs\pet_api_semantic_v3_cuda_smoke.json
```

The local phone samples currently exercise the API contract and the detector's
nose-only fallback; they are not proof that fusion ran. For a strict regression,
add `--require-query-dual-branch`. The following prepared DogFaceNet example
fails unless AnyFace produced a real detection, both nose and face branches
were available, both fusion weights were positive, and Top-1 was correct:

    D:\CondaData\envs\torch312\python.exe tools\smoke_test_pet_api.py `
      --enroll "pet-a=..\..\DogFaceNet_alignment\images\180&tit=Wolf.00854.jpg" `
      --enroll "pet-a=..\..\DogFaceNet_alignment\images\180&tit=Wolf.00855.jpg" `
      --enroll "pet-b=..\..\DogFaceNet_alignment\images\231&tit=Dorl.01084.jpg" `
      --enroll "pet-b=..\..\DogFaceNet_alignment\images\231&tit=Dorl.01086.jpg" `
      --query "..\..\DogFaceNet_alignment\images\180&tit=Wolf.01229.jpg" `
      --expected-pet-id pet-a `
      --require-query-dual-branch `
      --output logs\pet_api_semantic_v3_dual_branch_smoke.json

The accepted run observed branch availability `[true, true]`, fusion weights
`[0.1177, 0.8823]`, and returned `pet-a`.

Full regression suite:

```powershell
D:\CondaData\envs\torch312\python.exe -m unittest discover -s tests -v
```

Real ONNX/CUDA smoke test after starting the server:

```powershell
curl.exe http://127.0.0.1:8000/health

curl.exe -X POST "http://127.0.0.1:8000/v1/identify?top_k=2" `
  -H "X-API-Key: $env:PET_REID_API_KEY" `
  -F "file=@data\local_pet_gallery_v1\images\validation\local-1\003_mmexport1787622883567.jpg"
```

The persistent state lives under `--storage-dir`: SQLite stores identities and
descriptors transactionally, while original images are stored under
content-addressed SHA-256 paths. The database is bound to the embedding model
hash and refuses to mix descriptors created by another checkpoint or ONNX file.

