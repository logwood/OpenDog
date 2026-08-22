# AnyFace + nose-print + PetFace ArcFace wiring

The multimodal path treats AnyFace and SAM 2 as frozen geometry providers. It
does not use detector features as identity evidence.

```text
full-resolution image
  -> AnyFace face box + 5 landmarks
     -> differentiable aligned face crop -> PetFace dog.pt -> 512-D face feature
     -> nose point/scale -> SAM 2 soft mask
        -> differentiable raw + masked nose crops -> IMAG -> 2048-D nose feature
  -> quality gate -> normalized 2560-D descriptor
```

The initial gate is `0.75 * nose_quality` versus `0.25 * face_quality`. A
missing branch is assigned zero weight. The gate itself is trainable.

## Inference

From the repository root:

```powershell
D:\CondaData\envs\torch312\python.exe tools\multimodal_inference.py `
  image_a.jpg image_b.jpg `
  --config-file configs\multimodal_inference.yaml `
  --output-dir logs\multimodal_run
```

The command writes:

- aligned face crops;
- raw and softly masked nose crops;
- binary masks and overlays;
- safe `.npz` descriptors and JSON metadata;
- all per-animal pair scores when multiple images or multiple animals are
  present.

The cache namespace includes every model/checkpoint timestamp and the fusion
settings. `--no-cache` forces a fresh full pass.

## Local end-to-end boundary

`LocalEndToEndPetIDModel` accepts full-resolution image tensors plus cached
AnyFace boxes/landmarks and SAM masks. ROI extraction uses
`affine_grid`/`grid_sample`, so gradients continue through both crops, the two
identity encoders, the quality gate, and the fused classification/triplet
losses. NMS and SAM mask decoding stay outside autograd.

For conservative adaptation, construct the training model with:

```python
model = build_local_identity_model(cfg, device="cuda", for_training=True)
```

Set `MULTIMODAL.NUM_CLASSES` to the local identity count. The default ArcFace
adaptation exposes only `layer4` and `fc`; its earlier ResNet stages and
BatchNorm statistics remain frozen. The model returns a `losses` dictionary
when `targets` are passed.

Full-image identity labels are required to train the fusion. The released
nose-only challenge data can still use the fallback nose branch, but it cannot
supervise the face branch or the quality gate.

## DogFaceNet alignment identities

The DogFaceNet alignment archive encodes the target identity in each filename
(`identity.original-name.jpg`) and provides left-eye, right-eye, and nose
coordinates in `labels.csv`. Prepare frozen geometry once:

```powershell
D:\CondaData\envs\torch312\python.exe tools\prepare_dogfacenet_alignment.py `
  ..\..\DogFaceNet_alignment `
  --archive ..\..\DogFaceNet_alignment.zip `
  --min-eye-distance 128 `
  --output-dir logs\dogfacenet_alignment_geometry
```

CRC matching uses the original ZIP metadata to associate Unicode filenames
that Windows extracted with mojibake. No source image is renamed. For images
containing multiple dogs, the AnyFace detection nearest to the CSV eye/nose
landmarks is the only animal assigned the filename identity.
EXIF orientation is applied to both the image and CSV coordinates before
matching, which is required for rotated phone photos.
On the current archive, the 128-pixel high-detail tier contains 1,230 usable
annotated images across 338 identities with at least two images each.

Train the locally end-to-end identity portion from resized full images and the
cached geometry:

```powershell
D:\CondaData\envs\torch312\python.exe tools\train_multimodal_dogfacenet.py `
  logs\dogfacenet_alignment_geometry\manifest.json `
  --steps 1000 --identities-per-batch 4 --images-per-identity 2
```

The training loader reconstructs the full-image nose mask, pads variable-size
images, and samples P identities by K images. AnyFace and SAM 2 remain frozen;
ROI crops, IMAG, the selected PetFace ArcFace tail, fusion gate, and local
classification/triplet heads participate in backpropagation.

CUDA training selects BF16 automatically when supported. FP16 is deliberately
avoided on these high-contrast nose crops because the IMAG branch can overflow;
every step aborts before an optimizer update if its loss or gradient norm is
not finite. Use `--no-amp` for the slower FP32 path.

Evaluate the trained fusion itself (rather than the unmodified base encoders)
on every same-identity and different-identity pair in a prepared manifest:

```powershell
python tools/evaluate_multimodal_dogfacenet.py `
  logs/dogfacenet_alignment_geometry/manifest.json `
  logs/multimodal_dogfacenet_train/model_final.pth `
  --output-dir logs/multimodal_dogfacenet_eval
```

This writes `evaluation.json` plus `pairs.csv`, including fused, nose, and face
cosine scores, pair AUC, leave-one-out rank-1 retrieval, and closed-set
classifier accuracy. Thresholds reported here are dataset diagnostics, not a
production unknown-dog threshold.

Omit the checkpoint to evaluate the frozen pretrained IMAG + PetFace ArcFace
combination without any local fine-tuning. The optional limits make a compact
visual sanity check:

```powershell
python tools/evaluate_multimodal_dogfacenet.py `
  logs/dogfacenet_alignment_geometry/manifest.json `
  --max-identities 5 --max-images-per-identity 3 `
  --visualization logs/frozen_pretrained_pairs.png
```

The ordinary image inference CLI can load the same trained fusion checkpoint:

```powershell
python tools/multimodal_inference.py image_a.jpg image_b.jpg `
  --identity-weights logs/multimodal_dogfacenet_train/model_final.pth
```

Its descriptor metadata then contains the checkpoint's top identity scores,
while pair comparisons use the trained fused descriptor. The checkpoint class
names belong to its training dataset; identities registered after training
should still be resolved through an embedding gallery with unknown rejection.
