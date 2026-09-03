# Local prototype gallery

> Compatibility workflow: the package-building commands below target the old
> Semantic V3 descriptor space. The production-baseline UnifiedPetReID V3
> service uses the
> independent incremental gallery at
> `data/gallery_store/pet_api_gallery_unified_v3_v1`; enroll it through the
> API described in `docs/PET_API.md`. Descriptor files from these two spaces
> cannot be mixed.

This workflow adds new pet identities without fine-tuning the neural encoders on
a handful of images. The locked multimodal checkpoint produces descriptors;
gallery references are averaged per identity and L2-normalized into a safe
NumPy model package.

Run these commands from `src/Pet-ReID-IMAG`. Set `$Python` to the Python
interpreter that contains PyTorch and the selected ONNX Runtime package.

## 1. Deduplicate and split labeled identity folders

```powershell
& $Python tools\prepare_local_pet_gallery.py `
  --identity local-1=..\..\data\local_gallery\local-1 `
  --identity local-2=..\..\data\local_gallery\local-2 `
  --gallery-images-per-identity 2 `
  --output-dir ..\..\data\processed\pet-reid-imag\local_pet_gallery_v1
```

The split is fixed before inference. Exact SHA-256 duplicates are recorded but
not copied into the library. Original input folders are never changed.

## 2. Build the gallery model

```powershell
& $Python tools\build_pet_gallery_model.py `
  ..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\dataset_manifest.json `
  ..\..\models\selected\dogfacenet_semantic_v3_v1\model_final.pth `
  --config-file ..\..\models\selected\dogfacenet_semantic_v3_v1\config.yaml `
  --backend onnx `
  --onnx-model ..\..\models\selected\dogfacenet_semantic_v3_v1\onnx\pet_embedding.onnx `
  --production-only `
  --output-dir ..\..\models\selected\local_pet_gallery_semantic_v3_onnx_v1
```

The package contains locked-joint, frozen-fusion, nose-only, and face-only
reference descriptors so the same validation set can be used for ablation.

## 3. Validate the held-out split

```powershell
& $Python tools\validate_pet_gallery.py `
  --gallery-model ..\..\models\selected\local_pet_gallery_semantic_v3_onnx_v1\gallery_model.json `
  --manifest ..\..\data\processed\pet-reid-imag\local_pet_gallery_v1\dataset_manifest.json `
  --backend onnx --onnx-provider cuda --production-only `
  --output-dir ..\..\artifacts\evaluations\local_pet_gallery_validation
```

## 4. Identify arbitrary future images

```powershell
& $Python tools\validate_pet_gallery.py `
  ..\..\data\queries\inbox `
  --gallery-model ..\..\models\selected\local_pet_gallery_semantic_v3_onnx_v1\gallery_model.json `
  --backend onnx --onnx-provider cuda --production-only `
  --output-dir ..\..\artifacts\evaluations\new_image_run
```

For a labeled validation directory whose immediate subfolder names are gallery
identity names, add `--labels-from-parent`. Unlabeled runs still return Top-1,
runner-up, cosine scores, and margin, but do not claim accuracy.

This is closed-set retrieval: the script always ranks a known gallery identity.
Unknown-dog rejection requires a separately calibrated threshold and is not
enabled by this small two-identity set.
