# Local prototype gallery

This workflow adds new pet identities without fine-tuning the neural encoders on
a handful of images. The locked multimodal checkpoint produces descriptors;
gallery references are averaged per identity and L2-normalized into a safe
NumPy model package.

## 1. Deduplicate and split labeled identity folders

```powershell
D:\CondaData\envs\torch312\python.exe tools\prepare_local_pet_gallery.py `
  --identity local-1=D:\Pet-ReID-IMAG_repro_attempt_2026-08-09\1 `
  --identity local-2=D:\Pet-ReID-IMAG_repro_attempt_2026-08-09\2 `
  --gallery-images-per-identity 2 `
  --output-dir data\local_pet_gallery_v1
```

The split is fixed before inference. Exact SHA-256 duplicates are recorded but
not copied into the library. Original input folders are never changed.

## 2. Build the gallery model

```powershell
D:\CondaData\envs\torch312\python.exe tools\build_pet_gallery_model.py `
  data\local_pet_gallery_v1\dataset_manifest.json `
  logs\multimodal_protocol32_selected\model_final.pth `
  --config-file configs\multimodal_inference.yaml `
  --output-dir models\local_pet_gallery_v1
```

The package contains locked-joint, frozen-fusion, nose-only, and face-only
reference descriptors so the same validation set can be used for ablation.

## 3. Validate the held-out split

```powershell
D:\CondaData\envs\torch312\python.exe tools\validate_pet_gallery.py `
  --gallery-model models\local_pet_gallery_v1\gallery_model.json `
  --manifest data\local_pet_gallery_v1\dataset_manifest.json `
  --output-dir logs\local_pet_gallery_v1_validation
```

## 4. Identify arbitrary future images

```powershell
D:\CondaData\envs\torch312\python.exe tools\validate_pet_gallery.py `
  D:\new-images `
  --gallery-model models\local_pet_gallery_v1\gallery_model.json `
  --output-dir logs\new_image_run
```

For a labeled validation directory whose immediate subfolder names are gallery
identity names, add `--labels-from-parent`. Unlabeled runs still return Top-1,
runner-up, cosine scores, and margin, but do not claim accuracy.

This is closed-set retrieval: the script always ranks a known gallery identity.
Unknown-dog rejection requires a separately calibrated threshold and is not
enabled by this small two-identity set.
