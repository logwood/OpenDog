# Local BIFOR layout

This directory is a checkout of the official BIFOR repository at commit
`47b27892e0062a31e7ba0c894fa9caec7928172c`. The upstream source is kept
unchanged; `config.workspace.yaml` is the only workspace-specific runtime
configuration.

## Assets

Paths below are relative to this directory:

- Dataset: `../../data/BIFOR/YT-BB-Dog`
  - `train`: 2,000 identities, 19,932 images
  - `test`: 723 identities, 7,104 images
  - total: 2,723 identities, 27,036 images
- Random-background companion set:
  `../../data/BIFOR/YT-BB-Dog_random_bckg/YT-BB-Dog`
  - `test`: 723 identities, 7,064 images
- BIFOR feature extractor f(2):
  `../../models/pretrained/BIFOR/f2/bifor.pth`
- Background network f(1):
  `../../models/pretrained/BIFOR/f1/f(1)/background_net/background_net.ckpt`

The original archives remain at `../../data/YT-BB-dog.zip` and
`../../models/pretrained/Weights.zip`.

## Interface relevant to fusion

`models/Bifor.py` defines a ConvNeXt-Small with its classifier removed. It
accepts ImageNet-normalized RGB images resized to 224 x 224 and returns one
768-dimensional feature vector per image. The checkpoint under `f2` is the
feature extractor to use for inference or downstream fusion. The `f1`
checkpoint is a background-similarity network used by the BIFOR training
sampler; it is not a second inference branch.

## Official evaluation

Create an environment matching `requirements.txt`, then run from this
directory:

```powershell
python main.py --config=config.workspace.yaml
```

`config.workspace.yaml` evaluates the official f(2) checkpoint on the
YT-BB-Dog test split. It uses `num_workers: 0` for reliable Windows execution
and does not emit the optional nearest-neighbour JSON file.
