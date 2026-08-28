# Upstream sources checked on 2026-08-09

- Repository: https://github.com/muzishen/Pet-ReID-IMAG
- Paper: https://arxiv.org/abs/2205.15934
- Author assets (data + weights): https://drive.google.com/drive/folders/1_7pdSRTvD_XdTu8z0MxrM9PDoEuX-tjf?usp=drive_link
- Challenge: CVPR 2022 Biometrics Workshop Pet Biometric Challenge / Tianchi
- ResNeSt-101 ImageNet weights: https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth
- ResNeSt-200 ImageNet weights: https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest200-75117900.pth

The upstream repository is Apache-2.0. See `UPSTREAM_LICENSE_APACHE_2.txt`.

## Repository identities preserved during workspace consolidation

The workspace was consolidated under the outer `logwood/OpenDog` repository on
2026-08-28. The original repositories and baselines are:

- Pet-ReID-IMAG: `https://github.com/muzishen/Pet-ReID-IMAG.git`, commit
  `7a131552ccea575a70ef4f9d4dc8948687798485`.
- AnyFace: `https://github.com/IS2AI/AnyFace.git`, commit
  `bed844d23be03334f7fc12b2c4fb3ba2ac530bab`.
- PetFace: `https://github.com/mapooon/PetFace.git`, commit
  `69e7be7a82bece98dfaf832ad79d3c80f1a844f9`.
- SAM 2: `https://github.com/facebookresearch/sam2.git`, commit
  `2b90b9f5ceec907a1c18123530e92e794ad901a4`.

Complete verified bundles, working-tree patches, staged patches and untracked
file lists are stored under `archive/git/2026-08-28/`. The original nested Git
metadata is recoverably quarantined under
`archive/quarantine/2026-08-28/inner-git/`; it is not part of the source tree.
The local source differs from the baselines through the compatibility,
multimodal, joint-fusion, ONNX, gallery, API, Java and frontend work recorded in
the outer repository and the snapshot inventories.

Key upstream facts verified:
- README reports a ResNeSt multi-scale solution, Phase A 91.7% / Phase B 86.27%, and provides data + weights through cloud-drive links.
- Test entry point: `bash predict.sh`, which calls `pet_id/train_net.py --config-file ./configs/fusion_submit.yaml --eval-only --commit`.
- Four feature branches used by fusion: s101_224, s101_256, s101_288, s200_224.
