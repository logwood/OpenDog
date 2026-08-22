# Upstream sources checked on 2026-08-09

- Repository: https://github.com/muzishen/Pet-ReID-IMAG
- Paper: https://arxiv.org/abs/2205.15934
- Author assets (data + weights): https://drive.google.com/drive/folders/1_7pdSRTvD_XdTu8z0MxrM9PDoEuX-tjf?usp=drive_link
- Challenge: CVPR 2022 Biometrics Workshop Pet Biometric Challenge / Tianchi
- ResNeSt-101 ImageNet weights: https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest101-22405ba7.pth
- ResNeSt-200 ImageNet weights: https://github.com/zhanghang1989/ResNeSt/releases/download/weights_step1/resnest200-75117900.pth

The upstream repository is Apache-2.0. See `UPSTREAM_LICENSE_APACHE_2.txt`.

Key upstream facts verified:
- README reports a ResNeSt multi-scale solution, Phase A 91.7% / Phase B 86.27%, and provides data + weights through cloud-drive links.
- Test entry point: `bash predict.sh`, which calls `pet_id/train_net.py --config-file ./configs/fusion_submit.yaml --eval-only --commit`.
- Four feature branches used by fusion: s101_224, s101_256, s101_288, s200_224.
