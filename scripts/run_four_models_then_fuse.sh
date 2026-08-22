#!/usr/bin/env bash
set -euo pipefail
# Run from the upstream Pet-ReID-IMAG repository root after placing author weights/data.
# Copy the supplied upstream_configs/*.yaml into ./configs if your clone does not already contain them.
python3 pet_id/train_net.py --config-file ./configs/s101_224_submit.yaml --eval-only --save-features
python3 pet_id/train_net.py --config-file ./configs/s101_256_submit.yaml --eval-only --save-features
python3 pet_id/train_net.py --config-file ./configs/s101_288_submit.yaml --eval-only --save-features
python3 pet_id/train_net.py --config-file ./configs/s200_submit.yaml --eval-only --save-features
# Upstream final command. Apply patches/evaluator_gallery_fix.diff first, or use fuse_and_score.py.
python3 pet_id/train_net.py --config-file ./configs/fusion_submit.yaml --eval-only --commit
