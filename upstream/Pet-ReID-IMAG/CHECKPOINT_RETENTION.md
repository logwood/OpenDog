# Checkpoint retention

Cleanup date: 2026-08-23

Cleanup result: removed 62 redundant checkpoints (39.22 GiB). Nineteen
checkpoints remain across all workstreams (12.77 GiB).

## Latent experiments

- `logs/modern_latent_workspace_s101_224_d192/model_0001.pth`: early MHA reference.
- `logs/modern_latent_workspace_s101_224_d192/model_best.pth`: MHA best, epoch 31, ROC-AUC 0.996559.
- `logs/modern_mesh_workspace_s101_224_d192_balanced/model_0001.pth`: learned-MESH differentiated phase.
- `logs/modern_mesh_workspace_s101_224_d192_balanced/model_0007.pth`: learned-MESH collapse-onset reference.
- `logs/modern_mesh_workspace_s101_224_d192_balanced/model_best.pth`: learned-MESH best, epoch 27, ROC-AUC 0.996412.
- `logs/ablation_mesh_mix_fixed005_s101_224_d192/model_0001.pth`: fixed-gate differentiated phase.
- `logs/ablation_mesh_mix_fixed005_s101_224_d192/model_0007.pth`: fixed-gate collapse reference.
- `logs/ablation_mesh_mix_fixed005_s101_224_d192/model_best.pth`: fixed-gate best, epoch 11, ROC-AUC 0.995154.

## Baselines and released ensembles

- `logs/retrained_s101_224/model_recent_0.pth`: latest locally retrained plain baseline, epoch 24.
- `logs/s101_224/model_final.pth`
- `logs/s101_256/model_final.pth`
- `logs/s101_288/model_final.pth`
- `logs/s200_224/model_final.pth`

Multimodal/DogFaceNet checkpoints are a separate active workstream and were intentionally excluded from this cleanup.

Smoke checkpoints and redundant periodic/final copies were removed. Their configs, metrics, TensorBoard events, and text logs remain available.
