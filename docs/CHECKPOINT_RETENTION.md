# Checkpoint retention and cleanup status

Last inventory refresh: 2026-08-28

## Current state

No checkpoint or ONNX file was deleted or moved during the workspace
reorganization. The earlier 2026-08-23 cleanup remains historical context:
it removed 62 redundant checkpoints (39.22 GiB) after its own review and left
19 checkpoints (12.77 GiB) at that time. Multimodal work after that cleanup
created additional retained models.

The current reproducible inventory is generated with:

```powershell
python .\scripts\generate_workspace_metadata.py
```

It currently records 65 checkpoint/ONNX files totaling 24,372,065,243 bytes:

- KEEP: 46
- REVIEW: 12
- QUARANTINE candidates: 7
- duplicate SHA-256 groups: 7

The machine-readable inventory is
`artifacts/reports/checkpoint_inventory.json`; the human review is
`artifacts/reports/checkpoint_cleanup_preview.md`. A QUARANTINE label is only
a recommendation. None of those seven files has been moved.

The retrospective run inventory is
`artifacts/reports/legacy_run_inventory.json`. It covers all 160 legacy run
directories and all 48 legacy checkpoint files. Twenty-eight directories
contain checkpoints and every one has an explicitly selected checkpoint and
selection basis; one additional checkpoint stored directly under the legacy
root is represented as a shared historical evaluation artifact. Validation
reports zero missing or duplicate inventory paths. Historical fields that
cannot be reconstructed safely are kept as null and named in each run's
`missing_historical_fields`, rather than being invented.

## Retained diagnostic checkpoints

These pre-cleanup paths now live below `artifacts/runs/legacy/`:

- `modern_latent_workspace_s101_224_d192/model_0001.pth`: early MHA reference.
- `modern_latent_workspace_s101_224_d192/model_best.pth`: MHA best, epoch 31,
  ROC-AUC 0.996559.
- `modern_mesh_workspace_s101_224_d192_balanced/model_0001.pth`: learned-MESH
  differentiated phase.
- `modern_mesh_workspace_s101_224_d192_balanced/model_0007.pth`: learned-MESH
  collapse-onset reference.
- `modern_mesh_workspace_s101_224_d192_balanced/model_best.pth`: learned-MESH
  best, epoch 27, ROC-AUC 0.996412.
- `ablation_mesh_mix_fixed005_s101_224_d192/model_0001.pth`: fixed-gate
  differentiated phase.
- `ablation_mesh_mix_fixed005_s101_224_d192/model_0007.pth`: fixed-gate
  collapse reference.
- `ablation_mesh_mix_fixed005_s101_224_d192/model_best.pth`: fixed-gate best,
  epoch 11, ROC-AUC 0.995154.

Baseline/released ensemble references also remain below
`artifacts/runs/legacy/`: `retrained_s101_224/model_recent_0.pth` and the
four `s101_224`, `s101_256`, `s101_288`, `s200_224`
`model_final.pth` files.

Deployment models and their metadata live under `models/selected/`. The
`dogfacenet_semantic_v3_bifor_lowrank_v1` package is a validated research
candidate, not the default deployment. Its descriptor space is incompatible
with the default Semantic V3 gallery, so references must be enrolled again
before it is used. The new package and its frozen body detector account for the
two additional retained binary files in this inventory refresh.
Pretrained assets live under `models/pretrained/`. Their binary payloads are
ignored by Git; `models/registry.json` records selected-model identity,
source, hash and role.

All five repository recovery bundles under `archive/git/2026-08-28/` pass
`git bundle verify`; each bundle now has a colocated verification log,
including `nested/BIFOR-bundle-verify.txt`.

## Review procedure

Before moving any candidate:

1. regenerate the inventory and require an idempotent second run;
2. review every REVIEW/QUARANTINE rationale and duplicate-hash group;
3. confirm no active config, manifest, resume pointer or document references it;
4. obtain explicit human approval;
5. move candidates to `archive/quarantine/<date>/checkpoints/` and record the
   original-to-quarantine mapping;
6. rerun the full Git, model, inference, API, Java and frontend verification.

Moving files to quarantine on the same disk does not free disk space. Permanent
deletion or transfer to another disk is a separate, explicitly authorized
operation.
