# Agent V1 / MegaDescriptor model record

Agent V1 keeps the existing BIFOR 512-D descriptor as the primary identity
space and adds `BVRA/MegaDescriptor-B-224` as a frozen, independent 1024-D
body-shape expert. Features are stored and compared in separate SQLite
namespaces; only scores are fused.

MegaDescriptor details:

- Architecture: `swin_base_patch4_window7_224`.
- Checkpoint SHA-256:
  `655791158167f07773a890368f7db2fced85d569b9bccbbe7e5194e5051e2459`.
- Runtime preprocessing: RGB BIFOR body crop, direct bicubic resize to
  224×224, ImageNet mean/std.
- Loading: offline only. The checkpoint's old timm Swin stage names are
  deterministically converted, then all tensors are loaded with
  `strict=True`.
- Calibration: `zero_shot_monotonic_v1`; the fused score is not a
  probability. The current two-identity gallery is too small to fit honest
  Platt or isotonic calibration.

## License restriction

The MegaDescriptor weights are licensed **CC BY-NC 4.0**. They are restricted
to non-commercial use unless the rights holder grants separate permission.
This applies even though other code and models in the workspace use different
licenses.

Authoritative references:

- https://huggingface.co/BVRA/MegaDescriptor-B-224
- https://github.com/WildlifeDatasets/wildlife-tools
- https://openaccess.thecvf.com/content/WACV2024/html/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.html

## Formal accuracy evaluation (2026-08-29)

The deployed raw-image path was evaluated with two references per identity.
Features were extracted by the production BIFOR ONNX runtime and the frozen
MegaDescriptor encoder, then cached by image and model SHA-256. Thresholds were
fit only on the calibration split.

The old joint100 20-way blind split is now classified as an
**in-distribution diagnostic**, not unseen-identity evidence: 19 of its 20
identities later entered the joint800 training set. It produced 59/60 BIFOR,
46/60 MegaDescriptor, and 60/60 Agent Top-1, but must not be cited as
cross-identity generalization.

The strict result filters every identity seen by either the 100-identity BIFOR
fusion training stage or the 800-identity Semantic V3 training stage. This
leaves 57 identity-disjoint test identities, 228 images, and 114 queries:

| Method | Top-1 | Top-5 | MRR / mAP |
|---|---:|---:|---:|
| BIFOR-only | 110/114 (96.4912%) | 113/114 (99.1228%) | 0.974591 |
| MegaDescriptor-only | 86/114 (75.4386%) | 100/114 (87.7193%) | 0.817503 |
| Agent score fusion | 106/114 (92.9825%) | 113/114 (99.1228%) | 0.956656 |

Against BIFOR, Agent V1 fixed one query and regressed five. The paired exact
McNemar p-value is 0.21875; this does not establish a significant difference,
but the direction and the separate calibration result both argue against
making the current zero-shot fusion the default.

At the threshold selected for 95% known recall on calibration data, the strict
open-set test produced:

| Method | AUROC | FAR | FRR | Unknown rejection |
|---|---:|---:|---:|---:|
| BIFOR-only | 0.964035 | 10.0000% | 10.5263% | 90.0000% |
| MegaDescriptor-only | 0.807895 | 98.3333% | 2.6316% | 1.6667% |
| Agent score fusion | 0.950585 | 25.0000% | 7.8947% | 75.0000% |

Decision: keep BIFOR as the production identity score. Retain MegaDescriptor
features and the Agent path as an experimental expert for guarded rescue,
reranking, or future calibrated gating; do not use the current roughly
half-weight score fusion as an unconditional default.

Reproducible artifacts:

- Strict protocol/report:
  `artifacts/runs/agent_v1/formal_unseen57_v1/protocol.json` and
  `artifacts/runs/agent_v1/formal_unseen57_v1/report.json`.
- Strict protocol canonical SHA-256:
  `a38203d6bd8fdc1a0a8d241978296b0d086c10ff3ef51779f912b3972ef71670`.
- In-distribution diagnostic:
  `artifacts/runs/agent_v1/formal_joint100_20_v1/report.json`.
- Evaluator:
  `src/Pet-ReID-IMAG/tools/evaluate_agent_formal_protocol.py`.
