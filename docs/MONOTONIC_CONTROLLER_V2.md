# Monotonic Evidence Controller experiment (historical revision V2)

> `V2` identifies this controller experiment only; it is unrelated to the
> UnifiedPetReID V4/V3 model generations and to the HTTP `/v1` contract.

## Status

V2 is implemented, evaluated, and exported to ONNX. It keeps BIFOR and
MegaDescriptor frozen and replaces the V1 DeepSets MLP with monotonic logistic
cores plus a strongly regularized linear residual.

This is an experimental checkpoint, not a production promotion. The repaired
controller has a useful selective-policy result, but it does not beat raw BIFOR
on either open-set ranking or overall correctness ranking. Mega consultation
also remains unsupported by the available positive examples.

The 110 fresh test identities are spent. The first locked execution exposed an
implementation bug, and the same identities were then evaluated once after the
causal fix. The repaired numbers below are therefore a post-fix diagnostic, not
a new blind result. No further model or cost tuning was performed from those
numbers.

## Method

The controller consumes 38 scalar measurements from frozen BIFOR,
MegaDescriptor, gallery support, body/crop quality, illumination, and viewpoint
evidence. It predicts seven intermediate judgments:

- BIFOR correctness
- MegaDescriptor correctness
- unknown-identity probability
- expected MegaDescriptor gain
- expected recapture gain
- gallery stability
- temporal consistency

BIFOR Top-1 and margin have non-negative contributions to the BIFOR-correct
logit and non-positive contributions to the unknown logit. When MegaDescriptor
is available, its Top-1 and margin have non-negative contributions to the
Mega-correct logit. Other measurements enter only through a linear residual.
Mega-derived measurements are masked before the expert is called.

Actions maximize learned success probability minus explicit business cost.
The costs used here are 0.04 for consultation, 0.02 for rejecting an unknown,
and 0.25 for human review. They are not visual match thresholds.

`log_gallery_size` remains in the 38-value deployment schema but is masked from
every residual head. The controller fit episodes have fixed gallery cardinality,
so a gallery-size effect is not identifiable from this protocol.

## Protocol

- Protocol SHA-256:
  `5d038b5de31330bc3133cf18a99a2a6da778e8684536ed4357e1e5b04eb7be4d`
- Controller fit: 80 identities, four images per identity.
- Calibration: 16 identity-disjoint identities, four images per identity.
- Fresh diagnostic known set: 80 identities, two gallery images plus one query.
- Fresh diagnostic unknown set: 30 identities, one query per identity.
- Pairwise identity overlap between all four splits: zero.
- Overlap with 821 BIFOR training identities: zero.
- All V1 controller/test identities and historical protocol identities were
  excluded before selecting the 110 fresh identities.
- The three-image diagnostic set cannot evaluate recapture gain or temporal
  consistency; those targets are explicitly masked.

Four-fold identity-level cross-validation selected weight decay 0.001. Its
mean validation BCE was 0.34569, compared with 0.35402, 0.35387, and 0.35078
for the other candidates. Final training stopped at epoch 37 with calibration
BCE 0.40461.

## Gallery-size failure and fix

The first execution produced fold BCE values from 677 to 3147, constant 0/1
probabilities, AUROC 0.5 for every evaluable head, and 100% human review. This
was not a model-capacity result.

`log_gallery_size` was numerically constant in the fit split, but float32
variance calculation produced a standard deviation of only `2.48e-5` rather
than exactly zero. Calibration used a smaller gallery. Standardization therefore
amplified the split difference by tens of thousands, and an unidentified random
linear coefficient saturated every logit. Masking that unidentifiable column
reduced fold BCE to 0.338-0.366 and restored non-constant predictions.

A regression test now verifies that changing gallery size alone cannot alter
V2 logits. The complete failed artifacts are preserved with the suffix
`pre_fix_gallery_size_bug` in the V2 artifact directory.

## Post-fix diagnostic results

The diagnostic set contains 110 episodes: 80 known and 30 unknown.

| Metric | Result |
|---|---:|
| BIFOR known Top-1 | 73/80, 91.25% |
| MegaDescriptor known Top-1 | 46/80, 57.50% |
| Autonomous coverage | 85/110, 77.27% |
| Autonomous accuracy | 78/85, 91.76% |
| Autonomous errors | 7 |
| Human review | 25/110, 22.73% |
| Accept BIFOR | 67/110, 60.91% |
| Reject unknown | 18/110, 16.36% |
| MegaDescriptor consultation | 0/110, 0% |
| Unknown rejection recall | 14/30, 46.67% |
| Known false rejection | 4/80, 5.00% |
| Accuracy if review is correct | 103/110, 93.64% |

At the same post-hoc coverage, ranking only by raw BIFOR Top-1 yields 70/85
correct (82.35%), versus 78/85 (91.76%) for the learned policy. Eight fewer
errors is material on this sample, but this baseline is a one-sided selective
accept rule rather than a separately calibrated two-sided accept/reject policy.
Together with the spent test identities, this prevents a generalization claim.

### Learned-head diagnostics

| Head | Pre-expert AUROC | Post-expert AUROC | Pre-expert Brier |
|---|---:|---:|---:|
| BIFOR correct | 0.9430 | 0.9408 | 0.0926 |
| Mega correct | 0.8505 | 0.9667 | 0.1601 |
| Unknown | 0.9408 | 0.9421 | 0.1025 |
| Expert gain | 0.8073* | 0.7890* | 0.0135 |
| Gallery stable | 0.8981 | 0.9012 | 0.1269 |

`*` Expert-gain AUROC has only one positive episode and is not reliable.
Recapture-gain and temporal-consistency test metrics are intentionally absent.

### Strong-baseline audit

| Task | Raw BIFOR | Learned pre-expert head | Outcome |
|---|---:|---:|---|
| Known vs unknown | Top-1 AUROC 0.9504 | Unknown AUROC 0.9408 | raw wins |
| Prediction correct, all episodes | Top-1 AUROC 0.9552 | Correctness AUROC 0.9430 | raw wins |
| Prediction correct, known only | Top-1 AUROC 0.9159 | Correctness AUROC 0.9178 | 0.002 difference; inconclusive |

The monotonic constraints prevent the severe V1 regression, but the learned
heads still do not improve the strongest raw rankings. The policy improvement
comes from combining correctness and novelty into accept/reject/review actions,
not from a uniformly better confidence score.

## MegaDescriptor and recapture limits

The fit split contains only 26 positive staged expert-gain rows, corresponding
to 13 unique episodes. Calibration has zero expert-gain positives, and the
diagnostic test has one. The rational learned action is therefore never to pay
for MegaDescriptor, even though the post-call Mega-correct AUROC of 0.9667 shows
that choosing between experts after a call is learnable.

The current three-image test protocol has no independent recapture image.
Recapture and temporal heads are retained for the future four-view data path but
must not be interpreted as evaluated here.

## ONNX verification

- Input: dynamic batch by 38 raw scalar evidence values.
- Output: dynamic batch by seven probabilities in the recorded output order.
- Random verification batch: 17 by 38.
- Maximum absolute PyTorch/ONNX Runtime error:
  `7.152557373046875e-07`.
- Mean absolute error: `5.0397673589941405e-08`.
- Checkpoint SHA-256:
  `d2686408cc4e2a5d2d7942df6eb12f77cce155920122810eacfd99fba151bedb`.
- ONNX SHA-256:
  `8aa75093ea410a1cef24b84c7d086931a844df95a0515b521f84b6bb799633ec`.

## Validation completed

- Protocol canonical hash validation passed.
- Feature-cache rerun: 654/654 hits, zero encodes.
- Python compilation passed for the V2 model, trainer, and tests.
- Four manual regression tests passed: no pre-expert Mega leakage, BIFOR/unknown
  monotonicity, gallery-size masking, and consult-then-accept-Mega action flow.
- `git diff --check` reported no whitespace errors in the working changes.

## Artifacts and reproduction

- `artifacts/runs/agent_v2/monotonic_controller_v2/protocol.json`
- `artifacts/runs/agent_v2/monotonic_controller_v2/report.json`
- `artifacts/runs/agent_v2/monotonic_controller_v2/controller_v2.pt`
- `artifacts/runs/agent_v2/monotonic_controller_v2/controller_v2.onnx`
- `artifacts/runs/agent_v2/monotonic_controller_v2/test_decisions.json`

Protocol verification only:

    D:\CondaData\envs\torch312\python.exe src/Pet-ReID-IMAG/tools/train_evaluate_monotonic_controller_v2.py --protocol-only

Full cached reproduction:

    D:\CondaData\envs\torch312\python.exe src/Pet-ReID-IMAG/tools/train_evaluate_monotonic_controller_v2.py

## Next valid experiment

Do not tune on these 110 identities and call the result blind. A valid V3
requires newly collected identities, preferably with at least four independent
views per dog. The data collection should deliberately enrich cases where
MegaDescriptor fixes BIFOR and where recapture changes the decision. Until then,
the deployable conservative choice is to keep raw BIFOR as the primary model,
use the controller for structured evidence and review routing, and leave paid
expert consultation disabled by the learned policy.
