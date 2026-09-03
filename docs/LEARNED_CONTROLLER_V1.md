# Learned Evidence Controller experiment (historical revision V1)

> `V1` identifies this controller experiment only; it is unrelated to the
> UnifiedPetReID V4/V3 model generations and to the HTTP `/v1` contract.

## Status

The end-to-end experiment is complete, including an identity-disjoint protocol,
feature extraction, automatic counterfactual labels, controller training,
calibration, locked evaluation, and ONNX export.

This is an experimental result, not a production promotion. The learned
controller did not establish a reliable improvement over the strong raw BIFOR
confidence baseline.

## Method

The frozen BIFOR and MegaDescriptor models produce gallery scores, per-reference
support, and capture diagnostics. A small DeepSets encoder consumes the
unordered candidate set and a query-level context vector. Independent
one-vs-all heads estimate:

- BIFOR correctness
- Mega correctness
- success after consulting Mega
- success after recapture
- unknown-identity risk
- gallery stability
- expected Mega gain
- expected recapture gain
- temporal consistency

The same network is trained twice per episode: once with Mega evidence masked
and once with Mega evidence available. Before consultation, candidate selection
and every Mega-derived input are zeroed, so the controller cannot see future
expert evidence.

The deployed intermediate judgments are learned combinations of these heads:
match reliability, novelty risk, gallery support, expert expected gain,
recapture expected gain, and temporal consistency.

Actions are selected by maximum predicted success probability minus explicit
action cost. There is no hand-written match-score threshold. Action costs remain
explicit because they represent business preferences rather than visual
evidence.

## Locked protocol

- Protocol SHA-256:
  afa785fde9c109d5be2de6cf963fe3df808e32f96c241f2adc0798e1996c7560
- Four images per identity.
- Controller train: 72 identities, 288 images.
- Controller validation/calibration: 24 identities, 96 images.
- Locked known test: 28 identities, 112 images.
- Locked unknown test: 10 identities, 40 images.
- Pairwise identity overlap: zero.
- Overlap with the 821 BIFOR training identities: zero.
- Every identity already used by historical manifests or the historical Agent
  experiment V1 protocols
  was excluded.
- The historical unseen57 result was not used for training or calibration.

The 38 locked-test identities are now spent. They must not be reused as a
fresh blind set for a redesigned V2 controller.

## Results

The locked test contains 76 sequential episodes: 56 known and 20 unknown.

| Metric | Result |
|---|---:|
| BIFOR known Top-1 | 51/56, 91.07% |
| Learned policy autonomous coverage | 56/76, 73.68% |
| Learned policy autonomous accuracy | 48/56, 85.71% |
| Review rate | 20/76, 26.32% |
| Overall accuracy if review is correct | 68/76, 89.47% |
| Mega consultation rate | 0% |
| Recapture rate | 3/76, 3.95% |
| Direct unknown rejection rate | 0% |

At the same post-hoc test coverage of 56/76, selecting only by raw BIFOR Top-1
score gives 47/56 correct (83.93%). The learned policy gives 48/56, one fewer
error. This sample is far too small to interpret the one-case difference as a
reliable improvement.

### Learned-head diagnostics

| Head | Locked AUROC | Brier |
|---|---:|---:|
| BIFOR correct | 0.8196 | 0.1448 |
| Mega correct | 0.7730 | 0.2330 |
| Consult success | 0.7759 | 0.1497 |
| Recapture correct | 0.7271 | 0.1664 |
| Unknown | 0.7112 | 0.1565 |
| Gallery stable | 0.8973 | 0.2066 |
| Expert gain | 0.7089 | 0.0382 |
| Recapture gain | 0.8858 | 0.0343 |
| Temporal consistency | 0.7592 | 0.1637 |

The combined table contains both pre- and post-expert examples. For the actual
pre-expert decision, the learned unknown head has AUROC 0.6732.

### Strong-baseline audit

The raw BIFOR evidence generalizes better than the learned reliability heads:

| Task | Raw BIFOR baseline | Learned pre-expert head |
|---|---:|---:|
| Known vs unknown | Top-1 score AUROC 0.8670 | Unknown AUROC 0.6732 |
| BIFOR prediction correct | Top-1 score AUROC 0.9192 | Correctness AUROC 0.8196 |
| BIFOR prediction correct | Margin AUROC 0.9043 | Correctness AUROC 0.8196 |

Therefore V1 should not replace raw BIFOR confidence for novelty or correctness
ranking. The multi-task network overfits the 72 controller-training identities
despite early stopping at epoch 8.

### Why no Mega consultation or direct rejection occurred

- Mega fixed only 11 unique training episodes out of 576 (22 positive staged
  examples out of 1152). With a consultation cost of 0.04, not calling Mega
  is often the rational learned action.
- A review has assumed success 1.0 and cost 0.25, so its utility is 0.75.
  The calibrated unknown probability did not exceed review often enough to make
  direct rejection optimal.
- These are policy outcomes, not hard-coded score thresholds.

## ONNX verification

- Exported model size: 175,874 bytes.
- PyTorch checkpoint size: 180,277 bytes.
- Maximum absolute PyTorch/ONNX Runtime difference over a dynamic 2 by 7
  candidate batch: 1.7881393432617188e-07.
- ONNX output order is recorded in the checkpoint and report.

## Reproduction

Create or verify only the locked protocol:

    D:\CondaData\envs\torch312\python.exe src/Pet-ReID-IMAG/tools/train_evaluate_learned_controller.py --protocol-only

Run feature extraction, training, evaluation, and ONNX export:

    D:\CondaData\envs\torch312\python.exe src/Pet-ReID-IMAG/tools/train_evaluate_learned_controller.py

Features are committed row by row to the shared SQLite cache, so interrupted
extraction resumes without recomputing completed images.

## Recommended V2

1. Treat raw BIFOR score and margin as mandatory strong baselines and learn only
   a regularized residual correction.
2. Use identity-level cross-fitting on the 72/24 development identities instead
   of selecting a larger network from one validation split.
3. Constrain reliability and unknown heads monotonically with respect to the
   raw BIFOR score, or compare against a regularized logistic/GAM controller.
4. Oversample naturally occurring Mega-fix and recapture-fix episodes without
   duplicating identities across evaluation splits.
5. Select action costs or a risk-coverage budget using validation data only.
6. Collect additional four-view identities before claiming a new blind V2
   result; the current 38 test identities cannot be reset.

## Method references

- SelectiveNet, ICML 2019:
  https://proceedings.mlr.press/v97/geifman19a.html
- Learning to Defer, ICML 2020:
  https://proceedings.mlr.press/v119/mozannar20b.html
- Calibrated Learning to Defer with One-vs-All Classifiers, ICML 2022:
  https://proceedings.mlr.press/v162/verma22c.html
- Learning to Defer to Multiple Experts, AISTATS 2023:
  https://proceedings.mlr.press/v206/verma23a.html
- Deep Sets, NeurIPS 2017:
  https://papers.nips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
