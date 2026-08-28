# DogFaceNet joint800 v1

This package contains the model locked before the first 200-way blind feature
extraction.

## Protocol

- 800 training identities, 4 images each (3,200 images)
- 200 identity-disjoint test identities, 4 images each (800 images)
- 2 gallery images averaged into one normalized prototype per test identity
- 2 held-out queries per test identity (400 queries)
- every query competes against all 200 identity prototypes
- exact SHA-256 and identity overlap between train and test: zero
- checkpoint selection: fixed final checkpoint after 5,000 steps; no test-based
  checkpoint selection

## Results

| Model / branch | Top-1 | Top-5 | AUC |
|---|---:|---:|---:|
| Frozen fusion baseline | 336/400 (84.00%) | 381/400 (95.25%) | 0.988841 |
| Nose branch | 260/400 (65.00%) | 317/400 (79.25%) | 0.953306 |
| Face branch | 376/400 (94.00%) | 397/400 (99.25%) | 0.998405 |
| Locked trained fusion | **380/400 (95.00%)** | **397/400 (99.25%)** | **0.997980** |

Against frozen fusion, the trained fusion corrected 46 queries and regressed on
2, for a net gain of 44 correct queries. Of its 20 Top-1 errors, 17 still place
the true identity at ranks 2-5 and 3 place it below rank 5; the worst true rank
is 11.

## Integrity and scope

The checkpoint SHA-256 is
`4f76e7f57bd9683193e3fdfa7d117f9f7663eac0a7d0ba9d3b02217fc74e3ffa`.
See `lock_record.json` and `blind_completion.json` for the signed-off protocol
paths, hashes, and complete metrics.

This demonstrates identity-disjoint, closed-set retrieval within the
DogFaceNet-alignment data domain. It does not by itself prove unknown-identity
rejection, cross-dataset performance, or resistance to near-duplicate/同一拍摄批次
leakage beyond exact-file SHA-256 deduplication.
