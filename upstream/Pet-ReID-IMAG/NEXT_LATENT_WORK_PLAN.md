# Latent workspace next-work plan

Plan date: 2026-08-23

## Current conclusion

- The MHA workspace reached ROC-AUC 0.996559, but its final eight slots collapsed.
- Learned MESH reached ROC-AUC 0.996412. At the end, the learned initial slots had
  cosine similarity 0.9766 and the final slots had cosine similarity 1.0.
- Fixing the inter-slot mix gate at 0.05 did not prevent collapse. At iteration
  14800, the initial slots were still distinct (`raw_slot_cosine` 0.0110), but
  the final slots had cosine similarity 1.0.
- In that fixed-gate run, the C2 and C5 read-map cosine similarities were 0.9729
  and 0.9781. The slots therefore receive almost identical read vectors before
  the shared GRU contracts them further.
- The current implementation is not a literal reproduction of the original
  SA-MESH data flow. The reference model performs several refinements against
  the same input and learns both input and slot marginals. This project performs
  one refinement at each of four different backbone stages and fixes the slot
  marginal to be uniform. The GRU is a collapse amplifier, but is not yet proven
  to be the sole cause.
- The fixed-gate ablation used `MAX_EPOCH: 12`, which also shortened the cosine
  learning-rate schedule. Its validation curve is useful, but not a strictly
  controlled comparison with the 35-epoch learned-MESH run.

## Phase 1: prove where collapse happens without full training

1. Add stage-local diagnostics for C2-C5:
   - slot cosine before the read;
   - transport entropy before and after MESH optimization;
   - cosine similarity of the per-slot read vectors;
   - slot cosine after the GRU proposal;
   - slot cosine after inter-slot mixing and the FFN;
   - GRU proposal-to-state delta ratio.
2. Run those diagnostics on the retained early, collapse-onset, and best MESH
   checkpoints (`model_0001`, `model_0007`, and `model_best`).
3. Add a deterministic parity test for Sinkhorn/MESH against the reference
   equations. It must verify marginal error, decreasing transport entropy,
   finite gradients, and permutation equivariance.

Decision gate: do not start another full training run until the exact first
operation that pushes slot cosine upward is known.

## Phase 2: calibrate competition before changing the recurrent state

Use a fixed validation mini-batch and sweep MESH cost scale, temperature, inner
learning rate, and refinement count. This is a forward-only calibration, not a
training sweep. A candidate is acceptable only if it:

- materially lowers post-MESH entropy and read-map cosine;
- keeps Sinkhorn marginal error small and all values finite;
- does not create a single winner or empty slots;
- adds no more than about 15% step time over the current MESH path.

If the present parameters already pass these checks, keep them. If they do not,
fix the competition path first and record the selected scale in a dedicated
config rather than changing several values in the training command.

## Phase 3: one controlled recurrent-update ablation

Replace the hard GRU state replacement with a bounded residual proposal:

```text
proposal = GRU(read_update, old_slots)
new_slots = old_slots + alpha * (proposal - old_slots)
```

Start with a fixed `alpha = 0.10`. Keep the MESH parameters selected in Phase 2
and keep the inter-slot mix gate fixed at 0.05. Do not make `alpha` trainable in
this first run; otherwise another gate can silently grow and confound the result.

The experiment must retain the original 35-epoch scheduler, seed 20260811,
batch size, augmentation, losses, backbone freeze of 1000 iterations, and
validation split. Stop the process after the epoch-12 evaluation for the first
decision, while leaving `MAX_EPOCH` at 35 so the learning-rate trajectory remains
comparable.

Checkpoints for this ablation: best plus every two epochs, with a rolling limit
of two periodic files. The future launcher will be:

```powershell
.\scripts\train_mesh_residual_gru_ablation.ps1
```

## Pass and stop criteria

Continue beyond epoch 12 only when all of the following hold:

- final slot cosine remains below 0.95 at iteration 10000 and below 0.98 at
  iteration 14800;
- multiple slots have visibly different read maps on the same retained image set;
- validation ROC-AUC at iteration 14903 is at least 0.9939 (within 0.0005 of the
  scheduler-matched learned-MESH result of 0.99443);
- loss, gradients, memory, and step time remain healthy.

Stop early if slot cosine is already above 0.99 for three consecutive health
records, if MESH read vectors are already identical before the GRU, or if AUC
falls by more than 0.001 without a clear specialization benefit.

## Fallback if bounded GRU still collapses

Run exactly one second architecture ablation: remove the GRU and use a
LayerNorm-projected, fixed-LayerScale additive update. Do not add diversity loss
yet. An auxiliary cosine loss can force numerical separation without producing
meaningful roles, so it is reserved for later only if read maps are demonstrably
different but the state still contracts.

After a stable multi-slot mechanism exists, return to the 1:N evaluator and
gallery indexing work. Model-mechanism experiments and deployment evaluation
should remain separate until then.
