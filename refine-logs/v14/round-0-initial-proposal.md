# V14 initial proposal — Fold-Robust Retrieval-Regret Router

## 1. Problem statement

V8 Phase-A proved that the frozen pretrained-tail CNN, Transformer and Mamba
residual experts are complementary at query level, but V13 failed to learn a
deployable policy from its pointwise nine-way utility target. The new V13
target-learnability diagnostic shows that this target is nearly uniform
(normalized entropy 0.999832; mean maximum probability 0.11528 versus 0.11111
uniform), has a median Top-1/Top-2 utility gap of only 0.0003407, and changes
slot ordering across folds (rank correlations -0.50 to 0.40).

The falsifiable hypothesis is:

> The remaining Router bottleneck is the pointwise action-distillation
> objective, which discards the relational retrieval structure and permits one
> training fold to be sacrificed. A Router trained directly on fold-local
> cross-camera retrieval regret, with worst-fold aggregation, will transfer
> more reliably than V13's utility-KL without changing experts, fusion energy,
> data, seed, or deployment inputs.

## 2. Existing prerequisites retained unchanged

V14 is not a new backbone version. It reuses the already implemented and tested
structural corrections:

- RGB/NIR/TIR share flip, padding and crop geometry; RandomErasing remains
  modality-specific only after alignment;
- exact frozen Signal 3072D baseline prefix;
- V8 Phase-A pretrained-tail CNN, Transformer and Mamba residual experts;
- matched-token residual construction;
- hierarchical weights
  `P(modality|x) * P(expert|modality,x)` with missing modalities hard-zeroed;
- exact-prefix blockwise fusion at the already frozen bounded `alpha=0.2`;
- unchanged corruption-response quality objective;
- all-fit deployment features as Router input and identity-OOF teacher
  embeddings only as train-time supervision.

HFER remains disabled in this qualification stage. It may be enabled only
after a deployable Router proves useful; otherwise exchange would confound the
single tested hypothesis.

## 3. Single main mechanism

### 3.1 Fold-local differentiable retrieval risk

For each source fold `f`, the Router predicts query-local joint weights
`w_i` from the frozen all-fit deployment features. Those weights compose the
actual deployed embedding from the identity-OOF teacher baseline and residual
bank:

```text
z_i(w_i) = Normalize([z0_i, 0.2 * ||z0_i|| * Normalize(w_i * delta_i)])
```

Distances are never computed between two teacher folds because their features
come from different OOF generators. Within one fold, for query `i`, define the
hardest cross-camera positive and nearest negative:

```text
d_pos(i) = max  ||z_i - z_j||_2, y_j=y_i and cam_j!=cam_i
d_neg(i) = min  ||z_i - z_k||_2, y_k!=y_i
```

The fold risk is the parameter-free smooth ranking violation:

```text
R_f(w) = mean_i softplus(d_pos(i) - d_neg(i)).
```

This is the batch-hard retrieval geometry used by ReID, applied to the whole
registered fold rather than to a pointwise pseudo-label. Embeddings are L2
normalized exactly as in evaluation.

### 3.2 Worst-fold regret aggregation

For the Router that holds out fold `h`, the two remaining source folds are
`a,b`. On each source fold, construct the fixed-policy comparator using the
slot selected from the training identities exactly as V13 did. Define:

```text
G_f(w) = R_f(w) - stop_gradient(R_f(w_fixed)).
L_identity = max(G_a(w), G_b(w)).
```

The fixed term changes which fold is worst but introduces no gradient. The
maximum prevents the optimizer from improving the easier source fold by
sacrificing the harder one. It adds no tunable temperature, margin, or fold
weight. The unchanged V13 quality loss is added with its already frozen weight
1.0:

```text
L = L_identity + L_quality.
```

The proposal deliberately removes `teacher_identity_utility`, utility
softmax, and `UTILITY_TEMPERATURE` from training. The utility tensor remains
read-only audit evidence only.

## 4. Why this is a coherent research contribution

The mechanism aligns three levels that V13 separated:

1. **deployment path:** the loss composes the exact Signal prefix and the same
   nine routed residual blocks used at inference;
2. **retrieval objective:** supervision is a differentiable cross-camera
   positive/negative ordering loss rather than a near-uniform action label;
3. **generalization constraint:** the two training teacher coordinate systems
   remain separate and the worst fold, not the pooled mean, controls the step.

Batch-hard triplet learning provides the ReID retrieval surrogate, while
worst-group optimization motivates protecting the poorest predeclared fold.
The proposed contribution is their deployment-aligned use for an
expert-modality Router, not a claim that triplet loss or worst-group risk is
new by itself.

## 5. Minimal execution protocol

### M0 — public-seam tests

- exact equality of the fold-risk implementation to a hand-computed toy case;
- cross-camera positives only;
- no distances across fold IDs;
- gradients reach modal and expert Router heads;
- missing modality mass is exactly zero;
- V13 exact fusion-prefix and fixed-alpha tests remain green.

### Q0 — real-cache zero-step qualification

Using the existing paired cache only, with no optimizer step:

- compute uniform and fixed-policy fold risks separately for all three folds;
- verify all folds have valid cross-camera positives and negatives;
- verify the composed loss is finite and produces finite nonzero gradients for
  every trainable Router tensor at initialization;
- verify Phase-A SHA unchanged and dev/official accesses remain zero.

This gate checks executability, not improvement. Failure stops V14.

### Q1 — one seed-42 identity-OOF Router qualification

- same three identity folds, hidden width 128, LR 3.5e-4, weight decay and 100
  epochs/fold as V13;
- for each held-out fold, train only on the other two folds using the proposed
  worst-fold retrieval regret plus unchanged quality loss;
- compare against the train-fold-selected fixed slot with the same four V13
  held-out metrics: expected utility, target-winner Top-1, replay AP and replay
  margin;
- require every metric non-inferior in every fold and all four identity-cluster
  bootstrap 95% lower bounds strictly above zero;
- retain corrupted-modality mass decrease, missing-mass zero, frozen Phase-A
  SHA, dev0 and official0 gates.

Q1 failure seals the method: no epoch/LR/loss-weight/margin/temperature/fold
scan, no final refit, no dev, no official test.

### Conditional final refit and one dev evaluation

Only after Q1 passes, fit one final Router on all 571 fit queries. Each of the
three OOF coordinate systems contributes one separate fold risk and the final
objective is their maximum plus the unchanged quality loss. Save one combined
Phase-A+Router checkpoint and perform one fixed 141-fit/30-dev evaluation.

Promotion requires all existing gates:

- fused mAP strictly above exact Signal 58.0109;
- fused mAP strictly above current deployable best V8 Phase-B 58.4050;
- fused mAP and Rank-1 strictly above CNN, Transformer and Mamba outputs;
- fused mAP at least 65.0;
- no official-test access.

No ablation, multi-seed run, baseline rerun, or official test occurs before
these gates pass.

## 6. Falsifiers and claim ceiling

The hypothesis is falsified if real-cache gradients are absent/nonfinite, Q1
sacrifices any fold/metric, bootstrap lower bounds are not positive, or the
conditional dev result misses any promotion gate. A failure is attributed to
the fixed relational routing mechanism, not to insufficient epochs.

Before a successful dev result, supported wording is limited to a tested
fold-robust Router objective. No causal-calibration, full three-way HFER
collaboration, official benchmark, SOTA, or broad generalization claim is
allowed. Even after success, HFER and the three full paper claims require
separate main-stage evidence; they are not inferred from Router qualification.

## 7. Primary grounding

- Hermans, Beyer and Leibe, *In Defense of the Triplet Loss for Person
  Re-Identification*, arXiv:1703.07737.
- Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts: On
  the Importance of Regularization for Worst-Case Generalization*,
  arXiv:1911.08731.
- Chen et al., *Ranking Measures and Loss Functions in Learning to Rank*,
  NeurIPS 2009.

