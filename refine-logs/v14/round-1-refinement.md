# V14 round-1 refinement — Fold-Robust Retrieval-Regret Router

## 1. Narrowed status and hypothesis

V14 is one train-only falsification of a relational Router objective. It is not
yet a deployable contribution. The V13 diagnostic supports deleting the
pointwise utility-KL target; it does not show that V14 is likely to reach the
65 mAP dev gate.

The single hypothesis is:

> With experts, Router inputs, fusion and quality control frozen, optimizing
> the Router on fold-local cross-camera retrieval regret and protecting the
> worst source fold will transfer more reliably than V13's nearly uniform
> pointwise action target.

## 2. Frozen prerequisites and exact OOF boundary

V14 retains without modification:

- shared RGB/NIR/TIR flip, padding and crop geometry, followed by independent
  modality erasing;
- exact frozen Signal 3072D baseline prefix;
- V8 Phase-A pretrained-tail CNN, Transformer and Mamba residual experts;
- matched-token residuals;
- hierarchical `P(modality|x) * P(expert|modality,x)` routing;
- hard-zero missing-modality mass;
- exact-prefix blockwise fusion with fixed bounded `alpha=0.2`;
- V13 hidden width, learning rate, weight decay, epoch count and corruption-
  response quality loss at weight 1.0.

The Router inputs are frozen **all-fit deployment features by design**. Only
the teacher/replay baseline and residual embeddings are identity-OOF. V14 is
therefore a deployment-input / OOF-teacher train-only qualification, not a
complete-path identity-OOF feature-generalization experiment. This fact is a
required receipt field in M0, Q0 and Q1.

HFER stays disabled so the experiment tests one mechanism. It can be considered
only after a Router becomes deployably useful.

## 3. Fold-bound risk API

Every differentiable retrieval calculation must use the conceptual API:

```text
risk(fold_id, rows_in_fold_id, features_from_generator_fold_id, weights)
```

The API enforces all of the following:

- every row has `fold_indices == fold_id`;
- baseline and residual tensors are the OOF teacher tensors registered for the
  same `fold_id`;
- query and reference/gallery embeddings come only from those rows;
- embeddings are composed through the exact V13 deployment fusion and L2
  normalized before distance computation;
- positives require the same identity and a different camera;
- negatives require a different identity;
- no distance is ever computed across two OOF generators.

For query `i` in fold `f`:

```text
d_pos(i) = max_j ||z_i-z_j||_2,
                 y_j=y_i and camera_j!=camera_i
d_neg(i) = min_k ||z_i-z_k||_2, y_k!=y_i
R_f(w)   = mean_i softplus(d_pos(i)-d_neg(i)).
```

No margin, temperature, listwise relaxation or new loss weight is introduced.

## 4. Training-only fixed comparator

For a Q1 Router that holds out fold `h`, let source folds be `a,b`. Select one
shared fixed slot using only source-fold retrieval risks:

```text
s* = argmin_s max(R_a(one_hot(s)), R_b(one_hot(s))),  s in {0,...,8}.
```

This nine-action enumeration is the definition of the fixed comparator, not a
hyperparameter scan. It does not access held-out fold `h`; all distances remain
inside `a` or `b`.

For each source fold:

```text
G_f(w) = R_f(w) - stop_gradient(R_f(one_hot(s*))).
```

The identity objective is:

```text
L_identity = max(G_a(w), G_b(w))
L_total    = L_identity + L_quality.
```

The fixed constants select the worst relative source fold but carry no
gradient. V13 `teacher_identity_utility`, its softmax temperature and its KL
loss are absent from optimization. Utility and action-winner values remain
read-only diagnostics only.

## 5. Correctly aligned Q1 evaluation

On held-out fold `h`, apply the trained Router and the source-only `s*` to the
same fold-bound OOF teacher embeddings. Define positive improvement:

```text
risk_gain_h = R_h(one_hot(s*)) - R_h(learned).
```

Also compute per-query replay AP and retrieval margin for learned and fixed
embeddings, yielding:

```text
ap_gain_h     = AP_h(learned) - AP_h(fixed)
margin_gain_h = margin_h(learned) - margin_h(fixed).
```

The three primary Q1 retrieval gates are:

1. every fold has `risk_gain_h > 0`;
2. every fold has `ap_gain_h >= 0` and `margin_gain_h >= 0`;
3. identity-cluster bootstrap 95% lower bounds for pooled paired risk gain, AP
   gain and margin gain are all strictly greater than zero.

Replay Rank-1, V13 expected utility and V13 target-winner Top-1 are reported as
diagnostics but do not determine V14 success. This removes the rejected
objective/gate mismatch without hiding their behavior.

The held-out fold's own best fixed-slot risk/AP/margin is also reported as a
non-gating oracle diagnostic. It is never used to select `s*`, train the Router
or decide Q1; it only bounds the wording of any later fixed-policy comparison.

The unchanged safety gates also remain mandatory:

- corrupted RGB, NI and TI each receive lower modal mass than their clean
  counterparts;
- missing-modality maximum mass is exactly zero;
- every Phase-A tensor SHA is unchanged;
- dev and official-test access counts are zero.

Any failure seals V14: no temperature, margin, epoch, LR, fold, loss-weight or
threshold scan; no final refit or dev.

## 6. Minimal execution sequence

### M0 — public-seam tests

- toy fold risk equals a hand-computed result;
- same-camera positives are excluded;
- invalid fold/row/feature bindings are rejected;
- Q1 and final-refit risk paths both prohibit cross-fold distances;
- gradients reach every Router parameter tensor;
- missing modality remains zero;
- exact V13 prefix and fixed-alpha fusion remain unchanged;
- receipts explicitly state `router_input_scope=all_fit_deployment` and
  `teacher_embedding_scope=identity_oof`.

### Q0 — real-cache zero-step gate

- compute per-fold uniform and all nine fixed-slot risks using only the exact
  paired cache;
- verify cross-camera positive and negative support for every retained query;
- verify the worst-fold loss is finite and gives finite nonzero gradient to
  every Router parameter tensor at seeded initialization;
- verify all fold binding receipts, Phase-A SHA, dev0 and official0;
- optimizer steps remain zero.

Q0 checks executable signal only, not gain.

### Q1 — sole seed-42 OOF qualification

For each held-out fold, select `s*` from the two source folds, train 100 epochs
with the fixed V13 optimizer settings, and apply the revised retrieval and
safety gates. This consumes one preregistered Q1; failure stops the family.

### Conditional final refit and one dev evaluation

Only after Q1 passes, select the all-fit comparator by:

```text
s_all = argmin_s max_f R_f(one_hot(s)).
```

Train one final Router with:

```text
max_f [R_f(w) - stop_gradient(R_f(one_hot(s_all)))] + L_quality.
```

Each of the three `R_f` calls uses only `rows_in_fold_f` and
`features_from_generator_f`; scalar fold regrets may be compared, but feature
vectors and pairwise distances are never mixed across folds. Save one combined
Phase-A+Router checkpoint, then perform exactly one fixed 141-fit/30-dev
evaluation.

Dev promotion still requires fused mAP strictly above Signal 58.0109, current
deployable V8 Phase-B 58.4050 and all three branch outputs; fused Rank-1 must
also strictly exceed all branches; fused mAP must be at least 65.0. Official
test, ablations, baseline rerun and multiple seeds remain forbidden beforehand.

## 7. Falsifiers and claim ceiling

The V14 hypothesis fails if Q0 lacks valid gradients, any held-out fold has
nonpositive risk gain, held-out AP or margin decreases, any paired bootstrap
lower bound is nonpositive, or any safety/access gate fails. A Q1 pass supports
only that this fixed Router objective transfers across the three registered fit
folds under all-fit deployment inputs and identity-OOF teacher/replay metrics.
It does not prove complete-path OOF generalization, HFER
collaboration, dev improvement, SOTA, causal calibration or broad robustness.

Only a conditional dev pass could support deployable improvement. HFER and the
three eventual paper contributions still require later main-stage evidence.

## 8. Primary grounding

- Hermans, Beyer and Leibe, *In Defense of the Triplet Loss for Person
  Re-Identification*, arXiv:1703.07737.
- Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts: On
  the Importance of Regularization for Worst-Case Generalization*,
  arXiv:1911.08731.
- Chen et al., *Ranking Measures and Loss Functions in Learning to Rank*,
  NeurIPS 2009.
