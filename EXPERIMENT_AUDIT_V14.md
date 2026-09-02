# TriFusion V14 Experiment Audit

**Date**: 2026-09-02

**Auditor**: GPT-5.5 xhigh, read-only independent review

**Overall verdict**: WARN

**Integrity status**: warn
**Terminal status**: `Q0_PASS_Q1_FAIL_DO_NOT_PROMOTE`

## Checks

### A. Ground-truth provenance: PASS

V14 risk/AP/margin use dataset identity and camera labels to define
cross-camera positives, negatives, junk filtering and ordinary ReID ranking.
The V13 `teacher_identity_utility` appears only in non-gating diagnostics.

Evidence: `modeling/trifusion/signal_preserving_v14.py:27-76`,
`tools/train_v14_fold_robust_router.py:83-126`, `:385-417`.

### B. Score normalization: PASS

No metric is divided by prediction maxima, minima or means. Retrieval risk is
batch-hard softplus over distances between L2-normalized embeddings; replay AP
uses the existing ordinary ReID distance ranking. Risk values near 0.69 are the
natural softplus scale, not percentage accuracy.

### C. Result existence and numeric consistency: WARN

Both evidence JSON files exist and are internally consistent: Q0 fold queries
sum to 571 and has zero optimizer steps; Q1 fold queries sum to 571 and has
`3×100=300` steps. Runner/config hashes match the recorded files.

The warning has two reporting causes: the tracker was stale at audit time, and
Q1 `status: PASS` denotes runner completion while the scientific gate is false.
The terminal report and updated tracker therefore use the unambiguous status
`Q0_PASS_Q1_FAIL_DO_NOT_PROMOTE`.

### D. Executed path / dead-code check: PASS

Q1 executed all three OOF Router fits. Final refit/checkpoint creation is inside
`if gate["passed"]`; the failed result has `final_training=null`,
`combined_checkpoint=null`, `next_phase_authorized=false`, dev0 and official0.
No phantom dev result exists.

### E. Scope, leakage and claim boundary: PASS

For each held-out fold, the comparator and Router training use only the other
two folds. The held-out best fixed slot is diagnostic only. Fold binding rejects
row/generator mismatch, and no distance is computed across OOF generators.
The documented scope correctly states that Router inputs are all-fit deployment
features and only teacher/replay embeddings are identity-OOF.

### F. Evaluation classification: PASS

- M0: synthetic engineering worked examples.
- Q0: real-cache zero-step engineering qualification.
- Q1 risk/AP/margin: real-GT OOF fit replay, not 30-dev deployment evaluation.
- Expected utility/action Top-1: teacher-proxy diagnostics.
- Held-out best fixed slot: diagnostic oracle only.
- Corruption/missing checks: synthetic engineering quality controls.

## Claim impact

V14 supports only Q0 executability. Q1 is a negative result: fold0 AP declines,
fold2 risk and margin decline, and all three bootstrap lower bounds are
negative. It supports no reliable Router-transfer, dev, HFER, official-test or
SOTA claim. Final refit, dev, ablations, multiple seeds and same-family scans
remain unauthorized.
