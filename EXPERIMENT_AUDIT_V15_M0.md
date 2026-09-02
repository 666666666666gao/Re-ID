# TriFusion V15 M0 Experiment Audit

**Date**: 2026-09-02

**Auditor**: GPT-5.5 xhigh, read-only independent review

**Overall verdict**: WARN

**Integrity status**: `warn_packaging_only`

**Scientific/protocol status**: PASS

**Q1 authorized**: YES

## Checks

### A. Ground-truth provenance: PASS

M0 uses RGBNT201 train records and dataset identity/camera labels for supervised
loss and cross-camera retrieval risk. Model outputs are not used as ground
truth. The exchange-off comparator is stop-gradient and state-clean.

### B. Score/loss normalization: PASS

Retrieval risk uses L2-normalized embeddings, pairwise distances and explicit
identity/camera masks. The corrected lower bound is the analytic fixed-off
comparator floor, not a prediction-statistic normalization.

### C. Result/log/hash/dirty-state: WARN

The result and console JSON agree. The receipt is `PASS`, binds commit
`1f2de44f0c7c953bea7d75921be509ce9704f84c`, records an empty runtime diff,
and binds runner/config hashes. The packaging-only warning records that local
evidence and tracker files were not yet committed when the audit ran. They are
committed separately before Q1; unrelated pre-existing local edits remain
excluded.

### D. Executed path / dead-code check: PASS

The explicit M0 gate now requires zero overflow, unchanged frozen state and
both exchange stages live in the 8-step capacity run. The 100-step gate still
requires all trainable tensors to have received nonzero gradients and the
floor-adjusted loss ratio to pass.

### E. Scope and claim boundary: PASS

The config remains B64/K8 and seed42 with dev, official test, reranking,
multi-seed and scans forbidden. Receipt access counters are dev0/official0.
Q1 is a mechanism qualification only; D1 remains blocked.

### F. Evaluation classification: PASS

M0 is real-GT train-only engineering qualification, not a dev performance
evaluation. It supports Q1 authorization only.

## Decisive finding

`PASS_WITH_PACKAGING_WARN`. Both former invalid-gate issues are corrected:

1. Capacity now gates on both exchange stages being live, while retaining
   `107/110` and the three untouched-in-eight-steps tensors as diagnostics.
2. The overfit ratio subtracts both the label-smoothing floor (`0.578383`) and
   fixed matched-regret floor (`0.474426`). The combined floor is `1.052809`
   and the recomputed ratio is `0.051554 <= 0.10`.

No dev/official leakage or metric fraud was found. Q1 is authorized from the
hash-bound passing M0 receipt. D1, dev claims, official evaluation, ablations,
multiple seeds and hyperparameter scans remain unauthorized.
