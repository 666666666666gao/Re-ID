# TriFusion V8 Phase-A experiment audit

**Date:** 2026-09-02

**Auditor:** GPT-5.5 xhigh, independent read-only review

**Scope:** V8 pretrained-tail expert-formation preflight, capacity, overfit and
20-epoch formation probe

## Overall verdict: WARN

## Integrity status: warn

The narrow diagnostic claim is supported: the three Phase-A residual experts
contain query-level complementary identity signal and warrant one fit-only
Router feasibility phase. The result does not support deployable routing,
HFER, 65 mAP, official-test performance, SOTA or broad generalization.

## Checks

### A. Ground-truth provenance: PASS

- `protocols/rgbnt201_dev_v1.json:126` declares that selection does not use
  test labels.
- `tools/run_signal_baseline_dev.py:89` loads the 141/30 identity registry and
  verifies disjoint train/dev identities and expected record counts.
- `utils/reid_evaluation.py:31` derives matches from dataset identities and
  cameras after same-identity/same-camera exclusion.
- `tools/diagnose_v6_oracle_complementarity.py:94` computes per-query AP and
  Rank-1 from dataset query/gallery identities and cameras.
- The V8 probe receipt records `oracle_uses_ground_truth=true`,
  `oracle_is_deployment_result=false` and `official_test_access_count=0`.

The Oracle is real held-out-dev ground truth, explicitly diagnostic and
non-deployable; it is not generated from model predictions.

### B. Score normalization: PASS

- `utils/metrics.py:293` performs standard feature L2 normalization, Euclidean
  distance and CMC/mAP evaluation.
- `tools/diagnose_v6_oracle_complementarity.py:129` L2-normalizes features
  before `torch.cdist`.
- `modeling/trifusion/signal_preserving_v8.py:454` normalizes expert residual
  embeddings, and `:499` gives the diagnostic bank a fixed baseline-relative
  feature energy.

No reported metric is divided by the model's own maximum, minimum or mean.

### C. Result existence and numeric consistency: WARN

All four V8 evidence JSON files exist, and the reported values recompute and
match after rounding:

- exact 3072D Signal parity and baseline mAP `58.0108866`;
- capacity `203/203` gradient tensors, zero overflow and unchanged Signal;
- overfit excess-loss ratio `0.0005344774 ≤ 0.1`;
- probe 20 epochs, 840 steps, final-only dev evaluation, unchanged Signal and
  zero official accesses.

The warning is provenance packaging, not a metric mismatch. At audit time the
four JSON files were untracked, while the checkpoint, history and run identity
remained remote artifacts referenced by SHA/path. This change tracks the JSON
receipts and result report; the large checkpoint remains intentionally outside
Git.

### D. Dead code and gate execution: PASS

- The formation gate is defined and called in
  `tools/run_signal_preserving_v5.py:943` and `:1233`.
- The complementarity evaluator calls the per-query score and Oracle summary
  routines at `tools/run_signal_preserving_v5.py:1056`.
- The V8 residual criterion is live at
  `modeling/trifusion/signal_preserving_v8.py:719`, and the runner includes its
  losses.
- CLI `formation_probe` dispatch reaches the training/evaluation path.

### E. Scope and dev leakage: PASS

- One dataset, one frozen 141-fit/30-dev split and one seed 42 were used.
- The 20-epoch loop reads only `train_loader`; no dev evaluation occurs during
  training.
- The final checkpoint is saved before the single dev complementarity
  evaluation.
- The receipt records `model_selection=none_final_epoch_only` and
  `dev_evaluations_during_training=0`.
- Official-test access remains zero.

### F. Evaluation type: PASS / real_gt

- Preflight and final retrieval metrics: held-out-dev `real_gt`.
- Formation Oracle: held-out-dev `real_gt`, diagnostic and non-deployable.
- Capacity/overfit: supervised fit-set engineering gates, not retrieval claims.

## Action items

- [x] Track the four V8 Phase-A JSON receipts and this audit/report.
- [x] Keep Oracle wording explicitly diagnostic and non-deployable.
- [ ] Preserve remote `run_identity.json`, `history.json` and checkpoint by
  their recorded SHA/path; do not commit the large checkpoint.
- [ ] Train only a fit-only Router next; do not use dev Oracle labels for
  Router training, tuning or model selection.

## Claim impact

- **Supported with qualifier:** the three Phase-A experts contain query-level
  complementary signal.
- **Supported next step:** one frozen-expert, fit-only Router feasibility phase.
- **Unsupported:** learned routing, HFER, 65 mAP, official-test performance,
  SOTA and multi-seed robustness.
