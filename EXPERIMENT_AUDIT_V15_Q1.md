# TriFusion V15 Q1 Experiment Audit

**Date**: 2026-09-02

**Auditor**: GPT-5.5 xhigh, read-only independent review

**Overall verdict**: WARN

**Integrity status**: `valid_q1_fail_with_packaging_warn`

**Scientific verdict**: Q1 gate failed; V15 sealed

**D1 authorized**: NO

## A–F checks

- **GT provenance: PASS.** RGBNT201 train records provide identity/camera GT;
  heldout evaluation uses cross-camera eligible records and ordinary ReID AP.
- **Normalization: PASS.** Embeddings use L2 normalization and pairwise
  distance. No self-normalized metric was found; off regret is stop-gradient.
- **Result/log/hash: WARN (packaging only).** Console and JSON agree, runner,
  config and M0 hashes match, and the receipt binds clean commit `71152d3` and
  empty runtime diff. The warning records that evidence was uncommitted during
  review; it is committed before publication.
- **Executed path: PASS.** All three folds trained and evaluated exchange-on
  and exact exchange-off. The earlier optimizer0 import failure is not counted
  as an experiment result.
- **Scope: PASS.** Q1 is fit-only identity-OOF with dev0/official0. The frozen
  plan requires sealing V15 after failure.
- **Evaluation type: PASS.** `real_gt_train_only_identity_oof_mechanism`, not
  dev, official or proxy accuracy.

## Decisive finding

The Q1 failure is valid and is not another gate implementation error. Fused
gains are `+0.09518/-0.83106/+0.16052`; weighted aggregate is `-0.172068`;
bootstrap 95% lower bound is `-0.950330`; aggregate CNN/Transformer gains are
negative and only Mamba is positive. Fused remains above the CRDE branches,
but this single condition cannot satisfy the conjunction of registered gates.

`status=PASS` means runner completion. `gate.passed=false`,
`next_phase_authorized=false` and `d1_executed=false` are authoritative. The
only supported claim is that Q1 completed under the registered train-only
identity-OOF protocol and failed. No D1, 65 mAP, deployable gain, official,
SOTA or ablation claim is supported.
