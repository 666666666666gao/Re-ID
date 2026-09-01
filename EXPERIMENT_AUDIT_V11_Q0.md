# V11-Q0 Experiment Integrity Audit

- **Date**: 2026-09-02
- **Auditor**: GPT-5.5 xhigh, independent read-only reviewer
- **Overall verdict**: `WARN`
- **Integrity status**: `warn`
- **Scientific verdict**: `FAIL_TO_QUALIFY / STOP_V11_Q0`

## A. Ground-truth provenance and fold isolation — WARN

Identity folds and held-out checkpoint identities are verified in
`tools/probe_v11_dinov2_oof_residual_complement.py:240-267`. AP and Rank-1 use
dataset identity/camera labels and exclude same-identity/same-camera junk in
`tools/diagnose_v6_oracle_complementarity.py:91-104`.

The expert adapters exclude the held-out identities during fold training, but
the frozen Signal field feeding them was trained on all fit identities. The
result records this explicitly. Thus only adapter training is identity-OOF;
the complete feature path is not identity-unseen.

## B. Score normalization — PASS

Each residual expert and fixed source block receives ordinary L2 feature
normalization before Euclidean retrieval. No metric is divided by a prediction
maximum, minimum or mean. The 100 mAP is saturation/leakage evidence, not
score-normalization fraud.

## C. Result existence and provenance — WARN

The result exists and its fold counts, 571-query total, metrics, gate values,
`qualification_status=FAIL` and `next_phase_authorized=false` are internally
consistent. The source commit, script/test/config/result hashes are packaged in
the separate provenance wrapper. Large fold checkpoints and DINO weights
remain remote-only; their recorded hashes cannot be independently recomputed
from a fresh local clone.

## D. Executed path — PASS

The live CLI loads every fold checkpoint, collects expert and DINO features,
computes fold-local scores, aggregates query results, computes the diagnostic
Oracle and applies the preregistered gate. Unit tests cover residual-bank
composition, fold aggregation and fail-closed saturation behavior.

## E. Scope and claim boundary — WARN

The run used fit identities only and reports zero training, dev and official
access. The material leakage caveat means the result cannot support end-to-end
identity-isolated generalization. It is a limited real-GT qualification
diagnostic, not a benchmark result.

## F. Evaluation type — PASS

Classification: `real_gt_fit_identity_oof_residual_only`. The hard Oracle uses
ground truth only to quantify headroom and is not deployable.

## Required action

1. Mark V11-Q0 `COMPLETE-FAIL` and close Q1/Q2/dev.
2. Package the raw result and source provenance.
3. Keep remote-only binary limitations visible.
4. Do not perform post-hoc DINO scans, training, dev, official or ablations.
5. Describe 100 mAP as saturation, never as a deployable improvement.
