Read-only audit completed. I did not modify files or run experiments.

**A. Ground-Truth Provenance / Fold Isolation: WARN**

Evidence:

- `probe_v11_dinov2_oof_residual_complement.py:240-244` builds identity folds and selects fit cross-camera records.
- `probe_v11_dinov2_oof_residual_complement.py:253-267` verifies each fold checkpoint matches its held-out identities.
- `build_v8_oof_router_targets.py:299-306` trains fold experts with held-out identities excluded.
- `diagnose_v6_oracle_complementarity.py:91-104` computes AP/Rank-1 from dataset identity and camera labels, excluding same-id/same-camera junk.
- `probe_v11_dinov2_oof_residual_complement.py:363` and result JSON `:139` explicitly record `frozen_signal_field_seen_all_fit_identities: true`.

GT is real dataset identity/camera metadata, not model-generated GT. Fold isolation is valid only for the fold-trained expert residual states. It is not end-to-end identity isolation because the frozen Signal field feeding those residuals was trained on all 141 fit identities.

**B. Score Normalization / Self-Normalization Fraud: PASS**

Evidence:

- `probe_v11...py:23-29` L2-normalizes each residual expert before concatenation.
- `probe_v10_dinov2_fit_utility.py:64-73` L2-normalizes fixed source blocks for equal-block concat.
- `diagnose_v6_oracle_complementarity.py:129-139` L2-normalizes embeddings, computes pairwise distances, then AP from labels.
- `diagnose_v6_oracle_complementarity.py:43-45` hard Oracle is per-query max AP across sources, not a deployed score.

No metric is divided by the model’s own max/min/mean output. The 100 mAP is suspicious, but not metric fraud. It is expected fit-identity saturation/leakage from the residual bank under an all-fit frozen Signal field, and the V11 gate catches it as saturation.

**C. Result / Provenance Consistency: WARN**

Evidence:

- Result exists and says `status: COMPLETE` at JSON `:242`, `qualification_status: FAIL` at `:193`, `next_phase_authorized: false` at `:166`.
- Fold/query counts are internally consistent: fold queries `190/179/202` at `:60`, `:98`, `:136`; gate query count `571` at `:185`.
- Key metrics are present: residual bank 100 mAP at `:157-160`, DINO 14.1323 at `:149-151`, concat 95.8582 at `:145-147`.
- Gate numbers match the failure: best fixed 100 at `:174`, concat gain `-4.1418` at `:175`, Oracle gain `0.0` at `:181`, DINO unique wins `0` at `:188-190`.
- Remote-only provenance is recorded but not locally verifiable: DINO weight path/hash at `:18-19`, fold checkpoint paths/hashes at `:25-26`, `:63-64`, `:101-102`, source summary path/hash at `:240-241`.
- `EXPERIMENT_TRACKER.md:21` still says V11-Q0 is `READY`, inconsistent with the completed result.

Additional repo observation: the local evidence JSON is untracked in this checkout, and the V11 result schema does not embed a V11 script SHA/commit. That is a packaging/provenance warning, not a metric-integrity failure.

**D. Executed vs Dead Code: PASS**

Evidence:

- V11 CLI calls `run()` at `probe_v11...py:403-404`.
- The live V11 path loads fold checkpoints, collects features, scores all outputs, aggregates folds, computes Oracle, and applies the gate at `:253-336`.
- The result object written by that path is defined at `:342-388`.
- Tests cover residual-bank normalization, fold-score aggregation, and gate saturation/failure behavior at `test_probe_v11...py:7-97`.

No reported V11 metric path appears dead. Some V10/V6 CLI entrypoints are historical/helper paths, not claimed V11 execution paths.

**E. Scope / Leakage / Claim Boundary: WARN**

Evidence:

- Result scope is “three-fold fit-identity OOF residual-only retrieval” at JSON `:237`.
- No dev or official access: JSON `:5`, `:167`; code writes these counters at `probe_v11...py:376-377`.
- No training in V11-Q0: JSON `:168`, `:244`.
- Plan explicitly excludes Signal/Phase-B embeddings from Q0 metrics while admitting the frozen Signal token field is common input at `EXPERIMENT_PLAN...md:216-219`.
- Tracker boundary says Signal/Phase-B do not enter the scientific gate and failure closes V11 at `EXPERIMENT_TRACKER.md:26`.

Scope is correctly bounded, but the leakage caveat is material: this supports only expert-head residual OOF conditioned on an all-fit frozen Signal field, not end-to-end identity-isolated generalization.

**F. Evaluation Classification: PASS**

Classification: `real_gt_fit_identity_oof_residual_only`.

Evidence:

- Code writes this classification at `probe_v11...py:347`.
- Result records it at JSON `:21`.
- Protocol uses held-out identity triplets with same-id/same-camera exclusion and zero invalid positives at `rgbnt201_dev_v1.json:52-58`.

This is real-GT diagnostic fit evaluation, not official test, not dev benchmark, not synthetic proxy, and not deployable Oracle.

**Overall**

- `overall_verdict`: WARN
- `integrity_status`: warn
- `scientific_verdict`: FAIL_TO_QUALIFY/STOP_V11_Q0

Claim impact:

- DINO complement claim: unsupported. DINO has 0 unique AP wins and Oracle gain is 0.
- V11-Q0 qualification: failed. Do not proceed to Q1/Q2/R001.
- Identity isolation claim: must be qualified as fold-trained expert residual isolation only, conditioned on a frozen all-fit Signal field.
- Oracle: diagnostic only, not deployable.

Action items:

1. Mark V11-Q0 terminal as `COMPLETE-FAIL` in the tracker.
2. Commit/package the result evidence and add V11 script/git provenance.
3. Keep remote checkpoint/DINO/source-summary hashes recorded; do not imply local binary verification.
4. Stop V11-Q0: no DINO scans, no fusion/head training, no dev, no official, no ablations.
5. In any writeup, say 100 mAP is saturation/leakage evidence, not a deployable gain and not metric fraud.
