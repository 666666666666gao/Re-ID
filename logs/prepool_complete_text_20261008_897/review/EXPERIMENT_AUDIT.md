# H2a experiment integrity audit

2026-10-08 — **WARN; integrity_status: warn; 0 blockers.** Same-family, provisional review. Requested reviewer: gpt-6-astra / max; actual backend and effort: **UNATTESTED**.

Original receipts support all three fresh full50 endpoints: **150 epochs, 6,484 formal updates, 0/3 registered primary advances**. The negative utility decision is supported.

| Dataset | Best epoch | mAP | R1 | ΔmAP versus sealed raw-semantic |
|---|---:|---:|---:|---:|
| RGBNT201 | 8 | 74.3864 | 79.1866 | +0.0116 |
| MSVR310 | 38 | 50.5468 | 68.0203 | +0.0046 |
| RGBNT100 | 26 | 83.5272 | 95.8017 | −0.5631 |

All four metrics use the same fused mAP-selected checkpoint. The fixed advance condition remains ΔmAP ≥ +0.5 points with R1 not decreasing (`EXPERIMENT_PLAN_20261007_194152.md:9`; `tools/report_prepool_dense_correspondence.py:109`).

| Check | Status | Evidence |
|---|---|---|
| A. GT provenance | PASS | Dataset identity labels; complete camera/scene filtering. `tools/run_correspondence_roles.py:79`; upstream `utils/metrics.py:68,137`. |
| B. Score normalization | PASS | Standard AP/CMC; no prediction-statistic score normalization. Camera/scene scorers at `tools/train_rgbnt100_signal_oof.py:253` and `tools/train_msvr310_signal_oof.py:223`. |
| C. Existence/completion | WARN | Original text, hashes and terminal records agree; binaries were not independently rescored locally. Original controller `campaign.json:1188`; report `SUMMARY.json:7`. |
| D. Actual callers | PASS | Claimed metric functions are called by the original training/evaluation/report chain. `tools/run_prepool_dense_correspondence.py:87,125`; `tools/run_foundation_recipe.py:248,300`. |
| E. Scope | WARN | One seed (42); official development scores select epochs. Plan lines 32,59. |
| F. Classification | PASS | Retrieval is `real_gt`; geometry and attention diagnostics are `self_supervised_proxy`. Prepool model lines 75,112,135. |

Independent checks verified 95 requested inputs, all 428 source digests, 72 original files, six sealed control curves, all formal batches/steps and 12,568 paired-query rows. **1,436 checks passed.** Maximum score reaggregation difference: **2.824e-6 points**, below the unchanged **1e-5** tolerance.

Keep four qualifications: development/single-seed scope; text-only local validation and unresolved canonical precheck; different selected epochs in control comparisons; and proxy metrics without identity/part-label meaning. RGBNT100 own-global equals the original global curve at epoch26, while global-best selects epoch7. Their difference does not establish a changed shared-global trajectory. Near-uniform RGBNT201 late attention also prevents interpreting very low mapped L1 as semantic correspondence.

Retain the negative result and close this fixed implementation under the registered plan. No integrity-driven source change is required. These receipts establish neither SOTA, cross-family acceptance nor full-goal completion. The prior unavailable review remains preserved.
