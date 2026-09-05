# EXPERIMENT_AUDIT_V24_Q1

Audit date: 2026-09-06

Project root: `C:/Users/gb/.trifusion_github_publish_22c3bee`

Requested reviewer: GPT-5.5 xhigh. The dispatch request records `model: gpt-5.5`, `reasoning_effort: xhigh`, and registered agent `/root/audit_v24_q1` (`.aris/traces/experiment-audit/2026-09-06_run15/001-v24-q1.request.json:7-11`). The dispatch observation records that this is a root conversation observation, not backend identity attestation, and `resolved_backend_independently_attested` is false (`.aris/traces/experiment-audit/2026-09-06_run15/dispatch_observation_20260906.json:3-13`). GPT-5.5, if actually resolved by the backend, is still GPT-lineage and therefore same-family relative to the GPT-lineage executor, not a cross-family review. The original auditor was dispatched; this auditor did not create further child agents.

Overall verdict: **WARN**

Engineering integrity: **PASS**. Fixed M0 qualification: **PASS, engineering only**. Scientific qualification: **FAIL**. Evaluation classification: **real_gt** / `real_gt_train_internal_complete_path_oof_reused_development_qualification`.

The correct high-level claim is that V24 Q1 completed the preregistered train-internal complete-path OOF qualification and failed its fixed scientific gate. The run should not be used to claim method efficacy, D1 advancement, fixed-dev success, official/test success, or deployment readiness.

## Follow-up 002 corrections

Status: **metadata corrected; scientific verdict unchanged**

This follow-up read the preserved round-1 request, response metadata, reports, dispatch observation, original audit request, and input manifest directly (`.aris/traces/experiment-audit/2026-09-06_run15/FOLLOWUP_CHECKLIST_002.txt:1-29`, `.aris/traces/experiment-audit/2026-09-06_run15/001-v24-q1.meta.json:1-23`). Two round-1 findings required correction:

- Round 1 said no GPT-5.5 or cross-family reviewer was callable here (`.aris/traces/experiment-audit/2026-09-06_run15/001-EXPERIMENT_AUDIT_V24_Q1.md:7`). The supplied dispatch records do not support an unavailable-model claim. They support requested `gpt-5.5`/`xhigh` and accepted registered agent `/root/audit_v24_q1`, while explicitly not attesting resolved backend identity (`.aris/traces/experiment-audit/2026-09-06_run15/001-v24-q1.request.json:7-11`, `.aris/traces/experiment-audit/2026-09-06_run15/dispatch_observation_20260906.json:3-13`).
- Round 1 JSON recorded `recomputation.bootstrap.seed` as `20260906` (`.aris/traces/experiment-audit/2026-09-06_run15/001-EXPERIMENT_AUDIT_V24_Q1.json:451-452`). The registered plan, config, runner, and array audit use seed `42` (`configs/RGBNT201/TriFusion-signal-preserving-v24-source-prototype-rtx3090.yml:72-75`, `refine-logs/v24/EXPERIMENT_PLAN.md:118-123`, `tools/train_signal_preserving_v24.py:475-491`, `tools/audit_v24_terminal_arrays.py:286-294`). This report corrects the seed metadata to `42`. The bootstrap lower bound remains `-0.6948068678403989`, so the scientific verdict is unchanged.
- Round 1’s identity gain row values were numerically correct, but the final report did not clearly separate independently recomputed identity rows from executor verifier claims. This follow-up recomputed the identity rows directly from supplied per-query AP arrays, `query_indices`, and `gallery_manifest` identity labels. The recomputed rows match the verifier rows with max absolute identity mean-gain discrepancy `0.0` mAP points.

## A. Ground truth and isolation

Status: **PASS**

The audited run uses real identity and camera labels inside train-internal complete-path OOF folds. The split builder excludes heldout identities from fit/source records and leaves heldout labels/cameras available for evaluation (`tools/build_v12_complete_path_oof_targets.py:36-55`). The V24 runner initializes source-prototype memory from source records before training (`tools/train_signal_preserving_v24.py:109-130`) and then trains/evaluates per fold/endpoint with heldout records (`tools/train_signal_preserving_v24.py:422-467`). The retrieval evaluator removes same-identity same-camera junk and requires a cross-camera same-identity match (`tools/audit_v17_full_gallery.py:14-43`, `tools/diagnose_v6_oracle_complementarity.py:81-108`).

Recomputed scope from the supplied arrays:

| fold | gallery records | eligible queries | excluded queries | heldout identities |
|---:|---:|---:|---:|---:|
| 0 | 1000 | 190 | 810 | 47 |
| 1 | 1051 | 179 | 872 | 47 |
| 2 | 1075 | 202 | 873 | 47 |
| total | 3126 | 571 | 2555 | 141 |

These values match the array verifier and complete-comparison scope claims (`evidence/trifusion_v24_q1_array_verification_20260906.json:1165-1185`, `evidence/trifusion_v24_q1_complete_comparison_20260906.json:57931-57939`).

## B. Mathematics and statistics

Status: **PASS**

I replayed metric arithmetic over the supplied JSON AP and first-match-rank arrays using local Python 3.13.12 and NumPy 2.5.2. The recomputation used the evaluator formulas in `tools/audit_v24_terminal_arrays.py:20-23`, the loss checks in `tools/audit_v24_terminal_arrays.py:36-61`, and the identity-cluster bootstrap definition in `modeling/trifusion/signal_preserving_v13.py:253-278`. The raw summary values are at `evidence/trifusion_v24_q1_seed42_6a4ac2c.json:147613-147705`.

Aggregate metrics recomputed from the supplied arrays:

| output | control mAP | candidate mAP | gain | control R1 | candidate R1 | control R5 | candidate R5 | control R10 | candidate R10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_only | 77.487603 | 77.487603 | 0.000000 | 79.334501 | 79.334501 | 89.492119 | 89.492119 | 93.520140 | 93.520140 |
| fused | 79.534978 | 80.026285 | 0.491307 | 82.136602 | 82.311734 | 89.842382 | 90.192644 | 94.045534 | 94.395797 |
| cnn | 78.288484 | 77.543382 | -0.745102 | 81.786340 | 80.560420 | 88.441331 | 88.091068 | 91.593695 | 90.542907 |
| transformer | 76.857705 | 78.208971 | 1.351266 | 80.385289 | 81.085814 | 89.316988 | 90.718039 | 93.520140 | 93.169877 |
| mamba | 78.454237 | 77.875034 | -0.579203 | 81.085814 | 79.859895 | 91.768827 | 91.068301 | 94.570928 | 94.395797 |

Fixed Q1 gate replay:

| gate | result |
|---|---|
| fused aggregate gain >= 1 pp | false |
| all fold fused gains nonnegative | false |
| all expert aggregate gains nonnegative | false |
| fused bootstrap lower bound positive | false |
| fused beats candidate baseline and experts | true |
| next phase qualified | false |

The fused gain is **+0.491307 mAP points**, below the +1 pp gate. Fold fused gains are **-0.125144**, **+0.166401**, and **+1.359050**, so the all-fold gate fails. Expert aggregate gains are cnn **-0.745102**, transformer **+1.351266**, and mamba **-0.579203**, so the all-expert gate fails. The identity-cluster bootstrap lower bound is **-0.694807**, so the bootstrap gate fails. Query-paired fused changes are 205 improved, 163 declined, and 203 equal, with 8 Rank-1 repairs and 7 Rank-1 breaks (`evidence/trifusion_v24_q1_complete_comparison_20260906.json:2740-2776`).

Follow-up 002 independently recomputed the registered identity-cluster bootstrap from raw fused AP arrays and gallery identities: compute candidate-control AP deltas for the 571 eligible fused queries, map each query through `query_indices` to `gallery_manifest[*].identity`, sort clusters with `np.unique`, then run `np.random.default_rng(42).choice` over the 21 identity clusters for 10,000 resamples. Each sampled identity contributes all of its query deltas; the statistic is the mean over concatenated sampled query rows, so identities are weighted by sampled query counts. The quantile convention is `np.percentile(means, 2.5)`, matching the runner (`modeling/trifusion/signal_preserving_v13.py:265-276`). Arithmetic elapsed time for this bootstrap pass was 0.160515 seconds. The lower bound is `-0.6948068678403989`, with discrepancy `0.0` against the raw Q1 summary (`evidence/trifusion_v24_q1_seed42_6a4ac2c.json:147698-147701`).

Numerical mismatch against summary aggregates is **0.0**. The supplied array verifier reports maximum absolute numeric difference of **1.1102230246251565e-16** percentage points and all 120 epoch loss rows exact within recorded precision (`evidence/trifusion_v24_q1_array_verification_20260906.json:1162-1164`, `evidence/trifusion_v24_q1_array_verification_20260906.json:1439-1447`). The displayed result markdown matches the JSON aggregate and gate values (`results/TRIFUSION_RGBNT201_V24_COMPLETE_Q1_2026-09-06.md:13-21`, `results/TRIFUSION_RGBNT201_V24_COMPLETE_Q1_2026-09-06.md:95-119`).

## C. Provenance and immutability

Status: **WARN**

The input-byte manifest has **0** current-file SHA256/byte mismatches across its listed files (`.aris/traces/experiment-audit/2026-09-06_run15/input_file_sha256.json:1-244`). The execution summary binds the run to commit `6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33` and to per-source execution hashes (`evidence/trifusion_v24_q1_seed42_6a4ac2c.json:3-28`). The terminal verifier records file evidence, receipt equality, prototype memory receipts, exit code, and no additional optimizer/retrieval work during verification (`evidence/trifusion_v24_q1_terminal_file_verification_20260906.json:10-120`, `evidence/trifusion_v24_q1_terminal_file_verification_20260906.json:497-512`). Dispatch acceptance is not independent backend identity attestation (`.aris/traces/experiment-audit/2026-09-06_run15/dispatch_observation_20260906.json:3-13`).

There is one provenance warning: four source entries have current working-tree SHA256 values that differ from execution raw-byte SHA256 values but match after CRLF-to-LF normalization against the local Git blobs at the execution commit.

| path | execution bytes | current bytes | same after LF normalization |
|---|---:|---:|---|
| `modeling/trifusion/criterion.py` | 6101 | 6231 | true |
| `modeling/trifusion/experts/mamba.py` | 7270 | 7479 | true |
| `modeling/trifusion/experts/semantic_residual.py` | 15428 | 15882 | true |
| `protocols/rgbnt201_dev_v1.json` | 5376 | 5685 | true |

Because these normalize to the same Git content, this is a provenance **WARN**, not an integrity **FAIL**. The harder limitation is that this audit did not load checkpoint tensors, prototype-memory tensors, image files, feature tensors, or distance tensors. Those binary claims remain supported by source code and supplied verifier receipts, not by direct tensor replay.

## D. Execution integrity

Status: **PASS**

The runner defines a fixed configuration and asserts seed, source hashes, no dev/official access, no reranking, no gradient accumulation, and no new inference parameters (`tools/train_signal_preserving_v24.py:36-56`, `tools/train_signal_preserving_v24.py:310-346`). It builds models by strict-loading V12 checkpoints and binding the source-prototype candidate with zero new inference parameters (`tools/train_signal_preserving_v24.py:60-96`). Training uses weak/strong paired views, computes the weak pass, detaches weak features, computes strong original/prototype losses, performs one optimizer step, and updates prototype memory after optimizer update (`tools/train_signal_preserving_v24.py:157-198`).

Full endpoint training is fixed at 20 epochs with asserted step counts of 580, 560, and 540 per endpoint across folds (`tools/train_signal_preserving_v24.py:263-307`). Final checkpoints are saved, strict-reloaded, and evaluated read-only (`tools/train_signal_preserving_v24.py:422-451`). The terminal aggregation asserts total optimizer steps 3360, gallery records 3126, eligible queries 571, and view pairs 6720 before writing Q1 status (`tools/train_signal_preserving_v24.py:468-493`).

Log replay confirms 120 epoch rows, six final metric events, exact M0 event match, terminal aggregate match, and terminal status `Q1_FAIL`. This matches the supplied array/log verifier (`evidence/trifusion_v24_q1_array_verification_20260906.json:1439-1447`).

## E. Scope and claim boundaries

Status: **WARN**

The experiment is complete for V24 Q1, but the scope is deliberately narrow: single seed 42, train-internal complete-path OOF, reused-development qualification, final-epoch-only selection, no fixed dev, no official/test, and no D1 (`refine-logs/v24/EXPERIMENT_PLAN.md:85-93`, `refine-logs/v24/EXPERIMENT_PLAN.md:118-129`, `refine-logs/v24/EXPERIMENT_TRACKER.md:16-18`, `refine-logs/v24/EXPERIMENT_TRACKER.md:28-40`). The result report correctly states Q1_FAIL and no advancement (`results/TRIFUSION_RGBNT201_V24_COMPLETE_Q1_2026-09-06.md:3-10`, `results/TRIFUSION_RGBNT201_V24_COMPLETE_Q1_2026-09-06.md:117-129`). The complete comparison records no dev/official access and no D1 attempt (`evidence/trifusion_v24_q1_complete_comparison_20260906.json:57931-57943`).

## F. Claim impact

Status: **WARN**

Supported claims:

- V24 Q1 completed the fixed 3-fold x 2-endpoint execution plan.
- The evaluation is real-GT train-internal OOF, not synthetic or pseudo-label evaluation.
- M0 passed as an engineering gate.
- The reported metric arrays and summary arithmetic are internally consistent.
- The candidate has zero new inference parameters, with the scope caveat that training uses source-prototype memory.

Rejected or unsupported claims:

- Scientific success or method efficacy for V24 Q1.
- Advancement to D1, fixed development, official, or test protocols.
- Generalization beyond the audited train-internal reused-development qualification.
- Cross-family model acquittal of this audit.

## Limitations

- I did not load images, model weights, checkpoint tensors, prototype memory tensors, features, or distance matrices because the audit request prohibits model/tensor/image loading and inference.
- I replayed AP/rank aggregation from supplied AP and first-match-rank JSON arrays. AP from raw feature distances could not be recomputed because the necessary distance or feature arrays are not supplied and inference is prohibited.
- Checkpoint strict-load, unchanged-source-checkpoint, and prototype-memory binary claims are verified through source paths and supplied verifier receipts, not through binary tensor loading.
- Dispatch acceptance is not independent backend identity attestation. The supplied records support requested `gpt-5.5`/`xhigh` and accepted registered agent `/root/audit_v24_q1`, but do not prove the resolved backend model. GPT-5.5, if actually resolved, is GPT-lineage and same-family relative to the GPT-lineage executor.

## Machine-readable companion

The companion JSON report is `EXPERIMENT_AUDIT_V24_Q1.json` and includes the input-byte manifest, recomputation values, claim verdicts, source line-ending drift entries, and the A-F checklist statuses.
