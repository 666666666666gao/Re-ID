# Experiment audit: six fixed-best region-use diagnoses

**Verdict: WARN — no unresolved blocking finding.** The 30 saved score rows and all reported numerical comparisons are consistent. A factual epoch typo was found, corrected by the executor, and independently checked. Remaining warnings concern the evidence boundary and a preserved transient source change.

Review independence: **same-family**. Acceptance: **provisional**. Requested reviewer: `gpt-6-astra`, effort `max`; actual runtime model and effort are **UNATTESTED**. This report does not certify them.

Artifact root W: `D:/Program Files/UserCache/gb/codex_tmp/independent_evidence_draft/evidence_comparison903`. Paths below are relative to W. `S/` means `sealed_source_snapshot911/`; `R/` means `readonly_results919/`; `C/` means `frozen_region_use_draft912/`. Every reference is to an exact file and line. The full final SHA map is in EXPERIMENT_AUDIT.json.

## Deterministic evidence

- 3556/3556 main checks and 24/24 bounded text-correction checks passed: **3580 final passing checks**. All 31 requested files exist.
- Verified 465 local source pins, 13 intake-file pins, six result SHA against campaign, and 489 hashed local input files. The admission receipt reports 90 remote artifact checks; these were not repeated locally.
- Independently aggregated 31,420 saved AP/rank records across 30 score rows into mAP/R1/R5/R10 (120 values). Recomputed all 24 comparisons, including the six required fused comparisons: 96 Rank-1 groups, 25,136 paired query records, 1,056 identity aggregates, AP contributions, percentiles and worst-5% deltas.
- All 30 CSV rows and 18 numeric Markdown table rows match. All three 50-epoch independent histories were read and their selected-best and selected-epoch differences checked. No raw distance binary was reranked.
- Pure stdlib Python used: `C:/Users/gb/AppData/Roaming/uv/python/cpython-3.13-windows-x86_64-none/python.exe`. The independent algorithms are retained in `check_saved_evidence.py` and `check_text_delta.py`; their saved outputs belong to the input states recorded in those receipts. The main historical check SHA map is intentionally preserved; the final map overlays only the two verified changed text files.

## A. Ground truth and protocol — PASS within stated boundary

The actual protocol root is selected by `S/tools/queue_native_research.py:20`, used by `C/tools/diagnose_region_use_best.py:60`. Records supply dataset identity, camera and scene; no model output creates retrieval labels. The metadata path is `S/tools/official_three_dataset_data.py:8` and `S/tools/diagnose_native_research_best.py:45`. All 27,266 protocol train/query/gallery metadata records agree with identity/camera/scene parsed from their dataset filenames. This checks manifest labels, not image contents or original inventory completeness.

| Dataset | Q | Full G | Environment exclusion | Query identities | Legal positives per Q |
|---|---:|---:|---|---:|---:|
| RGBNT201 | 836 | 836 | Same identity + same camera | 30 | 7–21 |
| RGBNT100 | 1715 | 8575 | Same identity + same camera | 50 | 50–175 |
| MSVR310 | 591 | 1055 | Same identity + same scene | 52 | 1–31 |

Dataset parsing is explicit at `S/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:79`, `RGBNT100.py:76`, and `msvr310.py:81` in that same directory. Per-query scoring sorts all gallery distances, removes only same-identity/same-environment entries, retains all different-identity negatives, and requires a legal positive (`S/tools/train_rgbnt100_signal_oof.py:253`; `S/tools/train_msvr310_signal_oof.py:223`). Every query has one or more legal positives.

All five scores call `S/tools/diagnose_native_research_best.py:51`, which verifies the loaded scorer path and checks mAP/R1/R5/R10 against the pinned author implementation within 1e-5. The author exclusions and AP/CMC code are `S/comparators/Signal-cd1b0a6/utils/metrics.py:68` and `:137`. This is **real_gt** evaluation as established by source and saved metadata, not synthetic-proxy evaluation. Author-scorer agreement remains inherited from completed source-bound execution, since raw matrices were not local audit inputs.

## B. Score normalization and formulas — PASS

`S/tools/run_official_three_dataset_roles.py:230` L2-normalizes G/C/F vectors and computes squared Euclidean distance, giving cosine ordering. This is representation normalization, not division of reported performance by prediction maxima.

For each of nine corresponding role/modality spaces, R normalizes the 16 actual 128-wide region vectors, takes each query region’s maximum cosine over gallery regions and vice versa, averages both directions and all spaces. This is symmetric many-to-one MaxSim, not one-to-one correspondence or part/visibility ground truth (`C/tools/diagnose_region_use_best.py:32`). J uses distance `2 - 2 * 0.5 * (global_cosine + R)` (`:105`–`:109`). The coefficient and the symmetric region formula match the existing training score underlying the loss margin (`S/modeling/trifusion/region_use_objective.py:6` and `:30`). Training weighting, legal triplet masks and loss are not claimed to be an inference scorer. No score fitting, sweep or prediction-max metric normalization appears in this active path.

## C. Results, numbers and terminal status — PASS after text correction

Six DIAGNOSIS files match campaign SHA at `R/logs/region_use_fixed_best_diagnosis_20261013_918/campaign.json:22`, `:33`, `:44`, `:55`, `:66`, `:77`; the full intake pins are `R/INTAKE_MANIFEST_919.json:3`. Each DIAGNOSIS `:7` binds the original selected epoch, `:8` its checkpoint SHA, `:10` zero updates, `:11`/`:12` equal state SHA, and `:13` recorded fused-distance max difference 0.

The maximum difference between recomputed saved fused metrics and seal metrics is **2.95416606377e-06 percentage points**, below the registered 1e-5. Maximum same-model-global versus independent-same-epoch saved metric difference is **5.20918806046e-06 points**, also below 1e-5. This is tolerance agreement, not byte or weight identity.

| Dataset/condition | Epoch | F−G mAP points | R1 repairs/new errors | J−G mAP points |
|---|---:|---:|---:|---:|
| RGBNT201 uniform | 8 | +0.491385 | 6/6 | −3.329888 |
| RGBNT201 ambiguity | 8 | +0.333237 | 8/7 | −2.871146 |
| RGBNT100 uniform | 26 | −0.679685 | 10/28 | −11.562607 |
| RGBNT100 ambiguity | 26 | −0.619533 | 9/23 | −13.158692 |
| MSVR310 uniform | 38 | −0.000352 | 0/1 | −17.508367 |
| MSVR310 ambiguity | 38 | +0.003630 | 0/1 | −17.642067 |

Recomputed fused AP groups are sourced at each RGBNT201 DIAGNOSIS `:22170`, RGBNT100 DIAGNOSIS `:45224`, and MSVR310 DIAGNOSIS `:16020`, under the campaign path above. Their group counts partition all queries, and their AP contributions sum to F−G mAP. Report tables are `readonly_summary919/REPORT.md:9`, `:20`, `:34`; CSV begins at `readonly_summary919/SCORES.csv:2`.

The controller records COMPLETE and six child exits 0; supervisor EXIT is 0 at `R/logs/region_use_fixed_best_diagnosis_admin918/EXIT.json:6`. The saved late observation records both parent PIDs and six children absent (`OBSERVATION_LATE_918.stdout:1`), and the observer command exit is 0 (`OBSERVATION_COMMAND_LATE_918.json:2`). Admission source count, artifact count, GPU-memory gate and storage pass are `launch918/ADMISSION.json:4`–`:9`. These are receipt checks, not a fresh remote process or GPU query.

**TEXT-1, corrected:** Original REPORT line 29 and analyzer line 65 said independent RGBNT100 “epoch9 best.” The supplied history has best epoch **7**, mAP **84.53378422597531**, and both the 50-row maximum and SUMMARY agree (`R/trained-model/native_research_v6_20261003_794_full_global_only_RGBNT100/training.json:149`, `:679`; `readonly_summary919/SUMMARY.json:55234`). The exact edit was `epoch9 best` → `epoch7 best`. The final text delta check proves only this substitution in both files, verifies original SHA against the already-audited versions, and checks numerical inputs unchanged. Original copies are retained as `REPORT.md.before_TEXT1` and `analyze_readonly_results919.py.before_TEXT1` in this review directory. No score correction remains.

## D. Active hook, definitions and gallery use — PASS

The current entry calls `frozen.extract` (`C/tools/diagnose_region_use_best.py:89`). Its forward hook stacks `evidence.cnn`, `.transformer` and `.mamba` from the actual `roles` output and checks shape `(3,3,16,128)` (`S/tools/diagnose_same_evidence_comparison.py:45`–`:59`). The training head captures the same fields (`S/modeling/trifusion/region_use_heads.py:14`). These are final region values, not query/key tensors.

The exact feature path computes correction via `read_evidence`, raw fused as `shared_global + gain * correction`, and fused as its L2 normalization (`S/modeling/trifusion/independent_native_roles.py:66`–`:74`; `S/modeling/trifusion/correspondence_evidence_readout.py:65`). The current consumer checks that identity (`C/tools/diagnose_region_use_best.py:92`), processes all Q/G records, checks saved Q/G metadata equality (`:96`–`:98`), builds all five distances (`:101`–`:109`) and calls the scorer for each (`:110`). Every active score appears in all six results. Historical helper/visualization branches are not evidence of an uncalled current score.

Best-selection checks bind to the original training history/checkpoint rather than selecting by this diagnosis (`C/tools/diagnose_region_use_best.py:74`–`:84`). The controller has six cases and no retry/reselection path (`C/tools/queue_region_use_best_diagnosis.py:34`–`:74`). Source checks and recorded state equality support a read-only execution interpretation; this reviewer did not inspect live tensors. Pairwise R/J change the comparison/resource regime and are not the single-1536-vector deployment or teachers.

## E. Scope and claim limits — PASS

The corrected report states one seed42, development-consumed benchmarks, fixed selected bests and no teacher/compression claim (`readonly_summary919/REPORT.md:27`, `:30`, `:43`). All six J scores are lower than G in mAP and Rank-1. That constrains this comparator; it does not establish the absence of all complementary local evidence or the cause of deployment weakness. The correction/region standalone caveat is present and appropriate.

The same-model G comparison uses independent-history epochs 8/26/38. Independent best epochs are 8/**7**/38 respectively. RGBNT100’s gap to its independent best therefore cannot all be assigned to shared-backbone harm. Saved text-metric agreement does not imply feature or weight equality, and same-model G is not a separate ablation. No significance, cross-seed robustness or general method-causality claim is supported or made. No new training or teacher experiment is authorized by this verdict.

## F. Classification and retained review history

**Evaluation type:** `real_gt`. **Audit mode:** source tracing plus saved per-query result verification. **Overall:** WARN, same-family/provisional, zero unresolved blockers.

TRACE-1: During initial source reading, analyzer line 32 briefly excluded query/identity arrays while the saved SUMMARY retained them. The executor reported reverting that temporary edit without rerunning the producer. The reviewer directly checked the restored line and SHA; all full comparison objects match the saved DIAGNOSIS files. The pre-restoration source SHA was not captured and is not invented. This is preserved in `FIRST_OBSERVED_DISCREPANCY.json`. No claim of source immutability across the entire review is made.

REVIEWER-1: The first deterministic run produced 3 failed protocol-byte bindings out of 3,550 checks because the reviewer picked an older protocol directory. Inspecting the actual queue binding identified the correct directory. The correct SHA matches all six initializer bindings; the old/new parsed protocol contents are identical. The first checker and output remain intact as `check_saved_evidence.initial.py` and `DETERMINISTIC_CHECKS.initial.json`; the corrected 3,556-check run passes. This is an auditor path error, not a hidden experiment failure.

Raw image integrity, independently reranked distances, live state/gradient equality, repeated-run determinism and original historical generator execution remain outside this audit. The recorded zero fused-distance difference is one comparison and must not be upgraded to checkpoint/distance-file byte equality or repeatability proof. No unresolved exact edit is required after TEXT-1 was closed.

## Key final SHA-256

| Artifact | SHA-256 |
|---|---|
| `readonly_diagnosis918/INPUT_SEAL.json` | `8e5c37e57c088a8ab53344266195bbc7183ccd27a94d0f0910ff9c20998b4b41` |
| `analyze_readonly_results919.py` | `ae5953251cf061cfa938fdb3f0223603785ec57cd7241c28ff368bdd8d2f01b9` |
| `readonly_summary919/SUMMARY.json` | `f74bfbe8014c4879c242b486e39cb1d634390932d1e8758549018022623d9190` |
| `readonly_summary919/SCORES.csv` | `74e9c7d79c399a6a60534cdca544741f5336ee41f905352483b1a5b320c02626` |
| `readonly_summary919/REPORT.md` | `ede99d2592ef8fb7dd9eba0f39ae74593def52dd90f7dddabd4d8fc52f3c4676` |

All additional audited hashes, check counts, exact finding references and limitations are recorded in `EXPERIMENT_AUDIT.json`, `DETERMINISTIC_CHECKS.json` and `TEXT_DELTA_CHECK.json`.
