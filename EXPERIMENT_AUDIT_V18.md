# Experiment Audit Report: V18 Paired-View Projection

**Date**: 2026-09-05
**Auditor**: Codex independent experiment-integrity auditor, direct primary-artifact read, no subagents by task instruction
**Project**: TriFusion RGBNT201 V18 paired-view projection
**Scope**: V18 implementation, frozen V18 plan, terminal Q1 evidence, endpoint/calibration receipts, postrun binding receipt, complete comparison artifact, projection-geometry diagnostic, V18 result writeup, and handoff sections 33-34.

## Overall Verdict: WARN

## Integrity Status: warn

The available evidence supports that V18 ran the frozen train-internal complete-path Q1 protocol and produced a real negative result. I found no fake ground truth, no self-normalized metric inflation, no dead metric path for the reported numbers, no hidden dev/official evaluation, and no evidence that a favorable subset was selected.

The overall integrity verdict remains **WARN** because the large remote checkpoint/source-cache binaries were not independently available as local bytes for this audit. The postrun binding receipt reports complete-file remote SHA-256 checks and JSON equality, but that remains a binding receipt, not independent auditor access to those remote bytes. The scientific advancement verdict is separate and is **FAIL_NO_ADVANCEMENT**: V18 completed Q1 but failed two frozen advancement gates.

## Scientific Advancement Verdict: FAIL_NO_ADVANCEMENT

The frozen plan required all Q1 gates to pass before D1/dev advancement: aggregate fused mAP gain at least +1.0, all fold fused gains nonnegative, all expert aggregate gains nonnegative, bootstrap lower bound positive, and projected fused above baseline plus all three branches (`docs/V18_PAIRED_VIEW_PROJECTION_PLAN_2026-09-05.md:49-54`). The Q1 summary records `Q1_FAIL`, `d1_executed=false`, and `next_phase_qualified=false` (`evidence/trifusion_v18_q1_seed42_2a71e20.json:3`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:287070-287078`).

| Frozen Q1 gate | Status | Evidence |
|---|---:|---|
| Projected fused aggregate gain >= +1.0 mAP point | FAIL | Recomputed and stored gain is +0.9215042005674405 (`evidence/trifusion_v18_q1_seed42_2a71e20.json:287053-287056`), below the predeclared +1.0 threshold (`docs/V18_PAIRED_VIEW_PROJECTION_PLAN_2026-09-05.md:49-51`). |
| All fold fused gains >= 0 | PASS | Stored and recomputed fold gains are +1.755786, +0.110911, +0.855081 (`evidence/trifusion_v18_q1_seed42_2a71e20.json:287060-287064`). |
| All expert aggregate gains >= 0 | PASS | Stored expert gains are CNN +0.249724, Transformer +0.903566, Mamba +1.837550 (`evidence/trifusion_v18_q1_seed42_2a71e20.json:287053-287058`). |
| Identity-cluster bootstrap 10000, seed42, 95% lower > 0 | FAIL | Stored lower bound is -0.117338 mAP with 21 clusters and 10000 resamples (`evidence/trifusion_v18_q1_seed42_2a71e20.json:287065-287069`); source calls `identity_cluster_bootstrap_lower_bound(..., seed=42, resamples=10000)` (`tools/train_signal_preserving_v18.py:284-285`). |
| Projected fused beats baseline and all branches | PASS | Projected fused mAP is 81.482001; baseline is 77.487603, CNN 79.548593, Transformer 79.417463, Mamba 80.702741 (`evidence/trifusion_v18_q1_seed42_2a71e20.json:287020-287050`). |

The bootstrap number itself is sourced to the terminal Q1 artifact. I independently recomputed all aggregate metrics and fold gains from the per-query arrays, and I verified the bootstrap source path and negative stored gate. Exact NumPy PCG bootstrap replay was not performed locally because the available local Python entry points do not provide NumPy and installing packages or running new inference/training was outside the read-only audit scope. A standard-library identity-cluster resampling check using the same cluster scheme also produced a negative lower bound, so the gate sign is not borderline for the advancement decision.

## Primary Evidence Files and Bindings

| Artifact | Status | Evidence |
|---|---:|---|
| Frozen V18 plan | Present and hash-bound | Q1 summary stores plan SHA `bb7bb3ca...e6d7` (`evidence/trifusion_v18_q1_seed42_2a71e20.json:9`); postrun binding matches the plan file SHA (`evidence/trifusion_v18_postrun_bindings_20260905.json:26-30`). |
| V18 runner and module | Present and hash-bound | Q1 summary records runner/module SHA values (`evidence/trifusion_v18_q1_seed42_2a71e20.json:7-8`); postrun binding matches both source files (`evidence/trifusion_v18_postrun_bindings_20260905.json:14-25`). |
| Terminal Q1 summary | Present | Q1 summary is `v18-paired-view-projection-main-v1`, `Q1_FAIL`, seed42, commit `2a71e209...` (`evidence/trifusion_v18_q1_seed42_2a71e20.json:2-9`). Local SHA-256 observed: `8c5f99fcd4ba218ac2925a01123e377415c8443b7ed89de9ec0da5f400415f20`. |
| Training terminal log | Present | Calibration lines appear for folds 0-2 (`C:/Users/gb/.codex_tmp/trifusion_v18_terminal_20260905/training.log:16`, `C:/Users/gb/.codex_tmp/trifusion_v18_terminal_20260905/training.log:26`, `C:/Users/gb/.codex_tmp/trifusion_v18_terminal_20260905/training.log:36`); Q1 final metrics are emitted for all six fold/endpoint combinations (`training.log:148`, `training.log:187`, `training.log:226`, `training.log:265`, `training.log:304`, `training.log:343`); terminal summary is `Q1_FAIL` (`training.log:344`). Local SHA-256 observed: `f4c6f61e8b2a021c49f89f0f5399441dfa31a24b4a37391aef7368f6b37cd9bc`. |
| Endpoint receipts | Present and equal to embedded Q1 objects | Postrun binding lists six endpoint receipt SHA values, optimizer steps, strict reload, and `endpoint_receipt_equals_summary=true` (`evidence/trifusion_v18_postrun_bindings_20260905.json:147-203`). Local JSON-equivalence recomputation also returned true for all six receipts. |
| Calibration receipts | Present and equal to embedded Q1 objects | Fold 0 calibration records source-only records/cache and heldout use 0 (`C:/Users/gb/.codex_tmp/trifusion_v18_terminal_20260905/fold_0_calibration.json:148-150`, `fold_0_calibration.json:70048-70050`); local JSON-equivalence recomputation returned true for all three calibration receipts. |
| Remote checkpoint/source-cache binding | Bound by receipt, not independently byte-read locally | Postrun receipt states the verification type is remote complete-file SHA and JSON equality with no new inference or optimizer (`evidence/trifusion_v18_postrun_bindings_20260905.json:2-6`), and lists all final checkpoints/source caches as SHA-matched (`evidence/trifusion_v18_postrun_bindings_20260905.json:56-145`). This is the reason for WARN rather than PASS. |
| Complete comparison artifact | Present and source-bound | The artifact points to the Q1 source SHA and `Q1_FAIL` (`evidence/trifusion_v18_complete_comparison_20260905.json:1-3`), records all-query AP/rank change counts (`evidence/trifusion_v18_complete_comparison_20260905.json:4-45`), and states no raw inference was added and all 571 legal queries/all 21 identities were used (`evidence/trifusion_v18_complete_comparison_20260905.json:584-585`). Local validation found source SHA match and no per-fold metric row mismatches against Q1. |
| Projection geometry diagnostic | Present and source-bound | Diagnostic JSON records `ALL_SIX_FINAL_HEAD_REPLAYS_MATCH`, read-only train-only scope, Q1 summary SHA, optimizer0/checkpoint-writes0/dev0/official0 (`evidence/trifusion_v18_projection_geometry_20260905.json:2-11`). Local SHA-256 observed: `55865c3b10c55871f9ccda48e84f6872750b3d1fad649d7b0bfff60ce9f9ad4f`. |

## Checks

### A. Ground Truth Provenance: PASS

The evaluation uses dataset identity and camera labels carried in the RGBNT201 records, not labels generated from model outputs. `_parse_record` derives the camera ID from the dataset filename and constructs the RGB/NI/TI triplet paths, failing if the modalities are not paired (`tools/run_signal_baseline_dev.py:27-37`). The protocol loader rejects non-blind dev protocols and validates the frozen 141-fit/30-dev identity registry (`tools/run_signal_baseline_dev.py:89-95`). The complete-path OOF builder splits records by heldout identity and preserves heldout records as gallery/query candidates (`tools/build_v12_complete_path_oof_targets.py:29-56`); it loads the frozen 141 train identities from `train_171` and checks the expected triplet count (`tools/build_v12_complete_path_oof_targets.py:156-173`).

V18 source calibration uses only `split["train_records"]`, extracts identities/cameras from those source records, asserts no identity overlap, and records `heldout_records_used_for_calibration=0` (`tools/train_signal_preserving_v18.py:63-99`). The Q1 calibration evidence records source record counts 2126/2075/2051 and heldout calibration use 0 (`evidence/trifusion_v18_q1_seed42_2a71e20.json:172`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:70223`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:140274`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:70074`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:140125`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:210176`).

Retrieval scoring uses `identities` and `cameras` arrays supplied from records (`tools/train_signal_preserving_v18.py:145-150`). `full_gallery_scores` marks eligible queries only when a cross-camera positive exists, then passes query/gallery identities and cameras to the ReID scorer (`tools/audit_v17_full_gallery.py:14-28`). The AP scorer removes same-identity same-camera junk and treats identity equality as the match criterion (`tools/diagnose_v6_oracle_complementarity.py:91-108`). No audited path derives ground truth from predictions or baseline outputs.

### B. Score Normalization: PASS

The metric computation uses standard retrieval metrics, not normalization by the model's own maximum/minimum output. V18 L2-normalizes feature vectors before pairwise distance computation (`tools/train_signal_preserving_v18.py:148-150`). The AP scorer sorts distances, removes same-camera same-identity junk, computes cumulative precision at true identity matches, and returns AP/Rank arrays (`tools/diagnose_v6_oracle_complementarity.py:91-108`). `full_gallery_scores` reports mAP as mean AP times 100 and Rank-k as the fraction of first-hit ranks inside k times 100 (`tools/audit_v17_full_gallery.py:34-43`).

I found no division by a model-specific max score, min score, or prediction statistic in the metric path. The high fold-1 values are accompanied by lower fold-0/fold-2 values in the same Q1 artifact, and all per-query arrays are available for recomputation.

### C. Result File Existence and Claim Binding: WARN

All claimed V18 terminal numbers I checked are present in primary artifacts and match after recomputation from per-query arrays. The terminal Q1 summary records seed42, commit `2a71e209...`, `envelope_enabled=false`, rank1 projection, 20 epochs per endpoint, no model selection, dev0, and official0 (`evidence/trifusion_v18_q1_seed42_2a71e20.json:2-23`). It records M0 as passed with 8-step capacity, 22/22 nonzero gradient tensors, no missing gradients, frozen state unchanged, directions unchanged, and peak reserved 1810 MiB (`evidence/trifusion_v18_q1_seed42_2a71e20.json:286851-286871`). The overfit gate records overflow0, 22/22 gradients, frozen/directions unchanged, and excess loss ratio 0.0006935 (`evidence/trifusion_v18_q1_seed42_2a71e20.json:286873-286984`).

The six final endpoint receipts exist locally and are embedded in the Q1 summary. Each records the expected projection flag and final reload/evaluation state; for example fold 0 uncentered has `projection_enabled=false`, 20 epochs, 580 optimizer steps, overflow0, 22 nonzero gradients, no missing gradient tensors, frozen state unchanged, final checkpoint SHA, strict reload, and read-only evaluation (`C:/Users/gb/.codex_tmp/trifusion_v18_terminal_20260905/fold_0_uncentered_receipt.json:406`, `fold_0_uncentered_receipt.json:410-413`, `fold_0_uncentered_receipt.json:624-625`, `fold_0_uncentered_receipt.json:1751`, `fold_0_uncentered_receipt.json:8741-8744`). Fold 0 projected has the same run shape with `projection_enabled=true` (`fold_0_projected_receipt.json:406`, `fold_0_projected_receipt.json:410-413`, `fold_0_projected_receipt.json:8741-8744`). Postrun binding records all six receipt hashes, optimizer steps 580/580/560/560/540/540, strict reload, and equality to the Q1 summary (`evidence/trifusion_v18_postrun_bindings_20260905.json:147-203`).

The aggregate Q1 table in the result writeup matches Q1 JSON: baseline 77.487603, projected fused 81.482001, branch metrics, and fused gain 0.921504 (`results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:15-21`; source values at `evidence/trifusion_v18_q1_seed42_2a71e20.json:286987-287058`). The per-fold result table matches the comparison artifact and Q1 terminal log (`results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:23-56`; `evidence/trifusion_v18_complete_comparison_20260905.json:321-386`; `training.log:148`, `training.log:187`, `training.log:226`, `training.log:265`, `training.log:304`, `training.log:343`). The all-query AP/rank change counts match the complete comparison artifact (`results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:58-70`; `evidence/trifusion_v18_complete_comparison_20260905.json:4-45`).

The WARN is about packaging/byte access, not number mismatch. The postrun receipt binds remote files by SHA and records complete-file equality (`evidence/trifusion_v18_postrun_bindings_20260905.json:2-6`, `evidence/trifusion_v18_postrun_bindings_20260905.json:56-145`), but I did not independently read the remote checkpoint/source-cache bytes locally. This matches the task instruction not to infer remote checkpoint byte verification from local receipts alone. A secondary provenance note: some nested `build_provenance` metadata still says `signal_preserving_v17_dtred` because V18 wraps the V17 correction head; the V18 runner overwrites the active binding architecture and projection fields (`tools/train_signal_preserving_v18.py:52-60`), and the Q1 top-level architecture/projection fields identify the actual V18 run (`evidence/trifusion_v18_q1_seed42_2a71e20.json:2-4`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:215590`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:224335`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:238342`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:247232`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:261504`, `evidence/trifusion_v18_q1_seed42_2a71e20.json:270744`).

### D. Metric Live Execution and Dead Code: PASS

The claimed Q1 metrics come from a live runner path. V18 imports `full_gallery_scores` (`tools/train_signal_preserving_v18.py:16`) and its `evaluate` function extracts features, L2-normalizes them, computes pairwise distances, and calls `full_gallery_scores` for every output (`tools/train_signal_preserving_v18.py:133-151`). In the Q1 loop, the runner builds every fold/endpoint, trains with `_fit_endpoint(..., envelope_enabled=False)`, strict-reloads the checkpoint, calls `evaluate`, appends every output's AP/rank arrays, prints the Q1 metric line, and writes the endpoint receipt (`tools/train_signal_preserving_v18.py:233-269`). The terminal log contains those six printed `Q1_final` records (`training.log:148`, `training.log:187`, `training.log:226`, `training.log:265`, `training.log:304`, `training.log:343`) and the terminal summary (`training.log:344`).

The projection-geometry diagnostic is also live with respect to the saved final heads. It asserts Q1 summary SHA, loads the prior frozen caches and checks their hashes, loads each final checkpoint by SHA, reconstructs the V18 head, computes features, calls `full_gallery_scores`, and asserts the recomputed score object equals the saved Q1 output (`tools/diagnose_v18_projection_geometry.py:24-44`, `tools/diagnose_v18_projection_geometry.py:51-60`, `tools/diagnose_v18_projection_geometry.py:87-93`). Its JSON records all six replays as exact and state unchanged (`evidence/trifusion_v18_projection_geometry_20260905.json:8029-8032`, `evidence/trifusion_v18_projection_geometry_20260905.json:35163-35166`, `evidence/trifusion_v18_projection_geometry_20260905.json:70722-70725`, `evidence/trifusion_v18_projection_geometry_20260905.json:97270-97273`, `evidence/trifusion_v18_projection_geometry_20260905.json:132435-132438`, `evidence/trifusion_v18_projection_geometry_20260905.json:161384-161387`). I found no claimed metric that depends on an uncalled metric function.

### E. Scope Assessment: PASS

The actual scope matches the frozen plan and the written claims. The plan fixed complete train-internal OOF Q1 as three folds x two endpoints x 20 epochs, seed42, no intermediate retrieval or epoch selection (`docs/V18_PAIRED_VIEW_PROJECTION_PLAN_2026-09-05.md:44-45`), all 47 heldout identities per fold retained in gallery, 3126 gallery records total, and all 571 legal queries used (`docs/V18_PAIRED_VIEW_PROJECTION_PLAN_2026-09-05.md:46-48`). The runner defines endpoints `uncentered` and `projected` (`tools/train_signal_preserving_v18.py:28-30`), loops folds/endpoints, trains each endpoint, writes final-only checkpoints, and computes aggregate over accumulated arrays (`tools/train_signal_preserving_v18.py:230-301`). It enforces final-only model selection and dev/official access counts in the report header (`tools/train_signal_preserving_v18.py:166-175`).

Recomputed scope from the Q1 gallery manifests and query arrays is fold0 1000 gallery/190 eligible/810 excluded, fold1 1051/179/872, fold2 1075/202/873. Totals are 3126 gallery, 571 eligible queries, 2555 excluded-as-query records, and 47 heldout identities per fold. These match the Q1 summary totals (`evidence/trifusion_v18_q1_seed42_2a71e20.json:287079-287080`) and comparison artifact scope (`evidence/trifusion_v18_complete_comparison_20260905.json:584-585`). The result writeup explicitly says the numbers are train-internal OOF and cannot be compared directly to 30-dev, official test, or SOTA (`results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:11-13`, `results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:96-104`).

Section 33 of the handoff matches the terminal artifacts: it states 3360 optimizer steps, 0 overflow, no intermediate selection, Q1 summary SHA, complete train-internal OOF scope, failed advancement, D1 false, and dev/official0 (`docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:1778-1804`). Section 33.3 matches the geometry diagnostic scope and aggregate claims (`docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:1843-1873`; source/evidence at `tools/diagnose_v18_projection_geometry.py:133-139` and `evidence/trifusion_v18_projection_geometry_20260905.json:2-11`, `evidence/trifusion_v18_projection_geometry_20260905.json:230317-230365`).

Section 34 is a SOTA/resource refresh note, not a V18 execution result. Its local source document explicitly says the public values are not server-reproduced, not exhaustive, and not fair same-protocol evidence (`docs/SOTA_REFRESH_2026-09-05.md:1-5`), and that the SOTA refresh does not change frozen V18 training or advancement conditions (`docs/SOTA_REFRESH_2026-09-05.md:73-75`). I did not independently re-download external SOTA PDFs or repositories in this bounded V18 audit, so no new SOTA verification is implied.

### F. Evaluation Type: PASS - real_gt train-internal OOF; diagnostic replay only

Classification: **real_gt** for the V18 Q1 retrieval evaluation, with the claim ceiling **train-internal complete-path OOF only**. It uses dataset-provided identities/cameras from the RGBNT201 record paths and protocol split (`tools/run_signal_baseline_dev.py:27-37`, `tools/build_v12_complete_path_oof_targets.py:156-173`) and standard ReID matching by identity after same-camera same-identity junk removal (`tools/diagnose_v6_oracle_complementarity.py:91-108`). It is not synthetic proxy, self-supervised proxy, simulation-only, or human evaluation.

The geometry diagnostic is a **read-only train-only postmortem replay**, not a new validation result. The diagnostic JSON records `scope=read_only_train_only_postmortem_not_new_validation`, optimizer0, checkpoint-writes0, dev0, official0 (`evidence/trifusion_v18_projection_geometry_20260905.json:2-11`). The source also writes those fields and no optimizer/training loop is created (`tools/diagnose_v18_projection_geometry.py:133-139`).

## Independent Recomputations Performed

I recomputed the following from the JSON arrays and manifests without running training or inference:

| Recomputed item | Result |
|---|---|
| Per-fold stored metric arrays vs stored fold metrics | Maximum absolute delta `7.105427357601002e-14`. |
| Aggregate metrics vs Q1 aggregate | Maximum absolute delta `9.947598300641403e-14`. |
| Query index consistency across all outputs/endpoints | No mismatches. |
| Gallery/query scope | Fold0 1000/190, fold1 1051/179, fold2 1075/202; total 3126 gallery and 571 legal queries. |
| Fold fused gains | `[1.7557860965124235, 0.11091134470029829, 0.8550812350757298]`. |
| Standalone calibration receipts vs embedded Q1 calibrations | All three JSON-equivalent; each has 94 fit IDs, 47 heldout IDs, heldout calibration use 0, frozen unchanged. |
| Standalone endpoint receipts vs embedded Q1 endpoints | All six JSON-equivalent; epochs 20, overflow0, strict reload true, read-only evaluation true, missing nonzero gradients 0. |
| Complete comparison artifact vs Q1 | Source SHA matches Q1; no per-fold metric-row mismatches; AP-improved/worse/equal counts partition all 571 queries for each output. |
| Geometry replay artifact vs Q1/source | Q1 summary SHA and diagnostic script SHA match; all six endpoint replays exact; optimizer0/checkpoint-writes0/dev0/official0. |

## V18 Terminal Result

Stored and recomputed aggregate metrics, percent units:

| Output | Uncentered mAP/R1/R5/R10 | Projected mAP/R1/R5/R10 | Projected minus uncentered mAP |
|---|---:|---:|---:|
| baseline_only | 77.487603 / 79.334501 / 89.492119 / 93.520140 | 77.487603 / 79.334501 / 89.492119 / 93.520140 | 0.000000 |
| fused | 80.560497 / 83.712785 / 88.966725 / 92.294221 | 81.482001 / 84.938704 / 90.017513 / 93.520140 | +0.921504 |
| cnn | 79.298869 / 81.961471 / 88.791594 / 90.718039 | 79.548593 / 82.486865 / 89.667250 / 91.068301 | +0.249724 |
| transformer | 78.513897 / 81.961471 / 90.192644 / 95.271454 | 79.417463 / 83.187391 / 91.593695 / 94.570928 | +0.903566 |
| mamba | 78.865192 / 82.837128 / 87.390543 / 90.893170 | 80.702741 / 83.362522 / 90.367776 / 92.469352 | +1.837550 |

Primary source lines: `evidence/trifusion_v18_q1_seed42_2a71e20.json:286987-287058`.

## V18 Geometry Diagnostic Claims

The geometry diagnostic supports only a narrow postmortem explanation, not a new advancement result. The diagnostic source recomputes all six final heads from SHA-bound caches/checkpoints and asserts exact score equality with Q1 (`tools/diagnose_v18_projection_geometry.py:24-44`, `tools/diagnose_v18_projection_geometry.py:51-60`, `tools/diagnose_v18_projection_geometry.py:87-93`). It asserts projected corrected residual coefficients are below `1e-6` (`tools/diagnose_v18_projection_geometry.py:96-104`). Local recomputation from the diagnostic JSON found maximum absolute projected direction coefficient `1.6763806343078613e-08`.

| Output | Positive distance delta | Negative distance delta | Nearest-margin delta | Negative same-camera ratio uncentered -> projected | R1 repaired/broken |
|---|---:|---:|---:|---:|---:|
| baseline_only | 0.000000 | 0.000000 | 0.000000 | 67.6007% -> 67.6007% | 0 / 0 |
| fused | -0.006439 | +0.002267 | +0.008706 | 63.5727% -> 60.0701% | 10 / 3 |
| cnn | -0.009467 | -0.000633 | +0.008834 | 63.5727% -> 60.2452% | 12 / 9 |
| transformer | -0.002700 | +0.001041 | +0.003741 | 66.5499% -> 63.2224% | 11 / 4 |
| mamba | -0.007448 | +0.005348 | +0.012796 | 63.5727% -> 58.3187% | 13 / 10 |

Primary source lines: `evidence/trifusion_v18_projection_geometry_20260905.json:230317-230365`. The result writeup's geometry claims match these values (`results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:113-124`), and handoff section 33.3 matches them with the correct read-only boundary (`docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:1843-1873`).

The diagnostic does not support claims that camera is the sole causal factor, that rank/direction/epoch changes are authorized, or that V18 qualifies for D1/dev/official evaluation.

## Claim Impact

| Claim | Impact | Reason |
|---|---:|---|
| V18 M0 engineering gate passed | Supported | Q1 summary records M0 capacity and overfit pass fields with no missing gradients, unchanged frozen state/directions, and low overfit excess ratio (`evidence/trifusion_v18_q1_seed42_2a71e20.json:286851-286984`). |
| V18 completed full frozen Q1, three folds x two endpoints x 20 epochs, seed42 | Supported | Frozen plan defines that protocol (`docs/V18_PAIRED_VIEW_PROJECTION_PLAN_2026-09-05.md:44-48`); Q1/header and postrun endpoint bindings show seed42, final-only selection, six endpoints, total 3360 optimizer steps (`evidence/trifusion_v18_q1_seed42_2a71e20.json:2-23`; `evidence/trifusion_v18_postrun_bindings_20260905.json:147-205`). |
| Projected fused improves over matched uncentered fused by +0.921504 mAP | Supported | Stored and recomputed from per-query arrays; source values at `evidence/trifusion_v18_q1_seed42_2a71e20.json:287027-287032` and gain at `evidence/trifusion_v18_q1_seed42_2a71e20.json:287053-287056`. |
| V18 Q1 scientifically advances to D1/dev | Unsupported | Aggregate gain < +1.0 and bootstrap lower bound < 0; `next_phase_qualified=false`, `d1_executed=false` (`evidence/trifusion_v18_q1_seed42_2a71e20.json:287065-287078`). |
| V18 provides a new 30-dev, official-test, SOTA, or deployable result | Unsupported | Q1/result writeup explicitly scope it as train-internal OOF and dev/official0 (`results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:11-13`, `results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md:96-104`; Q1 dev/official0 at `evidence/trifusion_v18_q1_seed42_2a71e20.json:22-23`). |
| V18 projection geometry replay verifies the saved Q1 arrays | Supported with packaging qualifier | Diagnostic source asserts exact equality and JSON records six exact replays (`tools/diagnose_v18_projection_geometry.py:87-93`; `evidence/trifusion_v18_projection_geometry_20260905.json:8029-8032`, `evidence/trifusion_v18_projection_geometry_20260905.json:35163-35166`, `evidence/trifusion_v18_projection_geometry_20260905.json:70722-70725`, `evidence/trifusion_v18_projection_geometry_20260905.json:97270-97273`, `evidence/trifusion_v18_projection_geometry_20260905.json:132435-132438`, `evidence/trifusion_v18_projection_geometry_20260905.json:161384-161387`). The remote binary-byte caveat still applies. |
| Handoff section 33 terminal and diagnostic claims | Supported with same boundaries | Section 33 values match Q1/comparison/geometry source artifacts (`docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:1778-1873`). |
| Handoff section 34 SOTA/resource refresh | Not a V18 experiment-integrity claim | Local refresh doc explicitly marks public values as not server-reproduced and not exhaustive (`docs/SOTA_REFRESH_2026-09-05.md:1-5`) and says the refresh does not change V18 conditions (`docs/SOTA_REFRESH_2026-09-05.md:73-75`). External SOTA sources were not independently rechecked in this bounded audit. |

## Action Items

- Preserve V18 as a completed negative Q1 result: do not run D1/dev/official, and do not relax rank, direction estimation, epoch count, LR, fold, or gate thresholds under the V18 plan.
- If a future audit must become PASS rather than WARN, make the final checkpoints, frozen source checkpoints, and source-cache bytes independently available to the auditor, or provide a reproducible immutable artifact bundle whose bytes can be hashed directly by the auditor.
- Keep V18 claims limited to train-internal complete-path OOF: projected fused gives +0.921504 mAP over the matched uncentered endpoint but fails the frozen advancement contract.
- Keep the projection-geometry result framed as read-only postmortem diagnosis. It supports average margin/same-camera-neighbor changes and exact replay of saved arrays, but it does not establish a causal single-factor explanation or authorize another V18 rerun.
