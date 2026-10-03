# Fresh experiment integrity audit: F3 metric feature scale

**Overall: WARN. `integrity_status: warn`; `blocking_findings: []`.** The six formal trainings and six strict evaluations, followed by the first original CPU report, have consistent source and primary-record evidence. The original controller failed; a separately attributed storage-only finish completed the missing evaluation/report. This is not retrospective success of the failed worker.

**Review route: fresh gpt-6-astra / max / fork-none; `review_independence: same-family`; `acceptance_status: provisional`.** This is a source and primary-text audit, not cross-family assurance or model/tensor replay. The reviewer independently inspected the primary files and executed local standard-library hash, JSON, count, arithmetic and AST checks. No SSH, experiment imports, GPU, inference, scoring/report execution, deletion or evidence edits were performed. Only these two audit outputs were written; private trace metadata remains the parent’s responsibility.

## Exact scope and result

F3 is six fresh seed42 runs: normalized versus raw **Triplet input** on RGBNT201, RGBNT100 and MSVR310. Both arms use normalized BN/CE input and normalized 1536-dimensional deployment features. Each formal run completes 50 epochs; there are 300 formal epochs and 20,416 formal steps, preceded by six separate eight-step M0 probes.

All 265 sealed source files and all 126 primary text files (49,731,550 bytes) match their catalog/intake hashes. The inherited 243-file foundation and 259-file F2 source bindings remain intact. All 237 sealed Python files parse as AST. The companion JSON records SHA256 and byte counts for **424 inspected/hash-checked inputs**, including supplementary original transport, retirement and finish-observer evidence. It separately lists the 24 externally attested current F3 binaries; their hashes were not recomputed from local tensor files.

| Check | Verdict | Main conclusion |
|---|---|---|
| A — GT provenance | PASS | Full official metadata, valid positives, correct camera/scene filtering and MSVR distractors. |
| B — normalization | PASS | Feature normalization and ordinary GT-based AP/CMC denominators; no score manipulation found. |
| C — existence/fidelity | WARN | All records consistent; remote binary assurance and composite completion have explicit limits. |
| D — executed paths | PASS | Triplet-only intervention and unchanged deployment occur in the actual call path. |
| E — scope/causality | WARN | Matched one-seed panel; official selection and bootstrap/cost limits remain. |
| F — evidence classification | PASS | Six formal results are real_gt; M0/pair checks are engineering probes. |

No source patch or numerical report correction is indicated by this audit. The warning qualifications below are required when using the results.

## A. Ground truth and full evaluation — PASS

Filename-derived identity, camera and scene fields agree with supplied protocol records; split indices/paths and train label mappings were recounted. Train/test identities are disjoint. Every query has a positive after the legitimate filter. Counts are:

| Dataset | Train / query / gallery records | Train / query / gallery IDs | Minimum valid positives | Gallery-only IDs / records |
|---|---:|---:|---:|---:|
| RGBNT201 | 3951 / 836 / 836 | 171 / 30 / 30 | 7 | 0 / 0 |
| RGBNT100 | 8675 / 1715 / 8575 | 50 / 50 / 50 | 50 | 0 / 0 |
| MSVR310 | 1032 / 591 / 1055 | 155 / 52 / 155 | 1 | 103 / 464 |

RGBNT scorers exclude same-identity/same-camera items; MSVR excludes same-identity/same-scene items. Different-identity gallery distractors remain. The full evaluation loader does not shuffle or subsample. Although an inherited transform-building helper constructs a doubled template, the actual loader constructs a single-copy ImageDataset from the exact supplied records. No model-generated GT is on this path.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/official_three_dataset_data.py:8`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_official_three_dataset_roles.py:58`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/train_signal_preserving_v18.py:33`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_correspondence_roles.py:79`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/train_rgbnt100_signal_oof.py:253`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/train_msvr310_signal_oof.py:223`.

The metadata checks do not establish byte-level fidelity of remote dataset images. The official-protocol roots changed from F2 only as recorded; all other protocol content agrees.

## B. Normalization and score denominators — PASS

F3 computes raw and L2-normalized features from the same forward without detaching the training-loss path. BN/CE always receives normalized features, Triplet receives the designated raw/normalized features, and deployment always returns the normalized embedding. Distance construction uses normalized embeddings and squared Euclidean distance. AP divides precision at GT-positive ranks by the number of valid positives; CMC averages across valid GT queries. These are standard GT denominators, not model-statistic score scaling.

Attention normalization remains internal to feature extraction: the pinned CLIP MultiheadAttention output feeds its residual block, not a metric denominator (`C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/comparators/Signal-cd1b0a6/modeling/clip/model.py:172`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/comparators/Signal-cd1b0a6/modeling/clip/model.py:223`).

Both independently implemented camera/scene scorers and the pinned author scorer are reached and compared at the unchanged 1e-5 tolerance. The float64 independent CMC arithmetic can differ slightly from author float32 CMC; the recorded values remain within that tolerance. No reranking, post-hoc score rescaling or metric denominator based on prediction statistics was found.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_metric_feature_scale.py:26`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_official_three_dataset_roles.py:230`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_correspondence_roles.py:95`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/comparators/Signal-cd1b0a6/utils/metrics.py:95`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/comparators/Signal-cd1b0a6/utils/metrics.py:155`.

## C. Existence, numbers and completion — WARN

All six M0 receipts record exactly eight steps, all 155 trainable tensors with gradients and zero strict save/reload prediction difference. Formal runs bind fresh public visual/camera/head initial states rather than warm-starting the M0 probes. All formal step and batch records are contiguous and agree with the epoch histories. B64/K8 labels, cameras and paths agree with actual train metadata. The formal totals are 2,649 steps per RGBNT201 arm, 6,559 per RGBNT100 arm and 1,000 per MSVR310 arm. Identity sampling can drop insufficient final groups, so full50 is a complete 50-epoch sampler budget, not a claim that every image appears exactly once per epoch.

| Dataset | Variant | Selected epoch | Steps | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | normalized | 26 | 2649 | 62.580552 | 62.440193 | 75.837320 | 83.014351 |
| RGBNT100 | normalized | 12 | 6559 | 77.584292 | 94.169098 | 95.335275 | 95.860058 |
| MSVR310 | normalized | 24 | 1000 | 50.356738 | 67.512691 | 81.387478 | 86.294419 |
| RGBNT201 | metric_raw | 38 | 2649 | 59.734344 | 60.287082 | 72.966510 | 82.416266 |
| RGBNT100 | metric_raw | 48 | 6559 | 75.451985 | 94.752187 | 95.510203 | 95.860058 |
| MSVR310 | metric_raw | 23 | 1000 | 52.759551 | 71.404397 | 84.940779 | 89.847714 |

The selected epochs are the unique mAP maxima in their 50-epoch histories and conform to the sealed latest-tie rule. All four reported metrics use that same selected checkpoint. The largest strict reload difference is **0.000001452251595424059 mAP points**, for normalized MSVR310; all other selected-versus-strict metric differences are zero. Every error is below the unchanged 0.00001-point requirement. All train/evaluate terminal JSON records agree with their receipts. Report table rounding, all costs/peaks/norm ranges, nonzero-Triplet counts, last-minus-selected metrics and pair arithmetic agree with their underlying records. Loss decomposition differs only by ordinary float32 rounding (maximum about 4.77e-7); source-order LR and epoch mean arithmetic agree.

AMP starts at scale 256, asserts finite loss/gradients, and rejects a scale decrease after scaler.step/update. The optimizer covers every trainable parameter. M0 checks gradient support, visual/camera updates and frozen-state preservation. Completed histories support those executed source guards; there is no separate per-step AMP scale telemetry.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:197`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:233`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:250`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:264`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:287`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/report_metric_feature_scale.py:33`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/results/metric_feature_scale_complete_20261003/REPORT.md:5`. The companion JSON gives all six exact training/step/batch/strict-receipt paths and metric values.

### Original failure and the separately attributed finish

1. Original controller **2691826** is archived **FAILED**. Worker **2762610** exited **1** after the RGBNT100 metric_raw training process **2762617** completed all 50 epochs with exit0. The unchanged 10GiB disk guard fired before its independent evaluation row/process was created. Original parent/child bytes and traceback remain archived.
2. Exactly **24 closed historical M0 probes** were retired, totaling **8,444,122,668 bytes**. The executable path verifies closed summaries/owners, each exact real path/size/SHA and M0 receipt, protected historical formal bests, public CLIP, 23 then-existing current F3 binaries and the 265-file source seal before deleting only prelisted files. The receipt records all 24 deleted. The 24 historical formal bests and current F3 M0/best/distance artifacts remain protected. Historical deleted M0 tensors can no longer be replayed directly.
3. The first finish deployment failed **before Popen** because the local retirement JSON had 2,490 CRLF line endings (134,866 bytes), while the remote receipt had LF (132,376 bytes). Decoded JSON is equal, and replacing CRLF with LF produces exact remote bytes. The original failure is retained. The second deployment binds the actual received remote bytes and uses a new asset directory; helper/spec/plan bytes are unchanged. No scientific source, tolerance or model setting changed.
4. Finish controller **3070294**, separately recorded on physical **GPU1**, launched the first missing evaluation **3070549** once, against the original epoch48 checkpoint. Evaluation exit0 is recorded, all six strict rows verify, and the first **unmodified original CPU report** exits0. No training restart occurred. Finish completion is **2026-10-03 11:51:59.274711 +08:00**. A separate observer at **11:53:24.349609 +08:00** records `controller_live=false`.
5. The aggregate completed job contains `original_failure`, immutable original-failure SHA and explicit composite provenance. Its original command/GPU2 fields describe the original training context; its updated PID/exit describe the finish controller. Do not present that row as the original worker succeeding.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_storage_finish766_20261003/original_campaign.json:2`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_storage_finish766_20261003/original_campaign.json:856`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_storage_finish766_20261003/original_child_campaign.json:6`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_storage_finish766_20261003/original_worker_failure.log:6`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/queue_foundation_recipe.py:130`; `C:/Users/gb/.codex_tmp/retire_closed_m0_766.py:22`; `C:/Users/gb/.codex_tmp/retire_closed_m0_766.py:51`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/retirement/TRANSPORT_DIAGNOSIS.json:3`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766.py:44`; `C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py:19`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_storage_finish_source766_v2_20261003/finish_metric_feature_scale766.py:81`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_storage_finish_source766_v2_20261003/finish_metric_feature_scale766.py:101`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_storage_finish_source766_v2_20261003/finish_metric_feature_scale766.py:114`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_20261003_v1/campaign.json:901`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_20261003_v1/campaign.json:928`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_storage_finish766_20261003/STATUS.json:277`; `C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/finish_observer/STATUS.json:12`.

The interval check covers 18 recorded model processes: six M0, six training and six strict evaluations. It finds maximum concurrency four, physical GPUs 0–3 only, no same-card overlap and all M0 completions before the first formal start, using GPU1 for the actual finish evaluation. No broader remote-process inventory or live environment replay was performed by this review.

### Numerical assurance boundary

All 24 current remote binary size/SHA attestations agree with receipts; the 23 protected pre-finish files retain their bindings and the newly produced last official distance file completes 24. No tensor file was opened here. Pair score differences, identity counts, identity-macro/weighted AP arithmetic and repairs-minus-errors/Rank1 arithmetic were independently checked from JSON. Individual query AP, repair/error membership and bootstrap resamples were not regenerated from remote arrays; their existence and values remain primary-report evidence supported by the inspected analysis source and hashes. This is why C is WARN despite no detected numerical blocker.

## D. Actual model, evaluator and report path — PASS

Executed path: `run_metric_feature_scale.main` calls its configure, replaces the foundation builder/loader/loss, then invokes `run_foundation_recipe.main`. Imported F2 captured the original functions; its own main-guard configure does not run on import. F3 calls the original current-package builder, wraps its Signal global model/current BN classifier, and changes only the Triplet feature input. The norm diagnostics detach only their measurements.

Current construction binds the public CLIP visual tensors, fresh camera parameters and fresh 1536-dimensional head. The pinned Signal implementation runs with USE_A/USE_B false and concatenates three 512-dimensional modal features. The temporary construction scaffold does not activate roles or adapters in the retained global-only forward. Inherited author-recipe branches, old F2 trainer behavior and legacy evaluation functions are not substituted for F3.

Actual evaluation is `foundation.evaluate -> runner.official_metrics -> extract -> the full evaluation loader`, then normalized distance construction, independent camera/scene scoring and pinned-author scoring. Strict serialization loads the full model state with schema/dataset/condition/protocol checks. Actual terminal reporting is `report_metric_feature_scale.main -> base.verify` for every accepted row, then `analyze_correspondence_distances.compare` on saved distances; it requires the complete matrix, report count 1 and an absent output directory. The finish invokes that exact sealed report.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_metric_feature_scale.py:36`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_metric_feature_scale.py:64`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_training_feature_scale.py:17`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_training_feature_scale.py:84`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_clean_clip_joint.py:62`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_visual_update_control.py:30`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:180`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:300`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_correspondence_roles.py:66`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/report_metric_feature_scale.py:22`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/analyze_correspondence_distances.py:28`.

## E. Matched contrast and claim limits — WARN

Fresh initial model state, visual/camera/head initialization, trainable capacity, protocol, head, optimizer, losses, augmentation recipe, B64/K8, seed42 and all 50-epoch budget match within each pair. Capacity is 86,407,680 parameters for RGBNT201, 86,224,896 for RGBNT100 and 86,386,176 for MSVR310, with 155 trainable tensors each. Both fresh arms also have identical actual logged label/camera/path batch-order bytes to each other and the historical F2 normalized arm. The initial paired forward records identical CE logits/deployment features while the intended Triplet feature differs. Metadata order parity does not constitute archived augmented-pixel replay.

The fixed gate is **delta mAP >=0.5 points and delta Rank1 >=0**. The registered rule is unchanged; it is a development criterion, not significance. The following deltas use the independent scorer arithmetic from the original CPU report:

| Dataset | Raw minus normalized mAP | Rank1 | Repairs / new errors | Identity-macro AP delta | Fixed-model identity bootstrap95% interval | Gate |
|---|---:|---:|---:|---:|---|---|
| RGBNT201 | -2.846208 | -2.153110 | 63 / 81 | -2.792201 | [-7.424828, +1.681966] | FAIL |
| RGBNT100 | -2.132307 | +0.583090 | 52 / 42 | -2.252955 | [-4.766854, +0.378859] | FAIL |
| MSVR310 | +2.402813 | +3.891709 | 45 / 22 | +2.545318 | [+0.066222, +5.077635] | PASS |

Only MSVR310 passes. RGBNT100 has a small Rank1 increase but worse mAP; RGBNT201 loses both. These unfavorable results must remain in any conclusion. The bootstrap uses 2,000 identity resamples with seed42 for fixed selected models; it does not measure training-seed uncertainty, multiple-selection uncertainty or a fresh held-out test. Each checkpoint is selected on the official set, and F3 follows earlier consumed official results.

The supported interpretation is a **conditional Triplet-input effect within this fixed current-package/normalized-BN-CE panel**. F2 changed both CE and Triplet inputs, so F2/F3 differences are not independent seed replicates or automatic isolated CE causality. This does not establish a general gain, SOTA, a new module, roles/adapters validity or an explanation of the full F1 author/current gap. Training time includes epoch evaluations, excludes construction/final strict evaluation and varies with placement/concurrency; memory is observed peak allocated memory. Neither supports a controlled deployment-speed claim.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md:7`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md:20`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md:37`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/check_metric_feature_scale_pair.py:26`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/check_metric_feature_scale_pair.py:50`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_training_feature_scale.py:50`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:133`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:252`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/analyze_correspondence_distances.py:58`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/report_metric_feature_scale.py:44`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/report_metric_feature_scale.py:59`.

## F. Evidence classification — PASS

All six formal retrieval evaluations are **real_gt**, based on full official query/gallery metadata and images consumed by the recorded evaluation. M0 reload/gradient and initialization-pair parity tests are **engineering probes**, not retrieval evidence. Identity diagnostics use real GT on fixed models. No synthetic_proxy, self_supervised_proxy, simulation_only or human_eval result is presented as the formal retrieval outcome.

Evidence: `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:264`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py:300`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/check_metric_feature_scale_pair.py:65`; `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_correspondence_roles.py:88`.

## Findings, corrections and hash record

**Blocking findings: none within the audited source/text scope.** Nonblocking findings W1–W5 are: one-seed official selection; remote-binary/array replay limits; composite completion with preserved original failure; permanent loss of direct replay for 24 retired historical M0 probes; and finite metadata/AMP/cost evidence. Their exact claim impacts and file:line evidence are in the companion JSON.

No source patch or recalculation is requested. Preserve the existing report/plan qualifications, report the storage-only finish explicitly, and carry the same-family/provisional and replay limitations into any downstream claim. Deterministic checks are accepted only for the precise local hash/count/JSON/arithmetic/AST scope; this does not upgrade the scientific claim or review independence.

The complete input hash inventory is `C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/reviewer_audit766/EXPERIMENT_AUDIT.json:1`. Key immutable bindings:

| Input | SHA256 |
|---|---|
| C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_storage_finish766_20261003/original_campaign.json | `c8664424fcf1f7769f9c87d21860e2d2c2a1aca88516449db7413e63604dc9aa` |
| C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_feature_scale_storage_finish766_20261003/original_child_campaign.json | `4edc2472993bf9613dbd6ae07cfcbf7063546511d48b47d3515a7ae517700e49` |
| C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/closed_m0_retirement766_20261003/RETIREMENT.json | `63956ff16251c212b37ec1264c58878e4ccab46ffed1721f6f8da499ebba4645` |
| C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_storage_finish_source766_v2_20261003/finish_metric_feature_scale766.py | `22807e06f47b387e77218f2dc7ec72393a265d6fd184495542345507749ca2f6` |
| C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/logs/metric_storage_finish_source766_v2_20261003/FINISH_SPEC.json | `8a3bf5d8a9575150f415360fd30dba87da565eeb8531521636432cd71e7e1dfc` |
| C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/results/metric_feature_scale_complete_20261003/SUMMARY.json | `1899c3fbb53e29f4a8e53bf063317c83a1232aba8885b5f0c66dea5a213c1050` |
