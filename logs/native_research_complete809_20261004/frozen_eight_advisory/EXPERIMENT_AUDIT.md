**Experiment integrity audit — native_research_v6**

**Overall: WARN.** The supplied eight completed V6 endpoints are internally consistent real-ground-truth results. No fake GT, self-referential score normalization, mismatched selected-epoch CMC, or retroactive parity PASS was found on the inspected path. The nine-arm experiment and terminal CPU report are incomplete in this snapshot; remote binary existence and runtime reproduction were not established. Claims remain limited to single-seed, post-selection, unequal-capacity system comparisons. The added training-loss floor is mathematically and numerically consistent but does not measure CE and Triplet separately.

Reviewer: **gpt-6-astra / max**; review_independence=**same-family**; acceptance_status=**provisional**. Generated 2026-10-03 23:43:25 UTC. Local publication HEAD: `10b77166c655b550cfd951c38b6e93d9d1a64324`. This is an advisory static/source/receipt audit, with no model execution or server connection.

Evidence roots: R = `C:/Users/gb/.trifusion_github_publish_22c3bee`; S = `C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source`; P = `C:/Users/gb/.codex_tmp/independent_evidence_draft`. Every citation below links to the exact local file and starting line; a displayed range describes the inspected span. Actual file hashes are recorded at the end and in the JSON report.

All 43 primary paths in the request were read directly. The additional loss analysis, its script, its bound loss source and all eight referenced training-step logs were also inspected. Only the two requested audit reports were written.

| Check | Verdict | Meaning |
|---|---|---|
| A. Ground truth provenance and full protocol evaluation | **PASS** | Dataset metadata supplies GT; full registered query/gallery. |
| B. Metric denominators and score normalization | **PASS** | Conventional AP/CMC denominators; no self-reference. |
| C. Result existence, bindings and completion status | **WARN** | Eight formal receipts agree; final endpoint/report and binary authentication unresolved. |
| D. Active, inactive and not-yet-executed paths | **WARN** | Active path traced; historical and pending checks distinguished. |
| E. Scientific scope and claim strength | **WARN** | Descriptive one-seed, selected, unequal-capacity comparisons only. |
| F. Evaluation category | **PASS** | real_gt, within the explicit evidence boundary. |

**Observed formal results**

These are values in the eight local COMPLETE receipts, checked against all 50 history entries and the latest-tie maximum-mAP rule. The maximum difference for selected epoch and all CMC values is zero. Each row is seed 42/full 50; mAP and CMC are percentage points. The JSON report retains full precision and all reported checkpoint/distance hashes.

| Dataset | Arm | Selected epoch | mAP | R1 | R5 | R10 | Receipt |
|---|---|---:|---:|---:|---:|---:|---|
| RGBNT201 | global_only | 8 | 74.2967 | 78.9474 | 88.2775 | 91.8660 | [R/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_official_metrics.json:25) |
| RGBNT201 | semantic | 7 | 71.8981 | 74.4019 | 84.3301 | 90.1914 | [R/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_official_metrics.json:25) |
| RGBNT201 | native | 20 | 72.1273 | 75.1196 | 85.1675 | 89.3541 | [R/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_official_metrics.json:25) |
| MSVR310 | global_only | 38 | 50.5421 | 68.0203 | 80.5415 | 85.4484 | [R/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_official_metrics.json:25) |
| MSVR310 | semantic | 49 | 50.9636 | 69.2047 | 80.7107 | 86.1252 | [R/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_official_metrics.json:25) |
| MSVR310 | native | 38 | 50.6755 | 68.6971 | 81.5567 | 85.9560 | [R/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_official_metrics.json:25) |
| RGBNT100 | global_only | 7 | 84.5338 | 96.6181 | 97.3178 | 97.9592 | [R/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_official_metrics.json:25) |
| RGBNT100 | semantic | 5 | 83.4395 | 96.0933 | 96.6764 | 96.9679 | [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_official_metrics.json:25–32](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_official_metrics.json:25) |

RGBNT100 native has a supplied **M0_PASS**, not a formal retrieval endpoint. Its one eight-step probe records 299/299 covered trainable tensors, eight effective optimizer updates, BN counters 8, fourteen detail parameters with cumulative active gradients/nonzero final updates and strict-reload difference 0. [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:75–96](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:75); [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:2204–2225](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:2204). The last supplied status reports the fresh50 job at 1/50 around 06:43 +08:00; this is a historical snapshot, not a live-process observation.

| Paired comparison | Dataset | ΔmAP | ΔR1 | +0.5mAP / nondeclining-R1 gate |
|---|---|---:|---:|---|
| semantic − global_only | RGBNT201 | -2.3985 | -4.5455 | Not met |
| native − semantic | RGBNT201 | 0.2291 | 0.7177 | Not met |
| semantic − global_only | MSVR310 | 0.4215 | 1.1844 | Not met |
| native − semantic | MSVR310 | -0.2881 | -0.5076 | Not met |
| semantic − global_only | RGBNT100 | -1.0943 | -0.5248 | Not met |

Neither completed native-minus-semantic pair meets the registered developmental gate. This supports a mixed/no-consistent-gain description for the completed snapshot, not a universal operator-failure conclusion. RGBNT100 native has no inferred score.

**A. Ground truth provenance and full protocol evaluation — PASS**

Within the supplied protocol snapshots, retrieval labels are dataset identity and camera/scene metadata, not model outputs. Both project and upstream author scorers consume full query/gallery arrays. Filename and manifest checks found no mislabeled metadata or invalid queries; original image bytes and the source inventory itself were not supplied for authentication.

| Dataset | Train | Query | Gallery | Train IDs | Query IDs | Valid positives/query | Minimum negatives/query | Invalid queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RGBNT201 | 3951 | 836 | 836 | 171 | 30 | 7–21 | 806 | 0 |
| MSVR310 | 1032 | 591 | 1055 | 155 | 52 | 1–31 | 1021 | 0 |
| RGBNT100 | 8675 | 1715 | 8575 | 50 | 50 | 50–175 | 8375 | 0 |

Independent metadata checks covered 27,266 records and 3,142 queries, with zero failures and no train/gallery identity overlap. RGBNT201 legitimately reuses the test collection as query and gallery; same-identity/same-camera exclusion removes self/same-camera matches. The manifests are bound by their actual SHA-256 values, while image-byte authenticity remains unaudited.

Evidence:

- [R/tools/official_three_dataset_data.py:8–39](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py:8) — Builds paths and labels from protocol records; uses the selected dataset evaluation loader.
- [R/tools/run_correspondence_roles.py:60–108](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_correspondence_roles.py:60) — Extracts all query/gallery rows, passes protocol IDs/cameras/scenes to both scorers, checks upstream module path and metric equality, saves distances with metadata.
- [R/tools/train_rgbnt100_signal_oof.py:253–268](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_rgbnt100_signal_oof.py:253) — Camera scorer defines relevance by identity and excludes same identity plus camera.
- [R/tools/train_msvr310_signal_oof.py:223–238](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr310_signal_oof.py:223) — Scene scorer defines relevance by identity and excludes same identity plus scene.
- [S/comparators/Signal-cd1b0a6/utils/metrics.py:42–106](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/utils/metrics.py:42) — Upstream MSVR relevance/exclusion/AP/CMC use dataset IDs and scenes.
- [S/comparators/Signal-cd1b0a6/utils/metrics.py:125–168](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/utils/metrics.py:125) — Upstream camera protocol uses dataset IDs and camera exclusion.
- [S/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:73–85](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:73) — Author identity/camera filename parsing.
- [S/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:63–84](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:63) — Author identity/camera filename parsing.
- [S/comparators/Signal-cd1b0a6/data/datasets/msvr310.py:74–87](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/data/datasets/msvr310.py:74) — Author identity/camera/scene filename parsing.
- [R/logs/training_feature_scale_protocols_20261002/RGBNT201.json:78309–78316](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json:78309) — Registered full counts: train 3951, query/gallery 836/836.
- [R/logs/training_feature_scale_protocols_20261002/MSVR310.json:38538–38545](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/MSVR310.json:38538) — Registered full counts: train 1032, query/gallery 591/1055.
- [R/logs/training_feature_scale_protocols_20261002/RGBNT100.json:218978–218985](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT100.json:218978) — Registered full counts: train 8675, query/gallery 1715/8575.

Checks completed:

- Parsed all three full protocol JSON files; 27,266 train/query/gallery records and 3,142 query rows.
- Checked filename identity/camera/scene against records, contiguous record indices, train labels, train/gallery identity disjointness, and each query's positive/negative counts.
- Observed zero metadata failures and zero invalid queries. Different-identity gallery records remain negatives, including those in the same camera/scene.
- Found no teacher, predicted embedding, model score or generated reference substituted for retrieval ground truth on the active V6 scoring path.

Limits:

- Dataset file paths and source-inventory SHA provide provenance claims, not a local image-byte authenticity check.
- The currently inspected protocol builder is not in SOURCE_SCOPE; it is contextual source, not proof that these exact builder bytes created the manifests.

**B. Metric denominators and score normalization — PASS**

AP divides summed precision at relevant ranks by the number of valid positives; mAP and CMC average over valid queries. Every manifest query is valid. Multiplication by 100 converts ratios to percentage points. No division by the model's own score, a selected arm, or a baseline was found. Feature L2 normalization changes the distance representation, not the score denominator.

Claim impact: The supplied formal numbers are conventional percentage mAP/CMC under the specified protocol. The loss-floor extension is a separate analytic training-loss diagnostic, not a replacement metric.

Evidence:

- [R/tools/train_rgbnt100_signal_oof.py:253–268](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_rgbnt100_signal_oof.py:253) — Project camera AP and CMC denominators and percentage conversion.
- [R/tools/train_msvr310_signal_oof.py:223–238](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr310_signal_oof.py:223) — Project scene AP and CMC denominators and percentage conversion.
- [S/comparators/Signal-cd1b0a6/utils/metrics.py:83–106](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/utils/metrics.py:83) — Upstream valid-query and positive-count denominators.
- [S/comparators/Signal-cd1b0a6/utils/metrics.py:143–168](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/utils/metrics.py:143) — Upstream camera AP and valid-query CMC/mAP.
- [R/tools/run_correspondence_roles.py:93–102](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_correspondence_roles.py:93) — Project/upstream equality check at unchanged tolerance 1e-5; converts upstream ratios to percentages.
- [R/tools/run_official_three_dataset_roles.py:230–238](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_official_three_dataset_roles.py:230) — L2 normalization and squared Euclidean distance construction.
- [R/tools/run_independent_native_evidence.py:118–122](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py:118) — Logs raw feature norm separately from deployment feature norm.

**C. Result existence, bindings and completion status — WARN**

All eight local full50 training receipts and their official receipts exist, parse, and agree exactly on latest-tie mAP-best epoch and all four metrics. The ninth supplied artifact is an M0 engineering probe, not a formal result. Checkpoint/distance hashes are recorded but their binary bytes and physical remote existence are outside this local audit.

Claim impact: Supports eight receipt-consistent formal endpoints and the final M0 engineering status. Does not establish formal9/9, a completed final CPU report, remote binary availability, resumed process state, or an independent re-evaluation.

Evidence:

- [R/tools/run_foundation_recipe.py:246–257](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:246) — Selects maximum official mAP, replacing on ties; saves one matching best checkpoint and distances.
- [R/tools/run_foundation_recipe.py:287–312](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:287) — Fresh build, exact condition/strict reload, history argmax and all-metric consistency, binary hashing and final receipt writing.
- [R/tools/queue_foundation_recipe.py:70–107](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_foundation_recipe.py:70) — Queue requires M0, 50 contiguous epochs, seed42, selected metrics, actual runtime file/hash checks.
- [R/tools/queue_native_research.py:115–140](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_native_research.py:115) — Each arm M0 then fresh50/evaluation; only after all nine accepted can final report be invoked.
- [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:3](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:3) — M0_PASS status, not COMPLETE retrieval.
- [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:75–96](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:75) — One eight-step probe, 299/299 gradient coverage, zero reload difference, best_epoch null.
- [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:107–108](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:107) — Eight effective optimizer updates recorded.
- [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:2204–2225](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json:2204) — Author BN counters and fourteen detail parameters/update diagnostics.
- [R/refine-logs/native_research_v6/EXPERIMENT_TRACKER.md:3–15](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/EXPERIMENT_TRACKER.md:3) — Published snapshot says formal8/9, all9M0, native100 pending and final CPU report not invoked.
- [R/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:16082–16090](C:/Users/gb/.trifusion_github_publish_22c3bee/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:16082) — Explicit remote-only checkpoint/distance physical verification boundary; native100 pending.
- [R/refine-logs/native_research_v6/SOURCE_SCOPE.json:312–316](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/SOURCE_SCOPE.json:312) — V6 runner/queue/check/report/plan bound in source map.

Checks completed:

- Eight histories contain exactly epochs 1..50, finite loss and matching selected mAP-best metrics. Latest-tie selection was independently recomputed.
- All official receipts report seed42, 50 epochs, matching condition/protocol/initializer binding and same-epoch mAP, Rank-1, Rank-5 and Rank-10. Maximum discrepancy is 0.
- Verified actual local protocol, training, official-receipt and step-log SHA-256 values. All source bindings used by the added loss analysis match SOURCE_SCOPE.
- Inspected only the supplied final RGBNT100-native M0 directly. The earlier eight M0 PASS claims are document/queue-contract evidence here, not eight additional independently audited M0 receipts.

Actions/claim boundaries:

- Keep RGBNT100 native formal metrics empty until its original full50, strict reload/evaluation, queue verification and terminal report are actually complete.
- Preserve the referenced M0 probes and exact checkpoint/distance files until the existing verification/report dependency closes.
- At any later authorized runtime acceptance, check the reported SHA-256 values against actual binary files; this audit does not perform that action.

**D. Active, inactive and not-yet-executed paths — WARN**

The V6 wrapper installs the partitioned independent-native author-head/author-loss path before the foundation training entry runs. Legacy OOF/teacher training, normalized-head fallback behavior and alternative current-recipe loss are not the active V6 path. Final CPU paired analysis is implemented but not executed in the supplied snapshot; M0 diagnostics are engineering checks and cannot substitute for it.

Claim impact: Source presence demonstrates an implemented route, not that every proposed diagnostic ran. The prior backward-repeatability FAIL remains FAIL; V6 deliberately changes its prerequisite scope without retroactive PASS.

Evidence:

- [R/tools/run_native_research.py:8–30](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_research.py:8) — V6 schema/entry wrapping and installation into partitioned entry.
- [R/tools/run_native_partitioned.py:14–60](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_partitioned.py:14) — Uses original independent-native builder and foundation training with shared full-batch visual partitioning.
- [R/tools/run_independent_native_evidence.py:100–122](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py:100) — Author optimizer/loss selected; wrapper forces author recipe and records raw/deployment norms.
- [R/tools/run_independent_native_evidence.py:177–212](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py:177) — Installs hooks; CLI seed 42/full 50 and fresh build/evaluation dispatch.
- [R/modeling/trifusion/evidence_author_heads.py:21–73](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/evidence_author_heads.py:21) — Freezes historical normalized classifier heads, uses forward_features plus author BN/classifiers/raw features, author loss and optimizer.
- [R/tools/run_foundation_recipe.py:155–162](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:155) — Active author branch sums head totals; alternative branch's explicit id/triplet fields are not V6 component measurements.
- [R/tools/run_correspondence_roles.py:79–108](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_correspondence_roles.py:79) — Only reusable official scoring function is active; historical training at 139-242 is not V6.
- [S/comparators/Signal-cd1b0a6/utils/metrics.py:173–276](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/utils/metrics.py:173) — Metric wrapper/visualization classes and their dummy-state paths are not called by V6 free-function scoring.
- [R/tools/build_v12_complete_path_oof_targets.py:244–258](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/build_v12_complete_path_oof_targets.py:244) — Historically named _build_signal_teacher constructs fresh Signal; its name alone is not a teacher-weight dependency.
- [R/tools/queue_native_research.py:81–140](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_native_research.py:81) — V6 performs initialization witness/pair check and per-arm M0; no old backward-repeatability prerequisite call.
- [R/tools/queue_native_partitioned.py:180–191](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_native_partitioned.py:180) — Old partitioned campaign backward check is historical V5 behavior, not V6 queue execution.
- [R/tools/report_native_partitioned.py:22–59](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_partitioned.py:22) — Terminal report gates on completed9/9/18 records, verifies steps/batch order, builds comparisons and cost summaries.
- [R/tools/report_native_research.py:15–39](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_research.py:15) — Adds query AP/rank details only when final report is called.
- [R/refine-logs/native_research_v6/EXPERIMENT_TRACKER.md:13–15](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/EXPERIMENT_TRACKER.md:13) — Old parity remains unresolved and CPU report remains pending.
- [R/refine-logs/CURRENT_GOAL.md:59](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/CURRENT_GOAL.md:59) — Raw norms/head totals are not a separately measured loss decomposition or detail-exit amplitude.

Actions/claim boundaries:

- Continue distinguishing implemented, receipt-observed, historical and pending checks in reports.
- Do not describe unlogged detail attention/entropy/exit amplitude or CE-versus-Triplet components as measured.
- Keep the original 1e-5 checks and historical failure records; no tolerance change or history correction is warranted by this audit.

**E. Scientific scope and claim strength — WARN**

The current documents generally state the appropriate limitations. The evidence is a seed42, official-set-selected comparison of complete trainable systems with matched shared initialization and author-origin recipes. Added capacity, retrained shared backbone/adapters and possible RNG differences prevent a causal claim that native detail evidence alone caused the score differences. The observed deltas are mixed and do not establish necessity, stable generalization or SOTA.

Claim impact: Accept descriptive, receipt-bound differences for completed endpoints and the statement that current evidence does not consistently support the intended gain. Do not claim universal operator uselessness, causal failure mechanism, statistical significance, training-seed robustness, untouched test-set generalization, necessity or SOTA.

Evidence:

- [R/refine-logs/native_research_v6/EXPERIMENT_PLAN.md:7–11](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/EXPERIMENT_PLAN.md:7) — Matched source/public CLIP/camera/head/shared initialization, full protocols, author recipe and batch scope; native adds159296 parameters.
- [R/refine-logs/native_research_v6/EXPERIMENT_PLAN.md:16–26](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/EXPERIMENT_PLAN.md:16) — Fresh50 after M0, mAP-best same-epoch CMC, +0.5mAP/R1 gate is developmental; capacity/necessity/multiseed evidence pending.
- [R/tools/run_independent_native_evidence.py:36–90](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py:36) — Shared initial hashes and fresh weights; all152 visual tensors/camera are trainable; seed 42/full 50 recipe metadata.
- [R/modeling/trifusion/evidence_author_heads.py:6–18](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/evidence_author_heads.py:6) — global_only retains shared backbone/adapters; it is not a no-adapter F1 base.
- [R/modeling/trifusion/image_native_evidence.py:15–46](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/image_native_evidence.py:15) — Native adds trainable convolution/query/key/value/output detail path over all512 positions; not annotated parts or an explicit local-window guarantee.
- [R/tools/run_foundation_recipe.py:246–257](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:246) — Official retrieval evaluation each epoch and selection of maximum mAP consume the official set.
- [R/tools/analyze_correspondence_distances.py:57–75](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_correspondence_distances.py:57) — 2000 fixed-model identity resamples; identity-macro mean/interval explicitly distinct from training seeds.
- [R/refine-logs/CURRENT_GOAL.md:7–16](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/CURRENT_GOAL.md:7) — Separates global/semantic/native questions and stronger pending goals.
- [R/refine-logs/CURRENT_GOAL.md:52–59](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/CURRENT_GOAL.md:52) — Discloses official-set use for R&D and distinguishes diagnostic evidence from causal mechanisms.
- [R/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:15935–15937](C:/Users/gb/.trifusion_github_publish_22c3bee/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:15935) — Unequal capacity, trainable visual backbone, no equal-capacity necessity control and timing boundaries disclosed.
- [R/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:16073–16092](C:/Users/gb/.trifusion_github_publish_22c3bee/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:16073) — Mixed results, final endpoint/report pending and goal unmet.
- [R/refine-logs/CURRENT_GOAL.md:14–22](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/CURRENT_GOAL.md:14) — Minor stale B128-awaiting-measurement sentence is contradicted by the latest M0-PASS section.

Actions/claim boundaries:

- Keep native-minus-semantic separate from semantic-minus-global_only; preserve the fact that global_only includes shared adapters.
- Label any eventual bootstrap interval as fixed-model identity-macro uncertainty, not training-seed uncertainty or automatically an interval for query-weighted mAP.
- Treat +0.5mAP with nondeclining R1 as the stated project gate, not a significance test.
- Clarify CURRENT_GOAL.md:14 on the next authorized documentation edit: B128 is now supported by the recorded real-batch M0, while RGBNT100-native full50 remains pending in this snapshot.
- Stronger causal/necessity/stability/SOTA claims require their separately authorized matched-capacity/complete-removal/multiseed/strong-reference evidence; no new experiment is requested or launched here.

**F. Evaluation category — PASS**

This is supervised retrieval evaluation against recorded dataset identities with camera/scene exclusions. The real_gt category describes the label/scorer path; it does not certify image authenticity, independent held-out method selection, physical remote checkpoints, or a runtime reproduction.

Evidence:

- [R/tools/official_three_dataset_data.py:8–18](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py:8) — Training label or identity and camera/view come from dataset records.
- [R/tools/run_correspondence_roles.py:85–107](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_correspondence_roles.py:85) — Model outputs supply distances; protocol records supply ground-truth metadata.
- [S/comparators/Signal-cd1b0a6/utils/metrics.py:125–168](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/utils/metrics.py:125) — Ground truth is identity equality after same-camera exclusion.

**Additional loss-floor analysis — PASS within its analytic scope**

The active author criterion uses epsilon 0.1 label smoothing, ID weight 0.25 and nonnegative no-margin soft-margin Triplet weight 1.0. RGBNT201 has 171 classes/one head; MSVR310 155/three heads; RGBNT100 50/three heads. The sum is over author raw-feature heads. The alternative implementation that separately logs `id` and `triplet` is not the V6 branch.

For the smoothed target q, with q_y=0.9+0.1/K and q_j=0.1/K, CE(q,p)=H(q)+KL(q||p). Therefore, in exact real arithmetic,

`L − n_heads × 0.25 × H(q) = 0.25 × Σ KL(q||p_head) + Σ Triplet_head ≥ 0`.

The residual bounds the summed mean Triplet term after the same averaging; it is **derived**, not separately measured Triplet. It does not establish Triplet=0, identify which term explains a large residual, or establish a causal scale/overfitting/detail mechanism. Logged floating-point subtraction is not a certified real-number upper bound. The report explicitly preserves this distinction.

| Dataset/arm | Ideal weighted CE floor | Best epoch | Best loss | Best residual | Last loss | Last residual | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| RGBNT201/global_only | 0.208593149 | 8 | 0.277299907 | 0.068706758 | 0.225934504 | 0.017341355 | [P/native_research_completed_loss_floor/ANALYSIS.json:862–877](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:862); [P/native_research_completed_loss_floor/ANALYSIS.json:878–893](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:878) |
| RGBNT201/semantic | 0.208593149 | 7 | 0.303814583 | 0.095221434 | 0.226628955 | 0.018035806 | [P/native_research_completed_loss_floor/ANALYSIS.json:1730–1745](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:1730); [P/native_research_completed_loss_floor/ANALYSIS.json:1746–1761](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:1746) |
| RGBNT201/native | 0.208593149 | 20 | 0.249552707 | 0.040959558 | 0.226195151 | 0.017602002 | [P/native_research_completed_loss_floor/ANALYSIS.json:2598–2613](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:2598); [P/native_research_completed_loss_floor/ANALYSIS.json:2614–2629](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:2614) |
| MSVR310/global_only | 0.618081529 | 38 | 2.496271712 | 1.878190183 | 2.462797420 | 1.844715891 | [P/native_research_completed_loss_floor/ANALYSIS.json:3568–3583](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:3568); [P/native_research_completed_loss_floor/ANALYSIS.json:3586–3601](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:3586) |
| MSVR310/semantic | 0.618081529 | 49 | 2.459225110 | 1.841143580 | 2.472505518 | 1.854423989 | [P/native_research_completed_loss_floor/ANALYSIS.json:4542–4557](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:4542); [P/native_research_completed_loss_floor/ANALYSIS.json:4560–4575](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:4560) |
| MSVR310/native | 0.618081529 | 38 | 2.469467112 | 1.851385583 | 2.436677354 | 1.818595824 | [P/native_research_completed_loss_floor/ANALYSIS.json:5516–5531](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:5516); [P/native_research_completed_loss_floor/ANALYSIS.json:5534–5549](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:5534) |
| RGBNT100/global_only | 0.526548419 | 7 | 0.672285387 | 0.145736968 | 0.539489506 | 0.012941087 | [P/native_research_completed_loss_floor/ANALYSIS.json:6490–6505](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:6490); [P/native_research_completed_loss_floor/ANALYSIS.json:6508–6523](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:6508) |
| RGBNT100/semantic | 0.526548419 | 5 | 0.832198180 | 0.305649761 | 0.542115742 | 0.015567324 | [P/native_research_completed_loss_floor/ANALYSIS.json:7464–7479](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:7464); [P/native_research_completed_loss_floor/ANALYSIS.json:7482–7497](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:7482) |

Independent full-file checks covered 400 epochs and 16,323 step records. All 24 receipt/step hash bindings, five source bindings, three protocol hashes/class counts, recorded configs, selected epochs, duplicated first/best/last objects and metric values agree. Recomputed epoch loss means, head means, floor/residuals and metric values have maximum difference 0. Sum-of-head-means versus recorded total differs by at most 1.1069433991650612e-7, consistent with separate floating-point summation. All residuals are positive. The Python analysis script was read, not executed.

- [S/comparators/Signal-cd1b0a6/layers/softmax_loss.py:16–33](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/layers/softmax_loss.py:16) — Active epsilon 0.1 target smoothing and mean CE implementation.
- [S/comparators/Signal-cd1b0a6/layers/triplet_loss.py:113–135](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/layers/triplet_loss.py:113) — No-margin SoftMarginLoss on hard-mined raw-feature distances; nonnegative in real arithmetic.
- [S/comparators/Signal-cd1b0a6/layers/make_loss.py:13–56](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/layers/make_loss.py:13) — Selects label-smoothed CE plus Triplet with config weights.
- [R/tools/run_foundation_recipe.py:155–162](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py:155) — Only active author head totals are logged; alternative component-logging branch is unused.
- [R/tools/run_independent_native_evidence.py:118–122](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py:118) — Forces author recipe for V6.
- [C:/Users/gb/.codex_tmp/analyze_native_completed_loss_floor.py:26–40](C:/Users/gb/.codex_tmp/analyze_native_completed_loss_floor.py:26) — Binds five source files and trace/local publication head.
- [C:/Users/gb/.codex_tmp/analyze_native_completed_loss_floor.py:57–99](C:/Users/gb/.codex_tmp/analyze_native_completed_loss_floor.py:57) — Binds receipts/steps/protocol/config and computes the floor/residual for50 epochs.
- [P/native_research_completed_loss_floor/ANALYSIS.json:7503–7504](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json:7503) — Explicit analytic identity, precision boundary and no component/causal/seed/SOTA claims.

**Source bindings and execution boundary**

All314 expected source bytes were found locally. Twenty-seven working-copy files are not byte-identical to the manifest: their CRLF-normalized SHA and their exact archived copies match the expected hashes. Actual repository hashes remain recorded; no file was normalized or changed. This local resolution is not proof of remote runtime imports.

Of 314 manifest entries, 180 matched repository bytes directly, 107 were resolved from the archived source where the repository file was absent, and 27 repository CRLF variants were resolved from exact archived copies. All 43 requested primary inputs exist; no primary input was silently substituted. The currently inspected protocol-builder source is unbound contextual evidence. Installed `mamba_ssm`, CUDA and PyTorch binaries are outside this audit. [S/modeling/trifusion/experts/mamba.py:29–32](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/modeling/trifusion/experts/mamba.py:29).

| Path/check | Attribution |
|---|---|
| queue_native_research -> run_native_research -> run_native_partitioned -> run_independent_native_evidence -> run_foundation_recipe | active path supported by source and V6 receipt bindings |
| AuthorHeadEvidence.forward -> core.forward_features -> raw1536 -> author BN/classifier head(s) -> author make_loss -> sum heads | active; one head RGBNT201, three heads vehicle datasets; shared visual/camera/adapters trainable |
| run_correspondence_roles.official_metrics -> full records_for/extract -> L2 distance -> project camera/scene scorer plus author free eval_func/eval_func_msrv | active formal scorer; parity recorded, not re-executed by reviewer |
| check_native_research_pair and per-arm production M0 checks | V6 engineering prerequisite; only supplied final native100 M0 directly inspected |
| queue_native_partitioned old backward-repeatability gate | historical V5 prerequisite removed explicitly for V6, not retroactively passed |
| train_*_signal_oof legacy OOF training/evaluation; run_correspondence_roles historical teacher training; normalized legacy classifier forward; author metric wrapper visualization classes | not active V6 methods, despite helper imports |
| report_native_research -> report_native_partitioned -> analyze_correspondence_distances.compare | implemented but final terminal report pending in supplied snapshot |
| CE/Triplet split; detail-only exit norms or attention entropy; equal-capacity necessity controls; multiseed training | not measured/completed in this evidence packet |

**Specific findings and actions**

- **C1 (WARN)** — Eight formal endpoints and one final M0 are supported; ninth formal endpoint and terminal CPU report remain pending in the snapshot. No formal9/9 or completed paired CPU diagnosis claim. Action: Preserve pending status and existing queue dependencies; do not fill native100 metrics from M0/partial history.
- **C2 (WARN)** — Checkpoint, distance and public-CLIP hashes are reported in receipts, but those binary bytes were not independently opened or hashed locally. Receipt consistency is not remote-file availability or runtime reproduction. Action: Keep that boundary explicit; later authorized acceptance must verify physical files.
- **C3 (WARN)** — 27 repository working-copy byte hashes differ from SOURCE_SCOPE through CRLF line endings; exact archived source copies match all 27 expected hashes. Use archived bytes for exact scope attribution; do not silently describe the entire working copy as byte-identical. Action: Record both actual and expected hashes; no source change was made or required for this advisory audit.
- **E1 (WARN)** — One training seed, official-set epoch/method consumption and unequal trainable capacity limit causal/generalization claims. Mixed descriptive system deltas do not prove detail necessity, stability, SOTA or a universal failure. Action: Retain present restrained language; separate future stronger evidence from current endpoints.
- **D1 (WARN)** — Terminal paired analysis and several finer diagnostics are source-only/pending/unlogged. Do not infer execution or measured components from an available code branch. Action: Label implementation/receipt/historical/pending status explicitly.
- **E2 (minor)** — CURRENT_GOAL.md:14 still says RGBNT100 B128 awaits measurement, while lines20-22 and the supplied final M0 show B128 engineering success. Small documentation ambiguity; latest section explicitly supersedes historical observations. It does not invalidate formal scores. Action: Clarify the stale sentence during the next authorized doc update; this reviewer made no project edit.
- **L1 (PASS_with_boundary)** — Loss-floor derivation, all400 epoch values and all16323 step records are consistent with bound source and receipts. Analytical residual/upper bound is supported; direct CE/Triplet decomposition is not. Action: Retain exact-real-arithmetic and precision language and no-component/no-causal limitations.

**Unresolved limitations**

- All 43 requested primary paths were read directly; no listed primary input was missing. Large JSON manifests and receipts were fully parsed, not judged from a truncated preview.
- Only the current V6 portion of the 16092-line handoff was audited for present claims; historical lines were treated as history rather than silently revalidated.
- Original image bytes, original inventory manifest, public CLIP binary, checkpoints, distance tensors and remote runtime state were not authenticated here.
- Earlier eight M0 receipts, initialization-witness/pair-check raw receipts and batch-order JSONL bytes were not independently inspected as inputs. Their claims are documentary/queue-contract evidence; eight full-training step logs were inspected for the added loss audit.
- All314 manifest source entries were hash-resolved locally, but every transitive source file and installed library/kernel was not semantically reviewed. Hash agreement does not prove remote import execution.
- Source has the terminal nine-arm report and bootstrap, but no completed terminal result was present in this snapshot; no distance-array rescoring was performed.
- One model-family reviewer only. This is same-family/provisional advice, not cross-family acceptance or runtime reproduction.

**Local inspection failures preserved**

- An initial PowerShell foreach pipeline expression failed with ParserError/exit1; corrected by assigning the collection before ConvertTo-Json.
- Several oversized combined reads were truncated or showed encoding garbling. Relevant files/gaps were reread in bounded chunks or parsed with PowerShell/.NET; unprinted giant previews were not accepted as complete evidence.
- Two Python read/check attempts failed with 'No pyvenv.cfg file' from the E:/Scripts/python.exe shim. No Python/model code ran and no environment repair was attempted; static checks used PowerShell/.NET instead.
- Oversized protocol/hash JSON output containing a truncation warning could not be parsed. Replaced with bounded summaries from full-file parsing and complete record/hash checks.
- A one-object handoff response was briefly treated as an array, causing a TypeError. Corrected the object access and read the scoped lines.
- rg given a Windows literal wildcard path returned OS error123. Reissued with a directory and -g filename filter.
- An earlier large JavaScript report-construction cell failed to parse with SyntaxError: Unexpected token '{' before any report write. Rebuilt the reports in smaller valid cells.
- The first combined loss-script/ANALYSIS.json preview exceeded the output limit. Read the full script, parsed the complete7505-line JSON and independently checked every epoch and cited source/receipt binding.

No failed model run was performed by this reviewer. Failed inspection/report-construction attempts above did not alter project inputs or acceptance tolerances.

**Actual audited input SHA-256 values**

These 82 entries are actual local file-byte hashes, including the request, all 43 primary inputs and the additional directly inspected evidence. Reported remote binary hashes belong to the formal-result records in EXPERIMENT_AUDIT.json and are not presented as locally measured here.

| Input | Bytes | SHA-256 |
|---|---:|---|
| [P/native_research_integrity_audit/REQUEST.txt](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_integrity_audit/REQUEST.txt) | 8521 | `de862eee27a3f65e70d0222dea748c3ea83e9d208193410162474e280091350b` |
| [R/tools/run_native_research.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_research.py) | 781 | `b48725622371fc6d9671c464ce97252e2e06eba465e213ce221d84c798e52cc2` |
| [R/tools/queue_native_research.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_native_research.py) | 8654 | `b138144780a31dc8a5a4922cc4e15dbc0537d0f75ce47d823434d3d3a3c5bc65` |
| [R/tools/report_native_research.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_research.py) | 1749 | `1117d3ad9e186ec941c8d84ab4363aa28b7ce9c78d492ca7671c95338121e77a` |
| [R/tools/run_native_partitioned.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_native_partitioned.py) | 2289 | `2c3abac37b122760895b97ccf01d4fefe2cfb8223b35b6e2e335a1a4144f72fa` |
| [R/tools/queue_native_partitioned.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_native_partitioned.py) | 12948 | `e3de037c378791734d4cf5cb07d705941b39f1a98038f0cdc24556ede3837c15` |
| [R/tools/report_native_partitioned.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_partitioned.py) | 4541 | `ecf9fab59aaaaafe88aa1e598977e31f08b0ec5f42bd2ffb5e4e8abd4623b5d7` |
| [R/tools/run_independent_native_evidence.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_independent_native_evidence.py) | 10367 | `2429af8d2f803d70b9d15113a430609ead28621eb4341a746c3025cc5685fe23` |
| [R/tools/run_foundation_recipe.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_foundation_recipe.py) | 18242 | `c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758` |
| [R/tools/queue_foundation_recipe.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/queue_foundation_recipe.py) | 12850 | `3afe051358aa25d99cb3900b4c6fdb13d5d79d9dcee76d01018fccd27e27d302` |
| [R/tools/run_clean_clip_joint.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_clean_clip_joint.py) | 9804 | `178e9793f2cf8261eb44a278e053d5caabd4f2f37dba48a65952b2547338a746` |
| [R/tools/run_correspondence_roles.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_correspondence_roles.py) | 15677 | `e50865fb923297cd61cf38b33ec2bc95154f8c503b5dfe5de9823cad3f03d7ef` |
| [R/tools/train_rgbnt100_signal_oof.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_rgbnt100_signal_oof.py) | 22653 | `4462b73139e034d450d955c5b4a994aaa967548e9cc503d97bb753a14ea03b22` |
| [R/tools/train_msvr310_signal_oof.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr310_signal_oof.py) | 20480 | `c25579d931df34481fef321558f0b9a1f9179d72a25d6047dbbfd3682d4ae60f` |
| [R/tools/analyze_correspondence_distances.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_correspondence_distances.py) | 5031 | `833ebeb47cb5840422710fc00df1a974338cb04078249720d867f18b945f997a` |
| [R/refine-logs/native_research_v6/SOURCE_SCOPE.json](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/SOURCE_SCOPE.json) | 39296 | `62f2c63c5c497c0f22787576fe308708da1f4b098c4347460f43e480a2fada00` |
| [R/refine-logs/native_research_v6/EXPERIMENT_PLAN.md](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/EXPERIMENT_PLAN.md) | 4179 | `06407ddd77a59cdf92c7b5b1313ba1c97b9993a5973f2adc1dfd89fe47249a44` |
| [R/refine-logs/native_research_v6/EXPERIMENT_TRACKER.md](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/native_research_v6/EXPERIMENT_TRACKER.md) | 1972 | `8e93a98676ebc95358f2a3e039b29ea1e0db8ce0e9b8137892a09dce2efd93ff` |
| [R/refine-logs/CURRENT_GOAL.md](C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/CURRENT_GOAL.md) | 9922 | `e8298bd1e66a4a38d53b9c8d9588cb9d318f7967a48863327546d1502b3a0ea8` |
| [R/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md](C:/Users/gb/.trifusion_github_publish_22c3bee/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md) | 2286969 | `7cf158bf27f004d3bd29d4ce07449ababb156f7f578bb471c262e1112f4ebb5f` |
| [R/logs/training_feature_scale_protocols_20261002/RGBNT201.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT201.json) | 1958131 | `b2409c2d992d04adcf0c78354d306afd6af7ae9507738fd4e83de480e9a9b90f` |
| [R/logs/training_feature_scale_protocols_20261002/MSVR310.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/MSVR310.json) | 1060914 | `015a3cafef8e36ba71da2784de6835959c0549c442624c8fb9e9b37bd8c8229d` |
| [R/logs/training_feature_scale_protocols_20261002/RGBNT100.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/training_feature_scale_protocols_20261002/RGBNT100.json) | 4850644 | `b2c14adf947b06b27b9576a712f96192c1393ad6ed3ec1be5259b63f88300169` |
| [R/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_training.json) | 22397 | `77d1f8173dfd36f0f2511a395e90488998b143ab30012da941a47d13cb32b21b` |
| [R/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_official_metrics.json) | 1458 | `d9a397682dc7115cc4542385a8f000bded59d1e468d6c58a9ba9ec58a6d8a148` |
| [R/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_training.json) | 22385 | `2aa01cf1cd6bd44c3183a58aa7a6735e2f4f37f21f8381401d48d6c4553db706` |
| [R/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_official_metrics.json) | 1452 | `d98a93a754b6d3208e2ea8f9410ea32347c932641ea01dcde0417b757d6f0ef4` |
| [R/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_training.json) | 22308 | `fa597d2ebcbefdeb961da1a4bc077d82e7b34269c7d428edcdfe159d963acc9b` |
| [R/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_official_metrics.json) | 1449 | `4c741ec115510ecac7b255a0247760885ff7eff34aac78bac159b20604547dd1` |
| [R/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_training.json) | 22452 | `a87fd1cd8558c9d17c43b42446864a5fbd2c233c4456c2a7b401fba9c3e2c1d7` |
| [R/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_official_metrics.json) | 1459 | `d5c7995a71eda99cbfd4f94add0650c05cec360462db1859fb04794c10d1df73` |
| [R/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_training.json) | 22505 | `c21f0ab91e5334eda811947dad607db7e86a938a156921e83dfca707d7653a48` |
| [R/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_official_metrics.json) | 1452 | `828ded91e88c6eb25555f00ce8eabf82cee2d880fc0582c794af8ed673f9a8f4` |
| [R/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_training.json) | 22494 | `276e510c526850e90d0756c0072b22300d5e2b9480c5d69d7d83bf3037dc0a2d` |
| [R/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_official_metrics.json) | 1447 | `fab563c6ecb35d2d4e6f1ebafbc6987828debe20e210b9e493f0b6246b59cd80` |
| [R/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_training.json) | 22483 | `1f34d05bbfab945a388bbb0f411ee4b2251cd7f80c68960f0b3dbd510bad876f` |
| [R/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_official_metrics.json) | 1458 | `d8f61fc5b2add3683f523a8d8a159cc5bd3e93798a57d0160f0449fb1948d476` |
| [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_training.json) | 22457 | `360385ce6055f83b396ab450199a33c6ddb260e3c425af4db458ea8d2cc4b3bb` |
| [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_official_metrics.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_official_metrics.json) | 1451 | `d9fc9e77d7c69ff30575ba6b299251ea29d18fd72fc3e81097de4b47ed5780cf` |
| [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_m0_native_RGBNT100_training.json) | 90577 | `d5bbf31fce9b8dab3855d7705a437b21112b0b0a0b9fd6ae491d6d99ab19d181` |
| [S/comparators/Signal-cd1b0a6/utils/metrics.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/utils/metrics.py) | 22888 | `91604acb7d978462c16904910a645d7d1697c15bf150da529099004c8b5eecb2` |
| [S/comparators/Signal-cd1b0a6/layers/make_loss.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/layers/make_loss.py) | 3674 | `7eb7ab953c2bbc52d0c298ef971d0c3ff7000e5e50756fbb5a5fd63ac4c109aa` |
| [S/comparators/Signal-cd1b0a6/data/datasets/make_dataloader.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/data/datasets/make_dataloader.py) | 10233 | `54c47d798a20955bde224d9c78b42149c3c2ef008b02659141829a553194434f` |
| [S/comparators/Signal-cd1b0a6/config/defaults.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/config/defaults.py) | 6411 | `40257d349153e56e345983c99b1638d7c3d77966dfd2f5496fe03c86dc8dbd57` |
| [R/tools/official_three_dataset_data.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/official_three_dataset_data.py) | 1885 | `c3180a88d12440709319d2025928fc699ff3b7c020b11697d468453fbfc68062` |
| [R/tools/run_official_three_dataset_roles.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_official_three_dataset_roles.py) | 24829 | `f30bb00a0b71429e9d11dd196cd1d543202a133c663ad66e175d2f82e28d448d` |
| [R/tools/build_official_three_dataset_protocols.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/build_official_three_dataset_protocols.py) | 5495 | `d0a81699b31303fae1c8dadf4d397c13b308d9c7074902c67860047ff41eb151` |
| [R/tools/check_native_research_pair.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/check_native_research_pair.py) | 424 | `f368de9c455971da5d4fd50757ce576088b0db526ecd992dac2a3f0d1a43ae70` |
| [R/tools/check_partitioned_native_pair.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/check_partitioned_native_pair.py) | 4602 | `9d5a0c95f988d5d5e80139bb0f09313505eef3e0d5e387a0dafc840488cf856a` |
| [R/tools/run_training_feature_scale.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_training_feature_scale.py) | 3442 | `473853f58e4ed70bf123509afe2861fa8a9988631c04f0ce04f3260c55fd9081` |
| [R/modeling/trifusion/independent_native_roles.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/independent_native_roles.py) | 5375 | `1565590fdfb281fb51b43d06497cd526ffe5bbdc135727ebaa609615dc7832ff` |
| [R/modeling/trifusion/evidence_author_heads.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/evidence_author_heads.py) | 3287 | `613b249dae9e1d62d418b94398a0d6ccbcf48e72c88b26b27dd5b9254e9c24cd` |
| [R/modeling/trifusion/partitioned_evidence_clip.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/partitioned_evidence_clip.py) | 3614 | `3bad53abd6777602e1f1ea9e7b03f10a796116b577501ee5ec0aec6e683b436c` |
| [R/modeling/trifusion/image_native_evidence.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/image_native_evidence.py) | 2142 | `0b5a2ab7fa3f2e3dc276f651e86a83bbff162e25e65e16f321b7882f05ecbd58` |
| [R/tools/run_signal_baseline_dev.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_signal_baseline_dev.py) | 12601 | `083967f2a38267415b2992da98c2ad9429ebb793f387b45daac1dbb4eb16f365` |
| [R/tools/build_v12_complete_path_oof_targets.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/build_v12_complete_path_oof_targets.py) | 32774 | `999356e18d489dce99042f1e7873eac15b811b64255f1aee9550c52ae1c7ee06` |
| [R/modeling/trifusion/role_global_tokens.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/role_global_tokens.py) | 4968 | `f41ff1af1fb1b6303a9af63cf039263e2572c16b23813f92a0297f014e71dd4f` |
| [R/modeling/trifusion/correspondence_roles.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_roles.py) | 15524 | `e2a51154b5b327341f35e453dd32a821b091a49a669b49a7c70e79e6e2f2591b` |
| [R/tools/train_signal_preserving_v18.py](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_signal_preserving_v18.py) | 17540 | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` |
| [R/modeling/trifusion/aligned_data.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/aligned_data.py) | 10062 | `3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5` |
| [S/comparators/Signal-cd1b0a6/configs/RGBNT201/Signal.yml](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/configs/RGBNT201/Signal.yml) | 914 | `a13bd5169a4c710fa79efc87db909409015a0b2611086c0fbf4c1842f0cfcc44` |
| [S/comparators/Signal-cd1b0a6/configs/MSVR310/Signal.yml](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/configs/MSVR310/Signal.yml) | 934 | `357ca3a989e943ea5348e6206c0d0f69b3c0d35ad9fbadbf68d2786439b8a915` |
| [S/comparators/Signal-cd1b0a6/configs/RGBNT100/Signal.yml](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/configs/RGBNT100/Signal.yml) | 946 | `02641ce2941dcd50df77d379aab54650cae4a2a6b15ba1bcd43deba970b6900f` |
| [S/comparators/Signal-cd1b0a6/layers/triplet_loss.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/layers/triplet_loss.py) | 5839 | `f502c44bec642bbf72ce6d0f58c9f9a0084beef9cbb451c2d2786bb13f94960a` |
| [S/comparators/Signal-cd1b0a6/solver/make_optimizer.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/solver/make_optimizer.py) | 2021 | `ada1023a79fcaaed092bce0849a61db9d0253cf8974c494bfb3082d0fd35fee1` |
| [S/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py) | 3527 | `7b2b30b339123e92bbead5c092afa7570a06f423570c4249acfdd39294fc4a88` |
| [S/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py) | 3416 | `fe38cdb9d0905b4637f215b804e933a226aefcd9a865e9f73e564a84ef2393ac` |
| [S/comparators/Signal-cd1b0a6/data/datasets/msvr310.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/data/datasets/msvr310.py) | 3449 | `10b0197e7c97d1067bd63f25afef3cf0c4e64944ff4028c90d1f2f426a200eb9` |
| [S/modeling/trifusion/experts/mamba.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/modeling/trifusion/experts/mamba.py) | 7270 | `c516c7ad937e5eee6a4ed1e3ec33c2afe3522b751d296bd2e4910e4f27a20ee5` |
| [S/modeling/trifusion/state.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/modeling/trifusion/state.py) | 4854 | `5fe89c4885507e74ca1833cc18822a4b57e4c2d6fa9a6c6b2e127d1e74b70db9` |
| [P/native_research_completed_loss_floor/ANALYSIS.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_completed_loss_floor/ANALYSIS.json) | 286838 | `c55d13bd055a50263814663abf1fa6bb5beb2caddca642a32d2aeca04e97990d` |
| [C:/Users/gb/.codex_tmp/analyze_native_completed_loss_floor.py](C:/Users/gb/.codex_tmp/analyze_native_completed_loss_floor.py) | 7737 | `652359537e8b46bdd13659dd8b97dccc89944789723fc9cff4317d9ea44e4894` |
| [P/native_research_closed_eight_training_traces/ANALYSIS.json](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_research_closed_eight_training_traces/ANALYSIS.json) | 208030 | `d14ef6312dd69645d2bd20f00bc7d24d9dc75b4c625cb599c55a8837dbb19fd8` |
| [S/comparators/Signal-cd1b0a6/layers/softmax_loss.py](C:/Users/gb/.codex_tmp/independent_evidence_draft/native_original_control785_terminal_intake/_source/comparators/Signal-cd1b0a6/layers/softmax_loss.py) | 2048 | `92192d9c3ff4e2fd3920c6e9a9637e0caa1e66cffbbc0be8b4049e5dcdb0bf24` |
| [R/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_full800_20261003/native_research_v6_20261003_794_full_global_only_RGBNT201_training_steps.jsonl) | 13302375 | `cffbbcc702aa6155803e92540322fa78f34053830fa6591a5f4da80eeb38fbf1` |
| [R/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_semantic801_20261004/native_research_v6_20261003_794_full_semantic_RGBNT201_training_steps.jsonl) | 17712400 | `278651290972c374d07cbafdedbd3274f7fd65801f706e8c6d5813e9b0af2428` |
| [R/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_first_dataset802_20261004/native_research_v6_20261003_794_full_native_RGBNT201_training_steps.jsonl) | 18569087 | `04bc2ef410eba74f2e53b5661c1d0bbeeb6665b974902a28b074a19f64cf6d9f` |
| [R/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_global803_20261004/native_research_v6_20261003_794_full_global_only_MSVR310_training_steps.jsonl) | 2695967 | `5f389899bb07dea42f70e74b97684cba96b6b4e05c4e31d0aaa125dd419f95da` |
| [R/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_msvr_semantic804_20261004/native_research_v6_20261003_794_full_semantic_MSVR310_training_steps.jsonl) | 3565304 | `2888ef894a5439e6b6879679b841951e8bc86733daa5c66b68ff91d34ad3d6ea` |
| [R/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_second_dataset805_20261004/native_research_v6_20261003_794_full_native_MSVR310_training_steps.jsonl) | 3735060 | `d60f800a29c42a9fc2434e5ddaef8cc26b9dddddbaf5fc698d347289406a18f5` |
| [R/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_global807_20261004/native_research_v6_20261003_794_full_global_only_RGBNT100_training_steps.jsonl) | 15984571 | `275e8ef41b4b6eca9e3feafc6492e7d726b42acffb990ed2e5805e507ee1773f` |
| [R/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_training_steps.jsonl](C:/Users/gb/.trifusion_github_publish_22c3bee/logs/native_research_rgb100_semantic808_20261004/native_research_v6_20261003_794_full_semantic_RGBNT100_training_steps.jsonl) | 21119157 | `7cd97c67815d32039a55e3e360ef7b254cd240ddea7164de0beead16e7bf3ce5` |

**Exact-byte differences in the repository working copy**

For each entry below, the expected hash equals both the actual exact archived-copy hash and the diagnostic LF-normalized hash. The actual repository bytes differ and were not changed. Archived-copy root is S.

| Relative source | Actual repository SHA-256 | Expected / actual archived-copy SHA-256 |
|---|---|---|
| [R/modeling/trifusion/criterion.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/criterion.py) | `9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f` | `0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a` |
| [R/modeling/trifusion/experts/mamba.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/experts/mamba.py) | `8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83` | `c516c7ad937e5eee6a4ed1e3ec33c2afe3522b751d296bd2e4910e4f27a20ee5` |
| [R/modeling/trifusion/__init__.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/__init__.py) | `5d50f2288fdb623810dc9b80f83c17525f888ff3565745430538b262ffd02da8` | `ad617b86663e319f7702e9231fdec9c5f079f4d39ec5dabb90ced469414d910c` |
| [R/modeling/trifusion/builder.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/builder.py) | `70a94158fca9424b6ca35e4d64adbf2d4329d03d898e63a723a6ebf1ab6068e6` | `8b3620e6be3727709253589a9e59b3efd20a611ba0eec4e8f6b4d0d17eeea4eb` |
| [R/modeling/trifusion/circ_scoring.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/circ_scoring.py) | `bf73f0aac908addf4d2428c1a7aa443c6f9a279780808acbfc167145303d252c` | `db9ae624446674812618ff8c6f7e8d773a576ea31ab1ecb0d41bc3c33ca2da3e` |
| [R/modeling/trifusion/data.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/data.py) | `e66d526aa94f85c1c5dbd694019171531f3331305e757ceca0933a7c22a36c75` | `d0795f8c74e1a35d44d50b408f41c585e90fb57f9267915923f9de5a7aa850be` |
| [R/modeling/trifusion/encoder.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/encoder.py) | `a2e83b4c042979c9a88b0d8c993426840b56b58f18b9336c0923022bb9df68f8` | `80fd6adc6d80b232296412ff8647a3d7c6a66d64bda699f545e477b94e72a46b` |
| [R/modeling/trifusion/experts/__init__.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/experts/__init__.py) | `19a4e48d170b2ebb97ff239c10106333a541b8c80b7ea5b71addc9460d20a164` | `49709dfc6fd4ca781fa6d24f81476596f716163222ab01fe5d9eae105d8fc286` |
| [R/modeling/trifusion/experts/cnn.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/experts/cnn.py) | `e207c9b81171b83e5a6d09aa92123d1fd50b2e415cadd1311769bc77ca279df1` | `f823cfd59589e7fa8dbe066b26eec662d93a9d01b2376f05e7a4b6f43409996a` |
| [R/modeling/trifusion/experts/semantic_residual.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/experts/semantic_residual.py) | `c1831a3de031be033624420791cecf2e7a3945e009655089cb48ee4b9912edd6` | `c8cef9717fd7bd1e5e50b428ac92762455defac2e25857c9e0dfaf82729c2a93` |
| [R/modeling/trifusion/experts/tiny.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/experts/tiny.py) | `56b81761bb935ca5c5f24f71e57791bcf5312c7f6bcade9c355fca4d90b6699f` | `d5462c7259e5a17f7e50b87d29f44e2ecb49c4a50ccf5b59f34e8870c3e1cf90` |
| [R/modeling/trifusion/experts/transformer.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/experts/transformer.py) | `0497bf0c6f0eb8da42e0319fb2894f70db6aae49d060ddab6b21ae04df3897c2` | `8580c6e761a2fbfd54a70d0bd231fef0695cccecbec328f643a6c5112deb674e` |
| [R/modeling/trifusion/fusion.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/fusion.py) | `4910991148534cd2f3e138d022ec42c7025a233eb498cd3d6d9c4e46d8c5af8c` | `58d3672fb571e9acce8e7a8e82c8cd355fdb0f831f5439aa2d5b6e918b491a8d` |
| [R/modeling/trifusion/intervention_targets.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/intervention_targets.py) | `398704988eb15dcd5acab8b7e1094e287e5e3616651c7aff66c07e8a582dea7b` | `dcfab01a167411e98b148caeb55a9e29cd1eae453bae5a2cfe2ae9166c866f38` |
| [R/modeling/trifusion/interventions.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/interventions.py) | `08a37c74be6c189ba45a9d48a37d37f5f76440a3b33f3f9fa70c3265ad3c1b90` | `b32386352edc6f37a0ef62cf4140eb22d5e400b9eefc580e649447b10cc549e1` |
| [R/modeling/trifusion/model.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/model.py) | `af577e82d2778132ec35a06c91c984c796d6fe051543177a6c509f141f651fde` | `02e08b4ef6102431808bf850c98117e2169330fc954999086241693a22d92bf5` |
| [R/modeling/trifusion/peer_teaching.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/peer_teaching.py) | `2c15e9e74a9256980413226bdc603a8c35be0620a40cae616d1ef27f885f9014` | `d5ec38a6ebb26d8012be7fa5ebced3dcd5e16de4576e29fef2e89d92dc970b3b` |
| [R/modeling/trifusion/protocol.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/protocol.py) | `f942938909ec328ac8522c0504b6dd8a4d1355b8346e99fe44dbfc8fc450fa54` | `519ff4386addb3b83a0416bca25f75ed5a54029a4fcd949231b5eb96db4ef3bc` |
| [R/modeling/trifusion/relay.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/relay.py) | `51bc192d543eb8770068f9f5beb34d1d5724d43efbf6fe1c98cff4b9afc2ad5e` | `a24194bc81b72444dcbf3cea3450ee27f7d541787b77fd5d5cf4bdf13799d645` |
| [R/modeling/trifusion/reliability.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/reliability.py) | `469cba5e7f67cd3757df71c920a1dd103768b29ebfeddd1d713cf04752e7745c` | `eb9fbed2994ef31038bd31cd129af0b0761c5adc88902ca918d20a05ae431207` |
| [R/modeling/trifusion/semantic_tokenizer.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/semantic_tokenizer.py) | `1f9d0213ecc04c7ec75d5cb2c8485eb6f5e8f5343e62d2a953ecf081b1831a12` | `ea5e2181aaf20be10802b626a17a5dcd6de59ba88d44fa5e833827f8a5931278` |
| [R/modeling/trifusion/standalone.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/standalone.py) | `2509beda00e153b09cd9ae850aa352662b7db091e9f2989c9aa1b95eb41bfeb8` | `2017048e0a2493acd76da3946793a02707f9dd543a96cc30994b38887506d8e2` |
| [R/modeling/trifusion/state.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/state.py) | `d6491db289e68e0b97a8d532edc933252ab60fdaad963af85f603dec9002e2b9` | `5fe89c4885507e74ca1833cc18822a4b57e4c2d6fa9a6c6b2e127d1e74b70db9` |
| [R/modeling/trifusion/tokenizer.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/tokenizer.py) | `de2cfee1d6c5f274e34492f3eb55628f7522f303ec32717a8a29426fb18a005d` | `ff180350cea3ea8c5a9a7c57e4f4c9b1d7f30cc70dde5405037270be2392c306` |
| [R/modeling/trifusion/training_phases.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/training_phases.py) | `3f012311d50adead08425230a7f0befaa6a58872ae8588814cfcf1fb062b06a0` | `66a7666ad04d7e32392864227df967b86d8aeeba9a62b6ebeff280c96dd89dee` |
| [R/modeling/trifusion/variants.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/variants.py) | `61826544395a48a62f43bf5e997b4fef0ab91d4b13e664e919ff9262bd325b69` | `ca456227c5249e28461dafb67a68f7103a9b1fea292dcd2f214fb92bab1b91ef` |
| [R/modeling/trifusion/warm_start.py](C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/warm_start.py) | `62a01fa8a80cb25b6880538cf58d00f0f063545e4228edc9869f03835e064a79` | `68591452f89d6303baf81b9957c30b48135a33ce9e7a5544a40ede2b92f5a05d` |

Final local consistency check: all 82 audited input hashes were rechecked unchanged, and 154 evidence references were checked against actual line counts. The report JSON records the complete check results.

