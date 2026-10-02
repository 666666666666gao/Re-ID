# Experiment integrity audit

**Overall verdict: WARN. No blocking integrity finding for the qualified six-arm report.** This is a **same-family, provisional** review, not cross-family assurance. Deterministic checks are accepted only within the source, primary-text, metadata and arithmetic scope described below.

No blocking integrity error for the narrow six-arm report; local hash/count/batch/metric arithmetic checks pass. Official selection, single seed, external binaries and cfg_yaml overrides require qualification.

Reviewer: gpt-6-astra; reasoning max; fork none; task /root/audit_training_feature_scale_complete754. Spawn provenance was confirmed by the parent. Review date: 2026-10-03. Private trace metadata remains under parent/root ownership.

## Scope and integrity of inputs

Read and hashed all **259 sealed source files** and **109 listed original primary texts**, plus the request and intake/catalog records: **372 audited input hashes**. Every JSON/JSONL and original log was read; actual evaluation/scorer/report scripts were read line by line and the active model, dataset, configuration and runtime-helper paths were traced. Unused legacy functions were not credited as executed.

Only standard-library reading, hashing, parsing, counting and arithmetic were used. No experiment module was imported; no model, GPU, evaluator/report program, SSH, binary deserialization, image rereading or bootstrap regeneration was run. No source, primary record or private trace was changed. Only the two requested audit outputs were written.

All 372 input SHA256 values are recorded in the companion [EXPERIMENT_AUDIT.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/reviewer_audit754/EXPERIMENT_AUDIT.json>), in audited_input_hashes. Its external_binary_attestations array separately records 24 remote binary receipts with locally_reread=false. Matching remote hashes establishes receipt consistency, not an independent local binary rehash.

## A–F result

| Check | Verdict |
|---|---|
| A. Ground truth provenance | PASS |
| B. Score normalization | PASS |
| C. Result existence and numerical fidelity | WARN |
| D. Dead code and invocation | PASS |
| E. Scope and causality | WARN |
| F. Evaluation classification | PASS |

### A. Ground truth provenance — PASS

Retrieval GT comes from fixed dataset identity/camera/scene records, not model outputs. All filename-derived metadata and all training label maps were independently checked. Train IDs are disjoint from query/gallery IDs.

RGBNT201 has 3951/836/836 train/query/gallery rows and 171/30/30 identities; RGBNT100 has 8675/1715/8575 rows and 50/50/50 identities; MSVR310 has 1032/591/1055 rows and 155/52/155 identities. All 3142 queries have valid positives. MSVR310 retains all 103 gallery-only identities.

Active filters remove same identity AND same camera (RGBNT) or same identity AND same scene (MSVR); different-ID distractors remain. The actual evaluator invokes the pinned Signal author scorer and independent AP/CMC scorers. F2 versus canonical protocol objects match except for permitted dataset_root.

Evidence:

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/official_three_dataset_data.py:8](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/official_three_dataset_data.py:8>): Protocol record-to-loader mapping.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_correspondence_roles.py:79](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_correspondence_roles.py:79>): Full-gallery extraction, protocol GT and both scorers.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:61>): Filename ID/camera parsing.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:63](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/data/datasets/RGBNT100.py:63>): Filename ID/camera parsing.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/data/datasets/msvr310.py:67](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/data/datasets/msvr310.py:67>): Filename ID/camera/scene parsing.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:68](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:68>): MSVR filter.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:137](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:137>): RGBNT filter.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/logs/training_feature_scale_protocols_20261002/RGBNT201.json:184](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/logs/training_feature_scale_protocols_20261002/RGBNT201.json:184>): All train/query/gallery metadata; counts at 78309.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/logs/training_feature_scale_protocols_20261002/RGBNT100.json:63](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/logs/training_feature_scale_protocols_20261002/RGBNT100.json:63>): All metadata; counts at 218978.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/logs/training_feature_scale_protocols_20261002/MSVR310.json:168](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/logs/training_feature_scale_protocols_20261002/MSVR310.json:168>): All metadata; counts at 38538.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:112](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:112>): Root-only protocol difference.

Scope limit: Metadata/source provenance is verified, not the remote image bytes or dataset publisher archive. No newly enumerated dataset file listing or SSH was allowed.

### B. Score normalization — PASS

No active metric is divided by a prediction max/min/mean/norm. AP uses GT relevant-item counts, CMC averages query hits, and mAP averages query AP; values are percentage points.

L2 feature normalization before squared Euclidean distance is representation processing. The intervention switches features delivered to both BN/CE and margin-0.3 Triplet; deployment remains L2 in both arms. CLIP attention/LayerNorm are not score normalization.

Evidence:

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py:30](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py:30>): Training/deployment boundary.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:155](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:155>): Actual CE+Triplet loss.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/modeling/trifusion/criterion.py:17](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/modeling/trifusion/criterion.py:17>): Batch-hard Triplet.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_official_three_dataset_roles.py:230](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_official_three_dataset_roles.py:230>): L2/squared Euclidean distance.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/train_msvr310_signal_oof.py:223](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/train_msvr310_signal_oof.py:223>): Independent scene AP/CMC.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/train_rgbnt100_signal_oof.py:253](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/train_rgbnt100_signal_oof.py:253>): Independent camera AP/CMC.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:93](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:93>): MSVR GT denominator.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:153](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/utils/metrics.py:153>): RGBNT GT denominator.

Scope limit: Source and text/arithmetic audit, not regenerated distance rankings.

### C. Result existence and numerical fidelity — WARN

All 259 source files and 109 primary texts match actual local SHA256; all primary byte sizes match. Actual relative-path/hash maps equal the catalog, both intakes and terminal manifest. All JSON/JSONL contents and logs were read; every epoch/step/batch record was checked.

Six initialization witnesses and three two-record forward witnesses exist. Six M0 jobs each record 8 steps, all 155 trainable tensors active, and zero reload-output difference. All M0 parent completions precede the first full launch; full jobs rebuild from public initial state, not M0 weights.

Six full jobs have 50 ordered epochs each, 20416 aligned formal step/batch records and 1306624 B64 exposures. All batches are 8 identities x 8 instances, all filename/label/camera tuples map to train metadata, and all official train filenames occur across 50 epochs. Per-epoch counts, means, logs and LR schedules agree. Loss/norm values are finite. Source logs after finite-gradient and nondecreasing AMP-scale assertions.

Later-tie mAP-best epochs recompute to normalized/raw 26/26 (201), 12/47 (100), 24/23 (MSVR). All four final metrics equal selected-epoch values exactly (maximum numeric difference 0), passing the unchanged <1e-5 percentage-point gate. This establishes recorded metric parity, not local bitwise checkpoint/distance equality.

All official receipts, matrix rows, report histories and rounded Markdown values match. All 24 launcher events agree with 12 terminal COMPLETE/exit-0 parent jobs; all 18 child subprocesses are COMPLETE/exit-0. report_invocations=1, report_exit_code=0, matching report.log and SUMMARY support one report. Historical SOURCE_INTAKE RUNNING/report0 at 02:40 is not terminal evidence.

The 24 output binary hashes agree across remote attestations and local receipts but were not locally recomputed. Public CLIP weights, image bytes and output tensors were not reread. Per-query AP/ranks, repair/harm events and bootstrap samples were not regenerated. Their aggregate arithmetic, identity-weighted deltas and bindings agree; binary numerical regeneration remains unverified.

Evidence:

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/manifest.json:6](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/manifest.json:6>): 259-source seal.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/manifest.json:267](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/manifest.json:267>): Nine initialization/forward hashes.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:704](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:704>): 24 remote binary attestations.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:2063](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:2063>): Explicit remote/no-replay limit.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/SOURCE_INTAKE.json:270](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/SOURCE_INTAKE.json:270>): Historical running status.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/campaign.json:961](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/campaign.json:961>): One report and terminal exit/time.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/report.log:1](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/report.log:1>): Report completion output.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:5](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:5>): Six arms/300 epochs/20416 steps.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:174](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:174>): Full model-state save and strict load at 185.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:189](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:189>): Fresh build, optimizer/AMP assertions and logging.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:248](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:248>): Actual best-checkpoint selection.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:263](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:263>): M0 activity and reload check.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:287](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:287>): Final strict reload and tolerance.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_foundation_recipe.py:118](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_foundation_recipe.py:118>): Child execution and exits.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:183](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:183>): M0 barrier, full phase, one report.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/report_training_feature_scale.py:23](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/report_training_feature_scale.py:23>): Report source/batch/receipt checks.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py:28](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py:28>): Array-bound diagnosis path.

Scope limit: Accepted deterministic scope is local bytes, complete text, counts and arithmetic only; semantic/remote binary assurance remains provisional. No optimizer-state counter or model replay was audited.

### D. Dead code and invocation — PASS

F2 queue installs its command/source/coordinator into the foundation queue; workers invoke run_training_feature_scale in prepare/m0/train/evaluate modes. This installs FeatureScaleFoundation plus actual batch/norm instrumentation.

Inference calls signal(training=False), encodes three images with CLIP, concatenates 1536-D globals and applies L2. Training return_aux=True passes either raw or normalized features to both BN/classifier and Triplet. Norm records approximately 1 versus 13.09-20.60 corroborate the active intervention.

The evaluator calls run_correspondence_roles.official_metrics/extract, the squared-distance helper, pinned author eval_func/eval_func_msrv, and independent camera/scene scoring. The report calls analyze_correspondence_distances.compare.

Role/adapter/teacher objects are transient construction objects removed from the final wrapper. Their forward, auxiliary losses, old OOF entry points and old Signal classifiers are not used. The fresh shared BN/classifier is active. No uncalled metric, TTT, reranking, role-module or SOTA result is credited.

Evidence:

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:123](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:123>): Actual F2 command.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:137](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py:137>): Queue installation.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py:22](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py:22>): Final model wrapper.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py:84](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py:84>): Trainer installation and main.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_clean_clip_joint.py:37](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_clean_clip_joint.py:37>): Public initialization and module disable.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_clean_clip_joint.py:62](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_clean_clip_joint.py:62>): 152 visual tensors checked against public input.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/modeling/trifusion/correspondence_roles.py:49](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/modeling/trifusion/correspondence_roles.py:49>): Old Signal parameters frozen during construction.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/modeling/make_model.py:197](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/modeling/make_model.py:197>): Actual head-bypassing Signal forward.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/modeling/clip/model.py:447](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/comparators/Signal-cd1b0a6/modeling/clip/model.py:447>): Actual CLIP forward.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_correspondence_roles.py:66](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_correspondence_roles.py:66>): Active extraction/evaluator.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/report_training_feature_scale.py:50](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/report_training_feature_scale.py:50>): Actual diagnostic call.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:654](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:654>): Normalized norm evidence.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:3286](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:3286>): Raw norm evidence.

Scope limit: Call-path/record audit, not profiling or model replay. Legacy imported functions are not assumed executed.

### E. Scope and causality — WARN

Three datasets x two fresh arms, seed42, 50 epochs. Paired initial full-state/visual/camera/head hashes, capacity, data and budget match; formal and M0 batch files are byte-identical and LR sequences agree. Actual filename/label/camera order is proved; augmented pixel tensors and cross-device kernel execution were not recorded for exact comparison.

BN/CE and Triplet inputs change together. This is a current-package feature-scale control, not Triplet-only causality, full F1 package-gap explanation, role necessity or a new-module contribution.

Official query/gallery mAP is scored each epoch and selects the best checkpoint (later ties). This is benchmark-consumed development evidence, not untouched holdout generalization. Equal selection policy does not eliminate selection bias.

The fixed raw-minus-normalized gate (mAP >=0.5 pp and Rank1 >=0) recomputes as FAIL for RGBNT201, FAIL for RGBNT100 and PASS for MSVR310. Both negative datasets remain; neither rounding nor identity-macro substitution is used.

Bootstrap resamples fixed-model identity mean AP changes, seed42, 2000 replicates. It is an identity-macro interval, not a query-weighted official-mAP or training-seed interval. All three intervals include zero; MSVR gate passage is not significance, seed stability or universal benefit.

Per-run training+epoch-evaluation elapsed time sums to 21994.386657 seconds (6.109551849 hours); parent first-M0-to-report wall time is 3.754712843 hours, excluding preparation. Maximum recorded concurrency is four on GPUs0-3; memory matches receipts. These are timing boundaries, not inference throughput or controlled speed comparisons.

cfg_yaml retains legacy Adam and BASE_LR/warmup/loss fields. Active non-author code uses AdamW, visual5e-6/other3.5e-4, weight_decay1e-4, five-epoch warmup, fixed50 cosine, unit CE+Triplet weighting and margin0.3. Source and LR records establish actual behavior; cfg_yaml alone is not an effective recipe. This ambiguity affects both arms equally.

Evidence:

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:7](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:7>): Joint BN/CE+Triplet scope.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:11](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:11>): Fixed recipe/budget/tolerance.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:23](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:23>): Unchanged gates and claim ceiling.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:25](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md:25>): Official selection and seed limitation.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:148](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:148>): Actual AdamW groups.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_signal_preserving_v5.py:1621](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_signal_preserving_v5.py:1621>): Actual LR formula.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:248](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:248>): Official epoch selection.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py:57](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py:57>): Identity-macro/bootstrap formula.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:3963](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:3963>): 201 gate false; interval at 4011.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4174](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4174>): 100 gate false; interval at 4222.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4485](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4485>): MSVR gate true; interval at 4533.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training.json:643](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training.json:643>): Legacy cfg_yaml fields.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4803](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4803>): Narrow claim statement.

Scope limit: No multi-seed, untouched-test, exact augmented-pixel or speed claim accepted.

### F. Evaluation classification — PASS

Official retrieval is real_gt. M0 supervised finite-loss checks use real labels but establish engineering activity, not retrieval efficacy.

Initial-forward/reload equality uses model outputs as numerical consistency references. Under the requested taxonomy this is synthetic_proxy solely for the reference-output comparison, not for the real images or official retrieval GT. These probes are explicitly not dataset accuracy claims.

Hash/count/arithmetic checks are engineering verification. No human-eval or simulation-only retrieval result is claimed.

Evidence:

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_correspondence_roles.py:88](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_correspondence_roles.py:88>): Real protocol GT.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/check_training_feature_scale_pair.py:28](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/check_training_feature_scale_pair.py:28>): Two real training records; output equality only.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/initial_forward_pair_RGBNT201.json:30](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/initial_forward_pair_RGBNT201.json:30>): No optimizer or official scoring.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:263](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:263>): Engineering M0/reload.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_m0_normalized_RGBNT201/training.json:51](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_m0_normalized_RGBNT201/training.json:51>): Recorded activity/reload.

Scope limit: Engineering consistency is not performance evidence.

## Primary numerical results

Metrics are percentages from original strict-reload official receipts. All six runs have 50 full epochs, seed 42. Original full-precision values are preserved in the JSON audit; the following table is presentation rounding only.

| Dataset | Training features | Selected epoch | mAP | Rank-1 | Rank-5 | Rank-10 | Steps |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | normalized | 26 | 62.580552 | 62.440193 | 75.837320 | 83.014351 | 2649 |
| RGBNT100 | normalized | 12 | 77.584292 | 94.169098 | 95.335275 | 95.860058 | 6559 |
| MSVR310 | normalized | 24 | 50.356740 | 67.512691 | 81.387478 | 86.294419 | 1000 |
| RGBNT201 | raw | 26 | 59.687899 | 60.885167 | 73.205739 | 80.622011 | 2649 |
| RGBNT100 | raw | 47 | 75.192936 | 94.402331 | 95.918369 | 96.384841 | 6559 |
| MSVR310 | raw | 23 | 51.577616 | 70.558375 | 85.617596 | 89.170897 | 1000 |

Raw minus normalized gates were recomputed from full-precision official metrics, with the unchanged threshold of mAP gain >=0.5 percentage points AND Rank-1 gain >=0. No rounding or identity-macro substitution is used for these decisions.

| Dataset | Delta mAP (pp) | Delta Rank-1 (pp) | Registered gate | Recorded identity-macro 95% interval |
|---|---:|---:|---|---|
| RGBNT201 | -2.892652488 | -1.555025578 | FAIL | [-8.480107771, 3.181634262] |
| RGBNT100 | -2.391355350 | +0.233232975 | FAIL | [-5.381174074, 0.932553599] |
| MSVR310 | +1.220876029 | +3.045684099 | PASS | [-1.197075636, 4.365897958] |

Only MSVR310 passes the registered gate. Both negative datasets are retained. Every identity-macro interval includes zero. These fixed-model, 2,000-replicate identity bootstrap intervals do not measure training-seed variation; the original bootstrap samples and distance rankings were not regenerated locally.

### Primary receipt evidence by run

**RGBNT201 / normalized**

Nonzero Triplet steps: 1882; recorded feature-norm range: [0.9999998211860657, 1.0000001192092896]; elapsed training plus epoch evaluation: 1896.074327 s; peak allocated memory: 11704324096 bytes; trainable parameters/tensors: 86407680/155. Selected-epoch versus strict-reload metric maximum difference: 0.

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/official_metrics.json:20](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/official_metrics.json:20>): Original metrics.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training.json:41](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training.json:41>): All fifty epoch rows.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training_steps.jsonl:2649](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training_steps.jsonl:2649>): Last step; all rows checked.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training_batch_order.jsonl:2649](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training_batch_order.jsonl:2649>): Last batch; all rows checked.

**RGBNT100 / normalized**

Nonzero Triplet steps: 842; recorded feature-norm range: [0.9999998211860657, 1.0000001192092896]; elapsed training plus epoch evaluation: 5871.136725 s; peak allocated memory: 11702802432 bytes; trainable parameters/tensors: 86224896/155. Selected-epoch versus strict-reload metric maximum difference: 0.

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/official_metrics.json:20](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/official_metrics.json:20>): Original metrics.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/training.json:41](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/training.json:41>): All fifty epoch rows.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/training_steps.jsonl:6559](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/training_steps.jsonl:6559>): Last step; all rows checked.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/training_batch_order.jsonl:6559](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/training_batch_order.jsonl:6559>): Last batch; all rows checked.

**MSVR310 / normalized**

Nonzero Triplet steps: 992; recorded feature-norm range: [0.9999998211860657, 1.0000001192092896]; elapsed training plus epoch evaluation: 1304.533802 s; peak allocated memory: 11702207488 bytes; trainable parameters/tensors: 86386176/155. Selected-epoch versus strict-reload metric maximum difference: 0.

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/official_metrics.json:20](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/official_metrics.json:20>): Original metrics.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/training.json:41](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/training.json:41>): All fifty epoch rows.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/training_steps.jsonl:1000](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/training_steps.jsonl:1000>): Last step; all rows checked.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/training_batch_order.jsonl:1000](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_MSVR310/training_batch_order.jsonl:1000>): Last batch; all rows checked.

**RGBNT201 / raw**

Nonzero Triplet steps: 1122; recorded feature-norm range: [14.621070861816406, 20.60223960876465]; elapsed training plus epoch evaluation: 2461.064590 s; peak allocated memory: 11705700352 bytes; trainable parameters/tensors: 86407680/155. Selected-epoch versus strict-reload metric maximum difference: 0.

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/official_metrics.json:20](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/official_metrics.json:20>): Original metrics.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/training.json:41](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/training.json:41>): All fifty epoch rows.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/training_steps.jsonl:2649](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/training_steps.jsonl:2649>): Last step; all rows checked.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/training_batch_order.jsonl:2649](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT201/training_batch_order.jsonl:2649>): Last batch; all rows checked.

**RGBNT100 / raw**

Nonzero Triplet steps: 704; recorded feature-norm range: [13.090947151184082, 20.42727279663086]; elapsed training plus epoch evaluation: 9122.063616 s; peak allocated memory: 11702802432 bytes; trainable parameters/tensors: 86224896/155. Selected-epoch versus strict-reload metric maximum difference: 0.

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/official_metrics.json:20](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/official_metrics.json:20>): Original metrics.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/training.json:41](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/training.json:41>): All fifty epoch rows.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/training_steps.jsonl:6559](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/training_steps.jsonl:6559>): Last step; all rows checked.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/training_batch_order.jsonl:6559](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/training_batch_order.jsonl:6559>): Last batch; all rows checked.

**MSVR310 / raw**

Nonzero Triplet steps: 938; recorded feature-norm range: [17.30005645751953, 20.57608413696289]; elapsed training plus epoch evaluation: 1339.513597 s; peak allocated memory: 11702207488 bytes; trainable parameters/tensors: 86386176/155. Selected-epoch versus strict-reload metric maximum difference: 0.

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/official_metrics.json:20](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/official_metrics.json:20>): Original metrics.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/training.json:41](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/training.json:41>): All fifty epoch rows.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/training_steps.jsonl:1000](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/training_steps.jsonl:1000>): Last step; all rows checked.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/training_batch_order.jsonl:1000](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_raw_MSVR310/training_batch_order.jsonl:1000>): Last batch; all rows checked.

## Blocking and nonblocking findings

Blocking findings: **none for the qualified result scope**. This does not approve broader claims that these controls do not test.

**W1 — Official benchmark selects checkpoints.** (nonblocking WARN)

Claim impact: Post-selection development evidence, not untouched generalization/SOTA.

Minimal correction: Keep official-test mAP-best selection explicit beside results.

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:248](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:248>): Official epoch selection.

**W2 — One seed and all identity-bootstrap intervals cross zero.** (nonblocking WARN)

Claim impact: MSVR gate passage is not significance/stability; two datasets fail.

Minimal correction: Retain all datasets, seed42 and identity-macro interval meaning.

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py:57](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py:57>): Identity-macro/bootstrap formula.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:3963](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:3963>): 201 gate false; interval at 4011.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4174](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4174>): 100 gate false; interval at 4222.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4485](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json:4485>): MSVR gate true; interval at 4533.

**W3 — Binary/image content remains externally attested.** (nonblocking WARN)

Claim impact: No independent local tensor/ranking/image regeneration.

Minimal correction: Call this a source/primary-text audit with remote binary hashes. Stronger binary validation requires actual separately authorized binary reading.

- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:704](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:704>): 24 remote binary attestations.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:2063](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json:2063>): Explicit remote/no-replay limit.

**W4 — cfg_yaml retains overridden optimizer settings.** (nonblocking WARN)

Claim impact: Reproduction from cfg_yaml alone misstates effective optimization.

Minimal correction: Document effective settings from source/LR logs; preserve original receipts.

- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:148](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py:148>): Actual AdamW groups.
- [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_signal_preserving_v5.py:1621](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_signal_preserving_v5.py:1621>): Actual LR formula.
- [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training.json:643](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT201/training.json:643>): Legacy cfg_yaml fields.

## Claim assessment

| Claim | Assessment | Reason |
|---|---|---|
| Six fresh matched feature-scale controls completed. | supported_with_qualifiers | Source/text/receipt evidence; binary content not locally replayed. |
| MSVR310 raw arm passes the registered gate. | supported_with_qualifiers | One seed, official-selected checkpoint; no significance/stability established. |
| Raw improves all datasets. | unsupported | 201 and 100 mAP decline and fail gates. |
| Triplet-only causality or whole F1 gap explained. | unsupported | BN/CE and Triplet inputs change jointly; other F1 variables not isolated. |
| New module/role contribution, SOTA, seed robustness or untouched-test generalization. | unsupported | No such evidence in this experiment. |
| Strict reload means bitwise equality of all checkpoint/distance tensors. | unsupported | Strict loading is in source and metrics exactly match, but no local binary comparison. |

## Evaluation types

| Evidence | Classification | Claim ceiling |
|---|---|---|
| official retrieval | real_gt | Benchmark retrieval, subject to official selection and single-seed limits |
| M0 supervised loss checks | real_gt | engineering only |
| initial/reload reference-output comparison | synthetic_proxy | numerical consistency only; real inputs, not benchmark GT |

Initial/reload references are model-output consistency targets on real images. Calling those comparisons synthetic_proxy does not make the benchmark labels synthetic or establish retrieval efficacy.

## Accepted deterministic verification

- acceptance_status: "accepted_only_for_exact_local_scope".
- source_hashes_match: 259.
- primary_hashes_and_sizes_match: 109.
- source_maps_equal: true.
- formal_epochs: 300.
- formal_steps: 20416.
- formal_sample_exposures: 1306624.
- m0_steps: 48.
- all_m0_before_full: true.
- matched_batch_bytes_all_pairs: true.
- all_batches_B64_K8_and_GT_metadata_consistent: true.
- all_train_filenames_seen_across_50_epochs: true.
- selected_epochs_recomputed: true.
- strict_reload_recorded_metric_max_abs_delta_pp: 0.0.
- lr_max_abs_recalculation_delta: 0.0.
- all_log_histories_match: true.
- parent_jobs_complete_exit0: 12.
- child_subprocesses_complete_exit0: 18.
- launcher_events: 24.
- report_invocations: 1.
- report_exit_code: 0.
- peak_recorded_concurrency: 4.
- gpu_indices: [0, 1, 2, 3].
- summed_training_and_epoch_evaluation_seconds: 21994.386657.
- binary_hash_cross_record_matches: 24.
- binary_hashes_recomputed_locally: 0.

Audit harness note: Initial exact LR assertion used a different multiplication/division operation order; exact source order yields max difference 0. No experiment record/source/gate was changed.

No checkpoint optimizer-state counter or binary distance equality was independently inspected. Source control flow and records support actual stepping and strict metric parity; binary replay remains outside the authorized audit.

## Minimal corrections

- No code change, threshold change or retraining needed for honest reporting.
- Preserve negative datasets, official selection, seed42, joint CE/BN+Triplet scope and bootstrap limits.
- State external binary boundary and effective optimizer source; preserve primary records.

## Input hash anchors

The complete 372-file hash ledger is in the companion JSON. These anchors bind the request, catalogs, plan, central runtime and terminal reporting records:

| Input | SHA256 |
|---|---|
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/AUDIT_INPUT_PATHS.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/AUDIT_INPUT_PATHS.json>) | sha256:144cdfb2d9883aa4cdf37043b489359ee5ae67b387c85470f92f4159b773563a |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/AUDIT_REQUEST.txt](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/AUDIT_REQUEST.txt>) | sha256:aa88a1e5b60961fe9a789a54d8ed7b23b34dcce44102a00c4e8cada3557fe257 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/INTAKE.json>) | sha256:bd895673325553e71f41f25c54f6c53f0f87b7b47ec6e2876710a27406e1c137 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/accepted_matrix.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/accepted_matrix.json>) | sha256:f1df5ecc85daa346229dfc94f90ca20039558a80640f4c7dc183dd5e556dcc7a |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/campaign.json>) | sha256:90306d936d93a6d5221bace09476e69837cd180f77ed8fb0d7b10b19e72c9197 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/manifest.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/manifest.json>) | sha256:e7e6df04d8632147d1fb16398566474665e0b094d6431ea36d96a34621f17d18 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_normalized_MSVR310/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_normalized_MSVR310/campaign.json>) | sha256:7fab75434dc1c1f8d245656cecf34ed6c3a879902cc7f87797c8643b697b92b9 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_normalized_RGBNT100/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_normalized_RGBNT100/campaign.json>) | sha256:8218f7b7f88de7f9e9480ae3ac414884a5ac2aa018cab1205f474a2d82ad2305 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_normalized_RGBNT201/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_normalized_RGBNT201/campaign.json>) | sha256:701e6d3caa698e7320d39d6a204429890ec35ea23e4fe689a9c8818ace71a0c0 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_raw_MSVR310/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_raw_MSVR310/campaign.json>) | sha256:f16570ab48125fc17360d0bfa896193ada03da54b1ecab032eabee1cfb21728e |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_raw_RGBNT100/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_raw_RGBNT100/campaign.json>) | sha256:1ada8903f4dcad765672a1d97d6fd4dfb7d000362e9b680c5eac0f149c5eb685 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_raw_RGBNT201/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_full_raw_RGBNT201/campaign.json>) | sha256:5ba7d73c8145620cbe806a7c7675475a62efdf2d3e37da5659e85fa26ff7c502 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_normalized_MSVR310/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_normalized_MSVR310/campaign.json>) | sha256:109f8aabd72f70378b6c23c414f36c9f0509e0a959e706481f35c1f3072dde5f |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_normalized_RGBNT100/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_normalized_RGBNT100/campaign.json>) | sha256:88ce315ecf5e1904e7941b36e29ff5569351d1eb6ae7e8b53ce746d0c4c7569f |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_normalized_RGBNT201/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_normalized_RGBNT201/campaign.json>) | sha256:0abb562b01880207fafc85810e8940fc1a17c7259bb61e2e39296b86cd3dc457 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_raw_MSVR310/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_raw_MSVR310/campaign.json>) | sha256:90074df6f5d8e3fc5bd8524568fbe31c6c5a345c2c7b555e1271b7436b8b1d5d |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_raw_RGBNT100/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_raw_RGBNT100/campaign.json>) | sha256:86e8883ff7be0892bbd07062fde684b3287c738f91bfd5b362bf62c8d1757141 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_raw_RGBNT201/campaign.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/logs/training_feature_scale_20261003_v2/training_feature_scale_20261003_v2_m0_raw_RGBNT201/campaign.json>) | sha256:a878d0018bd62f58e4972b39159bfdc83a4ca87fe20034b1f23a372065094563 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/REPORT.md](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/REPORT.md>) | sha256:32e26f91e1d0beaab19f0968cb912cbd86d60593764438f5c3d2110831ff3f33 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json](<C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json>) | sha256:202addf168eb43b6d1942e29a646a2b14b418ab54dba5c3f97405709ac544b17 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/SOURCE_INTAKE.json](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/SOURCE_INTAKE.json>) | sha256:e446b628da46959c939ab5e7de9a30801b6dd6bfd5f053595d439fbb082f8ee0 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/clean_clip_joint_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/clean_clip_joint_v1/EXPERIMENT_PLAN.md>) | sha256:20acb704e0f7dd34f77fb611e57ef641da0fbce24401ebe9cf26c60a9d1d0d7c |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/foundation_recipe_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/foundation_recipe_v1/EXPERIMENT_PLAN.md>) | sha256:9ba24f07c014b0238cb690b55f3883094f5c5953adeae18f32ef74a2dfffed87 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/native_detail_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/native_detail_v1/EXPERIMENT_PLAN.md>) | sha256:370cb36478d03847648a2ba681a1b1dab2a9594cd7f3c4cc341a0b4b98562aa9 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/role_global_tokens_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/role_global_tokens_v1/EXPERIMENT_PLAN.md>) | sha256:d8f2dfc64268267734f6aaff6ce9d53d3533a8517a903c6a52e40d9b352d2776 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/semantic_native_evidence_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/semantic_native_evidence_v1/EXPERIMENT_PLAN.md>) | sha256:adec43cee2f3d0f84bddcd1aca25cf649832a970534ae40dfc5fbffd92177bde |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/shared_private_evidence_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/shared_private_evidence_v1/EXPERIMENT_PLAN.md>) | sha256:a152b8e224504eb4b3e2b90e563b6ff7a1a1da896f60ddc85a808044f3a58a7f |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/slot_competition_roles_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/slot_competition_roles_v1/EXPERIMENT_PLAN.md>) | sha256:3d6906fa25340d64d8d98cf4a5e1e40ccef9e1d9576a993c522d08c1c81af37b |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md>) | sha256:04e3f88fe9e1e11351a34e9e6aab375ae0e3946653f3abba1e9f2c33903900f0 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/visual_update_control_v1/EXPERIMENT_PLAN.md](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/refine-logs/visual_update_control_v1/EXPERIMENT_PLAN.md>) | sha256:a8df22383478a59fd68c1c7f57719120896f2c5327844ce264f52f4cb330a984 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/analyze_correspondence_distances.py>) | sha256:833ebeb47cb5840422710fc00df1a974338cb04078249720d867f18b945f997a |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/queue_training_feature_scale.py>) | sha256:43de0eae7b9660de12d8729fdee06efa82f6adfcaf74599c6fda5faf00e8ae16 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/report_training_feature_scale.py](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/report_training_feature_scale.py>) | sha256:977281df7c340d5b9b9d2b6aa36d33b018810bb7cdf935c2186a68a1215d413c |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_foundation_recipe.py>) | sha256:c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758 |
| [C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py](<C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source/tools/run_training_feature_scale.py>) | sha256:473853f58e4ed70bf123509afe2861fa8a9988631c04f0ce04f3260c55fd9081 |

