# Experiment audit: MSVR310 fixed-source Smooth-AP coverage

Date: 2026-09-20. Overall verdict: **WARN / CLOSED_WITH_LIMITS**. Deterministic saved-array and result checks: **PASS**. Engineering: **PASS_WITH_LIMITS**. Review independence: **same-family**; acceptance: **provisional**.

The registered source diagnostic is complete and its source-ranking/candidate-coverage claims match all saved results. No scientific code, metric, denominator, masking or reported-number correction is required. WARN records the actual provenance limits below; it is not a failure of the reconstructed ranking arithmetic and does not promote the sealed Q1 result.

Requested reviewer: `gpt-6-astra`, effort `max`, fresh task (`fork_turns=none`). Actual tool-visible identity: `/root/audit_smooth_coverage_resume_20260920`. No UUID or independently verified backend identity is available. The prior auditor left no verdict or complete independent receipt. Its script was inspected and reused as an incomplete independent input, not as an accepted conclusion.

## Completed scope and receipts

- Fixed six epoch20 checkpoints, 3 folds x control/Smooth-AP x clean/one seed42 augmentation: **12 conditions**, original **8,256 source triplet-record forwards**, **0 optimizer updates**.
- Source galleries contain **672 / 683 / 709** records, with **103 / 103 / 104** identities and **390 / 393 / 417** cross-scene-eligible records. The unique dataset has **1,032 train records / 155 identities**. Each record occurs in two source folds.
- Independently checked **60 Float32 feature arrays / 41,280 array rows**, **82,560 complete-gallery query rows**, **1,996,800 candidate/full paired anchor exposures**, and **120 output/filter conditions**.
- Independently reconstructed all six **260-step memory ledgers**, current-record deduplication, last-occurrence LRU order, age<=8, capacity512, warmup65, 64 positions and eight identities x eight positions per batch. Every selected record belongs to the corresponding source fold; matched endpoints have identical record/pixel/pool ledgers.
- Recomputed all **40 aggregates**, **20 full-source endpoint pairs** and **180 CSV rows**. All **159 original text files / 13,715,377 bytes** match the saved intake manifest; all 120 current remote full-query files also match those local bytes.
- Current bytes match **99 recursive project/config bindings**, **17 pinned Signal source files**, the pinned Signal commit and diff hash, and **six whole-checkpoint hashes**. Five local-vs-remote source differences are exclusively CRLF vs LF with identical ASTs; both versions are preserved.
- R2 CPU checker `1925` (wrapper `1924`) exited **0** at `2026-09-20T10:13:44.247966+00:00`; elapsed **706.165 s**, peak RSS **400404 KiB**. It imported no Torch, initialized no CUDA, read no images and performed no model forwards or updates. Only audit scripts/logs/receipts were written to the separate remote audit directory.
- Maximum AP reconstruction error **0**; maximum direct-difference vs registered algebraic distance error **2.6645352591003757e-15**; maximum aggregate rounding difference **1.4210854715202004e-14**.

Complete evidence: `remote_artifacts/attempt02_terminal/independent_verify.jsonl`, `terminal.json`, `intake.json`; `independent_verification_summary.json`; `local_analysis_verification.json`; `provenance_verification.json`; `remote_source_binding.jsonl`. Input inventories and all SHA256 values are in `EXPERIMENT_AUDIT.json`, `local_input_inventory.json` and `dependency_inventory.json`.

## Checks A-F

### A. Ground truth provenance: PASS

All 1032 train records match the pinned dataset filename label manifest; 155 identities; source partitions match the registered label-only rule. No model outputs supply identity or scene labels.

Evidence: `snapshots/ground_truth/tools/audit_vehicle_query_protocol_labels.py:13-18`; `snapshots/ground_truth/tools/build_msvr310_train_oof_protocol.py:13-53`; `provenance_verification.json`

### B. Score normalization: PASS

Feature L2 normalization defines geometry; AP uses legal positive count, and mAP uses eligible query/exposure denominators. No score is divided by the model own maximum, minimum or mean.

Evidence: `snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:38-58`; `snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:127-131`; `snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:175-200`

### C. Result existence and arithmetic: PASS

All 60 arrays, 82560 full rows, 1996800 candidate rows, 120 condition summaries, 40 aggregates, 20 paired summaries and their 180 CSV rows verified. 159 text files match the historical intake manifest; six checkpoints match pinned whole-file hashes.

Evidence: `independent_verification_summary.json`; `local_analysis_verification.json`; `remote_source_binding.jsonl`

### D. Executed metric path: PASS

Source loader -> exact_signal_forward -> output_mapping emits five arrays; analyze invokes rank_row for every full record and candidate exposure. Persisted rows and summaries were independently reconstructed.

Evidence: `snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:85-155`; `snapshots/repo/tools/diagnose_msvr_smooth_ap_coverage.py:158-215`; `remote_artifacts/attempt02_terminal/independent_verify.jsonl`

### E. Scope assessment: PASS

Accurately scoped fixed terminal source diagnostic: one dataset, seed42, six epoch20 checkpoints, two fixed views, full source galleries. Source membership and repeated anchor weighting are disclosed; neither generalization qualification nor full objective-gradient measurement is claimed.

Evidence: `snapshots/repo/refine-logs/msvr310_smooth_ap_source_coverage_v1/DIAGNOSTIC_PLAN.md:3-13`; `snapshots/repo/results/MSVR310_SMOOTH_AP_SOURCE_COVERAGE_2026-09-09.md`; `local_analysis_verification.json`

### F. Evaluation type: PASS — real_gt, source-only diagnostic

Identity and scene labels come from dataset filename metadata, consistent with the pinned MSVR310 loader (`snapshots/signal_msvr310.py:73-97`) and the archived label manifest. The source embeddings are model outputs; they are not ground truth. This diagnostic uses the source complement of each fold, even though the reused registry is named OOF. It is not a heldout/official evaluation. The custom metric preserves the upstream same-ID AND same-scene exclusion and AP denominator (`snapshots/signal_metrics.py:65-107`), while adding separately registered all-identity/source-self and stable Float64 rules.

## Metric semantics and weighting

`rank_row` excludes the anchor's global record, never merely one duplicated batch position. Candidate pools are the sorted union of current and memory record indices, deduplicated once. In cross-scene mode, same-ID/same-scene records are removed rather than relabeled as negatives. Every different-identity gallery record remains a distractor, including single-scene identities with no valid query positive. Ineligible queries have AP/Rank1 `null`; their records remain in other queries' galleries. AP is the mean of precision at every legal positive's rank; Rank1 is the first remaining hit. `inverted_positives` counts positive positions with at least one strictly closer negative, not the number of all inversion pairs.

All ranks use the registered Float64 algebraic squared-distance matrix and a stable ascending global-record tie break. Independent direct-difference geometry was checked for every array. Across array-query rows, **0** exact adjacent non-self distance ties were observed, including **0** cross-identity adjacent ties. Direct-formula order changed in **0** array-query rows; complete-gallery AP changed in **0** output/filter/query rows, with maximum AP delta **0**. This sensitivity check does not redefine the registered ordering.

Full-source mAP weights each eligible source member once within each fold, then pools AP values without mixing embedding coordinate systems. Candidate-vs-full mAP weights the original repeated anchor positions and uses only positions eligible in both pools. Each endpoint/view has 49,920 positions: 49,872 common all-identity positions (48 unique-pool missing-positive cases) or 20,496 common cross-scene positions (168 full-positive/pool-missing cases). Those 48 cases do not mean the original K8 batches lacked training positives: training retained distinct sampled positions and random views. Cross-table differences are not candidate effects because the weighting differs.

## Reported results, fully retained

Member-weighted fused full-source mAP:

| View / positive rule | Control | Smooth-AP | Eligible source members per endpoint |
|---|---:|---:|---:|
| clean / all-identity | 99.105917 | 99.419141 | 2064 |
| clean / cross-scene | 97.185653 | 98.224923 | 1200 |
| augmented / all-identity | 98.591449 | 98.934077 | 2064 |
| augmented / cross-scene | 95.444495 | 96.672922 | 1200 |

Augmented fused candidate/full mAP on common eligible repeated positions:

| Endpoint | Positive rule | Candidate | Full source | Common exposures |
|---|---|---:|---:|---:|
| control | all-identity | 99.604749 | 99.198965 | 49872 |
| control | cross-scene | 98.150400 | 96.402936 | 20496 |
| smooth_ap | all-identity | 99.682293 | 99.318497 | 49872 |
| smooth_ap | cross-scene | 98.665998 | 97.384815 | 20496 |

All 20 source endpoint comparisons (AP gain in percentage points; no branch/view selection). Improved/worsened members require AP delta greater than +1e-12 / less than -1e-12; other members are unchanged at this tolerance:

| View | Output | Positive rule | AP gain | Improved members | Worsened members |
|---|---|---|---:|---:|---:|
| clean | baseline_only | all-identity | +0.000000 | 0 | 0 |
| clean | baseline_only | cross-scene | +0.000000 | 0 | 0 |
| clean | fused | all-identity | +0.313224 | 185 | 31 |
| clean | fused | cross-scene | +1.039270 | 175 | 28 |
| clean | cnn | all-identity | +0.490707 | 268 | 56 |
| clean | cnn | cross-scene | +1.481515 | 244 | 44 |
| clean | transformer | all-identity | +0.344154 | 221 | 45 |
| clean | transformer | cross-scene | +1.113747 | 207 | 34 |
| clean | mamba | all-identity | +0.410076 | 290 | 74 |
| clean | mamba | cross-scene | +1.239278 | 269 | 56 |
| augmented | baseline_only | all-identity | +0.000000 | 0 | 0 |
| augmented | baseline_only | cross-scene | +0.000000 | 0 | 0 |
| augmented | fused | all-identity | +0.342628 | 269 | 90 |
| augmented | fused | cross-scene | +1.228427 | 251 | 70 |
| augmented | cnn | all-identity | +0.592307 | 386 | 140 |
| augmented | cnn | cross-scene | +1.758941 | 345 | 104 |
| augmented | transformer | all-identity | +0.468687 | 310 | 102 |
| augmented | transformer | cross-scene | +1.429036 | 277 | 80 |
| augmented | mamba | all-identity | +0.494377 | 383 | 163 |
| augmented | mamba | cross-scene | +1.607211 | 348 | 118 |

The highlighted augmented cross-scene fused gain is **+1.228427 pp**, 95.444495 -> 96.672922, with Rank1 96.583333 -> 97.916667. Its 1,200 source members contain 251 improvements, 70 regressions and 879 unchanged AP values at the stated tolerance; strictly inverted positive positions decrease 1,214 -> 1,002. The original report retains these adverse rows and explicitly denies a generalization or calibration-failure conclusion. All 16 non-baseline aggregate AP gains are positive, while all four baseline comparisons are exactly unchanged; this remains one fixed source study, not independent replication.

## Actual execution chain and provenance limits

The inspected chain is coverage `contract/extract` -> Smooth-AP `context` -> role-set -> history-gradient -> fresh-coordinate -> instance-memory -> source-style configuration context -> Signal configuration. The coverage extractor actually uses the ordinary role `build_model/reload_model`, not the optional V27 source-style wrapper. Source records use `records_for(..., True)`; `source_loader` creates a sequential complete source loader; `SharedGeometryTripletTransform` applies common triplet geometry and independent erasing for the augmented view. Seed42 is reset for each endpoint/view. Signal's `train_collate_fn` preserves filenames and camera/scene metadata. `_training_batch` passes images, all-present modality masks and camera IDs to the model, not identity GT for ranking generation.

The actual inference route is `exact_signal_forward` -> `torch.func.functional_call` under `no_grad` -> V8 hierarchical frozen Signal + three role residual paths -> output mapping -> per-output feature L2 normalization. The temporary detached SIM dispatch view does not update its registered parameter. Strict reload binds fold/source IDs/config SHA/role state/frozen baseline aliases, and extraction asserts full model state SHA unchanged, absent gradients and paired pixel/baseline equality. Those historical assertions are consistent with every receipt and the inspected source; they were **not independently re-executed** in this CPU-only audit. Architecture/source snapshots are preserved, including the external Signal loader/model files; compiled extensions and GPU kernels were not independently inspected or replayed.

The original math/extract/analyze pipeline and executor verifier have complete exit0 receipts. The extractor's 8,256 forwards are original runtime cost, not new work by this reviewer. The saved arrays consume **811,605,504 bytes including NPY headers**. The original models, arrays and large candidate records remain on the server. The historical result report's gradient-preflight progress paragraph is outside this audit's subject and is not treated as a current statement about the separate running gradient task.

- This audit reads saved arrays, records, source and whole checkpoint bytes; it does not independently regenerate features from source image pixels or instantiate/reload the model.
- Historical input pixel equality, model-state invariance, absent gradients, zero optimizer updates and zero heldout/official image reads have consistent runtime receipts and code paths; no independent historical syscall/GPU telemetry exists here.
- The diagnostic pools use one fixed representation per unique record; original training used repeated sampled positions, changing augmentations/dropout and historical views. Diagnostic AP is not a reconstruction of the dynamic Smooth-AP objective.
- All metrics use registered Float64 algebraic squared distances and global-record stable tie order. This does not replace the separately registered FP32 Q1 protocol.
- Only one dataset and seed42 are covered. Source records recur in two source folds; member rows and anchor exposures are not independent subjects or new images.
- Binary model/NPY/candidate artifacts remain remote; local report, source snapshots, complete row hash receipts and full independent logs are preserved. No portable image-to-feature reproduction bundle is claimed.
- The requested reviewer model/effort is gpt-6-astra/max, but only the canonical task name is tool-visible. No backend UUID or independently verified model identity is claimed. Same-family review remains provisional.
- Earlier sealed Q1/M0 studies and the separate active objective-gradient experiment were not re-audited; their qualification or promotion status is unchanged.

## Attempts, corrections and claim impact

R1 copied the prior auditor's unfinished checker exactly (SHA `a2d580137b449ebe0f0d5d28bd3640b87080a7e538a3c08c066c0d721edddb69`). It exited1 before any array check because a protocol count was a NumPy int64 that JSON could not serialize. The complete original script, one-line stdout, traceback and terminal receipt are preserved in `remote_artifacts/failed_attempt01`. After confirming both processes had ended, R2 changed only three summary counts to Python int (SHA `00c6f19b2060f7925028c14a9fadd89a1d1147aac88fcf29333e14ffd261e491`), preserving all scientific assertions and arithmetic. R2 then completed the full requested verification once. Local runtime discovery/path-search errors are recorded in `initial_environment_errors.md`; no environment, experiment source or scientific result was changed.

No scientific correction is required. When integrating this report, replace audit-pending status pointers with **CLOSED_WITH_LIMITS / deterministic PASS / same-family provisional** and link this report, retaining the immutable pre-run plan and historical progress entries. Do not describe this as cross-family acceptance, official performance, generalization qualification, a replay of training's dynamic objective, or parameter-gradient verification. Do not reopen sealed Q1/M0 or authorize training based on this diagnostic.

No commit, push, master-document edit, model/GPU forward, new training, image read or weight deletion was performed by this audit. The existing `.aris/meta/events.jsonl` was already modified by other work and was left untouched; the review event is saved in this audit's private trace directory instead.

Final packaging note: the first trace-sealing attempt stopped on an overly broad whole-worktree status-equality assertion because `evidence/roadmap_primary_read_20260920/` appeared concurrently. The complete error and before/after observations are preserved in `trace_sealing_error.json`. No files in that directory were changed by this auditor. Packaging now records the actual shared-workspace difference; no scientific verification was rerun and no verdict changed.
