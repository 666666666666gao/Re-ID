# Ground-truth and frozen-source audit 669

**Overall verdict: WARN.** The inspected retrieval route uses dataset identity labels and standard mAP/CMC equations. All 27,266 protocol rows and all 3,142 query-filter records agree with the supplied directory catalogue. All 205 launch-manifest entries match their raw-byte hashes. This is a bounded source/protocol finding; it does not certify image authenticity, completed endpoints, or training stability.

Generated: 2026-09-29T23:55:35+08:00. Reviewer requested: **gpt-6-astra**, reasoning **max**, native Codex agent. Actual backend/model/effort attestation is **unavailable in the exposed runtime**. Review independence: **same-family**. Acceptance: **provisional**. No cross-family acceptance is claimed. Parent `/root` owns the full native request/response trace; its stored trace path was not exposed to this reviewer.

## Scope and evidence notation

The request is `C:/Users/gb/.codex_tmp/gt_source_audit_request_669.txt`. Only the two requested report files were written. No SSH, GPU work, training, inference, dependency installation, source/old-receipt edit, or project-module execution was performed. Local standard-library Python and PowerShell were used for catalogue arithmetic, AST inspection, raw SHA-256 and ZIP comparisons.

References below resolve as follows:

- **S/** = `C:/Users/gb/.codex_tmp/cross_depth_source_evidence_668_20260929/`
- **A/** = `C:/Users/gb/.codex_tmp/cross_depth_additional_sources_670_20260929/`
- **P/** = `C:/Users/gb/.trifusion_github_publish_22c3bee/`
- **C/** = `C:/Users/gb/.codex_tmp/`

All 27 initially listed primary artifacts were read directly, followed by relevant dependencies and the three supplemental modules supplied during review. The companion JSON records 219 input hashes, exhaustive catalogue results, source references and assurance metadata. Hash verification of a file does not imply that every dormant function in the 205-file envelope was semantically audited.

## Check results

| Check | Status | Finding |
|---|---|---|
| A. Ground-truth provenance | WARN | All label/split/catalogue checks pass; image bytes and authentic distribution provenance remain unproved. |
| B. Metric normalization | PASS | Active AP/CMC formulas use dataset positives and query counts; reported fractions are multiplied by 100. Original raw bytes match the manifest. |
| C. Results and claims | WARN — deferred | Endpoint existence, numerical claims, full50 completion and stability are outside this call. |
| D. Active/dead code | WARN | Active patched scorer route and depth binding are established; three imported modules were not launch-bound, and the recurrence control also changes depth aggregation. |
| E. Scope | WARN | Three-by-three seed42 registration and official-best selection are source facts, not completed-panel or unbiased generalization evidence. |
| F. Evaluation type | PASS | Active retrieval is `real_gt` at source/protocol level; engineering/model-target proxies are distinguishable and do not provide retrieval GT. |

## A. Ground-truth provenance

The directory collector reads the protocol only to choose dataset roots and physical splits, checks protocol SHA against the launch manifest, then records directory names and `[path, size, mtime_ns]` from a recursive listing. It does not run a model or derive labels from predictions (`C/collect_cross_depth_ground_truth_paths_669.py:8-30`). Its raw catalogue SHA is `72e68631b6d1a84e8374f1cd6b081d8e519be46d5b4804e6652406ff5927819c`, matching the summary receipt.

Every eligible primary filename was independently parsed, every protocol field compared, every companion modality path checked, and every query's retained-positive/exclusion count recomputed.

| Dataset | Train rows / IDs | Query rows / IDs | Gallery rows / IDs | Physical file-stat rows | Retained positives per query |
|---|---:|---:|---:|---:|---:|
| RGBNT201 | 3,951 / 171 | 836 / 30 | 836 / 30 | 14,363 | 7–21 |
| RGBNT100 | 8,675 / 50 | 1,715 / 50 | 8,575 / 50 | 18,965 | 50–175 |
| MSVR310 | 1,032 / 155 | 591 / 52 | 1,055 / 155 | 8,034 | 1–31 |

There are zero missing/extra/duplicate protocol primary rows, zero field mismatches, zero missing referenced files, and zero nonpositive referenced file sizes. Train IDs have zero overlap with query/gallery IDs. All train maps are contiguous sorted-identity bijections; all evaluation labels retain original identity numbers. The only unused physical files are RGBNT201's two `.DS_Store` files.

The original parsers select RGBNT201 `train_171/test/test`, RGBNT100 `rgbir/bounding_box_train/query/bounding_box_test`, and MSVR310 `bounding_box_train/query3/bounding_box_test`. Filename parsing exactly explains identity, camera and scene/view fields (`S/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:26-34,61-85`; `RGBNT100.py:24-37,63-84`; `msvr310.py:33-40,67-87`). Protocol evidence begins at `S/logs/official_three_dataset_protocols_20260923/RGBNT201.json:184`, `RGBNT100.json:63`, and `MSVR310.json:168`; counts are at lines 78309, 218978 and 38538 respectively.

RGBNT100's sorted class numbering is not the author's unspecified Python-set enumeration order. A local CPython 3.11 reconstruction from the sorted catalogue gives 50 permuted labels; for example identity 501 maps to protocol label 0 versus set-enumeration label 44. This is not an observed target error: the active loader consistently uses the protocol's train labels, a new classifier is initialized, and the frozen Signal classifier is bypassed by direct visual-encoder calls (`S/tools/official_three_dataset_data.py:8-18`; `S/modeling/trifusion/correspondence_roles.py:58-90,247-250`). This reproduction does not recover historical filesystem enumeration order or certify the baseline's training history.

For RGBNT201/100 the exclusion is **same identity AND same camera**, retaining different-identity same-camera distractors. MSVR310 uses **same identity AND same sNNN scene field**, with no extra camera/neighbour-view filter (`S/comparators/Signal-cd1b0a6/utils/metrics.py:55-69,135-138`). The plan calls MSVR's scene field a time-period label; this audit establishes the filename-field/filter correspondence, not chronological authenticity. All 3,142 query-filter records match. Every query filename is present in its gallery and its matching record is excluded by this rule. RGBNT201's query/gallery equality therefore does not create a retained self-match. MSVR's full gallery includes all 155 test identities despite only 52 query identities.

**Boundary:** this is a supplied post-collection directory catalogue, not direct image access. Names, sizes and mtimes cannot establish image hashes, correct identities in pixels, montage size/modality layout, an authentic dataset release, absence of content duplicates, or the bytes consumed during execution. The collector only enumerates the protocol-selected physical splits (`C/collect_cross_depth_ground_truth_paths_669.py:15-30`). These limitations keep A at WARN.

## B. Metric equations and raw-byte parity

Training's active fused evaluator extracts complete protocol records and calls the vendored author's `eval_func` or `eval_func_msrv`, plus separate `camera_scores`/`scene_scores` implementations (`S/tools/run_correspondence_roles.py:66-108`). The active final evaluator repeats this for fused, shared-global and joint-local outputs and requires scalar agreement within 1e-5 (`S/tools/run_correspondence_context_identity.py:175-240`).

After permitted filtering, AP is the mean of precision at each true-positive rank. mAP is the mean of query AP. CMC averages whether a true match has appeared by each rank. Denominators are retained ground-truth positives, ranks, and valid-query count (`S/comparators/Signal-cd1b0a6/utils/metrics.py:82-108,142-170`; `S/tools/train_rgbnt100_signal_oof.py:253-268`; `S/tools/train_msvr310_signal_oof.py:223-238`). Every catalogue query has positives, so no query needs the author's no-positive skip.

L2 feature normalization precedes squared Euclidean distances (`S/tools/run_official_three_dataset_roles.py:230-238`). Context attention softmax and feature normalization are representation operations. Reported mAP/Rank-1/5/10 simply multiply the author's fractions by 100 (`S/tools/run_correspondence_roles.py:100-102`; `S/tools/run_correspondence_context_identity.py:222-224`). No active reported metric divides by prediction maxima, minima or means. The author's `max_rank=50` limits the CMC vector/rank-list text, not AP's full-gallery positive denominator.

Raw SHA-256 checks, without newline normalization, passed for **205/205** manifest entries, the additional analyzer, manifest, archive and catalogue. All 205 ZIP entries also equal their local extracted bytes and manifest digests; the ZIP's only two extra entries are the manifest and analyzer.

| Artifact | SHA-256 |
|---|---|
| Launch manifest | `67e383e88f5151d12cbf3eae38111df42f7e1279ba7118b2a6f2cc4ca6044b14` |
| Frozen source ZIP | `08883eb638a08796ab07adcc4a86256743fdfea0891d9b12ce620a145c288d08` |
| Additional distance analyzer | `833ebeb47cb5840422710fc00df1a974338cb04078249720d867f18b945f997a` |
| Supplemental current-source ZIP | `097af86de8dbd9fa86b2a67ac674f26c6500d400e178785c6611943907aa2dd6` |

This PASS concerns inspected equations/calls and byte parity. No actual distance array, feature, checkpoint or reported performance value was replayed.

## C. Result existence and numerical claims — deferred

No endpoint receipts, training logs, checkpoint tensors or saved distances were inspected. Therefore this report does not certify completed endpoints, best-epoch values, improvement, reload success, production M0 success, full50 execution, or stability. The launch manifest's nine PENDING registration rows are not a current execution snapshot (`S/evidence_manifest.json:8-62`). The older numerical comparison in the experiment plan is outside this audit. C is WARN solely as an explicit deferred assessment, not a finding of phantom results.

## D. Actual call route, dormant code and depth controls

The queue launches `run_cross_depth_role_state.py` with seed42, 50 epochs, M1/M2 on, M3 off, context queries and no auxiliary target (`S/tools/queue_cross_depth_role_state.py:51-60`). Its main patches the context entry's model/build/evaluate/save/load globals; the context main then patches the base runner globals before dispatch (`S/tools/run_cross_depth_role_state.py:67-77`; `S/tools/run_correspondence_context_identity.py:244-254`; `S/tools/run_correspondence_roles.py:271-295`).

The active build strictly loads the dataset's frozen plain baseline, discards the temporary old V8 wrapper, then creates the cross-depth model (`S/tools/run_correspondence_roles.py:32-57`; `S/tools/official_three_dataset_model.py:41-82`). The active training loss is fused identity CE plus triplet, with disabled auxiliary loss exactly zero; the old runner's M3 prediction-loss training body is replaced (`S/tools/run_correspondence_context_identity.py:105-142`). `prediction_weight=0.1` and the auxiliary coefficient 1.0 are dormant metadata here, not active extra objectives.

The production model injects `production_mamba_factory`, which creates `mamba_ssm.Mamba`; the nearby CPU `TinySequenceMixer` is not selected (`S/tools/run_correspondence_roles.py:29,41-45`; `S/modeling/trifusion/experts/mamba.py:14-32`). The three snapshots are zero-indexed CLIP blocks 3/7/11, i.e. blocks 4/8/12. M1 adapters can change the adapted global output while Signal tensor parameters remain frozen (`S/modeling/trifusion/correspondence_roles.py:27-51,58-90`).

| Mode | Actual change |
|---|---|
| mixed_once | Uniformly mix the layer-normalized snapshots before the nonlinear role adapter, then execute one role step. |
| depth_mean | Process each depth independently with the shared role operators, then average the three role outputs. |
| depth_recurrent | Add each previous role output into the next step's corresponding input, then use only the final depth's output. |

Depth logits are frozen, and the address rule is computed once from the mixed Transformer-role CLS (`S/modeling/trifusion/cross_depth_role_state.py:8-18,49-71`). The CNN prior state is added after current-depth spatial convolution/sampling, before output normalization; it is not a recurrent convolutional grid. Transformer/Mamba prior states enter current sequences before their processing (`S/modeling/trifusion/cross_depth_role_state.py:20-47`). Regional readout remains shared (`S/modeling/trifusion/correspondence_evidence_readout.py:65-77`). The source uses the same parameter set/address rule, not guaranteed identical learned coordinates or global features across separately trained endpoints.

**Interpretation warning:** recurrent-versus-mean changes both state carry and final-vs-average depth aggregation (`S/modeling/trifusion/cross_depth_role_state.py:60-71`). It cannot isolate persistence alone. Describe the joint implementation difference; a persistence-only causal claim would need a separately registered matched aggregation control. This audit does not change or authorize changes to the frozen panel.

Checkpoint saving and reloading explicitly bind depth mode in addition to schema, dataset, seed, protocol SHA, baseline SHA, flags and condition; state keys are checked before strict loading (`S/tools/run_cross_depth_role_state.py:35-56`). The collector verifies complete metadata order, full distance shapes and all three CPU scores, plus the explicit best epoch and mode bindings (`S/tools/collect_cross_depth_role_state.py:65-103`). These are source-level requirements, not evidence they succeeded in any endpoint.

The old base train/evaluate/checkpoint bodies, the historical official/OOF training evaluators, author's R1_mAP wrapper classes, reranking/visualization, V18's historical evaluator and M3 teacher-target losses are dormant on this route. Only specific loader/scorer/helper functions are reused. The distance analyzer directly calls the real-label scorers when invoked, but the current queue/collector does not call the analyzer (`S/tools/analyze_correspondence_distances.py:28-75`).

**Launch-binding warning:** three modules imported at module load were absent from the immutable launch manifest:

- `tools/train_msvr310_trifusion_oof.py`
- `tools/train_official_three_dataset_roles.py`
- `tools/audit_v17_full_gallery.py`

The first two are imported by `S/tools/run_official_three_dataset_roles.py:17-18`; the third by `S/tools/train_signal_preserving_v18.py:16`. Their current supplemental raw bytes match receipt 670's SHA/size values. Relevant top-level imports/constants were inspected directly (`A/tools/train_msvr310_trifusion_oof.py:15-24`; `A/tools/train_official_three_dataset_roles.py:8-12`; `A/tools/audit_v17_full_gallery.py:3-16`). Their historical scoring/training bodies are not active here, and no further unseen project import was needed for the traced route. The supplied current bytes close semantic visibility, but cannot retroactively establish execution-time freezing; the three names remain absent from `S/evidence_manifest.json:64-269`. Current Git/blob claims in receipt 670 are collection evidence, not an independent historical execution attestation (`P/logs/cross_depth_additional_sources_670_20260929.json:5-28`).

## E. Scope and claim ceiling

The source registers three modes × three datasets and only seed42. It selects the largest official fused mAP over 50 epochs, using the later epoch for exact ties; other metrics are then taken from that one checkpoint (`S/tools/run_correspondence_context_identity.py:137-142`; `S/tools/collect_cross_depth_role_state.py:65-85`). Thus official query/gallery labels affect checkpoint selection even though gradient losses use train labels. “No test-label learning” must not become “test-label-blind selection.”

The plan itself acknowledges this as a one-seed official-best development comparison (`P/refine-logs/cross_depth_role_state_v1/EXPERIMENT_PLAN.md:23-35`). This audit provides no full-panel proof, no seed-variance estimate and no unbiased held-out generalization claim. The analyzer's identity bootstrap resamples fixed-model identities, not training seeds (`S/tools/analyze_correspondence_distances.py:57-75`).

## F. Evaluation classification

- **Active fused/global/local retrieval: real_gt.** Labels are read from frozen dataset records and used by the actual called scorer; they are not synthesized by the model (`S/tools/official_three_dataset_data.py:8-18`; `S/tools/run_correspondence_context_identity.py:207-224`).
- **M0 reload comparison: engineering model-output proxy.** It compares original and reloaded outputs on a small training batch, not retrieval accuracy (`S/tools/run_correspondence_context_identity.py:151-166`). The JSON marks this narrow comparison `synthetic_proxy`; it does not label dataset supervision as synthetic.
- **M3 teacher-target consistency: dormant model-target proxy.** M3 is disabled and the active context forward/loss does not produce or consume those targets (`S/modeling/trifusion/correspondence_roles.py:233-240,284-311`; `S/modeling/trifusion/correspondence_context_identity.py:69,92-109`).
- CPU structural checks and any `TinySequenceMixer` tests do not establish production retrieval performance; no such test result was certified in this call.

## Required reporting consequences

Preserve the catalogue-only data boundary and the three launch-unbound imports when reporting these runs. Preserve the exact final-vs-mean aggregation distinction and the single-seed official-best selection qualifier. Do not rewrite historical manifests to imply stronger launch provenance. Any future complete-freeze claim should bind its transitive import closure prospectively.

Completed performance endpoints, checkpoint/data/array replay and training stability require separate primary-artifact review. The current verdict remains **WARN, same-family, provisional**, with no observed fake retrieval GT or prediction-statistic metric normalization in the inspected active source.

