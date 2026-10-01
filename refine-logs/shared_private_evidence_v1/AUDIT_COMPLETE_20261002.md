# Shared/private complete714 experiment integrity audit

- Experiment date: 2026-10-01. Audit finalized: 2026-10-02 Asia/Shanghai.
- Project: `C:/Users/gb/.trifusion_github_publish_22c3bee`.
- Auditor: `gpt-6-astra`, reasoning `max`, fresh reviewer context.
- `review_independence: same-family`; `acceptance_status: provisional`.
- Reviewer task: `/root/audit_shared_private_complete714`.
- Overall verdict: **WARN**. Deterministic evidence consistency: **PASS**. Recorded engineering/M0 completion: **PASS within the inspected scope**. Registered scientific S1/S2/joint gates: **FAIL/FAIL/FAIL**.

This is an integrity audit, not a successful-improvement or SOTA verdict. I found no fabricated ground truth, self-normalized performance, missing accepted endpoint, or unsupported success promotion in the complete report. The warning concerns the stale tracker/handoff and the limits of single-seed, warm-start, official-test-selected evidence.

Paths below are relative to the project unless explicitly absolute. The accompanying verdict JSON records exact SHA256 hashes, expanded endpoint file:line evidence, the remote read-only inspection script/output, and the scope of deterministic checks. I did not run training, a neural forward, an optimizer, any metric function, bootstrap, collector, or report. I did not copy weights or distance arrays locally. Existing scores were compared between saved receipts, histories, checkpoint payloads and tables; I did not independently recompute AP/ranks or the reported bootstrap.

## A. Ground-truth provenance — PASS

The active evaluation derives identities, cameras and MSVR scene/time-block fields from fixed dataset records, not predictions. `tools/run_correspondence_roles.py:85` extracts the complete query and gallery; lines 88–99 read identity/camera/scene fields and pass them to both the local arithmetic scorer and the pinned upstream evaluator. `tools/official_three_dataset_data.py:8` uses original identities for evaluation and the train label map only for training.

The protocol builder declares the actual official directories and fixed counts at `tools/build_official_three_dataset_protocols.py:10`; it preserves every record and modality path at lines 44–55, asserts split identity disjointness at lines 36–40, validates every query has positives at lines 56–67, and documents the conjunction filter at line 79. Runtime counts are asserted again in `tools/run_official_three_dataset_roles.py:58`.

I independently enumerated the remote dataset files and compared every primary path, every modality-path existence, every identity/camera/scene parsed from filenames, and all record indices against the three stored protocols:

| Dataset | Training records / IDs | Query records / IDs | Gallery records / IDs | Gallery-only IDs | Legitimate positives/query |
|---|---:|---:|---:|---:|---:|
| RGBNT201 | 3951 / 171 | 836 / 30 | 836 / 30 | 0 | 7–21 |
| RGBNT100 | 8675 / 50 | 1715 / 50 | 8575 / 50 | 0 | 50–175 |
| MSVR310 | 1032 / 155 | 591 / 52 | 1055 / 155 | 103 | 1–31 |

All train identities are disjoint from query/gallery identities. Every registered query is valid. All 103 MSVR gallery-only identities remain as distractors. This inspection checked dataset files and metadata, not merely the executor's inventory description.

The authoritative pinned parser evidence is remote under `/data/gaob/Re-ID/Trifusion/comparators/Signal-cd1b0a6/`: `data/datasets/RGBNT201.py:61` parses identity/camera and uses train_171/test; `RGBNT100.py:63` parses montage identity/camera; `msvr310.py:67` parses identity, camera and scene. The remote `utils/metrics.py:68` removes only same identity AND same scene for MSVR; line 137 removes only same identity AND same camera for the RGBNT datasets. Different identities from the same environment remain.

Every saved distance tensor has the complete shape (836×836, 1715×8575 or 591×1055), is finite, and has all six label/environment arrays exactly equal to the corresponding protocol. The source performs these checks at `tools/collect_shared_private_evidence.py:87`; this audit independently checked the saved array structure and labels remotely.

This establishes agreement with the pinned, installed dataset/parser protocol. It does not establish exact protocol/resource parity with every external paper.

## B. Score normalization — PASS

The reported mAP is ordinary query-mean AP, with AP divided by the number of real positive matches and rank positions. CMC is the fraction of valid queries whose first positive is within the requested rank. See `tools/train_rgbnt100_signal_oof.py:253` and `tools/train_msvr310_signal_oof.py:223`. The pinned upstream evaluator uses the same real-label denominators at remote `comparators/Signal-cd1b0a6/utils/metrics.py:93` and `:153`. The multiplication by 100 at `tools/run_correspondence_roles.py:100` is percentage conversion.

L2 feature normalization at `tools/run_official_three_dataset_roles.py:233` and `modeling/trifusion/role_global_tokens.py:89` defines the retrieval representation/distance; it is not division of a reported score by the model's own best output. No such score normalization is present in the active path. Neither reranking nor a ground-truth-based inference selection is used.

Identity-macro AP change is a distinct diagnostic: `tools/analyze_correspondence_distances.py:57` first averages query AP changes within each real identity, then line 72 averages identities equally. It must not be substituted for query-weighted mAP. `REPORT.md:15` correctly labels that column separately.

## C. Result existence, selection and execution — WARN for documentation lag; deterministic chain PASS

All nine unique endpoints are backed by actual complete training, stage-exit, evaluation, checkpoint and distance artifacts. I checked all 450 epoch records against stdout and training histories, all 30,624 formal step rows for continuity, finite loss, ID-plus-triplet agreement and epoch-mean agreement, all nine selected epochs, and all nine evaluation stdout records against `official_metrics.json`. All 27 recorded M0/train/evaluate stages exited 0. Formal training steps are 2649 per RGBNT201 flow, 6559 per RGBNT100 flow, and 1000 per MSVR flow.

The nine official metric rows are:

| Dataset | Flow | Selected epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | coupled_roles | 1 | 74.050605 | 75.598085 | 85.047847 | 88.636363 |
| RGBNT201 | separated_roles | 1 | 73.824170 | 75.239235 | 84.449762 | 88.516748 |
| RGBNT201 | global_only | 1 | 73.744215 | 74.760765 | 84.808612 | 88.755983 |
| RGBNT100 | coupled_roles | 1 | 85.558109 | 95.276970 | 95.976675 | 96.559769 |
| RGBNT100 | separated_roles | 1 | 85.417589 | 95.335275 | 95.860058 | 96.384841 |
| RGBNT100 | global_only | 1 | 85.379916 | 95.218658 | 96.034986 | 96.501458 |
| MSVR310 | coupled_roles | 10 | 53.243449 | 67.343485 | 83.079529 | 89.001691 |
| MSVR310 | separated_roles | 10 | 53.072084 | 67.681897 | 83.079529 | 89.001691 |
| MSVR310 | global_only | 10 | 52.935430 | 67.851102 | 82.910323 | 88.663280 |

These are already-published saved numbers, not audit-generated scores. All nine metric-table rows and all nine comparison-table rows agree with `results/shared_private_evidence_complete_20261001/SUMMARY.json` and `REPORT.md:5`.

**Selection and same-weight metrics.** The frozen plan specifies highest official fused mAP, with the later epoch winning a tie (`refine-logs/shared_private_evidence_v1/EXPERIMENT_PLAN.md:35`). The actual loop evaluates every epoch and overwrites the one best checkpoint on `>=` at `tools/run_visual_update_control.py:188`. Evaluation selects `max(mAP, epoch)`, asserts the selected payload and historical metrics agree, strictly reloads the complete model, and evaluates that one model at lines 227–248. All four metrics come from the same weight file; there is no independent best epoch per metric. Early selected epochs do not mean early termination.

**Full-state evidence.** `tools/run_visual_update_control.py:93` saves every model state entry, including Signal/visual state, and line 108 uses `strict=True`. This is the active saver/loader; the older role-only checkpoint helper is not used here. I remotely loaded each saved checkpoint on CPU, verified its exact whole-file hash, dataset, condition, epoch, baseline/protocol bindings and four metric values. All checkpoints contain 152 FP32 visual tensors; the 7 RGBNT201 or 19 vehicle nonvisual Signal tensors equal the bound baseline exactly, while visual weights have changed. M0 probes match the full checkpoint's key set, shapes and dtypes. All saved states are finite.

The nine revised M0 receipts each record eight actual steps, all 334/334 role-flow or 208/208 global-flow trainable tensors with nonzero gradients, frozen-state preservation and reload difference 0. Their probe file hashes were independently checked remotely. Formal runs build fresh models and never read M0 probe weights: `tools/run_visual_update_control.py:112`; queue separation/reuse of only the three RGBNT201 M0 receipts is explicit at `tools/queue_shared_private_evidence.py:56`.

**Source/input binding.** All 59 source-artifact entries in SUMMARY have hash-matching local raw counterparts. Remotely, all 229 active source hashes and 222 sealed predecessor hashes match, as do report/waiter source hashes and both figure hashes. The 27 local source byte differences from the remote manifest are solely CRLF versus LF; I checked this explicitly and did not normalize or edit files. The missing local upstream/protocol copies were inspected at the authorized remote paths.

Useful exact terminal examples are:

- `logs/shared_private_complete714_20261001/raw/trained-model/shared_private_evidence_20261001_v2_shared_private_separated_roles_RGBNT100_seed42_full/training.json:3`, `:642`, `:674`: complete status, epoch 50, selected epoch 1.
- `logs/shared_private_complete714_20261001/raw/trained-model/shared_private_evidence_20261001_v2_shared_private_global_only_RGBNT100_seed42_full/training.json:3`, `:642`, `:674`: the other final endpoint.
- Every older endpoint's expanded training/official/campaign/log references and hashes are in the verdict JSON; the correct terminal archives are 711 for coupled RGBNT201/MSVR and separated RGBNT201, 712 for separated MSVR/global RGBNT201, 713 for coupled RGBNT100/global MSVR, and 714 for separated/global RGBNT100. Older partial snapshots are retained, not mistaken for terminal ones.
- `logs/shared_private_complete714_20261001/raw/logs/shared_private_evidence_20261001_v2/analysis_waiter_status.json:2` records CPU_REPORT_COMPLETE; lines 7, 14–26 record one reporter invocation, its PID, actual exit 0, and completion at 22:42:41.662988+08:00. `complete_analysis.log:1` records accepted=9 and S1/S2 FAIL.
- The one-invocation path is concrete: `tools/wait_shared_private_complete_analysis.py:87` waits for both actual terminal wrappers, lines 97–106 launch once through an exclusive log and wait for its actual exit. `tools/report_shared_private_evidence_complete.py:52` rejects an existing output directory. The archived process check at `logs/shared_private_complete714_20261001/raw/logs/shared_private_storage714_20261001.json:3` records all four relevant processes absent at 23:51:52. This supports one recorded successful report invocation, not a claim of externally attested process history.
- `logs/shared_private_complete714_20261001/ARCHIVE.json:3` binds the terminal snapshot and its report completion.

**Documentation finding C1 (non-blocking).** The supplied `refine-logs/shared_private_evidence_v1/EXPERIMENT_TRACKER.md:3` still states 7/9 accepted and zero report invocations; line 5 still states the workers/waiter are alive. The handoff's last section at `docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:14162` likewise records the old 7/9 milestone, with the next observation still pending at line 14174. They are accurately dated old snapshots, but they are no longer current completion documentation. Update the current tracker and append the terminal handoff; preserve the dated milestone files. The preregistered plan's pre-M0 status line is historical and should remain frozen, with completion recorded elsewhere.

## D. Metric execution / dead-code check — PASS for the active claims

The active call chain is:
`run_shared_private_evidence.main` → `run_visual_update_control.train/evaluate` → `run_correspondence_roles.official_metrics` → `camera_scores/scene_scores` AND pinned Signal `eval_func/eval_func_msrv`. Exact evidence: `tools/run_shared_private_evidence.py:63`, `tools/run_visual_update_control.py:189`, `:240`, and `tools/run_correspondence_roles.py:79`.

The upstream path is asserted at `tools/run_correspondence_roles.py:84`; `tools/official_three_dataset_model.py:34` configures Signal imports using `tools/run_signal_baseline_dev.py:57`. Thus the root project's `utils/metrics.py:111` and `utils/reid_evaluation.py:8` are not the active evaluator for these nine runs. Their definitions do not constitute execution evidence. The root legacy MSVR helper, visualization functions, old context runner's three-output evaluator, and old M3 prediction helpers are also not evidence of extra outputs/evaluations in this panel. This is deliberate inactive legacy code, with no corresponding results claimed by the complete report.

The collector actually calls complete-gallery scorers at `tools/collect_shared_private_evidence.py:97`; the queue calls verification at `tools/queue_shared_private_evidence.py:85` and the complete collector at line 152. The reporter calls `compare` at `tools/report_shared_private_evidence_complete.py:111`, and compare calls the scorers and writes real-GT repair/AP statistics at `tools/analyze_correspondence_distances.py:40`. Their successful terminal artifacts exist. The recorded maximum replay differences are below 1e-5 percentage points; small float64/float32 CMC differences explain the final decimal differences between diagnosis deltas and subtraction of displayed official scores.

This audit checked the call chain and recorded execution outputs. It did not call any of these scoring functions again.

## E. Scope, matched control and claims — WARN / qualified evidence only

The actual scope is three installed official dataset protocols × three flows × one adaptation-stage seed (42), with 50 full epochs each. The fixed build sets width 128, M1/M2 on, M3 off, no prediction loss, static extra global token, context-conditioned queries and 1536-dimensional fused retrieval: `tools/run_visual_update_control.py:66`; `tools/run_shared_private_evidence.py:55`. Inputs are 256×128 for RGBNT201 and 128×256 for vehicles (`tools/official_three_dataset_model.py:45`). AdamW, new-module LR 3.5e-4, visual LR 5e-6, weight decay 1e-4, warmup 5 and cosine 50, CE smoothing 0.1 and triplet margin 0.3 are bound in the plan/initializers and loop; no seed or recipe search is present in this panel.

**Matched S1 control.** The private adapter bank is an exact deepcopy at `modeling/trifusion/shared_private_evidence.py:21`. The only coupled/separated forward branch is direct private-mean writeback at line 39. Common backbone/head initialization excludes private adapters by name (`tools/check_shared_private_initialization.py:49`), and the witness verifies tensor equality and the full coupled/separated state hashes at lines 55–72. All training/M0 initializer bindings agree with that witness. The paired initial full-state hashes are:

| Dataset | Coupled and separated full initial state SHA256 | Trainable parameters per role flow |
|---|---|---:|
| RGBNT201 | 96a85fef436571c55440c03807fd730d251923ffafa7519894d939e516a2a4f7 | 89,917,706 |
| RGBNT100 | 62d001b975a5876578c5f581cf4e235b55b6a2037298fe94cbc77de25e5c71fa | 89,731,850 |
| MSVR310 | 36bc43bbe8bccb4177bdcaadda4aca8b49cedfa22686497bd5c0686baae5e219 | 89,893,130 |

The full initial state is evidenced by the saved witness/bindings and source construction; this audit did not rebuild nine initial models or independently regenerate initial-state hashes.

**Independent global control.** `tools/run_visual_update_control.py:30` drops role computation but retains the original shared backbone, neck and classifier. It is trained through its own full process/checkpoint, not sliced out of a role checkpoint. It still has shared adapters and differs in capacity: global trainable counts are 87,310,656 / 87,124,800 / 87,286,080. Repeating this prior recipe at the same seed is not a new training-seed replication. S2 cannot by itself isolate heterogeneous operator benefit from all additional capacity/cost.

**Failure and correction preserved.** The original failed M0 records actual exits 0/1/0 and no witness/formal launch at `logs/shared_private_preflight_failure_709_20261001/FAILURE.json:4`, `:95`. Its separated stdout actually ends with the 26-missing-gradient assertion at `logs/shared_private_preflight_failure_709_20261001/raw/logs/shared_private_preflight_20261001_v1/separated_roles.log:17`. All 24 files in the original intake still match their bound bytes and hashes.

The diagnostic's scope and small non-bitwise loss mismatch are explicit at `refine-logs/shared_private_evidence_v1/GRADIENT_DIAGNOSTIC_20261001.json:39`, `:49`, `:78`, `:458`. Its saved rows show the 26 missing gradients become nonzero under the recorded FP32 local derivative replay; this is numerical evidence at that captured state, not a universal causal proof or a historical bitwise reconstruction.

Comparing the original 229-source snapshot with the revised one changes only the model file and the appended numerical plan. The executable diff solely disables autocast inside both private MLP banks, converts their inputs to FP32 and casts output back at `modeling/trifusion/shared_private_evidence.py:32`. The paired correction does not relax the eight-step nonzero-gradient gate, change loss/LR/seed, or revise S1/S2. See `EXPERIMENT_PLAN.md:62`–68.

**Registered thresholds and negative results.** The original sealed plan at `logs/shared_private_preflight_failure_709_20261001/raw/refine-logs/shared_private_evidence_v1/EXPERIMENT_PLAN.md:43` and the revised plan at `refine-logs/shared_private_evidence_v1/EXPERIMENT_PLAN.md:43` retain the same rule: all three mAP deltas positive, all three R1 deltas nonnegative, and at least +0.5 pp mAP on RGBNT201 and MSVR310. The reporter implements it exactly at `tools/report_shared_private_evidence_complete.py:24`, combines all three datasets at lines 116–117, and keeps the third comparison family diagnostic only.

| Registered comparison | RGBNT201 ΔmAP / ΔR1 | RGBNT100 ΔmAP / ΔR1 | MSVR310 ΔmAP / ΔR1 | Joint across datasets |
|---|---:|---:|---:|---|
| S1 separated − coupled | −0.2264 / −0.3589 | −0.1405 / +0.0583 | −0.1714 / +0.3384 | FAIL |
| S2 separated − independent global | +0.0800 / +0.4785 | +0.0377 / +0.1166 | +0.1367 / −0.1692 | FAIL |

The RGBNT100 S2 component passes its registered per-dataset rule; it does not rescue the other datasets or the joint gate. All three additional coupled-versus-global comparisons are retained at `REPORT.md:23`, including MSVR R1 degradation, with the diagnostic qualifier at line 27. All nine final mAP values are below their own selected best, by 2.40–9.53 pp; all nine full trajectories and losses remain visible. `tools/report_shared_private_evidence_complete.py:129` plots every flow and epoch, and the inspected PNG agrees with the SVG/report source and hashes. The trajectory permits a descriptive late-stage retrieval deterioration observation, not an isolated unique mechanism attribution.

**AP/bootstrap interpretation.** Each comparison retains all 30/50/52 query identities, including zero changes; I checked their identity keys and query counts against the protocols. The 2000 draws at seed 42 resample those fixed-model identity means, not training seeds (`tools/analyze_correspondence_distances.py:57`–74). Every reported interval spans zero. They are percentile intervals for an identity-macro diagnostic after official checkpoint selection, not independent untouched-test significance, training stability, or proof of no possible benefit. The headline mAP is query-weighted and can differ from the macro diagnostic; the report keeps the distinction.

**Timing/storage.** `SUMMARY.json:3041` records campaign wall 12,762.786602 seconds; summed endpoint wall is 37,743.256675 seconds. These are different quantities because workers overlap. `tools/report_shared_private_evidence_complete.py:86` explains parent-observed preflight M0 timing and endpoint boundaries. Training step seconds exclude per-epoch evaluation; the separately named training-and-epoch-eval interval includes it. Peak allocated memory is measured after initialization at `tools/run_visual_update_control.py:148`, `:203`, excluding initialization transients/reserved memory. Equal epochs are not equal computation.

The 6,633,154,815 logical output bytes at `SUMMARY.json:3043` cover nine revised M0 and nine full output directories; I independently verified the same current total. They do not cover upstream weights/data/report, historical peak filesystem use, or the failed original preflight/diagnostic. Likewise these adaptation-stage timing totals are not total research cost: upstream ReID training, the original failed preflight, diagnostic and initialization witness are separate work. Carry that boundary into any cost comparison.

**Claim ceiling.** The report already disclaims training stability, clean public CLIP initialization and SOTA at `REPORT.md:34`, `:37`, `:41`–46. Strict loading of the entire trained ReID checkpoint occurs at `tools/official_three_dataset_model.py:63`; camera/nonvisual state is retained. Removing direct writeback does not prevent role gradients from modifying shared/visual parameters. Neither the adapter architecture nor this evaluation validates semantic shared/private region decomposition; the handoff correctly makes that distinction at line 14156. Do not turn this negative adapter-writeback test into a rejection of every shared/private method or an attribution to a unique underlying cause.

## F. Evaluation type — PASS: real_gt, with separate engineering evidence

- Official retrieval and saved-distance AP/repair diagnostics: **real_gt**.
- M0 gradient/frozen-state/reload checks and the initialization witness: engineering checks, not retrieval-accuracy claims.
- Original failure diagnosis: engineering numerical probe on real training batches, not an improvement score.
- The legacy M3 teacher-generated prediction targets in `modeling/trifusion/correspondence_roles.py:284` would be a self-supervised training target. M3 is disabled in this panel; those targets are not used as retrieval ground truth.

No synthetic_proxy, simulation_only or human_eval result is being passed off as the official retrieval benchmark here.

## Action items and claim impact

1. Update the current tracker and append the complete terminal handoff with 9/9, actual completion/report times, S1/S2/joint FAIL, and this provisional audit. Preserve the frozen plan and dated 7/9 snapshots.
2. Retain the one-seed adaptation-stage, trained ReID/camera initialization, official-best selection, fixed-model identity bootstrap and unequal global-control capacity qualifiers wherever these numbers are reused.
3. Preserve the original M0 failure and paired numerical correction. Do not revise the registered thresholds, select another epoch/seed, or reclassify the negative outcomes to obtain success.
4. Label the existing cost/storage numbers as the specified successful-v2-stage scopes; do not present them as full upstream-plus-failure research budget or peak storage.
5. Cite this review as direct source/artifact/hash validation. It is not a new independent AP/CMC/bootstrap recomputation or historical GPU execution attestation.

Supported: nine completed fixed full50 runs; same-checkpoint four-metric reporting; full legitimate gallery retention; qualified observed differences and failure of the exact registered S1/S2 criteria.

Needs qualifiers: matched-state direct-writeback comparison; separately trained global comparison; post-selection AP/repair/bootstrap diagnoses; adaptation-stage timing/memory/storage.

Unsupported by this evidence: successful shared/private improvement across three datasets, proven sufficient heterogeneous-role gain, clean public-initialization end-to-end training, multi-seed stability, semantic region decomposition, unique causal explanation, matched external SOTA, or completion of the broader scientific goal.

