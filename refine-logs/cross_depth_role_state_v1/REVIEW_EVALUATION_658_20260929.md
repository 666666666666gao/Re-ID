# Evaluation source audit — cross-depth role state

Date: 2026-09-29. Reviewer: native Codex `gpt-6-astra`, reasoning effort `max`, canonical task `/root/audit_cross_depth_eval_658`. Review independence: **same-family**. Acceptance: **provisional**. No opaque agent ID was exposed.

**Overall: WARN. Source design: PASS with stated limits. Remaining concrete code blockers: none in the reviewed paths. New runtime verification: NOT_RUN.** This is a source review and a check of existing local receipts, not a successful M0, a nine-endpoint result, or a remote artifact replay. One-seed and official-best limitations qualify later claims; they are not permission blockers.

The reviewer read the supplied evaluation/queue/model files directly, followed their live data/scorer/initializer dependencies, checked the archived panel arithmetic and six terminal text hashes, and re-read the new files after the executor's corrections. No SSH, GPU/model execution, training, environment modification, external reviewer, or scientific-source edit was performed by this reviewer. Only this report and its JSON companion were written. The JSON records all 37 audited file hashes.

## Concrete findings and their resolution

1. **Resolved acceptance failure — dormant auxiliary coefficient.** The initially reviewed new entry forced `auxiliary_id_weight=0.0`; its collector also demanded zero, while the reused auditor unconditionally demanded 1.0. Calling the auditor would therefore reject every otherwise complete endpoint. The executor removed the override, and the current collector requires 1.0. `auxiliary_target=none` still freezes the auxiliary head and produces exactly zero auxiliary loss. The original auditor is unchanged. Evidence: `tools/run_cross_depth_role_state.py:24-32`; `tools/run_correspondence_context_identity.py:35-40,112-116`; `tools/collect_cross_depth_role_state.py:45,103`; `tools/audit_correspondence_context_identity_losses.py:11-26`; `modeling/trifusion/correspondence_context_identity.py:87-90`. This is a receipt-compatibility correction, not an added loss.

2. **Resolved provenance gap — live dependencies omitted from the source freeze.** The initial 13-file list omitted the live scorers/loaders, protocol and distance helpers, initialization/schedule/loss code, configuration files, and constructor/state dependencies. The revised registration hashes the explicit tools/config list, all local `modeling/trifusion/*.py` recursively, all upstream Signal Python files recursively, the three Signal YAML files, and the three protocol JSON files. It checks that manifest before each stage and at closure; the collector checks it again. Evidence: `tools/queue_cross_depth_role_state.py:19-47,64-73,127-137,151`; `tools/collect_cross_depth_role_state.py:123`. Baseline bytes are independently checked on model construction and collection (`tools/official_three_dataset_model.py:53-69`; collector `:124-125`). This closes the identified scientific-source coverage gap. It does not attest the remote installed libraries, compiled Mamba/CUDA binaries, or pristine upstream working-tree status.

3. **Clarified prior-effect wording.** The plan now correctly identifies its three numbers as separate dataset effects averaged over the none/local supervision strata, not a mean across datasets (`EXPERIMENT_PLAN.md:5`). The numbers themselves were correct.

The revised collector also explicitly requires M0 epoch 1 with eight steps, batches 0–7, no auxiliary target/loss, and a hash-bound reload probe with the cross-depth schema and correct depth mode (`tools/collect_cross_depth_role_state.py:54-64`). No production M0 artifact was supplied or created in this review.

## A–F integrity checks

### A. Ground-truth provenance — PASS at source level

Retrieval labels come from the protocol's dataset record identities, cameras and scenes, not model predictions (`tools/run_correspondence_context_identity.py:207-212`). The protocol builder copies those values from the audited inventory, checks train/test identity separation and positive eligibility, and binds the inventory hash (`tools/build_official_three_dataset_protocols.py:32-79,90-99`). Training alone uses relabeled train identities; query/gallery retain original identities (`tools/official_three_dataset_data.py:8-18`).

The complete expected query/gallery sizes are RGBNT201 836/836, RGBNT100 1715/8575 and MSVR310 591/1055 (`tools/run_official_three_dataset_roles.py:58-68`). Extraction and saved matrices enforce those shapes. Camera scoring removes only same-identity AND same-camera pairs; MSVR scoring removes only same-identity AND same-scene/time-period pairs. Different-identity distractors remain (`tools/train_rgbnt100_signal_oof.py:253-268`; `tools/train_msvr310_signal_oof.py:223-238`).

The evaluator calls the Signal author scorer from its asserted module path for every output and requires agreement with the independent NumPy scorer (`tools/run_correspondence_context_identity.py:185-224`). The collector checks all saved identity/camera/scene arrays against every protocol record (`tools/collect_cross_depth_role_state.py:87-102`).

Limit: the runtime protocol directory and upstream `utils/metrics.py` are absent from this local mirror. This reviewer verified the source-level provenance chain, not the remote dataset files, inventory construction, or upstream scorer bytes.

### B. Score normalization — PASS

AP is mean precision at true-positive ranks; Rank-k is the fraction of queries whose first retained true match is within k. The only metric scale is multiplication by 100. No reported retrieval metric is divided by a model-output maximum, minimum or mean (`tools/train_msvr310_signal_oof.py:223-238`; `tools/train_rgbnt100_signal_oof.py:253-268`).

L2 feature normalization before squared Euclidean distance is retrieval geometry, not score inflation (`tools/run_official_three_dataset_roles.py:230-238`). Local/global/fused paths all use the same label-based scoring rule. Loss reconstruction adds the logged task terms directly; zero auxiliary loss is enforced for this experiment (`tools/audit_correspondence_context_identity_losses.py:20-38`).

### C. Result existence and claim matching — WARN; no phantom new result found

The new tracker says all nine endpoints are `NOT_STARTED`, and the plan explicitly says no formal result exists (`EXPERIMENT_TRACKER.md:3-11`; `EXPERIMENT_PLAN.md:3,9`). There is no supplied cross-depth M0, full50 training receipt, checkpoint, distance matrix, or accepted matrix to validate. The source collector cannot establish those facts until artifacts exist.

The preceding panel's local matrix exists and contains 15 unique dataset/condition rows, all `VERIFIED_COMPLETE` (`logs/context_identity_accepted_complete_670_20260929.json:1-8` and its rows). Its SHA-256 is `7fd21fa6398578ff22aba2f1812ca2c6a163287ae2447263811941b53180f958`. The companion completion receipt has 15 COMPLETE/exit-0 jobs, and the terminal archive receipt names the same matrix hash. The six archived final text files match their recorded hashes. These are checks of existing receipts; no old checkpoint or full distance array was replayed.

For each dataset, independently recomputed
`((context_none-static_none)+(context_local-static_local))/2`
gives:

| Dataset | Query main effect, mAP points |
|---|---:|
| RGBNT201 | +0.020999111887185506 |
| RGBNT100 | +0.0022817453653800612 |
| MSVR310 | +0.021399504644012524 |

Inputs are the four relevant fused-mAP rows per dataset at accepted-matrix lines 18/339/660/981, 125/446/767/1088, and 232/553/874/1195. The largest recorded prior fused/global/local CPU metric discrepancy is 2.937453785989419e-6 points. This last value summarizes the archived checks; it is not a new CPU recomputation of remote arrays.

### D. Live metric and selection paths — PASS at source level

The new entry replaces the context model/build/save/load/evaluate functions, then enters the reused runner (`tools/run_cross_depth_role_state.py:67-77`; `tools/run_correspondence_context_identity.py:244-254`). The reused training function calls official fused scoring every epoch and saves whenever mAP is at least the previous best. The collector independently selects the maximum `(mAP, epoch)`, so exact ties go to the latest epoch (`tools/run_correspondence_context_identity.py:93-146`; collector `:65-85`).

The best-checkpoint loader checks dataset, seed, protocol, baseline, module flags, condition, explicit depth mode, exact checkpoint key set and a strict tensor load (`tools/run_cross_depth_role_state.py:44-55`). The separate evaluation process rebuilds the same initializer, reloads that checkpoint, and evaluates fused, shared_global and joint_local from that single state (`tools/run_correspondence_context_identity.py:180-239`). The collector recomputes all three saved complete-gallery matrices on CPU and audits every training step (`tools/collect_cross_depth_role_state.py:86-114`). These are live call paths, not unused metric definitions. No runtime success is inferred from their existence.

### E. Scope and claim ceiling — WARN, correctly disclosed

The registered scope is three depth conditions × three datasets × seed42, full 50 epochs, with checkpoint selection by official fused mAP. All nine endpoints remain unrun in the reviewed tracker. This permits an eventual descriptive same-protocol comparison, not training-variance estimates, unbiased held-out generalization, broad robustness, SOTA, or a verified ten-point improvement.

The plan explicitly distinguishes downstream role state across snapshots from a state passed through every CLIP block, keeps the historical negative panel, uses an independent three-call mean control, and requires recurrent-versus-mean evidence before a persistence claim (`EXPERIMENT_PLAN.md:9-33`). The model implements one fixed-address prediction followed by one or three role steps; recurrence adds previous role evidence and returns the last depth, while mean averages the three independent outputs (`modeling/trifusion/cross_depth_role_state.py:20-71`). Parameter/RNG matching and real performance still need their production receipts.

### F. Evaluation type — PASS classification

The intended retrieval outputs are **real_gt**: dataset identity labels determine matches, while camera/time-period metadata defines exclusions. The auxiliary target is `none`; the global context controls candidate selection rather than supplying a retrieval target. M0 and any synthetic CPU operator check are engineering/structural evidence, not retrieval performance. No current cross-depth scientific result is classified as completed.

The subsequently supplied `logs/cross_depth_structure_658_20260929.json` records `SYNTHETIC_CPU_STRUCTURE_PASS` with synthetic inputs and `TinySequenceMixer`, source hash matching the reviewed model. Its 55 tensors/757824 parameters are the tested role-only substitute, not the full production model. Its zero control-parity differences and nonzero depth-input gradients are structural checks only. This reviewer read that receipt without rerunning it; it is not production Mamba M0 or ReID evidence.

## Nine-endpoint and queue contract

The condition map, command builder, child naming and collector agree on `mixed_once`, `depth_mean`, `depth_recurrent` across RGBNT201, RGBNT100 and MSVR310. The collector demands the exact ordered nine-job list and binds each completed child through dataset, variant, checkpoint depth mode, initialization, protocol and baseline (`tools/queue_cross_depth_role_state.py:18,51-60,65,125-126`; `tools/queue_correspondence_refinement.py:19,34-36`; collector `:31-85,117-147`).

The predecessor must already be COMPLETE with 15 exit-0 jobs and a hash-bound 15/15 accepted matrix before new registration (`tools/queue_cross_depth_role_state.py:115-122`). Workers execute M0 → train → evaluate and stop on nonzero exit. Before training, the worker checks the eight-step M0 history, gradient coverage, frozen-state/reload fields, probe hash and batch sequence. A child becomes COMPLETE only after final verification (`:72-106`). The controller's reused run loop records a failed child, stops launching pending jobs, lets already active children finish, and returns failure; it has no retry/resubmission path and polls every 240 seconds (`tools/queue_correspondence_refinement.py:135-181`). No kill/preemption command appears.

An assertion failure inside child verification can leave that child's last receipt at RUNNING, with its error preserved in the log; the parent observes the nonzero worker exit and marks the endpoint/campaign FAILED. Do not interpret that stale child text as a live process. This is a reporting limitation, not an automatic retry or a false accepted result.

## Claim impact and next evidence

- **Supported now:** the described source contract, correct prior-panel arithmetic, and closure of the two identified code findings.
- **Needs qualification:** eventual seed42 official-best comparisons and archived receipt-based checks of the prior panel.
- **Unsupported now:** cross-depth production M0 success, any completed new endpoint, better retrieval, persistent-role benefit, or runtime/compute equivalence.

Proceeding to the already authorized real M0/evaluation sequence does not require permission because of this audit. Retain the existing no-retry policy and collect the real nine-endpoint evidence before filling results or promoting the hypothesis. This source audit does not replace those checks.
