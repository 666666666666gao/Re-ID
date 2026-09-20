# MSVR310 cross-scene Smooth-AP M0 experiment audit

**Overall verdict: WARN / CLOSED_WITH_LIMITS. Deterministic M0 engineering checks: PASS.**

The full registered M0 completed, and independent remote CPU arithmetic reproduced every saved distance/mask/objective row and all six final capacity checkpoint states. No blocking arithmetic, provenance, scope, or missing-result defect was found. The warning records the remaining boundary: parameter-gradient trajectories, frozen-start state equality, and strict-reload outputs are runtime witnesses, not a fresh reconstruction of training.

Fresh reviewer: `/root/audit_cross_scene_smooth_ap_m0_20260921`; requested `gpt-6-astra`, reasoning `max`, `fork_turns=none`. Review independence is **same-family**; acceptance is **provisional**. Backend model identity is not independently attested. The parent records the native invocation and complete response in its review trace.

Execution: `d35864d6411591e05c8ac3e5164ebae48063ad99`. Config SHA-256: `e5326b52ebb12dced24ebfac788db2e2bb5bdca0b50c6c4dec64e596f1af05a1`. Remote run: `/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d`. Last documentation observation: `2026-09-20T16:49:01.969250+00:00`, HEAD `fbd5600551446065d21ab7bded92a33c54c352e0`. Source, config and protocol paths were unchanged from execution; subsequent changes were documentation/evidence.

| Check | Verdict | Finding |
|---|---|---|
| A. Ground-truth provenance | PASS | Dataset labels and actual source filenames; all three complete source/heldout identity bindings checked. |
| B. Score/reduction arithmetic | PASS | All 248 rows, all three fused objectives, eligible masks, branch hard losses and 14-term ledgers independently recomputed. |
| C. Files and completion | PASS | M0/M0_CPU exited 0; complete 248-step files, six checkpoint payloads and final state hashes verified. |
| D. Real execution paths | WARN | No dead objective found; runtime gradient/reload witnesses cannot be expanded into independent all-step gradient reconstruction. |
| E. Scope | PASS | One dataset, one seed, source-only engineering; no retrieval qualification inferred. |
| F. Evaluation type | PASS | `real_gt` source engineering and label accounting; T0 synthetic numerical fixtures are `simulation_only`. |

## A. Ground-truth provenance and identity isolation

Dataset filename labels, not model outputs. All 1032 protocol records agree with the frozen label manifest and loader parsing; the current source directory has the exact 3096 modality filenames. Three folds are identity-disjoint; all 1950 recorded B0 source steps and all 248 M0 steps use their registered source indices. Three B0 checkpoint payloads and six M0 payloads bind source/heldout identities exactly.

The GT chain is the source image filename → original MSVR310 loader identity/camera/scene parser → frozen label manifest → train-only protocol → source-index loader → current/history masks. The current directory contains exactly 1,032 filenames in each of vis/ni/th. The source IDs per fold are 103/103/104; heldout IDs are 52/52/51; source records are 672/683/709. Each fold starts from its own fixed epoch-50 Signal checkpoint and fresh seed-42 roles. The three baseline payloads and all 1,950 saved baseline training steps were checked for source isolation. No model output supplies a target label.

Sources: `tools/audit_vehicle_query_protocol_labels.py:13-18`; `tools/build_msvr310_train_oof_protocol.py:12-55,57-123`; `tools/train_msvr310_signal_oof.py:68-99`; `tools/train_msvr310_trifusion_oof.py:27-55`. Raw independent evidence: `provenance.stdout`, `ground_truth_disk.stdout`, `independent_m0.stdout`. The historical install/CRC receipt was read as provenance, not rerun; no image contents or official-test directories were read.

## B. Independent score, mask and denominator reconstruction

The audit used standalone NumPy/stdlib arithmetic with no project objective/verifier imports. PyTorch was used only for CPU checkpoint deserialization. From every saved row it rebuilt FIFO membership, maximum age 8, capacity 512, current-record exclusion, identities/scenes, position self masks, standard positives, cross-scene positives, ignored same-ID/same-scene candidates, retained different-ID negatives, eligible counts, positive ranks and total ranks. It checked hard/standard/cross objectives and the selected loss before/after the two-step M0 warmup.

Cross-scene loss is `1 - mean(AP over eligible anchors)`. Ineligible AP zeros are placeholders excluded from that mean. There is no prediction-max/min normalization. Each positive excludes its own comparison; same-scene different-identity negatives remain. Score conversion is fixed `1 - d^2/2`, tau 0.01, weight 1. The full intervention changes both relation masks and the eligible-anchor denominator, so its future effect cannot be attributed only to deleting easy positives.

| Independent quantity | Maximum absolute error | Contract tolerance |
|---|---:|---:|
| Standard per-anchor AP | 1.85272662934e-07 | 2e-6 |
| Cross-scene per-anchor AP | 2.59938854275e-07 | 2e-6 |
| Cross-scene scalar objective | 1.39805325317e-07 | 2e-6 |
| Pooled hard objective | 3.24100256077e-08 | 2e-6 |
| Three branch current hard losses | 1.54878944142e-07 | 2e-6 |
| 14-term weighted ledger | 5.58793544769e-07 | 1e-5 |

All 3,472 component scalars were checked for finite/nonnegative values and correct weighted summation. The fused objectives and three full-branch current hard losses were recomputed from distances. Classification logits and residual distances are not saved, so the ten remaining component scalars are ledger inputs rather than independently regenerated values.

The fixed-batch overfit gate uses the registered analytic label-smoothing floor 0.5857136327437849. Its denominator is the initial loss minus that analytic floor; this is an explicitly named engineering ratio with raw losses, not a retrieval metric rescaled by its own predictions.

| Endpoint | Initial loss | Final loss | Corrected excess-loss ratio |
|---|---:|---:|---:|
| control | 4.12212991714478 | 0.588187992572784 | 0.000699680023506816 |
| cross_scene | 4.12212991714478 | 0.588189363479614 | 0.000700067677764569 |

Both ratios are below 0.1. Sources: `tools/msvr_cross_scene_smooth_ap.py:11-54`; `tools/msvr_smooth_ap.py:30-62`; `tools/run_signal_preserving_v5.py:99-135,1580-1617`; `independent_m0_remote.py`; `independent_m0.stdout`.

## C. Complete terminal evidence

The wrapper records T0 exit 0 at 00:25:36, M0 exit 0 at 00:36:52, and M0_CPU exit 0 at 00:37:03 on 2026-09-21 UTC+08. M0 status is `PASS_ENGINEERING_ONLY`; the complete project CPU verifier status is `PASS_COMPLETE_CROSS_SCENE_SMOOTH_AP_M0`. Independent checks consumed those artifacts without invoking that verifier. The current tracker agrees with those completed phases and correctly leaves M0_Audit running until this verdict is received.

- M0 summary SHA-256: `c59e39cc2192bfb11b12a58f53eb8b3853455bfd41744c74aaf4742272ba1506`.
- M0_CPU SHA-256: `80f52c03337e0c497efe25f9a888de9dae299717733f0de19114ee90052aa1d8`.
- Six capacity runs × 8 updates + two fixed-batch overfit runs × 100 updates = **248/248** checked updates.
- **15,872** current anchor exposures, **4,945,920** saved distance values, **744** fused-objective scalar definitions, **744** branch current-hard scalar checks.
- Six capacity checkpoint hashes and payloads pass. Each full 472-tensor state reconstructs from 241 exact B0 aliases plus 231 saved role tensors, and its SHA equals the training/final-reload state record.
- Eight training files and complete JSONL/distance pairs, six receipts, epoch means, actual selected objectives, paired sample/pixel hashes and overfit gates all agree. The two overfit paths have no final checkpoint by design; their final model-state record remains runtime evidence.
- Recursive static intake checked 156 registered bindings across 119 files, plus six B0 checkpoint/array bindings. All matched. The initially observed running snapshot is preserved and never treated as terminal.

The exact eight-run row/distance counts, all six checkpoint SHA values, file receipts and per-step numerical errors are in `independent_m0.stdout`. Source: `tools/train_msvr_cross_scene_smooth_ap.py:322-388`; `tools/train_msvr310_source_style.py:164-174`; `terminal_intake.stdout`; `final_document_state.stdout`.

## D. Actual call paths and runtime-witness boundary

The live trainer invokes the same objective adapter for current graphs (line 129), historical leaf partials (154) and direct full graphs (176); endpoint selection is consistent. Historical candidate VJPs are added before one optimizer step. Complete saved rows show the intended selections and accounting. However the six direct/VJP comparisons, 203/203 cumulative nonzero gradients, frozen-start hashes, RNG/buffer equality and strict reload outputs are runtime witnesses. Independent CPU arithmetic validates their scalar consistency; no saved per-step parameter gradients or inputs exist here for independent reconstruction of the full gradient trajectory.

All six direct comparisons occur at step 4, the first actual history group. Their largest recorded direct/VJP relative L2 error is **2.175816305864426e-5**, below 0.005. All six have finite scalar norm identities and consistent recorded decomposition; 5,760 historical VJP record-forwards are accounted for. This audit did not generate those gradients and does not describe these six witnesses as independently regenerated tensor comparisons. The 203/203 nonzero-gradient claim is cumulative over each run, not a claim that every parameter has a nonzero gradient on every step.

Checkpoint state reconstruction is independent. Strict reload output equality, original frozen-state equality, runtime RNG/buffer preservation and the gradient path itself retain the stated witness limit. Sources: `tools/train_msvr_cross_scene_smooth_ap.py:129-176,179-264`; `tools/msvr_freshness_probe.py:43-85`; `tools/probe_msvr_role_set_gradients.py:29-50`; `tools/probe_msvr_history_candidate_gradients.py:43-66`.

## E. Scope and complete zero-row accounting

This is one-seed, one-dataset source-only engineering evidence. It does not measure heldout retrieval. The registered 780-batch source-support sequence was independently replayed from text labels and FIFO rules: 49,920 repeated anchor exposures. Post-warmup eligible exposures are 5,144 / 5,176 / 5,056 out of 12,480 per fold. All nine zero-eligible support batches are retained; four are after warmup: fold0 step180, fold1 step221, fold2 step133 and step232. The other five are warmup rows. The completed 248-step M0 contains **zero** zero-eligible batches, so its zero-batch graph behavior is supported by the explicitly synthetic T0 test and static call path, not by a claimed observed M0 zero case.

The two overfit paths repeatedly use one fold0 batch and have no historical candidates because every stored record is in the current batch. Historical behavior is exercised by the six capacity paths. M0 records 0 heldout forwards and 0 official image reads. Candidate AP is not full-gallery mAP, and no performance, scientific promotion, robustness, or SOTA conclusion follows from engineering completion. The original failed Smooth-AP family remains unaltered. Sources: `EXPERIMENT_PLAN.md:11-24,34-50`; `provenance.stdout`; `independent_m0.stdout`.

## F. Evaluation classification

- M0 mask/loss supervision: **real_gt**, dataset-provided identities/scenes, source-only engineering.
- Source-support/T0 label replay: **real_gt**, label-only candidate accounting, no model execution.
- T0 finite-difference/tie/zero-eligible tests: **simulation_only**, explicitly synthetic numerical unit fixtures, not retrieval evidence.
- Checkpoint hashes and saved scalar checks: engineering consistency; they are not a separate predictive evaluation.

No model-generated reference is presented as real GT. Sources: `tools/check_msvr_cross_scene_smooth_ap.py:15-70`; `tools/check_msvr_cross_scene_smooth_ap_math.py:12-91`.

## Limits, action items and audit execution record

There are **no blocking issues** within the registered M0 arithmetic/checkpoint evidence. Keep the runtime limits attached to every engineering claim. Once this report and the parent trace are archived, update the M0_Audit tracker to `CLOSED_WITH_LIMITS / WARN`, with deterministic engineering PASS. Q1 requires its own complete endpoint/full-gallery and independent terminal evidence; this report contains no Q1 assessment.

Audit execution performed zero model forwards, backward calls, optimizer updates and image-content reads. All weight/array reads and numerical work remained on remote CPU; no model was instantiated and no binary was downloaded. The actual 3,096 source filenames were enumerated without reading image contents. No project, master, Git, environment configuration or running process was edited by this reviewer.

A local supplemental snapshot unpack initially failed on a deeply nested Windows path. The complete remote stdout already existed; flat numbered local filenames solved materialization. `audit_helper_failure_01.json` records the actual failure and correction. No experiment or model computation was repeated. All seven remote receipt stderr captures are empty.

Core archivable evidence is listed, with sizes and SHA-256 values, in `audit_evidence_manifest.json`. Every entry is a text receipt, report, or auditor-written Python script. It excludes the private transport/credential helpers and all binary artifacts. The raw intake JSON files contain the source snapshots; included unpack helpers can materialize them. `independent_m0.stdout` preserves all 248 independent row results, including unfavorable values; no row was filtered.
