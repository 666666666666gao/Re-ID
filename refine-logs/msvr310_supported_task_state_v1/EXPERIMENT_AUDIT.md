# Experiment Audit Report — supported task-state V1 complete M0

Date: 2026-09-21. **Overall verdict: WARN. Integrity status: warn. Original engineering gates: PASS. Identified engineering blockers: none.**

Fresh native Codex review, requested `gpt-6-astra` / `max`. `review_independence: same-family`; `acceptance_status: provisional`; actual backend identity is **unattested**. Deterministic checks below are accepted only within their explicit saved-artifact scope. Earlier kernel/integration review verdicts were not read or used as proof of this M0.

Execution is bound to commit `cb4f4c37fbedd357ccc0ae4768161d049a280c5d`, config SHA-256 `4ebedda4a10d5e535d6820aa05684b7f707f7a0b8308d454b2316e9b3a61f3c3`, M0 summary `1385158d9aee97e04a7652e591a3c7b52a716f2954398481daa8a1bd4299af33`, and original M0_CPU receipt `c09ddf3854713fc451e3554a62df47ead8fe3338a4324ea6158dc5fa69ba4bdf`. Remote HEAD during this audit was `a8999c828b024cace9b84d32c28a7f7c5ba9dd9b`; that later HEAD does not replace the execution identity. Current bound execution-source/config hashes passed.

## Actual read and verification scope

All ten scripts specifically named in the request were read line by line. Called label loading, source-fold construction, objective, loss aggregation, historical replay, checkpoint, metric and gate definitions were inspected. All 28 intake text files were parsed and hashed. The audit independently rehashed all **45 original CPU-bound artifacts**, plus four run text/log files, and **120 distinct current source/config/dependency files** on the server. It loaded eight saved optimizer/Scaler payloads and six capacity role checkpoints on CPU, verified three frozen baseline checkpoint hashes, and reconstructed the six complete saved capacity model state hashes. No binaries were downloaded.

Source citations below resolve under `snapshots/` (short Python basenames mean `snapshots/tools/`). Result citations resolve under the immutable intake at `D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_task_state_m0_complete_20260921`. The plan/tracker snapshots are under `snapshots/refine-logs/msvr310_supported_task_state_v1/`.

The independent CPU payload read all **4,945,920 saved distance elements**, checked finite/nonnegative arrays, and independently recomputed fused hard/current, standard Smooth-AP, cross-scene Smooth-AP and weighted-loss arithmetic for every one of the **248 steps**. The role-distance arrays were read and checked for numeric validity; this is not a reconstruction from original image features. All 780 T0 sampler/queue metadata rows were replayed without images or models. Full original CPU receipt code was inspected; this audit's independent code and results are separate in `remote_readonly_check.py`, `remote_checks.json`, `local_checks.py`, `local_checks.json`, `reference_checks.json`, and `original_m0_gate_checks.json`.

The server checks used two CPU threads, hidden CUDA, zero model forwards, zero optimizer updates and zero image reads. They did not inspect private credential-helper contents or touch live Q1/repository/run files. Prior performance results and review verdicts were not used as integrity evidence for this run; the baseline checkpoint identity is used only to bind current initialization/frozen state. Original M0 and M0_CPU stage exits are both 0 (`pipeline_at_intake.json:26-64`); Q1 is incomplete in that captured pipeline (`:66-85`) and is outside this audit.

## A. Ground-truth provenance — PASS, with access-evidence boundary

The 1,032 training records exactly match the bound dataset label manifest (`c835d204…e3c8`) and frozen protocol (`4ff4c60b…7cb4`). Labels are parsed from dataset filenames, including identity, scene and camera, rather than derived from model outputs: `snapshots/tools/audit_vehicle_query_protocol_labels.py:13-18`; `snapshots/tools/build_msvr310_train_oof_protocol.py:12-55`. The label-only stratified round-robin fold construction was independently replayed; all source/held-out identity and record sets are disjoint and the complete distractor-bearing gallery/query metadata matches the registered protocol (`build_msvr310_train_oof_protocol.py:57-123`; `local_checks.json`).

The three registered source folds contain 672/683/709 records and 103/103/104 identities; their held-out galleries contain 360/349/323 records and 52/52/51 identities. The planned 210/207/183 query counts are protocol metadata, **not completed M0 retrieval evaluations** (`remote_checks.json`, `fold_scope`). Every sampled M0 and admitted historical record was checked against the correct source fold; masks use original identity/scene labels. The sampler preserves B64/K8 with replacement where needed (`train_msvr310_signal_oof.py:68-99`; `train_msvr_supported_task_state.py:89-118`).

M0 calls source records, source preflight and source reload checks; its held-out extraction/evaluation branch is skipped (`train_msvr_supported_task_state.py:392-438`). The summary reports held-out forwards and official image reads both zero (`m0/summary.json:26300-26301`). This is corroborated by the inspected execution path and complete recorded sample identities; it is not an independent OS-level file-access trace. The original label inventory includes official-split **metadata**, but this audit and M0 do not turn that into official image/model access.

Frozen Signal initialization is bound to the source-only B0 fold checkpoints, whose payload identities and hashes were independently checked. Model-building rejects mismatched folds and freezes baseline parameters (`train_msvr310_trifusion_oof.py:27-65`). The six saved complete model hashes reconstruct exactly; preflight/output parity and frozen-before/after assertions are original runtime witnesses. No new model forward was performed by this auditor.

## B. Score definitions and normalization — PASS

Cross-scene positives are same identity/different scene; same identity/same scene entries are ignored, and all different-identity entries remain valid, including same-scene negatives. AP is averaged over eligible current anchors only. Ineligible anchors store zero and a wholly unsupported batch returns a connected zero. Historical candidates are columns, not anchors. The score transform `1 - d²/2` uses unit-feature distance, fixed tau 0.01 and valid rank sums, with no denominator taken from prediction maxima/minima (`msvr_cross_scene_smooth_ap.py:29-54`; `msvr_instance_memory.py:40-80`).

The independently recomputed maximum fused-objective discrepancy is **1.03871775847e-07**, below the original 2e-6 bound; maximum weighted-loss discrepancy is **5.94804684617e-07**, below 1e-5. All 14 scalar objectives are present and their fixed weights reconstruct the logged total (`verify_msvr_supported_task_state.py:137-142`; `remote_checks.json`). These are training objective checks, not retrieval scores.

The original overfit excess-loss ratio uses the analytic label-smoothing CE entropy floor, not a model-dependent rescaling. Raw initial/final losses remain visible: both arms start at 4.122129917144775, the floor is 0.5857136327437849, and final losses are 0.5881898403167725 / 0.5882197022438049. Recomputed excess ratios are **0.000700202514028135 / 0.0007086466350339654**, both below the unchanged 0.1 gate (`run_signal_preserving_v5.py:1580-1618`; `m0/summary.json:16820-16829`, `:26278-26287`). Small ratios here indicate fitting the repeated source batch, not 99.9% retrieval performance.

Relative L2 gradient denominators are explicitly numerical reference checks, not reported performance. The Q1 retrieval routine is standard per-query AP/Rank-k with same-identity/same-scene removal and upstream metric parity (`train_msvr310_signal_oof.py:223-238`; `train_msvr310_trifusion_oof.py:196-241`); it did not produce M0 retrieval results.

## C. Artifact existence and exact correspondence — WARN for documentation only

The eight endpoint records total 248 steps, and standalone training/receipt JSON matches the embedded M0 summary. File size/hash bindings match the original complete CPU receipt; all optimizer files, model checkpoints and distance arrays exist remotely. The recorded original engineering gate booleans were recomputed from the evidence and all pass. Maximum reserved GPU memory in the original M0 records is 12,230 MiB; all eight endpoints report 203/203 nonzero trainable tensors and zero AMP overflows (`original_m0_gate_checks.json`; `local_checks.json`; `m0.log:25-194`).

| Endpoint | Steps | Actual unique records / identities | Steps with history | Original peak reserved MiB |
|---|---:|---:|---:|---:|
| fold_0_control | 8 | 333 / 63 | 5 | 12078 |
| fold_0_split | 8 | 333 / 63 | 5 | 12172 |
| fold_1_control | 8 | 316 / 61 | 5 | 12186 |
| fold_1_split | 8 | 316 / 61 | 5 | 12172 |
| fold_2_control | 8 | 350 / 62 | 5 | 12182 |
| fold_2_split | 8 | 350 / 62 | 5 | 12230 |
| overfit_control | 100 | 53 / 8 | 0 | 7248 |
| overfit_split | 100 | 53 / 8 | 0 | 7272 |

All four paired runs have identical initialization bindings and sampled indices/pixel hashes. The overfit pair repeats one fold-0 batch, containing 53 unique records of eight identities, for 100 steps. It has **zero historical candidates** because the repeated current records are excluded. Capacity has 30 historical steps in total, 3,448 candidate exposures and 5,760 original historical-VJP record forwards. These forwards were verified from records; the auditor did not execute them.

Five older local source files use CRLF while their registered remote source hashes are LF. Both raw local hashes and LF-normalized diagnostic hashes are recorded in `config_binding_checks.json`; actual remote bytes match the registered hashes. No runtime normalization or source editing was performed. The current ten task-state files/config/plan match exactly.

The tracker still contains a top historical `No model training` statement and `M0 NOT_RUN` row (`snapshots/refine-logs/msvr310_supported_task_state_v1/EXPERIMENT_TRACKER.md:3-14`) despite a dated terminal M0 milestone at `:43-45`. This is stale current-status presentation, not evidence that the result files are missing. Refresh its current summary while preserving the preregistration plan's bound original bytes. No unsupported Q1 or official numerical claim was found in the scoped task-state plan/tracker.

## D. Called paths and gradient/state semantics — PASS within witness scope

The wrapper's T0/M0/M0_CPU stages are actually called and receipt-backed (`run_msvr_supported_task_state.py:31-65`). M0 computes and records both standard and cross-scene AP diagnostic values; its active training objective is hard Triplet for the first two updates, then cross-scene AP (`train_msvr_supported_task_state.py:138-181`, `:307-327`). The inherited EMA-balancing controller and full `verify_balance` helper are not active; only role grouping, direct gradient helpers, support counting and scalar pair validation are reused. No claimed M0 metric depends on an uncalled retrieval function.

Current R is a direct derivative of the fused ranking scalar. A is differentiated directly from a separately assembled scalar with fused ranking multiplied by zero; it is **not** accepted as `total - ranking` (`train_msvr_supported_task_state.py:176-208`). Historical R uses a detached current-coordinate leaf derivative and replayed encoder VJP, added to current R before the optimizer update (`:158-167`, `:227-273`). Saved original role-entry CPU/CUDA RNG and frozen fields are reused (`msvr_freshness_probe.py:48-60`; `probe_msvr_history_candidate_gradients.py:43-50`); the trainer asserts exact reencoded coordinates, unchanged RNG/buffers and unchanged current gradient while replaying historical groups (`train_msvr_supported_task_state.py:231-249`). Original RNG bytes and per-step gradient vectors were not retained for fresh trajectory reconstruction.

Each capacity endpoint performs its one direct full-graph reference at step 4, the first single-group historical step. All six original full-reference comparisons meet 0.005 relative L2; maximum is **1.577269128216772e-5**. The 6 endpoints × 3 roles × 5 components give **90 direct component checks**, comprising current rank, historical rank, full rank, direct auxiliary and assembled gradient; maximum relative L2 is **3.12850281798043e-6**. Zero-reference handling uses absolute 1e-8. These references and tolerances are present and arithmetically consistent (`train_msvr_supported_task_state.py:185-216`, `:259-286`; `msvr_supported_gradient_balance.py:53-62`; `reference_checks.json`). They are original model-runtime witnesses; this audit validates the records, not 90 new autodiff executions. Multi-group later steps have coordinate/VJP witnesses, not a fresh full-graph reference at every step.

Task buffers are explicitly divided by AMP scale; combined parameter gradients are unscaled before checks; every task buffer and every parameter gradient is validated before any moment/parameter mutation (`train_msvr_supported_task_state.py:287-300`; `msvr_task_state_optimizer.py:35-48`). Split uses one rank and one auxiliary clock/moment pair. A supported zero R remains an observation. Unsupported R neither advances its task clock/moments nor applies its old momentum; A continues. Shared roles use R+A in common state, and 14 head/neck tensors use the original current total gradient. Decoupled weight decay is applied once after the chosen direction (`msvr_task_state_optimizer.py:22-64`).

All eight saved optimizer states were independently loaded with `map_location=cpu`, without model construction. The 189 role parameters partition into CNN/Transformer/Mamba **42/54/93**, plus 14 head/neck tensors. Control has 406 moment tensors per endpoint; split has 784; **4,760** tensors were checked for FP32 dtype, actual model-parameter shape, finiteness, nonnegative second moments, and clock consistency. Final role moment norms match final step records to the original 1e-9 relative/absolute scale, with maximum observed absolute discrepancy **2.77555756156e-17**. Parameter groups are lr 0.00035, decay 0.0001, beta (0.9, 0.999), eps 1e-8. All Scalers have scale 256, growth factor 2, backoff 0.5, interval 2000 and growth tracker 8/100 as appropriate. Shared/head/A clocks are 8/100; all split R clocks are also 8/100 because all real M0 steps are observed (`remote_checks.json`; `verify_msvr_task_state_records.py:58-90`).

Original disk save/reload compares each final moment exactly and checks Scaler/parameter groups (`msvr_task_state_records.py:23-54`). This audit independently checks saved state structure and final-record correspondence. It does **not** claim a resumed real-model training test or a reconstruction of intermediate moments. Overfit models have no separate final model checkpoint; their optimizer states are saved, but final model-state assertions remain runtime witnesses.

## E. Scope and causal interpretation — WARN

This is one dataset, seed 42, three source fold pairs at eight capacity steps plus one source fixed-batch pair at 100 steps. It validates the original engineering gate, with one direct-reference step per capacity endpoint. All **248 actual M0 rank observations are supported**; actual unsupported rank steps are **zero**. Synthetic task-state checks explicitly cover absent steps 1/5/6/11 and an observed-zero step 8, native-AdamW comparison, CPU Scaler kwargs, finite rejection and state_dict continuation (`check_msvr_task_state_math.py:24-115`; `t0.json:71-125`). Their toy parameter updates must not be described as actual unsupported source-batch validation. T0 synthetic results were read and bound; this audit did not rerun optimizer updates.

The registered treatment starts in hard-Triplet warmup and retains its moments at AP activation; no clock reset occurs. Q1's registered 65-step warmup is separate from M0's two-step warmup (`EXPERIMENT_PLAN.md:13-16`; config memory block; trainer `:92`, `:156`, `:269`). The original two groups of five scientific gates are preserved through the reused paired-summary path and independent verifier (`train_msvr_supported_task_state.py:359-364`; `train_msvr_instance_memory.py:250-257`; `train_msvr310_source_style.py:196-220`; `verify_msvr_supported_task_state.py:294-338`), but those gates have no M0 performance result.

For actual first capacity updates, split/control role update-norm ratios span **1.6469892151–1.7242741446** (`local_checks.json`, matched pairs). Summing independent preconditioned directions changes effective update magnitude; warmup is also treated. The plan already registers the comparison as an entire optimizer rule, not isolated state-history causation (`EXPERIMENT_PLAN.md:5`, `:13-16`, `:29`).

`actual_parameter_updates` records norms before/after the update, their difference norm and the cosine between the two parameter vectors (`train_msvr_supported_task_state.py:290-304`). It does **not** preserve per-task update vectors, cosine of the update with R/A, or cross-arm update directions. Scalar moment norms cannot recover these. Thus the plan's phrase about update “directions” must be qualified; the vector-level mechanism and the full training trajectory remain unavailable.

Matched initialization/pixels also do not imply bitwise gradient identity. The first-step Mamba scalar gradient summaries differ across arms; the largest observed absolute scalar difference is **4.0325888398760066e-7** (fold 1). CNN/Transformer summaries match exactly. The cause was not diagnosed, no model replay was run, and cross-arm gradient-vector equality is not established. This is not a failed registered engineering gate.

No held-out retrieval gain, Q1 completion, official performance, generalization, multiseed robustness or three-dataset success follows. A later complete Q1 needs its own fixed-endpoint evidence, scientific gates and full audit.

## F. Evaluation classification — PASS

- **real_gt, source-supervised engineering:** actual capacity and overfit model checks use dataset identity/scene labels. Claim ceiling here remains engineering feasibility, because held-out retrieval is not evaluated in M0.
- **simulation_only / synthetic arithmetic:** T0 toy optimizers, masks, AP references and finite differences. These are explicitly synthetic and do not report model quality.
- **synthetic_proxy, parity only:** standalone Signal and zero-update/reload feature comparisons use model outputs as references to test equivalence. They are labeled engineering parity, not ground-truth retrieval performance.

## Claim impact and actions

Accept the qualified statement: **the exact registered M0 completed and passed its original engineering gates, with complete original CPU verification and fresh independent saved-artifact checks.** Source provenance, candidate masks, matched initialization/source pixels, direct R/A construction, persisted final task state and the original finite/decay/head rules are supported at the evidence levels above.

Block claims of real unsupported-batch M0 coverage, reconstructed full parameter-gradient/update trajectory, isolated state-history benefit, completed Q1 retrieval gains or official performance. There is no new experimental fix or rerun requested by this audit. Retain the limitations in reporting, refresh the stale tracker current-status presentation, and audit the full Q1 only on its complete original endpoint/CPU evidence.

## Audit execution transparency

The first remote checker passed its arithmetic/structure assertions but had a reporting-only loop-variable bug that labeled endpoint rows with the final parameter name. `remote_readonly_check_r1.py` and `remote_readonly_check.stdout` preserve that run. The corrected checker was independently rerun with the same read-only scope; **use `remote_checks.json` / `remote_readonly_check_r2.stdout` as authoritative**, which explicitly binds all eight endpoint names/directories. The two executions took 15.61 and 15.26 seconds.

An additional local exploratory assertion demanded bitwise equality of first-step gradient summaries. It failed for Mamba, was not part of the registered gates, and has been retained as an observed limitation, not hidden or reclassified as an experimental pass. `local_checks_r1.py` preserves that attempted assertion; `local_checks.py` measures and reports the differences. No sealed experimental file or gate was changed.

All input hashes are in `EXPERIMENT_AUDIT.json`; source snapshots and check code are retained. Semantic review remains same-family/provisional/backend unattested. This audit is an evidence audit, not a guarantee against every possible integrity failure.
