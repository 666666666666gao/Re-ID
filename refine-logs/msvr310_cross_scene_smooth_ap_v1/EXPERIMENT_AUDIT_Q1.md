# MSVR310 cross-scene Smooth-AP complete Q1 integrity audit

**Overall verdict: WARN / CLOSED_WITH_LIMITS. Deterministic evidence checks: PASS. Scientific qualification: FAIL under the unchanged registered gates.**

Date: 2026-09-21. Fresh reviewer: `/root/audit_cross_scene_smooth_ap_q1_20260921`; native request `model=gpt-6-astra`, `reasoning_effort=max`, `fork_turns=none`. Review independence is **same-family**, acceptance is **provisional**, and backend model identity is not independently attested. The native request and metadata were read directly. This is the reviewer's own assessment of primary files, not the executor's verdict.

The full three-fold, two-endpoint, fixed-epoch-20 experiment exists and its saved rankings, masks, objectives, checkpoint states and original gates agree with independent reconstruction. The candidate's fused mAP improves by **0.5671825364 percentage points** over its matched control, below the registered 1-point requirement. Its identity-bootstrap lower bound is **−0.0772393104 points**. The paired gates pass **2/5**, and the candidate-versus-Signal gates pass **1/5**. No threshold was modified, no favorable fold was substituted, and no failed-version training was rescued.

The warning preserves three material boundaries: saved gradient/runtime assertions do not reconstruct a training trajectory; the paired common-objective warmup is not bitwise identical; and this is one seed on repeatedly used internal development folds. The initial status-document lag was resolved by the executor during the audit and is explicitly separated from the final publication state below. No blocking fabrication, missing result, score-normalization or source/heldout leakage defect was found within the audited evidence.

## Scope and reproducibility

Execution commit: `d35864d6411591e05c8ac3e5164ebae48063ad99`. Config SHA-256: `e5326b52ebb12dced24ebfac788db2e2bb5bdca0b50c6c4dec64e596f1af05a1`.

Remote run: `/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d`. Repository references below are relative to `/root/autodl-tmp/trifusion-v2/TriFusion-ReID`, with equivalent local source in `C:/Users/gb/.trifusion_github_publish_22c3bee`. Remote source snapshots and their exact original hashes are mapped in `remote_source_manifest.json`; normalized text-snapshot hashes are explicitly separate from original-file hashes.

Independent scripts import no project objectives or verifiers. NumPy/stdlib rebuild objective and ranking arithmetic; PyTorch is used only for CPU tensor deserialization and the registered 56-thread feature-to-distance arithmetic. There were **0 model forwards, 0 optimizer updates, 0 official-test dataset reads, 0 environment installs**, and no repository or original-evidence edits. Arrays, images and checkpoints stayed remote. The separate M0 audit and completed source census were not rerun; only their saved bindings and the obligations relevant to Q1 were inspected.

| Check | Verdict | Main finding |
|---|---|---|
| A. Ground-truth provenance | PASS | Dataset filename labels, complete source/heldout identity isolation, full galleries and actual scene filtering. |
| B. Score and objective normalization | PASS | Raw AP/CMC and fixed Smooth-AP definitions reproduce; eligible-anchor reduction is explicit. |
| C. Result existence and provenance | PASS | Complete immutable artifacts and numerical claims pass; the refreshed terminal tracker is correct, with a minor stale AGENTS heading. |
| D. Live objective/evaluation paths | WARN | Relevant paths are called; current/history gradient reconstruction remains a runtime-witness boundary. |
| E. Scope and claims | WARN | Full registered scope is covered, but one seed, adaptive development and pre-intervention numerical divergence limit scientific inference. |
| F. Evaluation classification | PASS | Q1 is `real_gt` internal identity OOF evaluation; T0 numerical fixtures are `simulation_only`. |

## A. Ground-truth provenance and complete path isolation — PASS

The provenance chain is dataset filename → MSVR310 identity/camera/scene parser → frozen label manifest → deterministic training-only identity folds → source-index loader → current/history labels and heldout scene-filtered retrieval. Targets are not derived from predictions. The upstream loader reads identity from `img[0:4]`, camera from `img[11]`, and scene from `img[6:9]`; the actual upstream evaluator removes **same identity AND same scene**, despite inherited person-ReID wording in its comments.

All **1,032 training triplets / 3,096 modality filenames** exist and agree with the frozen manifest. The audit inspected filenames and existing installation/label receipts, not image contents or current official query/gallery directories. The 155 identities comprise 60 multi-scene query identities and 95 single-scene gallery-only distractor identities. Every heldout record remains in its fold's gallery; only query rows lacking a legal different-scene positive are excluded from the denominator.

| Fold | Source IDs / records | Heldout IDs / gallery | Legal queries | Query-ineligible records retained | Gallery-only IDs |
|---|---:|---:|---:|---:|---:|
| 0 | 103 / 672 | 52 / 360 | 210 | 150 | 32 |
| 1 | 103 / 683 | 52 / 349 | 207 | 142 | 32 |
| 2 | 104 / 709 | 51 / 323 | 183 | 140 | 31 |
| Total heldout union | — | 155 / 1,032 | 600 | 432 | 95 |

All 1,950 saved prerequisite B0 source-training steps and all 1,560 Q1 source steps belong to their own fold's source indices. All three B0 checkpoint payloads bind the corresponding source and heldout identity lists. The six Q1 endpoints load the appropriate fixed epoch-50 B0 checkpoint and fresh seed-42 roles, with `role_weights_loaded=false`; they do not load prior candidate weights or M0 terminal states. Q1 initial bindings match the saved M0 initial bindings, and the Q1→M0 summary/CPU hashes match. Role training and retrieval occur within each fold; no cross-fold feature-distance comparison is used.

Evidence: `tools/audit_vehicle_query_protocol_labels.py:13-18`; `tools/build_msvr310_train_oof_protocol.py:12-123,143-152`; `tools/train_msvr310_signal_oof.py:31-76,89-110`; `tools/train_msvr310_trifusion_oof.py:27-55,196-241`; upstream `comparators/Signal-cd1b0a6/data/datasets/msvr310.py:67-88` and `utils/metrics.py:53-69`. Independent results: `AUDIT_EVIDENCE.json:33`, `:68`, `:148`; raw `05_remote_provenance.stdout`, `10_remote_receipt_links.stdout`.

## B. Raw scores, target masks and all-step objective arithmetic — PASS

Retrieval uses conventional L2-normalized features and squared Euclidean distance, followed by unscaled AP/CMC reported as percentages. No metric denominator uses the model's own maximum, minimum or mean prediction. The architecture's baseline-norm matching is part of feature construction, not a post hoc metric rescaling. All **30 fold/endpoint/output distance matrices**, **2,069,520 distance/ranking elements**, and **6,000 query/output rows** were checked. Recomputing distances from the saved feature tensors reproduces every saved distance bitwise under the registered arithmetic; complete `argsort` permutations reproduce, and independent positive-hit rank arithmetic matches AP to at most `4.4408920985e-16`.

Training uses `s=1-d²/2`, fixed temperature 0.01 and fused weight 1. The standard endpoint excludes only the current self position from its all-same-identity positives. The candidate endpoint uses same-identity/different-scene positives, ignores every same-ID/same-scene position in both rank sums, and retains all different-identity negatives regardless of scene. Duplicate current views remain distinct positions where the contract permits them; historical records are unique, at most 8 steps old, capacity 512, and exclude all currently sampled record IDs. Self and each positive's own comparison are correctly excluded.

The candidate loss is `1 - mean(AP over eligible current anchors)`. Ineligible per-anchor stored AP zeros are excluded from this mean. Those samples remain under the other 13 supervision terms and remain negative candidates for other identities. An all-ineligible batch returns a graph-connected exact zero rather than changing the sampling or substituting positives. Independent integer-mask checks cover every step, including retained same-scene negatives and the ineligible current samples' continued presence as negative candidates.

All **116,501,504 saved training distances** were consumed. This reconstructs all three fused objective definitions at every step, three branch current hard-triplet terms, queue membership/ages/masks, selected objectives, 14-term weighted ledgers, all 120 epoch means and the fixed learning-rate schedule. The other ten components' logits/residual distances are not saved, so their scalar values are ledger inputs, not independently regenerated loss values.

| Independently checked quantity | Maximum absolute difference | Required tolerance |
|---|---:|---:|
| Standard per-anchor Smooth-AP | 2.9494505660e-7 | 2e-6 |
| Cross-scene per-anchor Smooth-AP | 6.6910120189e-7 | 2e-6 |
| Cross-scene scalar / selected fused objective | 1.6373415901e-7 | 2e-6 |
| Pooled hard-triplet scalar | 3.7997961055e-8 | 2e-6 |
| Branch current hard-triplet scalars | 1.0961666702e-7 | 2e-6 |
| 14-term weighted total | 4.4951836298e-7 | 1e-5 |
| Epoch mean / learning rate | 0 | 1e-12 / 1e-15 |

The first 65 updates use the common current-batch hard objective. Update 66 switches to each endpoint's AP definition and starts enqueueing; historical candidates first participate at update 67. Every endpoint has 260 steps / 20 epochs with 13 steps per epoch. The AdamW learning-rate schedule is the registered five-epoch warmup followed by the original cosine expression, not an inferred replacement schedule.

Per endpoint, post-warmup eligible exposures are **5,144 / 5,176 / 5,056**, totaling **15,376 / 37,440 = 41.0684%**. Actual candidate zero-eligible active batches occur at fold 0 step 180, fold 1 step 221, and fold 2 steps 133 and 232. In each, the selected fused scalar and saved historical upstream norms are zero, no historical VJP group is run, the other supervised total is positive, and the registered update sequence continues. All nine zero-support step positions per endpoint are retained, including five warmup positions; across both endpoints there are 18 rows, of which 8 are after warmup. Only four are active zero-objective candidate rows.

Evidence: `tools/msvr_cross_scene_smooth_ap.py:11-62`; `tools/msvr_smooth_ap.py:30-62`; `tools/msvr_instance_memory.py:8-37`; `tools/train_msvr_cross_scene_smooth_ap.py:95-160,243-270`; `tools/run_signal_preserving_v5.py:99-135,1621-1627`; `modeling/trifusion/signal_preserving_v8.py:483-517,690-742`. Independent results: `AUDIT_EVIDENCE.json:201,215,241,369,789`; `07_remote_arithmetic.py:36`, `09_remote_final_checks.py:25`; complete step and zero-row CSVs.

## C. Existence, provenance, completion and report arithmetic — PASS

Original Q1 completed at **2026-09-21 04:19:46 +08**, Q1_CPU at **04:20:11 +08**, both exit 0. The pipeline is `COMPLETE_VERIFIED_Q1_FAIL`. Original wrapper and all five stage PIDs are absent at the final observation. All **119 original run files** have identical size and SHA-256 between the first and final audit intakes; the complete text intake matches current remote originals for all 30 corresponding files. The local intake embeds the terminal pipeline snapshot in `inventory.json`; it need not contain a standalone `pipeline.json`, and the auditor obtained the original pipeline directly.

The registered recursive chain has **134 exact hash bindings**, with **88 repository file bytes** equal to execution commit `d35864d`. The separately followed project-source closure has **66/66** current files matching that execution commit. Of these 66, 54 local sources match the remote bytes exactly and 12 differ only by CRLF/LF; all normalized text agrees. These comparisons do not substitute a new code version or silently rewrite pinned bytes. A CRLF-bearing remote source caused an audit text-snapshot SHA assertion to fail; the corrected forensic mapping records original-source and LF-text hashes separately, retaining the failed attempt.

All six Q1 role checkpoint file hashes and payloads were independently checked. Each final 472-tensor state reconstructs from **241 B0 aliases + 231 saved role-state tensors**, equals the reported final/strict-reload hash, and preserves the frozen B0 state and seven zero neck biases. This checks stored state content independently. Original model forward outputs and per-step frozen-state trajectories remain runtime evidence, as specified in D.

The Q1 summary's `project_commit=fdc6305...` is the worktree HEAD observed when Q1 started, while the wrapper pins `d35864d...`. The audited configuration and source bytes match the execution commit; this is a documented HEAD-versus-execution distinction, not evidence of an algorithm change.

All v2 executor report tables, all five CSVs, all 24 source-log phase descriptions, and associated means/distributions/costs were independently checked: **57,813 scalar comparisons**, maximum difference `4.4408920985e-14`. The CSV checks cover 3,000 paired query-output rows, 60 wide identity rows, 30 fold-output rows, all 120 epochs and all 1,560 full component rows. Separately, all 600 endpoint/identity/output means and eight endpoint-versus-Signal query-change groups in the original summary agree. The v2 report correctly distinguishes matched-control changes from each endpoint's changes versus Signal.

Document timing is explicit. The initial local snapshots, captured at **2026-09-21 04:32:33 +08**, are `snapshots/0016_EXPERIMENT_TRACKER.md:10-13` and `snapshots/0009_AGENTS.md:3-5`; those snapshots contain Q1 RUNNING / CPU NOT_STARTED. They are retained as historical evidence. The auditor subsequently read the current publication at **HEAD 17bba30a7a832e9667172f6aa4ebd882d25febc0**: `document_refresh/EXPERIMENT_TRACKER.md:10-14` correctly records Q1_FAIL / AUDIT_PENDING, Q1_CPU PASS, and Q1_Audit RUNNING pending this verdict; `document_refresh/AGENTS.md:5` correctly records terminal Q1_FAIL. Their current hashes match the supplied synchronization receipt checked at **04:35:51 +08**. Only the AGENTS heading at line 3 still says Q1 running; this is a minor editorial inconsistency, not an incorrect terminal result or an integrity blocker. `DOCUMENT_STATE_REFRESH.json` records the exact observation, snapshots, hashes and remote-receipt basis. Record audit closure after delivery and preserve the frozen registered plan bytes.

Primary hashes:

- Q1 summary: `3aa13b43b4a3c6d665486077dcd64daf0b9693a78aae75b3c92de04c3dd0879f`.
- Q1 CPU receipt: `905d44f4eaea5f5a9c43fc2c5029af22c896f33d50cfb51c252031bf81dd9fff`.
- M0 summary link: `c59e39cc2192bfb11b12a58f53eb8b3853455bfd41744c74aaf4742272ba1506`.
- M0 CPU link: `80f52c03337e0c497efe25f9a888de9dae299717733f0de19114ee90052aa1d8`.

All **448 audited input hashes**, including remote binary identities without binary downloads, are in `audited_input_hashes.json`. Evidence: `tools/run_msvr_cross_scene_smooth_ap.py:31-70`; `tools/train_msvr_cross_scene_smooth_ap.py:302-388`; `tools/train_msvr310_source_style.py:164-173`; `tools/train_msvr_instance_memory.py:94-110`; `AUDIT_EVIDENCE.json:68,148,857,978,982`.

## D. Live paths, current/history derivatives and runtime limits — WARN

There is no dead relevant objective or evaluation path. The trainer calls the shared objective adapter for the current graph at line 129, historical leaf partials at line 154, and the M0 direct full graph at line 176. Current-to-current distance uses both occurrences of the differentiable current embeddings. Historical refresh re-encodes saved frozen source fields at current role parameters with the original role-entry CPU/CUDA RNG. The leaf computation fixes current coordinates only for the historical partial; the original current graph still receives its own anchor and candidate derivatives. Historical VJPs over the selected encoder parameters are added to the current gradients before exactly one optimizer-step call.

The same endpoint objective is selected in all three sites. The fusion has no trainable parameters, so the historical candidate derivative correctly targets the role encoder. The other 13 supervision definitions and weights are unchanged, although their numerical values naturally differ after training trajectories diverge. Historical records are candidates, never anchors: 64 current anchors and 0 historical anchors per step.

`fork_rng` restores CPU/CUDA RNG after replay. Buffer checks surround no-grad refresh and historical VJPs; the code also asserts re-encoded coordinate equality, preservation of current `.grad`, and exact final gradient addition. All 1,560 saved norm/decomposition rows are finite and internally consistent; maximum norm-identity discrepancy is `2.1316282073e-14`. Those facts verify the recorded scalar ledger, not independently reproduced tensors.

**No Q1 direct full-graph comparison was executed:** Q1 `extra_direct_check_record_forwards=0` in all six endpoints, consistent with the `mode=='capacity'` guard. The prior M0 audit records six single-history-group direct/VJP witnesses. Those are bounded M0 witnesses and cannot be promoted to Q1 all-step gradient reconstruction. The Q1 **203/203 nonzero-gradient tensors** value is a cumulative union over each endpoint, not an every-step nonzero guarantee. Per-step parameter vectors, logits, exact original frozen fields and complete RNG/optimizer trajectories were not persisted for a new full backward replay. This audit generated none.

The gradient summaries labeled `total_vs_history` compare current-path gradients of the active total loss with the additional historical ranking contribution. They do not isolate ranking loss F versus the other supervision O, and endpoint ratios cannot identify a causal loss multiplier. Ineligible anchors can still receive feature gradients when they occur as negative candidates for eligible queries; a zero anchor row is not proof of zero parameter influence everywhere.

Evidence: `tools/train_msvr_cross_scene_smooth_ap.py:42-52,120-185,190-264,273-285`; `tools/msvr_role_set_relations.py:60-64`; `tools/msvr_freshness_probe.py:43-60`; `tools/probe_msvr_role_set_gradients.py:29-50`; `tools/probe_msvr_history_candidate_gradients.py:43-66`; prior `EXPERIMENT_AUDIT_M0.md:68-74`. Independent results: `AUDIT_EVIDENCE.json:215,241`; raw `07_remote_arithmetic.stdout`.

## E. Complete scientific scope, original gates and cost — WARN

This is one MSVR310 training-internal identity OOF study, seed 42, three folds, two independently initialized endpoint executions per fold, fixed epoch 20. Each endpoint is evaluated on the complete 1,032-record heldout-gallery union and all 600 legal queries from 60 query identities. No per-epoch retrieval result was generated; the 120 epoch records are source training summaries. The same internal protocol has informed preceding development, so this is not an untouched confirmatory external holdout. The fixed-seed identity bootstrap samples whole identities 10,000 times with replacement, retains within-identity query weighting, and uses the 2.5th percentile with linear interpolation. It measures conditional query/identity resampling uncertainty for these saved models, not training-seed uncertainty or correction for adaptive method development.

| Output | Control mAP | Candidate mAP | Paired gain pp | Control R1 | Candidate R1 |
|---|---:|---:|---:|---:|---:|
| Signal / baseline_only | 53.129381 | 53.129381 | 0.000000 | 63.000000 | 63.000000 |
| fused | 52.838309 | 53.405492 | +0.567183 | 61.166667 | 62.000000 |
| cnn | 50.608598 | 50.343302 | −0.265296 | 60.000000 | 59.333333 |
| transformer | 51.555161 | 51.417266 | −0.137895 | 60.666667 | 60.166667 |
| mamba | 51.403658 | 51.997190 | +0.593532 | 60.000000 | 61.000000 |

The candidate fused mAP exceeds Signal by **0.2761112172 pp**, but its Rank-1 is **1 pp below Signal**. “Fused strictly best” is the registered mAP gate, not a claim of superiority on every retrieval metric.

| Original condition | Matched-control comparison | Candidate versus Signal |
|---|---|---|
| Fused gain ≥1 pp | FAIL: +0.567183 | FAIL: +0.276111 |
| Every fold fused gain ≥0 | PASS: +1.335700 / +0.026182 / +0.297229 | FAIL: +4.118151 / −3.059038 / −0.360241 |
| All three role gains ≥0 | FAIL: CNN and Transformer decline | FAIL: all three are below Signal |
| Identity bootstrap lower >0 | FAIL: −0.077239 | FAIL: −1.805773 |
| Candidate fused strictly best by mAP | PASS | PASS |

Fused improves AP for 299 queries, declines for 218 and is unchanged for 83; it repairs 11 Rank-1 errors and introduces 6. Its per-identity gains are positive/negative/zero for 28/26/6 identities. Every identity and unfavorable row remains in the artifacts.

There is an additional measured qualification: despite matching initial bindings, sampled records and pixel hashes, the first 65 common-hard warmup losses differ. Maximum paired total-loss differences by fold are **0.001278042793 / 0.001622676849 / 0.002021312714**, starting at steps **3 / 2 / 3**; 63/64/63 warmup steps differ. The audit does not assign an unverified kernel-level cause. It is incorrect to describe the two full training trajectories as bitwise paired or to attribute the entire endpoint gap to a numerically exact counterfactual intervention. Update 66 already uses different objectives; the 66-step “before first history” window must not be described as all common-objective warmup.

The last 65 source steps show lower same-definition candidate diagnostics for both standard AP loss (0.006668543→0.005573188) and cross-scene AP loss (0.029118386→0.014568746), while current hard loss increases (0.062468318→0.064545732) and expanded hard loss increases (0.158681301→0.161146010). These are repeated minibatch/pool observations, not complete source-gallery or official retrieval metrics. Active total losses contain different objectives and cannot be compared as the same objective value. Neither these diagnostics nor partial fold signs justify loss-weight changes or promotion.

Cost accounting is explicit. Q1 records 1,560 optimizer updates and 99,840 current anchor exposures over 1,032 unique source records across folds. Extra fresh-role record computations total 585,600 and historical VJP record computations 583,168, giving 1,268,608 current-plus-extra role record computations. Historical VJP counts differ between control/candidate: **292,608 / 290,560**, because zero historical upstream groups are skipped. Equal updates and data sequences therefore do not mean equal computation. The source fit-epoch time is **6,664.869 / 6,565.623 s**, and total Q1 elapsed time is **13,364.189 s**. These single-run times are descriptive, not a general efficiency result. Required B0's 1,950 prior updates and M0's 248 updates are separate prerequisite costs, not newly executed here.

The original run stores **1,182,950,157 bytes**: Q1 959,870,022, M0 219,430,041 and root text/logs 3,650,094. This is within the registered additional 3-GiB ceiling; final observed free space is 5,640,347,648 bytes. Peak recorded allocated GPU memory is 6,120.097 MiB. No artifact deletion was needed.

Evidence: registered `EXPERIMENT_PLAN.md:24-32,38-50`; `tools/train_msvr310_source_style.py:196-221`; `tools/train_msvr310_trifusion_oof.py:244-284`; `modeling/trifusion/signal_preserving_v13.py:253-279`; `AUDIT_EVIDENCE.json:587,658,768,775,780,781,857,959,980`.

## F. Evaluation type — PASS / real_gt

Q1 retrieval is **real_gt, training-internal identity OOF, complete legal-query/full-gallery evaluation**, not official testing. Training Smooth-AP uses real dataset identity/scene labels and is a surrogate optimization objective; it is not a model-generated ground-truth proxy or a substitute official mAP. T0 synthetic tensors/finite-difference fixtures are **simulation_only numerical tests**, and the registered source-support counts are label-derived repeated-exposure diagnostics. Their scopes remain separate.

Evidence: `protocols/msvr310_train_oof_v1.json` evaluation contract; `tools/build_msvr310_train_oof_protocol.py:124-168`; `tools/check_msvr_cross_scene_smooth_ap_math.py:33-91`; `tools/check_msvr_cross_scene_smooth_ap.py:15-70`; `tools/train_msvr_cross_scene_smooth_ap.py:311-319`.

## Failed checks and audit provenance

All failed helper/check attempts are retained in `AUDIT_ATTEMPTS.json`, with scripts and stdout/stderr. Attempts 01 and 02 failed to follow two explicitly named recursive configuration links. Attempt 06's extra bitwise warmup assertion failed on real data and is retained as a warning, not omitted; attempt 07 reports the complete discrepancy without changing a registered gate. Attempt 11 confused normalized text-snapshot bytes with original CRLF-bearing source bytes; attempt 12 preserves both hash meanings in a new snapshot directory. The executor's failed report v1 used a gallery position to index a query-only ranking list; its failure receipt/code are copied, and every corrected v2 CSV row was checked independently. None of these actions reran or repaired training.

## Claim impact, action items and blockers

- **Supported:** complete fixed-epoch Q1 completion; all saved distance/mask/objective/ranking arithmetic; exact checkpoint contents and baseline retrieval parity; the observed aggregate +0.567183 pp matched mAP gain with its negative identity-bootstrap lower bound.
- **Requires the stated qualification:** current-coordinate historical derivatives, per-step frozen/RNG assertions, strict reload forwards and cumulative gradient coverage; one-seed OOF and adaptive-development scope; measured input pairing with non-bitwise warmup.
- **Unsupported:** all-step independent parameter-gradient reconstruction; every-step 203/203 nonzero gradients; guaranteed unseen-distribution or multi-seed improvement; causal loss multipliers from gradient ratios; all-role improvement; official/SOTA success; inference that the unchanged original gates passed.

Record the review and finish the mutable tracker/AGENTS/result handoff with **Q1_FAIL / audit WARN-CLOSED_WITH_LIMITS**, including correction of the AGENTS line-3 heading. The current terminal Q1/CPU status is already correct; leave frozen protocol/configuration bytes unchanged. Archive the scripts, failed attempts, complete CSVs and hashes together. Keep all warnings visible in the scientific report and do not convert CPU reconstruction into a model-training replay claim.

There is **no unresolved deterministic integrity blocker to archiving this terminal result**. There is a **scientific promotion blocker**: the registered paired and Signal conditions fail. Official-test evaluation, failed-family tuning/retraining and pre-success ablations are not authorized by this audit. Any successor requires its own evidence-grounded preregistration; this report does not prescribe or authorize a new experiment.
