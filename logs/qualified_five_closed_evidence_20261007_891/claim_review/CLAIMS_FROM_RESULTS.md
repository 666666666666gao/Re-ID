# Result-to-claim review: qualified five endpoints, 2026-10-07

**Verdict: C1 no; C2 no; C3 yes within the documented execution scope; C4 no. The broad three-dataset/SOTA goal remains ACTIVE_UNMET.** The five selected endpoints and all twelve registered comparisons miss the project advance line. Small positive observations are retained below; they do not establish the intended stable, meaningful three-dataset gains. This is a completed negative efficacy result for the tested objectives, not evidence that the experiments were never run.

Reviewer requested model: `gpt-6-astra`; requested reasoning effort: `max`. Review context: fresh delegated reviewer `/root/claim_five891`. `review_independence: same-family`; `acceptance_status: provisional`; `runtime_attestation: UNATTESTED`. Actual backend/model/effort are not independently attested. No additional reviewer or external model was invoked by this reviewer.

**[INTEGRITY: WARN]** The completed audit was received and inspected before finalization: `integrity_status: warn`, `verdict: WARN`, `blocking_count: 0`, `warning_count: 3`. Its author is `/root/audit_five891`, with `review_independence: same-family`, `acceptance_status: provisional`, and audit `runtime_attestation: unavailable`. The three warnings retain consumed-official/single-seed selection, the missing RGBNT100 repair endpoint, and transcript-backed assurance limits. It found no blocking defect in the expressly qualified five-endpoint record. Remote binaries, physical image inventory and actual model runtime were not independently replayed. Its local deterministic PASS is accepted only for exact byte/text checks. The canonical evidence helper remains `UNRESOLVED`; neither those deterministic checks nor this WARN promotes the semantic claims to accepted. The audit arrived after the independent semantic conclusions above were formed and did not change them. [S13]

## What was tested and actually implemented

The comparison adds one training objective to the existing semantic model, preserving its parameter set, public initialization policy, author global/role tasks, seed 42, fifty epochs and L2-normalized 1536-dimensional deployment. Here, “unchanged parameters” means no added trainable parameter set, not unchanged learned tensor values. The independent `raw_global_only` control omits semantic roles and is therefore a distinct-capacity reference; the matched `raw_semantic` comparison isolates the training-objective change more directly. The full model's best checkpoint was selected on the consumed official development benchmark, and metrics must remain attached to that checkpoint. [S1], [S2], [S3]

The MD-inspired adaptation uses the batch-wide largest same-label distance and smallest different-label distance. Its loss is the fused positive-distance ratio plus one minus the fused negative-distance ratio, each denominator containing fused, detached-global and detached-correction distances. These are batch extrema, not per-query legal cross-environment relations. The adaptation is not a reproduction of the complete MDReID model, and this review makes no formula-novelty judgment. [S4] (lines 18–32)

The literal repair/keep implementation normalizes global and fused features, defines a legal positive as the same identity in another protocol environment, and combines that positive with every different-identity negative. The environment is the scene for MSVR310 and camera for RGBNT201/RGBNT100. With cosine margins `m_g` and `m_f`, it selects repair relations using detached `m_g <= 0`, then minimizes `relu(0.1 - m_f)`; this is an **absolute positive 0.1 target**. It is not a relative-improvement expression such as `relu(m_g + 0.1 - m_f)`; earlier candidate relative-margin formulas must not be attributed to this run. On `m_g > 0`, keep minimizes `relu(m_g - m_f)`. It averages within supported queries in each cell and gives repair and keep equal 0.5 weights. The detached reference is the current model's global feature, whose author global task still trains; it is not a frozen independently trained global-only baseline. [S4] (lines 35–69); [S5] (lines 20–32); [S6] (lines 62–81 and 101–111)

The checked driver changes observation of the isolated M0 VJP, disabling autocast for that measurement; non-M0 training calls the original training objective. No regional deletion-responsibility supervision or complete regional P1/P2/P3 mechanism is tested. A training keep penalty is not a guarantee of retaining held-out retrieval rankings. [S7] (lines 22–44)

## All five received endpoints

Numbers are percentages; deltas below are percentage points. These best-checkpoint metrics are reported together, without selecting a different epoch for each CMC rank.

| Dataset | Objective | Best epoch | mAP | R1 | R5 | R10 | E50 mAP | E50 R1 | Best-to-E50 mAP drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RGBNT201 | MD ratio | 8 | 74.469385 | 79.186600 | 88.397127 | 91.985649 | 72.009691 | 74.521530 | 2.459694 |
| RGBNT201 | repair/keep | 8 | 74.652074 | 79.665071 | 88.277513 | 91.746414 | 71.498628 | 73.923445 | 3.153446 |
| MSVR310 | MD ratio | 38 | 50.554177 | 68.020302 | 80.541456 | 85.617596 | 50.490374 | 67.681897 | 0.063803 |
| MSVR310 | repair/keep | 38 | 50.544641 | 67.851102 | 80.541456 | 85.448390 | 50.481473 | 67.681897 | 0.063167 |
| RGBNT100 | MD ratio | 5 | 83.910644 | 95.626825 | 96.909618 | 97.492713 | 73.677626 | 92.128283 | 10.233018 |

Source: five official receipts and complete histories in [S1], [S2], independently cross-checked in [S10]. Epoch-50 R5/R10, every epoch, and all exact numbers remain in the JSON evidence and companion review JSON. Selected-to-last deterioration is descriptive; without a matched control trajectory analysis it does not identify which objective caused the deterioration. In particular, RGBNT100's 10.233018 points are a decline from its selected endpoint, not a gain.

## All twelve registered paired comparisons

`S` denotes matched raw semantic; `G` denotes independent raw global-only; `MD` denotes the paired MD-ratio endpoint. Every row below has `phase_progress=false` under the predeclared line `delta mAP >= 0.5` and `delta R1 >= 0`.

| Dataset | Candidate / control | ΔmAP | ΔR1 | ΔR5 | ΔR10 | R1 repairs / new errors | Query AP improved / worsened |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | MD / S | +0.094523 | +0.358852 | 0 | +0.239234 | 7 / 4 | 277 / 215 |
| RGBNT201 | MD / G | +0.172721 | +0.239234 | +0.119617 | +0.119617 | 6 / 4 | 266 / 206 |
| RGBNT201 | repair / S | +0.277213 | +0.837321 | −0.119617 | 0 | 9 / 2 | 272 / 206 |
| RGBNT201 | repair / G | +0.355411 | +0.717703 | 0 | −0.119617 | 9 / 3 | 291 / 169 |
| RGBNT201 | repair / MD | +0.182690 | +0.478469 | −0.119617 | −0.239234 | 8 / 4 | 253 / 242 |
| MSVR310 | MD / S | +0.011952 | +0.169205 | 0 | +0.169205 | 1 / 0 | 150 / 91 |
| MSVR310 | MD / G | +0.012048 | 0 | 0 | +0.169205 | 1 / 1 | 205 / 125 |
| MSVR310 | repair / S | +0.002416 | 0 | 0 | 0 | 0 / 0 | 29 / 12 |
| MSVR310 | repair / G | +0.002512 | −0.169205 | 0 | 0 | 0 / 1 | 178 / 140 |
| MSVR310 | repair / MD | −0.009537 | −0.169205 | 0 | −0.169205 | 0 / 1 | 96 / 143 |
| RGBNT100 | MD / S | −0.179661 | −0.233236 | +0.291545 | +0.291545 | 17 / 21 | 677 / 691 |
| RGBNT100 | MD / G | −0.623141 | −0.991254 | −0.408163 | −0.466472 | 16 / 33 | 657 / 726 |

The five primary candidate/semantic comparisons advance 0/5; all comparisons advance 0/12. All available legal query entries were retained: 836 for RGBNT201, 591 for MSVR310, and 1,715 for RGBNT100 per comparison. Counts above and metric deltas were independently recomputed from every serialized per-query AP/rank entry; this is a text-level recount, not a new tensor inference or raw-distance ranking evaluation. Tiny printed differences between official float32 CMC receipts and exact integer-count ratios do not alter any gate. [S1], [S8], [S10]

The project decision line is a practical phase rule, **not a statistical significance threshold**. Missing it does not prove an exactly zero population effect.

## Bootstrap evidence and the stability boundary

The report uses 2,000 resamples of identity-macro AP differences from fixed selected models, with RNG seed 42; it does not train 2,000 models or provide independent training seeds. Identity counts are 30, 52 and 50. Its estimand is identity-macro AP, whereas reported mAP weights queries. These intervals are post-selection diagnostics on a repeatedly consumed benchmark, with no multiplicity-adjusted or selection-safe confirmatory inference. [S8] (lines 52–74)

| Dataset | Candidate / control | Identity-macro ΔAP | Recorded 95% percentile interval |
|---|---|---:|---:|
| RGBNT201 | MD / S | +0.087038 | [−0.216309, +0.383402] |
| RGBNT201 | MD / G | +0.154234 | [−0.038992, +0.381688] |
| RGBNT201 | repair / S | +0.270357 | [−0.130552, +0.719599] |
| RGBNT201 | repair / G | +0.337553 | [+0.002010, +0.714322] |
| RGBNT201 | repair / MD | +0.183319 | [−0.205057, +0.604598] |
| MSVR310 | MD / S | +0.007205 | [−0.006883, +0.024950] |
| MSVR310 | MD / G | +0.000766 | [−0.043624, +0.036308] |
| MSVR310 | repair / S | +0.001510 | [+0.000028, +0.003650] |
| MSVR310 | repair / G | −0.004930 | [−0.044127, +0.025171] |
| MSVR310 | repair / MD | −0.005695 | [−0.023159, +0.007801] |
| RGBNT100 | MD / S | −0.186415 | [−1.185743, +0.769750] |
| RGBNT100 | MD / G | −0.377224 | [−1.646597, +0.963184] |

Two intervals are strictly positive. Those observations must remain visible; it would be wrong to say that every interval includes zero. Neither positive interval supplies the missing practical gain, three-dataset replication or training-seed stability. The recorded intervals were inspected together with their generating source; their random draws were not rerun by this reviewer.

## Training activity, qualification and the missing sixth endpoint

All 9,839 saved formal steps were inspected and aligned with their saved batch identities. Independent label/environment counting also matches the repair runs' static legal-support fields. The following are step/exposure measurements, not unique-person counts or held-out retrieval effects. [S9], [S10]

| Dataset / objective | Formal steps | Positive incremental-loss steps | Correction/global norm ratio at best | Ratio at E50 |
|---|---:|---:|---:|---:|
| RGBNT201 / MD | 2,649 | 2,649 | 0.058247 | 0.117911 |
| RGBNT201 / repair | 2,649 | 1,657 | 0.061599 | 0.037538 |
| MSVR310 / MD | 706 | 706 | 0.005072 | 0.005034 |
| MSVR310 / repair | 706 | 706 | 0.005277 | 0.005229 |
| RGBNT100 / MD | 3,129 | 3,129 | 0.295475 | 0.666420 |

RGBNT201 repair has 57,416 eligible query exposures of 169,536 (33.866553%); 52,929 of the eligible exposures have at least two legal positives (92.185105%). Repair has support and nonzero loss in 189/2,649 steps; keep has support in 2,557 and nonzero loss in 1,653. There are 39,278 repair relations out of 11,089,680 legal relation exposures. At E50 its mean incremental loss is 0.0000017704, compared with 0.003241728 at the selected epoch.

MSVR310 repair has 21,708 eligible query exposures of 45,184 (48.043555%); 17,124 eligible exposures have at least two legal positives (78.883361%). Repair is supported and nonzero in 640/706 steps; keep is supported and nonzero in every step. There are 75,342 repair relations out of 2,793,000 legal exposures. Thus a blanket “no legal relations” or “auxiliary loss never active” explanation is contradicted for the received repair runs. Sparse repair support on RGBNT201 remains a measured limitation; it is not an established causal explanation of all datasets.

Positive scalar losses, large correction norms and M0 gradient qualification do not prove complementary retrieval information. Full training did not save isolated per-role gradients on every step. The saved official-distance path contains fused distances and labels/environment arrays; it does not contain a new own-global/correction/fused retrieval decomposition for these five checkpoints. No own-global quality degradation, equality, or g/c/f causal attribution can be inferred from the norm-ratio table or the separately trained global-only controls. [S7], [S11] (lines 103–107)

**RGBNT100 repair/keep is absent.** Its eight-step M0 evidence has legal relations and active repair/keep cells in every batch; all 1,024 repeated query exposures have multiple-positive support. Nevertheless, the third role's isolated unscaled query and key gradient norm sums are both zero across all eight steps, with `unused=False`. The registered qualification gate failed, so there is no formal fifty-epoch run, endpoint metric or substitute checkpoint. The new five-job controller's success does not erase the original six-job controller failure. These eight batches refute a no-evaluable-relations explanation for those batches; they do not identify a numerical cause or predict formal efficacy. [S12], [S3]

## Claim judgments and calibrated revisions

**C1 — no; confidence high for this scoped judgment.** The MD ratio gives small selected-point gains on RGBNT201, negligible changes on MSVR310 and regression on RGBNT100 against both specified references. None reaches the phase line. Revision: “In a single-seed, fifty-epoch training-objective comparison on the existing semantic model, the MD-inspired batch-ratio adaptation did not deliver the predeclared practical gain on any of the three datasets.” This does not prove universal ineffectiveness or invalidate the complete MDReID method. Stable repeated-training gains, confirmatory evaluation and a successful three-dataset outcome are missing; these gaps do not authorize rerunning or tuning the sealed failed objective.

**C2 — no; confidence high for this scoped judgment.** Repair/keep produces modest selected RGBNT201 net repairs but still creates 2/3 new Rank-1 errors against semantic/global controls and fails the mAP line. MSVR310 is effectively unchanged against semantic and loses one Rank-1 success against independent global-only. The RGBNT100 formal endpoint does not exist. Revision: “The implemented absolute-margin repair/keep objective produced modest RGBNT201 improvements and near-zero MSVR310 changes, with residual ranking harms; RGBNT100 failed M0 qualification and has no formal efficacy result.” General ranking preservation, a three-dataset benefit and training-seed stability are unsupported.

**C3 — yes, as a documented execution/completeness claim; confidence high on counts and internal consistency.** Five runs contain epochs 1–50, totaling 250 epochs and 9,839 steps; the report contains exactly twelve planned complete-query comparisons and the explicit missing endpoint. Initialization bindings, official receipts and source scope consistently identify seed 42, no added parameters and unchanged L2_1536 deployment. Candidate objective pairs share byte-identical batch-order files. Matching to historical raw semantic batch bytes is supported by the frozen acceptance function and received accepted rows; this reviewer did not independently reopen historical control batch bytes or remote checkpoint tensors. The attached integrity audit separately checked order and initializer digests against sealed controls. Batch-order equality does not independently prove equality of every stochastic augmentation value. Therefore the accurate revision is: “Five qualified objective endpoints completed their full planned schedule and comparison report, with unchanged semantic parameter/input/deployment design and recorded matched raw-semantic batch ordering; the sixth endpoint remains a documented M0 failure.” This is not six successful endpoints, capacity equality to global-only, invariant learned weights or full independent runtime attestation. [S13], [S14] (lines 56–82)

**C4 — no; confidence high.** There is no new regional responsibility supervision, no complete P1/P2/P3 intervention, no full MDReID reproduction, no novelty evaluation, no robust multi-seed/independent-development evidence, no ten-point improvement and no broad SOTA success. Revision: “These results close a training-objective comparison and constrain future mechanism design; the original regional three-dataset/SOTA objective remains ACTIVE_UNMET.” The project must not be redefined as successful because its negative-result documentation is complete.

## One bounded next scientific hypothesis

**H1: At these fixed selected checkpoints, residual correction is active but insufficiently selective for the model's own global retrieval errors; newly created errors may offset the rescues.** This is an untested hypothesis with medium motivation and low current causal confidence. It is grounded in the coexistence of nonzero training objectives, dataset-dependent correction amplitudes, observed rescue/harm tradeoffs and the missing own-global component evaluation. The independently trained global-only reference cannot resolve it.

The bounded test is one fixed-checkpoint decomposition for exactly the five selected checkpoints, using each checkpoint's own global feature `g`, existing correction `c`, and unchanged deployed fusion `f`. Use the same official query/gallery protocol and exclusions; bind every tensor/receipt to the existing checkpoint SHA; report all full-query AP/CMC, paired `f` versus its own `g` repairs/new errors and identity-level deltas. Correction-only retrieval is descriptive; it is not an inference replacement or a new selected model. Do not fit, change gains, tune thresholds, select new checkpoints, add seeds, or restart training. Any reported result remains a post-selection diagnostic on consumed data. The same evidence should check whether the own-global component is weak; the current review cannot assume it is unchanged or degraded.

H1 is weakened if own-global-to-fused comparisons show consistently useful error-selective gains without the predicted offsetting harms; that would direct attention to another unresolved source of the end-to-end gap. Failure of this diagnostic to establish complementarity does not by itself prove that regional deletion supervision would solve the problem. **Status: NOT_RUN in this review; no new g/c/f metric is asserted.** The original broad goal remains ACTIVE_UNMET regardless of the diagnostic outcome.

Do not reopen the closed near-capacity/native/reconstruction/slot-uniform/SIM, independent-head, detach/joint-L2 controls. Do not use numerical precision, loss weights, gains, learning rates, margins, seeds or gate thresholds to rescue these sealed failures. A future distinct mechanism requires its own scientific evidence; no supplementary training or publication is authorized or executed by this review.

## Verification, source citations and trace

Direct verification matched 45 mapped primary files and all 36 precheck-listed input byte hashes. Five complete histories/receipts, all 9,839 saved steps, every registered per-query entry, aggregate identity means, and static repair relation support were cross-checked with a standard-library CPU script. The first script invocation stopped because the source map contains two historical protocol editions; the script was corrected to the exact `training_feature_scale_protocols_20261002` paths bound by the run's recorded SHA and passed on its second invocation. No evidence file was modified. This was a reviewer helper error, not an experiment failure. [S10]

Primary full-report SHA256: `45a5f8ab701e03973e8733a89dde691afe9ae867af07fb3fd7bb9775e6c05572`.
Implemented-objective SHA256: `4859d1f19d9f9a9c944caefee5c7194c754444053b6e443af9d39e40a807a4cf`.
The full input SHA manifest, verifier source, parsed verdict, request, full response, routing action and execution history are in `.aris/traces/result-to-claim/2026-10-07_five891/` under this review directory. All writes are restricted to this review directory. No SSH/GPU/NN/install/Git or project source mutation occurred.

[S1]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_full_complete889/primary_text/0022_SUMMARY.json
[S2]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_full_complete889/LOCAL_RECEIPT_SUMMARY.json
[S3]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0417_QUALIFIED_FIVE_PLAN_20261007_888.md
[S4]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0385_incremental_role_objectives.py
[S5]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0324_global_task_role_heads.py
[S6]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0386_run_incremental_role_objective.py
[S7]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0406_run_incremental_role_objective_checked.py
[S8]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0256_analyze_correspondence_distances.py
[S9]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_training_text_analysis891/SUMMARY.json
[S10]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_result_to_claim891/.aris/traces/result-to-claim/2026-10-07_five891/independent_text_crosscheck.json
[S11]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0009_run_correspondence_roles.py
[S12]: C:/Users/gb/.codex_tmp/independent_evidence_draft/checked_m0_support_analysis889/SUMMARY.json
[S13]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_integrity_audit891/EXPERIMENT_AUDIT.json
[S14]: C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0415_queue_incremental_qualified_five.py
