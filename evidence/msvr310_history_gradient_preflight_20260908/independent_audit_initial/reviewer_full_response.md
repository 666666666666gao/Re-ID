**Overall verdict: WARN — the saved preflight supports its limited numerical-mechanism claims. I found no incorrect gradient formula, fabricated ground truth, missing local result, or mismatched report-table value.** The warnings concern incomplete verifier coverage, the scope of the direct-graph proof, unavailable remote tensors, and a stale tracker.

Reviewer: fresh `gpt-6-astra` / `max`; `review_independence: same-family`; `acceptance_status: provisional`. Work was read-only except for the permitted independent audit script and JSON. No remote access, model execution, training, checkpoint changes, or additional seeds were used.

The independent checks are saved in:

- [audit.py](C:/Users/gb/.codex_tmp/history_gradient_preflight_independent_audit_20260908/audit.py)
- [independent_checks.json](C:/Users/gb/.codex_tmp/history_gradient_preflight_independent_audit_20260908/independent_checks.json)

The JSON records **19,782 passed deterministic checks**, no failed checks, input hashes, complete row coverage, and explicit unavailable-input boundaries. Its SHA-256 is `81b4a93c05c81cc51e2450100bd1fd0970778bc2c652b8d89f93b589b775074e`.

**A. Ground-truth provenance — PASS, within the recorded source-data boundary**

The identity targets come from the dataset label inventory, not model predictions. The protocol builder selects only `record_manifest["bounding_box_train"]` and checks identities, cameras, and scenes against filenames before constructing the records: `tools/build_msvr310_train_oof_protocol.py:17`, `:21`, `:40`, `:44`, `:48`. The protocol explicitly identifies the split as official training data and distinguishes its internal identity OOF evaluation from official testing: `protocols/msvr310_train_oof_v1.json:4`, `:6`, `:14`, `:22`.

I independently verified:

- The original label-manifest hash equals `protocol["label_evidence_sha256"]`.
- The protocol-builder hash equals `protocol["builder_sha256"]`.
- All **1,032 protocol records** match the original training labels and RGB paths.
- All three modality paths for every record have matching filename identity, scene, and camera fields.
- The deterministic scene-eligibility/round-robin split reconstructs exactly.
- Every saved current record and historical record belongs to the relevant source fold; none belongs to that fold’s heldout set.

The source folds contain **672 / 683 / 709 records** and **103 / 103 / 104 identities**. The corresponding heldout sets contain **360 / 349 / 323 records** and **52 / 52 / 51 identities**.

The model path receives remapped classification labels through `records_for`, while mining uses the original dataset identity. This preserves identity equality: `tools/train_msvr310_signal_oof.py:68`, `:73`, `:76`; `tools/probe_msvr_history_candidate_gradients.py:177`. Remapped **class 0 is present for eight current-record exposures in each of the nine states**; it is not excluded. The synthetic fixture also includes identity 0, although its `legal_class_zero` output is a runtime declaration rather than an independently saved class-specific gradient proof.

Candidate masking is correct for the stated objective:

- Current positives are same identity, excluding the diagonal.
- Current negatives are different identities.
- Historical positives/negatives use identity equality.
- Scene labels are used for diagnostics, not to relabel negatives.
- Historical copies of current records are excluded.
- The queue keeps the latest occurrence of each record; `ViewFields.positions` retains the final position within its original B64.

Evidence: `tools/msvr_instance_memory.py:15`, `:19`, `:29`, `:47`, `:55`, `:72`; `tools/msvr_freshness_probe.py:59`.

The new execution path calls `records_for(..., True)` and contains no retrieval stage: `tools/probe_msvr_history_candidate_gradients.py:266`; `tools/run_msvr_history_candidate_gradients.py:25`. The zero heldout/official **image-forward counters** are consistent with that path. This should not be broadened into “no heldout artifact access of any kind”: inherited context reads prior Q1 summaries and hashes prior retrieval artifacts, including `tools/train_msvr310_source_style.py:50`. Those provenance reads are distinct from new heldout-image inference.

**B. Metric definitions, normalization, and gradient mathematics — PASS; numerical-proof coverage is limited as detailed in D**

The differentiated objective is the intended **mean over the current 64 anchors**, with current peers and historical candidates participating in hard mining. Both implementations normalize the current embedding, compute Euclidean distances, apply the same identity masks, use margin `0.3`, and average the same hinge objective. Historical coordinates are already normalized when stored.

Evidence: `tools/probe_msvr_history_candidate_gradients.py:26`, `:29`, `:32`, `:40`; `tools/msvr_instance_memory.py:40`, `:43`, `:46`, `:69`.

The current batch remains differentiable on both sides of `cdist(unit, unit)`. Therefore \(g_U\) includes anchor and current-candidate contributions. Restoring \(g_V\) does not add historical anchors. The report describes this correctly at `results/MSVR310_HISTORY_CANDIDATE_GRADIENT_PREFLIGHT_2026-09-08.md:9`.

The historical VJP reconstruction is mathematically sound:

1. Differentiate the expanded loss with respect to a historical feature leaf.
2. Re-encode each relevant original B64 with the saved frozen fields and role-entry RNG.
3. Scatter the corresponding upstream coefficients to the retained record positions.
4. Differentiate their inner product with the re-encoded features.
5. Sum group contributions.

Evidence: `tools/probe_msvr_history_candidate_gradients.py:194`, `:197`, `:84`, `:95`, `:98`, `:99`, `:101`. Unique historical records and final-position mapping make assignment into `coefficients[positions]` appropriate here; I found no duplicated-position accumulation defect.

The original 14-component objective is computed by the actual criterion and `weighted_training_loss`. Historical contribution is added with the original fused-Triplet weight, which is **1.0** in the bound configuration: `modeling/trifusion/signal_preserving_v8.py:721`, `:727`, `:735`; `tools/run_signal_preserving_v5.py:110`, `:114`, `:123`; `tools/probe_msvr_history_candidate_gradients.py:157`, `:190`, `:191`, `:76`.

The cosine comparisons use corresponding parameter tensors within each role’s encoder block. They do not compare CNN and Mamba vectors in different parameter spaces. They also do not represent the entire model’s classifier/neck parameter vector: `tools/probe_msvr_history_candidate_gradients.py:154`, `:72`.

Normalization is legitimate and disclosed:

- Raw parameter norms accompany cosine and norm-ratio diagnostics.
- Cosine divides by the two compared vector norms.
- The direct-graph check uses a conventional relative error denominator, `max(direct_norm, combined_norm)`.
- Zero-norm cosines are represented as `None`.
- Ratios and cosines are averaged per batch, as stated in the report, rather than being presented as the cosine of averaged gradients.

Evidence: `tools/probe_msvr_history_candidate_gradients.py:59`, `:65`, `:212`; `tools/analyze_msvr_history_gradient_text.py:35`, `:51`; report `:37`.

The RNG/buffer handling matches the intended fixed-state measurement. Historical replay restores CPU/CUDA RNG, skips classification necks, and current-path buffers are restored after every batch. The actual encoder/fusion path uses LayerNorm/GroupNorm and frozen tail blocks; the mutable classification BN necks are outside historical encoding. Evidence: `tools/probe_msvr_history_candidate_gradients.py:43`, `:219`, `:233`, `:239`; `modeling/trifusion/signal_preserving_v8.py:607`, `:641`; `modeling/trifusion/experts/semantic_residual.py:44`, `:46`. The saved equality checks remain runtime witnesses; I did not regenerate their model tensors.

The added mathematical note also passes:

- \(D_U=2(x-y)^TAx\) and \(D_V=-2(x-y)^TAy\) correctly sum to \(2(x-y)^TA(x-y)=0\) for skew-symmetric \(A\).
- Common orthogonal rotation preserves every pairwise distance, hence the distance-based hard-mining/hinge scalar.
- The role-slot factor \(1/\sqrt{18}\) follows from the actual normalization chain: normalized modality slots → normalized three-modality role → normalized three-role bank → equal-energy fused normalization.
- The note explicitly excludes fixed-classifier ID-loss invariance and says no rotation was fitted or demonstrated in MSVR310.

Evidence: `docs/MSVR310_PARTIAL_VS_TOTAL_METRIC_GRADIENT_2026-09-08.md:17`, `:25`, `:29`, `:33`, `:35`, `:37`, `:43`; `modeling/trifusion/signal_preserving_v8.py:454`, `:458`, `:499`, `:511`; probe `:185`.

**C. Artifact existence, hashes, arithmetic, and chronology — WARN**

All **25 manifest-listed preflight texts exist**, decode as UTF-8, and match both their declared size and SHA-256. Their total is exactly **805,652 bytes**. All nine saved receipts exactly equal their corresponding summary entries.

The complete independent pass covered:

| Item | Independently checked result |
|---|---:|
| Saved steps | 72 |
| Historical batches | 45 |
| Historical role rows | 135 |
| Declared distance elements, from all offsets/shapes/file sizes | 625,920 |
| Extra role-record forwards | 9,792 |
| Engineering report rows | 9, all values match |
| Role-statistic report rows | 27, all 108 statistic values match |
| CSV rows | 135, every column matches |
| CPU aggregate arrays | All nine states and three roles match |
| Reaggregation means, ranges, signs, and quantiles | All match |
| Maximum gradient-sum norm-identity error | `6.026005765920966e-12` |
| Maximum pairwise comparison norm-identity error | `1.943921143653686e-15` |

The report’s engineering and role tables are therefore arithmetically supported by the full saved text population, not selected examples. Evidence locations: report `:21`, `:41`, `:67`; all nine `preflight/fold_*/steps.jsonl` files, lines 1–8; `reaggregation/all_role_history_steps.csv`.

Across all 135 role rows, independently recomputed text statistics are:

| Diagnostic | Minimum | Maximum | Mean |
|---|---:|---:|---:|
| \(\|g_V\|/\|g_U\|\) | 0.4815581782 | 1.1524469636 | 0.8424395504 |
| \(\cos(g_U,g_V)\) | −0.9391284811 | −0.3526926507 | −0.7570334449 |
| \(\cos(g_U,g_U+g_V)\) | 0.0712399406 | 0.8769592183 | 0.5530773399 |
| Task-total before/after cosine | 0.5799934544 | 0.9828298554 | 0.8680481100 |

All **135 historical norms are nonzero and exceed their recorded same-graph repeat difference**. All 135 current/history cosines are negative; all 135 current/combined and task-total before/after cosines are positive. These are consistency checks of recorded runtime statistics, not independently reconstructed parameter gradients.

The bound Q1 summary/CPU hashes match. I also checked all **29 files in the previous Q1 intake manifest** by size and hash. Both endpoints’ five-gate decisions reconstruct as **0/5** from their sealed metric/lower-bound values. I did not independently resample the prior bootstrap.

There are three qualifications:

1. **Remote binary inputs are unavailable locally.** The nine `distances.f32` hashes and actual matrix entries could not be independently verified, although all declared shapes, offsets, byte totals, and CPU receipt totals agree. Checkpoint tensors, pixels, upstream feature gradients, and parameter VJPs were also not locally reconstructed. The report correctly states this at `:11` and `:33`.

2. **Five inherited local source files are CRLF copies of bound remote LF files.** All top-level gradient and coordinate project-file bindings match exactly. In the recursively followed lineage, these five files match the expected hashes only after CRLF→LF normalization: `criterion.py`, `state.py`, `builder.py`, `experts/mamba.py`, and `experts/semantic_residual.py`, all under `modeling/trifusion/`. The prior recorded source-byte evidence documents this distinction, for example `evidence/trifusion_msvr310_trifusion_v1_prelaunch_source_bytes_20260906.json:113`. I did not modify or normalize any source file. Do not describe every inherited local raw-byte hash as exact.

3. **The tracker is stale.** `EXPERIMENT_TRACKER.md:6` still says preflight RUNNING, `:7` says source NOT_STARTED, and `:8` says CPU NOT_STARTED. The saved pipeline instead records preflight exit 0 at 05:24:58.144320, preflight CPU exit 0, and source PID 3799 starting afterward: `pipeline.json:46`, `:47`, `:64`, `:65`, `:70`, `:84`. The report’s completed-preflight account is correct; the tracker needs a dated status update.

The 05:32:41 observation is a historical snapshot. It proves what was recorded then, including fold-0 initial at epoch 7, not the process’s current live status. No completed full-source result is present in this audited intake.

**D. Executed code and uncovered verification — WARN**

All functions in the four new execution/verifier/analyzer scripts have call sites, and the saved pipeline/logs demonstrate execution of T0, all nine preflight states, the CPU verifier, and the source-stage launch. I found no reported metric generated solely by dead code.

The important uncovered areas are concrete:

- **T0 does not exercise the production grouped VJP implementation.** `math_check` constructs random current features and one historical linear transform, then compares direct autograd with `grad_outputs`. It does not call `candidate_vjp`, `encode_graph`, the AMP-scaled `gradients` helper, or the group/position scatter path: probe `:109`, `:124`, `:127`. Its result is correctly named synthetic chain-rule verification, but it is not a complete production reconstruction test.

- **The real direct-graph witness is one single-history-group batch per state.** The guard `if proof is None` and explicit `history_groups == 1` assertion enforce this: probe `:203`, `:204`, `:216`. All nine saved proofs occur at **step 4**. Later steps execute **2, 3, 4, and 5 history groups**, but do not receive another direct-graph comparison. The maximum recorded relative error is `7.623875794974142e-05`, below the preregistered `0.005`; see `preflight/fold_1_control/receipt.json:326`, `:334`, `:335`. The report accurately limits this claim at `:17`.

- **The CPU verifier reads only six of the thirteen logged statistics fields.** It does not validate:
  `current_wrong_order_anchors`,
  `expanded_hinge_positive_anchors`,
  `maximum_memory_age`,
  `memory_cross_scene_positive_pairs`,
  `memory_negative_pairs`,
  `memory_negative_violations_against_batch_hard_positive`,
  `memory_positive_pairs`.

  Evidence: producer `tools/msvr_instance_memory.py:70`; verifier `tools/verify_msvr_history_candidate_gradients.py:64`, `:71`, `:73`, `:74`. My complete text audit closes the four pair-count/age fields that are recoverable from metadata. The three matrix-dependent omitted counts remain independently unchecked locally. Thus the sealed plan’s “all counts” language at `EXPERIMENT_PLAN.md:22` is broader than the current verifier’s implementation.

- **The 14-component scalar `row["loss"]` is not independently reassemblable from this intake.** The probe saves the weighted total and fused-Triplet statistics but omits the component values: probe `:191`, `:224`, `:228`. The CPU verifier reconstructs current/expanded Triplet scalars, not the complete task total: verifier `:71`, `:72`. Static code supports the weighting, but a claim that CPU independently recomputed every total-loss value would be unsupported.

- **Some queue branches are unexercised by the preflight.** Maximum historical age is 5 and maximum selected historical records is 221. It never reaches age-8 expiry, 512-entry eviction, or a zero-upstream group skip. Every available group was selected in these 45 historical batches. This is a coverage limit, not evidence of a wrong queue implementation.

These findings do **not** demonstrate an incorrect \(g_V\). They constrain the strength of “verified complete gradient implementation” claims.

**E. Scope and interpretation — PASS for the report’s stated claims; scientific conclusions remain limited**

The report consistently labels this as a short, fixed-state preflight and rejects retrieval or training-gain conclusions: report `:3`, `:7`, `:17`, `:69`, `:75`; plan `:13`, `:28`.

Actual per-state data coverage is:

| Fold | Observed source records | Total source records | Observed source identities |
|---|---:|---:|---:|
| 0 | 333 | 672 | 63 |
| 1 | 316 | 683 | 61 |
| 2 | 350 | 709 | 62 |

The three states within each fold use identical current indices, pixel hashes, and history metadata. Across folds, this amounts to **738 distinct dataset records** and **129 distinct identities**, with repeated exposures across states/folds. These are not 738 independent experiments.

Only seed 42 is involved. Initial, control endpoint, and fresh-memory endpoint are fixed parameter states, not samples along the original optimizer trajectory. No optimizer step occurs. Historical age means batch recency, not parameter staleness. Same-graph repeated backward differences measure numerical repeatability; they do not measure variability across training runs, parameter states, or stochastic re-encodings.

Negative \(\cos(g_U,g_V)\) alone does not establish harmful task conflict, a causal explanation of Q1 failure, an ascent step under the actual optimizer, or a reason to apply PCGrad. The report and added mathematics note correctly preserve those distinctions.

The rotation note is clearly theoretical and appropriately bounded at `docs/MSVR310_PARTIAL_VS_TOTAL_METRIC_GRADIENT_2026-09-08.md:3`, `:41`, `:43`, `:47`. Its input SHA is `aa2920f3970a69cabf568a7bccbd3586670c27e7447e04a10f426590332b06d8`.

**F. Evaluation-type classification — PASS, with separate types**

| Subevaluation | Appropriate classification | Claim ceiling |
|---|---|---|
| T0 random-vector/linear-map calculation | `synthetic_proxy` | Synthetic algebra/autograd consistency |
| Source identity mining and Triplet statistics | `real_gt`, source-only diagnostic | Behavior under dataset-provided training identities |
| Current/history gradient cosine, norm, repeat-noise, direct-graph comparisons | `self_supervised_proxy` for numerical consistency, conditioned on a real-GT loss | Recorded gradient/mechanism behavior; no retrieval-performance claim |
| Hashes, queue replay, CSV/JSON arithmetic | Provenance/arithmetic verification; not a benchmark evaluation | Artifact consistency |
| Orthogonal-rotation note | Analytic example, not a measured evaluation | A counterexample to overinterpreting negative cosine |
| Previous sealed Q1 | `real_gt`, train-internal identity OOF | Its existing qualified internal retrieval result; not official-test performance |
| Full 9×260 source stage | Incomplete in the audited package | No completed-stage result may yet be claimed |

No newly reported subevaluation is an official retrieval evaluation.

**Actions and claim impact**

1. Update the tracker to distinguish completed preflight/CPU from the source-stage snapshot. Preserve the sealed plan and raw evidence.
2. Retain the explicit “runtime witness” qualifier for gradients, pixel equality, state hashes, and direct-graph agreement.
3. Narrow “all counts/all losses independently checked” to the actual verifier coverage. A separate read-only matrix postcheck can close the omitted matrix-derived counts without changing the running experiment. The omitted 14-component total-loss decomposition cannot be recovered from this intake alone.
4. Keep the direct-graph statement at **nine single-history-group witnesses**. Multi-group execution and text closure do not supply a separate direct-model proof.
5. Keep the rotation explanation as a theoretical possibility; do not label measured historical gradients as “rotation-removal gradients.”

Supported claims are: the preflight completed; its full text arithmetic is consistent; recorded historical gradients are nonzero above recorded repeat differences in all 135 role rows; they change the recorded gradient direction under the stated fixed-state objective. Claims of improved training, new-identity retrieval gain, a proven cause of Q1 failure, full-source completion, or independently reconstructed multi-group model gradients remain unsupported.