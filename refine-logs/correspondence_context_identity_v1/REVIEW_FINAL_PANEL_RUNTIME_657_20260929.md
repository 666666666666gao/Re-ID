# Complete context/local panel runtime-artifact audit

Date: 2026-09-29. Reviewer: `/root/audit_context_analyzer_657`, continuing the previously fresh native Codex review. This is a follow-up by the same reviewer, not a new fresh-context review. Initial invocation requested `gpt-6-astra`, reasoning effort `max`, `fork_turns=none`; the caller transcription is at `logs/context_identity_review_invocations_657_20260929.json:8`. Backend identity is not independently attested (`:5`). `review_independence: same-family`; `acceptance_status: provisional`.

**Overall: WARN. Artifact consistency and deterministic arithmetic: PASS. Concrete code blockers: none found.** The earlier missing-panel limitation is resolved in the new artifacts: the accepted matrix contains **15/15**, and a completed analysis with all **15 registered contrasts** exists. This review verifies the archived text, hashes, bindings, and available arithmetic; it does not independently rerun the unavailable distance tensors, reload weights, or attest the original remote process.

The complete panel **does not support the registered claim of consistent cross-dataset gain from the full context/local candidate**. It loses to both the same-structure `static_none` control and the extra-supervision `context_global` control on RGBNT201 and RGBNT100. This scientific outcome is distinct from analyzer validity and successful completion of the experiment.

Only this report and its companion JSON were written. No SSH, training, GPU work, model forward, code change, original evidence rewrite, or pair-file rewrite was performed.

## A–F checks

| Check | Status | Finding and exact evidence |
|---|---|---|
| A. GT provenance | WARN | The unchanged evaluator reads dataset IDs/cameras/scenes and scores all three outputs against them (`tools/run_correspondence_context_identity.py:207`, `:214`; `tools/collect_correspondence_context_identity.py:73`). The final two receipts retain the same bound protocol and baseline (`logs/context_identity_context_local_RGBNT100_670_20260929/official_metrics.json:12`; global counterpart `:12`). Original protocol JSONs, raw data/inventory and upstream evaluator remain absent locally; no unconditional end-to-end GT-provenance PASS. |
| B. Score normalization | PASS | Standard AP/CMC percentages remain unchanged (`tools/train_rgbnt100_signal_oof.py:253`; `tools/train_msvr310_signal_oof.py:223`). All 48 factor entries recompute exactly from accepted official metrics using the four registered formulas (`tools/analyze_correspondence_context_identity.py:26`; `logs/context_identity_factor_complete_657_20260929/summary.json:10`, `:170`, `:330`). No self-normalized performance score found. |
| C. Files, numbers, statuses | WARN | All 15 complete rows, all campaign worker exit codes, all six newly archived file hashes, final two metrics/histories/scalars, all pair-file hashes and bindings, and preservation of the prior 13 accepted rows pass. Matrix `logs/context_identity_accepted_complete_670_20260929.json:6`; final cells `:1081`, `:1402`; campaign `logs/context_identity_campaign_complete_670_20260929.json:2`, `:970`; archive `logs/context_identity_archive_670_20260929.json:3`. The 30 remote checkpoint/distance binaries are not present locally, so binary existence/hash and ranking replay remain unavailable to this reviewer. |
| D. Live calls and outputs | PASS, within archived call/output scope | Unlike the earlier partial-panel review, final factors and 13 newly produced pair reports now exist; the summary has complete status and the matching matrix/source digests (`summary.json:2`). The unchanged source computes factors, reuses sealed pairs, otherwise calls `compare`, and writes summary last (`tools/analyze_correspondence_context_identity.py:104`, `:110`, `:114`, `:121`). This establishes source-to-output consistency, not independent observation of the remote process or re-execution of its assertions. |
| E. Actual scope and claims | WARN | All five conditions are represented on all three datasets, one seed 42 and 50 epochs each. Paired tables cover 836/1,715/591 queries and 30/50/52 identities. Official-test mAP selected the checkpoints each epoch (`tools/run_correspondence_context_identity.py:138`); identity bootstrap concerns fixed selected models, not training seeds (`tools/analyze_correspondence_distances.py:62`, `:75`). No robust multi-seed or untouched-test claim follows. The registered consistency condition (`refine-logs/correspondence_context_identity_v1/EXPERIMENT_PLAN.md:49`) is not supported by the observed signs. |
| F. Evaluation type | PASS | Fused, shared-global and joint-local retrieval are `real_gt`; AP changes, repair counts and identity summaries are derived `real_gt` post-selection diagnoses. Loss records are supervised training diagnostics; initialization/reload/hash checks are engineering checks, not retrieval success. No model-generated target is being substituted for identity GT (`tools/run_correspondence_context_identity.py:112`, `:207`). |

## Complete panel and historical preservation

The complete matrix was collected at **18:00:24.397859 +08:00**, with seed 42, exactly 15 rows and `VERIFIED_COMPLETE` for every member of the 3×5 Cartesian set (`logs/context_identity_accepted_complete_670_20260929.json:3`). The campaign reports all 15 worker jobs `COMPLETE`, each `exit_code=0`, and terminal completion at **18:00:24.400838 +08:00** (`logs/context_identity_campaign_complete_670_20260929.json:970`). Each embedded worker result agrees with its matrix row for condition, selected epoch, metrics, diagnostic metrics, checkpoint hash and distance hash.

All **13 previously accepted row objects** are exactly unchanged from snapshot 667. The old accepted matrix, old code/log inputs, sealed pair files, and prior source-review reports also retain their previously audited SHA-256 values. The previous source review remains a valid dated account of its then-incomplete panel; it was not rewritten to imply knowledge of future results.

The matrix, archive and analysis agree on matrix SHA-256 `7fd21fa6398578ff22aba2f1812ca2c6a163287ae2447263811941b53180f958` (`logs/context_identity_archive_670_20260929.json:3`; `summary.json:5`). The analyzer still hashes to `91ccb8995f98624a42c2e08c4afc391e651aa4a535d345430497894dc477cb77`, and the comparison tool to `833ebeb47cb5840422710fc00df1a974338cb04078249720d867f18b945f997a`; both match the summary (`summary.json:6`). All nine declared runtime source hashes remain unchanged. The prior caveat remains: the nine-file manifest is not a transitive lock of every imported evaluator/helper.

The analyzer's full15 schema/count/set/status gate now evaluates true on the supplied matrix (`tools/analyze_correspondence_context_identity.py:45`). The prior real partial5 rejection and local partial13 rejection remain historical evidence of the gate; neither is misrepresented as the completed run. The summary timestamp **18:01:26.485287** is the `at` value initialized before comparisons (`tools/analyze_correspondence_context_identity.py:90`), not a precise finish timestamp. Newly produced pair timestamps span **18:01:26.714942–18:01:44.591428 +08:00**.

## Final two RGBNT100 endpoints

Both newly archived training histories contain epochs 1–50, seed 42, best-official-mAP policy, and the latest-epoch maximum-mAP selection rule. Both select **epoch 1**, with all four reported metrics coming from that same selected checkpoint (`logs/context_identity_context_local_RGBNT100_670_20260929/training.json:640`; global counterpart `:640`).

| Condition | mAP | Rank-1 | Rank-5 | Rank-10 | Logged steps |
|---|---:|---:|---:|---:|---:|
| context_local | 84.5125006310 | 94.9271142483 | 95.3935861588 | 95.6268250942 | 6,559 |
| context_global | 85.3363012190 | 95.5685138702 | 96.2682187557 | 96.3848412037 | 6,559 |

These numbers match the final histories, official receipts, accepted matrix, campaign results and summary cells (`logs/context_identity_context_local_RGBNT100_670_20260929/official_metrics.json:16`; global counterpart `:16`). Both complete diagnostic metric dictionaries also match every copied occurrence.

I independently reconstructed all **13,118 new scalar steps**. Epoch/batch sequences, epoch means, finite values, loss decomposition, first/best/last summaries and nonzero counts agree. Each run has 6,559 nonzero auxiliary-ID steps; each maximum loss reconstruction error is **2.384185791015625e-7**. Combined with the unchanged prior 37,922 steps, the complete panel contains **51,040** logged training steps. This is scalar accounting, not a gradient-contribution measure.

All six hashes in the new archive match the actual files (`logs/context_identity_archive_670_20260929.json:10`, `:21`). The two new initializers match the existing RGBNT100 initial state and every same-initializer field checked by the analyzer, including the reused entry hash; both have 2,509,834 trainable parameters (`training.json:25`, `:26` in each final-two directory). Per-dataset initial-state equality and both parameter-count groups now hold across all five conditions.

## Factor and pair verification

All **48 factor scalar entries** (3 datasets × 4 factors × 4 metrics) and all **15 summary cell objects** exactly match independent arithmetic/copies from the accepted matrix. The mAP effects, in percentage points, are:

| Dataset | Mean query effect | Mean local-ID effect | Query × local-ID difference-in-differences | Local versus global ID |
|---|---:|---:|---:|---:|
| RGBNT201 | +0.020999 | −0.591536 | +0.015911 | −0.243563 |
| RGBNT100 | +0.002282 | −0.615205 | +0.013780 | −0.823801 |
| MSVR310 | +0.021400 | +0.181943 | −1.396767 | +0.380162 |

Evidence: `logs/context_identity_factor_complete_657_20260929/summary.json:10`, `:170`, `:330`; definitions: `tools/analyze_correspondence_context_identity.py:30`. These are descriptive condition effects on selected endpoints; they are not independent additive contribution claims or causal identification of a unique mechanism.

The summary has exactly the five registered contrasts for each dataset, in the source order. For every one of the 15 referenced reports, I verified:

- the actual report SHA matches the summary;
- the summary copies all report fields other than the deliberately omitted identity table exactly;
- both endpoint epoch/checkpoint/distance/receipt bindings match the complete matrix;
- per-endpoint metrics agree with the official values within **2.937453785989419e-6 pp**, below the registered 1e-5 tolerance;
- each metric delta equals candidate minus control;
- `(repairs − new_errors) × 100 / query_count` equals Rank-1 change;
- identity counts, positive query counts, macro AP, query-weighted AP, and improved/worsened identity counts agree with all **660 identity-table rows**;
- all four edges of the first-four-condition AP cycle close separately for each identity, with maximum numerical residual **4.884981308350689e-15 pp**.

The 660 rows are five repeated contrasts over 30+50+52 identities, not 660 independent identities. Rank-1/CMC in the official receipts use upstream floating-point values while pair recomputation uses integer-rank percentages; the observed few-millionths-of-a-point differences explain, for example, the nearly zero RGBNT201 mean Rank-1 query effect. They must not be interpreted as material improvement.

The report directory contains exactly **13 new pair JSONs plus `summary.json`**. The two remaining comparisons are explicitly `sealed_report_reused` and reference the original RGBNT201 and MSVR310 files with their original timestamps and hashes (`summary.json:544`, `:1114`). Original SHA-256 values remain:

- RGBNT201: `aaf0bac540234e98e8712815f65285ba2099f400896f68f3354fcf149f428fdc`.
- MSVR310: `c50322fc949f76322f8b57c133d105780db2b87b421ed44dcb9f162cb15bbf6c`.

Thus the archived outputs substantiate explicit sealed reuse and preservation; there are no newly written duplicate pair files for these two comparisons. The unchanged source checks bindings before reuse, creates a new output directory and writes the complete summary last (`tools/analyze_correspondence_context_identity.py:75`, `:89`, `:110`, `:121`). This is consistent with the required immutable-output workflow; it is not a filesystem history attestation.

## Scientific disposition and remaining limits

The complete candidate's direct changes against the two registered controls are:

| Dataset | context_local − static_none, mAP / Rank-1 pp | context_local − context_global, mAP / Rank-1 pp |
|---|---:|---:|
| RGBNT201 | −0.570537 / −1.076549 | −0.243563 / −0.119615 |
| RGBNT100 | −0.612923 / −0.058311 | −0.823801 / −0.641400 |
| MSVR310 | +0.203343 / +0.507611 | +0.380162 / +0.169200 |

These are direct arithmetic differences between accepted cells (`summary.json:38`, `:198`, `:358` and their subsequent cells). The two negative datasets prevent a claim of consistent three-dataset mAP/Rank-1 improvement under the plan's criterion (`EXPERIMENT_PLAN.md:49`). The small positive averaged query mAP effects do not remove the negative MSVR310 interaction or prove robust sample-conditioned selection. Rejecting this claim does not establish that context queries or local supervision can never help another registered design.

**Remaining unavailable evidence:** original complete distances and checkpoints (30 binaries), bound raw protocol/inventory/data, and upstream evaluator source are not present locally. Per-query rankings, repair membership, and their raw-data lineage were not independently reconstructed. The 2,000-resample NumPy bootstrap endpoints were not replayed: identity tables were available and their means/counts were checked, but no usable NumPy installation was available in the inspected local interpreters. Interval endpoints were checked only for finite ordered values and exact copying, not awarded a numerical-replay PASS. No library was installed or environment altered.

The summary's explicit single-seed, post-selection and unequal-effective-capacity boundaries remain necessary (`summary.json:1347`). No remote PID or GPU state was queried by this audit. The campaign and result files are runtime artifacts with internally matching provenance, not independent live-process observation. Review-invocation records accurately label themselves caller transcriptions, same-family and provisional (`logs/context_identity_review_invocations_657_20260929.json:2`).

**Disposition:** accept the archived full-panel text/hash/arithmetic closure within these limits; retain overall WARN and provisional semantic acceptance. No code fix or repeat training is indicated. Preserve the negative scientific result and old source-review bytes. The parent should append this follow-up request/response to the review trace and publish the complete-panel status without recasting the experiment as a successful cross-dataset improvement.

Final summary SHA-256: `4aba5bd1e354fc5fa508c42892baaf5b71e40092d8c35fd19e72b7a00e6572d7`.

Preserved earlier reports: Markdown SHA `1138cccdd3599b3573defe08d22c219963642c8b5cceaa92d4fb6ef2625b4ae2`; JSON SHA `c9a0aaa1351829902fa055f79fca1d03055ff3fb2c6b2bad78dacadd79ed2782`.
