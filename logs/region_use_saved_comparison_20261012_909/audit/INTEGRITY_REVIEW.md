# Saved query comparison909: integrity review

**Overall WARN; no open repair or blocking finding.** Saved numerical comparison is supported within its declared scope. No arithmetic, query join, GT-metadata or selected-best discrepancy was found. The delimiter defect is closed by 16 targeted document checks. WARN remains for saved-text/runtime provenance and selected-development scope; no new execution gate was created.

Fresh Codex reviewer `/root/audit_saved_comparison909`; requested Astra/max, actual model and reasoning effort **UNATTESTED**. Semantic review is **same-family / provisional**. Deterministic acceptance is confined to the listed checks.

The full received request is preserved in `REVIEW_REQUEST_ACTUAL.txt`; 60 source/text inputs are byte-preserved under `evidence_snapshot/` with SHA256 in `AUDITED_INPUT_MANIFEST.json`. Only saved text and source were read. No producer/report execution, production import, NumPy, torch, tensor/checkpoint read, SSH, GPU call, source/seal/campaign change or runtime gate occurred. The broken ambient `python` launcher returned `No pyvenv.cfg file`; the installed CPython 3.12.12 interpreter then ran the independent standard-library verifier.

## Verified results

All deltas are ambiguity minus uniform in percentage points. Aggregate mAP/CMC are recalculated from saved per-query AP and first ranks; individual AP is **not** regenerated from absent complete ranked lists or distance tensors.

| Dataset | Queries / IDs | Best U / A | Delta mAP | Delta R1 | AP better / worse / equal | Identity macro delta AP | Worst ceil5% N / mean delta AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| RGBNT201 | 836 / 30 | 8 / 8 | -0.1581480577 | +0.1196172249 | 222 / 271 / 343 | -0.1782876270 | 42 / -5.2577030159 |
| MSVR310 | 591 / 52 | 38 / 38 | +0.0039826023 | 0 | 32 / 29 / 530 | +0.0032798636 | 30 / -0.0109994890 |

RGBNT201 has 6 first-rank repairs and 5 new errors; contributions from still correct, still wrong, repair and new error are -0.1208409252, -0.0558241663, +0.0437890099 and -0.0252719761 points, summing to -0.1581480577. MSVR310 has no first-rank transitions: 401 still correct and 190 still wrong contribute +0.0038559093 and +0.0001266930 points. Its 30-query worst tail contains all 29 negative changes and one zero change; it is not a damaged-only average.

| Dataset / fixed global stratum | N | Within-stratum delta AP | Contribution to dataset delta mAP |
|---|---:|---:|---:|
| RGBNT201 / correct | 660 | -0.1444277659 | -0.1140219204 |
| RGBNT201 / wrong | 176 | -0.2095991521 | -0.0441261373 |
| MSVR310 / correct | 402 | +0.0056687622 | +0.0038559093 |
| MSVR310 / wrong | 189 | +0.0003961671 | +0.0001266930 |

Every full and stratum field, including each identity row/macro, transition count/contribution and ceil5% signed tail, matches the saved JSON with absolute tolerance 1e-10. Saved author CMC uses float32; integer first-rank recomputation agrees with the original <1e-5 percentage-point tolerance. All copied output query values match the four original tables exactly.

## A-F checks

### A_gt_provenance: PASS

Protocol identities and camera/scene fields are dataset metadata; every path identity/environment and every legal-positive count was independently checked. Full query order matches each saved table. Different-identity distractors remain, including those in the same environment. Image bytes and distance tensors were outside scope.

Evidence: `evidence_snapshot/sources/tools/official_three_dataset_data.py:8`; `evidence_snapshot/sources/tools/run_correspondence_roles.py:85`; `evidence_snapshot/sources/tools/train_rgbnt100_signal_oof.py:253`; `evidence_snapshot/sources/tools/train_msvr310_signal_oof.py:223`.

### B_normalization: PASS

AP averages precision at every legal positive across the complete legal gallery; mAP/CMC average all valid protocol queries. New deltas use 100*(ambiguity AP-uniform AP). Full transition contributions divide by dataset N; nested transition contributions inside a stratum divide by that stratum N despite the reused all_query field name. Only the outer stratum contribution divides by dataset N. Identity macro gives equal weight to each identity present in that subset; tail size is ceil(N/20), including zero changes if needed. No prediction-extrema/mean denominator inflates retrieval scores.

Evidence: `evidence_snapshot/workspace/compare_saved_region_use909.py:15`; `evidence_snapshot/workspace/compare_saved_region_use909.py:85`; `evidence_snapshot/sources/comparators/Signal-cd1b0a6/utils/metrics.py:95`; `evidence_snapshot/sources/comparators/Signal-cd1b0a6/utils/metrics.py:155`.

### C_result_provenance_join_and_format: WARN

All numeric/query/identity/tail/stratum fields agree with the original independent stdlib checks. All declared comparison/intake/source hashes matched. Query indices, identities and shared global reference join exactly. D2 is now CLOSED: the only follow-up report changes remove one delimiter cell on each of lines 13 and 20; every data/scope/source/SUMMARY byte remains unchanged. Binary hash strings remain consistent producer assertions, not independent rehashes from binaries.

Evidence: `evidence_snapshot/workspace/compare_saved_region_use909.py:41`; `evidence_snapshot/workspace/compare_saved_region_use909.py:63`; `evidence_snapshot/workspace/comparison909/SUMMARY.json:4`; `evidence_snapshot/workspace/comparison909/SUMMARY.json:504`; `evidence_snapshot/workspace/comparison909/SUMMARY.json:9649`; `evidence_snapshot/workspace/comparison909/REPORT.md:13`; `evidence_snapshot/workspace/comparison909/REPORT.md:20`; `followup_snapshot/REPORT.md:13`; `followup_snapshot/REPORT.md:20`; `followup_snapshot/DOCUMENT_DELIMITER_FIX.json:5`.

### D_invoked_code: WARN

The preserved comparison source has only stdlib imports and invokes stats for full rows and both strata; its SHA matches the saved SUMMARY and its render order explains the preserved initial report. The four original endpoint reports have saved command arrays, exit0 and matching stdout, tracing main -> paired -> camera_scores/scene_scores. No separate comparison909 command/exit receipt is among the supplied artifacts, so source/output consistency is not independent execution attestation. No comparison producer or prior reporter was rerun.

Evidence: `evidence_snapshot/workspace/compare_saved_region_use909.py:78`; `evidence_snapshot/workspace/compare_saved_region_use909.py:85`; `evidence_snapshot/workspace/compare_saved_region_use909.py:97`; `evidence_snapshot/workspace/compare_saved_region_use909.py:103`; `evidence_snapshot/workspace/results908/logs/region_use_resume906/CPU_REPORTS_908.json:3`; `evidence_snapshot/sources/tools/report_region_use_endpoint.py:71`; `evidence_snapshot/sources/tools/report_prepool_dense_correspondence.py:18`.

### E_scope_selection_and_claims: WARN

Exactly four completed endpoint receipts: two datasets x two conditions x one seed42 x 50 epochs. Latest maximal official mAP selects epoch 8/8 for RGBNT201 and 38/38 for MSVR310; every CMC comes from that same selected checkpoint. Official query/gallery metrics were used each epoch and the benchmarks were consumed in development. Global-rank1 strata are descriptive post-selection groups, not training ambiguity weights or a causal partition. No evidence for RGBNT100/full cohort, significance/stability/equivalence, teacher, a new inference comparator, same-model decomposition, unique failure cause or binary replay.

Evidence: `evidence_snapshot/workspace/comparison909/REPORT.md:3`; `evidence_snapshot/workspace/comparison909/REPORT.md:24`; `evidence_snapshot/sources/tools/run_foundation_recipe.py:248`; `evidence_snapshot/sources/tools/run_foundation_recipe.py:295`; `evidence_snapshot/sources/refine-logs/region_use_supervision_v1/EXPERIMENT_PLAN.md:15`.

### F_evaluation_type: PASS

Saved GT-scored retrieval evidence, reused in a descriptive comparison. It is not synthetic proxy GT, a new neural evaluation, a mathematical-causality experiment or independent binary/image replay.

Evidence: `evidence_snapshot/sources/tools/run_correspondence_roles.py:88`; `evidence_snapshot/sources/tools/report_prepool_dense_correspondence.py:51`; `evidence_snapshot/workspace/compare_saved_region_use909.py:99`.

## Formatting finding and retained source

**D1: original source renderer defect; output row placement corrected, original source preserved.** The loop at `evidence_snapshot/workspace/compare_saved_region_use909.py:107` emits an overview row and then a stratum section before the next overview row. The initial MSVR310 overview row therefore appears at `evidence_snapshot/workspace/comparison909/REPORT_INITIAL_RENDER.md:15` inside the first stratum table. Keeping this historical source unchanged is appropriate; do not rerun or rewrite it to conceal the original render.

**D2: CLOSED after targeted document verification.** The initial snapshot at `evidence_snapshot/workspace/comparison909/REPORT.md:13` and `:20` retains five delimiter cells for four-column stratum tables. The corrected `followup_snapshot/REPORT.md:13` and `:20` remove one cell on each line. All three current tables now have matching header/delimiter/data widths: 6, 4 and 4. The only report edits were those exact two cell removals; no content, data, scope or line-ending changes occurred.

`DOCUMENT_DELIMITER_FIX.diff` records the exact follow-up diff. `TARGETED_DOCUMENT_CHECKS.json` reports **16/16 PASS**: initial WARN and pre-fix bytes preserved; original render unchanged; SUMMARY and executed source byte-identical; exact edits at lines 13/20; table widths correct; final report equals the original render with only the MSVR310 overview-row move; all receipt/SHA links valid; and all eight follow-up snapshots verified. The current report SHA256 is `63d8e14a1337693a3c4c2e2fcde5be3cd33a5db917c0b527213d201b34d9021e`.

The **original 190-check run remains preserved unchanged** in `initial_review/DETERMINISTIC_CHECKS.json`: 187 PASS and 3 related layout FAIL checks for D2. Those historical FAIL results are not relabeled or rerun. `initial_review/INTEGRITY_REVIEW.md` and `.json` retain the initial WARN, and the original 60 snapshots remain intact. `DETERMINISTIC_CHECKS.json` is now an index linking that original run and the successful targeted follow-up, with current status PASS. The overall semantic verdict remains WARN for the disclosed evidence/scope boundaries; `blocking_findings` and open document findings are empty.

Five additional control-provenance checks, completed from preserved text before the targeted follow-up request, also pass (`CONTROL_PROVENANCE_SUPPLEMENT.json`): the declared control-seal digest and all four shared-global metric references. Their CMC differences are below 1e-5 points because the sealed author values use float32. These checks are separate from the 16 document checks and did not rerun the query aggregation or production evaluator.

## Provenance and claim ceiling

RGBNT201 protocol has all 836 query/gallery entries and 30 query identities, with 7-21 legal positives and 815-829 legal gallery entries per query. MSVR310 has all 591 queries, 1,055 gallery entries, 52 query identities and 155 gallery identities, with 1-31 legal positives and 1,047-1,054 legal gallery entries per query. All metadata paths/identities/environments and query orders agree; physical files and rankings remain uninspected.

All four receipts record seed42/full50, latest-maximum official mAP selection, and the same selected checkpoint for every CMC. Official query/gallery metrics enter selection at every epoch (`evidence_snapshot/sources/tools/run_foundation_recipe.py:248`); the strict final pass reloads that selected checkpoint (`:295`). This is not an untouched test split. The saved campaign witness still has unfinished RGBNT100 jobs; this review makes no current campaign completion inference.

The two global-rank1 strata are based on the same **independent global model** reference across conditions; they are not each candidate's own-global decomposition or its per-relation ambiguity training weights. Their sum is an arithmetic identity, not a causal explanation. Nested transition fields inside each stratum use that stratum's denominator; the outer stratum contribution alone uses the full dataset denominator. Unweighted identity macros can differ from query means because query counts vary.

The review supports the exact descriptive saved query comparison. It does not establish mathematical-causality, unique failure mechanism, significance/stability/equivalence, a stronger teacher, an inference comparator, all-three/full-cohort results, or independent binary/image/runtime replay. Prior review908 is read for provenance boundaries rather than substituted for this audit. Only the relevant copied sources are checked against their declarations; the entire inherited 463-file source inventory is not recertified.

Original executed source SHA256: `fae60622b5b00cf1111ba56cb452d8b5a79cda802b6fd55b36e34165f4da0e8b`.
Comparison SUMMARY SHA256: `0c18f2968adae23366cb4c6dad65ad7831865df0dee65177d362f20915caf115`.
Initial render SHA256: `40ad7bbdd261e879c5254b9d8fb262c60a7834734cab7ef20cf249e0ef2a5176`.
First formatted report SHA256: `c3eb5050ed9f90432da32d0593c33e031e2affda9de342cc4a1768eaa7d92aa6`.

Private audit helper path corrections and retained pre-correction tool artifacts are documented in `AUDIT_TOOL_NOTES.md`. All 60 authoritative snapshot paths are unique and each copied digest matches the original audited digest. This repaired audit storage only; numeric checks were not rerun.

The follow-up request is preserved verbatim in `REVIEW_REQUEST_FOLLOWUP_ACTUAL.txt`; its full response and honest UNATTESTED model/effort attribution are recorded as trace call 002. `FOLLOWUP_INPUT_MANIFEST.json` identifies the eight final/retained document/source versions separately from the initial manifest.
