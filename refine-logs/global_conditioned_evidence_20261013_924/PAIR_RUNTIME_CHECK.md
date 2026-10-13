# Paired saved-record runtime check

Paired saved-record evidence accepted: **true**. Unresolved blocking evidence defects: **none**. Overall WARN for bounded assurance; deterministic stored-record arithmetic PASS.

Requested gpt-6-astra/max; actual model/effort UNATTESTED; same-family/provisional.

## source_and_all_input_pins - PASS

Actual paired summary source SHA equals current reviewed source and PAIR_SOURCE_CHECK. Its exact11 input-file hashes equal both accepted endpoint summaries, two actual CPU EXITs, two step/two batch files, terminal observation and both accepted runtime audit files. All9 artifact pins also still equal explicit C-path accepted-audit entries. D-resolved input path aliases resolve to the same files; no fallback or audit key substitution occurred.

Evidence: `build_evidence_pair929.py:12-32,110`; `EVIDENCE_PAIR_SUMMARY_929.json (source/input lines in artifact_line_references)`; `PAIR_SOURCE_CHECK.json:1`; `RUNTIME_AUDIT.json:1`; `SECOND_RUNTIME_AUDIT.json:1`.

## saved_query_arithmetic - PASS

Independently recomputed all836 aligned query records, real control/candidate mapping random to gradient, endpoint mAP/CMC, official deltas, strict-sign AP counts, Rank1 repairs/harms, four group counts/contributions normalized by836, all30 identity means/macro and lowest42-query mean. All match stored output. This is saved-record arithmetic, not sorting of distance tensors.

Evidence: `EVIDENCE_PAIR_SUMMARY_929.json (metric/query/group/identity lines in artifact_line_references)`; `build_evidence_pair929.py:50-82,98-105`.

## recorded_order_support_scalars - PASS

Two training-order files are byte-equal;2649 paired step indices/LRs and recorded support/activity/exclusion fields agree. Teacher scalar maximum differences recomputed from stored records equal output. Equality of these scalar records does not establish same-model global-vector equality.

Evidence: `evidence_gradient_results928/output/training_steps.jsonl:1-2649`; `evidence_random_results929/output/training_steps.jsonl:1-2649`; `build_evidence_pair929.py:84-95`.

## execution_and_independence_boundary - WARN

Parent reports original native507ac9 EXIT0 and one successful stdlib combination. Reviewer did not independently consume that native exit transcript or execute the script; actual produced file, reviewed source and input binding, and independently recomputed arithmetic support bounded artifact acceptance. No new tensor ranking, NN or optimizer update. One seed/dataset; no causal/robustness/equivalence or full-goal claim.

Evidence: `EVIDENCE_PAIR_SUMMARY_929.json (status/zero-call/boundary fields)`; `build_evidence_pair929.py:106-115`; `Parent task message: native507ac9 EXIT0`.

## Exact produced-record references

- EVIDENCE_PAIR_SUMMARY_929.json: metrics: 8
- EVIDENCE_PAIR_SUMMARY_929.json: delta_metrics: 22
- EVIDENCE_PAIR_SUMMARY_929.json: official_delta_metrics: 28
- EVIDENCE_PAIR_SUMMARY_929.json: rank1_repairs: 34
- EVIDENCE_PAIR_SUMMARY_929.json: rank1_group_ap_contributions: 36
- EVIDENCE_PAIR_SUMMARY_929.json: query_changes: 54
- EVIDENCE_PAIR_SUMMARY_929.json: query_ap_improved: 6744
- EVIDENCE_PAIR_SUMMARY_929.json: identity_changes: 6746
- EVIDENCE_PAIR_SUMMARY_929.json: identity_macro_mean_delta_ap_points: 6898
- EVIDENCE_PAIR_SUMMARY_929.json: worst_five_percent_query_ap_delta_mean_points: 6899
- EVIDENCE_PAIR_SUMMARY_929.json: recorded_teacher_max_abs_difference: 6903
- EVIDENCE_PAIR_SUMMARY_929.json: source_sha256: 6914
- EVIDENCE_PAIR_SUMMARY_929.json: input_sha256: 6915
- EVIDENCE_PAIR_SUMMARY_929.json: boundary: 6928

## Audited bytes

| File | SHA256 |
|---|---|
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/EVIDENCE_PAIR_SUMMARY_929.json` | `946277cf7799f39f748b5c767e2da4588ac447ff36647406f58568b59b61c6c7` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/build_evidence_pair929.py` | `639319f4725a05d0fc685a4fd5526f55d27abff1db069dba2329ba05ad55f529` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_endpoint_report928/PAIR_SOURCE_CHECK.json` | `b69c4d8b56b45b905beff68b80bbe8d64efd8a7f8a47d0b6d489774acc4934eb` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_endpoint_report928/RUNTIME_AUDIT.json` | `6a6416adfc9ed7dffe74c0726d4dce5318714957f5126d1fc1339f76f8174840` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/EVIDENCE_GRADIENT_CPU_SUMMARY_928.json` | `511f6e00aa0c4df2f464170dc1064d727cc851d351764a60bdacee3b92a8b110` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/EVIDENCE_GRADIENT_CPU_EXIT_928.json` | `b75968b009cc064edc2988a84c8ddce28fb56b0e0ef97ea2b6ba48e08e25669e` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_gradient_results928/output/training_batch_order.jsonl` | `e446282654c74949fb594856631b246642593b73229331c30b555418a0358993` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_gradient_results928/output/training_steps.jsonl` | `c93175b6e796763bb62ab7befb7bd94e129038ab1be73cd7234aa9cf9addfd96` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/review_endpoint_report928/SECOND_RUNTIME_AUDIT.json` | `9418e94fa3948c247935ee2552ad1c07ded5de35c93715725b76b0ffeea6efaf` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/EVIDENCE_RANDOM_CPU_SUMMARY_929.json` | `57921c4e11bbd7b18219513cd95e83247f8125516fcf1307d3252a56819e5c28` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/EVIDENCE_RANDOM_CPU_EXIT_929.json` | `1bdad2969ff4e3ab610bdf72a577f2dcd37e316de35496868eae9f3d7adaa3c5` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_random_results929/output/training_batch_order.jsonl` | `e446282654c74949fb594856631b246642593b73229331c30b555418a0358993` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_random_results929/output/training_steps.jsonl` | `15dd311c7377d23e5be17aa6b70b2ebde859e583b0a025ebcade59a74c8998ec` |
| `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/EVIDENCE_TERMINAL_OBSERVATION_929.json` | `ea0cd31693fdd50416a9af1c159c7f536ff79c82b475c412c94fb9db10f44978` |

## Limits

- No paired script/report/model/torch execution, SSH, tensor load, publication or production mutation by reviewer.
- No fresh dataset or tensor-ranking replay; per-query evidence starts from accepted stored AP/first-rank records.
- Original native507ac9 exit is parent-reported; independent review covers actual supplied artifact and deterministic arithmetic.
- No same-model G/C/F, teacher-vector equality, causal localization, seed robustness, equivalence, all-Signal superiority or full-goal acceptance.
