# Fixed-global difficulty 914: bounded experiment audit

**Verdict: WARN. Blockers: 0. Required corrections: none.**

All 57,422 deterministic checks passed with zero mismatches. This supports the saved-query description and its report within the limits below.

Requested reviewer: `gpt-6-astra`, reasoning `max`. Actual service model and effort: **UNATTESTED**. Native fresh reviewer `/root/audit_fixed_difficulty914`; **same-family / provisional**.

The CPU verifier used the supplied Python 3.13 interpreter with `-X utf8 -B`, independent 50-digit Decimal sums and an index-keyed query join. It did not execute/import the producer. Its real command, timestamps, hashes and exit 0 are in `RUN_01.json`; output is in `RUN_01.log`.

| Dataset | Aligned queries | Global first wrong | A-U overall mAP points | A-U repairs/new errors |
|---|---:|---:|---:|---:|
| RGBNT201 | 836 | 176 | -0.158148058 | 6/5 |
| RGBNT100 | 1715 | 58 | +0.060151674 | 15/11 |
| MSVR310 | 591 | 189 | +0.003982602 | 0/0 |

The 6,284 endpoint records form 3,142 aligned dataset-query keys. All 12 strata, 6 pooled groups, 162 numeric result fields, 186 exact fields and 18 report rows agree. The new result has eight valid input pins; the six summaries and accepted aggregate also match prior-audit hashes.

Within-bin delta = sum of AP changes in percentage points / bin size. All-query contribution = the same sum / total query count. Thus contribution = within-bin delta × bin size / total size. Add the four strata or the two pooled groups, never both partitions together. Rank-1 repair/error counts use the comparator named by each contrast.

The maximum derived numeric-field discrepancy is 1.08e-14 points (arithmetic tolerance 1e-10). Integer-count R1 deltas differ from accepted float32 CMC deltas by at most 2.79e-6 points, below the existing 1e-5 bound. No tolerance was widened.

## A_ground_truth_provenance: WARN

The new analyzer reads saved independent_global per-query identity, AP and first-positive-rank fields. It creates no pseudo-GT. These ranks are GT-dependent post-development descriptors. The prior bounded audit classifies the original scoring path as real_gt but explicitly did not re-authenticate raw benchmark annotations or distances. This review inherits that provenance limit and does not claim a new dataset/raw-ranking certification.

Evidence: D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:32; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:39; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:43; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\six_results913\AUDIT.json:25; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\six_results913\AUDIT.json:102; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\FIXED_GLOBAL_DIFFICULTY_914.md:3.

## B_score_normalization: PASS

Within-bin AP differences divide the sum of percentage-point AP changes by the bin count. Contributions to total mAP divide the same numerator by the full dataset query count. There is no prediction maximum/minimum/mean normalization. Independent 50-digit Decimal sums verify both denominators, every mean, and count/N times within-bin delta equals all-query contribution. The four strata form one partition and the two pooled groups form a separate partition; do not add all six rows together.

Evidence: D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:51; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:58; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:59; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:63; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:64; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\fixed_difficulty914\CHECKS.json.

## C_result_existence_and_arithmetic: PASS

All 11 supplied input artifacts and the request exist and were read as UTF-8. All eight producer input pins match; all six SUMMARY pins match ALL_SIX_RESULTS_912 and the prior audit, and the aggregate hash matches the prior audit. The input-set check plus these 21 hash comparisons comprise 22 pin checks. All 6,284 endpoint query records are valid, with unique contiguous indices; 3,142 dataset-query keys align exactly across conditions. Reconstructed all 12 strata and 6 pooled groups, including identity counts, all AP means/deltas/contributions and every Rank-1 repair/new-error count. All 18 Markdown table rows match exact four-decimal rendering. Recombined totals match the accepted all-six result deltas.

Evidence: D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\FIXED_GLOBAL_DIFFICULTY_914.json:4; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\FIXED_GLOBAL_DIFFICULTY_914.json:516; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\ALL_SIX_RESULTS_912.json:570; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\ALL_SIX_RESULTS_912.json:28957; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\six_results913\AUDIT.json:192; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\FIXED_GLOBAL_DIFFICULTY_914.md:7; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\fixed_difficulty914\RUN_01.json.

## D_executed_path_and_dead_code: PASS

Source was read line by line and AST-parsed. Both rank_bin and summarize are called in the result-construction path. The independent verifier imported only standard-library modules and never executed or imported the producer. The verifier actually completed with exit 0. Original producer execution chronology is not separately attested by this bounded audit; support is exact source/result pins plus independent reconstruction.

Evidence: D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:17; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:47; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:49; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:69; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:71; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:88; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\fixed_difficulty914\RUN_01.json.

## E_scope_and_wording: WARN

The report correctly says post-development, all original selected-best queries retained, identity counts nonadditive, and within-bin means distinct from global contributions. It excludes train-time margin/weight, GT deployment selection, multiple-seed uncertainty, same-model decomposition, stronger-teacher evidence and a unique failure cause. Fixed independent-global strata describe comparisons among original selected-best models; they cannot attribute a same-model branch effect or establish a training mechanism. Scope remains three datasets and one original seed42 per condition, using development-consumed benchmarks. No wording correction is required.

Evidence: D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:90; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:91; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\FIXED_GLOBAL_DIFFICULTY_914.md:1; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\FIXED_GLOBAL_DIFFICULTY_914.md:3; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\ALL_SIX_RESULTS_912.json:28970.

## F_evaluation_type: WARN

Saved-query description derived from dataset-GT evaluation records, with inherited source/receipt provenance. Independent-global AP is a comparator, not fabricated ground truth. Dataset annotations, gallery rankings and raw distance tensors were outside this review. Consequently real_gt describes the declared and previously inspected scoring path, not new raw-data authentication.

Evidence: D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:32; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\analyze_fixed_global_difficulty914.py:39; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\six_results913\AUDIT.json:25; D:\Program Files\UserCache\gb\codex_tmp\independent_evidence_draft\evidence_comparison903\review\six_results913\AUDIT.json:102.

## Claim boundaries

- Fresh native same-family review; acceptance provisional. gpt-6-astra/max were requested; actual service model and actual reasoning effort are UNATTESTED.
- Only supplied local analyzer, derived JSON/Markdown, accepted all-six aggregate, prior audit record and six saved SUMMARY files were examined. No full earlier 59-file audit replay.
- No SSH, network, GPU/NN work, optimizer/training, model or candidate change, producer execution/import, raw-distance or raw-annotation read, or source/result mutation.
- GT provenance is inherited from the prior bounded source/receipt audit; no independent authentication of raw benchmark labels, gallery contents, rankings or remote binaries.
- One original seed42 per condition; selected-best checkpoints and development-consumed benchmark records. No statistical significance, multiseed uncertainty or untouched-test generalization claim.
- Difficulty bins are post-development GT-based fixed independent-global ranks. They are not actual train-time margin/weights, a deployable selector, same-model causal decomposition, stronger-teacher evidence or a unique failure explanation.
- Four strata and two pooled groups are alternate partitions of the same queries. Identity counts may overlap across strata and are not additive.
- Original official CMC uses float32 percentages. Count-based Rank-1 deltas differ slightly from subtracting those percentages; the existing absolute 1e-5 percentage-point tolerance was retained, not widened.

## Actual execution and preservation

- `CHECKS.json` contains category counts, precise recomputed values, every stratum membership, totals, input hashes and zero failures.
- `RUN_01.json` / `RUN_01.log` preserve the actual deterministic command and exit 0.
- `CHECKS_FINAL.json` records final unchanged hashes for the request and all 11 supplied files plus audit artifact hashes. `WRITE_AUDIT.json` / `.log` record the artifact-writing command and exit.
- No failed invocation occurred in this audit. Prior audit failure artifacts were not modified or rerun.
- Writes are limited to this private review directory; SSH/GPU/NN/optimizer/training/source-result mutation counts are zero.
