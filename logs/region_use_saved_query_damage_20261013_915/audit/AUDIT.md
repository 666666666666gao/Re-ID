# Saved-query damage915 experiment audit

**Verdict: WARN. Blockers: 0. Required corrections: 0.**

Fresh same-family review; acceptance is provisional. Requested model/effort: gpt-6-astra / max. Actual service model/effort: UNATTESTED.

The independent verifier actually completed with exit 0: **72,244 checks, zero mismatches**. It read the source, saved inputs and reports directly and rebuilt the arithmetic without importing or rerunning the producer.

| Dataset | Condition | Queries / identities | Query mAP delta | Equal-identity macro AP delta | Identities gain / loss / unchanged |
|---|---|---:|---:|---:|---:|
| RGBNT201 | uniform | 836 / 30 | +0.4913853394 | +0.4512633715 | 20 / 8 / 2 |
| RGBNT201 | ambiguity | 836 / 30 | +0.3332372817 | +0.2729757445 | 23 / 5 / 2 |
| RGBNT100 | uniform | 1715 / 50 | -1.2480219903 | -0.7623997711 | 21 / 27 / 2 |
| RGBNT100 | ambiguity | 1715 / 50 | -1.1878703161 | -0.7668455380 | 22 / 26 / 2 |
| MSVR310 | uniform | 591 / 52 | -0.0003514773 | -0.0085116071 | 25 / 16 / 11 |
| MSVR310 | ambiguity | 591 / 52 | +0.0036311250 | -0.0052317434 | 27 / 15 / 10 |

Verified 3,142 distinct dataset-query keys / 6,284 endpoint records, 22 observed Rank-1 groups, all 264 endpoint identity rows, 28 analysis-report rows, six draft table rows and the draft numeric prose. All 16 supplied/request artifacts were hashed. MSVR310 has no observed repair group under either condition; no empty-group means were fabricated.

Exact rational accumulation independently checked the declared binary64 per-query AP differences. AP arithmetic tolerance was 1e-10 percentage points; maximum output-field discrepancy was 1.79e-14 points. Signs and counts used exact comparisons without an epsilon. The existing 1e-5 tolerance for accepted float32 CMC was retained; maximum discrepancy was 2.824e-6 points.

**A_ground_truth_provenance: WARN** — The analyzer consumes saved identity, AP and first-positive-rank records from the independent_global pair and creates no substitute labels. Independent-global AP is the comparator, not generated ground truth. Dataset-GT provenance is inherited; raw annotations and rankings were not re-authenticated.

**B_normalization_and_denominators: PASS** — Independently reconstructed every nonempty group and identity mean, full-N contribution, exact sign count, positive-only gain and negative-only loss. Checked within-group mean times group-count/full-N equals the contribution, both partitions sum to the overall delta, all identity means/macro/counts match, and the largest identity fraction uses the gross-negative denominator. No prediction-dependent normalization.

**C_results_alignment_and_reconstruction: PASS** — All producer input pins match, with SUMMARY/aggregate pins chained to accepted earlier artifacts. All 6,284 query records are valid and all 3,142 dataset-query keys align across conditions and the accepted A-minus-U records. Reconstructed all six outputs, 22 observed groups and every field in 264 identity rows. Totals/groups/identity means agree with saved SUMMARY and accepted all-six outputs; pooled results agree with the fixed-global difficulty analysis.

**D_executed_path_and_receipts: PASS** — Read the 129-line producer and AST-parsed it without importing/executing it. Its summarize function is called for whole endpoints, groups and identities; imports and text I/O are standard-library/explicit UTF8. The independent rational verifier actually completed with exit0. Historical executor receipt records exit0/chunk e5cb09; canonical source/input/output aliases match, but historical tool-event/original-stdout authentication is outside this fresh reviewer context.

**E_report_and_scope: WARN** — All 28 analysis table rows and 6 publication table rows match exact four-decimal rendering, and the draft numeric prose reconstructs. The boundaries correctly distinguish exact signs from significance, query mAP from identity macro, gross-negative share from net delta, and fixed independent-global comparison from same-model causality. MSVR repair groups are explicitly absent. Historical Signal/prior-objective and live operational statements are not newly certified. No correction is required for the bounded arithmetic or its descriptive wording.

**F_evaluation_type: WARN** — Post-development saved-query description from declared dataset-GT evaluation records. This is neither a new benchmark run nor synthetic-proxy GT. The real_gt label describes the inherited scoring provenance, not a new raw-data certification.

The RGBNT100 ambiguity still-correct group contributes -0.9127381844 mAP points despite retaining Rank-1. Identity 542 contributes 7.7283% of all negative AP changes, using gross loss rather than net mAP as denominator. The group and identity partitions each reconstruct the total separately.

**Boundaries**

- Fresh native Codex same-family review; acceptance provisional. gpt-6-astra and max were requested. Actual service model and reasoning effort are UNATTESTED.
- Read 15 supplied source/result/report/input/prior-audit/receipt/draft artifacts plus REQUEST.md; all 16 hashes recorded. Six saved SUMMARY files, accepted aggregate and fixed-global pooled results were the only numeric inputs.
- Exact scope: 3 datasets, 2 original seed42 selected-best conditions each, 3,142 distinct dataset-query keys / 6,284 endpoint query records, 22 observed Rank-1 groups, 264 endpoint identity rows, 28 analysis-report rows and 6 publication-draft table rows.
- No SSH, network, GPU, NN, optimizer/training, candidate/model change, producer import/execution, source/result mutation, raw-distance/annotation access, or replay of the earlier 59-file/463-source audit.
- Declared dataset-GT provenance is inherited from accepted saved-query records and the supplied prior audit. This review does not re-authenticate raw labels, gallery membership, rankings, checkpoints, remote binaries or official evaluation implementation.
- These are post-development descriptions of one original seed per endpoint, using each original selected-best model and development-consumed benchmark records. Exact sign counts are not statistical significance or multiple-seed/generalization evidence.
- The comparator is a fixed independent global model, not the global branch of the same candidate. No causal branch attribution, unique internal failure cause or deployable GT-based gate follows.
- Query mAP gives equal weight to queries. The identity macro gives equal weight to identities after averaging within identity. They answer different descriptive questions and neither sign supersedes the other.
- Gross loss sums only negative query AP changes. The largest-loss-identity fraction divides its negative-query loss by all negative-query loss, not net mAP and not the count of negative queries. It is not the identity with necessarily the lowest net mean.
- Exactly 22 nonempty groups exist: MSVR310 repair is absent for both conditions. No empty-group mean or completed zero-sample outcome was fabricated. Identity counts may overlap across Rank-1 groups and must not be added across those groups.
- The draft references prior repair/keep and multi-positive trials, the complete Signal comparison, and current training/device policy/state. Those historical and operational clauses are inherited context, not established or freshly certified by these supplied saved-query artifacts. No live-state check was performed.
- The original producer receipt is executor-supplied recorded history; its chunk/exit record was read and paths checked, but this reviewer cannot independently attest that earlier tool event or reconstruct unrecorded exact start/finish timestamps. The fresh verifier command, output, script hash and true exit0 were directly observed and preserved.

**Execution and artifacts**

- Independent verifier: [verify_damage915.py](verify_damage915.py)
- Complete counters, comparisons, hashes and reconstructed identities: [CHECKS.json](CHECKS.json)
- Actual command/start/finish/exit receipt: [RUN_01.json](RUN_01.json)
- Actual combined output: [RUN_01.log](RUN_01.log)
- Structured audit: [AUDIT.json](AUDIT.json)

Native tool invocation: exec_command chunk dd393b, actual exit 0. No failed verifier attempts occurred. Existing source and result artifacts were unchanged; only this private review directory was written.
