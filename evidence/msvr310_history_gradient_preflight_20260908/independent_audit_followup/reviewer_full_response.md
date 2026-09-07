**Remaining verdict: WARN, same-family/provisional.** The added evidence closes the specific missing-statistics verification action and the stale tracker action. It does not close the remaining limits on parameter-gradient reconstruction, the 14-component total loss, or multi-group direct-graph coverage. I found no new implementation error or numerical inconsistency in this follow-up.

I saved **127 passed local checks, with no failures**, separately in [followup_checks.json](C:/Users/gb/.codex_tmp/history_gradient_preflight_independent_audit_20260908/followup_checks.json). Its SHA-256 is `cb6f843fae8ae02618b42044f1e8635cdff4dc9e04bb82f4f17cc78f3db413a8`. The original audit files were not modified.

**Closed: all thirteen producer statistics now have an implemented CPU reconstruction path.**

The new verifier correctly reproduces the original producer’s definitions:

| Statistic group | New verifier evidence |
|---|---|
| Historical record and positive/negative pair counts | `tools/verify_msvr_history_gradient_all_statistics.py:70`, `:71` |
| Cross-scene positive pairs | `:72` |
| Negative violations relative to current hard positive plus margin | `:73` |
| Harder positive/negative anchor counts | `:74` |
| Current and expanded wrong-order anchor counts | `:75`, `:76` |
| Current and expanded Triplet means | `:77`, `:78` |
| Expanded positive-hinge anchors | `:79` |
| Maximum historical age | `:80` |

The code asserts that the computed and recorded key sets match and contain exactly thirteen fields, then checks every field for every row. Integer counts require equality; the two Triplet scalars use the existing `2e-6` numerical tolerance: `:81`, `:82`, `:84`, `:87`. The explicit `np.float32(.3)` margin at `:68` is consistent with the producer’s FP32 distance/hinge computation.

The three previously omitted matrix-derived counts—`current_wrong_order_anchors`, `expanded_hinge_positive_anchors`, and `memory_negative_violations_against_batch_hard_positive`—are therefore covered. The four previously omitted metadata-derived fields are covered too.

The verifier also binds its inputs adequately for this additive postcheck:

- Protocol, configuration, summary status, and original CPU-to-summary binding are checked at `:21`, `:22`, `:23`, `:24`, `:25`.
- Each state’s receipt must equal the summary entry, and every receipt-listed input file must match its size and SHA at `:31`, `:34`.
- It reads the saved float32 matrices at `:36`, checks every offset and shape at `:51`, checks finite entries at `:54`, and requires complete consumption of each matrix file at `:91`.
- The final counts must match the original CPU receipt at `:95`.
- The entry point actually calls `verify` at `:113`.

This is a separate script reading the existing preflight files and writing a new result. The code does not import Torch, execute model inference/backward, update parameters, or write to the input matrices or original receipts.

**Closed: the added execution receipt and result are consistently bound to the original audited preflight.**

I independently verified these exact local hashes:

| Artifact | SHA-256 |
|---|---|
| New verifier | `bb9c078f7800c2d5255c5922df68ed5072e6e93931347cb185b7dcce44987026` |
| Added result, 7,094 bytes | `434beb27dc6349a2d62ea64dedfa0efa17b41cfd2db8b11b7383a47923fd509b` |

The verifier hash matches both `execution_binding.json:16` and `all_statistics_postcheck.json:8`. The result size/hash match `execution_binding.json:17` and `:18`. The binding records exit code 0 and empty stderr at `:15` and `:19`; its command points to the separate transport script and the original preflight directory at `:5` and `:7`.

The result’s summary, original CPU, configuration, and protocol hashes all match the hashes preserved by my initial audit: `all_statistics_postcheck.json:4`, `:5`, `:6`, `:7`. Its stdout agrees with the result’s **72 batches, 625,920 distance elements, and 936 statistic checks**.

I also checked every state’s aggregate structure and bounds. The summed historical-record counts are **547, 519, and 658 per state for folds 0, 1, and 2**, respectively; these agree exactly with the original audited per-state matrix sizes. Positive and negative pairs partition all historical distances. Cross-scene counts, violation counts, anchor counts, and loss monotonicity are consistent throughout.

The `maximum_memory_age: 15` values in this new result are correctly nested under `sum_of_per_batch_statistics`: they mean \(0+0+0+1+2+3+4+5\), not a maximum age of 15. The revised report continues to give the actual preflight maximum as 5.

**The evidence boundary remains important:** the **936 checks are attested by the bound remote CPU execution receipt**. My local follow-up verified the implementation, hashes, recorded execution, coverage totals, and consistency of the aggregate values. I did not access or locally reconstruct the nine remote matrices. The new result states this narrower computational scope correctly at `all_statistics_postcheck.json:194`, and the revised report identifies the computation as a remote CPU postcheck at `results/MSVR310_HISTORY_CANDIDATE_GRADIENT_PREFLIGHT_2026-09-08.md:88`.

**Closed: original results and the original audit judgment are preserved.**

The original **7,978-byte report is an exact byte prefix of the revised report**. All **36 original numerical table rows—nine engineering rows and twenty-seven role rows—are unchanged**, so the original full-table arithmetic validation still applies.

The preserved original report and tracker match their hashes in the initial audit. The preserved `independent_checks.json` is byte-identical to the original audit artifact, retaining SHA-256:

`81b4a93c05c81cc51e2450100bd1fd0970778bc2c652b8d89f93b589b775074e`

The saved full reviewer response retains the A–F findings, WARN verdict, 19,782-check result, and action/claim-impact discussion. The revised report accurately describes this preservation at `:86`; it does not rewrite the original judgment as an unconditional PASS.

**Closed: the tracker now distinguishes completed preflight from the ongoing source measurement.**

The revised tracker records:

- Preflight complete: `EXPERIMENT_TRACKER.md:6`.
- Original preflight CPU complete: `:7`.
- Separate statistics postcheck complete: `:8`.
- Independent audit remains WARN: `:9`.
- Source-stage process reported running: `:10`.
- Source CPU not started: `:11`.

The earlier “preflight running/source not started” observations are now explicitly under the historical-observations label at `:17`. This resolves the chronology inconsistency identified in the initial review.

The specific **05:55:39 process/GPU/free-space observation** at `:15` has no corresponding observation artifact among the supplied follow-up paths. I therefore verified the corrected stage separation, but not those newer live process/resource details. If that exact snapshot is described as independently checked, its saved observation should be included in the evidence set.

**Open: the remaining substantive limits are correctly retained.**

The revised report’s `:90` explicitly preserves the limitations that this new CPU postcheck cannot address:

1. **The 14-component total loss cannot be independently reassembled.** Its individual components were not saved. Verifying both Triplet statistics does not reconstruct the other task components.
2. **Parameter gradients, exact pixel/re-encoding equality, and state equality remain runtime witnesses.** Matrix-statistic reconstruction is not a model backward reconstruction.
3. **The real direct-graph proof remains one single-history-group witness at step 4 per state.** Later 2–5-group VJP executions have no additional direct-model comparison.
4. **The short preflight does not exercise queue expiry, capacity eviction, or zero-upstream group skipping.** The full source records must establish any later branch coverage.
5. **The full source stage is not established as complete by this follow-up package.** Its final complete-scope result and CPU receipt remain separate deliverables.

The report also correctly retains the local CRLF/remote LF distinction and the difference between no new heldout-image forwards and no heldout-artifact access at `:92`. Its rotation-example qualifier at `:94` remains appropriate.

The precise updated claim is: **all thirteen logged memory statistics now have a complete, hash-bound remote CPU postcheck for the existing 72-batch preflight, while the original gradient tables remain unchanged and arithmetically verified.** This closes the identified statistics-coverage defect without establishing independent parameter-VJP reconstruction, full-source completion, or any training/retrieval gain.