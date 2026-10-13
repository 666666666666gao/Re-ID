# Result audit: saved evidence cosine terms

**Overall verdict: WARN, with local receipt consistency and final numerical claims PASS.** No unresolved code defect, numerical mismatch or blocker to the bounded descriptive report was found. This is a **same-family, provisional** source/receipt audit, not independent tensor reproduction.

Requested reviewer: `gpt-6-astra` with `max` reasoning. Actual model/effort: **UNATTESTED**. Audit timestamp: 2026-10-13T09:56:03.4638077Z.

SOURCE_AUDIT.md/.json remain frozen. Their source-only verdict has not been rewritten as runtime acceptance. This later report reads the supplied execution/claim artifacts directly. The reviewer performed no SSH, tensor loading, analysis execution, metric recomputation, NN/GPU/training work, installation, git mutation or production edits.

## A–F findings

| Check | Verdict | Finding and exact evidence |
|---|---|---|
| A. Ground truth/proxy provenance | PASS | Returned seal matches the protocol and prior diagnoses (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:21`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:23`); recorded counts match the local protocol checks (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:72`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:655`). Dataset identities/cameras define pairs, while correction means remain descriptive terms (`C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:62`, `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:69`). |
| B. Score normalization | PASS | Raw cosine/margin units and standard norm products; no benchmark rescaling. Query and pair weights are distinguished (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md:13`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md:26`). Copied metrics match prior diagnosis floats exactly (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:623`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:1206`). |
| C. Result existence/claims | PASS within receipt scope | Local exit and returned child exit are 0; stderr is empty. stdout equals parsed result, and request minus source equals returned seal (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_EXIT.json:2`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS.stdout:1`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/REQUEST.json:3`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:15`). All 16 numerical claim groups pass displayed-precision checks. Raw remote tensor/TERMS bytes were not independently read. |
| D. Called/dead code | PASS within receipt scope | Returned argv calls the final reviewed analysis without `-O` (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:3`); both arms populate every expected summary. The source calls the relevant functions and no model/training/retrieval evaluator. Zero counters match that path but are declarations, not independent telemetry (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:56`, `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:136`). |
| E. Scope | PASS with limits | Two E8 RGBNT201 seed42 arms; 836 identical query/gallery records, 30 identities, one seed and small error groups. The report retains non-AP/noncausal qualifications (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md:9`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md:35`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md:48`). No broader benefit or campaign-state certification follows. |
| F. Evaluation type | real_gt descriptive relation probe | The recorded type is appropriate (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:53`): real labels define relations, but the new numbers are descriptive algebra on fixed saved representations. No new model benchmark or causal intervention occurred in this path. |

## Provenance and execution receipts

The embedded REQUEST source, current reviewed source, result source field and source-stage hash all agree on:

`53ca320fbdeb7455b605ed9bb261f8a1c1211a5fd00f176705970c5175243a0f`

The request's nine artifact entries comprise the protocol, two diagnoses, four feature files and two distance files. Its two diagnosis hashes match local copies, its protocol hash matches the local protocol, and the six tensor hashes match prior diagnosis artifact fields. These last six are **receipt links**, not independent tensor-byte checks.

The supplied record has child PID **3934104**, remote start **2026-10-13T17:46:55.365887+08:00**, remote finish **2026-10-13T17:46:59.004805+08:00**, exit **0** (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:12`). The local SSH EXIT timestamp is **2026-10-13T17:46:48.444933+08:00** (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_EXIT.json:3`), earlier than both remote times. Thus no common wall-clock chronology is established. The final narrative now labels host times and rejects cross-host cost inference (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md:56`).

The program reports **1.712330429116264 s** via its remote perf_counter (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:1229`). Its timer starts after import/setup (`C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:127`), so this is not end-to-end latency or a benchmark. CUDA hiding and four CPU threads are supported by the reviewed wrapper/source (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/run_saved_terms.py:49`, `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:125`), not by independent environment telemetry.

Returned remote hashes are TERMS `192005527c171c1e34bf24dfeaad6fffdd2bedec1257547991bcbb264423c91e` and seal `16a2c2c823cabf25f04c51baf6cf640beb8e883d29cec4695922b2734c454af6` (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:1231`). The inner/outer seal fields agree. This audit has not independently read those remote files to certify their raw hashes.

## Recorded numerical gates

| Gate | Threshold | gradient | random |
|---|---:|---:|---:|
| Raw reconstruction | 0.000001 | 4.7178134288117235e-7 | 4.7441908890277773e-7 |
| Cosine term closure | 1e-12 | 1.9965989582135246e-15 | 1.9586112245950638e-15 |
| Saved-F dot vs saved-raw cosine | 0.000001 | 1.831568747778789e-7 | 1.959488303837631e-7 |
| Global saved-distance cosine | 0.00001 | 5.960464481979955e-7 | 5.960464481979955e-7 |
| Fused saved-distance cosine | 0.00001 | 6.049036592736812e-7 | 5.775944776909014e-7 |

All reported values pass unchanged source thresholds (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:65`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:648`; `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:29`, `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:47`, `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:49`, `C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_evidence_cosine_terms.py:77`). The report's “<2.1×10⁻¹⁵” closure and “<6.1×10⁻⁷” saved-distance claims are correct (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md:7`).

“Exact” is an algebraic statement about saved raw H cosine. Saved F and saved FP32 distance parity are tolerance-qualified. The seventh term is a residual of saved FP32 raw arithmetic under the same observed H denominator, not an independently estimated mechanism.

All **144** serialized count/mean/mean-absolute/min/max records were inspected for finite values, valid positive counts, component-count agreement and scalar range consistency; no violation was found. This is JSON consistency checking, not matrix recomputation.

## Claim-by-claim result narrative audit

All 58 final report lines were reread. Its final SHA is `9f46d1f3ea15920666ab01d3da7139f19f2dba4599bca0c98337a46c155bf487`.

| Narrative claim | Audit result |
|---|---|
| E8/seed42, one recorded CPU invocation, no new model/retrieval result (line 3) | Supported for this recorded invocation by command, source path and copied result metadata. Not an exhaustive process/attempt census. |
| Observed-denominator identity and error bounds (line 7) | Supported by final source and reported closures. |
| Mean terms are not identity/noise or causal components (line 9) | Correct limit; no intervention was evaluated. |
| Counts, masks, fixed-G negative and weighting (line 13) | Supported. Counts: 11,256 positives, 675,326 negatives, 12,314 excluded. Same-camera different-identity negatives remain. |
| Seven component means plus total (lines 17–24) | Every displayed number matches the corresponding serialized scalar to its printed precision: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:322`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:905`. |
| Positive/nonpositive baseline-margin groups (lines 26–31) | Both counts and four means match: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:200`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:260`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:783`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:843`. Pair weighting is correctly distinguished. |
| Saved rank groups and their means (lines 35–42) | All counts/means match and counts agree with prior saved ranks: `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:382`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:442`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:502`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:562`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:965`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:1025`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:1085`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:1145`. |
| Prior mAP deltas and gradient−random contrast (line 44) | +0.067795624, +0.172881271 and −0.105085647 points are correct rounded copies/contrast (`C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:637`, `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json:1220`). “Not promoted” remains a workflow decision. |
| Cancellation and pair-group direction (line 48) | Descriptively supported by shown signed means. “High common energy” is a prior-context antecedent, not a new energy statistic measured here; its earlier evidence was not supplied for this stage. |
| Next-step constraints and absence of content interventions (line 50) | Policy choice; this run contains no such intervention. The absence of evidence across all prior work is not independently certified. |
| Prior global parity, broader goal/training/retention state (line 52) | Supplied §930 global scores agree with independent-global scores. New source creates no weight or cleanup operation. Live remote training state, actual retained files and the full three-dataset goal status are outside this audit. |
| PID/times, CPU setup, source/input hashes (line 56) | Supported as host-specific receipts and reviewed control-flow assertions. No independent process/environment monitor. |
| Review scope/model attribution/revisions/no retry (line 58) | Same-family/provisional/UNATTESTED scope is correct. Embedded code is the revised source. One successful wrapper pass/no retry loop is shown; unprovided historical attempts cannot be excluded by this local evidence. |

The initial report used ambiguous “warm environment” wording and omitted the host-clock caveat. Both were corrected before freezing this result audit. Its original bytes are preserved at `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT_R1_BEFORE_WORDING.md`, SHA `6b74b80b5b736ab83b393eb0b92748b0e721603745d12d77a80f9dc32c1c3f44`. No analysis, request or numerical-result change was made as part of the wording revision.

## Numerical claim ceiling

| Arm | Equal-query fixed-G-negative margin mean | Prior G→F mAP change, points |
|---|---:|---:|
| gradient | −0.00011200842863951496 | +0.06779562420048535 |
| random | +0.0005324544141424804 | +0.1728812710489791 |

The supported statement is that **these recorded fixed-result statistics have those signs and values**, and their additive accounting passes the recorded tolerances. Positive/nonpositive pair means and saved rank groups describe different conditionings and weights.

The query/gallery records are identical, means are not identity-balanced, rank groups are selected from outcomes, and the probe retains one G-hardest negative rather than all F ranking interactions. Large opposing terms can cancel. Neither a positive mean nor an individual term implies a causal benefit, per-query improvement, a valid training objective, statistical robustness or broader retrieval superiority.

## Precise blockers

There are **no blockers to retaining the final bounded descriptive report**, and no unresolved numerical/source defect.

The following stronger claims remain blocked:

1. **Independent tensor reproduction:** six raw tensors, selected-negative indices and term matrices were not independently read/recomputed.
2. **Independently attested execution/no-retry history:** counters and command/exit are supplied receipts, consistent with source; no separate monitor or exhaustive attempt ledger was supplied.
3. **Actual reviewer model/effort:** requested Astra/max is not an attestation. Actual remains UNATTESTED.
4. **Causal/general performance claims:** two selected arms, one seed/dataset, shared observations and a fixed-negative proxy do not support them.
5. **Unified host chronology/cost:** the two wall clocks disagree.
6. **Broader campaign facts:** historical common-energy evidence, exhaustive intervention absence, live training state, retained remote artifacts and complete goal status require separate primary evidence if independently certified.

Keep the final report's explicit limits. No new GPU or training work is justified by this audit.

## Audited result-stage input SHA256

- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_RESULT.json`: `eab5617285cc21778d8bc4565e549841eb2e74e0ddd18ec21d80d6c2fde68c01`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS_EXIT.json`: `fa7475319ed53720b71cb1a257614fa0c2259f40f309ef230733baf5b0adba2e`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS.stdout`: `e2e1a24daf64d2d9d5436b2dbf3888ddcd9f149d85242a363bf7ce4d6ff87920`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/CPU_TERMS.stderr`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/REQUEST.json`: `ca41d6f1d71d9b04a62e566fe58fe99b8fc9de6d055d2165cdbdcbb54c8b96a4`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT.md`: `9f46d1f3ea15920666ab01d3da7139f19f2dba4599bca0c98337a46c155bf487`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/RESULT_REPORT_R1_BEFORE_WORDING.md`: `6b74b80b5b736ab83b393eb0b92748b0e721603745d12d77a80f9dc32c1c3f44`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/SOURCE_AUDIT.md`: `cd22d37de24455a9d7381806c7630c9faa297b1f8bc29e94a7c1422a0b4d4b16`
- `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/evidence_terms932/SOURCE_AUDIT.json`: `6b864dfdfb8ce83035c87851edb51dd9ffdc281d478d8e14dfc9d46be964aa3c`

The full JSON audit also includes every source-stage input hash, exact source references, 16 detailed numerical claim checks, contextual claim dispositions and the recorded row summaries.
