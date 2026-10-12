# Experiment audit: remaining911 and fifth saved endpoint

Date: 2026-10-13. Review route: fresh delegated Codex agent `/root/audit_remaining911`; requested `gpt-6-astra`, effort `max`; serving model and effort **UNATTESTED**. `review_independence: same-family`; `acceptance_status: provisional`.

## Verdict: WARN

No content blocker was found in the proposed one-condition controller or the fifth endpoint's saved text. The source-only controller review is satisfactory for its stated administrative scope. **This is not native admission or a six-of-six completion certificate.** Exact staging of the reviewed new controller, native source/artifact admission, and the remaining real M0/fresh50/first strict evaluation are still separate requirements.

The independent stdlib checker passed **27,324/27,324 non-source-pin checks**. Its broader raw checkout hash check remains **FAIL**: 329/463 exact source pins, 27 CRLF-only byte mismatches, and 107 files absent from the publication checkout. All 27 mismatching files produce the sealed hash after CRLF-to-LF replacement; no substantive mismatch was found among available bytes. No files were normalized or repaired. The original 134 failures remain in `DETERMINISTIC_CHECKS.json`, `CHECK_LEDGER.jsonl`, and `DETERMINISTIC_STDOUT_001.txt`; `SOURCE_HASH_LIMITS.json` explains them without replacing the failed run. The raw checker total is 27,787, with 27,653 passed and 134 failed. These are assertions over records, not independent experimental samples.

## Scope and evidence paths

All writes by this reviewer are within this review directory. No server/network/GPU query, production import, controller execution, model call, optimizer call, binary load, publication, source edit, or further agent call was made.

Path abbreviations below:

- `N` = `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/remaining911/`.
- `P` = `C:/Users/gb/.trifusion_github_publish_22c3bee/`.
- `I` = `C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/results911/`.
- `T` = `I/trained-model/region_use_v1_20261012_906_full_uniform_RGBNT100/`.
- `S` = `I/logs/region_use_endpoint_diagnosis_20261012_907/RGBNT100_uniform/SUMMARY.json`.

Exact local input byte hashes, sizes, and copied evidence paths are in `INPUT_HASHES.json`; key JSON line anchors are in `LINE_EVIDENCE.json`. The master was retained only as the bounded section 41.910 extract, original lines 18395–18414, in `MASTER_910_EXCERPT.txt`. Its historical claims were not treated as experimental proof.

## Controller and original failure

1. **Only the originally unattempted condition is constructed.** `N/tools/continue_region_use_remaining.py:18` reads the original campaign/exit; lines 22–26 require exit1, RGBNT100/ambiguity PENDING with no stages, and five COMPLETE jobs. Lines 34–43 create a separate one-job state and record original campaign/exit hashes. Lines 47–55 use one literal `prepare, m0, train, evaluate` loop and the original command builder with fixed RGBNT100/ambiguity arguments. There is no completed-job loop or retry loop.
2. **No original campaign overwrite or old endpoint replay is present.** Line 17 rejects an existing campaign; all writes at lines 45–75 target the new campaign. `P/tools/queue_region_use_probe.py:29` derives each output directory from the new campaign basename; `P/tools/run_foundation_recipe.py:190` rejects an existing training output. The admitted campaign name must be the planned `logs/region_use_remaining_20261013_911`, as stated in `N/EXPERIMENT_PLAN.md:7`.
3. **Original scientific functions and initialization remain in control.** The original command is reused (`queue_region_use_probe.py:33`), initializer fields are compared to the old semantic binding except the existing architecture/entry/scope exclusions (lines 43–54), and M0 acceptance is unchanged (lines 57–81). It requires eight logged batches matching the original order, all 285 trainable tensors active, BN counts eight, reload tolerance 1e-5, frozen-state preservation, and auxiliary support/activity. `P/tools/accept_region_use_m0.py:17` requires eight effective updates and all nine Q/K/V groups active. M0 probe retirement is the original scoped operation at `queue_region_use_probe.py:73`; this does not retire a formal best.
4. **Fresh50 and selected-best strict evaluation are preserved.** `P/tools/run_region_use_probe.py:101` fixes semantic/seed42/50 epochs. `P/tools/run_foundation_recipe.py:190` constructs a new model and output for each M0/full training invocation; lines 211–223 impose eight M0 batches or 50 epochs. Lines 249–255 save mAP-best, resolving ties to the later epoch. Lines 287–312 construct fresh evaluation state, strict-load that best, check all metrics against the same epoch, and emit the first planned strict result. `load(...):185` uses `strict=True`. No CMC column is selected from a different epoch.
5. **Source/seal ownership is preserved, with an explicit new-file limit.** Controller lines 27–30 check the old input seal, all 50 artifact digests, and original `panel.source_map()`. Each original `run_logged` calls `base.require_sources` (`queue_native_research.py:63`), which rechecks manifest source and initialization hashes (`queue_foundation_recipe.py:110`). The new controller is outside the original 463-file seal. Its self-hash recorded at controller line 39 is a provenance field, not a trusted external pin. Exact staged/admitted SHA must be `190d6a6dfaa02b3b68c7fec4cd9668c650bfde2c9b00f04759cbd68809435381`; plan SHA is `b58dfdc1e26f69e751e41479c6a24d46b0fea06b53cf1fb1722147ac3083695b`. No seal extension or production mutation is justified by this review.
6. **Capacity and stopping behavior have actual evidence.** Initial and stage checks preserve 2 GiB + 512 MiB = 2,684,354,560 bytes (`queue_region_use_probe.py:18`; new controller lines 31, 48). The added stage-capacity branch writes `STOPPED_BEFORE_STAGE_INSUFFICIENT_STORAGE` and returns once before appending/running a stage (lines 48–52). Child nonzero exits mark FAILED and return once (55–59). An initializer/acceptance assertion also terminates without a retry; it does not necessarily persist a terminal campaign status, so the supervisor EXIT remains authoritative. The inherited device wait sleeps 240 seconds and is admission waiting, not a failed-stage retry (`queue_native_research.py:49`).

The original failure is independently consistent across the saved files: `I/logs/region_use_resume906/EXIT.json:12` records exit1 at 23:57:40.989290. Original `campaign.json:3` remains RUNNING, but all first five jobs have four COMPLETE/exit0 stages; its sixth job is RGBNT100/ambiguity PENDING with an empty stage list at lines 938–941. `FIRST_100_NODE_909.json:2734` preserves the traceback to original controller line 106, the storage assertion immediately before creating a next-stage record. Its saved observation at 00:11:22.424501 has both original owners absent (lines 2709–2710), and its embedded state and exit exactly match the separate files. **This is a storage stop before the sixth condition was attempted, not a model mAP failure.** No claim about current live processes follows from that saved observation.

## A. Ground-truth provenance: WARN, source-supported real_gt

The source uses dataset identities and camera filtering, not model-generated references. `P/tools/build_official_three_dataset_protocols.py:32` consumes dataset inventory records; lines 49–64 copy identities/cameras and count legal positives. The sealed RGBNT100 protocol at `P/logs/training_feature_scale_protocols_20261002/RGBNT100.json:8` specifies train/query/gallery splits, and lines 218978–218983 record 8,675 train, 1,715 query, 8,575 gallery and same-identity/same-camera exclusion. Its hash equals the sealed artifact and fifth receipt protocol hash. The reviewer independently verified all 1,715 query identities, train/test identity disjointness, and positive/exclusion counts against that protocol JSON.

`P/tools/report_prepool_dense_correspondence.py:24` checks each saved distance hash, scores using stored dataset IDs/cameras, and requires identical control/candidate GT arrays (lines 27–34). `P/tools/train_rgbnt100_signal_oof.py:253` implements legal camera filtering, AP, and CMC from first positive ranks. The reviewer did not open original dataset images, inventory bytes, model weights, or distance binaries. Saved query-level AP and first ranks were the numeric inputs to this independent text audit; individual AP values were not regenerated from distance rankings.

Training support is also consistent with dataset GT: `modeling/trifusion/region_use_objective.py:21` constructs same-identity/different-environment positives and different-identity negatives. All 3,129 saved batches independently match sealed train path/label/camera records; B128/K16 structure, eligible-query counts, positive-pair counts, and legal-triplet counts exactly recompute from those records. Global-margin signs remain saved model-derived statistics, checked for legal bounds only.

## B. Score normalization: PASS within inspected source/text

AP is mean precision at legal relevant positions and CMC is the fraction of queries whose first positive rank is at most k (`train_rgbnt100_signal_oof.py:263`). There is no division of performance metrics by a model's own maximum or prediction statistic. Feature L2 normalization and the auxiliary objective's legal-triplet/query weighting are representation/training definitions, not rescaling of reported retrieval accuracy. The latter is explicitly training-only (`run_region_use_probe.py:48`; `modeling/trifusion/region_use_heads.py:19`).

The small CMC difference between official float32-style percentages and query-count arithmetic is below the existing 1e-5 point tolerance. For example, saved first-rank counts yield Rank-1 95.51020408163265 versus official 95.5102026462555; both refer to the same candidate, not an inflated score.

## C. Result existence and numerical consistency: PASS for saved text; WARN for binary attestation

All 11 intake-manifest entries exist and match their recorded SHA256. All six fifth-report input hashes match the available producer/seal/training text. The original campaign's fifth result exactly equals `T/official_metrics.json`. The four training text files are coherent: 50 ordered epochs, 3,129 steps/batches, exact per-epoch step counts and mean losses, finite values, global/fused/head/auxiliary loss arithmetic, and all 50 auxiliary trace totals. The complete batch-order byte hash matches the original semantic order digest already in the seal; the old order file was not reopened.

The selected epoch is 26 throughout history, training receipt, strict receipt, and summary. Metrics follow that single mAP-best checkpoint (`T/training.json:93`, `:697`; `T/official_metrics.json:8`, `:35`; `S:15`). Checkpoint SHA `5f40f20569f9b9f7bcb348749ff5287dd8f5ec10466330f5cc7c4c765d8682c2` and strict-distance SHA `f74d18f7929ba64c8924bfda0128ac1699fe42ff313f6f0eb1ab70f74496e3d8` agree across saved references; their binary bytes were not rehashed by this reviewer.

| Fifth endpoint | Best epoch | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|---:|
| RGBNT100 uniform | 26 | 83.285762 | 95.510203 | 95.976675 | 96.326530 |

Both comparisons contain the complete same 1,715 query order, 50 identities, and identical candidate AP/rank values. Saved AP means, Rank-1/5/10, metric deltas, direction counts, repairs/new errors, four rank-1 groups, group AP contributions, every identity mean, identity macro mean, and the worst 86-query (ceil 5%) tail all recompute. The evidence begins at `S:982`, with first comparison identities/queries at `:1012`/`:1264`, groups/tail at `:14987`/`:15005`; second comparison identities/queries at `:15036`/`:15288`, groups/tail at `:29011`/`:29029`.

| Comparison | ΔmAP | ΔRank-1 | Repairs/new errors | AP improved/worse/tied | Identity macro ΔAP | Worst 5% mean ΔAP |
|---|---:|---:|---:|---:|---:|---:|
| uniform − raw semantic | -0.804543 | -0.349854 | 35/41 | 624/800/291 | -0.571591 | -27.076664 |
| uniform − independent global | -1.248022 | -1.107872 | 24/43 | 603/804/308 | -0.762400 | -24.272613 |

The raw-semantic comparison's repaired/new-error/still-correct/still-wrong groups have 35/41/1603/36 queries and AP contributions +0.410845/-0.675577/-0.509879/-0.029932 points. The independent-global groups are 24/43/1614/34, contributing +0.334802/-0.477744/-1.017511/-0.087569. Contributions add to the respective total delta. No Signal comparison is present.

Best-to-last mAP drop is 0.9666028276476055. The recorded producer wall interval is 7,276.389425 seconds; training plus per-epoch evaluation is 7,248.36881 seconds; the sum of training-loop times is 3,346.0271131491754 seconds. Each recomputes using its stated interval rather than interchanging them (`S:29033`–`:29035`). Peak-memory metadata agrees between the saved training and report; no live memory measurement was made.

## D. Dead code and call path: PASS as source review; native status UNVALIDATED

The controller reaches the unchanged prepare/M0/train/evaluate call path, and the endpoint reporter calls `paired` for both named controls (`report_region_use_endpoint.py:71`). The scoring function produces the query AP/rank arrays consumed by all groups/identity/tail outputs. The new controller and 17 inspected supporting Python files parse as AST; none were imported or executed. AST and call-path inspection do not certify availability of the remote runtime, actual gradients, reload correctness, or a future launch.

The four sealed principal producers match exact original pins. The post-seal endpoint reporter is not one of the original 463 pins; its exact SHA is instead bound by the fifth summary input hash: `807746c6ab295fc39de40bd3a91028c57dc7d346f89e1351d803d5ac401a314d`. Four locally available artifact JSONs match the 50-artifact seal. The other remote dependencies were not locally recertified.

## E. Scope: WARN

Current original evidence is **five completed endpoints out of six planned, one seed42**, with the sixth originally unattempted. This fifth condition has negative mAP/CMC deltas to both named controls. It establishes neither useful additional identity evidence nor superiority to Signal. Its 50 identities are not 50 independent training seeds, and repeated training exposures are not independent test observations.

The official benchmark was used for per-epoch mAP selection and development; the report correctly discloses this at `report_region_use_endpoint.py:95` and `I/.../REPORT.md:7`. It is not an untouched test set. Neither these results nor a future six-of-six administrative completion establish generalization, causality, a unique failure mechanism, regional evidence utility, or broad multi-dataset superiority. Same-best global/region decomposition remains a separate possible future consumer, not a finding here.

The plan's cold-storage action and fresh idle admission are prelaunch operations, not accomplishments certified by this audit. The supplied saved Signal inventory has equal within-dataset byte sizes but distinct recorded SHA values; it supports the plan's refusal to treat them as duplicate cleanup candidates. Those remote Signal binary hashes were not independently recomputed here.

## F. Evaluation type and claim impact

`real_gt` is the source-defined retrieval evaluation type. This reviewer provides a **source and saved-text consistency audit** of that evaluation, not binary or model recertification.

- Supported with saved-evidence qualification: original storage assertion occurred before sixth-stage creation; original status is stale RUNNING with authoritative exit1; five endpoints completed; RGBNT100 uniform E26 saved metrics and paired arithmetic above.
- Supported as source-only intent: the new named controller attempts only the originally unattempted condition through unchanged eight-M0/fresh50/first-strict stages, with no retry or original-parent write.
- Needs external native evidence: exact staged new-script hash, native463/source50 dependency checks, capacity/idle admission, actual initialization/M0/full50/strict result for the remaining condition.
- Unsupported: six-of-six completed now, a successful sixth M0 now, result inflation, Signal advantage, scientific utility, unique mechanism, untouched-test performance, full training-seed uncertainty, serving-model/effort attestation.

## Required completion records

Use the exact new controller/plan bytes above for staging and the named new campaign. Preserve original EXIT/campaign/stdout and the five formal endpoints. Let original source/artifact checks and real M0 acceptance run unchanged; preserve authoritative supervisor exit on any stop. Record new campaign provenance explicitly if the sixth endpoint later completes. No additional defensive code, fallback, retry, or production refactor is recommended by this review.

Full request/response and route metadata are under `.aris/traces/experiment-audit/2026-10-13_remaining911/`. Local command failures and the original storage failure are recorded in `ORIGINAL_FAILURES.md`. The raw deterministic FAIL is intentionally retained, alongside the scoped all-pass arithmetic result and this WARN assessment.
