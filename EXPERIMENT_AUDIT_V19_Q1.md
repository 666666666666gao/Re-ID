# EXPERIMENT_AUDIT_V19_Q1

Date: 2026-09-05  
Auditor: independent experiment integrity auditor  
Requested audit: `C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-05_run04/AUDIT_REQUEST.txt`

## Conclusion

`integrity_status`: **warn**  
`engineering_integrity`: **pass**  
`scientific_qualification`: **fail**  
`overall_verdict`: **warn**  
`evaluation_type`: **real_gt_train_internal_complete_path_oof**  
`scientific_status`: **Q1_FAIL**

I independently audited the V19 Q1 terminal artifacts by reading the request, the listed primary reports/receipts, and the relevant source files directly. I did not use executor summaries, and I did not launch training, evaluation, model inference, checkpoint tensor loading, or downloads. The allowed recomputation scope was pure JSON/NumPy replay of reported arrays, masks, metrics, paired deltas, and the identity-cluster bootstrap.

The Q1 engineering record is internally consistent. The six endpoint receipts are present, match the embedded Q1 summary endpoints, preserve paired initialization/sample-order/baseline controls within each fold, report strict reload/read-only evaluation, and report the expected optimizer-step totals with no AMP overflow. I recomputed all aggregate AP/Rank metrics, fold metrics, paired AP/Rank deltas, gallery/query masks, fold fused gains, scientific predicates, and the NumPy identity-cluster bootstrap. The recomputed values match the terminal artifacts exactly within floating-point tolerance.

The scientific result is a failure under the preregistered Q1 gate. The trained private-tail endpoint improves fused aggregate mAP by only **+0.256035 pp**, fold gains are **+1.619524 pp**, **-0.900867 pp**, and **-0.001279 pp**, Mamba aggregate mAP declines by **-0.422824 pp**, and the 95% identity-cluster bootstrap lower bound is **-1.615129 pp**. Four of the five fixed scientific predicates are false. V19 therefore remains sealed at Q1_FAIL, with no support for D1, dev, official, or SOTA claims.

I assign `integrity_status: warn` rather than `pass` because the local audit could verify the JSON/log/report/receipt arithmetic and source-path bindings, but could not independently possess and hash the large remote-only checkpoints, CLIP/data artifacts, or target cache bytes. Two execution source hashes also match the local files only after LF normalization because the local checkout has CRLF bytes for `modeling/trifusion/criterion.py` and `protocols/rgbnt201_dev_v1.json`; the terminal file-verification receipt records LF-sized remote source bytes. This does not change the numerical Q1 conclusion, but it is a material byte-level limitation.

## Primary artifact status

| Artifact | Status | SHA-256 |
|---|---:|---|
| `AUDIT_REQUEST.txt` | read | `7b2898dd2b7f409ab06fa8de71732f9643204879274488dd9d9e244d3d76012e` |
| `evidence/trifusion_v19_q1_seed42_4b749cd.json` | read/recomputed | `e0c9c2e0683c934fd65ae594186d89452c9786e203e1f4b1a9b7612505316d59` |
| `evidence/trifusion_v19_complete_run_20260905.log` | read | `1b15ae13fdf41d0a81b0a4f21ecadd6cae542bce5aa3a9d459b331953afa36c5` |
| `evidence/trifusion_v19_terminal_array_audit_20260905.json` | read/cross-checked | `6c720a3638da37fdb5f11f9716cca6f0efe94bb8d2ef0f0198a5de73092fbc18` |
| `evidence/trifusion_v19_terminal_file_verification_20260905.json` | read/cross-checked | `d24eeccf3a06177cfc7117842f2601c6b1b17e5da628ed4ce2a08b85a7464163` |
| `evidence/trifusion_v19_complete_comparison_20260905.json` | read/cross-checked | `7f8cfae97bbb21829f3740440e527ca3e25f4a9eacd4d86ceaae72994b23f072` |
| `results/TRIFUSION_RGBNT201_V19_PRIVATE_TAIL_2026-09-05.md` | read | `8496643b6a50adf294139f4c45d1e10a8f93fa1c393b881e5fbbc1c667d4b0d0` |
| `modeling/trifusion/signal_preserving_v13.py` | bootstrap source read | `7b7c4abb220ed234608553c77aeedd5f8ef763abb04cce5b21f1bfce0f4daa62` |

The six endpoint receipt SHA-256 values were:

| Fold | Endpoint | Receipt SHA-256 |
|---:|---|---|
| 0 | frozen_private_tail | `3c53df49fc5f2641f00d687f75250a8964b39242afa2dd55afe3bb3ba7b02cac` |
| 0 | trained_private_tail | `b274c2244399185b254400757c6bce8a7985654b284f2ebcfb9e4c97d0ca431c` |
| 1 | frozen_private_tail | `f4d7c179cdffc1211208042907b5fca701ee43ee16493fd8140fff458a7dbca6` |
| 1 | trained_private_tail | `fbafa984dff70c33b27f914c0a0dd9de46ac17551d56cedc6d482715e2f4d381` |
| 2 | frozen_private_tail | `03f3b979c039b5558e695d61b0b9845229e3c6ebaa2947b0ed33ae00731ba1ef` |
| 2 | trained_private_tail | `e44a7137b94b82c3b23a9796a033b6c7a9755a616565e242913a1bc2e6fdd30c` |

## Recomputed aggregate metrics

All values below are percentages recomputed directly from the reported per-query arrays.

| Endpoint | Output | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---:|---:|
| frozen | baseline_only | 77.487603 | 79.334501 | 89.492119 | 93.520140 |
| frozen | fused | 80.240792 | 83.187391 | 90.017513 | 93.870403 |
| frozen | cnn | 79.915105 | 84.763573 | 88.966725 | 91.593695 |
| frozen | transformer | 78.150546 | 82.136602 | 90.542907 | 92.994746 |
| frozen | mamba | 77.801980 | 78.984238 | 89.316988 | 94.045534 |
| trained | baseline_only | 77.487603 | 79.334501 | 89.492119 | 93.520140 |
| trained | fused | 80.496828 | 84.238179 | 89.842382 | 93.695271 |
| trained | cnn | 80.054797 | 84.238179 | 90.367776 | 92.469352 |
| trained | transformer | 79.331729 | 83.187391 | 89.141856 | 92.469352 |
| trained | mamba | 77.379156 | 79.509632 | 89.667250 | 93.870403 |

## Recomputed matched aggregate mAP gains

| Output | Trained minus frozen mAP |
|---|---:|
| baseline_only | 0.000000 pp |
| fused | +0.256035 pp |
| cnn | +0.139692 pp |
| transformer | +1.181183 pp |
| mamba | -0.422824 pp |

## Recomputed fold fused mAP gains

| Fold | Eligible queries | Frozen fused mAP | Trained fused mAP | Gain |
|---:|---:|---:|---:|---:|
| 0 | 190 | 79.583894 | 81.203419 | +1.619524 pp |
| 1 | 179 | 86.785139 | 85.884272 | -0.900867 pp |
| 2 | 202 | 75.208196 | 75.206917 | -0.001279 pp |

## Recomputed Q1 scope

| Fold | Gallery records | Eligible queries | Excluded query-only records | Heldout gallery identities | Eligible query identities |
|---:|---:|---:|---:|---:|---:|
| 0 | 1000 | 190 | 810 | 47 | 7 |
| 1 | 1051 | 179 | 872 | 47 | 7 |
| 2 | 1075 | 202 | 873 | 47 | 7 |
| Total | 3126 | 571 | 2555 | 141 | 21 |

The recomputed query/gallery masks match the endpoint output indices and excluded query lists. Each eligible query has at least one same-identity, different-camera positive in its fold gallery. Same-identity/same-camera entries are treated as junk for AP and rank, matching the source logic.

## Recomputed bootstrap and scientific predicates

The bootstrap replay used the implementation in `modeling/trifusion/signal_preserving_v13.py:253-279`: identity clusters from the 571 eligible queries, `np.random.default_rng(42)`, 10,000 resamples, replacement sampling over the 21 identity clusters, concatenation of sampled identity rows, mean AP delta per resample, and a 2.5th percentile lower bound.

| Quantity | Value |
|---|---:|
| Identity clusters | 21 |
| Resamples | 10000 |
| Observed mean fused mAP gain | +0.256035 pp |
| 95% lower bound | -1.615129 pp |
| Max absolute difference vs Q1 summary | 0.0 |

| Scientific predicate | Recomputed value | Status |
|---|---:|---|
| Aggregate fused mAP gain is at least +1.0 pp | false | fail |
| All fold fused gains are nonnegative | false | fail |
| All expert aggregate gains are nonnegative | false | fail |
| Identity-cluster bootstrap lower bound is positive | false | fail |
| Trained fused beats trained baseline and all trained experts | true | pass |

## Recomputed paired changes

| Output | AP improved | AP declined | AP equal | Rank-1 repaired | Rank-1 broken |
|---|---:|---:|---:|---:|---:|
| baseline_only | 0 | 0 | 571 | 0 | 0 |
| fused | 180 | 186 | 205 | 19 | 13 |
| cnn | 180 | 193 | 198 | 8 | 11 |
| transformer | 214 | 199 | 158 | 25 | 19 |
| mamba | 157 | 202 | 212 | 10 | 7 |

The paired changes confirm a mixed query-level effect. Fused has a positive aggregate mAP gain but more AP declines than improvements, and the bootstrap lower bound remains negative.

## A-F checklist

### A. Ground-truth provenance — PASS

The Q1 evaluation is not official RGBNT201 test and not V19 dev. It is a real-ground-truth, train-internal, complete-path out-of-fold protocol over held-out train identities. The frozen protocol records train/dev/test disjoint checks and an evaluation filter that excludes same-identity/same-camera junk while retaining the same held-out list as query/gallery.

Evidence:

- `protocols/rgbnt201_dev_v1.json:2-14` records protocol checks, including train/dev/test disjoint and valid pairings.
- `protocols/rgbnt201_dev_v1.json:15-19` records train and dev triplet counts.
- `protocols/rgbnt201_dev_v1.json:52-58` defines the evaluation filter.
- `protocols/rgbnt201_dev_v1.json:126-130` states the protocol selection comes from train_171 identities and does not use test labels.
- `tools/build_v12_complete_path_oof_targets.py:29-55` defines the identity fold split and relabeling.
- `tools/train_signal_preserving_v18.py:145-152` uses record identities/cameras and full-gallery scoring.
- `evidence/trifusion_v19_q1_seed42_4b749cd.json:26-27` records `real_gt_train_internal_complete_path_oof` and reuse of V12 OOF.
- `results/TRIFUSION_RGBNT201_V19_PRIVATE_TAIL_2026-09-05.md:12-15` states the evaluation is train-internal OOF, not dev or official.

Claim impact: the real-ground-truth Q1 arithmetic is valid within this train-internal OOF scope. It does not support official, dev, or SOTA claims.

### B. Score normalization and AP/Rank semantics — PASS

The live evaluation path L2-normalizes features, computes pairwise distances, and evaluates AP/rank from identity/camera masks. I recomputed the reported AP/rank values from the stored arrays and matched the Q1 summary and terminal array audit.

Evidence:

- `tools/train_signal_preserving_v18.py:133-152` evaluates with `model.eval()`, collects baseline/fused/expert features, normalizes them, computes `cdist`, calls full-gallery scoring, and checks state unchanged.
- `tools/audit_v17_full_gallery.py:14-43` defines eligible queries and per-query score aggregation.
- `tools/diagnose_v6_oracle_complementarity.py:91-108` defines stable sort, same-identity/same-camera junk filtering, cumulative precision, and AP.
- `evidence/trifusion_v19_terminal_array_audit_20260905.json:31-116` independently records the same aggregate metrics, gains, fold gains, bootstrap lower bound, and scientific checks.

Claim impact: there is no evidence of prediction-derived score normalization leakage in the audited source path or stored arrays.

### C. Result existence and binding — WARN

The local terminal Q1 JSON, log, endpoint receipts, array audit, file verification, comparison, and result report are present and internally bound. All six standalone endpoint receipts match their embedded Q1 summary records. The terminal file-verification receipt binds the run summary, log, sources, V12 source checkpoints, final V19 checkpoints, and endpoint receipts.

Evidence:

- `evidence/trifusion_v19_q1_seed42_4b749cd.json:3-12` records Q1_FAIL status, execution commit, runner/config/plan hashes, Signal repository binding, and source file hashes.
- `evidence/trifusion_v19_terminal_file_verification_20260905.json:1-9` records verification success, Q1_FAIL status, execution commit, observed commit, summary SHA, and log SHA.
- `evidence/trifusion_v19_terminal_file_verification_20260905.json:117-212` records CLIP, V12 source checkpoint, and V19 final checkpoint hashes.
- `evidence/trifusion_v19_terminal_file_verification_20260905.json:215-257` records all six endpoint receipts and receipt-to-summary equality.
- `evidence/trifusion_v19_terminal_file_verification_20260905.json:259-266` records Signal binding and verifier scope.

The warning is byte-level, not arithmetic-level. The large checkpoint/data/CLIP artifacts are remote-only from the local audit perspective, so I could not independently hash those bytes. The M0 transfer receipt also states it is not a Q1 terminal artifact. Local source hash checking found two CRLF raw-byte mismatches that match the execution hashes only after LF normalization:

| Source | Execution SHA | Local raw SHA | LF-normalized result |
|---|---|---|---|
| `modeling/trifusion/criterion.py` | `0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a` | `9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f` | matches |
| `protocols/rgbnt201_dev_v1.json` | `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946` | `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d` | matches |

Claim impact: the Q1 result binding is sufficient for an engineering audit warning, but not a byte-perfect local possession pass for all large artifacts.

### D. Live execution path — PASS

The runner source enforces the fixed contract, builds V19 from strict V12 source checkpoints, creates paired frozen/trained endpoints, trains final-only for 20 epochs, strict-reloads checkpoints, evaluates read-only, and aggregates the stored arrays. The endpoint receipts record the paired controls and state-change checks.

Evidence:

- `tools/train_signal_preserving_v19.py:52-69` asserts the contract, seed, batch composition, endpoints, no dev, no official, no reranking, V12 source hash, and V12 fold receipts.
- `tools/train_signal_preserving_v19.py:72-114` loads V12 Signal/expert checkpoints, strict-loads them, wraps V19, verifies private-tail parameter/tensor counts, and records bindings.
- `tools/train_signal_preserving_v19.py:125-142` constructs the optimizer and criterion.
- `tools/train_signal_preserving_v19.py:145-175` validates finite outputs, baseline prefix preservation, gradients, and overflow.
- `tools/train_signal_preserving_v19.py:259-311` trains each endpoint for 20 epochs and records sample order, first batches, gradients, overflow, frozen state, and private-tail change state.
- `tools/train_signal_preserving_v19.py:411-467` runs all folds/endpoints, saves checkpoints, strict-reloads, evaluates, and writes endpoint receipts.
- `tools/train_signal_preserving_v19.py:468-506` aggregates metrics, fold gains, bootstrap, scientific checks, source hash recheck, and gallery/query totals.
- Endpoint receipts report strict reload/read-only evaluation and expected private-tail state changes.

Claim impact: V19 Q1 was a terminal trained/frozen paired engineering run, not just a dry-run or M0 diagnostic.

### E. Scope — PASS

The recomputed scope matches the preregistered Q1 complete-path OOF scope: three folds, all held-out identities retained as galleries, only query identities without cross-camera positives excluded from query metrics, 3,126 gallery records, and 571 eligible queries. This matches the plan, Q1 summary, terminal array audit, and result report.

Evidence:

- `refine-logs/v19/EXPERIMENT_PLAN.md:88-99` defines the Q1 full-identity OOF protocol and aggregate scope.
- `AGENTS.md:287-297` instructs retaining single-camera held-out identities as gallery distractors and excluding only queries with no valid cross-camera positive.
- `evidence/trifusion_v19_terminal_array_audit_20260905.json:8-29` records the fold gallery/query scope.
- `evidence/trifusion_v19_q1_seed42_4b749cd.json:78081-78082` records 3,126 gallery records and 571 eligible queries.
- `results/TRIFUSION_RGBNT201_V19_PRIVATE_TAIL_2026-09-05.md:12-15` records 3,126 gallery, 571 eligible queries, and 2,555 query-only exclusions.

Claim impact: the Q1 terminal result should be interpreted over the full train-internal OOF gallery scope, not a reduced diagnostic subset.

### F. Evaluation and diagnostic classification — PASS

The artifacts consistently classify V19 Q1 as train-internal complete-path OOF. The M0 capacity/overfit gates are correctly treated as train-only engineering diagnostics, not retrieval evidence. Vehicle data preparation is correctly treated as readiness-only, not a V19 performance result. The result report and handoff explicitly state no D1, no dev, no official, no reranking, no ensemble/multi-query, and no SOTA claim.

Evidence:

- `evidence/trifusion_v19_q1_seed42_4b749cd.json:26-35` records the evaluation type and disables dev, official, and D1.
- `AGENTS.md:61-68` classifies capacity/overfit gates as train-only engineering gates and gives the exact Signal dev floor.
- `evidence/trifusion_v19_terminal_array_audit_20260905.json:4` records the no-model/no-feature/no-distance-replay array audit scope.
- `evidence/trifusion_v19_terminal_file_verification_20260905.json:264-266` records no model/checkpoint tensor loading by the verifier.
- `docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:1986-1993` classifies vehicle data artifacts as readiness only.
- `docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:2054-2102` records V19 terminal Q1_FAIL, no D1/dev/official, and broader targets unmet.

Claim impact: the final supported claim is an engineering execution claim with a failed preregistered scientific gate, not a performance qualification claim.

## Claim classification

| Claim | Audit result | Basis |
|---|---|---|
| V19 Q1 terminal artifacts exist and are internally consistent | supported with warning | JSON/log/receipts/reports present; all recomputed arithmetic matches; large remote bytes not locally possessed |
| Frozen and trained endpoints are paired within each fold | supported | identical baseline outputs, initial state, sample order, and first-batch records within each fold |
| Private tails changed only in trained endpoints | supported | frozen endpoints report unchanged tails; trained endpoints report changed tails |
| Trained fused improves aggregate mAP over frozen fused | supported numerically | recomputed +0.256035 pp |
| V19 passes preregistered Q1 scientific qualification | rejected | four of five scientific predicates false |
| V19 should proceed to D1/dev/official evaluation | unsupported/rejected | Q1_FAIL and `next_phase_permitted=false` |
| V19 achieves dev, official, or SOTA improvement | unsupported | no D1/dev/official/SOTA evaluation was run or audited |

## Limitations

- I did not run training, evaluation, inference, checkpoint loading, or feature/distance replay.
- The audit is bounded to the primary listed artifacts and source files, plus pure JSON/NumPy recomputation.
- Large checkpoints, CLIP/data artifacts, and target cache bytes were not independently available as local bytes for hashing; their status remains receipt-bound.
- Two source files match execution hashes only under LF normalization because the local checkout uses CRLF bytes for those files.
- The evaluation is train-internal complete-path OOF. It is not dev, official, public-test, or SOTA evidence.

Final audit conclusion: **engineering_integrity pass**, **scientific_qualification fail**, **integrity_status warn**, **overall_verdict warn**.
