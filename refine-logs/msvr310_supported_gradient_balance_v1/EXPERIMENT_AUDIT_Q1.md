# Supported gradient balance R2 — complete independent Q1 audit

Date: 2026-09-21. **WARN / CLOSED_WITH_LIMITS**. Deterministic saved-evidence checks: **PASS**. Engineering audit blockers: **none**. Scientific qualification: **Q1_FAIL**, next phase **false**. These are separate conclusions.

Fresh delegated reviewer `/root/audit_supported_gradient_balance_q1_20260921`; requested gpt-6-astra/max/fresh-none. **Same-family / provisional; backend attestation unavailable.** No further delegation, training, optimizer update, model forward, official-test read, environment installation or binary download was performed.

The audit directory is `D:\Program Files\UserCache\gb\codex\tmp\trifusion_supported_balance_q1_independent_audit_20260921`. `R` below means `snapshots/repo/`; `U` means `snapshots/remote_text/root/autodl-tmp/trifusion-v2/comparators/Signal-cd1b0a6/`; `I` means the existing byte-exact intake `D:\Program Files\UserCache\gb\codex\tmp\trifusion_supported_balance_q1_complete_20260921`. Report/source snapshots, actual scripts, raw stdout/stderr, replay tables and hashes are retained here. The original request remains at the supplied `.aris/traces/experiment-audit/2026-09-21_supported_balance_q1/001-audit-request.md`.

## A. Ground-truth provenance — PASS

Dataset identities, camera and scene fields come from the MSVR310 **official training split filenames**, not model predictions. I reparsed all1,032 modality-triplet filenames and checked all3,096 current remote source path existences. The builder's dataset-label hash and filename offsets agree with the inspected upstream loader. The runtime reads explicit protocol source/gallery record indices through `records_for`; it does not instantiate the upstream dataset class that would also enumerate official query/test sets. Source: `latest_claims/tools/build_msvr310_train_oof_protocol.py:12-55`; R `tools/train_msvr310_signal_oof.py:68-99`; U `data/datasets/msvr310.py:68-89`.

The identity-disjoint split is reconstructed from sorted cross-scene-eligible and single-scene identity groups. All155 IDs and1,032 records enter exactly one heldout fold; training for each fold uses its complement. All600 legal query records/60 query identities are included. All95 single-scene identities/432 nonquery records remain in the complete galleries as distractors:

| Fold | Source IDs | Heldout IDs | Gallery records | Legal queries | Distractor IDs / records |
|---|---:|---:|---:|---:|---:|
| 0 | 103 | 52 | 360 | 210 | 32 / 150 |
| 1 | 103 | 52 | 349 | 207 | 32 / 142 |
| 2 | 104 | 51 | 323 | 183 | 31 / 140 |

The upstream metric function actually removes **same identity AND same scene** at U `utils/metrics.py:67-69`; its old person-ReID docstring does not define the executed rule. Every different-identity record, including same-scene negatives, remains. This rule was applied independently to every saved permutation. R `tools/train_msvr310_trifusion_oof.py:196-241` calls the imported author metric and stores agreement with its own scene scorer.

Full-path isolation was inspected through source-only B0 training, B0 checkpoint source/heldout IDs, seed42 fresh role initialization and all1,560 current/history memberships. The independent remote checker additionally read all1,950 recorded B0 training steps and verified their source memberships; the three B0 checkpoints, their final state hashes, original retrieval-array hashes and six terminal role checkpoint bindings passed. This checks existing evidence and is not a new baseline/M0/source-census run. R `tools/train_msvr310_trifusion_oof.py:27-55`; `remote_arrays_readonly.py:25-49`; `remote_arrays_attempt2.json:1`.

Image pixels and semantic annotation correctness were not freshly re-inspected. Per-step actual pixel hashes remain the original run's witnesses, but the full paired pixel-hash sequences match.

## B. Score normalization — PASS

Terminal mAP is raw mean query AP multiplied by100; AP uses precision at the real labeled positive positions divided by that query's labeled positive count. Rank-1/5/10 are raw fractions. Results pool all queries **after fold-local ranking**, without cross-fold distances or averaging fold mAPs equally. Unit feature normalization and squared Euclidean distances are the registered metric geometry, not rescaling by the model's own best output. No suspicious self-maximum score normalization was found.

The source loss uses `score=1-d²/2`, sigmoid temperature0.01, current-anchor positives from same-ID/different-scene pairs, and the **mean of eligible anchors only**. Same-ID/same-scene candidates including self are ignored, not made negative. The positive item being ranked is excluded from its own comparison and contributes the explicit rank-one offset. Missing-positive anchors retain their other supervision and remain candidates for other anchors. Zero-eligible batches return graph-connected zero for fused ranking. R `tools/msvr_cross_scene_smooth_ap.py:11-54`.

Every saved source fused distance was used to independently recompute standard Smooth-AP, cross-scene AP and hard diagnostics; all four stored spaces were shape/finite/nonnegative checked. Maximum per-anchor cross-scene difference is **1.13595797568e-06**, below the unchanged2e-6 bound. Largest reconstructed scalar objective difference is **1.50946067978e-07**. The13 other scalar terms and weighted total passed ledger checks, but absent logits/residual features prevent fresh reconstruction of each auxiliary loss. `remote_arrays_attempt2.json:1`; `replay_text_independent.py:118-125`.

EMA norm ratios, coefficient bounds, relative-gradient errors and actual AdamW parameter deltas are engineering quantities. They are not performance scores or task-specific update shares.

## C. Result existence, values and provenance — PASS with disclosed repairs

- All43 byte-exact intake files match their inventory. All43 files referenced by the terminal CPU receipt were freshly rehashed remotely, including saved arrays and six checkpoints.
- The remote binding traversal checked159 rows. All98 distinct bound repository paths match both pipeline execution commit `1381639f778f77f124a2726ee55c07610092a438` and Q1's observed publication commit `4bb1a7f6fe7b82a05632d8119d8b0c5fbd552815`. The mutable publication HEAD is not used as a substitute for these bindings. Five local files use CRLF bytes and therefore differ from frozen remote hashes; the remote LF bytes and both execution Git blobs pass. Raw local bytes are separately retained in the manifest.
- Config SHA256 `9ce36299299efbc09ccfb72f5b1c4fed20fc1026bd9fee1bd55a591d6950ec7b`; summary SHA256 `6f2d7a919dcb492348198dced490c92665a7dec381dd9644f0130d77792bde2f`; final posthoc CPU receipt SHA256 `df58328e0e2e89e912d151346892c4a1fa0d63eb7ef8e913c3c0fda8b8d0e843`.
- Original `pipeline.json` remains **STOPPED_AT_Q1_CPU**, with original CPU exit1 and preserved `q1_cpu.log`; no original `q1_cpu.json` was fabricated. Final completion is `q1_cpu_arithmetic_recheck/verification.json` plus actual exit0 `receipt.json`.

The failure chain is preserved: (1) original exact sqrt/power mismatch; (2) first repair failed the stats source hash; (3) hash-bound sqrt-only repair failed one weighted-norm identity. The final hash-guarded in-memory transformation changes only the sqrt operation to the executed `math.sqrt` and converts the weighted-norm formula's scalar coefficients to actual float32 representation. Threshold1e-7, exact state comparisons, M0 reference limits0.005/1e-8 and all scientific gates stay unchanged. R `tools/recheck_msvr_supported_balance_sqrt.py:11-31`; R `tools/verify_msvr_supported_gradient_balance_stats.py:7-9,60-77`; I repair logs; `failed_checks.json`.

On the actual remote Python3.10.14 runtime, six supported role records differ between sqrt and exponentiation by one binary64 ULP; all recorded values match sqrt. Local Python3.13 yields five such differences, so that platform-specific count is explicitly separated. The independently reproduced weighted-norm failure is fold0/balanced/step71/Mamba:5.4609169631e-7 original error versus5.0530124677e-7 threshold. Correct FP32 coefficients give4.6714214719e-9 error. All1,743 supported balanced role combinations pass the unchanged formula threshold after representation alignment. All9,360 applied coefficient conversions were checked using the installed CPU PyTorch2.5.1; CUDA remained uninitialized. This is arithmetic verification, not GPU gradient regeneration.

**Wording qualifier:** the first failed repair's source-binding assertion means the stats file was not byte-identical at that point. Current restored source bytes are exact and the final replay verifies them. The phrase “original execution files unchanged” must describe the preserved/restored final bound bytes, not claim that no verifier file was ever temporarily changed during posthoc work. Nothing here indicates training or checkpoint mutation from that failed CPU attempt.

The latest result table, all1,560 exported steps,120 epochs and4,680 role rows were checked against primary logs. Report numbers, late-source comparisons, query changes and coefficients match. The original ranking reader's repaired variant changes only terminal provenance handling; ranking/gate/bootstrap computation is unchanged (`ranking_reader.diff`). The latest tracker accurately records completed Q1/CPU and audit in progress; replacing that pending label with this audit is a publication action, not an unfinished computation.

## D. Executed paths and full runtime scope — PASS with limits

The actual trainer calls cross-scene objectives, replaces only `triplet_fused` after65 warmup steps, differentiates the current ranking objective and direct auxiliary sum, executes the original combined backward, then restores all historical ranking derivatives by grouped VJP. Only current records are anchors; historical records are candidates, **zero historical anchors**. The complete ranking vector is current plus history before role statistics and weighting. R `tools/train_msvr_supported_gradient_balance.py:119-184,222-268`.

Historical entries are unique by record index with newest exposure winning; current indices are excluded, maximum age8, capacity512. Frozen stored fields are re-encoded under current role parameters using the original encoder-entry RNG. RNG forks and buffer comparisons surround replay; zero upstream groups are skipped by the existing rule. R `tools/msvr_instance_memory.py:15-37`; R `tools/msvr_freshness_probe.py:48-60`; R `tools/probe_msvr_role_set_gradients.py:29-50`; R `tools/probe_msvr_history_candidate_gradients.py:43-50`; R trainer `228-249`.

Whole-role grouping covers42 CNN,54 Transformer and93 Mamba encoder tensors,189 total, disjoint and complete. The14 neck/classifier trainable tensors keep original current-total gradients. Supported balanced steps apply `wR*(Rcurrent+Rhistory)+wA*A_direct`; control/warmup/unsupported preserve original `current_total+Rhistory`. The controller uses EMA0.9, square-root ratio clipped0.25..4, epsilon1e-12 and weights totaling2 within0.4..1.6. All4,680 before/after states, support identities/relations, ratios, proposed/applied weights and scalar vector identities were independently checked. R `tools/msvr_supported_gradient_balance.py:10-43,65-106`; R trainer `267-320`.

Actual Q1 coverage is390 warmup steps,1,162 supported postwarmup steps and8 unsupported postwarmup steps. The latter occur in **both** arms at fold0 step180, fold1 step221, fold2 steps133/232. EMA remains unchanged, weights are1/1 and complete ranking norms are zero; the other supervision and optimizer updates continue. Ten additional warmup batches lack cross-scene positives but use the registered hard Triplet warmup, making18 zero-eligible batches across all phases. This extends actual branch coverage beyond M0, whose unsupported steps were absent.

Exactly one scaler unscale, finite check, AdamW step and scaler update occurs per step. No overflow was recorded. Seed42, fixed20 epochs/260 updates per endpoint, source-only initialization and M0 binding are verified. Role weights are freshly initialized and are not continued from M0 or an earlier final checkpoint. Retrieval is executed once after the fixed final reload; there is no per-epoch heldout selection. R trainer `290-346,358-449`.

All six final state hashes and frozen subsets were independently reconstructed from compact role states plus original B0 aliases. All30 saved feature spaces produce the exact stored float32 distances and full ranking permutations; an additional float64 distance calculation also agrees within2e-6. The independent CPU pass took **26.164480s**, zero forwards/updates, CUDA uninitialized. Strict original reload-forward output parity is a historical runtime witness, not repeated here.

The cumulative203/203 nonzero-gradient record is the union accumulated at trainer `291-294`; it does **not** establish every-tensor/every-step nonzero gradients. Q1 has **zero independent direct-reference vector checks**. The earlier M0's six capacity reference positions and90 role-component comparisons retain their narrow engineering scope; no M0 rerun was made. Original per-step vectors, logits and AdamW moment/scaler states are unavailable, so scalar identities cannot prove coordinate-wise derivatives or regenerate parameter update attribution.

Both arms have the same scalar objective definitions, but that scalar total is not a single globally weighted candidate objective: role-specific gradient coefficients generally cannot be represented by one common scalar weight. Finite-precision direct-sum versus combined-backward discrepancy is nonzero, and the original R1 subtraction failure remains failed. Supported direct-sum relative differences have medians0.000705..0.001028 and maximum0.001516 across the logged role/fold/endpoint summaries; these are diagnostics, not relaxed reference gates (`claims_cost_replay.json`).

Paired records/pixels and initialization match, but all three folds first differ in warmup scalar trajectories at step2. Maximum total-loss differences are0.0011240244/0.0014045238/0.0017092228. The balance controller is inactive then. This observation prevents a bitwise-isolated coefficient-only causal claim; its underlying runtime source was not independently localized. It is not evidence that source samples or checkpoint selection differed.

## E. Scope, outcomes and scientific gates — WARN / Q1_FAIL

This is one repeatedly developed **internal MSVR310 identity-OOF protocol**, seed42, three folds and two endpoints. Folds and role blocks are not independent training seeds. All600 queries/60 query IDs and complete galleries are covered, with6,000 query×endpoint×output rows,300 paired identity×output rows,120 real training epochs and4,680 role-step records retained.

| Output | Control mAP | Balanced mAP | Paired delta pp | Control R1 | Balanced R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 53.129380561 | 53.129380561 | +0.000000000 | 63.000000 | 63.000000 |
| fused | 53.399383646 | 53.452649370 | +0.053265725 | 62.166667 | 62.333333 |
| cnn | 50.262769794 | 50.280983059 | +0.018213266 | 59.166667 | 59.666667 |
| transformer | 51.500319334 | 51.356378809 | -0.143940526 | 60.500000 | 60.666667 |
| mamba | 52.025537658 | 52.148366770 | +0.122829112 | 61.166667 | 61.333333 |

Fused paired fold gains are **+0.129966707 / -0.107213849 / +0.146774279pp**. Paired identity-bootstrap lower bound is **-0.126141312pp**. The original paired gates are fail/fail/fail/fail/pass: gain≥1pp fails, all-fold nonnegative fails, all-role nonnegative fails (Transformer), bootstrap lower>0 fails, candidate fused strictly best passes.

Balanced versus fixed Signal gains fused+0.323268810pp, with fold gains+4.233440447/-3.163518323/-0.219742706; CNN/Transformer/Mamba are below Signal. Its bootstrap lower is **-1.817469561pp** and the Signal group also passes only1/5. Both complete gate groups are required, hence scientific **Q1_FAIL**.

Bootstrap independently reproduces the registered10,000 draws/seed42/linear2.5th percentile: sample60 identity clusters with replacement and divide summed AP differences by sampled query counts. It is not a macro identity average, seed-variance analysis or protection against repeated method development on these folds. Full raw negative results remain. Fused has240 AP improvements,203 declines,157 ties; Rank-1 repairs3 and new errors2.

Last65 cross-scene objective means decrease in each fold; batch-hard and expanded-hard diagnostics increase in each fold. The aggregate values and all late epochs match the report. This supports a descriptive source-objective observation, not an AdamW-causality claim or generalization mechanism. No heldout epoch scores were invented.

## Actual resource and storage accounting

| Endpoint | Fit epoch seconds | Peak allocated MiB | Peak reserved MiB | Fresh role forwards | History VJP record forwards |
|---|---:|---:|---:|---:|---:|
| fold_0_control | 2595.407 | 6566.926 | 7050 | 97600 | 97024 |
| fold_0_balanced | 2569.485 | 6566.803 | 7228 | 97600 | 97024 |
| fold_1_control | 2565.730 | 6569.351 | 7228 | 97600 | 97024 |
| fold_1_balanced | 2561.982 | 6569.609 | 7228 | 97600 | 97024 |
| fold_2_control | 2572.413 | 6569.459 | 7228 | 97600 | 96512 |
| fold_2_balanced | 2580.465 | 6568.664 | 7228 | 97600 | 96512 |

Q1 wall time is15,576.741760s (4.326873h); summed recorded fit epochs are15,445.481150s. Each arm performs780 current-rank and780 direct-auxiliary extra backward calls,292,800 fresh role-record forwards and290,560 historical-VJP record forwards. Thus both arms bear the same measured extra-call counts; equal counts do not imply bitwise trajectories or equal elapsed time. No extra direct-reference forward is made in Q1. There are9,080 historical VJP groups total and585,600 fresh role-record forwards including the six64-record zero-reencoding checks.

Current training exposures total99,840, plus355,244 repeated historical candidate exposures and1,032 unique official-training records in the protocol. Source records exposed per fold are672/683/709, repeated in both arms; exposure counts are not independent samples. Original source Signal prerequisite1950 updates, R2 M0248 updates and archived R1's3 completed updates are separate prior costs, not included as Q1 updates or treated as budget-matched free initialization.

The remotely inventoried whole run occupies **1,192,654,252 bytes**, of which Q1 subdirectory files occupy **968,227,607 bytes**. This includes retained evidence; no model/array/image was downloaded. No new inference parameter is introduced, but training derivatives, historical replay, logs and distance storage have real cost.

## F. Evaluation classification — PASS

Main terminal retrieval: **real_gt**, internal train-split identity-OOF. Source label/mask/loss computation also uses dataset labels. T0 synthetic differentiation and scalar fixtures: **simulation_only**, not performance. Direct derivative/RNG/buffer/head/overfit/coefficients: engineering proxies or saved runtime witnesses. No synthetic proxy was substituted for dataset retrieval GT.

## Claim impact and closure

The latest negative-result report is numerically supported. Its completeness claim applies to saved terminal arithmetic and covered records. Superiority, robust unknown-identity transfer beyond this internal protocol, independent multi-seed support, bitwise coefficient-only causality, task-specific AdamW fractions, official performance and whole-goal success are unsupported.

**No remaining engineering or audit computation blocker.** Publish this WARN/CLOSED_WITH_LIMITS audit and retain Q1_FAIL. Replace audit-pending documentation only after attaching the audit; qualify restored source wording as above. Preserve all original failures and current code/config/checkpoint hashes. Do not alter gates, select another epoch, retrain or rescue this failed fixed version on these results. No further experiment was launched or authorized by this audit.

The auditor's own failed attempts and false assumptions are explicit in `failed_checks.json`: default Python runtime failure, one mistaken snapshot path, mutable commit equality assumption, false warmup bitwise equality, and mistaken full-branch width. Scripts, original assertions and remote stderr remain. Corrected audit scripts used source-declared paths/widths and current evidence; no experimental tolerance or gate was changed.

Primary new outputs: `EXPERIMENT_AUDIT.json`, `independent_text_replay.json`, `remote_bindings.json`, `remote_arrays_attempt2.json`, `claims_cost_replay.json`, `all6000_query_endpoint_output_rows.csv`, `all300_identity_output_deltas.csv`, `all4680_gradient_role_rows.csv`, `all120_training_epochs.csv`, `failed_checks.json`, `final_response.md`. `artifact_manifest.json` hashes all final audit artifacts, and `input_manifest.json` plus report `audited_input_hashes` bind supplied inputs. Large arrays/checkpoints remain remote.
