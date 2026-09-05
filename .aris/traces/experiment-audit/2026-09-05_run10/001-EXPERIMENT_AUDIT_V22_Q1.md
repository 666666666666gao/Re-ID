# V22 terminal Q1 independent experiment-integrity audit

Date: 2026-09-05

Auditor: GPT-5.5 xhigh independent experiment integrity auditor

Verdict: **integrity WARN; engineering integrity PASS; fixed M0 qualification PASS; scientific qualification FAIL**.

The terminal Q1 artifacts are internally coherent and the engineering run completed as planned, but the fixed scientific outcome is negative. The candidate `camera_negative_residual` endpoint is worse than the matched `batch_hard_residual` control on fused mAP, every fold fused gain is negative, the identity-cluster bootstrap lower bound is negative, and all five preregistered Q1 scientific gates fail. Provenance is not a hard integrity failure, but it remains limited by LF-only local source matches for four text files and by remote-ledger-only possession of CLIP/V12/final checkpoint binaries.

I used only local JSON/NumPy/byte hashing and source reading. I did not run model training, inference, retrieval, image reads, checkpoint/tensor loads, GPU operations, remote commands, network calls, downloads, package installation, or new tests.

## Scope and evidence read

- Audit request: `.aris/traces/experiment-audit/2026-09-05_run10/AUDIT_REQUEST.txt:1` and constraints at `:55-70`; request SHA `793c10bee03fcb95fdb1758b9423a7698a26dc99bdf641898efe4989be9fd60c`.
- Primary terminal Q1 summary: `evidence/trifusion_v22_q1_seed42_5ae096b.json`, SHA `b8cd7db81efc3827a91d165d47e001073785420baf8ddfc8507a2eead9c3d6a3`, literal `status` at `:3`, literal `evaluation_type` at `:30`, and terminal aggregates/gates at `:82064-82156`.
- Full run log: `evidence/trifusion_v22_complete_run_20260905.log`, SHA `a091390a25ced0cfd336fce3c5bd6c51565fcad6becfbf3419778bcb9b7a2f1a`, with preflight lines 25/44/63, M0 line 91, 120 Q1 epoch rows at lines 101-325, and terminal status line 326.
- Local source/config/plan/test/protocol/evidence files listed in the request were byte-read and hashed; full per-file hashes are in `EXPERIMENT_AUDIT_V22_Q1.json` under `recomputation.primary_file_hashes`.

## A. Ground truth and scope — PASS

The source split path is traceable. `tools/build_v12_complete_path_oof_targets.py:29-56` constructs fit and heldout records with fit-only relabeling and explicit overlap reporting, while `tools/build_v12_complete_path_oof_targets.py:156-173` loads the frozen protocol `train_ids` from `train_171`. Filename-derived identity and camera parsing is in `tools/run_signal_baseline_dev.py:27-54`; the protocol checks/counts are in `protocols/rgbnt201_dev_v1.json:2-18`.

The Q1 model loader checks architecture, seed, B64/K8, 20 epochs, MCNL margins, no dev/official/rerank access, source checkpoint hashes, CLIP hash, and V12 summary status in `tools/train_signal_preserving_v22.py:35-56`. It then strictly loads fold-specific Signal and expert states and asserts their fit/heldout IDs match the split in `tools/train_signal_preserving_v22.py:59-89`.

I independently derived legal query indices from every gallery manifest using the rule “same identity exists in a different camera.” The derived masks exactly match every recorded `query_indices` and `excluded_no_cross_camera_positive` array; `query_mask_mismatches` is empty.

| Fold | Gallery records | Gallery identities | Eligible queries | Eligible query identities | Excluded only from query | Camera counts |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 1000 | 47 | 190 | 7 | 810 | {'0': 694, '1': 212, '2': 38, '3': 56} |
| 1 | 1051 | 47 | 179 | 7 | 872 | {'0': 710, '1': 243, '2': 25, '3': 73} |
| 2 | 1075 | 47 | 202 | 7 | 873 | {'0': 769, '1': 281, '2': 25} |

Total heldout gallery records: 3126. Total eligible queries: 571. Total excluded-only-from-query records: 2555. Heldout identity union across folds: 141; heldout identity sum across folds: 141.

For all six endpoint bindings, fit identity count is 94, heldout identity count is 47, fit/heldout overlap is 0, and heldout IDs equal the gallery-manifest IDs. Each paired fold has identical initial state SHA, sample order SHA, first-eight batch receipts, binding, and baseline output. The stale literal metadata key `first_8_batches_exactly_match_v21_m0_indices_and_paths` appears in `evidence/trifusion_source_camera_metadata_20260905.json:34`, but the generating code reads the supplied M0 object at `tools/audit_source_camera_supervision.py:55`; I treat that as a stale key label, not V21 evidence.

## B. Mathematics, losses, scores and gates — FAIL

The arithmetic is consistent, but it proves a negative scientific result. The loss source in `tools/train_signal_preserving_v22.py:100-128` keeps the common V8 identity/triplet objective and switches only the selected residual metric: ordinary residual triplet for `batch_hard_residual`, MCNL for `camera_negative_residual`. V8 label-smoothed ID plus normalized batch-hard triplet is defined in `modeling/trifusion/signal_preserving_v8.py:690-742`, and the ordinary batch-hard triplet uses FP32 `torch.cdist` in `modeling/trifusion/criterion.py:17-34`.

MCNL is defined in `modeling/trifusion/camera_negative_v22.py:8-50`: it forms real same-ID positives, different-ID same-camera negatives, different-ID other-camera negatives, requires all three groups for a valid row, normalizes residual embeddings, computes FP32 Euclidean distances, and averages the two hinge terms on valid rows. The config fixes residual triplet/MCNL weight 0.25, triplet margin 0.3, label smoothing 0.1, and MCNL margins 0.1 at `configs/RGBNT201/TriFusion-signal-preserving-v22-camera-negative-rtx3090.yml:38-48`.

Retrieval evaluation normalizes features before distance computation (`tools/train_signal_preserving_v18.py:133-151`) and uses `full_gallery_scores` (`tools/audit_v17_full_gallery.py:14-43`) plus project AP/rank scoring (`tools/diagnose_v6_oracle_complementarity.py:81-108`). This is feature normalization. I found no self max/min score normalization or post-hoc metric normalization in the evaluated path.

Independent metric replay used `mAP = mean(AP) * 100` and `Rank-k = mean(first_match_rank <= k) * 100`. Every endpoint/output/fold metric, aggregate, gain and gate matched the Q1 summary. Bootstrap replay used `np.random.default_rng(42)`, 21 identity clusters, 10000 resamples, whole-identity sampling with replacement, concatenated sampled query rows, and the 2.5 percentile multiplied by 100, matching `modeling/trifusion/signal_preserving_v13.py:253-279`.

| Output | Control mAP | Candidate mAP | Candidate-Control mAP pp | Control R1 | Candidate R1 | Control R5 | Candidate R5 | Control R10 | Candidate R10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_only | 77.487603116011 | 77.487603116011 | +0.000000000000 | 79.334500875657 | 79.334500875657 | 89.492119089317 | 89.492119089317 | 93.520140105079 | 93.520140105079 |
| fused | 80.640676532655 | 78.984454283112 | -1.656222249542 | 83.712784588441 | 82.311733800350 | 90.017513134851 | 90.718038528897 | 93.520140105079 | 94.045534150613 |
| cnn | 80.049438232661 | 79.152125933187 | -0.897312299474 | 84.588441330998 | 83.362521891419 | 88.966725043783 | 89.667250437828 | 91.243432574431 | 91.068301225919 |
| transformer | 79.676201851587 | 75.410761504161 | -4.265440347426 | 82.311733800350 | 79.509632224168 | 91.418563922942 | 89.316987740806 | 93.870402802102 | 93.345008756567 |
| mamba | 77.742993973068 | 78.060618565521 | +0.317624592453 | 78.283712784588 | 81.260945709282 | 88.441330998249 | 90.017513134851 | 93.870402802102 | 93.169877408056 |

Fold fused gains are `[-0.6760682853190474, -2.1499426404255075, -2.140644542781857]`. The identity-cluster bootstrap lower bound is `-3.8769957222550886` mAP. Scientific checks recomputed to `{'aggregate_fused_gain_at_least_1pp': False, 'all_fold_fused_nonnegative': False, 'all_expert_aggregate_nonnegative': False, 'fused_bootstrap_lower_positive': False, 'fused_beats_baseline_and_experts': False}`, matching the literal Q1 fields at `evidence/trifusion_v22_q1_seed42_5ae096b.json:82142-82152`.

| Output | AP improved | AP declined | AP equal | Rank-1 repaired | Rank-1 broken |
|---|---:|---:|---:|---:|---:|
| baseline_only | 0 | 0 | 571 | 0 | 0 |
| fused | 176 | 227 | 168 | 10 | 18 |
| cnn | 196 | 207 | 168 | 9 | 16 |
| transformer | 151 | 286 | 134 | 17 | 33 |
| mamba | 218 | 197 | 156 | 23 | 6 |

Per-identity fused gains were also recomputed from the same AP arrays:

| Encoded ID | Original ID | Queries | Fused gain mAP pp | Weighted contribution pp |
|---:|---|---:|---:|---:|
| 51 | 000159 | 30 | +1.653041832482 | +0.086849833581 |
| 62 | 000191 | 17 | -0.342908725262 | -0.010209191470 |
| 64 | 000195 | 31 | -3.211485177872 | -0.174353836277 |
| 66 | 000198 | 27 | -0.602652179812 | -0.028496688012 |
| 67 | 000200 | 32 | -5.448070292457 | -0.305320927073 |
| 68 | 000201 | 32 | -1.873859344153 | -0.105014884436 |
| 69 | 000204 | 27 | -7.757548668691 | -0.366819289062 |
| 70 | 000209 | 30 | -4.068475244981 | -0.213755266812 |
| 71 | 000212 | 30 | +6.880187279022 | +0.361480942856 |
| 72 | 000213 | 20 | +4.494964241581 | +0.157441829828 |
| 76 | 000217 | 17 | -2.461394078963 | -0.073281434925 |
| 79 | 000220 | 30 | +7.165336698493 | +0.376462523564 |
| 84 | 000228 | 28 | +0.000000000000 | +0.000000000000 |
| 87 | 000232 | 30 | -0.053636839351 | -0.002818047602 |
| 88 | 000233 | 30 | -11.504231620909 | -0.604425479207 |
| 89 | 000235 | 28 | -0.920383680084 | -0.045132649811 |
| 92 | 000239 | 24 | -1.125583671091 | -0.047309996683 |
| 96 | 000250 | 21 | -16.017605951225 | -0.589088835334 |
| 97 | 000251 | 30 | -0.548244935492 | -0.028804462460 |
| 99 | 000257 | 29 | -0.418998792055 | -0.021280148808 |
| 101 | 000261 | 28 | -0.455703708481 | -0.022346241397 |

The M0 fixed gate also replays exactly. Using 94 classes, smoothing 0.1, and identity weight sum 0.75, I recomputed entropy `0.7711772280616134`, floor `0.57838292104621`, and first/100th excess-loss ratio `0.013769174124866987`; the recorded M0 fields are at `evidence/trifusion_v22_q1_seed42_5ae096b.json:82057-82061`.

| Numeric check | Max difference |
|---|---:|
| Endpoint metric fields | 0.0 |
| Aggregates vs Q1 summary | 0.0 |
| Gains vs Q1 summary | 0.0 |
| Fold gains vs Q1 summary | 0.0 |
| Bootstrap lower bound vs Q1 summary | 0.0 |
| Epoch loss component arithmetic | 7.105646293581458e-09 |
| Epoch support arithmetic | 7.105427357601002e-15 |
| Metadata integer replay | 0 |
| M0 floor | 0.0 |
| M0 excess ratio | 0.0 |

## C. File existence and provenance — WARN

All listed primary local files existed and were hashed. Current local Q1 summary SHA is `b8cd7db81efc3827a91d165d47e001073785420baf8ddfc8507a2eead9c3d6a3`; current local run log SHA is `a091390a25ced0cfd336fce3c5bd6c51565fcad6becfbf3419778bcb9b7a2f1a`. Transfer-manifest replay over the locally transferred summary/log/file-verification/receipt files matched every recorded byte count and SHA: `True` (`evidence/trifusion_v22_terminal_transfer_manifest_20260905.json:1-65`).

Standalone receipt equality against embedded endpoints is `True`. The terminal remote verifier records six endpoint receipt equality checks at `evidence/trifusion_v22_terminal_file_verification_20260905.json:264-318`.

Execution commit literal is `5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36` (`evidence/trifusion_v22_q1_seed42_5ae096b.json:4`). Current local HEAD at report generation is `ad7841d57a063148f4b278aec5b87e3d3104af44`; the terminal verifier observed `25e9a1bb8ad36dd8be93c6ba9b4202ba405e6f05` at `evidence/trifusion_v22_terminal_file_verification_20260905.json:6`. I therefore bind the run to source hashes and the execution commit, not to the current local HEAD.

Source/config/plan/protocol hash comparison against the execution literals at `evidence/trifusion_v22_q1_seed42_5ae096b.json:5-28`:

| Path | Raw match | LF-normalized match | Current raw SHA | Execution SHA |
|---|---:|---:|---|---|
| modeling/trifusion/camera_negative_v22.py | True | False | 38b6ccc4e495d55af7e68617cdb6b54fdede9e3f177fa3c54a7f8d0db4465f97 | 38b6ccc4e495d55af7e68617cdb6b54fdede9e3f177fa3c54a7f8d0db4465f97 |
| modeling/trifusion/criterion.py | False | True | 9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f | 0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a |
| modeling/trifusion/experts/mamba.py | False | True | 8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83 | c516c7ad937e5eee6a4ed1e3ec33c2afe3522b751d296bd2e4910e4f27a20ee5 |
| modeling/trifusion/experts/semantic_residual.py | False | True | c1831a3de031be033624420791cecf2e7a3945e009655089cb48ee4b9912edd6 | c8cef9717fd7bd1e5e50b428ac92762455defac2e25857c9e0dfaf82729c2a93 |
| protocols/rgbnt201_dev_v1.json | False | True | f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d | d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946 |
| tools/train_signal_preserving_v22.py | True | False | ee17aeba9bc2a567b1163e6a57759a55423ee5f77edde21e6c7a81625fe45a53 | ee17aeba9bc2a567b1163e6a57759a55423ee5f77edde21e6c7a81625fe45a53 |
| configs/RGBNT201/TriFusion-signal-preserving-v22-camera-negative-rtx3090.yml | True | False | 5efd8e6dbf8cd70624902148a52701ad69ae5766825293efa878836a3e74f4f6 | 5efd8e6dbf8cd70624902148a52701ad69ae5766825293efa878836a3e74f4f6 |
| refine-logs/v22/EXPERIMENT_PLAN.md | True | False | ce2d1f93d5ef36a3eb53d2593fc548b31bff0ea4a3f737b19e1dcaf721d9723b | ce2d1f93d5ef36a3eb53d2593fc548b31bff0ea4a3f737b19e1dcaf721d9723b |

Four current local text files are LF-only matches rather than raw-byte matches: `['modeling/trifusion/criterion.py', 'modeling/trifusion/experts/mamba.py', 'modeling/trifusion/experts/semantic_residual.py', 'protocols/rgbnt201_dev_v1.json']`. This is the principal local source-provenance warning. The remote verifier lists `36 files including CLIP, V12 checkpoints, and final V22 checkpoints (`evidence/trifusion_v22_terminal_file_verification_20260905.json:222-318`); those remote binaries were not independently possessed or tensor-loaded in this local audit.

## D. Live execution and engineering — PASS

The executable source implements optimizer, AMP, gradient coverage and frozen-state checks in `tools/train_signal_preserving_v22.py:131-146`, preflight in `:149-171`, 20-epoch endpoint training in `:196-241`, and final checkpoint save/rebuild/strict reload/read-only evaluation in `:333-407`.

Parsed log evidence: preflights `[{'line': 25, 'fold': 0}, {'line': 44, 'fold': 1}, {'line': 63, 'fold': 2}]`, M0 `{'line': 91, 'passed': True}`, six Q1 final rows `[{'line': 130, 'fold': 0, 'endpoint': 'batch_hard_residual'}, {'line': 169, 'fold': 0, 'endpoint': 'camera_negative_residual'}, {'line': 208, 'fold': 1, 'endpoint': 'batch_hard_residual'}, {'line': 247, 'fold': 1, 'endpoint': 'camera_negative_residual'}, {'line': 286, 'fold': 2, 'endpoint': 'batch_hard_residual'}, {'line': 325, 'fold': 2, 'endpoint': 'camera_negative_residual'}]`, final status `{'line': 326, 'status': 'Q1_FAIL'}`, and `120` epoch JSON rows equal saved histories = `True`. The progress snapshot records process not live, Q1_FAIL, three completed paired folds, six receipts, 120 epoch log rows, elapsed seconds `4165.238463401794`, GPU `1, 24126, 0`, and exit code `0` at `evidence/trifusion_v22_progress_20260905_204357.json:1-58`.

| Fold | Endpoint | Epochs | Optimizer steps | Overflow | Nonzero/trainable tensors | Missing gradients | Frozen unchanged | Strict reload | Read-only eval |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | batch_hard_residual | 20 | 580 | 0 | 203/203 | 0 | True | True | True |
| 0 | camera_negative_residual | 20 | 580 | 0 | 203/203 | 0 | True | True | True |
| 1 | batch_hard_residual | 20 | 560 | 0 | 203/203 | 0 | True | True | True |
| 1 | camera_negative_residual | 20 | 560 | 0 | 203/203 | 0 | True | True | True |
| 2 | batch_hard_residual | 20 | 540 | 0 | 203/203 | 0 | True | True | True |
| 2 | camera_negative_residual | 20 | 540 | 0 | 203/203 | 0 | True | True | True |

Q1 optimizer steps sum to `3360`. M0 used 8 control capacity steps, 8 candidate capacity steps, and 100 fixed-batch candidate steps. T0 is synthetic-only and records dataset access 0, model instantiations 0, project optimizer steps 0, toy optimizer steps 0 at `evidence/trifusion_v22_t0_20260905.json:37-42`. Terminal verification artifacts record no new optimizer steps or retrieval evaluations (`evidence/trifusion_v22_terminal_transfer_manifest_20260905.json:63-64`; `evidence/trifusion_v22_terminal_log_and_loss_verification_20260905.json:249-251`; `evidence/trifusion_v22_terminal_file_verification_20260905.json:315-317`).

Per-arm source camera support, independently summed from frozen metadata, is the same for both endpoints: valid rows `98796`, same-camera-negative-missing rows `7852`, other-camera-negative-missing rows `872`, cross-camera-positive rows `17600`. Metadata integer replay covers 1680 batches and 107520 per-arm sample exposures with zero integer discrepancy against `evidence/trifusion_source_camera_metadata_integer_verification_20260905.json:2-22`.

## E. Scope and selection — WARN

The plan fixed 3 folds, 2 endpoints, 20 epochs, 120 epoch records and 3360 optimizer steps in `refine-logs/v22/EXPERIMENT_PLAN.md:68-87`, with final-only checkpoint evaluation and no intermediate epoch selection. The Q1 gates and no-post-failure scan/refit/dev/official rule are fixed in `refine-logs/v22/EXPERIMENT_PLAN.md:109-125`.

The artifacts retain all planned arms and failed outcomes: 30 fold/output/endpoint metric rows, all 21 eligible query identity rows, all query paired-change counts, all six training/checkpoint bindings, six standalone receipts, and complete 20-epoch histories. The result/tracker explicitly record Q1_FAIL and no D1/dev/official execution at `results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_2026-09-05.md:157-160` and `refine-logs/v22/EXPERIMENT_TRACKER.md:46-49`.

This check remains WARN because the complete Q1 scope is still a reused train-internal complete-path OOF development qualification (`evidence/trifusion_v22_q1_seed42_5ae096b.json:30-33`), not independent dev/test evidence. The warning is about claim scope, not hidden selection.

## F. Evaluation classification and claim impact — FAIL

Classification: T0 is `synthetic_cuda_loss_formula_and_autograd_contract_only` (`evidence/trifusion_v22_t0_20260905.json:42`); M0 is `source_only_engineering_m0_real_train_source_batches` in the immutable M0 snapshot (`evidence/trifusion_v22_m0_seed42_5ae096b.json:30`); terminal Q1 is `real_gt_train_internal_complete_path_oof_reused_development_qualification` (`evidence/trifusion_v22_q1_seed42_5ae096b.json:30`). Dev access, official test access and D1 execution are zero/false at `evidence/trifusion_v22_q1_seed42_5ae096b.json:40-42`.

The following claims are supported: terminal Q1 completed; fixed M0 engineering gate passed; all attempted Q1 arms and negative outcomes were retained. The following claims are rejected: MCNL improves fused mAP by +1pp, all fold fused gains are nonnegative, all expert aggregate gains are nonnegative, fused bootstrap lower bound is positive, candidate fused beats baseline and every expert, independent dev/official/test advancement, and SOTA. Mechanism/camera-causality/generalization claims are unsupported by this evidence.

## Recomputed numerical results

- Arithmetic runtime: `0.654931` seconds using local Python 3.12.14 and NumPy `2.3.5`.
- Q1 summary status: `Q1_FAIL`; elapsed seconds literal: `4165.238463401794`.
- Candidate-control fused mAP gain: `-1.6562222495424237` percentage points.
- Fold fused gains: `[-0.6760682853190474, -2.1499426404255075, -2.140644542781857]`.
- Bootstrap: lower_bound_95_mAP `-3.8769957222550886`, clusters `21`, resamples `10000`.
- All scientific checks: `{'aggregate_fused_gain_at_least_1pp': False, 'all_fold_fused_nonnegative': False, 'all_expert_aggregate_nonnegative': False, 'fused_bootstrap_lower_positive': False, 'fused_beats_baseline_and_experts': False}`.
- Max numeric difference vs summary fields: `0.0`.

## Limitations and exclusions

- Local read-only artifact audit except for the two requested reports; no model training/inference/retrieval, image reads, checkpoint/tensor loads, GPU, remote, network, downloads, package installation, or new tests.
- Distances and features were not recomputed because raw embeddings/distances were absent and model/tensor loading was outside scope. Metric replay starts from supplied AP/rank/query arrays and gallery identity/camera manifests.
- Remote CLIP, V12 checkpoints and V22 final checkpoints are evidenced by the terminal remote verifier ledger; they were not independently possessed as local bytes.
- Four current local text files differ from execution hashes as raw bytes but match after LF normalization: modeling/trifusion/criterion.py, modeling/trifusion/experts/mamba.py, modeling/trifusion/experts/semantic_residual.py, protocols/rgbnt201_dev_v1.json.
- Current local repository HEAD differs from the execution commit; source/config/plan identity is assessed by recorded execution hashes and current raw/LF-normalized content, not by HEAD equality.
- Q1 is reused train-internal complete-path OOF development qualification, not independent dev, official test, public benchmark SOTA, or causal mechanism evidence.

The audit conclusion is therefore: experiment-integrity **WARN**, engineering execution **PASS**, fixed M0 engineering qualification **PASS**, and V22 terminal Q1 scientific qualification **FAIL**.
