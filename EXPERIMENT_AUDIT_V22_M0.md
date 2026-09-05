# V22 M0 Experiment Integrity Audit

**Date**: 2026-09-05  
**Auditor**: GPT-5.5 xhigh independent artifact audit  
**Scope**: local read-only audit of the listed V22 primary artifacts, with standard-library/NumPy/JSON arithmetic only. The only writes are this Markdown report and `EXPERIMENT_AUDIT_V22_M0.json`.

## Overall verdict

**Overall verdict**: WARN  
**Integrity status**: warn  
**Engineering integrity**: pass  
**Fixed M0 qualification**: pass  
**Scientific qualification**: fail  
**Actual evaluation type**: `source_only_engineering_m0_real_train_source_batches_and_metadata_replay_plus_synthetic_cuda_t0_with_nonterminal_q1_training_log_no_heldout_retrieval`

The V22 M0 engineering evidence is internally coherent and the fixed M0 gate passes. I found no evidence of fabricated M0 metrics, self-normalized score arithmetic, hidden heldout selection, or credited retrieval results in the supplied M0 snapshot. The audit remains `warn` because provenance has stage/possession limits: current local HEAD differs from the recorded execution commit, four current local source/protocol files match recorded execution hashes only after LF normalization, and several important remote possessions are represented only by a file-verification ledger rather than by independently loaded local tensors or images.

Scientific qualification is `fail` for this bounded M0 audit because the supplied artifacts contain no terminal Q1 heldout retrieval receipts, final checkpoints, AP/rank arrays, aggregate gains, bootstrap replay, dev result, official/test result, or SOTA evidence. The log/file-verification artifacts show Q1 had started after M0, but only as a nonterminal training-log observation.

## Stage timeline and capture separation

| Stage | Evidence | What is supported | What is not supported |
|---|---|---|---|
| Metadata replay | `evidence/trifusion_source_camera_metadata_execution_20260905.json:2-21`; `evidence/trifusion_source_camera_metadata_20260905.json:1-18` | Source label/camera metadata replay, no model/image/checkpoint/eval operations | V22 model performance |
| Preregistration | `evidence/trifusion_v22_preregistration_20260905.json:2-31` | Frozen runner/module/test/config/plan source hashes before launch | Any training/evaluation result |
| T0 | `evidence/trifusion_v22_t0_20260905.json:27-42` | Three synthetic CUDA MCNL contract tests passed, no dataset/model/project optimizer steps | Real B64 training or heldout retrieval qualification |
| Launch | `evidence/trifusion_v22_launch_20260905.json:2-31` | Original process PID 34656 launched at commit `5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36` with fixed config/plan/output dir | Terminal result |
| M0 progress capture | `evidence/trifusion_v22_progress_20260905_175506.json:2-32` | At 17:55:06, original process live, M0 passed, zero complete epoch rows in that progress artifact | Later Q1 progress |
| M0 file/log capture | `evidence/trifusion_v22_m0_file_verification_20260905.json:2-10`; `evidence/trifusion_v22_m0_file_verification_20260905.json:225-230` | M0 summary/log hashes and 3 complete Q1 epoch rows / 87 Q1 steps in the log by 17:57:12; partial current epoch explicitly excluded | Complete Q1, final checkpoint, heldout retrieval |
| Result/tracker publication | `results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_M0_2026-09-05.md:25-40`; `refine-logs/v22/EXPERIMENT_TRACKER.md:36-50` | M0 pass and Q1 running status; no retrieval conclusion | Dev/official/SOTA or terminal Q1 outcome |

## A-F checks

### A_gt_provenance: PASS — GT provenance and path isolation

Evidence:

- tools/run_signal_baseline_dev.py:27-37
- tools/build_v12_complete_path_oof_targets.py:29-56
- tools/build_v12_complete_path_oof_targets.py:156-173
- tools/train_signal_preserving_v22.py:59-89
- tools/train_signal_preserving_v22.py:149-171
- tools/audit_source_camera_supervision.py:51-62
- evidence/trifusion_v22_m0_seed42_5ae096b.json:30-43
- evidence/trifusion_v22_m0_array_verification_20260905.json:6-33
- evidence/trifusion_source_camera_metadata_20260905.json:1-52
- evidence/trifusion_source_camera_metadata_20260905.json:124439-124470
- evidence/trifusion_source_camera_metadata_20260905.json:244583-244613
- protocols/rgbnt201_dev_v1.json:15-19
- protocols/rgbnt201_dev_v1.json:126-132

Findings:

- Dataset identity and camera labels are parsed from RGBNT201 filenames and modality-triplet paths in source code, not generated from model outputs.
- Complete-path folds exclude heldout IDs and relabel only fit records; independent replay found 94 fit IDs and 47 heldout IDs per fold with zero fit/heldout overlap.
- M0 preflight compared both endpoints per fold for binding, initial state, raw batch receipts, output hashes and camera-pair support; all three fold pairings replayed true.
- Metadata replay covers 1680 source batches and 107520 sample exposures while recording zero model/image/checkpoint/eval operations.
- Real heldout retrieval GT is planned, but the supplied M0 snapshot contains no completed heldout retrieval fold.

Claim impact: Supports source-only engineering M0 provenance and path isolation. It does not support heldout retrieval, generalization, or official/test claims.

### B_mathematics: PASS — MCNL/loss arithmetic and fixed M0 gate

Evidence:

- modeling/trifusion/camera_negative_v22.py:8-23
- modeling/trifusion/camera_negative_v22.py:26-50
- tests/test_trifusion_camera_negative_v22.py:9-21
- tests/test_trifusion_camera_negative_v22.py:32-44
- tests/test_trifusion_camera_negative_v22.py:47-71
- modeling/trifusion/criterion.py:17-34
- modeling/trifusion/signal_preserving_v8.py:696-742
- tools/train_signal_preserving_v22.py:100-128
- tools/train_signal_preserving_v22.py:131-146
- tools/train_signal_preserving_v22.py:300-328
- configs/RGBNT201/TriFusion-signal-preserving-v22-camera-negative-rtx3090.yml:38-48
- evidence/trifusion_v22_m0_seed42_5ae096b.json:8164-8175
- evidence/trifusion_v22_m0_seed42_5ae096b.json:8191-8205
- evidence/trifusion_v22_m0_seed42_5ae096b.json:8543-8570
- evidence/trifusion_v22_m0_seed42_5ae096b.json:10858-10889
- evidence/trifusion_v22_m0_array_verification_20260905.json:35-42
- evidence/trifusion_v22_m0_array_verification_20260905.json:43-183

Findings:

- MCNL positives are same-ID non-self rows; same-camera and other-camera negatives are different-ID rows split by camera. The denominator is the mean over rows with positives and both negative groups present.
- MCNL normalizes embeddings and computes distances in FP32. This is feature normalization, not metric normalization or selection.
- The selected camera-negative endpoint uses common identity/fused/branch terms plus weighted MCNL residual terms; ordinary residual triplet is recorded as a diagnostic but not selected for that endpoint.
- Independent arithmetic recomputed the identity entropy floor and fixed first/100th excess-loss ratio exactly against the summary/sidecar.
- The fixed 100-step selected camera_residual_metric went from 0.1082095205783844 to 0.0, while unselected ordinary_residual_triplet rose from 0.000351776834577322 to 0.03214460238814354.

Claim impact: Supports the fixed engineering M0 pass and exact MCNL arithmetic. It does not establish retrieval performance or causal generalization.

### C_file_provenance: WARN — File existence and provenance

Evidence:

- evidence/trifusion_v22_preregistration_20260905.json:2-31
- evidence/trifusion_v22_t0_20260905.json:2-42
- evidence/trifusion_v22_launch_20260905.json:2-58
- evidence/trifusion_v22_m0_seed42_5ae096b.json:3-32
- evidence/trifusion_v22_m0_file_verification_20260905.json:2-10
- evidence/trifusion_v22_m0_file_verification_20260905.json:11-220
- evidence/trifusion_v22_m0_file_verification_20260905.json:223-230

Findings:

- Current local HEAD is f63889f57c4f1375d3fd9cc28e786b42d387e09b; M0 execution commit is 5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36; remote file-verification observation commit is 2fd6506e3295e401313a07afa6f5505c230eb178. These are distinct stages.
- Local raw-byte comparison against M0 source_file_sha256 found 13/17 raw matches and 4/17 LF-normalized-only matches.
- LF-normalized-only local matches: modeling/trifusion/criterion.py, modeling/trifusion/experts/mamba.py, modeling/trifusion/experts/semantic_residual.py, protocols/rgbnt201_dev_v1.json.
- Remote file verification reports 30 raw matches including CLIP, V12 summary and six source checkpoints, but this audit did not independently possess/load those remote tensors or images.
- All listed local primary evidence files exist and were hashed directly.

Claim impact: No contradiction in M0 arithmetic or engineering was found, but provenance claims require raw-vs-LF and remote-possession qualification.

### D_live_execution: PASS — Execution and engineering

Evidence:

- tools/train_signal_preserving_v22.py:244-285
- tools/train_signal_preserving_v22.py:286-328
- tools/train_signal_preserving_v22.py:329-332
- tools/train_signal_preserving_v22.py:333-407
- tools/train_signal_preserving_v22.py:174-193
- tools/train_signal_preserving_v22.py:196-241
- tools/train_signal_preserving_v22.py:340-365
- evidence/trifusion_v22_launch_20260905.json:2-31
- evidence/trifusion_v22_progress_20260905_175506.json:2-32
- evidence/trifusion_v22_m0_run_snapshot_20260905.log:25
- evidence/trifusion_v22_m0_run_snapshot_20260905.log:44
- evidence/trifusion_v22_m0_run_snapshot_20260905.log:63
- evidence/trifusion_v22_m0_run_snapshot_20260905.log:91
- evidence/trifusion_v22_m0_run_snapshot_20260905.log:101-103
- evidence/trifusion_v22_m0_file_verification_20260905.json:225-230

Findings:

- The executed M0 path calls three-fold/two-endpoint preflight, two 8-step capacity runs, and one 100-step fixed-batch camera-negative overfit run before Q1.
- M0 counts replay to 6 preflight models, 48 forward-only batches, 16 capacity optimizer steps, 100 fixed-batch optimizer steps, and 116 total project forward/backward pairs.
- M0 recorded all 203/203 trainable tensors with nonzero finite gradients, zero overflow, frozen state unchanged, and capacity peaks 6054.0/6200.0 MiB.
- T0 records three synthetic CUDA tests passing with dataset_access_count/model_instantiations/project_optimizer_steps/toy_optimizer_steps all zero.
- The captured log shows Q1 began after M0 and had 3 complete epoch rows / 87 optimizer steps by 17:57; partial current-epoch steps were explicitly not included and checkpoint_count_at_capture was 0.
- Save/strict-reload/evaluation code exists for terminal Q1, but no supplied report contains a completed fold receipt, final checkpoint or heldout metric from that path.

Claim impact: Supports engineering-pass M0 and nonterminal Q1-start observation. It does not support terminal Q1 training or evaluation claims.

### E_scope_selection: WARN — Scope and selection

Evidence:

- refine-logs/v22/EXPERIMENT_PLAN.md:68-87
- refine-logs/v22/EXPERIMENT_PLAN.md:89-107
- refine-logs/v22/EXPERIMENT_PLAN.md:109-127
- refine-logs/v22/EXPERIMENT_TRACKER.md:3-14
- refine-logs/v22/EXPERIMENT_TRACKER.md:16-33
- refine-logs/v22/EXPERIMENT_TRACKER.md:36-50
- results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_M0_2026-09-05.md:1-23
- results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_M0_2026-09-05.md:25-40
- evidence/trifusion_v22_m0_seed42_5ae096b.json:8164-8166
- evidence/trifusion_v22_m0_file_verification_20260905.json:225-230

Findings:

- The fixed plan requires 3 folds x 2 endpoints x 20 epochs, 3360 optimizer steps, 120 epoch rows, final-only checkpoints and heldout evaluation for Q1.
- The supplied M0 snapshot has literal folds: [] and no aggregate, matched_gains_mAP, scientific_checks, bootstrap, checkpoints or heldout metrics.
- The 17:57 snapshot separately accounts for nonterminal Q1 progress: 3 complete logged epochs and 87 optimizer steps, partial current-epoch steps excluded, checkpoint_count_at_capture 0.
- The result/tracker state that M0 has no retrieval-performance conclusion and Q1 is running. Complete-training, heldout-gain, dev, official or SOTA claims would exceed these files.

Claim impact: M0 engineering qualification is selectable; no scientific Q1 outcome is selectable from these files.

### F_evaluation_claims: WARN — Evaluation classification and claim impact

Evidence:

- evidence/trifusion_v22_m0_seed42_5ae096b.json:30-43
- evidence/trifusion_source_camera_metadata_20260905.json:1-18
- evidence/trifusion_v22_t0_20260905.json:34-42
- evidence/trifusion_v22_m0_file_verification_20260905.json:4-10
- evidence/trifusion_v22_m0_file_verification_20260905.json:225-230
- tools/train_signal_preserving_v22.py:379-405
- modeling/trifusion/signal_preserving_v13.py:253-279
- results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_M0_2026-09-05.md:16-23
- results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_M0_2026-09-05.md:37-40

Findings:

- Actual executed evidence types are source-only engineering M0 over real train-source batches, metadata-only source label/camera replay, synthetic CUDA T0 tests, and nonterminal Q1 training-log progress.
- The planned post-M0 Q1 evaluation would be real-GT train-internal complete-path OOF reused development qualification, but the supplied M0 files do not contain a terminal Q1 retrieval result.
- Identity-cluster bootstrap and scientific gates are implemented for the terminal Q1 path but are unexecuted in the supplied M0 snapshot.
- Mechanism claims may say MCNL was optimized and reached zero on the fixed M0 batch. They may not say camera information was removed, generalization improved, retrieval improved, or SOTA/dev/official goals were met.

Claim impact: Engineering M0 pass is supported. Scientific claims are not eligible from the supplied artifacts.

## Independent arithmetic replay

Arithmetic was run with Python 3.12.14 and NumPy 2.3.5, using only JSON, integer and floating-point operations over supplied artifacts. Runtime for the report-generation replay was `0.27603819998330437` seconds.

Protocol replay:

- `protocols/rgbnt201_dev_v1.json:15-19` records 3126 train triplets, 825 dev triplets and 3951 train_171 triplets.
- `protocols/rgbnt201_dev_v1.json:126-132` records `uses_test_labels: false` and empty `test_identity_overlap`.
- Independently derived counts: 141 train IDs, 30 dev IDs, 30 test IDs, train/dev overlap 0, train/test overlap 0, dev/test overlap 0.

M0 counts replay:

| Quantity | Recomputed value |
|---|---:|
| Preflight models | 6 |
| Preflight forward-only batches | 48 |
| Capacity optimizer steps | 16 |
| Overfit optimizer steps | 100 |
| Total M0 project optimizer steps | 116 |
| Total M0 forward/backward pairs | 116 |
| Completed Q1 folds in M0 summary | 0 |
| Complete Q1 epoch rows in captured log | 3 |
| Q1 optimizer steps in complete captured log epochs | 87 |
| Checkpoints at file-verification capture | 0 |

Preflight binding and support replay:

| Fold | Fit IDs | Heldout IDs | Overlap | Pairing true | First-8 valid-row range | First-8 same-neg missing rows | Total params | Trainable params | Trainable tensors |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 94 | 47 | 0 | True | 51-64 | 53 | 98800141 | 7841292 | 203 |
| 1 | 94 | 47 | 0 | True | 51-64 | 65 | 98800141 | 7841292 | 203 |
| 2 | 94 | 47 | 0 | True | 47-60 | 71 | 98800141 | 7841292 | 203 |

Source-camera metadata replay from 1680 recorded batches:

| Fold | Source records | Source IDs | Batches | Exposures | Cross-camera-positive rows | Directed cross-camera-positive pairs | Valid MCNL rows | Valid-row range | Incomplete batches | Single-cross-camera-group batches |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2126 | 94 | 580 | 37120 | 6240 | 21396 | 34266 | 42-64 | 295 | 540 |
| 1 | 2075 | 94 | 560 | 35840 | 6080 | 21406 | 33276 | 42-64 | 282 | 520 |
| 2 | 2051 | 94 | 540 | 34560 | 5280 | 17942 | 31254 | 44-64 | 346 | 520 |

Aggregate source-camera replay:

- Total batches: `1680`
- Sample exposures: `107520`
- Sums: `{"all_directed_positive_pairs": 752640, "both_camera_negative_groups_available_rows": 98796, "cross_camera_positive_rows": 17600, "directed_cross_camera_positive_pairs": 60744, "other_camera_negative_missing_rows": 872, "same_camera_negative_missing_rows": 7852}`
- Cross-camera-positive row fraction: `0.1636904761904762`
- Cross-camera-positive pair fraction: `0.08070790816326531`
- Missing either camera-negative group fraction: `0.08113839285714286`
- Min/max valid MCNL rows per batch: `42` / `64`
- Batches with incomplete support: `923`
- Batches with single cross-camera identity group: `1580`
- Differences from `evidence/trifusion_source_camera_metadata_integer_verification_20260905.json:5-22`: all replayed integer/fraction fields matched.

Loss and gate replay:

- Fit-class count: `94`
- Label smoothing: `0.1`
- Correct-class smoothed target: `0.9010638297872341`
- Other-class smoothed target: `0.0010638297872340426`
- Identity entropy: `0.7711772280616134`
- Identity weight sum: `0.75`
- Combined lower bound: `0.57838292104621`
- Registered first fixed-batch total loss: `0.7189050912857056`
- Registered 100th fixed-batch total loss: `0.5803177952766418`
- Excess-loss ratio: `(0.5803177952766418 - 0.57838292104621) / (0.7189050912857056 - 0.57838292104621) = 0.013769174124866987`
- Fixed gate: `0.013769174124866987 <= 0.1` is true.
- Maximum total-component reconstruction difference: `5.960464477539063e-08`
- Maximum MCNL component reconstruction difference: `7.450580596923828e-09`
- Maximum difference versus supplied array-verification scalar/range fields: `0`

Full overfit component ranges:

| Component | First | Step 100 | Min | Max | Last-20 mean |
|---|---:|---:|---:|---:|---:|
| `total` | 0.7189050912857056 | 0.5803177952766418 | 0.5803177952766418 | 0.7189050912857056 | 0.5803416132926941 |
| `common_identity_and_branch_triplet` | 0.6106956005096436 | 0.5803177952766418 | 0.5803177952766418 | 0.6439194083213806 | 0.5803416132926941 |
| `ordinary_residual_triplet` | 0.000351776834577322 | 0.03214460238814354 | 0.000351776834577322 | 0.055489540100097656 | 0.03298937734216452 |
| `camera_residual_metric` | 0.1082095205783844 | 0.0 | 0.0 | 0.1082095205783844 | 0.0 |
| `camera_valid_rows` | 56 | 56 | 56 | 56 | 56.0 |
| `camera_same_negative_missing_rows` | 8 | 8 | 8 | 8 | 8.0 |
| `camera_other_negative_missing_rows` | 0 | 0 | 0 | 0 | 0.0 |
| `camera_cross_camera_positive_rows` | 8 | 8 | 8 | 8 | 8.0 |
| `mcnl_cnn_positive_term` | 0.0 | 0.0 | 0.0 | 3.466968337306753e-05 | 0.0 |
| `mcnl_cnn_camera_term` | 0.16160520911216736 | 0.0 | 0.0 | 0.16160520911216736 | 0.0 |
| `mcnl_cnn_positive_active_rows` | 0 | 0 | 0 | 1 | 0.0 |
| `mcnl_cnn_camera_active_rows` | 47 | 0 | 0 | 47 | 0.0 |
| `mcnl_transformer_positive_term` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `mcnl_transformer_camera_term` | 0.12347975373268127 | 0.0 | 0.0 | 0.12347975373268127 | 0.0 |
| `mcnl_transformer_positive_active_rows` | 0 | 0 | 0 | 0 | 0.0 |
| `mcnl_transformer_camera_active_rows` | 53 | 0 | 0 | 53 | 0.0 |
| `mcnl_mamba_positive_term` | 0.0 | 0.0 | 0.0 | 0.0001170762989204377 | 0.0 |
| `mcnl_mamba_camera_term` | 0.14775311946868896 | 0.0 | 0.0 | 0.14775311946868896 | 0.0 |
| `mcnl_mamba_positive_active_rows` | 0 | 0 | 0 | 1 | 0.0 |
| `mcnl_mamba_camera_active_rows` | 40 | 0 | 0 | 40 | 0.0 |

## Local primary file hashes

| File | Bytes | Lines | SHA256 |
|---|---:|---:|---|
| `tools/train_signal_preserving_v22.py` | 25452 | 417 | `ee17aeba9bc2a567b1163e6a57759a55423ee5f77edde21e6c7a81625fe45a53` |
| `modeling/trifusion/camera_negative_v22.py` | 2358 | 50 | `38b6ccc4e495d55af7e68617cdb6b54fdede9e3f177fa3c54a7f8d0db4465f97` |
| `tests/test_trifusion_camera_negative_v22.py` | 3553 | 71 | `0d529426d2eb7dad8d5fdae1930866fae672e1a7aee24a8fd81793b0ab78eaa3` |
| `configs/RGBNT201/TriFusion-signal-preserving-v22-camera-negative-rtx3090.yml` | 3571 | 83 | `5efd8e6dbf8cd70624902148a52701ad69ae5766825293efa878836a3e74f4f6` |
| `refine-logs/v22/EXPERIMENT_PLAN.md` | 8418 | 136 | `ce2d1f93d5ef36a3eb53d2593fc548b31bff0ea4a3f737b19e1dcaf721d9723b` |
| `refine-logs/v22/EXPERIMENT_TRACKER.md` | 3963 | 51 | `2f9b6e0178da95b563b4401814534ced88c0cb8e6abba6bd8aa858151e51a40b` |
| `results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_M0_2026-09-05.md` | 5038 | 77 | `bd39807a534eb68b06564637fedcb8e9b226fba68390e3eedf8ea2eec362e6e6` |
| `evidence/trifusion_v22_m0_seed42_5ae096b.json` | 374648 | 10892 | `ad0a27abbba79f1c039f68ebcfcc64eba731916581a8dec67a5c64c19d212427` |
| `evidence/trifusion_v22_m0_run_snapshot_20260905.log` | 104518 | 103 | `47dc1a77075d588551f2fe73933da03a739815d40c09d3960ea9b9ef30491369` |
| `evidence/trifusion_v22_m0_file_verification_20260905.json` | 11298 | 231 | `70363bdec91170a59854f59669ecfebcded1d7edce144098ac5566b9f9d0d975` |
| `evidence/trifusion_v22_m0_array_verification_20260905.json` | 4666 | 188 | `55bab84d0098a30f707cab5a871d952a9c7a97133177877d6de3b11f654d1936` |
| `evidence/trifusion_v22_preregistration_20260905.json` | 1289 | 32 | `2f6a2a594e9953d8b7f0110d042cb985a3753fff6d7ef4db07fc23591dc00871` |
| `evidence/trifusion_v22_t0_20260905.json` | 1543 | 43 | `5301aef81deaa33f3fc2966969612f175cb6b4879594d92241f2386b925750f3` |
| `evidence/trifusion_v22_launch_20260905.json` | 3434 | 59 | `c486c4ce1cd37c42faa5b094de70f31011594b78f279ad05a85e6d5f97dc5c9c` |
| `evidence/trifusion_v22_progress_20260905_175506.json` | 913 | 33 | `017cc6470c35eb41bea807129b91511ea9c0cee5eb5b5b40e782521888a02f57` |
| `tools/audit_source_camera_supervision.py` | 7574 | 140 | `3f3e18d832a0f44837ca0a28551b5ecaf9a1081c8e0e3da7e298bc633b4ed6a0` |
| `evidence/trifusion_source_camera_metadata_20260905.json` | 6367917 | 360562 | `5a42be65a512534bb87f52a5f3f4385042157511803774579e65d96d94662d31` |
| `evidence/trifusion_source_camera_metadata_execution_20260905.json` | 4230 | 22 | `8c0193630c8ca5d8dd2a8e6480cca5cfc40f71020f23ba866dc4e04ee09ba319` |
| `evidence/trifusion_source_camera_metadata_integer_verification_20260905.json` | 899 | 23 | `014252d6d3dfa8fe512f770411c9eb43dd7dcd5ad17a49273814f0b957482703` |
| `evidence/trifusion_v19_generalization_geometry_20260905.json` | 47990970 | 1380110 | `0e40093688ed568b7e0584672e4a74098c5fba4e57df06fba4bab1b6405adbe6` |
| `modeling/trifusion/aligned_data.py` | 10062 | 296 | `3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5` |
| `modeling/trifusion/signal_preserving_v8.py` | 28592 | 755 | `97a7b5fe6dab882c2ed92ed1db95b9a8f48eae1d99a3a9b145b0cdfc4b8cefbc` |
| `modeling/trifusion/signal_preserving_v8_builder.py` | 3900 | 103 | `8afb028957ecff1a0a26497f7d0460bc240b0f8612a3c476b52a9a6667a3049d` |
| `modeling/trifusion/experts/mamba.py` | 7479 | 209 | `8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83` |
| `modeling/trifusion/experts/semantic_residual.py` | 15882 | 454 | `c1831a3de031be033624420791cecf2e7a3945e009655089cb48ee4b9912edd6` |
| `modeling/trifusion/criterion.py` | 6231 | 152 | `9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f` |
| `tools/train_signal_preserving_v17.py` | 61124 | 1616 | `333886d16f73987accddb70b0780661bb9400b8afd04af01b57e188d34e5228d` |
| `tools/train_signal_preserving_v18.py` | 17540 | 310 | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` |
| `tools/train_signal_preserving_v19.py` | 26363 | 516 | `5f000d3d3abe9636cae97e292db2c63a3a20f349c125dfe4b27bd3fca95bb9c5` |
| `tools/build_v12_complete_path_oof_targets.py` | 32723 | 857 | `fc13354b1065c46677122ee9cf63816087facca4e0e98ec5e6e2fb0dece141e4` |
| `tools/run_signal_preserving_v5.py` | 65190 | 1682 | `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5` |
| `tools/run_signal_baseline_dev.py` | 12601 | 351 | `083967f2a38267415b2992da98c2ad9429ebb793f387b45daac1dbb4eb16f365` |
| `protocols/rgbnt201_dev_v1.json` | 5685 | 309 | `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d` |

## Recorded execution source hashes versus current local bytes

The M0 summary records the dependency map at `evidence/trifusion_v22_m0_seed42_5ae096b.json:8-25`. Current local raw-byte replay found:

| Source dependency | Bytes | Recorded execution SHA256 | Current local raw SHA256 | Raw match | LF-normalized match |
|---|---:|---|---|---:|---:|
| `modeling/trifusion/aligned_data.py` | 10062 | `3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5` | `3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5` | true | true |
| `modeling/trifusion/camera_negative_v22.py` | 2358 | `38b6ccc4e495d55af7e68617cdb6b54fdede9e3f177fa3c54a7f8d0db4465f97` | `38b6ccc4e495d55af7e68617cdb6b54fdede9e3f177fa3c54a7f8d0db4465f97` | true | true |
| `modeling/trifusion/criterion.py` | 6231 | `0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a` | `9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f` | false | true |
| `modeling/trifusion/experts/mamba.py` | 7479 | `c516c7ad937e5eee6a4ed1e3ec33c2afe3522b751d296bd2e4910e4f27a20ee5` | `8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83` | false | true |
| `modeling/trifusion/experts/semantic_residual.py` | 15882 | `c8cef9717fd7bd1e5e50b428ac92762455defac2e25857c9e0dfaf82729c2a93` | `c1831a3de031be033624420791cecf2e7a3945e009655089cb48ee4b9912edd6` | false | true |
| `modeling/trifusion/signal_preserving_v13.py` | 9347 | `7b7c4abb220ed234608553c77aeedd5f8ef763abb04cce5b21f1bfce0f4daa62` | `7b7c4abb220ed234608553c77aeedd5f8ef763abb04cce5b21f1bfce0f4daa62` | true | true |
| `modeling/trifusion/signal_preserving_v19.py` | 4059 | `69989cced25950ff476759bc93df945b167e8ce88a4340676e8adbbd9f88918e` | `69989cced25950ff476759bc93df945b167e8ce88a4340676e8adbbd9f88918e` | true | true |
| `modeling/trifusion/signal_preserving_v8.py` | 28592 | `97a7b5fe6dab882c2ed92ed1db95b9a8f48eae1d99a3a9b145b0cdfc4b8cefbc` | `97a7b5fe6dab882c2ed92ed1db95b9a8f48eae1d99a3a9b145b0cdfc4b8cefbc` | true | true |
| `modeling/trifusion/signal_preserving_v8_builder.py` | 3900 | `8afb028957ecff1a0a26497f7d0460bc240b0f8612a3c476b52a9a6667a3049d` | `8afb028957ecff1a0a26497f7d0460bc240b0f8612a3c476b52a9a6667a3049d` | true | true |
| `protocols/rgbnt201_dev_v1.json` | 5685 | `d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946` | `f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d` | false | true |
| `tools/audit_v17_full_gallery.py` | 10425 | `856881f984ab8793788d291018a60046a47e309c625a65394b1a3ff4e670d8a9` | `856881f984ab8793788d291018a60046a47e309c625a65394b1a3ff4e670d8a9` | true | true |
| `tools/build_v12_complete_path_oof_targets.py` | 32723 | `fc13354b1065c46677122ee9cf63816087facca4e0e98ec5e6e2fb0dece141e4` | `fc13354b1065c46677122ee9cf63816087facca4e0e98ec5e6e2fb0dece141e4` | true | true |
| `tools/diagnose_v6_oracle_complementarity.py` | 9754 | `919b624156c57f92fa75f79a06fe7c872a02d730fa8cd021ce9d7b4e498b1db2` | `919b624156c57f92fa75f79a06fe7c872a02d730fa8cd021ce9d7b4e498b1db2` | true | true |
| `tools/run_signal_preserving_v5.py` | 65190 | `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5` | `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5` | true | true |
| `tools/train_signal_preserving_v17.py` | 61124 | `333886d16f73987accddb70b0780661bb9400b8afd04af01b57e188d34e5228d` | `333886d16f73987accddb70b0780661bb9400b8afd04af01b57e188d34e5228d` | true | true |
| `tools/train_signal_preserving_v18.py` | 17540 | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` | `f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8` | true | true |
| `tools/train_signal_preserving_v19.py` | 26363 | `5f000d3d3abe9636cae97e292db2c63a3a20f349c125dfe4b27bd3fca95bb9c5` | `5f000d3d3abe9636cae97e292db2c63a3a20f349c125dfe4b27bd3fca95bb9c5` | true | true |

Remote file-verification evidence at `evidence/trifusion_v22_m0_file_verification_20260905.json:11-220` reports 30 raw matches, including CLIP, V12 summary and six V12 source checkpoints. I treat that as a remote ledger. This audit did not load checkpoint tensors or read image bytes.

## Claim impact

| ID | Assessment | Claim | Limit |
|---|---|---|---|
| C1 | supported | V22 fixed M0 engineering gate passed. | Source-only M0, not heldout retrieval. |
| C2 | supported | MCNL formula and line-domain handling match the intended row-wise objective. | Test artifact was read, not re-run; no distance recomputation from absent embeddings. |
| C3 | supported | The fixed 100-step selected objective overfit criterion passed. | Uses the registered first and 100th pre-update total losses. |
| C4 | rejected | The unselected ordinary residual metric also improved. | ordinary_residual_triplet increased from 0.000351776834577322 to 0.03214460238814354. |
| C5 | supported | Q1 was running after M0 and had logged partial nonterminal training progress. | Only 3 complete epoch rows / 87 optimizer steps were evidenced at capture; no terminal outcome. |
| C6 | unsupported | V22 improved heldout retrieval/generalization or passed Q1 scientific gates. | No completed folds, final checkpoints, heldout AP/rank metrics, aggregates, bootstrap, or scientific_checks are present. |
| C7 | rejected | D1/dev/official/test or SOTA evidence exists for V22 from these files. | Recorded dev_access_count=0, official_test_access_count=0, d1_executed=false. |

## Limitations retained

- No model training, inference, checkpoint tensor load, image read, feature extraction, distance recomputation, remote command, network call or download was performed by this audit.
- The raw RGBNT201 image tree, CLIP weights, V12 run summary and six V12 source checkpoints are remote-ledger possessions in this audit; I did not independently possess or load their raw tensors/images locally.
- Current local repository HEAD differs from the V22 execution commit; the audit distinguishes current local files, recorded execution source hashes, remote file verification and publication/report files.
- Four recorded code/protocol dependencies match current local files only after CRLF-to-LF normalization, not raw bytes.
- The V19 geometry file was used only as the stored source label/camera manifest for metadata replay; prior V19 findings were not reused as V22 conclusions.
- The supplied M0 snapshot is nonterminal with respect to Q1. It does not contain final checkpoints, heldout retrieval arrays, aggregate metrics, scientific gates or bootstrap values.
- T0 was audited as a recorded synthetic CUDA math/autograd test artifact and source file, not by rerunning GPU tests.

