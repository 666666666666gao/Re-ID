# Experiment Audit: MSVR310 TriFusion Original Three-Role Terminal Comparison

Generated: 2026-09-06T09:26:08.251932+08:00 (Asia/Shanghai).

Verdict: **integrity PASS with limits, engineering PASS, scientific FAIL**. The completed comparison is internally reproducible from the provided text/JSON evidence and timestamped receipts. It does not support the original three-role method scientifically under its own fixed gates.

## A-F verdicts

| Category | Verdict | Meaning |
|---|---:|---|
| A - Protocol, labels, and data isolation | PASS | 600 valid internal queries, 60 query identities, 1032 gallery records, source/heldout disjoint per fold, and no official test/dev access. |
| B - Metric and score computation | PASS | AP/Rank recomputed from categorical labels after same identity/same scene filtering and complete saved rankings. |
| C - Independent replay and number/file agreement | PASS | Independent stdlib/NumPy replay matched aggregate/fold metrics, query AP/Rank, bootstrap, loss/epoch/LR, protocol counts, and error census; manifest pre/post hash checks passed. |
| D - Execution boundary, reuse, and source binding | PASS | Original fold0 failure and R3 continuation are separated; fold0 training/verified features reused, folds1/2 trained once, no metric-selection retraining found in text receipts. |
| E - Scope and claim qualification | WARN | Evidence is complete for the declared internal OOF comparison, but not an official MSVR310 591/1055 test result; raw tensor/checkpoint binaries are represented by remote receipts only. |
| F - Scientific support | FAIL | All five fixed scientific conditions are false; fused mAP is 1.011990 pp below Signal baseline and the identity bootstrap lower bound is negative. |

## Audit boundary

- I used the `experiment-audit` checklist directly as the terminal reviewer, but did not delegate further because the task explicitly forbids further delegation.
- The immutable input manifest hash matched the requested value: `af1df38e530edd0a7702d7fdd6544b11efa1e317a8697d0a3e02edd9ce5a9059`. Pre and post hash checks covered 102 files and 25,929,381 bytes; both reported no mismatches.
- Execution was local text/JSON/stdlib/NumPy only. Python: `C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe`. NumPy path: `D:\Program Files\UserCache\gb\uv\archive-v0\92ICCcZmeDTDf2G1EDSF0\Lib\site-packages`. NumPy version: `2.5.2`.
- I did not use torch, model runtime, tensor/image libraries, remote commands, network, browser, package installation, or further delegation.
- Signal B0 and original M0 were treated as separately audited prerequisites. I inspected their commit/config/protocol/checkpoint/initialization bindings only and did not repeat their prior 1950/124-update numerical audits.
- The package does not contain raw `.pt`/`.pth` tensor/checkpoint binaries. Stored-array and checkpoint parity beyond text/JSON rankings is therefore receipt-audited against terminal remote hashes and shapes, not locally reloaded from binary tensors.
- Initial audit observation: `.aris/traces/experiment-audit/2026-09-06_run19/dispatch_observation.json` was not present when I first checked. It now exists and records `recorded_at=2026-09-06T09:21:14.338716+08:00`, after dispatch, with `resolved_backend_independently_attested=false` and `review_family_scope=GPT-family Type-A; no cross-family attestation`. This corrects the provenance wording only: requested/accepted metadata and delayed `list_agents` observation still do not attest backend identity or cross-family independence.

## Replay command and artifacts

Complete stdout, command line, start/end timestamps, and exit code are preserved in `.aris/traces/experiment-audit/2026-09-06_run19/terminal_audit_replay_stdout.txt`.

```powershell
& 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' '.aris\traces\experiment-audit\2026-09-06_run19\replay_msvr310_terminal_audit.py' --manifest '.aris\traces\experiment-audit\2026-09-06_run19\input_manifest.json' --output '.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_replay_result.json' --query-output '.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_query_outputs.jsonl' --training-output '.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_training_steps.jsonl' --epoch-output '.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_epoch_means.jsonl' --hash-pre-output '.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_hashcheck_pre.jsonl' --hash-post-output '.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_hashcheck_post.jsonl'
```

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `.aris\traces\experiment-audit\2026-09-06_run19\replay_msvr310_terminal_audit.py` | 47636 | `af5b454e0130caa7d672315a54d1ab5e21dd9cc12d1f66ed5dd00dbc528aa0be` |
| `.aris\traces\experiment-audit\2026-09-06_run19\build_terminal_audit_report.py` | 30935 | `e2cd78b985fc34441ee6968a38cd9e67d419a48dd5e142000537936416fd06ba` |
| `.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_replay_stdout.txt` | 2693 | `c9314664439a22297e386b09c9646f2da52a24f5e4750d5941d717eb92a9a85e` |
| `.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_replay_result.json` | 43855 | `4934804314a1634bfd5f3fba52496346f282fefd8ace9e6d33f09ce390ccca43` |
| `.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_epoch_means.jsonl` | 16795 | `0d456134379dc9eb630283bdd6c267b09798c0d3e2d9612de1fab20aa74e7f91` |
| `.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_hashcheck_post.jsonl` | 35566 | `43311cd7de1e7d23a09eeb2d025655382c0060329cb270009192d6a1ee4f2ff6` |
| `.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_hashcheck_pre.jsonl` | 35464 | `5414c51f15ec2a1b35bce3bb11a706594554191c3e5d7f624f7295573a126a1b` |
| `.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_query_outputs.jsonl` | 1862577 | `0719fe30bf6d3cc7059212202f5208dad23a0db23d3d9d8a4cf56c31f45c8409` |
| `.aris\traces\experiment-audit\2026-09-06_run19\terminal_audit_training_steps.jsonl` | 323663 | `af395a8585d0b6c82e173d2eb2d9fb116b2afb37ca1ec5428950542b7e1f4f59` |

## Independent metric replay

- Recomputed query-output evaluations: 3000 = 600 queries x 5 outputs.
- Distinct query identities: 60.
- Max aggregate metric absolute difference to comparison summary: 0.0.
- Max fold metric absolute difference to fold receipts: 0.0.
- Max AP absolute difference to receipt query AP lists: 0.0.
- Rank mismatch count upper bound from replayed first positive rank: 0.
- Bootstrap lower-bound difference to summary: 0.0.

Metric implementation basis: `scene_scores` sorts distances, removes gallery records with same identity and same scene as the query, computes AP from positive positions, and records first positive rank (`tools/train_msvr310_signal_oof.py:223-238`). TriFusion evaluation normalizes features, computes pairwise distances, verifies baseline-only distances against B0, writes complete rankings, and calls the same scene-filtered metric (`tools/train_msvr310_trifusion_oof.py:196-241`).

| Output | mAP | Rank-1 | Rank-5 | Rank-10 | Gain vs Signal mAP pp |
|---|---:|---:|---:|---:|---:|
| baseline_only | 53.129380560871 | 63.000000000000 | 77.000000000000 | 82.833333333333 | 0.000000000000 |
| fused | 52.117390116586 | 60.833333333333 | 75.333333333333 | 83.166666666667 | -1.011990444285 |
| cnn | 49.707347521850 | 59.166666666667 | 75.500000000000 | 80.833333333333 | -3.422033039021 |
| transformer | 50.332406611819 | 59.166666666667 | 75.666666666667 | 82.833333333333 | -2.796973949052 |
| mamba | 50.791739829709 | 59.166666666667 | 76.166666666667 | 82.833333333333 | -2.337640731162 |

| Fold | Queries | Query identities | Gallery | Signal mAP | Fused mAP | Fused gain pp |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 210 | 20 | 360 | 49.311695077953 | 51.717207585519 | 2.405512507566 |
| 1 | 207 | 20 | 349 | 47.942264565656 | 44.182541627756 | -3.759722937900 |
| 2 | 183 | 20 | 323 | 63.377724617823 | 61.552100328126 | -1.825624289697 |

Query changes versus baseline-only:

| Output | AP improved | AP declined | AP unchanged | Rank-1 repaired | Rank-1 new errors |
|---|---:|---:|---:|---:|---:|
| fused | 267 | 285 | 48 | 24 | 37 |
| cnn | 231 | 332 | 37 | 30 | 53 |
| transformer | 228 | 331 | 41 | 24 | 47 |
| mamba | 248 | 314 | 38 | 28 | 51 |

Identity bootstrap: lower bound -2.939109553971 pp, seed 42, resamples 10000, clusters 60, percentile 2.5, method `linear`. The implementation resamples whole identities while retaining query weights (`modeling/trifusion/signal_preserving_v13.py:253-280`).

Scientific checks from independent replay:

- `all_fold_fused_gains_nonnegative`: False
- `all_full_branches_not_below_signal`: False
- `fused_gain_at_least_1pp`: False
- `fused_strictly_best`: False
- `identity_bootstrap_lower_positive`: False

## Protocol, isolation, and scope

- Aggregate protocol replay: 600 valid queries, 60 query identities, 1032 gallery records, 432 excluded query records retained in gallery, 95 gallery-only distractor identities.
- All protocol paths came from the training split: True. Official test paths seen: 0.
- Fold isolation replay:

| Fold | Source IDs | Source records | Heldout IDs | Gallery | Queries | Query IDs | Excluded query records | Source/Heldout disjoint |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 103 | 672 | 52 | 360 | 210 | 20 | 150 | True |
| 1 | 103 | 683 | 52 | 349 | 207 | 20 | 142 | True |
| 2 | 104 | 709 | 51 | 323 | 183 | 20 | 140 | True |

This supports only the declared internal train-split OOF comparison. It is not the official MSVR310 591-query/1055-gallery evaluation, and the timestamped report itself says the comparison is internal and no RGBNT201 dev/official test access was used (`results/TRIFUSION_MSVR310_ORIGINAL_ROLES_COMPARISON_20260906_085329.md:10-12`).

## Training scalar replay

- Replayed optimizer steps: 780; summary optimizer-step agreement: True.
- Epoch rows replayed: 60; max epoch mean-loss difference: 0.0; max LR difference: 0.0.
- Source record exposures: 49920; same-identity positive pairs: 174720; cross-scene positive pairs: 45539.
- Max weighted loss recomposition difference from serialized components: 5.650023622294498e-07. The max row was below 1e-6; no count, B64/K8, epoch mean, LR, source-membership, or fold total mismatch was found.

Loss composition was independently recomputed from the stored scalar parts using the source formula (`tools/run_signal_preserving_v5.py:99-142`). Learning rate was independently recomputed from warmup/cosine source logic (`tools/run_signal_preserving_v5.py:1621-1628`).

| Fold | Optimizer steps | Epochs | Source exposures | Unique source records exposed | Source IDs exposed | Same-ID pairs | Cross-scene pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 260 | 20 | 16640 | 672 | 103 | 58240 | 15116 |
| 1 | 260 | 20 | 16640 | 683 | 103 | 58240 | 15098 |
| 2 | 260 | 20 | 16640 | 709 | 104 | 58240 | 15325 |

Full per-step replay is in `terminal_audit_training_steps.jsonl` (780 rows). Full per-epoch mean/LR replay is in `terminal_audit_epoch_means.jsonl` (60 rows).

## Fold0 repair, original reuse, and R3 execution

The evidence supports the engineering narrative with explicit boundaries:

- Original execution `1c444cdf72e13fd041afd0c641dc8f522faa5844` trained fold0 for 260 optimizer steps, stopped at baseline feature parity, wrote no retrieval metric outputs, and did not start folds1/2 (`evidence/trifusion_msvr310_trifusion_v1_comparison_failure_receipt_20260906.json:3-8,52`).
- R3 execution `1ff7e2d8ed2ed56668f814e229aca347f57b9b5d` reused fold0 training and verified arrays, trained only folds1/2, and asserted the final 520 new steps, 780 formal optimizer steps, 1032 evaluated gallery records, and 672 new heldout forwards (`tools/resume_msvr310_trifusion_exact_inference.py:82-170`).
- The R3 wrapper pinned HEAD, runner SHA, plan SHA, and absence of an existing resume directory before launch (`evidence/trifusion_msvr310_trifusion_v1_resume_r3_wrapper_20260906.py:9-24`). The launch scope says fold0 training/verified360 features were reused and only folds1/2 were trained (`evidence/trifusion_msvr310_trifusion_v1_resume_r3_launch_20260906.json:1-12`). The run log begins with fold0 reused/new_steps=0, then shows fold1 and fold2 epoch logs (`evidence/trifusion_msvr310_trifusion_v1_resume_r3_run_20260906.log:1-80`).
- Text receipts in the final summary record fold0 `training_reused_from_original_run=true`, fold0 `new_heldout_record_forwards=0`, fold1/2 `training_reused_from_original_run=false`, and final `new_heldout_record_forwards=672` (`evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json:19-22,23727-23729,55700-55704,87494-87498,95589`).
- The exact inference helper temporarily uses a functional dispatch view for the frozen SIM in-projection weight and checks that original parameter object, flags, and grad state remain unchanged (`tools/msvr310_exact_signal_inference.py:4-22`).
- The full360 verification reproduced the unrepaired difference, then confirmed repaired baseline features and 210x360 distances bitwise equal to B0 while model state, frozen state, signal state, parameter flags, and original batch sizes stayed unchanged; it performed zero optimizer/backward/checkpoint/ranking operations (`tools/verify_msvr310_exact_signal_inference.py:41-112`; `evidence/trifusion_msvr310_trifusion_v1_exact_signal_inference_verification_20260906.json:3,11-19,73-80`).

The SIM dispatch diagnosis supports the mechanism within its fixed-input profiler scope: original/repeat/restored requires-grad states keep bitwise parity, while `freeze_sim_only`, build-before-load, load-final-roles, and original-after-load states change the SIM result by max 1.9073486328125e-06 (`evidence/trifusion_msvr310_trifusion_v1_sim_operation_parity_diagnosis_20260906.json:76-87,89-143`). The applicable source evidence is the noncontiguous projection `linear()` path: `Tensor linear` begins at `evidence/msvr310_pytorch251_Linear.cpp.txt:73`, reaches `at::matmul(input, weight.t())` at line 111, and then the `matmul` implementation enters `should_fold` / expanded `bmm` dispatch depending on shape, contiguity, and the smaller operand `requires_grad` state (`evidence/msvr310_pytorch251_Linear.cpp.txt:73-120`; `evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2030-2075,2108-2120,2157-2166`). The earlier `Linear.cpp:180-231` and `Linear.cpp:820-824` excerpts are einsum/tensordot background and are not evidence for this observed projection path. Captured native MHA self-attention fastpath is blocked earlier by `query is not key` at source line 107; the later MHA `requires_grad` fastpath gate is therefore not the observed cause here (`evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:100-110,160-175`).

A precise caveat: the baseline-parity diagnosis shows that unrepaired `standalone_after_wrapping`, `hierarchical_baseline`, and `full_model_baseline` are not bitwise equal to stored B0, with max absolute difference 1.9073486328125e-06 (`evidence/trifusion_msvr310_trifusion_v1_baseline_parity_diagnosis_20260906.json:1-70`). The valid engineering claim is that the inference-only functional view restores exact B0 and preserves role outputs/state, not that the naturally frozen full-model path was already exact.

## Stored-array and remote-binary receipt checks

I could not load binary `.pt`/`.pth` files locally because the immutable package excludes raw tensor/checkpoint binaries. I checked all provided terminal receipt rows against fold receipt hashes, expected shapes, and all text flags. The terminal file verifier itself reports zero local model/tensor runtime in that check and `PASS_WHOLE_FILES_AND_ALL15_ARRAYS` (`evidence/trifusion_msvr310_trifusion_v1_terminal_files_verification_20260906.json:1-8,385-395`).

| Fold | Output | Feature shape | Distance shape | Distance recompute receipt | Ranking sort receipt | Max distance diff |
|---:|---|---:|---:|---:|---:|---:|
| 0 | baseline_only | [360, 3072] | [210, 360] | True | True | 0.0 |
| 0 | fused | [360, 7680] | [210, 360] | True | True | 0.0 |
| 0 | cnn | [360, 4608] | [210, 360] | True | True | 0.0 |
| 0 | transformer | [360, 4608] | [210, 360] | True | True | 0.0 |
| 0 | mamba | [360, 4608] | [210, 360] | True | True | 0.0 |
| 1 | baseline_only | [349, 3072] | [207, 349] | True | True | 0.0 |
| 1 | fused | [349, 7680] | [207, 349] | True | True | 0.0 |
| 1 | cnn | [349, 4608] | [207, 349] | True | True | 0.0 |
| 1 | transformer | [349, 4608] | [207, 349] | True | True | 0.0 |
| 1 | mamba | [349, 4608] | [207, 349] | True | True | 0.0 |
| 2 | baseline_only | [323, 3072] | [183, 323] | True | True | 0.0 |
| 2 | fused | [323, 7680] | [183, 323] | True | True | 0.0 |
| 2 | cnn | [323, 4608] | [183, 323] | True | True | 0.0 |
| 2 | transformer | [323, 4608] | [183, 323] | True | True | 0.0 |
| 2 | mamba | [323, 4608] | [183, 323] | True | True | 0.0 |

Receipt summary from my replay: array_check_count=15, all_flags_true=True, retrieval_array_hashes_match_receipts=True, checkpoint_hashes_match_receipts=True, local_binary_arrays_available=False.

## Dense outputs versus residual-only outputs

The evaluated CNN/Transformer/Mamba outputs are dense extended branches, not standalone residual-only vectors. The V8 source concatenates the baseline embedding with the residual bank and returns expert branch embeddings through `fusion.branch_embeddings[expert]`; retrieval outputs are `baseline_only`, `fused`, or an expert branch (`modeling/trifusion/signal_preserving_v8.py:492-516,620-666,669-676`). The fused output is 7680D and branches are 4608D in the stored-array receipt table above.

## B0/M0 and source binding inspection

- Signal B0 binding inspected: status `COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION`, project commit `bb01d60b6e1517ee6f5dc9120faefd17d75401e5`, fixed epoch50, 1950 optimizer steps, 1032 heldout image forwards, no official test access, aggregate mAP 53.1293805608712 (`evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:4,181765-181793`).
- Original M0 binding inspected: status `PASS_ENGINEERING_ONLY`, project commit `1c444cdf72e13fd041afd0c641dc8f522faa5844`, discarded M0, source-only training, role weights not loaded, 124 optimizer steps, zero heldout forwards (`evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:4,31,3480-3492,12810-12812`).
- Source binding: the manifest maps five actual remote LF project snapshots and 17 Signal source snapshots; the source binding receipt documents CRLF/LF byte differences but local LF equals remote, remote equals git blob, and AST equality for the five changed remote project files, with zero model calls or optimizer updates in that binding step (`evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json:1-86`).

## Error census

I independently replayed the post-hoc error census from the query rows and rankings. Counts match, but the census is descriptive only and cannot establish an exclusive cause. The census file itself states the same limit (`evidence/trifusion_msvr310_trifusion_v1_complete_error_census_20260906.json:1-67,45981`).

| Output | Queries | Rank-1 repaired | Rank-1 new errors | New errors same camera | New errors same scene | All Rank-1 errors | All Rank-1 errors same camera | All Rank-1 errors same scene |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fused | 600 | 24 | 37 | 10 | 4 | 235 | 83 | 87 |
| cnn | 600 | 30 | 53 | 11 | 7 | 245 | 76 | 85 |
| transformer | 600 | 24 | 47 | 10 | 10 | 245 | 77 | 102 |
| mamba | 600 | 28 | 51 | 11 | 7 | 245 | 76 | 83 |

Baseline rank-1 errors: total=222, same_camera=85, same_scene=98. Identity directions: improved=30, declined=26, unchanged=4.

## All 60 identity results

| Identity | Queries | Baseline mAP | Fused mAP | CNN mAP | Transformer mAP | Mamba mAP | Fused delta pp |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 22.488323 | 23.613495 | 20.141506 | 19.618637 | 23.378192 | 1.125172 |
| 5 | 20 | 71.302141 | 57.862554 | 50.819326 | 57.950983 | 55.158979 | -13.439587 |
| 7 | 7 | 35.727401 | 20.657917 | 16.978832 | 18.524894 | 19.717111 | -15.069484 |
| 9 | 28 | 41.797538 | 41.313987 | 42.353975 | 36.153394 | 42.233111 | -0.483551 |
| 11 | 11 | 14.288880 | 16.849922 | 19.524125 | 11.466311 | 22.329018 | 2.561042 |
| 13 | 9 | 57.322280 | 57.976311 | 51.901214 | 57.012729 | 57.003191 | 0.654030 |
| 15 | 10 | 25.151760 | 42.248626 | 35.069329 | 39.699570 | 44.227010 | 17.096866 |
| 17 | 8 | 56.693612 | 59.775526 | 52.662950 | 55.670932 | 55.236498 | 3.081914 |
| 19 | 14 | 81.260675 | 86.797095 | 84.477713 | 77.276347 | 92.153084 | 5.536420 |
| 21 | 9 | 5.976463 | 10.807053 | 8.613602 | 20.090297 | 7.539281 | 4.830590 |
| 23 | 13 | 56.028628 | 51.464766 | 52.846323 | 46.477361 | 48.185975 | -4.563861 |
| 25 | 14 | 85.640268 | 81.486151 | 79.251645 | 80.761010 | 77.091678 | -4.154117 |
| 27 | 8 | 12.758510 | 15.243865 | 20.673520 | 10.284961 | 23.848670 | 2.485355 |
| 29 | 7 | 42.386748 | 64.594289 | 57.520346 | 58.558780 | 71.155670 | 22.207541 |
| 31 | 4 | 77.083333 | 46.180556 | 46.714744 | 50.446429 | 42.719156 | -30.902778 |
| 33 | 16 | 95.171096 | 96.051112 | 94.117103 | 97.223057 | 95.432113 | 0.880016 |
| 35 | 8 | 79.620994 | 67.573986 | 68.062285 | 69.190358 | 64.018072 | -12.047009 |
| 38 | 12 | 96.944149 | 96.256439 | 94.833993 | 94.681862 | 91.682788 | -0.687711 |
| 40 | 13 | 64.448586 | 63.677222 | 64.252122 | 60.490108 | 58.330171 | -0.771364 |
| 46 | 21 | 30.780486 | 26.137553 | 24.482314 | 28.336299 | 24.545700 | -4.642933 |
| 60 | 5 | 62.770563 | 49.377955 | 49.020488 | 61.917897 | 45.110457 | -13.392607 |
| 64 | 14 | 40.962791 | 42.271718 | 42.177512 | 41.012101 | 37.842547 | 1.308927 |
| 68 | 14 | 52.833644 | 44.616909 | 36.540522 | 42.605764 | 47.093175 | -8.216735 |
| 71 | 4 | 38.194444 | 64.722222 | 70.783730 | 30.863095 | 53.264791 | 26.527778 |
| 76 | 8 | 70.920755 | 77.709951 | 77.571224 | 71.420235 | 78.023858 | 6.789196 |
| 78 | 7 | 100.000000 | 100.000000 | 97.345238 | 99.523810 | 100.000000 | 0.000000 |
| 80 | 7 | 69.250523 | 64.011188 | 65.957581 | 55.086136 | 62.054286 | -5.239336 |
| 82 | 6 | 80.925926 | 84.391534 | 76.812169 | 89.880952 | 77.777778 | 3.465608 |
| 85 | 10 | 1.469956 | 1.818379 | 1.457163 | 2.561392 | 1.624775 | 0.348423 |
| 87 | 10 | 10.743522 | 7.943046 | 3.967637 | 12.707420 | 10.022353 | -2.800476 |
| 91 | 5 | 46.749703 | 63.253968 | 62.333333 | 30.039683 | 38.059567 | 16.504265 |
| 93 | 4 | 22.852018 | 13.032698 | 27.667215 | 8.291415 | 6.047126 | -9.819320 |
| 97 | 16 | 62.215821 | 60.929305 | 58.268769 | 57.501467 | 62.444105 | -1.286517 |
| 99 | 31 | 80.026005 | 79.142454 | 75.534775 | 81.332522 | 77.961427 | -0.883550 |
| 101 | 10 | 62.147453 | 50.396505 | 41.347812 | 43.731249 | 50.703525 | -11.750948 |
| 105 | 10 | 81.839483 | 78.729069 | 73.855392 | 71.712474 | 78.394250 | -3.110414 |
| 109 | 3 | 3.834799 | 14.478114 | 46.284271 | 5.246257 | 11.269841 | 10.643315 |
| 113 | 5 | 15.224864 | 7.478237 | 3.732949 | 12.556752 | 9.897377 | -7.746627 |
| 115 | 4 | 100.000000 | 100.000000 | 100.000000 | 95.833333 | 95.833333 | 0.000000 |
| 121 | 9 | 40.031405 | 45.872411 | 42.902533 | 36.697949 | 47.001179 | 5.841006 |
| 130 | 12 | 63.543964 | 54.914212 | 46.941011 | 51.064355 | 58.009856 | -8.629752 |
| 136 | 8 | 21.358278 | 21.999860 | 21.748657 | 25.979373 | 20.893427 | 0.641582 |
| 138 | 6 | 53.724707 | 59.082481 | 58.036828 | 66.514479 | 47.612136 | 5.357773 |
| 140 | 9 | 34.613472 | 38.776850 | 35.829345 | 35.270565 | 36.611916 | 4.163378 |
| 151 | 3 | 83.333333 | 100.000000 | 100.000000 | 75.000000 | 75.000000 | 16.666667 |
| 155 | 5 | 100.000000 | 100.000000 | 98.333333 | 100.000000 | 98.333333 | 0.000000 |
| 157 | 6 | 62.483778 | 44.641835 | 28.661955 | 50.262460 | 40.868625 | -17.841943 |
| 161 | 5 | 90.190476 | 74.833333 | 60.117361 | 85.333333 | 67.611111 | -15.357143 |
| 165 | 8 | 15.408751 | 16.243401 | 11.058570 | 17.076207 | 17.492192 | 0.834650 |
| 167 | 5 | 32.228595 | 49.581669 | 39.567017 | 58.310041 | 42.526652 | 17.353074 |
| 171 | 16 | 89.172665 | 92.083564 | 92.110490 | 92.733809 | 86.405902 | 2.910900 |
| 173 | 6 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 100.000000 | 0.000000 |
| 175 | 12 | 44.184221 | 52.434110 | 47.345334 | 46.320315 | 55.388404 | 8.249889 |
| 179 | 7 | 98.095238 | 99.285714 | 91.150794 | 100.000000 | 94.126984 | 1.190476 |
| 195 | 9 | 3.960659 | 4.134510 | 4.409743 | 4.518340 | 2.989751 | 0.173851 |
| 229 | 12 | 81.723333 | 70.693458 | 58.193370 | 69.904810 | 73.468078 | -11.029875 |
| 231 | 12 | 28.102833 | 37.385400 | 41.338705 | 36.205519 | 28.702143 | 9.282567 |
| 234 | 11 | 0.975793 | 1.088602 | 0.999503 | 1.091765 | 1.229316 | 0.112809 |
| 238 | 13 | 16.498400 | 13.595330 | 13.933520 | 9.282813 | 18.674075 | -2.903070 |
| 254 | 16 | 26.794100 | 14.956427 | 14.600921 | 17.198176 | 14.280869 | -11.837672 |

## Actual issues and required qualification

1. **Scientific support fails.** Do not claim the original three-role method improves MSVR310 under this comparison. Fused mAP is 52.1173901165861 versus Signal/B0 53.1293805608712, a -1.0119904442850967 pp drop; all five predefined scientific checks are false (`tools/train_msvr310_trifusion_oof.py:244-285`; `evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json:94833-94894`).
2. **The exact-helper boundary must stay explicit.** The unrepaired frozen full-model/SIM paths are not bitwise exact to B0. The valid statement is: the inference-only functional view restores exact original B0 for feature extraction while preserving original parameter values/flags, role residuals, and training receipts.
3. **Remote binary evidence is limited.** The local immutable package has no raw tensor/checkpoint binaries. The 15 stored-array path claims and checkpoint hashes are supported by terminal remote receipts and local text/JSON consistency, not by local binary reload. This remains a qualification only; I did not request local tensor/image copies and did not require any new tensor runtime.
4. **Backend reviewer attestation remains unproven.** `001-terminal.request.json` records requested model `gpt-5.5` and reasoning `xhigh`; `dispatch_observation.json` now exists, but it was recorded after dispatch and marks `resolved_backend_independently_attested=false`. This supports only delayed root-side dispatch/list-agents provenance, not backend identity or cross-family independence (`.aris/traces/experiment-audit/2026-09-06_run19/001-terminal.request.json:2,7-8`; `.aris/traces/experiment-audit/2026-09-06_run19/dispatch_observation.json:1-10`).

## Evidence references

- `scene_ap_rank`: `tools/train_msvr310_signal_oof.py:223-238`
- `trifusion_eval`: `tools/train_msvr310_trifusion_oof.py:196-241`
- `comparison_summary`: `tools/train_msvr310_trifusion_oof.py:244-285`
- `run_binding`: `tools/train_msvr310_trifusion_oof.py:288-413`
- `resume_reuse`: `tools/resume_msvr310_trifusion_exact_inference.py:82-109`
- `resume_new_folds`: `tools/resume_msvr310_trifusion_exact_inference.py:111-170`
- `exact_helper`: `tools/msvr310_exact_signal_inference.py:4-22`
- `exact_verifier`: `tools/verify_msvr310_exact_signal_inference.py:41-112`
- `loss_formula`: `tools/run_signal_preserving_v5.py:99-142`
- `lr_formula`: `tools/run_signal_preserving_v5.py:1621-1628`
- `identity_bootstrap`: `modeling/trifusion/signal_preserving_v13.py:253-280`
- `dense_outputs`: `modeling/trifusion/signal_preserving_v8.py:492-516,620-666,669-676`
- `pytorch_linear_algebra`: `evidence/msvr310_pytorch251_LinearAlgebra.cpp.txt:1925-1958,2030-2075,2108-2120,2157-2166`
- `pytorch_linear`: `evidence/msvr310_pytorch251_Linear.cpp.txt:73-120`
- `pytorch_mha`: `evidence/msvr310_sim_operation_pytorch_mha_source_20260906.py.txt:100-110,160-175,621-639,667-678`
- `failure_receipt`: `evidence/trifusion_msvr310_trifusion_v1_comparison_failure_receipt_20260906.json:3-8,52`
- `failed_log`: `evidence/trifusion_msvr310_trifusion_v1_comparison_failed_run_20260906.log:145-166`
- `comparison_complete`: `evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json:1-22,94826-94894,95557-95589`
- `exact_verification_json`: `evidence/trifusion_msvr310_trifusion_v1_exact_signal_inference_verification_20260906.json:3,11-19,73-80`
- `sim_diagnosis_json`: `evidence/trifusion_msvr310_trifusion_v1_sim_operation_parity_diagnosis_20260906.json:76-87,89-143`
- `baseline_parity_json`: `evidence/trifusion_msvr310_trifusion_v1_baseline_parity_diagnosis_20260906.json:1-70,522-535`
- `terminal_files_json`: `evidence/trifusion_msvr310_trifusion_v1_terminal_files_verification_20260906.json:1-60,150-180,360-395`
- `r3_wrapper`: `evidence/trifusion_msvr310_trifusion_v1_resume_r3_wrapper_20260906.py:9-24`
- `r3_launch`: `evidence/trifusion_msvr310_trifusion_v1_resume_r3_launch_20260906.json:1-12`
- `r3_log`: `evidence/trifusion_msvr310_trifusion_v1_resume_r3_run_20260906.log:1-80`
- `source_binding`: `evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json:1-86`
- `signal_b0`: `evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:4,181765-181793`
- `m0`: `evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json:4,31,3480-3492,12810-12812`
- `error_census`: `evidence/trifusion_msvr310_trifusion_v1_complete_error_census_20260906.json:1-67,45981`
- `terminal_request`: `.aris/traces/experiment-audit/2026-09-06_run19/001-terminal.request.json:2,7-8`
- `replay_result`: `.aris/traces/experiment-audit/2026-09-06_run19/terminal_audit_replay_result.json:1-1274`


## Report-only correction round 2

- Correction generated: 2026-09-06T09:32:28.465757+08:00 (Asia/Shanghai).
- Preserved verdicts and all numerical replay values. No 3000-query replay, 780-step replay, training, torch/model runtime, tensor/image library, network, or remote command was run.
- Rechecked immutable manifest inputs without replay: 102 files, 25,929,381 bytes, mismatch_count=0, manifest_sha256=`af1df38e530edd0a7702d7fdd6544b11efa1e317a8697d0a3e02edd9ce5a9059`.
- Corrected dispatch provenance from current absence to initial absence plus delayed observation.
- Corrected PyTorch source references for the observed noncontiguous projection `linear -> matmul -> should_fold/mm/bmm` path and made the MHA fastpath non-cause explicit.
- Removed the prior future-audit wording that could be read as asking for local raw binaries; the remote-binary receipt limit remains only a qualification.
