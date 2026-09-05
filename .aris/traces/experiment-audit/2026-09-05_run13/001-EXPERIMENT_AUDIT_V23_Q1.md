# EXPERIMENT_AUDIT_V23_Q1

Date: 2026-09-05  
Auditor: Codex constrained same-family local audit. The audit request asked for GPT-5.5 xhigh (`.aris/traces/experiment-audit/2026-09-05_run13/AUDIT_REQUEST.txt:1`), and also required no delegation plus no model/tensor/image/GPU/remote/network/installation actions (`.aris/traces/experiment-audit/2026-09-05_run13/AUDIT_REQUEST.txt:55-58`). This is not a cross-family acquittal.  
Request SHA256: `287dc8bdd75493ae5b6506e326f33637ecd0421beef23cefc8cf96d735519e80`

Overall verdict: **WARN**. Engineering integrity passes. Fixed M0 qualifies only as an engineering can-train/preservation check. Scientific qualification fails. The evaluation type is real-ground-truth train-internal complete-path OOF reused-development qualification, not fixed dev and not official test.

| check | status | conclusion |
|---|---:|---|
| A. GT and source isolation | PASS | Real train identities/cameras are used; heldout train-fold separation and no dev/official/test access are consistent. |
| B. Mathematics | WARN | AP/rank arithmetic, matched gains, fold gains, and gates recompute from supplied arrays; exact NumPy bootstrap stream was not regenerated locally. |
| C. Provenance and receipts | WARN | Source/config/plan/receipt binding is internally consistent; four local raw source files differ only by CRLF normalization; checkpoint binaries were not independently loaded. |
| D. Execution and dead code | PASS | Active evaluator path is connected and terminal log/receipt counts agree with 3,360 Q1 optimizer steps and exit code 0. |
| E. Scope and qualification | WARN | Claims are correctly scoped to train-internal OOF; Q1 fails scientific gates and does not qualify D1/dev/official progression. |
| F. Claim impacts | WARN | Negative Q1 result and engineering execution are supported; improvement, mechanism, dev, or official claims are not supported. |

## A. GT and source isolation - PASS

The source/evaluation chain supports real GT identity and camera use inside the train-internal split. The V12 complete-path builder constructs fold records by separating heldout identities from fit identities and reports overlap (`tools/build_v12_complete_path_oof_targets.py:29-56`). RGBNT201 records parse identity and camera from actual filenames (`tools/run_signal_baseline_dev.py:27-54`). The protocol records 3,126 train triplets and keeps train/dev/test identities separated (`protocols/rgbnt201_dev_v1.json:1-19`, `protocols/rgbnt201_dev_v1.json:123-164`). The gallery/query rule is the normal cross-camera ReID rule (`protocols/rgbnt201_dev_v1.json:52-58`).

The V23 plan explicitly fixes three train-internal folds with 94 fit and 47 heldout train identities per fold, reuses V12 complete-path OOF targets, and states this is not independent validation (`refine-logs/v23/EXPERIMENT_PLAN.md:41-49`). The terminal summary records seed 42, no dev/official/rerank, complete-path OOF reuse, and dev/official counts of zero (`evidence/trifusion_v23_q1_seed42_9f4a10b.json:1-45`). The terminal scope aggregates 571 heldout train queries across 141 heldout identities and records D1 as false (`evidence/trifusion_v23_q1_seed42_9f4a10b.json:81254-81280`).

Independently reconstructed fold scopes from the supplied terminal gallery manifests:

| fold | gallery | queries | excluded same-id same-camera pairs | heldout identities |
|---:|---:|---:|---:|---:|
| 0 | 1000 | 190 | 810 | 47 |
| 1 | 1051 | 179 | 872 | 47 |
| 2 | 1075 | 202 | 873 | 47 |

## B. Mathematics - WARN

The active metric path is mathematically coherent. V18 evaluation normalizes embeddings, computes pairwise distances, calls full-gallery scoring, and keeps the model in read-only evaluation state (`tools/train_signal_preserving_v18.py:133-152`). Full-gallery scoring creates eligible queries, excludes same-identity same-camera junk, and returns AP/rank metrics (`tools/audit_v17_full_gallery.py:14-43`). The per-query routine sorts distances, removes same-id same-camera matches, and computes AP and first-match rank (`tools/diagnose_v6_oracle_complementarity.py:81-108`).

The terminal array auditor defines mAP and Rank-k directly from AP/rank arrays (`tools/audit_v23_terminal_arrays.py:18-21`), reconstructs query/exclusion masks (`tools/audit_v23_terminal_arrays.py:68-80`), recomputes metrics/gains/bootstrap/gates (`tools/audit_v23_terminal_arrays.py:117-166`), and records the pass state with NumPy 2.5.2 (`evidence/trifusion_v23_q1_array_verification_20260905.json:1-7`). I independently recomputed the AP/rank aggregate metrics, fold fused gains, matched mAP gains, receipt equality, and gate booleans from the supplied arrays using Node.js v24.13.0. The maximum metric difference from the summary was `5.684341886080802e-14`. Loss-component roundoff was at most `3.3728824289092074e-8`.

Exact NumPy bootstrap resampling was not independently regenerated in this run: local default Python returned `No pyvenv.cfg file`, and `E:/python.exe` could not import NumPy. The stored verifier reports the identity-cluster bootstrap lower bound as `-1.509454322847345` and the result is negative, so this limitation does not affect the scientific failure verdict. It does prevent a full PASS for checklist B.

Recomputed aggregate metrics:

| endpoint | embedding | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|---:|
| frozen_zero_spectral_adapter | baseline_only | 77.487603116011 | 79.334500875657 | 89.492119089317 | 93.520140105079 |
| frozen_zero_spectral_adapter | fused | 80.507515546712 | 84.413309982487 | 90.192644483363 | 93.169877408056 |
| frozen_zero_spectral_adapter | cnn | 79.471874953647 | 83.537653239930 | 88.966725043783 | 91.593695271454 |
| frozen_zero_spectral_adapter | transformer | 78.529832158357 | 80.910683012259 | 88.966725043783 | 92.119089316988 |
| frozen_zero_spectral_adapter | mamba | 77.796840675490 | 80.035026269702 | 89.316987740806 | 94.045534150613 |
| trained_spectral_adapter | baseline_only | 77.487603116011 | 79.334500875657 | 89.492119089317 | 93.520140105079 |
| trained_spectral_adapter | fused | 80.254811045804 | 83.537653239930 | 91.418563922942 | 94.921190893170 |
| trained_spectral_adapter | cnn | 80.540530605339 | 84.413309982487 | 89.842381786340 | 92.994746059545 |
| trained_spectral_adapter | transformer | 77.927403684145 | 81.786339754816 | 89.842381786340 | 92.819614711033 |
| trained_spectral_adapter | mamba | 77.529976218872 | 78.633975481611 | 87.915936952715 | 93.520140105079 |

Recomputed gains and bootstrap gate input:

| quantity | value |
|---|---:|
| matched baseline_only mAP gain pp | 0.000000000000000 |
| matched fused mAP gain pp | -0.252704500907456 |
| matched cnn mAP gain pp | 1.068655651691316 |
| matched transformer mAP gain pp | -0.602428474211266 |
| matched mamba mAP gain pp | -0.266864456617540 |
| fold 0 fused mAP gain pp | -0.000121211614271 |
| fold 1 fused mAP gain pp | -0.999667315709658 |
| fold 2 fused mAP gain pp | 0.171629750993020 |
| recorded identity-cluster bootstrap 95pct lower pp | -1.509454322847345 |

Scientific gates recomputed from the arrays and recorded bootstrap result:

| gate | pass |
|---|---:|
| aggregate fused gain >= 1 pp | false |
| all fold fused gains nonnegative | false |
| all expert aggregate gains nonnegative | false |
| fused bootstrap lower positive, from recorded NumPy result | false |
| fused beats baseline and every expert | false |

## C. Provenance and receipts - WARN

The preregistration binds the V23 runner, config, plan, source files, seed, and V12 source summary before execution, with Q1 not yet executed and dev/official counts zero (`evidence/trifusion_v23_preregistration_20260905.json:1-23`). The launch file records execution commit 9f4a10b, PID, screen, argv, verified files, V12/CLIP hashes, planned optimizer counts, a single original process, and dev/official zero (`evidence/trifusion_v23_launch_20260905.json:1-118`). Terminal file verification records summary/log hashes, execution source hashes, V12/V23 checkpoint hashes, receipt equality to summary, signal commit/diff, preregistration matching, mutable tracker exception, and explicitly states no tensor loading (`evidence/trifusion_v23_q1_terminal_file_verification_20260905.json:1-45`, `evidence/trifusion_v23_q1_terminal_file_verification_20260905.json:229-341`). Intake independently records raw SFTP byte/hash evidence for the result files and zero new optimizer/retrieval/checkpoint tensor loads during intake (`evidence/trifusion_v23_q1_intake_20260905.json:6-81`).

Receipt files read locally and compared byte-parsed objects against the terminal summary:

| receipt | equals summary | strict reload | read-only eval | SHA256 |
|---|---:|---:|---:|---|
| evidence/trifusion_v23_q1_receipts/fold_0_frozen_zero_spectral_adapter_receipt.json | true | true | true | 7fe91e7173613e42421832648d5f31c3eeeb4d78f3417f1bd14bc4c32179ce8e |
| evidence/trifusion_v23_q1_receipts/fold_0_trained_spectral_adapter_receipt.json | true | true | true | ef54222e23ed1fc8489f8628589491c76082dedd0d2023d211d578bb4668b8dc |
| evidence/trifusion_v23_q1_receipts/fold_1_frozen_zero_spectral_adapter_receipt.json | true | true | true | ecb8fe210e6349e6349de7d3c6d45df31dd37e159d2d747e39b32d903b719f6c |
| evidence/trifusion_v23_q1_receipts/fold_1_trained_spectral_adapter_receipt.json | true | true | true | 761fb52579908d65333d95b75205bb8e7b171289109dfb1af585ee1078246bc5 |
| evidence/trifusion_v23_q1_receipts/fold_2_frozen_zero_spectral_adapter_receipt.json | true | true | true | 633ebb7ec30aed38f667af75b6e9ff8a151473e465db3cfe648ea0a310c00ad7 |
| evidence/trifusion_v23_q1_receipts/fold_2_trained_spectral_adapter_receipt.json | true | true | true | 4eeebad700dad1535c9da3ec6ee4d317b06838a28d367f62f553cdf67b94bd7c |

Raw local file bytes for four files differ from execution hashes because the local working tree uses CRLF line endings for those files. LF-normalized local content matches the execution hash; local git blob comparison against commit 9f4a10b showed no semantic source drift for those files. This is a provenance warning, not evidence of scientific invalidity.

| file | local raw SHA256 | LF-normalized SHA256 | execution SHA256 | LF matches execution |
|---|---|---|---|---:|
| modeling/trifusion/experts/mamba.py | 8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83 | c516c7f6e185a2d93152d9f18ac65e0548fd5320d71102e40f4d85af7216fc8e | c516c7f6e185a2d93152d9f18ac65e0548fd5320d71102e40f4d85af7216fc8e | true |
| modeling/trifusion/experts/semantic_residual.py | c1831a3de031be033624420791cecf2e7a3945e009655089cb48ee4b9912edd6 | c8cef88b44ba155aaecdfdd5d51ecf8372b3b5d40fbfcae9eb17a82141791b4b | c8cef88b44ba155aaecdfdd5d51ecf8372b3b5d40fbfcae9eb17a82141791b4b | true |
| modeling/trifusion/criterion.py | 9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f | 0b2a63705cd52197fb99b0e6d5c6b774423179df9ff459bc78b99e2fd5b72444 | 0b2a63705cd52197fb99b0e6d5c6b774423179df9ff459bc78b99e2fd5b72444 | true |
| protocols/rgbnt201_dev_v1.json | f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d | d91629c908baad72b5208d938065616459cf79a6a5e0f8aa2c3f6eb66dd85949 | d91629c908baad72b5208d938065616459cf79a6a5e0f8aa2c3f6eb66dd85949 | true |

Checkpoint provenance remains limited: the terminal ledger and receipts bind hashes, but this audit did not independently possess, hash, or load the checkpoint binaries due to the explicit no tensor/model/remote constraint.

## D. Execution and dead code - PASS

V23 adds a per-stage spectral adapter initialized to zero and applied after tail-stage expert blocks (`modeling/trifusion/signal_preserving_v23.py:11-62`). The runner asserts the contract, strict-loads V12 signal/expert checkpoints, freezes baseline/source modules, wraps the V23 encoder, and asserts 1,777,536 trainable spectral adapter parameters (`tools/train_signal_preserving_v23.py:35-98`). Preflight verifies paired frozen/trained initialization, positive pairs, zero-adapter legacy equality, and unchanged state (`tools/train_signal_preserving_v23.py:146-175`). The fit endpoint reseeds, uses fixed loaders, trains 20 epochs, checks optimizer-step counts, gradient liveness, finite losses, and no overflow (`tools/train_signal_preserving_v23.py:200-235`). The Q1 loop saves final checkpoints, strict-reloads, evaluates, writes receipts, and prints final metrics (`tools/train_signal_preserving_v23.py:331-370`).

The metric functions used by the V23 claims are connected in the active path: V23 imports the V18 evaluator, the evaluator calls full-gallery scoring, and full-gallery scoring calls the per-query AP/rank routine (`tools/train_signal_preserving_v18.py:133-152`, `tools/audit_v17_full_gallery.py:14-43`, `tools/diagnose_v6_oracle_complementarity.py:81-108`). I found no phantom metric function in the active V23 Q1 evidence path. Historical files in the primary list are dependencies or precedent evidence, not claimed active metric code for Q1.

The log verifier confirms 120 epoch rows, 3,360 optimizer steps, six final events, exact history equality, M0, terminal event, and no new optimizer/evals (`evidence/trifusion_v23_q1_log_verification_20260905.json:1-12`). The exit file is zero (`evidence/trifusion_v23_q1_exit_20260905.txt:1-2`). The log terminal event records Q1_FAIL and final gates (`evidence/trifusion_v23_q1_run_20260905.log:325-326`).

## E. Scope and qualification - WARN

The plan preregistered five Q1 scientific gates and explicitly requires D1/dev/official only after all pass; failure means the result is sealed and no scans proceed (`refine-logs/v23/EXPERIMENT_PLAN.md:97-109`). The tracker reports Q1 complete/fail/sealed, all folds/endpoints complete, 3,360 Q1 optimizer steps, all scientific gates failed, and D1 not qualified (`refine-logs/v23/EXPERIMENT_TRACKER.md:3-38`). The result report records Q1_FAIL, fused mAP -0.252705 pp, fold gains -0.000121/-0.999667/+0.171630, bootstrap lower -1.509454, OOF-not-dev/official scope, and source/evidence scope warnings (`results/TRIFUSION_RGBNT201_V23_COMPLETE_Q1_2026-09-05.md:3-10`, `results/TRIFUSION_RGBNT201_V23_COMPLETE_Q1_2026-09-05.md:99-124`).

This is correctly qualified reporting. The warning is about scientific scope: single seed, train-internal reused OOF, and extra trainable adapter parameters cannot establish general scientific improvement or justify D1/dev/official progression. M0 is explicitly an engineering pass in the summary (`evidence/trifusion_v23_q1_seed42_9f4a10b.json:78810-78818`) and the terminal checks all fail scientifically (`evidence/trifusion_v23_q1_seed42_9f4a10b.json:81175-81280`).

## F. Claim impacts - WARN

The supported claim is that V23 Q1 ran to completion with engineering integrity and produced a negative train-internal result. The aggregate table shows trained fused mAP 80.254811 below frozen fused mAP 80.507516, while the trained CNN expert alone reaches 80.540531 (`results/TRIFUSION_RGBNT201_V23_COMPLETE_Q1_2026-09-05.md:17-21`). The identity and query paired-change tables show mixed behavior rather than stable improvement (`results/TRIFUSION_RGBNT201_V23_COMPLETE_Q1_2026-09-05.md:65-93`). The report generator is written to state the train-internal OOF scope, all metrics, and warnings (`tools/report_v23_complete_comparison.py:79-143`).

The following claims are not supported by this evidence: fused scientific improvement, mechanism validation, fixed-dev improvement, official-test improvement, model selection, threshold selection, or advancement to D1. The engineering M0 claim is supported only as an execution/preservation qualification.

## Input file hashes read in this audit

| file | bytes | SHA256 |
|---|---:|---|
| modeling/trifusion/signal_preserving_v23.py | 3908 | 89816298baedf6ab5d3f122fe16db69035420df8cc91b65e0bb0c8c6f9cde1af |
| tools/train_signal_preserving_v23.py | 24966 | 1b18edbd28e335469f6647a7095228e9e03cf8195b2739b4a9c54c62aedec42b |
| configs/RGBNT201/TriFusion-signal-preserving-v23-spectral-adapter-rtx3090.yml | 3397 | 9859342d044e8bb0b630bc8f6af345ee95dbf1eb445f7a2bffce4173df8c01fa |
| refine-logs/v23/EXPERIMENT_PLAN.md | 7282 | 7877e7fe6a857b965cac701220be387da4774c3e88dfa3682938df8dc32e4997 |
| refine-logs/v23/EXPERIMENT_TRACKER.md | 3190 | 957909b60d13a9e318a976c38e8f9aee65206da585593f15a07f4bf441cee681 |
| results/TRIFUSION_RGBNT201_V23_COMPLETE_Q1_2026-09-05.md | 9379 | c8b241a62dd2b1426b1a161c3f75a709db53d4d7e03a1221cc1a9a9f349107ff |
| evidence/trifusion_v23_preregistration_20260905.json | 1530 | 11c13c301db0aa27cc5d7c1b867cc86373dd62bfc238e210f176298c10ad2bc7 |
| evidence/trifusion_v23_launch_20260905.json | 5314 | b8db2fad72334988161cbb380be41b9f5846a97bb720240262eb4a4264bfc8aa |
| evidence/trifusion_v23_q1_seed42_9f4a10b.json | 2148943 | dbb58d0d614dc5e8007e8548508dec75e6a5225647450467e7b13a2c1111d9b0 |
| evidence/trifusion_v23_q1_run_20260905.log | 201215 | 60482e4297b484cf035a5e6417f2a3e49949d0fe609dba4b928971d39d7e9588 |
| evidence/trifusion_v23_q1_exit_20260905.txt | 2 | 9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa |
| evidence/trifusion_v23_q1_intake_20260905.json | 4151 | eb457353effb1f803c6e978fac9a661c12b68ef9b3eafe7325a2929d0734c218 |
| evidence/trifusion_v23_q1_terminal_file_verification_20260905.json | 17104 | 203505c88c2ac1cbf828300151fbdbccfd50e85f24e2ec900dda33e3d3121ca3 |
| evidence/trifusion_v23_q1_array_verification_20260905.json | 8902 | a39c30f0c4df1c72fa83329ebe811f78cd46ffb97e396d83e65781c8993756e5 |
| evidence/trifusion_v23_q1_log_verification_20260905.json | 523 | 1071cdc946d3f4f05abc5328e948ba45bb857badf50753f12d49caaf0f471df7 |
| evidence/trifusion_v23_q1_complete_comparison_20260905.json | 527772 | 6f9503b541a2b66a39451adc9dfaa9f241aa509c7384ebee9040cacd08acf27f |
| tools/verify_v23_terminal_files.py | 6098 | e8ae141d2fa926d215f3884b19f0475c8734ea95ac6bc911bee01b31ed8b4a92 |
| tools/audit_v23_terminal_arrays.py | 12876 | 860cab914a8367625a5ce0bef454a7bb5c353141794454f544bb924ea8821fb3 |
| tools/report_v23_complete_comparison.py | 11456 | 01ee60adbb484183b292b664b0f69e8448307e7b866067592d69d20cf48f0c68 |
| modeling/trifusion/signal_preserving_v8.py | 28592 | 97a7b5fe6dab882c2ed92ed1db95b9a8f48eae1d99a3a9b145b0cdfc4b8cefbc |
| modeling/trifusion/signal_preserving_v8_builder.py | 3900 | 8afb028957ecff1a0a26497f7d0460bc240b0f8612a3c476b52a9a6667a3049d |
| modeling/trifusion/state.py | 4990 | d6491db289e68e0b97a8d532edc933252ab60fdaad963af85f603dec9002e2b9 |
| modeling/trifusion/experts/mamba.py | 7479 | 8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83 |
| modeling/trifusion/experts/semantic_residual.py | 15882 | c1831a3de031be033624420791cecf2e7a3945e009655089cb48ee4b9912edd6 |
| modeling/trifusion/criterion.py | 6231 | 9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f |
| modeling/trifusion/aligned_data.py | 10062 | 3ea362d17660483b554cb599442b6377ace020fa114969b9bdd58906fbceedd5 |
| tools/train_signal_preserving_v19.py | 26363 | 5f000d3d3abe9636cae97e292db2c63a3a20f349c125dfe4b27bd3fca95bb9c5 |
| modeling/trifusion/signal_preserving_v19.py | 4059 | 69989cced25950ff476759bc93df945b167e8ce88a4340676e8adbbd9f88918e |
| tools/train_signal_preserving_v17.py | 61124 | 333886d16f73987accddb70b0780661bb9400b8afd04af01b57e188d34e5228d |
| tools/train_signal_preserving_v18.py | 17540 | f6bdda4631d710d0b6db5fe2a8df124d8dcbd294cb54adcf3ec261d523e96cf8 |
| tools/build_v12_complete_path_oof_targets.py | 32723 | fc13354b1065c46677122ee9cf63816087facca4e0e98ec5e6e2fb0dece141e4 |
| tools/run_signal_preserving_v5.py | 65190 | e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5 |
| tools/run_signal_baseline_dev.py | 12601 | 083967f2a38267415b2992da98c2ad9429ebb793f387b45daac1dbb4eb16f365 |
| tools/audit_v17_full_gallery.py | 10425 | 856881f984ab8793788d291018a60046a47e309c625a65394b1a3ff4e670d8a9 |
| tools/diagnose_v6_oracle_complementarity.py | 9754 | 919b624156c57f92fa75f79a06fe7c872a02d730fa8cd021ce9d7b4e498b1db2 |
| modeling/trifusion/signal_preserving_v13.py | 9347 | 7b7c4abb220ed234608553c77aeedd5f8ef763abb04cce5b21f1bfce0f4daa62 |
| protocols/rgbnt201_dev_v1.json | 5685 | f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d |
| evidence/trifusion_v12_complete_path_oof_seed42.json | 43930 | 9105b86a9c079c44843e9b118a21599746a87d279370341b8cf8d26bd18a8b69 |
| evidence/trifusion_v23_q1_receipts/fold_0_frozen_zero_spectral_adapter_receipt.json | 166376 | 7fe91e7173613e42421832648d5f31c3eeeb4d78f3417f1bd14bc4c32179ce8e |
| evidence/trifusion_v23_q1_receipts/fold_0_trained_spectral_adapter_receipt.json | 168389 | ef54222e23ed1fc8489f8628589491c76082dedd0d2023d211d578bb4668b8dc |
| evidence/trifusion_v23_q1_receipts/fold_1_frozen_zero_spectral_adapter_receipt.json | 162812 | ecb8fe210e6349e6349de7d3c6d45df31dd37e159d2d747e39b32d903b719f6c |
| evidence/trifusion_v23_q1_receipts/fold_1_trained_spectral_adapter_receipt.json | 165145 | 761fb52579908d65333d95b75205bb8e7b171289109dfb1af585ee1078246bc5 |
| evidence/trifusion_v23_q1_receipts/fold_2_frozen_zero_spectral_adapter_receipt.json | 174012 | 633ebb7ec30aed38f667af75b6e9ff8a151473e465db3cfe648ea0a310c00ad7 |
| evidence/trifusion_v23_q1_receipts/fold_2_trained_spectral_adapter_receipt.json | 175984 | 4eeebad700dad1535c9da3ec6ee4d317b06838a28d367f62f553cdf67b94bd7c |

## Limitations

- No model, tensor, image, checkpoint loading, GPU, remote, network, installation, or delegation action was performed.
- Exact NumPy 2.5.2 bootstrap resampling was not regenerated locally. The recorded verifier result is negative and internally consistent, but this audit cannot call that subcheck independently regenerated.
- Checkpoint binary hashes are accepted as terminal-ledger/receipt evidence; they were not independently rehashed from local checkpoint files in this audit.
- This is a same-family constrained audit, not a cross-family acquittal.
- The audit preserves the distinction among engineering M0, complete train-internal Q1, fixed dev, and official testing. No artifact selection, threshold change, or scientific advancement follows from M0.
