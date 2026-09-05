# Experiment Audit Report: V21 SAM M0

**Date**: 2026-09-05  
**Auditor**: GPT-5.5 xhigh independent local artifact audit  
**Project**: TriFusion RGBNT201 V21 SAM M0  
**Overall Verdict**: WARN  
**Integrity Status**: warn  
**Engineering Integrity**: pass  
**Fixed M0 Qualification**: fail  
**Scientific Qualification**: fail  
**Evaluation Type**: source-only engineering M0 over real train-source batches plus synthetic T0 toy tests; no heldout/dev/official/test retrieval was executed.

I read the listed primary files directly and followed the called dependencies. I did not use prior V20/V19 audit verdicts or executor summaries as evidence. I did not run model training, inference, evaluation, checkpoint tensor loads, image reads, feature or distance computation, tests, remote commands, downloads, or network calls. The only writes from this audit are this report and `EXPERIMENT_AUDIT_V21_M0.json`.

The fixed V21 M0 run failed its preregistered overfit gate. That is an experiment outcome, not an engineering-integrity failure. The engineering record is internally coherent: the run stopped at `M0_FAIL`, Q1 was not run, and the independent arithmetic replay exactly reproduces the M0 failure. Integrity is WARN because the current local checkout is not the recorded execution commit, four current local source files match execution hashes only after LF normalization, remote checkpoint/CLIP/V12 bytes were not locally possessed, and the summary's literal `evaluation_type` is broader than the executed M0 scope.

## A. Ground truth provenance and path isolation: PASS

Evidence:

- `protocols/rgbnt201_dev_v1.json:1-18` records protocol checks and triplet counts: 3126 train triplets, 825 dev triplets, 3951 train_171 triplets.
- `protocols/rgbnt201_dev_v1.json:126-132` records `selection.uses_test_labels=false` and `test_identity_overlap=[]`.
- `tools/run_signal_baseline_dev.py:27-54` parses raw RGBNT201 record identity from filename prefix, camera from the `cam` filename field, verifies RGB/NI/TI triplet files, and returns `(paths, label, camera_id, view)`.
- `tools/build_v12_complete_path_oof_targets.py:156-173` loads frozen protocol `train_ids`, reads `train_171`, relabels train records, and verifies the 3126 train-triplet count.
- `tools/build_v12_complete_path_oof_targets.py:29-56` builds each complete-path fold by excluding heldout identities from fit records and returning `identity_overlap`.
- `tools/train_signal_preserving_v21.py:201-206` executes contract loading, Signal config, record loading, fold construction from V12 `heldout_identity_ids`, and asserts `len(records)==3126` plus no split overlap.
- `tools/train_signal_preserving_v20.py:58-88` is the reused source/expert builder; it loads source checkpoint payloads, asserts `fit_identity_ids`/`heldout_identity_ids`, strictly loads Signal/expert state, freezes baseline, and records binding fields.
- `evidence/trifusion_v21_m0_seed42_3c39351.json:48-50`, `:51-216`, `:2657`, and `:5263` start the three recorded preflight folds. `evidence/trifusion_v21_m0_seed42_3c39351.json:7868` records `"folds": []`.

Independent replay from the local protocol JSON gives 141 train identities, 30 dev identities, 30 test identities, zero train/dev, train/test, and dev/test overlap, 3126 train triplets, and `selection.uses_test_labels=false`. The local artifact does not contain the remote image tree, so I did not independently read image bytes. I traced the label/camera provenance through the called parser and verified recorded M0 batch composition from the receipt filenames.

For the six preflight models, I recomputed every recorded pairing comparison from `evidence/trifusion_v21_m0_seed42_3c39351.json`: for folds 0, 1, and 2, `binding_equal=true`, `initial_state_sha256_equal=true`, `batch_receipts_equal=true`, `all_output_sha256_equal=true`, and all endpoint `state_unchanged` fields are true. Each fold has 94 fit identities, 47 heldout identities, and zero fit/heldout overlap; heldout identities across the three folds total 141 unique fold-internal identity indices. From all 48 recorded batch receipts, filename replay found 64 samples per batch, 8 identities per batch, 8 samples per identity, and exactly one cross-camera identity group in every batch.

This supports source-only M0 path isolation and pairing. It does not support any V21 heldout retrieval claim because the run stopped before Q1 and the literal `folds` array is empty.

## B. Losses, scores, and mathematical recomputation: PASS

Evidence:

- `configs/RGBNT201/TriFusion-signal-preserving-v21-sam-rtx3090.yml:31-47` fixes LR, weight decay, endpoint epochs/warmups, `SAM_RHO=0.05`, AMP scale, and the seven ID/triplet coefficients.
- `tools/train_signal_preserving_v21.py:59-69` computes the executed V21 loss as fused ID/triplet plus three branch ID/triplet plus three residual ID/triplet terms.
- `modeling/trifusion/signal_preserving_v8.py:690-742` defines smoothed identity cross-entropy and batch-hard triplet over normalized embeddings for fused, branch, and residual outputs.
- `modeling/trifusion/sam_training_v21.py:24-75` implements the one- or two-pass SAM/AdamW update, AMP scale handling, perturbation, parameter/BN restoration, unscale, optimizer step, and returned scalar fields.
- `tools/train_signal_preserving_v21.py:253-281` executes the two 8-step capacity runs, the 100-step fixed-batch SAM overfit run, entropy floor calculation, and M0 gate booleans.
- `evidence/trifusion_v21_m0_seed42_3c39351.json:7870-7878` records `m0.passed=false` and the five M0 check booleans.
- `evidence/trifusion_v21_m0_seed42_3c39351.json:7975-7985` records the rho 0.0 capacity totals and 6126 MiB peak; `:8082-8092` records the rho 0.05 capacity totals and 6284 MiB peak.
- `evidence/trifusion_v21_m0_seed42_3c39351.json:9202-9215` records the 100-step overfit totals, entropy floor, and excess-loss ratio.
- `evidence/trifusion_v21_m0_array_verification_20260905.json:14-23` records the verifier's entropy/ratio/gate summary; `:24-64` records the overfit component ranges.

The executed objective is the original V8 seven-ID/seven-triplet loss. There is no executed V20 cross-modal identity loss in M0: V21 imports `build_model` and `new_optimizer` from V20, but `tools/train_signal_preserving_v21.py:59-69` calls only the V8 criterion terms. Feature normalization occurs inside the triplet loss at `modeling/trifusion/signal_preserving_v8.py:707-710`; that is not metric normalization. No retrieval AP/Rank metrics exist for V21 M0, and no metric is divided by a model-output max/min. The only normalized scalar decision is the fixed M0 overfit ratio against the analytic entropy lower bound and first/last recorded losses.

SAM mechanics check out from source. `modeling/trifusion/sam_training_v21.py:26-31` zeroes gradients, records the GradScaler scale, evaluates the first loss, and backpropagates. `:31-42` computes the first gradient norm from scaled gradients and perturbs by `rho / scaled_norm`, so the AMP scale cancels. `:47-51` zeroes gradients, evaluates the perturbed loss, and backpropagates the update gradient. `:52-58` restores parameters and first-pass BatchNorm buffers. `:60-63` unscales, records the update gradient norm, steps the optimizer, and updates the scaler. The wrapper at `tools/train_signal_preserving_v21.py:72-85` records first/update gradient coverage and asserts seven BN counters increase by exactly one per step.

Independent arithmetic used Python 3.12.14 and NumPy 2.3.5; elapsed local arithmetic time was 0.011746s. From 94 classes and label smoothing 0.1, I computed `H=0.7711772280616133`; with identity coefficient sum `0.75`, the identity entropy floor and combined loss floor are `0.57838292104621`. The fixed gate ratio is:

`(0.5914160013198853 - 0.57838292104621) / (0.6110473871231079 - 0.57838292104621) = 0.39899872365870204`.

That exceeds the fixed threshold `0.1`, so `overfit_excess_ratio_at_most_point1=false`. The independent M0 gate replay exactly matches the literal summary checks. Maximum numeric difference versus the M0 summary and array-verification sidecar was `0.0`.

Overfit 100-step component ranges, independently recomputed from the raw summary arrays:

| Field | First | Last | Minimum | Maximum | Last-20 mean |
|---|---:|---:|---:|---:|---:|
| loss_at_parameters | 0.6110473871231079 | 0.5914160013198853 | 0.5813781023025513 | 0.7327114939689636 | 0.5843785017728805 |
| loss_for_update_gradient | 0.627093493938446 | 0.6106454133987427 | 0.5943088531494141 | 0.9014252424240112 | 0.6061754256486893 |
| first_gradient_norm | 0.15330751240253448 | 0.3367374539375305 | 0.08146421611309052 | 2.6079766750335693 | 0.22608692720532417 |
| update_gradient_norm | 0.6312003135681152 | 0.4366075098514557 | 0.3225419223308563 | 3.9590492248535156 | 0.7561642244458199 |
| actual_perturbation_norm | 0.05000000074505806 | 0.05000000447034836 | 0.04999999329447746 | 0.050000011920928955 | 0.05000000186264515 |

Capacity ranges were also replayed: rho 0.0 has 8 optimizer steps, 8 forward/backward pairs, perturbation norm exactly 0, loss range `0.6110473871231079-0.8435304164886475`, and gradient norm range `0.15330751240253448-3.2709567546844482`. Rho 0.05 has 8 optimizer steps, 16 forward/backward pairs, perturbation norm range `0.05000000074505806-0.05000000819563866`, loss-at-parameters range `0.6110473871231079-0.8659583926200867`, update-loss range `0.627093493938446-1.0011695623397827`, first-gradient norm range `0.15330751240253448-3.030792713165283`, and update-gradient norm range `0.6312003135681152-4.20034122467041`.

## C. File existence and provenance: WARN

Evidence:

- `evidence/trifusion_v21_preregistration_20260905.json:3-33` records the five preregistered source/config/plan/test files and `status="PREREGISTERED_NOT_LAUNCHED"`.
- `evidence/trifusion_v21_t0_20260905.json:3-40` records execution commit, source bindings, `pytest` command, pass output, and zero dataset/project-training access.
- `evidence/trifusion_v21_m0_seed42_3c39351.json:3-30` records `status="M0_FAIL"`, repository commit, runner/config/plan hashes, source file hash map, Signal commit, and Signal diff hash.
- `evidence/trifusion_v21_m0_terminal_file_verification_20260905.json:2-8` records verifier status, execution/observed commits, summary hash, and log hash.
- `evidence/trifusion_v21_m0_terminal_file_verification_20260905.json:17-43` records remote raw-byte matches for `criterion.py`, `experts/mamba.py`, and `experts/semantic_residual.py`; `:80-85` records the remote protocol JSON match.
- `evidence/trifusion_v21_m0_terminal_file_verification_20260905.json:164-224` records remote CLIP, V12 summary/checkpoint hashes, `new_checkpoint_count=0`, and the verifier's no-model/no-tensor-load scope.
- `results/TRIFUSION_RGBNT201_V21_SAM_M0_2026-09-05.md:30-36` reports the summary/log hashes and local math replay summary; `:40-53` indexes the evidence files.

All listed primary files exist locally and were read directly. Current local raw hashes for the main V21 artifacts match preregistration/T0/M0 fields: `tools/train_signal_preserving_v21.py` is `deafa4d6d2287928c9143d28f2bfb7f32e303939fa6db547f91219f3d708e0fa`, `modeling/trifusion/sam_training_v21.py` is `9233e8f806914eed623976c7f30830ecac80af274f842214c73055ded9009b56`, tests are `75f05728853af1fb82af23588362ef56bd573c1322bf2b423be7c6d1aad48add`, config is `f2f47acf54790dc69d9b0d7b5c94dcf9ecfd37bfcf3dd20dc8251e1e1a3600a3`, and plan is `ba17807f30e294618d2a21907a8fde0da82f24f6dab0d51073be49014cc40f71`. The summary raw SHA is `2ecc322270e4e1b82a77cf76e22ab76e359179fda9abc5ef7f2036db064d3c5d`; the log raw SHA is `0be3a21d007f1ff125779c13231c672ffaab67e884f02478b4be68e620f85194`.

The recorded execution commit is `3c393510f0e0a31bad602af8dd618a8dcdfe6ae6`; the current local checkout HEAD observed by read-only `git rev-parse HEAD` is `4f31651a6962372fb1d8c62b87e05a00abe953b0`. I therefore do not treat HEAD identity as current proof. I use file-level hashes.

Against the summary's literal `source_file_sha256` map at `evidence/trifusion_v21_m0_seed42_3c39351.json:8-27`, 15 dependencies match current local raw bytes. Four current local files match execution hashes only after CRLF-to-LF normalization:

| File | Expected execution SHA | Current local raw SHA | Match type |
|---|---|---|---|
| modeling/trifusion/criterion.py | 0b2a6370f434828d885945d1a46dd56f6668133ceed45ef359fe71eeca63740a | 9a028e5711981123310f5e0d441a35b7dbecaf33f9edcec4fd3773d070cf877f | LF-normalized |
| modeling/trifusion/experts/mamba.py | c516c7ad937e5eee6a4ed1e3ec33c2afe3522b751d296bd2e4910e4f27a20ee5 | 8b9cb420c42e4d70f8de7e4608637c81c505b97ca54228175462a7e92fdfcc83 | LF-normalized |
| modeling/trifusion/experts/semantic_residual.py | c8cef9717fd7bd1e5e50b428ac92762455defac2e25857c9e0dfaf82729c2a93 | c1831a3de031be033624420791cecf2e7a3945e009655089cb48ee4b9912edd6 | LF-normalized |
| protocols/rgbnt201_dev_v1.json | d916e7daaa1d55b179c1ec77e93128b6e6a8d1526adc9eac060ea8e733881946 | f7f6f8a906c06fec4918929fb5c17abfc197986b99e017aca1d3e15831c1541d | LF-normalized |

The terminal file-verification JSON records remote raw-byte matches for those files, plus remote CLIP and V12 checkpoint/source matches. Within this local no-network/no-checkpoint-load audit, those remote bytes are not independently possessed. This is the main provenance reason for WARN. It does not change the M0 arithmetic or the conclusion that Q1 did not run.

## D. Execution and engineering: PASS

Evidence:

- `tools/train_signal_preserving_v21.py:72-85` wraps one training step, selects trainable parameters, finds seven training `BatchNorm1d` modules, records BN counters, calls `training_step`, checks perturbation norm, and returns first/update gradient sets.
- `tools/train_signal_preserving_v21.py:113-138` implements `fixed_steps`, including seed reset, train mode, frozen-state snapshot, optimizer/scaler creation, optional fixed batch reuse, count rows, gradient coverage, overflow, BN, and restore totals.
- `tools/train_signal_preserving_v21.py:241-287` executes all preflights, both capacity runs, the fixed-batch overfit run, writes `m0`, records `M0_FAIL`, and returns before Q1 when M0 fails.
- `tools/train_signal_preserving_v20.py:91-96` constructs the actual AdamW/GradScaler optimizer stack used by V21.
- `modeling/trifusion/sam_training_v21.py:24-75` implements the optimizer update lifecycle.
- `tests/test_trifusion_sam_v21.py:10-23`, `:26-46`, and `:49-69` define the T0 toy tests for quadratic SAM, rho=0 AdamW equivalence/scale cancellation, and BN/frozen-parameter behavior.
- `evidence/trifusion_v21_t0_20260905.json:33-40` records `3 passed`, `dataset_access_count=0`, and `project_training_steps=0`.
- `evidence/trifusion_v21_launch_observation_20260905.json:3-30` records the live main process PID 32331 and launch command while status was RUNNING.
- `evidence/trifusion_v21_launch_transport_20260905.json:3-11` records the launch-manager timeout as `training_execution_failure=false` and zero new launches after timeout.
- `evidence/trifusion_v21_progress_20260905_1649.json:1` records terminal `M0_FAIL`, no live process, zero completed endpoints, zero epoch rows, zero Q1 optimizer steps, GPU idle, and the failed overfit ratio.

The optimizer target is every `requires_grad` parameter. The recorded capacity and overfit summaries show `trainable_tensors=203`, `first_nonzero_gradient_tensors=203`, `update_nonzero_gradient_tensors=203`, empty missing-gradient lists, `overflow_events=0`, `frozen_state_unchanged=true`, and `batchnorm_updates_per_step=1`. The capacity memory peaks are 6126 and 6284 MiB, below the fixed 24576 MiB limit.

Derived execution counts:

| Category | Count | Basis |
|---|---:|---|
| T0 toy optimizer updates | 6 | derived from test source calls: one quadratic SAM, one rho=0 training_step, one manual AdamW step, two SAM scale-cancellation updates, one BN SAM update |
| T0 project training steps | 0 | literal T0 JSON |
| M0 preflight forward-only batches | 48 | 3 folds × 2 endpoints × 8 receipt batches |
| M0 capacity optimizer steps | 16 | rho 0.0 8 + rho 0.05 8 |
| M0 capacity forward/backward pairs | 24 | 8 + 16 |
| M0 overfit optimizer steps | 100 | literal overfit field |
| M0 overfit forward/backward pairs | 200 | literal overfit field |
| M0 project optimizer steps total | 116 | derived 16 + 100 |
| M0 forward/backward pairs total | 224 | derived 24 + 200 |
| SAM restore batches total | 108 | derived capacity 8 + overfit 100 |
| Q1 optimizer steps/endpoints/epoch rows | 0 / 0 / 0 | `folds=[]` and array/progress records |
| New checkpoints | 0 | terminal file-verification field |

The launch transport record needs careful interpretation: `launch_transport_exit_code=1` is a manager timeout while waiting for `screen -DmS`, but the artifact also records the original live process and later terminal `M0_FAIL`. This supports a manager/transport issue, not a training execution failure.

Parameter/BN restoration exactness is partly a source/runtime assertion, not a full literal dump of tensor/buffer values. The literal fields record restore counts, BN updates per step, gradient coverage, overflow, and frozen-state status.

## E. Scope and selection: PASS

Evidence:

- `refine-logs/v21/EXPERIMENT_PLAN.md:99-125` defines T0/M0 scope and states M0 failure stops Q1.
- `refine-logs/v21/EXPERIMENT_PLAN.md:127-147` defines Q1 gates and terminal evidence requirements, contingent on M0 passing.
- `tools/train_signal_preserving_v21.py:253-287` implements M0 and returns after `M0_FAIL`; `:288-365` is Q1 code that was not executed for this artifact.
- `evidence/trifusion_v21_m0_seed42_3c39351.json:34-47` records planned endpoint fields and zero dev/official/D1 access; `:7868-7871` records empty `folds` and `m0.passed=false`.
- `refine-logs/v21/EXPERIMENT_TRACKER.md:15-23` records 342.520254 seconds, M0_FAIL, zero Q1 fold/endpoint/epoch, zero checkpoints, and 116/224 M0 budget counts.
- `refine-logs/v21/EXPERIMENT_TRACKER.md:30-40` records the failed fixed overfit gate and forbids replacing it with min/mean/rerun/relaxed thresholds.
- `results/TRIFUSION_RGBNT201_V21_SAM_M0_2026-09-05.md:18-28` reports the fixed gate failure and no rescue by intermediate/min/mean values; `:55-60` states no heldout/dev/official result.

All attempted M0 outcomes are retained: both 8-step capacity component lists and the complete 100-step fixed-batch component/loss trajectory are in the summary. The fixed selection is the 100th update-before loss ratio against the first update-before loss and analytic floor. The intermediate minimum loss `0.5813781023025513` and last-20 mean `0.5843785017728805` are descriptive; they are not selection criteria and cannot pass M0.

No main-arm comparison was attempted. The ordinary40/SAM20 equal-forward-backward plan, six final checkpoints, heldout masks, AP/rank metrics, bootstrap, all-identity query changes, D1 refit/dev, official/test, and SOTA claims remain unexecuted V21 scope. No claim should inherit scope or success from V20/V19 or from the V21 plan.

## F. Evaluation classification and claims: PASS

Evidence:

- `evidence/trifusion_v21_m0_seed42_3c39351.json:32-47` records the summary literal `evaluation_type`, OOF reuse flag, endpoint plan, no dev/official/D1 access, and `next_phase_qualified=false`.
- `evidence/trifusion_v21_m0_seed42_3c39351.json:7868-7871` records `folds=[]` and `m0.passed=false`.
- `evidence/trifusion_v21_m0_array_verification_20260905.json:5-13` records `M0_FAIL`, 116 M0 optimizer steps, 224 M0 forward/backward pairs, and zero Q1 endpoint/checkpoint counts.
- `evidence/trifusion_v21_t0_20260905.json:33-40` records T0 as synthetic tests with no dataset or project training.
- `results/TRIFUSION_RGBNT201_V21_SAM_M0_2026-09-05.md:26-28` states this does not prove SAM heldout retrieval harm and forbids scanning/rerun rescue; `:55-60` states no heldout/dev/official evidence and no deployable status change.
- `refine-logs/v21/EXPERIMENT_TRACKER.md:38-40` states no SAM heldout result and no proof that SAM retrieval generalization is harmful.

Actual executed evidence is source-only engineering M0. It uses real train-source RGBNT201 records for preflight/capacity/overfit and synthetic CUDA toy tests for T0. It is not heldout retrieval, development-set evaluation, official/test evaluation, or SOTA evidence. The summary's literal `evaluation_type="real_gt_train_internal_complete_path_oof"` is a runner/Q1 contract label; it overstates the actual executed M0 evidence type unless read with `folds=[]`, `M0_FAIL`, and zero Q1 counts.

Claim impacts:

- T0 mathematical optimizer/AMP/BN tests: supported, limited to synthetic toy tests.
- M0 paired preflight/capacity/overfit execution: supported.
- Fixed M0 qualification: false; failed because `0.39899872365870204 > 0.1`.
- SAM heldout mAP/Rank improvement or harm: unsupported; Q1 did not run.
- Generalization, causal sharpness/minima mechanism, deployment, dev65, official 85.3/87.9, or SOTA claims: unsupported by V21 M0.

## Evidence exclusions and limitations

- I did not run training, inference, evaluation, tests, tensor/checkpoint loads, image reads, feature/distance computation, remote commands, network, or downloads.
- The remote image dataset, CLIP weight, V12 run summary, and V12 source/expert checkpoints are not locally possessed/read in this audit. Their hashes are ledger evidence from `evidence/trifusion_v21_m0_terminal_file_verification_20260905.json`, not independent local byte reads.
- The current local checkout HEAD differs from the recorded execution commit. Current local source equivalence is per-file, and four dependencies match only after LF normalization.
- Parameter restoration and BatchNorm restoration are supported by source assertions and summary counters, not by full tensor/buffer dumps in the artifact.
- The literal summary `evaluation_type` should not be used alone for downstream classification; actual executed V21 M0 evidence is source-only engineering with no heldout retrieval.
