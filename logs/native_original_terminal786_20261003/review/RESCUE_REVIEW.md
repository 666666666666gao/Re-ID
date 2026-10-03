# Native original/original terminal rescue review

Reviewed 2026-10-03T19:24:48.1254362+08:00. **FAIL_STOP_GPU_PARITY_NO_LOCALIZED_REPAIR**.

The registered native original/original control failed its unchanged gradient gate on **83/295** tensors. The stop condition applies: seal the failure and stop GPU parity trials. I found **no concrete correctness defect that invalidates this interpretation and no localized source repair**. This is an engineering failure, not a retrieval-effectiveness result. The three-dataset scientific goal remains **ACTIVE_UNMET**.

This fresh native review follows experiment-bridge RESCUE_ON_FAILURE and the installed Codex policy. Parent-confirmed spawn arguments: `gpt-6-astra / max / fork_turns=none`. Actual serving backend was not independently verified. `review_independence=same-family`; `acceptance_status=provisional`.

## Evidence and actual terminal boundary

Primary root: [native_original_control785_terminal_intake](../native_original_control785_terminal_intake/INTAKE.json). I verified **5/5** primary byte counts and existing SHA256 values, totaling **310,773 bytes**; **317/317** captured-source hashes match the existing intake. Both directories have exactly the declared file counts, and LAUNCH/INTAKE source maps agree. Source bytes total **17,858,983**, locally measured; the source intake itself records hashes rather than expected byte sizes. These checks reuse the existing contract.

LAUNCH records 2026 port, physical visibility 0/1, controller 3475611 at 19:04:34.559385. JOB records one child, 3475612, started 19:04:34.612193 and exited **1** at **19:05:21.571946**. MEASURED was written at 19:05:20.263986. The traceback is `AssertionError: ['gradients']` at check_native_original_repeat.py:181. The 19:10:20.680173 intake records both processes absent and no weights. Collector exit0 is intake completion only. The controller exit is **not independently measured**; its source returns the child code. PASS.json is absent from the exact five-file primary set. The parent's earlier 19:09:07 observation was supplied as context, not independently opened in this review.

I also checked the two historical V5 primary files used for state/input comparison against their existing intake: both bytes and hashes match. The new control's initial state is `7f5ff300a45faed287a510dacc68d06c006931504a4510fffd8c47e43d93b852`. All recorded labels, cameras, paths and RGB/NI/TI image digests match the V5 native measurement: the first32 samples of one author B64/K8 batch.

## Independent recount

These are recounts of the **saved per-key fixed-gate results**, not reruns of tensor allclose. The full tensors are not serialized. The source uses atol=rtol=1e-4 for gradients and 1e-5 for forward/head outputs.

| Group | Tensors | Failed | Maximum absolute difference |
|---|---:|---:|---:|
| Signal | 155 | 76 | 0.005859375 |
| Shared adapters | 54 | 7 | 0.000244140625 |
| Roles | 70 | 0 | 4.76837158203125e-7 |
| Native detail | 14 | 0 | 1.7951242625713348e-7 |
| Readout | 2 | 0 | 0 |

All295 keys are unique, both gradients are present and finite, corresponding shapes/dtypes agree, and all recorded gradients are FP32. Failures span five non-block Signal tensors, visual blocks0–10 and adapter stages0–1; this does not locate the initiating operation. The accompanying JSON contains all83 failed rows and their distribution.

All **13 other gates passed**:

- Raw/fused/global/loss and the author score/feature comparisons have recorded maximum difference **0**.
- Gradient keys and buffer keys match; all **14** buffer comparisons are exact.
- CPU RNG and both visible CUDA RNG states compare equal.
- Author bottleneck batch count is **1** in each model.
- Full post-state digests match: `c58bf01d510f5a1e9161878de765a27398bd5b2032229561447f42446ca252e8`.
- All **306 parameters per model** remain equal to their initial values.
- Original capture hooks are removed; optimizer ownership covers exactly295 unique gradient keys, group comparison passes, and scale stays256.

Post-state equality is between the two models; it is not an unchanged-state claim versus initialization because BN advanced once. Some exact comparisons persist only their booleans, not the two original RNG/hook/group objects; source inspection establishes what was compared. Parent ANALYSIS.json matches my independent group/count/failed-row, gate, output, head and BN recount.

## Control flow and source assessment

The captured original builder is called twice directly. Its factory constructs fresh Signal instances; it does not cache a model or call the project's partition/checkpoint/CPU-save helper. State/config/bindings, FP32 storage, cuda:0 placement, trainability and historical inputs are checked before forward. Each model receives the same input objects and restored CPU/two-CUDA RNG before identical author optimizer/loss construction. The unused center-loss constructor consumes CPU RNG on both paths after the same restoration; no mismatch was found there.

Each model performs exactly one AMP FP16 forward, scaler256 backward and unscale. Author train mode is restored, original BN/CE/Triplet heads receive raw features and true labels, and all model parameters belong to the intended optimizer groups. Capture hooks use the original finally cleanup. There is no optimizer.step, scheduler.step, scaler.update, official scoring, save of weights, retry or training continuation on the executed path. The 306 unchanged-parameter comparisons support the no-update claim.

check_native_original_repeat.py:167–180 writes every already-required comparison before the final assertion at181; PASS is written only after it. The controller has one Popen and wait, then records the child result. The observed terminal boundary agrees with this source.

The actual role path is `IndependentNativeRoles → GlobalTokenRoles → FP32SlotCompetitionRoles.sample_context`, which performs full-patch Q/K/V attention under its explicit FP32 region. It does **not** call the older inherited `_sample/grid_sample` implementation. The native stem feeds another explicit FP32 attention and a single zero-initialized output projection. Only that detail output has a nonzero gradient on this first backward, consistent with its initialization. This neither satisfies eight-update/all14-tensor M0 support nor explains the failed repeat-gradient comparisons.

After the parent supplied installed-source evidence, I verified **5/5** supplemental file bytes/hashes in [native_installed_sources786_actual_python](../native_installed_sources786_actual_python/INTAKE.json), then read the Python dependencies. The factory imports real Mamba, exported as mamba_simple.Mamba in installed2.2.6.post3. Its default use_fast_path=True is conditional on resolved causal_conv1d availability; that value was not recorded by this control. Both visible branches reach selective_scan_cuda. The fast wrapper has autocast-dependent casts, compiled forward/backward calls and its own internal recomputation. This dependency behavior is not the project's rejected checkpoint wrapper.

CLIP's source upcasts LayerNorm and restores the incoming dtype, requests need_weights=False in attention, and the captured PyTorch functional source exposes the corresponding scaled-dot-product native boundary. These source facts do not identify the actual selected CUDA attention backend. The supplemental collection is after runtime, contains Python/metadata only and does not inspect compiled kernels or intermediate dtype/dispatch traces. I do not infer an atomics, layout, hardware, jitter, zero-branch or kernel cause, or guarantee determinism. Inherited seed/cudnn flags also do not supply that missing evidence.

## Blocking issues and permitted continuation

**B1:** Original-native repeat83/295 fails the fixed gradient prerequisite. The explicit FAIL clause in the captured control plan:21 requires sealing and stopping GPU parity trials. No original/V5 retry, additional parity arm, smaller batch, different precision/seed, changed threshold/backend/determinism setting or automatic M0/formal continuation is supported.

**B2:** None of the **nine full-author-batch M0s** is accepted, RGBNT100 B128 capacity is unproved, and this path supplies no accepted formal three-dataset result. Retain the original full50, author batch/recipe and scientific advancement requirements. The goal is strong same-protocol results on all three datasets, not passing a numeric check.

Concrete invalidating source defects: **none found**. Required patches: **none**. Nonblocking source defects: **none found**. Missing historical tensors and dependency runtime observations are evidence limits, not grounds for speculative patches.

V5's90/295 and this control's83/295 are separate executions; their difference is not a seven-tensor estimate of a partition effect. Existing evidence **cannot single out partition as the cause**, does not exonerate partition and does not repair V5. The earlier §776 original repeat exercised semantic only. **Old full gradients and pre-forward RNG were not saved**, so matching recorded metadata does not reconstruct historical gradients or historical RNG.

The only source-only continuation suggested is to reconcile the existing scientific plan/tracker with the failed prerequisite and the dependency boundaries now established, explicitly retaining the unresolved numerical mechanism, stop condition and complete three-dataset objective. No additional runtime probe or code change is proposed.

This reviewer performed local text/JSON/source reads and checks, then wrote only the designated reviewer directory. No SSH, target program, Torch/model import, CUDA, scorer, training, production edit, parameter update, weight save or subagent execution occurred.

