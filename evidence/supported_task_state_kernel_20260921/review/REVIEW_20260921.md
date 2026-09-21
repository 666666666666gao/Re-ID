# Supported task-state AdamW V1: bounded kernel review

2026-09-21. Verdict: **PASS_KERNEL_ONLY**. Blocking findings: **0**. Required kernel patches: **0**. Review independence: **same-family**. Acceptance status: **provisional**. The reviewer backend identity is **unattested**.

This conclusion covers the new 33-line experiment plan, the 64-line optimizer kernel, and the latest 119-line synthetic checker in `C:/Users/gb/.trifusion_github_publish_22c3bee`. It does not establish trainer integration, real-source M0 completion, CUDA or real-model AMP execution, checkpoint disk restoration, scientific qualification, or permission to launch training. The reviewer changed only the review artifacts in this directory and launched no remote workloads.

Reviewed inputs:

- `refine-logs/msvr310_supported_task_state_v1/EXPERIMENT_PLAN.md`
- `tools/msvr_task_state_optimizer.py`
- `tools/check_msvr_task_state_math.py`
- Latest executor-supplied CPU receipt: `D:/Program Files/UserCache/gb/codex/tmp/task_state_cpu_check_amp_20260921.json`; this supersedes the earlier non-GradScaler receipt for the review evidence.

**Math and state behavior are correct within the declared interface.** The kernel computes the Adam direction as `D = (m / (1 - beta1**n)) / (sqrt(v / (1 - beta2**n)) + eps)`. The first and second moment updates, bias correction, and epsilon placement agree algebraically with ordinary AdamW. Shared roles use the complete supplied `R + A` gradient and one moment pair. Split roles maintain separate rank and auxiliary moments and sum their directions. All parameter updates use the current group learning rate. The implementation applies `p *= 1 - lr * weight_decay` once and then subtracts the chosen direction once; it does not apply decay per task. See kernel lines 22-32 and 48-64 and the [PyTorch 2.5.1 AdamW source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py).

**AdaTask attribution is appropriately bounded.** Separate first and second moments and summed task directions match the central construction in Algorithm 2 of the [AdaTask paper](https://ojs.aaai.org/index.php/AAAI/article/download/26275/26047). The [author implementation](https://github.com/EnnengYang/AdaTask/blob/master/adatask.py) also separates task moments for shared parameters and applies ordinary total-gradient Adam to task-specific parameters. The new kernel's observed-task clocks and once-per-update AdamW decay are its registered adaptations. The plan uses “AdaTask-inspired” and explicitly avoids a novelty claim or an exact author-implementation reproduction claim.

**Absent support and observed zero are correctly distinct.** When `rank_observed=False`, the rank buffer must be exactly zero; split mode never calls the rank-direction function, so rank moments and count remain unchanged and no previous rank momentum enters the parameter update. Auxiliary state advances normally. When support is present with a zero rank gradient, the rank function is called: moments decay, the count advances, and residual momentum may contribute. Shared mode continues to update on the supplied total, retaining common history. This implements plan lines 13-15. The kernel takes the support flag from its caller; source eligibility and warmup-to-AP transition are trainer responsibilities still awaiting verification.

**Heads have the intended ordinary AdamW path.** Every parameter outside the supplied role list uses its own `head` state driven by `p.grad`, followed by the same single decay. This preserves the optimizer rule for heads in both arms. Equal head trajectories are not claimed, because changed role trajectories can change later head gradients. Kernel lines 61-64 and checker lines 14-21 and 34-50 support this result.

**In-memory save/load continuation is supported.** PyTorch 2.5.1's inherited optimizer loader recursively restores nested dictionaries and their tensors, so the nested task states and integer counters have a valid restore path. Reconstructing the optimizer with the same arm and parameter/role ordering is part of the interface: `split`, `roles`, and `role_ids` are constructor attributes, not self-describing fields in the optimizer state dictionary. The checker does reconstruct the same arm and order. No added serialization framework is needed for this scoped kernel. See checker lines 57-70 and the [PyTorch optimizer loader](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/optimizer.py).

**The AMP signature and CPU GradScaler path pass; CUDA and real-model AMP remain untested here.** PyTorch 2.5.1 forwards `GradScaler.step` keyword arguments to ordinary optimizer `step`, so the three keyword-only arguments are supported. GradScaler unscales/checks parameter `.grad` tensors only; the caller must separately unscale the R/A buffers and keep the supplied parameter gradients consistent with the assembled totals. All separate R/A buffers and all parameter gradients are checked before the kernel mutates any state or parameter. If GradScaler skips a step for nonfinite parameter gradients, the trainer must retain the registered stop policy. These are existing plan obligations, not requests for fallback logic. See kernel lines 35-47, plan line 16, and the [PyTorch 2.5.1 GradScaler source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/grad_scaler.py).

The latest checker lines 90-113 exercise actual `torch.amp.GradScaler('cpu')` scale/backward/unscale/keyword-step/update calls for six steps per arm, including one unsupported-rank step. Separately divided R/A buffers are supplied after unscaling parameter gradients. Each resulting parameter matches the corresponding direct unscaled custom-optimizer reference exactly. This verifies the scaler interface and scaling sequence on CPU; the earlier native comparisons supply the independent arithmetic reference. These finite CPU checks do not exercise CUDA autocast, a real model, or overflow-driven scaler skips.

**The synthetic checker supplies useful external arithmetic comparisons.** It uses native AdamW instances for the shared and separate task directions, resets only their reference parameters while retaining native moments, and applies decay once to the reference role. It tests four absent steps including an initially absent rank task, one supported zero step, a learning-rate change, a native head reference, resumed next-step equality, and rejection of a nonfinite rank buffer while supplied parameter gradients remain finite. The rejection check verifies parameters and every recorded moment/counter remain unchanged. The native reference is substantively independent of the custom direction helper.

The supplied execution receipt reports:

| Item | Shared arm | Split arm |
| --- | ---: | ---: |
| Synthetic trajectory steps | 12 | 12 |
| Observed rank steps | 8 | 8 |
| Maximum native role-parameter error | 0 | 2.980232238769531e-7 |
| Exact next-parameter-update resume checks | 12 | 12 |
| Nonfinite buffer atomic rejection | pass | pass |
| CPU GradScaler steps with exact unscaled-reference parameters | 6 | 6 |

The receipt reports PyTorch `2.5.1+cu121`, `cuda_initialized=false`, seed 42, zero model forwards, and zero training updates. The small split difference is consistent with different FP32 summation/update order and passes the checker's declared tolerances. Native head comparisons also passed their declared tolerance; the receipt does not record their maximum error, so an exact native-head equality claim is not supported by this receipt. The reviewer inspected the checker and receipt and did not independently re-execute the checks. The 24 resume comparisons assert exact next-step **parameter** equality after an in-memory state-dictionary load; they do not independently compare every post-step state tensor or exercise disk serialization.

**The plan's causal claims are honest.** It seals prior R2, removes EMA balancing in both new arms, uses the same direct decomposition for the two arms, and states that this is a new control rather than an exact R2 replay. It explicitly acknowledges treatment from the first warmup update, changed update scale from summing preconditioned directions, and the inability of a positive result to identify the separate causal contributions of moments, support clock, scale, or roles. Failure would limit the registered configuration. The registered result is presently NOT_RUN, and the old R2 result was supplied as context rather than independently re-audited in this review.

**Remaining work belongs to the already registered integration gates.** The later trainer review and real-source M0 must verify the actual complete current/history R and direct A construction, true source-support flags, unscaling of all buffers, role/head membership, update witnesses, and restored model/optimizer/scaler behavior. The reported moment-memory difference is arithmetically 40.90 MiB; actual role tensor counts and peak runtime memory were outside this review. Existing disk-budget and full-endpoint verification requirements remain pending. This kernel pass satisfies only the bounded review requested here and carries no launch clearance.

No concrete non-blocking code defects were found. The limits above describe untested scope rather than defects requiring extra defensive code.
