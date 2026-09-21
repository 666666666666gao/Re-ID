# Supported task-state V1: full M0 audit request

Dispatch only after the original M0 and M0_CPU stages have both exited zero and the full text intake exists. This is a request checklist, not an audit verdict.

Use a fresh agent with no conversation fork, gpt-6-astra, max reasoning. Apply experiment-audit and its installed execution policy. Record same-family, provisional semantic review and backend attribution limits. Independently read the artifacts; do not rely on executor conclusions.

## Paths

- Local repository: C:/Users/gb/.trifusion_github_publish_22c3bee
- Remote repository: /root/autodl-tmp/trifusion-v2/TriFusion-ReID
- Remote run: /root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3
- Local full text intake: D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_task_state_m0_complete_20260921
- Config: configs/MSVR310/TriFusion-supported-task-state-paired-v1.json
- Plan and tracker: refine-logs/msvr310_supported_task_state_v1/
- Pipeline: tools/run_msvr_supported_task_state.py
- Training: tools/train_msvr_supported_task_state.py
- Optimizer: tools/msvr_task_state_optimizer.py
- Serialization: tools/msvr_task_state_records.py
- T0: tools/check_msvr_supported_task_state.py
- Synthetic mathematics: tools/check_msvr_task_state_math.py
- Full verifier: tools/verify_msvr_supported_task_state.py
- Record verifier: tools/verify_msvr_task_state_records.py
- Reused bound helpers: tools/msvr_supported_gradient_balance.py and tools/verify_msvr_supported_gradient_balance_stats.py
- Prior code reviews: evidence/supported_task_state_kernel_20260921 and evidence/supported_task_state_integration_20260921
- Output: D:/Program Files/UserCache/gb/codex/tmp/trifusion_supported_task_state_m0_independent_audit_20260921

## Checklist

Apply all experiment-audit A-F checks with exact file:line evidence: dataset ground truth provenance, score definitions and normalization, real artifact existence, called evaluation paths, scope, and evaluation type. Separate synthetic arithmetic, actual model engineering checks, and retrieval claims.

Independently verify full 248-step coverage: six capacity endpoints and two overfit endpoints. Check source-only identity/scene provenance and no held-out or official image access; masks, full candidate policy, frozen Signal, matched initialization, optimizer/config/source hashes, and unchanged registered gates.

Check current plus historical ranking gradients, direct auxiliary gradients, all required reference checks and tolerances; do not accept total-minus-ranking as an exact auxiliary reference. Check AMP unscaling and finite checks occur before task-state mutation, one weight decay application, classifier-head behavior, and original RNG/buffer/history replay definitions.

Inspect saved optimizer/scaler states on the server, including both task clocks, moment tensors, parameter groups, and final-record correspondence. Distinguish actual unsupported batches from synthetic unsupported checks. Unsupported ranking must not advance its clock or apply its old momentum; supported zero ranking remains an observation. Check warmup and post-warmup definitions against the registered plan. Scalar moment norms do not reconstruct per-task update vectors or the full training trajectory.

Verify both arms' actual parameter update records, original M0 gates, complete CPU receipt and file bindings. Do not turn summed independently preconditioned directions into a pure state-history causal claim: effective step magnitude and warmup also change. Do not infer Q1 or official performance from engineering results.

Read-only audit: no GPU/model forwards, no optimizer updates, no reruns or modifications of live jobs or sealed results. CPU artifact checks are permitted. All model/optimizer binaries, images, and arrays remain remote; collect only text/code/receipts. Use the existing private SSH helper only through execution, never inspect or print its contents. Coordinate any needed remote command with the root agent. Produce EXPERIMENT_AUDIT.md and .json with blockers, limitations, deterministic checks, attribution, and allowed claims.
