# Task-state interface preparation — NOT AN EXPERIMENT CONTRACT

2026-09-21. Full R2 Q1 audit still in progress. No optimizer implementation, training, deployment, sweep or new method qualification. This note scopes a user-suggested conditional successor; it does not select or register one.

Current source tools/train_msvr_supported_gradient_balance.py: role parameter selection74–80, AdamW83–86, full current/history rank and direct auxiliary combination267–268, gradient deletion288, unscale290 and one step295. Encoder189 tensors; heads14. Existing full R/A are AMP-scaled, with scale256 in this run. They are deleted before optimizer step. A task-state implementation must explicitly retain/unscale those buffers if used; reading p.grad afterward cannot reconstruct them.

Read-only remote checkpoint4fee97298b318c5183ce5cf60880382480487546bf0c561976fd24a511adfdb9:

| Role | Tensors | FP32 elements |
|---|---:|---:|
| CNN |42|2229120|
| Transformer |54|1388928|
| Mamba |93|1742604|
| Total |189|5360652|

One FP32 m/v pair costs42,885,216B; two task pairs85,770,432B; incremental moments40.898529MiB. This arithmetic excludes gradients, activations, heads, counters and allocator. It is not measured peak GPU memory. No GPU initialized, no model forward/update, no checkpoint downloaded.

Before any future implementation, a registered single hypothesis must settle:

1. Control optimizer and whether R2 EMA coefficients are retained; distinguish task-state separation from simultaneous changes to loss/weights/support.
2. Warmup state continuity versus state reset, and rank task meaning when fused objective switches after65steps. Duplicating common moments is not automatically a valid task decomposition.
3. Full current+history rank must enter rank moments once; direct auxiliary enters its own moments. Heads retain original total gradient; frozen Signal has no state.
4. Unsupported step versus supported zero gradient: whether rank clock/moments hold and rank momentum is not applied; this is a candidate definition, not demonstrated benefit.
5. Task effective clocks versus global LR schedule; sum/mean of preconditioned updates; weight coefficients before/after moments; epsilon placement and once-only decoupled decay.
6. Actual training uses GradScaler. All active gradient buffers must have explicit finite/scale handling and parameters/moments must advance consistently. Current execution had overflow0; no need for speculative compatibility paths.
7. Save actual optimizer state if claiming task-history diagnostics. Current R2 checkpoints have no AdamW m/v, so they cannot reconstruct past task updates. Any new stateful diagnostic must be labeled as newly observed, not a recovered old trajectory.

AdaTask attribution and primary sources: refine-logs/msvr310_supported_gradient_balance_v1/ADATASK_PRIMARY_SOURCE_NOTE_20260921.md. Separate task moments are established prior art. Missing-support clock treatment is only a candidate adaptation and not a novelty claim. This note adds no retrieval result.
