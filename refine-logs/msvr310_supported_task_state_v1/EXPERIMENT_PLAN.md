# Supported task-state AdamW V1 — selected single hypothesis

2026-09-21. IMPLEMENTATION_IN_PROGRESS / NOT_RUN. Prior R2 is sealed Q1_FAIL, +0.0532657247 mAP, 1/5. Goal remains ACTIVE/UNMET.

Hypothesis: AdaTask-inspired separate task preconditioning, with explicit absent-ranking support, can improve heldout cross-scene retrieval over a common-state optimizer. This is a joint optimizer-rule comparison, not proof that common AdamW states caused prior failure. Independent task moments and summation are existing AdaTask ideas. The support rule is an adaptation, not an established novelty claim.

## Fixed comparison

- MSVR310, existing three identity folds, seed42, all six fixed endpoints, complete gallery including distractors, unchanged scene evaluation. No official test. Original five scientific gates unchanged; no rescue or intermediate selection.
- V8 network, frozen Signal, 7680D fixed fusion, source records/augmentation, B64/K8, queue512/age8, history reencoding/full historical VJP, current64 anchors/history0, cross-scene AP tau0.01/eligible-anchor mean, other13 objectives and 20 epochs unchanged.
- Equal task weights in both arms. Do not retain R2 EMA balancing. Both arms compute R=current fused-metric derivative+historical derivative and direct A=other13 derivative, then use identical direct decomposition. This new control is not an exact replay of sealed R2 or cross-scene controls.
- Shared arm: one AdamW m/v pair for R+A in role parameters. Split arm: separate m/v for R and A; sum their preconditioned directions, as in AdaTask. Heads use ordinary total-gradient AdamW in both arms. beta=(0.9,0.999), eps=1e-8; original LR/decay/schedule preserved.
- Use split states from the first update, including hard-Triplet warmup (65 Q1 steps, original M0 warmup lengths). Warmup R is observed even if zero. At AP activation, retain all moments and clocks; no reset, duplicated common moments or special warm-start. Thus treatment acts during warmup too; warmup trajectories are not expected to match.
- Post-warmup rank_observed means at least one eligible cross-scene anchor. If absent, split rank m/v/clock remain unchanged and no old rank momentum is applied. Auxiliary advances. Shared arm continues on A with its existing common history. If supported but R is zero, advance rank moments/clock normally and allow residual momentum.
- Task bias correction uses its observed count; LR schedule uses original epoch. Decoupled weight decay applied once per parameter update. Preconditioned task directions summed, not averaged or normalized. This can alter total step size: record update norms/directions and do not claim a pure scale-controlled benefit.
- All gradients FP32 after AMP unscale, complete R/A assembled before task updates. Check every gradient including separate task buffers before any state/parameter mutation. Existing nonfinite-stop policy retained. Scaler checks combined parameter gradients; task-buffer finite checks additionally prevent hidden cancellation. No speculative overflow recovery. One logical optimizer step.

## Evidence and engineering gates

1. T0 CPU math: shared arm versus native torch AdamW; split arm versus independent two-task native AdamW directions; nonzero/zero/absent support, state clocks, single decay, head path, changing LR, state save/load continuation. These are synthetic arithmetic checks, not model results.
2. Real-source M0: existing capacity/100-step overfit and full checks, complete history direct-reference checks, actual state and parameter update witnesses. Verify unsupported behavior synthetically if no unsupported real M0 step occurs; never fabricate real coverage.
3. Full fixed Q1 only after engineering gates and fresh code review pass. Save final model plus optimizer/scaler states remotely, all per-step objectives/support/counters and late-epoch training tables. No claim that old runs' optimizer states are recovered.
4. CPU full endpoint verification and independent integrity review precede scientific reporting. Full queries/identities/folds/roles and original gate report required.

## Budget and decision

Existing R2 six-end training took about4.33 hours. Provision4–6 GPU hours plus M0/CPU verification; reestimate from actual timing. Two role moment pairs occupy85,770,432 bytes versus42,885,216 for shared; difference40.90MiB, excluding gradients/activations. This is arithmetic, not measured peak memory. Save-state disk budget must be checked before launch; preserve all current initial/final/audit dependencies.

Negative result limits this registered optimizer configuration, not all task-aware optimization. Positive result supports this entire rule only; separating support clock, step-size and role contributions belongs to later permitted mechanism experiments. No multiseed, no pre-main-result ablations.

## Attribution

See ../msvr310_supported_gradient_balance_v1/ADATASK_PRIMARY_SOURCE_NOTE_20260921.md for primary paper and fixed author-code inspection. Independently implement formulas; do not copy author implementation. No additional paper-performance claim is made here.
