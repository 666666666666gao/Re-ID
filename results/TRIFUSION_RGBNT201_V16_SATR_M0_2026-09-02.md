# TriFusion RGBNT201 V16 SATR M0

## Terminal verdict

`M0_FAIL_Q1_BLOCKED`

V16 Signal-Anchored Triadic Repair completed its seed-42 train-only M0 on the
remote RTX3090. Capacity, optimization, exact-prefix, frozen-state and paired
endpoint checks passed, but the preregistered fixed-initial relation-activity
gate failed. Q1, D1, dev and official test were not run.

This run produces no new retrieval metric. The best deployable same-protocol
result remains V8 Phase-B at `58.4050 mAP / 59.3939 Rank-1`, which is `6.5950`
mAP below the frozen 65 mAP gate.

## Engineering results

| Check | Result |
|---|---:|
| SATR/no-SATR initial model state, all folds | exact match |
| Trainable-name set, seed contract, first 8 transformed batches | exact match |
| Exact Signal prefix | PASS |
| Frozen Signal/shared-tail state | unchanged |
| Real capacity batch | B64/K8 |
| Capacity gradients | 203 / 203 tensors |
| Capacity AMP overflows | 0 |
| Peak allocated / reserved | 5715.68 / 5962 MiB |
| 100-step gradients | 203 / 203 tensors |
| Fixed-batch loss | 0.622885 → 0.581252 |
| Floor-aware excess-loss ratio | 0.064479 ≤ 0.10 |
| Dev / official accesses | 0 / 0 |

The RTX3090 has ample memory for the registered physical batch. Memory is not
the reason V16 stopped.

## Failed activity gate

Coverage is eligible relations divided by valid cross-camera Signal-hard
queries over the frozen first eight B64/K8 batches.

| Fold | Receiver | Frozen probe | Clean M0 replay | Gate `[0.5%,25%]` |
|---:|---|---:|---:|---|
| 0 | CNN | 1.5625% | 3.1250% | PASS |
| 0 | Transformer | 3.1250% | **0.0000%** | FAIL |
| 0 | Mamba | 4.6875% | 3.1250% | PASS |
| 1 | CNN | 3.1250% | **0.0000%** | FAIL |
| 1 | Transformer | 1.5625% | **0.0000%** | FAIL |
| 1 | Mamba | 14.0625% | 7.8125% | PASS |
| 2 | CNN | 4.1667% | 2.7778% | PASS |
| 2 | Transformer | 1.3889% | **0.0000%** | FAIL |
| 2 | Mamba | 5.5556% | 11.1111% | PASS |

The formal M0 endpoint pair reproduced identical post-transform RGB/NI/TI
tensor hashes and sampler indices, so this is not an SATR-versus-no-SATR
fairness failure. It is a mismatch between the proposal-time threshold probe
and the now-versioned runner draw. The earlier threshold evidence did not bind
its sampler indices or transformed-batch hashes; its Signal-margin quantiles
also differ from the formal replay. Therefore its positive per-receiver
coverage cannot be reproduced as the registered M0 contract requires.

## Decision boundary

The activity interval and relation gap are not relaxed after seeing this
failure. V16 is stopped before Q1. No threshold, worker-count, RNG-order,
epoch, learning-rate or loss-weight search is authorized under this experiment
identity. A successor must preregister an executable, hash-bound activity
probe and avoid making the scientific mechanism depend on one undocumented
augmentation draw.

## Evidence

- `evidence/trifusion_v16_satr_m0_seed42_20260902.json`
  (`sha256=8bf94eb5b90b8661d58fffa709c03f40b59c4aab0c70bb1b56b8cef62677cbbe`)
- Remote artifact:
  `/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v16_satr_m0_seed42_20260902/run_summary.json`
- Frozen proposal probe: `refine-logs/v16/threshold-freeze-readonly.md`
