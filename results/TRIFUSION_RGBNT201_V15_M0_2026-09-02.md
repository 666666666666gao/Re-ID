# TriFusion RGBNT201 V15 M0

## Terminal verdict

`PASS_Q1_AUTHORIZED`

The corrected seed-42 train-only M0 passed on the remote RTX3090 at clean
commit `1f2de44f0c7c953bea7d75921be509ce9704f84c`. It accessed neither the
30-identity dev split nor official test. This is an engineering/learnability
gate, not a retrieval result or SOTA claim.

## Results

| Gate | Result |
|---|---:|
| Step-0 exact exchange-on/off parity | PASS |
| Same input tensor pointers | PASS |
| Exact Signal prefix | PASS |
| Exchange stages present | 2 |
| Initial six edge scales | all zero |
| B64/K8 capacity, live stages | 0 and 1 |
| Capacity nonzero gradients | 107 / 110 |
| Capacity AMP overflows | 0 |
| Peak allocated / reserved | 9276.38 / 9798 MiB |
| 100-step gradients | 110 / 110 |
| Fixed-batch loss | 4.095560 → 1.209675 |
| Label-smoothing floor | 0.578383 |
| Matched-regret floor | 0.474426 |
| Combined floor | 1.052809 |
| Excess-loss ratio | 0.051554 ≤ 0.10 |
| Frozen state unchanged | PASS |
| Dev / official accesses | 0 / 0 |

The prior M0 receipt was not a valid failure: its runner required every tensor
to receive nonzero gradient within eight steps although the frozen contract
required finite gradients and both exchange stages to be live, and it omitted
the unavoidable matched-comparator regret floor. Both issues were corrected by
public tests before this rerun. The three capacity tensors not reached in eight
steps remain visible as diagnostics; all 110 were reached by step 100.

## Evidence

- `evidence/trifusion_v15_m0_seed42_1f2de44.json`
- `evidence/trifusion_v15_m0_seed42_1f2de44.console.log`
- `evidence/trifusion_v15_m0_seed42.json` (invalid-gate historical receipt)
- `evidence/trifusion_v15_m0_regret_floor_diagnostic_seed42.json`
