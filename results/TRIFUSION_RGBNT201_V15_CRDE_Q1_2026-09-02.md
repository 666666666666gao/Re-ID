# TriFusion RGBNT201 V15 CRDE Q1

## Terminal verdict

`M0_PASS_Q1_FAIL_DO_NOT_PROMOTE`

The unique seed-42 complete-path identity-OOF Q1 completed on the remote
RTX3090. The scientific gate failed, so V15 is sealed and D1 was not run.
Q1 is fit-only mechanism qualification; its 88–89 mAP values are not the
30-identity dev protocol and cannot be compared with the deployable 65 mAP
gate or public official-test results.

## Raw fold results

All values below are exchange-on minus the exact same-fold, same-checkpoint
no-exchange comparator in mAP percentage points.

| Fold | Fused | CNN | Transformer | Mamba | Gate-relevant observation |
|---:|---:|---:|---:|---:|---|
| 0 | +0.0952 | -0.0258 | -0.3291 | +0.9375 | only one receiver positive |
| 1 | **-0.8311** | -0.0836 | -0.6967 | -0.8020 | fused and all receivers regress |
| 2 | +0.1605 | -0.3470 | +0.1904 | +0.6480 | two receivers positive |
| Weighted aggregate | **-0.1721** | **-0.1576** | **-0.2606** | +0.2898 | fused/CNN/T regress |

## Aggregate on/off metrics

| Output | No exchange mAP | CRDE mAP | Delta | CRDE Rank-1 |
|---|---:|---:|---:|---:|
| fused | 88.6915 | 88.5194 | -0.1721 | 91.5937 |
| CNN | 87.0976 | 86.9400 | -0.1576 | 89.3170 |
| Transformer | 88.4012 | 88.1406 | -0.2606 | 91.7688 |
| Mamba | 87.6929 | 87.9826 | +0.2898 | 91.5937 |

The fused identity-cluster bootstrap observed gain is `-0.1721` mAP and its
95% lower bound is `-0.9503` over 21 identities and 10,000 resamples.

## Gate result

| Registered condition | Result |
|---|---|
| Fused nonnegative in every fold | FAIL |
| Weighted fused gain at least +1.0 mAP | FAIL |
| Fused bootstrap 95% lower bound above zero | FAIL |
| Aggregate CNN/T/M gains all positive | FAIL |
| At least two positive receivers per fold | FAIL |
| Aggregate fused strictly above CRDE branches | PASS |
| Integrity/access contracts | PASS |

## Findings

1. **Observation:** fold 1 reduces all four outputs, while folds 0/2 have only
   small fused gains. **Interpretation:** the learned exchange is not stable
   across unseen identity groups. **Implication:** CRDE does not establish the
   intended robust collaboration claim.
2. **Observation:** only Mamba has a positive aggregate matched gain.
   **Interpretation:** the exchange can help one receiver in some folds but is
   receiver-asymmetric. **Implication:** three-way mutual benefit is false for
   this fixed method.
3. **Observation:** every fold trains all 110/110 tensors with zero overflow,
   unchanged frozen state, and dev0/official0. **Interpretation:** the negative
   result is not explained by an incomplete run, dead exchange, OOM or protocol
   leakage. **Implication:** additional epochs cannot be assumed to fix it.

Within the frozen boundary there is no follow-up training experiment. Any
successor must be a new preregistered representation hypothesis, not a CRDE
LR/epoch/regret/edge-scale/checkpoint scan.

## Resource and provenance facts

- 60/60 epochs, 1,669 optimizer steps, zero overflow.
- Peak allocated/reserved: 9,185.62 / 12,324 MiB.
- Runtime: 2,411.81 s (40.20 min).
- Clean commit: `71152d3848c05177da0af30b0b921c6a3aa9942a`.
- Repository diff SHA: empty SHA-256.
- Dev / official access: 0 / 0; D1 executed: false.

## Evidence

- `evidence/trifusion_v15_q1_seed42_71152d3.json`
- `evidence/trifusion_v15_q1_seed42_71152d3.console.log`
- `evidence/trifusion_v15_q1_seed42_4cdc1ec_optimizer0.console.log`
- `RESULT_TO_CLAIM_V15.md/.json`
- `EXPERIMENT_AUDIT_V15_Q1.md/.json`
