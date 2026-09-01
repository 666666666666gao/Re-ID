# TriFusion V8 frozen-router probe — 2026-09-02

This read-only probe asks whether V8 should freeze the selected V7 experts and
change only the utility Router and residual energy. The answer is no: the probe
finds a real equal-energy gain, but neither a learnable fit-domain routing
target nor enough held-out performance to pass the 65 mAP development gate.

## Protocol

- Source: V7 selected epoch-1 checkpoint, SHA-256
  `8bcdf3583e121dd7a7b0071743b8fd34f93a82cc710bd07e26b53c3693609a2b`.
- Model parameters are frozen; model training and optimizer steps are zero.
- Only fit identities with cross-camera positives can define retrieval AP:
  21 identities and 571 queries. Evaluation uses all 30 disjoint dev identities
  and 825 queries.
- An analytic 18-to-3 least-squares teacher maps the frozen reliability
  features to each query's residual-expert AP gain. It is fit on fit identities
  and evaluated on disjoint dev identities.
- The deployable probe concatenates the exact Signal baseline with a routed
  residual bank whose norm equals the baseline norm. No alpha or temperature
  scan is performed.
- FP32 extraction with cuDNN benchmark disabled produced byte-identical core
  receipts in two consecutive runs. Official-test access is zero.

## Routing result

| Split/router | CNN target | Transformer target | Mamba target | Accuracy | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| fit target/teacher | 100.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| dev target/fit teacher | 55.27% | 17.45% | 27.27% | 55.27% | 55.27% |
| dev target/V7 Router | 55.27% | 17.45% | 27.27% | 27.39% | 55.27% |

The fit teacher has only one target class and therefore predicts CNN for every
dev query. It cannot beat the fit-derived majority policy. The V7 Router is
worse: it predicts Mamba for 95.39% of dev queries and CNN for none.

## Retrieval result

| Output | mAP | Rank-1 |
|---|---:|---:|
| exact Signal baseline | 58.0109 | 57.4545 |
| original V7 fused (FP32 probe) | 58.3285 | 57.9394 |
| residual-only CNN | 59.1317 | 59.3939 |
| residual-only Transformer | 54.8594 | 54.4242 |
| residual-only Mamba | 57.8991 | 57.2121 |
| equal-energy current Router | 59.5902 | 58.9091 |
| equal-energy uniform | **59.6188** | **59.1515** |
| equal-energy fit utility teacher | **59.6188** | **59.1515** |

Equal residual energy improves the exact Signal baseline by 1.6079 mAP and the
strongest fixed residual by 0.4871 mAP. This confirms that V7's approximately
0.2 residual-energy ratio suppresses useful information. However, 59.6188 is
still 5.3812 below the frozen 65 mAP gate. The separate ground-truth
residual-only Oracle is 62.7435 mAP and is also below 65.

## Decision

The frozen-router probe gate fails because the fit teacher does not beat the
majority policy on disjoint dev identities and the deployable fused result does
not reach 65. A V8 that only freezes the current experts and retrains the Router
is rejected before main training. The next main hypothesis must improve expert
representations and their task division, while retaining the exact Signal
baseline output. This probe is diagnostic, not an ablation or SOTA result.

Evidence:

```text
evidence/trifusion_v8_frozen_router_probe_seed42.json
```
