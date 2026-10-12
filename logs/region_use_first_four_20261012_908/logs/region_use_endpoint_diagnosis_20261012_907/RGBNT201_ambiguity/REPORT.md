# Actual-region supervision: one original endpoint

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Last mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | ambiguity | 8 | 74.6299 | 79.0670 | 88.2775 | 91.8660 | 71.9604 |

One seed42, original full50/strict selected-best distances, complete legal query/gallery and matched raw-semantic order. No model inference, optimizer, extra seed, test-time comparison, stronger-teacher claim or post-hoc rescue. Source support/auxiliary loss and AP tails are descriptive; no unique failure cause. Same-model global/regional decomposition and uniform-versus-ambiguity remain separate consumers, not fabricated here. Official benchmark consumed in development.

ambiguity minus raw_semantic: mAP +0.2550, R1 +0.2392; first repairs/new errors 8/6.

ambiguity minus independent_global: mAP +0.3332, R1 +0.1196; first repairs/new errors 8/7.
