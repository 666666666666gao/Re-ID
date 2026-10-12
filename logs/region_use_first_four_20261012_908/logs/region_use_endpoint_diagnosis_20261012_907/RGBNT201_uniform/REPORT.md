# Actual-region supervision: one original endpoint

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Last mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | uniform | 8 | 74.7880 | 78.9474 | 88.6364 | 92.2249 | 71.6362 |

One seed42, original full50/strict selected-best distances, complete legal query/gallery and matched raw-semantic order. No model inference, optimizer, extra seed, test-time comparison, stronger-teacher claim or post-hoc rescue. Source support/auxiliary loss and AP tails are descriptive; no unique failure cause. Same-model global/regional decomposition and uniform-versus-ambiguity remain separate consumers, not fabricated here. Official benchmark consumed in development.

uniform minus raw_semantic: mAP +0.4132, R1 +0.1196; first repairs/new errors 7/6.

uniform minus independent_global: mAP +0.4914, R1 +0.0000; first repairs/new errors 6/6.
