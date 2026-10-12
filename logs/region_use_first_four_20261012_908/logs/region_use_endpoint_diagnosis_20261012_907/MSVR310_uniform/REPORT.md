# Actual-region supervision: one original endpoint

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Last mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| MSVR310 | uniform | 38 | 50.5418 | 67.8511 | 80.5415 | 85.4484 | 50.4903 |

One seed42, original full50/strict selected-best distances, complete legal query/gallery and matched raw-semantic order. No model inference, optimizer, extra seed, test-time comparison, stronger-teacher claim or post-hoc rescue. Source support/auxiliary loss and AP tails are descriptive; no unique failure cause. Same-model global/regional decomposition and uniform-versus-ambiguity remain separate consumers, not fabricated here. Official benchmark consumed in development.

uniform minus raw_semantic: mAP -0.0004, R1 +0.0000; first repairs/new errors 0/0.

uniform minus independent_global: mAP -0.0004, R1 -0.1692; first repairs/new errors 0/1.
