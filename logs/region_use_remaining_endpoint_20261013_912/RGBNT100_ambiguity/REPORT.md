# Actual-region supervision: one original endpoint

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 | Last mAP |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT100 | ambiguity | 26 | 83.3459 | 95.7434 | 96.3265 | 97.3761 | 82.4386 |

One seed42, original full50/strict selected-best distances, complete legal query/gallery and matched raw-semantic order. No model inference, optimizer, extra seed, test-time comparison, stronger-teacher claim or post-hoc rescue. Source support/auxiliary loss and AP tails are descriptive; no unique failure cause. Same-model global/regional decomposition and uniform-versus-ambiguity remain separate consumers, not fabricated here. Official benchmark consumed in development.

ambiguity minus raw_semantic: mAP -0.7444, R1 -0.1166; first repairs/new errors 40/42.

ambiguity minus independent_global: mAP -1.1879, R1 -0.8746; first repairs/new errors 31/46.
