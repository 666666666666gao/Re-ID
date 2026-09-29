# Cross-depth role state tracker

2026-09-29 21:22 actual snapshot: four production M0 passed and full50 jobs are running; five pending; zero complete formal endpoints.

| Condition | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| mixed_once | M0_PASS / TRAIN5/50 | M0_PASS / TRAIN1/50 | M0_PASS / TRAIN6/50 |
| depth_mean | M0_PASS / TRAIN4/50 | PENDING | PENDING |
| depth_recurrent | PENDING | PENDING | PENDING |

All nine endpoints use seed42, complete50, official fused mAP-best, strict reload and full-gallery verification. Results are not filled from intermediate best scores. Two source reviews finished with no current code blockers and same-family/provisional scope warnings; environment reused without dependency edits. Epoch counts are from the recorded snapshot, not live counters.


Actual launch 2026-09-29T21:18:38.471873+08:00, PID3133064; snapshot 2026-09-29T21:22:40.084511+08:00. Four real M0 passed and full50 active; five pending; formal endpoints0/9. No intermediate metrics accepted.
