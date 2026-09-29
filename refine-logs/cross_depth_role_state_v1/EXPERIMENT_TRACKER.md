# Cross-depth role state tracker

Actual22:19:17 collector:4/9 formal endpoints verified. Actual22:18 process snapshot:4 COMPLETE,4 RUNNING,1 PENDING,0 FAILED; all four GPUs show100% utilization. Epochs below belong to that snapshot, not current completion estimates.

| Condition | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| mixed_once | VERIFIED_COMPLETE / best2 | M0_PASS / TRAIN31/50 | VERIFIED_COMPLETE / best15 |
| depth_mean | VERIFIED_COMPLETE / best2 | M0_PASS / TRAIN10/50 | VERIFIED_COMPLETE / best15 |
| depth_recurrent | M0_PASS / TRAIN33/50 | M0_PASS / TRAIN1/50 | PENDING |

All9 conditions retain seed42, complete50, official fused mAP-best, strict reload and complete gallery. Four formal endpoints and eight M0 endpoints are distinct counts. Previous three accepted row objects remain exactly unchanged; old tracker2154 and immutable handoff review input retain the first-formal audit bindings.

RGBNT201 mixed/mean:72.5888817496/74.1626799107 and72.5327582009/73.9234447479. MSVR mixed/mean:53.0316750259/68.1895077229 and52.7232371173/68.6971247196. No recurrent formal benefit, multi-seed generalization or SOTA claim. Recurrent vs mean changes both state transfer and last/mean readout; it does not isolate state transfer alone.

Evidence: logs/cross_depth_accepted_663/664_20260929.json, progress664, archive664, pair_msvr310_mean664; four full bundles and eight raw M0 bundles. First-three formal fresh review is WARN/same-family/provisional; it did not replay remote payloads or review the fourth endpoint. Parent3133064 and205 frozen runtime files unchanged. Observer665 PID3198355 is scheduled for22:32; no early repeated checks or training restarts.
