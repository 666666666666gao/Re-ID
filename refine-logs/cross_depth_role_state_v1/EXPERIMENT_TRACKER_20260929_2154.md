# Cross-depth role state tracker

2026-09-29 21:54:54 collector: three of nine formal endpoints verified; four child campaigns running and two pending. Six real production M0 receipts are archived. Epoch counts below come from the21:54:00 process snapshot; the recurrent201 child started by collection but its M0 has not yet been archived.

| Condition | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| mixed_once | VERIFIED_COMPLETE / best2 | M0_PASS / TRAIN18/50 | VERIFIED_COMPLETE / best15 |
| depth_mean | VERIFIED_COMPLETE / best2 | M0_PASS / TRAIN2/50 | M0_PASS / TRAIN5/50 |
| depth_recurrent | RUNNING / M0 not yet archived | PENDING | PENDING |

All nine endpoints use seed42, complete50, official fused mAP-best, strict reload and full-gallery verification. Results are not filled from intermediate best scores. The completed RGBNT201 mixed_once and depth_mean tensors share the same initialization and trainable parameter count. Their formal results are72.5888817496/74.1626799107/82.8947365284/88.0382776260 and72.5327582009/73.9234447479/82.7751219273/88.0382776260. MSVR310 mixed_once is53.0316750259/68.1895077229. No recurring-state benefit or three-dataset success is claimed.


Actual launch2026-09-29T21:18:38.471873+08:00, PID3133064. Evidence: logs/cross_depth_accepted_663_20260929.json, logs/cross_depth_progress_663_20260929.json, logs/cross_depth_archive_663_20260929.json and logs/cross_depth_pair_rgbnt201_mean_663_20260929.json. The earlier21:22 tracker remains preserved in EXPERIMENT_TRACKER_20260929_2122.md. The first accepted MSVR row remains exactly unchanged in the later matrix. Source reviews and the first-four-M0 artifact follow-up are same-family/provisional with unreplayed remote tensor and pending-mode limits. Environment and205 runtime source files remain unchanged.
