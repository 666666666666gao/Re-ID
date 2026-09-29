# Cross-depth role state tracker

Actual22:32:55 collector:5/9 formal verified, prior4 objects unchanged. Actual22:32 snapshot:5 COMPLETE/4 RUNNING/0 PENDING/0 FAILED, all four GPUs99-100% utilized. These epochs are historical snapshot values.

| Condition | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| mixed_once | VERIFIED_COMPLETE / best2 | M0_PASS / TRAIN38/50 | VERIFIED_COMPLETE / best15 |
| depth_mean | VERIFIED_COMPLETE / best2 | M0_PASS / TRAIN15/50 | VERIFIED_COMPLETE / best15 |
| depth_recurrent | VERIFIED_COMPLETE / best2 | M0_PASS / TRAIN7/50 | M0_PASS / TRAIN0/50 |

All9 production M0 bundles archived with8 batches each, frozen Signal, reload0 and118/118 gradient union. Five formal bundles have complete50, a single official mAP-best checkpoint, strict reload and full-gallery CPU parity. Total9947 formal scalar rows. No active-endpoint temporary score is accepted.

RGBNT201 mixed/mean/recurrent mAP-R1:72.5889/74.1627,72.5328/73.9234,72.7638/74.2823. Primary recurrent-minus-mean+.2311mAP/+.3589R1, repair3/new0, identity21up/4down. State carry and last-depth versus mean readout both change; no isolated carry cause, multi-seed stability or three-dataset gain claim. Fixedmodel identity bootstrap does not provide training variance or untouched-test inference.

Evidence: accepted665, progress665, archive665, primary pair_rgbnt201_recurrent665, five full bundles/nine raw M0 bundles. First3 fresh review remains WARN/same-family/provisional and did not replay remote payloads. Original first-formal handoff/tracker preserved. Parent3133064 and205 frozen runtime files unchanged. Observer666 PID3216915 is scheduled22:58; no early repeated progress reads or trainer restarts. Goal ACTIVE/UNMET.
