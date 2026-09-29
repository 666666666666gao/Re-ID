# Cross-depth role state: full registered panel

All nine registered seed42 endpoints completed 50 epochs. Each row uses one checkpoint selected by highest official fused mAP, then strictly reloaded for full query/gallery scoring. This is a second training stage over a frozen dataset-trained pure CLIP ReID baseline. Fresh complete numerical audit is PENDING.

| Dataset | Condition | Best epoch | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | mixed_once | 2 | 72.5889 | 74.1627 | 82.8947 | 88.0383 |
| RGBNT100 | mixed_once | 1 | 85.1237 | 95.0437 | 95.7434 | 96.0350 |
| MSVR310 | mixed_once | 15 | 53.0317 | 68.1895 | 82.2335 | 87.6481 |
| RGBNT201 | depth_mean | 2 | 72.5328 | 73.9234 | 82.7751 | 88.0383 |
| RGBNT100 | depth_mean | 1 | 85.1700 | 95.0437 | 95.6851 | 95.9767 |
| MSVR310 | depth_mean | 15 | 52.7232 | 68.6971 | 83.0795 | 87.4789 |
| RGBNT201 | depth_recurrent | 2 | 72.7638 | 74.2823 | 83.1340 | 88.0383 |
| RGBNT100 | depth_recurrent | 1 | 85.1635 | 95.0437 | 95.6851 | 95.9184 |
| MSVR310 | depth_recurrent | 8 | 52.0987 | 67.3435 | 83.0795 | 86.6328 |

| Dataset | Recurrent minus depth mean mAP | R1 | Fused minus shared global mAP (recurrent) |
|---|---:|---:|---:|
| RGBNT201 | +0.2311 | +0.3588 | +0.2301 |
| RGBNT100 | -0.0065 | +0.0000 | +0.0006 |
| MSVR310 | -0.6246 | -1.3536 | -0.5015 |

Interpretation boundaries:

- Recurrent versus mean changes both state carry and final-depth versus mean aggregation; it does not isolate state carry.
- Global/local values are outputs of the same selected model, not independent training controls.
- Official data were used for epoch and historical method selection. This is not untouched-test generalization or multi-seed stability.
- The launch manifest binds 205 files. Supplemental dependency bytes captured later cannot establish execution-time freezing retroactively.
- Source and filename-catalogue review is separate from full numerical performance review; pixel content and official image-release authenticity are not certified.
- Training, weights, input images and saved distance arrays remain on the remote server.

Primary matrix: `cross_depth_accepted_671_20260930.json`, SHA `5bbe27b41feb9a4b2ea5f863d07e18159cce934c191fcad2e7eeb8c55ae51aec`.
Collector timestamp: `2026-09-30T00:07:51.462110+08:00`. Existing seven accepted row objects remain unchanged.
