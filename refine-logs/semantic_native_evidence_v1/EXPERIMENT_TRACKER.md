# EV1 execution tracker

Actual snapshots: preflight completed 2026-10-02 14:40:17 +08:00; full6 queue registered and launched once 14:46:38; first-wave state observed 14:50:26; waiter/M0 readback 14:52:33. Sources reviewed PASS (same-family/provisional); source243 unchanged. No formal endpoint is complete in this snapshot.

| Dataset | Condition | Initialization | M0 | Full50 | Strict reload | mAP / R1 |
|---|---|---|---|---|---|---|
| RGBNT201 | semantic | Verified | PASS 281/281, reload 0 | Running GPU0 | Pending | — |
| RGBNT201 | combined | Verified | PASS 287/287, reload 0 | Running GPU3 | Pending | — |
| RGBNT100 | semantic | Verified | PASS 281/281, reload 0 | Running GPU1 | Pending | — |
| RGBNT100 | combined | Verified | Pending worker | Registered/Pending | Pending | — |
| MSVR310 | semantic | Verified | PASS 281/281, reload 0 | Running GPU2 | Pending | — |
| MSVR310 | combined | Verified | Pending worker | Registered/Pending | Pending | — |

Actual durable CPU waiter PID1310752: WAITING_FULL6, report invocations0, poll240 seconds. Formal manifest SHA8555b297cdbe2a017fff3cea68430a26a135ed9afc0ffd0dfe1e9c7bed5d9842. Parent PID1301010. Receipt archive: logs/semantic_native_start731_corrected_20261002/.

EV-A / EV-B: no result. Original N1 remains FAILED with five accepted and one invalid endpoint; old native-high is historical only. Combined adds93,248 trainable parameters over semantic; this is not capacity-matched. Estimated6–9 GPUhours / wall3–4hours from launch, to be revised from completed endpoints. Goal ACTIVE_UNMET.
