# EV1 execution tracker

Actual16:00 snapshot: four accepted full50/strictreload endpoints; twoRGBNT100 continue unchanged. Source243 unchanged, report invocations0.

| Dataset | Condition | Status | Epochs | Formal mAP / R1 |
|---|---|---|---:|---|
| RGBNT201 | semantic | COMPLETE | 50 | 69.8595978 / 71.6507196 (epoch10) |
| RGBNT100 | semantic | RUNNING | 31 | Pending |
| MSVR310 | semantic | COMPLETE | 50 | 52.6870403 / 71.9120145 (epoch24) |
| RGBNT201 | combined | COMPLETE | 50 | 69.2028873 / 70.5741644 (epoch10) |
| RGBNT100 | combined | RUNNING | 15 | Pending |
| MSVR310 | combined | COMPLETE | 50 | 51.7131982 / 70.8967865 (epoch24) |

EV-A201/MSVR completed paired conditions FAIL; no full6 scientific report yet. Existing controller1301010 and unique CPU waiter1310752 remain the owners; do not restart on SSH timeout.
2025 extra-resource preparation: existing stack packed, transfer interrupted, validation pending. No EV1 migration or successor. Next scheduled read16:45. GoalACTIVE_UNMET.

## Terminal snapshot — 2026-10-02

Terminal: all six full50 endpoints and strict reload completed; one CPU report invocation completed. EV-A FAIL, EV-B FAIL. No successor registered. Historical snapshots above and the original audited tracker mirror remain unchanged.

| 数据集 | EV1条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | semantic | 10 | 69.8595978 | 71.6507196 | 80.7416260 | 85.4066968 |
| RGBNT201 | combined | 10 | 69.2028873 | 70.5741644 | 80.3827763 | 85.2870822 |
| RGBNT100 | semantic | 12 | 79.1431140 | 94.9854255 | — | — |
| RGBNT100 | combined | 30 | 78.8446032 | 95.1603472 | — | — |
| MSVR310 | semantic | 24 | 52.6870403 | 71.9120145 | — | — |
| MSVR310 | combined | 24 | 51.7131982 | 70.8967865 | — | — |

Public report: logs/semantic_native_complete734_20261002/raw/results/semantic_native_evidence_complete_20261002/SUMMARY.json and REPORT.md.
Integrity: logs/semantic_native_complete734_20261002/EXPERIMENT_AUDIT.md (fresh Astra/max, same-family/provisional).
2025 resource validated independently; no EV1 migration/retry. Combined adds93,248 parameters; single seed42 and consumed official benchmarks remain explicit limits. GoalACTIVE_UNMET.
