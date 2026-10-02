# F1 基础配方执行表

六端完整50轮/严格评价COMPLETE；CPU报告一次exit0；fresh integrity WARN，精确资源范围改为2026四卡。

| Dataset | Recipe | Best epoch | mAP | R1 | R5 | R10 | Steps |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | author | 27 | 73.4728 | 77.1531 | 85.8852 | 89.9522 | 2649 |
| RGBNT100 | author | 9 | 84.0319 | 96.2099 | - | - | 3129 |
| MSVR310 | author | 38 | 50.8388 | 68.6971 | - | - | 706 |
| RGBNT201 | current | 26 | 62.5806 | 62.4402 | 75.8373 | 83.0144 | 2649 |
| RGBNT100 | current | 12 | 77.5843 | 94.1691 | - | - | 6559 |
| MSVR310 | current | 24 | 50.3567 | 67.5127 | - | - | 1000 |

完整主张见logs/foundation_complete739_20261002/CLAIMS_FROM_RESULTS.md；基础配方差值不能计为新模块增益。原计划与249运行绑定不修改。
