# 跨光谱槽位传输：局部质量与模态对统一质量

| 数据集 | 质量分配 | best轮 | mAP | R1 | 正式步数 |
|---|---|---:|---:|---:|---:|
| RGBNT201 | slot_mass | 8 | 74.3388 | 78.5885 | 2649 |
| RGBNT201 | uniform_mass | 8 | 74.2569 | 78.8278 | 2649 |
| MSVR310 | slot_mass | 38 | 50.5466 | 68.0203 | 706 |
| MSVR310 | uniform_mass | 38 | 50.5477 | 68.0203 | 706 |
| RGBNT100 | slot_mass | 26 | 83.4181 | 96.2682 | 3129 |
| RGBNT100 | uniform_mass | 26 | 83.8952 | 96.3848 | 3129 |

Single seed42 on consumed official benchmarks. Original RAW duties; slot-specific versus pair-uniform real mass. Both change old48 Mamba to private3x16. Same-P total matrix mass matches, not message energy or trained budget. No correspondence truth, text, new loss, seed stability or SOTA claim.
