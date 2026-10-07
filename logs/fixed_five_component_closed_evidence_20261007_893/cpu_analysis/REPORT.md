# 固定五份best的global / 修正 / 融合分解

| 数据集 | 目标 | own-g mAP | c单独mAP | f mAP | f−own-g ΔmAP / ΔR1 | 首位修复 / 新增 | query AP改善 / 恶化 |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | md_batch_ratio | 74.2967 | 2.5841 | 74.4694 | +0.1727 / +0.2392 | 6 / 4 | 266 / 206 |
| RGBNT201 | repair_keep | 74.2967 | 9.9627 | 74.6521 | +0.3554 / +0.7177 | 9 / 3 | 291 / 169 |
| MSVR310 | md_batch_ratio | 50.5421 | 6.4681 | 50.5542 | +0.0120 / +0.0000 | 1 / 1 | 205 / 125 |
| MSVR310 | repair_keep | 50.5421 | 6.6670 | 50.5446 | +0.0025 / -0.1692 | 0 / 1 | 178 / 140 |
| RGBNT100 | md_batch_ratio | 84.3692 | 35.3562 | 83.9106 | -0.4586 / -0.4665 | 8 / 16 | 699 / 663 |

Standard-library CPU recheck of every saved score/query/identity row. No tensors loaded locally, no NN, fitting, checkpoint selection, new threshold or source modification. Fixed-best consumed-protocol diagnosis, not independent ablation/causal attribution/training-seed uncertainty.
