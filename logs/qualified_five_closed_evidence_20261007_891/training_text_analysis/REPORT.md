# 完整训练文本中的目标活动与关系支持

| 数据集 | 目标 | steps | 正增量loss步 | best轮修正/global均值 | E50均值 |
|---|---|---:|---:|---:|---:|
| RGBNT201 | md_batch_ratio | 2649 | 2649 | 0.058247 | 0.117911 |
| RGBNT201 | repair_keep | 2649 | 1657 | 0.061599 | 0.037538 |
| MSVR310 | md_batch_ratio | 706 | 706 | 0.005072 | 0.005034 |
| MSVR310 | repair_keep | 706 | 706 | 0.005277 | 0.005229 |
| RGBNT100 | md_batch_ratio | 3129 | 3129 | 0.295475 | 0.666420 |

RGBNT201：合法query曝光占33.866553%；其中多正例占92.185105%；repair/keep有支持批次189/2649、2557/2649；非零loss批次189/2649、1653/2649。

MSVR310：合法query曝光占48.043555%；其中多正例占78.883361%；repair/keep有支持批次640/706、706/706；非零loss批次640/706、706/706。

CPU saved-text calculations only. Exposures are repeated query/relations, not unique identities. Mean loss and correction amplitude are not gradient or complementary retrieval evidence. M0 isolated qualification is not full-training per-role gradient proof. No new model forward, optimizer, SSH, rerun or threshold change.
