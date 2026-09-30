# 角色 Patch 读取实验执行表

2026-09-30，登记阶段；正式训练尚未启动。

| Run | 数据集 | 模式 | 预算/选点 | 状态 |
|---|---|---|---|---|
| L201 | RGBNT201 | local_memory：9 Patch | seed42 / 50轮 / fused mAP-best | CPU通过，待真实M0 |
| L100 | RGBNT100 | local_memory：9 Patch | seed42 / 50轮 / fused mAP-best | CPU通过，待真实M0 |
| LMSVR | MSVR310 | local_memory：9 Patch | seed42 / 50轮 / fused mAP-best | CPU通过，待真实M0 |
| F201 | RGBNT201 | full_memory：128 Patch | seed42 / 50轮 / fused mAP-best | CPU通过，待真实M0 |
| F100 | RGBNT100 | full_memory：128 Patch | seed42 / 50轮 / fused mAP-best | CPU通过，待真实M0 |
| FMSVR | MSVR310 | full_memory：128 Patch | seed42 / 50轮 / fused mAP-best | CPU通过，待真实M0 |

上一提示六端已完整验收；carry的201条件差为负，后继门失败。没有启动其条件global-only，不将本新实验改名为旧实验成功版本。
