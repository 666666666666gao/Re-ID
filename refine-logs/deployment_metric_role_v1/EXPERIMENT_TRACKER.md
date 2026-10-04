# 当前执行登记

§41.837：代码/CPU计算、配置绑定和验收清理合成检查通过，旧332 source/187控制不变，新scope339。§41.838实际队列已启动；首端真实M0通过，fresh50运行；没有正式完成结果。

| 数据集 | 条件 | 阶段 | 状态 |
|---|---|---|---|
| RGBNT201 | semantic | prepare/M0通过，fresh50运行 | train3610653，从02:16:15 |
| RGBNT201 | native | 同上 | 已登记待运行 |
| MSVR310 | semantic | 同上 | 已登记待运行 |
| MSVR310 | native | 同上 | 已登记待运行 |
| RGBNT100 | semantic | 同上 | 已登记待运行 |
| RGBNT100 | native | 同上 | 已登记待运行 |

只有26GPU0/1，无功率/温度动作。旧控制不重跑，逐端严格验收后退役探针；正式每端仅best。Goal active/unmet。

2026-10-05 §41.838：部署几何六端队列已启动，RGBNT201 semantic真实8步M0通过并进入fresh50；其余五端依次待运行。首端281/281有效梯度、BN8次、重载maxdiff0。无新正式结果，仅26GPU0/1，不管功率/温度，Goal ACTIVE / UNMET。
唯一预计节点observer首次02:54:04，运行中的339source不改。
