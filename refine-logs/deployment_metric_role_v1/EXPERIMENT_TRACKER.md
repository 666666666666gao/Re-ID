# 当前执行登记

§41.837：代码/CPU计算、配置绑定和验收清理合成检查通过，旧332 source/187控制不变，新scope339。真实prepare/M0/fresh50尚未启动。

| 数据集 | 条件 | 阶段 | 状态 |
|---|---|---|---|
| RGBNT201 | semantic | prepare→8M0→fresh50→first strict | 已登记待运行 |
| RGBNT201 | native | 同上 | 已登记待运行 |
| MSVR310 | semantic | 同上 | 已登记待运行 |
| MSVR310 | native | 同上 | 已登记待运行 |
| RGBNT100 | semantic | 同上 | 已登记待运行 |
| RGBNT100 | native | 同上 | 已登记待运行 |

只有26GPU0/1，无功率/温度动作。旧控制不重跑，逐端严格验收后退役探针；正式每端仅best。Goal active/unmet。
