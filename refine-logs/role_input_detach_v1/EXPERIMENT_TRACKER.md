# 角色读取梯度边界执行表

| Run ID | 阶段 | 目的 | 条件 | 数据 | 指标 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| D201-S | M0→fresh50→strict | 对照原semantic的读取梯度路径 | detached semantic | RGBNT201 | mAP/R1/R5/R10 | MUST | PREPARED_NOT_LAUNCHED |
| D201-N | M0→fresh50→strict | 对照原native的读取梯度路径 | detached native | RGBNT201 | mAP/R1/R5/R10 | MUST | PREPARED_NOT_LAUNCHED |
| D310-S | M0→fresh50→strict | 对照原semantic的读取梯度路径 | detached semantic | MSVR310 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |
| D310-N | M0→fresh50→strict | 对照原native的读取梯度路径 | detached native | MSVR310 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |
| D100-S | M0→fresh50→strict | 对照原semantic的读取梯度路径 | detached semantic | RGBNT100 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |
| D100-N | M0→fresh50→strict | 对照原native的读取梯度路径 | detached native | RGBNT100 | mAP/R1 | MUST | PREPARED_NOT_LAUNCHED |

仅26GPU0/1，顺序一组两卡；原V6九端/§812六best诊断均已完成并封存。控制复用其原接受结果，不调用已退役旧M0二进制。没有新训练成绩；每端真实8-update M0通过才从相同初始化fresh50。完整Goal active/unmet。


## §41.814 真实执行起点

六项角色读取梯度边界对照已真实启动。RGBNT201 semantic 自己的8次有效更新M0通过，281/281张量有非零有限梯度，作者BN计数8、重载差0，匹配初始state/cfg/容量；fresh50已开始。当前M0 1/6、正式完成0/6，没有新的完整50轮分数。只用26GPU0/1，不管功率温度；科学Goal active/unmet。

观察：2026-10-04T09:54:15.361783+08:00；首项PID1313574，fresh50始于2026-10-04T09:53:44.383147+08:00。完整证据与不确定性边界见logs/role_input_detach_launch814_20261004。
