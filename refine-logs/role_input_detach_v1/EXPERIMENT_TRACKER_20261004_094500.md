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
