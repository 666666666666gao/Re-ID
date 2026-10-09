# Frozen text prior pilot tracker

当前是六端执行草案，R2方法已闭合且组件source复核完成，生产接入/prefix/NN门未闭合。没有新NN或formal。

| Run ID | Milestone | Purpose | Variant | Dataset | Metrics | Priority | Status |
|---|---|---|---|---|---|---|---|
| T0 | component | 12/full77 causal输出及输入VJP | both text packages/templates | none | fixed output/VJP tolerances | MUST | NOT_RUN |
| T1 | M0/full50 | text先验净效用 | pretrained | RGBNT201 | mAP/R1/R5/R10 | MUST | PLANNED_NOT_LAUNCHED |
| T2 | M0/full50 | 同结构固定random包控制 | random | RGBNT201 | mAP/R1/R5/R10 | MUST | PLANNED_NOT_LAUNCHED |
| T3 | M0/full50 | text先验净效用 | pretrained | MSVR310 | mAP/R1 plus full CMC | MUST | PLANNED_NOT_LAUNCHED |
| T4 | M0/full50 | 同结构固定random包控制 | random | MSVR310 | mAP/R1 plus full CMC | MUST | PLANNED_NOT_LAUNCHED |
| T5 | M0/full50 | text先验净效用 | pretrained | RGBNT100 | mAP/R1 plus full CMC | MUST | PLANNED_NOT_LAUNCHED |
| T6 | M0/full50 | 同结构固定random包控制 | random | RGBNT100 | mAP/R1 plus full CMC | MUST | PLANNED_NOT_LAUNCHED |
| T7 | report | 完整全部配对/query/成本 | all complete or missing | three | all original/results | MUST | NOT_RUN |

六旧semantic/global只复用封存控制；完整流程种子及论文级强近邻另立合同，目前不启动。


2026-10-10：用户恢复；生产source R2通过，prefix/M0/full50仍NOT_RUN。固定六端顺序和原seed42/50轮合同不变，只用26GPU0/1。


## 2026-10-10 §901：初次六端停在观察器，续接只修正M0诊断

原900队列00:39:22启动，00:50:08最后一个M0验收失败；00:57:58实际确认控制器及监督进程均退出，controller exit1、CPU报告exit0。六端各真实8次M0更新，累计48次，producer活动/重载检查通过；下游验收全部在同一isolated-gradient条件失败。正式训练0轮、0端，没有新mAP。保留原失败、六份probe及全部来源；不重放旧控制。

真实optimizer五个ψ/W参数在后七步均有非零、有限的unscaled梯度，但M0独立role VJP没有使用AMP loss scale，ψ梯度报告0。fresh init_scale256、growth_interval2000、固定8步、无跳步或scale下降，从安装源码和实际回执推出该窗口scale256；没有声称直接记录过旧scaler值或定位首个underflow算子。fresh rescue建议仅将诊断role_loss乘256，返回梯度转FP32后除256。实际训练loss不变，原活动、BN、buffer、初始化、批次与重载门槛不放宽。

901是新命名的缺失端续接：相同六端、原seed42、各fresh8M0后fresh50、1536维、原raw职责训练与作者配方。queue通过显式prior-campaign读取已终态的900，检查恰有六个M0缺失且无已接受正式端；只有run/queue两份source变化，原429、prefix/report不变，复用实际qualified prefix原字节及已绑定CLIP来源，不再执行prefix NN。source-only复核不代替新真实M0。仍只26物理GPU0/1，单作业占两卡；不查询25/GPU2/3、功率或温度。正式训练每端只保留同一mAP-best；失败probe暂为修复证据，闭合后按消费者状态清理。
