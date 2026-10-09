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
