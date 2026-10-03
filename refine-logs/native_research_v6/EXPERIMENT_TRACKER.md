# 逐端研究V6执行表

2026-10-03：用户选择第1项；实现完成，真实源码复核/部署/初始化/M0/正式训练待执行。旧九端FAIL不改判。

| 数据集 | global_only | semantic | native |
|---|---|---|---|
| RGBNT201 | NOT_RUN | NOT_RUN | NOT_RUN |
| MSVR310 | NOT_RUN | NOT_RUN | NOT_RUN |
| RGBNT100 | NOT_RUN | NOT_RUN | NOT_RUN |

每格分别记录实际M0、full50与strict评价状态；不以计划或初始前向代替正式成绩。仅26物理GPU0/1一个双卡任务。源码检查不是数值/容量/检索验收。
