

## §41.870 — 未启动任务状态字段检查修正（2026-10-06T18:12:15.507340+08:00）

869 supervisor699872/controller699874于18:05:34.940349 EXIT1，18:09:10实查两者均不存。失败在旧任务检查、campaign目录创建及任何NN/初始化前：原868两份PENDING full任务均没有steps键，masked在调用panel.run前遇到容量断言，尚未把full步骤写入状态文件；all_patch从未进入。旧queue初始PENDING任务本来就不写此字段，直接j['steps']是新控制器错误。七端350轮/13194更新及八M0仍有效；新869正式更新为0、报告为0，不能把snapshot的accepted0误解为抹除原七端。原失败/源码c1a70890/overlay及日志保留。

唯一队列逻辑修正为PENDING且not j.get('steps')，同时新独立v2 overlay保存当前来源，原374和v1 overlay不变。真实868两个旧任务经实际AST gate重放通过；空列表通过，非空步骤及COMPLETE均被拒绝。fixture R1错误预期一份旧full带空steps而失败，在归档/文档/NN变更前停止并保留。这不是假设性防御层、模型修复、新M0或独立审计。新870仍只复用7full/8M0/9真实initializer，执行RGBNT100 masked/all_patch fresh50和缺失all_patch8M0，原50/RAW/heads/AMP256/采样/Adam/top-k/seed/过滤/容差/2GiB阶段门与3634392494B启动预算均不变。

本节发布时新870尚未启动，仅26物理GPU0/1，一个NN，无25/GPU2/3/功率/温度操作；无需重跑任何旧科学结果。既有negative科学边界、五十轮曲线/同一mAP-best规则与Goal ACTIVE/UNMET保留。
