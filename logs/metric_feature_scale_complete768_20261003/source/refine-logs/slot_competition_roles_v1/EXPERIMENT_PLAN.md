# 完整Patch读取的槽位竞争配对实验

2026-10-01，训练前登记。前序六端已结束且原收益门FAIL；此计划是新结构假设，不是失败后调门或追加seed。

已有全query只读诊断显示三数据集full128采样内容槽位余弦约0.9864/0.9800/0.9857。待检验：各槽位独立对Patch归一化是否促成近似共同汇总，槽位竞争分配是否改善增量身份信息及检索。这是机制假设，不是相似性已经证明的唯一根因。

Slot Attention、DINOSAUR、PLOT是直接先例，见前序SLOT_COMPETITION_PRIOR_ART_20261001.md。竞争、共享槽位不原创；本计划仅检验简化分配干预，不复制其GRU迭代、重建或额外身份目标。

## 唯一实验差异

两端score形状B×3模态×16槽位×128Patch，固定完整支持、相同Q/K/V/out投影、位置和参数。

- independent：对每个槽位的128Patch作softmax。
- competitive：每个Patch先对16槽位作softmax，再对每个槽位的Patch权重归一化为加权均值。

两端归一化都显式FP32，最终权重转回V的dtype。没有epsilon、clamp、fallback或空槽位分支；真实M0须检查有限输出/梯度。竞争分配具有对所有槽位共享的每Patch logit偏移不变性，不保证不同内容、语义部件或检索增益。

## 固定合同与执行

复用原纯CLIP ReID数据集权重、loader/增强/AMP/AdamW/学习率；M1第4/8/12层、CNN→Transformer→真实Mamba、16槽位、内部128/fused1536、固定位置、global仅条件查询、模态×槽位组读出。M3关闭、额外ID头关闭，ID+Triplet、seed42，每端50角色轮，逐轮官方fused mAP-best同一权重全部指标、严格重载、完整gallery/camera或MSVR时间段过滤、无rerank。

**新跑两个归一化×三个数据集的全部六端。**不拿历史full成绩替换fresh independent控制；FP32明确实现、独立入口和当次执行配对一并匹配。先完成代数CPU合同，再每端真实loader8批M0、非零梯度/冻结基线/严格重载，才开始完整50轮。每端新目录，不续训历史中途权重，不自动重试。四卡按实际空闲分配，240秒队列观察；预计约4–6 GPU小时加M0/验证，以实际首轮修正。沿用现有tri_reid环境、不安装依赖，保存原故障记录及基线权重。

旧210份运行源不修改；新module/entry/collector/queue/check/plan单独绑定，checkpoint显式保存归一化、architecture、protocol与baseline SHA，拒绝模式串用。相同数据集两端初始state及参数量必须一致。保留六端退出、50轮history、梯度、best重载和三输出距离的CPU复算。

## 预登记后继门与诊断

只有competitive−independent在三个数据集mAP均为正、R1均不降，且201/MSVR mAP均至少+0.5pp，才登记独立global-only/多seed；失败不改门、不扫温度/系数/seed。原Patch支持门继续失败。

完整端后按前序定义检查query/gallery槽位内容及注意力相似性、有效Patch数、首位修复/新增、身份AP分布、best→末轮。需要同时改善检索，不能将注意力更分散当成功。global/local是同模型输出分解；single研发seed和已消费官方集不构成未选择泛化/SOTA/三角色必要性证据。整体研究目标仍未达到。
