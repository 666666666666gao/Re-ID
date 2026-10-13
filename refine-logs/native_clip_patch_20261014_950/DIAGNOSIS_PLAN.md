# 原CLIP patch形成路径：固定权重一次诊断

目的：§949已证明当前16槽位的局部比较不构成强教师，不能继续以更复杂比较器救分。先测原CLIP输出是否保留不同的身份关系；不预设压缩、背景或某个算子是原因。

只用26物理GPU0/1、原warm conda、RGBNT201 joint_delta的原E8 best。原三端full50/首次strict/CPU报告已终态；不重跑旧训练或M0。source488和原权重、初始化、评价元数据保持。

一次原backbone完整query/gallery编码，在同一前向中读取三种候选：原最终LN/预训练投影后的128×512 patch；末层实际QK注意力输出经原LN/proj；以相同末层输入、Q/V/输出参数计算QQ注意力后经原LN/proj。后两者不含最终残差/FFN，也绕过最后外部角色adapter的snapshot，不能声称仅更改attention因果对照或完整ClearCLIP复现。global原前向不改。

每组相同token归一化、三模态平均、完整合法gallery/camera过滤；对照全部patch-pair cosine均值与双向MaxSim。没有global融合系数、test拟合、重排子库、teacher学习或参数更新。512维/128token比旧128维/16槽位具有不同容量，不能唯一归因投影或读出。

实际接入要求：CLS与原G一致；源/输入SHA前后相同；模型state前后相同、grad空；global距离与上次固定best差≤1e-5，四项指标和作者评价一致。保留每种距离、逐query AP及身份变化；大特征只在内存记录shape/bytes/SHA，不写约4GB缓存。

来源：[ClearCLIP](https://arxiv.org/abs/2407.12442)，[作者开源仓库](https://github.com/mc-lan/ClearCLIP)，固定commit ad68a404d55d48d27330b93554eb64a234ff717f的custom_attn。QQ数学实现是标准attention，保留其dense读取启发，不下载权重、安装库或照搬分割框架。API403保留，随后git固定commit读取原文件；现有两项CPU fixture与作者函数差0，仅组件数学核验，不等于完整模型通过。

预计编码约一分钟、完整矩阵比较另需数分钟。原调用保存REQUEST/PID/EXIT，首个结果观察在180–300秒节点；若传输失败，先核原调用结果，不重启NN。失败不改容差、不换best/seed/mode；需修订时另命名记录。

决策：若原patch或辅助输出可用，再为训练设计保留原global、实际value受任务监督的机制与匹配控制；若仍弱，不能假定其为强教师或再做融合倍率搜索。全部是diagnostic local-only，原正式成绩不变、三数据集超过Signal目标ACTIVE_UNMET。
