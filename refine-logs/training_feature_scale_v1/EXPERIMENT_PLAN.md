# F2：训练特征尺度单干预对照

准备状态，无M0或正式成绩。F1完整六端已完成；F1是多项差异同时变化的基础配方包，不能直接认定L2训练归一化造成全部差距。本阶段只分离一个明确训练边界，不是新研究模块或SOTA结果。

## 两条件与合同

三个数据集各normalized/raw两个fresh端，共六项正式训练。双方使用F1-current的无角色/无adapter/无SIM/AlignM纯公开CLIP、fresh camera/一组BN和分类头，完全相同完整初始state、1536维及参数量。唯一差别是送入BN分类头和margin0.3 Triplet的特征：normalized先整体L2，raw保留原始值。它同时改变CE/BN输入和Triplet几何，不能进一步声称只分离了Triplet。

推理始终整体L21536维、平方欧氏距离、无rerank/TTT、完整原query/gallery与camera/time过滤，包括MSVR全部103个gallery-only干扰身份。无新文本、掩码、DINO、角色、细节CNN、蒸馏、loss公式、router或memory。

seed42/full50；三集B64/K8、workers4、当前共享几何增强、AdamW weight_decay1e-4、视觉base_lr5e-6/其余3.5e-4、5轮warmup及原50轮cosine、AMP scale256。每项取同一mAP-best权重，later ties，全部CMC跟随同一权重；保留原1e-5个百分点严格重载门，不因失分改变配置。

## 执行与可复核性

仅2026 GPU0–3，最多四个独立单卡任务，沿用现有conda/data。旧F1留2025且不重跑；本批normalized也是fresh当前对照，不用跨主机旧F1替代。新protocol仅将dataset_root明确指向2026，所有其他metadata及行顺序与旧固定记录相同，保留原protocol不修改。

六个初始化见证比较完整state及visual/camera/current头，实测初始正式forward一致；每条件八个真实batch M0有限梯度、所有实际训练参数活动、visual/camera变化、冻结参数不变与全state重载通过后，正式六端重新从公开初值开始。六项M0全部有效后才开正式阶段，不能从M0权重继续训练。

所有实际formal batch的身份、camera、RGB文件名与顺序逐项保存，六端完成时比较同数据集两条件全部顺序。记录训练特征范数、CE/Triplet、所有实际LR、step/epoch轨迹、显存、完整state/训练时best距离和严格reload距离。原F1分析器遗漏绑定的真实问题，已要求本批事前纳入tools/analyze_correspondence_distances.py；不回填旧249manifest。失败保留、不减batch、不改seed、不改门或盲重试。

## 科学判断

全部六端full50/strict评价完成后唯一CPU报告；raw-normalized每数据集事前推进门仍为>=0.5mAP且R1不下降，三集分别判断，不四舍五入或用身份宏平均替代官方mAP。正式受控差值只证明这一个训练尺度边界在当前配方下的作用，不能宣称解释整个F1包、角色必要性、多seed稳定性或SOTA。若无有效增量，停止此方向，不扫margin或倍率救分。

整批训练与逐轮评价估计约5–8 GPU小时、四卡wall约2–3小时；M0/初始化和终态CPU另外计。依据F1-current已观测耗时作估计，不作终态证据。控制器240秒轮询，人工按预计里程碑/180–300秒观察。官方数据已用于研发与epoch选点，单42，固定身份bootstrap不代表训练种子方差。

## 当前入口

- tools/run_training_feature_scale.py：仅改变训练特征尺度，复用封存F1循环。
- tools/queue_training_feature_scale.py：六M0先行、六full50后行，唯一报告。
- tools/report_training_feature_scale.py：真实GT/完整图库配对及全部batch顺序审计。

实现/source review/M0/registration/launch需记录真实状态；本文件不是已经执行的证据。

## 2026目标目录校正（§41.741）

实际只读检查发现旧F1证据目录未纳入Git稀疏检出；已恢复并持久加入检出范围，旧manifest字节保持不变。F1/249三份canonical协议使用25的数据根；26继续保留原SEALED_SOURCE243的三份协议。F2只对这三个明确已知差异使用原243哈希，其余246份旧源码仍逐字节验证；新F2协议的metadata/行顺序与26canonical相同，canonical不覆盖。此项只是目标来源校正，不改变训练/math、数据划分或门槛。

41,360条唯一引用图像路径存在，MSVR全部103个gallery-only身份保留；这是路径存在性检查，不是图像字节复核或模型执行。源码审查续接仍未形成完整结论；当前四卡另有DeMo-DualAxis任务，待审查完成、容量再次实查可用后才能初始化及M0，不抢占、不新增25训练。
