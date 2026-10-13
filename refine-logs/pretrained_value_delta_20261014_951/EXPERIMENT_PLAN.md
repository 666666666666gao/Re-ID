# 实际预训练value上的角色增量：两端新训练

问题来自§949/950：处理后128维区域的独立mAP很弱，原CLIP末层实际QK-output经LN/proj的512维内容在同一固定best上具有72.8013局部mAP；原G74.2967未改变。这支持检验value形成/使用路径，不证明唯一压缩根因，也不构成强教师或最终融合增益。

借ClearCLIP把全局与dense形成路径分开的思想，保留原CLIP/G计算；根据本项目实测采用实际QK输出，而不是照搬其QQ方案。无额外模型、文本、掩码、预训练权重或库。

新角色职责：CNN/T/M仍生成16×128区域，但其实际输出用于选择预训练512维value，不再经自由384→1536读出生成修正。现有128维query/key投影产生实际角色读取权重；以相同位置/value计算参考读取；权重之差读取原512维value。3角色/16槽位固定均值，各模态512维按原G顺序拼接。原自由readout的589824参数删除，无闲置容量；训练参数比原raw semantic少589824、张量少1，不称完全同容量原架构消融。

两端仅参考不同：semantic_reference用原语义读取后的区域生成参考query；uniform_reference用128位置均匀权重。两端状态、参数量、初始化、采样、损失、gain初值、value来源和输出维度相同。两者的权重差逐行总和近0，只保证固定权重下对value公共加项不敏感，不能称去除全部风格、保证互补或不掉分。均匀参考也不冒充独立global。

value由原末层实际attention输出经原LN/proj、detach取得；global按原作者任务更新；角色读取stages/context/G也detach，raw角色任务只更新roles与gain。沿用已完成global-role职责分离、原raw作者头/CE+softmargin Triplet、seed42、RGBNT201 B64/K8、完整50轮和唯一mAP-best，全部CMC跟同权重。两端新公开CLIP初始化，不继承§950 E8权重；不改loss/head/采样/学习率/精度/输出维度，不扩车辆或seed。

每端prepare验证共享/视觉/camera/head和原raw semantic相同及两端全state相同，真实8步M0先确认全部需要训练的参数活动、effective updates8、角色到G无梯度、实际attention_change接受角色任务梯度、BN8与state重载一致。通过后才fresh50和首次strict。没有通过的端保留缺失/原失败，不临时救seed/gain或改阈值。

主要比较semantic_reference−uniform_reference，以及各端−独立global_only、原raw semantic和全Signal。原raw semantic与新模块同时有value、读出和参数量变化；不得归因单一因素。推进线仍ΔmAP≥0.5且R1不下降，不等于统计显著性。报告全部尝试、完整曲线、best→末轮、首位修复/新增错误及身份AP变化。

只26物理0/1、单个NN进程使用两卡，串行2端。warm环境不变、不查温度/功率、不安装/重建或抢占。预计每端约40–50分钟；M0节点≥180秒，其后按近ETA一次观察，必要轮询180–300秒。保存best和必要依赖；入场前只清理已D归档SHA/大小验证的关闭实验远端重复，保留历史receipt并明确恢复义务。

来源与有效性边界：标准attention/参考相减/预训练坐标不是新发明；项目待证明的是实际value上的角色选择是否产生已有G之外有用身份关系。不是旧LMF重新训练、原生CNN追加、末端gain调整或未经证明的late-interaction教师。三数据集超过Signal目标ACTIVE_UNMET。
