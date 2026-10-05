# 近似同容量语义来源对照

状态：PREPARED_NOT_FORMALLY_TRAINED。仅新增三端，不重跑已封存 raw controls；Goal ACTIVE / UNMET。

## 唯一问题与解释边界

在原作者 raw global/role 分工目标下，新增读取器的作用是否需要原生图像信息，还是额外语义变换容量已经能解释它？前一轮 joint-L2 role metric 未晋级，不能在该目标上继续救分。本轮恢复的是既有已封存 raw 目标，不改变其历史结果。

主张上限：比较原生 CNN 来源与额外语义来源的完整检索结果。近似参数量不等于 FLOPs、源分辨率、梯度路径或表示分布完全匹配；即使原生胜出也不能据此证明唯一因果根源、独创新颖性、多种子稳定性或 SOTA。

## 最小设计

保持原128-token语义读取、16区域、CNN→Transformer→Mamba桥接、512候选的额外独立 Q/K/V 读取、唯一零初始化输出出口、1536维部署向量。仅将额外读取器的图像 CNN 来源换成当前 CNN-role 语义 patch，经主动参与计算的128→202→202→128 MLP后双线性插值为512候选；两个图像方向分别对应16×8→32×16、8×16→16×32。

原生 stem：93,248参数；新 MLP：93,048参数。完整读取器159,296对159,096，共14个实际参数张量，差200（约0.126%）。不挂闲置参数凑容量。宽度202来自静态容量计算，不读取官方分数选择宽度。

| 数据集 | 新训练端 | 复用参照 | 首要差值 |
|---|---|---|---|
| RGBNT201 | semantic_capacity | raw native / raw semantic / independent global-only | capacity−native、capacity−semantic |
| MSVR310 | semantic_capacity | 同上 | 同上 |
| RGBNT100 | semantic_capacity | 同上 | 同上 |

控制来自 `refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json` 的9行、187文件。新入口沿用现有内部 `native` factory slot，但 initializer/condition/manifest 明示 `semantic_capacity` 来源，不能误标为原生实验。

## 固定执行合同

公开CLIP、新camera/head、seed42、作者来源各数据集配方、完整50轮、每轮原合法全图库评价，同一mAP-best附带全部CMC；当前raw global/role两目标、角色输入stop-gradient、普通raw模态Triplet保持。只用2026物理GPU0/1，单进程两卡分段、完整author batch；无功率/温度动作。无新loss、LR/margin/gain/seed扫描、外部数据、N2/N3、重计算/确定性工程修复。

新CPU组件检查两个方向、初始零出口、8步累计有限非零梯度、全部参数真实更新及严格重载。正式端开始前做真实完整author batch初始化：未改state与旧native逐项一致、原native初始化摘要匹配封存控制、raw/L2/global/logits初始输出相同。随后各自一次8步M0，仍要求全部可训练张量梯度支持、visual/camera更新、BN计数8、optimizer恰好覆盖及原重载容差。新M0仅使用实际159,096参数预算，不放宽任何支持或数值条件。

M0失败端不正式训练，不按分数/梯度日志换种子、倍率或学习率；旧失败保留。完整训练后第一次严格独立评价通过，才封存并删除该端M0 probe，仅保留best及评价证据。

空间预算：三份best＋最多一份live probe，各384MiB，另保留2GiB；入口事前至少3,758,096,384字节。实际磁盘仍由运行阶段检查，不认为其他项目空间恒定。只退役已完整消费且不在187依赖中的诊断数组，保留必要best、原始正式距离和失败证据。

## 报告与停止规则

三端均正式完成才能称3/3。新模型对raw native、raw semantic、独立global-only全部九个配对均报告mAP/R1（201附R5/R10）、完整50轮轨迹、全部query差异/首位修复和新增错误/身份宏平均、训练成本；实际training batch顺序必须逐字节相同。项目推进线沿用ΔmAP≥0.5且ΔR1≥0，不能替代统计显著性。

若语义容量对照与native相近或更好，原生内容必要性主张不成立；若二者都不超过semantic/global，应停止把更多读取容量当答案。结果未收齐时只报实际已完成项，不补分、不调用已退役探针验证器，不中途堆N2/N3救分。
