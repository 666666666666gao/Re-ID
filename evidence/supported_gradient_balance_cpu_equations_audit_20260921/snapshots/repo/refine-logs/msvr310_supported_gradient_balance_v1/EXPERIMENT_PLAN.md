# MSVR310 支持感知的角色内梯度平衡：固定配对合同

日期：2026-09-21。实现修订R2：DIRECT_AUXILIARY_PREPARATION_NOT_RUN。R1执行92a75e4已T0通过，但M0第4步辅助梯度减法近似误差0.006824872超过预先登记0.005，原队列停止；仅3次更新，无Q1。原R1源码、合同、配置及失败原文完整封存。R2保持同一科学假设、全部超参数及门槛，每步直接求辅助导数；同族独立复审通过后从原固定初始化重新执行T0/M0，不续训失败的3步。仅seed42。

## 问题、依据与唯一假设

现有跨场景 AP 配对 fused 增益 +0.567183 mAP，三个 fold 非负，但 CNN/Transformer 下降，身份 bootstrap 下界 -0.077239，按原合同未晋级。完整结果参见 results/MSVR310_CROSS_SCENE_SMOOTH_AP_Q1_2026-09-21.md。全量独立审查已完成，WARN/CLOSED_WITH_LIMITS，确定性复核通过，边界见原审计。

此前标准 Smooth-AP 固定终点诊断发现排名梯度通常小于其他目标的合成梯度，且多数方向不冲突。该诊断不是本轮 cross-scene 模型的梯度比例，更不是 AdamW 实际更新份额。当前 cross-scene 训练保存的 total_vs_history 比较的是“当前总梯度与历史排名梯度”，不能把它读成排名/辅助梯度比。

唯一待验证假设：在跨场景关系、完整历史导数和推理结构固定时，按各角色完整排名/辅助梯度的有支持移动统计进行温和、有界的量级平衡，能否改善未知身份检索，并保留原角色能力。不是已经证明辅助目标导致失败，不预设三个角色都缺少排名作用。

最低支持仍是原两组科学门全部通过；来源损失降低、梯度比例接近或控制器运行正常均不能代替检索。若失败，只否定此固定优化规则，不扫描倍率、EMA、终点或重复救回旧版本。

## 两端与不变量

- control：原跨场景 Smooth-AP，完整当前/历史排名导数与其他监督直接相加。
- balanced：同一个标量目标与关系集合，仅在三个 encoder 角色参数块上改变排名/辅助梯度的组合系数。
- 两端均从同一折 source-only Signal 与 seed42 角色初始化独立训练，不能从上一轮 epoch20 继续。
- 原 V8 稠密三角色、7680D固定融合、冻结 Signal/tail、七组分类/度量监督、tau=0.01、scene 掩码及 eligible-anchor 均值全部不变。
- 其余13项不是13个独立目标同时作用于每个角色。单角色的直接辅助依赖为 fused ID、自己完整分支 ID/Triplet、自己残差 ID/Triplet，共5项；仍将原其余13项的加权总和定义为辅助目标，不改变其内部权重。
- B64/K8、512唯一历史记录、最大年龄8、65步预热、三折各两端20epoch/260更新、当前64anchor/历史0anchor、当前坐标重编码和完整历史VJP保持。
- 两端都计算同样的分解与控制器统计，只有候选端实际应用新系数。不能让一端省略测量、另一端承担额外反传后宣称等计算。
- 分类 neck/head 的14个可训练张量保持原当前batch总目标梯度；189个 encoder 张量按原 cnn_/transformer_/mamba_ 前缀分组。检查三个集合互斥且覆盖所有 encoder 参数。冻结参数不参与调节。

## 精确候选规则（固定常量，不从 Q1 扫描）

角色 e 的完整排名梯度 R_e = R_current,e + R_history,e，辅助梯度 A_e 来自原其余13项，仅当前batch。两者均在 AMP unscaled 的真实量级上统计，范数是角色全部 encoder 张量共同的 L2 范数，不逐张量各自归一化。

前65个训练步骤两端保留原 hard Triplet 更新，控制器不更新统计、不应用系数。M0 沿用2步预热。

AP 生效且 n_valid > 0 时，对每个角色使用：

```
G_R = ||R_e||_2
G_A = ||A_e||_2
首个有支持步骤：EMA_R = G_R, EMA_A = G_A
以后有支持步骤：EMA_R = 0.9 * EMA_R + 0.1 * G_R
                   EMA_A = 0.9 * EMA_A + 0.1 * G_A
r = clip(sqrt((EMA_A + 1e-12) / (EMA_R + 1e-12)), 0.25, 4.0)
w_R = 2*r/(1+r)
w_A = 2/(1+r)
候选 encoder 梯度 = w_R * R_e + w_A * A_e
```

系数范围均为[0.4,1.6]，和为2；不能把这解释为角色梯度范数或 AdamW 步长不变。平方根使响应比直接逆范数温和。对称范围不预设排名总是较弱。epsilon 只定义零范数处的表达式，系数始终有界；不将小梯度自动解释为有价值的新信号。

已实测存在4个全批无跨scene正例步骤。因此 n_valid == 0 时，两端均按原辅助目标正常更新，原图连接零排名保留；控制器 EMA 不更新，实际系数为1/1。不能用该零值更新“排名收敛”统计，也不构造假正例、改变候选或跳过优化器。

若有支持但 R_e 为零，仍按上述连续有界式处理并明确记录零梯度；不把它伪装成无支持。无额外阈值、按身份标签定制、损失学习速度估计或部署 Router。

这些是普通范数统计与组合工具，不单独主张原创；也不是完整 GradNorm、MMPareto 或 OGM-GE 复现。没有加入方向投影、噪声或额外损失。不同角色系数通常不等价于某个统一标量加权损失，方法应称为参数块梯度调节。

## 当前/历史分解及实际更新

1. 当前前向、候选重编码、三个 AP/hard 统计沿用现实现。
2. 对当前实际排名项计算 AMP-scaled 当前 encoder 导数，再对原其余13项直接求辅助 encoder 导数，保留当前图；普通总损失 backward 仍产生所有 encoder 与分类头梯度。两端同样增加辅助求导，不增加优化器更新。
3. 实际A_current使用对其余13项直接求导。R1实测表明combined backward减rank与direct auxiliary不完全相等，差异不只归因于最后一次FP32相减，混合精度反传的累加路径也可能贡献。原相减值保留subtraction_auxiliary_vs_direct诊断，不再作为实际A或近似等式门。M0另外独立调用一次辅助求导作为参考，不能alias本步A到参考。
4. 释放当前图后，沿用原历史随机状态/分组重放，恢复完整 R_history。历史组的零上游跳过已存在，不作为新设计。
5. 合并完整 R 后才统计和组合。不能先调当前排名、再追加未加权历史 VJP。
6. control实际保留原“当前总梯度＋历史梯度”，balanced使用完整R与直接A的上述组合，头部梯度逐位保留。有限精度下direct R+A不强制等于原combined梯度，新增direct_sum_vs_original量化差异；CPU仅在balanced有支持步核验加权direct R/A范数，控制/无支持步核验逐位保持原sum。
7. 只执行一次 scaler.unscale、finite检查、AdamW step/update。统计与系数无计算图；不改 AdamW 动量或二阶矩。保留 RNG、buffer、重编码一致性检查。

逐步记录：支持 anchor/身份/scene 数、R/A 范数与余弦、current/history 排名范数及余弦、EMA前后状态、系数、组合梯度范数、AMP尺度、实际角色参数更新范数、更新前参数范数、lr、溢出及全部14项。实际更新范数是含AdamW动量/预条件/衰减的观测，不分摊成排名贡献比例。

## 工程与科学验证

T0（远端张量运算，本地仅代码/文本）：固定小例验证权重范围、EMA初始化/只在支持步更新、无支持原梯度、分组覆盖、角色交换等变、AMP缩放不改变统计与系数、历史必须纳入R、头部梯度不变。数学测试不表示真实模型性能。

M0：仍三折两端8更新及fold0两端100固定batch，共248更新。203/203累计非零、0overflow、冻结路径不变、精简checkpoint严格重载全部输出、原校正overfit比≤0.1。

在六个容量端首个实际历史组，保留原直接完整图/VJP核验，并直接核验current R、历史R、full R、当前直接A与另一独立辅助求导参考，角色相对误差阈值仍≤0.005；零参考要求绝对误差≤1e-8。候选实际组合与direct_rank/direct_aux加权参考比较；control实际组合与原full direct_loss参考比较，不能把它强行当作分开反传的R+A。原R1减法误差保留，门不放宽；任一实际参考核验失败仍停止工程阶段，不修改科学门。

M0 CPU 全量重算 AP、掩码、支持数、保存的范数统计→EMA→权重及应用账本。CPU不声称从标量恢复参数梯度；实际梯度见证范围须说明。通过后才能开始完整Q1。

Q1：三折两端完整1560更新，固定epoch20，完整图库；保留全部fold/query/身份/角色，Rank-1修复和新增错误、全部120个训练epoch账本。没有逐epoch heldout选模。终态CPU核验后独立审查。

沿用原 paired 与 Signal 两组各五门：fused增益≥1pp、每折≥0、三个角色≥0、身份bootstrap下界>0、candidate fused严格高于对应基线及各角色；10000次身份bootstrap/seed42。两组全通过才晋级。既有检索结果不更名为消融，不挑匹配控制。

## 资源、来源及执行前置

上一轮完整Q1约3.71小时，R2两端均多一次当前排名及一次直接辅助导数，初估M0 15–30分钟、Q1 4–6小时，以实际容量测量修订执行时间估计，不修订训练长度。记录额外反传次数、历史前向数、显存和墙钟时间；新增推理参数0不等于训练无成本。

远端模型/图像/数组保持不下载。沿用tri_reid/RTX3090，不装新环境。启动前实查GPU与空间；至少4GiB可用、最大新增3GiB作为当前预算。保留初始化、终点和复核数组，不因失败删权重。

来源：evidence/support_aware_gradient_neighbours_20260921/PRIMARY_SOURCE_NOTE.md、SOURCE_RECEIPT.json、PROJECT_OBJECTIVE_SCOPE.md。MMPareto/GradNorm/OGM-GE及其限制沿用已核对原文/作者代码，不复制无许可证代码，不宣称新颖性已经成立。

上一轮科学审查已闭环；R1静态审查通过但真实M0按门停止。R2直接辅助修订须独立复审并同步固定源码/配置后，从原初始化执行T0/M0。R2尚未训练，R1没有检索结果。主结果前不做消融、多seed或官方测试。全项目Goal保持ACTIVE/UNMET。


## 实现文件与记录边界

控制器 tools/msvr_supported_gradient_balance.py；两端训练器 tools/train_msvr_supported_gradient_balance.py；T0数学 tools/check_msvr_supported_gradient_balance_math.py 及完整来源合同 tools/check_msvr_supported_gradient_balance.py；CPU tools/verify_msvr_supported_gradient_balance.py 和统计重算 tools/verify_msvr_supported_gradient_balance_stats.py；持久队列 tools/run_msvr_supported_gradient_balance.py。cross-scene及更早前置源码保持字节不变；本新实验R1原字节已封存，R2按本节明确修订。

R2每步额外1次当前排名和1次直接辅助参数求导；每个容量端首历史组另有4次直接分量求导（当前排名、历史排名、完整排名、辅助），以及沿用的直接总梯度检查。这些是工程测量成本，两端一致。当前/历史求导只针对189个encoder张量，分类头14张量保留原backward结果。


## R1失败与R2修订范围

R1运行 /root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_seed42_92a75e4，05:36:03 M0退出1，原wrapper STOPPED_AT_M0。fold0 control第4步辅助参考first_norm0.3899933414205923、second_norm0.3900544174096418、difference0.0026616547754863508、cos0.9999767264664654、相对误差0.006824872357540746，超过0.005。T0通过不代表真实混合精度求导等价。

只将实际辅助向量的取得方式改为直接求导；原控制更新路径、科学公式/常量、源关系、历史VJP、optimizer与全部门槛保持。额外开销两端一致并记录current_auxiliary_backward_calls。原减法与direct-sum偏差持续记录，没有把失败抹掉或放宽真实reference门。有限精度下两端除有支持组合系数外，还存在direct分量与combined反传累加的差别，不能声称逐位只差一个标量；同一数学梯度组合的实现修正不构成新科学结果。
