# V19 私有语义尾部主实验方案（2026-09-05，执行前冻结）

本方案仅检验一个新的表征假设；V18的Q1_FAIL及其全部条件保持不变。
依据为V18全571-query诊断、全21身份结果和实际V8共享冻结尾部代码。
本方案不声称已证明共享尾部是唯一原因，也不预先声称有收益或创新性。

## 主张与比较

主假设：让CNN局部细节、Transformer全局CLS、Mamba空间/跨模态角色各自
学习其后续语义变换，能改善完全身份隔离条件下的身份区分能力。最低证据是
对匹配冻结尾部对照通过下述完整Q1门，而不是只看均值或个别身份。

比较两端均从同一fold的V12 Signal与V8 expert最终checkpoint严格恢复，均
实例化三专家×CLIP索引9/10/11的九个私有副本。角色模块与分类头均继续训练。
唯一训练结构差别是frozen_private_tail冻结私有尾部，trained_private_tail
训练私有尾部；两端初始参数、输出、采样和增强相同。原Signal全部参数始终
冻结；完整3072D direct+SIM及camera SIE不变，baseline-only输出必需。

这增加63,790,848个可训练参数（108个tensor），并非等可训练参数量比较。
两端总实例参数量相同、训练参数量不同。正结果最多支持该固定完整训练结构
的收益，不能单凭此比较把收益归于角色分工或排除容量效应。用户要求主实验
达到SOTA后再做消融；当前不运行冻结层数、宽度、学习率、epoch或种子矩阵。

## 网络与初始化

- CLIP第8索引block后取得冻结anchor token，原Signal继续其完整路径产生
  冻结reference与baseline。每专家使用自己的9/10/11索引block，交替使用
  已有CNN/Transformer/Mamba角色模块，最后保持原角色head、BN及分类头。
- 原CNN水平局部池化、Transformer CLS、Mamba跨模态空间池化及专家减去
  frozen reference的定义不变。每模态512D、每专家1536D；完整融合为
  3072D Signal + 4608D等能量残差银行，单专家输出为3072+1536D。
- 不叠加V18投影、DTRED、Router、HFER、mask、文本、外部backbone、reranking
  或测试时训练。V19 module与runner独立，既有封存源码不重构。
- 仅使用V12三折中相应94-source身份训练出的Signal/expert初始化；六个源
  checkpoint及V12汇总均按config SHA验证，payload身份集合必须精确一致。
  对source权重的继续训练不构成新baseline重训。

## 固定训练合同

配置：`configs/RGBNT201/TriFusion-signal-preserving-v19-private-tail-rtx3090.yml`。
启动时同时验证本方案和配置的外部SHA，并记录实际runner/module/依赖源码SHA。

| 项目 | 冻结值 |
|---|---|
| seed | 42；单卡RTX3090 |
| 实际batch | B64/K8，无梯度累积 |
| 数据 | 固定141-fit registry；每fold94-source/47-heldout，完全路径隔离 |
| sampler/增强 | 现有CrossCameraIdentitySampler、SharedGeometryTripletTransform，workers4 |
| 两端预算 | 各20完整epoch；无中途检索、无best选择，只评价最终checkpoint |
| optimizer | AdamW，weight_decay=0.0001 |
| 角色模块和分类头LR | 0.00035 |
| 私有预训练尾部LR | 0.0000035，固定为角色LR的1/100，不扫描 |
| scheduler | 沿用V8 learning_rate_multiplier：5epoch线性warmup、其后cosine |
| AMP | FP16，初始GradScaler256；gradient checkpointing沿用V8 |
| CuDNN | deterministic=true，benchmark=false；两端相同 |
| eval batch | 128，原256×128 eval resize/normalize |

私有尾部使用较低LR是保护已训练语义参数的预先选择，并非已验证最优值。
V18历史重训出现相同序列而非位级结果一致，旧seed helper同时开启benchmark；
V19在其自身seed入口明确关闭benchmark，历史代码/结果不修改。此设置不
自动构成全CUDA算子位级确定性保证；实际初始输出与增强hash仍需逐项核验。

损失原样采用V8的七路ID/Triplet权重：fused ID0.25/Triplet1；
三个branch分别ID1/12、Triplet0.25；三个residual分别ID1/12、Triplet0.25。
margin0.3，label smoothing0.1，完全沿用V8归一化batch-hard定义。
ID权重和为0.75，因此M0解析平滑下界为0.75H，而非V18的2H。

## M0：训练内工程检查

1. 三折两端各取固定前8个真实B64/K8 source batch。原V8与V19所有五路输出
   必须精确一致，初始state SHA、输出SHA和增强receipt两端逐项相等。
   所有batch均有跨camera正例和身份负例；原Signal前缀精确，eval状态不变。
2. 所有九个私有block与原Signal、彼此之间均无storage共享；私有尾部必须
   实测63,790,848参数/108 tensor。实验端与对照的训练tensor数差为108；
   角色及head的绝对训练tensor数由真实builder列出，不照抄旧版本计数。
3. fold0分别运行两端8个不同source batch的真实训练步，记录显存和完整
   非零梯度覆盖；所有实际训练tensor必须在8步中收到有限非零梯度，
   overflow0，峰值reserved<24576MiB。对照尾部不得改变，实验端必须改变；
   全部冻结参数/Signal buffer保持SHA一致。
4. 新初始化实验端在固定source batch完整100步，固定两组base LR、无warmup。
   全训练梯度覆盖、overflow0、冻结state不变；以末步和首步计算
   (loss_final - 0.75H)/(loss_initial - 0.75H) <= 0.1。
   M0状态不进入Q1，正式训练每端从源checkpoint重新构建。

M0失败则保存失败并停止Q1；实现错误可根据具体报错修复，不能通过换学习率、
换batch/worker/RNG或放宽科学/过拟合条件将真实负结果改成通过。

## Q1：全量身份OOF资格比较

M0通过后串行执行三折×两端×20epoch，预计3360步。每端保存完整非baseline
state（含BN buffer和冻结/训练私有尾部），同时绑定不可变Signal source。
重新构建并strict reload，重载完整state SHA必须等于实际最后训练state。
两端完整sample-order SHA和前8增强batch、初始state SHA均必须相同。

每fold全部47-heldout身份保留为gallery，包括单camera身份干扰项。仅无
跨camera正例的query不进入指标分母：共3126 gallery records、571合法query、
21个跨camera身份、2555条仅排除于query分母的记录。不得把gallery筛成21身份。
全部五路baseline/fused/CNN/Transformer/Mamba报告mAP、Rank1/5/10及逐query
AP/rank，不使用跨fold特征距离。汇总按全部571 query加权。

固定科学晋级条件必须全部满足：

- 实验端相对本次匹配对照的fused aggregate mAP增益 >=1.0个百分点；
- 三个fold的fused mAP增益均 >=0；
- CNN、Transformer、Mamba各自aggregate mAP增益均 >=0；
- paired query AP差按身份聚类bootstrap10000次、seed42，95%下界>0；
- 实验端fused aggregate mAP严格高于同checkpoint baseline及三个expert。

继续复用既已消费的训练内OOF开发资格，不能称新的独立验证或官方结果。
bootstrap输入显式float64；报告完整21身份，不删负收益身份或择优分支。
若任一科学条件失败则封存，不做D1/dev/official，也不扫描尾部层数或LR。
全部端完成后才能作主实验结论；单fold/端中间值只说明进度。

## 运行顺序、成本与后续

| 阶段 | 内容 | 条件 | 初始估计 |
|---|---|---|---|
| T0 | 远端定向单元测试：初始输出/重载/独立storage/真实更新范围 | 全部通过 | <2分钟 |
| M0 | 三折配对、两端8步、实验端100步 | 上述所有检查 | 5–15分钟，按实测速率修正 |
| Q1 | 三折两端完整20epoch与终态全gallery评价 | M0通过自动执行 | 2–4 GPU小时，按M0/首epoch修正 |
| 终态审计 | 全部查询/来源/范围/路径与检查点核验 | Q1完整终态 | 结果后执行 |
| D1 | 141-fit refit后只访问一次30-dev终态 | 仅Q1通过，另行固定实施合同 | 本runner不执行 |
| official/跨数据集/消融 | 达到固定开发门后的主比较 | 后续合格模型 | 当前不执行 |

当前30-dev可部署最好为V8 Phase-B58.4050/59.3939，exact Signal58.0109/57.4545；
65开发门未过，官方SOTA目标仍未实现。RGBNT100和MSVR310仅安装校验，未训练
或检索。Q1数字不能与外部官方表格直接比较。

启动前要求GPU至少22000MiB空闲，远端现有约10GB磁盘可保存预计约2–3GB的
六个最终非baseline checkpoint及JSON；数据与权重仍只在服务器。单次进度
采样通常间隔180–300秒；已获得稳定epoch时长后按预估里程碑检查。
