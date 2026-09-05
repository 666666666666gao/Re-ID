# V19候选：专家私有语义尾部（设计草案，尚未冻结或训练）

本文件记录V18之后的一个表征方向，不能作为已完成实现、已通过M0或已获检索
增益的证据。V18保持Q1_FAIL，其全部结果和科学条件不变。

## 来自当前代码和完整实验的动机

V18完整六端重放确认模块生效，但21身份中仍6个负收益；CNN新增Rank1错误的
负例平均靠近0.024874，远大于正例靠近0.002015。后续需要同时保留身份区分
与跨视角鲁棒性，不能只增加被删除的方向数或调整方向估计。

当前V8/V12专家在CLIP第8索引block之后分支，却在三个后续stage共用同一套
冻结CLIP尾部：`modeling/trifusion/signal_preserving_v8.py`的
`PretrainedTailTriExpertEncoder.forward`在每个stage将相同tail_block用于全部
expert；可训练CNN/Transformer/Mamba模块只能在这些冻结语义变换之间调整token。
V17/V18又将整个V8专家源冻结，仅训练其后的向量修正头。这是实际参数约束，
尚不能仅凭代码证明它就是唯一性能原因。

候选主假设：让每个专家的语义尾部根据自身局部/全局/跨模态角色学习，可产生
比共享冻结尾部更有身份区分能力的表征，同时保留完整exact Signal路径。

## 最小架构范围

- 继续从同一V12三折的source-only Signal和V8 checkpoint初始化；不重新训练
  Signal，不下载新backbone、数据描述、mask或外部模型。
- 为CNN/Transformer/Mamba各复制CLIP索引9/10/11三个block。原Signal模块保持
  冻结且独立；三专家保持现有角色模块、head和等能量追加银行，完整3072D
  Signal前缀不变。
- 远端fold0 Signal checkpoint的实际state核得每block7,087,872个参数，共12个
  tensor，均为attn/ln1/mlp/ln2，无额外闲置adapter参数。三专家共9个block，
  增加63,790,848个参数；这是一项显著容量变化，必须在比较中明确披露。
- 主对照与实验端都实例化相同私有block副本，从同一state开始；对照冻结其
  尾部，实验端训练其尾部。CNN/Transformer/Mamba原有角色模块和分类头采用
  相同训练策略，区分冻结与可训练尾部带来的结果。
- 不叠加V18投影、V17包络、Router、HFER、reranking或测试时训练。

## 实现前仍需冻结的细节与执行顺序

先完成真实builder、参数/内存核查以及Loss/optimizer分组的静态合同，再固定
一次主实验方案。需要为预训练尾部明确唯一学习率，记录与新增角色模块的分组；
不能在检索结果之后扫描学习率、冻结层数或epoch。公共V8/V17/V18已封存代码
优先保持不变，用新的wrapper/builder承载修改。

M0应证明初始检索路径/增强batch配对、完整Signal前缀、所有实际训练tensor梯度、
B64/K8容量、固定batch过拟合及无overflow。私有tail参数不与Signal共享storage，
训练不能更改原Signal或被冻结的对照tail。原V18记录22/22不能照抄到本版本。

M0通过后按三折×两端、seed42、完整20epoch终点进行主比较；全部3126 gallery、
571 query、五路输出都必须报告。继续使用已消费的训练内OOF开发资格，不声称
新的独立验证集。候选晋级条件沿用+1.0mAP、各fold/专家非负、identity-bootstrap
下界>0、fused胜过同checkpoint Signal及各专家；实施时须在启动前正式冻结。
未通过训练内条件不得做D1或访问30-dev/official。所有训练/推理只在远端GPU。

当前状态：草案；无V19代码、M0、训练、结果或晋级回执。

后续更新（2026-09-05）：本草案已落实为独立V19 wrapper/runner；冻结执行
合同见`refine-logs/v19/EXPERIMENT_PLAN.md`及其时间戳副本，当前状态以主交接
§35和V19 tracker为准。本段保留先前草案状态，不能作为实验结果。
