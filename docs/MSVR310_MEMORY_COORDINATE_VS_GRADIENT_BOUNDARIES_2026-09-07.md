# MSVR310历史候选：坐标刷新、反传范围与anchor范围

2026-09-07，依据当前固定执行源码ab67d4c和原始论文核查。完整来源测量仍运行；本文不增加训练任务、不选参数，也不把预检观察外推为完整检索结论。

## 当前实现实际比较什么

`tools/msvr_instance_memory.py:40–69`中，当前batch距离为`cdist(unit, unit)`，历史距离为`cdist(unit, memory.detach())`。因此当前batch既作anchor，也作当前正负候选，二者都可参与反传；历史候选不参与反传。

`tools/msvr_freshness_probe.py:63–85`通过当前encoder重算同一历史视图，但在`no_grad`中执行，结果仍detach。新旧历史坐标使用同一候选集合、同一当前参数状态、同一当前batch。`parameter_probe`比较的是这两种坐标下实际encoder参数梯度，并重复同一旧loss作为数值噪声参照。实际optimizer仍使用已注册batch/stale规则，fresh loss从未更新参数。

所以此前报告中“当前anchor计算图”应完整理解为**当前整个batch计算图，包含当前peers的梯度**，不能理解为每条距离只有anchor一端可导。历史一端始终停止梯度。

## 四个不同的实验轴

| 设计 | 候选集合 | 历史坐标 | 历史样本反传 | 作为loss anchor的样本 |
|---|---|---|---|---|
| 原batch-hard control | 当前batch | 无历史 | 无 | 当前64条 |
| 已封存实例记忆V1 | 当前batch＋合法历史 | 保存时的旧编码 | 否 | 当前64条 |
| 当前测量中的fresh对照 | 与stale完全相同 | 当前参数重新编码 | 否，仅测量 | 当前64条 |
| 假设中的历史反传 | 可保持同一集合 | 当前编码 | 是，尚未实现/验证 | 仍可仅当前64条 |
| 所有候选同时作anchor | 同一集合也可以 | 取决于实现 | 取决于实现 | 当前＋历史，目标权重改变 |

后二行不是已登记的新实验。不能将“历史坐标刷新”“历史参数梯度恢复”“增加anchor覆盖”一次改动后，只将效果归因于缓存新鲜度。停止历史梯度是已有记忆方法的计算取舍，本次没有据此认定原实现有bug。

## 链式法则对应的差别

记当前batch的单位表示为`U(theta)`，同一历史视图在当前参数下的表示为`V(theta)`；`L(U,V)`仍只对当前64个anchor求平均。按所采用的自动微分子梯度，在相同表示数值处：

```text
g_fresh_detached = J_U^T · dL/dU
g_fresh_both     = J_U^T · dL/dU + J_V^T · dL/dV
```

`dL/dU`已经包含当前样本作为anchor及当前候选的贡献。当前测量比较`g_stale_detached`和`g_fresh_detached`；没有测得第二项`J_V^T · dL/dV`。其大小、方向以及加入后对泛化是否有帮助均未知。

即使补上第二项，也只是同一个“64个anchor、扩展候选”目标的完整参数梯度；并不自动等价于让全部历史候选也作anchor的对称大batch Triplet。这里是普通链式法则与本项目接口解释，不声称新定理。

## GradCache的已有方法边界

Gao等的《Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup》发表于RepL4NLP 2021（ACL workshop，非ACL主会）。方法先无图编码全部分块，计算并保存表示梯度，再逐块重算编码器、累积参数梯度，最后进行一次优化更新。其目的是降低编码器激活存储需求，表示与损失计算本身仍有批大小开销。[原论文§3.2–3.3](https://aclanthology.org/2021.repl4nlp-1.31.pdf)

作者实现核到提交`906f03835fbc183132a9db32612a9e8f180ca3b4`。`build_cache`保存表示梯度，`forward_backward`通过表示与已缓存梯度的内积进行反传；`RandContext`恢复首遍前向的CPU/GPU随机状态。代码的optimizer步骤在缓存函数外执行。本项目没有安装或复制该实现。[作者源码](https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/grad_cache.py)、[随机状态实现](https://github.com/luyug/GradCache/blob/906f03835fbc183132a9db32612a9e8f180ca3b4/src/grad_cache/context_managers.py)

因此，未来若需要分块恢复历史样本梯度，应明确引用已有GradCache思想；仅“缓存冻结字段＋分块反传”不足以宣称原创。XBM/XBN/AXBN对应的历史坐标近似与矩匹配边界另见`MSVR310_MEMORY_DRIFT_PRIOR_ART_2026-09-07.md`。

## 本项目的具体移植约束与当前决定

当前`SignalPreservingExpertFormationV8`在fusion之后通过七个BN neck计算分类logits（`modeling/trifusion/signal_preserving_v8.py:607–650`）；度量损失使用fusion表示（同文件721–739）。而本次历史重编码只调用encoder/fusion，跳过分类neck。这是从当前代码确认的接口区别。

由此推断：如果未来把整个`return_aux=True`前向直接套到两遍分块编码上，额外BN运行及不同batch统计可能改变训练定义。必须验证真实输入、随机状态、buffers、AMP尺度、归一化和参数更新时机的一致性，不能只凭普通链式法则宣称与本项目大batch逐位等价。此风险来自实际存在的BN路径，不是在本次运行中新增防御代码。

当前先完成固定六端1560步来源测量及全CPU。如果新旧坐标差异接近重复反传噪声，应降低坐标刷新优先级；如果差异明显，只能确认历史坐标影响训练信号，是否改善泛化仍需另行固定配对实验。历史样本反传和增加anchor覆盖不在本次合同内。不能用本文增加新模块，或将新的来源终点送入已消费官方测试选参。
