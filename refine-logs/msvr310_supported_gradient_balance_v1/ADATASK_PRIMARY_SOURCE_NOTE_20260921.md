# AdaTask 原文核查与有支持 R/A 状态候选边界

日期：2026-09-21。性质：有界文献与源码核查；未实现优化器、未运行模型或实验、未接触远端训练。

**结论：按任务分开 Adam 一阶和二阶矩已有 AdaTask 先例；按跨场景正例支持冻结排名状态/时钟并禁止无支持时应用旧排名动量，是本项目待验证的适配定义。该区别既不能证明 shared AdamW 已导致当前失败，也不能单凭未在本篇出现就证明新颖性。**

## 1. 原始来源与核查范围

- 论文：Yang 等，*AdaTask: A Task-aware Adaptive Learning Rate Approach to Multi-task Learning*，AAAI 2023；核查 [arXiv v2 PDF](https://arxiv.org/pdf/2211.15055v2)，版本日期 2023-05-18，重点为第4–5页 Eq. 7 / Algorithm 2、第6页 Table 1，以及第10页 Appendix B（页码按 PDF 顺序）。
- 作者仓库：[EnnengYang/AdaTask](https://github.com/EnnengYang/AdaTask)。只读查询 HEAD 并读取固定提交 `84853de15dedda72d65629c8cedfcd55cbb720cf` 下的 [adatask.py](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py) 与 [main_cityscapes.py](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/main_cityscapes.py)。没有执行作者代码；该发布快照不等于已验证论文训练时使用的原始提交。
- 项目定义来源：[当前固定合同](EXPERIMENT_PLAN.md)、[当前本地工作记录](EXPERIMENT_TRACKER.md)，以及本次提出的后继候选要求。本地记录标记 seed42 R2 Q1_RUNNING；本次没有刷新远端状态。合同早期段落的 NOT_RUN 为历史文本，不应覆盖工作记录的后续状态。

## 2. 论文已定义的 Adam 规则

用常见 Adam 记号重写 Algorithm 2，令 `beta1` 对应论文 `gamma2`，`beta2` 对应论文 `gamma1`，`v` 对应论文 `G`。每个共享参数、每个任务 k 各有独立的 m、v：

```text
m[k,t] = beta1 * m[k,t-1] + (1-beta1) * g[k,t]
v[k,t] = beta2 * v[k,t-1] + (1-beta2) * g[k,t]^2
m_hat[k,t] = m[k,t] / (1-beta1^t)
v_hat[k,t] = v[k,t] / (1-beta2^t)
delta[k,t] = -eta * m_hat[k,t] / (sqrt(v_hat[k,t]) + eps)
theta[t+1] = theta[t] + sum_k delta[k,t]
```

这里先分任务预条件再求和；没有额外除以任务数。Algorithm 2 显式包含 bias correction，每步遍历所有任务，指数用共同 t；未定义按“合法训练支持事件数”推进的任务时钟，也未列出无支持/缺失任务分支。论文算法未给出 AdamW 解耦权重衰减项。[论文 Algorithm 2](https://arxiv.org/pdf/2211.15055v2#page=5)

作者实现可更具体地裁定以下问题：

| 项目 | 固定作者代码实际行为 | 对移植的含义 |
|---|---|---|
| 任务状态 | 每个共享参数保存 `exp_avg_t` / `exp_avg_sq_t` | 分开一阶矩和二阶矩已有明确先例 |
| 时钟 | 每个参数仅一个 `state['step']`，任务循环外加1 | 不是每任务支持事件时钟 |
| Bias correction | 两个 beta 均用该共享 step | 缺失任务后单独计时须另外定义 |
| 更新组合 | 已收集各任务梯度，再循环 `p.addcdiv_` | 数学上相加，不在每次参数变化后重新求任务梯度；浮点加法顺序仍有影响 |
| 任务权重 | `task_weight[i] * losses[i]` 后求导 | 权重进入矩估计；不是预条件之后再乘权重 |
| 私有参数 | 对加权总损失求导，使用普通单套 m/v | 不能说作者对所有参数都无条件拆 K 套状态 |

依据：[状态、计步与更新，第64–112行](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py#L64)；[梯度获取与任务权重，第42–55行](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py#L42)。

**权重衰减：** 构造器接受 `weight_decay=0` 并放入 defaults，但 step 中未读取它，既没有向梯度加入 L2 项，也没有对参数作解耦衰减。故不能仅看参数名就说此代码实现 AdamW，也不能说传非零值即可生效。Cityscapes 入口未传 weight_decay，采用默认0；论文结果不能为本项目非零衰减的迁移提供直接验证。[构造器](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py#L10)、[完整 step](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py#L58)、[Cityscapes 优化器创建](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/main_cityscapes.py#L58)。

## 3. 缺失任务、零梯度和旧动量

作者代码没有支持计数输入。数值零张量照常进入矩更新、共享 step 照常前进；因而从其代码直接推出：若旧 m 非零，本步 g=0 仍有 `m_new=beta1*m_old`，并可能形成非零参数更新。这里是公式推论，非已运行数值测试。[作者更新循环](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py#L76)

收集器允许把缺失 `.grad` 记为 None，但 step 不检查 None，直接传给 `add_` / `addcmul_`；因此这不是可用的“缺失任务跳过”实现。并且清梯度函数将已有 grad 清零而非设为 None，某参数未被后续任务使用时还可能留下数值零槽位。**autograd 的 None、零张量和数据层面的无合法正例不是同一件事。** 对 None 路径的判断来自静态代码，未执行触发错误。[第30–48行](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py#L30)、[第81–89行](https://github.com/EnnengYang/AdaTask/blob/84853de15dedda72d65629c8cedfcd55cbb720cf/adatask.py#L81)

本轮核查未发现原文或这份实现有“缺失支持则冻结 m/v/任务时钟，同时不应用旧任务动量”的完整规则。这是有界否定：不是声称其他论文、作者未发布实现或更广泛优化器文献从未有过。

## 4. 用户 Cityscapes 表核对

以下数值与 Table 1 完全一致。Task A 为深度估计，Task B 为语义分割；主实验为 MTAN + Adam。论文报告三随机种子均值；Appendix B 指定200 epochs、初始学习率0.0001、100 epoch减半、batch8、最后10个epoch测试指标平均。此协议与 TriFusion 固定终点检索不同。[论文 Table 1 与 Appendix B](https://arxiv.org/pdf/2211.15055v2)

| 方法 | 深度 Abs Err ↓ | 深度 Rel Err ↓ | 分割 mIoU ↑ | 分割 Pix Acc ↑ |
|---|---:|---:|---:|---:|
| EqualWeight | 0.0152 | 47.00 | 75.01 | 93.40 |
| GradNorm | 0.0145 | 44.16 | 75.18 | 93.38 |
| AdaTask | 0.0128 | 36.85 | 75.02 | 93.40 |

这些数字支持“该设置深度误差下降、分割基本维持”的描述；AdaTask 的 mIoU 仍低于 GradNorm，不能写成每项均优。这里只核对论文报告，不是复现，也不能外推成 MSVR310 收益预测。

## 5. 当前 R2 与后继候选分别是什么

当前合同定义每角色 `R = R_current + R_history`，`A` 为其他原有辅助目标的**直接导数**。全局其余13项不等于每角色有13个直接辅助依赖；按现行合同，单角色实际直接依赖5项：fused ID、自己完整分支 ID/Triplet、自己残差 ID/Triplet。在有支持步骤，用角色块范数 EMA（0.9）得到有界组合系数，再交给同一个 AdamW；不拆其一阶/二阶矩。无支持时冻结的是范数控制器 EMA，优化器仍按原辅助路径执行。故“控制器 EMA 冻结”不等于“rank optimizer 状态冻结”。这些是当前固定合同的定义，不是本次新改动。[当前合同第21、28–64行](EXPERIMENT_PLAN.md)

后继候选只可暂述为 **AdaTask 启发的、有支持 R/A 独立优化状态适配**，尚未实现或验证：

| 设计点 | 来源与状态 |
|---|---|
| 在同一角色参数上，R/A 各维护一阶和二阶矩，分开预条件再组合 | AdaTask 已有核心思想；R/A 是本项目目标分组 |
| R 必须含 current + history 完整导数；A 直接求导 | 本项目既有合同，不因换优化器而缩减 |
| 无合法跨scene正例时，rank m/v/支持时钟全部保持不变，rank更新严格为0 | 本次候选规定；不能仅喂入零梯度代替 |
| 有合法支持但 rank梯度为0时，照常观测、推进rank时钟并按矩公式更新 | 本次候选对“观测到零”与“未观测”的明确区分；可能仍应用历史rank动量 |
| 无支持时 auxiliary继续更新；每次实际参数更新只作一次解耦decay | 本次候选的 AdamW 适配规则；作者发布代码未实现该decay |

要形成可审查的后继合同，还必须明确：是否保留当前有界系数、系数在矩前还是矩后、任务时钟和全局LR调度时钟的区别、头部/私有参数策略，以及有限精度组合路径。上述选择均没有在本笔记中擅自决定。

## 6. 仍需诊断，不能用文献代替的证据

1. **当前因果判断未成立。** 小 R/A 原始范数比不直接等于 AdamW 实际更新贡献比。加权后同一二阶矩含混合梯度平方的交叉项，不能从两项范数和一个总更新范数唯一反推出各任务贡献。本地合同也已明确这一测量边界。
2. **若后续获准诊断，需状态与向量证据。** 固定同一 checkpoint、batch、支持掩码、完整 R/A 和真实优化器状态，才可比较现行更新与预先定义的候选更新；分开记录方向、量级、旧动量和衰减。仅在该冻结点的反事实运算仍只是局部机制诊断，不是长期检索收益的因果证明。该诊断本次没有运行。
3. **稀疏支持规则有现实动机，但收益未知。** 当前合同已经区分无支持与有支持零梯度；将该区别延伸到 optimizer状态有明确语义，仍可能改变有效学习速度、历史遗忘与总步长，必须靠将来固定协议验证。
4. **创新边界仍开放。** 分任务 m/v 不能再主张为新；“本篇未给出”的支持时钟/跳过规则，尚需独立的相关工作检索，不能据此宣布方法创新成立。
5. **执行边界保持。** 当前 seed42 R2 Q1 按原合同完成并统一核验；本笔记不授权中途换优化器、读中间成绩择法、改门槛或启动消融/新训练。

本次唯一产物是本笔记。未复制作者实现到项目代码，未修改现行训练源码、配置、合同或跟踪状态，未提交或推送。
