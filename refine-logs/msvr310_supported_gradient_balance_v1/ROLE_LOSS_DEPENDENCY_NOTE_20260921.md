# R2 单角色损失依赖与优化器边界

2026-09-21。静态代码核查；不修改正在运行的 R2，不增加 GPU 诊断，不读取部分检索成绩。

## 已核对的依赖

当前执行提交为 `1381639f778f77f124a2726ee55c07610092a438`。在原 V8 平行角色结构中，三条角色的可训练 stage/head 参数彼此独立；共享 tail 参数冻结。各自残差只进入自己的完整分支和残差分类头，同时三者一起进入 fused 表示。

本轮已逐字比较以下四个本地文件与该执行提交的 Git blob，全部一致；未执行新张量试验：

| 文件 | SHA256 |
| --- | --- |
| `modeling/trifusion/signal_preserving_v8.py` | `97a7b5fe6dab882c2ed92ed1db95b9a8f48eae1d99a3a9b145b0cdfc4b8cefbc` |
| `tools/run_signal_preserving_v5.py` | `e162184f68778b4991db0f97f26c5fda273b2ad2f7c8db2bbb2d53775eb717e5` |
| `tools/train_msvr_supported_gradient_balance.py` | `3ab635f97e08edb439c352a742189a0f82cd8f76d73bd1a34ab1a7eb57da5187` |
| `tools/msvr_supported_gradient_balance.py` | `27ae6eb74a83d405eabb4a30a60156226d9dc93e70d78847cf24b0f57bd774c0` |

全模型有七组 ID/metric 共十四项监督。将 fused metric 替换为当前跨 scene Smooth-AP 后，其余十三项仍存在，但对任意单个角色编码器参数块 `encoder.<role>_*`，直接连接的辅助标量仅为以下五项：

| 辅助标量 | 对该角色的依赖 |
| --- | --- |
| `id_fused` | 通过 fused 表示连接全部三个角色 |
| `id_<role>` | 通过该角色的 Signal＋残差完整分支 |
| `triplet_<role>` | 通过该角色完整分支 |
| `id_residual_<role>` | 通过该角色纯残差 |
| `triplet_residual_<role>` | 通过该角色纯残差 |

另两个角色的八项局部监督没有到这个角色编码器的计算路径。共享冻结 tail 接受各自输入的反传，并不因此把另一角色的 loss 连到本角色可训练参数。上述是结构依赖，不表示五项在每一步都非零，也不表示可由项数推导梯度占比。

代码依据：

- [角色各自 stages、heads 和前向](../../modeling/trifusion/signal_preserving_v8.py)：293–340、391–463 行。
- [固定拼接与分类接口](../../modeling/trifusion/signal_preserving_v8.py)：483–518、640–656 行。
- [十四项监督定义](../../modeling/trifusion/signal_preserving_v8.py)：690–743 行；[监督加权](../../tools/run_signal_preserving_v5.py)：99–135 行。

## 当前实现及解释

R2 对全体十三项辅助标量的加权和直接求导，再按角色参数分组；这与仅求该角色五项的数学依赖一致，并没有把其他角色梯度跨坐标相加。没有必要为文字修正改写训练器。

- [训练器](../../tools/train_msvr_supported_gradient_balance.py)：74–86 行选择 189 个 encoder 参数张量、14 个分类 neck/head 张量，建立一套 AdamW；175–180 行直接求当前辅助梯度；267–295 行组合、保留头部梯度、执行一次实际更新。
- [角色内组合](../../tools/msvr_supported_gradient_balance.py)：65–96 行。排名先合并当前与历史候选导数，再使用支持条件下的有界 EMA 系数。控制端和无支持步骤保留原组合路径。

当前三个角色各有两项范数 EMA，但 AdamW 仍对每个参数维护**合成梯度**的一阶、二阶矩。它不是排名/辅助任务各自一套优化状态，也不是完整 GradNorm 或 MMPareto。

固定终态的排名/辅助原始梯度范数比不能解释为 AdamW 更新份额。R2 保存的实际参数变化范数也包含已有动量、二阶矩及权重衰减，不能直接分摊到当前排名与辅助目标。

## 后继边界

用户提出的 AdaTask 类任务独立状态、排名有效任务时钟，先做原文核查并保留为候选。当前 R2 六端、CPU 终态及审查完成之前，不改其目标、优化器、EMA、温度或训练长度，不依据已完成的局部端点选择下一方法。

即使未来采用任务状态分离，也需要明确：完整当前＋历史排名导数先合并；无支持与支持但零梯度不同；任务 bias correction 的时间轴；任务更新求和还是平均；权重衰减一次；分类头原路径；额外状态内存与实际步幅。上述是候选需要登记的定义，不是已经证明共享 AdamW 状态导致当前失分。

## 终点保存能支持什么分析

本轮补查实际保存入口：训练器403行调用 `train_msvr310_source_style.checkpoint`，后者164–174行只保存角色模型状态、冻结基线别名、绑定信息、身份划分和配置SHA，没有保存 AdamW 的 `state_dict`、一/二阶矩或 AMP scaler 状态。训练器另存的 `gradient_balance_state` 是范数EMA控制器状态，不是优化器状态；逐步 `actual_parameter_updates` 是比较统计，不是完整参数更新向量。

因此，当前合同产物支持固定终点检索重载和已登记的梯度/参数变化统计，但**不能仅凭这些终点文件精确复盘原训练的 AdamW 历史状态或任务更新分量**。这是保存代码的静态核查，不是本轮额外加载远端checkpoint。最终文件内容仍由原CPU与终态审查核对。

不在正在执行的R2中追加状态保存、重跑已完成端点或宣称可恢复未保存的状态。若后继假设需要真实优化器历史的反事实比较，应在新的来源侧流程中预先登记状态捕获；新建空优化器或从终点继续若干步得到的状态，不等于原轨迹状态。该限制不改变当前R2检索合同及其完整结果的有效性。
