**MSVR310 跨场景 Smooth-AP 实现静态代码审查**

日期：2026-09-21。结论：**PASS_WITH_LIMITS / NO_BLOCKING_STATIC_DEFECT_FOUND**。未发现需要阻止注册 T0、M0 的具体代码缺陷，未提出源代码补丁。此结论仅表示所读实现与注册定义相符；T0、M0、M0_CPU 尚须实际运行通过，不能把本报告作为工程通过、Q1 晋级或新检索成绩。

审查者 canonical ID：`/root/review_cross_scene_smooth_ap_20260921`。由父代理确认的实际 spawn 请求为 `model=gpt-6-astra`、`reasoning_effort=max`、`fork_turns=none`。这些是调用请求参数，未独立认证后端型号。审查为 `review_independence: same-family`、`acceptance_status: provisional`。

工作树：`C:/Users/gb/.trifusion_github_publish_22c3bee`。观察 HEAD 为 `d32845ad975bd2de96eb1e9d4a45043aeb65f03c`；新代码当时尚在工作树，结论针对所读文件内容。配置为 `configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json`，所读 SHA256 为 `e5326b52ebb12dced24ebfac788db2e2bb5bdca0b50c6c4dec64e596f1af05a1`。

合同为 `refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_PLAN.md`，记录表实际名为 `EXPERIMENT_TRACKER.md`；任务中的 `TRACKER.md` 已解析为此现存文件。审查中执行方补入了 T0 对已归档来源支持逐批比对，已按最终磁盘版本重读。新 T0 文件 SHA 为 `6ec509da5dd11138a7852ea1281ad5cc5fcb8f33203d5b2907a3cdcf1d3ed564`。

**BLOCKING 发现：无。**

所有六个新增 Python 文件、配置、合同和执行记录已逐行阅读；训练器、CPU 验证器、T0 和外层运行器与原登记 Smooth-AP 文件做了差异比对。只执行文本/JSON 读取、Git 状态/差异及现有 SHA 合同核对，没有运行 Python、公式测试、张量、模型、数据图像或远程命令，没有启动实验，没有修改源码。

本地现有绑定核对为：新配置 9/9 项匹配；原 Smooth-AP 配置登记的 10/10 项匹配；归档 source-support 的 3/3 个输入绑定匹配。原六个 Smooth-AP 脚本及其配置 Git 差异为空。这是核对现有不变量，没有增加新散列体系；远端递归依赖、checkpoint 和环境绑定未在本审查中执行。

**确认正确的实现行为**

| 检查点 | 静态结论与证据 |
|---|---|
| 唯一干预 | control 选择原 all-identity Smooth-AP，cross_scene 选择新跨场景 AP，两个端点仅 fused 度量项的关系与 reduction 不同。见训练器 127–160 行。 |
| 正负关系与排名 | `same_id & ~same_scene` 定义正例；`~(same_id & same_scene)` 定义有效候选；同身份同场景完全忽略，其他身份即使同场景也保留。每个正例自身比较剔除，排名的起始 1 保留，分数为 `1-d²/2`、tau=0.01。当前 self 自然属于同身份同场景。见目标文件 31–51 行。 |
| eligible reduction | 仅 `counts>0` 的当前 anchor 进入等权平均。无正例 anchor 的 AP=0 仅作存储占位，不进入均值；没有移除其作为其他 anchor 候选的列。见目标文件 36–54 行。 |
| 全批零合法 anchor | 返回 `distance.sum()*0`，保留 current/history 距离图连接，数学导数为零。训练器只替换 `triplet_fused`，其余 13 项、backward、optimizer 更新和 active 队列入队继续执行。历史 upstream 为零的组可略过 VJP，最终总更新仍执行。见目标文件 53 行及训练器 159–161、190–248、266 行。 |
| 三个调用位置一致 | 当前主损失（129 行）、历史 leaf partial（154–157 行）和 direct 完整图（175–180 行）都调用相同 `objectives` 并传入同一身份与 scene；端点选择一致。不存在某一路残留 hard 控制目标或错误跨场景定义。 |
| 当前与历史梯度 | 当前距离由同一 current feature 同时作为行、列计算，保留两侧导数。历史 leaf 计算只固定当前坐标，随后以已保存角色入口 RNG、当前参数重编码历史字段，累积 candidate VJP 后按既定 weight=1 加入缩放梯度，一次 unscale/update。历史只有候选列，没有历史 anchor。见训练器 149–157、199–248 行及沿用的 `fused_distances`、`refresh_all`、`encode_graph`。 |
| 历史新鲜度与位置合同 | 队列保持唯一历史记录、排除当前记录副本、512 容量和 8 步年龄；当前重复采样视图仍是独立位置。重编码输出与 fresh values 逐项断言一致，RNG 和 buffer 保持。见原 `msvr_instance_memory.py` 8–37 行、`probe_msvr_role_set_gradients.py` 29–50 行及训练器。 |
| 固定设置与成对性 | fresh seed42 角色初始化及 source-only Signal 初始化沿用原路径；B64/K8、AdamW、增强、20 epoch/260 更新、调度和匹配像素未改。comparison 前 65 更新为原批内 hard；零基 step65 开始 AP 和入队，历史最早 step66 参与。训练记录由 protocol 的真实 identity/scene 获取，没有把 scene 换成 camera。 |
| T0 定义覆盖 | 新公式检查包含独立标量、有限差分、位置置换、忽略项零导数、无正例 anchor 仍作负候选、全零 eligible 图连接、历史候选导数及 tie 算术；原 standard 检查继续调用。新 T0 对全部 780 来源批次比较历史数量、跨场景正例总数、eligible 数及零批次标记，绑定已归档输入。见数学检查 12–90 行、T0 18–65 行。此处描述测试代码，未执行。 |
| 实际日志与 CPU 目标重算 | 每步保存四空间距离、真实 ID/scene、队列位置、standard/cross AP、合法正例 counts、eligible 数、14 项标量及实际目标名称。NumPy 独立重算 hard、standard 与 cross AP/条件均值、队列和账本；AP/损失容差 2e-6。见训练器 132–139、186–264 行，验证器 17–193 行。 |
| M0 与 heldout 次序 | wrapper 顺序为 T0→M0→M0_CPU→Q1→Q1_CPU，任何非零退出停止。Q1 trainer 在 gallery forward 前核对成功 M0、M0_CPU 和同配置绑定。M0 为三折两端各 8 更新以及 fold0 两端各 100 更新，共 248；保留 203/203 累计非零、无 overflow、Signal 冻结、严格重载、原过拟合比≤0.1，以及各 capacity 端首个真实历史组 direct/VJP≤0.005。见运行器 31–65 行、训练器 305–381 行。 |
| GT 与原科学门 | 提取和评价沿用真实 dataset identity/scene，不把模型输出当标签；完整图库保留其他身份干扰项。配对组与 candidate 对 Signal 组的原五项门槛全部保留，只有两组均通过才晋级。身份 bootstrap 仍为 seed42/10000；固定 epoch20、三折和五输出全部进入结果。见原 `evaluate`、`paired_summary` 及新验证器 238–340 行。 |

**NONBLOCKING 边界**

- 审查是新上下文同系列审查，接受状态 provisional；不能描述为跨模型系列独立确认。
- 本审查没有执行任何公式测试、M0 或 CPU 数值验证。既有文件绑定匹配无法证明远端可运行、显存/磁盘满足、FP32/AMP 数值一致或参数梯度正确。
- 保存距离的 CPU 验证独立重算 fused 三种目标及记录的梯度统计关系。四空间距离均被保存并检查形状/有限值；其余 13 项是加权标量账本核对，未从 logits/特征重新生成。逐步梯度仍是 runtime witness；每个 capacity 端只检查首个实际历史组 direct/VJP，不能扩大成全程参数梯度独立重建。
- conditional mean 是当前候选池的 eligible-anchor 均值，本干预同时改变关系与分母。将来若得到收益，不能单独归因于删除容易正例，也不能把来源候选 AP 或重复 anchor 曝光当作未知身份完整图库成绩。
- 每步 eligible/counts 已完整保留零批次信息，但 terminal `training_checks` 没有汇总这些跨场景曝光数。最终报告应从全部行汇总 active 阶段零批次；区分 control 的 cross AP 诊断值和 cross_scene 的实际优化目标。此项不需要为本审查新增训练逻辑。

**必须完成的实际检查**

1. **T0**：在指定远端环境执行所有已写合成公式、有限差分、掩码、置换和零梯度检查，以及全 780 来源队列行检查；通过现有递归输入绑定。不能将本静态审查记为 T0 PASS。
2. **M0**：完整执行 248 个注册 source 更新，核验匹配初始化/像素、首历史组 direct/VJP≤0.005、历史角色贡献、累计 203/203 非零、有限梯度、无 overflow、冻结 Signal、严格重载、容量和两个固定 100 步过拟合比≤0.1。不能用中间较好步替代终点。
3. **M0_CPU**：完整重算所有 248 行距离、GT 掩码/counts、hard/standard/cross AP、14 项权重账本、队列与 checkpoint 证据。M0 与同配置 CPU 证明全部通过前，不进入 Q1。
4. **条件成立后的完整 Q1/CPU/审计**：六个 fresh fold/endpoint 模型固定 epoch20，共 1560 更新，每个端点保留 600 query/1032 gallery，两个端点合计 2064 gallery record forwards；完整保留五输出、全部 fold/query/identity、mAP/Rank-1 和负结果。归档来源队列预期每端预热后 eligible 曝光为 5144/5176/5056，零合法 batch 为 1/1/2；这是当前定义的支持约束，不是性能结果。实际 selected cross_scene 零损失分支须从 active 记录核对，其余监督和更新仍完成。最终两组原五项门与独立终态审计照常执行。

本审查仅写入指定独立审查目录中的 Markdown/JSON。没有训练、远程执行、源码修改或历史实验重启。

背景记忆仅用于定位既有 worktree 与保持协议边界；旧记忆中的诊断运行状态未用作当前状态证据。当前结论基于本次读取的代码、合同和登记文件。
