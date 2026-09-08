# MSVR310 封存来源角色关系覆盖：独立完整审计

审计生成时间：2026-09-08T09:55:06.483001+00:00。整体结论 **WARN**；原始数组和全部结果字段的确定性核验 **PASS**；新方法有效性/未知身份收益 **本分析未检验**。

本次审计由独立上下文的原生 Codex 审阅任务直接读取原始路径完成，canonical task 为 `/root/audit_msvr_role_relation_coverage`。请求路由是 `gpt-6-astra`、独立字段 `max`、`fork_turns=none`。当前界面没有提供可独立核验的实际后端模型/effort attestation 或 agent UUID，故这些 observed/UUID 字段保留 null，不将请求路由伪写成实际后端证明。语义审阅明确保留 **same-family / provisional**；可独立复现的确定性文件/数值核查另记为 accepted deterministic evidence。没有使用其他模型后端。

执行范围严格只读：本审阅没有修改科学源文件或远端文件，没有模型前向、反向、更新、checkpoint 保存/载入、图像读取、训练、包安装或新方法登记。所有新审计脚本、完整原始输出、快照、哈希和本文只写入本审计目录。传入的连接恢复脚本仅在内存命名空间静默执行，其文本和凭据未复制进审计产物。

## 核心结论和实际完成证据

完整重建 **54 条件结果、37,152 个 query-condition-protocol 行**，覆盖 **27 条件 receipt、108 个 NPY 数组、1,597,847,040 字节**。逐行递归检查了 **1,030,549 个非浮点标量、58,752 个浮点标量、125,388 个字典结构及 176,256 个列表结构**。离散字段无差异；58,469 个浮点值直接相等，其余 283 个的最大差为 **1.5543122344752192e-15**，远低于固定比较容差 1e-12。不是仅对 saved summary/hash 做相等性检查。独立脚本还对实际被选极值/并集做了 **286,131 次直接向量差欧氏距离见证**，最大差 **1.8873791418627661e-15**。证据：`deterministic_verification.json:2440`、`deterministic_verification.json:2438`、`deterministic_verification.json:2429`。

原 wrapper 在 **2026-09-08 17:20:55.044692 +08:00** 启动，**17:21:07.884876 +08:00** 结束，exit **0**；wrapper 耗时 **12.837832 秒**，分析 summary 记录 **12.540962 秒**。本次实时核查原 PID **31867/31871 均不存在**，日志完整从 2 到 54 条件。执行提交是 `c46be4eb6bf8aa096527abb9f45b59282fc09085`；审计读取时 HEAD 已前进至 `7c69636b6deea894c981a1416572a22b82e2aed1`，但 `git show` 核验执行提交中的分析器、wrapper、合同和计划与当前远端及本地字节一致。旧来源提取绑定 `4e57e542e22c895ac7af96c01357b6a49e4739e4`，相关历史源码和补充特征映射源码也与其提交字节一致。证据：`local_snapshots/intake/pipeline.json:2`、`local_snapshots/intake/pipeline.json:18`、`local_snapshots/intake/pipeline.json:20`、`remote_inventory.json:167`、`remote_inventory.json:591`，完整 commit/blob 结果在 `remote_inventory.json` 和 `supplementary_input_hashes.json`。

远端新结果目录递归枚举实际只有以下四个文件，均与本地 intake 清单、实际本地文件和本次再次远端读取一致：

| 文件 | 实际字节 | SHA-256 |
|---|---:|---|
| `pipeline.json` | 794 | `56bbcd103b5611b89498ee84ec3f6a1362fb878fc703a86de72f1674f53d84fd` |
| `analysis.log` | 1,976 | `58114e036232cf7e426f6d08c912ba49350e31a3279dccf212240f835ecc9480` |
| `analysis/summary.json` | 87,606 | `f1d98d2e8615e71ca6223c632b16fc9bb80d163e796c4a39b74b924d83468f40` |
| `analysis/all_query_relations.jsonl` | 29,971,182 | `09365b53e73de38c8e9b8f60b393c2f4b73b2d21d71fabda7517a5ec469347c4` |

分析合同 SHA-256 为 `c52c86e49bde1a73d2ea75b4182c4ae5db1c1911e820c3e30f9eba19fe3dd051`。旧来源 summary/CPU verification/pipeline、协议、分析器、计划和 wrapper 的七项合同输入绑定全部通过；27 receipt 内容与旧 summary 条件逐字段一致，135 个选定 receipt/array 输入条目与新 summary 的输入集合恰好相同。独立脚本总计读取并对 **177 个直接远端输入前后重哈希**，没有发生变化；另核对了 15 份本地/远端主文本和 5 份补充源码。证据：`configs/MSVR310/Role-relation-coverage-v1.json:15-22`、`independent_replay.py`、`actual_remote_input_hashes.json`、`supplementary_input_hashes.json`。

## A–F 实验完整性检查

### A. ground_truth_provenance — PASS

从既有 bounding_box_train 标签清单独立核对全部 1032 个 record 的 identity/scene/camera/三模态路径，并重建 155 身份、60 跨场景身份和 95 单场景身份的固定三折划分。

两种正例 mask 和真实不同身份负例均由标签产生，无模型输出生成 GT；仅源身份参与当前距离。官方评估器不是这项来源关系描述的适用终点。

证据：`tools/audit_vehicle_query_protocol_labels.py:13-18`；`tools/build_msvr310_train_oof_protocol.py:12-55`；`protocols/msvr310_train_oof_v1.json:4-8`；`tools/analyze_msvr_role_relation_coverage.py:61-77`；`independent_replay.py:100-119`。

### B. score_normalization — PASS

NPY 是 float32 单位化特征；独立复算将其转换为 float64 并再次按向量 L2 范数归一化。距离为单位向量欧氏距离，margin 固定 0.3。

未把评价指标除以本模型最大值、最小值或均值。所有计数是原始实例/查询成员计数；均值使用真实有效查询数量。

证据：`tools/diagnose_msvr_source_relations.py:43-58`；`tools/analyze_msvr_role_relation_coverage.py:46-54`；`tools/analyze_msvr_role_relation_coverage.py:67-96`；`configs/MSVR310/Role-relation-coverage-v1.json:12`。

### C. result_existence_and_completion — WARN

实际 pipeline COMPLETE_UNAUDITED_SOURCE_REANALYSIS、exit 0；分析 summary 完整，27 条日志结束在 54 条件；两个原 PID 实时均已不存在。

远端输出目录实际只有 4 个文件，全部字节/哈希与 intake 和本地原件一致；54 条件和 37152 行的所有字段均已从原数组独立复算。

审计时 tracker 仍是启动/初查 RUNNING 快照，未写完成时间和本次审计终态。它不是数字伪造，但需在本审计之后关闭文档状态。

证据：`local_snapshots/intake/pipeline.json:2`；`local_snapshots/intake/pipeline.json:18`；`local_snapshots/intake/pipeline.json:20`；`remote_inventory.json:167`；`local_snapshots/intake/analysis/summary.json:2020`；`deterministic_verification.json:2440`；`refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_TRACKER.md:3-10`。

### D. called_code — PASS

wrapper 的真实 command 指向 tools.analyze_msvr_role_relation_coverage，运行结果记录相同脚本和合同哈希。

全部当前关系计算位于该脚本 main 的实际循环中，输出全部 54 条件；已按原始数组核对，因此不是未调用的死代码结果。

旧诊断的模型提取调用是历史成本；当前分析脚本仅导入 NumPy/标准库，不调用该提取函数。

证据：`tools/run_msvr_role_relation_coverage.py:20-30`；`tools/analyze_msvr_role_relation_coverage.py:57-106`；`tools/analyze_msvr_role_relation_coverage.py:109-113`；`tools/diagnose_msvr_source_relations.py:146`；`remote_inventory.json:591`。

### E. scope — PASS

实际完整范围为 3 folds × 3 旧固定状态 × 3 固定诊断 query views × 2 mask protocols；4 输出/108 数组；单 seed 42。

1032 独特训练 record、155 独特训练身份；每条 record 在两个 source folds 出现，条件/视图/协议重复不是独立身份样本。

此完整性只覆盖已登记的封存来源分析。它不是当前 history-gradient 终点、原训练模式历史队列回放、Q1 检索结果、多种子验证或新训练终点。计划和 summary 已保留这些限制。

证据：`refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_PLAN.md:7-26`；`local_snapshots/intake/analysis/summary.json:2014`；`configs/MSVR310/Role-relation-coverage-v1.json:3-22`；`deterministic_verification.json:2282`。

### F. evaluation_type — PASS

关系标签是数据集真实身份/scene 标签；输出是已见 source 身份上的描述性嵌入几何统计。

它既不是模型生成标签的 synthetic_proxy，也不是未知身份泛化实验；real_gt 标签来源不自动授予官方性能或方法有效性结论。

证据：`protocols/msvr310_train_oof_v1.json:4-8`；`tools/analyze_msvr_role_relation_coverage.py:63-77`；`refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_PLAN.md:9-11`。

以上项目级相对路径以 `C:/Users/gb/.trifusion_github_publish_22c3bee` 为基准；为避免后续文档关闭覆盖审阅现场，完全相同的主文件原始字节同时保存在 `local_snapshots/project/` 或 `remote_snapshots/project/`。本报告中的审计目录相对路径均以本文所在目录为基准。

## 标签、候选集合、视图和状态的独立核对

标签证据 SHA 为 `c835d20478b817a54b7710463269186af2619cab3e38850534b01f3aaee6e3c8`，协议 SHA 为 `4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4`。审阅者没有调用协议 builder，而是直接由训练清单文件名解析 identity/scene/camera，对三模态相对路径逐项一致性核对，并按“跨场景/单场景身份分别排序、各自 round-robin”独立还原全部折划分和 source label map。未依赖特征、Q1 失败身份、检索排序或模型输出进行候选选择。

| Fold | source records | source identities | heldout identities（本次不用） | cross-scene 有效 source queries | cross-scene 无正例 source records |
|---|---:|---:|---:|---:|---:|
| 0 | 672 | 103 | 52 | 390 | 282 |
| 1 | 683 | 103 | 52 | 393 | 290 |
| 2 | 709 | 104 | 51 | 417 | 292 |

每条件 query 与同 fold/同 state 的 **clean gallery** 使用完全相同的 source 原始 record 顺序。identity_exclude_record 正例为同身份且非 query 自身；cross_scene 再要求 scene 不同。负例永远是所有不同身份，包含同 scene 的不同身份与单 scene 身份。全部 37,152 行都输出真实正/负候选数，未把无正例 record 从 gallery 删除。累计真实正候选成员 **260,352**，负候选成员 **25,204,140**；无跨场景正例的 gallery record 在不同查询的负集合内累计出现 **5,307,066** 次。mask 序列独立 SHA 见 `deterministic_verification.json` 的 `masks_sha256`。这些是重复成员计数，不是独特 record/identity 数。

NPY 的实际 shape 为每折记录数乘 fused 7680 或角色 4608，dtype 全部 float32、全部有限、L2 范数非零。四个输出是完整 fused 与三个 **Signal+角色** 向量；不是三个互相独立的完整 backbone，也不是 pure residual 输出。`tools/train_msvr310_trifusion_oof.py:68-73` 的映射与 `modeling/trifusion/signal_preserving_v8.py:499-514` 的拼接定义明确支持这一点。

所有 27 条件的旧模型 state hash 均分别与旧 paired source-style Q1 的 initial/control/source_style 对应状态一致；三视图同状态的 hash 相同。27 份 inputs.jsonl 的实际字节/哈希、batch 顺序、source record 完整覆盖、跨状态 pixel hashes、augmented/coupled_style pixel hashes 及额外视觉调用计数也独立核对。这里仅将已封存旧 Q1 的状态/初始化收据用作来源绑定，没有重新评判其检索结果，也没有载入 checkpoint。具体实现证据：`tools/diagnose_msvr_source_relations.py:105-181`；对应确定性断言保存在 `independent_replay.py`。

“固定诊断视图”需要精确理解：外层模型进入 eval；coupled_style 分支故意把 baseline 包装层置为 training 并强制 style plan active，使用与 augmented 匹配的输入像素，令角色 field 发生风格扰动。这不等于普通无扰动 eval，更不等于原 260 步真实训练模式/历史队列重放。`tools/diagnose_msvr_source_relations.py:126-148`、`modeling/trifusion/source_style_v27.py:65-113` 和 `tools/msvr310_exact_signal_inference.py:7-21` 明确区分外层 eval、style 分支与 no_grad 的旧提取行为。

## 全范围数值结果与精确含义

| 量（query/record 成员计数） | identity_exclude_record | cross_scene | 总计 |
|---|---:|---:|---:|
| 所有 query 行 | 18,576 | 18,576 | 37,152 |
| 有合法正例的 query | 18,576 | 10,800 | 29,376 |
| 角色负例并集成员 | 36,596 | 21,224 | 57,820 |
| 角色正例并集成员 | 31,495 | 18,493 | 49,988 |
| 额外负 record | 19,045 | 10,992 | 30,037 |
| 额外正 record | 13,203 | 7,883 | 21,086 |
| 额外负 record 与 fused 最难正例构成 active hinge | 18,320 | 10,963 | 29,283 |
| 额外负 record 与 fused 最难正例构成非正间隔 | 3,969 | 3,364 | 7,333 |
| 至少一个额外 active 负例的 query | 12,801 | 7,633 | 20,434 |
| 至少一个额外非正间隔负例的 query | 3,105 | 2,610 | 5,715 |
| 角色子集 hinge 更低 | 1,262 | 747 | 2,009 |
| 角色子集 hinge 相等 | 17,314 | 10,053 | 27,367 |
| 角色子集 hinge 更高 | 0 | 0 | 0 |
| 全集 fused hinge active | 18,203 | 10,790 | 28,993 |
| 角色并集包含两端 fused 极值 | 17,281 | 10,052 | 27,333 |
| CNN 独有负 record query | 7,646 | 4,424 | 12,070 |
| Transformer 独有负 record query | 8,176 | 4,706 | 12,882 |
| Mamba 独有负 record query | 6,964 | 4,068 | 11,032 |

表内角色 union 只按实际 record 去重；同一个负身份的不同 record 可以同时入选。“role unique negative”表示该角色的选中 record 不等于另外两个角色的选中 record；它不表示独特身份、全数据唯一错误，或该角色拥有独立参数梯度。每个角色只取一个最难正例和一个最近负例，没有 top-k 搜索或身份/scene 配额。证据：`tools/analyze_msvr_role_relation_coverage.py:81-98`，独立原始结果 `independently_recreated_query_rows.jsonl` 和 `independently_recreated_conditions.json`。

负例并集覆盖 1/2/3 个不同身份的 query 数分别为：identity_exclude_record **9,612 / 7,305 / 1,659**，cross_scene **5,568 / 4,225 / 1,007**。覆盖 1/2/3 个不同 scene 的 query 数分别为 **10,840 / 6,158 / 1,578** 与 **5,335 / 4,267 / 1,198**。两种协议有效 query 上的平均全集/子集 hinge 分别为 **0.2454884771402472 / 0.24510042224906947** 与 **0.27873004506275745 / 0.2783632678322085**；均值分母为各自 18,576 和 10,800 个有效成员，不把无正例行当成零 loss 混入。

所有 29,376 个有效 query 的 fused、CNN、Transformer、Mamba 正/负极值在本次 float64 计算中均没有精确并列；原始保存的 fused tie counts 全部为 1。实现对精确并列取 source 原顺序首个，独立重建用 Python 稳定 min/max 得到相同选择。额外 record 也没有与 fused 极值距离并列，因此在这批数组中确实是严格非极值关系。这个实测结果不允许把“将来训练不可能遇到 tie”当成假设。

### 子集不等式、标量、梯度和泛化的边界

令合法全集正/负候选为 P、N，角色提出的非空子集为 A⊆P、B⊆N，fused 距离为 d。则 max(A)≤max(P)、min(B)≥min(N)，所以 `[max(A)−min(B)+0.3]+ ≤ [max(P)−min(N)+0.3]+`。本次逐行核对为 **2,009 更低 / 27,367 相等 / 0 更高**。成立的前提是同 query、同 gallery、同参数状态/距离、同合法候选全集和同 hard-max/relu 标量公式。换候选池、缓存/梯度图、距离、归一化、权重、聚合规则或重新执行模型，都不属于这个不等式直接检验的范围。

额外 active 关系指与同一 fused 最难正例配对时 `d_pos−d_extra_negative+0.3>0`；非正间隔指 `d_extra_negative−d_pos≤0`。后者是前者的子集，不能混称错误检索性能。一个非极值 active 关系可以被 fused 的单一 hard-max 目标遮蔽；其存在只证明几何关系未被该单一索引选中。它不证明实际训练 loss 已改变。本次只是计算了把全集限制到角色并集后的反事实标量，根本没有修改训练目标或执行更新。

标量相同也不自动证明参数梯度相同：ties 的 index-max/amax 分配、relu 边界、detach/缓存候选、模型状态、计算图和权重都会影响梯度。在本次无 tie 的有效数组中，正的相等 hinge 都保留了两端唯一 fused 极值；此外有 **34** 个相等零 hinge 的 query 没有同时保留两端极值。但这些数组不携带真实训练 autograd 图，所以不据此报告当前训练的参数梯度相等或改变。只有在同一可微局部、同计算图/张量依赖/权重、同唯一 active 极值等额外前提成立时，才可讨论对应局部导数；本次未执行此类训练模式验证。

即使未来另行构造 sum、soft weighting 或其它目标令额外关系获得梯度，这也属于新的训练假设，不能从当前统计直接推断有效。当前 query 是旧状态对应的已见 source 身份，多个状态、视图和协议高度重复，没有未知身份检索门、收益对照或多 seed 证据。**本审计不提出、不登记、不授权下一训练损失、top-k、scene 规则或组合改动，也不包含 novelty/SOTA 判断。**

## 零新模型计算与历史成本的范围

本轮分析器完整源码只有标准库/NumPy 导入与已保存 NPY/JSON 读取，wrapper 调用该模块并配置 OPENBLAS/MKL/OMP 为 4。新输出目录没有 checkpoint 或任何模型新产物。本审计自己的原始数组复算以 stdin 脚本执行，`-B`/PYTHONDONTWRITEBYTECODE=1，CUDA_VISIBLE_DEVICES 为空，四线程环境；没有导入 torch 或任何项目模块，并使用审计 hook 拒绝所有文件写模式。确定性计算 **89.697854 秒**，最大 RSS **3,264,956 KiB**。完整 read-only 打开路径记录在 `deterministic_verification.json:2441`，全部路径为元数据/源码/NPY/JSONL，没有图像或 checkpoint。

旧来源提取已发生的成本不能被本轮零模型计算覆盖：封存 summary 记录 **18,576 source_record_forwards、306 forward_batches、306 additional_visual_passes**；其代码确实在旧提取时调用 exact_signal_forward，coupled-style 使用额外模态视觉 pass。旧提取的 model-state unchanged 和 gradients absent 来自被哈希绑定的代码/运行 receipts；本次只验证这些收据内部及与旧状态绑定的一致性，没有重新执行旧提取来构造新的 runtime 证明。旧 CPU census 的 972 输出/协议 cell 是历史已有内容，本次新验证范围为所选 4 输出下的 54 条件/37152 行，不能混为历史整个 census 的再次运行。

零前向/零更新/零新 checkpoint 的结论针对被审计分析命令及其输出目录，不声称做过全服务器历史进程追踪。读取含官方文件名的既有标签证据 JSON 不等于打开官方图像；本次候选构建只使用其中 bounding_box_train 数据。

## 唯一需要关闭的文档项及交付结论

`EXPERIMENT_TRACKER.md:3-10` 在本审计快照中仍记录启动/初查 RUNNING，未写最终退出时间和审计结果。这与实际文件和退出状态不构成数值冲突，应在归档完整审计后更新为完成状态并保留 **确定性 PASS、整体 WARN、same-family/provisional、source-only descriptive**。无须为这个文档状态问题重跑数组分析或训练。本审阅没有替执行者修改 tracker。

支持的表述是：“固定旧来源状态和诊断视图的完整角色关系复算发现额外非极值 active fused 关系；同候选池角色子集从未增大全集 hardest hinge。”不支持的表述包括：“新的训练损失已有效”“当前 history-gradient 终点已获得新梯度”“Q1/未知身份提升已证明”或“标量相等保证训练梯度相等”。

## 附录：全部 54 条件的核心计数（不是抽样）

协议缩写 I=identity_exclude_record，C=cross_scene。每行的完整全部 counters 在 `independently_recreated_conditions.json`；每个 query 的所有字段在 `independently_recreated_query_rows.jsonl`，没有省略无正例记录。

| 条件 | 协议 | 总行 | 有效 | 额外负 record | active 额外负 | 非正间隔额外负 | hinge 更低 | hinge 相等 | hinge 更高 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fold_0_initial_clean | I | 672 | 672 | 559 | 559 | 250 | 31 | 641 | 0 |
| fold_0_initial_clean | C | 672 | 390 | 344 | 344 | 195 | 20 | 370 | 0 |
| fold_0_initial_augmented | I | 672 | 672 | 600 | 600 | 262 | 34 | 638 | 0 |
| fold_0_initial_augmented | C | 672 | 390 | 352 | 352 | 204 | 20 | 370 | 0 |
| fold_0_initial_coupled_style | I | 672 | 672 | 624 | 624 | 257 | 43 | 629 | 0 |
| fold_0_initial_coupled_style | C | 672 | 390 | 386 | 386 | 200 | 34 | 356 | 0 |
| fold_0_control_final_clean | I | 672 | 672 | 750 | 687 | 33 | 40 | 632 | 0 |
| fold_0_control_final_clean | C | 672 | 390 | 422 | 416 | 32 | 22 | 368 | 0 |
| fold_0_control_final_augmented | I | 672 | 672 | 726 | 674 | 44 | 51 | 621 | 0 |
| fold_0_control_final_augmented | C | 672 | 390 | 416 | 413 | 40 | 29 | 361 | 0 |
| fold_0_control_final_coupled_style | I | 672 | 672 | 778 | 740 | 73 | 59 | 613 | 0 |
| fold_0_control_final_coupled_style | C | 672 | 390 | 450 | 448 | 62 | 35 | 355 | 0 |
| fold_0_style_final_clean | I | 672 | 672 | 699 | 650 | 62 | 40 | 632 | 0 |
| fold_0_style_final_clean | C | 672 | 390 | 403 | 398 | 59 | 25 | 365 | 0 |
| fold_0_style_final_augmented | I | 672 | 672 | 702 | 655 | 62 | 35 | 637 | 0 |
| fold_0_style_final_augmented | C | 672 | 390 | 400 | 400 | 58 | 14 | 376 | 0 |
| fold_0_style_final_coupled_style | I | 672 | 672 | 737 | 701 | 82 | 46 | 626 | 0 |
| fold_0_style_final_coupled_style | C | 672 | 390 | 431 | 431 | 74 | 24 | 366 | 0 |
| fold_1_initial_clean | I | 683 | 683 | 535 | 535 | 261 | 27 | 656 | 0 |
| fold_1_initial_clean | C | 683 | 393 | 320 | 320 | 209 | 18 | 375 | 0 |
| fold_1_initial_augmented | I | 683 | 683 | 585 | 585 | 264 | 30 | 653 | 0 |
| fold_1_initial_augmented | C | 683 | 393 | 349 | 349 | 210 | 18 | 375 | 0 |
| fold_1_initial_coupled_style | I | 683 | 683 | 627 | 627 | 285 | 40 | 643 | 0 |
| fold_1_initial_coupled_style | C | 683 | 393 | 378 | 378 | 221 | 25 | 368 | 0 |
| fold_1_control_final_clean | I | 683 | 683 | 744 | 707 | 55 | 57 | 626 | 0 |
| fold_1_control_final_clean | C | 683 | 393 | 399 | 396 | 50 | 29 | 364 | 0 |
| fold_1_control_final_augmented | I | 683 | 683 | 753 | 725 | 82 | 74 | 609 | 0 |
| fold_1_control_final_augmented | C | 683 | 393 | 426 | 426 | 74 | 45 | 348 | 0 |
| fold_1_control_final_coupled_style | I | 683 | 683 | 773 | 747 | 119 | 60 | 623 | 0 |
| fold_1_control_final_coupled_style | C | 683 | 393 | 429 | 429 | 99 | 37 | 356 | 0 |
| fold_1_style_final_clean | I | 683 | 683 | 685 | 643 | 80 | 40 | 643 | 0 |
| fold_1_style_final_clean | C | 683 | 393 | 366 | 362 | 73 | 16 | 377 | 0 |
| fold_1_style_final_augmented | I | 683 | 683 | 696 | 671 | 95 | 53 | 630 | 0 |
| fold_1_style_final_augmented | C | 683 | 393 | 398 | 398 | 87 | 34 | 359 | 0 |
| fold_1_style_final_coupled_style | I | 683 | 683 | 718 | 693 | 129 | 59 | 624 | 0 |
| fold_1_style_final_coupled_style | C | 683 | 393 | 405 | 405 | 111 | 33 | 360 | 0 |
| fold_2_initial_clean | I | 709 | 709 | 595 | 595 | 260 | 27 | 682 | 0 |
| fold_2_initial_clean | C | 709 | 417 | 348 | 348 | 223 | 16 | 401 | 0 |
| fold_2_initial_augmented | I | 709 | 709 | 639 | 639 | 254 | 31 | 678 | 0 |
| fold_2_initial_augmented | C | 709 | 417 | 370 | 370 | 217 | 22 | 395 | 0 |
| fold_2_initial_coupled_style | I | 709 | 709 | 674 | 674 | 283 | 38 | 671 | 0 |
| fold_2_initial_coupled_style | C | 709 | 417 | 381 | 381 | 239 | 20 | 397 | 0 |
| fold_2_control_final_clean | I | 709 | 709 | 819 | 757 | 81 | 50 | 659 | 0 |
| fold_2_control_final_clean | C | 709 | 417 | 474 | 471 | 78 | 28 | 389 | 0 |
| fold_2_control_final_augmented | I | 709 | 709 | 819 | 771 | 92 | 55 | 654 | 0 |
| fold_2_control_final_augmented | C | 709 | 417 | 482 | 481 | 86 | 38 | 379 | 0 |
| fold_2_control_final_coupled_style | I | 709 | 709 | 844 | 819 | 134 | 74 | 635 | 0 |
| fold_2_control_final_coupled_style | C | 709 | 417 | 488 | 488 | 116 | 42 | 375 | 0 |
| fold_2_style_final_clean | I | 709 | 709 | 798 | 748 | 108 | 58 | 651 | 0 |
| fold_2_style_final_clean | C | 709 | 417 | 479 | 478 | 103 | 40 | 377 | 0 |
| fold_2_style_final_augmented | I | 709 | 709 | 789 | 748 | 122 | 55 | 654 | 0 |
| fold_2_style_final_augmented | C | 709 | 417 | 450 | 449 | 115 | 29 | 388 | 0 |
| fold_2_style_final_coupled_style | I | 709 | 709 | 777 | 746 | 140 | 55 | 654 | 0 |
| fold_2_style_final_coupled_style | C | 709 | 417 | 446 | 446 | 129 | 34 | 383 | 0 |

## 附录：独立实现和可复核产物

`independent_replay.py` 完全由审阅者编写，不 import 被审实现；使用独立标签列表构造、Python 稳定 min/max、record 集合并集、独立递归全字段比较及另一路直接向量差距离见证。其原始 stdout 中包含每个重建 query、所有条件 counters、全部实际数组哈希和验证结果。`summarize_replay.py` 将它们展开为直接查看的 JSONL/JSON，并独立比对本地 intake 与远端原件的绑定。

| 审计产物 | 实际字节 | SHA-256 |
|---|---:|---|
| `independent_replay.py` | 21,215 | `f853cf9b4c7b28c18c878d1a91eb4b8e2cfe878a497679f6c4979df6284648a9` |
| `independent_replay.stdout.jsonl` | 57,157,230 | `d77b53323843e3da0ccb4ddf0b5782bf1390033fded6a443b27f9aa9ce74819b` |
| `independent_replay.execution.json` | 650 | `bf272f6830e25791a2fbf9b2ff31ea4e60e06e7119a2e00b0b78e1029689960c` |
| `independently_recreated_query_rows.jsonl` | 29,971,120 | `a9b907103deeadda012fbe6f2870ce87b538bd1c161eb359349105a74c3e3730` |
| `independent_extremum_witnesses.jsonl` | 26,076,635 | `f96ddb0c4c789d59dd2b9cd2f20ac1bf87168d821db930507211032290279cd1` |
| `independently_recreated_conditions.json` | 51,823 | `d4ec82b17e9c7bb3c70901b50df53fe079d84e971595b4e6c7ba678d083581a9` |
| `deterministic_verification.json` | 103,968 | `6f92e108e39ee8145224093efd1f6b33faefa539017a84e60decb3df4c94bc75` |
| `actual_remote_input_hashes.json` | 42,434 | `2bfa927d1fc2ca8b2a2a86ba8a6c68cc616000541db11fdb10ca02feb98e3357` |
| `local_input_hashes.json` | 8,299 | `8e68856e67c3421ebf0b07c9913b55e5230184e9a0db0f42dd7ad8bc3c76ac0b` |
| `remote_text_input_hashes.json` | 26,366 | `58e4fa76427a5da9f059d1f6838752576ded25444741585a0e73e775e7a68c55` |
| `supplementary_input_hashes.json` | 2,175 | `af5b44dca2d52cc7f392cfa3dbfc2f72526444acdb4a308656ca03d0e9306463` |

审阅完整响应原文即 `reviewer_full_response.md`，并按原字节保存在 `trace/001-role-relation-coverage.response.md`；机器结果是 `audit.json`。完整脚本/原始输出/快照哈希清单最终保存在 `artifact_hashes.json`。trace 请求保存的是原始审阅请求，没有执行者结果解释。由于用户明确要求所有写入只在本审计目录，常规 `.aris/traces` 和 `.aris/meta/events.jsonl` 的本轮记录重定向到本目录的 `trace/`，没有向科学项目写入。
