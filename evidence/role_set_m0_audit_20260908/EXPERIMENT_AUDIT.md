# MSVR310 role-set v1 完整 M0 独立工程完整性审计

日期：2026-09-08。审计者：gpt-6-astra / max，新上下文原生 Codex reviewer；`review_independence: same-family`，`acceptance_status: provisional`。模型归属来自调用规范，未取得独立后端身份认证。

**总体判定：WARN。工程判定：PASS。确定性核验：PASS。Q1 科学资格：未评价。**

完整登记 M0 的文件、来源记录、距离、关系、损失账本、终点权重和阶段退出均能对应。未发现本轮 M0 的数字错报、虚构文件、身份目标来自模型输出、归一化作弊或需要修改科学代码的问题。WARN 来自可复现边界：部分关键模型级结论只有原运行断言和范数汇总，现有保存产物不能独立重建原模型梯度、缓存/RNG 或重载前后输出。

审计对象为执行提交 `26c97390704c629237687d263b4381f5584cbe97`，远端运行 `/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739` 的 T0/M0/M0_CPU。源码引用相对于 `C:/Users/gb/.trifusion_github_publish_22c3bee`；审计 JSON/脚本引用相对于本报告目录。远端精确文本快照保存在 `snapshots/remote/`，本地原始副本在 `snapshots/project/` 与 `snapshots/intake/`。完整输入 SHA 在 `EXPERIMENT_AUDIT.json`，快照 SHA 在 `snapshot_manifest.json`。

## 分项结论

| 检查 | 状态 | 结论 |
|---|---|---|
| A. 标签与完整路径身份隔离 | PASS | 真实数据文件名标签、三折 source/heldout 与 Signal 来源绑定相符；无伪 GT |
| B. 目标、归一化与 loss floor | PASS | 固定均值 hinge、原 hard tie、预热和其余 13 项账本均可重算 |
| C. 执行、文件、更新与终点 | PASS | 全 248 步和六权重状态匹配，阶段退出与原文本一致 |
| D. 模型路径与梯度 | WARN | 活跃调用链明确；原参数梯度、缓存/RNG 与输出逐位证明只能作为有界运行见证 |
| E. 范围、公平性与成本 | PASS | 完整执行所登记的 M0，配对资源一致；没有科学晋级结论 |
| F. 证据分类与声明 | PASS | 区分真实标签、工程自比较、合成数学和独立确定性重算 |

## A. 标签与完整路径隔离 — PASS

原始标签证据 `evidence/vehicle_query_protocol_labels_20260905.json` 的 SHA 与协议登记 `c835d20478b817a54b7710463269186af2619cab3e38850534b01f3aaee6e3c8` 一致；1032 条训练记录的 identity/camera/scene 均与文件名和原始标签逐项一致，vis/ni/th 三模态共 3096 个路径均存在且完整配对。仅枚举训练文件名和元数据，没有打开图片字节。训练身份与官方 query/gallery 的标签身份集合不相交。依据：`tools/build_msvr310_train_oof_protocol.py:12-55`，`tools/train_msvr310_signal_oof.py:68-76`，`supplementary_checks.json:5`。

三折来源身份为 103/103/104，来源记录 672/683/709；对应留出身份 52/52/51，完整图库记录 360/349/323，合法 query 210/207/183。独立从 scene eligibility 分层轮转规则重建划分，并逐一重算所有图库/合法正例元数据。这些是既定协议的标签检查，不是本次已运行的留出评价。依据：`tools/build_msvr310_train_oof_protocol.py:24-123`，`independent_arithmetic.json:9`。

来源 Signal 是各折固定 50 epoch 的 B0 checkpoint；独立加载三个权重，核对 source/heldout/fold 元数据、文件 SHA 和完整 Signal 状态 SHA，并检查其保存的每折 650 个来源训练 batch，共 1950 步均只引用该折来源记录。M0 `build_model` 每端重新构建，载入该折 Signal，角色 seed42 随机初始化且 `role_weights_loaded=False`；不读取旧角色终点或全训练身份 teacher。`HierarchicalFrozenSignalBackbone` 冻结同一 Signal，三个 frozen tail 引用也来自它，因此被测基座/角色整条路径均继承该折来源隔离。依据：`tools/train_msvr310_trifusion_oof.py:27-55`，`modeling/trifusion/signal_preserving_v8.py:33-77`，`tools/build_v12_complete_path_oof_targets.py:384-411`，`independent_checkpoints.json:7`。

训练分类标签是 source label map 的连续编号，关系掩码用真实 identity；二者同属同一等价关系，类别 0 合法。scene 在训练关系日志中用于描述，未误用于排除所有同 scene 负例。M0 的加载器、预检和重载抽样均走 source records；留出读取只在 `m0=False` 分支。依据：`tools/train_msvr_role_set.py:85-115`、`:319-355`，`tools/train_msvr_instance_memory.py:53-91`。

## B. 目标、归一化与账本 — PASS

每个 anchor 保留同一最难正例；负例集合取 fused 与三个完整角色（Signal+角色）各自最近合法负例的位置去重并集。使用单位特征的未平方欧氏距离、margin 0.3。记集合大小为 n，原 hard hinge 为 h，去掉 fused 首选位置后的额外位置集合为 E，则逐 anchor 目标为 `(h + sum(relu(d_pos - d_neg[j] + 0.3), j in E)) / n`，再对 64 anchors 求均值。索引和角色距离只负责提议，不反传；最终 loss/两侧导数在 fused 空间计算。依据：`tools/msvr_role_set_relations.py:20-64`。

该分母是所选关系数，不是模型输出最大值等分数重标定。标准特征 L2 归一化和既有等能量拼接出现在表示定义中，也不是把检索成绩缩放到高值。额外关系与最难关系降权是同时发生的干预，不能把将来的差异独归因于异构角色。依据：`modeling/trifusion/signal_preserving_v8.py:494-516`，`refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:11-15`。

原 hard 项分别做 current/history 极值，再用 `torch.maximum/minimum` 合并，保留跨 current/history 相等时的导数分配。独立 CPU toy 指定了应为 ±1/8 的精确导数，实际代码完全相符；另 12 组独立构造距离的集合/均值标量和距离导数均与显式逐项参考式相符，并重做合成候选 VJP 链式法则检查。它们仅是合成数学检查。依据：`supplementary_checks.json:72`，`supplementary_checks.json:120`，`supplementary_checks.py:1-104`。

M0 前两次更新用原批内 triplet，第三次起替换 fused 项；第一次历史候选出现在第四次更新。登记 Q1 的对应预热为 65 步，不能从 M0 的短预热推断长训练行为。其余 13 项仍从同一个 `ExpertFormationV8Criterion` 计算，权重沿用原合同。独立重算所有 248 步保存的 fused 距离目标，最大误差 **3.24100253857e-08**；独立从已保存 14 个分量重建加权总损失，最大误差 **5.7121117969e-07**。未重新生成其余 13 项的模型前向。依据：`tools/train_msvr_role_set.py:125-160`，`modeling/trifusion/signal_preserving_v8.py:690-742`，`tools/run_signal_preserving_v5.py:99-142`，`local_report_checks.json:2`。

103 类、label smoothing 0.1 的加权 CE 熵下界为 **0.5857136327437849**，ID 总权重为 0.75。两端首 loss 均为 4.122129917144775；末 loss 为 control 0.5881941318511963、role_set 0.5881949663162231。由 `(last-floor)/(first-floor)` 得 **0.0007014160404002304 / 0.0007016520038614711**，均低于登记 0.1。没有使用中途最低 loss 或模型自身最大值。依据：`tools/run_signal_preserving_v5.py:1580-1617`，`independent_arithmetic.json:210`。

## C. 原始文件、执行与六终点 — PASS

递归配置展开包含八份配置，其中 R1/原 source 配置作为来源链保留；137 条 project/fixed/Signal 源码 hash 绑定全部在远端匹配，另核对 CLIP、baseline summary、protocol、metadata 等嵌套 SHA。新合同与 14 个直接 project bindings 共 15 个文件和原执行提交逐字节相同；核心被调用依赖与执行提交语法树一致。Signal commit 和 diff hash 也匹配。依据：`remote_inventory.json:606`，`binding_and_launch_checks.json:25`，`binding_and_launch_checks.json:1-29`。

本地五个旧源码依赖的工作副本为 CRLF，而固定远端为 LF，导致本地直接字节 SHA 不同；独立换行归一和 AST 比较相同，远端原字节符合合同。不能用本地 checkout 字节冒充该五项运行字节。依据：`remote_inventory.json:22676`。

29 份 M0 原始文本共 **5,717,997 字节**，本地接收副本、仓库证据副本与远端都与 intake manifest 匹配。summary SHA 为 `203f2e572d40e00d0d41842289746ad2c077d8d7e97ada5fe237214f1b161a38`。原 CPU 收据列出的 **37 文件**被再次重新散列；六个角色 checkpoint 共 **196,617,690 字节**，所有原始距离数组共 **19,783,680 字节**。依据：`local_inventory.json` 中 `intake_checks`，`independent_checkpoints.json:368`，`local_report_checks.json:1-20`。

每个 checkpoint 的 231 个角色状态项和 241 个 frozen baseline 别名项被从真实来源 Signal 重建；六个完整状态 SHA、冻结状态 SHA、Signal 状态 SHA 均与训练终点/严格重载收据一致。训练参数域是 203 个张量，其中 encoder 189，另 14 为七组可训练 BN weight/分类器；冻结 BN bias 七项和 buffer 单独核对。每折两端 trainable parameter count 相同，fold0/1 为 8,076,300，fold2 为 8,102,412，差异来自来源分类数。依据：`tools/train_msvr310_source_style.py:164-174`，`tools/train_msvr_instance_memory.py:94-110`，`independent_checkpoints.json:66`。

原阶段收据：T0 用时 4.8872329947 秒、M0 **614.7843318302 秒**、M0_CPU 8.3967608288 秒，均 exit 0，结束后原 wrapper 才启动 Q1。八个 epoch 日志事件和保存的 training history 全等。只读进程观察时间为 2026-09-08T12:04:35.518926+00:00：原 wrapper 35302 和 Q1 36320 存活，已完成阶段的原 PID 不存在。没有打开 Q1 目录、Q1 log 或分数。依据：`binding_and_launch_checks.json:241`，`remote_inventory.json:22484`，`local_report_checks.json:2`。

## D. 实际梯度路径与有界见证 — WARN

`ViewFields.capture` 保存当前真实 source batch 的 frozen anchor/reference/baseline 字段，以及角色入口的 CPU/CUDA RNG；同 record 重复采样时 queue 留最后一次视图。历史重编码读取原完整 64 样本组，在当前 encoder 参数下重跑，保留其原 batch/RNG 语义；不是将旧特征作为当前坐标。`refresh_all` 两端都计算 fused 与三个角色，保存距离也确有四矩阵。依据：`tools/msvr_freshness_probe.py:48-60`，`tools/msvr_instance_memory.py:15-37`，`tools/probe_msvr_role_set_gradients.py:29-50`。

当前完整目标先 backward；候选侧把新鲜历史向量作为 leaf 求上游导数，再在原历史完整 batch 上用 `encode_graph` 计算 encoder VJP。循环只跳过上游为零的历史组；随后把历史导数准确加到现有当前参数梯度，unscale 后执行一次 AdamW step。原 current graph 包括当前样本既作 anchor 又作 current candidate 的导数，历史路径只有 candidate 导数，没有历史 anchor。分类器/neck 不决定 fused 检索嵌入，所以历史 VJP 限于 encoder 189 张量符合当前模型定义。依据：`tools/train_msvr_role_set.py:146-247`，`tools/probe_msvr_history_candidate_gradients.py:43-50`，`modeling/trifusion/signal_preserving_v8.py:614-653`。

六个容量端均在第 4 步首个单历史组执行直接完整图检查：重新计算 fused 与三个完整角色，四输出逐位一致断言；原总 loss 的直接 encoder 导数与当前导数+历史 VJP 相比较。最大记录相对 L2 误差为 **1.6810499980311218e-05**，小于 0.005。它是原完整 14 项目标在 **189 个 encoder 参数张量**上的比较，并不是在 203 个全部训练张量上的直接图比较。依据：`tools/train_msvr_role_set.py:166-183`、`:227-241`，`independent_arithmetic.json:247`；例如 `snapshots/project/evidence/msvr310_role_set_m0_complete_20260908/m0/fold_0_control/memory_steps.jsonl:4`，其余五端见 direct_recorded_norm_checks 列表。

独立算术核对全部 248 步的梯度范数关系、余弦恒等式、所选历史组、上游范数长度、应用前后统计和直接比较比率。可核验的记录计数是：fresh role 重编码 **6272**，历史 VJP **5760**，直接检查 **384** 个 record-forwards。它们是按原组记录重算的计数，不是本审计重新执行的模型前向。依据：`independent_arithmetic.py:1-254`，`local_report_checks.json:10-18`。

限制必须保留：

- 没有保存真实参数梯度向量、每步模型状态、原 RNG 或 frozen-field 缓存，所以范数/计数一致不能替代真实模型 Jacobian 的独立复现。
- 203/203 指原运行的累计非零梯度覆盖。名字域/终点权重可核对，逐步全部非零既没有被证明，也不是报告中的主张。
- 首历史组之外，原代码检查 fused VJP 重编码相等、RNG/buffer 不变和梯度累加；没有全历史组直接完整图或三角色图输出的独立证明。
- 六次严格模型重载和五种检索输出逐位相同是原 runner 在每端八条来源记录上的运行断言。原始 before/after 输出数组没有保存；本审计重建的是状态映射及 SHA，没有重跑模型输出。
- 初始角色权重与两个 overfit 最终权重没有独立保存文件；相应初始/最终 hash 和冻结状态仍为运行见证。

这些边界在当前登记计划/结果中已有明确限定，本审计没有发现需要补写成失败的已登记全程直接图要求。旧梯度探针没有历史全角色图比较的历史限制也不能被本轮首组检查追溯改写。依据：`refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:29-33`，`results/MSVR310_ROLE_SET_V1_M0_2026-09-08.md:9-12`。

## E. 完整长度、配对、公平性和成本 — PASS

| 端点 | 更新 | 唯一来源记录 | 距离元素 | 历史候选曝光 | 最大历史数 / 年龄 | VJP record-forwards |
|---|---:|---:|---:|---:|---:|---:|
| fold_0_control | 8 | 333 | 271,104 | 547 | 195 / 5 | 960 |
| fold_0_role_set | 8 | 333 | 271,104 | 547 | 195 / 5 | 960 |
| fold_1_control | 8 | 316 | 263,936 | 519 | 176 / 5 | 960 |
| fold_1_role_set | 8 | 316 | 263,936 | 519 | 176 / 5 | 960 |
| fold_2_control | 8 | 350 | 299,520 | 658 | 221 / 5 | 960 |
| fold_2_role_set | 8 | 350 | 299,520 | 658 | 221 / 5 | 960 |
| overfit_control | 100 | 53 | 1,638,400 | 0 | 0 / 0 | 0 |
| overfit_role_set | 100 | 53 | 1,638,400 | 0 | 0 / 0 | 0 |

总计 **248 更新、4,945,920 距离元素、738 条跨端去重来源记录**；每步仍为 64 个当前样本位、8 身份×8 视图。全部四对序列（含 overfit）逐步记录索引和三模态增强像素 SHA 相等。重复曝光不作为独立样本。像素 tensor 没有保存，因此只确认日志 SHA 配对，未重新生成增强图像。依据：`independent_arithmetic.json:2`，`tools/train_msvr_role_set.py:109-115`、`:356-375`。

容量端的实际最大历史候选数为 221、最大年龄 5；登记容量为 512、最大年龄 8。另 780 个完整 source batch 的独立无模型 replay 验证年龄 8、顺序、当前 record 排除与过期规则；这不等于 GPU 容量端运行过年龄 8 或 512 个历史候选。两个 100-step overfit 固定 batch 只有 53 个唯一 record，因当前 record 的历史副本被排除，历史候选和历史 VJP 均为 0。其通过仅证明固定来源 batch 的优化能力，不证明长历史反传过拟合。依据：`independent_arithmetic.json:9`，`independent_arithmetic.json:210`，`tools/msvr_instance_memory.py:15-37`。

两端共用新鲜坐标、完整历史反传、V8 架构、AdamW、LR0.00035、weight decay0.0001、AMP 初始 scale256、既有其他损失和配对来源；没有新训练模块。角色提议代码即使在 control 端也实际执行；改变的是所选 fused 关系目标及其候选上游。运行的累计非零梯度/AMP/frozen assertions 均完成；容量端峰值 allocated **11266.8193359375 MiB**、峰值 reserved **12116 MiB**，未越登记 24 GiB。依据：`tools/train_msvr_role_set.py:71-105`、`:125-160`、`:242-283`，`independent_arithmetic.json` 的 `endpoint_checks`。

8 个训练循环记录耗时合计 **459.3965334482 秒**；整个 M0 阶段用时 **614.7843318302 秒**，包含构建/预检/序列化等开销，不能把两者混写。封闭的 M0/T0/CPU 共 **42 文件、222,116,266 字节**（不含正在变化的 pipeline/Q1）。启动盘空闲 **11,129,188,352 字节**，启动 GPU 占用收据为 **1 MiB**。本轮未测量新的 GPU 性能、包裹整个 Q1 的盘占用或未来运行时长。依据：`binding_and_launch_checks.json`，`local_report_checks.json:14-20`。

科学门仍是完整三折两端、固定 20 epoch、1560 更新后两组既有五项全部成立；单种子身份 bootstrap 不是多训练种子。M0 没有新的 600-query 检索数组或科学成绩，原 CPU 收据的 retrieval element count=0。本报告不推断 Q1 结局。依据：`refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:19-41`，`tools/verify_msvr_role_set.py:250-307`。

## F. 证据类型与声明影响 — PASS

| 证据 | 类型 | 能支持的声明 |
|---|---|---|
| 数据集文件名 identity/scene、来源训练监督 | `real_gt` | 标签/隔离/监督来源正确；不自动成为检索成绩 |
| Signal 相等、重载相等、直接图/VJP 相比较 | `synthetic_proxy` 工程自比较 | 有界实现一致性，不能称作任务 GT 或方法优越性 |
| 随机矩阵、类 0、tie、合成链式法则 | `simulation_only` | 数学实现检查，没有真实模型或一般化结论 |
| 保存距离与标量重算、hash、标签队列重建 | 确定性审计证据 | 精确说明重算域；不替代模型级原运行见证 |

声明影响逐条见 `EXPERIMENT_AUDIT.json:claims`。完整 M0、标量/距离/终点状态均支持；203 非零覆盖、初始权重一致、原模型逐位输出和直接图数值关系必须保留“原运行见证”限定。全 248 步真实模型梯度独立复现、角色多样性单因果归因、Q1 资格与官方/SOTA 等声明均未建立。

## 审计过程、失败尝试与复现边界

所有独立脚本、输出、快照和本报告只写入本审计目录。未训练、未构造模型、未做模型/GPU forward/backward、未安装包、未改进程或科学文件、未读官方图片、未读 Q1 结果。远端 CPU 设置最多 2 个 compute threads；checkpoint 只在 CPU 用 `weights_only=True` 加载。CPU 合成 tensor autograd 不含模型对象。

保留了三项审计自身的问题及修复记录：私有 SSH helper 的 stdout.reconfigure 与 StringIO 不相容；初版 inventory 在无 signal binding 的配置上误取 `signal_source`；文本快照的 CRLF 归一导致本地快照字节不符。前两项分别在连接准备/清单收集阶段停止，第三项是审计快照序列化问题。对应 `attempt_01_transport_failure.txt`、`attempt_02_remote_inventory.*`、`attempt_03_snapshot_failure.txt`。22 个受换行影响的文本随后经只读二进制 SFTP 收齐，原规范化副本保留在 `attempt_03_normalized_snapshots/`；最后快照字节差异为 0。没有将不可用检查记为 PASS。

`README.md` 给出脚本/命令顺序，所有成功远端执行命令与脚本 SHA 在 `*.command.json`。凭据及 SSH helper 原文没有复制到审计目录。结论无需修改在运行代码、重启或追加实验；下游保持本报告的证据范围，并在原完整 Q1 终态另行审计。

最后的交付核验另发现报告生成器先对 LF 字符串计算 verdict_id，而 Windows 实际写入 CRLF，以及四处报告行范围超出文件末尾/一处通配引用不能直接定位。均已在审计报告层修正，原失败保存在 attempt_04_report_packaging_failure.txt 和 attempt_04_reference_validation.json；最终使用实际文件字节 SHA，并核验所有明确引用存在、范围有效。科学输入和核验结论未变化。
