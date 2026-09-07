# MSVR310 fresh-coordinate V1 complete Q1 audit

总体结论：**WARN；完整文本一致性核验 PASS，科学终态为 Q1_FAIL，保持不晋级。** 本次审查为独立上下文、只读的 Codex 同模型家族审查，记录 `review_independence: same-family`、`acceptance_status: provisional`。未启动训练、未写入文件、未调用外部模型，未独立重算任何未取得的模型二进制或反向传播。

以下路径均相对于 `C:/Users/gb/.trifusion_github_publish_22c3bee`。为减少重复，`Q/` 表示 `evidence/msvr310_fresh_coordinate_complete_q1_20260908/`。我已确认仓库中全部 29 个原始文本文件与最初提供的 `.codex_tmp/msvr_fresh_coordinate_complete_q1_20260908/` 文件逐一 SHA 相同。

**A. Ground Truth Provenance：PASS，保留原始数据未访问的限制。**

- MSVR 标签来自数据集文件名中的 identity、camera、scene，未从模型输出生成。证据：`tools/audit_vehicle_query_protocol_labels.py:13-18`。
- OOF 协议只取 `bounding_box_train`，根据标签确定跨 scene 查询资格和身份分折。证据：`tools/build_msvr310_train_oof_protocol.py:12-55`、`:57-123`；`protocols/msvr310_train_oof_v1.json:4-22`。
- 我独立检查了全部 **1032 条记录、155 个身份、3096 条模态路径**，所有 identity/camera/scene 均与文件名及完整标签清单一致。标签清单 SHA 为 `c835d20478b817a54b7710463269186af2619cab3e38850534b01f3aaee6e3c8`，协议构建器 SHA 也一致。标签清单位置：`evidence/vehicle_query_protocol_labels_20260905.json:6541` 起的完整训练清单。
- 全部 source/heldout 身份和记录隔离、query 资格、gallery 构成均由我重新计算。无正例的 **95 个身份、432 条记录**全部保留为 gallery 干扰记录；合法 query 为 **600 条、60 个身份**。证据：`protocols/msvr310_train_oof_v1.json:22576-22582`。
- 当前 Q1 的 evaluator 确实调用 Signal 上游 `eval_func_msrv`，并与独立 `scene_scores` 比较。证据：`tools/train_msvr310_trifusion_oof.py:196-241`。全部 30 个 fold/端/输出的上游差异字段均在原阈值内，最大已记录差异为 **2.96583498027303×10⁻⁶ pp**。
- **限制**：我访问的是完整标签清单、协议及执行代码；原始图片、远端 comparator 的 17 个源文件及实际 GT 文件系统未独立读取。官方读取 0 与代码构造的 `bounding_box_train` 路径及收据一致，但不是独立的系统级文件访问审计。

**B. Score Normalization：PASS。**

- 检索前对每个特征向量作 L2 normalization，再计算平方欧氏距离；AP 分母为真实正例数量，mAP 为全部合法 query 的 AP 均值，CMC 为首次正确匹配的位置统计。没有用预测分数的最大值、最小值或均值归一化最终性能指标。证据：`tools/train_msvr310_signal_oof.py:223-238`；`tools/train_msvr310_trifusion_oof.py:212-230`。
- 扩展 Triplet 同样使用单位特征和真实身份正负关系，历史候选 detach；当前 peers 保留梯度。向量归一化不是指标归一化。证据：`tools/msvr_instance_memory.py:40-80`。
- 身份 bootstrap 使用完整身份簇重采样并保留 query 权重，固定 seed42、10000 次、2.5% 线性分位数。证据：`modeling/trifusion/signal_preserving_v13.py:253-279`；固定合同 `configs/MSVR310/TriFusion-source-style-paired-v1-r2.json:69-79`。
- 我从完整排序重新计算全部 AP/CMC、身份统计及 bootstrap，原始数值与收据一致，未发现人为放大指标。

**C. Result File Existence / Claim Matching：WARN；数值和终态存在性 PASS，当前 tracker 尚待同步。**

- 全部 **29 个原始文本、79,009,736 字节**逐一 size/SHA 校验通过。依据：`Q/intake_manifest.json:1`。
- Pipeline 为 `COMPLETE_VERIFIED_Q1_FAIL`；五个阶段全部 exit0。Q1 用时 **6904.015165943652 秒**，Q1 CPU 子进程用时 **15.094488112255931 秒**。证据：`Q/pipeline.json:66-113`。
- Q1 summary SHA 为 `419c27a0be8d391b6dd1a68e63b7515f72c94124be4695592e6f586160ba1167`，CPU SHA 为 `91892b5d28b72f1b5a852d31dfa5cf3d4dbbb87d1c30c59b9aba69426c141c19`；相互绑定一致。证据：`Q/pipeline.json:112-113`、`Q/q1_cpu.json:2-4`。
- 我独立复核了新 Q1 报告中的全部主要结果、分折结果、query/身份变化、训练诊断、最后 65 步表及 **60 行身份表，共 463 个数值表格单元**，均按报告精度匹配。
- 我重新聚合了 `local_complete_reaggregation.json` 的全部六端和 120 epoch，**10,764 个数值叶项一致**；新增训练目标 JSON 及 120 epoch CSV 的 **2940 个数值项一致**。
- 最后审查的 Q1 报告 SHA 为 `7c236f64b30acb75e07648b7d5e67dede75bf7368002510d22243c187906940f`。报告第 5 行已披露预历史轨迹差异，第 75 行已区分启动提交和 Q1 启动时 HEAD，这两项修订与证据一致。
- **尚需同步**：`refine-logs/msvr310_fresh_coordinate_v1/EXPERIMENT_TRACKER.md:8-11` 仍写 Q1 `RUNNING`、Q1 CPU `NOT_RUN`。这是当前状态表落后于终态，不能继续作为最新状态呈现。其后带历史时间的 01:36、02:03、02:26 观察应保留为历史，不应改写成当时已经完成。
- M0 报告中的“尚无完整 Q1”属于明确的早期历史状态；本次终态证据不使那些带时间的旧观察成为虚假结果。
- CPU 收据绑定 **49 个文件**，其中本地取得 **25 个文本**，未取得 **24 个二进制：6 个 checkpoint、6 个 retrieval arrays、12 个 f32**。证据：`Q/q1_cpu.json:95-291`。因此不能把本次文本审查描述为独立模型/距离二进制复现。

**D. Executed Evaluator / Dead Code：PASS，范围限于可核查的调用链与完整执行文本。**

当前调用链清楚，指标函数确实进入终态流程：

- wrapper 依次执行 T0 → M0 → M0 CPU → Q1 → Q1 CPU：`tools/run_msvr_fresh_coordinate.py:26-64`。
- Q1 从相同初始化重新构建各端，固定 260 更新，保存 checkpoint、严格重载，然后提取完整 gallery 并调用 evaluator：`tools/train_msvr_fresh_coordinate.py:240-283`。
- 实際更新在预热后将 `triplet_fused` 换为选中端的扩展项，然后计算原加权总损失并反传：`tools/train_msvr_fresh_coordinate.py:110-149`。
- 历史字段重编码复用原冻结 field、历史入口 RNG 和最后位置，检查 buffers/RNG 未变，返回 detach 特征：`tools/msvr_freshness_probe.py:48-85`。
- 旧/新梯度比较及重复同图反传分别执行：`tools/msvr_freshness_probe.py:97-112`。
- CPU 核查训练矩阵、坐标选择、损失、checkpoint 重构、检索数组和排序：`tools/verify_msvr_fresh_coordinate.py:17-154`、`:176-238`。
- 两组五项门在执行路径中实际计算并决定 `next_phase_qualified`：`tools/verify_msvr_fresh_coordinate.py:253-301`。
- 我对全部 **120 条训练 epoch 日志**逐项核对，均与六端 training history 一致；CPU 日志全部字段也与最终收据一致。

没有发现当前 Q1 宣称使用但实际未调用的指标函数。**CPU 的“完整训练核验”是保存距离、标签关系、标量和终点状态核验；其余 13 项训练 loss 没有独立模型前向重算，参数梯度也没有独立反传复现。** 这一边界由 `tools/verify_msvr_fresh_coordinate.py:113-117`、`:303-310` 及 `Q/q1_cpu.json:293-296` 明确支持。

**E. Scope Assessment：WARN。**

我独立核实的实际范围如下：

| Fold | source 身份/记录 | gallery 身份/记录 | query 身份/记录 | gallery-only 干扰身份/记录 |
|---|---:|---:|---:|---:|
| 0 | 103 / 672 | 52 / 360 | 20 / 210 | 32 / 150 |
| 1 | 103 / 683 | 52 / 349 | 20 / 207 | 32 / 142 |
| 2 | 104 / 709 | 51 / 323 | 20 / 183 | 31 / 140 |

证据：`protocols/msvr310_train_oof_v1.json:15524-15535`、`:19128-19139`、`:22490-22501`。六个端实际都覆盖了各自**全部 source 记录**。

完整运行和匹配条件核验结果：

- **1560 个更新、120 个 epoch、780 组配对 batch、2340 组三模态像素 SHA 比较**通过。
- 每端 **194 个含历史更新**，六端合计 **1164**；历史 anchor 曝光 **74,496**，历史候选曝光 **355,244**。
- 每 fold 两端历史曝光分别完全匹配为 **58,133 / 59,505 / 59,984**；实际年龄覆盖 **1–8**。
- 各 fold 实际最大候选历史集合为 **356 / 372 / 362**；最大存储队列为 **391 / 408 / 406**。配置 capacity512 不代表真实运行填满了 512。
- 每端额外重编码 **97,600 条角色记录**，其中 **64 条**为 step0 零更新检查；六端 **585,600**。固定视图漂移另 **7680 条**。全部计数配对相同。证据：`Q/local_complete_reaggregation.json:12-23`、`:143`、`:152-163`、`:283`、`:292-303`、`:423`、`:432-443`、`:563`、`:572-583`、`:703`、`:712-723`、`:843`；计数定义见 `tools/train_msvr_fresh_coordinate.py:96-109`、`:173-205`。
- 全部 **3492 个 role/update** 的旧/新梯度差异范数大于对应重复同图噪声，未定义余弦 **0**。这些是运行时见证。
- 每端累计 **203/203** trainable 张量至少一次非零梯度，不是每一步全部非零。累计集合实现见 `tools/train_msvr_fresh_coordinate.py:142-146`、`:192-203`。
- 完整排序重放覆盖 **2,069,520 个 rank position、6000 个端/query/输出结果**；两份 CSV 的全部 **3000 行 query×输出、300 行身份×输出**及所有字段均一致。
- 过滤后保留的同 scene 异身份负例 rank position 合计 **143,920**；无正例干扰身份记录的保留曝光 **865,140**。这些是重复排序曝光，不是独立样本量。

**重要限制：配对匹配并不等于位级相同训练轨迹。** 全部 66 个无历史步骤中，每 fold 都从 step2 开始有 65 步 loss 不同；最大绝对差分别为 **0.0011534691 / 0.0019909143 / 0.0018357038**。第一处含历史步骤为 step67。直接例子：

- `Q/q1/fold_0_control/training.json:250-252`：step2 loss **4.05465841293335**；
- `Q/q1/fold_0_fresh_memory/training.json:250-252`：step2 loss **4.054688930511475**；
- 六端 `memory_steps.jsonl:1-66` 均为空历史，`:67` 开始有历史。

不能仅根据这些文本确定差异的底层来源，也不能把同图重复梯度噪声当作跨独立训练重复的噪声界。新报告第 5 行已准确披露。单 seed、重复使用内部 OOF、未评估 run-to-run 方差的限制继续存在；本次约 −0.0716 pp 的差值不能支持确定的因果效应或普遍结论。

**F. Evaluation Type Classification：PASS，分类应在最终审计文件中明确列出。**

| 内容 | 分类 | 可支持的结论 |
|---|---|---|
| Q1 600-query 检索 | `real_gt`，内部身份 OOF | 这次固定 seed42/source OOF 比较的性能与原门结果 |
| M0/训练中的真实身份监督 | `real_gt`，training-only | 训练工程及优化诊断 |
| 零更新特征一致性、旧/新坐标及梯度比较 | `synthetic_proxy` / `numerical_consistency` | 模型数值行为和运行时梯度见证 |
| T0 构造向量的数学检查 | `synthetic_proxy` | 数学实现的一致性 |
| 官方测试、多种子、其他数据集性能 | 未执行 | 无对应性能结论 |

依据：`tools/msvr_instance_memory.py:102-136`；`tools/msvr_freshness_probe.py:62-112`；`tools/train_msvr310_signal_oof.py:223-238`；`results/MSVR310_FRESH_COORDINATE_V1_M0_2026-09-08.md:46-48`。梯度比较不是有独立外部真值的梯度精度评测，也不是无 GT 的自监督训练实验。

我独立复算的核心数值为：

| 输出 | control mAP | fresh mAP | fresh − control，pp | control R1 | fresh R1 |
|---|---:|---:|---:|---:|---:|
| Signal | 53.12938056 | 53.12938056 | 0 | 63.000000 | 63.000000 |
| fused | 51.78835925 | 51.71677883 | −0.07158042 | 61.666667 | 61.500000 |
| CNN | 50.44809997 | 50.45127183 | +0.00317187 | 61.166667 | 62.000000 |
| Transformer | 50.68304601 | 50.13669073 | −0.54635529 | 58.666667 | 58.833333 |
| Mamba | 50.44113785 | 50.88445387 | +0.44331602 | 62.000000 | 60.666667 |

- 配对 fused 分折增益：**+0.01329176 / −0.77169879 / +0.62296244 pp**。
- 配对身份 bootstrap 下界：**−0.5063560012 pp**。
- fresh fused 相对 Signal：**−1.4126017354 pp**；分折为 **+0.87987410 / −4.48012058 / −0.57349531 pp**；bootstrap 下界 **−3.0006230305 pp**。
- **配对五门 0/5，相对 Signal 五门 0/5。** “严格最高”失败来自 fused 低于 Signal；fresh fused 确实高于 CNN、Transformer、Mamba 三个角色。
- fused：**279/251/70** 条 query 的 AP 改善/下降/不变，Rank1 修复/新增错误 **9/10**，身份平均 AP 改善/下降/不变 **33/22/5**。

证据：`Q/ranking_replay.json:20-71`、`:74-146`、`:156-161`，以及我对全部原始排序和两份完整 CSV 的独立复算。

**需要完成的收束与主张边界：**

1. 将 tracker 的当前状态更新为完整 Q1 终态及 CPU PASS，保留带时间的旧观察。
2. 在最终审计记录保留 **24 个远端二进制未访问**、真实图片未访问、CPU 标量/梯度边界、累计203/203语义及预历史轨迹差异。无需为本审计重跑训练。
3. 49 个递归合同绑定的项目文件中，44 个本地工作树 SHA 直接一致；其余 5 个为 CRLF/LF 差异，**b4501fa Git blob 的 SHA 全部精确匹配合同**。不要将工作树“全部字节相同”作为表述；也不需要修改运行合同。
4. 启动提交 `b4501fa` 与 Q1 启动时 HEAD `ced43dd2` 的差异只涉及文档和证据，固定训练源码/合同没有差异；新报告第75行的澄清已通过。
5. 支持的结论是：**该固定 fresh-coordinate 方案改善了记录中的若干来源训练诊断，但未通过预登记检索门，未证明未知身份检索收益。** 不支持“缓存陈旧没有影响”“所有记忆学习无效”“已确定造成 −0.0716 pp 因果下降”，也不支持官方性能、跨数据集普遍性或 SOTA。

本次未发现假 GT、指标自归一化、虚构终态或抽样冒充完整评测的证据；总体 WARN 来自上述可复现性和研究范围限制。

## Executor disposition after review

已采纳提交HEAD区别与预历史非逐位轨迹限制。当前tracker和主交接随本次发布更新到终态，带时间旧观察保留。未修改模型、训练脚本、合同、原始证据或任何检索门；无需重跑。审查时报告SHA记录为修订前快照，后续仅增加审计链接/边界和状态同步。递归49绑定文件为44工作树原始SHA匹配、5既有CRLF/LF差异，启动Git blob全部匹配；不声称工作树所有递归字节相同。
