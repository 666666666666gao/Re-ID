# MSVR310 role-set Q1 完整实验独立审计

审计完成时间：2026-09-08T23:14:27.926246+08:00。审计角色：fresh-context Codex reviewer，请求模型 `gpt-6-astra`、reasoning `max`；review_independence=`same-family`，acceptance_status=`provisional`。本会话无法另外证明底层服务的模型身份；不声称跨模型家族接受。

**Overall verdict: WARN。Integrity status: WARN。Engineering: PASS（有明确覆盖边界）。Deterministic checks: PASS。Scientific: Q1_FAIL，两组均 0/5，next_phase_qualified=false。**

未发现已核对数值的虚构、指标自归一化、来源/留出身份交叉或门槛篡改。此结论不等于完整训练轨迹已经被独立重建：逐步图像增强、特征生成、参数梯度、优化器状态和全程 RNG/buffer 行为仍有运行见证边界。微小配对均值增益不能升级为稳定有效性结论。

本报告只审计完成的 role-set Q1 及其当前输入链。M0 文本纳入接收完整性和阶段依赖检查，**不重做 M0 审计，也不复审/重跑早先封存实验**。另行提出的 positive-coverage 诊断不合并到本核心结论。

## 输入、版本与可复现收据

以下路径别名只用于让 file:line 引用简洁；每个别名均唯一对应绝对目录：

- `R/` = `C:/Users/gb/.trifusion_github_publish_22c3bee/`；代码引用均可在本审计 `snapshots/` 的同相对路径找到。
- `I/` = `C:/Users/gb/.codex_tmp/role_set_q1_complete_20260908/`。
- `A/` = `C:/Users/gb/.codex_tmp/role_set_q1_independent_audit_20260908/`。
- `S/` = `A/remote_texts/signal/`，是实际远端 Signal 源码文本快照。
- 远端 run = `/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739`。

完整读取、UTF-8 解码并解析 57 份注册文本，81,195,809 B；全部注册大小/SHA一致。JSON/JSONL 的全部节点、全部 1,560 条 Q1 训练记录和其完整内存元数据进入检查；并非仅抽样头尾行。M0 仅做完整接收/解析。原始大文本没有全部打印到对话；`local_intake_inspection.json` 记录逐文件大小、哈希、行数、键和完整解析统计。

远端独立计算 117 项递归注册输入哈希，全部匹配；其中重复引用保留为独立绑定记录。比较 59 个注册项目代码/配置文件在 wrapper commit `26c97390704c629237687d263b4381f5584cbe97`、Q1 summary commit `23a5b48d019b7351630bc670d14399d413fc767e`、当时远端 HEAD `864a2c9fd33399c60a6990cbb6e4c9ffe609da14` 的字节，全部一致。因此 commit 字段变化有实际代码内容证据解释，不能仅凭版本字符串不同判定实验代码漂移。Signal commit/diff、BASELINE config、protocol 和阶段先后另由 `remote_final_binding_check.json` 直接确认。

当前 Q1 summary SHA：`fc6493b643ab378e2941fb335ac5ebf3093c0d31673bdda7d4564db724598de2`。config SHA：`5e5ad4663f4c3c58e11b81b475048d6ab6c53675ac2dd0ade60c84f07a62986d`。Q1 CPU receipt SHA：`b19622644d45007560e9f930ea52dffea921044f456f8f82d4fc9935d5d55726`。

本审计的接受证据来自独立计算，不来自原 verifier 的 PASS 字符串：

| 独立计算范围 | 实际结果与边界 |
|---|---|
| 递归输入及当前 Q1 文件哈希 | 117 项注册输入 + 原 Q1 CPU 清单全部 43 个文件匹配，含 6 个终点 checkpoint、6 个 retrieval 数组、6 个距离二进制与完整文本 |
| 数据协议 | 全 1,032 条真实文件名三模态记录、155 身份、确定性分折、全部合法 query 和图库重建；没有打开图像 |
| 训练证据 | 1,560 更新记录，116,501,504 个保存距离元素，完整队列/年龄/身份与 scene 元数据、四空间提议与位置集合、14 项 loss 加权账本 |
| 目标数值 | 最大全部目标误差 3.73305132362e-08；总 loss 账本最大误差 4.71870104768e-07 |
| 距离坐标导数 | 全 1,560 行的 hard 与 role-set 两目标，分析式 dL/dD 对实际目标代码 CPU autograd 最大绝对差 `3.1044085878340066e-10`；这不是特征或参数梯度重放 |
| 最终检索 | 全 30 个 fold/endpoint/output，2,069,520 个距离/排名位置；完整 AP、首正例名次、R1/R5/R10、60 身份聚合和两组门槛 |
| 独立距离公式 | 从最终特征用 Float64 范数与矩阵乘法复算，最大差 1.18152844153e-06；没有把它标成原 56 线程 FP32 addmm 位级重现 |
| 变化与来源表 | 3,000 条 query-output、300 条 identity-output 的每个 CSV 字段，24 个 source phase block、14 项均值以及成本聚合全部一致 |

主要实现/收据：`remote_independent_q1_replay.py` / `independent_remote_replay.json`，`remote_distance_gradient_audit.py` / `independent_distance_gradient_audit.json`，`local_claim_checks.py` / `independent_local_claim_checks.json`。成功远端主检查用 2 CPU 线程，约 19.897 秒；距离导数扩展约 10.886 秒。模型 forward=0、训练/optimizer update=0、图像读取=0、GPU 初始化=0；二进制始终留在远端。只向指定审计目录写代码、文本、CSV、报告和收据。

## A. Ground-truth provenance — PASS

GT 来自数据集路径/文件名给定的 identity、scene、camera，而不是模型输出。协议构建器首先绑定既有 label evidence SHA，再读取 `bounding_box_train` manifest；对每条记录重新解析文件名 identity/scene/camera（`R/tools/build_msvr310_train_oof_protocol.py:12-55`）。实际上游 MSVR310 parser 也按相同字段读取（`S/data/datasets/msvr310.py:65-86`）。独立远端检查枚举了 train/vis 下全部实际记录，核对三模态文件存在及名称/目录标签，无缺项、额外候选或改标签迹象。

分折按是否有跨 scene 正例对排序身份分层 round-robin；图库含每个留出身份的所有记录，只有无合法跨 scene 正例的记录不作 query。全量重建匹配，不是根据模型分数挑选困难/容易身份。来源与留出在每折的完整记录路径和原始身份上互斥（`R/tools/build_msvr310_train_oof_protocol.py:57-123`）。

| fold | source 身份 / 记录 | held-out 身份 / gallery | 合法 query / 身份 | 仅作干扰项的身份 / 记录 |
|---|---:|---:|---:|---:|
| 0 | 103 / 672 | 52 / 360 | 210 / 20 | 32 / 150 |
| 1 | 103 / 683 | 52 / 349 | 207 / 20 | 32 / 142 |
| 2 | 104 / 709 | 51 / 323 | 183 / 20 | 31 / 140 |

总 gallery 1,032、合法 query 600、query 身份 60；95 个单 scene 身份的 432 条记录保留为图库干扰项。距离只在本折计算，最终汇总 query AP，未跨折比较特征。

完整模型路径的 source 隔离：`records_for` 仅依据当前 fold source/gallery 索引构造路径，source 分类标签由当前折 label map 提供（`R/tools/train_msvr310_signal_oof.py:68-99`）；loader 不调用上游 dataset 工厂创建官方 query/gallery。模型从该折 source-only Signal checkpoint 严格加载，逐项断言 source/held-out IDs，再从 seed42 新建角色（`R/tools/train_msvr310_trifusion_oof.py:27-55`，`R/tools/build_v12_complete_path_oof_targets.py:384-411`）。我另外检查了三份 B0 checkpoint 的 ID 绑定以及原 B0 全部 1,950 个训练 step 的采样索引均属于本折 source；没有复审其旧科学结果。

当前 Q1 所有 1,560 个 source batch 以及历史候选均属于 source。最终评估才构建本折完整 gallery（`R/tools/train_msvr_role_set.py:319-354`）；各阶段实际顺序为 T0→M0→M0_CPU→Q1→Q1_CPU，M0/CPU收据哈希及先后关系匹配。全程 `official_image_reads=0` 字段本身是初始化值，不是 OS 级访问审计器；代码访问路径、采样/图库 manifest 和阶段证据支持本轮未使用官方测试，但不能声称独立证明了历史进程的每一次系统调用。图像内容真伪、原始数据发布者的标注质量及预训练语料成员关系不在本次验证范围。

## B. Score normalization — PASS

检索先对 embedding 做标准 L2 归一化，再计算平方欧氏距离（`R/tools/train_msvr310_trifusion_oof.py:210-218`）。这属于特征几何定义，不是把 AP/CMC 除以模型自己的最大/均值以放大成绩。AP 的分母是合法正例数，mAP/CMC 的分母是合法 query 数（`R/tools/train_msvr310_signal_oof.py:223-238`）；身份 bootstrap 采用抽样整身份并保留 query 权重（`R/modeling/trifusion/signal_preserving_v13.py:253-278`）。没有发现评价指标的自参照归一化。

实际上游 `eval_func_msrv` 使用 same-ID AND same-scene 过滤、全部不同身份干扰项、标准 AP/CMC（`S/utils/metrics.py:13-108`）。当前 pipeline 真实调用它，并把相同距离交给独立显式 query scorer，核对所有输出的上游差值（`R/tools/train_msvr310_trifusion_oof.py:219-241`）。本审计再次从保存的排序重算所有 AP/CMC，不依赖其 PASS。

排序采用已安装 NumPy 的默认 argsort，训练提议采用第一个 argmin；没有另加稳定排序或 ID 级 tie-break 改协议。所有保存排序与当前远端 argsort 精确一致。合法检索中的少量相同距离没有正负标签混合 tie（混合 tie 数=0），因此本次观察到的精确距离 tie 不改变 AP/CMC；不承诺任意平台的近等浮点排序逐位一致。

## C. Result existence and numerical agreement — WARN（实际文件/数值 PASS；tracker 滞后）

全部六端固定 epoch20/260 更新结果、checkpoint、检索数组、完整排名存在，且当前重新哈希一致。原 terminal pipeline 五阶段退出均为 0，所有原 PID 当前不存在。文件证据与 `Q1_FAIL` 一致；没有需要不存在文件来支撑的已核对科学数值。

| 输出 | control mAP | role_set mAP | 配对差值 pp | control R1 | role_set R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 53.129380561 | 53.129380561 | +0.000000000 | 63.000000 | 63.000000 |
| fused | 52.392266133 | 52.490152084 | +0.097885952 | 59.833333 | 60.333333 |
| cnn | 49.998831962 | 50.004771399 | +0.005939436 | 59.833333 | 59.333333 |
| transformer | 50.389904027 | 50.253322215 | -0.136581812 | 59.333333 | 58.333333 |
| mamba | 51.007376534 | 50.970124010 | -0.037252524 | 59.166667 | 59.000000 |

配对 fused 三折差值：`-0.02243479335133003 / +0.22733669166214554 / +0.08953105253196014` pp；seed42/10,000 整身份重采样的 2.5% 下界为 `-0.02059987959543969` pp。候选相对 Signal 的 fused 差值 `-0.6392284763979887` pp，下界 `-2.5799538402211346` pp。候选高于三个角色，但低于 Signal。

两组五项全部重新计算：

| 既有门槛 | control→role_set | 候选→Signal |
|---|---|---|
| fused 至少 +1 pp | FAIL：+0.097885952 | FAIL：-0.639228476 |
| 三折 fused 增益均非负 | FAIL：fold0 负 | FAIL：fold1/2 负 |
| 三个完整角色均非负 | FAIL：Transformer、Mamba 负 | FAIL：三个角色均低于 Signal |
| 身份 bootstrap 下界 >0 | FAIL：-0.020599880 | FAIL：-2.579953840 |
| 候选 fused 严格高于 Signal 和三个角色 | FAIL：低于 Signal | FAIL：低于 Signal |

门槛来自注册合同 `R/refine-logs/msvr310_role_set_v1/TRAINING_PLAN.md:35-41`；实际聚合调用链见 `R/tools/train_msvr_role_set.py:288-293` → `R/tools/train_msvr_instance_memory.py:250-256` → `R/tools/train_msvr310_source_style.py:196-224`，候选对 Signal 五项见 `R/tools/train_msvr310_trifusion_oof.py:244-285`。身份 bootstrap 是同一训练种子的样本不确定性，不是多种子训练可靠性。

全 600 query 的 fused AP 改善/下降/不变为 209/233/158；R1 修复3、新错0。60身份改善/下降/不变为28/25/7。完整两份 CSV 每个字段（含正例末位、最近负例记录/身份/scene）都独立匹配。

WARN 原因：本次读取的 `R/refine-logs/msvr310_role_set_v1/EXPERIMENT_TRACKER.md:3` 仍写 22:03 时 5/6 完成、独立审计待完成；`R/AGENTS.md:3-9` 亦为旧运行状态。因此 tracker 不是当前 DONE/终态记录。与之相比，实际结果已于 22:38–22:39 完成，执行侧终态报告明确是 Q1_FAIL、审计待结论，数值一致。应同步文档当前状态，不应删除旧观察或把科学失败改成成功。

## D. Active versus dead metric code — WARN（历史旁路存在；当前所报指标均有调用）

活跃链是 `train_msvr_role_set.run`→`old.extract`（`R/tools/train_msvr_instance_memory.py:39-50`）→`exact_signal_forward`→本折全部 features→`evaluate`→`scene_scores` 和上游 `eval_func_msrv`→`paired_summary`。summary、receipt、ranking 中全部所报 AP/CMC/gates 都能连接到真实执行路径，并被本次重算确认。

但导入模块里仍含未用于本 Q1 的旧 `train_roles`/旧 runner、另一 `extract`/`evaluate_gallery`，以及上游通用 camera evaluator `S/utils/metrics.py:111`、`R1_mAP_eval` 的其他可视化/通用指标路径。这些不是本 Q1 的指标来源。按技能“存在未调用指标函数即 WARN”的字面规则保留 WARN；没有把未执行 metric 函数当作已产出结果。`ViewFields`/`encode_graph`/`compare` 复用活跃，旧诊断模块自己的 run/main 不活跃。未重新运行旧科学实验验证这些旁路。

## E. Scope and claim limitations — WARN

实际是一份 MSVR310 官方训练 split 内的完整路径身份 OOF 比较：一训练种子、三折、两端、固定20 epoch、每端260更新。共有155图库身份、60合法 query 身份；source/held-out覆盖的实际 scene 列表已保存在 `independent_remote_replay.json`。不能称为多个数据集、多个训练种子、全 Goal、官方测试、SOTA 或独立盲测确认。既有 held-out split 在开发链中重复使用；单次身份 bootstrap 不校正方法迭代选择，也不提供训练随机性重复验证。

主要 paired 条件是同初始化绑定、同采样记录、同三模态像素 SHA；Q1 初始角色权重未作为独立初始 checkpoint 保存。日志与模型构建源码支持初始一致，但本审计没有在 CPU 新建该 CUDA 模型并重新算初始状态。因此“绑定相同”可独立核对，“实际初始角色张量被独立重建并逐位比较”不可声称。

而且三折 loss 都从 step2 即不完全一致，前65步总 loss 最大差分别为 `0.0014867782592773438 / 0.002425074577331543 / 0.0011173486709594727`。这些真实差值来自完整两端训练记录；其具体数值非确定性来源本次没有定位。合同已明确不预先承诺独立轨迹位级相同（`TRAINING_PLAN.md:19`）。这一现象不是注册违约，但阻止将 +0.0979 pp 单独解释成严格隔离的确定性因果效应。

## F. Evaluation-type classification — PASS

- Q1 全部 AP/CMC、身份聚合和科学门槛：`real_gt`，具体是 `train_internal_identity_oof_not_official_test`；没有以模型产生的 reference 当作检索 GT。
- 当前/历史编码相等、状态哈希、RNG/buffer/重载一致性：`self_supervised_proxy` 的工程一致性检查，无 GT 性能语义。
- 注册 T0 随机特征上的 loss/链式法则合成检查：`synthetic_proxy` 的明确数学检查，不是数据集性能。元数据 queue replay 使用真实标签但不做检索。
- M0 工程结果是保留的阶段依赖，不构成本审计新评估；不能把其过拟合或直接图检查当作正式性能。

## 队列、负例集合、损失与导数的细节

队列按真实 record 唯一化、保留最近一次视图，年龄大于8清除，读取时移除所有当前 record 的历史副本；当前 batch 的重复曝光仍是不同位置。实现 `R/tools/msvr_instance_memory.py:8-37`，原 field 的位置映射取最后一次重复记录位置 `R/tools/msvr_freshness_probe.py:48-60`。独立复算完整 membership/order/stored_step/age/identity/scene，与1,560行逐项相等。实际队列容量峰值 fold0/1/2 为391/408/406，过滤后历史候选峰值356/372/362，实际年龄1–8。512是上限设置，当前Q1没有填满512，也没有触发按容量的淘汰；不能把它说成满容量压力测试。当前每batch重复记录曝光为3–41个，不能将64个 anchor 位置当作64条独立图片。

第1–65步两端使用原 batch-hard；第66步开始换候选目标并写入队列，但当步读取到的 history 仍为空；第67步首次有历史候选。`R/tools/train_msvr_role_set.py:88-118,146-160,264` 的顺序与全部原记录一致。无历史与无干预是两个不同条件。

正 mask：同 identity、当前排除自身位置；训练不要求正例跨scene。负 mask：不同 identity，包括同scene负例；不加入 scene/identity 配额。fused加三个完整4608D角色分别提议最近负位置，按位置并集去重。当前重复record两个位置可能都保留，不是按record或负identity去重。`R/tools/msvr_role_set_relations.py:12-56`。

记当前/历史并集最难正距离为 hp、最难负距离为 hn，四提议去重集合为 S，fused第一个最近负位置为 n0：

`Lhard_i = ReLU(hp - hn + 0.3)`；
`Lset_i = [Lhard_i + sum_(n in S\{n0}) ReLU(hp - D_fused[i,n] + 0.3)] / |S|`；最后对64个anchor平均。

角色提议索引 detached，所有实际 hinge 均在 fused 未平方欧氏训练空间。第一个 hard 项使用原 current/history split max/min 导数，避免把其 tie 语义换成 argmin 单位置导数；current 内 max/min 取首位置，current/history 相等极值由 maximum/minimum 平分导数。完整实际距离中没有观察到 current/history 极值 tie，所以该 tie 分支的语义来自源码检查，不能宣称 Q1 实际覆盖了它。全部实际行的 dL/dD 已另与解析式核对。

其余13项是原 fused ID + 三组 full-role ID/Triplet + 三组 residual ID/Triplet 中除 fused Triplet 之外的项，权重保持 `ID_FUSED=.25, TRIPLET_FUSED=1, ID_BRANCH=ID_RESIDUAL=1/12, TRIPLET_BRANCH=TRIPLET_RESIDUAL=.25`。七组真实分类/Triplet监督见 `R/modeling/trifusion/signal_preserving_v8.py:690-742`；加权入口 `R/tools/run_signal_preserving_v5.py:99-142`。全部数值账本重算一致，但其他13项数值是保存的前向见证，没有由图像和模型独立生成。

## 来源覆盖与权重：数据支持到哪里

候选预热后3折585步/37,440 anchor曝光：选81,645负位置，平均2.180689/anchor；比fused最近负多44,205位置，其中38,553历史位置。21,500 anchor曝光带来额外负identity；额外负identity曝光27,235。还存在521个“不同位置但同record”的重复选择曝光、16,449个“不同record但同负identity”的冗余曝光。因此覆盖不能只用位置集合大小代替身份覆盖。

额外 active hinge 37,730次，涉及25,848 anchor曝光。执行侧文本只有每anchor的active总数，确实不能凭该总数归因给具体角色；**但完整远端四距离矩阵可进一步独立重建活跃位置**。本审计用它确认：预热后候选CNN/Transformer/Mamba独有提议且hinge活跃的曝光分别为 `11831 / 12731 / 10227`。它们是该位置在四提议中独有、该 scalar hinge>0的数学事件，不是对相应角色参数净梯度、净学习收益或独立训练样本数的证明。该扩展在 `independent_distance_gradient_audit.json`，没有加入任何新晋级门。

对同一固定几何，额外负例不比最难负例更难，故同hp下 `Lset_i <= Lhard_i`；mean同时减少最极端关系的系数。候选预热后 `1/|S|` 的平均为 `0.544589120`（不是1/平均集合大小）。因此增加覆盖与最难关系降权同时发生，不能仅归因“角色多样性”。

预热后 candidate 总loss 1.047168680，小于 control 1.058057711，但采用了不同fused项；相同定义的expanded-hard均值反而为candidate0.166097508、control0.165848547，role-set均值为candidate0.155443942、control0.155318383。最后5epoch同样分别0.141739062>0.141091582、0.131121171>0.130590732。不能用不同目标的total下降推出已解决困难来源关系或泛化更好，也不能靠这几个聚合值识别唯一失败原因。

## 当前/历史参数梯度、RNG与优化器的验证边界

代码包含的梯度路径是正确分域的：当前总loss backward覆盖203个可训练张量；历史候选叶子偏导只来自所选 fused Triplet，通过重新编码的历史group VJP加入189个encoder参数。剩余14个可训练张量是当前路径的七个neck weight与七个classifier weight；额外历史目标不经过neck/classifier。固定fusion无可训练参数、冻结Signal不接收历史导数。证据 `R/tools/train_msvr_role_set.py:76-96,148-160,189-247` 与 `R/modeling/trifusion/signal_preserving_v8.py:545-611`。

原view缓存仅保存冻结field和原encoder-entry RNG，当前坐标由现模型重新编码；group按原stored_step重放完整64个record，保证batch/RNG语境。history叶子偏导在当前特征detached下求，再按唯一record的最后视图位置施加；原history特征不作为训练参数。加入历史VJP前，代码断言当前梯度、RNG与buffer不变，之后断言当前+历史之和逐位一致，unscale再一次optimizer.step（`R/tools/train_msvr_role_set.py:195-247`；`R/tools/probe_msvr_history_candidate_gradients.py:43-66`；`R/tools/probe_msvr_role_set_gradients.py:29-50`）。

CPU能独立证明：完整group元数据、保存上游norm非负有限、所有norm/cosine/difference代数自洽、zero/nonzero norm对应的group选择和成本、两端参数绑定、6个终点完整/冻结状态哈希、最终baseline特征/距离与source B0逐位相同。不能独立证明：每步实际编码等于当前模型、原增强像素SHA确由对应像素产生、全部真实VJP张量、优化器每步状态变换，以及每个运行断言确实覆盖了所有未保存tensor。Q1没有任何direct full-graph参数梯度独立比较；代码的直接比较只在M0 capacity首历史group触发（`R/tools/train_msvr_role_set.py:166-184,226-233`）。本次没有再运行M0。

实际203/203是累计曾有非零梯度，不是每步203全部非零。189也不是另一个模型的总可训练张量数。历史候选直接 dL/dD 核对和最终检索重算提升了确定性证据范围，仍不能改称全程参数梯度重现。

## 成本、工程结论与记录问题

Q1固定1,560更新，M0既有248更新使新训练合同总数1,808；本审计新更新0。每端三折新鲜角色record-forward292,800，历史VJP control265,600、candidate284,096（+6.963855%）；fit epoch时间之和6108.169539s→6357.269051s（+4.078137%）。峰值allocated两端均6118.908691MiB；reserved值逐端保存在独立收据。Q1整个stage为12597.781153秒，包含边界上的模型构建/重载/检索时间，不能将fit时间当整个pipeline耗时。

这些forward计数含每个group重放64记录的整组成本；不是独立图像数量，也不是精确FLOPs/每个内部checkpoint重算次数。history的训练开销不能用“无新推理参数”掩盖。远端观察到本run全部119文件约1,174,811,942B，低于注册最大预期3GiB。新checkpoint不含queue或optimizer状态，支持固定终点核验；它们本身不是可恢复每步AdamW轨迹的快照。

工程PASS的含义：已注册六个终点完成、固定训练长度、保存证据有限数值与完整性、匹配的paired输入、终点状态/检索可核验；原非零累计覆盖、AMP、RNG/buffer及历史参数grad行为仍按有范围的运行见证解释。没有完整模型重训练或参数轨迹重放，故语义/完整性评审保留WARN/provisional。

## 必须保留的审计尝试

1. `remote_independent_q1_replay.attempt1.py/.json/.stderr`：审计器把七个BatchNorm neck的21个运行buffer错误计入frozen集合，报冻结hash断言。修正来自实际v8 neck定义与frozen_state_sha参数域；实验输入未改。
2. `remote_independent_q1_replay.attempt2.py/.json/.stderr`：全部计算走完后输出JSON遇NumPy int64不能序列化，未留下接受收据；不能当成功结果。只修正审计器序列化。
3. `remote_independent_q1_replay.py` 与 `attempt3.json/.stderr`：全部检查成功，stderr空。
4. `remote_distance_gradient_audit.attempt1.py/.json/.stderr`：真实无history的warmup中，审计器请求对未使用的空history tensor求导而失败。按实际H=0只求current距离导数后，attempt2成功。
5. `remote_distance_gradient_audit.py` 与 `attempt3.json/.stderr`：在已通过导数核验上增加完整数组可重建的角色独有active位置统计，成功；不是训练重跑。
6. `local_claim_checks.attempt1.log`：全量作者数值/CSV检查一次通过；collector和最终binding检查成功。工具显示曾截断大文件输出，原stdout/快照完整保留，审计不依赖被截断显示。

所有失败是本次审计工具的真实尝试，原样保留；没有清理失败文件或声称无失败。命令外层PowerShell可能在最后Write-Output后返回0；判定以transport的实际`LOCAL_TRANSPORT_EXIT`、远端stderr和有效JSON终态联合为准。

## Claim impact 与后续处理

| 声明 | 审计判断 |
|---|---|
| 六端完整Q1完成且其检索/门槛数值可复算 | supported，工程与确定性范围如上 |
| 角色可提出额外record/identity负关系，部分额外hinge活跃 | supported，训练曝光/保存几何的描述性结论 |
| 完整新鲜坐标、历史候选VJP设计已执行 | supported with qualifier：代码及有范围的运行见证，未独立重现全程参数梯度 |
| 新目标稳定提升未知身份检索、达到晋级条件或超越Signal | unsupported；两组均0/5，候选低于Signal |
| 效果单独由角色多样性/来源覆盖产生 | unsupported；最难关系权重同时变动，且实际paired轨迹预热即分歧 |
| 已独立证明所有训练反传/原像素/RNG逐位一致 | unsupported；仅有限直接检查/运行断言和保存产物可核对 |
| 官方测试/SOTA/三数据集完整Goal达成 | unsupported，本轮完全不提供此证据 |

处理项仅是记录与声明校正：同步tracker的实际终态与本审计路径；保留Q1_FAIL及两组门槛；原样保存失败尝试和远端必要产物；精确写明可重建与运行见证边界。**本审计不要求补跑、改变实验、扫超参或用另一个checkpoint挽救结果，也不指定后继科学结论。**

机器可读报告：`EXPERIMENT_AUDIT.json`。全部输入哈希：`audited_input_hashes.json`。完整原文：`full_response.md`。请求/响应/模型路线元数据在本目录`trace/`，未写入仓库或提交。
