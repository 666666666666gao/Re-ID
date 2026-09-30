# role_global_tokens_v1 源码与预检审查

审查日期：2026-10-01。审查员：gpt-6-astra，reasoning effort=max；fresh native Codex reviewer，review_independence=same-family，acceptance_status=provisional。

**总体结论：WARN。** 在所审源码、合同和真实预检记录中，未发现具体阻止正确执行的代码错误，必改代码项为零。源码与合同的一致性检查通过；现有证据仅能支持 CPU 结构检查和 CLI 导入通过，不能证明生产 M0、正式性能、完整九端执行或科学门槛通过。

本次只写本 REVIEW.md/REVIEW.json。未改源码、合同、配置、交接或 Git；未启动 GPU、M0、正式训练、观察器或新神经重放，未安装环境，未下载权重、图像或数组。远端只读检查仅使用 gaob@172.19.12.138:2026、指定 SSH key，ProxyCommand=none、BatchMode=yes、ConnectTimeout=10，Python 为 /data/gaob/Re-ID/conda-envs/tri_reid/bin/python。历史性能没有重新完整审计。

## 判定

| 检查 | 状态 | 结论 |
|---|---|---|
| A：真实标签、完整图库、过滤 | PASS | 协议标签与实际文件名一致；三图库完整，MSVR 时间场景过滤与作者代码一致 |
| B：指标归一化 | PASS | L2 仅用于表示；AP/CMC 没有用模型自身分数作归一化分母 |
| C：文件、调用与 CPU 证据范围 | WARN | 预检日志真实且哈希一致；完整模型、direct 读出、生产 Mamba 和正式结果尚无执行证据 |
| D：初始化、结构、训练、保存/重载、收集与队列 | WARN | 静态执行链一致；生产数值与全模型初始化一致性仍待注册 M0/九端证据 |
| E：合同和对照解释 | PASS | fresh 三条件、历史结果隔离及 direct/局部语义限制写明；没有性能或新颖性晋级结论 |
| F：评价类型 | PASS | 计划中的正式检索是 real_gt；已有 CPU 随机输入与线性替身是 simulation_only 工程夹具 |

## A：标签、图库和作者协议

实际读取三个远端协议，逐条将 identity/camera/scene 与文件名解析值比较，核对 train_label_map，检查训练与测试身份不重叠；每个 split 的全部协议路径与实际图像目录条目集合相等。检查只读取目录和文本元数据，未打开图像。所有查询都有合法正例，合法正例数与协议 query_rows 一致。

| 数据集 | train / query / gallery | 有效查询 | 每查询合法正例范围 | 额外图库干扰身份 / 记录 |
|---|---:|---:|---:|---:|
| RGBNT201 | 3951 / 836 / 836 | 836 | 7–21 | 0 / 0 |
| RGBNT100 | 8675 / 1715 / 8575 | 1715 | 50–175 | 0 / 0 |
| MSVR310 | 1032 / 591 / 1055 | 591 | 1–31 | 103 / 464 |

`tools/official_three_dataset_data.py:8-18` 对训练使用真实标签映射、对检索使用真实 identity；`tools/run_correspondence_roles.py:66-108` 从全部 query/gallery records 提取，不按模型输出创建标签或缩减图库。固定数量还由 `tools/run_official_three_dataset_roles.py:58-68` 检查。

RGBNT201/RGBNT100 仅排除“同身份 AND 同相机”；MSVR310 仅排除“同身份 AND 同 scene/时间场景”，不排除不同身份的同相机/同场景干扰项。执行路径分别是 `tools/train_rgbnt100_signal_oof.py:253-268`、`tools/train_msvr310_signal_oof.py:223-238`；实际远端作者代码 `comparators/Signal-cd1b0a6/utils/metrics.py:68` 和 `:137` 使用同一掩码。新评价继承入口逐路径调用两套计分并核对，见 `tools/run_correspondence_context_identity.py:175-240`。作者函数的 camera 文档注释不能覆盖 MSVR 的实际 scene 分支。

## B：表示归一化和计分

`modeling/trifusion/role_global_tokens.py:25-27,79-94` 对每模态 global、context、fused、correction 作表示 L2 归一化；`tools/run_official_three_dataset_roles.py:230-238` 用归一化表示计算平方欧氏距离。这不是将 mAP/CMC 除以本模型的最大值或均值。

作者及独立 scorer 均由真实身份匹配排序计算 AP，按合法正例数/查询数作标准平均；乘100只转换为百分数。训练 epoch 计分与最终三个输出的 full-gallery 计分均实际接入，见 `tools/run_correspondence_roles.py:79-108`、`tools/run_correspondence_context_identity.py:214-228`。本次没有对不存在的新距离矩阵重算性能。

## C：已有证据及其边界

`PREFLIGHT_20261001.json:18-75` 记录四个子进程 exit_code=0：CPU fixture、训练入口 help、队列 help、collector help。6项新源码/合同 SHA、4个日志 SHA 与远端实际文件以及本地副本完全一致。fixture 日志 `preflight_0.log:1` 对两个 grid 记录相等初始角色状态、相等角色输出、Transformer 输入 [6,17,128]、三角色输出 [2,3,16,128]、token 投影梯度有限非零；非零投影下 token 对改变 global 有响应，static/direct 的角色模块输出不变。三个 help 日志只证明解析/导入可到达 help。

**W1（证据范围）：fixture 并未构造 GlobalTokenTriFusion。** `tools/check_role_global_tokens_cpu.py:23-24` 构造 GlobalTokenRoles，并用 nn.Linear 替代 Mamba；`:43-46` 只对 token 角色投影反传，`:59-60` 只验证 direct 的 projected_global 数值会变。它没有执行完整 backbone/read_evidence/direct correction/fused loss，也没有证明 static/direct 完整训练梯度、完整 TriFusion 状态相等、生产 Mamba、CUDA AMP、八批全参数梯度或生产 strict reload。日志明确 production_mamba_tested=false、real_training_data=false；不得称作真实 M0 PASS。

远端检查时 logs/ 和 trained-model/ 下没有 role_global_tokens 命名的活动/结果目录；与合同 `EXPERIMENT_PLAN.md:3,31,41` 的未运行状态相符。不存在新实验的正式 mAP、已完成 epoch 或已接受端点。本次没有重跑 CPU fixture 或神经计算，接受的是原有、哈希绑定的预检记录。

## D：实际执行链

### 初始化、17-token 和 direct

`tools/run_role_global_tokens.py:82-92` 先将 ContextIdentityTriFusion 绑定为 GlobalTokenTriFusion，并替换 build/evaluate/save/load；`tools/run_correspondence_context_identity.py:244-254` 再将实际 runner 的模型工厂与训练/评价绑定到这些函数。保存的 BASE_BUILD 指向原函数，不形成 build 递归。collector 的 verify 在 worker 三阶段完成后实际被调用，见 `tools/queue_role_global_tokens.py:125-129`。

`tools/run_correspondence_roles.py:32-57` 在加载冻结的纯 ReID baseline 后重新 seed42，构造实际工厂并记录全模型初始 state SHA 与 trainable 参数总数。三 token mode 没有改变构造的模块集合；`role_global_tokens.py:16-23,64-77` 使用同一零初始化、无 bias 的512→128投影和同一固定单位向量 buffer，继承状态 strict 合并，fork_rng 保持外部随机序列。依赖替换遵循相同模式，见 `patch_memory_roles.py:52-66`、`slot_competition_roles.py:36-47`、`slot_competition_fp32_roles.py:23-33`。这支持完整初始化/参数匹配的源码判断，但现有 CPU 记录仅直接验证角色模块，不能冒充已测得的全模型匹配。

`role_global_tokens.py:45-60` 将一个投影 token 与16区域拼为17-token Transformer 输入，丢弃新增 token 的输出，再将16区域按原顺序构成48-token Mamba 输入。输出的 CNN/Transformer/Mamba 区域数均为16；读出仍是1536D。全128 Patch 支持、固定位置与 independent FP32 attention 由继承链和 `slot_competition_fp32_roles.py:10-20` 固定。

direct 模式在 `role_global_tokens.py:84-89` 显式调用 projected_global，将每模态投影重复16次并经同一个 read_evidence 累加到 correction；`correspondence_evidence_readout.py:65-77` 是可微分的区域平均与同一读出矩阵乘法，无 detach。因而完整 fused 损失在源码上可通向 direct 投影、读出及 adapted global。零投影与无 bias 的读出使 direct 的新增初始校正为零；三种完整初始输出相等在结构上成立。实际生产梯度仍必须由 M0 证明。

### 真实 M0、full50 与严格重载

生产工厂固定为 `modeling/trifusion/experts/mamba.py:29-32` 的 mamba_ssm.Mamba，见 runner `:41-45`，不会使用 CPU 夹具替身。实际共享训练函数是 `tools/run_correspondence_context_identity.py:68-171`：

- M0 用第一个训练 epoch 的前8批，不查询官方检索；累计要求每个 trainable tensor 有非零梯度，逐批有限 loss/gradient，AMP 不降 scale，末尾确认冻结 baseline 未变。
- M0 保存后重新 build，要求完整 binding 相等，以新 schema/mode 严格加载，在保留的两条真实训练输入上比较输出；queue 与 collector 另要求最大绝对误差≤1e-5。它不是整个训练集上的重载证明。
- train 由新进程、不同输出目录、相同 baseline 和 seed42 从头构造；没有读取 M0 权重。使用完整50 epoch、AdamW、5 epoch warmup、原 cosine schedule、fused CE(smoothing0.1)+Triplet(margin0.3)，无 M3/auxiliary ID。每 epoch 官方 fused mAP 的 >= 比较选最后一个精确平分；只覆盖一个 best_map.pth。
- 最终 evaluate 重新 build、比对 initializer，并加载 best_map；从该同一 checkpoint 提取 fused/shared_global/joint_local 全量距离，核对作者计分与选中 epoch。

checkpoint 自有 schema 为 trifusion-role-global-tokens-v1。`tools/run_role_global_tokens.py:42-68` 绑定 dataset/seed/protocol/baseline/variants/condition/token_mode/memory/attention/dtype，并要求非冻结保存 state 的键集合完全相等、load_state_dict(strict=True)。相同形状的其他 mode 也会被拒绝。冻结上游由 SHA 锁定的 baseline 严格重建，非冻结参数和 buffer 由 checkpoint 保存；训练/检索 receipt 保留旧 context schema，但明确新 architecture 与 mode，collector 按同一设计检查，未发现 schema 接线不一致。

### 九端 collector 和队列

`tools/collect_role_global_tokens.py:19-141` 接收的字段与实际 runner 保存字段对应：完整 M0、50 epoch、选中 epoch、checkpoint/距离 SHA、新 schema/mode、initializer 源哈希、协议标签、三路全图库尺寸/有限值、独立 CPU 排名，以及无 auxiliary 的逐步损失重构。`:144-181` 固定三条件×三数据集的九端顺序，并在每数据集已完成条件之间核对完整初始模型 SHA 与参数总数；缺失/未完成项不能变成 VERIFIED_COMPLETE。coordinate 在 `queue_role_global_tokens.py:185-189` 要求9/9才写最终 accepted_matrix。

`queue_role_global_tokens.py:69-79,91-124` 的真实阶段顺序是 m0→train→evaluate，每阶段 process.wait() 后才记录对应退出与继续；继承 scheduler 在 `queue_correspondence_refinement.py:135-181` 用 Popen.poll() 取得/回收实际 worker 退出，失败后停止新 pending 分配、等待既有活动任务，无 retry 分支。POLL_SECONDS=240（同文件 `:23`），只调度0–3号中显存占用<500MiB且未由自身占用的卡，没有抢占或重置。

`queue_role_global_tokens.py:139-176` 要求完整六端前驱和指定 accepted_matrix SHA，核对全部213项前驱来源后，锁定本次源码、合同、全部本地模型/上游 Python、三协议与上游配置；worker 每阶段前再次核对。只读实际核对前驱 campaign 为 COMPLETE，6个 worker 均 COMPLETE/exit0，accepted_matrix 为6/6，SHA 与新合同相等，213项远端来源无不符。三实际 baseline 文件 SHA 均等于 `queue_correspondence_roles.py:21-25` 固定值；CLIP 实际 SHA 另记于 JSON。未读取旧单卡/旧 IP。

**W2（结果解释与成本字段）：** collector 的 `training_and_epoch_eval_seconds`（`:133-134`）仅覆盖正式训练加逐 epoch 评价，不含独立 M0/最终评价和之前 ReID 训练。最终总成本需结合现有 campaign 的真实阶段起止时间；不能只用该字段声称总 GPU 成本。新 collector 也没有自动产生合同全部 repairs/new-errors、identity-equal AP 与全部50轮分析；这些是完整结果报告的待办，不是已完成证据。

## E：合同与科学解释

合同 `EXPERIMENT_PLAN.md:13-25` 正确披露：static 输入变动维度不同，参数数相同不等于有效容量相同；direct 共享投影和既有 readout，只是具体 bypass 对照，不与 token 计算等价。shared_global 是受 M1 adapters 影响的 adapted global；token/direct 的 joint_local 是归一化总 correction，direct 显式含 global 投影，不能解释为独立训练或纯局部身份信息。字段名虽保留，未来 accepted_matrix/图表须连同该语义读；`run_role_global_tokens.py:32,78` 已在 initializer/official receipt 给出说明。

新 queue 创建九个 fresh 输出，不复用历史16-token endpoint，合同 `:23` 明确禁止替代。上一板 report `results/SLOT_COMPETITION_FP32_COMPLETE_2026-10-01.md:24,61` 保持 advancement gate FAIL 和发展性单 seed 边界；其 accepted_matrix 只用于前驱锁，未作为新17-token static 控制。

合同 `:37` 的门槛仍是 token 在三数据集的 mAP 均超过 static/direct、每项 Rank-1 不下降，并在 RGBNT201/MSVR310 对每个控制至少+0.5 mAP。源码中的 COMPLETE/VERIFIED_COMPLETE 是执行与证据完整性，不能等同该科学门槛通过。本审查不作 gate、novelty、SOTA、训练稳定性、三角色必要性或 untouched-test 显著性结论。

## 仍需的证据与必改项

**必改代码项：无。** 未提出 fallback、额外异常处理、兼容层、假设性分支或无关重构。

W1须由原合同规定的各端真实生产 M0 解决：8批、全 trainable tensor 累计非零梯度、冻结状态、严格重载。全模型跨三条件初始状态与参数数须以真实 initializer/collector 证据确认。之后即使训练完成，也须等完整9/9与正式分析，才能判断预注册门槛；不得用本 WARN 或 CPU_STRUCTURE_PASS 代替这些步骤。

W2是最终报告时的范围要求，不阻止当前源码正确执行，也不要求新实验或扩大训练矩阵。所有性能仍为 NOT_RUN/NOT_ESTABLISHED。

## 证据锁与限制

审查的30个跨端源/证据文件中28个逐字节相同；mamba.py 与 criterion.py 仅本地 CRLF/远端 LF 不同，文本规范化 SHA 相等，未修改文件。24个所涉 Python 源 AST 解析通过。6项新源码/合同及4个预检日志均逐字节吻合。完整本地/远端 SHA、三协议、baseline/CLIP、前驱和本次只读元数据检查记录在 REVIEW.json。

前驱 accepted_matrix SHA：e3433db691dbb1070595386c542f27782cd88b45f5f6faa8660b246e7c2fd4b8。
预检 receipt SHA：3a6d62a4c42f1e7045905f14e7b8c7c718a9ff7e7fcdd4f96897f98635f20c76。

本审查只复核所述源码/元数据/原有预检；没有独立执行生产模型、重算历史完整性能或验证不存在的未来产物。fresh 仅指独立上下文，仍是 same-family/provisional，非跨模型家族接受。

