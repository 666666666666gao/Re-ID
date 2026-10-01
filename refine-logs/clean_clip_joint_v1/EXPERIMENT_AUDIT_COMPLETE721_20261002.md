# Experiment audit — complete six endpoints

Auditor: fresh gpt-6-astra/max; same-family/provisional. Full response follows; source and saved-tensor checks are distinct from unreplayed runtime assertions.

**总体结论：WARN。六端执行完成和现存结果文件的完整性核验通过；预登记 J1 为 FAIL。未发现需要撤销六行结果的实质矛盾，但不能据此宣称三角色机制有效、初始化具有独立因果收益或达到 SOTA。**

本次为 fresh、same-family/provisional Codex 审查。仅读取文件、检查哈希/源码/字节码、解析日志和在 CPU 加载保存张量；没有导入或运行模型，没有重放训练、推理、采样器、评分器或报告程序，也没有修改实验及发布文件。

以下引用采用固定路径缩写：

- `S` = `C:/Users/gb/.codex_tmp/clean_clip_audit_source721_20261002`
- `P` = `C:/Users/gb/.trifusion_github_publish_22c3bee`
- `R` = `/data/gaob/Re-ID/Trifusion`
- `U` = `P/logs/clean_clip_complete721_20261002/raw/results/clean_clip_joint_complete_20261002/SUMMARY.json`
- `Q` = `S/refine-logs/clean_clip_joint_v1/EXPERIMENT_PLAN.md`

**审计覆盖与证据绑定**

逐行阅读了目录列出的全部评价脚本，以及实际训练、初始化、模型构造、数据加载、完整保存/重载、队列和报告调用链；区分了被当前入口覆盖的旧构造函数及未调用的旧评价代码。读取了六端原始 training/evaluate 终端文本、各阶段 campaign、完整 training.json、全部 step traces、六个 M0 的回执及原始终端日志、完整报告和 COMPLETE721 tracker。

确定性检查结果：

- 目录列出的 **98 个本地文件**尺寸及 SHA256 全部相符。
- **233 个封存源码/配置/协议文件**同时匹配 SOURCE_INTAKE、正式 manifest 和远端当前字节。
- 完整报告的 **48 个输入哈希**全部核实；其中 15 个未在当前本地增量包中的输入通过远端只读核实。
- manifest SHA256：`4683c9b78d4beb6f76f7910a22b1a10c02978519aba1206549fd4260f2252701`。
- 完整 SUMMARY SHA256：`784e3938a0203ea043543804c6e2933b43877405d7ca624116d59e59d35538fd`。
- 报告程序及独立分析程序虽不在训练的 233 文件集合中，但其远端字节、本地副本、waiter 登记及报告输入哈希一致。见 `U:1293`、`U:1304`，以及 `P/logs/clean_clip_complete721_20261002/raw/logs/clean_clip_analysis_waiter_20261002.json:6`。
- 实际 Signal 仓库 HEAD 为 `cd1b0a672d1fe642e7608731cb4899a19dda7d51`。作者 metrics.py、三个数据集解析器、主要 CLIP/Signal 构造文件与该提交的 Git blob 直接比对一致。`make_model_clipreid.py` 唯一差异是把原机器的硬编码权重路径改为 `cfg.MODEL.PRETRAIN_PATH_T`。

**A．真实标签与完整评价集合：PASS**

标签来自真实数据文件名及原作者规则，没有使用模型预测生成 GT：

- RGBNT201：身份取文件名前六位，camera 取文件名字段并减一；query/gallery 均为完整 test。见 `S/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:26`、`:79`。
- RGBNT100：身份/camera 按作者正则解析，使用原 query 和 bounding_box_test。见同目录 `RGBNT100.py:29`、`:65`、`:76`。
- MSVR310：使用 query3；身份、camera、scene/time-block 取原文件名字段。见同目录 `msvr310.py:33`、`:82`。

本次重新解析了全部协议行的标签和路径，逐个确认对应文件存在且非空，并将实际分割目录中的完整文件集合与协议集合比较：

| 数据集 | train 记录/身份 | query 记录/身份 | gallery 记录/身份 | gallery-only 身份/记录 |
|---|---:|---:|---:|---:|
| RGBNT201 | 3951 / 171 | 836 / 30 | 836 / 30 | 0 / 0 |
| RGBNT100 | 8675 / 50 | 1715 / 50 | 8575 / 50 | 0 / 0 |
| MSVR310 | 1032 / 155 | 591 / 52 | 1055 / 155 | **103 / 464** |

MSVR310 的全部 gallery-only 干扰身份保留。训练身份与 gallery 身份不相交。全部 query 在规定排除规则之后仍有真实正样本，因此作者评分器的“无正样本 query 跳过”分支没有形成隐性子集。

协议定位：`S/logs/official_three_dataset_protocols_20260923/RGBNT201.json:78310`、`RGBNT100.json:218979`、`MSVR310.json:38539`；完整 records 分别始于 `:184`、`:63`、`:168`。

六个距离文件中 query/gallery 的身份、camera、scene 数组均与完整协议顺序逐项相等。限制是：本次没有重新认证数据集发行包、检查每张图片的像素来源；协议中的历史 inventory 哈希本身也未追溯至原发行记录。结论是现有真实文件及标签的完整协议评价，而不是对上游发行链作全面认证。

**B．分数定义、归一化和分母：PASS**

实际调用的是：

`run_visual_update_control.train/evaluate`
→ `run_correspondence_roles.official_metrics`
→ 原作者 `eval_func` / `eval_func_msrv`，同时调用独立 `camera_scores` / `scene_scores`。

见 `S/tools/run_visual_update_control.py:189`、`:240`；`S/tools/run_correspondence_roles.py:79–108`。作者模块来源还有实际路径断言 `:84`。

两套实现均：

- camera 数据集排除同身份、同 camera；
- MSVR 排除同身份、同 scene/time-block；
- AP 按真实正样本所在位置计算 precision，再以真实正样本数平均；
- mAP 对有效 query 平均，CMC 为真实首次匹配位置的命中比例，最终乘 100。

精确位置：作者 `utils/metrics.py:68`、`:95–106`、`:137`、`:155–168`；独立实现 `train_rgbnt100_signal_oof.py:253–268`、`train_msvr310_signal_oof.py:223–238`。

没有预测统计量作分母、选择性 query 子集、身份宏平均替代主表 mAP 或 reranking。身份宏平均和 bootstrap 仅在后验配对分析中报告。

限制：遵守禁止评分重放的范围，本次没有重新排序距离矩阵并独立重算 AP/CMC。分数的定义和实际调用路径已验证；数值检查是现存文件、checkpoint、历史及报告之间的精确绑定与算术一致性，原运行时双实现一致性仍包含保存断言证据。

**C．结果存在、选点、训练轨迹与完整状态：WARN**

六端实际回执与 checkpoint、history、原终端输出、accepted matrix 和完整报告一致。下面按原作者回执四舍五入至六位：

| 数据集 | 系统 | best epoch | mAP | R1 | R5 | R10 | 正式 step |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | global_only | 13 | 68.111669 | 69.736844 | 79.066986 | 84.569377 | 2649 |
| RGBNT201 | roles | 13 | 69.876396 | 71.411484 | 79.186600 | 84.928232 | 2649 |
| RGBNT100 | global_only | 3 | 78.491274 | 93.877554 | 95.102042 | 95.568514 | 6559 |
| RGBNT100 | roles | 20 | 80.923060 | 94.285715 | 95.276970 | 96.209913 | 6559 |
| MSVR310 | global_only | 16 | 51.722411 | 67.174280 | 83.587140 | 88.494080 | 1000 |
| MSVR310 | roles | 16 | 52.004202 | 68.697125 | 83.587140 | 88.155669 | 1000 |

行证据：`U:19`、`:238`、`:92`、`:311`、`:165`、`:384`。各端原始 `R/trained-model/clean_clip_joint_20261002_v1_clean_clip_{variant}_{dataset}_seed42_full/official_metrics.json:16–29` 保存选点、四项指标及 checkpoint/distance 哈希。

确定性核实：

- 每端 history 连续覆盖 1–50，共 **300 epochs、20,416 条正式 step**。
- 每轮 batch 编号连续；每个 loss 有限，`loss ≈ id + triplet`；step 均值与 history 一致，50 条原终端 epoch JSON 与 history 一致。
- 按 `max(mAP, epoch)` 重新从历史选点，六端结果正确；四项指标均来自同一个最高 mAP checkpoint。源码 `>=` 保证并列取较晚 epoch。见 `run_visual_update_control.py:188–193`、`:236–241`。
- 六个 M0 各有八条连续 batch 记录；RGBNT201 两个 M0 来自 preflight，没有被误算为正式训练。M0 回执的 `epochs:50` 是参数字段，实际执行量由一条 history 和八条 step 确定。
- 六个 best checkpoint 均在 CPU 成功读取，包含完整 state：每端 **152 个 FP32 视觉张量、camera、实际 neck/classifier**，全部状态有限；协议、condition、seed、epoch、metrics 和文件哈希与回执相符。
- 保存的 M0 probe 与 best 的键集合相同；M0 保存视觉/camera 张量摘要匹配回执，未训练的 Signal 状态在 M0 与 best 之间逐张量相同；best 视觉和 camera 摘要不同于初始化见证。
- 六个实际距离文件有限，尺寸为 836×836、1715×8575、591×1055，各两份。

完整保存及 strict reload 的实际入口是 `run_visual_update_control.py:93–108`，不是旧 `run_correspondence_roles.py` 中省略 Signal 的 checkpoint 实现。

WARN 的具体边界：

1. 公共初始张量逐位一致、152 视觉张量与公开权重一致、全部 trainable 梯度累计非零和重载前向误差为零，具有实际执行源码及已绑定运行回执支持，但未保存全部初始张量、逐参数梯度或原重载输入/输出，无法在本次禁止模型重放范围内独立重建这些运行时事实。见 `run_clean_clip_joint.py:62–89`、`prepare_clean_clip_joint.py:42–49`、`run_visual_update_control.py:205–220`；远端 M0 回执例如 `R/trained-model/clean_clip_preflight_20261002_v1_RGBNT201_global_only_m0/training.json:88–94`。
2. checkpoint 到距离矩阵的语义生成关系由原严格重载、评价调用和哈希回执支持；本次没有重新前向验证该关系。日志连续性也不等于独立重放每个优化器更新。
3. **目录存在六处后缀笔误**：catalog `:550`、`:557`、`:564`、`:571`、`:578`、`:585` 写成 `official_distances.pth`；实际绑定、生成及已核实文件均为 `official_distances.pt`。见 `run_visual_update_control.py:239`、`queue_clean_clip_joint.py:105`。这是目录问题，不是距离结果缺失。

公开权重实际 SHA256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`，与封存作者 CLIP 下载索引 `S/comparators/Signal-cd1b0a6/modeling/clip/clip.py:35` 一致。干净入口直接从该公开权重构造；旧 ReID checkpoint 构造函数被覆盖，未处于当前调用路径。见 `run_clean_clip_joint.py:37–106`、`:175–179`。

**D．死代码排除、实际报告调用与终态：PASS**

当前训练调用、严格重载后的正式评价、作者/独立 scorer，以及报告调用 `compare()` 的链均已追踪。旧 OOF 评价、旧 partial-state 保存、其他 plotting/KDE/统计定义不能当作当前结果实现。报告实际入口见 `P/tools/report_clean_clip_joint_complete.py:55–79`，配对分析见 `P/tools/analyze_correspondence_distances.py:28–75`。

唯一登记 waiter 记录：

- `invocations:1`；
- 04:06:11 启动；
- 04:06:42 完成，exit code 0；
- 状态 `CPU_REPORT_COMPLETE`。

见 `P/logs/clean_clip_complete721_20261002/raw/logs/clean_clip_analysis_waiter_20261002.json:2–22`。原日志仅有一条完成记录，报告/控制器/waiter 的相关 PID 当前均不存在。waiter 的防重复目录检查、独占日志和实际 `Popen/wait` 路径见 `P/tools/wait_clean_clip_complete_analysis.py:21`、`:45–54`。

字节码补查：233 清单内 217 个 Python 文件中有 120 个 CPython 3.10 缓存；119 个缓存 code object 与封存源码编译结果相等。唯一不同的 `signal_preserving_v8.pyc` 同时具有失效的时间戳和长度头，因此正常 import 不会接受它。没有发现有效但与源码不同的缓存。

此前对 Signal `modeling/__init__.py` 的疑点已排除：它是完整的 35 字节导入语句，Git blob、当前源码与缓存 code 一致；旧 `co_filename` 是搬迁前路径。缓存 SHA 为 `8e393022d1fe9d6151ceca334f197a60590502f45ddc3a98ca9c4d3db8a79ce7`。

限制：这些证据支持登记执行路径的一次报告；无法从现存日志证明机器历史上绝无未登记的额外调用，也不构成对过去进程的系统级执行证明。

**E．实验范围、容量/算力和 J1：WARN；J1 = FAIL**

原登记门明确要求三集 mAP 增量 >0、R1 增量 ≥0，同时 RGBNT201 和 MSVR310 各至少 +0.5 mAP。见 `Q:49`。

按完整报告的未四舍五入值及固定门限重新做算术判断：

| 数据集 | ΔmAP，百分点 | ΔR1，百分点 | 登记判定 |
|---|---:|---:|---|
| RGBNT201 | +1.764727 | +1.674641 | PASS |
| RGBNT100 | +2.431786 | +0.408163 | PASS |
| MSVR310 | +0.281791 | +1.522843 | **FAIL** |

见 `U:491`、`:701`、`:1011`。**MSVR310 未达到 +0.5，故总 J1 必须 FAIL**，与 `U:9` 一致。配对分析的 R1 使用 float64 比例，原作者 CMC 为 float32；两者的尾数差低于登记比较容差，不影响判定。

可以支持的描述是：在这次 seed42、匹配初始化及训练配方下，roles 系统在三个数据集均得到正的 mAP 点估计；预定跨三集足够增量门未通过。

不支持：

- CNN、Transformer、Mamba 各自必要性或分别贡献；
- 超出额外容量的机制收益；
- 干净初始化相对旧温启动的单因素因果收益；
- 完整训练流程的多 seed 稳定性；
- SOTA 或完整三数据集总目标已达成。

两系统都含原九个共享适配器；global-only 不是未经适配的纯 CLIP。roles 每集多 **1,701,002 个登记 trainable 参数**，相同输出维度和 epochs 不代表等容量或等计算。六端训练加逐轮评价累计 21,871.911352 秒，控制器观察 wall time 10,327.448314 秒；这些边界不包含全部构造和前置成本。见 `U:1343–1357`。

官方集已经用于全部 50 轮选点及此前方法开发；身份 bootstrap 只是固定模型的后验不确定性，不能代替训练 seed 稳定性。上述范围在 `Q:13`、`:26`、`:43–55` 和报告中已明确披露。

**F．评价类型：PASS，`real_gt`**

六个正式端属于 **real_gt** 检索评价：真实数据集身份和 camera/time-block 标签用于完整 query/gallery 的 mAP/CMC。没有发现 synthetic/self-supervised proxy 冒充正式性能。

M0 是初始化、梯度、更新和重载的工程检查，不是性能端；配对 bootstrap 是既定模型的后验分析。分类依据为 `run_correspondence_roles.py:85–107` 和已验证协议完整数组。

**G．新的 native-detail 实验边界：WARN**

未发现当前六端文件或有效字节码存在足以阻止**另行预登记 N1** 的实质完整性问题。但当前证据没有验证 N1/N2/N3，不能把这些机制标为已实施或已有效。

必须保留的具体边界：

- 当前 FP32 content attention 的 `sample_context()` 接收 `positions` 却不使用它；固定地址本身不能证明语义对应或 native-detail 收益。见 `S/modeling/trifusion/slot_competition_fp32_roles.py:10–20`。
- 当前 CNN→Transformer→Mamba 路径和 role0 semantic projection 有实际训练路径；新的 native-detail 接口若绕过它们，必须在新实验自己的代码/M0检查中证明所有预期 trainable 参数仍有有效路径。见 `S/modeling/trifusion/role_global_tokens.py:29–59`。
- high/low-detail 对照应在新登记中固定参数化、公共初始化、候选数量和读出，明确真正改变的输入信息。不能靠本面板官方结果继续调阈值或选择变体。
- **全部 20,416 条正式 step 只有 epoch、batch、loss、id、triplet、lr；其中 id 是 CE 标量，不是身份列表。** 因而无法恢复历史实际 batch 的身份/camera/scene 组合，也不能用重新运行 sampler 的模拟冒充历史 N3 A/B 支持。见 `run_visual_update_control.py:182–184`；现有说明 `P/refine-logs/clean_clip_joint_v1/EXPERIMENT_TRACKER_20261002_TRACE_COVERAGE720.md:24–26` 正确。
- 数据级多环境身份数量只是潜在支持：RGBNT201 51/171、RGBNT100 50/50、MSVR310 60/155；它不能证明当时每个 batch 有足够跨环境正样本。

COMPLETE721 当前准确区分了执行完成、待审计和未启动后继：`P/refine-logs/clean_clip_joint_v1/EXPERIMENT_TRACKER_20261002_COMPLETE721.md:3`、`:12`。

**无需训练重放的最小补证动作**

1. 更正审计目录的六个 `.pth`→`.pt` 路径，保留现存结果文件及哈希。
2. 将已在远端核实的 preflight/初始化见证、缺失的 M0 小型回执/trace/终端文本补入只读证据包，并登记原文件哈希。无需复制模型或再次执行报告。
3. 保留本次 Git blob 与字节码元数据结论；不需要删除或重建缓存。
4. 在最终审计记录中明确区分“直接核实保存张量/字节”与“源码加原运行时断言”。未保存的初始全状态、逐参数梯度、重载输入/输出和历史 batch 元数据，不能通过文字补写变成独立验证事实；后续新登记实验可以在首次运行时保存这些证据。
5. 封存本面板及 **J1 FAIL**；新的 N1 使用独立计划、来源绑定及 M0，不用已消费官方结果挽救这组已完成实验。

这些 WARN 限制主张强度和历史运行事实的独立可验证程度；**不改变六端已完成、现存六行结果绑定一致以及 J1 失败的结论。**
