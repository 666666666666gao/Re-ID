# 固定五 best 诊断源码审查 · 第一轮

结论：**FAIL，1 项阻断问题**。范围为 SOURCE_ONLY；同模型家族独立上下文审查，接受状态 provisional。请求的评审路由为 Astra/max，实际运行模型与 effort 均 **UNATTESTED**。未 SSH、未查询 GPU、未导入神经模块、未构造模型、未训练或运行诊断；未修改生产源码或 Git。

审查代码 SHA256：`e1903440e9de0df3d928600c9e02032e969bd118c98b5b44cf84598cd7e19e1e`。计划 SHA256：`83c81e09061729e1fd623641b4faad62586bb305f2f55a5130aa17475cf09732`。此结论固定于这两个版本；第一轮原始代码、计划、完整请求/响应及数据核验留在私有 trace。

## 必须修正的阻断问题

**B1：MSVR310 和 RGBNT100 的评价批处理访问空 ENVIRONMENTS 字典。** 新入口调用 `entry.configure()`，然后直接构造、加载并提取特征，但绕过原 `main()` 的环境表填充步骤（[新入口 51—69 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:51)；[原填表 106—111 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0386_run_incremental_role_objective.py:106)）。

`configure()` 把共享 runner 的 `_training_batch` 替换为增量版本；这个函数会按 `raw[4]` 的文件名索引 `ENVIRONMENTS`（[替换位置 84—91 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0386_run_incremental_role_objective.py:84)，[字典初始化和索引 23—39 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0386_run_incremental_role_objective.py:23)）。复用的 `extract()` 先调用 `runner._eval_batch`，而其 MSVR310/RGBNT100 分支动态调用这个被替换的 `_training_batch`（[extract 120—124 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0314_diagnose_native_research_best.py:120)，[_eval_batch 60—63 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0009_run_correspondence_roles.py:60)）。因此直接调用 `core.forward_features()` 只能绕开外层辅助头，绕不开之前的批处理。

只抽取实际函数 AST、以普通 Python 字典和占位批处理运行的测试复现了：

| 数据集 | 原始首个 query 文件名 | 当前入口的数据路径结果 |
|---|---|---|
| RGBNT201 | `000258_cam1_0_00.jpg` | 走 `image_batch`，不读取环境表 |
| MSVR310 | `0002_s002_v0_000.jpg` | `KeyError` |
| RGBNT100 | `0502_c0001_000.jpg` | `KeyError` |

因此在先前检查通过的前提下，前两个 201 作业可继续，第三个 MSVR310 作业将在首个提取批次中断；五任务队列无法完成。这是已有调用链的确定问题，不是假设性边界情况。

最小修正：在新入口读完 protocol 后、提取之前，复用原 `main()` 的完整环境表填充步骤，仍用 MSVR310 scene、其他数据集 camera，并保留原有同名键一致性断言。不要加 fallback、改外层模型、loss 或评价规则。对原协议字典执行这一原有填表步骤后，隔离测试中的 MSVR310/100 首批均能取得正确环境值；这只是数据路径验证，并未修正生产源码或证明模型运行成功。

## 其余已核查内容

- **初始化与严格加载：源码链匹配。** 新入口先绑定 checked 包装，再执行原 configure 链；最终构造类为 `IncrementalObjectiveHeads`，核心是 `DetachedSemanticTriFusion`，foundation 使用增量 schema/condition 和 checked `build_core`。所有五份初始化 JSON 与原正式 accepted rows 的完整 binding 相等，checked `entry_sha256` 为 `70121fe68b6a89830ca98835e84ace604b04986e125881eb571744dc7549c17d`。objective、cfg、public CLIP、protocol、头、初始 state 和分区元数据均按原值保留。[checked 15—19 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0406_run_incremental_role_objective_checked.py:15)、[增量 configure](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0386_run_incremental_role_objective.py:84)、[foundation build 103—107 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0243_run_foundation_recipe.py:103)、[strict load 180—186 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0243_run_foundation_recipe.py:180)。这不是一次真实模型初始化。
- **模式、设备和状态保护：源码匹配。** `model.eval()` 经 AuthorHeadEvidence 的 `train(False)` 明确还原 Signal/head eval；原 first6 cuda:1、last6/head cuda:0 保留。提取在 inference_mode 下无 AMP，每批调用一次 core，并取四份 CPU FP32 `[N,1536]` 特征。完整 state_dict（含 buffer）SHA 及无梯度检查包围提取；没有 optimizer、训练或权重保存调用。[head 模式 42—47 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0267_evidence_author_heads.py:42)、[分区 9—61 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0304_partitioned_evidence_clip.py:9)、[提取 120—139 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0314_diagnose_native_research_best.py:120)、[SHA 523—528 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0028_run_signal_preserving_v5.py:523)。这些路径仍受 B1 阻断。
- **全量范围和指标：源码匹配。** query/gallery 为 836/836、591/1055、1715/8575，五任务总计 16,926 条记录。原 loader 不按 query 身份裁剪 gallery，保留 MSVR 合法干扰身份；顺序及六项 identity/camera/scene 元数据与原矩阵和独立 global 矩阵逐项比较。沿用真实标签的 scene/camera 过滤；同一 best 的四指标均用原 `<1e-5` 点容差，分解恒等式用原 `1e-6` atol/rtol。[数据顺序 8—39 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0025_official_three_dataset_data.py:8)、[协议计数 58—68 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0027_run_official_three_dataset_roles.py:58)、[score/paired 45—94 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0314_diagnose_native_research_best.py:45)、[新诊断 62—106 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:62)。
- **任务和资源：源码匹配。** 恰好五个顺序独立子进程，排除失败的 RGBNT100 repair_keep；首个失败即停，不重试、不换 best。GPU 查询限定 0/1 的 index/memory.used，未查询功率或温度。沿用诊断 `>2 GiB` 门，原正式训练 `5,192,548,352 B` 门未改。特征/距离/标量张量计算为 `624,523,020 B`（约 0.582 GiB，另加序列化和文本），与约 0.6 GiB 估计一致。[新协调器 116—163 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:116)、[旧正式门 17—18 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0415_queue_incremental_qualified_five.py:17)。
- **封存计数：成立，但最终 seal 待提供。** 已验证 423 件原执行快照、45 件正式文本的实际本地字节 SHA，且全部 45 件文本与原 REMOTE.json 中对应 payload 相同；五份初始化与正式回执匹配。旧 scope 恰为 420，新增代码、计划、复审 MD/JSON 为 424，独立 seal 不计入自身 source 哈希。代码会在作业前后核查所列 source/artifact、原正式 controller EXIT0，不删除或改写旧权重/回执。[输入门 32—40 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:32)、[结束核查 110—111 行](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:110)。最终 424 源及五 best/距离/receipt、五初始化、global 行、45/48 输入联合、CLIP/协议的完整 seal 尚未作为本轮输入，不能声称它已通过实物核验。

## 非阻断说明与证据边界

1. 作者 `eval_func_msrv` 会追加当前目录的 `re.txt`。新子进程 cwd 为 ROOT，未切换工作目录；两个 MSVR 作业各评四组矩阵，共将写 4,728 个 query block 到 ROOT/re.txt，位于各作业输出目录之外。[作者 71—78 行](C:/Users/gb/.codex_tmp/independent_evidence_draft/qualified_five_audit_sources891/0213_metrics.py:71)、[子进程 cwd](C:/Users/gb/.trifusion_github_publish_22c3bee/tools/diagnose_incremental_fixed_best.py:134)。没有证据表明 ROOT/re.txt 是受保护输入，因此不另列阻断，但不能称本路径完全隔离所有写入。
2. Windows 发布目录不是完整可运行目标树：旧 scope 中有 107 路径本地缺失，另 27 文件只存在 CRLF/LF 差异。本审查实际使用经过 SHA 校验的原执行快照。不能整树覆盖服务器或宣称本地发布树已逐字节等于原 420；部署前后的真实服务器 seal 检查仍必要。
3. 原 checkpoint/距离/CLIP 的二进制只具有已供原始回执中的物理 SHA 证据，本轮没有读取远端二进制。原45/48封存输入与五 selected 的完整 SHA、所有实际审查输入 SHA 详见同名 JSON。当前 host/conda、磁盘、空闲 GPU/OWN NN 和唯一 observer 是启动事实，本轮未验证。
4. 同模型 g 仍只是组件诊断，c 单独检索才归一化；配对结果是已选官方单种子描述。失败100 repair_keep 和历史 FAIL/STOP 原样保留。源码审查、AST 测试和身份配对都不是模型运行成功、因果归因、完整三集改进、创新或 SOTA 证据。

时间：2026-10-07T14:37:29.487718+08:00。私有 trace：`C:\Users\gb\.codex_tmp\independent_evidence_draft\fixed_five_source_review892\.aris\traces\experiment-bridge\2026-10-07_fixed_five892`。**修复 B1 并完成一次有迹可查的复审后，才能推进部署。**
