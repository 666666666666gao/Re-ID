# Experiment Audit — MSVR310 固定来源 Smooth-AP 目标梯度诊断

**最终结论：WARN / CLOSED_WITH_LIMITS。确定性全量核验 PASS；工程结论 PASS_WITH_LIMITS。** 原始三阶段及全部注册行完成，GT、目标分母、行覆盖和报告数值未发现需要更正的问题。WARN 来自未保存的完整source参数向量及固定终态诊断的解释边界，不是检索性能失败的替代判定。原 `Q1_FAIL` 保持；本诊断没有新的heldout/official成绩或SOTA结论。

审计签发时间：2026-09-21T00:14:57.584596+08:00。源运行日期：2026-09-20。审计员实际 canonical identity：`/root/audit_smooth_objective_source_20260920`。配置路线为 gpt-6-astra / max，same-family / provisional；没有独立认证backend型号或UUID。确定性工具结果可在其有限范围内接受，语义评判仍是同家族临时结论。

## 输入范围与出处

- `REPO` = `/root/autodl-tmp/trifusion-v2/TriFusion-ReID`；`SOURCE` = `/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920`；`SIGNAL` = `/root/autodl-tmp/trifusion-v2/comparators/Signal-cd1b0a6`。下文这些前缀后的 `file:line` 是原文件的精确行位置；本目录保存了对应文本快照。未加前缀的文件位于本隔离审计目录。
- 静态阶段独立核对157项递归绑定，接收117份文本/38,550,382字节；终态阶段接收55份文本/91,156,469字节。它们包含相同文件的不同阶段快照，不能简单相加为唯一文件数。完整科学输入表包含189个唯一远端路径，见 `AUDITED_INPUTS.json`；模型、图像、NPY、距离和梯度张量均留在远端。
- 五份继承Python文件存在本地CRLF/远端LF差异，逐项LF内容及AST相同，远端实际哈希符合合同，详见 `local_remote_source_comparison.json`。不把本地整工作树当作执行字节参照。
- 早期master/AGENTS只按捕获时的历史进度封存。调用方并发更新其他文件不属于本次科学输入不变声明。未重开旧Q1或预检模型执行；只读取其已存在的checkpoint和保存数组。

## 原始终态与独立确定性覆盖

| 阶段 | 原wrapper / 子进程 | 原始终态（北京时间） | 退出 |
|---|---|---|---:|
| source | 1230 / 1231 | 2026-09-20 23:45:14 | 0 |
| verification | 1706 / 14798 | 2026-09-20 23:48:51 | 0 |
| analysis | 2251 / 15038 | 2026-09-20 23:50:59 | 0 |

三条原始pipeline均为 `COMPLETE`，上述六个 `/proc` PID路径在最终只读核查时均不存在。真正执行的checker/analyzer哈希分别为 `79802db8ef471e8fc495f625aa15321f2d9c1034b13afd82a8f08b1541fb2ef4` / `a4337f88e93200d7738018a4288022a3a98f10a34c724fdda13a33d6849a0653`，与原wrapper及分析合同一致。证据：`terminal_final_bindings_r1.stdout:1`；`SOURCE` 的三个 sibling `_pipeline.json` / `_verification_pipeline.json` / `_analysis_pipeline.json:1`。

审计员自己的封存核验器 `remote_terminal_verification.py` 不导入项目的目标核验器或分析器，使用NumPy/stdlib独立实现距离目标及聚合；在远端CPU一次完整执行，7.443秒、exit0、stderr空，SHA `04a30cb001a47f8e8f8ca4fbd107505bcaf3c875eaa80223dcb8a0995e8141c1`。原核验器复用项目目标函数，其PASS与这次额外的独立公式复算要区分。

| 范围 / 检查 | 完整核对结果 |
|---|---:|
| 条件 / 步 / epoch事件 | 6 / 1,560 / 120 |
| 保存距离值 | 29,125,376 |
| 角色行 / 全量窗口组 / 组-指标行 | 4,680 / 72 / 1,008 |
| 展示表行 | 全部18行，及两份完整CSV全部单元 |
| source原记录、像素摘要、memory与配对绑定 | 全1,560步，逐行通过 |
| 原checkpoint完整状态 | 六端各472张量，结合原Signal alias后SHA一致 |
| 保存预检梯度 | 3,402张量；第4步，每端一个指定历史状态 |
| 独立距离/加权loss最大绝对误差 | 1.5211602066855789e-07 |
| 保存预检向量统计最大逐角色误差 | 5.5511151231257827e-16 |
| 范数/余弦/差范数恒等式最大误差 | 8.8620503223546443e-16 |
| current / full 分解误差最大值（原分母） | 0.00160711764072 / 0.00171668277424，均小于0.005 |
| 每端第67步单历史组直接检查最大相对误差 | 0.000130492894757，小于0.005；原运行见证 |
| current / history 重复梯度最大相对差 | 0.000264255380785 / 0.000232125926939；保存标量见证 |
| 当前 / 额外角色记录前向 | 99,840 / 1,146,048；不是普通训练成本 |

全部原始行、原始端点receipt和分析结果按字节/SHA封存。确定性详情见 `INDEPENDENT_VERIFICATION_SUMMARY.json`、`terminal_verification_r1.stdout`、执行receipt及 `remote_terminal_verification.py:26-282`。审计额外模型前向、参数反传、优化器更新、官方图像读取均为0。

## A–F 审计判定

### A. 真实数据 GT 与 source/heldout 隔离 — PASS

真实 bounding_box_train 文件名中的 identity/camera/scene 与三模态存在性全部重建，1032 记录/155 身份完全匹配协议。标签分层轮转和三个 source label map 独立重建；全部诊断当前/历史记录均属于该折 source，未以预测构造 GT。

**边界：** 核查了实际标签路径及完整记录身份隔离；未重读或重放源图像内容。历史像素相同是原保存哈希与运行断言的绑定。

**证据：** `REPO/tools/build_msvr310_train_oof_protocol.py:12-55`；`REPO/tools/build_msvr310_train_oof_protocol.py:57-116`；`REPO/tools/train_msvr310_signal_oof.py:68-99`；`SIGNAL/data/datasets/msvr310.py:67-87`；`remote_static_inventory.py:60-85`；`remote_terminal_verification.py:116-137`。

### B. 目标、范数、余弦和误差分母 — PASS

单位特征距离转为 cosine-equivalent score 用于 Smooth-AP 目标；F/O 与 F/Total 是同一 encoder 角色块的导数范数比，cosine 是同坐标内积。分解误差除以两个分量范数之和是数值恒等式误差尺度，未被当作检索成绩。null/零值没有 epsilon 替换或选择性删除。所有实际分母及完整标量代数通过。

**边界：** 这些量不是训练 loss 权重、AdamW 更新份额、泛化贡献比例或性能归一化；不同目标的 loss 数值不可直接比较。

**证据：** `REPO/tools/msvr_role_set_relations.py:60-64`；`REPO/tools/msvr_smooth_ap.py:30-62`；`REPO/tools/probe_msvr_history_candidate_gradients.py:53-66`；`REPO/tools/probe_msvr_smooth_ap_objective_gradients.py:38-45`；`remote_terminal_verification.py:59-84`；`remote_terminal_verification.py:153-180`；`SUPPLEMENTAL_SCALAR_CHECKS.json:1`。

### C. 终态、文件、全量行和数值一致性 — PASS

三条原始 pipeline 均 COMPLETE/exit0，原六个 PID 路径已消失。六条件各 260 步、120 个 epoch 事件、29125376 距离、4680 角色行、72 组/1008 组指标、所有 CSV 单元和18条展示行全部验证。checkpoint/原Q1行绑定、文件字节/SHA和分析摘要链一致。

**边界：** 原核验器重用项目目标函数；本次额外使用独立 NumPy/stdlib 公式。历史进度文档仅封存为当时状态，不声称整个并发工作树不变。

**证据：** `SOURCE/summary.json:2`；`SOURCE/source_verification.json:2-55`；`SOURCE/analysis/analysis.json:2-16`；`SOURCE/analysis/REPORT.md:3-28`；`remote_terminal_verification.py:26-45`；`remote_terminal_verification.py:86-197`；`remote_terminal_verification.py:215-271`；`terminal_final_bindings_r1.stdout:1`。

### D. 执行路径成立，完整参数导数仍有重建边界 — WARN

真实 criterion 的14项、其余13项常量替换、current/history VJP、分组重编码、RNG和buffer断言均在实际 source 入口；三阶段 wrapper 和被调用 checker/analyzer SHA 相符。六端 checkpoint 完整状态和预检3402已存梯度张量统计独立复算通过。但1560步参数向量、历史冻结字段和上游VJP向量未完整保存，不能独立重做全部模型导数。

**边界：** source 第67步每端一个历史 group 的直接全图检查是原运行见证；预检第4步保存向量统计不是独立验证向量生成。零 upstream group 跳过、逐批随机/缓冲不变和图像到距离过程依赖一致代码与原运行断言。

**证据：** `REPO/modeling/trifusion/signal_preserving_v8.py:690-742`；`REPO/tools/run_signal_preserving_v5.py:99-135`；`REPO/tools/probe_msvr_smooth_ap_objective_gradients.py:84-203`；`REPO/tools/msvr_freshness_probe.py:43-85`；`REPO/tools/probe_msvr_history_candidate_gradients.py:43-106`；`SOURCE/summary.json:883-907`；`remote_prerequisite_arrays.py:37-83`；`remote_terminal_final_bindings.py:11-23`。

### E. 固定终态、单种子、source 诊断的解释上限 — WARN

seed42、三折×两端、固定终态模型、20轮原source序列/260批、warmup65、memory512/8、AMP scale256、零更新。189 encoder 张量覆盖CNN42/Transformer54/Mamba93；全优化器203张量还含neck/classifier。四个窗口重叠，fold和重复批也不是独立重复。不同端点来自不同最终参数状态。

**边界：** 不能从本诊断恢复原训练轨迹/AdamW矩状态，不能分离目标公式与端点表征的因果影响，不能推断heldout收益或据比值倒数/负余弦选择损失放大与投影。原Q1_FAIL不变。

**证据：** `REPO/configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json:2-12`；`REPO/refine-logs/msvr310_smooth_ap_objective_gradients_v1/DIAGNOSTIC_PLAN.md:3-15`；`REPO/tools/probe_msvr_smooth_ap_objective_gradients.py:89-95`；`REPO/tools/probe_msvr_smooth_ap_objective_gradients.py:216-223`；`REPO/tools/train_msvr_smooth_ap.py:73-80`；`REPO/tools/train_msvr_smooth_ap.py:186-245`；`REPO/tools/train_msvr310_source_style.py:164-174`；`SOURCE/analysis/analysis.json:7-16`；`SOURCE/analysis/REPORT.md:26-28`。

### F. real_gt 的固定source目标梯度诊断 — PASS

主诊断的监督来自真实数据身份标签；输出是局部目标导数及其保存标量，不是heldout或官方检索评估。独立检查器的闭式算术夹具是明确标注的synthetic math check，不能混为数据集结果。

**边界：** 真实GT不自动使导数诊断获得性能或SOTA声明资格；没有新的heldout/official检索结果。

**证据：** `REPO/tools/probe_msvr_smooth_ap_objective_gradients.py:1-4`；`REPO/tools/probe_msvr_smooth_ap_objective_gradients.py:94-123`；`REPO/tools/msvr_smooth_ap.py:30-35`；`SOURCE/analysis/REPORT.md:3`；`SOURCE/source_verification.json:52-54`；`verifier_selfcheck.json:1`。

## 分母的实际边界补查

原合同误差是 `||gF+gO−gT|| / (||gF||+||gO||)`，不能误写成相对总梯度误差。审计对全部1,560行另算 `||difference|| / max(||sum||,||total||)`：current/full最大值分别为0.0018931088889053457 / 0.0018828334088093705。总量与分量范数和的最小比例分别0.61764431670219 / 0.694995778185913，未出现分母因严重抵消而掩盖巨大总量相对误差的情形。

由逐角色F/O范数及余弦解析重建全体 `||F+O||`，与保存full decomposition范数的最大相对差为6.534357739438535e-10。当前与历史重复差范数之和除以F范数的最大值为0.000421236748250839。它们只检查保存标量的算术与噪声规模，没有恢复原向量，也没有新增实验准入门。证据：`SUPPLEMENTAL_SCALAR_CHECKS.json:1`、`supplemental_scalar_checks.py:1`。

## 全部活动窗口描述（step 66–260，每格195条）

F为同一角色块上的完整fused当前+历史导数，O为其余13项加权和的导数。以下18格全部列出；同时独立核对all、warmup和last65，未筛选有利fold/角色/窗口。分位数描述这些相关批次的分布，不是跨seed置信区间。

| fold | endpoint | role | F/O中位数 | F/O余弦中位数 | 负余弦数 / 195 | last65 F/O中位数 |
|---|---|---|---:|---:|---:|---:|
| 0 | control | cnn | 0.659347256 | 0.362579162 | 0 | 0.596388828 |
| 0 | control | transformer | 0.760940405 | 0.331815733 | 0 | 0.710066622 |
| 0 | control | mamba | 0.619499775 | 0.340791258 | 2 | 0.590398810 |
| 0 | smooth_ap | cnn | 0.108227351 | 0.238659864 | 7 | 0.141831721 |
| 0 | smooth_ap | transformer | 0.125272914 | 0.251877853 | 15 | 0.151650346 |
| 0 | smooth_ap | mamba | 0.105536807 | 0.212293618 | 14 | 0.126565544 |
| 1 | control | cnn | 0.655049048 | 0.364648469 | 1 | 0.598270394 |
| 1 | control | transformer | 0.787886630 | 0.324152619 | 3 | 0.750704993 |
| 1 | control | mamba | 0.606574178 | 0.331948955 | 1 | 0.589105268 |
| 1 | smooth_ap | cnn | 0.146948437 | 0.205059325 | 8 | 0.158732340 |
| 1 | smooth_ap | transformer | 0.174303569 | 0.217557190 | 13 | 0.192601068 |
| 1 | smooth_ap | mamba | 0.129413988 | 0.182966318 | 15 | 0.152762665 |
| 2 | control | cnn | 0.628204856 | 0.387886053 | 1 | 0.583343184 |
| 2 | control | transformer | 0.718893248 | 0.358751905 | 2 | 0.699977797 |
| 2 | control | mamba | 0.603683253 | 0.375264756 | 2 | 0.588349209 |
| 2 | smooth_ap | cnn | 0.137404979 | 0.234437953 | 8 | 0.149850591 |
| 2 | smooth_ap | transformer | 0.148566280 | 0.221428319 | 14 | 0.172503438 |
| 2 | smooth_ap | mamba | 0.123093251 | 0.206943897 | 21 | 0.155978378 |

原报告全部未舍入值见 `SOURCE/analysis/REPORT.md:7-24` 和全量JSON/CSV。本审计表仅显示9位小数。全部4,680角色行的F、O、Total范数为正，F/O与F/Total比值和余弦均无未定义项；1,191条零历史范数完整保留。活动阶段F/O负余弦为control12 / Smooth-AP115条；F/Total负余弦为0 / 37条。分母各为1,755个角色-批次记录，不能称独立样本。

这些终态及来源视图下，Smooth-AP端的F/O中位数在九个对应fold-role格中都低于control端。这是已保存运行时局部导数的描述，不能排除端点参数状态、目标形式、训练阶段或优化器历史的影响，也不能据此推导一个应使用的新loss倍率。

## 可接受声明与不可接受扩张

| 声明 | 影响 |
|---|---|
| 全部注册source条件和账本已完成，完整分析数字与原始保存行一致 | supported：deterministic all-row verification |
| 这些固定终态模型上的fused监督具有非零encoder导数，并呈报告中的相对规模和方向 | supported_with_qualifier：runtime scalar witnesses with complete arithmetic checks; saved preflight vectors separately recomputed |
| 当前+历史fused、其余13项、总目标的数值分解符合原.005约束 | supported_with_qualifier：all scalar identities and runtime decomposition ratios verified; not a source model backward replay |
| 1560批全参数梯度已经独立重构 | unsupported：per-step full source vectors/fields were not persisted |
| Smooth-AP的目标尺度导致原Q1泛化失败，或F/O比值给出了AdamW贡献/应取loss倍率 | unsupported：fixed different final states, no optimizer trajectory or counterfactual intervention |
| 本诊断产生新的heldout/官方性能、SOTA或主目标达成 | unsupported：source-only derivative diagnostic; original Q1_FAIL remains sealed |

## 行动项与原有记录

- I1：保留 runtime gradient witness 的表述；不要把完整标量复算、单历史组直接检查或预检向量统计重算写成1560步独立模型梯度重构。
- I2：描述每个终态和来源窗口的局部导数规模/方向，明确189-tensor范围、不同端点状态、固定scale和单seed。不得据此报告AdamW贡献百分比、推翻Q1_FAIL或宣布性能提升。
- I3：保留全部72组、零/负记录及窗口重叠说明；分位数只描述这些批次分布，不称跨seed置信区间。
- I4：调用方可在其获授权的状态文档中链接本报告与三条原始终态；保留原进度快照。审计员未修改master/AGENTS。

原始诊断没有要求修正实验实现、GT掩码、loss权重、科学门槛或报告数字；新增派生解读的两处措辞已按下述范围修订，也没有依据本审计授权重新训练、重新测量、选择中间权重或启动后继方法。

审计侧失败完整保留其源码与错误记录：连接prelude R1在远端payload前因StringIO不支持reconfigure失败，改为兼容该真实调用的静默TextIOWrapper；闭式夹具R1把B64/K8负例数误算成57，预期值从57/61更正为56/60，独立终态核验算法未改。文件为 `execute_remote_readonly_r1.py`、`static_inventory_r1.prelude_failure.json`、`verify_checker_selfcheck_r1.py`、`verifier_selfcheck_r1.failure.json`。完整终态核验R1一次通过，没有隐去失败的科学结果或只报告有利重试。最终文档引用范围检查另发现两处审计员file:line末行超过文件长度，已更正为实际行；失败结果、原校验源码和更正前报告保留于 `final_document_validation_r1.failure.json`、`validate_final_outputs_r1.py`、`final_draft_before_reference_correction/`。该检查不涉及实验代码或科学结果变化。

本审计未更改运行进程、代码/配置、master、AGENTS、环境、权重或Git历史。全请求/响应由root保留在 `.aris/traces/experiment-audit/2026-09-20_run01`，包括静态审查和本次终态continuation，canonical identity与same-family/provisional归属一致。

## 待发布派生解读的完整核对

审计另从已独立核验的全部4,680原始角色行重新计算派生 `pooled_descriptive_statistics.json` 的六个endpoint-window组合、全部计数、中位数/最小值/最大值、defined/undefined数及两个误差极值，所有数值逐值完全相同。九行配对展示表的全部数字也与原行按六位小数舍入一致，末65位置九组中位数范围与报告一致。这里的合并统计是相关角色-批次的描述，不能代替三折或跨seed独立推断。证据：`verify_derived_interpretation.py:1`；`DERIVED_INTERPRETATION_VERIFICATION.json:1`。

CNN和Transformer全部3,120条角色记录的current/history重复差异为0；Mamba 1,560条的最大绝对重复差异为0.00005039154397355579 / 0.00002629540187454565。这是原保存标量的完整复核，未将Mamba运行称为逐位确定，也不是梯度生成的独立重构。全部六个合并窗口均不存在F范数为零或未超过该条current/history重复差异之和的行。复算过程未按符号或大小筛选。

调用方接收目录的全部33份文本再次按实际本地字节计算SHA，并与本审计直接从远端封存的同一路径、字节数和SHA逐一匹配，总计37,743,954字节。这个33份是调用方的原始source归档范围；本审计55份终态文本还包含原Q1行与执行依赖，两者不是遗漏关系。

原待发布报告存在两处范围措辞。第35行“所有分支梯度都相互冲突”超出了被测的同一角色块内部F/O比较；第29行“真实学习信号”可能被理解成已验证学习或泛化收益。审计提出后一处精确化和前一处必要限定后，root已修订为固定终态/source视图上的非零encoder导数，以及各角色参数块内F/O并非普遍反向，并明确不能评价不同角色之间的梯度冲突或原训练全程。修订后两项均已解决，没有改动数字或pooled JSON。证据：`snapshots/interpretation_review_round2/MSVR310_SMOOTH_AP_SOURCE_OBJECTIVE_GRADIENTS_2026-09-20.md:29`、`:35`。原版本完整文本保留于本轮工具追踪，其初次观察SHA为 `93ac261af5e670648dbd03c0c4cf7add1fb1497401b1ca5c2a22531fb1bb3e8e`；未在root修订前保留原版本本地字节快照，因此不声称存在该快照。

最终修订报告SHA为 `4aac7787d6f4de93e25c96f72536ff3761db521f3ea83f23623282b4a0b5474a`，pooled JSON SHA为 `e050b5d987db7e02e056c8e84b0a83db9e3a8d5966405993d01004199b406d04`，精确字节副本已封存在 `snapshots/interpretation_review_round2/`。报告第3/41行现在标明WARN/CLOSED_WITH_LIMITS及预定审计链接；正式归档仍由root执行。报告所提原来源排序/Q1结论是历史上下文，不是本次新增检索评估。新增输入及33个哈希匹配项见 `DERIVED_INTERPRETATION_VERIFICATION.json`，科学remote输入仍为189个唯一远端路径。审计整体A–F及same-family/provisional边界均保持。

## 关键封存哈希

| 文件 | SHA256 |
|---|---|
| SOURCE/summary.json | `f1cb20221786a6aad5147f15333b136f3cd13222e82ca48df0a1876d12e747c2` |
| SOURCE/source_verification.json | `806ee90114c18d3e01878e5852bb09c8d4db57e2af40b74ce4a3404dc3a27ee7` |
| SOURCE/analysis/analysis.json | `208d2c1f5d0171c793d77d9c4c51b45785dc5933d9521d6dd9341b3b6de96063` |
| AUDITED_INPUTS.json | `d6f15019813c184602883a1cd920f9fa6dc03a2ba91b2aac3537042e527fdfa5` |
| INDEPENDENT_VERIFICATION_SUMMARY.json | `51aa28a765c94531d180126e07a697d66f975a9f073faa2dbce318b2c8fc99ca` |
| remote_terminal_verification.py | `04a30cb001a47f8e8f8ca4fbd107505bcaf3c875eaa80223dcb8a0995e8141c1` |
| terminal_verification_r1.stdout | `02f6a1cd995952cfd5197130d8c2e9d075b9f77ed7e51d8a4c6250ee49bf68a9` |
