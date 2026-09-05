# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-08-31 02:16 | /run-experiment | AGENTS.md | implementation | WSL2、GPU、conda、数据与公平性约束 |
| 2026-08-31 02:19 | /run-experiment | tools/audit_rgbnt201.py | implementation | RGBNT201 身份、模态配对与 JPEG 完整性审计工具 |
| 2026-08-31 02:20 | /run-experiment | ../artifacts/rgbnt201_audit_20260831.json | implementation | RGBNT201 数据审计结果，全部通过 |
| 2026-08-31 02:21 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260831_022125.md | implementation | claim-driven 论文实验计划的永久版本 |
| 2026-08-31 02:21 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 实验计划 latest copy |
| 2026-08-31 02:21 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_022125.md | implementation | 可执行实验追踪器的永久版本 |
| 2026-08-31 02:21 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 实验追踪器 latest copy |
| 2026-08-31 02:27 | /run-experiment | tools/audit_msvr310.py | implementation | MSVR310 身份划分、配对与 JPEG 完整性审计工具 |
| 2026-08-31 02:28 | /run-experiment | ../artifacts/msvr310_audit_20260831.json | implementation | MSVR310 数据审计结果，全部通过 |
| 2026-08-31 02:30 | /research | docs/RESEARCH_AUDIT_2026-08-31.md | research | 新颖性碰撞、RGBNT201 协议、公开工件和分轨 SOTA 审计 |
| 2026-08-31 02:32 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260831_023224.md | implementation | 主源查新后冻结的 claim-driven 实验计划永久版本 |
| 2026-08-31 02:32 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 主源查新后实验计划 latest copy |
| 2026-08-31 02:32 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_023224.md | implementation | 新基线与数据状态同步后的追踪器永久版本 |
| 2026-08-31 02:32 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 新基线与数据状态同步后的追踪器 latest copy |
| 2026-08-31 02:35 | /run-experiment | ../artifacts/environment_smoke_20260831.json | implementation | tri_reid 的 PyTorch/CUDA/CuDNN 与真实 GPU 前反向 smoke 结果 |
| 2026-08-31 02:51 | /run-experiment | ../artifacts/mdreid_rgbnt201_loader_smoke_20260831.json | implementation | MDReID 官方 RGBNT201 loader 单批三模态接口 smoke 结果 |
| 2026-08-31 03:01 | /run-experiment | tools/audit_rgbnt100.py | implementation | RGBNT100 原始三谱、官方 composite 划分与全量 JPEG 审计工具 |
| 2026-08-31 03:02 | /run-experiment | ../artifacts/rgbnt100_audit_20260831.json | implementation | RGBNT100 17,250 triplets、50/50 身份划分与 70,715 JPEG 全部通过 |
| 2026-08-31 03:05 | /run-experiment | environment.yml | implementation | WSL2 tri_reid conda/CUDA 环境入口 |
| 2026-08-31 03:05 | /run-experiment | requirements-lock.txt | implementation | 复现实验的完整 pip 版本锁 |
| 2026-08-31 03:05 | /run-experiment | scripts/build_mamba_sm89.sh | implementation | 固定官方提交、应用 SM89 patches、源码编译并烟测 Mamba |
| 2026-08-31 03:06 | /run-experiment | tools/smoke_mamba.py | implementation | Mamba CUDA 前向、反向与全参数梯度门禁 |
| 2026-08-31 03:07 | /run-experiment | ../artifacts/mamba_cuda_smoke_20260831.json | implementation | Mamba 2.2.6.post3 / causal-conv1d 1.6.0 的 SM89 有效烟测 |
| 2026-08-31 03:08 | /run-experiment | tools/reproduce_mdreid.py | implementation | 绕开上游损坏入口、严格复现 MDReID checkpoint 的独立驱动器 |
| 2026-08-31 03:09 | /run-experiment | ../artifacts/mdreid_rgbnt201_eval_20260831.json | implementation | MDReID 本机同协议 82.0868 mAP / 85.1675 R1 parity 证据 |
| 2026-08-31 03:09 | /run-experiment | ../artifacts/download_manifest_20260831.json | implementation | 数据、权重、审计、源码提交与下载哈希总清单 |
| 2026-08-31 03:09 | /run-experiment | docs/ENVIRONMENT_REPRODUCTION.md | implementation | 环境重建、兼容性决策与基线复现说明 |
| 2026-08-31 03:09 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_030917.md | implementation | 数据、环境、Mamba 与 MDReID parity 完成后的追踪器永久版本 |
| 2026-08-31 03:14 | /experiment-plan | docs/METHOD_SPEC_V1.md | research | HFER、URGC、RDPT 的公式、软件契约、证伪门和完整消融矩阵 |
| 2026-08-31 03:14 | /tdd | docs/TDD_SEAMS.md | implementation | 核心三分支实现前待用户确认的公共测试接缝 |
| 2026-08-31 03:15 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260831_031302.md | implementation | 数据、环境、baseline parity 与方法冻结状态同步后的永久实验计划 |
| 2026-08-31 03:18 | /run-experiment | tools/smoke_demo.py | implementation | 不修改上游仓库的 DeMo 三谱完整训练图前反向诊断器 |
| 2026-08-31 03:19 | /run-experiment | ../artifacts/demo_train_smoke_20260831.json | implementation | DeMo 完整训练图 loss、322/322 参数梯度与 1184.57 MiB 显存证据 |
| 2026-08-31 03:22 | /run-experiment | tools/probe_demo_train_step.py | implementation | 真实 RGBNT201、官方身份采样/损失/AMP 的单优化步容量探针 |
| 2026-08-31 03:24 | /run-experiment | tools/run_demo_baseline.py | implementation | 保持上游模型不变、显式覆盖 CLIP/数据路径的 DeMo 可复现训练入口 |
| 2026-08-31 03:26 | /run-experiment | scripts/run_demo_rgbnt201_seed42.sh | implementation | RGBNT201 DeMo seed42、B32/K4、50 epoch 的固定复现命令 |
| 2026-08-31 03:27 | /run-experiment | tools/summarize_demo_run.py | implementation | 解析 DeMo 逐 epoch 三模式指标、异常与 checkpoint 的 JSON 审计器 |
| 2026-08-31 03:28 | /run-experiment | ../artifacts/demo_real_step_probe_b32_20260831.json | implementation | RGBNT201 B32/K4 官方损失 8 步 AMP 稳定性与 6894.18 MiB 容量证据 |
| 2026-08-31 03:28 | /run-experiment | ../artifacts/demo_rgbnt201_sanity_seed42_summary.json | implementation | DeMo 1-epoch 训练、三模式评估、保存和 checkpoint 哈希闭环证据 |
| 2026-08-31 03:28 | /run-experiment | ../runs/demo_rgbnt201_seed42_b32k4/ | implementation | R012 50-epoch DeMo seed42 正式复现运行目录（进行中） |
| 2026-08-31 03:28 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_032800.md | implementation | R012 正式运行启动后的追踪器永久版本 |
| 2026-08-31 03:31 | /experiment-plan | tools/build_rgbnt201_dev_protocol.py | implementation | 基于 train_171 跨摄像头元数据和固定哈希盐构建无测试泄漏 dev 协议 |
| 2026-08-31 03:32 | /experiment-plan | protocols/rgbnt201_dev_v1.json | research | 141-fit/30-dev 身份、官方过滤与完整审计的冻结协议 |
| 2026-08-31 03:32 | /experiment-plan | protocols/rgbnt201_train_ids_v1.txt | research | 哈希预注册协议的 141 个训练身份列表 |
| 2026-08-31 03:32 | /experiment-plan | protocols/rgbnt201_dev_ids_v1.txt | research | 全部具有跨摄像头正样本的 30 个开发身份列表 |
| 2026-08-31 03:32 | /experiment-plan | ../artifacts/rgbnt201_dev_protocol_v1_audit.json | research | 零测试泄漏、零身份交叠、825/825 有效 query/gallery 的协议证据 |
| 2026-08-31 03:48 | /run-experiment | tools/run_demo_baseline.py | implementation | 增加独立测试批量参数，规避 8 GB GPU 上 B128 评估分页停顿 |
| 2026-08-31 03:49 | /run-experiment | ../artifacts/demo_rgbnt201_tb128_stall_audit_20260831.json | research | TB128 epoch2 评估 857s 停顿、终止与不晋级决定的哈希审计 |
| 2026-08-31 03:48 | /run-experiment | ../runs/demo_rgbnt201_seed42_b32k4_tb64/ | implementation | R012 从初始化启动的 DeMo seed42、B32/K4/TB64、50-epoch 正式运行目录 |
| 2026-08-31 03:59 | /monitor-experiment | ../artifacts/demo_rgbnt201_seed42_b32k4_tb64_gate2_20260831.json | research | 连续两轮三模式评估有效、最大 224.226s<300s 的 TB64 实时门禁快照 |
| 2026-08-31 03:59 | /monitor-experiment | docs/ENVIRONMENT_REPRODUCTION.md | implementation | 增补 TB64 两轮门禁、BAR1/显存压力警告与非终态声明 |
| 2026-08-31 03:59 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_035900.md | implementation | RGBNT100、dev 协议、TB128 负结果与 TB64 两轮门禁同步后的只读追踪器 |
| 2026-08-31 03:59 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 同步后的实验追踪器 latest copy |
| 2026-08-31 04:03 | /research | docs/SOURCE_INTAKE_2026-08-31.md | research | 原始调研文件的哈希、需求提炼、用户三分支覆盖决策与项目工件映射 |
| 2026-08-31 04:07 | /research | docs/RESEARCH_AUDIT_2026-08-31.md | research | 补核 NEXT、STMI、CoT-ReID、PRISM 的精确指标、测试资源、协议与工件可复现性 |
| 2026-08-31 04:16 | /implement | evidence/ | implementation | 将数据、环境、MDReID parity 与 DeMo 系统门禁 JSON 按 SHA-256 原样纳入版本化证据包 |
| 2026-08-31 04:25 | /implement | tools/capture_demo_live_provenance.py | implementation | 对早于新启动门的 R012 生成明确非 launch-attestation 的受限进程/源码 provenance receipt |
| 2026-08-31 04:33 | /implement | docs/BASELINE_PROTOCOL_AUDIT_2026-08-31.md | research | 固化 DeMo 每轮读取 test、按 test mAP 选 best 的源码证据及 fixed-epoch 双报告策略 |
| 2026-08-31 04:42 | /monitor-experiment | evidence/demo_rgbnt201_epoch10_checkpoint_receipt_20260831.json | research | 冻结 test 评估前保存的 DeMo epoch10 checkpoint 大小、SHA 与时序证据 |
| 2026-08-31 04:46 | /monitor-experiment | evidence/demo_rgbnt201_seed42_b32k4_tb64_epoch10_snapshot_20260831.json | research | 10 epoch/30评估/双checkpoint哈希快照；如实保留 ori latency>300s 导致 valid=false |
| 2026-08-31 04:46 | /monitor-experiment | refine-logs/EXPERIMENT_TRACKER_20260831_044637.md | research | 冻结 epoch10 指标、固定checkpoint与系统延迟 WARN 的不可变实验追踪器快照 |
| 2026-08-31 05:00 | /implement | tools/summarize_demo_run.py | implementation | 实时汇总新增已训练epoch×三模式精确覆盖门，部分评估一律fail-closed |
| 2026-08-31 05:00 | /monitor-experiment | docs/ENVIRONMENT_REPRODUCTION.md | research | 记录epoch12的34→35→36条评估边界验证，并保持正式300s延迟失败不变 |
| 2026-08-31 05:06 | /implement | evidence/demo_summarizer_fail_closed_verification_20260831.json | research | SHA绑定epoch12实时边界及空/断档/越界/重复/partial/complete七项CLI合同探针 |
| 2026-08-31 05:06 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 登记R019 fail-closed汇总门并将版本化证据包更新为19项 |
| 2026-08-31 05:21 | /codebase-design | docs/IMPLEMENTATION_BLUEPRINT_V1.md | implementation | 将冻结方法落成三完整专家、统一可靠性、双层同步中继、方向教学、512维融合、消融开关与8 GiB资源门禁 |
| 2026-08-31 05:38 | /novelty-check | docs/NOVELTY_CHECK_2026-08-31.md | idea-discovery | 最近六个月主源撞车、独立 GPT-5.5 xhigh 复核、评分与 PROCEED WITH CAUTION 决策 |
| 2026-08-31 05:45 | /novelty-check | docs/RESEARCH_AUDIT_2026-08-31.md | idea-discovery | 增补 MRUF、TIER-MoE、TMUR、TIGER 与动态教师近邻，并收紧论文定位 |
| 2026-08-31 05:45 | /experiment-plan | docs/METHOD_SPEC_V1.md | implementation | v1.1：HFER、身份外折完整干预 CIRC、同后验 URGC 与 RDPT 辅助晋升门 |
| 2026-08-31 05:45 | /experiment-plan | docs/IMPLEMENTATION_BLUEPRINT_V1.md | implementation | v1.1：global joint router、离线 target builder、统一控制消融与8 GiB串行干预设计 |
| 2026-08-31 05:46 | /tdd | docs/TDD_SEAMS.md | implementation | 同步 CIRC target-cache 可复现接缝；核心实现继续等待精确接缝同意 |
| 2026-08-31 05:47 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260831_054739.md | implementation | v1.1 claim-driven 实验计划永久版本 |
| 2026-08-31 05:47 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | v1.1 实验计划 latest copy |
| 2026-08-31 05:47 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_054739.md | implementation | CIRC/URGC 因果、校准、容量和正式种子执行追踪永久版本 |
| 2026-08-31 05:47 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | v1.1 实验追踪器 latest copy |
| 2026-08-31 06:03 | /monitor-experiment | evidence/demo_rgbnt201_epoch20_checkpoint_receipt_20260831.json | research | epoch20 固定 checkpoint 的保存时序、SHA、三模式指标与非 test 选点边界 |
| 2026-08-31 06:03 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 同步 epoch20 里程碑与 CIRC 第二轮实现就绪性审查门 |
| 2026-08-31 06:14 | /novelty-check | docs/NOVELTY_CHECK_2026-08-31.md | idea-discovery | 四轮独立复核闭环为 PROCEED WITH CAUTION / 最终 implementation-readiness PASS |
| 2026-08-31 06:14 | /experiment-plan | docs/METHOD_SPEC_V1.md | implementation | 闭合 all-171、共享 Beta、helpful 语义、sampled-edge 成本及全套校准门 |
| 2026-08-31 06:14 | /experiment-plan | docs/IMPLEMENTATION_BLUEPRINT_V1.md | implementation | 与最终 PASS 规范同步的测试先行实现蓝图，继续等待精确接缝同意 |
| 2026-08-31 06:14 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 最终 reviewer PASS 后的 latest 计划；时间戳副本保持逐字节一致 |
| 2026-08-31 07:22 | /monitor-experiment | evidence/demo_rgbnt201_epoch30_checkpoint_receipt_20260831.json | research | epoch30 固定 checkpoint 的保存时序、SHA、30轮/90评估完整性与非 test 选点边界 |
| 2026-08-31 07:22 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | 同步 epoch30 里程碑、21项证据包与 R012 持续运行状态 |
| 2026-08-31 08:36 | /research-lit | docs/RESEARCH_AUDIT_2026-08-31.md | research | 增量核验 FUSE、Signal、PEFT-BoA、ICPL、MGRNet、UPCL 与 Hyper-ReID 的指标、协议和官方工件 |
| 2026-08-31 08:36 | /research-lit | refine-logs/EXPERIMENT_PLAN.md | implementation | 固定 DeMo/MDReID/PEFT-BoA 三类基线角色及最终动态 SOTA 刷新门 |
| 2026-08-31 08:36 | /research-lit | refine-logs/EXPERIMENT_TRACKER.md | implementation | 登记 PEFT-BoA/Signal 比较器和 R071A 投稿前前沿复查 |
| 2026-08-31 08:45 | /tdd | tests/test_audit_peft_boa_source.py | implementation | PEFT-BoA receipt 解析、配置/shape 漂移和 test-selection 风险的 fail-closed 单测 |
| 2026-08-31 08:45 | /tdd | tools/audit_peft_boa_source.py | implementation | 固定官方提交并执行真实 RGBNT201 CPU loader batch 的只读来源审计器 |
| 2026-08-31 08:45 | /tdd | evidence/peft_boa_source_audit_20260831.json | research | PEFT-BoA clean source、配置、loader、环境偏差与训练阻塞证据 |
| 2026-08-31 08:45 | /tdd | docs/ENVIRONMENT_REPRODUCTION.md | implementation | 记录 PEFT-BoA 下载位置、复查命令及非训练就绪边界 |
| 2026-08-31 08:57 | /research | evidence/peft_boa_protocol_audit_20260831.json | research | 冻结 PEFT-BoA 论文数值来自 test-selected epoch80、fixed epoch120 为 82.2/85.8 的源码/日志证据 |
| 2026-08-31 08:57 | /research | docs/BASELINE_PROTOCOL_AUDIT_2026-08-31.md | research | 将 DeMo 与 PEFT-BoA 的 test-selected/fixed checkpoint 报告边界统一审计 |
| 2026-08-31 08:57 | /tdd | docs/PEFT_BOA_REPRODUCTION_SPEC.md | implementation | 固定 B64/K4 容量门、seed1111 fixed120、全状态恢复与一次 test 评估合同 |
| 2026-08-31 08:57 | /tdd | docs/TDD_SEAMS.md | implementation | 增补待用户确认的 PEFT-BoA fixed-endpoint 恢复 CLI 公共接缝 |
| 2026-08-31 09:23 | /run-experiment | environment/peft_boa_environment.yml | implementation | 独立 PEFT-BoA Python 3.10.21 / torch 2.1.1+cu118 conda 重建入口 |
| 2026-08-31 09:23 | /run-experiment | environment/peft_boa_requirements-lock.txt | implementation | 与已安装上游环境 `pip freeze` 完全一致的全量版本锁 |
| 2026-08-31 09:23 | /run-experiment | evidence/peft_boa_environment_smoke_20260831.json | research | pip、CPU tensor、真实 RGBNT201 loader 通过及 GPU/训练未就绪边界证据 |
| 2026-08-31 09:23 | /run-experiment | docs/ENVIRONMENT_REPRODUCTION.md | implementation | 增补隔离环境重建命令、验证结果和非 GPU/非指标声明边界 |
| 2026-08-31 09:43 | /run-experiment | evidence/signal_source_protocol_audit_20260831.json | research | Signal clean source、B64/K8 真实 CPU loader、公开 fixed-path 日志、test-selection 与缺失权重边界证据 |
| 2026-08-31 09:43 | /run-experiment | docs/BASELINE_PROTOCOL_AUDIT_2026-08-31.md | research | 将 Signal fixed-e50 路径、test-selected best 和未本机复现的报告边界纳入统一协议 |
| 2026-08-31 09:43 | /run-experiment | docs/ENVIRONMENT_REPRODUCTION.md | research | 记录 Signal 固定源码、可接通数据链、CLIP 依赖及当前 checkpoint/GPU 阻塞 |
| 2026-08-31 10:10 | /run-experiment | environment/mfrnet_environment.yml | implementation | MFRNet Python 3.8.20 隔离环境的 conda 基础重建入口 |
| 2026-08-31 10:10 | /run-experiment | environment/mfrnet_requirements-lock.txt | implementation | torch1.12+cu113 与 MFRNet 最小评估运行时的无用户站点版本锁 |
| 2026-08-31 10:10 | /run-experiment | environment/mfrnet_tutel_source.txt | implementation | 固定 Tutel v0.3.2 commit 与源码归档 SHA 的第二阶段构建输入 |
| 2026-08-31 10:10 | /run-experiment | evidence/mfrnet_rgbnt201_checkpoint_audit_20260831.json | research | MFRNet 官方权重下载、隔离环境、B64/K8 loader、297/297 strict load 与 test-selection 边界证据 |
| 2026-08-31 11:08 | /run-experiment | evidence/mfrnet_eval_batch_semantics_audit_20260831.json | research | Tutel capacity/batch-priority 的确定性 batch 分割反例与 MFRNet B128 parity 边界 |
| 2026-08-31 11:08 | /tdd | docs/MFRNET_CHECKPOINT_REPRODUCTION_SPEC.md | implementation | 提议 MFRNet preflight/official128 公共接缝、显存门禁、原始日志与原子 receipt 合同 |
| 2026-08-31 11:08 | /tdd | docs/TDD_SEAMS.md | implementation | 将待确认的 MFRNet B128 checkpoint 评测 CLI 纳入接缝协议 |
| 2026-08-31 11:20 | /experiment-plan | protocols/claim_gates_v1.json | implementation | 在 R020+ 结果前冻结三创新点效应量、identity-cluster 统计、校准、鲁棒性和 SOTA 门 |
| 2026-08-31 11:20 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260831_112010.md | implementation | v1.2 永久版本：量化 claim gates、必要控制、stop/go 与本地算力工期 |
| 2026-08-31 11:20 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | v1.2 latest copy，与 timestamped 版本逐字节一致 |
| 2026-08-31 11:20 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_112010.md | implementation | 新增 HFER 方向/深度/同构/容量、CIRC 粒度/泄漏/温度及统计运行项 |
| 2026-08-31 11:20 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest tracker copy，与 timestamped 版本逐字节一致 |
| 2026-08-31 11:25 | /experiment-plan | evidence/claim_gates_v1_preregistration_20260831.json | implementation | 绑定量化门、版本化计划及 50 个 R020+ 零完成结果的 pre-result ordering receipt |
| 2026-08-31 11:25 | /experiment-plan | evidence/SHA256SUMS | implementation | 将 claim-gates 预注册 receipt 纳入第 29 份版本化证据 |
| 2026-08-31 11:25 | /experiment-plan | evidence/README.md | implementation | 记录预注册 receipt 的证明范围与非有效性边界 |
| 2026-08-31 11:25 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260831_112526.md | implementation | evidence count 更新后的永久 tracker 版本 |
| 2026-08-31 11:25 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest tracker copy，与 112526 timestamped 版本逐字节一致 |
| 2026-08-31 11:41 | /research | docs/BASELINE_SELECTION_AND_LICENSE_AUDIT_2026-08-31.md | research | 固定 Signal 高指标 MIT baseline、DeMo 实现底座及无仓库许可证比较器的隔离复用边界 |
| 2026-08-31 11:41 | /research | evidence/baseline_license_audit_20260831.json | research | 七个本地官方仓库的精确 commit/tree/license/hash 与 baseline 角色机器可读收据 |
| 2026-08-31 11:41 | /research | evidence/SHA256SUMS | implementation | 将 baseline license receipt 纳入第 30 份版本化证据 |
| 2026-08-31 11:41 | /research | refine-logs/EXPERIMENT_TRACKER_20260831_114139.md | research | baseline 许可证角色纠正后的永久 tracker 版本 |
| 2026-08-31 11:41 | /research | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest tracker copy，与 114139 timestamped 版本逐字节一致 |
| 2026-08-31 11:57 | /tdd | evidence/tdd_seam_consent_20260831.json | implementation | 记录用户精确“接缝同意”、五份冻结合同哈希及 TriFusion/PEFT/MFRNet 接缝授权边界 |
| 2026-08-31 11:57 | /tdd | docs/IMPLEMENTATION_BLUEPRINT_V1.md | implementation | 将核心实现状态从等待同意切换为按 TDD 纵向切片实施 |
| 2026-08-31 11:57 | /tdd | refine-logs/EXPERIMENT_TRACKER.md | implementation | R004 开始、R020-R023/R026 解锁；实验结果与 GPU 门状态不变 |
| 2026-08-31 12:03 | /tdd | utils/reid_evaluation.py | implementation | 手算 CMC/mAP、同身份同相机剔除、无正样本跳过/全无正样本失败的统一评估内核 |
| 2026-08-31 12:03 | /tdd | tests/test_reid_evaluation.py | implementation | 四次红绿循环并覆盖旧入口 ragged CMC 回归；全仓 21/21 PASS |
| 2026-08-31 12:41 | /tdd | modeling/trifusion/ | implementation | 六个冻结公共接缝、三阶段深协作、HFER/CIRC/URGC、可选 RDPT、criterion 与真实 CLIP 构建器 |
| 2026-08-31 12:41 | /tdd | tools/build_circ_targets.py | implementation | 身份外折全网络干预记录的 fail-closed 编译、两条 hash-edge 审计及三类原子收据 |
| 2026-08-31 12:41 | /tdd | evidence/trifusion_core_tdd_20260831.json | implementation | 39/39 PASS、真实 CLIP/Mamba 默认构建、95,893,482 参数与 GPU 门失败边界收据 |
| 2026-09-01 19:40 | /run-experiment | environment/signal_environment.yml | implementation | 远端实建 Signal Python3.10.13/torch2.1.1+cu118 环境；移除已证实无关且失败的 ninja 构建工具 |
| 2026-09-01 19:40 | /run-experiment | environment/signal_requirements-lock.txt | implementation | 与远端 Signal 训练环境 `pip freeze` 逐字节一致的 86 包运行时锁 |
| 2026-09-01 19:40 | /run-experiment | environment/SIGNAL_BASELINE.md | documentation | Signal 环境重建命令及 grad-cam/visdom/ninja 三项证据化排除说明 |
| 2026-09-01 19:40 | /run-experiment | comparators/signal_cd1b0a6_clip_path.patch | implementation | 将 Signal 作者机器硬编码 CLIP 路径参数化为现有 cfg 字段的单行补丁 |
| 2026-09-01 19:40 | /run-experiment | tools/run_signal_baseline_dev.py | implementation | Signal 原模型/损失/调度器的 seed42 B64/K8 141-fit/30-dev 训练与 3072D 终局回执入口 |
| 2026-09-01 20:10 | /monitor-experiment | evidence/signal_baseline_dev_terminal_seed42.json | research | Signal seed42 50/50 epoch 终局：58.0109/57.4545/69.9394/76.6061，完整3072D、SIE、official0 |
| 2026-08-31 12:49 | /run-experiment | evidence/signal_environment_smoke_20260831.json | research | MIT Signal 固定源码、真实 RGBNT201 B64/K8 三模态 loader 与 GPU 门失败边界收据 |
| 2026-08-31 13:11 | /tdd | tools/run_peft_boa_resumable.py | implementation | PEFT-BoA B64/K4容量、双代全状态恢复、e80/e120预评测导出及fixed120单次测试驱动 |
| 2026-08-31 13:11 | /tdd | tools/run_mfrnet_checkpoint_eval.py | implementation | MFRNet immutable preflight、官方B128执行、日志哈希与终态分类驱动 |
| 2026-08-31 13:11 | /tdd | evidence/baseline_runners_tdd_20260831.json | implementation | 45/45全仓PASS、两个真实preflight仅因1025MiB GPU门阻塞且零指标的边界收据 |
| 2026-08-31 13:36 | /tdd | modeling/trifusion/data.py | implementation | 仅使用train_171的141-fit/30-dev真实RGB/NI/TI加载器与B16/K4采样合同 |
| 2026-08-31 13:36 | /tdd | tools/run_trifusion_experiment.py | implementation | TriFusion capacity、单batch过拟合、60epoch train-only dev及双代全状态恢复驱动 |
| 2026-08-31 13:36 | /tdd | configs/RGBNT201/TriFusion.yml | implementation | core_pre_circ B16/K4、60epoch、零official-test开发配置 |
| 2026-08-31 13:36 | /tdd | evidence/trifusion_training_readiness_20260831.json | implementation | 51/51 PASS、真实数据接口及1035MiB显存门禁前阻塞且零指标的训练就绪收据 |
| 2026-09-01 12:23 | /run-experiment | ../artifacts/trifusion_shared_semantic_circ_urgc_directional_final_seed42/ | research | seed42、epoch60、官方测试一次评估的正式结果；fused 59.1478 mAP / 63.2775 R1 |
| 2026-09-01 12:23 | /tdd | tools/repair_trifusion_directional_final_completion.py | implementation | 缺失导入后的 audit-only 完成修复器；禁止优化器 step 和官方测试重评 |
| 2026-09-01 12:23 | /tdd | tests/test_directional_final_completion_repair.py | implementation | 最终修复状态、一次访问、定向授权和失败回执的 fail-closed 测试 |
| 2026-09-01 12:23 | /analyze-results | results/TRIFUSION_RGBNT201_FINAL_SEED42_2026-09-01.md | research | 四路正式指标、目标差距、训练/校准分析与下一步边界 |
| 2026-09-01 12:23 | /result-to-claim | findings.md | research | 高置信度 claim_supported=no 与禁止消融的路由结论 |
| 2026-09-01 12:23 | /experiment-audit | EXPERIMENT_AUDIT.md | research | 独立 GPT-5.5 完整性审计；整体 WARN、指标链 PASS、旧交接状态需更新 |
| 2026-09-01 12:23 | /experiment-audit | EXPERIMENT_AUDIT.json | research | 完整性审计机器可读结论 |
| 2026-09-01 12:23 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | research | 正式结果、修复账本、证据哈希、测试状态和负结论的统一交接入口 |
| 2026-09-01 12:56 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260901_125629.md | implementation | V2 anchor-preserving、stage-updated、information-preserving 主方法恢复计划 |
| 2026-09-01 12:56 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest V2 主方法恢复计划 |
| 2026-09-01 12:56 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260901_125629.md | implementation | V2 main-only train/dev 执行 tracker |
| 2026-09-01 12:56 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest V2 执行 tracker |
| 2026-09-01 13:29 | /tdd | modeling/trifusion/cascade_v2.py | implementation | 隔离实现APSD锚点保持、SURE逐阶段质量刷新、QIPF 5120维无损融合及效用排序监督 |
| 2026-09-01 13:29 | /tdd | modeling/trifusion/cascade_v2_builder.py | implementation | 真实CLIP初始化、93.97M参数与三创新点provenance的独立V2构建入口 |
| 2026-09-01 13:29 | /tdd | tools/run_trifusion_cascade_v2.py | implementation | 不改变冻结V1 runner/协议哈希的独立V2训练启动器 |
| 2026-09-01 13:29 | /tdd | tests/test_trifusion_cascade_v2.py | implementation | 三项V2机制、真实CLIP构建、宽检索头、路由排序与label smoothing的红绿测试 |
| 2026-09-01 13:29 | /run-experiment | configs/RGBNT201/TriFusion-cascade-v2-hfer-uniform-rtx3090.yml | implementation | seed42、B32/K4、train-only V2 uniform-selector 配置 |
| 2026-09-01 13:48 | /run-experiment | ../artifacts/trifusion_cascade_v2_isolated_hfer_uniform_seed42_capacity/ | research | 隔离V2的93.97M参数、峰值6248MiB reserved、8步梯度覆盖100%真实3090容量门 |
| 2026-09-01 13:48 | /run-experiment | ../artifacts/trifusion_cascade_v2_isolated_hfer_uniform_seed42_overfit/ | research | 隔离V2固定批100步损失比例0.088862的学习能力门 |
| 2026-09-01 13:29 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260901_132912.md | implementation | M0全通过并将V2-R003切换为READY的永久追踪版本 |
| 2026-09-01 13:29 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest V2执行追踪器，记录真实3090门禁证据 |
| 2026-09-01 13:48 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260901_134819.md | implementation | V1哈希链恢复、141 PASS及隔离V2门禁证据的永久追踪版本 |
| 2026-09-01 13:48 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest V2执行追踪器，V2-R003已READY |
| 2026-09-01 15:50 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260901_155059.md | implementation | V3 task-anchor主实验冻结摘要：direct CLIP anchor、有界三专家残差、身份质量路由 |
| 2026-09-01 15:50 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest V3主实验计划；登记V1/V2完整失败证据及V3晋级门 |
| 2026-09-01 15:50 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260901_155059.md | implementation | V3 main-only执行状态冻结副本 |
| 2026-09-01 15:50 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest V3执行追踪器；V3-R000进行中 |
| 2026-09-01 17:11 | /tdd | tools/diagnose_trifusion_task_anchor_v3.py | implementation | 冻结 V3 最佳 checkpoint 的 anchor、三专家、路由残差、能量和熵只读诊断器 |
| 2026-09-01 17:11 | /tdd | tests/test_trifusion_task_anchor_v3_diagnostic.py | implementation | 诊断视图重建、缺失模态与外部 checkpoint 拒绝的红绿测试 |
| 2026-09-01 17:11 | /analyze-results | evidence/trifusion_task_anchor_v3_diagnostic_seed42_f32990b.json | research | epoch14 精确指标 parity、anchor/residual 指标、残差范数和路由熵冻结证据 |
| 2026-09-01 17:11 | /result-to-claim | findings.md | research | V3 claim_supported=no、已证实与推断根因边界及 V4 路由决定 |
| 2026-09-01 17:11 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260901_171116.md | implementation | V4 非破坏式等能量身份效用路由残差银行的主方法恢复永久计划 |
| 2026-09-01 17:11 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest V4 主方法计划 |
| 2026-09-01 17:11 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260901_171116.md | implementation | V3 负结果闭环与 V4 main-only 执行追踪永久版本 |
| 2026-09-01 17:11 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest V4 执行追踪器 |
| 2026-09-01 18:01 | /tdd | modeling/trifusion/task_anchor_v4.py | implementation | 非破坏式等能量三专家残差银行、身份效用路由监督及 V4 输出/criterion |
| 2026-09-01 18:01 | /tdd | modeling/trifusion/task_anchor_v4_builder.py | implementation | 真实 CLIP/V3 专家主干复用、6144 维 fused 与 95.20M 参数 V4 构建器 |
| 2026-09-01 18:01 | /run-experiment | tools/run_trifusion_task_anchor_v4.py | implementation | 独立 V4 审计启动器、预训练参数分组、AMP256 稳定资源档案与源码哈希 |
| 2026-09-01 18:01 | /run-experiment | configs/RGBNT201/TriFusion-task-anchor-v4-core-rtx3090.yml | implementation | seed42、B32/K4、AMP256、60epoch、utility-router loss 的主实验冻结配置 |
| 2026-09-01 18:01 | /tdd | tests/test_trifusion_task_anchor_v4.py | implementation | 残差独立性、等能量、零残差距离、身份效用目标、路由梯度与真实 CLIP 构建红绿测试 |
| 2026-09-01 18:01 | /run-experiment | evidence/trifusion_task_anchor_v4_readiness_seed42.json | research | 3090 capacity/overfit 收据哈希、0 overflow、100% 梯度覆盖与 official0 的 readiness 证据 |
| 2026-09-01 18:01 | /run-experiment | evidence/README.md | research | 补充 V4 readiness 证据范围与非 dev/非 SOTA 边界 |
| 2026-09-01 18:01 | /run-experiment | evidence/SHA256SUMS | implementation | 登记 V4 readiness 机器可读收据 SHA-256 |
| 2026-09-01 18:01 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260901_180105.md | implementation | V4 M1 实测门禁回填后的永久主方法计划 |
| 2026-09-01 18:01 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest V4 主方法计划，与 180105 永久版本逐字节一致 |
| 2026-09-01 18:01 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260901_180105.md | implementation | V4 M1 全门通过、V4-R003 READY 的永久追踪版本 |
| 2026-09-01 18:01 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest V4 执行追踪器，完整 dev 主实验已解锁 |
| 2026-09-01 19:00 | /monitor-experiment | evidence/trifusion_task_anchor_v4_dev_terminal_seed42.json | research | V4 60/60 dev 终态、epoch27 四路指标、official0、恢复/结果哈希与失败门 |
| 2026-09-01 19:00 | /analyze-results | findings.md | research | V4 fused 低于 Mamba 0.6628 mAP、距65门21.5969及 Signal baseline 缺口 |
| 2026-09-01 19:00 | /result-to-claim | findings.md | research | 纠正 V3/V4 anchor 混淆后的 claim_supported=no 与 baseline-first 路由 |
| 2026-09-01 19:00 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260901_190000.md | implementation | V5 完整 Signal baseline floor、梯度隔离、双输出与晋级/回退永久计划 |
| 2026-09-01 19:00 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest baseline-preserving V5 主方法计划 |
| 2026-09-01 19:00 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260901_190000.md | implementation | V4 负结果闭环与 V5 baseline-first 永久 tracker |
| 2026-09-01 19:00 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | latest V5 baseline-first 执行追踪器 |
| 2026-09-01 19:00 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | research | 续写服务器统一交接：V3/V4 终态、Signal 完整特征缺口与 baseline 保底合同 |
| 2026-09-01 19:00 | /handoff | README.md | research | 公开入口回填 V4 负结果和 Signal baseline-first 状态 |
| 2026-09-01 20:29 | /tdd | modeling/trifusion/signal_preserving_v5.py | implementation | 冻结完整 Signal 3072D baseline，三阶段三专家协作、质量/身份效用路由与非破坏式 2304D 残差银行 |
| 2026-09-01 20:29 | /tdd | modeling/trifusion/signal_preserving_v5_builder.py | implementation | 使用现有 CNN/Transformer/Mamba 完整专家、两次 HFER 与三次可靠性刷新的 V5 独立构建器 |
| 2026-09-01 20:29 | /tdd | tests/test_trifusion_signal_preserving_v5.py | implementation | baseline 前缀、Signal 冻结、专家/路由梯度与两阶段交换合同；远端 4 passed |
| 2026-09-01 20:29 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 纠正 Signal 运行中旧状态，记录 V5 已实现/未完成边界、包名冲突和接续技能 |
| 2026-09-01 21:01 | /tdd | tests/test_run_signal_preserving_v5.py | implementation | 冻结配置、五路晋级门、损失权重、过拟合比例和 warmup+cosine 公开 runner 合同 |
| 2026-09-01 21:01 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v5-rtx3090.yml | implementation | seed42、B32/K4、60epoch、完整 Signal checkpoint、五路输出与 65mAP 主门冻结配置 |
| 2026-09-01 21:01 | /run-experiment | tools/run_signal_preserving_v5.py | implementation | preflight/capacity/overfit/dev 远端启动器；同 checkpoint 五路单次评估、best-fused 严格重载与 official0 |
| 2026-09-01 21:01 | /tdd | modeling/trifusion/signal_preserving_v5_builder.py | implementation | 冻结实测无目标的三组 private projection，保留 213/213 可达梯度张量 |
| 2026-09-01 21:01 | /run-experiment | evidence/trifusion_signal_preserving_v5_preflight_seed42.json | research | 825/825 dev 逐元素 Signal parity、四指标精确一致、official0 |
| 2026-09-01 21:01 | /run-experiment | evidence/trifusion_signal_preserving_v5_capacity_seed42.json | research | 真实 B32/K4 8-step、213/213 梯度、0 overflow、3542MiB reserved、Signal SHA 不变 |
| 2026-09-01 21:01 | /run-experiment | evidence/trifusion_signal_preserving_v5_overfit_seed42.json | research | 真实固定批 100-step loss ratio 0.02102、0 overflow、Signal SHA 不变 |
| 2026-09-01 21:01 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 回填 V5 三项工程门 PASS、实测缺梯度修复和完整 dev 待办 |
| 2026-09-01 21:52 | /tdd | tools/diagnose_signal_preserving_v5.py | implementation | 冻结 best checkpoint 的残差能量、路由熵、分支相似度、距离排序与参数更新只读诊断器 |
| 2026-09-01 21:52 | /tdd | tests/test_diagnose_signal_preserving_v5.py | implementation | 残差范数、分支余弦和归一化路由熵的最小回归合同；远端 1 passed |
| 2026-09-01 21:52 | /monitor-experiment | evidence/trifusion_signal_preserving_v5_dev_terminal_seed42.json | research | V5 60/60 dev、epoch51 五路严格重载、Signal SHA 不变、5498 steps、0 overflow、official0 |
| 2026-09-01 21:52 | /analyze-results | evidence/trifusion_signal_preserving_v5_diagnostic_seed42.json | research | 全825 dev 的残差能量、距离变化、Top-10 重合、路由熵与模块更新量只读诊断 |
| 2026-09-01 21:52 | /result-to-claim | .aris/traces/result-to-claim/2026-09-01_run03/ | research | V5 claim_supported=partial：支持 exact baseline 工程子主张，不支持融合增益、65mAP 或 SOTA |
| 2026-09-01 21:52 | /analyze-results | results/TRIFUSION_RGBNT201_V5_DEV_SEED42_2026-09-01.md | research | V5 五路指标、失败门、检索几何诊断与下一步 main-only 边界 |
| 2026-09-01 21:52 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 服务器统一交接续写 V5 完整训练终局、诊断根因、证据哈希与接续顺序 |
| 2026-09-01 21:52 | /handoff | README.md | documentation | 公开入口更新 V5 终局指标、失败门和检索排序近似不变结论 |
| 2026-09-01 21:52 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | latest 计划关闭 V5 official/消融并限定一个诊断驱动 main-only 修正 |
| 2026-09-01 21:52 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | research | V5 readiness、dev、诊断和 result-to-claim 四项终态回填 |
| 2026-09-01 22:30 | /tdd | modeling/trifusion/signal_preserving_v6.py | implementation | exact Signal 前缀、无自由倍率等能量残差激活、residual-only 监督与路由合同 |
| 2026-09-01 22:30 | /tdd | modeling/trifusion/signal_preserving_v6_builder.py | implementation | 复用三完整专家、两次 HFER、三次可靠性刷新并构建 V6 五路输出 |
| 2026-09-01 22:30 | /tdd | tests/test_trifusion_signal_preserving_v6.py | implementation | baseline 字节保持、残差能量激活、残差梯度、路由梯度和真实 builder 公共接缝 |
| 2026-09-01 22:30 | /tdd | tests/test_run_signal_preserving_v6.py | implementation | residual-only 损失权重、seed42/B32K4/60epoch 与 V6 独立 receipt identity 合同 |
| 2026-09-01 22:30 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v6-rtx3090.yml | implementation | 单一 V6 seed42 主实验配置；沿用 V5 优化设置且不含可扫描残差倍率 |
| 2026-09-01 22:30 | /run-experiment | tools/run_signal_preserving_v6.py | implementation | 复用审计 runner 的 V6 独立入口；V5/V6 显式 fail-closed 分派与版本化 schema |
| 2026-09-01 22:45 | /run-experiment | evidence/trifusion_signal_preserving_v6_preflight_seed42.json | research | V6 全825 dev exact Signal parity、独立 schema、optimizer0、official0 |
| 2026-09-01 22:45 | /run-experiment | evidence/trifusion_signal_preserving_v6_capacity_seed42.json | research | RTX3090 B32/K4 8-step、218/218梯度、3554MiB reserved、0 overflow |
| 2026-09-01 22:45 | /run-experiment | evidence/trifusion_signal_preserving_v6_overfit_seed42.json | research | 同一真实批100-step loss ratio0.05655、Signal SHA不变、official0 |
| 2026-09-01 23:20 | /monitor-experiment | evidence/trifusion_signal_preserving_v6_dev_terminal_seed42.json | research | V6 60/60 dev、epoch8五路严格重载、Signal SHA不变、5498 steps、0 overflow、official0 |
| 2026-09-01 23:20 | /analyze-results | evidence/trifusion_signal_preserving_v6_diagnostic_seed42.json | research | 全825 dev的残差指标、检索距离变化、Top-10重合、路由熵/权重与模块更新只读诊断 |
| 2026-09-01 23:20 | /analyze-results | results/TRIFUSION_RGBNT201_V6_DEV_SEED42_2026-09-01.md | research | V6五路指标、失败门、训练回落、检索几何变化和路由失配结论 |
| 2026-09-01 23:20 | /result-to-claim | .aris/traces/result-to-claim/2026-09-01_run04/ | research | V6 claim_supported=no/high/provisional：只支持exact Signal和dev +0.7212，不支持best-expert、65mAP或SOTA |
| 2026-09-01 23:20 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 回填V6完整失败终态，并只解锁V7 marginal-gain routing main-only修正 |
| 2026-09-01 23:20 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | research | V6 dev、诊断、result-to-claim终态与V7单主实验边界 |
| 2026-09-01 23:20 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 服务器统一交接续写V6完整训练、五路指标、路由根因、证据路径和V7接续边界 |
| 2026-09-01 23:20 | /handoff | README.md | documentation | 公开入口更新V6终局指标、失败门和路由失配结论 |
| 2026-09-02 04:51 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_045122.md | implementation | V9 orthogonal triadic relay synthesis 时间戳计划；从 V8 弱路由失败转向新表示生成 |
| 2026-09-02 04:51 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V9 claim map、TDD/readiness、单次 final-only dev 与条件 official/消融固定入口 |
| 2026-09-02 04:51 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_045122.md | implementation | V8 Phase-B 终态与 V9 fail-closed run order 快照 |
| 2026-09-02 04:51 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V9-R000 开始；后续 capacity/overfit/dev/official/消融依次受门禁约束 |
| 2026-09-02 05:10 | /tdd | tests/test_trifusion_signal_preserving_v9.py | implementation | V9 peer sensitivity、逐轮正交、exact prefix、三路协同输出及冻结梯度公共接缝 |
| 2026-09-02 05:10 | /tdd | modeling/trifusion/signal_preserving_v9.py | implementation | 两轮 receiver-specific orthogonal relay、triadic synthesis、V9 五路模型与训练 criterion |
| 2026-09-02 05:10 | /tdd | modeling/trifusion/signal_preserving_v9_builder.py | implementation | 从冻结 V8 Phase-A-plus-Router 构建单一 V9，登记三项论文机制与参数边界 |
| 2026-09-02 05:25 | /tdd | tests/test_run_signal_preserving_v9.py | implementation | seed42/B64K8/final-only 配置与 train-only runner/import/访问边界合同 |
| 2026-09-02 05:25 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v9-rtx3090.yml | implementation | 单一 V9 配置；冻结 V8 combined checkpoint、两轮中继、60 epoch 与原65门 |
| 2026-09-02 05:25 | /run-experiment | tools/run_signal_preserving_v9.py | implementation | train-only preflight/capacity/overfit/final training，状态 SHA 与0 dev/official receipts |
| 2026-09-02 05:35 | /tdd | tests/test_evaluate_signal_preserving_v9_dev.py | implementation | 65门、严格超过五输出与单次冻结评估无 optimizer 合同 |
| 2026-09-02 05:35 | /run-experiment | tools/evaluate_signal_preserving_v9_dev.py | implementation | final checkpoint 六路真实 dev、状态不变、beta/relay 诊断及0 official receipt |
| 2026-09-02 05:40 | /run-experiment | evidence/trifusion_v9_preflight_seed42.json | research | V9 真实 train-only exact prefix/orthogonal relay/frozen state 门 PASS |
| 2026-09-02 05:40 | /run-experiment | evidence/trifusion_v9_capacity_seed42.json | research | V9 B64/K8 8-step，59/59 梯度、2020MiB reserved、0 overflow |
| 2026-09-02 05:40 | /run-experiment | evidence/trifusion_v9_overfit_seed42.json | research | V9 真实固定批100-step excess-loss ratio 0.000518，冻结状态不变 |
| 2026-09-02 05:40 | /run-experiment | results/TRIFUSION_RGBNT201_V9_READINESS_2026-09-02.md | research | V9 三项 train-only readiness、主门与访问边界汇总 |
| 2026-09-02 05:46 | /monitor-experiment | evidence/trifusion_v9_train_seed42.json | research | V9 seed42 B64/K8 60/60、2520 steps、0 overflow、dev0/official0 与冻结 state SHA |
| 2026-09-02 05:46 | /analyze-results | evidence/trifusion_v9_dev_seed42.json | research | final checkpoint 唯一冻结 dev：fused56.5339，低Signal1.4770、低Phase-B1.8711、official0 |
| 2026-09-02 05:46 | /analyze-results | results/TRIFUSION_RGBNT201_V9_DEV_SEED42_2026-09-02.md | research | V9 六路指标、训练完整性、beta/relay 诊断与终局封存决定 |
| 2026-09-02 05:46 | /result-to-claim | .aris/traces/result-to-claim/2026-09-02_run04/ | research | V9 claim_supported=no/high；不支持协同、65、official、SOTA或Mamba必要性 |
| 2026-09-02 05:46 | /experiment-audit | EXPERIMENT_AUDIT_V9.md | research | V9独立审计WARN/warn/FAIL_TO_PROMOTE；GT/归一化/评价类型PASS，远端artifact包装WARN |
| 2026-09-02 05:46 | /experiment-audit | EXPERIMENT_AUDIT_V9.json | research | V9完整性审计机器可读终局 |
| 2026-09-02 05:46 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V9主门失败回填与后继fit-only identity-OOF正效用前置门 |
| 2026-09-02 05:46 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | research | V9-R000/R001/R002 PASS、R003 FAIL、R004/R005关闭，未授权V10 |
| 2026-09-02 05:46 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_054650.md | implementation | 与latest逐字节一致的V9终局计划永久快照 |
| 2026-09-02 05:46 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_054650.md | research | 与latest逐字节一致的V9终局tracker永久快照 |
| 2026-09-02 05:46 | /run-experiment | evidence/SHA256SUMS | implementation | 登记V9 preflight/capacity/overfit/train/dev五份轻量证据SHA-256 |
| 2026-09-02 05:46 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 统一交接续写V9训练、唯一dev负结果、独立审计和后继研究边界 |
| 2026-09-02 06:03 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V10共享CLIP+DINO语义场、三专家分工及identity-OOF效用保护的条件式主假设；先做零训练fit-only资格门 |
| 2026-09-02 06:03 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | research | V10-Q0 READY；Q1/Q2/dev全部受冻结DINO互补与身份隔离正效用门约束 |
| 2026-09-02 06:03 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_060304.md | implementation | 与latest逐字节一致的V10条件式主假设永久快照 |
| 2026-09-02 06:03 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_060304.md | research | 与latest逐字节一致的V10资格门tracker永久快照 |
| 2026-09-02 06:18 | /tdd | tests/test_probe_v10_dinov2_fit_utility.py | implementation | DINO输入/strict key/CLS+patch mean/等块融合/资格门真实RED→GREEN合同 |
| 2026-09-02 06:18 | /run-experiment | tools/probe_v10_dinov2_fit_utility.py | implementation | 冻结Phase-B+DINO fit-only特征、Oracle和fail-closed资格探针；optimizer0/dev0/official0 |
| 2026-09-02 06:18 | /analyze-results | evidence/trifusion_v10_dinov2_fit_qualification_seed42.json | research | Q0终态：PhaseB/DINO/concat mAP100/7.6284/92.2120、Oracle gain0、gate false |
| 2026-09-02 06:18 | /analyze-results | results/TRIFUSION_RGBNT201_V10_DINOV2_FIT_QUALIFICATION_2026-09-02.md | research | V10-Q0指标、工程链、失败门、饱和边界和停止决定 |
| 2026-09-02 06:18 | /result-to-claim | .aris/traces/result-to-claim/2026-09-02_run05/ | research | V10-Q0 claim_supported=no/high；不支持互补或继续V10，也不外推否定DINO |
| 2026-09-02 06:18 | /experiment-audit | EXPERIMENT_AUDIT_V10_Q0.md | research | WARN/warn/FAIL_TO_QUALIFY；GT/归一化/路径/scope PASS，远端binary包装WARN |
| 2026-09-02 06:18 | /experiment-audit | EXPERIMENT_AUDIT_V10_Q0.json | research | V10-Q0独立审计机器可读终局 |
| 2026-09-02 06:18 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | 回填V10-Q0失败并关闭Q1/Q2/dev及所有DINO事后扫描 |
| 2026-09-02 06:18 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | research | V10-Q0 COMPLETE-FAIL；后续V10全部关闭，V11未授权 |
| 2026-09-02 06:18 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 统一交接续写V10冻结DINO资格负结果和研究边界 |
| 2026-09-02 06:18 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_061819.md | implementation | 与latest逐字节一致的V10-Q0终局计划永久快照 |
| 2026-09-02 06:18 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_061819.md | research | 与latest逐字节一致的V10-Q0终局tracker永久快照 |
| 2026-09-02 06:29 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_062924.md | implementation | V11非饱和identity-OOF双基础残差互补资格计划永久快照 |
| 2026-09-02 06:29 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V11固定折内residual-only/DINO门与条件式主模型路径latest |
| 2026-09-02 06:29 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_062924.md | implementation | V11 Q0/Q1/Q2/final门禁tracker永久快照 |
| 2026-09-02 06:29 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V11-Q0 READY；后续实现、训练和dev均受门禁约束 |
| 2026-09-02 06:39 | /tdd | tests/test_probe_v11_dinov2_oof_residual_complement.py | implementation | residual bank、折内query聚合与fail-closed资格判定公共接缝RED→GREEN合同 |
| 2026-09-02 06:39 | /run-experiment | tools/probe_v11_dinov2_oof_residual_complement.py | implementation | 三折held-out residual-only/DINO固定资格探针，optimizer0/dev0/official0 |
| 2026-09-02 06:55 | /analyze-results | evidence/trifusion_v11_dinov2_oof_residual_complement_seed42.json | research | V11-Q0终态：bank/DINO/concat mAP100/14.1323/95.8582，non-saturation false、gate false |
| 2026-09-02 06:55 | /analyze-results | evidence/trifusion_v11_dinov2_oof_residual_complement_seed42_provenance.json | research | 绑定a29692a及probe/test/config/result SHA和远端result路径 |
| 2026-09-02 06:55 | /analyze-results | results/TRIFUSION_RGBNT201_V11_DINOV2_OOF_RESIDUAL_QUALIFICATION_2026-09-02.md | research | V11-Q0折指标、饱和来源、失败门、工程完整性和停止决定 |
| 2026-09-02 06:55 | /result-to-claim | .aris/traces/result-to-claim/2026-09-02_run06/ | research | V11-Q0 claim_supported=no/high；不支持DINO互补、后续实现或dev |
| 2026-09-02 06:55 | /experiment-audit | .aris/traces/experiment-audit/2026-09-02_run05/ | research | GPT-5.5只读完整性审计完整trace |
| 2026-09-02 06:55 | /experiment-audit | EXPERIMENT_AUDIT_V11_Q0.md | research | WARN/warn/FAIL_TO_QUALIFY；100mAP为all-fit Signal饱和而非指标欺诈 |
| 2026-09-02 06:55 | /experiment-audit | EXPERIMENT_AUDIT_V11_Q0.json | research | V11-Q0独立审计机器可读终局 |
| 2026-09-02 06:55 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V11-Q0失败回填并关闭Q1/Q2/dev与DINO事后扫描 |
| 2026-09-02 06:55 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_065520.md | implementation | 与latest逐字节一致的V11-Q0终局计划永久快照 |
| 2026-09-02 06:55 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V11-Q0 COMPLETE-FAIL；Q1/Q2/R001关闭，V12未授权 |
| 2026-09-02 06:55 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_065520.md | implementation | 与latest逐字节一致的V11-Q0终局tracker永久快照 |
| 2026-09-02 06:55 | /analyze-results | findings.md | research | 登记V11完整路径非隔离、固定DINO无独有增益和后续约束 |
| 2026-09-02 06:55 | /run-experiment | evidence/SHA256SUMS | implementation | 登记V11原始结果和provenance wrapper SHA-256 |
| 2026-09-02 06:55 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 统一交接续写V11资格失败、100mAP饱和来源与停止边界 |
| 2026-09-02 07:09 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_070902.md | implementation | V12完整路径identity-OOF教师计划永久快照 |
| 2026-09-02 07:09 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V12 Signal50+专家20三折隔离门与条件式Router/dev latest |
| 2026-09-02 07:09 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_070902.md | implementation | V12 Q0/Q1/dev fail-closed执行tracker永久快照 |
| 2026-09-02 07:09 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V12-T0 READY；GPU作业受公共接缝和preflight约束 |
| 2026-09-02 07:16 | /tdd | tests/test_build_v12_complete_path_oof_targets.py | implementation | V12完整路径fold隔离、连续重映射与fail-closed资格门公共接缝 |
| 2026-09-02 07:34 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v12-complete-path-oof-rtx3090.yml | implementation | V12 seed42 Signal50+专家20三折train-only固定配置 |
| 2026-09-02 07:34 | /run-experiment | tools/build_v12_complete_path_oof_targets.py | implementation | V12完整路径identity-OOF教师preflight、三折训练、margin cache与资格receipt |
| 2026-09-02 08:47 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v12-complete-path-router-rtx3090.yml | implementation | 复用冻结V8 Router合同，仅替换为V12完整路径OOF margin cache及其SHA |
| 2026-09-02 09:05 | /analyze-results | evidence/trifusion_v12_complete_path_preflight_seed42.json | research | V12 fold0真实B64/K8单步preflight、191 gradients、0 overflow、dev0/official0 |
| 2026-09-02 09:05 | /analyze-results | evidence/trifusion_v12_complete_path_oof_seed42.json | research | V12三折完整路径OOF终态：非饱和、多样性、Oracle与隔离全门PASS |
| 2026-09-02 09:05 | /analyze-results | evidence/trifusion_v12_complete_path_router_seed42.json | research | V12 Router终态：margin/Top1门FAIL、quality PASS、combined null、dev0/official0 |
| 2026-09-02 09:05 | /experiment-audit | evidence/trifusion_v12_complete_path_execution_provenance_seed42.json | research | 绑定V12项目commit、config/runner/log/result与remote cache/checkpoint SHA |
| 2026-09-02 09:05 | /analyze-results | results/TRIFUSION_RGBNT201_V12_COMPLETE_PATH_OOF_ROUTER_2026-09-02.md | research | V12 Q0窄资格正证据与Q1 Router失败的完整终态报告 |
| 2026-09-02 09:05 | /result-to-claim | .aris/traces/result-to-claim/2026-09-02_run07/ | research | 独立GPT-5.5审阅partial/high；只支持Q0 train-only资格，不授权dev |
| 2026-09-02 09:05 | /experiment-audit | EXPERIMENT_AUDIT_V12.md | research | 独立完整性审计WARN/warn/Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE |
| 2026-09-02 09:05 | /experiment-audit | EXPERIMENT_AUDIT_V12.json | research | V12完整性审计机器可读终局 |
| 2026-09-02 09:05 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | research | 回填V12 Q0 PASS、Q1 FAIL并关闭dev/official/消融/多种子 |
| 2026-09-02 09:05 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | research | V12 T0/P1/Q0/Q1与独立review全部终态 |
| 2026-09-02 09:05 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 统一交接续写V12完整路径教师、Router负结果、哈希与接续边界 |
| 2026-09-02 09:05 | /handoff | README.md | documentation | 公开入口更新V12 Q0资格通过、Q1失败和当前deployable best |
| 2026-09-02 09:13 | /research-refine | refine-logs/REFINE_STATE_20260902_091348.json | implementation | V13路径一致关系型反事实路由 refinement 状态永久快照 |
| 2026-09-02 09:13 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V13 refinement latest 状态 |
| 2026-09-02 09:13 | /research-refine | refine-logs/round-0-initial-proposal.md | implementation | V13 initial anchored proposal：actual-path counterfactual utility + relational Router input |
| 2026-09-02 09:21 | /research-refine | refine-logs/round-1-review.md | implementation | GPT-5.5 xhigh round1 原始评审：6.75/10 REVISE |
| 2026-09-02 09:21 | /research-refine | refine-logs/score-history.md | implementation | V13 refinement 评分演进 |
| 2026-09-02 09:21 | /research-refine | refine-logs/round-1-refinement.md | implementation | V13 完整修订：paired OOF teacher / deployment student、固定 alpha、OOF replay gate |
| 2026-09-02 09:21 | /research-refine | refine-logs/REFINE_STATE_20260902_092108.json | implementation | V13 round1 refinement 状态永久快照 |
| 2026-09-02 09:21 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V13 round1 refinement latest 状态 |
| 2026-09-02 09:26 | /research-refine | refine-logs/round-2-review.md | implementation | GPT-5.5 xhigh round2 原始评审：8.00/10 REVISE |
| 2026-09-02 09:26 | /research-refine | refine-logs/round-2-refinement.md | implementation | V13 完整修订：claim边界、read-only transfer、identity-cluster paired hard gates |
| 2026-09-02 09:26 | /research-refine | refine-logs/REFINE_STATE_20260902_092628.json | implementation | V13 round2 refinement 状态永久快照 |
| 2026-09-02 09:26 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V13 round2 refinement latest 状态 |
| 2026-09-02 09:30 | /research-refine | refine-logs/round-3-review.md | implementation | GPT-5.5 xhigh round3 原始评审：9.05/10 READY |
| 2026-09-02 09:30 | /research-refine | refine-logs/FINAL_PROPOSAL_20260902_093054.md | implementation | V13 final proposal 永久快照 |
| 2026-09-02 09:30 | /research-refine | refine-logs/FINAL_PROPOSAL.md | implementation | V13 deployment-aligned counterfactual distillation latest proposal |
| 2026-09-02 09:30 | /research-refine | refine-logs/REVIEW_SUMMARY_20260902_093054.md | implementation | V13 三轮评审解决记录永久快照 |
| 2026-09-02 09:30 | /research-refine | refine-logs/REVIEW_SUMMARY.md | implementation | V13 review summary latest |
| 2026-09-02 09:30 | /research-refine | refine-logs/REFINEMENT_REPORT_20260902_093054.md | implementation | V13 refinement完整报告永久快照 |
| 2026-09-02 09:30 | /research-refine | refine-logs/REFINEMENT_REPORT.md | implementation | V13 refinement report latest |
| 2026-09-02 09:30 | /research-refine | refine-logs/REFINE_STATE_20260902_093054.json | implementation | V13 completed state 永久快照 |
| 2026-09-02 09:30 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V13 completed state latest |
| 2026-09-02 09:33 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_093332.md | implementation | V13 claim-driven执行计划永久快照 |
| 2026-09-02 09:33 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V13 M0→Q0→Q1→conditional dev latest计划 |
| 2026-09-02 09:33 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_093332.md | implementation | V13 fail-closed tracker永久快照 |
| 2026-09-02 09:33 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V13 execution tracker latest |
| 2026-09-02 09:37 | /tdd | tests/test_trifusion_signal_preserving_v13.py | implementation | V13 shared fusion/counterfactual/bootstrap public seam worked examples |
| 2026-09-02 09:37 | /tdd | tests/test_build_v13_deployment_aligned_targets.py | implementation | V13 Q0 fail-closed target/transfer gate seam |
| 2026-09-02 09:37 | /tdd | docs/TDD_SEAMS.md | documentation | 已同意offline builder scope下的V13 public seam扩展 |
| 2026-09-02 09:41 | /tdd | modeling/trifusion/signal_preserving_v13.py | implementation | V13 shared fusion、query-side actual-path utility与identity-cluster bootstrap最小实现 |
| 2026-09-02 09:41 | /tdd | tools/build_v13_deployment_aligned_targets.py | implementation | V13 Q0 target diversity/action-transfer fail-closed gate最小实现 |
| 2026-09-02 09:49 | /tdd | tools/build_v13_deployment_aligned_targets.py | implementation | V13 paired cache、fold checkpoint reuse、preflight/Q0 actual-path builder与receipts |
| 2026-09-02 09:49 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v13-deployment-aligned-rtx3090.yml | implementation | V13 seed42 fixed-alpha paired target/router frozen contract |
| 2026-09-02 10:34 | /analyze-results | evidence/trifusion_v13_target_learnability_diagnostic_seed42.json | experiment | V13 fit-only零训练target learnability原始诊断，dev0/official0 |
| 2026-09-02 10:34 | /analyze-results | results/TRIFUSION_RGBNT201_V13_TARGET_LEARNABILITY_DIAGNOSTIC_2026-09-02.md | analysis | V13近均匀target、fold漂移与局部可观测性分析 |
| 2026-09-02 10:34 | /analyze-results | findings.md | analysis | 追加V13 target-learnability结论与V14唯一允许方向 |
| 2026-09-02 10:34 | /research-refine | refine-logs/v14/round-0-initial-proposal.md | implementation | V14 fold-local retrieval regret与worst-fold Router初始方案 |
| 2026-09-02 10:34 | /research-refine | refine-logs/REFINE_STATE_20260902_103424.json | implementation | V14 round0 review状态永久快照 |
| 2026-09-02 10:34 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V14 refinement latest状态 |
| 2026-09-02 10:42 | /research-refine | refine-logs/v14/round-1-review.md | implementation | GPT-5.5 xhigh V14 round1原始评审：7.20/10 REVISE |
| 2026-09-02 10:42 | /research-refine | refine-logs/v14/round-1-refinement.md | implementation | V14修订：held-out risk/AP/margin门、fold-bound API、诚实OOF边界 |
| 2026-09-02 10:42 | /research-refine | refine-logs/v14/score-history.md | implementation | V14 refinement评分演进 |
| 2026-09-02 10:42 | /research-refine | refine-logs/REFINE_STATE_20260902_104236.json | implementation | V14 round1状态永久快照 |
| 2026-09-02 10:42 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V14 round1 latest状态 |
| 2026-09-02 10:45 | /research-refine | refine-logs/v14/round-2-review.md | implementation | GPT-5.5 xhigh V14 round2评审：9.25/10 READY |
| 2026-09-02 10:45 | /research-refine | refine-logs/FINAL_PROPOSAL_20260902_104547.md | implementation | V14最终方案永久快照 |
| 2026-09-02 10:45 | /research-refine | refine-logs/FINAL_PROPOSAL.md | implementation | V14 fold-robust retrieval-regret Router latest方案 |
| 2026-09-02 10:45 | /research-refine | refine-logs/REVIEW_SUMMARY_20260902_104547.md | implementation | V14两轮评审解决记录永久快照 |
| 2026-09-02 10:45 | /research-refine | refine-logs/REVIEW_SUMMARY.md | implementation | V14 review summary latest |
| 2026-09-02 10:45 | /research-refine | refine-logs/REFINEMENT_REPORT_20260902_104547.md | implementation | V14 refinement完整报告永久快照 |
| 2026-09-02 10:45 | /research-refine | refine-logs/REFINEMENT_REPORT.md | implementation | V14 refinement report latest |
| 2026-09-02 10:45 | /research-refine | refine-logs/score-history_20260902_104547.md | implementation | V14评分演进永久快照 |
| 2026-09-02 10:45 | /research-refine | refine-logs/score-history.md | implementation | V14 score history latest |
| 2026-09-02 10:45 | /research-refine | refine-logs/REFINE_STATE_20260902_104547.json | implementation | V14 completed状态永久快照 |
| 2026-09-02 10:45 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V14 completed latest状态 |
| 2026-09-02 10:48 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_104830.md | implementation | V14 claim-driven M0→Q0→Q1→conditional dev永久计划 |
| 2026-09-02 10:48 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V14最小执行路线latest计划 |
| 2026-09-02 10:48 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_104830.md | implementation | V14 fail-closed tracker永久快照 |
| 2026-09-02 10:48 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V14执行状态latest tracker |
| 2026-09-02 10:55 | /tdd | tests/test_trifusion_signal_preserving_v14.py | implementation | V14 fold-bound risk、minimax comparator与gradient public seams |
| 2026-09-02 10:55 | /tdd | tests/test_train_v14_fold_robust_router.py | implementation | V14 held-out retrieval Q1 gate public seam |
| 2026-09-02 10:55 | /tdd | modeling/trifusion/signal_preserving_v14.py | implementation | V14 cross-camera fold-bound retrieval risk最小实现 |
| 2026-09-02 10:55 | /tdd | tools/train_v14_fold_robust_router.py | implementation | V14 Q0/Q1/source-minimax/worst-fold runner |
| 2026-09-02 10:55 | /tdd | configs/RGBNT201/TriFusion-signal-preserving-v14-fold-robust-router-rtx3090.yml | implementation | V14 seed42冻结合同，无utility temperature |
| 2026-09-02 10:55 | /tdd | docs/TDD_SEAMS.md | documentation | 登记V14已同意public seams |
| 2026-09-02 11:15 | /run-experiment | evidence/trifusion_v14_q0_seed42.json | experiment | V14 real-cache零步Q0 PASS，optimizer0/dev0/official0 |
| 2026-09-02 11:15 | /run-experiment | evidence/trifusion_v14_q1_seed42.json | experiment | V14唯一seed42 Q1终态负结果，无refit/dev |
| 2026-09-02 11:15 | /analyze-results | results/TRIFUSION_RGBNT201_V14_FOLD_ROBUST_ROUTER_2026-09-02.md | analysis | V14 Q0/Q1原始表、质量/资源/claim边界 |
| 2026-09-02 11:15 | /experiment-audit | EXPERIMENT_AUDIT_V14.md | review | GPT-5.5 xhigh独立完整性审计WARN/warn |
| 2026-09-02 11:15 | /experiment-audit | EXPERIMENT_AUDIT_V14.json | review | V14完整性审计机器可读结果 |
| 2026-09-02 11:15 | /result-to-claim | RESULT_TO_CLAIM_V14.md | review | V14 claim_supported=no/high终局判定 |
| 2026-09-02 11:15 | /result-to-claim | RESULT_TO_CLAIM_V14.json | review | V14 result-to-claim机器可读结果 |
| 2026-09-02 11:15 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_111508.md | implementation | V14 terminal tracker永久快照 |
| 2026-09-02 11:15 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V14 terminal tracker latest |
| 2026-09-02 11:15 | /analyze-results | findings.md | analysis | 追加V14终态负结果与禁止扫描边界 |
| 2026-09-02 11:15 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 主交接追加V14 §25终态 |
| 2026-09-02 11:32 | /research-refine | refine-logs/v15/round-0-initial-proposal.md | implementation | V15 CRDE初始锚定方案 |
| 2026-09-02 11:32 | /research-refine | refine-logs/REFINE_STATE_20260902_113220.json | implementation | V14完成态永久快照 |
| 2026-09-02 11:32 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V15 refinement进行中状态 |
| 2026-09-02 11:38 | /research-refine | refine-logs/v15/round-1-review.md | review | V15 GPT-5.5 xhigh首轮6.9/REVISE原始审查 |
| 2026-09-02 11:38 | /research-refine | refine-logs/v15/score-history.md | implementation | V15评分演进 |
| 2026-09-02 11:38 | /research-refine | refine-logs/REFINE_STATE_20260902_113858.json | implementation | V15 proposal阶段状态快照 |
| 2026-09-02 11:38 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V15 round1 review状态 |
| 2026-09-02 11:39 | /research-refine | refine-logs/v15/round-1-refinement.md | implementation | V15反事实/BN/两层交换收敛后的完整修订案 |
| 2026-09-02 11:41 | /research-refine | refine-logs/REFINE_STATE_20260902_114107.json | implementation | V15 round1 review状态快照 |
| 2026-09-02 11:41 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V15 round1 refinement状态 |
| 2026-09-02 11:44 | /research-refine | refine-logs/v15/round-2-review.md | review | V15 GPT-5.5 xhigh二轮8.35/REVISE原始审查 |
| 2026-09-02 11:44 | /research-refine | refine-logs/v15/score-history.md | implementation | V15二轮评分演进 |
| 2026-09-02 11:44 | /research-refine | refine-logs/REFINE_STATE_20260902_114401.json | implementation | V15 round1 refinement状态快照 |
| 2026-09-02 11:44 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V15 round2 review状态 |
| 2026-09-02 11:45 | /research-refine | refine-logs/v15/round-2-refinement.md | implementation | V15冻结regret系数与同tensor反事实后的完整终审案 |
| 2026-09-02 11:45 | /research-refine | refine-logs/REFINE_STATE_20260902_114557.json | implementation | V15 round2 review状态快照 |
| 2026-09-02 11:45 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V15 round2 refinement状态 |
| 2026-09-02 11:47 | /research-refine | refine-logs/v15/round-3-review.md | review | V15 GPT-5.5 xhigh终审9.15/READY原始审查 |
| 2026-09-02 11:47 | /research-refine | refine-logs/v15/score-history.md | implementation | V15三轮评分终态 |
| 2026-09-02 11:47 | /research-refine | refine-logs/FINAL_PROPOSAL_20260902_114746.md | implementation | V15 9.15/READY最终方法方案永久版 |
| 2026-09-02 11:47 | /research-refine | refine-logs/REVIEW_SUMMARY_20260902_114746.md | implementation | V15三轮审查摘要永久版 |
| 2026-09-02 11:47 | /research-refine | refine-logs/REFINEMENT_REPORT_20260902_114746.md | implementation | V15方法演进与边界报告永久版 |
| 2026-09-02 11:47 | /research-refine | refine-logs/score-history_20260902_114746.md | implementation | V15评分历史永久版 |
| 2026-09-02 11:47 | /research-refine | refine-logs/REFINE_STATE_20260902_114746.json | implementation | V15 round2 refinement状态快照 |
| 2026-09-02 11:47 | /research-refine | refine-logs/FINAL_PROPOSAL.md | implementation | V15最终方法方案latest |
| 2026-09-02 11:47 | /research-refine | refine-logs/REVIEW_SUMMARY.md | implementation | V15审查摘要latest |
| 2026-09-02 11:47 | /research-refine | refine-logs/REFINEMENT_REPORT.md | implementation | V15 refinement报告latest |
| 2026-09-02 11:47 | /research-refine | refine-logs/score-history.md | implementation | V15评分历史latest |
| 2026-09-02 11:47 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V15 9.15/READY完成态 |
| 2026-09-02 11:50 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_115051.md | implementation | V15 M0→Q1→conditional D1冻结执行计划 |
| 2026-09-02 11:50 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_115051.md | implementation | V15执行tracker永久版 |
| 2026-09-02 11:50 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V15冻结执行计划latest |
| 2026-09-02 11:50 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V15执行tracker latest |
| 2026-09-02 11:52 | /tdd | docs/TDD_SEAMS.md | documentation | 登记V15已同意CRDE/反事实/Q1 public seams |
| 2026-09-02 12:20 | /tdd | modeling/trifusion/signal_preserving_v15.py | implementation | V15两层异构role-delta交换、matched off/on前向、有效query regret与统一criterion |
| 2026-09-02 12:20 | /tdd | modeling/trifusion/signal_preserving_v15_builder.py | implementation | 冻结Signal/V8专家并仅开放CRDE与source-local heads的V15 builder |
| 2026-09-02 12:20 | /tdd | tests/test_trifusion_signal_preserving_v15.py | implementation | V15零交换、同步无self、异构mixer、配对前向、masked risk与criterion接缝 |
| 2026-09-02 12:20 | /tdd | tests/test_train_signal_preserving_v15.py | implementation | V15 Q1全门联合判定接缝 |
| 2026-09-02 12:20 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v15-crde-rtx3090.yml | implementation | V12三折与V8 Phase-A SHA绑定的seed42 B64/K8 V15冻结合同 |
| 2026-09-02 12:20 | /run-experiment | tools/train_signal_preserving_v15.py | implementation | 独立M0与三折Q1 runner；Q1失败不含自动D1路径 |
| 2026-09-02 14:45 | /research-refine | refine-logs/v16/round-0-initial-proposal.md | implementation | V16 Signal锚定三方关系修复初始锚定方案 |
| 2026-09-02 14:45 | /research-refine | refine-logs/REFINE_STATE_20260902_144550.json | implementation | V16 proposal状态永久快照 |
| 2026-09-02 14:45 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V16 refinement进行中状态 |
| 2026-09-02 14:58 | /research-refine | refine-logs/v16/round-1-review.md | review | V16 GPT-5.5 xhigh首轮7.1/REVISE原始审查 |
| 2026-09-02 14:58 | /research-refine | refine-logs/v16/score-history.md | implementation | V16评分演进 |
| 2026-09-02 14:58 | /research-refine | refine-logs/REFINE_STATE_20260902_145844.json | implementation | V16 round1 review状态永久快照 |
| 2026-09-02 14:58 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V16 round1 review状态 |
| 2026-09-02 15:00 | /research-refine | refine-logs/v16/threshold-freeze-readonly.md | implementation | V16三折fit-only零步阈值冻结证据 |
| 2026-09-02 15:00 | /research-refine | refine-logs/v16/round-1-refinement.md | implementation | V16公平双端点与Signal-hard relation完整修订案 |
| 2026-09-02 15:00 | /research-refine | refine-logs/REFINE_STATE_20260902_150037.json | implementation | V16 round1 refinement状态永久快照 |
| 2026-09-02 15:00 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V16 round1 refinement状态 |
| 2026-09-02 15:07 | /research-refine | refine-logs/v16/round-2-review.md | review | V16 GPT-5.5 xhigh二轮8.13/REVISE原始审查 |
| 2026-09-02 15:07 | /research-refine | refine-logs/v16/score-history.md | implementation | V16二轮评分演进 |
| 2026-09-02 15:07 | /research-refine | refine-logs/v16/threshold-freeze-readonly.md | implementation | 补全V16阈值候选披露并改为fixed-initial activity gate |
| 2026-09-02 15:07 | /research-refine | refine-logs/REFINE_STATE_20260902_150718.json | implementation | V16 round2 review状态永久快照 |
| 2026-09-02 15:07 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V16 round2 review状态 |
| 2026-09-02 15:09 | /research-refine | refine-logs/v16/round-2-refinement.md | implementation | V16 fixed-initial activity与paired draw封口完整修订案 |
| 2026-09-02 15:09 | /research-refine | refine-logs/REFINE_STATE_20260902_150907.json | implementation | V16 round2 refinement状态永久快照 |
| 2026-09-02 15:09 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V16 round2 refinement状态 |
| 2026-09-02 15:13 | /research-refine | refine-logs/v16/round-3-review.md | review | V16 GPT-5.5 xhigh终审9.10/READY原始审查 |
| 2026-09-02 15:13 | /research-refine | refine-logs/v16/score-history.md | implementation | V16三轮评分终态 |
| 2026-09-02 15:13 | /research-refine | refine-logs/FINAL_PROPOSAL_20260902_151345.md | implementation | V16 9.10/READY最终方法方案永久版 |
| 2026-09-02 15:13 | /research-refine | refine-logs/REVIEW_SUMMARY_20260902_151345.md | implementation | V16三轮审查摘要永久版 |
| 2026-09-02 15:13 | /research-refine | refine-logs/REFINEMENT_REPORT_20260902_151345.md | implementation | V16方法演进与边界报告永久版 |
| 2026-09-02 15:13 | /research-refine | refine-logs/score-history_20260902_151345.md | implementation | V16评分历史永久版 |
| 2026-09-02 15:13 | /research-refine | refine-logs/REFINE_STATE_20260902_151345.json | implementation | V16完成态永久快照 |
| 2026-09-02 15:13 | /research-refine | refine-logs/FINAL_PROPOSAL.md | implementation | V16最终方法方案latest |
| 2026-09-02 15:13 | /research-refine | refine-logs/REVIEW_SUMMARY.md | implementation | V16审查摘要latest |
| 2026-09-02 15:13 | /research-refine | refine-logs/REFINEMENT_REPORT.md | implementation | V16 refinement报告latest |
| 2026-09-02 15:13 | /research-refine | refine-logs/score-history.md | implementation | V16评分历史latest |
| 2026-09-02 15:13 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V16 9.10/READY完成态 |
| 2026-09-02 15:21 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_152139.md | implementation | V16 TDD→M0→matched Q1→conditional D1冻结计划 |
| 2026-09-02 15:21 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_152139.md | implementation | V16执行tracker永久版 |
| 2026-09-02 15:21 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V16冻结执行计划latest |
| 2026-09-02 15:21 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V16执行tracker latest |
| 2026-09-02 15:27 | /tdd | docs/TDD_SEAMS.md | documentation | 登记既有接缝同意覆盖的V16 hard-pair/SATR/builder/Q1公共接缝 |
| 2026-09-02 16:05 | /tdd | modeling/trifusion/signal_preserving_v16.py | implementation | V16 Signal-hard pair、two-peer SATR、protect与固定ReID criterion |
| 2026-09-02 16:05 | /tdd | modeling/trifusion/signal_preserving_v16_builder.py | implementation | 复用V8三专家、零新增推理模块的V16 builder |
| 2026-09-02 16:05 | /tdd | tests/test_trifusion_signal_preserving_v16.py | implementation | V16 pair、detach/gradient与criterion公共接缝 |
| 2026-09-02 16:05 | /tdd | tests/test_train_signal_preserving_v16.py | implementation | V16 M0/Q1/D1 fail-closed gates |
| 2026-09-02 16:05 | /tdd | tests/test_trifusion_signal_preserving_v8.py | implementation | 追加V16复用V8拓扑且零推理协作builder测试 |
| 2026-09-02 16:05 | /run-experiment | configs/RGBNT201/TriFusion-signal-preserving-v16-satr-rtx3090.yml | implementation | seed42 B64/K8、V12/V8 SHA与SATR常数冻结合同 |
| 2026-09-02 16:05 | /run-experiment | tools/train_signal_preserving_v16.py | implementation | 配对M0、三折Q1与条件D1独立runner |
| 2026-09-02 16:30 | /run-experiment | evidence/trifusion_v16_satr_m0_seed42_20260902.json | experiment | V16 M0工程门通过但fixed-initial activity失败，dev0/official0 |
| 2026-09-02 16:30 | /analyze-results | results/TRIFUSION_RGBNT201_V16_SATR_M0_2026-09-02.md | analysis | V16 M0覆盖差异、资源结果与Q1封锁结论 |
| 2026-09-02 16:30 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_163000.md | implementation | V16 M0失败终态tracker永久快照 |
| 2026-09-02 16:30 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V16 M0失败终态latest tracker |
| 2026-09-02 16:30 | /handoff | docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md | documentation | 主交接追加V16 SATR M0终态、资源与封锁边界 |
| 2026-09-02 16:30 | /handoff | AGENTS.md | documentation | 工作区规则追加V15/V16终态与禁止重放边界 |
| 2026-09-02 17:11 | /research-refine | refine-logs/v17/round-0-initial-proposal.md | implementation | V17 DTRED问题锚、关系包络蒸馏与no-reranking初始方案 |
| 2026-09-02 17:11 | /research-refine | refine-logs/REFINE_STATE_20260902_171121.json | implementation | V17 proposal阶段永久状态快照 |
| 2026-09-02 17:11 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V17 proposal阶段latest状态 |
| 2026-09-02 17:19 | /research-refine | refine-logs/v17/round-1-review.md | review | V17 GPT-5.5 xhigh首轮7.4/REVISE原始审查 |
| 2026-09-02 17:19 | /research-refine | refine-logs/v17/score-history.md | implementation | V17首轮评分演进 |
| 2026-09-02 17:19 | /research-refine | refine-logs/REFINE_STATE_20260902_171907.json | implementation | V17 round1 review永久状态快照 |
| 2026-09-02 17:19 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V17 round1 review latest状态 |
| 2026-09-02 17:22 | /research-refine | refine-logs/v17/round-1-refinement.md | implementation | V17 one-sided正负平衡envelope完整修订案 |
| 2026-09-02 17:22 | /research-refine | refine-logs/REFINE_STATE_20260902_172200.json | implementation | V17 round1 refinement永久状态快照 |
| 2026-09-02 17:22 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V17 round1 refinement latest状态 |
| 2026-09-02 17:25 | /research-refine | refine-logs/v17/round-2-review.md | review | V17 GPT-5.5 xhigh二轮8.3/REVISE原始审查 |
| 2026-09-02 17:25 | /research-refine | refine-logs/v17/score-history.md | implementation | V17二轮评分演进 |
| 2026-09-02 17:25 | /research-refine | refine-logs/REFINE_STATE_20260902_172507.json | implementation | V17 round2 review永久状态快照 |
| 2026-09-02 17:25 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V17 round2 review latest状态 |
| 2026-09-02 17:29 | /research-refine | refine-logs/v17/round-2-refinement.md | implementation | V17冻结loss scalar、teacher来源回执与Oracle边界完整修订案 |
| 2026-09-02 17:29 | /research-refine | refine-logs/REFINE_STATE_20260902_172900.json | implementation | V17 round2 refinement永久状态快照 |
| 2026-09-02 17:29 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V17 round2 refinement latest状态 |
| 2026-09-02 17:42 | /research-refine | refine-logs/v17/round-3-review.md | review | V17 GPT-5.5 xhigh终审9.03/READY原始审查 |
| 2026-09-02 17:42 | /research-refine | refine-logs/v17/score-history.md | implementation | V17三轮评分终态 |
| 2026-09-02 17:42 | /research-refine | refine-logs/REFINE_STATE_20260902_174258.json | implementation | V17完成态永久状态快照 |
| 2026-09-02 17:42 | /research-refine | refine-logs/REFINE_STATE.json | implementation | V17 9.03/READY完成态 |
| 2026-09-02 17:42 | /research-refine | refine-logs/FINAL_PROPOSAL_20260902_174258.md | implementation | V17 DTRED 9.03/READY最终方法方案永久版 |
| 2026-09-02 17:42 | /research-refine | refine-logs/FINAL_PROPOSAL.md | implementation | V17 DTRED最终方法方案latest |
| 2026-09-02 17:42 | /research-refine | refine-logs/REVIEW_SUMMARY_20260902_174258.md | implementation | V17三轮审查摘要永久版 |
| 2026-09-02 17:42 | /research-refine | refine-logs/REVIEW_SUMMARY.md | implementation | V17审查摘要latest |
| 2026-09-02 17:42 | /research-refine | refine-logs/REFINEMENT_REPORT_20260902_174258.md | implementation | V17方法演进与风险永久版 |
| 2026-09-02 17:42 | /research-refine | refine-logs/REFINEMENT_REPORT.md | implementation | V17 refinement报告latest |
| 2026-09-02 17:42 | /research-refine | refine-logs/score-history_20260902_174258.md | implementation | V17评分历史永久版 |
| 2026-09-02 17:42 | /research-refine | refine-logs/score-history.md | implementation | V17评分历史latest |
| 2026-09-02 17:47 | /experiment-plan | refine-logs/EXPERIMENT_PLAN_20260902_174747.md | implementation | V17 T0→M0→Q1→条件D1冻结执行计划 |
| 2026-09-02 17:47 | /experiment-plan | refine-logs/EXPERIMENT_PLAN.md | implementation | V17冻结执行计划latest |
| 2026-09-02 17:47 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER_20260902_174747.md | implementation | V17执行tracker永久版 |
| 2026-09-02 17:47 | /experiment-plan | refine-logs/EXPERIMENT_TRACKER.md | implementation | V17执行tracker latest |


## 2026-09-05 V17 terminal and full-gallery continuation

- Source execution: V17 M0/Q1 `535ef2f`; supplementary evaluator `5ca978c`, numeric JSON fix `0888f45`.
- Raw receipts: `evidence/trifusion_v17_dtred_m0_seed42_535ef2f.json`, `evidence/trifusion_v17_dtred_q1_seed42_535ef2f.json`.
- Complete-gallery receipt: `evidence/trifusion_v17_full_gallery_fixed_20260905.json`; all six final endpoints, 3126 gallery rows, 571 eligible queries, optimizer0/dev0/official0.
- Remote verification: `evidence/trifusion_v17_terminal_verification_20260905.json` (13 tests and raw receipt/log SHAs).
- Result and independent audit: `results/TRIFUSION_RGBNT201_V17_DTRED_2026-09-05.md`, `EXPERIMENT_AUDIT_V17.md/json`.
- Updated main handoff through sections28/29 and `docs/SOTA_REFRESH_2026-09-05.md`.
- V17 is terminal negative, without D1 or a new deployable/official result. Large weights remain remote-only.

| 2026-09-05 12:12 | /experiment-plan | refine-logs/v19/EXPERIMENT_PLAN_20260905_121232.md | implementation | V19 private semantic tail preregistration; no results yet |
| 2026-09-05 12:12 | /experiment-plan | refine-logs/v19/EXPERIMENT_PLAN.md | implementation | V19 private semantic tail preregistration; no results yet |
| 2026-09-05 12:12 | /experiment-plan | refine-logs/v19/EXPERIMENT_TRACKER_20260905_121232.md | implementation | V19 private semantic tail preregistration; no results yet |
| 2026-09-05 12:12 | /experiment-plan | refine-logs/v19/EXPERIMENT_TRACKER.md | implementation | V19 private semantic tail preregistration; no results yet |

| 2026-09-05 12:29 | /run-experiment | refine-logs/v19/EXPERIMENT_TRACKER_20260905_122939.md | evidence | T0/M0 complete; Q1 running, no retrieval claim |
| 2026-09-05 12:29 | /run-experiment | evidence/trifusion_v19_m0_seed42_4b749cd.json | evidence | Immutable M0 snapshot; remote SHA verified |

| 2026-09-05 12:46 | /experiment-audit | EXPERIMENT_AUDIT_V19_M0.md/json | evidence | Independent M0 engineering PASS; integrity WARN; Q1 pending |
| 2026-09-05 12:46 | /run-experiment | refine-logs/v19/EXPERIMENT_TRACKER_20260905_124646.md | evidence | M0 audit complete; Q1 unchanged and running |

| 2026-09-05 12:48 | /run-experiment | refine-logs/v19/EXPERIMENT_TRACKER_20260905_124940.md | evidence | First complete paired fold, 2 of 6 endpoints; overall Q1 still RUNNING |

| 2026-09-05 13:14 | /run-experiment | refine-logs/v19/EXPERIMENT_TRACKER_20260905_131521.md | evidence | Two complete paired folds; 4/6 endpoints, overall Q1 still RUNNING |

| 2026-09-05 13:55 | /run-experiment | refine-logs/v19/EXPERIMENT_TRACKER_20260905_135536.md | evidence | V19 full six-endpoint Q1_FAIL; independent terminal audit pending |
| 2026-09-05 13:55 | /run-experiment | results/TRIFUSION_RGBNT201_V19_PRIVATE_TAIL_2026-09-05.md | evidence | Complete five-output, three-fold and 21-identity terminal results |
| 2026-09-05 13:55 | /run-experiment | evidence/trifusion_v19_q1_seed42_4b749cd.json | evidence | Complete source-bound Q1 summary; all six receipts and terminal verification accompany it |

| 2026-09-05 14:25 | /experiment-audit | EXPERIMENT_AUDIT_V19_Q1.md/json | evidence | Independent complete Q1 engineering PASS, integrity WARN, scientific FAIL; actual bootstrap replay |
| 2026-09-05 14:25 | /run-experiment | refine-logs/v19/EXPERIMENT_TRACKER_20260905_142550.md | evidence | V19 terminal audit complete; all-six-model read-only geometry diagnosis complete |
| 2026-09-05 14:25 | /run-experiment | results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md | evidence | All source/heldout records and all nine modality pair geometries; independent diagnosis audit pending |

| 2026-09-05 14:46 | /experiment-audit | EXPERIMENT_AUDIT_V19_GEOMETRY.md/json | evidence | Complete independent arithmetic replay; engineering PASS, integrity WARN, V19 Q1_FAIL unchanged |
| 2026-09-05 14:46 | /experiment-plan | refine-logs/v20/EXPERIMENT_PLAN_20260905_144458.md | implementation | Frozen per-expert cross-modal identity objective and matched full main comparison; not launched |
| 2026-09-05 14:46 | /experiment-plan | refine-logs/v20/EXPERIMENT_PLAN.md | implementation | Current frozen V20 plan, byte-identical timestamped copy |
| 2026-09-05 14:46 | /experiment-plan | refine-logs/v20/EXPERIMENT_TRACKER_20260905_144458.md | implementation | T0/M0/Q1 pending; D1 locked |
| 2026-09-05 14:46 | /experiment-plan | evidence/trifusion_v20_preregistration_20260905.json | implementation | Config/plan/runner/module hashes and AST receipt; no model execution |

| 2026-09-05 14:55 | /run-experiment | evidence/trifusion_v20_m0_seed42_3cea5bf.json | evidence | Immutable M0_PASS snapshot; Q1 running, no retrieval claim |
| 2026-09-05 14:55 | /run-experiment | refine-logs/v20/EXPERIMENT_TRACKER_20260905_145411.md | evidence | T0 three CUDA tests and M0 passed; independent M0 audit in progress |

| 2026-09-05 15:23 | /experiment-audit | EXPERIMENT_AUDIT_V20_M0.md/json | evidence | Independent M0 engineering PASS, integrity WARN, scientific not evaluated; two verbatim rounds retained |
| 2026-09-05 15:23 | /run-experiment | refine-logs/v20/EXPERIMENT_TRACKER_20260905_152333.md | evidence | First complete paired fold, 2/6 endpoints; negative fold retained; Q1 continues |
| 2026-09-05 15:23 | /run-experiment | evidence/trifusion_v20_first_paired_fold_20260905.json | evidence | All first-fold five-output final arrays and receipts; remote SHA verified |

| 2026-09-05 16:12 | /run-experiment | refine-logs/v20/EXPERIMENT_TRACKER_20260905_161219.md | evidence | V20 all six endpoints complete, Q1_FAIL; independent terminal audit pending |
| 2026-09-05 16:12 | /run-experiment | results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md | evidence | All three folds, five outputs, 21 identities and paired query changes |
| 2026-09-05 16:12 | /run-experiment | evidence/trifusion_v20_q1_seed42_3cea5bf.json | evidence | Terminal raw arrays; six receipts, complete log, remote 32-file SHA and actual NumPy bootstrap audit |

| 2026-09-05 16:38 | /experiment-audit | EXPERIMENT_AUDIT_V20_Q1.md/json | evidence | Independent engineering PASS, integrity WARN, scientific FAIL; full bootstrap replay |
| 2026-09-05 16:38 | /run-experiment | refine-logs/v20/EXPERIMENT_TRACKER_20260905_163841.md | evidence | V20 Q1 audited and sealed; warnings preserved |
| 2026-09-05 16:38 | /experiment-plan | refine-logs/v21/EXPERIMENT_PLAN_20260905_163644.md | implementation | Frozen SAM20/AdamW40 comparison; equal forward-backward counts; not launched |
| 2026-09-05 16:38 | /experiment-plan | refine-logs/v21/EXPERIMENT_PLAN.md | implementation | Latest frozen V21 plan, identical timestamped copy |
| 2026-09-05 16:38 | /experiment-plan | refine-logs/v21/EXPERIMENT_TRACKER_20260905_163644.md | implementation | T0/M0/Q1 pending; D1 locked |

| 2026-09-05 16:48 | /run-experiment | refine-logs/v21/EXPERIMENT_TRACKER_20260905_164838.md | evidence | T0 all three passed; original PID32331 running M0; launch transport timeout isolated |

| 2026-09-05 16:55 | /run-experiment | refine-logs/v21/EXPERIMENT_TRACKER_20260905_165528.md | evidence | V21 M0_FAIL fixed overfit gate, Q1 not run; independent M0 audit pending |
| 2026-09-05 16:55 | /run-experiment | results/TRIFUSION_RGBNT201_V21_SAM_M0_2026-09-05.md | evidence | Complete M0 pairing, capacity and 100-step trajectory; no retrieval result |

| 2026-09-05 17:24 | /experiment-audit | EXPERIMENT_AUDIT_V21_M0.md/json | evidence | Independent engineering PASS, integrity WARN, fixed M0/scientific qualification FAIL; no retrieval |
| 2026-09-05 17:24 | /run-experiment | refine-logs/v21/EXPERIMENT_TRACKER_20260905_172433.md | evidence | V21 audit closed; exact M0 negative retained, Q1 not run |

| 2026-09-05 17:45 | /run-experiment | evidence/trifusion_source_camera_metadata_20260905.json | evidence | Source-only 1680-batch camera-label replay; no model execution |
| 2026-09-05 17:45 | /experiment-plan | refine-logs/v22/EXPERIMENT_PLAN_20260905_174555.md | implementation | Frozen V22 residual MCNL main comparison; not launched |
| 2026-09-05 17:45 | /experiment-plan | refine-logs/v22/EXPERIMENT_TRACKER_20260905_174555.md | implementation | T0/M0/Q1 pending, dev/official zero |

| 2026-09-05 17:53 | /run-experiment | refine-logs/v22/EXPERIMENT_TRACKER_20260905_175301.md | evidence | V22 T0 all 3 passed; original M0 PID34656 running; no retrieval result |

| 2026-09-05 18:01 | /run-experiment | refine-logs/v22/EXPERIMENT_TRACKER_20260905_180137.md | evidence | V22 complete M0_PASS; original complete Q1 running |
| 2026-09-05 18:01 | /run-experiment | results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_M0_2026-09-05.md | evidence | All M0 pairing, capacity and 100-step components; no Q1 retrieval result |

| 2026-09-05 18:27 | /experiment-audit | EXPERIMENT_AUDIT_V22_M0.md/json | evidence | Independent engineering/fixed M0 PASS; integrity WARN; bounded scientific qualification FAIL |
| 2026-09-05 18:27 | /run-experiment | refine-logs/v22/EXPERIMENT_TRACKER_20260905_182750.md | evidence | First paired fold negative; original complete Q1 continues |
