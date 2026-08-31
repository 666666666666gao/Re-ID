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
