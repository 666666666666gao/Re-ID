# Context/local identity 执行链来源复核

日期：2026-09-29。最终结论：**PASS（source-only / same-family / provisional）**。本轮发现的共同初始化验收连线缺口已修复并重新读取确认。当前结论仅说明所审查源码的执行链符合登记顺序；**生产 M0 尚未执行，本报告不是生产 M0、训练、检索成功或实际部署完成的证据。**

- review_scope: source-only
- review_independence: same-family（独立上下文，非跨模型家族审查）
- acceptance_status: provisional
- requested_model: gpt-6-astra
- requested_reasoning_effort: max
- actual_serving_model: 运行环境未暴露，不能据请求值声称已核验实际服务模型

## 范围

本轮读取三个新文件，并沿其 imports 和调用追踪原四卡 scheduler、M3 collector、训练/评价入口及固定计划：

- `tools/queue_correspondence_context_identity.py`
- `tools/collect_correspondence_context_identity.py`
- `tools/audit_correspondence_context_identity_losses.py`
- 复用的 `queue_correspondence_refinement.py`、`queue_correspondence_roles.py`、`queue_correspondence_role_prediction.py`、`collect_correspondence_role_prediction.py` 和 `collect_correspondence_roles.py`
- 已经复核的新模型/runner 与 `EXPERIMENT_PLAN.md`

没有 SSH、GPU 调用、模型前向或训练，没有修改实现代码。曾尝试只读的本地 Python import/AST/命令绑定检查，但解释器在启动阶段返回 `No pyvenv.cfg file`，检查没有运行；本报告不将该尝试记为 PASS。结论基于直接阅读源码及调用关系。仅写入本报告和固定名副本。

## BLOCKING：已关闭

初审发现，共同初始 state 的跨条件断言只存在于 collector 的 `collect()`；worker 仅执行单端 `verify()`，协调器原先在 `run_phase()` 返回 0 后直接写 `COMPLETE`。因此正常队列完成路径原先可能没有执行计划要求的跨条件初始化核查。该发现不表示已经出现权重不一致，而是一个实际的验收连线缺口。

已复核当前 queue 第 132–138 行：成功分支先调用 `collect(args.campaign)`，要求 `expected_endpoints == verified_complete == len(rows) == 15`，写入 `accepted_matrix.json`，随后才写 `COMPLETE`。collector 第 132–134 行对同数据集已验收端的 `initial_model_state_sha256` 执行一致性断言；15/15 要求与固定条件列表一起保证五个条件全部进入核查。现有来源/权重断言保留，未增加新的 hashing 方案。**该项关闭；未发现其它具体实现阻断项。**

## 已确认的源码执行链

1. **前序 M3 门槛。** 协调器核对指定前序 manifest；等待前序 `COMPLETE`，等待期间核对原 M3 协调器实际命令并每 240 秒检查。随后调用原 M3 collector，必须全部 12 端为 `VERIFIED_COMPLETE`，保存 `parent_matrix.json` 后才进入新 GPU 调度。原 M3 collector核对每端 M0/50轮/best/独立评价回执，并对完整图库距离使用官方身份及 camera/scene 元数据 CPU 复算。没有用部分端或中间 best 放行。

2. **固定 15 端与原 scheduler。** 五个登记条件各覆盖 RGBNT201、RGBNT100、MSVR310。新 `start_command` 绑定到原 `run_phase` 的模块作用域，仍使用原 240 秒、GPU 0–3、排除 active 卡及显存占用条件的调度；没有 kill、抢占、retry 或按成绩筛选配置的分支。

3. **每端真实生产入口。** worker 为 GPU 设置 `CUDA_VISIBLE_DEVICES`，依次以独立进程运行 `m0`、`train`、`evaluate`，逐项要求 exit 0 和对应完成状态。命令固定 seed42/full50/width128/M1+M2/关闭M3，以及正确的 query/auxiliary 条件。该入口沿原 baseline builder 和 `production_mamba_factory` 构造模型，没有引用 synthetic token encoder、linear Mamba 或 CPU 玩具检查结果来放行真实 M0。

4. **M0 与正式训练隔离。** M0 使用独立 `_m0` 目录；正式训练与评价使用同一 `_full` 目录。runner 的生产 M0 仍执行八批、累计有限非零梯度、冻结 baseline 状态比较和紧凑权重独立重载检查。正式训练重新初始化，不接续 M0 权重；collector 要求 M0 和训练 initializer 完全一致。

5. **best 与三路完整图库。** 每轮官方 fused mAP 按既有规则保存 best，平分时选择较晚轮；collector 用相同规则重算所选轮。独立评价加载同一 best，提取 fused/shared_global/joint_local 三路完整 query/gallery。collector核对 checkpoint、协议、condition、来源、baseline 和同一所选轮指标；检查距离形状、有限性及完整身份/camera/scene 元数据，并逐路 CPU 复算。上轮修复的作者 evaluator 导入顺序及精确来源断言继续保留。

6. **完整 loss 记账。** 单端 `verify()` 在 worker 写 `COMPLETE` 前调用 loss audit。audit 要求完整 1–50 轮、每轮连续 batch 编号和总步数一致，检查每步有限值，并重构 `loss = id + triplet + auxiliary_id`；auxiliary 权重固定为 1，none 条件辅助项必须为 0。逐轮均值与训练回执核对，first/best/last 来自同一训练历史，不跨轮拼列。标量支持度不被解释为梯度份额或因果贡献。

7. **失败与终态。** 子进程失败不继续该端后续模式；原 scheduler 检出失败后停止新增 pending，等待现有 active 结束，没有自动重试。单端 CPU/日志验收断言失败也使 worker 非零退出，不会写单端 `COMPLETE`。全部 worker 成功后，新增终态 collector 门槛再次要求完整 15/15，并核对同数据集跨条件初始 state；之后才写全局 `COMPLETE`。

## NON-BLOCKING 与边界

- 未提出额外 fallback、try/except、兼容层、假设性边界分支、无关重构或新增 provenance 方案。
- 新 queue/collector/loss audit 的源码已经具备上述路径；其生产执行尚未发生，真实 M0 的数值、梯度、真实 Mamba、AMP 和重载结果仍需实际回执。
- synthetic token encoder + linear Mamba 的 CPU 结构/初始化测试只能作为工程准备，不能代替任何一端生产 M0，不能产生成绩或证明方法有效。
- 原 M3 12 端仍按原配置完成；没有修改旧科学源码、旧 manifest、训练定义或已选权重。本次来源复核没有读取远端状态，不声称前序任务已经全部完成。

最终裁决：**执行链源码复核 PASS；same-family、provisional；生产 M0 待执行。**
