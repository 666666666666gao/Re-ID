# Context/local identity 空卡启动依赖复核

日期：2026-09-29。最终结论：**PASS（source-only / same-family / provisional）**。本轮发现的前序 GPU 阶段切换占用问题已作最小修复并重新读取确认。**真实生产 M0 尚未运行；本报告不是 GPU 状态实查、生产 M0、训练或检索结果。**

- review_scope: source-only；仅复核本次依赖调整及必要的 GPU 保留门
- review_independence: same-family（独立上下文，非跨模型家族审查）
- acceptance_status: provisional
- requested_model: gpt-6-astra
- requested_reasoning_effort: max
- actual_serving_model: 运行环境未暴露，不能据请求值声称已核验实际服务模型

## 本轮修订的执行约束

按本轮明确任务，启动门槛改为：原 M3 12 端全部已启动、无 FAILED、没有 PENDING，且 RGBNT201/MSVR310 各四端已经完整 CPU 验收；允许新任务使用释放的 GPU，原 RGBNT100 任务继续，不能抢占其 GPU。执行者提供的 13:15 观察指出最后一个原 RGBNT100 matched/predictor 仍在 GPU3；本审查者未通过 SSH 或 GPU 工具独立核实该实况。

启动时保存的是 `parent_matrix_at_launch.json`，不要求其中未完的 RGBNT100 端具有成绩；新 15 端全部完成后，仍必须原 M3 全 12 端完整 CPU 验收，才能保存最终 `parent_matrix.json` 并进入全局 COMPLETE。

这项启动顺序变更取代 13:08 版报告中的“先原 M3 全12验收、再启动任何新GPU任务”约束；它不改变模型、损失、原 M3 配置、15 条件矩阵或终态验收。`REVIEW_EXECUTION_20260929_1308.md` 保留为当时约束下的历史报告。

## BLOCKING：阶段切换占用问题已关闭

初审发现，复用 scheduler 的 `occupied` 只含新 context 队列自己的 active worker。原 M3 worker 用分开的子进程执行 train 和 evaluate；训练进程退出到独立 best 重载/评价完成模型加载之间，显存会释放，但原 worker 尚未完成。仅凭 `memory.used < 500` 判空可能在这一实际阶段切换窗口把 GPU3 分给新任务，干扰原评价。该问题来自本次允许两队列重叠的既定路径，不是任意进程或毫秒级竞态的假设性扩展。

已复核 `tools/queue_correspondence_context_identity.py` 第 58–72 行：新 worker 在设置 CUDA 环境及首次 GPU `Popen` 前，读取 manifest 绑定的原 campaign；要求其状态为 RUNNING/COMPLETE 且无 FAILED，并把同 GPU 上任何前序 RUNNING job 作为保留占用。若仍保留，则写 `WAIT_PREDECESSOR_GPU`，按原 240 秒间隔等待；等待期间没有 GPU 子进程启动。直到前序控制器不再将该 GPU 的旧 job 标记为 RUNNING，才进入原 M0→train→evaluate 流程。

原 scheduler 只有在前序 worker 进程结束并通过 `require_complete` 后才把该 job 写为 COMPLETE。因此 GPU3 的 train/evaluate 间隙不会绕过新保留门。新 scheduler 即使在显存间隙分配了一个等待 worker，它也不占用 GPU；该等待 worker 又被新 scheduler 视为 active，避免同队列继续给该卡分配其它 worker。**该阻断项关闭；没有发现本次改动的其它具体阻断项。**

## 已确认的其余连线

- **不延误原 pending。** coordinator 在启动新 `run_phase` 前要求原 campaign 无 FAILED、无 PENDING；原状态 RUNNING 时继续核对指定协调器 PID 的真实命令。原 M3 的固定 12 项由原 collector 按原 manifest/条件表核对，没有新建或改排原任务。
- **201/MSVR310 先完整验收。** 原 collector 的固定矩阵保证每个数据集有四个条件；新启动断言要求其中 RGBNT201/MSVR310 的全部记录为 VERIFIED_COMPLETE。这沿用原 full50、best、独立评价、完整图库元数据与 CPU 指标复算，而不是相信中间训练分数。
- **未完端不当成绩。** 启动快照直接保存原 collector 的 12 行结果。未完成的 RGBNT100 子 campaign 保留 UNACCEPTED/PENDING 表示，不从临时 best 或未完 epoch 抽取正式成绩，也不计为 VERIFIED_COMPLETE。
- **终态仍双重验收。** 新 `run_phase` 成功后，先重新 `collect_m3` 并要求 12/12，再保存最终 `parent_matrix.json`；随后运行新 collector，要求 15/15，执行同数据集五条件初始 state 一致性核查并保存 `accepted_matrix.json`，最后才写全局 COMPLETE。
- **原调度和每端链保持。** 原 `queue_correspondence_refinement.py` 未修改，仍为四卡、240 秒、显存判空、失败后不再启动 pending、无 retry。新 worker 的 GPU 保留门之后仍执行真实生产入口的 M0→full50→best 独立重载→三路完整图库 CPU 复算→loss 记账；此前已关闭的作者 evaluator 导入顺序和共同初始化验收问题没有被本次改动恢复。

## 边界

本轮只读新 queue 的 coordinator/worker 变动及原 scheduler/M3 worker 的相关源码，没有 SSH、GPU 查询、模型前向或训练，没有改模型、runner、旧队列或旧科学源，也没有运行新的 CPU 测试。仅写入本报告及固定名副本，保留 13:08 时间戳报告。

保留门是针对已经确认的原 M3 train/evaluate 子进程切换路径的最小约束，不是 fallback、retry、抢占或通用进程防御层。真实生产 M0 仍待执行；synthetic token encoder + linear Mamba 的 CPU 检查不能替代真实 M0，不能提供正式成绩。

最终裁决：**本次依赖调整及 GPU 保留门来源复核 PASS；same-family / provisional；生产 M0 尚未运行。**
