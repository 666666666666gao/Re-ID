# CPU-save 边界诊断终态救援复核

**结论：FAIL / STOP_CPU_SAVE_ROUTE_UNLOCALIZED。封存 v4 与本次诊断的真实失败，执行既定停止条件；当前没有直接因果证据支持 CPU-save 修补。此次诊断不放行 M0、50 epoch 或任何新 GPU 运行。唯一下一步建议是利用已取得的四卡容量事实，做一份保持完整作者 batch 的模型分段源码可行性说明。**

复核时间：2026-10-03 17:15 +08:00。fresh context、same-family、provisional。父代理确认实际 native spawn 请求为 model=gpt-6-astra、reasoning_effort=max、fork_turns=none；本子代理未独立验证底层实际 backend/model/effort 身份，不作跨家族接受声明。本次仅本地文件读取、完整 JSON 解析与字段复算；未连接服务器、导入模型、执行 CUDA、启动诊断/训练、修改生产代码或删除权重。

证据根目录为 ../cpu_boundary_terminal780，下文源码路径相对其 _source。完整读取并解析 1,826,450 字节的 primary/logs/cpu_saved_boundary_diagnostic_20261003_v1/observations/MEASURED.json，独立重算全部 281 梯度、1323 pack 与 1323 unpack；同时核对 INTAKE.json、JOB.json、完整 diagnostic.log 和指定四份源码。6 份 primary 文件均符合 intake 已有长度与 SHA 记录；没有创建新封存机制。历史边界读取自 ../rescue_cpu_saved778/RESCUE_REVIEW.md 与 .json，另核对 sealed v4、原对原控制和 V1 B128 OOM 的实际 primary。

**确定事实**

1. JOB.json 记录 2026 物理 GPU0，17:03:31 启动、17:04:31 完成，status=FAILED、child exit_code=1。diagnostic.log 的失败点是 diagnose_cpu_saved_boundaries.py:195 的既定参数梯度断言。MEASURED_BEFORE_FIXED_GATE 是断言前观测状态，不是 PASS；没有 COMPLETE.json。原 RUNNING.json 是启动记录，不能据此说任务仍在运行。
2. 实际范围是 semantic/RGBNT201，作者 B64/K8 首批的 first32、4 个身份，原实现一次与 CPU-save 一次，共两次 forward/backward、零 optimizer update、零权重。源码 :83–123 保持同一已取出的 batch、相同初始化、seed42、两臂复原同一 CPU/CUDA RNG、AMP FP16 和 GradScaler256。输入 labels/cameras/paths/images 摘要与保存的原对原控制相同；历史原始跨进程 RNG 相等仍未被证明。
3. raw、fused、shared_global、loss、author head score/feature，以及两个 backbone 返回边界的 forward，记录的最大绝对差均为 0。全部梯度为有限值、无 None 模式不一致；281 参数中 81 个固定 allclose(atol=rtol=1e-4) 失败，229 个存在非零差，整体最大差 0.00390625。148 个参数虽过门仍有非零差；不能将通过门槛写成逐位相等，也不能用单独绝对误差是否大于 1e-4 代替已有逐元素 allclose 判据。

| 参数组 | 参数数 | 固定门失败 | 非零差参数数 | 最大绝对差 |
|---|---:|---:|---:|---:|
| Signal | 155 | 75 | 150 | 0.00390625 |
| adapters | 54 | 6 | 14 | 0.000244140625 |
| roles | 70 | 0 | 65 | 2.384185791015625e-7 |
| readout | 2 | 0 | 0 | 0 |

4. stages 的 incoming_unscaled_gradient 最大差为 9.313225746154785e-10，非零，但固定 1e-4 门通过；两臂该梯度最大幅值均为 1.171603798866272e-6。shared_global 的 incoming 梯度最大差为 0。参数“失败集中在 backbone”不代表最早差异位于 backbone 内，两个反向入口不能称为完全相同。
5. 1323 个 pack/unpack 的 index 一一对应；完整逐项复算 before→CPU payload 与 before→restored 的 shape、stride、storage_offset、dtype、layout、contiguous，均只有 72 处 storage_offset 变化；restored device 也与原 device 相同。例如 index1265：offset3170304→0，shape[32,12,129,64]、stride[768,64,24576,1]、float16、torch.strided、非连续性均保持。CPU payload 的 device 变化及 requires_grad/grad_fn 的脱离属于保存载荷的观测，不是“梯度断开”证据；真实参数梯度已计算完成。
6. 6 项 post_checks 均实测 true：buffers、CPU RNG、CUDA RNG、state 一致，author BN 每臂一批，原 capture hooks 均移除。与 sealed v4 不同，这次在最终梯度断言之前确已计算并写出全部 post_checks（源码 :164–197）；不能把 v4 未执行的后置检查追认成历史通过。原两臂最终 CPU RNG 相同，但与初始 CPU RNG 不同，这与“两臂一致”不矛盾。
7. 记录的峰值 allocated 为原实现 7,761,206,272 B、CPU-save 2,072,029,696 B；reserved 分别为 9,124,708,352 B 与 3,099,590,656 B。这仅是此次有 instrumentation 的 first32 观测，不能推出作者 B128 全训练容量或九端 M0 已通过。sealed v4 原有 94/281 梯度失败、最大差 0.0078125 仍保留；本次 81/281 不构成“修复改善”，两次都有真实固定门失败。

**阻断项与因果判读**

- B1：本 CPU-save 实现尚未满足原固定参数梯度合同，不能原样重跑求通过，也不能进入 M0。历史原对原 281 参数控制确实 PASS（最大差 2.9103830456733704e-11），它既不是逐位恒等证明，也不覆盖本次失败。
- B2：根因未定位。cpu_saved_evidence_clip.py:7–16 只包原 forward；诊断 :28–47 委托实际 torch2.5.1+cu121 内置 pack/unpack 各一次，:94–99 替换包装而未嵌套，:104–143 对称保留边界并在 backward 后复制。实际内置源码已随 MEASURED 保存。不存在观测到的 stride/layout 变化，stages 反向入口也非零分歧，未满足此前“布局变化且边界一致”的优先归因条件。
- storage_offset 变化证明保存载荷的存储偏移表示发生变化，不能独自证明数值内容损坏、kernel 改选、归约顺序变化或任何一个参数的误差来源。相同 shape/stride/layout 也不能证明所有后端执行细节相同。本次没有干预隔离或算子执行轨迹，故不能指定 cv_embed、attention、Mamba 或 CPU pack 为唯一根因。
- B3：first32 零更新诊断没有解决 full-author-batch 容量。V1 的 RGBNT100 global_only B128 在 correspondence_roles.py:86 堆叠 captured tensors 时真实 OOM：请求 218 MiB，仅剩 78.56 MiB，进程占用 23.42 GiB。四卡低占用不会增加单卡容量；没有已验证的新实现供九端八更新 M0 使用。

**落实原定停止条件**

前次报告原文：“若布局相同或 instrumentation 不再复现，结果记为未定位并结束本次诊断，不自动追加另一内存方案或连续重跑。”本次失败复现，但 stride/layout 相同，且边界梯度已不同，足以执行停止条件。CPU-save 路线当前状态应记为 FAILED_UNLOCALIZED / SEALED，诊断运行记为 TERMINAL_FAILED，保留原始 MEASURED、JOB、log、源码与 v4 失败；不覆盖为 PASS，也不删除或重写旧状态文件来制造成功。

不批准追加诊断臂、节点 profiler、contiguous、empty_strided/custom offload、选择性搬运或另一轮无因果依据的局部修补；不改门槛、FP16/scaler、seed、作者 batch/K；不重跑未改版本。上述限制关闭当前失败救援循环，不意味着放弃原科学目标。

**仅建议的一项下一步：本地源码层面的模型分段可行性说明**

父代理新增的 ../capacity780/stdout.json 是 17:11:58 的只读资源快照：2026 四张 RTX3090，各24576 MiB，已用15/15/15/68 MiB、利用率0%；0–1与2–3为NODE，跨组SYS，无NVLink；RAM总131577172 kB、可用121397300 kB；/data可用17,769,156,608 B，/home可用591,721,623,552 B。这是时间点资源事实，并非卡位预约或后续运行许可；当前授权范围内没有更大单卡。

有证据支持调查的实现方向是单进程、按模型层段分布到既有2026卡上，完整 B64/B128 batch 通过各段，保持三个模态顺序和共享参数身份；把捕获张量汇集与 roles/readout/author heads 的驻留位置明确列出。源码已有12层顺序 CLIP（clip/model.py:462–482）、三个捕获层（3/7/11）及实际 OOM 的 stack 边界，提供了具体设计入口；这不是已证明能容纳、数值等价或有足够吞吐的实现。

这份短说明只需回答：每个模块/共享参数唯一驻留在哪张卡、跨段张量及 captured tensors 如何保留 autograd、完整 batch 的 BN 与 Triplet 在哪里一次计算，以及逐卡参数/激活/梯度/优化器状态的容量预算。author heads 源码 :49–60 对完整 raw 特征调用 neck/classifier；make_loss.py:37–56 对输入 feat/target 算 Triplet。因此普通 DDP batch 分片或梯度累积不能被直接写成原作者全 batch BN/Triplet 等价，四张24 GiB也不能按一张96 GiB使用。

本报告只建议形成这份本地设计以判断是否存在可保持合同的路径，不为“前6/后6 block 放两卡”等具体分点提供容量或等价背书；该分点仍须从捕获张量驻留、逐卡峰值预算和原计算顺序论证。不要求再次调查同一容量快照，不实施模型分段、不启动任何新 GPU 任务。若后续有具体实现，仍须另经源码审查与原固定数值门验证，之后全九端真实8-update M0先于任何50-epoch；没有任何放宽或诊断自动晋级。

**非阻断事实、合同与保留**

本诊断把负结果和原来缺失的后置观测完整写出，终态与失败日志一致；未发现需要为本次封存另补一次运行的证据。小于门槛的差异、较低显存和全部 post_checks 通过是有效观测，均不足以消除 B1–B3。

研究合同维持九端 global_only/semantic/native × RGBNT201/RGBNT100/MSVR310；201 B64/K8、100 B128/K16、MSVR B64/K4；AMP FP16、既定 scaler256 witness、forward atol=rtol=1e-5、parameter-gradient atol=rtol=1e-4；公共 CLIP、fresh camera/heads、作者 optimizer/loss 与 seed42 不变。仅2026物理GPU0–3、max4、不抢占；2025仅文本。全部九端真实八更新 M0 完成之前不得开始任何正式50 epoch。

本次诊断没有生成权重，sealed v4 intake 也确认输出权重目录为空，故当前无诊断权重可清。本复核没有删除文件。后续已完成且无依赖的工程权重及时退役；每个完成的正式端只留一个 mAP-best，所有指标来自同一 checkpoint，公共预训练、活动依赖和其他项目权重不动。所有实际失败文本与 JSON 保留。

审查输出属于同家族 fresh-context 临时意见；不是新的数值 PASS、B128 容量证明、M0 完成、正式成绩或新训练授权。
