# RESCUE_REVIEW

审查时间：2026-10-03。范围：**source-only / same-family / provisional**。fresh native Codex 子代理 `/root/review_memory_after_original_repeat776`；按 experiment-bridge 的 RESCUE_ON_FAILURE 执行。审查工作树 HEAD 为 `27a1d4bd7423a52bc658fb03ac48b92272879e8a`。本次只读源码、原始文本及官方资料；没有 SSH、模型/GPU 执行或科学源码修改。**当前没有新方案运行结果。**

## verdict

**选择一个最小候选：只在原 CrossLayerAdaptedCLIP 的完整 forward 调用外使用 `torch.autograd.graph.save_on_cpu(pin_memory=False)`，保存反向需要的张量到 CPU。可以进入新命名 helper/入口的实现与 fresh 源码审查；不构成部署、数值等价、容量或训练通过。**

**不能明确解释 V3 的具体梯度差异原因。** 原实现重复控制通过不改变 V2/V3 的固定门失败，不支持放宽任何门槛。停止继续盲改 checkpoint。

## blocking_issues

1. V3 semantic 的 92/281 项梯度未过 `atol=rtol=1e-4`，该失败永久保留。没有证据将它解释成已证明无害的后端噪声。
2. 本次审查不验收新 CPU 保存实现，当前没有新方案执行结果；尚无其 forward、所有梯度、buffer/RNG/BN、完整作者 batch、显存/主存、耗时与 strict reload 结果。以上阻止运行接受和正式训练。
3. 原 B128/K16 的活跃层快照及 stack 仍会保留在 GPU；CPU 保存并不保证该峰值消失。必须实测，不能由主存余量推导 PASS。
4. 新方案需要按实测主存和 GPU 容量安排并发；max4 是上限，不是四份 CPU 保存图必然能同时容纳的证明。磁盘先前不足，但最新已读取的退役收据显示在该时点恢复超过原固定预算；不再把那个旧磁盘快照列为当前已知阻断。

## observed_facts

- V1 原始失败日志 `terminal_intake771/primary/logs/independent_native_evidence_20261003_v1/independent_native_evidence_20261003_v1_m0_global_only_RGBNT100/m0.log` 定位到 `correspondence_roles.py:86` 的内层 `torch.stack(captured[index], dim=2)`。申请 218 MiB 失败；进程占用 23.42 GiB，PyTorch 已分配 22.52 GiB，GPU 剩余 78.56 MiB。它在 forward 阶段已经失败，单纯删除反向后对象不能解决这个发生时点。
- V2 按模态 checkpoint；V3 改成 12 个原 CLIP block 内部的 non-reentrant checkpoint，并保留 RNG。V3 的 `checkpoint(block._evidence_original_forward, ...)` 调用原 bound 方法，现有模块级 adapter forward hooks 在 checkpoint 外。源码没有显示能直接坐实的漏调用 adapter 或在重算时重复原模块 hook 的错误；也不能据此证明 CUDA 数值等价。
- 已读取 V3 `terminal_intake775/INTAKE.json`、原始 backward 日志和两份梯度 JSON。RGBNT201 的见证为完整 B64 中前 32 样本。global_only 209 项梯度全过；semantic 281 项中 92 项失败。raw/fused/global/loss 最大绝对差均为 0，heads 的既定 forward 检查先于失败断言且已经通过。
- 对已保存逐参数标量的只读汇总：92 项失败为 81 项 Signal/CLIP/camera 参数与 11 项早中层 adapter 参数。camera 差为 0.0009765625；全体最大差为 conv1.weight 的 0.005859375。其余 72 项角色/读出等参数无超门项，最大差为 4.76837158203125e-7。该分布是定位线索，不是首个分歧算子的证明。
- V3 在 semantic 梯度断言处中断；该分支后面的 buffer/RNG/BN/最终状态断言没有执行到，不能称它们已通过。V3 没有完整 backward PASS 文件、M0 或正式训练结果。
- 已实际读取 `original_repeat_intake776/primary` 的原控制日志、JOB、结果与梯度记录；其 intake 为 6 项原始文本、293 源码。JOB 开始于 15:09:22.781584+08:00，结束于 15:10:11.543323+08:00，exit0。semantic RGBNT201 单个见证批次为 B64 中前 32 样本、4 个身份。两模型恢复原 block.forward，281 项全过，最大差 2.9103830456733704e-11；Signal 和 adapters 的保存梯度差全为 0，5 项其他参数有极小非零差。BN count=1，buffer/RNG/状态相同，原 forward 已恢复。
- 核对的 9 个核心工作树文件与 original_repeat_intake776 中对应源码字节相同，包括两类模型、heads、两份 backward 检查和 V3 runner/queue。未声称逐行审查全部 293 文件。
- 现存 `cpu_component_witness/CPU_WITNESS.json` 记录 PyTorch `2.5.1+cu121`，这是已有环境版本证据，不是新 CPU 保存功能测试。官方 v2.5.1 源码包含该 API；默认非 pinned 路径用 `tensor.cpu()` 打包，反向用原 device 恢复，`pin_memory=False` 不请求异步恢复，不修改 dtype。它替换保存张量的存储，不重做前向。[PyTorch v2.5.1 graph.py](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/autograd/graph.py#L300-L355)

## cause_limits

原控制只说明新记录的这一个批次/状态/RNG 条件下，原实现的重复误差远低于现有门。它不是历史 V3 输入及未保存梯度张量的完整重建，也不证明原实现对所有 batch 普遍确定。

原始 JSON 只保存最终逐参数最大差与门结果，没有每个算子的中间梯度、重算张量值、实际内核/stride 轨迹，无法区分重算的数值路径、反向累加顺序或其他执行差别。camera 是第一个遍历到的失败参数，不是已定位的首个致因算子；某些差恰为 FP16 刻度，也不能据此宣布具体内核原因。

现有 semantic 的实际 `sample_context` 来自 `FP32SlotCompetitionRoles`，是 FP32 attention 读出；不能仅凭祖先类里存在 `grid_sample` 就把它当本次活动路径的确定原因。原作者 BN/heads 位于 backbone 之后，V3 block 重算范围不包含它们。不得把 BN 重复更新、RNG 漂移、Mamba 或 CUDA 非确定性写成已证实原因。

## selected_minimal_next_action

使用新文件和新入口，从原 `run_independent_native_evidence.build_core` 建模。只给该模型现有的 `model.evidence_model.backbone.forward` 添加一个调用边界：闭包保留其原 bound forward，在 `with torch.autograd.graph.save_on_cpu(pin_memory=False):` 内原样传入全部参数并调用一次。保留现有 nn.Module 注册结构与参数名；不把 backbone 放进新的子模块层级，不安装 V2/V3 checkpoint，不改封存 forward 的函数体。

上下文覆盖原 12 层 visual 前向及其中的现有 adapter hooks，并在返回 stages/global 时退出。角色读出、Mamba、native detail reader、author BN/classifier、loss、optimizer、GradScaler 和 backward 均沿用原路径。无梯度的调用本身没有需保存的反向张量，因此不需要新增训练/评估 fallback 分支。

选择 `pin_memory=False`，固定为第一候选；不加自定义 pack/unpack、streams、预取、压缩、去重或失败后切换逻辑。此方案不改变 forward 运算定义并避免 checkpoint 重算，但保存张量的对象/存储可能变化，因此仍须用原数值门验收，不能预告逐位一致。

预计代价是 GPU↔CPU 同步传输和 CPU 内存增加。官方实现按每次 save 复制；同一底层张量多次被保存可能产生多份 CPU 副本，参数本身仍驻 GPU。被 Python 引用的 captured 张量、stack 输出和后续角色计算不会自动离开 GPU。官方教程也明确将 CPU 保存视为内存与耗时的交换；不把教程其他网络的显存缩减或慢速倍数外推给本实验。[官方 saved-tensors 教程](https://docs.pytorch.org/tutorials/intermediate/autograd_saved_tensors_hooks_tutorial.html)

## required_fixed_runtime_checks

1. 先审查新 helper/入口：原封存源码保持不变，只有上述 forward 边界；无 checkpoint、无额外注册模块；原 state_dict 键、requires_grad、optimizer 参数覆盖与 cfg 相同。首次实际运行记录原环境版本、API 可用性和 backend 标志；不安装或切换 PyTorch。
2. 延续已登记的三数据集×三 variant 原实现/候选成对见证，匹配同一初始 state、完整作者 loader 批次及固定前 32 样本、CPU/CUDA RNG、FP16 autocast、init_scale=256 GradScaler。记录具体 batch 后按原顺序执行；不挑通过批次、不重复试到过。
3. raw/fused/global、所有 heads 和 loss 仍用 `atol=rtol=1e-5`；unscale 后所有可训练参数须有同名集、相同 None 状态、有限值，逐项 `atol=rtol=1e-4`。保留全部测量再断言；不只检查 camera 或首个失败项。一次失败即保存并停止这条候选。
4. 成对见证完成后核对 buffer/参数状态与 CPU/CUDA RNG 完全相同，作者 BN count=1，原模块 hooks 全部正常移除，且没有 checkpoint 重算路径。该 B32 见证只验证数值，不替代作者 batch 的容量。
5. 数值门通过后才运行既定 9 项 M0，各自使用完整作者 batch/K；RGBNT100 必须 B128/K16，其余沿用原作者 cfg。每项均完成既定 8 次有效更新、BN count=8、冻结参数不变、visual/camera 实际更新、native 14 项参数梯度与更新证据、strict reload。不能继承 V1 的 M0 PASS 充当新方案结果。
6. 在完整 M0 测量 GPU 峰值 allocated/reserved、进程 CPU RSS/峰值、同机 MemAvailable 和 swap 变化、每步 wall time，覆盖 forward/反向/optimizer/reload；记录是否完成实际 stack。由实测结果估计后续 50 轮时间与允许并发，不按显存释放量直接估算 CPU 所需容量。
7. 全部既定门通过且实际资源满足后，才能按原计划启动 9 端固定 seed42/50轮；batch、loss/heads、精度、学习率/调度、训练数据、正式指标和原保存策略保持一致。实际接受、M0和正式结果必须另有运行收据；本报告不提供任何运行 PASS。

## resource_limits

- 唯一允许计算资源：2026 的物理 GPU0–3，同时最多4；本 reviewer 没有访问服务器。2025 仅文本不可用条件不变，无 N2/N3、新设备、外部服务或训练数据。
- 已读取 15:14:16 intake：MemTotal=131577172 kB，MemAvailable=116887232 kB，SwapTotal/Free=8388604 kB。这是时点容量，不是 CPU 保存单图成本或四并发保证。
- 同一原始 intake 的 GPU 文本为 GPU0 18MiB/0%、GPU1 13774MiB/100%、GPU2 18MiB/0%、GPU3 156MiB/0%。因此本报告不把该收据写成“四卡空闲”；新运行需按实际空闲和主存容量安排，不抢占他人任务。
- 已读取 `closed_v1_probe_retirement776/RETIREMENT.json` 和 exit0：精确退役原失败 V1 的 3 份已封存 M0 reload probe，共1059051944B；原文本/哈希保留，直接二进制重放已退役。该操作由主代理完成，不是本 reviewer 所执行。
- 最新退役收据 free=10770534400B，高于原固定 campaign+reserve=10292822016B，余477712384B。保留原预算，启动时按当前容量再查；CPU 保存本身不写磁盘，不扩展为磁盘 offload。
- 长任务沿用240秒轮询并用完整 M0 的实测耗时估算首次观察点；不短间隔反复查、不自动重试、不用时间流逝宣称运行完成。

所有相对证据路径均以 `C:/Users/gb/.codex_tmp/independent_evidence_draft/` 为根；科学源码路径以所列工作树或 intake 的 source/ 为根。
