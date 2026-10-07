# CPU saved-tensor 边界诊断：源码审阅

审阅时间：2026-10-03T16:57:16+08:00。结论：**PASS / SOURCE_ONLY**。`BLOCKING: []`；`NONBLOCKING: []`。未发现当前实现中需要修复的具体缺陷。

审阅者：`/root/review_cpu_boundary_diagnostic779`，fresh context。父任务提供的实际 native 请求参数为 `model=gpt-6-astra`、`reasoning_effort=max`、`fork_turns=none`；底层 backend 身份未独立验证。`review_independence=same-family`，`acceptance_status=provisional`。

本次只读取本地源码、计划、已采集 primary/runtime/rescue，并执行标准库 AST 与字节检查。没有连接 SSH、导入 torch 或目标模块、构建模型、执行 forward/backward/GPU、修改生产源码或删除权重。

审阅对象是发布工作树中的 `tools/diagnose_cpu_saved_boundaries.py`、`refine-logs/cpu_saved_boundary_diagnostic_v1/EXPERIMENT_PLAN.md`，以及私有 `C:/Users/gb/.codex_tmp/deploy_cpu_boundary779.py`。同时检查了已有 CPU-save wrapper、backward checker、模型构造、数据、author heads/loss/optimizer 和 queue binding 调用链。历史证据来自 `cpu_saved_terminal_intake778`、`rescue_cpu_saved778` 和 `cpu_saved_runtime779`。

## 已核实的正确实现

1. **运行范围固定。** 诊断脚本第 69、101–123 行固定 semantic RGBNT201，在 `(reference, candidate)` 两臂各调用一次生产 forward/backward。`mode='prepare'` 仅复用构造/优化器接口，没有调用 campaign、M0、训练、正式评分或权重保存入口。AST 中只有第 122 行一个 backward 调用点，没有 `.step()`、`.save()` 调用；代码也不调用 scaler update。配置中的 `epochs=50` 不会触发训练循环。

2. **初始化、输入和损失保持原条件。** 第 78–89 行复用原始构造与 CPU-save 构造，核对历史 binding、两臂完整初始 state 和 cfg，取同一 author B64 批次的前 32 个样本。历史 semantic binding 明确为 seed42、B64/K8、281 个可训练参数张量；构造链使用 public CLIP 和新 camera/heads，不加载 ReID 模型权重。第 113–123 行在每臂构造相同 author optimizer/loss 前恢复同一 CPU/CUDA RNG，固定 autocast FP16 和 GradScaler256；loss 使用数据集真实 labels/cameras。

3. **saved-tensor 委托保持原实现。** 第 30–47 行以实际 `save_on_cpu(pin_memory=False)` 对象的 `pack_hook` / `unpack_hook` 各委托一次，每次调用记录输入、CPU payload 或恢复输出的标量元数据和序号。回调没有额外 tensor clone、布局修复、dtype 转换、stream 或原 GPU saved tensor 引用集合。第 94–99 行直接调用已保存的原始 bound forward，替换 v4 wrapper，没有嵌套两个 saved-tensor contexts。实际 runtime 采集为 PyTorch 2.5.1+cu121；其已采集实现确实通过 `saved_tensors_hooks.__init__` 暴露上述两个 delegate，非 pinned 分支为 `.cpu()` 和 `.to(device, non_blocking=False)`。

4. **两臂的边界观察对称。** 第 104–111 行在同一 backbone 返回值 `stages` / `shared_global` 上调用 `retain_grad()`，forward hook 不替换返回值，也不执行 `.cpu()` / `.item()` / 全局同步。第 128–135 行在原生产 backward 完成后复制边界值及梯度；optimizer unscale 只作用于参数梯度，因此保留的非叶边界梯度再除以固定 256 的处理正确。未增加第二次反向或新的 loss。边界比较是诊断量，不代替原参数梯度门槛。

5. **已知失败不会再挡住后续诊断记录。** 第 124–170 行测量全部可训练参数梯度、outputs/heads/loss、两个边界以及 buffers、CPU/CUDA RNG、两臂 BN counts、完整 state 和原 capture-hook 清理结果。第 192 行先写 `MEASURED.json`，第 193–197 行随后执行原 forward/head `1e-5`、gradient `1e-4` 的 atol/rtol 门槛和状态检查；原 semantic 有限梯度但不等价的失败能够先留下全量比较结果与状态检查结果。原始 RNG 字节列表和输入标签/相机/路径也被记录。初始化、有限性和结构前置断言保留，没有放宽门槛。

6. **私有启动器没有扩展实验。** `deploy_cpu_boundary779.py:10–19` 读取 publication/review，保留 intake 的 301 份源码约束并只追加本次 tool、plan、review MD；远程第 41–49 行核对已发布 HEAD、来源和旧 v4 的封存前状态/父进程退出及原初始化记录。第 51–54 行仅从 2026 physical GPU0–3 中选择当前显存占用小于 500 MiB 的一张卡，无抢占调用；SSH 端口固定 2026，无 2025 执行路径。新的 campaign 与旧 v4 分离，诊断 `--output` 为新 campaign 下尚不存在的 `observations`，不会与父目录的预创建冲突。

7. **子进程生命周期和退出码正确。** 启动器第 21–32 行的 controller 使用真实 `Popen` 和 `child.wait()`，记录实际 child PID/exit code，再按该 code 写 COMPLETE/FAILED 并退出。模型日志保存在 exclusive `diagnostic.log`，controller 源码保存在 `CONTROLLER_SOURCE.py`。外层第 63–64 行只确认 controller 的实际 Popen，不将其视作诊断完成；没有重试循环、自动再诊断或后续 M0/full50 启动。

## 检查证据与实际边界

- 诊断脚本、私有启动器、提取出的 controller，以及以惰性 receipt 占位值展开的远程源代码，均由 `ast.parse` 通过；没有导入或执行这些模块，也没有使用 py_compile。
- intake 的 301 份本地 `_source` 文件全部匹配其已有摘要。相对于该已采集源码，当前发布工作树有 167 份字节相同、27 份仅 CRLF 差异、107 份因稀疏工作树未展开，没有已发现的实质文本差异。缺失依赖从已采集原始源码读取；启动器继续要求实际远程 301 份满足既有精确约束。本次没有要求修改这些历史文件或增加摘要机制。
- 从 primary JSON 重新计数：semantic 281 个梯度中 94 个固定门槛失败，None 不一致为 0，最大绝对差为 0.0078125；raw/fused/global/loss 差为 0。真实 traceback 的首个失败参数是 `cv_embed`，差为 0.001220703125。该参数枚举位置不构成最早反向分歧定位。
- 本地初次 AST 命令误用了环境中不可用的 bare-python shim，仅返回 `No pyvenv.cfg file`，没有执行检查代码；随后使用已安装的 `uv run --offline --no-project --python 3.11 python -B -` 完成标准库检查，无新增依赖。

本报告未验证新的 GPU 数值结果、容量或 wall time，也未把历史资源快照视作执行时的空闲证明。观察会改变内存驻留/时序；metadata 差异不能单独证明唯一原因。保留边界梯度和两臂模型的诊断峰值也不等于 full-author-batch 的训练容量。

执行仍按既定先后顺序：先实际完成五份 publication779 校验，再由这一启动器执行一次诊断。审阅时 `five_copy779.json` 尚不存在，因此本报告不声称发布或启动已经发生。计划预计约 1–2 分钟，首次只读观察在 180 秒；本次未创建或运行 observer。诊断无论 PASS、FAIL 或不复现，都不能改写 sealed v4 FAIL、满足 M0 或自动恢复训练；不复现则停止，不追加重复诊断。2025 仍仅允许文本同步。

无需补丁。该结论仅通过本次 SOURCE_ONLY 实现审阅，不是新的数值等价或研究结果验收。
