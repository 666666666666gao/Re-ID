# 初始辅助梯度边界诊断：源码复核

结论：**PASS（SOURCE_ONLY）**。没有发现需要修改的阻断问题或非阻断缺陷。复核仅涉及本地源码、计划及已保存的 JSON/JSONL 文本；未执行神经网络、未构造模型、未 SSH，也未改动实现、模型或历史证据。

审查者：`/root/review_initial_aux882`；请求配置为 `gpt-6-astra / max`。独立上下文、同模型家族，接受状态为 provisional；实际服务后端和实际推理档位未获独立证明（backend not attested）。

审查对象：`tools/diagnose_initial_incremental_gradient_boundary.py`，依据 `INITIAL_BOUNDARY_PLAN_20261007_882.md`。沿调用链检查了原增量目标入口、global/role heads、foundation 构造与训练循环、角色采样和读出、两 GPU 分段实现及现有完整 state 摘要函数。

1. **初始化和历史绑定正确。** 第 25–36 行固定 RGBNT201、repair_keep、semantic、seed42 和原公开 CLIP 路径，通过原 `configure()` 与 `foundation.build()` 构造。后者先核对已保存 witness；诊断再核对原 `training.json` 的 initializer。底层 `run_clean_clip_joint.py:37–83` 使用原 seed 设置、公有视觉权重核对和新 camera/role/head 初始化，没有加载旧 M0 或正式 ReID 权重。已保存本地副本 `0051_RGBNT201_repair_keep.json` 与 `0066_training.json` 的 binding 全字段相同，初始 state 为 `93775618f02cbab681550773d271d9d8b6cdd7a44e28c35eeae5118802cf7c7e`，B64/K8。新模型是否实际重建为该状态仍由运行时断言验证，源码审查不冒充该运行结果。

2. **样本边界正确。** 第 39–43 行调用原 source-train loader，取一批并逐项比较历史首批 paths 和 labels。原 loader 的数据源为 `records_for(protocol, 'train')`，训练变换及 B64/K8 来自原配置；已保存首批有 64 个路径和 64 个标签。未调用写历史 batch-order 的包装器。路径/标签相同只证明样本身份及顺序相同，不证明像素增强、RNG 或旧八步计算图相同；计划和结果边界已明确不作这种主张。

3. **hook 仅观察，目标对齐正确。** 第 50–59 行的 hook 仅保存原输出引用，隐式返回 `None`，没有 detach、原位修改或替换 forward 输出。三个无 bias 的 query 线性层和三个 key 线性层在原 `GlobalTokenRoles.forward()` 中各使用一次；原 FP32 sampler 保持不变。RoleEvidence hook 捕获实际供 `read_evidence()` 使用的 cnn、transformer、mamba 张量。targets 共 17 项：shared_global 1 项、correction/三个角色出口/六个 Q/K 出口共 10 项、六个参数；`offset = 11` 与两个切片、命名顺序完全对应。角色出口的导数包含其下游跨角色连接，这是原图的总导数。

4. **训练模式和 AMP 保留原路径。** 第 60–63 行调用 `model.train()` 并使用原 CUDA float16 autocast。`AuthorHeadEvidence.train()` 会恢复作者 Signal 和 BN 的训练模式；`GlobalTaskRoleHeads` 继续执行原 global head 与使用分离参数、克隆 buffer 的 fused head。Q/K sampler 及 repair_keep loss 原有的局部 FP32 区域仍保留。辅助目标使用原函数、原 0.1 margin 和原等权定义，global 参考及角色输入的 stop-gradient 均未改动。

5. **一次图、两次独立 VJP、零优化符合计划。** 第 61–72 行只有一次模型前向，随后对同一 loss/图依次计算乘 1 和乘 256 的 VJP。没有 optimizer、`.backward()` 或 `.step()`。`allow_unused=True` 配合单独的 unused 字段，区分无连接和数值零；目标/梯度 dtype 被保留，统计先转 FP32 再除尺度。shared_global 梯度必须为 None，已有梯度必须有限；没有要求 scaled 梯度非零的资格门。256 与原生产 GradScaler 初值相同，但独立辅助 VJP 不是完整 combined-task 生产反向的重放。

6. **buffer 与完整模型张量状态检查正确。** 第 37–38 行先记录现有完整 state 摘要及全部 named buffers；第 83–87 行确认所有参数 `.grad` 仍为 None，恢复包括 BN running statistics 和计数器在内的 buffer，随后核对完整 state。原摘要函数遍历全部 `state_dict()` 张量，故参数及持久 buffer 都包含在该检查内。这是代码规定的运行时成功条件，本次未声称已实际执行恢复。

7. **历史结论和写入范围保留。** 输出为 `INITIAL_AUXILIARY_BOUNDARY_COMPARISON_COMPLETE`，附 1 forward / 2 VJP / 0 updates 以及明确的范围说明；没有改写旧 training receipt、旧 gate 或保存新权重。当前矩阵的六份生产状态全部为 `M0_PASS`，六份独立未缩放 auxiliary gate 全部失败。此次检查无论结果如何都不能改判原 0/6、证明 underflow 根因、永久断路、检索收益或全训练资格。代码中的 `checkpoint_loaded=False` 在该公开初始化上下文中表示未加载 M0/正式 ReID checkpoint，公开 CLIP 输入仍按原构造路径读取。

静态验证：目标脚本 AST 解析通过；语法树中模型调用一处，VJP 调用一处位于固定两个尺度的循环中。仅使用 Python 标准库解析文本，没有导入 torch 或项目模块。

BLOCKING：无。NON-BLOCKING：无。无需补丁。下一步的真实输入/源码发布、远端执行和实际断言结果不在本次 SOURCE_ONLY 验证范围内。
