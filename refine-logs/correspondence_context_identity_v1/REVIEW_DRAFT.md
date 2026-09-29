# Context/local identity 草稿来源复核

日期：2026-09-29。最终结论：**PASS（仅来源复核，provisional）**。初审唯一阻断项已最小修复；本结论不构成生产 M0 通过、正式部署授权或方法有效性证据。

- review_scope: source-only
- review_independence: same-family（独立上下文，非跨模型家族审查）
- acceptance_status: provisional
- requested_model: gpt-6-astra
- requested_reasoning_effort: max
- actual_serving_model: 未由运行环境暴露，不能据请求值声称已核验实际服务模型

## 范围

初审读取固定实验计划、新模型与 runner，以及继承的 correspondence_roles、correspondence_evidence_readout 和原 runner，并追踪相关初始化、训练标签与官方评价代码。本轮只重新检查唯一阻断项的修复，没有重新开展模型测试或扩大复核范围。

复核对象：

- `EXPERIMENT_PLAN_20260929_1246.md`
- `modeling/trifusion/correspondence_context_identity.py`
- `tools/run_correspondence_context_identity.py`
- `modeling/trifusion/correspondence_roles.py`
- `modeling/trifusion/correspondence_evidence_readout.py`
- `tools/run_correspondence_roles.py`

## BLOCKING：已关闭

初审发现，新 `evaluate()` 在 `build()` 配置作者 Signal source 前导入 `utils.metrics`。独立评价进程会先绑定仓库内同名模块；随后修改 `sys.path` 不会替换已缓存模块，作者文件断言因而失败。

已复核当前 `tools/run_correspondence_context_identity.py` 第 184–189 行：先调用 `build(args, protocol)`，再导入 `author_metrics`，并保留其 `__file__` 必须精确等于 `args.signal_source / "utils/metrics.py"` 的断言。修复只调整导入顺序，没有 fallback、缓存绕过或断言放宽。**该阻断项关闭；未发现其它具体实现阻断项。**

## 初审确认的实现内容

- 当前 CNN → Transformer → Mamba 路径保留原有 `mamba_norm`、双向平均及区域读出，没有额外 Mamba 残差。
- global 条件仅改变候选内容评分；局部 value 和桥接 slot 未显式加入 global 值。local 出口来自区域 correction，fused 出口才加回 global。
- 三角色查询投影零初始化，继承角色 state 被复制；新增角色和辅助头构造位于 CPU `fork_rng` 中。各条件的构造分支支持共同初始 state 与后续 RNG 一致性；这属于来源判断，不替代实际检查。
- no-aux 头冻结且不执行。未发现静态可确认的、应训练却没有使用路径的参数；实际累计梯度仍须由生产 M0 验证。
- local/global 辅助 CE 分别使用归一化联合局部/global 表示，使用同一真实训练身份标签、头结构、权重和 label smoothing；fused CE、triplet 与辅助 CE 分开记录。
- `BASE_BUILD` 与 runner namespace 重绑定连接到新模型、训练、评价及 checkpoint 函数。紧凑 checkpoint 保留新模块状态和 condition 检查，既有 provenance/状态集合断言保留。
- 最终评价从同一 best checkpoint 提取完整 query/gallery 的 fused、shared_global 和 joint_local 距离，共用官方身份、camera/scene 元数据；三条路径均检查作者与 CPU 指标一致，fused 结果还核对被保存的 best 指标。

## NON-BLOCKING 与未完成门槛

没有提出额外防御逻辑或无证据的实现修改。生产 M0、正式 queue/collector 仍未完成，必须保持待执行状态；真实 Mamba、AMP、累计有限非零梯度、冻结 baseline 不变和独立重载一致性均不能由本次来源复核代替。另行执行的 synthetic token encoder 与 linear Mamba CPU 结构测试也不是生产 M0 证据。

原 M3 12 条件训练/重载继续使用原配置；本报告不修改旧科学源码、训练定义或已选权重，也不替代剩余实验和完整结果分析。新方法目前没有正式结果。

本审查者没有运行 GPU、训练、SSH 或访问凭据，没有修改模型/runner；本轮仅写入本报告及其固定名副本。
