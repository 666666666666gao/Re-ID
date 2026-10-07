# 新增证据目标：独立源码审查 878

日期：2026-10-07。结论：**SOURCE_ONLY PASS**。BLOCKING：0；NON-BLOCKING 缺陷：0；当前实现所需修改：无。

审查者：fresh Codex `gpt-6-astra`，reasoning effort `max`；`review_independence: same-family`，`acceptance_status: provisional`。按 experiment-bridge Phase 2.5 独立读取实际源码，未修改实现，未执行 SSH、模型构造、模型前后向、优化器或 GPU 任务。此结论不是实际初始化、M0、重载或检索效果通过。

审查依据为 `EXPERIMENT_PLAN_20261007_071500.md`，审查对象为当前 `modeling/trifusion/incremental_role_objectives.py`、`tools/run_incremental_role_objective.py` 及其实际调用链。这里只判断计划所定义的机制适配，不认证完整 MDReID 复现、公式原创性或完整 P1/P2/P3。

## BLOCKING

无。未发现当前目标计算、配置接线、输入元数据或梯度归属方面的具体错误。

## NON-BLOCKING

无需要修改实现的缺陷。下述 M0 验收范围属于计划尚待执行的工作，不记作已完成结果。

## 已核实的源码行为

1. **目标只加一次。** `run_incremental_role_objective.py:62-81` 先调用原 global+role 作者任务，再于头损失之外加一次 increment。`run_global_task_role.py:25-31` 保留 global 与 fused 两个作者任务；`run_foundation_recipe.py:155-159` 的车辆三头循环不包含新增联合损失，没有把它重复三次。

2. **MD 式比例适配符合登记公式。** `incremental_role_objectives.py:13-32` 使用未归一化 raw 欧氏距离、平方距离下界 `1e-12`，同 ID 掩码保留 self；每个分支分别取全 batch 正例最大距离、异 ID 负例最小距离。global 与 correction 比较分支 detach，fused 分支保留导数；分母仍包含有梯度的 fused 距离。当前作者 identity sampler 的 B/K 配置提供多个身份，负例断言符合既有训练路径。

3. **repair/keep 的关系与归约正确。** `incremental_role_objectives.py:35-70` 的三维张量表示同一 query 的所有合法正例×所有异 ID 负例；正例要求同 ID 且环境不同。参考 global cosine margin `<=0` 进入 repair，目标 `0.1`；`>0` 进入 keep，保留原 global margin。每个 cell 先对 query 内关系平均，再对该 cell 有支持的 query 平均，最终各乘 `0.5`。空 query/cell 贡献可微零值，不会产生除零，也不会取消原作者任务。

4. **新增损失没有通向 shared global 的反向路径。** `global_task_role_heads.py:20-32` 用 detached global 加 live correction 重建训练 fused；`role_input_detach.py:5-7` 同时截断角色输入 stages、context 和 shared global；两种新目标也 detach 参考 global。比例目标对参考 correction 的 detach 不会截断 raw_fused 中的 correction 路径。M0 的 `autograd.grad` 独立检查 g 为 `None` 并记录 c、六个 Q/K 的范数，不会把这些诊断梯度累加到参数 `.grad`。

5. **配置链没有递归或被后续 configure 覆盖。** 逐层核对 incremental → global_task_role → role_input_detach → native_research → native_partitioned → independent_native_evidence → foundation 后，最终 `foundation.build_core`、`foundation.loss_values`、`foundation.condition` 分别是新增入口的函数，schema 同步到新增实验；内层使用 `IncrementalObjectiveHeads` 与 `DetachedSemanticTriFusion`。构造链中的 `original_build_core` 保存原函数对象，未递归调用被替换的同名入口。

6. **模型与初始化仍是原 semantic 路径。** 新 head 子类只在辅助字典中传递环境，没有新增参数、buffer 或随机初始化。`run_independent_native_evidence.py:36-92` 经 `run_clean_clip_joint.py:37-107` 构造公共 CLIP、新 camera/head 和原 semantic 角色，再按既有方案分配两个 GPU 段。seed42、作者配置、50 epochs、优化器与增强保持原路径。borrowed detached heads 的参数停止梯度、BN buffer 使用克隆，新增入口未改变既有 global/head 与 role 的职责。

7. **训练环境绑定正确，未用 MSVR view 代替 scene。** 新入口从全部 protocol records 建立 basename→环境映射，MSVR 取 `scene`，另两集取 `camera`；训练用实际 `raw[4]` 的 basename 查找，原 camera embedding 输入未改。已读取执行者的 metadata witness 源码与 JSON：三集分别有 4,787 / 2,087 / 17,250 个映射项，无环境冲突，首个已存训练 batch 的全部路径可解析。此见证只证明这些元数据事实，不证明全训练 batch 的新增损失活动。

8. **评价与 M0 audit 不会因缺环境字段报错。** 车辆 `_eval_batch` 经同一个被替换的 `_training_batch`，可从全 protocol 映射解析 query/gallery 名称；RGBNT201 `_eval_batch` 直接生成 images/camera 输入。两者以及 foundation 的二样本 M0 audit 均调用 `model(batch)`，即 `return_aux=False`；新 head 在该路径不读取 `retrieval_environments`。因此 M0 audit 只保留 images/camera 是正确的。推理仍返回原 1536 维归一化 fused。

9. **保存与严格重载绑定一致。** 新 condition 包含 objective、权重1、margin0.1、cell 权重及真实环境语义；`run_foundation_recipe.py:103-107,174-186` 通过最终安装的 build/condition/schema 校验初始化见证与完整模型 state。M0 fresh build 后再次设置 CURRENT_MODEL 不影响已结束的八步损失诊断。正式评价仍按同一 mAP-best 的权重报告各项 CMC，并使用 protocol 的真实身份/camera/scene 和原官方度量，不把 global 参考当成评价真值。

## 尚待实际验收的边界

当前新模型初始化、六个八步 M0 和正式训练均未运行。已有四个 CPU 合成向量检查的源码与 receipt 已阅读，分别支持 keep、repair、空支持和比例公式的有限数值/导数行为；这些检查由执行者完成，不能替代新模型或六个 Q/K 的实际梯度证据。本审查没有重跑这些检查。

新增入口将独立 c/QK 梯度写入每步日志，但继承的 `M0_PASS` 本身检查的是**总损失**参数活动。正式推进时应按计划核验八步日志：g 的独立梯度均为 None，c 和六个 Q/K 各自在八步内有有限非零活动，并完成配对初始化/批次、BN计数8、严格重载及六端全部通过的验收。不能仅凭继承的 M0_PASS 字段认定新增目标活动已通过。当前日志提供了这些核验所需字段；不因此要求修改原 trainer 或增加通用检查框架。

独立审查执行的是源码追踪。一次本地 AST-only 命令因工作站 Python 返回 `No pyvenv.cfg file` 而未执行，未计为语法检查通过；没有为此安装环境或启动远端任务。

最小后续动作：保持当前实现，执行计划中已登记的实际 M0，并核验上述现有日志与回执。SOURCE_ONLY PASS 不提前放行正式训练，不证明科学收益、三集数值保护 global 或完整研究目标达成。
