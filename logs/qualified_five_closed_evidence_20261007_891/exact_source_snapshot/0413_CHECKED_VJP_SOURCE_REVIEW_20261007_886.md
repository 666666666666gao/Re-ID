# Checked VJP 源码审查（2026-10-07 / 886）

**最终 Verdict: PASS — 0 blocking，0 nonblocking。Scope: SOURCE_ONLY。**

初审发现1个实际阻断；对修复补丁作了一次复审后关闭。初审发现和修复过程保留如下及JSON的 `review_history` / `resolved_blocking`，没有改写为从未发现问题。

请求 reviewer：gpt-6-astra / max；fresh-agent `/root/review_checked_vjp886`，same-family / provisional。实际 backend 与 reasoning effort 未独立证明（UNATTESTED）。

本次仅本地源码、Python 3.10语法AST及提供的REMOTE.json阅读。35/35个不同文件AST PASS；8/8个原/新损失AST比较PASS；补丁涉及路径6/6 AST PASS。未执行项目import、NN、SSH、优化器更新或训练。

## 初审发现与修复复审

### B001 — 已修复：报告子进程未继承 checked source_map

初版 `queue_incremental_checked_full.py` 只在父进程替换 `full.source_map`。旧full coordinator却在独立Python进程中直接执行原报告；原报告main将 `base.source_map` 重新绑定到原 `panel.source_map`，读旧 `FULL_SOURCE_SCOPE.json`，而 `queue_foundation_recipe.py:112` 要求来源映射与新manifest完全相等。该路径会在六个full50完成后阻断15个配对报告。初审结论是FAIL。

修复在 `tools/queue_incremental_role_objective_full.py:17` 增加 `REPORT_ENTRY`，其第109行唯一报告subprocess.run使用此常量。`tools/queue_incremental_checked_full.py:20` 指向新 `tools/report_incremental_checked.py`。新报告进程第12–14行先配置checked，再把 `report.panel.source_map` 设为 `checked.source_map`，然后调用原 `report.main()`；原main第21行因此把相同checked函数传给 `base.require_sources`。来源映射在实际报告进程内正确绑定，原来源校验仍在。

复审确认：撤销REPORT_ENTRY的常量和命令替换后，full coordinator AST与HEAD完全相同；原报告源码git diff为空，统计和15pairs未改。既有393、397、398、402、406五个source scope都不含这个原先仅准备的full coordinator或原report；旧执行训练/M0代码未改。计划已明确说明这处实际修复。B001关闭，最终PASS。

## 其余核查

- **entry_import_and_alias_chain: PASS。** 新脚本将 ROOT 加入 sys.path，再导入原 incremental entry。其上游 run_visual_update_control/run_correspondence_roles 在导入 trifusion 之前加入 ROOT/modeling。entry.inner 精确落到 run_independent_native_evidence；configure 将 checked build_core/loss_values 经 native_research→native_partitioned→independent_native_evidence 传播至 foundation。prepare 也调用已替换的 checked build_core。
- **m0_forward_loss_and_single_increment: PASS。** 归一化 AST 的8项比较全部相同：基础 global+fused loss 调用、objective 分支、activity记录、最终 original+increment、QP选择、八targets及VJP参数、c norm和QP norm记录。新M0调用的是 entry.original_loss_values，未重复调用已加increment的旧entry.loss_values；每步只添加一次增量。
- **autocast_and_targets: PASS。** 唯一数值观测修改在独立 torch.autograd.grad 周围加入 torch.autocast(cuda, enabled=False)。仍为未缩放 increment、g/c/6个QK targets、retain_graph=True/allow_unused=True。三组无bias query和key Linear产生恰好6参数，来自同一CURRENT_MODEL。g None断言及逐张量finite检查保留；新增unused/context记录。
- **production_training: PASS。** 非M0直接调用旧entry.loss_values；生产forward/loss仍在原float16 autocast内，GradScaler init256的backward位于context外。原optimizer、scheduler、seed42、50轮、原batch、1536维推理及same-head结构均未改；checked入口没有新增Module或Parameter。
- **binding_and_initialization: PASS。** checked build只在原binding上写实际checked入口entry_sha256和观测协议。原构造仍从公开CLIP与fresh camera/heads开始；old semantic initializer除architecture/entry_sha256/scope外所有字段逐项比较。原模型state/参数数目/config等字段仍参与对比。
- **m0_gates_and_batch_pairing: PASS。** checked accept_m0添加8行context False和unused False断言后调用原accept_m0；原8次effective更新、全trainable累计非零、BN8、frozen不变、visual/camera更新、严格重载≤1e-5、g无梯度、c及六QK八步累计有限非零和完整首八batch行相等门全部保留。只有通过全部门才退役工程probe。
- **fresh_six_m0_then_six_full: PASS。** M0仍为3数据集×2目标并且不自动启动full；full require_m0仍验证6个COMPLETE/exit0/acceptance原件/正norm和probe已退役。独立新campaign/output禁止覆盖。full train重新build公开初始模型，只使用M0的初始化witness，不读取M0权重。
- **full50_selection_and_strict_first_evaluation: PASS。** 原50轮全部history、以mAP及后轮tie规则选择同一best、checkpoint全state strict=True重载、全部mAP/R1/R5/R10与best同源及1e-5比较、完整50轮旧rawsemantic batch-order逐字节相等都保持。失败无自动重启或放宽。
- **ground_truth_and_report_arithmetic: PASS。** 原评价逐条protocol GT identity对全query/gallery评分，MSVR310按scene、201/100按camera过滤，核对author evaluator，不rerank。原report保留每目标对raw_semantic/raw_global_only及repair_keep对MD，共3×(2+3)=15配对，以及每query、identity、修复/新增错误、完整曲线/E50和时间。统计源码未改；B001经新报告入口路由修复。
- **resource_gates: PASS。** M0 2684354560 bytes即2GiB+512MiB、full 5192548352 bytes不变；仅物理GPU0/1、max1，原240秒device等待保持。
- **closed_control_evidence_boundary: PASS。** 只读取本地REMOTE.json及诊断源码：一个新公开初始201 AMP图、原8targets/scale1、两次VJP，True重新进入context时六QK零/非unused，False时六QK非零；g无梯度/c非零，0optimizer且模型state还原。不是旧八步同pixels/cache复演，不追认旧FAIL，不代表新M0/full通过或检索收益。
- **report_subprocess_checked_scope: PASS。** 原full coordinator新增REPORT_ENTRY并在唯一报告subprocess.run使用它；checked_full将REPORT_ENTRY指向report_incremental_checked.py。新报告进程导入checked和原report，在调用原main前设置report.panel.source_map=checked.source_map；main随后将base.source_map设为同一checked函数，require_sources从新CHECKED_VJP_SOURCE_SCOPE验证新manifest。原report统计源码未改；full coordinator AST撤销REPORT_ENTRY两处替换后与HEAD完全相同。

## 实际构造与调用链

`checked.__main__ → incremental.main/configure → global_task_role → role_input_detach → native_research → native_partitioned → independent_native_evidence → foundation`。最终foundation的 `build_core/loss_values` 指向checked函数；保存的original构造函数链仍到 `clean.build_core` 的公开初始构造。`clean.control.GlobalTokenTriFusion` 是DetachedSemanticTriFusion，`inner.AuthorHeadEvidence` 是原IncrementalObjectiveHeads；新入口没有额外参数。`inner.train → partitioned.train → captured foundation.train` 保留原生产循环，evaluate仍为原foundation.evaluate。

M0命令索引2切换到checked脚本。full.m0与checked.original是同一导入模块，因此train/evaluate命令正确切换；现在唯一报告子进程也通过REPORT_ENTRY明确进入checked报告入口。

## Blocking / Nonblocking

当前均无。B001保存在已解决历史中。

## 范围和状态

CHECKED_VJP_SOURCE_SCOPE按任务约定待此次审查后登记，尚未把其登记或实际训练标为完成。PASS仅表示当前源码实现通过本次审查，不能宣称新六端M0/full通过。旧六端0/6、原失败、48更新及旧正式0保持。有限控制的True分支是在同一新初始图上重新进入autocast，既非旧八步pixels/cache复演，也非检索收益证据。
