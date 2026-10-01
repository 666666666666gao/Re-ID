# N1完整报告与等待器源码复核

gpt-6-astra/max，same-family/provisional；两次实际回复原文如下。

## 初次审查

源码审查结论：**REQUEST_CHANGES，1 项阻塞缺陷；其余主要执行逻辑正确。** Same-family/provisional，仅源码与已有证据审查，不是运行或结果 PASS。

- **阻塞：描述性比较被新增为门槛。** [report_native_detail_complete.py:80](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_detail_complete.py:80) 为 high−global_only 计算阈值与 `gate_passed`，[第 140 行](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_detail_complete.py:140) 输出 `overall: PASS/FAIL`。这不属于登记的 N1-A/B。最小修复：保留差值及配对诊断，将其标为描述性比较，不计算或输出任何通过条件；A/B 汇总保持不变。
- **非阻塞：配对诊断边界文字失真。** 复用的 [analyze_correspondence_distances.py:75](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/analyze_correspondence_distances.py:75) 固定写着 “Different capacity”，也进入参数匹配的 high−low 结果。在新报告中仅修正该比较的边界文字，保留输入计算成本不同、官方选点与固定模型 bootstrap 限制。
- **非阻塞：旧晚期退化仅间接保留。** [report_native_detail_complete.py:109](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_detail_complete.py:109) 采用旧 matrix 行，保留时间、峰值显存及参数信息，但遗漏已存在旧报告中的 `final_minus_best_metrics` 等字段。固定旧 SUMMARY 仍完整保留，因此没有证据丢失；直接使用 `old_report['rows']` 即可完整带入，无需重新计算。

已核对正确：

- `native.install()` 修改的正是 `panel.verify/verify_m0/require_sources` 实际读取的模块全局；导入旧报告的 `elapsed` 不会撤销这些绑定。旧控制只读固定报告与矩阵，不会按 N1 schema 再验证。
- 新六端要求完整组合、父子 COMPLETE/exit0、全部 50 轮、最高 mAP 并列取较晚轮，以及训练/重载结果一致；实际复用保存全 state、`strict=True` 重载后生成正式回执。
- N1-A/B 数值条件准确。CPU 比较只读取已保存距离；GT、完整图库、同身份同 camera/scene 排除及修复/新增错误算式正确。三个固定协议仍匹配已有绑定，MSVR310 的 103 个图库独有身份保留。
- 等待器固定 240 秒；FAILED 时零报告；COMPLETE 后唯一 `Popen`，保留退出码和日志，不自动重试。
- 新六端完整损失/mAP 曲线、末轮退化、参数/时间/显存及来源记录已覆盖；官方选点、seed42、bootstrap 和非 SOTA 限制明确。

验证包括两文件 AST、旧六行与指定 SHA 报告一致、六份初始化绑定、两份实际 M0 条件/schema、11 项已登记源码绑定。没有编辑、部署、导入执行模型、运行采样器/评分器或调用报告。当前启动证据仍为四端训练中、两端待启动；新报告运行资格与科学结果均待实际完成。

## 最小修正后复核

复核结论：**SOURCE_ONLY_PASS；阻塞缺陷 0，未发现新增非阻塞代码问题。** Same-family/provisional。

已确认实际修正：

- [报告第 80 行](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/report_native_detail_complete.py:80)：high−global_only 使用 `gate=None`，不计算阈值或 `gate_passed`，仅显示 `Descriptive only`。正式汇总仍仅包含 N1-A/B，原判定算式未变。
- 第 82–87 行：high−low 正确注明参数与初始化匹配、输入计算成本不同，并保留官方选点及固定模型 bootstrap 限制。
- 第 117 行：完整引用已固定的 `old_report['rows']`，六端原有成本与晚期退化字段均保留。
- [等待器](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/wait_native_detail_complete_analysis.py) 未改变：240 秒轮询，FAILED 不调用报告，COMPLETE 后一次调用，保留失败退出码与日志，不自动重试。

两文件 AST 检查通过。未编辑、部署或执行模型、采样器、评分器、报告。运行与科学结果仍须等待六端实际完成及最终结果审计；本结论不代表科学 PASS。
