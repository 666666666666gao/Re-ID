# V6 逐端研究执行：独立部署前审查

**结论：PASS_SOURCE_AND_STDLIB_MOCK；BLOCKING = 0。** 当前四个入口和继承调用链正确实现已选的逐端执行口径。此结论仅关闭源码审查门槛，不是实际 M0、容量、50轮训练或科学收益通过。

审查时间：2026-10-03T22:24:17.609521+08:00。实际模型：gpt-6-astra；reasoning_effort：max。fresh-context；review_independence：same-family；acceptance_status：provisional。

## 关键核验

1. **实际 CLI 调用链正确。** `run_native_research.configure` → partitioned.configure → independent.configure；`independent.main` 再配置后执行 `independent.train` → partitioned.train → 原 foundation.train。返回后写两卡内存和一次 production_m0_diagnostics。精确函数 AST mock 验证 V5/V6 各一次诊断写入，无递归、无漏写。
2. **M0 与正式训练从同一登记初始化分别新建。** 独立进程、seed42、公开CLIP、新camera/head；fresh50不读M0权重。完整作者batch/增强/BN/Triplet及两卡放置继承不变。首次实际整batch输入、共有state、shared输出及semantic/native zero-exit前向仍须通过。
3. **真实更新与状态检查保留。** AMP初始scale256、unscaled梯度/loss有限、8次有效optimizer step、全部可训练张量累计活动、视觉/camera变化、作者BN计数8和native14张量活动/变化均有检查。保存完整model state（含BN buffers），独立构造strict load及原1e-5输出比较。optimizer/scheduler/scaler各端新建；本任务不声称optimizer续训恢复。
4. **完整50轮与统一best语义正确。** 同mAP取较晚epoch；只覆盖一份best_map.pth，所有CMC跟随同一权重。新evaluate进程严格重载，完整query/gallery按真实GT和原camera/scene过滤评分，与独立实现核对。
5. **逐端顺序、资源与失败停止正确。** mock成功路径39条命令，201→MSVR→100，各集三初始化+paired forward，再global_only/semantic/native各M0→train→evaluate。无额外backward探针、无全九M0屏障；每次Popen固定CUDA_VISIBLE_DEVICES=0,1、单个双卡子任务，资源等待240秒。模拟201/native M0失败后第11条命令即停止，FAILED及退出码保留、报告0次。
6. **终态报告交付补齐。** 原18-job/9端验证、batch顺序一致性、GT评分和配对诊断公式不变；新wrapper落盘完整query identity/AP/首位rank，camera及scene分支的mock输出正确。曲线、身份分布、修复/新增错误、best到末轮变化、参数/内存/耗时原记录保留。

## 审查期间已修正

- 初稿关于“V5未写M0诊断”的解释不符合实际CLI调用链。执行者已删去V6重复train包装并改正计划，最终版本仅复用原回执链。
- 原compare虽计算逐query AP/rank，却未返回落盘。执行者已在V6报告层按原scorer补齐query_changes，未改模型、loss、优化器或评分公式。

## 验证与边界

`MOCK_RESULT.json` 为PASS；精确AST函数执行只用stdlib和假计算/假进程，实际模型forward=0、optimizer update=0、远端命令=0、包安装=0。审查者未修改项目源码。

首次PATH Python因缺pyvenv.cfg未运行；首次有效Python harness因mock遗漏STDOUT失败。仅修正mock常量，使用新fixture后通过，原fixture及`HARNESS_EXECUTION_NOTES.md`保留。这两项不属于模型失败或模型证据。

远端309文件预检及最终SOURCE_SCOPE由执行者准备，本审查未独立远端核验。实际GPU/磁盘、初始配对和每端M0仍须运行通过，尤其B128容量尚未证实。旧V1–V5 FAIL/STOP不改判；用户选择只取消额外反向重复性前置目标。

## 最终审查输入 SHA256

| 文件 | SHA256 |
|---|---|
| tools/run_native_research.py | `b48725622371fc6d9671c464ce97252e2e06eba465e213ce221d84c798e52cc2` |
| tools/queue_native_research.py | `b138144780a31dc8a5a4922cc4e15dbc0537d0f75ce47d823434d3d3a3c5bc65` |
| tools/check_native_research_pair.py | `f368de9c455971da5d4fd50757ce576088b0db526ecd992dac2a3f0d1a43ae70` |
| tools/report_native_research.py | `1117d3ad9e186ec941c8d84ab4363aa28b7ce9c78d492ca7671c95338121e77a` |
| refine-logs/native_research_v6/EXPERIMENT_PLAN.md | `06407ddd77a59cdf92c7b5b1313ba1c97b9993a5973f2adc1dfd89fe47249a44` |
