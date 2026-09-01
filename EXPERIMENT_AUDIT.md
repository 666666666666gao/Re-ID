# Experiment Audit Report

**Date**: 2026-09-01
**Auditor**: GPT-5.5 xhigh（独立只读审查）
**Project**: TriFusion RGBNT201

## Overall Verdict: WARN

## Integrity Status: warn

正式官方指标及其哈希链真实且内部一致。未发现伪造 ground truth、按模型自身统计量归一化指标、幽灵结果、使用 test 标签选模、修复阶段重跑官方评估或执行优化器 step。警告来自审计时的交接文档仍保留“正式实验未启动/官方访问 0”的旧状态；该文档必须在发布前更新。

## Checks

### A. Ground-Truth Provenance: PASS

- RGBNT201 身份和摄像头标签来自数据集文件名，而非模型输出：`modeling/trifusion/data.py:67`、`:72`、`:99`、`:102`。
- 正式训练与测试分别来自 `train_171` 和 `test`，代码拒绝身份交集：`modeling/trifusion/data.py:386`、`:396`、`:400`、`:405`。
- ReID 评估按身份/摄像头标签剔除同身份同摄像头 junk：`utils/reid_evaluation.py:31`。
- 数据审计登记 `valid=true`，test 为 30 身份、836 triplets、摄像头 1/2：`evidence/rgbnt201_audit_20260831.json`。

### B. Score Normalization: PASS

- 最终评估仅对特征做标准 L2 归一化，再计算欧氏距离：`tools/run_trifusion_experiment.py:2260`。
- mAP/CMC 来自排序距离和真实 ID/camera 标签：`utils/reid_evaluation.py:27`。
- 未发现用预测自身 max/min/mean 对最终指标做归一化。特征 L2 归一化不是指标膨胀。

### C. Result File Existence: WARN

正式结果文件存在并一致：

- `official_test_metrics.json`：epoch 60；query/gallery 各 836；official access/evaluation 各 1；fused `59.14784166853979 mAP / 63.27751196172249 Rank-1`。
- `official_test_access_guard.json`：`status=COMPLETE`，metrics SHA-256 为 `a75d51aa5e17bc11c8c27246fc005fac5c764b4813b147c9706ed2ca5b0eeb85`。
- `run_summary.json`：`status=PASS`、`single_seed_target_exceeded=false`、`sota_claim_supported=false`。
- `.resume/latest.json` 的完成证据与正式 metrics/checkpoint/guard 哈希一致。

审计时 `docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md` 仍写正式训练未启动和 access 0，故判 WARN。本次提交已将该文档更新为真实完成状态。

### D. Dead-Code Path and Repair: PASS

- 正式路径实际调用 `_evaluate_official_fixed_endpoint_once(..., evaluate=evaluate)`：`tools/run_trifusion_experiment.py:2562`；`evaluate()` 实际调用 `evaluate_reid()`：`:2223`。
- 一次性守卫先登记访问，再写 metrics/guard：`tools/run_trifusion_experiment.py:211`。
- 修复器将官方评估函数替换为调用即失败：`tools/repair_trifusion_directional_final_completion.py` 中 `_forbidden_official_evaluation`。
- 修复器将 `torch.optim.AdamW.step` 替换为调用即失败。
- `repair-0002/completion_receipt.json`：`training_reexecuted=false`、`official_test_reexecuted=false`、`optimizer_steps=0`、`status=PASS`。

### E. Scope Assessment: WARN

- 实际范围：一个数据集、一个 seed 42、一个 epoch-60 固定终点、无 reranking、无 baseline 复现、无消融。
- 只支持报告本次单种子正式指标。
- 不支持 SOTA、目标超越、多种子稳健性、全面实验或消融结论。

### F. Evaluation Type: real_gt

正式 ReID 指标属于 `real_gt`：使用 RGBNT201 文件名提供的身份/摄像头标签。路由校准是独立的训练目标描述性校准，不是因果或身份留出校准证据。

## Action Items

- [x] 更新统一交接文档中的正式结果、官方访问计数和检查清单。
- [x] 将 `official_test_metrics.json`、`official_test_access_guard.json`、`run_summary.json`、`repair-0002/completion_receipt.json` 登记为权威结果链。
- [x] 明确 `single_seed_target_exceeded=false`，不声明 SOTA，不进入消融。
- [ ] 如果论文必须使用“官方 evaluator”一词，应补充与上游 RGBNT201 evaluator 的只读等价性回执；当前审计验证的是本地 official-style evaluator。

## Claim Impact

- 单种子 epoch-60 RGBNT201 正式指标：supported。
- 官方测试恰好访问/评估一次：supported。
- SOTA、目标超越、融合优于分支：unsupported。
- CIRC/URGC 校准：只能表述为训练目标上的描述性方向性证据；`query_gallery_symmetry_claim_eligible=false` 必须保留。
