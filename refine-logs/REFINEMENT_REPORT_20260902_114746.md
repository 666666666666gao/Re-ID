# V15 Refinement Report

**Date**：2026-09-02  
**Rounds**：3 / 5  
**Final Score**：9.15 / 10  
**Final Verdict**：READY

## Problem Anchor

在固定 RGBNT201 `141-fit/30-dev`、远端单3090、seed42与 exact Signal 合同下，
让 CNN/Transformer/Mamba 生成选择/路由以外的新协同表示；先过 complete-path
identity-OOF，唯一D1达到65且严格胜出前不消融、不official、不称SOTA。

## Output Files

- Review summary：`refine-logs/REVIEW_SUMMARY.md`
- Final proposal：`refine-logs/FINAL_PROPOSAL.md`
- Round files：`refine-logs/v15/round-*.md`
- Score history：`refine-logs/score-history.md`

## Score Evolution

| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.0 | 6.5 | 6.5 | 7.0 | 7.0 | 7.5 | 6.0 | 6.9 | REVISE |
| 2 | 9.0 | 8.5 | 8.5 | 8.0 | 8.0 | 8.0 | 7.0 | 8.35 | REVISE |
| 3 | 9.5 | 9.2 | 9.1 | 9.0 | 9.0 | 9.1 | 8.8 | 9.15 | READY |

## Method Evolution Highlights

1. 从三项并列贡献收敛为一个机制：delta-only pre-tail CRDE 与 matched regret。
2. 删除无后续 pretrained interpreter 的第三层 exchange，以及 Router、alpha/
   beta、late synthesis 和外部 backbone。
3. 把 counterfactual 收紧为同一增强 tensor 上 frozen eval/no-grad、无 head、
   pre-BN、state-clean 的 exact same-fold comparator。
4. 固定 `REGRET_WEIGHT=1.0`，并把同tensor、head hook、frozen buffer/BN状态纳入
   M0 contract。

## Pushback / Drift Log

| Round | Reviewer concern | Response | Outcome |
|---:|---|---|---|
| 1 | 贡献像 V8+exchange+regret 三件套 | V8降为substrate，regret并入CRDE | accepted |
| 2 | 未有结果所以venue risk高 | 保留风险与严格Q1/D1，但区分proposal READY和claim evidence | accepted in round3 |

## Remaining Weaknesses

READY 不代表指标已提高。CRDE 仍可能在 Q1 或 D1 失败；Q1 OOF 使用 V12
fold-specific坐标，只是D1的必要授权，不是V8 all-fit deployment替代证据。

## Raw Reviewer Responses

完整原始响应逐字保存在：

- `refine-logs/v15/round-1-review.md`
- `refine-logs/v15/round-2-review.md`
- `refine-logs/v15/round-3-review.md`

这些文件均以 `<details>` 包含 GPT-5.5 xhigh 的完整 standalone review，未做摘要
替换；本报告不重复嵌入以避免产生第二份可能分叉的原文。

## Next Steps

按 `experiment-plan` 冻结 M0→Q1→conditional D1 顺序，然后用 TDD 实现；任何
门失败立即封存，不访问后续 split。
