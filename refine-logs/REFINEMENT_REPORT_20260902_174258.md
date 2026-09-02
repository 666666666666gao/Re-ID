# V17 DTRED Refinement Report

**Date**：2026-09-02

**Rounds**：3 / 5

**Final score**：9.03 / 10

**Final verdict**：READY

## Final Thesis

将 CNN、Transformer、Mamba 三种冻结专家在 source-training pairs 上形成的
label-conditioned 最优关系，作为正相似度下界与负相似度上界，通过单向、正负
平衡的 envelope constraint 蒸馏到三个 corrected receivers 和一个普通
no-reranking fused embedding。

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.0 | 6.8 | 6.9 | 8.0 | 6.8 | 8.5 | 6.0 | 7.4 | REVISE |
| 2 | 9.5 | 8.5 | 8.2 | 8.3 | 7.4 | 9.0 | 6.8 | 8.3 | REVISE |
| 3 | 9.8 | 9.2 | 9.0 | 8.8 | 8.2 | 9.5 | 7.8 | 9.03 | READY |

## Method Evolution Highlights

1. 精确target回归改成one-sided envelope，student超越teacher时不受罚。
2. 正负pair各占一半；fused与三branch总权各占一半，loss scale完全冻结。
3. Q1 teacher只使用source-fold identities；teacher来源、violation、branch diversity
   与unique AP wins进入同run receipt。
4. test-time graph/rank fusion被明确排除，部署仍是single-sample embedding。

## Review Record

- Round 1 raw：`refine-logs/v17/round-1-review.md`
- Round 2 raw：`refine-logs/v17/round-2-review.md`
- Round 3 raw：`refine-logs/v17/round-3-review.md`
- Final proposal：`refine-logs/FINAL_PROPOSAL.md`

## Remaining Weaknesses

V8 query-level branch Oracle为64.785 mAP，DTRED必须学到per-pair synthesis而不是
仅模仿一个branch；这是高风险经验假设。Q1/D1失败时必须封存，不能扫描参数。

## Next Step

进入 `/experiment-plan`，随后TDD实现M0和identity-OOF Q1；Q1全门通过才运行D1。
