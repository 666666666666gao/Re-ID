# V16 SATR Refinement Report

**Date**：2026-09-02  
**Rounds**：3/5  
**Final Score**：9.10/10  
**Final Verdict**：READY

## Score Evolution

| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8.5 | 6.5 | 7.0 | 7.5 | 7.5 | 5.5 | 6.5 | 7.10 | REVISE |
| 2 | 9.0 | 8.0 | 8.1 | 8.2 | 8.3 | 6.8 | 7.2 | 8.13 | REVISE |
| 3 | 9.4 | 9.3 | 9.0 | 8.8 | 9.0 | 9.3 | 8.7 | 9.10 | READY |

## Final Thesis

- exact Signal 为每个 query 定义同一 batch-hard cross-camera relation；
- 只有另外两支都胜过 Signal 与 receiver 时，receiver 才补足共同 margin 下界；
- 不匹配 feature/logit/token，不增加推理 Router/exchange；
- matched no-SATR 双端点把继续训练与 SATR 效果分开。

## Method Evolution Highlights

1. 从“泛关系蒸馏”收敛成 failure-driven 的 two-peer/Signal intersection；
2. 从不公平 frozen comparator 改成完全 matched 的双端点；
3. activity gate 与结果 gate 分离，避免成功修复后 coverage 下降被误判。

## Pushback / Drift Log

- 拒绝 Router、EMA、额外 teacher/backbone、DINO/VLM；这些不解决 V15 已观察到
  的 hidden-direction fold instability，并会增加贡献扩张。
- 接受公平 comparator 增加的计算，因为它是机制归因的最低成本。

## Remaining Weaknesses

SATR 的新颖性边界窄，且 two-peer intersection 可能只覆盖少量 query。READY
只表示可进入实验；若 Q1 未通过，不得用工程可运行替代科学结论。

## Raw Reviewer Responses

- `refine-logs/v16/round-1-review.md`
- `refine-logs/v16/round-2-review.md`
- `refine-logs/v16/round-3-review.md`

## Next Step

进入 `/experiment-plan`，随后按 TDD 实现 M0 和 Q1；D1 仍为条件执行。

