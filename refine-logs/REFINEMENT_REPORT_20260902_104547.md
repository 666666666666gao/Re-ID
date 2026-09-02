# V14 Refinement Report

- 问题：V8 三专家有 query 级互补性，但 V13 Router 无法稳定学习近均匀、
  fold-dependent 的点式 utility target。
- 初始路线：fold-local retrieval risk + worst-source-fold regret。
- 轮数：2/5。
- 最终评分/判定：9.25/10，READY。

## 方法演进

1. 删除点式 utility-KL，保留 utility 仅作诊断。
2. 把 Q1 主门从 action target 改为 held-out risk gain，并以 AP/margin 防止
   surrogate 改善却损害检索排序。
3. 固定 comparator 改成 source-only minimax fixed slot，完全不读取
   held-out fold。
4. 以 fold-bound API 禁止跨 OOF generator 的 feature/distance 混用。
5. 明确 Router input 为 all-fit，只有 teacher/replay 为 identity-OOF；缩窄
   所有 transfer 与 generalization 措辞。

## 剩余弱点

- V8 branch Oracle 仍低于 65 mAP，V14 即使通过 Router 门也未必达到 dev 门。
- HFER 未在本阶段启用，因此 Q1 不能支持最终“三分支深层协同”主张。
- 单 seed 是用户冻结约束，不能给出跨 seed 方差结论。

## 输出

- 最终方案：`refine-logs/FINAL_PROPOSAL.md`
- 评审摘要：`refine-logs/REVIEW_SUMMARY.md`
- 原始轮次：`refine-logs/v14/round-1-review.md`、
  `refine-logs/v14/round-2-review.md`

下一步：依据 `/experiment-plan` 冻结 M0→Q0→Q1→条件 dev 计划，然后实现。

