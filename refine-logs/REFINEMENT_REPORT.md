# V14 Refinement Report

两轮 GPT-5.5 xhigh 评审将 V14 从 7.20/10 REVISE 修正到 9.25/10 READY。
决定性变化是：训练与验收都对齐到 fold-local retrieval risk/AP/margin，
source-only comparator 不接触 held-out fold，并用 fold-bound API 禁止跨 OOF
坐标距离。最终干净方案见 `refine-logs/FINAL_PROPOSAL.md`；V14 只获准进入
train-only qualification，尚无 dev 或 SOTA 结论。
