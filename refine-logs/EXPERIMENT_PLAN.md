# Experiment Plan

V13 固定顺序：M0 TDD/preflight → M1 Q0 paired actual-path target → 条件式M2 Q1 identity-heldout policy+OOF replay → 条件式M3一次dev。Q0/Q1任何门失败即停止。主结果达到65 mAP并严格胜出后才允许两项deletion checks。完整计划见同目录 `EXPERIMENT_PLAN_20260902_093332.md`。

核心配置：seed42；远端RTX3090；无baseline重跑、多种子、HFER/DINO、参数扫描、主结果前消融或official；alpha固定0.2；Q1 aggregate以identity-cluster paired bootstrap 95% lower bound>0判定。
