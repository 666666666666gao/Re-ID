# V14 Experiment Plan

V14 的唯一主张是：在每个 identity-OOF teacher 坐标系内直接优化
cross-camera retrieval regret，并以最坏 source fold 控制 Router，能否比
source-only fixed policy 更可靠地迁移到 held-out fit fold。

执行顺序严格为：

1. **M0**：public-seam TDD；
2. **Q0**：exact paired cache、optimizer steps=0 的梯度/绑定/SHA 门；
3. **Q1**：唯一 seed42 三折 train-only Router qualification；
4. **D1**：仅 Q1 全过后，一次 final refit + 一次 dev。

Q1 要求每折 risk gain>0、AP/margin 不降，三项 identity-bootstrap 95%
下界>0，并通过质量、missing、SHA、dev0/official0 门。任一失败即封存，
不扫温度/LR/epoch/margin/fold/权重，不 final refit/dev。

D1 只有一个门：fused mAP≥65，且 mAP/Rank-1 严格超过 exact Signal、当前
deployable best 与三分支。此前不做 baseline rerun、多种子、消融或 official
test。完整冻结计划见 `refine-logs/EXPERIMENT_PLAN_20260902_104830.md`。
