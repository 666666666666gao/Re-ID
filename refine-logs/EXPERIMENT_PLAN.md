# TriFusion V3 强基线锚定主实验计划

**问题**：V1 已完整训练 60 epoch，但正式 fused 仅为 59.1478 mAP / 63.2775 Rank-1；三个分支均约 59 mAP，融合未超过最佳分支。V2 同样完整训练 60 epoch，train-only dev 最佳 fused 41.0476 mAP，低于 Transformer 41.3275，说明 5120 维升维和复杂路由没有恢复强基线。
**冻结诊断**：差距的主因不是训练未完成，也不是显存或评估器，而是旧模型没有保留 Signal 类方法的 task-adapted CLIP 直接检索路径，且三个专家高度同质。
**版本**：V3 task-anchor-1；仅 seed 42；仅云端 RTX3090；不复现 baseline；先主实验，过登记目标后才做消融。

## Claim Map

| Claim | Minimum convincing evidence | Anti-claim / gate |
|---|---|---|
| C1：直接 task-adapted CLIP 身份锚点加有界异构残差，能在不破坏强语义主路径的前提下获得互补局部、全局和长程信息 | zero-residual 模式与 anchor 距离矩阵严格相同；每个残差范数不超过对应 anchor 范数乘可学习尺度；dev fused 同时超过 anchor 和最佳分支 | 不能把提升归因于 5120 维：V3 固定为 3072 维，低于 V2 的 5120 维 |
| C2：分阶段双向交换、身份感知跨谱对齐和质量后验联合路由，能把三专家差异转化为融合收益 | 三专家均有有限非零梯度；fused>best branch；跨模态对齐损失有效；路由非恒定且对缺失模态正确屏蔽 | 只增加参数但不产生融合增益视为失败；正式指标未过登记目标不得声称 SOTA |

## 三个主创新点（工作命名）

1. **TACA：Task-Adaptive CLIP Anchor**。RGB、NIR、TIR 共用完整 CLIP ViT-B/16；每个模态的官方 512 维 projected CLS 直接拼接成 1536 维身份锚点，并以 `5e-6` 小学习率任务适配，不再把强主干只当作被后续模块重写的 token 源。
2. **BCER：Bounded Collaborative Expert Residuals**。CNN、Transformer、Mamba 从同一 CLS 保持的局部 token 场出发，分别提取二维高频、全局关系和四向长程信息；三阶段 HFER 双向交换。每个专家残差显式限制为不超过相应 anchor 范数乘 sigmoid 尺度，从结构上抑制随机专家覆盖强预训练语义。
3. **IQR Fusion：Identity-aligned Quality Routing**。以身份感知多正样本跨谱对齐约束 RGB/NIR/TIR anchor，以 `r(1-u)` 后验对三个专家的逐模态残差路由；最终表示为 `[1536-d direct anchor, 1536-d routed residual]`，保留 anchor 与专家贡献而非再次压成单一分支。

上述名称是工程身份；未经新的主源查新不声称“首次”。

## 主实验顺序

| Run ID | Split / mode | Success gate | Status |
|---|---|---|---|
| V2-R003 | 141-fit/30-dev，60 epoch | fused≥65 且 fused>best branch | COMPLETE—FAIL：best e44 fused 41.0476，Transformer 41.3275，official access=0 |
| V3-R000 | synthetic + real CLIP TDD / full regression | anchor、norm bound、三支梯度、builder、runner 全通过 | IN PROGRESS |
| V3-R001 | RGBNT201 train-only B32/K4 capacity | ≤120M 参数；无 OOM/nonfinite/overflow；梯度覆盖通过 | TODO |
| V3-R002 | 固定真实 batch 100-step overfit | 总损失显著下降且全程有限 | TODO |
| V3-R003 | 141-fit/30-dev，seed42，完整 60 epoch | fused mAP≥65；fused>anchor 且 fused>best branch；official access=0 | BLOCKED BY R000–R002 |
| V3-R004 | full171 固定 60 epoch + official exactly once | 预先冻结 checkpoint/选择规则；严格超过 85.3 mAP / 87.9 Rank-1 才支持目标 | BLOCKED BY DEV GATE |

## 固定训练配置

- 位置：仅远端 RTX3090 24GB；Windows/WSL 只用于代码传输和文档。
- 数据：RGBNT201；开发阶段只用 train_171 的 141-fit/30-dev，official test access 必须为 0。
- batch：真实 B32/K4，不用梯度累积替代 batch-hard；eval B64；AMP initial scale 512；activation checkpointing。
- 优化：AdamW；CLIP 与 CLIP 初始化投影 LR `5e-6`；新增专家/路由/分类器 LR `3.5e-4`；60 epoch；warmup 5。
- 检索：raw-before-neck；不 rerank；不 TTA/TTT。
- 资源：capacity 门要求启动前至少 22000 MiB 空闲，但训练实际可使用 3090 的绝大部分显存，不存在 500 MiB 限制。

## 决策边界

- V3-R003 是完整 60 epoch 主方法开发实验，不是短跑；若失败，如实保留并继续只在 train/dev 修结构。
- anchor 指标由冻结 checkpoint 的独立 train-only representation diagnostic 计算；必须与 fused/三分支使用同一 dev 协议。
- 只有 V3-R003 过门才创建新的正式实验身份；正式 official test 对该身份只访问/评估一次。
- 不做多种子、不复现 baseline、不提前做消融。只有正式 fused 同时严格超过 85.3/87.9 后才规划消融。
- Signal 80.3/85.2 是上游日志值，不称本地复现；MDReID 82.0868/85.1675 仅为已完成的评估链校验，不复制其无仓库级许可证代码。

## 当前检查表

- [x] V1/V2 完整训练与失败模式已审计
- [x] V3 两项 claim、三项机制和晋级门已冻结
- [x] TDD 证明 direct anchor、zero-residual 等价、残差范数上界及三支梯度
- [ ] 真实 CLIP builder 与内部全回归完成
- [ ] RTX3090 capacity / overfit 通过
- [ ] V3-R003 完整 60 epoch dev 达到门槛
- [ ] 过 dev 门后冻结并执行一次正式评估
- [ ] 过 85.3/87.9 后才进入消融和论文 SOTA 主张
