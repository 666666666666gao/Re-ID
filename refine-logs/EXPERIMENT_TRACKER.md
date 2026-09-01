# TriFusion V3 Experiment Tracker

| Run ID | Purpose | System | Split | Priority | Status | Evidence / gate |
|---|---|---|---|---|---|---|
| V1-FINAL | 历史正式失败证据 | shared semantic CIRC/URGC | full171→official once | LOCKED | COMPLETE—FAIL | e60 fused 59.1478 mAP / 63.2775 R1；best branch CNN 59.1561；official access/eval 1/1 |
| V2-R003 | 验证 5120 维 V2 是否恢复强语义 | cascade V2 uniform | 141-fit/30-dev | LOCKED | COMPLETE—FAIL | 60 epoch 完整；best e44 fused 41.0476/40.6061，低于 Transformer 41.3275；last fused 40.2379；official access=0 |
| V3-R000 | task-anchor V3 行为合同与回归 | TACA+BCER+IQR | synthetic + real CLIP | MUST | IN PROGRESS | 定向 6 PASS/1 SKIP；真实 CLIP builder 与全回归待最终提交验证 |
| V3-R001 | 真实容量和梯度门 | V3 B32/K4 AMP | RGBNT201 train-only | MUST | TODO | 参数≤120M；GPU free≥22000 MiB；无 OOM/overflow/nonfinite |
| V3-R002 | 学习能力门 | V3 fixed-batch 100 steps | RGBNT201 train-only | MUST | TODO | loss 显著下降；official/dev evaluation=0 |
| V3-R003 | 完整主实验开发选择 | V3 core seed42 | 141-fit/30-dev | MUST | BLOCKED | 60 epoch；fused≥65 且超过 anchor 与最佳分支；official access=0 |
| V3-R004 | 新正式实验 | frozen V3 seed42 | full171→official once | MUST | BLOCKED | 仅 V3-R003 过门；严格 >85.3 mAP / >87.9 R1 后才考虑消融 |

## 当前因果判断

- “未训练完”已排除：V1 与 V2 都到 epoch 60。
- “评估器/数据错误”不是主因：隔离 checkpoint 曾在同协议得到 82.0868 mAP / 85.1675 R1。
- 最大已证实问题是分支同质和融合无增益：V1/V2 fused 均未超过最佳分支。
- V3 的关键修复不是单纯加模块，而是把 task-adapted CLIP projected CLS 设为不可绕过的直接检索锚点，并让三专家只提供显式范数受限的协同残差。
