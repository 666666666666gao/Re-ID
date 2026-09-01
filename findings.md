# Research Findings

## 2026-09-01 — TriFusion RGBNT201 seed-42 主结果

- 测试内容：共享 CLIP 语义主干 + CNN/Transformer/Mamba 三专家 + HFER + CIRC + URGC；RGBNT201 `postfreeze-final`；epoch 60；官方测试一次。
- 正式结果：fused `59.1478 mAP / 63.2775 Rank-1`；CNN `59.1561 / 63.7560`；Transformer `59.1219 / 62.6794`；Mamba `58.8748 / 62.4402`。
- 目标差距：相对登记目标 `85.3/87.9`，fused 低 `26.1522 mAP / 24.6225 Rank-1`。
- 判定：`claim_supported = no`，高置信度。不能声称达到目标、SOTA 或融合优于分支。
- 失败信号：CNN 略高于 fused；路由平均概率在条件/专家/模态间几乎固定为 `0.24997`，九路贡献近似均匀平均。三个最终融合投影两两余弦相似度均高于 `0.99992`，四路结果过于接近，构成专家同质化证据。
- 主要结构原因：九个“专家×模态”512 维贡献被直接加权求和压成一个 512 维向量；共享 CLIP 的独立 CLS 输出没有原样进入检索头，而是与 patch 混合后取均值；私有多样性和同伴教学损失均关闭。
- 收敛判断：epoch 60 的 fused ID/triplet loss 已降至 `0.01823/0.00562`，但 test mAP 只有 `59.1478`，说明是泛化失败而非单纯没训练完。
- 低场景证据：训练目标路由校准中 `modality_missing` 最差（Brier `0.22338`、ECE `0.07178`）；未对官方 test 做分场景重复评估，因此不能把它写成该场景的 ReID mAP。
- 设计风险：HFER 第二次交换仍使用 stage-1 质量后验，最终融合前才刷新；Mamba 当前主要做模态内扫描，跨模态传播主要来自通用 HFER。
- 约束：不进入消融；不做多种子；不复现 baseline；不得再次使用本次官方测试做选模或调参。
- 后续：先做 train/dev-only 的主方法失败分析，再设计并预注册新的主版本。需要身份留出的路由校准证据后，才能提出泛化校准主张。
- 完整结果：`results/TRIFUSION_RGBNT201_FINAL_SEED42_2026-09-01.md`。

## 工程事件

- 原正式启动在一次官方评估后，因 `build_rgbnt201_record_eval_loader` 未导入而在训练集路由审计阶段失败。
- `repair-0001` 完成路由审计但因未复用定向授权上下文而在汇总资格检查失败，已事务回滚。
- `repair-0002` 复用冻结定向授权，只运行训练集路由审计；`optimizer_steps=0`、`training_reexecuted=false`、`official_test_reexecuted=false`，完成回执 PASS。

## 2026-09-01 — TriFusion V3 task-anchor dev 主门负结果

- 完整性：远端 RTX3090、seed42、B32/K4、141-fit/30-dev、60/60 epoch 完成；94,757,973 参数；无 OOM、fatal 或 nonfinite；60 次 dev 评估；official test access=0。
- 最佳结果：epoch14 fused `42.8978 mAP / 43.8788 Rank-1`；CNN `42.8402/44.0000`；Transformer `43.0168/44.0000`；Mamba `42.9259/43.8788`。fused 比 65 mAP dev 门低 `22.1022`，并低于 Transformer `0.1190 mAP`。
- 末轮结果：epoch60 fused `37.9848/36.8485`；训练 triplet 已接近 0，但 dev 从早期峰值下降，支持身份外过拟合风险。
- 冻结最佳分解：anchor `42.4787/43.8788`；routed residual 单独 `42.8225/44.8485`；fused `42.8978/43.8788`。诊断与登记 fused/branch 指标逐项 delta=0。
- 已证实结构瓶颈：三个残差两两余弦为 `0.5462–0.6014`，说明专家并未完全塌缩；但 learned scales 和实际 expert/anchor norm ratio 全部饱和在 `0.2529–0.2567`，routed residual/anchor norm ratio 仅 `0.2124–0.2187`，约对应最终拼接距离中 `4.3%–4.6%` 的残差平方能量。路由归一化熵 `0.99977–0.99991`，权重近似均匀三分之一；fused/branch cosine `0.9909–0.9922`。
- 判定：`claim_supported=no`，独立复核置信度 high。V3 只支持“残差学到身份信息但被融合机制压制”的诊断，不支持“三专家协同增益”、dev 晋级或 SOTA 主张。
- 下一步：只允许一个 V4 主方法结构修正——非破坏式保留三个专家残差块，以无自由倍率的等能量校准让残差银行与 anchor 对检索距离贡献可比，并用训练批次身份效用监督路由。保持同一 dev 门，不做 baseline、多种子、消融或 official test。
- 证据：`evidence/trifusion_task_anchor_v3_diagnostic_seed42_f32990b.json`，SHA-256 `c30e11e6471325f3c811e967daa6f5cb296d87d7c9df5809096c5f94a4e779fe`。

## 2026-09-01 — TriFusion V4 主训练就绪门

- 单一修正已实现：保留 `[CNN, Transformer, Mamba] × [RGB, NI, TI]` 九个独立残差块，不沿专家维求和；整个 4608 维残差银行无自由倍率地归一到 1536 维 direct CLIP anchor 的样本级 L2 能量；最终 fused 为 6144 维。
- 路由监督：用训练批次内 detached 的逐样本 batch-hard 身份间隔形成三专家效用目标，并通过 `peer_logits` 槽反传到质量路由；不读取 dev/test 标签。
- TDD：V4 专项 6/6、相邻模块 23/23、排除四个缺失外部 baseline 仓库的内部全回归 146 passed / 7 skipped。
- RTX3090 容量门：B32/K4、AMP scale256、8 步；95,197,266 参数；峰值 6043.58 MiB allocated / 6548 MiB reserved；366/366 可训练参数张量梯度覆盖；0 overflow；official access=0。
- 固定批门：100 步总损失 `14.91096→0.99563`，ratio `0.0667716≤0.10`；0 overflow；official access=0。
- 边界：这些只证明工程和学习能力就绪，不证明开发集增益、SOTA 或论文主张。下一步仅运行 seed42 的完整 60-epoch held-out dev 主实验。
- 证据：`evidence/trifusion_task_anchor_v4_readiness_seed42.json`。
