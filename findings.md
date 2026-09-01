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
