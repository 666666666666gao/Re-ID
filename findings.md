# Research Findings

## 2026-09-01 — Signal-preserving V6 主训练就绪门

- 单一诊断驱动修正：移除 V5 learned residual scale；把路由后联合残差银行按样本无自由倍率地校准到 exact Signal baseline 能量；用 residual-only ID/triplet 直接训练三个专家，并用 residual-only batch-hard 身份效用监督路由。
- TDD：V5/V6 core+runner 联合 `16 passed, 3 warnings`，warnings 仅为 timm 弃用提示。
- preflight：全 825/825 dev 的 3072D baseline 逐元素相等，指标精确保持 `58.0109/57.4545/69.9394/76.6061`；optimizer0、official0。
- capacity：RTX3090、B32/K4、8 step；96,917,971 参数、5,723,154 可训练参数；218/218 梯度张量；0 overflow；峰值 `3403.44 MiB allocated / 3554 MiB reserved`；Signal SHA 不变。
- overfit：同一真实 B32/K4 批 100 step，loss `4.06445→0.22984`，ratio `0.05655≤0.10`；218/218 梯度、0 overflow、Signal SHA 不变、official0。
- 边界：只证明工程与固定批学习能力。下一步仅允许一次 seed42、60-epoch held-out-dev；仍禁止消融、多种子和 official test。
- 证据：`evidence/trifusion_signal_preserving_v6_{preflight,capacity,overfit}_seed42.json`。

## 2026-09-01 — Signal-preserving V5 60-epoch dev 主门负结果

- 完整性：远端 RTX3090、seed42、B32/K4、141-fit/30-dev、60/60 epoch、5498 optimizer steps、0 AMP overflow；最佳 epoch51 严格重载五路指标逐项一致；Signal 参数 SHA 在训练前、训练后和重载后完全不变；official test access=0。
- 同 checkpoint 结果：baseline `58.0109/57.4545`；fused `58.0168/57.4545`；CNN `58.0181/57.4545`；Transformer `58.0137/57.4545`；Mamba `58.0135/57.4545`（mAP/Rank-1）。
- 主门失败：fused 仅比 baseline 高 `0.00587 mAP`，又比 CNN 低 `0.00130 mAP`，并比 65 mAP dev 门低 `6.98324`；因此不进入 official test、不做消融，也不支持三分支融合增益或 SOTA 主张。
- 只读诊断：三分支参数都有明显更新，但 fused 追加残差范数仅为 baseline 的 `2.747%`；融合距离与 baseline 距离 Pearson 相关为 `1.0`，平均绝对距离变化 `0.000202`，Top-10 邻居重合率 `99.9879%`。三个分支残差两两余弦接近 0，说明专家有差异，但差异几乎没有进入最终检索排序。
- 路由诊断：归一化熵 `0.9600`，仍偏高；三个训练后残差 scale 仅约 `0.106`。当前瓶颈不是专家没有训练，而是路由与小尺度残差共同造成输出几何几乎等同 baseline。
- 独立 result-to-claim：`claim_supported=partial`、置信度 high。只支持“精确保留 Signal 并完成稳定训练”的工程子主张；不支持协同增益、65 mAP dev 晋级、SOTA 或三项创新有效性。
- 下一步：只允许一个 main-only 架构修正，使互补专家信息对检索距离产生实质影响，同时继续保留 exact `baseline_only` 输出；修正后重新通过 TDD/capacity/overfit，再运行一次 seed42 held-out-dev。禁止消融、多种子和 official test。
- 证据：`evidence/trifusion_signal_preserving_v5_dev_terminal_seed42.json` 与 `evidence/trifusion_signal_preserving_v5_diagnostic_seed42.json`。

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
- 约束：不进入消融；不做多种子；不得再次使用本次官方测试做选模或调参。此前“不复现 baseline”的约束已被用户 2026-09-01 19:00 的最新指令覆盖；现在只允许先做远端 Signal baseline 保底，不允许 baseline 网格扫描或官方 test 选点。
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

## 2026-09-01 — TriFusion V4 60-epoch dev 主门负结果

- 完整性：远端 RTX3090、seed42、B32/K4、141-fit/30-dev、60/60 epoch 完成；状态 `PASS/complete`；60 次 dev 评估；无 fatal/nonfinite；official test access=0。
- 最佳 checkpoint：epoch27，SHA-256 `47fea7f42a5673e42deb1d67540cca6338af62b028be4d69daedfe309de1e852`。
- 最佳 checkpoint 原始结果：fused `43.4031 mAP / 42.7879 Rank-1`；CNN `40.9147/39.5152`；Transformer `41.6819/40.1212`；Mamba `44.0659/43.5152`。
- 主门判定：fused 比 Mamba 低 `0.6628 mAP / 0.7273 Rank-1`，比 65 mAP dev 门低 `21.5969`；因此 `claim_supported=no`，official test 与消融继续封闭。
- 相对 V3：V4 fused mAP 只提高约 `0.5053`；这证明等能量残差银行没有把结构差异转化为足够的融合增益。
- 过拟合信号：epoch60 fused 回落到 `40.1199/40.0000`，Mamba 为 `41.0375/42.7879`。这是完成后的早峰回落，不是“还没训练完”。
- baseline 边界：V4 没有测得同 checkpoint 的完整 Signal baseline-only 指标。其 1536D anchor 仅是三模态 projected-CLS，缺少 Signal 推理的 1536D SIM 交互特征和 camera SIE；不得把它称为 Signal baseline，也不得把 43.4 dev 与上游 official-test `80.3/85.2` 直接相减。
- 最新路线：先建立完整、独立可检索的 Signal 3072D baseline-only 路径；分阶段冻结 baseline，避免专家梯度破坏；同 checkpoint 同 dev 协议同时评估 baseline-only 与 fused，只有 fused 不低于 baseline 且通过冻结主门才晋级，否则拒绝 fused 且不主张融合增益。不实现额外运行时 fallback。
- V4-specific integrity：当前只有真实 GT、`run_summary=PASS` 和完整哈希链，尚无独立 V4 integrity audit，故完整性结论标为 provisional；负结果 verdict 置信度 high，baseline 缺失是主要根因的因果判断仅为 medium。
- 证据：`evidence/trifusion_task_anchor_v4_dev_terminal_seed42.json`；远端原始目录 `/root/autodl-tmp/trifusion-v2/artifacts/trifusion_task_anchor_v4_core_dev_seed42_3fbedbb`。
