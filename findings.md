# Research Findings

## 2026-09-02 — V11-Q0 residual-only OOF 仍被 all-fit Signal 场饱和

- 三个专家 checkpoint 的 adapter 训练身份严格隔离，距离也只在各 held-out fold 内计算；但它们共享的 frozen Signal token field 已见全部141个fit身份，因此完整表示路径并不identity-unseen。
- 571 query 聚合 mAP/Rank-1：CNN residual=`98.5115/98.4238`，Transformer=`100/100`，Mamba=`99.9416/100`，residual bank=`100/100`，DINO=`14.1323/9.4571`，fixed concat=`95.8582/96.4974`。
- concat 比最强fixed source低 `4.1418 mAP`；Oracle gain=0；unique AP wins residual-bank/DINO=`570/0`；non-saturation门失败。
- 独立 result-to-claim=`no/high`；独立审计=`WARN/warn/FAIL_TO_QUALIFY`。100 mAP不是self-normalization或假GT，而是all-fit Signal场泄漏/饱和证据。
- 封存V11：不实现Q1/Q2、不训练、不访问dev/official、不做消融或DINO事后扫描。后继必须使完整测量路径对held-out身份未见，且不能复跑baseline。

## 2026-09-02 — V10-Q0 frozen DINOv2 资格门失败

- 范围：仅 141-fit 中 21 个跨摄像头身份、571 query；optimizer0、training=false、dev0、official0。DINOv2 ViT-B/14 只删除预训练专用 `mask_token` 后 strict load，输入固定252×126、token163×768，Phase-A/Router/DINO state 均不变。
- 冻结指标 mAP/Rank-1：Phase-B=`100/100`；DINOv2=`7.6284/6.1296`；固定等块拼接=`92.2120/95.9720`。该 fit 协议对已在 fit 身份上训练的 Phase-B 完全饱和，不能作为 dev 泛化指标。
- Phase-B/DINO hard Oracle 仍为100/100，Oracle gain=0；unique AP wins=`571/0`；concat 比 Phase-B 低 `7.7880 mAP`。concat≥+1、Oracle≥+2和双源独有胜例全部失败。
- 独立 result-to-claim=`no/high`；只支持当前冻结 DINO 特征未提供可用互补、固定拼接有害，不支持“DINOv2 普遍不适合 RGBNT ReID”。
- 独立审计=`WARN/warn/FAIL_TO_QUALIFY—STOP_V10_Q0`。GT/协议、归一化、实际路径、scope和评价类型PASS；WARN 仅因审计时JSON未追踪、二进制权重远端保存。
- V10 封存：不实现Q1，不训练、不访问dev，不扫描模态子集、分辨率、block、token pooling、权重或训练头。未来若再使用DINO，必须作为新预注册假设并设计非饱和、身份隔离的train-only门。
- 证据：`evidence/trifusion_v10_dinov2_fit_qualification_seed42.json`；报告：`results/TRIFUSION_RGBNT201_V10_DINOV2_FIT_QUALIFICATION_2026-09-02.md`。

## 2026-09-02 — V9 orthogonal triadic relay 主门负结果

- 工程合同成立：exact Signal 与完整 V8 Phase-B embedding 是逐元素前缀；Phase-A/Router state SHA 在训练和评估前后不变；两轮 peer relay 实际执行，最大绝对 relay/receiver cosine=`1.01e-7`。
- 远端 RTX3090、seed42、真实 B64/K8 完成 60/60 epoch、2,520 optimizer steps、0 overflow；训练 loss=`3.45323→0.62362`，训练期间 dev0/official0。final checkpoint 只进行一次冻结 dev，optimizer0、official0。
- 同 checkpoint 指标：baseline `58.0109/57.4545`；Phase-B `58.4050/59.3939`；V9 fused `56.5339/57.2121`；CNN/Transformer/Mamba=`55.8825/51.3416/54.6342 mAP`。
- 主门 FAIL：fused 比 baseline 低 `1.4770 mAP`，比 Phase-B 低 `1.8711 mAP`，并比 65 门低 `8.4661 mAP`。超过三个已经退化的 V9 专家不能作为协同增益证据。
- beta mean/min/max=`0.498794/0.462330/0.499998`，接近上限0.5；这只是观察，未做消融，不能把失败因果归结为 beta 饱和。
- 独立 result-to-claim=`no/high`；独立审计=`WARN / warn / FAIL_TO_PROMOTE`。GT、评价归一化、实际路径和评价类型均 PASS；WARN 来自远端 checkpoint 包装、审计时 JSON 未追踪/文档滞后，以及 config 门字段不是运行时单一数据源，不改变负结果。
- V9 封存：不做 official、消融、多种子或 beta/epoch/LR/residual/checkpoint 扫描。任何新结构访问 dev 前，必须先在 fit-only 身份隔离折上证明新增表示有正检索效用，并能从训练侧抑制有害追加。
- 证据：`evidence/trifusion_v9_{preflight,capacity,overfit,train,dev}_seed42.json`；报告：`results/TRIFUSION_RGBNT201_V9_DEV_SEED42_2026-09-02.md`；审计：`EXPERIMENT_AUDIT_V9.md`。

## 2026-09-02 — V8 OOF-margin Router Phase-B 正向但未晋级

- 连续 OOF identity-margin target 覆盖 571 个 fit-only query；CNN/Transformer/Mamba 独有 slot winner=`38/350/183`，RGB/NI/TI=`215/59/297`，slot Oracle 比最佳固定 slot 高 `0.164303`。Oracle 使用真实身份标签，只是训练域诊断上限。
- 三折身份隔离 Router 加 all-fit refit 共执行 400 个 Router optimizer step，Phase-A 专家和 Signal 全程冻结且 state SHA 不变。OOF learned expected margin=`0.102034`，仅比 fixed=`0.101720` 高 `0.000314`；Top-slot accuracy=`17.8634%`，仅比 majority=`17.6883%` 高 `0.1751` 个百分点，因此只能称为很弱的泛化证据。
- 质量语义门通过：缺失模态最大权重严格为 0；单独模糊 RGB/NI/TI 后，对应模态平均质量分别从 `0.306154/0.298051/0.395795` 降到 `0.117502/0.102016/0.166562`。
- 唯一一次冻结 held-out-dev 评估：baseline/fused mAP=`58.0109/58.4050`，CNN/Transformer/Mamba=`57.6071/56.3031/56.6260`；fused 比 baseline 高 `0.3941 mAP / 1.9394 Rank-1`，并严格超过三个固定专家。
- 主门仍 FAIL：fused 比 65 mAP 低 `6.5950`，`promotion_gate=false`、`next_phase_authorized=false`。dev 评估 optimizer0、official access0，checkpoint/state SHA 在评估前后不变。
- 独立 result-to-claim=`partial/medium`：只支持当前完整 Phase-B 配置在单 seed、固定协议上的小幅部署增益；不能把增益单独归因为 learned Router，也不支持充分互补、HFER、official、SOTA 或跨数据集泛化。
- 当前 V8 Phase-B 封存为“正向但未晋级”。不启用 HFER，不做消融、多种子、official test 或 Router/alpha/epoch/LR 扫描。若继续冲击 65，必须预注册新的表示级主假设并重新通过 train-only 门。
- 证据：`evidence/trifusion_v8_oof_router_margin_targets_seed42.json`、`evidence/trifusion_v8_oof_margin_router_phase_b_seed42.json`、`evidence/trifusion_v8_oof_margin_router_dev_seed42.json`；报告：`results/TRIFUSION_RGBNT201_V8_OOF_MARGIN_ROUTER_PHASE_B_2026-09-02.md`。

## 2026-09-02 — V8 pretrained-tail Phase-A 专家互补门通过

- 结构：从冻结 Signal/CLIP 第 8 block 分叉，三路分别共享冻结的 CLIP tail 9/10/11，并加入 CNN 横向局部细节、Transformer CLS 全局关系和 Mamba 空间/跨模态长程残差；Router 与 HFER 在 Phase-A 均关闭。
- 工程门：exact Signal preflight、真实 B64/K8 capacity、100-step overfit 全部 PASS；203/203 可训练张量有梯度，overfit 超额损失 ratio=`0.000534≤0.1`，Signal state 不变，峰值 reserved 约 `6.0 GiB`。
- 训练边界：seed42、20 epoch、840 optimizer steps、最终 epoch 才进行一次 held-out-dev 评估，训练中 dev evaluations=0、overflow=0、official access=0；耗时 `933.24s`，峰值 reserved=`6214 MiB`。
- 可部署固定输出仍弱：baseline/fused mAP=`58.0109/58.0972`；CNN/Transformer/Mamba branch=`57.6071/56.3031/56.6277`。fused 只比 baseline 高 `0.0863 mAP` 且 Rank-1 下降，不能作为融合成功结论。
- branch ground-truth Oracle=`64.7850 mAP / 65.9394 Rank-1`，比最强固定 baseline 高 `6.7741 mAP`；CNN/Transformer/Mamba 独有 AP 胜例=`201/170/138`，leave-one-out 边际=`+1.2043/+1.9592/+0.8435 mAP`。
- residual-only Oracle=`63.4813/66.9091`，比最强 residual 高 `9.6153 mAP`；三专家独有 AP 胜例=`257/232/199`，leave-one-out 边际=`+3.1128/+4.9698/+2.6370 mAP`。这证明表征互补存在，但 Oracle 使用真实标签、不是部署结果。
- 独立 result-to-claim=`partial/medium`：只支持“值得开展一次冻结专家、fit-only Router 可行性阶段”；不支持 learned routing、HFER、65 mAP、official 或 SOTA。branch Oracle 本身仍比 65 低 `0.2150 mAP`，单纯硬选分支不够。
- 下一步先完成 V8 专属完整性审计，再做 fit-only 层级 Router 与质量干预门；Router 未证明泛化前不启用 HFER，不做消融、多种子或 official test。
- 证据：`evidence/trifusion_signal_preserving_v8_expert_formation_{preflight,capacity,overfit,probe}_seed42.json`。

## 2026-09-02 — V8 frozen-router 表征上限探针未晋级

- 目的：在不更新模型的条件下，检验“冻结 V7 三专家、只重训身份隔离效用 Router 并恢复等能量 residual”是否值得进入 V8 主训练。source checkpoint 为 V7 epoch1 `8bcdf358...09a2b`；FP32 两次重放核心 JSON 完全一致；optimizer0、official0。
- 教师数据边界：141-fit 中只有 21 个身份具有跨摄像头正样本，共 571 个合格 query；30-dev 为 825 个 query。fit 的 residual-only 最佳专家标签 100% 为 CNN，因此检索级教师在训练域没有可学习的专家类别变化。
- 身份隔离迁移：dev 真实赢家分布为 CNN/Transformer/Mamba `55.27%/17.45%/27.27%`。fit 教师在 dev 上恒选 CNN，准确率 `55.27%`，等于多数类先验；V7 当前 Router 预测分布为 `0%/4.61%/95.39%`，准确率仅 `27.39%`。
- 融合几何：把 residual bank 从 V7 的约 0.2 能量恢复为与 Signal 等能量后，均匀融合和 fit 教师融合都达到 `59.6188 mAP / 59.1515 Rank-1`；比 baseline 高 `1.6079 mAP`，比最强固定 residual-CNN 高 `0.4871`，证明低 residual 能量确实压制融合。
- 晋级门失败：`59.6188` 仍比 65 mAP 低 `5.3812`，效用教师也没有超过多数类先验。结合 residual-only GT Oracle `62.7435<65`，仅冻结专家并更换 Router/能量不能完成主目标，不启动这一 V8。
- 下一步必须改变专家表征能力和训练分工，同时保留 exact Signal 输出；不得把该探针称为主方法结果、消融、部署 Router 或 SOTA。
- 证据：`evidence/trifusion_v8_frozen_router_probe_seed42.json`、`results/TRIFUSION_RGBNT201_V8_FROZEN_ROUTER_PROBE_2026-09-02.md`。

## 2026-09-02 — Signal-preserving V7 60-epoch dev 主门负结果

- 完整性：远端 RTX3090、seed42、B64/K8、固定 141-fit/30-dev、60/60 epoch、2,520 optimizer steps、0 overflow；严格重载五路指标逐项一致；Signal state SHA 训练前后及重载后不变；official test access=0。
- 同 checkpoint 指标：baseline `58.0109/57.4545`；fused `58.3293/57.9394`；CNN `58.2773/57.4545`；Transformer `58.3028/58.0606`；Mamba `58.3476/57.8182`（mAP/Rank-1）。
- 主门失败：fused 比 baseline 高 `0.3184 mAP`，但比 Mamba 低 `0.0183`，并比 65 mAP 门低 `6.6707`。因此 `claim_supported=no`，不进入 official test、不做消融或多种子，也不支持协同优越性或 SOTA。
- 不是训练不足：按预注册 fused dev mAP 选中的最佳 checkpoint 是 router-warmup epoch1；joint 阶段最佳 epoch11 仅 `57.9804 mAP`，epoch60 为 `57.7550`。联合训练持续降低 held-out 检索性能。
- 只读诊断：epoch1 的联合 Router 归一化熵 `0.99791`、模态熵 `0.99994`，alpha `0.198947±0.0000015`，预测槽位与逐槽身份效用目标的 Top-1 一致率只有 `14.0625%`。fused/baseline 距离相关仍为 `0.999786`，Top-10 overlap `99.6364%`。
- 三专家仍有真实差异：residual-only CNN/Transformer/Mamba 固定 mAP 为 `59.1317/54.8594/57.8991`，ground-truth Oracle 为 `62.7435`，比最强 residual 分支高 `3.6118`；每个专家的 leave-one-out 边际贡献均为正。Oracle 不是部署结果。
- 因果边界：当前证据支持“残差专家存在查询级互补，但样本级 Router 未学会选择，且 joint 优化破坏已有 residual 检索能力”。不支持把失败归因于 CNN/Transformer/Mamba 天然不兼容，也不支持继续原样训练。
- 独立 result-to-claim：`claim_supported=no`、置信度 high；只允许报告 exact Signal preservation 和 `+0.3184 mAP` 的窄 baseline 增益。V7-specific independent integrity audit 尚缺，因此完整性标记仍为 warn/provisional。
- 证据：`evidence/trifusion_signal_preserving_v7_dev_terminal_seed42.json`、`evidence/trifusion_signal_preserving_v7_diagnostic_seed42.json`、`results/TRIFUSION_RGBNT201_V7_DEV_SEED42_2026-09-02.md`。

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

## 2026-09-01 — Signal-preserving V6 60-epoch dev 主门负结果

- 完整性：远端 RTX3090、seed42、B32/K4、固定 141-fit/30-dev、60/60 epoch 完成；5,498 optimizer steps、0 overflow、严格重载逐项一致、Signal state SHA 不变、official access=0。V6-specific independent integrity audit 尚未完成，因此完整性为 provisional。
- 最佳 checkpoint：按冻结的 fused dev mAP 选择 epoch8，SHA-256 `32bba88cc0204cec6b563ce0a8c6239c828c46eec50647d357b2e9f30031ee2e`。
- 五路 mAP：baseline `58.0109`；fused `58.7321`；CNN `59.1022`；Transformer `57.7962`；Mamba `58.7298`。
- 主门：fused 比 baseline 高 `0.7212 mAP`，但比 CNN 低 `0.3701`，且比 65 mAP 低 `6.2679`。`claim_supported=no`，official test、消融和多种子继续封闭。
- V5 瓶颈已解决一半：残差/baseline 范数比从 `0.02747` 提升为精确 `1.0`；fused/baseline 距离相关从 `1.0` 降为 `0.96875`，Top-10 overlap 从 `99.9879%` 降为 `95.3939%`，说明 V6 确实改变检索几何。
- 当前首要瓶颈是路由对齐：最强 CNN residual-only mAP 为 `56.9267`，但只获约 `0.228–0.245` 权重；较弱 Transformer/Mamba 获得更高权重。路由熵 `0.97435`、std `0.00843`，整体接近静态。
- 次要瓶颈是泛化：epoch8 达到最佳后，训练损失继续从 `0.3667` 降至 epoch60 的 `0.02797`，fused dev 却回落到 `56.5679`。训练已经完成，不能解释为“没训练完”。
- 下一步：只允许一个 V7 main-only 结构修正，使路由目标表达各专家相对 exact baseline 的边际身份收益；保留 exact Signal、三完整专家、两次 HFER、三次可靠性刷新和无自由倍率残差银行。不扫 epoch、batch、学习率、温度或残差倍率。
- 证据：`evidence/trifusion_signal_preserving_v6_dev_terminal_seed42.json`、`evidence/trifusion_signal_preserving_v6_diagnostic_seed42.json`、`results/TRIFUSION_RGBNT201_V6_DEV_SEED42_2026-09-01.md`。
