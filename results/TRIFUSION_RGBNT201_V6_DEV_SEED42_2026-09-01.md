# TriFusion Signal-preserving V6 RGBNT201 dev 结果（seed 42）

## 结论

V6 在远端 RTX3090 上完成唯一一次 seed42、60-epoch、141-fit/30-dev 主实验，并严格重载按 fused dev mAP 选择的 epoch8 checkpoint。工程运行完整，Signal 状态逐张量不变，official test access=0；但主方法门失败，`claim_supported=no`。

V6 fused 比 exact Signal baseline 高 `0.7212 mAP`，证明等能量残差银行确实改变了检索几何；但它比同 checkpoint 最强 CNN 分支低 `0.3701 mAP`，且比冻结的 65 mAP dev 门低 `6.2679`。因此当前结果只支持“精确保留 Signal 且在 held-out dev 上小幅改善 baseline”，不支持“三专家融合优于全部专家”、正式评估资格或 SOTA 主张。

## 固定协议与完整性

- 设备：远端单张 NVIDIA RTX3090 24 GB；本地 Windows/WSL 未运行训练或评估。
- 数据：RGBNT201 固定 141-fit/30-dev，825 query / 825 gallery；未访问 official test。
- 训练：seed42，B32/K4，60/60 epoch，5,498 optimizer steps，0 AMP overflow。
- 资源：峰值 allocated `3663.23 MiB`，峰值 reserved `6084 MiB`，耗时 `2319.35 s`。
- 选点：只按 fused dev mAP 选择 epoch8；最终五路指标均由该 checkpoint 严格重载复评，逐项 parity=true。
- checkpoint SHA-256：`32bba88cc0204cec6b563ce0a8c6239c828c46eec50647d357b2e9f30031ee2e`。
- Signal state：训练前、训练后和重载后均为 `97234c510f993dd2936986b9765d774b12674bf9b08dff933ce840d2c6c45a92`。
- 独立 V6-specific integrity audit 尚未完成，故完整性标签为 `provisional`；这不改变主门失败的数值事实。

## 最佳 checkpoint 五路指标

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| exact Signal baseline-only | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| fused | 58.7321 | 57.5758 | 69.2121 | 76.7273 |
| CNN | **59.1022** | **59.6364** | **70.3030** | 76.1212 |
| Transformer | 57.7962 | 57.4545 | 68.3636 | 76.0000 |
| Mamba | 58.7298 | 57.6970 | 69.5758 | 76.3636 |

主门分解：

- fused − baseline：`+0.7212 mAP`；
- fused − CNN：`−0.3701 mAP`；
- fused − Transformer：`+0.9359 mAP`；
- fused − Mamba：`+0.0023 mAP`；
- fused − 65 mAP：`−6.2679`。

V6 严格高于 baseline、Transformer 和 Mamba，但未高于 CNN，也未达到 65 mAP，因此 dev gate=false。

## 训练曲线解释

| Epoch | Train loss | Fused mAP | 说明 |
|---:|---:|---:|---|
| 1 | 3.3475 | 56.2684 | 初始阶段 |
| 8 | 0.3667 | **58.7321** | 冻结规则选出的最佳点 |
| 20 | 0.09138 | 56.2539 | dev 已明显回落 |
| 40 | 0.08870 | 57.0742 | 未恢复最佳点 |
| 60 | 0.02797 | 56.5679 | 训练完成但泛化低于早期峰值 |

这不是“没训练完”。60/60 epoch 已完成，训练损失继续下降而 dev 早期见顶后回落，支持过拟合或身份外泛化不足的解释。

## 冻结 checkpoint 只读诊断

诊断处理全部 825 个 dev 样本，`training_executed=false`、`optimizer_steps=0`、official access=0。

- CNN、Transformer、Mamba 和 fused 的追加残差/baseline 样本级范数比均精确为 `1.0`，V5 的能量压制已被解除。
- fused 相对 baseline 的距离 Pearson 相关从 V5 的 `1.0` 降至 `0.96875`，平均绝对距离变化为 `0.04861`，Top-10 邻居重合率从 `99.9879%` 降至 `95.3939%`。V6 已实质改变检索排序。
- 三专家残差两两余弦接近 0：CNN–Mamba `−0.0011`、CNN–Transformer `0.0299`、Transformer–Mamba `0.0327`，说明残差保持异构性。
- residual-only mAP：CNN `56.9267`、Mamba `56.2747`、Transformer `53.6715`。
- 路由归一化熵均值 `0.97435`、标准差 `0.00843`，仍接近静态分配。最强 CNN 残差只获得约 `0.228–0.245` 权重；较弱 Transformer 为 `0.325–0.367`，Mamba 为 `0.405–0.430`。
- 所有模块参数组均有非零更新；当前瓶颈不是分支未训练，而是路由没有把最终权重与专家的实际增量检索价值对齐。

## Observation → Interpretation → Implication → Next step

1. **Observation**：fused 比 baseline 高 `0.7212 mAP`，检索距离相关降至 `0.96875`。
   **Interpretation**：等能量激活已解决 V5 “残差几乎不起作用”的问题。
   **Implication**：不得回退到 V5 的 learned scale，也不需要扫描残差倍率。
   **Next step**：保留 exact Signal 前缀、三完整专家、两次 HFER、三次可靠性刷新和无自由倍率残差银行。

2. **Observation**：CNN `59.1022` 高于 fused `58.7321`，但 CNN 获得最低路由权重。
   **Interpretation**：当前 residual-only identity-gap 目标与最终 `[baseline,residual]` 表示的边际检索收益不一致。
   **Implication**：现有质量/效用路由没有把专家互补性转化为最优融合。
   **Next step**：下一主版本只修正路由对齐，使监督直接表达专家相对 exact baseline 的边际身份收益；不做超参网格、消融或多种子。

3. **Observation**：最佳点在 epoch8，后续训练损失下降而 dev 回落。
   **Interpretation**：身份外泛化是次级瓶颈。
   **Implication**：不能把低指标归因于训练未结束。
   **Next step**：先完成路由结构修正；继续沿用冻结选点规则，不根据本次曲线热改 epoch、batch 或学习率。

## Result-to-claim 判定

- `claim_supported: no`
- `confidence: high`
- `integrity_status: provisional`
- 可支持：exact Signal preservation；严格重载；official access=0；V6 fused 在固定 dev 上比 baseline 高 `0.7212 mAP`。
- 不可支持：fused 优于所有专家、达到 65 mAP、正式 official-test 资格、三项创新已获性能验证或 SOTA。

下一步只允许一个诊断驱动的 V7 main-only seed42 train/dev 运行；在其通过同一 dev 门前，official test、消融和多种子仍然封闭。

## 证据路径

- `evidence/trifusion_signal_preserving_v6_dev_terminal_seed42.json`
- `evidence/trifusion_signal_preserving_v6_diagnostic_seed42.json`
- 远端完整目录：`/root/autodl-tmp/trifusion-v2/artifacts/trifusion_signal_preserving_v6_dev_seed42_e4a6bff`
