# Signal-preserving V5 RGBNT201 dev 主结果（seed 42）

日期：2026-09-01

硬件：NVIDIA RTX 3090 24 GiB

协议：固定 141-fit/30-dev，60 epoch，按 fused dev mAP 选点

结论：**精确保住 Signal baseline，但未产生有意义的融合增益；不进入 official test 或消融。**

## 运行完整性

- seed 42，B32/K4，60/60 epoch，5498 optimizer steps，0 AMP overflow。
- 最佳 epoch 51；保存后严格重载，五路指标逐项一致。
- Signal state SHA 在训练前、训练后和 checkpoint 重载后完全相同：`97234c...5a92`。
- official-test access count：0。
- 峰值显存：3656.94 MiB allocated / 6228 MiB reserved。
- 总耗时：2195.40 秒（36.59 分钟）。

## 同 checkpoint 五路结果

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| fused | 58.0168 | 57.4545 | 69.9394 | 76.6061 |
| CNN | 58.0181 | 57.4545 | 69.9394 | 76.6061 |
| Transformer | 58.0137 | 57.4545 | 69.9394 | 76.6061 |
| Mamba | 58.0135 | 57.4545 | 69.9394 | 76.7273 |

fused 相对 baseline 仅 `+0.00587 mAP`，相对 CNN 为 `-0.00130 mAP`，相对冻结 65 mAP dev 门为 `-6.98324`。因此三个主条件中只满足“高于 baseline/Transformer/Mamba”，不满足“高于所有专家”和“达到 65 mAP”。

## 只读协同诊断

诊断严格加载 epoch51 checkpoint，处理 825 个 dev 样本，不训练、不创建 optimizer、不访问 official test。

| 诊断量 | 数值 | 含义 |
|---|---:|---|
| fused suffix / baseline 范数 | 0.027471 | 追加信息能量很小 |
| fused 距离绝对变化均值 | 0.0002017 | 检索距离几乎不变 |
| fused 距离 Pearson 相关 | 1.000000 | 与 baseline 几何近乎一致 |
| Top-10 邻居重合率 | 0.9998788 | 排序几乎完全相同 |
| 路由归一化熵 | 0.960005 | 路由选择性仍弱 |
| CNN/T/M 残差两两余弦 | -0.00148 至 0.00832 | 三专家残差存在差异，并非同质化 |

CNN、Transformer、Mamba、relay、reliability 和 fusion 参数相对初始化均有非零更新。三个训练后 residual scale 分别约为 `0.1060/0.1080/0.1058`。所以本次失败不能归因于“三分支没有训练”；直接证据指向最终残差缩放和高熵路由使互补信息未能改变检索排序。

## 主张边界

独立 result-to-claim 结论为 `partial`，置信度 high：

- 支持：V5 在训练三分支时精确保留 Signal 3072D baseline，并完成稳定、可重载、official0 的完整 dev 运行。
- 不支持：三分支协同带来检索增益、fused 优于全部专家、达到 65 mAP dev 门、三项创新已被性能验证或达到 SOTA。

下一步只允许一个基于上述诊断的 baseline-preserving main-only 架构修正；重新通过 TDD、capacity、overfit 后，才运行一次新的 seed42 dev。当前禁止消融、多种子和 official test。

## 证据与远端位置

```text
run summary  SHA256 58fb5ebb30f4a72b02d2377e52d55e20fa070f7a3b2f831d6a52987d32f8c4ab
diagnostic   SHA256 4aeafcfa29219ba51fbb81accf9d8d14528e096fe82026baa922b222e9473555
best ckpt    SHA256 43f4806437545520d91b2fe70349b6036dbb3949e6d6351d79a24c3aa7f539c0

/root/autodl-tmp/trifusion-v2/artifacts/
  trifusion_signal_preserving_v5_dev_seed42_18f81c3/
```

版本化轻量证据：

```text
evidence/trifusion_signal_preserving_v5_dev_terminal_seed42.json
evidence/trifusion_signal_preserving_v5_diagnostic_seed42.json
```
