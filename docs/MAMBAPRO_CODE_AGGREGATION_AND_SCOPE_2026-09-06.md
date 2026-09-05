# MambaPro 固定代码的聚合机制与复现边界（2026-09-06）

记录时间：2026-09-06T02:33:13.022271+08:00。固定 commit f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149，MIT License。
取得完整 98 文件的树，以及 14 份源码/配置/说明文本共 137,173 bytes；各文件均与固定树 Git blob 和完整 SHA 对应。
证据为 evidence/mambapro_source_text_inspection_20260906.json。
本次只做源码追踪，没有运行作者模型、下载 checkpoint、安装依赖或进行 TriFusion 修改。

## 论文机制及实际调用

论文将冻结 CLIP 的适配/提示与后续 Mamba 聚合分工；先模态内聚合，再将三模态拼成 3N 长序列。
聚合后的每个模态 Patch 均值与对应 CLS 拼接、归一化并降维，接身份监督。
这与当前 TriFusion 并行角色残差银行的结构不同。[原论文式13–21](https://arxiv.org/html/2412.10707v1)

固定源码可直接追踪上述路径：

| 位置 | 实际行为 |
|---|---|
| modeling/make_model.py:143–163 | CLIP 路径 feat_dim=512；AAM 固定堆叠两层 |
| modeling/fusion_part/mamba.py:624–634 | 每层先 SS2D_intra，再 SS2D_inter |
| 同文件:430–446 | 三个模态分别运行模态内 SSM |
| 同文件:575–584 | 沿 Token 维按 RGB/NIR/TIR 拼接，单次 SSM 看完整 3N 序列 |
| modeling/fusion_part/AAM.py:27–45 | CLS 保留，聚合 Patch 各取均值；每模态将 CLS+均值经 LN/Linear 从 1024 降至512，再拼为1536D |
| modeling/make_model.py:183–235 | 训练按配置返回原 CLS 监督及聚合监督；MAMBA=True 推理时返回聚合特征 |
| engine/processor.py:65–71 | 对实际返回的每组 score/feature 分别计算配置的 ID/Triplet |

[固定融合入口](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/modeling/fusion_part/AAM.py)；
[固定 SSM 调用](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/modeling/fusion_part/mamba.py)；
[固定主模型](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/modeling/make_model.py)

config/defaults.py:35 的 MAMBA_BI=False，RGBNT201/MSVR310 两份配置都没有覆盖它。
因此固定默认实现的模态内与 3N 模态间扫描均为单向；源码提供的反向条件路径不等于默认启用。
“完整3N序列”也不意味着每个输出Token都以双向方式接收所有后续位置的信息。
这一点是固定代码的作用域说明，不能据此改写其他论文变体或未经验证的训练命令。

## 与当前三分支设计的对应

当前 V8/V24 的 Mamba 在每个位置上运行长度3的模态序列，见 signal_preserving_v8.py:195–203；
MambaPro 的模态间序列为整个 3N。N=128 时分别是长度3与384，拓扑不同。
当前 Mamba 最终读取与冻结 reference 做差后的 Patch 均值；
MambaPro 把原 CLS 与聚合 Patch 均值共同送入每模态降维头。
当前 CNN/Transformer/Mamba 是三条各自继续冻结尾部的角色路径，固定拼接为残差银行；
MambaPro 的 Mamba 位于语义提取之后，输出一个聚合检索表示。

这些差异只提供结构参照，不证明替换为长序列聚合就能提升 TriFusion。
如未来采用 CNN/Transformer 提供内容、Mamba 汇聚的结构，需要重新固定角色输出、残差/reference、
原 Signal 前缀、监督和计算预算；不能把它称作已经完成的当前三专家消融或已验证的新版本。

## 直接运行前必须绑定的实际条件

PFA 的源码为升维再降维 d→2d→d，使用 QuickGELU；
不是当前 V23 的 768→128→768 模态专属窄 MLP。
共享视觉主干中的冻结函数虽名为 mark_only_lora_as_trainable，
实际以参数名包含 adapter 来保留可训练参数，不能仅凭函数名声称实现了 LoRA。
[适配器定义](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/modeling/clip/model.py#L183)；
[冻结规则](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/modeling/clip/LoRA.py#L307)

两份配置均为 B64/K4、Adam 3.5e-4、60 epoch、10 epoch warmup，输入分别256×128及128×256；
MSVR310 的 DIRECT=0，与 RGBNT201 的 DIRECT=1 不同，实际训练监督头组合也不同。
本项目 B64/K8、seed42 与固定终点合同应单独登记，不能混称作者原条件。
源码默认按验证 mAP 保存 best；本项目官方测试不能用于逐 epoch 选择。
[RGBNT201 配置](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/configs/RGBNT201/MambaPro.yml)；
[MSVR310 配置](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/configs/MSVR310/MambaPro.yml)；
[保存逻辑](https://github.com/924973292/MambaPro/blob/f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149/engine/processor.py#L150)

CLIP 实际读取路径在 modeling/make_model_clipreid.py:175–186 硬编码为 ../PTH/ViT-B-16.pt；
配置中的 PRETRAIN_PATH_T 不是该 CLIP 分支最终使用的路径。
MSVR310 实际过滤仍为同身份同 scene（utils/metrics.py:87）。
这些是将来固定运行入口需要核对的现有代码事实，当前未作运行适配。

本次没有实例化参数、测量 FLOPs/延迟/显存、核验本机模型成绩或证明该固定仓库在现有环境可直接运行。
文献报告值继续保留文献标签；当前正在进行的 V24 比较、科学门和失败封存规则不变。
