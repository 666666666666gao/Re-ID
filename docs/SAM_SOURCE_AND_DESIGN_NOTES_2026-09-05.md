# SAM 作者来源与 V21 设计依据（2026-09-05）

## 原始来源

作者论文 [Foret 等，ICLR 2021](https://arxiv.org/abs/2010.01412v3) 的
Algorithm 1 先计算参数梯度，再沿其单位方向扰动参数，在扰动点计算用于更新
原参数的梯度。§3.1 为非 SAM 对照给出双倍 epoch 预算；Appendix C.3
同时报告固定 rho=0.05 的结果。论文的图像分类/微调结果不证明 RGBNT201
收益；这里不沿用其超参数搜索或最佳 checkpoint 选择。

作者仓库固定到 commit
[dae9904c4cf3a57a304f7b04cecffe371679c702](https://github.com/google-research/sam/tree/dae9904c4cf3a57a304f7b04cecffe371679c702)，
实查于 2026-09-05。[训练代码](https://github.com/google-research/sam/blob/dae9904c4cf3a57a304f7b04cecffe371679c702/sam_jax/training_utils/flax_training.py#L484)
保留第一遍的模型状态、丢弃第二遍的新状态；rho=0 对照实际只有一次反传。
代码采用 [Apache 2.0 许可证](https://github.com/google-research/sam/blob/dae9904c4cf3a57a304f7b04cecffe371679c702/LICENSE)。
当前 PyTorch 实现按论文公式独立编写，没有复制作者通用模型兼容分支。

服务器实查为 PyTorch 2.5.1+cu121。安装包
torch/amp/grad_scaler.py 的 unscale_ 只允许同一 optimizer 每步调用一次；
[对应版本官方文档](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/amp_examples.rst)
说明了此限制。V21 第一遍用共同缩放的梯度求单位方向，第二遍之后才 unscale，
不修改 GradScaler 内部状态。

## 来自本项目的证据与推断边界

V19 六个最终模型的完整 source/heldout 诊断显示，source 七路分类与五路检索
均为 100%，heldout 明显低于 source。V20 跨模态身份监督完整三折主比较随后
fused -1.010871 mAP、Transformer -4.779791 mAP，固定 Q1 门失败。这些事实
支持继续检验身份泛化问题，未证明尖锐极小值是唯一原因，亦不证明 SAM 有效。
不能把 Mamba 单路收益当成 V20 融合成功，不扫描 V20 温度/系数或重训旧版本。

新假设限定为：保留原 V8 的 Signal/CNN/T/M 表示和七路 ID/Triplet 目标，
改变训练梯度的取得位置，是否能提高未参与该 fold 完整训练路径的身份检索。
没有新推理参数、文本、数据集混合、外部 backbone 或跨模态辅助项。
这是已发表优化思想在当前训练管线上的主比较，不作方法本身新颖性声明。

## 必须按现有代码处理的状态

- signal_preserving_v8.py:563–573、607–612 构造七个训练 BatchNorm1d 颈部，
  644–655 的前向实际调用它们。SAM 第二遍前向会再次更新 running_mean、
  running_var 和 num_batches_tracked。因此保存第一遍后的三个 buffer，第二遍
  反传完成后精确恢复；训练仍用当前 batch 统计，未将 BN 改成 eval。
- 仅对 requires_grad=true 的原有参数施加扰动。更新前用原参数副本 copy_
  恢复，保证 AdamW 在原参数处应用第二遍梯度，而不依赖浮点加减的可逆性。
  只需保存原 7,841,292 个训练参数，约 31 MB，无新模型状态保存协议。
- 现有 CNN/Transformer/Mamba 角色模块没有激活的 dropout 随机层，attention
  dropout 为默认 0。同一真实 batch 的增强已在 loader 完成，两遍复用该 batch。
  没有引入通用 RNG 恢复器、异常回退、参数分组兼容层或新推理分支。

## 比较预算的解释

拟采用同一源 checkpoint 下 ordinary AdamW 40 epoch、SAM 20 epoch：
每个 SAM batch 有两次前向/反传，普通对照每个 batch 只有一次，因此完整三折
每端各 3360 次前向/反传对，总计 6720 次。优化步数分别为 3360 和 1680，
总计 5040；普通对照遍历数据两倍。两端学习率保持 0.00035，
warmup 分别 10/5 epoch，随后按各自完整预算 cosine。

这匹配的是规定的前向/反传次数，不能称实际 wall time、epoch、优化步数、
数据暴露次数均相同。SAM 额外参数复制和状态处理的真实耗时须报告。
仅使用每端最后 checkpoint，不评价或选择对照第 20 epoch。
固定 rho=0.05，不扫描；三折五路和全部 query/gallery 范围保持。
该比较能回答规定训练计算预算下的整体效果，不能唯一分离 SAM 与不同
更新/数据暴露次数的因果贡献。

以上为研究和设计依据。新版本以执行前 EXPERIMENT_PLAN 和实际源码/配置 SHA
为准，本文本身不代表 T0/M0/Q1 已执行或已通过。
