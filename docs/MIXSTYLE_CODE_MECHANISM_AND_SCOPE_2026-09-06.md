# MixStyle 代码机制及 TriFusion 接入边界（2026-09-06）

记录时间：2026-09-06T03:56:55.565546+08:00。状态 SOURCE_RESEARCH_ONLY；没有新增训练、模型前向、权重下载或检索。
核查对象为 [ICLR 2021 原论文 v1](https://arxiv.org/pdf/2104.02008) 和
[作者仓库固定提交](https://github.com/KaiyangZhou/mixstyle-release/tree/16f7cf1fe2c7b1b3c660c72b32817ebb0545397a)，
不能将当前仓库的所有配置都称为原论文完全相同的实验条件。

本次只取得 12 份 UTF-8 源码/配置文本，共 85,559 bytes，逐个与固定提交 Git blob 核对。
commit 为 16f7cf1fe2c7b1b3c660c72b32817ebb0545397a；仓库 MIT，Copyright 2021 Kaiyang Zhou。
完整路径、Git blob SHA1、全文件 SHA256 和本地文本位置见
evidence/mixstyle_source_text_inspection_20260906.json。没有 clone 或导入该项目。

## 代码实际混合什么

[reid/models/mixstyle.py](https://github.com/KaiyangZhou/mixstyle-release/blob/16f7cf1fe2c7b1b3c660c72b32817ebb0545397a/reid/models/mixstyle.py#L80)
对 NCHW 张量分别求每图每通道的空间均值和标准差，统计量 detach；归一化内容仍保留梯度。
每样本一个 lambda，来自 Beta(alpha, alpha)，对另一 batch 样本的均值、标准差分别作凸组合：

    z_i = (x_i - mu_i) / sigma_i
    mu_mix = lambda_i * mu_i + (1-lambda_i) * mu_j
    sigma_mix = lambda_i * sigma_i + (1-lambda_i) * sigma_j
    y_i = z_i * sigma_mix + mu_mix

原实现默认 p=0.5、alpha=0.1、eps=1e-6；评估模式直接返回输入，没有新的推理参数。
var 调用没有显式指定方差修正参数；若后续复用，不能只依据原文 HW 分母便声称与实际代码数值完全相同。
这里没有改变代码或为了假设风险添加分支。

random 模式使用 randperm(B)。crossdomain 模式反转 batch，再分别打乱两半；算子本身没有读取域标签，
其不同域保证依赖上游两半来自两个不同环境。任意 B64/K8 batch 不具备这个保证。
[resnet_ms2.py](https://github.com/KaiyangZhou/mixstyle-release/blob/16f7cf1fe2c7b1b3c660c72b32817ebb0545397a/reid/models/resnet_ms2.py#L252)
确实选择 crossdomain，配套 cfg_r50_domprior.yaml 选择 RandomDomainSampler/num_cams=2。
本次没有重新取得外部 torchreid sampler 实现，因此这里报告配置合同，不声称重放证明其 batch 顺序。

默认 ReID ResNet 工厂 resnet50_fc512_ms12_a0d1 插在 layer1/layer2 之后，alpha=0.1；
并非默认插满四个 block。原随机版本 cfg_r50.yaml 为 60 epoch/stepsize20，
当前域先验版本为 180 epoch/stepsize60，其他显式训练项包括 SGD、LR0.05、B32。
这两份当前配置不能充当只有混合方式不同的容量/预算匹配对照，也不能直接套用到项目的 CLIP、B64/K8。

## 原论文直接支持的边界

原任务为 Market1501 与 Duke 之间的行人 ReID，使用 ResNet/OSNet。
Table 3(b) 的 ResNet-50 mAP 为：baseline19.3、res1 22.6、res12 23.8、
res123 22.0、res1234 10.2、res14 11.1、res23 20.6。
作者将末层混合下降与临近平均池化的统计量承载类别语义联系起来。
这不是 CLIP 或 RGBNT201 的实测结论；本次通过 PDF 文本读取表格和第3.4节，没有声称独立图像可视化复核。
[原论文第7页](https://arxiv.org/pdf/2104.02008)

## 与当前项目的关系：推论，不是新实测

当前 TriFusion 三分支从 CLIP block8 后的语义序列开始，Mamba 最后读取残差 Patch 均值；
因此直接把较深语义通道的均值当成纯环境风格并混合，缺少等价于原方法低层接入的依据。
这条风险由论文深层结果与本项目实际读取位置共同支持，但不能推断所有 CLIP 接入位置都会失败。

V24 的新诊断表明 clean source 的 fused 原型分类及非self样本正负间隔已经充分拟合；
它没有测量任意强环境增强下的分布。增加有效训练环境变化仍是可研究假设，
但当前没有证据选定 MixStyle 插层、强度或损失，也没有授权它绕过固定工程门和配对 Q1。
不扫描已封存 V24 的增强参数，不将 source100% 或外部论文提升替代未知身份检索证据。

后继方案若采用特征风格混合，需先登记具体作用位置、内容保留依据和匹配对照，
保留冻结 Signal 的独立原始检索输出，并检查增强是否仍保留真实身份信息。
本次只完成机制与适用边界研究；没有生成 V25 配置、模型或实验结论。
