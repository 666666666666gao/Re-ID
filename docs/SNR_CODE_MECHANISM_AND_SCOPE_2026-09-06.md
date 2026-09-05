# SNR 源码机制与可用范围（2026-09-06）

本次只读取原论文链接、作者主页和六份固定源码/说明文本，没有把 SNR 加进 TriFusion，也没有启动新训练。
V24 原 Q1、M0 独立审计和 MSVR310 优先级保持不变。

## 论文与仓库不是同一套可运行 ReID 实验

CVPR 2020 论文研究跨域行人重识别，以实例归一化去除风格，再把残差中身份相关的成分回补；
这是“去风格可能误删身份信息”的直接研究来源。
[原论文](https://arxiv.org/abs/2005.11037)

作者主页的 CVPR 2020 code 链接指向 microsoft/SNR；该仓库也被后续 2021
Domain Generalization and Adaptation 扩展论文引用。
当前固定 commit 为 f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b。287 个文件的完整路径树没有 ReID 专用命名入口；
顶层训练任务为 classification、detection、segmentation。不能把文件树检查扩大成历史上所有版本都没有 ReID 实现的结论。
分类 README 明确对应 PACS/ResNet18 和后续扩展论文；不是 RGBNT201、CLIP 或 TriFusion 的运行合同。
[作者主页](https://jinx-ustc.github.io/jinxin.github.io/)；
[固定分类目录](https://github.com/microsoft/SNR/tree/f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b/classification)

## 本次实际读取的分类实现

- classification/PACS/resnet_SNR.py:20–58：残差先做全局平均池化，经两层 1×1 映射、ReLU、Sigmoid 得到通道门；
  缩减率为16，返回门控残差、互补残差和门值。
- 同文件:439–452、504–527：前三个 ResNet stage 后使用带 affine 的 InstanceNorm2d；
  R=F-IN(F)，正常路径为 IN(F)+g(R)R，互补路径为 IN(F)+(1-g(R))R。
- 同文件:541–554：不同阶段的原归一化/回补/互补特征经各阶段共享分类器，产生分类概率。
- classification/PACS/model_resnet.py:420–430、475–480：对分类概率计算 batch 平均熵，
  用 SoftMarginLoss 鼓励 H(useful)<H(IN)<H(useless)，前三层各乘0.01，
  另加0.01的最后一层 useful 分类 CE，以及主分类 CE。
- 这是本次读取的 PACS 分类因果损失代码，不是对 CVPR 2020 ReID 论文度量损失的独立等价复现证明；
  也不能把文件中的 reid 变量名当作已包含行人检索训练入口的证据。

MIT LICENSE 已和源码一并取得。六份文本共58,584字节，全部与固定 Git tree 的 blob SHA-1 相符；
各自 SHA256 与位置见 evidence/snr_source_text_inspection_20260906.json。
未取得预训练权重、未运行下载代码、未安装仓库依赖，也未声称旧 PyTorch 环境适配已经完成。
[固定模块源码](https://github.com/microsoft/SNR/blob/f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b/classification/PACS/resnet_SNR.py)；
[固定损失实现](https://github.com/microsoft/SNR/blob/f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b/classification/PACS/model_resnet.py)

## 对当前项目的约束

可借鉴的是“归一化后允许身份信息回补”的假设。
CLIP 语义 Patch 的空间通道统计不等价于 ResNet 卷积风格统计，CLS 也没有同样的空间归一化语义；
当前没有证据支持将该实现直接塞进全部 Token 后称作等价 SNR。
如后续选择这条路线，应重新登记插入位置、reference/残差定义、原能力保留和完整配对比较，
并继续以新身份完整图库排序为准，不能以熵降低或所谓因果损失降低替代检索结果。
本次没有决定将其作为 V24 之后的正式主版本，没有改写 V24 的失败封存或门槛。


## 原始 ReID 公式复核与项目读取方式（2026-09-06T02:26:36.380613+08:00）

本次补读原始 arXiv v1 的第3节。原 ReID 约束比较空间池化后的正/负对距离：
记归一化、身份回补、互补回补后的向量为 f0、f+、f-，d=(1-cos)/2，sp 为 Softplus，则

- L+ = sp(d(f+a,f+p)-d(f0a,f0p)) + sp(d(f0a,f0n)-d(f+a,f+n))；
- L- = sp(d(f0a,f0p)-d(f-a,f-p)) + sp(d(f-a,f-n)-d(f0a,f0n))。

原文在四个 ResNet 阶段加入该项，权重为 0.1/0.1/0.5/0.5，并保留 ID 与 batch-hard Triplet。
这与前述 PACS 熵约束有明确区别。[原论文第3节，式6–8](https://arxiv.org/pdf/2005.11037)

以下是根据公式和本项目代码作出的代数推论，不是作者已验证的 CLIP 结论。
设 X 有 N 个 Patch，IN 沿空间维归一化每个通道，gamma/beta 与通道门 a 不随 Patch 位置变化，则：

    mean_N(IN(X)) = beta
    mean_N(IN(X) + a * (X - IN(X))) = beta + a * (mean_N(X) - beta)

因此，若把 IN 直接附加在当前 Mamba 的最终残差 Patch 上、紧接着仍按
signal_preserving_v8.py:449–452 求均值，这个读取位置会失去样本独有的残差均值；
加回通道门控残差后，该均值读取只看到门控后的原均值。
不能据此声称这个读取位置已经获得了新的空间细节或验证了风格消除。
该结论严格限于中间没有其他空间/Token 非线性算子的情形，不能扩大为 SNR 无效或所有插入位置都会退化。
当前 CNN 的分区池化、Transformer 的 CLS、Mamba 的全 Patch 均值也不能按相同读取机制处理。

另一个工程推论是：原文四个 Softplus 项有正的常数下界。仅从 d∈[0,1] 已可得到
L+ + L- ≥ 4 log(1+exp(-1)) = 1.253046750073。
这是保守下界，不声称可以同时达到各项最小值，更不是已经确定的新 M0 最优损失。
如未来加入该目标，固定批次工程指标必须在训练前明确其原始损失项及非零下界，
不能继续把只有标签平滑 CE 的旧下界当作全部新目标的零点，也不能训练后修改门槛。

复核与推论范围记录在 evidence/snr_original_reid_formula_review_20260906.json。
没有运行 SNR、改动 V24、决定下一版本，或对作者方法作复现成功声明。
