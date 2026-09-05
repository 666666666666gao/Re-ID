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
