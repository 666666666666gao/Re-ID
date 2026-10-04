# RoDI作者PDF：资源与协议边界

固定作者提交 `2f38911c49d42d4ca259d440a851b8d77dddccbe` 的[PDF](https://github.com/lsh-ahu/RoDI/blob/2f38911c49d42d4ca259d440a851b8d77dddccbe/assets/RoDI.pdf)已取得；13页，文件与Git blob哈希核验。目视复核PDF页6、7、12、13。二进制、页面图及全文提取留本地私有目录，不再分发。

| 主表1版本 | RGBNT201 mAP/R1/R5/R10 | MSVR310 mAP/R1 | RGBNT100 mAP/R1 |
|---|---|---|---|
| CLIP ViT-B/16 | 84.1/87.2/92.0/93.2 | 64.1/77.2 | 88.5/97.6 |
| DINOv3 ViT-B/16 distilled | 85.3/87.9/93.0/94.8 | 71.8/84.8 | 89.0/99.1 |

页7表2的201累计消融mAP：CLIP 76.0→79.3→82.7→84.1；DINOv3 81.0→82.3→84.0→85.3。不同预训练版本分列。

页6说明B64/K8、Adam、学习率3.5e-4、weight decay1e-4、warmup10轮、单RTX4090。该说明不足以核实精确query/gallery成员、camera/scene过滤、选checkpoint规则、总训练轮数和预训练权重哈希。固定仓库仍无可执行训练/评价代码。这是论文参照，不是同协议本地复现或SOTA已达成证明；不修改当前实验。
