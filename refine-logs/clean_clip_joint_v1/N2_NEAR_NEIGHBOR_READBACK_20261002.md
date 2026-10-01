# N2直接近邻：实现范围与后续对照边界

2026-10-02。仅为原始来源核读与对照准备，没有运行作者模型、实现N2或判定新颖性。MDReID复用已核的作者提交；DeMo、MODAL由两个fresh只读research-lit分片核读，模型gpt-6-astra/max。分片是来源提取，不是性能或新颖性审稿。父进程下载10份原始文本并核对5个作者Git blob；摘要与SHA见同目录SOURCE_CHECK。

| 近邻 | 原方法实际组织的信息 | 后续同配方控制应保留什么 | 不应混淆的边界 |
|---|---|---|---|
| MDReID | 编码期间的模态特有与共享Token，及组合表示 | 共享/特有/组合路径及其监督职责 | 不是private adapter写回开关；不能只放两个线性投影就称作者复现 |
| DeMo | R/N/T/RN/RT/NT/RNT七种模态输入子集，每种由专用query读取global与patch | 七种真实输入支持、内容注意力与专家拼接 | 模态子集不是七个空间区域；不能只保留七个算子名称 |
| MODAL | 全局class token的三个专有、三个双模态共享、一个三模态共享稀疏code，再读取局部patch | 七类信息单元与稀疏展开；明确全局输入或区域输入 | 不是逐区域分解；完整方法含文本，image-only版本只能称机制适配 |

MDReID的组合损失停止特有/共享比较分支梯度，并取batch级困难距离极值；它不是逐query mAP增量，也不是同一正负对的间隔差。已缓存版本缺少仓库级代码许可，不复制其实现进入TriFusion。[作者模型](https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/modeling/make_model.py)、[实际组合损失](https://github.com/stone96123/MDReID/blob/3525ac2da1a2a90a5a160c930fac674b4f226f6c/layers/triplet_loss.py)。

DeMo的已核代码对七种子集均执行专家，采用dense softmax而非top-k dispatch；先按channel分head，再执行各head的专家。默认最终检索拼接3C原模态与7C专家表示，CLIP C=512时为5120维。若改成1536维读出，必须标注同配方适配，不能冒称作者默认结果。发布配置还存在100训练30轮、MSVR每身份4张等与论文叙述的区别；原配方与统一50轮控制应分列。[作者HDM/ATMoE](https://github.com/924973292/DeMo/blob/b4f323a430b32e3a1637c3e7acb25868cb52e9cd/modeling/moe/AttnMOE.py)、[推理读出](https://github.com/924973292/DeMo/blob/b4f323a430b32e3a1637c3e7acb25868cb52e9cd/modeling/make_model.py)。

MODAL核读的是arXiv v1预印本。MFSD分解视觉/文本global class token；TIDF以分解code为query、视觉patch为key/value。完整文本路径不能与image-only控制混写。本次没有建立官方可运行代码或checkpoint链接的证据，不能声称精确复现；原文末端descriptor与缺失模态激活定义也有待作者实现澄清。当前只研究完整三模态，不为这些未纳入任务的缺失模态情形添加额外分支。[原文§III与实现说明](https://arxiv.org/html/2608.15096v1)。

未来直接对照须使用同一公开起点、视觉更新边界、数据/过滤、50轮日程、主监督和选点规则；输出宽度、参数量与额外资源无法相同时单列。作者完整配方用于外部复现；同预算机制适配用于分离结构作用。N2的区域机制仍须分别对照无分解、全量对齐及这些近邻，不能因这里核读了原文就认定主张成立。

当前六项clean-CLIP实验源233不变。先完成六项、唯一完整CPU报告及fresh结果审计，再登记N1原生细节路径；N2/N3未实现。N3还需要实际训练批次跨环境合法正负支持，不能把数据集级多环境身份数当成监督覆盖。
