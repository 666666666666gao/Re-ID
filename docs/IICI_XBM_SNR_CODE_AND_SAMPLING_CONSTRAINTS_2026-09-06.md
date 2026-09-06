# IICI、XBM、SNR 作者代码与真实采样约束核查

核查时间：2026-09-06T20:13:14.726891+08:00。本轮读取20份作者代码/许可证文本，并完成全部source标签容量分析。没有训练作者模型、接入组件或改变当前RGBNT100固定20epoch输入。作者源码只在本机临时审阅目录，项目归档提交、SHA和分析。

| 论文/代码 | 作者与发表 | 已核实机制 | 项目边界 | 一手来源 |
|---|---|---|---|---|
| IICI | Menglin Wang、Xiaojin Gong；ACM MM2023 | camera/subcamera原型判别、weak-only EMA、weak/strong原型损失、多级MCNL | 原入口要求每训练身份只属于一个camera；V24已测试核心weak-update/strong-prototype思路 | [论文](https://arxiv.org/abs/2311.01155)、[作者入口](https://github.com/Terminator8758/IICI/blob/d60e09bad6637b076a3c1347dfe59745b4cd76b3/train.py) |
| Cross-Batch Memory | Xun Wang、Haozhi Zhang、Weilin Huang、Matthew R. Scott；CVPR2020 | 缓存历史实例特征，用当前特征查询更多实例负例；缓存detach，无额外EMA编码器 | 新增的是同一负身份的实例多样性；V24已包含所有source身份原型 | [CVF论文](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)、[作者循环](https://github.com/msight-tech/research-xbm/blob/223ecdc25f71ef1721a58bc87cc567025a32bc92/ret_benchmark/engine/trainer.py) |
| SNR | Xin Jin等；CVPR2020 ReID论文，另有2021通用DG/DA扩展 | IN去风格，通道门控回补判别残差 | 本轮代码来自Microsoft通用DG/DA的PACS分类实现；不是RGBNT201或原ReID复现 | [ReID论文](https://arxiv.org/abs/2005.11037)、[作者通用仓库](https://github.com/microsoft/SNR/tree/f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b) |

本轮没有新的三光谱任务性能数值，不把这些论文在其他任务的结果当成本项目增益。近期同类公开指标另见SOTA_PRIMARY_REFRESH_2026-09-06.md。

## 1. V24已经覆盖的部分

IICI原入口在原型初始化时断言每身份仅有一个camera；默认ResNet的Market命令不拆subcamera，MSMT才启用每epoch低层特征聚类。因此“所有IICI运行都进行了风格聚类”不成立。[作者MSMT脚本](https://github.com/Terminator8758/IICI/blob/d60e09bad6637b076a3c1347dfe59745b4cd76b3/train_msmt.sh)

项目V24按真实identity-camera建立108个原型，均衡各camera后形成94个全局身份原型。weak视图按每个identity-camera组一次EMA更新，strong fused7680D接受全局及同camera原型分类。两端共享双视图和原采样器，以系数0/1配对；完整Q1与失败门已经封存，不能换名再试同一机制。依据为source_prototype_v24.py、dual_view_data_v24.py、train_signal_preserving_v24.py和原完整比较JSON。

V24不是完整IICI复现：后者还对weak特征计算原型损失，并有ISCS标签前提及可选subcamera/多级MCNL。V24失败不等于整篇IICI无效。新的实际P/K曝光合同、实例级跨batch记忆和source风格聚类仍是不同干预；再次增加MCNL不作为本轮新路线。

## 2. XBM的可借鉴点与代码边界

作者示例在1000次更新后先将当前特征detach入队，再叠加batch与memory度量损失；示例为128D、队列55000。队列以最后一个target是否非零判断已满，本项目class0合法、fused是7680D，不能直接继承这两个约定。[队列源码](https://github.com/msight-tech/research-xbm/blob/223ecdc25f71ef1721a58bc87cc567025a32bc92/ret_benchmark/modeling/xbm.py)

V24已有全部93个其他source身份的原型负例。实例记忆真正要检验的是：同一负身份的困难视角是否被原型均值掩盖。clean-source原型分类100%或缓存余弦接近1，不能替代真实增强训练视图下的实例难例证据。后继若实施，应只缓存该折source实例，记录索引和年龄、按真实身份过滤，最终仍用完整图库比较。

XBM实际LICENSE为CC-BY-NC-4.0；IICI首页及本次文件未见仓库级许可证声明；Microsoft通用SNR实际LICENSE为MIT。本轮没有复制作者实现到项目。[XBM许可证](https://github.com/msight-tech/research-xbm/blob/223ecdc25f71ef1721a58bc87cc567025a32bc92/LICENSE)、[SNR许可证](https://github.com/microsoft/SNR/blob/f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b/LICENSE)

SNR示例在二维CNN特征的空间轴做IN；CLIP的逐token LayerNorm轴不同。把恢复残差思想迁移到语义Patch需要重新定义作用位置、归一化轴及身份监督，应称为新实现，不能称为原方法等价复现。

## 3. 两个跨camera身份/批次的完整数量约束

分析覆盖三折282个source身份成员、6252条source记录成员，并逐折核对已归档1680个原采样batch。每折94身份中14跨camera、80单camera。这里只考察B64/K8、每批两个跨camera身份这一目标；没有实现采样器、扫描超参数或运行模型。

| Fold | 每epoch批次 | 原分组可供跨camera组 | 原调度实际使用组 | 两组/批次所需 | 超出原分组的最低额外组数 | 跨camera记录位置重复次数下界 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 29 | 41 | 39 | 58 | 17 | 83 |
| 1 | 28 | 41 | 38 | 56 | 15 | 56 |
| 2 | 27 | 40 | 33 | 54 | 14 | 63 |

最后一列是所需跨camera样本位置数减去该折所有真实跨camera记录数的聚合下界，不是具体调度器实测值，也不等于上一列乘8。原数据还有未使用尾部记录，两种数量应分别报告。

只改原available排序，不能确保每批两个跨camera身份。实现目标必须增加跨camera组抽取并承认身份曝光改变，同时保留全部80个单camera身份参与训练，完整图库中的干扰身份也保留。单camera组总量足够填其余6组，但这只是必要数量条件，具体调度仍需全epoch重放。

身份内固定4+4是另一个干预：三折无记录复用时最多31、34、29组。不能把4+4与每批两身份同时加入后，只归因为其中一个改变。后继应先选定唯一干预，再登记全部身份覆盖、重复曝光、跨camera正对和完整图库终态比较；标签数量本身不能证明mAP改善。

## 4. 下一步顺序

当前RGBNT100固定主训练和后续官方完整比较优先完成。已有内部增益不当作官方结果。

后继优先把尚未测试的真实监督曝光定义成可执行、可重放的合同，再决定新配对训练。实例记忆作为另一项独立假设，避免与采样变化、SNR同时叠加。weak/strong原型路线已有V24完整负结果，不能继续当成尚未实现的关键修复。

证据：evidence/trifusion_iici_xbm_snr_author_code_review_20260906.json记录20份文本的作者提交/SHA和项目对照；evidence/trifusion_rgbnt201_camera_sampling_feasibility_plan_20260906.json冻结分析输入；同名无plan结果JSON记录全部身份计数；tools/census_rgbnt201_camera_sampling_feasibility.py为纯标准库入口。所有新增模型、张量、图片、优化器和检索指标计算均为0。
