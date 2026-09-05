# 表征泛化方向的公开实现核对（2026-09-05）

这是V19全部终态后的资源笔记，不是新的实验方案或性能证据。V19的完整诊断
运行范围已另行固定，下面的候选均未加入模型，未启动新训练。

## 监督对比学习

[SupContrast作者实现](https://github.com/HobbitLong/SupContrast/blob/master/losses.py)
接受多view特征和真实身份标签，以同标签样本构造正例，明确排除自身比较。
这提供了检查“每个专家内部RGB/NI/TI是否有共同身份几何”的训练目标参考。
当前V8/V19只对三模态拼接残差、baseline+残差分支及融合施加ID/triplet；没有
直接的模态间对比项。若全部诊断显示相同身份的模态方向不一致，后继可检验
一个身份感知、专家内的跨模态目标，不把不同专家强迫成相同表征。是否采用、
温度/权重/预算和完整门槛均须在新实验前固定，不能根据V19个别身份调参。

## 模态原型一致性

[UPCL作者仓库](https://github.com/ZhouZhongao/UPCL)公开了训练代码；
[NeurIPS2025论文](https://proceedings.neurips.cc/paper_files/paper/2025/file/82a0696bea2c4ebf726fc796eaca7a55-Paper-Conference.pdf)
以模态原型的均值构造共同身份原型，分别约束各模态，并针对多人/车辆类别联合
任务加入聚类约束。其主设置联合训练RGBNT201和RGBNT100，不能把该表格直接
列为当前单数据集、固定141-fit条件下的同协议比较。这里仅借鉴可核查的模态
一致性思路；当前没有引入prototype memory、类别聚类或跨数据集联合训练。

## 风格随机化

[MixStyle作者代码及ReID实现](https://github.com/KaiyangZhou/mixstyle-release/tree/master/reid)
通过训练时混合样本的特征统计量改善跨域泛化，仓库提供CNN上的实例。
其原始假设针对底层视觉风格；当前分支输入是CLIP block8后的语义token，
不能未经验证就把该处的均值/方差视为纯camera风格。V18移除一条camera相关
方向后仍有身份受损，也不构成任意风格随机化均有效的证据。目前不采用此改动。

以上来源用于形成可检验假设，不能替代三折完整配对训练、同checkpoint五路
结果与严格开发/官方主比较。V19封存、seed42、无消融/扫描和不修改Signal的
既有要求仍有效；主目标尚未实现。
