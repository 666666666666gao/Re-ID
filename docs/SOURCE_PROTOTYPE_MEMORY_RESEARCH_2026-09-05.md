# 训练侧环境不变性与负例覆盖：实际代码核对

此文落实用户2026-09-05更新复核的研究优先级，尚未登记或启动新实验。
V23原六端过程、代码和门限保持不变。研究重点是保留真实身份关系，
用source-only原型扩大身份竞争范围，并使增强视图向可靠身份表征学习。

## IICI实际实现

作者固定源码commit d60e09bad6637b076a3c1347dfe59745b4cd76b3。
train.py:207-231以完整训练特征的身份均值初始化归一化class_memory，
并显式断言每身份仅一摄像头。当前项目每fold有14个跨摄像头身份，
所以该断言和单个class_camera不能原样移植，不能把这些真实正例改标为负例。

sct_memory.py:16-37由普通增强features的反向过程更新身份原型，
先用momentum混合再归一化；增强features在112-126附近只对原型的detach快照计算分数，
不更新该原型。128-140附近仅在当前环境所属身份原型中计算身份softmax。
train.py实际默认temp0.05、momentum0.2；不能把类构造器默认temp0.07当作实际入口值。
其普通增强仍含erase0.5；额外增强含brightness0.2、erase0.6，
MSMT17另用contrast0.15。增强不是两份完全无扰动/任意颜色变化的图像。
Market ViT脚本显式关闭额外增强，与ResNet脚本不同。

[IICI作者源码](https://github.com/Terminator8758/IICI/blob/d60e09bad6637b076a3c1347dfe59745b4cd76b3/train.py)，
[IICI论文](https://arxiv.org/abs/2311.01155)。

其多级MCNL是另一个组件；V22已有完整负结果，不将它移入后继。
这里可采用的思想是环境内身份区分及原型更新/强增强监督分离，
不是宣称已经复现IICI或继承其跨数据集收益。

## XBM实际实现

作者固定源码commit 223ecdc25f71ef1721a58bc87cc567025a32bc92。
trainer.py:105-115在START_ITERATION之后将detach特征和真实标签入队，
保留当前batch损失，再计算当前特征对memory的损失。
sample_config.yaml使用55000队列/1000起点；
defaults.py为8192/2000，不能混合成一个实际配置。
xbm.py实际按128D分配，队列满的判定使用最后标签是否为0。
当前项目身份编码含0且表示维度不同，后继不能原样复制这些实现假设。
source-only身份原型与FIFO样本队列也不是同一种记忆。

[XBM作者源码](https://github.com/msight-tech/research-xbm/blob/223ecdc25f71ef1721a58bc87cc567025a32bc92/ret_benchmark/modeling/xbm.py)，
[XBM原论文](https://arxiv.org/abs/1912.06798)。

这项研究支持扩大负例覆盖的动机，尚未证明本项目特征具备论文所需的slow drift。
若后继采用记忆，必须登记刷新方式、年龄和重复样本处理；
若采用全source原型初始化，则直接完成全部合法条目，避免未初始化项进入分母。

## 与当前实证的对应和后继边界

原CrossCameraIdentitySampler先选择一个跨相机身份，
再按“单相机优先、剩余组多者优先”补齐其余身份
（modeling/trifusion/aligned_data.py:176-219）。现有全量元数据确认
1580/1680批只有一组跨相机身份，跨相机正对占8.070791%。
因此新监督覆盖合同应统计真正跨相机正关系，保留全部94个source身份和14个
跨相机source身份；不得只丢弃其余身份，也不得仅检查每batch非空。

优先固定一个统一的source-only身份原型目标：
正常增强负责更新，强增强受到同身份目标监督；
全体source身份提供不同ID竞争者，同环境子集要求排除摄像头捷径。
对多摄像头身份保留全局ID，环境成员关系必须覆盖其实际全部摄像头。
这是待形成完整新计划的候选设计，不是已确定的损失权重、采样器或训练预算。

两个端点须同起点、同视图、同预算，保持三角色和推理路径可比较；
每fold及每endpoint单独初始化、清空并保存记忆状态。
单独记录原七组目标、原型目标、梯度覆盖、正负关系数量和新增计算。
不把测试图库、dev、heldout特征写进记忆，也不跨fold坐标系计算距离。
固定完整图库的五路结果、所有21身份及原五项Q1门；
主结果成立后再做容量/目标/角色消融。

来源文件13份原始和LF归一化SHA保存在
evidence/source_prototype_memory_source_inspection_20260905.json。
源码获取时上游IICI克隆附带了两份权重；没有加载它们，已仅保留文本源码，
把当次完整克隆及Git对象移入Windows回收站。无本地模型、tensor或图像执行。
