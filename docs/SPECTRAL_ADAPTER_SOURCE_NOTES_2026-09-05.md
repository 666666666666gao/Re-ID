# 模态专属适配器的主源核查与V23移植边界（2026-09-05）

## 问题证据与待检验假设

V22完整Q1科学门全FAIL；共同初始化的同完整图库诊断表明普通继续训练的fused
仅从80.590328到80.640677 mAP，CNN/T/M相对初始化变化不一致。
本次证据排除了用旧eligible-only约88mAP直接减当前full-gallery fused来证明
训练退化的推断；不证明缺少模态参数就是唯一瓶颈。

当前V8源码modeling/trifusion/signal_preserving_v8.py:264–465：
CLIP尾部按B×3打包，同一组CNN/T/M角色模块分别处理RGB/NI/TI；
Mamba另外进行跨模态序列处理，但不存在模态专属的语义尾部残差MLP。
V19尝试的是每个expert独立训练完整CLIP尾部，仍在模态间共享；其固定负结果封存。
新假设是在冻结尾部之后增加按已知模态分派的残差MLP，能改善全部专家的heldout检索。
这是待检验假设，不能由上面的源码结构或跨版本负结果直接推导为有效方法。

## 已读主源和实际入口

- [ICPL官方仓库](https://github.com/lsh-ahu/ICPL-ReID)，本次clone HEAD
  47c3d128b16c1183cf8aa66cfa76de9eef334bed，LICENSE为MIT。
- [作者主页论文](https://aihuazheng.github.io/publications/pdf/2025/2025-ICPL-ReID-Identity-Conditional_Prompt_Learning_for_Multi-Spectral_Object_Re-Identification.pdf)
  给出冻结主干、学习模态特定适配参数的动机与多谱适配器。完整ICPL还包含文本prompt、
  身份prototype及相互对齐训练；本文不把一个适配器称为ICPL复现。
- [RGBNT201配置](https://github.com/lsh-ahu/ICPL-ReID/blob/47c3d128b16c1183cf8aa66cfa76de9eef334bed/configs/Multi_Modal/ICPL/RGBNT201/ICPL.yml)
  MODEL.VISUAL_BRANCH=multi，MLP_ADAPTER=True，ADAPTER=parallel_adapter，
  MLP_ADP_MID_DIM=768、scale=1，B64/K4、50epoch，eval period1。
  因此不能按README的low-rank文字就声称这个实际配置使用128维或瓶颈低秩。
- [make_model.py](https://github.com/lsh-ahu/ICPL-ReID/blob/47c3d128b16c1183cf8aa66cfa76de9eef334bed/model/make_model.py)
  :174–212建立三个视觉模型分别处理三模态；:231–250实际调用parallel adapter构造器。
  model_mm_adapter.py虽定义同模型内RGB/NIR/TIR专属MLP，但该文件不是此入口调用的构造器。
- [parallel adapter源码](https://github.com/lsh-ahu/ICPL-ReID/blob/47c3d128b16c1183cf8aa66cfa76de9eef334bed/model/clip_adapter/model_parallel_adapter.py)
  :253–280是down→ReLU→up，up权重/偏置与down偏置置零，初始残差为零；
  :225–233将适配分量与MLP结果相加。
- [optimizer源码](https://github.com/lsh-ahu/ICPL-ReID/blob/47c3d128b16c1183cf8aa66cfa76de9eef334bed/solver/make_optimizer.py)
  :14–20显示还训练prompt/classifier/image projection/ln_post/camera embedding/BN，
  不能把ICPL完整训练称为仅训练MLP适配器。

## 本次移植边界

V23从已有合法fold source权重初始化，不读取ICPL已发表RGBNT201训练权重。
每个CLIP尾部阶段9/10/11完整输出之后、原角色算子之前，按RGB/NI/TI分别运行
x_m + U_m ReLU(D_m x_m)。每阶段三个MLP，在CNN/T/M三个角色之间共享这组参数。
D:768→128，U:128→768，含bias；U和两个bias为零，D为Kaiming初始化；
固定scale1、无新增dropout或额外loss。中间宽度128沿用本项目既有adapter宽度，
不是ICPL配置、rank扫描或V19私有尾部参数微调。该实现不是原论文MLP并行插点的逐行复现。

三个阶段×三模态×(768×128+128+128×768+768)=1777536个新参数/36个tensor。
两端都有相同模块及初始state；控制端固定零适配器，候选训练适配器。
原Signal、CLIP主干与尾部均冻结，原角色模块继续训练。
控制/候选可训练参数7841292/9618828，故不能声称排除了参数容量影响。
主目标成功之前不运行参数匹配消融；只检验该具体新路径的完整主实验效用。

RoDI当前官方仓库只列README和assets，未提供可直接执行的训练代码；
其85.3/87.9来自论文DINOv3版本。ICPL/本次CLIP适配方案均不等于该结果，
所有当前SOTA/协议限制保持原交接文档，不据此宣布达到目标。
