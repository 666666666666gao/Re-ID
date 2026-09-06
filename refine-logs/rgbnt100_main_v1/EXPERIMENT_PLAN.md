# RGBNT100 全训练集与官方固定主比较 v1

登记时间：2026-09-06T17:44:01.445667+08:00。状态 **PREPARATION_ONLY_NOT_RUN**。
前置为原三角色内部三折完整终态及全量执行侧核验：mAP89.524175→91.316540，
原五条件全部通过，50身份bootstrap下界+0.951777609pp。该内部比较已经结束，不再训练或换选其三份权重。

## 要验证的主张

唯一主要主张：在全部50个官方训练身份上，固定训练完成的原完整三角色模型，
能否在全部官方query与gallery上较同源Signal稳定改善身份检索。

内部跨身份正结果是开展这项新主结果的依据，不是官方成绩。
保留RGBNT201/MSVR310负结果；本次不晋级其失败版本，不做角色/容量/融合消融或多seed。
额外角色训练和计算的替代解释尚未排除，在主目标成功后再按用户限制处理。

## 固定数据、模型和训练

- 训练：RGBNT100全部8675 montage，50个身份501/503/.../599；不使用内部三折已训练权重。
- 官方query：全部1715个文件，50个偶数身份502/504/.../600。
- 官方gallery：全部8575个文件，与query相同50身份；保留全部不同身份干扰。
- 评价只排除same identity AND same camera，包含同camera的不同身份负例。
- 三模态切片沿原RGB/NIR/TIR各256x128，输入高128x宽256；camera1–8映射0–7，view=-1。
- seed42；B64/K8/workers4；三模态同步几何、独立erase，沿现有已验证loader。
- Signal从同一CLIP预训练重新训练全部50身份，固定30epoch；原作者Adam、
  task LR0.0007、CLIP LR5e-6、5epoch warmup/noise-cosine、原ID/Triplet/Patch及已登记stable Gram。
- Gram在FP32计算，sqrt(abs(det).clamp_min(1e-12))保持R2定义；不得冒称未修改原零点公式。
- 从该固定epoch30 Signal构造原完整V8 CNN/Transformer/Mamba，角色/七分类头fresh seed42，
  冻结Signal与共享CLIP；固定20epoch/5warmup，AdamW0.00035/wd0.0001，沿原七ID/Triplet权重。
- 不使用V23模态MLP、V24原型、Router/HFER、蒸馏、私有尾部或新检索融合。
- 总成本分别记录Signal30epoch与新增角色20epoch；不作为等总训练预算比较。
- 50类输出头变化须在本次M0实际核对参数、梯度和保存/重载，不能借33/34类M0替代。

## 执行次序和证据

| 阶段 | 内容与完成条件 | 状态 |
|---|---|---|
| T0 | 固定全18965文件的路径/标签/SHA/RGB montage头；全部1715query合法正例及排除集合，和原训练inventory一致 | NOT_RUN |
| Signal M0 | fresh全50类，完整第一个训练epoch，195项梯度有限/有效更新/无AMP下降，source前向与严格重载一致 | NOT_RUN |
| Signal B0 | fresh固定30epoch，全训练记录覆盖，逐步loss/索引/AMP/LR记录，保存唯一epoch30；此时不读官方模型分数 | NOT_RUN |
| 角色M0 | 从上述固定Signal出发，8步B64/K8容量及fresh角色固定batch100步；203梯度/冻结state/Signal parity/全五输出严格重载，excess ratio<=0.1 | NOT_RUN |
| 角色主训练 | fresh同M0初态但不加载其已训练权重，固定20epoch/全部训练身份，保存唯一终点 | NOT_RUN |
| 官方完整比较 | 两模型和全部训练回执固定后，各一次完整1715query与8575gallery前向；五输出及独立Signal parity | NOT_RUN |
| 终态核验 | 所有权重/数组/每query全部8575图库位置的排序、8575×1715×5距离、8575条query-output AP/CMC、50身份bootstrap和每条实际训练更新 | NOT_RUN |

T0仅冻结公开基准文件和标签、检查文件头及已有训练SHA；不执行模型或产生检索分数。
工具tools/build_rgbnt100_official_protocol.py已准备，当前仅本地AST通过，尚未读取新官方文件。
Signal全50类训练驱动tools/train_rgbnt100_signal_main.py已准备且仅AST通过；角色训练、官方检索和全部终态核验入口仍待完成绑定。当前没有新训练配置或启动。

角色M0的100步使用fresh角色；正式角色训练再次fresh初始化。
Signal前缀要求全部官方query/gallery特征与距离逐元素等于对应独立Signal，
沿已核验exact_signal_forward处理冻结SIM的mm/bmm数值路径，不放宽误差门。
任何模型/优化器工程错误保留原现场并诊断，不因观察超时重启，不为通过门临时改参数。
科学负结果完整报告，不按官方指标调epoch/seed/branch/权重重跑。

## 固定终点和完整报告

固定Signal epoch30与角色epoch20后，才允许官方模型前向和指标计算。
以所有1715 query的mAP、Rank-1/5/10、所有50身份的AP变化、Rank-1修复/新增错误为报告。
所有输出的完整排序和特征/距离保留；按输出分别写排名gzip文件，
预计官方五输出总排名规模大于既有单fold文件，分输出保存便于全量核验及GitHub单文件发布。
模型、张量、图像运行均在远端；本地只做文本/JSON/NumPy标量核验和传输。

本官方主比较的Signal增益支持条件为：
1. fused mAP至少比同协议Signal高1.0个百分点；
2. 三个完整分支mAP均不低于Signal；
3. 全50身份query加权bootstrap下界严格>0，seed42/10000次/2.5%linear；
4. fused mAP严格高于baseline及三个完整分支。

这里没有三fold门，因为本次是完整训练后的单一官方测试，不伪造“每fold均通过”。
mAP与Rank-1分别报告；不因一个指标最高而声称另一个也最高。

SOTA是独立要求：按相同数据协议与输入资源，与已核对的公开表比较。
现有归档参照包括RoDI-CLIP RGBNT100 88.5/97.6和PMKD-DINOv2 91.6/98.0；
后者用更强预训练及多阶段蒸馏，不能描述为等资源对照。
本轮官方页面可读，PMKD大PDF通过网页工具超过大小限制、直接下载断连；
这些数字仍归因已有原表归档，尚未在本轮重新提取原表，不编造新的核验。
同协议本机Signal增益、CLIP文献竞争力、跨资源公开SOTA三层分开；单seed不代表跨种子稳定性。
RGBNT201 dev65与官方主目标继续未达，不能用本数据集成绩替代。

## 预算、磁盘和边界

新产物计划放在/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_trifusion_main_v1_seed42_20260906，
使用刚清理后约26.62GiB空闲的数据卷，独立目录，不覆盖原OOF、M0、B0或失败证据。
启动前核对该卷>=8GiB可用、GPU无其他训练，按各阶段真实文件大小更新剩余量。
预计Signal主训练约25–40分钟、角色约50–60分钟、全量评估/核验约10–20分钟；
以全50类M0的实际速度修正，保留所有原30/20epoch，不缩减query或gallery。
长任务按估计阶段终点观察，180–300秒以上间隔，不逐epoch轮询。

独立审计服务限制仍在，没有新独立verdict；执行侧核验如实标记，不阻止已授权的工程执行。
当前仅登记下一主结果，未访问官方模型结果、未训练、未新增checkpoint。
