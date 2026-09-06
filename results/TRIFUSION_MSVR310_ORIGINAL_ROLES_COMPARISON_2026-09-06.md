# MSVR310 原三角色完整比较：工程核验通过，五项科学条件均未通过

记录时间：2026-09-06T08:53:29.912750+08:00。状态 **COMPLETE_COMPARISON_SUPPORT_FAIL_AUDIT_CLOSED_WITH_LIMITS**。
正式终态08:39:25.099313+08；wrapper75993 exit0，08:42:26确认进程结束/GPU空闲。
R3执行1ff7e2d，原第0折训练执行1c444cd；只复用其260更新与已验证完整特征，后两折新增520更新。
原失败run的RUNNING JSON、日志、receipt及checkpoint保持原字节，未重训第0折。

## 完整同协议结果

只使用MSVR310官方training部分构造的三折身份隔离内部比较：600query/60query身份，
1032gallery/155heldout身份，95个单scene身份仍作为真实干扰图库；不是官方591query/1055gallery结果。
原scene过滤、三模态128x256/B64/K8、seed42及固定epoch20均保留；未读取RGBNT201 dev/官方test。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 | 相对Signal mAP |
|---|---:|---:|---:|---:|---:|
| baseline_only | 53.129381 | 63.000000 | 77.000000 | 82.833333 | +0.000000 |
| fused | 52.117390 | 60.833333 | 75.333333 | 83.166667 | -1.011990 |
| cnn | 49.707348 | 59.166667 | 75.500000 | 80.833333 | -3.422033 |
| transformer | 50.332407 | 59.166667 | 75.666667 | 82.833333 | -2.796974 |
| mamba | 50.791740 | 59.166667 | 76.166667 | 82.833333 | -2.337641 |

CNN/Transformer/Mamba是3072D Signal+1536D角色残差的完整4608D扩展分支；
不把此表称为残差单独检索成绩。fused7680D高于三个扩展分支，但仍低于Signal3072D。
当前三角色均稠密执行，非Sparse MoE；共享冻结tail参数仍在三条路径分别计算。
无V23模态MLP、V24原型、Router/HFER、额外投影/对比/MCNL；原V8三角色结构及七组监督不改。

| fold | Signal mAP | fused mAP | 差值 | query/gallery |
|---|---:|---:|---:|---|
| 0 | 49.311695 | 51.717208 | +2.405513 | 210/360 |
| 1 | 47.942265 | 44.182542 | -3.759723 | 207/349 |
| 2 | 63.377725 | 61.552100 | -1.825624 | 183/323 |

五项原登记条件全部false：总体fused至少+1pp、各fold非负、三个完整分支均不低于Signal、
60身份query加权bootstrap下界>0、fused在五输出严格最优。原seed42/10000次、linear2.5%下界
**−2.9391095539708862pp**。这是未通过增益门；没有据此宣称统计上显著劣于Signal，未计算该主张所需上界。
60身份中30改善/26下降/4不变；query级fused267改善/285下降/48不变，Rank-1修复24/新增37。
所有60身份及全部600query均保存在原JSON与完整错误普查，不以少数例子代替全量结果。

## 已排除的工程原因与诊断边界

原SIM冻结requires_grad后，PyTorch2.5.1非连续输入投影从mm切换到bmm，造成小数值差；
该问题已通过可逆操作诊断和完整360特征/210x360距离核验修复。推理helper仅用一个共享原数据的
临时投影权重视图恢复B0 dispatch，整个调用no_grad，原注册参数flags/state不变，训练代码不改。
三折终态Signal特征与距离全部逐元素等于B0。此次科学失败不能归咎于那一项已修复的基线差异。

固定等能量拼接仍让残差承担最终一半相似度：s_fused=0.5s_Signal+(s_C+s_T+s_M)/6。
冻结原Signal可保留基线输出，却不保证新排序改善。本次观测到三种扩展分支均低于Signal，说明
新增角色表示在这些未知车辆身份上尚无稳定增益；不能只靠训练梯度/低损失宣称获得互补身份信息。

三个source折全部103/103/104身份与672/683/709条记录均实际曝光。全部780步同身份无序正对
174720对，其中跨scene45539对，占26.0639880952%。scene与RGBNT201 camera定义不同，不直接类比比例。
本次fused37个新增Rank-1错误，仅10个最先错误身份与query同camera、4个同scene；
Signal全部222个Rank-1错误中同camera85个，fused235个中同camera83个。
这些是完整排序关联描述，不支持照搬RGBNT201的同相机新增错误比例来认定车辆失败的唯一原因。
局部空间、光谱/视角信息的具体混淆机制仍需新干预验证；没有由这些统计推出某个卷积核或学习率为主因。

## 训练与核验成本

固定原三fold各20epoch/260更新，正式总780更新/49920三模态记录曝光；第0折只计一次。
R3新增520更新/33280曝光/672gallery记录前向，复用验证过的第0折360特征完成首次排名。
R3程序计时694.6885192021728秒，不包含原第0折训练；总研究成本还包含B0前提1950更新、
原角色M0124更新、一次完整1440记录四路径诊断、576次SIM记录操作诊断及720条full-role验证。
这是额外三角色训练与推理成本，不是与Signal等计算预算比较。
总/可训练参数来自相同M0架构：fold0/1为99,065,869/8,076,300，fold2为99,095,053/8,102,412；
本次训练峰值reserved6030/6030/6146MiB。未测当前完整FLOPs或独立推理延迟，不以参数量暗示低计算成本。

远端终态核验8.637015929445624秒：三checkpoint内容/state/fold标签、原失败文件、全部20项目/17Signal
源绑定、三折15组保存特征/距离/全排序、所有receipt/training文件一致。15组距离重算均逐元素相同。
本地JSON/stdlib/NumPy核验0.4124450999661349秒：全部3000query-output、780训练步/60epoch、
60身份与bootstrap重新计算。最大指标差1.4210854715202004e−14、bootstrap差8.881784197001252e−16，
epoch均值差0；保存分项损失双精度重组最大差5.650023613412714e−7，未保存AMP中间dtype，
没有添加新的浮点容差晋级门。两项核验0模型/图像前向、0训练；远端张量未下载到本地。

## 当前处理与下一步

封存此固定原三角色MSVR310试验为科学条件失败：不重跑epoch/seed/参数、不做消融、不访问官方test。
独立终态审计run19已闭合：integrity PASS_WITH_LIMITS、engineering PASS、scientific FAIL；A/B/C/D PASS、E WARN、F FAIL。
后续新假设应直接验证未知身份下困难竞争样本的区分与信息保留，结合RGBNT201和MSVR310两份负证据；
不把RGBNT201的21个资格身份或相机正例稀疏当成所有数据集失败的充分解释。
RGBNT100已安装但尚无本项目训练/检索结果，仍按已登记三核心数据集路线独立建立协议与基线。
当前RGBNT201保留dev58.4050/59.3939，dev65和官方SOTA目标都未达到。

主终态JSON SHA c3831a0e95423767cf152e332ed671a8d91d1779c286a0afce8bf522391bfbac。
完整证据见evidence/trifusion_msvr310_trifusion_v1_comparison_complete_20260906.json、
terminal_files_verification、terminal_scalar_verification和complete_error_census对应20260906文件。

## 独立终态审计闭合

2026-09-06T09:36:57.748608+08:00：run19两轮完成。102份不可变输入/25,929,381字节均原SHA；
独立复算全部3000query-output、780训练步/60epoch、60身份及bootstrap和完整错误普查，1.2726868秒，主要指标差0。
第2轮只纠正审计报告的延迟dispatch观察、Linear.cpp行号与MHA/底层matmul区别、远端二进制边界措辞；未重复数值计算或修改实验。
两轮请求/原始回答、首轮及最终报告、原始复算代码/结果/完整输出均在run19归档。GPT同族Type-A、后端未独立证明，
张量/checkpoint只在远端；审计对15数组及checkpoint的验证依赖原源码和远端回执，此证据边界保留。
最终报告EXPERIMENT_AUDIT_MSVR310_TRIFUSION_TERMINAL.md/.json；闭合回执evidence/trifusion_msvr310_trifusion_v1_terminal_audit_closure_20260906.json。
