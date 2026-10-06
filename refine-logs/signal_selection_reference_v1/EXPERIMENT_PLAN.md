# 作者SIM选择作用的匹配直接近邻对照

**问题锚点：** 三光谱身份检索中，新增证据的独立活动和判别性尚未转为稳定净增益；必须区分内容选择、额外容量/头/部署几何及当前角色组织的作用。

**日期：** 2026-10-06。基于§863完整六端和fixed-best诊断、固定数组CPU几何诊断的资格失败，以及pinned Signal `cd1b0a672d1fe642e7608731cb4899a19dda7d51`真实源码。本计划是基础参照，零原创机制主张。

## 研究问题与控制

唯一主要问题：作者SIM中的模态内/间hard top-k二值mask，是否优于相同ModalInteractive、独立var训练头和3072维部署的全patch输入？两臂同容量、初始参数、global/var损失、Adam与作者采样/增强/日程、同一public CLIP及新camera/head；只改变TokenSelection输出是作者mask乘patch还是原patch。不能把SIM整体收益归给mask。

| 臂 | 选择 | 交互与头 | 部署 |
|---|---|---|---|
| masked | pinned作者模态内/间top-k并集的零值mask | 原512维ModalInteractive/MHA/FFN/LN；原global头＋独立1536 var BN/ID头 | L2([global1536;SIM1536]) |
| all_patch | 相同TokenSelection state保留，但直接返回三个patch输入 | 完全相同 | 完全相同 |
| global_only | 没有SIM | 原无adapter的global作者头 | L2(global1536) |

旧F1终态的run_dir实际指向2025 `/data2/gb/Re-ID/Trifusion`，不在本次26权重清单中，2025原I/O pending不得探测。最终固定为三集×三个条件九端，重新在26匹配训练无SIM global-only；旧F1只作历史数值/初始化来源范围的参照，不重定位或冒称已得到其26二进制/距离验收。SIM两臂仍为同容量/同3072直接主配对；与global-only比较还包含额外头、容量、视觉任务和1536→3072部署差异，不能独归内容选择。

这不是完整Signal（不含GAM/LAM/AlignM），也不是当前1536维TriFusion主模型；没有共享adapter、CNN/Transformer/Mamba角色、native CNN、容量MLP或新loss。两臂均为作者机制来源的可核验参照，不能写成原论文所有条件精确复现。新SIM/var头用CPU私有seed42初始化且恢复原RNG；不是作者完整构造顺序的复现。公共视觉、camera及global头与F1作者基础记录单独核对。

源码实情：W_q/W_k只影响离散top-k索引，W_v未使用；六个权重/bias保留初始值并明确requires_grad=False，原作者autograd路径也没有其任务梯度。不是把这些闲置参数算作活动容量。其余交互/var头活动梯度与实际变化由真实8步M0验证。作者zero mask不移除位置，也不对MHA设置key_padding_mask，不能冒称零位置完全不参与attention。

## 执行与评价合同

- 三数据集各global_only/masked/all_patch，九端，seed42、公有CLIP；新camera/global/var头，不读训练好的Signal/ReID权重。
- 作者配方：201 B64/K8、100 B128/K16、MSVR B64/K4；原raw各头CE＋soft-margin Triplet、作者optimizer/LR/WD/增强/50轮日程和原AMP。var联合1536 Triplet只加一次；车辆global仍三套512 Triplet。原F1分数是辅助基础参照，跨它还包含新头、交互、输出维度及联合视觉目标差异。
- 每臂fresh process构造，避免YACS跨数据集残留。完整batch不缩小；只26 GPU0/1，视觉前6块GPU1、后6和全部SIM/heads GPU0，一模型过程。逐步记录实际labels/camera/view及RGB basename序列；它不是原图/增强字节等同性证明。
- 每端prepare→实际8M0→fresh50→首次strict完整评价；M0任一真实失败保留并停止该端，不按正式分数换seed/top-k/margin/gain/LR救分。原反向parity FAIL不追认修复。
- 每端完整50轮，同一mAP-best报告全部四项；主指标201四项、车辆mAP/R1。完整query/gallery与原camera/MSVR scene过滤，所有干扰身份保留，不rerank/测试更新。
- 主masked−all_patch，以及masked/all_patch−新同配方global_only单列；预登记+0.5mAP且R1不降仅为项目推进线。单seed和消费官方基准不能证明稳定性/SOTA。旧F1不重训、不访问25，也不作为当前可重载的严格控制。
- 保存原50轮轨迹、逐step/head损失、actual batch顺序、同一best全query修复/新增错误及身份分布、实测成本、source/weight/distance SHA。M0及formal全state保存，全部消费者闭合后只保留winner/依赖权重。

## 里程碑与预算

1. SOURCE：真实source冻结、两臂state/RNG/容量检查、CPU模块选择路径与活动检查；当前只有草稿，无全模型初始化或M0。
2. 九端真实初始化及M0；同数据集common global state/optimizer coverage/BN更新、原global权重来源与新SIM活动证明。
3. 九次完整50轮＋首strict；预计9.5–12小时、约19–24两卡GPU小时，实际以M0测量调整ETA，仅时间估算。
4. 原保存距离一次完整配对报告；解释top-k作用与SIM整体恢复幅度，不把输出维度或独立head变化隐藏。

存储上界预登记每份best/probe360MiB：九best＋一probe＋2GiB=5,922,357,248B。真实初始化后核实际state字节/探针大小；启动时仍须实际free达门，不承诺目前约5.6GB已通过。只能资格退役已闭合且四指标被保留winner支配的明确自训权重，不放宽门。功率/温度不查询或设置；GPU2/3及其他项目不动，2025原I/O pending不访问。长任务按预计里程碑或180–300秒观察，超时不重启；NN/唯一报告活跃时不热同步source/Git。

## 后续解释

- masked明显超过all_patch：支持作者选择确有作用，后续新区域证据机制须直接超越它。
- 两者相近且均超过F1：只能支持交互/head/几何整体，不能宣布hard选择必要。
- 两者均弱：不证明所有身份证据选择无效，也不因失分增加N2/N3救本基线。
- 各集方向不同：全部结果保留，禁止用201正值覆盖车辆集。

架构、近邻归属及真实头/维度先清楚，再形成原创候选；三个成功主模块目前仍未成立。Goal ACTIVE/UNMET。


## §41.865 execution revision — 2026-10-06T11:02:39.828452+08:00

Queue and the sole all-query CPU report now implemented. Each dataset prepares three fresh-process initializers, verifies both SIM initial states/capacity and the common plain foundation state, then runs each actual8M0 → fresh50 → first strict evaluation. Retire only that M0 probe after accepted full receipt; all nine formal bests stay until consumers close. Six explicitly qualified historical own weights retired, freeing 2127963598B; actual free 6749700096B at retirement. Startup budget unchanged5,922,357,248B and each stage requires2GiB.

The entry now logs M0 per-tensor actual parameter changes, and requires all12 SIM interaction tensors plus var classifier/BN weight to change, in addition to original finite/live/reload gates. Original CPU component8-step evidence remains component-only. Common initializer and BN/optimizer/runtime gates have not run. No source modification after launch. Report requires9/450epochs and all9 comparisons, exact labels/camera/view/basename batch metadata equality; full official arrays, same-best CMC, query repairs/new errors and identity changes, fixed-model bootstrap only.

Original storage qualificationV2 stopped before mutation because its generic loop incorrectly forbade a protected retained winner as well as targets. V3 narrows the protection exclusion to target weights; all winner SHA preservation remains required. No old producer or experiment was rerun. This source revision is root self-checked, not independently reviewed or runtime accepted.
