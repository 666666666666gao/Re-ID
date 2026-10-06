# TriFusion 当前执行目标

更新：2026-10-06T12:04:15.690161+08:00，§41.867。完整目标ACTIVE/UNMET。

在RGBNT201、RGBNT100、MSVR310匹配训练/评价下取得稳定净增益，超过独立global-only、同预算V8/V27及资源注明的强参照，形成必要机制和完整流程多种子。工程通过、作者配方、继承Signal能力、单seed局部涨分不当作原创/SOTA。历史goal工具的旧服务器/20轮记录不再作为当前执行指令。

## 当前状态和唯一继续

row_mass_role_transport_v2六端/300轮12968更新与24固定best全query对照已闭合：主0/3，全0/15推进，不救参；五旧权重退役，201slot因R10保留。固定数组CPU几何资格FAIL没有直和结果。旧parity/STOP/失败均保留。

Signal作者SIM masked/all_patch是来源邻近参照，零原创主张：同3072维/同容量/同头，仅改变hardmask，独立global-only无adapter且1536维。公共CLIP、新camera/head、作者raw配方/原AMP/50轮/seed42；车辆原三个512globalTriplet＋一次1536varTriplet，不用旧joint-L2。完整query/gallery/camera及MSVRscene过滤保持。

原866已EXIT1：global201首8M0→fresh50→firststrict接受73.4728/77.1531（27轮best）；masked8M0也通过，正式53更新后评价继承1536断言失败，未出正式指标。当前1/9完成、原report0。只追加与原循环相同的3072/1536提取检查，其他entry AST不变，CPU真实调用链fixture通过不是模型验收。

唯一后继867行政继续，复用原global完整50及两个已接受M0，除entrySHA外fresh初始化全部字段精确匹配才可复用。masked53无checkpoint，fresh50不能称optimizerresume，旧成本保留。余7实际M0/8fresh50＋firststrict，再只一次CPU9端/450轮/全query配对。禁止重跑旧接受global或按官方分数换LR/gain/margin/topk/seed、叠N2/N3。当前新源待发布/一次启动。

## 固定资源与证据

仅gaob@172.19.12.138:2026 /data/gaob/Re-ID/Trifusion，温复用既有tri_reid/data/public。仅物理GPU0/1，一NN，first6块GPU1/last6+headsGPU0；GPU2/3/其他项目不动。功率温度不查/设/监/设门。2025 /data2/gb/Re-ID/Trifusion原I/O pending不探测或恢复。NN/唯一报告活动时不热同步source/Git。观察按预计里程碑或180–300秒；超时不重启。

每端完整50轮、一份mAP-best携同权重全部CMC，不拼epoch/seed；记录曲线、逐query修复/新增错误、身份收益、成本和SHA。官方基准已消费，bootstrap/50epoch不代替完整训练种子。正式global和maskedM0是现依赖；best仅消费者闭合且实际SHA/保留winner核验后才按用户授权退役。所有模型/图像/数组留远端，代码/文本/同一主文档Local/GitHub/26/Desktop核字节，25pending单列。


### 2026-10-06 §41.868 当前执行边界


## §41.868 — training/evaluation metadata repair and explicit four-result reuse, 2026-10-06T14:41:08.275546+08:00

Original867 supervisor/controller stopped EXIT1 at13:45:42 after MSVRglobal full50/firststrict both exited0. Official panel stays3/9 until new administrative reuse acceptance: RGB201 all3 accepted; MSVRglobal best38 mAP50.8388266656/R168.6971247196 has completed50/706 updates but original acceptance failed. Original campaign RUNNING and fulljob PENDING are stale flags, not liveness. Original receipt/log/exit/status remain unchanged.

Actual MSVR metadata had2056rows vs706updates:1350extra rows are50×27 ordered query/gallery evaluation batches. `_eval_batch` for vehicles calls the wrapped `_training_batch`, whose logger omitted a phase guard. Every evaluation block and all training rows checked against protocol; no skipped optimizer update. Entry now appends/increments metadata only while model.training, with converter call/return unchanged. All other entry AST is unchanged. Actual-function CPU RED→GREEN logs2056→706; original accepted_row remains RED by default, explicit canonical706path passes its unchanged count/best/SHA checks. CPU fixtures do not establish new M0/runtime or independent review.

New resume entry uses explicit origins for four existing50/firststrict and four accepted8M0s; only five missing8M0/full50 run. Nine fresh initializer witnesses and existing paired state guards must pass; reused binding fields match except logging-only entrySHA. Explicit sidecar provenance preserves originalraw2056. Reuse acceptance must precede retirement of the still-live MSVR M0 probe. Startup budget after oldprobe retirement:5newbest+oneprobe at360MiB each+2GiB=4,412,407,808B; each stage still2GiB. This accounts only remaining storage, does not lower the reserve. All9best stay through final report; report executes once using all9 explicit batchpaths and origin map, full450epochs/completequery/gallery/all9pairs.

Eleven own closed fixed20 inferior weights actually retired after actual weight/receipt/distance/protocol/author SHA checks, allfour-metric dominance within same method/dataset/protocol/author+Signal state, RAW187/currentprobe verification. Retired379,316,962B; retained31candidate weights/allauthor weights/current50dependencies. Original cleanup helper NameError occurred before QUALIFIED/delete; empty originalfolder verified, separateR2 complete. Historical11 binary replay is retired, original logs/receipts/distance arrays remain. Source check fixture R1 failed on Windows path separator before mutation; R2 pass is source-level only. No model/recipe/seed/topk/AMP/loss/filter/best/tolerance change. Only2026 GPU0/1, oneNN; no25/power/temperature action. New868 not launched. GoalACTIVE/UNMET.


### §41.869 current execution override


## §41.869 — 七端结果、磁盘停机与仅缺失两端续接（2026-10-06T18:01:50.850805+08:00）

本节是§868启动快照的实际终态补充。原868控制器在2026-10-06 17:17:07退出1：RGBNT100 masked的真实8步M0已经通过（173/173可训练张量活动、重载差0、四个BN计数均为8），但开始正式训练前触发原2 GiB磁盘门。masked/all_patch两份full的steps仍为空，未开始训练；all_patch M0也未执行。17:24:05实查原父进程与子进程均已退出，campaign的RUNNING为遗留状态。原EXIT1、stderr、未启动状态和此前866/867失败全部保存，不追认成PASS，不解释为指标失败。

七端各完成50轮并通过第一次严格独立评价，共350轮、13,194次正式更新；八次M0共64次更新。全部CMC跟随同一mAP-best：

| 数据集 | 条件 | best轮 | mAP | R1 | R5 | R10 | 实际更新数 |
|---|---|---:|---:|---:|---:|---:|---:|
| RGBNT201 | global_only | 27 | 73.47275487 | 77.15311050 | 85.88516712 | 89.95215297 | 2649 |
| RGBNT201 | masked | 29 | 72.78543583 | 75.47847033 | 84.80861187 | 90.66985846 | 2649 |
| RGBNT201 | all_patch | 7 | 72.54842495 | 76.67464018 | 86.00478172 | 90.07176757 | 2649 |
| MSVR310 | global_only | 38 | 50.83882667 | 68.69712472 | 81.38747811 | 85.95600724 | 706 |
| MSVR310 | masked | 47 | 53.24525745 | 70.38916945 | 82.06430078 | 86.80202961 | 706 |
| MSVR310 | all_patch | 48 | 53.27061314 | 70.72758079 | 81.38747811 | 87.30964661 | 706 |
| RGBNT100 | global_only | 9 | 84.03193834 | 96.20991349 | 96.61807418 | 97.08454609 | 3129 |

RGBNT100使用作者B128/K16，本端实际3129次更新，不能套用旧B64/K8的6559。它重现同种子F1作者基础结果，不是新种子稳定性证据。masked与all_patch正式分数缺失，不能填入预测值。

RGBNT201主要masked−all_patch为+0.23701087 mAP/−1.19616985 R1；MSVR为−0.02535570 mAP/−0.33841133 R1。两者均未达到事前固定的mAP≥+0.5且R1不下降条件。MSVR两份SIM相对plain约+2.4点同时涉及2,342,400参数、额外head、1536→3072维与目标变化，不能全归于选择。当前是作者来源近邻对照，不是新的TriFusion贡献；不能据此宣布三集结论或SOTA。

MSVR masked/all_patch的best分别在47/48轮，best至末轮仅下降0.09443167/0.04270520点；本批没有旧joint-L2版本的24—26点后期坍退，不能混用历史训练行为。完整CLI训练时间分别1333.252873/1220.149641秒；旧intake中1310.311575/1197.034279字段实际是training.json起止区间（训练循环及epoch评价/保存），不是完整CLI。原字段保留，新增分析说明边界。成本比约1.09270来自共享机器上的单次顺序运行，不是隔离性能基准。

选择器源码复核：模态内/间top-k为并集，未选value置零后仍进入完整长度MHA；没有attention/padding mask，零token仍占softmax质量，训练后的投影bias可能产生共同value。这是静态结构性质；实际mask覆盖未保存，不能宣称所有patch都被选中、top-k等于最终并集数量或已定位失分根因。

MSVR全局原2056行日志、706行语义修订及原manifest保持不变。后续发现706行数据字段相同但序列化key顺序不同；登记的serialization-only路径补充将global报告指向7b16ed8e版本，保持值/顺序不变，并与masked/all_patch逐字节一致。原路径48ed83d5及未修订manifest留存，report原byte等值检查未放宽；没有新增NN、评价或报告重放。

及时清理：此前11份fixed20旧劣权重379,316,962B、三个闭合MSVR M0合计1,061,965,852B、三个冗余/劣global1,048,149,101B；RGBNT100 global M0344,978,388B只在正式first-strict接收后退役；一份旧RGBNT100 role356,154,041B；本次101份验收完毕的旧role池1,519,366,499B。近期累计退役4,709,930,843B，保留作者权重、RAW187、所有七份当前best、当前masked M0、所有非支配赢家及明确重建依赖。旧global同state的复制只证明231张量/缓冲一致，PTH整体schema/字节不同；原二进制退役后不得宣称可按旧文件重放。所有文本与官方距离仍保留。

旧池资格R1/R2各在KeyError/缺first-strict回执处停止，未删除；R3明确排除未验收项。R3本地300秒传输读取超时，但远端17:48:19已写完整资格文件，后续只取回实存文件，没有重跑资格。17:54:35一次退役101份，释放1,519,366,499B，可用3,940,270,080B。资格17:48:19完整核验RAW187；退役步骤不重复187全量哈希，检查保护seal及文件仍存，不能称其又执行了一次完整核验。

新入口tools/resume_signal_selection_disk.py登记为窄续接：复用七份full/first-strict、八份M0及868已通过的九份实际初始化JSON；仅运行RGBNT100 masked/all_patch各自从公开CLIP和新camera/head开始的fresh50，以及缺失的一次all_patch M0。masked复用M0不等于用M0权重初始化正式训练。先实际校验来源与binding（历史四项只允许logging entrySHA差异），保留原374来源seal，单独4文件续接overlay；不改原plan/tracker/source字节。原报告尚为0；仅在九端450轮/十八阶段全接收后调用一次原CPU全query九配对报告。

启动容量预留2 GiB+三个360 MiB保存/探针位置+六份实测59,074,461B的RGBNT100距离数组，共3,634,392,494B；逐阶段仍检查原2 GiB。此为当前保存行为的保守条件预算，共享文件系统额外使用不可由它保证。实际可用必须在启动前再次检查；不能临时放宽磁盘门。此次仅新增队列，不修改模型、forward、loss、Adam、raw/head、采样、增强、AMP256、50轮、seed42、top-k112、过滤、best或容差。

本节发布时新续接尚未启动；root源码自检和既有回执核验不冒充独立审计、新M0或科学PASS。仅26物理GPU0/1，一次NN；不探测25/GPU2/3，不查询或设置功率温度。25历史镜像I/O pending保留。架构与容量归因、必要机制、三集稳定收益及完整流程多种子仍未达成，Goal ACTIVE/UNMET。


## §41.870 — 未启动任务状态字段检查修正（2026-10-06T18:12:15.507340+08:00）

869 supervisor699872/controller699874于18:05:34.940349 EXIT1，18:09:10实查两者均不存。失败在旧任务检查、campaign目录创建及任何NN/初始化前：原868两份PENDING full任务均没有steps键，masked在调用panel.run前遇到容量断言，尚未把full步骤写入状态文件；all_patch从未进入。旧queue初始PENDING任务本来就不写此字段，直接j['steps']是新控制器错误。七端350轮/13194更新及八M0仍有效；新869正式更新为0、报告为0，不能把snapshot的accepted0误解为抹除原七端。原失败/源码c1a70890/overlay及日志保留。

唯一队列逻辑修正为PENDING且not j.get('steps')，同时新独立v2 overlay保存当前来源，原374和v1 overlay不变。真实868两个旧任务经实际AST gate重放通过；空列表通过，非空步骤及COMPLETE均被拒绝。fixture R1错误预期一份旧full带空steps而失败，在归档/文档/NN变更前停止并保留。这不是假设性防御层、模型修复、新M0或独立审计。新870仍只复用7full/8M0/9真实initializer，执行RGBNT100 masked/all_patch fresh50和缺失all_patch8M0，原50/RAW/heads/AMP256/采样/Adam/top-k/seed/过滤/容差/2GiB阶段门与3634392494B启动预算均不变。

本节发布时新870尚未启动，仅26物理GPU0/1，一个NN，无25/GPU2/3/功率/温度操作；无需重跑任何旧科学结果。既有negative科学边界、五十轮曲线/同一mAP-best规则与Goal ACTIVE/UNMET保留。
