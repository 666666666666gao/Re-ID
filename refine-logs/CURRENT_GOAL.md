# TriFusion 当前 Goal 执行约束

更新：2026-10-05 §41.855：五formal/250轮12262步/12可用配对闭合，推进0/12；固定诊断四接受＋MSVR一原失败数组CPU描述，100首次诊断parent/observer已退出0，零重试/训练更新。所有当前角色未同时改善同模型global的mAP/R1，本轮不晋级；待登记唯一新科学干预。退役6已消费历史gallery缓存514949895B、free约3.01GB，best/当前依赖保留。仅26GPU0/1，无功率温度动作，2025 I/O pending。Goal ACTIVE / UNMET。

## 总目标与贡献边界

在RGBNT201、RGBNT100、MSVR310匹配协议下超过baseline及注明预训练、外部资源和完整训练成本的强参照，形成机制必要性、同容量及完整流程稳定证据。普通配方提升、继承Signal、单个seed最佳点或工程检查不代替主方法贡献。完整历史只追加docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md，旧失败不回改。

## 已闭合，不能重复调用

- V6九端§809/固定best§812；角色读取detach六端§822/固定best§823；职责任务六端§835/固定best§836均已完成。
- 当前职责任务六端：logs/global_task_role_v1_20261004_824，全部fresh50/首次strict/唯一18对CPU报告；300轮/12968步；相对独立global的推进0/6。201 semantic/native74.3749/74.4182，MSVR50.5422/50.5523，10084.0903/83.1632 mAP。
- 固定best实测：201角色使同模型global mAP提高0.078198/0.121584；MSVR提高0.000096/0.010216；100下降0.278914/0.802213。100自己fused-best E5/E26与独立global E7不同，跨checkpoint差异单独解释。detail输出非零不证明身份价值。
- 332旧source和187控制输入封存在refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json。原M0二进制已按依赖闭合退役；不能调用依赖这些binary的旧verify/report或重放旧模型。旧parity FAIL/STOP保留，本研究训练不额外修parity。

## 当前训练与固定诊断均已终态，下一科学干预待登记

refine-logs/deployment_metric_role_v1/EXPERIMENT_PLAN.md与SOURCE_SCOPE.json。semantic/native各自三个数据集，共六端；仅改变角色metric feature为当前output.fused=L2(sg(g)+gain*c)。原global作者raw任务、raw BN classification、同一head值/clone buffers、梯度所有权、结构/state/容量、初始化、视觉/camera更新、optimizer/LR/WD/scheduler、soft-margin/gain/seed42、完整batch/增强、noAMP和推理保持上一轮。

车辆从各512维raw模态Triplet改为整1536联合L2，因此同时改变模态联合几何，不能称仅L2。原作者各头loss求和保留，车辆三份joint Triplet等价3倍，同三份CE不变。没有新head、辅助目标或系数。

原supervisor3606472/startticks39300650已04:07:25退出1，其三端闭合及MSVR native M0失败不回改。原100 semantic完成full50后磁盘门阻断首次评价，旧磁盘parentEXIT1保留；新行政continuation完成原semantic首次strict及从未启动native自己的8M0/fresh50/首次strict。当前可用五个正式端/250轮12262步及唯一12对CPU报告均闭合，推进0/12；MSVR native无formal权重。原固定诊断父进程FAILED及两201接受回执保留；两个从未启动100已在新行政父进程504070下首次诊断完成，parent EXIT0。MSVR原失败只做一次已保存数组CPU描述，不补签原接受回执或重跑模型。现在四接受＋一失败数组描述，原341/271和所有科学合同不变，所有本轮producer/observer已终态。本轮角色度量归一化不晋级为通用修复；后续依据全部证据登记唯一新科学干预，不调整本轮seed/LR/gain/margin救分。所有同步必须先完成，再启动；活跃期间不进行remote仓库同步。原训练队列/报告/退役操作不能重复调用。

每端独立prepare核对上一轮initializer→8步真实M0→fresh50→首次strict。M0沿用有限loss、全部训练张量至少一次非零梯度/实际更新、BN8次、native14张量活动、完整state严格重载。不能用CPU witness替代实际M0。失败保持，不据官方结果改seed/LR/gain/margin/batch或堆N2/N3。

主要比较候选−封存同variant职责任务；另报独立global-only和native−semantic，全量15对报告。推进+0.5mAP且R1不降只是项目门槛。每端完整50轮一份mAP-best，所有CMC跟随它；同时保留50轮、query修复/新增错误、身份分布和实际成本。单seed、官方基准已参与研发，身份bootstrap不替代训练种子方差。

## 资源与保存

只2026 gaob@172.19.12.138:2026，/data/gaob/Re-ID/Trifusion，现有tri_reid/data/公共权重复用，无环境重建。只物理GPU0/1一对，前6CLIP层GPU1/后6与heads GPU0，最大并行1。201 B64/K8、MSVR B64/K4、100 B128/K16。预计六端6–7小时，按节点或180–300秒观察，不误把超时当停止、不重复启动。

用户最新指令：不查询、设置、监控或以功率/温度作为门槛。GPU2/3和2025不运行本项目模型。2025 /data2/gb/Re-ID/Trifusion原I/O失败仍pending，不恢复/探测；本地/Desktop/GitHub/2026执行副本SHA核对，不能冒称2025一致。

每个formal仅保留best_map.pth。新队列逐端完整严格验收及SHA封存后只退役该M0探针，失败端不删；最终report验证不可变验收文件/journal，不调用退役binary依赖。六best+最多一probe+2GiB reserve预算4,966,055,936B，必要控制/作者/初始化全部保留。模型、原图、NPY/PT距离留远端；代码、文字、CSV/SVG、SHA同步。

后续在可用五端训练、固定诊断及明确失败/缺失记录闭合后再决定；MSVR native原失败不得为凑齐六端而重试；优先验证真实同容量、角色完整路径删除重训、强参照与完整流程多种子，不为三个框强保无效机制。目标实际达成前不complete，正常长任务等待不blocked。

首端保存曲线补充§41.840：47/50轮mAP配对为正；R5/R10多数轮为负。依赖轮次不当独立种子，E8主结果和推进判定不变；剩余全部配对与固定best诊断闭合后再决定新干预。

历史M0存储退役§41.841：70份通过回执/权重SHA且不属于当前依赖的工程探针已退役807,017,706B；旧二进制直接重放不可再调用。全部正式best/作者/初始化/当前控制保留。完整six及唯一15对报告闭合后再判断下一干预。

§41.842：原六端15对全量CPU报告仍0次，因MSVR native无正式回执不能运行。仅未启动100配对继续；不把新2/2称原6/6，后续汇总披露至多5正式＋1M0失败。历史非优20轮权重按同组最高mAP保留赢家后有SHA/距离/回执证据退役；seed42/当前控制/作者/初始化保留。

唯一当前活动队列§41.843：logs/deployment_metric_role_pending100_20261005_842；supervisor3997841，最多1个GPU0/1任务，100semantic/native原合同。旧837 campaign FAILED终态SHA88e2575a7d3693afc0e9e02d0ed3352e61e0eb02447263b8a5c4ff1612db62f9不改。local observer72982首次04:32:44；未取得该两端M0，不提前称通过。旧4份已消费gallery缓存SHA核对后退役843MB，距离/query/报告/模型best保留。

最新实际收取§41.844：04:33:35 pending100 semantic M0通过，fresh50PID4002738；唯一observer46650首次06:27:54.475551，依据旧同variant7126.702006秒估计。节点前不重复启动/查询。旧72982已退出0；原MSVR native没有formal，不回改或重试。新probe356161472B等自身full50/首次strict验收后退役。

§41.845仅完成三端保存轨迹分析：各50轮global loss/global范数均值对各自控制差0；201多轮mAP正但高阶CMC多负，MSVR semantic 49/50轮mAP负。依赖epoch不代替多种子，原best/门槛不改；模型与当前队列不改。

§41.846部分报告入口REPORT_AVAILABLE_FIVE.py已登记，source-only自复核/AST/封存terminal接线通过，未执行。固定五端/12可用全query对/3缺失对，不改原FAILED/PENDING或report_invocations。仅在100两端实际验收后运行一次，缺失不是6/6完成。


§41.848当前执行与保存边界：更新：2026-10-05 §41.848；100semantic full50/3129步/首次strict已闭合，84.0827mAP/95.1603R1，两推进FAIL。原磁盘parentEXIT1/campaignSHA保留；退役41份历史冗余释放2.725GB。新行政续接native M0299/299通过，07:03:13 fresh50PID150146运行。唯一observer17277首次09:02:14.819772，之后240秒。源339/控制187不变；仅26 GPU0/1，无功率温度动作。4/6正式端，原报告846不执行，新混合来源12对报告仅待验。Goal ACTIVE / UNMET；下方早期状态按对应历史节阅读。


§41.849本轮完成保存文本配对分析；当前实验不改。更新：2026-10-05 §41.849；100semantic已封存全50轮配对：41轮mAP/46轮R1低于raw控制，best→末轮24.34点 vs raw1.81点；记录global损失/范数逐轮相等，不作因果或种子显著性。当前native fresh50PID150146/observer17277不变，首次远端09:02:14.819772，随后240秒；本轮同一handle已确认活动，无提前native查询。4/6正式端，12对混合来源报告未执行；源339/控制187不改，只26 GPU0/1，无功率温度动作，2025 I/O pending。Goal ACTIVE / UNMET；以下旧状态按历史节阅读。


§41.850：更新：2026-10-05 §41.850：六份执行源码SHA核验确认单位范数soft-margin Triplet的理想数学下界约0.126928；不据合计loss判断拟合或监督有效性，不改变当前训练。native原计时器17277首观察09:02:14.819772，之后240秒；正式4/6，五端/12对CPU报告未执行；MSVR native原M0失败保留。仅26 GPU0/1，无功率温度动作；2025 I/O pending。Goal ACTIVE / UNMET。


§41.855当前边界：本节之前所有启动/等待文字按对应历史读取。当前五formal和四接受fixed诊断＋一失败数组描述已闭合，GPU队列/observer均已退出，下一科学干预未登记或启动；不重新执行原报告/NN/退役操作。
