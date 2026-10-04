# TriFusion 当前 Goal 执行约束

更新：2026-10-05 §41.842。原队列退出1：完整正式3/6；MSVR semantic E38 50.5121/67.8511；MSVR native真实8步缺细节stem首层2张量非零梯度，M0失败/full未启动。只登记未启动100两条件继续，未启动，原失败不改。仅26GPU0/1，无功率/温度；Goal ACTIVE / UNMET。

## 总目标与贡献边界

在RGBNT201、RGBNT100、MSVR310匹配协议下超过baseline及注明预训练、外部资源和完整训练成本的强参照，形成机制必要性、同容量及完整流程稳定证据。普通配方提升、继承Signal、单个seed最佳点或工程检查不代替主方法贡献。完整历史只追加docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md，旧失败不回改。

## 已闭合，不能重复调用

- V6九端§809/固定best§812；角色读取detach六端§822/固定best§823；职责任务六端§835/固定best§836均已完成。
- 当前职责任务六端：logs/global_task_role_v1_20261004_824，全部fresh50/首次strict/唯一18对CPU报告；300轮/12968步；相对独立global的推进0/6。201 semantic/native74.3749/74.4182，MSVR50.5422/50.5523，10084.0903/83.1632 mAP。
- 固定best实测：201角色使同模型global mAP提高0.078198/0.121584；MSVR提高0.000096/0.010216；100下降0.278914/0.802213。100自己fused-best E5/E26与独立global E7不同，跨checkpoint差异单独解释。detail输出非零不证明身份价值。
- 332旧source和187控制输入封存在refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json。原M0二进制已按依赖闭合退役；不能调用依赖这些binary的旧verify/report或重放旧模型。旧parity FAIL/STOP保留，本研究训练不额外修parity。

## 唯一当前实验

refine-logs/deployment_metric_role_v1/EXPERIMENT_PLAN.md与SOURCE_SCOPE.json。semantic/native各自三个数据集，共六端；仅改变角色metric feature为当前output.fused=L2(sg(g)+gain*c)。原global作者raw任务、raw BN classification、同一head值/clone buffers、梯度所有权、结构/state/容量、初始化、视觉/camera更新、optimizer/LR/WD/scheduler、soft-margin/gain/seed42、完整batch/增强、noAMP和推理保持上一轮。

车辆从各512维raw模态Triplet改为整1536联合L2，因此同时改变模态联合几何，不能称仅L2。原作者各头loss求和保留，车辆三份joint Triplet等价3倍，同三份CE不变。没有新head、辅助目标或系数。

当前原supervisor3606472/startticks39300650已04:07:25退出1，不重启。正式3端闭合，M0通过3端/失败1端；MSVR native缺detail_reader.stem.0 weight/bias梯度，保留失败/full未启动。旧observer86112正常收取终态退出0。仅未prepare的RGBNT100 semantic/native另登记有限队列，当前未启动，计划见PENDING_RGBNT100_PLAN.md；原FAILED/PENDING历史不覆盖。

每端独立prepare核对上一轮initializer→8步真实M0→fresh50→首次strict。M0沿用有限loss、全部训练张量至少一次非零梯度/实际更新、BN8次、native14张量活动、完整state严格重载。不能用CPU witness替代实际M0。失败保持，不据官方结果改seed/LR/gain/margin/batch或堆N2/N3。

主要比较候选−封存同variant职责任务；另报独立global-only和native−semantic，全量15对报告。推进+0.5mAP且R1不降只是项目门槛。每端完整50轮一份mAP-best，所有CMC跟随它；同时保留50轮、query修复/新增错误、身份分布和实际成本。单seed、官方基准已参与研发，身份bootstrap不替代训练种子方差。

## 资源与保存

只2026 gaob@172.19.12.138:2026，/data/gaob/Re-ID/Trifusion，现有tri_reid/data/公共权重复用，无环境重建。只物理GPU0/1一对，前6CLIP层GPU1/后6与heads GPU0，最大并行1。201 B64/K8、MSVR B64/K4、100 B128/K16。预计六端6–7小时，按节点或180–300秒观察，不误把超时当停止、不重复启动。

用户最新指令：不查询、设置、监控或以功率/温度作为门槛。GPU2/3和2025不运行本项目模型。2025 /data2/gb/Re-ID/Trifusion原I/O失败仍pending，不恢复/探测；本地/Desktop/GitHub/2026执行副本SHA核对，不能冒称2025一致。

每个formal仅保留best_map.pth。新队列逐端完整严格验收及SHA封存后只退役该M0探针，失败端不删；最终report验证不可变验收文件/journal，不调用退役binary依赖。六best+最多一probe+2GiB reserve预算4,966,055,936B，必要控制/作者/初始化全部保留。模型、原图、NPY/PT距离留远端；代码、文字、CSV/SVG、SHA同步。

后续只有完整六端和固定诊断闭合后才能决定；优先验证真实同容量、角色完整路径删除重训、强参照与完整流程多种子，不为三个框强保无效机制。目标实际达成前不complete，正常长任务等待不blocked。

首端保存曲线补充§41.840：47/50轮mAP配对为正；R5/R10多数轮为负。依赖轮次不当独立种子，E8主结果和推进判定不变；剩余全部配对与固定best诊断闭合后再决定新干预。

历史M0存储退役§41.841：70份通过回执/权重SHA且不属于当前依赖的工程探针已退役807,017,706B；旧二进制直接重放不可再调用。全部正式best/作者/初始化/当前控制保留。完整six及唯一15对报告闭合后再判断下一干预。

§41.842：原六端15对全量CPU报告仍0次，因MSVR native无正式回执不能运行。仅未启动100配对继续；不把新2/2称原6/6，后续汇总披露至多5正式＋1M0失败。历史非优20轮权重按同组最高mAP保留赢家后有SHA/距离/回执证据退役；seed42/当前控制/作者/初始化保留。
