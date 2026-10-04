# TriFusion 当前 Goal 执行约束

更新：2026-10-05 §41.840。首端完整50轮保存曲线补充核对：47/50轮mAP高于匹配控制，但R5、R10分别35/50、34/50轮下降；正式best仍为E8，推进仍未通过。 正式1/6、M0终态2/6；最近模型snapshot02:58:48，原native从02:58:05进入fresh50，observer37914首次03:36:44。只26GPU0/1，无功率/温度操作，Goal ACTIVE / UNMET。

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

当前原supervisor3606472/startticks39300650队列正式1/6、M0终态2/6。首201semantic E8为74.9363/78.7081，相对匹配控制mAP+0.5614/R1−0.1196，未通过推进。02:58:48 snapshot记载native train3741795从02:58:05进入fresh50；唯一新observer session37914首次03:36:44.855310、之后240秒。旧66644已退出0，不重启；不要提前读取中间分数。

每端独立prepare核对上一轮initializer→8步真实M0→fresh50→首次strict。M0沿用有限loss、全部训练张量至少一次非零梯度/实际更新、BN8次、native14张量活动、完整state严格重载。不能用CPU witness替代实际M0。失败保持，不据官方结果改seed/LR/gain/margin/batch或堆N2/N3。

主要比较候选−封存同variant职责任务；另报独立global-only和native−semantic，全量15对报告。推进+0.5mAP且R1不降只是项目门槛。每端完整50轮一份mAP-best，所有CMC跟随它；同时保留50轮、query修复/新增错误、身份分布和实际成本。单seed、官方基准已参与研发，身份bootstrap不替代训练种子方差。

## 资源与保存

只2026 gaob@172.19.12.138:2026，/data/gaob/Re-ID/Trifusion，现有tri_reid/data/公共权重复用，无环境重建。只物理GPU0/1一对，前6CLIP层GPU1/后6与heads GPU0，最大并行1。201 B64/K8、MSVR B64/K4、100 B128/K16。预计六端6–7小时，按节点或180–300秒观察，不误把超时当停止、不重复启动。

用户最新指令：不查询、设置、监控或以功率/温度作为门槛。GPU2/3和2025不运行本项目模型。2025 /data2/gb/Re-ID/Trifusion原I/O失败仍pending，不恢复/探测；本地/Desktop/GitHub/2026执行副本SHA核对，不能冒称2025一致。

每个formal仅保留best_map.pth。新队列逐端完整严格验收及SHA封存后只退役该M0探针，失败端不删；最终report验证不可变验收文件/journal，不调用退役binary依赖。六best+最多一probe+2GiB reserve预算4,966,055,936B，必要控制/作者/初始化全部保留。模型、原图、NPY/PT距离留远端；代码、文字、CSV/SVG、SHA同步。

后续只有完整六端和固定诊断闭合后才能决定；优先验证真实同容量、角色完整路径删除重训、强参照与完整流程多种子，不为三个框强保无效机制。目标实际达成前不complete，正常长任务等待不blocked。

首端保存曲线补充§41.840：47/50轮mAP配对为正；R5/R10多数轮为负。依赖轮次不当独立种子，E8主结果和推进判定不变；剩余全部配对与固定best诊断闭合后再决定新干预。
