# 用户恢复训练：201单项角色分类入口对照

2026-10-13。用户新指令“继续训练”撤销此前训练停止；Goal已active。先执行§936实现/§937配置链已核查的一个201 semantic/seed42条件，不把恢复授权写成旧失败改判或三集成功。

唯一干预：角色分类接收L2(h)后仍经现有detached train-BN/classifier；角色Triplet保持raw h，global作者任务、读取边界detach、原CNN/Transformer/Mamba和结构化读出、gain初值、推理L2(g+gain*c)不变。新wrapper不增加参数。它是有限训练接口诊断，不作为论文原创模块或完全同检索几何。

来源/数据/预算：固定公共CLIP、新camera/head与原共享适配/semantic角色；不加载已训练Signal ReID或本次M0权重做温启动。沿用作者来源201 B64/K8、soft-margin Triplet、ID权重0.25、原作者optimizer/LR、原50轮与实际AMP路径，不更改原环境。每端同一mAP-best报告mAP/R1/R5/R10，同mAP选择末次并列best，不跨轮拼列。官方集已消费，当前是开发比较。

封存raw semantic控制74.3749/78.8278，独立global-only74.2967/78.9474，完整Signal80.3029/85.1675。主要推进线相对匹配raw semantic mAP至少+0.5点且R1不降；该阈值非统计显著性，且未超过完整Signal时总体目标仍未完成。车辆与多seed不加入此队列，未接近前不调整gain/margin/层数或换seed救分。

执行一次prepare→真实8步M0→fresh50→首次严格重载评价。prepare完整binding与旧raw initializer比较，只排除实际新增的分类/metric policy元数据及原架构/entry/scope记录；模型state、视觉/camera初始化、参数量、cfg/采样与分段保持。M0复用原8有效更新、所有trainable至少一次非零有限梯度、BN计数8、视觉/camera改变、冻结参数不变、严格重载容差1e-5。M0不合格不进入正式训练，原失败保留；不从M0接训50轮。

资源固定26物理GPU0/1，一个原双卡分段进程，前6层GPU1，后6层/heads GPU0；不使用25/2/3，不设置或查询温度/功率，不安装、重建、抢占或杀其他任务。warm已有conda规格不变，既有ledger/witness适用，当前实际production M0仍须运行。未配置W&B/通知，不发送外部消息。

启动前实际memory-only preflight：GPU0/1各15MiB，磁盘3308871680B。新campaign事前保留2GiB原reserve +2×384MiB（M0/正式best）+128MiB辅助，共3087007744B；预计额外两份权重实际约717MB。所有旧best/来源保持，不因预算删除唯一依赖。正式训练原逻辑覆盖best_map.pth；M0在首次full严格验收与原verify前保留，之后可按SHA转存/退役重复探针，不放宽旧门槛。

源SHA封存后独立控制器一次启动，各阶段等待上一阶段原child实际退出，不重启失败阶段或因前端观察超时重新起队列。训练检查按预计M0完成节点约2分钟及full50约60分钟；长任务观察180–300秒，临近ETA再读取，不连续轮询。真实进程启动后记录PID/命令/CUDA与原生EXIT；控制器异常由外部唯一supervisor记录真实退出，不按campaign RUNNING残留重跑。

完成后做保存距离的全query修复/新增错误、身份AP、首位/整体排序与完整50轮曲线诊断；如不达推进线，依据实际结果判断分类入口作用，不自动扩车辆或调参。必要时按原同best提取G/C/F，另行登记，不能用toy/配置PASS冒充检索成功。代码/GitHub与唯一master及时同步，原失败/原best保护。目标仍是三集全面超过Signal和针对真实问题的有效原创机制。
