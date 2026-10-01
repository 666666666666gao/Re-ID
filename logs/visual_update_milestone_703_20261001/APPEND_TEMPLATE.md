

## §41.702 视觉更新配对面板的首批完整端与连续队列

记录时间：{{RECORDED_AT}}；下列进度来自一次保存于{{SNAPSHOT_AT}}的顺序文本快照，不冒充文档写入时的实时状态。只使用gaob@172.19.12.138:2026及现有tri_reid环境。既有十二端、seed42、完整50轮、官方fused mAP选best、严格重载及完整图库过滤均保持不变；没有改变学习率、选点门槛或追加种子。

### 完成与运行情况

父队列确认完整端{{PARENT_COMPLETE}}/12，子任务自身已验证完整端{{CHILD_COMPLETE}}/12。未完成端不得使用中途best填入正式结果。下面的正式行都来自已完成m0/train/evaluate且子验证VERIFIED_COMPLETE、实际退出0的一份mAP-best权重；父进程尚未收讫的状态另行保留。

{{PROGRESS_TABLE}}

{{FORMAL_TABLE}}

### 当前能回答的配对问题

{{PAIR_TABLE}}

C1仍要求两个独立读出设置中的六个low_lr−frozen配对全部满足原门槛；C2仍要求三个low_lr下roles−独立global_only配对全部满足原门槛。当前完整门尚未验收。单个端或单个数据集的局部差值不替代完整跨数据集结果，也不构成SOTA或稳定性结论。旧冻结结果的存储dtype及对应初始化合同不同，不能用来替代本批新frozen对照。

### 队列、观察与证据保存

现有controller PID2342941、workflow wrapper2337321，观察wrapper2372978/observer2372984继续按240秒工作；没有启动第二套观察器或队列。当前CPU完整汇总调用次数{{REPORT_INVOCATIONS}}，状态{{WAITER_STATUS}}。该汇总仍只在全部十二端、真实父子退出0及accepted_matrix=12后执行一次；本次仅封存现有文本和子任务验证，不重新加载神经权重、运行推理、重放距离数组或执行完整report main。

本次证据目录：logs/visual_update_milestone_702_20261001/，原始文本{{RAW_FILES}}份逐字节/SHA核对。封存包含既有观察器不可变快照、父/子状态、训练逐步及逐轮记录、正式回执、原同步701证明、等待器实际状态和本次保存脚本；模型、距离数组及图像留在远端。当前空闲磁盘{{FREE_BYTES}}字节，是本次时间点的free值，不是历史峰值或准确GPU小时。所有222条活动源绑定及原221条绑定、既定manifest/witness/preflight和完整汇总源码继续保持原SHA。

此前公共视觉重置六端的负结果与WARN审计仍然封存；本批普通视觉微调是对照，不包装为创新。整个三数据集超过同协议基线及经资源/协议核实的SOTA目标仍为ACTIVE/UNMET。
