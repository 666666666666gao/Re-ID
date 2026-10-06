# 本地入口原型复核状态

状态：PREPARING，根代理自查；不是独立模型评审或GPU验收。code-review的固定Git差异流程不适用这份未提交的本地新工具；未伪造该技能的双代理审查。

已对照实际源码：默认GlobalTaskRoleHeads推理直接调用evidence_model.forward_features取fused。原型使用同一个接口，捕获g/c/raw_fused而不调用训练头或第二次视觉编码。原角色transport位于私有3×16扫描后、角色输出归一化前，self_only出口返回float32 private H；同参数证据与对照不删除Mamba。

本地py_compile退出0，仅证明语法。参数/buffer摘要、原6份best的原1e-5指标重放、score/Q/质量身份、完整q/g与代码seal尚未实际运行。原始六端生产文件、模型、队列和Git没有改变。

具体修订：避免把previous Capture误注册为额外子网络；output/seal路径在改变cwd前解析为绝对路径；明确检查原supervisor已终态、报告SHA、全部六端及唯一报告。self_only原生内部last_diagnostics仍可能计算peer，记录有效写入为0而不把该字段冒充真实输出。

仍需完成：CPU工具对原模式/两个反事实及同模型global建立完整query修复、新增错误和身份AP分布；聚合既有训练日志与范数轨迹。NN入口当前只产生模式数组/基础指标，不能将其SUMMARY.COMPLETE当成整个四块协议或论文主张通过。

成本边界已修订：同步后的模式时间仍包含捕获/诊断计算和CPU评分，不能当纯部署推理速度。数组初步预留2GiB，并在实际部署前据真实协议计数核对；现阶段没有真实空间验收或诊断seal。

部署前必须用原队列全部终态形成真实六权重/源/报告seal；当前只准备协议和入口。固定状态研究不改变当前训练比较的推进线、选epoch方式或消耗过官方集的事实。
