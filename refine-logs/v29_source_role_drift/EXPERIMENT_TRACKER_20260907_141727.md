# V29来源角色漂移诊断

状态REGISTERED_NOT_RUN，2026-09-07T14:15:46.836244+08:00。
固定initial/control_final/bounded_final，两个原注册输入条件，1680批/10080前向；无更新/检索。
本地显式枚举/旋转反例T0及AST/F821检查通过；远端CUDA T0/模型前向/CPU完整汇总均未执行。
合同configs/RGBNT201/TriFusion-v29-source-role-drift-diagnostic.json；计划refine-logs/v29_source_role_drift/EXPERIMENT_PLAN.md。
预估90–110分钟前向+5–20分钟CPU，数组3.115GiB，启动至少6GiB可用。
保持V29 Q1_FAIL0/5，不开新loss或超参数扫描；三数据集目标继续。
