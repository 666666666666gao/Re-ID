# 本轮root源码复核

范围：新的诊断入口和复用的diagnose_native_research_best / run_independent_role_heads调用链；三份实存距离的CPU schema与当前完整receipt。由本轮root执行，不宣称独立模型家族审阅或已运行通过。

- 三端variant仅semantic，原entry.configure/build/load和initializer binding保留；原固定best不重新选择。
- 每个split一次core.forward_features，同时计算g/c/h/f；模型eval、无optimizer、state SHA前后相同及参数无grad。
- 三种完整距离与合法camera/scene评分保留，独立global只读旧数组；继承固定容差1e-5和公式检查1e-6。
- 唯一输出差别：样本统计文件替代不必要的完整特征向量缓存；三种距离全部保留。实际三种tensor载荷192342312B，256MiB预算内。
- 原385源文件不修改，新增入口/本计划/本复核单独封入新source map；原RAW187与三份当前正式best不删。
- 原NN/唯一报告已EXIT0。新真实加载/前向/数值验收尚未执行，静态AST及调用链复核不计作运行PASS。

结论：SOURCE_ONLY PASS；没有新增网络、梯度修复、fallback、安装或训练。实际运行必须另外验收。
