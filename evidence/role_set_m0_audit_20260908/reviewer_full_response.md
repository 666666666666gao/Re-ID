独立只读审计已完成。**总体 WARN；工程 PASS；确定性核验 PASS；Q1 科学资格未评价。** 审计采用新上下文原生 Codex reviewer，`review_independence: same-family`、`acceptance_status: provisional`。未发现 M0 数字错报、虚构文件、目标来自模型输出或需要修改科学代码的问题。

- **来源与完整执行匹配：** 137 条递归 hash 绑定、29 份原始文本（5,717,997 字节）、三个来源 Signal checkpoint 的身份绑定、780 个独立重建来源 batch、完整 248 步均通过。M0 覆盖 738 条跨端去重来源记录；重复视图仍按原采样定义保留。
- **原始数组和终点重建通过：** 独立复算 4,945,920 个距离元素、全部关系提议/去重/mask/queue/损失账本；最大目标误差 3.2410e-08、最大总损失账本误差 5.7121e-07。六个终点完整/冻结/Signal 状态 SHA 重建匹配；原 CPU 收据 37 文件再次散列匹配。
- **过拟合算术通过：** 加权 CE floor 为 0.5857136327437849，control/role_set 校正末首比为 0.0007014160404002304 / 0.0007016520038614711，均低于 0.1。两个固定 batch 各只有 53 条唯一记录，历史候选为 0，因此这部分不能证明历史 VJP 的过拟合能力。
- **WARN 的具体范围：** 203/203 是原运行累计非零梯度见证；直接完整图比较仅覆盖六个容量端各第 4 步首个历史组、189 个 encoder 张量，记录最大相对 L2 误差 1.6810499980311218e-05。独立核对了范数/组别/比率算术，但原梯度向量、缓存/RNG、初始角色状态和重载前后输出数组没有保存，不能声称全程模型级独立重建。M0 实际历史最大 221 个候选、年龄 5；年龄 8 来自无模型的完整 T0 元数据重放。
- **范围和成本明确：** M0 阶段 614.7843 秒；峰值 allocated 11266.8193 MiB、reserved 12116 MiB；封闭 M0/T0/CPU 产物 222,116,266 字节。角色负关系覆盖与最难 hinge 降权同时改变，不能单因果归于角色多样性。

所有审计产物只写入 [审计目录](C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908)。主要文件：

- [完整审计报告](C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/EXPERIMENT_AUDIT.md)
- [机器可读审计结论及完整输入 SHA](C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/EXPERIMENT_AUDIT.json)
- [248 步独立数组核验](C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/independent_arithmetic.json)
- [独立 checkpoint 重建核验](C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/independent_checkpoints.json)
- [命令、脚本与失败尝试说明](C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/README.md)
- [本回复原文](C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908/reviewer_full_response.md)

本审计未训练、未运行模型/GPU前后向、未改科学文件或进程、未安装包、未读官方图片或部分 Q1 分数；CPU 算术限制为 2 线程。原 wrapper/Q1 在只读观察时存活，未作任何干预。本轮审计不要求补跑、重启或改代码，后续结论继续等待原完整 Q1 终态。