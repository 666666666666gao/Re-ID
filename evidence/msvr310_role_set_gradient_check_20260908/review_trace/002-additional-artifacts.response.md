审计完成：总体 **WARN**，确定性核验 **PASS**；`review_independence: same-family`，`acceptance_status: provisional`。当前证据支持“固定初始化上的角色度量目标梯度发生变化”，不支持训练有效性、完整 M0/Q1 或检索提升。

完整报告和证据已写入指定目录：

- [完整审计报告](C:/Users/gb/.codex_tmp/msvr_role_set_gradient_independent_audit_20260908/EXPERIMENT_AUDIT.md)
- [机器可读结论](C:/Users/gb/.codex_tmp/msvr_role_set_gradient_independent_audit_20260908/EXPERIMENT_AUDIT.json)
- [独立远端数组与来源核验](C:/Users/gb/.codex_tmp/msvr_role_set_gradient_independent_audit_20260908/remote_verification.json)
- [本地收据与汇总核验](C:/Users/gb/.codex_tmp/msvr_role_set_gradient_independent_audit_20260908/local_verification.json)
- [独立合成梯度验证](C:/Users/gb/.codex_tmp/msvr_role_set_gradient_independent_audit_20260908/remote_math_verification.json)

| 检查 | 结论 | 关键证据与判断 |
| --- | --- | --- |
| A. GT 来源 | **PASS** | 身份来自真实 source 标签；全部 1,032 条记录的文件名标签、3,096 个模态路径、身份分折、实际采样/历史队列和三份 Signal checkpoint 身份绑定均独立核对。`tools/probe_msvr_role_set_gradients.py:78-89`；`tools/train_msvr310_trifusion_oof.py:27-54`。 |
| B. 指标归一化 | **PASS** | 标量为明确的均值 hinge；原始梯度范数、差值范数和标准余弦均保留，没有把自身输出最大值用作检索成绩分母。均值目标带来的 hardest 权重降低已披露。`tools/msvr_role_set_relations.py:25-56`；`tools/probe_msvr_history_candidate_gradients.py:53-66`。 |
| C. 文件/数字/状态 | **PASS** | 全部 14 个远端原始产物与 intake 的文件名、大小和 SHA 一致；24 批、834,560 个距离元素和 aggregate 全部数字重算一致；两阶段退出 0，原三 PID 已结束。`intake/pipeline.json:24-51`；`remote_verification.json:3154-3231`。 |
| D. 实际执行与梯度见证 | **WARN** | 实际调用新目标、当前/历史 VJP 和直接图校验的路径成立，但历史全部角色逐位一致的计划要求未被完整覆盖；原始 GPU 梯度与字段/RNG 未保存，无法独立重建整次反传。`tools/probe_msvr_role_set_gradients.py:53-68,93-96,104-159`。 |
| E. 范围与资源 | **PASS** | seed42、三折固定初始化、每折 8 批；0 更新/新权重/留出前向。产物 3,643,844 B 在预算内，峰值 allocated 为 11,355.177246 MiB。没有把该小检查称为完整训练或检索收益。`configs/MSVR310/Role-set-gradient-check-v1.json:3-11`；`remote_verification.json:52-153,3226`。 |
| F. 评估类型 | **PASS** | 身份掩码为 `real_gt` 来源；GPU/距离重放为有真实 source 输入的实现一致性代理；合成矩阵测试为 `simulation_only`。没有官方/留出任务性能评价。`tools/check_msvr_role_set_relations.py:62-68`；`tools/verify_msvr_role_set_gradients.py:79-80`。 |

上表代码路径相对仓库；`intake/` 相对原接收目录，独立核验 JSON 相对审计目录。完整报告给出了所有绝对根路径、逐项引用和冻结快照。

独立计算确认三折实际覆盖 **333/672、316/683、350/709** 条 source record，合计 **1,536** 次当前视图曝光、**15** 个含历史 batch、**1,271** 次额外 active 负关系曝光。折间记录存在重叠，合并后是 **738** 个不同 record ID。CNN、Transformer、Mamba 都在 **24/24** 批、含历史的 **15/15** 批中出现大于所记录重复反传噪声的目标梯度差异。我的 float64 标量复算最大误差为 **3.5048772884e-08**；原 CPU float32 复算记录为 **2.9802322388e-08**，两者属于不同精度的重算残差。

WARN 的具体边界是：全部角色逐位比较只在每折**未入队的首批**执行，历史组显式核对的是 fused；真实模型直接图/VJP 比较只覆盖每折第 **4** 批的首个单历史组；重复噪声来自**同图重复反传**；所选梯度域为 **189 个 encoder 张量**，不能据此宣称原 **203 个训练张量/14 项总损失**全部通过。相关运行见证的最大直接图/VJP 相对误差为 **1.273639003e-09**，远低于原 0.005 容差，但这不等于独立 GPU 梯度复现。

我已重新阅读最终结果报告与实施草案：结果报告第 **27–29** 行已明确写出上述覆盖不足和重复反传限制，实施草案第 **21** 行也限定未来新增检查的抽样范围。因此，**报告措辞修正已关闭；原始证据覆盖限制仍保留，整体 WARN/provisional 不变**。均值目标同时改变关系覆盖与 hardest 权重，未来即使有效，也不能从此对照单独归因于三角色提议。

审计共核对 **156 项声明哈希绑定、115 个远端文件**。五个已有 Windows 文件仅为 CRLF/LF 字节差异，实际执行的远端字节均符合注册哈希；报告逐项列明。两次审计器自身的失败尝试、对应脚本和错误记录均已保留在 [audit_attempts.json](C:/Users/gb/.codex_tmp/msvr_role_set_gradient_independent_audit_20260908/audit_attempts.json) 与 `failed_attempts/`；两次都不是实验断言失败。

本轮没有启动训练、读取图像内容、修改科学产物或搬动二进制数组。新未注册训练器/验证器草案及探针结束后的 transport 清理账本不在这次完成态审计范围内，未因此获得实现验收。
