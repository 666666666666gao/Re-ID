# V24 experiment tracker

更新时间：2026-09-06T01:08:02.643895+08:00；状态M0_RUNNING_PREFLIGHT_PASSED。
实际训练进度证据时间：2026-09-06T01:04:14.071192+08:00。

| 项目 | 状态 | 证据 |
|---|---|---|
| source成员关系 | DONE_METADATA_ONLY | evidence/trifusion_v24_source_membership_contract_20260906.json |
| 代码/配置/计划登记 | DONE_BOUND_6a4ac2c | evidence/trifusion_v24_preregistration_20260906.json |
| 远端CUDA T0六测试 | DONE_PASS | evidence/trifusion_v24_t0_20260906.json/log/xml |
| 六端完整预检 | DONE_PASS_ENGINEERING_ONLY | evidence/trifusion_v24_progress_summary_20260906_010414.json |
| 完整M0容量/固定100步 | RUNNING_ORIGINAL_PID52030 | evidence/trifusion_v24_progress_20260906_010414.json |
| 全六端Q1 | NOT_YET_STARTED_IN_OBSERVATION | 同一原过程仅在M0完整通过后进入 |
| D1/dev/official/消融 | NOT_QUALIFIED_NOT_RUN | 尚无M0完整资格或Q1检索结果 |

T0实际6.623946666717529秒，6passed，3条timm FutureWarning原样保留。
T0为7个合成原型对象和1个合成几何变换，无真实模型/checkpoint/数据集或项目optimizer。
00:56:14.521639+08:00启动原PID52030，screen v24_source_prototype_6a4ac2c。
执行commit6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33；18项代码/配置/方案/来源权重SHA启动前全通过。

01:04实际确认原过程存活，GPU5362MiB使用/18766MiB空闲/100%；exit不存在。
三fold两端完整预检已完成，完整初始model/memory及8个双视图batch/五输出SHA严格配对。
真实参数98,800,141/可训练7,841,292/203tensor，两端相同；新增推理参数0。
source/heldout94/47隔离；108原型及真实相机成员关系与完整标签普查一致。
六预检模型共12504条source特征前向、102次source batch前向，另96次双视图预检前向。
原型范数记录最大偏差1.1920928955078125e-07，初始化更新计数/年龄均0；这是记录算术核对，不是本地模型重演。
summary最后保存elapsed332.6613943576813秒，此后容量/固定批次阶段尚未整体写回。
观察中的0epoch/0更新仅指Q1日志，不能推断M0尚无优化。
完整M0原定116更新/232前反传及固定第100步门不变；目前没有完整M0 ratio或结论。
预计启动后10-20分钟完成M0，实际完成后修正Q1的138-180分钟预算；按阶段观察，不重启。
V23保持Q1_FAIL封存；当前固定dev最佳和原总目标未达状态不变。
