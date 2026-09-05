# V21 Experiment Tracker

状态：T0_PASS_M0_RUNNING；更新时间：2026-09-05T16:36:44.082755+08:00。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V21-T0 | 三项CUDA数学/AMP/BN单测 | DONE_PASS | evidence/trifusion_v21_t0_20260905.json |
| V21-M0 | 六模型配对、两端容量、SAM固定batch100步 | RUNNING | 原PID32331 / screen v21_sam_3c39351 |
| V21-Q1 | 三fold ordinary40/SAM20完整训练 | TODO | M0全部通过 |
| V21-AUDIT | 全量终态复算及独立审计 | TODO | 六端终态 |
| V21-D1 | 141-fit refit与30-dev终态 | LOCKED | 全五个Q1科学门通过后另行冻结 |

仅seed42远端GPU；B64/K8、原V8结构与七路ID/Triplet保持。
对照普通AdamW40epoch，SAM20epoch/rho0.05；前向/反传次数匹配，各3360次，
总5040优化步/6720对前向反传。数据暴露/更新次数不同，不能声称全部条件相同。
全部3126gallery/571query/21身份、三fold五路及负结果完整报告，最终checkpoint
严格重载；无中途选择或扫描。M0不通过不执行Q1，Q1失败不晋级D1/dev/official。
T0三项数学检查已通过；M0执行中，Q1尚无结果。计划和固定配置执行前SHA绑定。
V20已独立审计为工程PASS/完整性WARN/科学FAIL并封存；用户整体目标未完成。

V21远端三项CUDA数学测试全部PASS，pytest4.28秒（总管理耗时5.818秒），
源码执行commit3c393510f0e0a31bad602af8dd618a8dcdfe6ae6。
解析SAM梯度、一次普通AdamW、AMP scale消去、BN第一遍统计精确保留均通过。
测试只使用合成数学张量，无数据集/项目训练；测试内部有toy optimizer更新，
不能把T0写成所有优化器step总数为0。原始回执见evidence/trifusion_v21_t0_20260905.json。

原始screen 32330.v21_sam_3c39351于16:43:57CST启动，训练PID32331，目录
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v21_sam_seed42_3c39351，
日志为同路径+.log。16:46:20实际观测PID存活2分23秒，三fold配对preflight已
写入、M0尚无最终回执；GPU6630MiB/100%。当前M0执行中，尚无Q1检索结果。

启动管理脚本使用screen -DmS（大写D），该形式detached但不fork返回，
因此subprocess.run等待原screen结束，SSH管理读取超时退出1。
这不是训练执行失败：随后单独SSH实际确认原PID正常推进，未重启或复制训练。
权威启动观测evidence/trifusion_v21_launch_observation_20260905.json，
管理故障单列evidence/trifusion_v21_launch_transport_20260905.json。
若原管理脚本在训练结束后写出trifusion_v21_launch_20260905.json，其
launched_at/status只能视作延迟管理记录，不得覆盖screen实际启动时间或终态。
后续启动工具应使用小写-dmS，此处保持原进程及执行源码不动。

冻结config SHA f2f47acf54790dc69d9b0d7b5c94dcf9ecfd37bfcf3dd20dc8251e1e1a3600a3，
plan SHA ba17807f30e294618d2a21907a8fde0da82f24f6dab0d51073be49014cc40f71，
runner SHA deafa4d6d2287928c9143d28f2bfb7f32e303939fa6db547f91219f3d708e0fa。
全部T0绑定逐项匹配；启动前GPU24126MiB空闲，磁盘8,357,470,208字节空闲。
M0原估4–8分钟，下次有意义观测16:49CST；如M0全通过，原进程自动继续
ordinary40/SAM20的完整三折，Q1约130–160分钟，按实际epoch再修正ETA。
没有新训练结果、dev/official/D1或SOTA声明，目标仍active且未达成。
