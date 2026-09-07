# MSVR310 当前历史坐标更新 V1 已启动

新固定实验于2026-09-08T01:32:35.718115+08在远端screen启动一次。执行b4501fa795155eb613fb3bcb07a07c5fae4c6a79，配置SHA32e22d3a0cd858e96430e4b9f9b0a2093c0fdf30f026d8d334ad92d46cfb94aa。代码/配置已同步到GitHub与远端；原封存源码不改。

run `/root/autodl-tmp/trifusion-v2/artifacts/msvr310_fresh_coordinate_v1_seed42_b4501fa`；screen `msvr_fresh_coordinate_b4501fa`。wrapper192704，T0原PID192706于01:32:40退出0，自动启动M0 PID192718。最近2026-09-08T01:36:06.658087+08:00实查原wrapper192704与m0 PID192718及命令行均存在；M0已写入4/6个capacity端，完整248步与CPU尚不能提前PASS。

T0 PASS_FRESH_COORDINATE_CPU_CONTRACT：全部780batch候选过滤/年龄与CPU数学通过，三fold历史候选暴露58133/59505/59984，每折194个有历史更新；模型前向0。T0文件SHA6e1ba4dbf54164d97551ca3770ddec90883e32784f25428a9e4e3e793a9a04a0。所有新源码及继承依赖hash保持。T0并非训练或检索通过。

Control使用陈旧历史坐标，fresh_memory使用当前角色重编码历史坐标。两端额外重编码、固定漂移诊断、旧/新梯度与重复噪声检查匹配，保留历史detach与当前peer梯度。M0预设六端8步＋两端100步，共248更新；全部CPU通过才进入六端1560步Q1及完整CPU。两组原有五项门、seed42、全部图库及MSVR scene过滤不变，官方图片读取0。

启动主盘4,843,671,552B；最近实查4706992128B，GPU读数6252, 0（MiB,%）。复用tri_reid环境，没有安装或重建。原初始化/必要终点/二进制与图像留远端，本次删除权重0；既有清理不重复计数。

预计M010–15分钟，约01:43–01:48完成后CPU；Q1及CPU预计再2–3小时，按实际速度更新。以180–300秒或预计里程碑观察原进程，不因工具超时重复启动。完整Goal保持ACTIVE/UNMET，此处尚无新检索结论。

计划：refine-logs/msvr310_fresh_coordinate_v1/EXPERIMENT_PLAN.md。证据：evidence/msvr310_fresh_coordinate_launch_20260908/。
