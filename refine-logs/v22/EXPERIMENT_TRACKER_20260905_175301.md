# V22 Experiment Tracker

更新时间：2026-09-05T17:53:01.270567+08:00；状态T0_PASS_M0_RUNNING。

| ID | 内容 | 状态 | 证据/依赖 |
|---|---|---|---|
| V22-SOURCE | 全1680batch相机监督元数据与整数复核 | DONE_METADATA_ONLY | evidence/trifusion_source_camera_metadata_20260905.json |
| V22-T0 | 三项CUDA数学/行域/梯度契约 | DONE_PASS | evidence/trifusion_v22_t0_20260905.json |
| V22-M0 | 六模型配对、两端容量、100步过拟合 | RUNNING | 原PID34656 / launch receipt |
| V22-Q1 | 三fold两端各20epoch完整检索 | PENDING_M0 | 固定3360优化步、120epoch记录 |
| V22-AUDIT | 独立实际结果审计 | PENDING_RESULTS | 实际回执后执行 |
| V22-D1 | 141-fit refit与30-dev | NOT_QUALIFIED_NOT_RUN | 五项Q1门全通过后登记执行细节 |

V21已独立审计并封存M0_FAIL，不能据其零检索结果判断SAM检索有害。
V22是三个residual度量目标的单一替换，MCNL两个margin均0.1，原残差系数0.25；
原表示/七路ID/融合与单分支triplet、数据、优化器、epoch和seed保持。
T0已通过，真实M0运行中；尚无M0/Q1最终结果或新checkpoint/dev/official检索。
元数据不是模型性能。整体目标未完成。

## 实际T0与原M0启动

V22三项远端CUDA数学契约全部PASS，pytest3.02秒、总管理3.957356秒；
这是合成loss/autograd测试，模型实例、数据集、项目及toy optimizer更新均0。
全部runner/module/tests/config/plan SHA与冻结回执相同，T0源码commit5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36。

17:51:07.422CST以小写screen -dmS启动唯一原始训练进程34656，
screen34654.v22_camera_negative_5ae096b；17:51:08实际确认PID和完整argv。
目录/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v22_camera_negative_seed42_5ae096b，
日志同路径+.log，退出码将写同路径+.exit。启动前GPU24126MiB free/1MiB used，
磁盘8342507520字节空闲。当前只是M0已启动，没有Q1检索结果或新模型改进声明。
权威T0及启动回执为evidence/trifusion_v22_t0_20260905.json及trifusion_v22_launch_20260905.json。

M0预估4–7分钟，下一次有意义检查17:54–17:55；通过后原进程自动执行完整六端，
Q1初估75–95分钟、约19:10–19:30完成，首个完整paired-fold约18:20，
须按实际epoch速度修正。失败则保留完整M0轨迹并停止Q1，不重启或更换门槛。
