# V25 experiment tracker

更新时间：2026-09-06T23:10:34.125041+08:00。训练执行97468dd；原训练PID112550于23:08:42仍确认活跃。
完整M0 PASS，Q1原过程继续。最近完整进度观察23:04:45：
36/120epoch，fold0-control20/20并完成终点检索，fold0-candidate16/20；
其余fold由同一原过程顺序执行，完整三折比较尚未完成。

| 阶段 | 状态 | 证据 |
|---|---|---|
| SOURCE-METADATA / T0 | COMPLETE_PASS | 全3360批、全部source身份/图像 |
| M0 | COMPLETE_PASS | 48只读batch+116优化步，203梯度、overflow0、超额比0.0398826204 |
| Q1 | RUNNING | 原PID112550；全部六端固定20epoch；不根据首折调参 |
| 终态完整核验 | QUEUED_WAITING | screen v25_terminal_verify_97468dd，wrapper114796活跃 |
| D1 / official / ablations | NOT_QUALIFIED_NOT_RUN | 原门槛保持 |

终态核验队列于23:08:42建立，每180秒读取原训练terminal是否完成。
等待期间不读取任何检索tensor。原wrapper写入完整terminal后，自动单次执行
tools/verify_v25_complete_terminal.py；CUDA_VISIBLE_DEVICES为空，全部数组核验在服务器CPU。
无自动重试，不重启训练；结果写入run_dir/terminal_verification.json，
完整日志terminal_verification.log，退出值terminal_verification.exit，队列状态terminal_verification_queue.json。
队列活跃不代表核验已执行或PASS。

预计原六端在23:45–次日00:00完成，再等待最多180秒进入核验。
数据盘23:04:45 free21.863GiB，
系统free10.365GiB。
24个已删冗余权重仍缺席，12个保留模型路径及大小匹配；本次新删除0。
