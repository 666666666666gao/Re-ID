# V23 experiment tracker

更新时间：2026-09-05T22:04:05.629104+08:00；状态M0_RUNNING_ORIGINAL_PROCESS。

| ID | 内容 | 状态 | 证据 |
|---|---|---|---|
| V23-SOURCE | ICPL实际入口与移植边界 | DONE_SOURCE_ONLY | docs/SPECTRAL_ADAPTER_SOURCE_NOTES_2026-09-05.md |
| V23-T0 | 五项远端CUDA合成契约 | DONE_PASS | evidence/trifusion_v23_t0_20260905.json/log/xml |
| V23-M0 | 三fold两端54前向/116优化步 | RUNNING | 原PID44684；尚无完整M0结果 |
| V23-Q1 | 三fold两端20epoch完整五输出 | WAIT_M0_PASS | 原进程仅在M0通过后自动继续 |
| V23-D1 | 141-fit refit与30-dev | NOT_QUALIFIED_NOT_RUN | 无Q1科学资格 |

22:01:32.893788+08:00启动原进程44684，screen v23_spectral_adapter_9f4a10b。
执行commit9f4a10b6162b9658ba103cd92466411ebb6ccd8f；
配置SHA9859342d044e8bb0b630bc8f6af345ee95dbf1eb445f7a2bffce4173df8c01fa；
计划SHA7877e7fe6a857b965cac701220be387da4774c3e88dfa3682938df8dc32e4997。
启动前GPU空闲24126MiB、使用1MiB，18项执行/初始化文件按实际完整字节核验。
T0实际9.800828秒，5passed/0failure/0error/0skip，
6个合成模型、1个独立适配stage、6步toy优化；真实数据/项目训练步0。
保留3条timm弃用警告；这些不属于科学结果或真实M0门。

run目录 /root/autodl-tmp/trifusion-v2/artifacts/trifusion_v23_spectral_adapter_seed42_9f4a10b，
同名.log/.exit/.launch.json；不用观察调用超时判断失败或重启进程。
M0预计22:06–22:09附近结束；通过后完整Q1预计75–100分钟，以实际epoch速度更新。
首次M0窗口之前完成证据同步和终态比较准备，不高频查询。
目前没有V23科学指标，没有D1/dev/official访问，旧版本封存与总目标未达状态不变。
