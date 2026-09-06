# V25 experiment tracker

更新时间：2026-09-06T22:42:56.382205+08:00；原过程于2026-09-06T22:39:45.136173+08:00启动。
代码执行commit97468dd；screen v25_camera_coverage_97468dd；wrapper112548，child112550。

| 阶段 | 状态 | 证据与范围 |
|---|---|---|
| SOURCE-METADATA / T0 | COMPLETE_PASS | 三折两端20epoch，3360batch，全source覆盖 |
| M0 | RUNNING | 22:40:53原PID活跃、GPU3072MiB，第0折两端预检完成 |
| Q1 | PENDING_M0_PASS | 同原进程自动完成六端20epoch，共3360更新 |
| D1 / official / ablations | NOT_QUALIFIED_NOT_RUN | 原五项科学门保持 |

启动前重新核对所有注册文件与Git提交一致、CLIP及6个V12原权重全文件SHA。
空闲GPU24126MiB，数据盘free22.015GiB。
第0折两端模型初始SHA一致，总参数98800141、可训练7841292、203tensor。
本次状态只证明原过程活跃及一折只读预检完成，尚未通过完整M0，尚无Q1指标。
预计22:44–22:49进入M0终态；依据阶段时间修正Q1预计75–100分钟。
在180–300秒或预估里程碑观察原PID，不因观察超时重启。
日志：/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd.log
终态：/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd/run_summary.json。
