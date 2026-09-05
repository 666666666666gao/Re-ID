# V24 experiment tracker

更新时间：2026-09-06T02:42:14.150734+08:00。状态 M0_PASS_Q1_RUNNING；独立 M0 审计已完成，整体 WARN、工程 PASS。
最近实际 GPU 观察：2026-09-06T02:39:11.049481+08:00；不将文档更新时间当成新的训练观察。

| 项目 | 状态 | 证据 |
|---|---|---|
| source 成员合同 / 代码配置计划登记 | DONE_BOUND_6a4ac2c | evidence/trifusion_v24_preregistration_20260906.json |
| CUDA T0 六机制用例 | DONE_PASS_SYNTHETIC_ONLY | evidence/trifusion_v24_t0_20260906.json/log/xml |
| 三折六端预检与完整 M0 | DONE_PASS_ENGINEERING_ONLY | results/TRIFUSION_RGBNT201_V24_COMPLETE_M0_2026-09-06.md |
| 全 M0 数组与文件核验 | DONE_116_ROWS_47_FILES | evidence/trifusion_v24_m0_array_verification_20260906.json; evidence/trifusion_v24_m0_file_verification_20260906.json |
| 独立 M0 审计 | DONE_WARN_ENGINEERING_PASS | EXPERIMENT_AUDIT_V24_M0.md/json; evidence/trifusion_v24_m0_audit_closure_20260906.json |
| 六端 Q1 | RUNNING_TWO_PAIRED_FOLDS_COMPLETE | refine-logs/v24/PROGRESS_20260906_023911.md |
| 每折 fused 非负必要门 | VIOLATED_BY_FOLD0 | 首两折 fused 增益 -0.1251440758834974/+0.1664007443172011 pp |
| 完整 Q1 聚合 / bootstrap / 独立终态审计 | PENDING | 最后一折两端继续原配置 |
| D1 / dev / official / 消融 | NOT_QUALIFIED_NOT_RUN | 第一折已违反必要门 |

执行 commit 6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33；原 PID52030，screen v24_source_prototype_6a4ac2c。
00:56:14 启动，原进程未重启。完整 M0 固定第100步 excess 比0.059999361786045424；
116更新/232视图前后传，203/203梯度、冻结不变、无overflow、全部工程门通过。
参数两端均98,800,141总量/7,841,292可训练/203张量；新增推理参数0。
独立 M0 审计依据01:10快照，不覆盖后续完整 Q1，保留同家族与远端二进制回执的限制。

两对完成各20epoch、共80epoch/2280更新；02:39日志共82epoch/2334更新，最后一折对照端已完成2epoch。
fold0/1分别1000/1051完整gallery、190/179query，完整五输出和配对采样/原型元数据已核验。
第二折 fused91.07494864936395→91.24134939368115，CNN/T/M增益-0.5048841150709649/-0.7798281723704292/-0.5867659616460372。
第一折原记录未变；不以两个折的局部结果代替三折最终科学判断。
按实测阶段耗时，全终态预计03:18–03:25，下一检查03:17附近。
原六端完整合同、五项门、seed42 和失败封存规则不变，不修改或重启 V24。
