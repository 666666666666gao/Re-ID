# MSVR310 原三角色架构 source-only M0 完整结果

记录时间：2026-09-06T06:24:17.950707+08:00。状态 **M0_PASS_COMPARISON_STOPPED_INDEPENDENT_AUDIT_PENDING**。
执行1c444cdf72e13fd041afd0c641dc8f522faa5844；原M0 wrapper69455 exit0，
三fold各8步容量和一个全新fold0固定100步拟合完成，共124更新/7936训练记录曝光，
0heldout/RGBNT201dev/官方前向。M0只证明工程与固定拟合条件，不是检索收益。

## 不变模型与全部工程条件

每fold只加载本fold固定车辆B0 Signal，冻结所有Signal/tail，原V8角色/七分类头从seed42新初始化。
高128/宽256、8x16 Patch、原CNN/Transformer/Mamba三路径，固定等能量拼接与七组ID/Triplet；
没有RGBNT201角色权重、Router、HFER、V23/V24干预或新loss。
每fold可训练张量203/203具有有限非零梯度，全部阶段0overflow；
整个Signal及所有冻结参数state SHA不变，角色状态确有更新。
三fold各8条clean source的独立Signal3072D与baseline前缀一致；
保存完整M0checkpoint并严格重载后，五输出均逐元素一致，完整模型state SHA相同。

| Fold | source身份 | 总参数 | 可训练参数 | 容量更新 | 容量秒数 | 峰值allocated/reserved MiB | checkpoint SHA-256 |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 103 | 99065869 | 8076300 | 8 | 11.631840 | 5726.967/5948.000 | 73890a30e04c3e594222e15de4de4ef73fe00b0ce1d8a5a7abb081e21fdd963a |
| 1 | 103 | 99065869 | 8076300 | 8 | 10.305954 | 5724.926/6084.000 | 8211a2ec80edba77197a4b4dbe1756cbec85aeb3ac9a96d253090c885a8165a6 |
| 2 | 104 | 99095053 | 8102412 | 8 | 10.933421 | 5727.030/6084.000 | ebaae4b4798dd886cb82fa4a9733a8196d9d465bbfea61ef739bdb9a6b609e31 |

可训练参数随本fold分类头类别数变化，不是增加新的角色模块。三路径全部执行，不能只按8M可训练参数描述推理成本。

## 固定100步拟合门

原七头label-smoothing解析熵下界为0.585713632744，initial/final loss为
4.122129917145/0.588196277618。按唯一最后一步计算，
(final-floor)/(initial-floor) = **0.000702022804**，小于固定0.1；
没有采用中途最小loss、延长步数或改变batch。该模型与三折容量模型分别从新初始化构建，
不把这些M0训练后权重传入正式比较。

## 原始记录、核验与成本

全部124步保留原损失组成、真实global record indices和AMP scale；每步确为8身份x8样本。
| 阶段 | Fold | 实际独立source身份/记录 | 记录曝光 | 同身份pair / 跨scene正pair |
|---|---:|---|---:|---|
| capacity | 0 | 63/333 | 512 | 1792/440 |
| capacity | 1 | 61/316 | 512 | 1792/482 |
| capacity | 2 | 62/350 | 512 | 1792/543 |
| overfit | 0 | 8/53 | 6400 | 22400/9200 |

这些是短source工程阶段的实际覆盖，固定100步使用同一个增强batch，不表示整个source数据覆盖。
三foldclean核验合计72次三角色记录前向及24次独立Signal记录前向；每记录含三幅模态图像。
程序配置后总计时246.289107秒，不包含前置import/config/hash，不能称完整wrapper wall time。
四个训练阶段epoch计时合计140.289673秒；
最大allocated/reserved为5727.030/6284.000MiB。

远端完整核对17个文件，包括3个完整checkpoint、独立fold training/receipt与summary相等、
20个实际项目输入绑定；执行纯文件核验3.472877秒，0新模型/图像/张量执行。
本地stdlib重算124步loss组成、全部真实采样、epoch均值、熵下界与固定门，
最大loss组合差4.12265459104e-07，epoch均值差0，
耗时0.020010秒。原AMP中间dtype未保存，保留该误差，不增加事后阈值。

本地只有文本/JSON/源码；完整权重留远端。以上为执行側核验，独立experiment-audit待进行；
不能据此声称独立图像/二进制复现或未知身份检索提升。

## 启动前失败与正式比较

首次148f5a7尝试在创建远端目录/进程前因criterion.py严格SHA失败，训练调用0。
5份历史CRLF/LF差异已确认远端等于Git原blob且AST相同；R2只固定实际远端字节绑定，
原配置、计划、wrapper及失败证据均保存；runner/model/优化/门没有更改，无runtime fallback。

正式比较原wrapper **70422** 于2026-09-06T06:21:14.660925+08:00启动，执行仍1c444cd。
每折重新初始化原角色，训练完整20epoch；baseline使用固定车辆Signal，不重训B0。
按M0约1.1–1.4秒/更新估计18–25分钟，首次阶段检查约启动6分钟后；不是承诺完成时间。
保留完整600query/1032gallery和全部五输出，只在各fold固定最后checkpoint评估一次；
原五项支持门、source/heldout身份隔离、scene过滤、无跨fold距离及全部负结果保留。
当前还没有完整比较检索成绩。独立审计与RGBNT201主目标保持未完成状态。

## 主证据

- evidence/trifusion_msvr310_trifusion_v1_m0_complete_20260906.json，SHA e021303b51d6af0b8bc49717016744e0ad483af419496652e0525b188d3d646b。
- evidence/trifusion_msvr310_trifusion_v1_m0_file_verification_20260906.json。
- evidence/trifusion_msvr310_trifusion_v1_m0_scalar_verification_20260906.json。
- evidence/msvr310_trifusion_v1_m0_receipts/，包含各fold与固定过拟合的全部原训练记录。
- 远端artifacts/msvr310_trifusion_source_oof_v1_seed42_1c444cd/m0。

2026-09-06T06:41:47.899579+08:00状态补记：原正式比较在fold0训练完成后因baseline与B0特征逐元素差异停止，
0检索指标、fold1/2未开始。此前M0结论不变；最初时间戳版供独立审计且保持原SHA。
