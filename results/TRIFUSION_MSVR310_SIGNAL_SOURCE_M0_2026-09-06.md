# MSVR310 Signal 源基线工程检查与正式启动

记录时间：2026-09-06T05:00:34.202298+08:00。M0 R2 三折完整通过，状态 **PASS_ENGINEERING_ONLY**。
正式 B0 于 2026-09-06T04:50:55.255082+08:00 启动，wrapper PID 63945，
执行代码 bb01d60b6e1517ee6f5dc9120faefd17d75401e5。04:57:26 的只读检查显示第一折已到46/50epoch，
原进程仍运行；没有完整折或检索终态报告可引用。这是新数据集的源基线建设。

## 三折 M0 实测

| Fold | source身份 | source训练更新 | 可训练参数 | 梯度张量/训练张量 | 8步平均loss | 峰值分配显存MiB |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 103 | 8 | 90172417 | 195/195 | 13.373951554 | 11379.005 |
| 1 | 103 | 8 | 90172417 | 195/195 | 13.083325505 | 11377.287 |
| 2 | 104 | 8 | 90175489 | 195/195 | 13.955506921 | 11377.322 |

合计24次实际优化、1536条训练记录曝光、48次clean source特征前向，耗时77.374851秒。
计时从配置构建后开始，不冒充wrapper完整墙钟时间。三折均无AMP overflow，全部实际训练张量均获梯度。
每折严格重载state_dict，重载前后8条clean source的3072D特征逐元素相同。
787968个TokenSelection参数的训练前后SHA不变；模型其余已训练状态发生变化。
原第一次M0另有8次更新，独立梯度诊断另有1次反传、0次更新；这些失败/诊断成本没有抹除。

本工程检查没有固定100步过拟合门，没有held-out/dev/official图像前向。
三折可用source身份为103/103/104；短M0实际曝光63/61/62身份，不能说全部source身份都在M0前向过。
正式基线使用独立新初始化，不继承M0 checkpoint；三个M0 checkpoint保存在远端供证据追溯。

## 原失败与修订范围

R1在fold0完成8步后因6个声明trainable的选择器张量没有梯度退出1。
独立零更新诊断和原作者useA.py表明Q/K只通过离散索引及二值mask作用，W_v未调用；
安装的Adam只更新grad非None项。R2明确冻结这六个本来不被更新的张量，并核对其值不变。
原前向、loss、LR、50epoch、seed和评价规则保留；在资格断言前保存training.json是针对R1实际缺失诊断细目的修正。

R1没有保存完整逐步训练数组；其fold0平均loss13.362928748与R2的13.373951554不同。
因此不声称已证明两次8步轨迹逐元素一致；修订依据是已读取的调用链和优化器语义。
R1原日志、退出、配置和commit仍保存。

## 证据核对与限制

远端核对21个完整文件（含三个完整checkpoint）的SHA；6个独立training/receipt JSON与总报告一致，
7份项目绑定文件和17份Signal实际源码SHA保持。传回本地的是文本及JSON，没有checkpoint或特征张量。
本地JSON算术核对全部24步B64/K8、source原始索引、真实ID映射、195组实际训练参数和各loss分量；
mean loss重算误差0，loss组合最大绝对差1.300722360e-6。
原AMP中间操作dtype未保存，该差如实报告，不追改M0门槛。
这些是执行侧文件/标量核验；完整基线后仍需独立审计，不等价于独立模型前向复现。

原始结果 [evidence/trifusion_msvr310_signal_v1_m0_r2_20260906.json](../evidence/trifusion_msvr310_signal_v1_m0_r2_20260906.json)，SHA 79c0e2b1c981c4bb10548f0249dca684c113feb43151cc4bdcc2a2e9bf2887ae。
逐文件核验见 evidence/trifusion_msvr310_signal_v1_m0_r2_file_verification_20260906.json；
逐步核验见 evidence/trifusion_msvr310_signal_v1_m0_r2_array_verification_20260906.json；
启动证据见 evidence/trifusion_msvr310_signal_v1_baseline_original_launch_20260906.json。首次进度读取器event名误匹配导致epoch_rows=0，
原始log_tail明确到epoch46，原记录与单独更正说明均保留；不影响训练代码或结果。

## 正式 B0 的固定边界

三折各50epoch，seed42/B64/K8，H128/W256，原MSVR多步LR与scene过滤。
只用官方训练部分155身份/1032条记录建立身份隔离OOF；完整gallery360/349/323，
最终query210/207/183，共600条。单scene身份继续作为gallery干扰身份。
每折重新从固定通用CLIP建模，使用完整3072D direct+SIM，最终epoch50只评一次；
结果不可称原作者官方测试复现或TriFusion有效性验证。
B64/K8和同步几何与作者K4/原增强差异已在固定计划披露。

原始20–35分钟估计按04:57实测约8秒/epoch收窄到05:13–05:16附近；
不按训练loss提前终止，不依据某折结果改配置。没有官方测试、RGBNT201固定dev、
新三分支、消融或新seed操作。V23/V24封存和RGBNT201未达主目标的状态保持。
