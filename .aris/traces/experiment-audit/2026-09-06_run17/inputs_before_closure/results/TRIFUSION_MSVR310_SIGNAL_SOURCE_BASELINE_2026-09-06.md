# MSVR310 Signal 完整训练内部源基线

记录时间：2026-09-06T05:17:36.048722+08:00。状态 **BASELINE_COMPLETE_INDEPENDENT_AUDIT_PENDING**。
原wrapper63945正常退出0，执行bb01d60b6e1517ee6f5dc9120faefd17d75401e5。
完成三个独立source模型，各50epoch/650更新，总150epoch/1950更新；
仅seed42，终点固定epoch50，完整600query的加权mAP **53.129380561**，
Rank-1 **63.000000000%**。
这是MSVR310官方训练集内部的身份隔离基线，未访问官方测试图片。
新TriFusion方法资格未评估，RGBNT201主目标仍未达到。

## 完整三折结果

所有指标单位为百分比，mAP是逐query AP的算术均值；总计按600条query合并。
不把三个fold的mAP不加权平均当作本合同总指标，不在fold模型间计算距离。

| Fold | source身份/记录 | heldout身份 | 完整gallery | query身份/记录 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---|---:|---:|---|---:|---:|---:|---:|
| 0 | 103/672 | 52 | 360 | 20/210 | 49.311695078 | 59.523809524 | 71.428571429 | 77.619047619 |
| 1 | 103/683 | 52 | 349 | 20/207 | 47.942264566 | 60.869565217 | 76.811594203 | 81.159420290 |
| 2 | 104/709 | 51 | 323 | 20/183 | 63.377724618 | 69.398907104 | 83.606557377 | 90.710382514 |
| 全部query加权 | 每折独立 | 155 | 1032 | 60/600 | 53.129380561 | 63.000000000 | 77.000000000 | 82.833333333 |

每折source均含40个跨scene身份；全部95个单scene heldout身份的432记录作为gallery干扰保留。
只剔除没有跨scene正例的query；排序时只移除“同身份且同scene”的gallery。
camera0–7用于Signal SIE，不代替scene过滤。不同身份同scene负例保留。
三个fold的性能范围说明划分难度存在差异；当前是一个固定seed、一个固定协议的描述结果，
不能据此宣称跨数据集改进机制成立或稳定优于公开方法。

## 训练、状态和成本

正式基线共124800条source训练记录曝光；每fold的672/683/709记录和103/103/104真实身份均实际出现过。
全部150epoch每epoch真实13个B64/K8 batch，1950个step都有完整原始索引和loss组成。
三个模型各自从相同固定CLIP文件重新初始化；initial_state SHA分别与对应M0初始状态一致，
正式训练未继承M0、RGBNT201、V24或其他fold的已学习权重。
50epoch终点严格重载state_dict，重载后的完整gallery只前向一次；query按固定位置读取该同次特征。
评估前后模型状态SHA不变，全部保存的训练state SHA及checkpoint文件SHA见下表。

| Fold | 可训练参数 | 全部参数 | 实际训练/梯度张量 | checkpoint SHA-256 |
|---|---:|---:|---|---|
| 0 | 90172417 | 90963457 | 195/195 | fd17dd2e4c1a5576bf02b70e2b7faced979794d955a4e510cb76fd144d9c4b00 |
| 1 | 90172417 | 90963457 | 195/195 | a11e0fd6c531ea55e43040285671e8adbd3481c7dc5510309fe0d9ea0c974253 |
| 2 | 90175489 | 90966529 | 195/195 | fa7531cc3de677cec99852e588dcaa0851d0c1b5dc34bb65e0d4cc23ff95c06b |

三折无overflow；787968个已证实不被原Adam更新的TokenSelection参数保持原值。
训练标量时间按epoch累计分别407.841615、
411.711371、414.373363秒；
程序配置后总计时1299.025203秒（约21.65分钟）。
该计时不包含进程启动前配置/hash准备，不能称完整端到端wall time。
最大分配显存11382.037MiB。
B0成本之外，原R1 M0为8更新，R2 M0为24更新，另有1次零更新梯度诊断；全部保留，不计成正式epoch。

## 完整文件与逐query核验

远端核对23个完整文件，包括三个最终checkpoint、三份原始特征/距离文件，
六个独立training/receipt JSON与summary一致。7份项目输入、17份Signal实际源码、
原Signal commit/diff、完整CLIP权重文件及M0门回执SHA均重新核对。
核验执行代码192881ce517997afa7c7e18d79e3f873f023d4a1，
耗时2.895723秒，不运行新模型、不读图片、不反传、不优化、不写checkpoint。

由三份保存的3072D特征重算距离，三fold与原距离文件均逐元素一致，最大差0。
从原距离导出全部600query的完整gallery排序索引，再用原真实身份/scene标签独立于训练函数重算AP/Rank。
本地核对全部600个AP/首正例排名、60身份均值、完整干扰图库与所有1950step；
指标最大绝对差3.33066907388e-16，loss分量组合最大差1.30072236004e-06。
原AMP中间dtype未保存，保留该组合误差，不追改阈值；本地核验0.133057秒。
本地仅有文本、JSON和离散排名索引；checkpoint、特征与距离张量留在远端。

这些是执行侧文件和算术核验；独立experiment-audit仍PENDING，不把它们称跨模型审计或独立图像复现。
首次只读进度reader误查event名所产生的epoch_rows=0及其更正记录保留；
完整150条实际event日志已逐一与最终history相等核对。

## 固定边界

沿用原MSVR Signal：128高/256宽、8×16 Patch、完整3072D direct+SIM、DIRECT=0的模态分类头，
可训练视觉主干、原车辆Adam参数组/分类头100倍基础LR，20/40epoch多步衰减。
项目B64/K8与三模态同步几何相对作者K4/原增强存在已登记差异，不称严格逐项作者复现。
原始M0失败与R2修订见TRIFUSION_MSVR310_SIGNAL_SOURCE_M0_2026-09-06.md。
固定主计划与config、runner未改；没有选择更高epoch/单折、改seed、reranking、vehicle TriFusion或消融。

本协议600query/1032完整gallery来自官方训练部分；原官方测试591query/1055gallery是另一测量条件。
内部mAP接近某个公开数字不代表复现官方结果，不做跨协议直接差距相减。
RGBNT100尚未产生本项目训练或检索成绩；V23/V24负结果封存，RGBNT201dev最好仍V8 Phase-B58.4050/59.3939，
dev65和官方85.3/87.9目标未达到。后继配对方法须另立合同，本基线不自动开放官方测试。

## 主证据

- 原始结果：evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json，SHA 22a4f3642e88088a8dfcb4acddb610d6566c12d6b765cb7b344b28acdbcea6eb。
- 完整排名：evidence/trifusion_msvr310_signal_v1_baseline_saved_rankings_20260906.json，
  SHA 14a14081d2d99b5932184bc898277f473a18021b127d22149e2ec4b2b5362cb2。
- 文件核验：evidence/trifusion_msvr310_signal_v1_baseline_file_verification_20260906.json。
- 全数组核验及全部60身份：evidence/trifusion_msvr310_signal_v1_baseline_array_verification_20260906.json。
- 完整日志：evidence/trifusion_msvr310_signal_v1_baseline_run_20260906.log。
- 独立fold训练/receipt/作者re.txt：evidence/msvr310_signal_v1_baseline_receipts/。
- 远端原始目录：/root/autodl-tmp/trifusion-v2/artifacts/msvr310_signal_source_oof_v1_seed42_bb01d60/baseline。
