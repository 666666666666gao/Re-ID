# §948：跨光谱三个同参数条件已收齐

原supervisor578010/controller578011从23:51:29到01:59:47 EXIT0，02:03节点原观察器确认二者终态。三个条件分别从相同初始化完成prepare、8步M0、fresh50及首次strict，所有命令EXIT0；没有训练重启、换best、补分或新seed。每一端使用同一mAP-best E8报告四项。

| 条件 | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| joint_delta | 74.5331 | 79.1866 | 88.2775 | 91.8660 |
| add_delta | 74.4934 | 79.4258 | 88.5167 | 91.9856 |
| joint_absolute | 74.4006 | 79.1866 | 88.2775 | 91.6268 |

一次CPU保存距离报告成功，全50记录、首strict、三条件初始化state SHA及batch order一致性、旧semantic/global控制批次SHA核读；完整query/身份变化见COMPLETE_PAIRED_REPORT_948.md和SUMMARY。报告原transport回收误读child/SOURCE_SEAL.json（实际seal在parent，child在manifest），于计算结束后失败；原报告EXIT0保留，独立只读收取成功。CPU report执行1次，transport/collection2次，不能写成总调用一次，更不能把行政回收失败当训练失败或重复计算一次。

核心配对：joint_delta−add_delta仅+0.0397mAP、−0.2392R1，修复2/新增4；joint_delta−joint_absolute +0.1324mAP、R1不变，修复3/新增3。两组均未过事前0.5mAP且R1不降的项目推进线；这是推进决策，不是统计显著性或数学无效证明。没有足够证据主张product/reference各自成立，更不是完整2×2交互消融。

相对原raw semantic，joint_delta/add_delta/joint_absolute mAP分别+0.1582/+0.1186/+0.0258；相对独立global分别+0.2364/+0.1968/+0.1040。相对旧semantic还同时改变了共同slot池化/dense读出并增加589836参数，不能全部归给跨光谱乘积。全Signal既有参照80.3029/85.1675未超过，当前差距仍约5.77mAP；无车辆新成绩或多seed。

全部150正式epoch、7947训练batch steps；不冒称全程逐step参数有效更新都已有独立witness。完整训练成本包含逐轮评价/保存，不能用history.seconds之和代替。三条末轮与实际墙钟：

- joint_delta: 50轮/2649训练batch steps；完整训练/逐轮评价 2411.063s；E50 mAP/R1 71.1558/73.4450，best→末轮 mAP下降3.3773。
- add_delta: 50轮/2649训练batch steps；完整训练/逐轮评价 2406.572s；E50 mAP/R1 71.5276/73.8038，best→末轮 mAP下降2.9658。
- joint_absolute: 50轮/2649训练batch steps；完整训练/逐轮评价 2398.059s；E50 mAP/R1 71.6259/74.4019，best→末轮 mAP下降2.7747。

这三条仍有约3点后期退化，继续保留同一E8 best；不能称训练未完成或据此认定唯一遗忘/背景根因。

report后真实free2919043072B，低于完整区域保存+2GiB reserve需求。实际D盘三M0 probe均size/SHA核验，才退休服务器exact重复文件1082206476B；free变4001263616B，三份best各360733443B和所有唯一记录保留。后续旧M0-dependent verifier必须先恢复D原probe。归档是重复退休，不是抹掉历史验收依赖。

按§947预登记，固定best三端只读G/C/F及原16区域诊断已请求原native2569，估计约5分钟；尚未在本节收取终态。新版source/输入seal包括已完成报告及retention receipt，exclude已经退休且不再消费的M0 binaries。严格距离/metric parity、模型state不变、0更新仍须实际通过；不会用诊断填正式分数或容差调宽。后续六项同数据CPU mean/MaxSim比较尚未执行。

训练阶段已收齐，不再重跑本三条件或围绕rank/gain/margin/seed救分。应据固定best实际用途判断额外区域证据是否有信息、当前比较是否用坏/用漏；失效机制不会扩到车辆或计入三项贡献。Full三数据集超过Signal Goal ACTIVE_UNMET，用户继续授权保留，仅26物理GPU0/1/原conda、不查询T/P、不使用25/2/3。
