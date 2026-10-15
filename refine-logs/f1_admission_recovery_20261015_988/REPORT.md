# F1接续资源恢复与旧100seed43闭合

旧RGBNT100 seed43已真实完成50轮、6564更新及完整1715Q/8575G独立评价，原控制器1653898 exit0；E13同一mAP-best为86.9928725/97.2594752/97.7842566/98.0758017，相对完整Signal为+.6686739/-.2915452/0/+.0583090个百分点。CPU闭合距离诊断为7修复、12新增错误，首位净少5；其中4个新增错误为同camera负例。已保存全部50轮、best SHA f1b8fccae217c13d804f177a3c9c314e61284c2602fc9fb737030220905a6bc6、距离SHA1c1bfcff0320e8e4ec869b389c895e9afc00fa9c31dddf2a6371d9046a0afdf5及独立诊断。旧seed42 rho.75仍优于这一原rho1端点；不能混取指标，也不将该旧Signal初始化结果充当F1主线。44继续原完整50轮，21:03只消费原64699，不新增45。

15:33旧43闭合时实际磁盘仅2,270,175,232字节。15:37新F1控制器仍实际存活，三job全部PENDING，无prepare目录，尚未发生新NN；因此不能沿用§987已排队就会成功接续的推断。原代码在phase更新之前assert free>3GiB，另线程等待21:03；容量未满足及未进入新模型是实际证据，worker future具体异常未直接捕获，仅作源码推断。15:39核对原controller cmdline和全PENDING后，仅主动SIGTERM未启动的3384245，监督器明确记录-15；没有终止或重启任何训练，旧记录保留。

按已观测的资源问题作最小调度修正：不降低3GiB门槛，改为任务明确WAITING_FOR_DISK，每240秒复查同一个资源条件，满足后继续。controller总体RUNNING表示控制器在工作，中间进度以每个job实际phase为准。另在prepare增加真实完整Q/G的F1基线复现，先验证三集原F1四项指标，再开始新增优化；这是评分/初始化检查，不改变网络、模块、loss、seed42或额外50轮训练。第一轮仍统一默认内部参数；用户允许后续按集调整内部宽度/层数/比例，不能将其误写为LR/B搜索。

八份已闭合、全50轮的旧shared-private、signal-selection、input-detach及normalized-role best先逐份核对原metrics checkpoint SHA、50轮history，再保存exact-D并复核大小/SHA，之后仅删除对应远端重复文件。原训练/指标/距离/源码及所有唯一权重保留；F1三份、Signal/publicCLIP、强R2及刚闭合43best未动。每个旧权重消费者前必须按RETENTION.json恢复原路径。首轮SCP第二份发生Connection reset，原失败保留，已从首份已验真副本继续，未删除任何未完成归档的远端文件。

新的§988登记只接续尚未开始的同三集主线：GPU0立即在旧43 COMPLETE后先201再MSVR，GPU1仍等旧44和21:03节点后跑100；每卡一项。先完整F1复现、实际8步M0/严格重载，再从原F1 fresh完整50轮和同一best实际全Q/G评价。注册发布时新的GPU/M0/正式成绩仍未发生，目标ACTIVE_UNMET；本次只是实际恢复资源和启动条件，不能宣称方法超Signal。

本次exact-D保留并退远端重复字节：2868734210；归档后实测空闲：5118128128字节。
