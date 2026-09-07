# TriFusion旧传输副本清理与模型保留核验

2026-09-07T23:43:28.110625+08:00。已删除182个历史Git传输bundle副本，合计637,229,963B（607.709849MiB）。它们均已导入当前仓库；删除前逐文件检查大小、mtime、SHA及包含提交，所有提交均是保留HEAD a0e5785f0353fa93bce2948ac092bccbc903a20c的祖先。删除前后git fsck --connectivity-only --no-dangling均通过。

全部目标是/root/autodl-tmp/trifusion-v2/transport的直接子文件，固定.bundle扩展，排除当前msvr_freshness*传输包。执行前全量复核固定计划SHA e9bc085253232ba89a34506abb1f8b6690470ac636abdb749f1e11284c1d9127；逐文件unlink，没有递归删除，没有清理仓库.git对象。

清理23:38:35完成；原182路径均不存在。主盘空闲4,670,595,072→5,308,112,896B，约4.35→4.94GiB。运行中的训练仍会产生少量文件，因此文件字节总和与文件系统空闲增量不是同一种计数。系统存储卷仍独立，23:39:24为11,129,233,408B，不能将两卷容量相加当作同一输出目录空间。

本次新增模型权重/检索数据删除为0。全量盘点279个pt/pth/safetensors序列化文件，清理后大小、mtime、device、inode全部相同；本轮没有重新哈希这279个文件，不将元数据核对冒充内容SHA核验。当前预检及正式来源权重、全部基线、已封存终点和二进制审计数据保留。此前24个冗余恢复权重清理释放24.90GiB的历史记录见results/TRIFUSION_DISK_WEIGHT_CLEANUP_2026-09-06.md，不重复计算成本次释放。

原wrapper185622及source186000在清理前后均存活；最近观察2026-09-07T23:42:41.033870+08:00，已完成来源端数1/6。实际执行仍是ab67d4c源码、f3a0634合同，不因文档HEAD变化而改训练。完整1560更新和终态CPU仍在等待，没有新的Q1或官方结果。

同期增加只读终态分组工具tools/analyze_msvr_freshness_epochs.py（SHA 7636fe83b9bce0e1a39dc835681d5676bb23cf238c438da1c47b8c57f2c70b29），在全部72预检行上核验通过：6个epoch组、48个年龄组、18个角色组。正式来源将保留全部120个epoch组，包括无历史的65步预热；年龄份额表示候选机会，不是赢家概率。梯度仍是运行时见证的完整文本汇总，不是CPU独立反传。工具尚未运行完整来源数据，不改变正在运行的源码/计划/优化规则。新鲜loss仍只诊断。

首轮只读盘点因远端未安装rg退出，随后对两个明确artifact目录使用Path.rglob盘点；未安装工具、未影响训练。清理计划、执行源码、完整回执、279文件保留核验及原进程观察在evidence/trifusion_obsolete_transport_cleanup_20260907/；全量预检分组在evidence/msvr310_freshness_epoch_age_preflight_20260907/。
