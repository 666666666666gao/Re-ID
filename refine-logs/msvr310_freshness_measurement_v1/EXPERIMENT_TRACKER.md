# MSVR310缓存坐标直接测量跟踪

| 阶段 | 固定范围 | 状态 |
|---|---|---|
| 静态/输入绑定 | 两新工具、固定输入、原采样/像素/源码 | 静态PASS；远端新前向NOT_RUN |
| 重编码预检 | 三fold×两更新规则×12步=72 | NOT_RUN |
| 预检CPU | 全72步/候选/距离/loss/权重 | NOT_RUN |
| 完整来源测量 | 六端×260步=1560，全历史项重编码 | NOT_RUN |
| 完整CPU与文本 | 全量距离/梯度见证/所有epoch | NOT_RUN |

新鲜loss只诊断，不用于更新；不读取heldout/official图片。不是旧训练逐位重放或新Q1。原失败不改，Goal active/unmet。

23:09固定ab67d4c/f3a0634启动原wrapper185622与preflight185624；23:11:43均验证存活。已完成fold0两端各12步及strict重载、真实零更新复用逐位相同，6端/72步全CPU尚未终态，不能提前PASS。run: /root/autodl-tmp/trifusion-v2/artifacts/msvr310_freshness_v1_seed42_ab67d4c。原进程继续，不重启、不改合同。
