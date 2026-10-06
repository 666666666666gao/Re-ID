# 首端实际M0完成，隔离活动门FAIL；固定尺度诊断已审查未运行

- original controller2208860/supervisor2208857 exit1 at07:29:02；sole observer59916 closed0 at07:35:36，五条件PENDING，无正式训练。
- md_batch_ratio201 production8更新 M0_PASS，281/281总loss梯度、BN8/reload0、完整初始化与旧8batch相同。隔离unscaled c有梯度、CNN q0/k0有梯度，其他4QK均零；原FAIL保留、资格0/6。probe358375672B作为当前固定诊断输入保留。
- 隔离VJP未缩放，但生产AMP反传初始scale256；不能据此认定Transformer/Mamba实际训练收不到新增梯度或已经定位underflow。
- 新fixed state一批/一forward/两VJP scale1与256，0更新，unused单列，BNbuffer恢复+state精确核验，SOURCE_ONLY PASS same-family provisional，NOT_RUN。
- full队列及15pair报告source审查通过、无部署且不具备六M0资格。当前不执行准备的12binary retirement资格脚本，不清理诊断probe。
- 07:43:47 old393与current45输入实存不变，原M0与终态九input固定；空闲3859419136B。不25、仅26物理0/1，无温度功率查询。Goal ACTIVE_UNMET。
