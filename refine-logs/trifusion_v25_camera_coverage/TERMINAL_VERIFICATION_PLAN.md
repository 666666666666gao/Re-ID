# V25 完整终态核验范围

登记时间：2026-09-06T23:00:03.502087+08:00；PREPARED_AWAITING_Q1_TERMINAL。
训练核心代码固定97468dd，原过程112550正在完成六端；本计划不修改训练代码或科学门。
核验器：tools/verify_v25_complete_terminal.py，SHA256 cc1cf0c1db52dc3b9bf770cd9cdc427d9142f751b62afda78bb5a73fa4e5a277。
本地仅AST检查；尚未运行终态核验。

待原训练和wrapper终止、exit0、完整Q1_PASS/Q1_FAIL报告出现后，在服务器tri_reid CPU上单次执行。
命令：python tools/verify_v25_complete_terminal.py --repo /root/autodl-tmp/trifusion-v2/TriFusion-ReID
--run-dir /root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd
--output /root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd/terminal_verification.json

- 逐文件核对冻结源码/配置/计划及执行Git原字节，原CLIP/6个V12起点和6个最终权重SHA。
- 核对全部3360行训练索引、路径、每epoch/step顺序、正例关系数、所有source记录曝光和120epoch全部loss均值。
- 只加载六份保存的CPU检索数组；核对每个输出全图库特征、所有距离元素与完整稳定排序。
- 逐条重算全部571query的AP、首次命中rank及aggregate mAP/Rank-1/5/10；same-ID same-camera过滤及完整干扰图库不变。
- 核对两端全部Signal特征/距离/排序逐位相等。
- 独立的数值算术使用全21身份query加权bootstrap，seed42/10000次，复算全部五项科学门。
- 输出全571query、全21身份、五个输出的配对差异、Rank-1修复和新增错误，不以子集替代。
- 新模型前向/优化/原始图像/权重tensor加载均0；CPU检索数组加载6份。
- 工程核验PASS不等于科学Q1_PASS，也不构成外部独立审计。
