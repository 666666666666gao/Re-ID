# 正式训练准备：代码与归档记录检查

2026-09-21。本次仅检查本地代码/配置和已归档文本，不声称远端权重现存或哈希已重新核验，不启动模型前向。

- RGBNT100：已有 `Signal-main-v1.json`、`TriFusion-main-v1.json`、`Official-main-v1.json`，完整50训练身份，8675训练记录；Signal固定30epoch、角色20epoch。官方1715query/8575gallery，五种输出。`train_rgbnt100_signal_main.py`、`train_rgbnt100_trifusion_main.py`、`evaluate_rgbnt100_official_main.py` 及对应验证器可作为现有入口。
- RGBNT100全量基线摘要：`/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_trifusion_main_v1_seed42_20260906/signal_baseline/summary.json`，配置登记SHA `7c82f15a1227e36b04b41091bcf57ebc8a23375682448d62cd263a58120ccc13`；下一步远端核对文件、checkpoint及训练身份。复用基线不等于复用旧方法成绩作为本次新训练结果。
- MSVR310：现有 `Signal-source-oof-v1.json` 明确是来源三折、50epoch；尚未在本次检查中找到对应完整官方训练入口。必须核对官方全量训练划分，补齐全量基线/角色运行与评价合同，不能拿三折checkpoint替代。
- RGBNT201：当前V8配置引用 `signal_baseline_dev_seed42_f7d4b30/Signalbest.pth`。`run_signal_baseline_dev.py` 使用fit/dev协议并加载best checkpoint。它不能未经审核直接作为本次固定终点全量训练基线。继续核对是否另有适用全量初始化；没有则补训。
- 旧RGBNT100评价器写死1715/8575、50测试身份与camera规则，不能直接换路径用于另外两套数据。复用数学与核验逻辑时必须绑定各自合法协议。

方法尚未选定。选型依赖当前完整Q1及已完成内部证据，准备检查不读取官方成绩来调参。上述缺口是正式执行的准备任务，不是让用户手工补材料的阻塞。
