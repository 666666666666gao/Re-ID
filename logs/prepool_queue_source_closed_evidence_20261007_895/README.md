# 端级验收隔离：源码与标准库流程证据

第一版把端级initializer/full验收直接放在控制器，stdlib精确函数重放确认失败会使后两端保持PENDING且无matrix。修订将二者放入原run_logged的独立accept子步骤，原验收条件未变；原两种失败成为missing并继续其余，source/disk共同门仍阻断后续子步骤。原审查FAIL不改。

这里的fixture文件/子进程边界为合成标准库对象，不是实际模型/训练/评价证据。目录只含确定性checker、结果、原/修订源快照；full review prompt及RAW留私有。实际运行身份UNATTESTED/same-family/provisional，SOURCE_ONLY PASS不代替CUDA初始化、M0、资源或完整50轮验收。
