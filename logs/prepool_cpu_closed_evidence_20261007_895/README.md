# 新同模态几何辅助目标：CPU接入证据

现有Torch2.5.1+cu121的CPU bf16 Linear固定二例证实：同外层autocast内，先no_grad教师缓存导致后续主分支无weight/bias梯度；仅教师cache_enabled=False后，两梯度恢复有限且前向相同。4组件forward、1backward、0optimizer，无CUDA/生产模型。

三集真实作者CPU loader/workers4的前8批主图像、标签、camera、view、paths、日志及CPU/Python/NumPy RNG精确相同。另有随机key的CPU辅助目标/VJP有限非零且unused-global梯度为空；这不是实际角色/主干更新证据。首个夹具在关闭包装generator时错误调用DataLoader内部_shutdown_workers；原EXIT1完整保留，新r2仅采用真实M0的第9批fetch/第8批log后关闭generator，EXIT0。

几何计数40组证据及首次张量索引失败见相邻prepool_geometry_closed_evidence_20261007_895。两个神经源的初始FAIL/仅教师缓存修补后的SOURCE_ONLY PASS分账，full review prompt/raw只在私有包；公开仅摘要、来源SHA和原/修订源码。尚无CUDA初始化/M0或新检索成绩。只26/无25或GPU2/3/功率温度/安装。
