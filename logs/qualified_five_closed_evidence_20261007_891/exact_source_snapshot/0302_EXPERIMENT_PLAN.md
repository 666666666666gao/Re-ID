# CPU-save边界诊断：只执行一次

§778真实v4在semantic RGBNT201梯度门失败：前向与loss一致，281梯度中94项失败；global-only209项一致。骨干83个Signal/11个adapter超阈值，但roles70项中的66项已有亚阈值差，不能由首个cv_embed失败或参数分组认定差异起点。v4停止在M0前，新权重目录为空，原失败封存。

Fresh rescue778（same-family/provisional，WARN_DIAGNOSIS_ONLY）只建议一次边界与保存张量诊断，没有批准修复或原样重跑。2026实际PyTorch2.5.1+cu121只读源码确认save_on_cpu对象提供pack_hook/unpack_hook，pin_memory=False内部直接tensor.cpu()及tensor.to(device)，可委托记录而不重写张量保存策略。

执行tools/diagnose_cpu_saved_boundaries.py：只semantic RGBNT201，原实现一臂、内置CPU-save带元数据记录一臂，共两个forward/backward、零optimizer更新。复用已验证publicCLIP/新camera/head/作者配方/seed42初始构造和实际B64/K8首32样本，原AMP FP16/GradScaler256/heads/loss/optimizer不变。完整B128内存与所有正式50轮不属于此诊断。输入、两臂同一初始CPU/CUDA RNG原始字节列表、参数/配置与来源回执留文本，不保存模型或原图。

记录器对候选骨干保存张量的每一次pack/unpack只委托实际内置函数一次，观察shape/dtype/layout/stride/storage_offset/device/contiguous与顺序。没有额外值复制、contiguous修复、压缩、pinning/stream、更改精度或GPU原张量引用保留。记录原实现与候选的stages/shared_global出口，retain_grad仅在两臂相同位置观察；callback不做cpu/item/同步，backward后再复制边界值及传入梯度。保存完整281项梯度、heads/输出/loss、buffers/RNG/BN/state/hook结果与元数据，所有固定断言放在MEASURED.json之后。

固定前向1e-5、参数梯度1e-4 allclose atol/rtol及状态精确门不变。边界梯度记录在scaled backward后除256换算为unscaled值，其比较用于定位，不能替代参数门或正式检索。仅metadata变化不足以确定因果；入口梯度已不同则不能只归因骨干CPU保存。instrumentation本身可改变时序，因此偶然PASS仅属本次记录，不能覆写v4或自动启动M0。若本次未定位，停止这一诊断，不追加另一臂/改gate/连续重试。

只用2026 physicalGPU0–3中的一张当前空闲卡，不抢占，2025仅文本同步。预计约1–2分钟，首次只读观察180秒；记录真实父/子退出与原stderr。无权重，无epoch、seed、loss、模型、正式campaign追加。诊断后基于完整实际证据决定后续唯一内存干预，所有9个完整作者batch M0仍未完成、正式9端50轮未启动。Goal ACTIVE_UNMET。
