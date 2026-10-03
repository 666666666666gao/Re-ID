# 已安装Mamba扩展的静态证据

结论：**COMPILED_ATOMIC_RISK_CONFIRMED_NOT_RUNTIME_CAUSE**。这是只读依赖核查，不是GPU重复试验、M0或性能结果。仅2026物理GPU0/1被授权；本次没有执行模型或CUDA kernel。

20:10:40的包记录核查：Mamba2.2.6.post3和causal-conv1d1.6.0的两个实际扩展，其字节数与摘要均匹配各自RECORD。两者direct_url均指向原/root下的本地构建目录，没有VCS/archive来源标识。这证明当前文件符合安装记录，不能证明官方源码与实际编译物完整等价。

20:15:08读取Mamba构建目录时，/root/autodl-tmp/trifusion-v2/vendor-sm86/mamba返回PermissionError，采集器退出1。原stderr/EXIT保留于build_source_failure，没有sudo、改权限、重建或覆盖失败。该脚本在Mamba处终止，没有检查另一构建目录的内容，不能将其可访问性填为实测。

20:17:42静态nm成功，扩展导出的scan/conv反向符号存在。20:19:51使用服务器已安装cuobjdump13.2.78，只解码现有selective-scan扩展；没有加载它、调用CUDA或改动训练环境。实际文件摘要与20:10记录一致；列出9个sm86 cubin。选定32线程/4项、非整长、variable B/C、delta-softplus、z的FP16-input/FP32-weight和FP32-input/FP32-weight两个反向函数。每个实际函数均含11处RED.E.ADD.F32.FTZ.RN.STRONG.GPU指令。完整dump、函数名、命令、版本、ELF列表、stderr及各条指令都保存于static_sass。工具在其余cubin找不到请求函数的16条warning照实保留；输出中两个精确请求的函数节都存在，不能把warning写成完全未解码，也不能省略。

这一证据把§787的“官方源码存在原子累加”推进为“当前安装二进制确实含对应配置的FP32原子加法”。静态指令的存在仍不证明历史backward执行了这些位置，不能把11处静态指令当成11次实际累加，也没有逐地址确认其中哪条对应dB/dC。尚未保存历史中间梯度或kernel dispatch，不能据此唯一解释83/295失败，更不能据此修复V5。

依赖解析以[NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)的静态反汇编工具职责为依据。工具版本与扩展构建版本不同，使用工具解码不代表升级训练CUDA。

保持原STOP_GPU_PARITY：不复跑original/V5、不加GPU对照、不改变阈值/精度/batch/seed/backend、不更新依赖，不进入九端M0/正式50轮。九端仍0个当前路径M0验收、0个正式结果，RGBNT100 B128容量未证明。未来若提出定位测量，应先给出具体可复核方案及审查，不能把本静态风险当成已定位缺陷或启动授权。Goal ACTIVE_UNMET。
