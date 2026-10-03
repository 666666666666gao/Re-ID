# 原作者完整批次的模型分段可行性说明

状态：仅本地源码设计，未实现、未复核具体分点、未执行GPU。§780真实CPU-save边界诊断仍为FAIL_STOP_CPU_SAVE_ROUTE_UNLOCALIZED；这份说明不修补或放行该版本，也不增加诊断臂。

现有容量事实：2026四张RTX3090，每卡24576MiB，没有更大单卡，无NVLink。原V1的RGBNT100完整B128在`CrossLayerAdaptedCLIP.forward`组装stages时实际OOM。不能把四卡合成一块96GiB空间，也不能直接用batch分片、DDP或梯度累积宣称作者BN和batch-hard Triplet保持等价。

## 单进程分段候选

原作者ViT按12个block顺序执行。候选是把视觉前段放到一个现有GPU、视觉后段及全部角色/head放到另一个GPU，同一个完整批次依次通过两段。第一版可研究前6/后6，但这只是待算预算和验证的候选；救援复核没有证明该分点可装下或等价。

- 前端候选放置：patch conv、class/position embedding、camera embedding、ln_pre、前6个原block及第4层的原角色adapter。后端候选放置：后6个原block、第8/12层adapter、ln_post/CLIP投影、角色/原生读取器、readout以及作者BN/classifier。
- 每个参数仅有一份，不复制CLIP或适配器，不把梯度经另一个网络合并。放置在optimizer构造前完成；保留参数名字、FP32存储、requires_grad、共享三模态身份及原参数分组。具体移动实现仍待审查。
- 只在确定的段边界搬运激活，用保留autograd的GPU间`to(device)`。原RGB/NIR/TIR调用顺序、batch/token/通道形状、原block、capture公式和直接写回公式保持。前端的第4层角色快照需明确跨段搬运到原stages组装设备，不能detach、提前释放反向所需值，或者任意复制成多个状态。
- 全部作者BN/head及Triplet在后端一次处理完整B，不按卡计算独立loss或BN统计。标签、相机和部署输出设备必须与后端一致。训练与评估复用同一放置，保存完整state，重载到匹配放置后再严格验收。
- 不使用checkpoint重放、CPU saved-tensor钩子、精度切换、新stream、模型/数据fallback或新的科学超参数。不能把迁移设备当作已证明的数值等价。

## 容量与数值证明仍缺什么

B128的一个`[129,128,768]` FP16激活为25362432B；完整stages为684785664B。原实现还有captured快照、内层stack、最终stack及反向保存值，不能只计最终输出。实际逐卡预算必须同时计入参数、AMP副本、完整保存激活、跨段/组装暂存、梯度和Adam状态；本说明未建立完整峰值预算，不用B32诊断的backward后allocated值冒充峰值。

具体实现前先完成来源明确的逐卡预算及放置/生命周期说明，再进行fresh源码复核。若实施，原前向1e-5和参数梯度1e-4门仍须通过，相同初始状态和CPU/各CUDA RNG必须实际记录；不能因为跨卡就放宽门。所有九端完整作者batch八次真实更新、BN统计、optimizer覆盖、14项native参数活动、完整state严格重载M0仍先于任何50轮正式训练。原V1–V4及这次诊断失败保持封存。

只允许2026物理GPU0–3/max4、不抢占。后续队列必须把一个两卡任务的两个设备同时纳入实际归属，而不是仍按单卡记录启动；该资源安排也未实现或运行。2025只同步文本。此阶段不生成权重；正式端仍仅保存一份mAP-best，已无依赖的工程probe按既有核验流程退役。
