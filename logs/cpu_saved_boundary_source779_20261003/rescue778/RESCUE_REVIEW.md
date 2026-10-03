# CPU saved-tensor v4 失败救援复核

时间：2026-10-03 16:34 +08:00。审查：fresh context、same-family、provisional。父代理回传的实际 native spawn 参数为 model=gpt-6-astra、reasoning_effort=max、fork_turns=none；底层 backend 身份未独立验证，不作跨家族接受声明。范围：本地封存源码和实际 primary；未连接服务器、导入模型、启动训练、修改实验代码或删除权重。

结论：**WARN / DIAGNOSIS_ONLY。v4 的实际固定梯度门禁失败，不能原样重启或进入 M0。建议下一步只做一次、两臂、零更新的 semantic RGBNT201 边界与 saved-tensor 元数据诊断。当前没有足够证据批准任何内存实现修补，也没有证据把 layout、stride 或反向顺序认定为唯一原因。**

审查根目录：`../cpu_saved_terminal_intake778/`；下文源码路径相对其 `_source/`。历史控制根目录：`../original_repeat_intake776/`；V3 根目录：`../terminal_intake775/`。

## 已核实事实

1. `primary/logs/independent_native_evidence_20261003_v4/cpu_saved_backward_RGBNT201.log` 的真实异常为 semantic 的 `cv_embed` 梯度 `allclose(atol=1e-4, rtol=1e-4)` 失败，最大绝对差 `0.001220703125`。这只是参数遍历中的第一个失败名，**不是反向传播中最早出现差异的算子**。源码 `tools/check_cpu_saved_backward.py:102–107` 直接证明遍历顺序。
2. 重新按两个 pre-assert JSON 汇总：global_only 209 个梯度参数零差异；semantic 281 个参数中 94 个失败（83 个 Signal、11 个 adapter），最大绝对差 `0.0078125`，无 gradient-None 不一致。两臂的 raw/fused/global/loss 最大差均为 0；heads 的固定门槛也在 gradient 断言之前通过。不能把最大绝对差与单独的 `1e-4` 比较当成完整判据；实际失败依据是保存的逐元素 `torch.allclose` 结果。
3. **门槛通过不等于逐位相等。** semantic 的 70 个 `roles` 参数全部过门槛，其中 66 个存在非零差，最大 `4.76837158203125e-7`；readout 的两个参数差为 0。Signal 共 155 个参数中 150 个非零差，adapter 共 54 个中 17 个非零差。因而“超阈值只在 backbone”不能证明差异最初产生于该 backbone 的 CPU saved-tensor 实现。
4. `check_cpu_saved_backward.py:108–116` 的 semantic buffers、CPU/CUDA RNG、BNcount、最终 model-state、hook-cleanup 断言在失败点之后，未执行；native 未进入。global_only 完成先前循环不等于整份 RGBNT201 backward witness 完成，最终 PASS JSON 未生成。
5. `INTAKE.json` 记录 parent absent、parent exit code 为 null、raw campaign 仍 `INITIALIZING/jobs=[]`、3 个初始化、1 个 eval pair、0 个完整 backward witness、0 个正式 job、0 次报告调用，输出权重目录为空。launcher traceback 记录子进程 exit 1；`EXIT.json` 的 exit 0 是 collector，不是父控制器成功。`queue_native_cpu_saved.py:176–198` 也说明此处在 M0 job 列表生成之前。
6. 历史原对原 semantic 控制实际 PASS，281 个梯度最大差 `2.9103830456733704e-11`。本次还核对其已保存的 labels/cameras/paths/images 摘要与 v4 semantic 全相同，initial-model-state 与 cfg 字段也相同。历史控制两臂间的状态与 RNG 检查成立；未持久化的跨进程原始 RNG 状态不能由这些字段重建。该控制加强了本次失败值得诊断的依据，仍不构成普遍确定性证明。V3 primary 另有相同固定门槛的实际 semantic 失败，`cv_embed=0.0009765625`、92/281 失败、最终输出/损失零差；没有将 V3 改写为通过。

## layout / stride 假设：有真实入口，没有直接因果证据

`modeling/trifusion/cpu_saved_evidence_clip.py:7–16` 唯一计算干预是在训练且启用梯度时，用 `save_on_cpu(pin_memory=False)` 包住同一个已绑定原 forward。`tools/run_native_cpu_saved.py:17–24` 不更换模块或参数。eval 调用原 forward，所以已通过的 full-batch eval pair 不检验 CPU saved-tensor 搬运。

模型里真实存在视图和分支：

- 作者 `comparators/Signal-cd1b0a6/modeling/clip/model.py:448–461,484–487` 有 reshape、NLD/LND permute、CLS 索引赋值和投影；`meta_arch.py:103,108–109` 有 camera 索引及 token 切片。
- `modeling/trifusion/correspondence_roles.py:64–72` 在三个层位分别生成三个 adapter 分支，既把 `(output+delta).permute(...)` 堆叠为角色输入，又返回 `output+sum(deltas)/3` 继续主干。`76–89` 对三个模态调用共享视觉模块并拼接全局输出。
- 同一文件的 permute 结果随后经 `stack`；因此仅见到 permute，不能推断最终被 autograd 保存的张量必为非连续布局，更不能推断 CPU 往返后布局已变化。

现有 primary 没有 pack 前、CPU payload、unpack 后的 shape/stride/storage_offset/layout；301 项项目快照也不包含执行环境的 `torch/autograd/graph.py` 或后端实现。**“搬运改变了保存张量的 stride，因而改变 kernel/归约”目前只能列为待测假设。** 不应据此预先加 `.contiguous()`、自定义 `empty_strided` 恢复、额外复制或选择性 offload。

## backward DAG / 顺序假设：必须区分图结构与执行调度

v4 没有 checkpoint、重放、forward 算子重排或显式新 stream；原 capture hook、三个模态顺序及两条返回路径保持。源码支持“主计算公式与显式分支结构保持”，不支持“backward DAG 已被重写”。共享参数、角色回流和主干直达路径确实形成多路梯度汇合：`independent_native_roles.py:66–74` 消费 stages 和 shared_global，而 `evidence_author_heads.py:14–18` 的 global_only 丢弃 stages。这个差别与当前 semantic 失败/global_only 通过相容，仍不足以定位原因。

saved-tensor 恢复带来的等待、内存地址或后端执行顺序是否实际改变，没有执行轨迹可证。即使 CPU/GPU RNG 相同，也不等价于全部后端归约逐位确定；反之，一次原对原 PASS 也不能排除这种可能。FP16 量级的梯度差、同样的最终 forward 及失败集中在视觉路径，均不单独证明上述任何一种机制。外部 roles 已有小差进一步要求先看边界，而不是直接指认 `cv_embed`、Mamba 或某个 attention kernel。

## 唯一建议的下一步：一次边界与 saved-tensor 诊断

这是待实现、待源码审查的诊断规格，**本复核未执行**。新独立诊断入口只跑 semantic RGBNT201：原实现一次、原实现加 CPU save 一次；总计两个 forward/backward、零 optimizer step，结束即停，不调用 campaign/M0/full50。

1. 复用现有真实 author B64 的 first32 输入、初始化和比较流程；seed42、AMP FP16、GradScaler256、作者 heads/loss/optimizer 构造不变。两臂使用同一已取出的 batch、同一初始状态及同一 CPU/CUDA RNG；记录既有输入身份字段，不建立新哈希体系，不以重新抽样求通过。
2. 准备 instrumentation 前，只读实际执行环境内置 `save_on_cpu`/`saved_tensors_hooks` 的实现与版本。诊断中委托同一内置 pack/unpack，各调用一次，记录 pack 前、CPU payload、unpack 后的 shape、dtype、layout、stride、storage_offset、device 和顺序整数编号。不要另外复制、contiguous、改 dtype、换 stream，或持有原 GPU saved tensor 的引用而抵消 offload。若实际 API 不支持该直接委托方式，停止并报告，不加兼容路径。
3. 两臂对 backbone 原返回值 `stages`、`shared_global` 做相同的诊断观测：比较实际 forward 张量，以及生产 backward 到达这两个边界的梯度。保留边界梯度用于 backward 完成后的比较，避免在 callback 内做 `.cpu()`、逐项 `.item()` 或全局同步；不增加新的 loss/backward 调用。仅两处边界，暂不挂整棵 DAG 的逐节点 profiler。
4. 在唯一最终固定断言前，把输出/heads、所有 281 个梯度、边界比较、saved-tensor 元数据差异，以及原有 buffers/RNG/BNcount/state/hook 检查结果一并写出。原有阈值不变；检查失败照实失败。额外观测会改变内存/时序，因此此次偶然 PASS 只能是诊断事实，不能覆盖 v4 失败或自动放行训练。

判读预先固定：若观察到 stride/layout 差且两个返回边界的前向/反向输入一致，才获得优先排查保存张量表示的直接证据，仍未证明唯一根因；若边界 forward 已不同，先定位隐含 forward 差异；若 forward 相同但进入 backbone 的边界梯度已不同，则差异在该边界已经存在，不能只靠修 CPU pack 宣称解决。若布局相同或 instrumentation 不再复现，结果记为未定位并结束本次诊断，不自动追加另一内存方案或连续重跑。

本次不建议预先进行内存干预。任何后续修补都应针对上述实际观测，另经原门禁验证；即便单次 diagnostic PASS，也不能代替全 9 初始化/比较、全 9 full-author-batch 八更新 M0，以及之后才允许的九个 50 epoch 正式实验。

## 资源与权重终态

仅既有 2026 物理 GPU0–3、max4、不抢占，诊断只需一张空闲卡；2025 保持 text only。训练 batch 保持 201 B64/K8、100 B128/K16、MSVR B64/K4，不改 precision、seed、优化器或数值门槛。长诊断以预估完成时间安排首次观察，后续 180–300 秒；不为诊断启动正式队列。

诊断只需小型文本/JSON 结论，不需要保存新模型权重。当前 v4 权重目录已由 intake 证实为空，无权重可清理。本复核未删文件。后续工程 probe 仅保留到其严格校验及终态报告结束；届时按已有依赖核验流程及时退役。每个完成的正式实验只留一个 mAP-best，所有报告指标来自该同一 checkpoint；公共预训练、活动依赖和其他项目权重不动。

审查边界：只读 source/primary 复核，不是新的数值等价 PASS、B128 容量证明、M0、正式结果或跨模型家族接受。
