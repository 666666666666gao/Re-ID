# Native original/original 控制源码审查

时间：2026-10-03T18:55:50.7788421+08:00。结论：**PASS / SOURCE_ONLY_PASS**。未发现部署前必须修复的具体源码缺陷；本次审查没有执行控制或证明数值门通过。

`review_independence=same-family`，`acceptance_status=provisional`。父任务随后明确确认，实际 native spawn 参数为 `gpt-6-astra / max / fork_turns=none`；服务端实际模型、后端和推理强度未独立验证，不声称跨模型家族验收。

## BLOCKING 与 NONBLOCKING

- **BLOCKING：无。**
- **NONBLOCKING 源码缺陷：无。**
- 所需补丁：无。以下未执行项目属于审查范围限制，不是新要求或已测失败。

## 核对结果

1. **原始模型与固定起点正确。** `check_native_original_repeat.py:45–79` 直接调用两次未改动的 `run_independent_native_evidence.build_core`，核对 cfg、完整初始 state、public CLIP、协议、作者源码、head 名与 trainability，并要求全部参数 FP32/cuda:0。沿捕获源码核对 builder、fresh Signal factory 和 native 构造，没有调用分区、checkpoint、CPU-save 或 undo helper。两次 fresh 构造对应两个模型，不复用前一次 forward graph。

2. **输入和 RNG 控制正确。** 脚本 `:83–95` 只取一次作者 B64/K8 loader 的前32项，labels/cameras/paths/三个图像摘要必须匹配旧 native 主记录。两次 pass 均恢复同一 CPU 与两张可见 CUDA RNG，保持原始 train mode、作者 head/loss 与 optimizer。原始 loss 构造会为未使用的 center criterion 消耗 CPU RNG，两次都在恢复后按相同路径执行；未添加 backend、layout、stream 或确定性开关。

3. **执行量与比较门正确。** `:97–166` 每模型一次 autocast FP16 forward、GradScaler256 backward、unscale；295 项梯度保留 key/None/shape/finite 与固定1e-4比较，raw/fused/global/loss 和每个 head score/feature固定1e-5。精确 buffer、CPU/双CUDA RNG、BN1、完整 post-state、所有参数未变、原始 capture hook清理及 optimizer ownership/groups 均有门。原始 `AuthorHeadEvidence.train` 恢复 Signal 的训练状态，capture hooks 在原 `finally` 中移除。

4. **失败可观测性与停止边界正确。** `:167–184` 在总门槛断言前写完数值与后续状态比较，失败梯度不会再次遮住 head/buffer/RNG/BN/state 结果；仅所有门通过才写 PASS。调用链没有 optimizer.step、scheduler.step、scaler.update、评分、权重保存、重试或自动后继。初始 state/input 不匹配则在 forward 前失败停止，运行异常交给 controller 留存 child log/exit。

5. **一次性 controller 正确。** `run_native_original_repeat_control.py:17–43` 固定原 remote ROOT 与 `CUDA_VISIBLE_DEVICES=0,1`，独占创建 child log，只 Popen一次并等待退出，记录实际 child exit，exit0必须有规定 PASS 标记，没有重试或训练后继。计划展示的模型命令是该 controller实际执行的 child命令。

6. **私有部署 helper 的静态路径正确。** `deploy_native_original_control784.py:11–78` 检查已发布 head、无阻断 review、现有312源码+5新增归属文件、两个脚本AST、旧controller消失、磁盘、公用CLIP和实际 GPU0/1均低于500MiB，创建唯一 output及317文件快照后只启动一个 controller，固定可见0,1。未选择2/3、抢占、重试或启动训练。SSH命令固定，嵌入值使用 Python repr；加载 known_hosts，没有自动接受新host key。这里审的是代码，尚未证明这些远端前置检查已实际通过。

## 主证据与依赖核对

- 本 reviewer直接重计已保存的 V5 native预断言JSON：295个唯一梯度key，90个已存 fixed_gate失败，None项0；raw/fused/global/loss记录差值均0，初始state为 `7f5ff300a45faed287a510dacc68d06c006931504a4510fffd8c47e43d93b852`。这只是已保存布尔结果重计，旧完整梯度未存，未重新计算 tensor allclose。
- 旧 `check_partitioned_backward.py:101–102` 的 head1e-5门在失败梯度 `:109` 前执行并通过；`:111–122` 的 buffer/RNG/BN/state/hook门未执行。保留这一勘误。
- 对12个与本调用链有关的本地原始依赖做直接逐字节比较，12/12与指定 `partition_terminal_intake781/_source` 相同；文件清单与字节数写入JSON。没有把这项核对扩大称为本 reviewer验证过远端312/317文件。
- 已读新脚本189行、controller43行、计划23行、私有helper78行，及所需捕获的 builder、输入、loss、optimizer、scheduler、模型与hook实现。稀疏本地仓库中缺少的作者源文件，按父任务要求读封存 `_source`。

## 实际工具范围与验收限制

本次只做本地 PowerShell源码/文本读取、rg搜索、JSON解析/重计、文件元数据和逐字节比较，并写指定 reviewer目录。一次本地 `python --version` 探测返回 `No pyvenv.cfg file`；未执行目标脚本、未导入 Torch/模型，本 reviewer没有独立跑AST编译，未把父任务已完成的AST冒充为自己的结果。没有SSH连接、CUDA、scorer、训练、更新、权重保存或生产文件编辑。

第783节五副本封存是父任务提供的前提。本 reviewer未独立验证远端资源、已部署317文件、未来第784节发布或实际启动；这些仍须按既有计划在启动前完成。此PASS仅覆盖源码门。

控制若通过，只能支持这次新受控的 original-native重复witness；**V5仍FAIL，不产生重启、M0、正式训练、具体修复或容量证明**。固定门若失败，应封存并停止GPU parity试验，仅继续源码分析；state/input/runtime失败也停止，不更换batch/seed/precision或重试。旧完整梯度和pre-forward RNG未保存，元数据匹配不能声称历史张量重建。九臂seed42/full50目标、全部M0门及RGBNT100 B128容量仍未完成。
