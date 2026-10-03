# V5 两 GPU 模型分段：部署前 fresh 源码审查

时间：2026-10-03 17:52（Asia/Shanghai）。结论：**WARN；blocking_issues = []**。

这是新实现的本地静态审查。没有发现必须阻止登记的初始化、等价 witness 和工程 M0 的源码错误。以下结论不构成 forward/backward 等价、B128 容量、M0、50 epoch 或科学有效性 PASS。后续必须保留现有门槛；九个完整 batch 的 M0 全部通过以前，不得开始任何正式训练。

审查由 fresh Codex reviewer 执行，父代理确认 native spawn 请求为 `model=gpt-6-astra`、`reasoning_effort=max`、`fork_turns=none`。`review_independence=same-family`，`acceptance_status=provisional`。请求参数不是独立后端身份证明；本审查没有独立核验实际 serving model 或实际 reasoning effort。

## 范围与实际核对

完整读取新增 `partitioned_evidence_clip.py`、`run_native_partitioned.py`、两个 pair/backward 检查脚本、queue、report、V5 计划与可行性文档，以及私有 `C:/Users/gb/.codex_tmp/deploy_native_partition781.py`。实际依赖使用 `cpu_boundary_terminal780/_source`，没有把稀疏发布镜像误当完整 runtime，也没有因 CRLF 差异修改任何模块。

重点沿源码核对了 `run_independent_native_evidence.py`、`run_foundation_recipe.py`、`queue_foundation_recipe.py`、初始化/seed/batch/评价调用链、`correspondence_roles.py`、`evidence_author_heads.py`、`independent_native_roles.py`、native reader、Signal `meta_arch.py`、CLIP `model.py`、作者 optimizer 与三个 YAML 配方、CPU 距离比较器。

实际执行仅限标准库 AST/JSON/既有封存清单核对：六份新增 Python 和私有 launcher 共七份 AST 通过；私有远端 body 用字面占位值还原后 AST 通过，未执行。既有 §780 `source_sha256` 的 304 份 `_source` 文件全部匹配。未连接 SSH，未导入 PyTorch/模型，未初始化 CUDA，未启动训练、诊断或 scorer，未修改生产源码/权重，未部署或发布。裸 `python` 首次调用因本地 `pyvenv.cfg` 缺失而在执行前失败；随后使用已有 uv 管理的 Python、offline 模式完成标准库检查，没有安装环境。

## 已确认正确的源码路径

1. **参数与设备。** `partitioned_evidence_clip.py:17–26` 在 optimizer 创建以前迁移 class/position/camera、conv、ln_pre、前六 block 和第一个 adapter bank；后六 block、后两个 adapter bank、ln_post/proj 与全部角色/原生/BN/head 保留在 cuda:0。`Parameter` 对象身份及原有完整 state 值分别有精确断言。没有第二份注册参数库、dtype 转换或梯度累积。

2. **camera、pre-hook 与 capture。** 作者 `meta_arch.py:101–103` 已执行 `cam_label.to(self.cv_embed.device)`，所以 pre-hook 只搬图像不会造成跨卡 camera 索引错误。作者 `VisionTransformer.forward` 实际逐个调用 block，block 6 的 pre-hook 只替换第一个位置参数，保留 modality/index 与关键字参数。新 capture 的 role stack、写回 `output + sum(deltas) / 3`、RGB/NI/TI 顺序和最终 depth/role/modal stack 与原源码相同。第 4 层 snapshot 的 `.to('cuda:0')` 和 block 边界传输没有 detach/no_grad；autograd 路径保留。临时 capture hook 仍在 finally 删除，两个传输 pre-hook 持续存在。

3. **完整 batch 与作者监督。** 原 `_training_batch` 仍把原 batch、labels/cameras 放在 cuda:0；新 forward 不切 batch。角色、native reader、完整 BN、作者 CE/Triplet 和部署 L2 1536D 路径未修改。只有独立 backward witness 明确取前 32 个样本；M0/正式循环断言 batch 等于绑定值。三个 YAML 仍为 B64/K8、B128/K16、B64/K4，seed42、FP32 参数存储、AMP FP16、GradScaler256。

4. **optimizer 与训练包装不递归。** 新模块先保存原 foundation train 函数，随后替换 foundation.train。实际链为 `entry.main → 原 entry.train → 新 foundation.train 包装 → 预先保存的原 foundation.train`，返回后原 entry.train 继续补 `production_m0_diagnostics`。新 optimization 重置两卡峰值后调用原 optimization。作者 optimizer 按全模型 named_parameters 建组，既有断言验证所有 trainable 参数恰好覆盖一次；设备迁移已经完成。GradScaler 的 scale/backward/unscale/step/update 链未改变，没有跨卡平均或两个独立 optimizer。§780 原始记录的 runtime 为 PyTorch 2.5.1+cu121；本审查没有执行该 runtime。

5. **初始化、witness 和 M0 门。** full-batch initial pair 比较共同 state、batch/cfg、shared global、semantic/native raw/fused/heads。backward witness 使用相同初始 state 和输入，恢复 CPU RNG 与两张 CUDA RNG，保留 1e-5 forward、1e-4 参数 gradient 门，并检查 buffers、BN、完整 state、hook 清理和最终双 CUDA RNG。每个 variant 失败会使 subprocess 失败，未添加改精度、改门或重试路径。原 M0 保留八次实际 optimizer post-hook、BN8、所有活动参数累计非零梯度、14 个 native 参数八步记录及最终变化、fresh full-state strict reload。queue 完成并逐项核验全部九个 M0 后才进入 full phase。

6. **保存和评价。** foundation 保存完整 CPU `state_dict`，fresh 构建时先恢复同一分段布局，再 `load_state_dict(strict=True)`。M0 输出 reload 用 1e-5；正式只覆盖 `best_map.pth`，按 mAP/最后同分 epoch 的既有规则选择，最终 fresh reload 核对同一 checkpoint 的指标和距离。评价读取真实 query/gallery identity 与 camera/MSVR scene，沿原作者 evaluator 与独立 scorer 比较，不以另一模型输出作 ground truth。

7. **资源与私有部署。** 私有 launcher 固定 2026、检查原失败 controller/child 已退出、源码/公开 CLIP/新路径/磁盘预算，并在 (0,1)/(2,3) 中选整对空闲 GPU；其两卡 `CUDA_VISIBLE_DEVICES` 显式进入初始化子进程。M0/full 调度每 240 秒检查两个物理 pair，两卡均低于 500 MiB 且不属于 active job 才派发，最多两个 job/四卡。worker 为实际模型子进程重新明确物理 pair；控制进程本身只用标准库。没有 kill/preemption、2025 模型启动、自动再跑或旧路线重启。输出固定 `/home/gaob/trifusion-native-evidence-v5`；磁盘预算保留 18×384 MiB + 600 MiB + 256 MiB 与 2 GiB reserve。发布 receipt/报告尚未就绪时 launcher 会停止；本次未运行它。

## 非阻断问题和交付边界

**N1 — 双卡峰值只在正常返回后落盘。** `run_native_partitioned.py:43–51` 先调用原 train，成功返回后才写 `partition_peak_memory`。发生 OOM、梯度门或 reload 断言失败时，此写入不会执行；原 traceback/worker exit 会保留，不能说失败路径已经留下两卡峰值 JSON。这个风险有实际 V1 B128 在 stages stack OOM 的证据，不是虚构边界。它不让失败模型通过 M0，但限制失败容量诊断。若补齐记录，只需在新包装层保证原异常继续传播并记录已初始化设备的峰值，不应改变模型、gate 或自动重试。

**N2 — pre-assert 文件不是所有检查对象的完整差异记录。** `check_partitioned_backward.py:79–98` 已记录每个 trainable 参数的 None 状态/最大差异/fixed-gate 结果和完整 witness 输入元数据，满足主要梯度失败定位需求；但没有把随后 `heads`、buffers、CPU/双 CUDA RNG 的比较明细一起写入该文件（检查位于 101–118 行）。这些条件仍会阻止 PASS，因此不是放宽 gate；如果它们先失败，原内存中的明细不会保留。不要把现有文件描述成“所有检查对象均已 pre-assert 落盘”。必要补充应仅限已有对象的观测记录。

**N3 — 自动终局报告尚未兑现完整 CMC 汇总。** `report_native_partitioned.py:33–59` 输出原四指标及 aggregate paired diagnosis；其实际 `compare` 返回值没有完整 CMC/逐 query AP 数组。上游 `official_metrics` 虽计算作者 CMC，落盘 metrics 仅有 mAP、Rank-1/5/10；完整同-best 距离及身份/环境标签仍保存。报告还没有把已存于 `official_metrics.json` 的 final_evaluation_seconds 汇入 summary。该缺口无需 GPU、重训、改 checkpoint 或新增科学模块即可用已有同-best 数组/receipt 补齐，不阻止工程 M0；在补齐以前不得声称“完整 CMC 和全部训练/最终评价成本报告完成”。不要为此重选 epoch/seed 或修改原评分规则。

## 容量、旧失败与结论限制

独立读取 V1 原始 OOM traceback：RGBNT100 global_only 的完整 B128 在 `correspondence_roles.py:86` stack 时申请 218 MiB 失败；当时 PyTorch allocated 22.52 GiB。V4 pre-gate 文件重数：global_only 209 参数无 fixed-gate failure；semantic 281 参数中 94 个失败。§780 measured JSON 重数为 281 参数中 81 个失败、1323 packs/unpacks、仅 72 项 storage_offset 差异，没有 stride/layout 变化；旧 CPU-save 路线仍停止，不能由本审查推翻。

§780 packs 按每模态 441 项、0..205/206..440 分段独立复算：各模态前段 1,194,906,976 B、后段 1,255,505,760 B；三模态乘 B32→B128 的 4 倍分别为 14,338,883,712 B、15,066,069,120 B，与新计划一致。没有扣共享 storage、重复保存或 AMP cache，但这仍是旧 B32 logical-value 计数的线性估算，不是新两卡 graph 的峰值上界。约 22 GiB 后端估计中的 roles/workspace 是估计，余量有限。capacity780 原始 inventory 为四张 24 GiB 3090，拓扑没有 NVLink；四卡不能合成单一 96 GiB 设备。

允许结论仅为：**源码路径可以进入登记的工程验证，带以上 WARN；runtime/容量/等价/M0/检索效果全部待实际证据。** 任何 witness 或完整 batch M0 失败均原样保留并停止，不据此下调门槛或启动正式训练。
