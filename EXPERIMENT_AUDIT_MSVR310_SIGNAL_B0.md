# MSVR310 Signal B0 实验完整性审计

日期：2026-09-06（Asia/Shanghai）  
审计对象：MSVR310 Signal source-only 三折 B0 基线  
审计方式：本地只读文本/JSON/log/source 检查，加一个独立 stdlib 标量重放脚本。未启动远端任务，未读取模型/张量/图像库，未修改输入或执行侧报告。

## 总体结论：WARN

完整性结论：WARN。没有发现伪 ground truth、预测自归一化、phantom result、指标未落盘或 source/heldout 泄漏；WARN 来自证据边界和方法资格边界：这是 MSVR310 官方训练集内部身份隔离 OOF 基线，单 seed42、固定 epoch50、非官方 test、非多 seed、非严格作者复现、非 TriFusion 新方法资格证明。远端 checkpoint、retrieval_arrays、CLIP 权重和图像像素只通过收据与哈希记录阅读，本地未直接读取这些二进制/图片内容。

工程结论：B0 工程与结果封存可以判定为完成：三折 source 模型各 50 epoch/650 optimizer step，总 150 epoch/1950 step，固定 epoch50 后一次性 held-out gallery 评估；完整 600 query / 1032 gallery / 60 query 身份的 AP/Rank 标量可由原协议标签和完整排序独立重算。

评价类型：`real_gt_train_internal_identity_oof_baseline`。它使用真实 MSVR310 训练标签作为 ground truth，但不是官方 query/gallery test 协议。

## 输入与重放

我校验了 `C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-06_run17/input_manifest.json` 中 71 个 primary input 的字节数和 SHA-256，全部匹配；该 manifest 的文件数组从 `input_manifest.json:3` 开始，样例字段见 `input_manifest.json:5-7`，训练脚本条目见 `input_manifest.json:215-217`，fold receipt/training 条目见 `input_manifest.json:230-272`。独立输出记录 `input_count=71`、`all_ok=true`、`mismatch_count=0`：`C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-06_run17/independent_replay_output.json:5-11`。

本次限定跟进重新读取原 manifest 和 71 个输入，结果仍为 `input_count=71`、`all_ok=true`、`mismatch_count=0`。本次没有重复 AP/Rank 算术，因为跟进范围只覆盖 reviewer dispatch/provenance 与二进制证据 action item，输入、评分规则和执行侧报告均未改变；原独立算术结果仍以 `audit_msvr310_signal_b0_replay.py` 和 `independent_replay_output.json` 为依据。

我写入并运行的独立脚本：

```powershell
& 'C:\Users\gb\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' 'C:\Users\gb\.trifusion_github_publish_22c3bee\.aris\traces\experiment-audit\2026-09-06_run17\audit_msvr310_signal_b0_replay.py'
```

第一次尝试导入 NumPy 时，指定解释器返回 `ModuleNotFoundError: No module named 'numpy'`。随后脚本改为纯 stdlib 标量计算；最终运行用时 0.17811989999609068 秒，输出 `result=PASS`。脚本中 manifest 校验、AP/Rank 重算和训练 step 核验入口分别在 `audit_msvr310_signal_b0_replay.py:26`、`audit_msvr310_signal_b0_replay.py:68`、`audit_msvr310_signal_b0_replay.py:95`、`audit_msvr310_signal_b0_replay.py:255`。AP 分母按真实正例数计算，位置精度求平均见 `audit_msvr310_signal_b0_replay.py:68-89`；完整 gallery 排序断言、scene/camera 口径差异记录、AP 对比见 `audit_msvr310_signal_b0_replay.py:152-177`；训练逐 step loss、source-only sampled indices、LR schedule 核对见 `audit_msvr310_signal_b0_replay.py:292-318`。重放输出的聚合结果与报告值完全一致：mAP 53.1293805608712、Rank-1 63.0、Rank-5 77.0、Rank-10 82.83333333333334，见 `independent_replay_output.json:158-168`。

我直接检查的证据：manifest、配置、协议 JSON、完整 categorical rankings JSON、summary/receipt/training JSON、source text snapshots、日志、re.txt、结果/计划/tracker 文本，以及执行侧 file/array verification 的 JSON 文本。  
我只通过收据读取的证据：远端 `signal_epoch50.pth` 三个 checkpoint、三份 `retrieval_arrays.pt`、CLIP 权重文件、MSVR310 图像像素、远端 GPU/Torch 运行时和远端文件本体。远端 file verification 声称核对了 3 个 checkpoint 与 3 个 retrieval array，并给出字节和 SHA，例如 fold0 checkpoint / retrieval array 在 `trifusion_msvr310_signal_v1_baseline_file_verification_20260906.json:18-24`；汇总计数在 `trifusion_msvr310_signal_v1_baseline_file_verification_20260906.json:103-108`。这些收据支持工程追溯，但不是我本地直接读取二进制后的证据。

## A. Ground Truth Provenance：PASS

ground truth 来自 MSVR310 原始训练标签/文件名解析，不是模型输出。原 Signal 数据集解析从目录和文件名读 identity、camera、scene：`C:/Users/gb/.codex_tmp/msvr310_signal_source_text_20260906/data/datasets/msvr310.py:67-88`，其中 `img[0:4]`、`img[11]`、`img[6:9]` 分别作为 identity、camid、sceneid。数据安装记录显示 MSVR310 已安装并配对，训练 split 为 155 个 identity / 1032 个 aligned triplet：`C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/msvr310_dataset_install_20260905.json:2-15`。

协议文件明确 scope 是“Existing official-training labels only; no image, feature, model, tensor or retrieval execution”，source split 是 `bounding_box_train`，split rule 是标签确定性 round-robin，不随机、不按特征/排名/困难身份选择：`C:/Users/gb/.trifusion_github_publish_22c3bee/protocols/msvr310_train_oof_v1.json:3-11`。标签支持证据记录 1032 条 train records、155 个 train identities、60 个跨 scene 正例身份和 600 个 scene-filter eligible query：`C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/trifusion_msvr310_source_label_support_20260906.json:6-7`、`trifusion_msvr310_source_label_support_20260906.json:70-71`。执行侧协议核验也记录全部 1032 records 与原训练标签匹配，source/heldout 分离，heldout union 覆盖训练记录一次，且原 train/test identities 分离：`C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/trifusion_msvr310_train_oof_protocol_verification_20260906.json:8-11`。

必要限定：这是官方训练部分内部 OOF ground truth，不是官方 query3 / bounding_box_test 评价；结果报告也明确“未访问官方测试图片”：`C:/Users/gb/.trifusion_github_publish_22c3bee/results/TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:6-9`。

## B. Metric Denominator And Normalization：PASS

没有发现把指标除以模型自身 max/min/mean 的归一化。评估分数使用标准 ReID AP：先按距离排序，再移除同身份同 scene，AP 分母是 `orig_cmc.sum()`，即该 query 的真实保留正例数：`C:/Users/gb/.codex_tmp/msvr310_signal_source_text_20260906/utils/metrics.py:68-100`。B0 训练脚本自己的 `scene_scores` 使用相同逻辑：`C:/Users/gb/.trifusion_github_publish_22c3bee/tools/train_msvr310_signal_oof.py:223-238`。mAP 是 per-query AP 的平均值乘 100，Rank-k 是 first positive rank 是否不超过 k 的平均：`tools/train_msvr310_signal_oof.py:235-237`、`tools/train_msvr310_signal_oof.py:360-362`。

评估前对特征做 unit L2 normalize，然后用平方欧氏距离排序：`tools/train_msvr310_signal_oof.py:253-257`。这会改变检索距离的尺度，但不是把结果指标按模型自身统计量归一化。脚本还调用上游 `eval_func_msrv` 并要求 mAP 差小于 `1e-10`、Rank 差小于 `1e-5`：`tools/train_msvr310_signal_oof.py:260-265`。执行侧 summary 的 upstream 差异为 0 或微小浮点差，见 `C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:54817-54820`、`115691-115694`、`176366-176369`。我方独立脚本按同一离散排序和 dataset label 重算，聚合指标与 summary 完全一致：`independent_replay_output.json:158-168`。

必要限定：报告中的所有指标单位为百分比；不能与不同 query/gallery 协议的公开官方数字直接相减，结果文本已经这样限定：`TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:13-14`、`79-82`。

## C. Result Existence / Number / Status：PASS

实际结果存在并与声明数字匹配。B0 summary 状态是 `COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION`：`C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/trifusion_msvr310_signal_v1_baseline_complete_20260906.json:4`；聚合指标记录在 `trifusion_msvr310_signal_v1_baseline_complete_20260906.json:181782-181793`。结果报告声明三折各 50 epoch/650 更新，总 1950 更新、600 query mAP 53.129380561、Rank-1 63.000000000%，并标明未访问官方测试图片：`TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:3-9`。三折与合并表格值见 `TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:16-21`，与 summary 和我方 replay 一致。

日志末尾有完成事件：`C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/trifusion_msvr310_signal_v1_baseline_run_20260906.log:225`。tracker 记录 B0 为 `COMPLETE_BASELINE_NOT_METHOD_QUALIFICATION`，执行侧核验为 DONE，独立审计在当时仍 PENDING：`C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/msvr310_signal_v1/EXPERIMENT_TRACKER.md:15-17`。这不是结果状态矛盾，而是本审计前的 tracker 状态；按任务要求我没有编辑 tracker 输入。

文件存在性方面，我校验了 manifest 中所有 71 个本地输入文件的 byte/SHA；执行侧 file verification 文本还记录远端 23 个文件、checkpoint/retrieval array 数、receipt/summary 一致性、source bindings：`trifusion_msvr310_signal_v1_baseline_file_verification_20260906.json:3`、`103-108`。我方独立 replay 记录 `input_count=71`、`mismatch_count=0`：`independent_replay_output.json:5-11`。

必要限定：checkpoint 和 retrieval arrays 本体未在本地 manifest 中提供；我没有直接 hash 它们，只读取 file verification 的收据。

## D. Metric Call Path / Output Retention：PASS

声明使用的 B0 指标路径被实际调用，且 per-query 输出被保留。训练脚本在 `evaluate_gallery` 内生成 gallery rows 与 query positions，抽取 ids/cameras/scenes，计算 normalized feature distance，然后调用本地 `scene_scores` 和上游 `eval_func_msrv`：`tools/train_msvr310_signal_oof.py:241-265`。它随后保存 `retrieval_arrays.pt`，并把 `metrics`、`average_precision`、`first_match_rank`、`gallery_manifest`、`query_rows`、`feature_width` 和 `retrieval_arrays_sha256` 写入 receipt/summary：`tools/train_msvr310_signal_oof.py:266-271`。summary 中每折都有 `average_precision` 和 `first_match_rank` 数组起点，例如 fold0 在 `trifusion_msvr310_signal_v1_baseline_complete_20260906.json:54385-54820`。

完整 categorical rankings 从原保存距离导出，并保留 gallery record indices、query gallery positions、sorted gallery positions：`C:/Users/gb/.trifusion_github_publish_22c3bee/evidence/trifusion_msvr310_signal_v1_baseline_saved_rankings_20260906.json:2-3`、`7`、`369`、`581`。执行侧 file verification 记录保存距离重算最大差 0 且 rankings 来自保存距离：`trifusion_msvr310_signal_v1_baseline_file_verification_20260906.json:120-122`、`134-136`、`148-150`。我方 replay 未调用执行侧 verifier，而是从 `sorted_gallery_positions` 和 protocol label 重算 600 个 AP/Rank；`max_ap_abs_diff` 在各 fold 为 `3.33e-16`、`2.22e-16`、`2.22e-16`：`independent_replay_output.json:85-97`、`114-126`、`143-155`。

必要限定：上游 `utils/metrics.py` 里还存在通用 `eval_func`、`R1_mAP`、`R1_mAP_eval` 等 helper；它们不构成本次 B0 数字的证据。B0 数字只依赖 direct `scene_scores` 与 `eval_func_msrv` 交叉核对路径。

## E. Scope / Seed / Protocol Limitations：WARN

范围必须严格限定。配置记录 `seed=42`、`epochs=50`，scope 是“New dataset Signal baseline only; not V24 promotion or a new TriFusion main method”：`C:/Users/gb/.trifusion_github_publish_22c3bee/configs/MSVR310/Signal-source-oof-v1.json:3-5`。计划文件明确本计划不包含正式官方评估、结构消融、多种子、车辆候选或失败版本重跑：`C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/msvr310_signal_v1/EXPERIMENT_PLAN.md:4-5`。同一计划还限定 source/heldout OOF、完整 gallery、camera/scene 口径、seed42/B64/K8、固定 epoch50 和不选 best：`EXPERIMENT_PLAN.md:19-22`、`44-48`、`61-65`。

B0 结果文本总体遵守这些限定：称为“官方训练集内部的身份隔离基线”，明确新 TriFusion 方法资格未评估，不能据此宣称跨数据集改进机制成立或稳定优于公开方法：`TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:8-9`、`23-27`。它也说明 B64/K8 与同步几何相对作者 K4/原增强存在差异，不称严格逐项作者复现：`TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:73-77`。

我给 E 项 WARN，是因为证据本身的外推能力有限，而不是因为当前结果文本严重越界。后续任何论文/README/issue 若把该结果写成官方测试复现、SOTA、多 seed 稳健性、跨数据集泛化或 TriFusion 方法有效性，都不被本审计支持。

## F. Evaluation Type Classification：PASS

分类为 `real_gt`，子类型为 `train_internal_identity_oof_baseline`。协议评价类型写明 `train_internal_identity_oof_not_official_test`：`protocols/msvr310_train_oof_v1.json:13-22`。summary 记录 `evaluation_type` 为 `train_internal_identity_oof_baseline`，`official_test_image_access=0`、`fixed_rgbnt201_dev_image_access=0`、`expert_training=0`、`epochs_selected_by_heldout=false`：`trifusion_msvr310_signal_v1_baseline_complete_20260906.json:181775-181779`。我方 replay 输出相同状态：`independent_replay_output.json:780-789`。

不是 `synthetic_proxy`，因为正例/负例来自 dataset label 与 scene；不是 `self_supervised_proxy`，因为 AP/Rank 有明确 identity label；不是 `simulation_only` 或 `human_eval`。

## 额外完整性检查

source/heldout identity 隔离：协议每折 source 与 heldout disjoint，执行脚本也在每折断言两者无交集：`tools/train_msvr310_signal_oof.py:311-314`。我方 replay 对每折重新检查 source/heldout disjoint、gallery union 覆盖全部 1032 records 一次，输出见 `independent_replay_output.json:68-84`、`99-113`、`128-142`、`170-173`。

scene/camera 过滤：协议要求 remove 是“same identity AND same scene”，retain 包括所有 different-identity gallery records：`protocols/msvr310_train_oof_v1.json:15-19`。上游 MSVR310 evaluator 实际在 `eval_func_msrv` 里用 scene removal：`utils/metrics.py:55-69`；B0 direct scorer 也用 `(same id) & (same scene)`：`tools/train_msvr310_signal_oof.py:227-230`。执行侧协议核验显示如果错用 camera-only，候选 query 的正例数会大量变化：fold0 325、fold1 310、fold2 291，见 `trifusion_msvr310_train_oof_protocol_verification_20260906.json:16-21`、`2548-2553`、`5003-5008`。我方 replay 在最终 valid query 子集也记录 scene/camera 口径差异：fold0 175、fold1 168、fold2 151，见 `independent_replay_output.json:80-82`、`109-111`、`138-140`。

完整 distractor gallery 保留：协议聚合记录 1032 unique gallery records、600 valid queries、432 个 excluded query records 仍留在 gallery、95 个 gallery-only distractor identities：`protocols/msvr310_train_oof_v1.json:22576-22582`。结果报告同样声明 95 个 single-scene heldout 身份的 432 records 作为干扰保留：`TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:23-25`。我方 replay 对三折分别确认 excluded records retained in gallery 为 150/142/140，single-scene distractor identities 为 32/32/31：`independent_replay_output.json:79-81`、`108-110`、`137-139`。

固定 endpoint 而非选择：配置要求 `fixed_epoch50; no heldout evaluation until final checkpoint`：`configs/MSVR310/Signal-source-oof-v1.json:56`。训练脚本非 preflight 时先检查 M0 R2 工程 PASS，再每折训练 50 epoch、保存 `signal_epoch50.pth`、严格重载后评估；最终 summary 写 `checkpoint_selection=fixed_epoch_50`、`epochs_selected_by_heldout=false`：`tools/train_msvr310_signal_oof.py:291-305`、`320-347`、`364-368`。summary 与我方 replay 均记录相同状态：`trifusion_msvr310_signal_v1_baseline_complete_20260906.json:181779-181793`、`independent_replay_output.json:775-778`。

初始化、参数和 update 账目：每折从 `new_model` 重新构建 Signal teacher，并冻结已证实不被 Adam 更新的 token selection 参数：`tools/train_msvr310_signal_oof.py:102-110`。训练循环记录 initial/final state、trainable/gradient tensor、optimizer groups、history、steps：`tools/train_msvr310_signal_oof.py:135-204`。我方 replay 核对每折 50 epoch、650 step、195/195 trainable/gradient tensors、无 `trainable_without_gradient`，总 1950 step，150 个日志 epoch row 与 history 一致：`independent_replay_output.json:695-778`。原日志显示每个 epoch 13 step，LR 在 epoch20/40 衰减，例子见 `trifusion_msvr310_signal_v1_baseline_run_20260906.log:22-41`、`61-71`、`115`、`135`、`205`、`215`，完成事件见 `baseline_run_20260906.log:225`。

原失败与后续修复：原 M0 R1 在 fold0 8 step 后失败，退出 1，原因是 6 个 trainable selection tensor 无梯度；R2 prereg 记录原失败、0-update 诊断、冻结六个 tensor 和不在此时执行新 M0/B0：`trifusion_msvr310_signal_v1_m0_r2_preregistration_20260906.json:4-27`。M0 报告说明 R2 三折工程通过、M0 无 held-out/dev/official 前向，B0 不继承 M0 checkpoint：`TRIFUSION_MSVR310_SIGNAL_SOURCE_M0_2026-09-06.md:16-24`、`28-35`。B0 结果报告也将 R1/R2/诊断成本保留且不计正式 epoch：`TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:50`。

source engineering 与 retrieval evidence 的区分：M0 是工程检查，不是检索结果，报告明确无 held-out/dev/official 图像前向：`TRIFUSION_MSVR310_SIGNAL_SOURCE_M0_2026-09-06.md:22-24`。B0 才有完整 held-out gallery retrieval，summary 记录三折 heldout forwards 360/349/323，总 1032：`trifusion_msvr310_signal_v1_baseline_complete_20260906.json:60889`、`121612`、`181761`、`181793`。

## Reviewer Dispatch 元数据

适用规则：`reviewer-independence.md` 要求 reviewer 直接判断 primary artifacts，不能接收 executor 对内容的预消化叙述：`C:/Users/gb/.codex/skills/shared-references/reviewer-independence.md:3-21`。`reviewer-routing.md` 说明默认 reviewer 走 `spawn_agent`/`send_input`，Codex reviewer 是同家族 reviewer，只能算 Type-A，不是跨家族 Type-B；Codex executor 的 Type-B 需要 Claude/Gemini overlay，Oracle Pro 仍属 GPT 家族：`C:/Users/gb/.codex/skills/shared-references/reviewer-routing.md:9-15`、`59-60`、`73-74`。

本次 root 确实派发了当前 reviewer：request 记录 `tool=spawn_agent`、`task_name=audit_msvr_signal_b0`、`model=gpt-5.5`、`reasoning_effort=xhigh`、`fork_turns=none`：`C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-06_run17/001-complete-baseline.request.json:5-10`。dispatch 观察记录 `tool=collaboration.spawn_agent`、`requested_model=gpt-5.5`、`requested_reasoning_effort=xhigh`、`fork_turns=none`、`returned_task_name=/root/audit_msvr_signal_b0`、`status=accepted_and_running`：`C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-06_run17/dispatch_observation.json:3-9`。meta 记录当前 `agent_id=/root/audit_msvr_signal_b0`、请求模型/推理强度、`status=ok`、`input_files_rechecked=71`：`C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-06_run17/001-complete-baseline.meta.json:5-8`、`20-21`。

这些文件只支持“root 请求并接受了一个 gpt-5.5/xhigh 配置的 Codex-family reviewer 任务”这一元数据结论，不构成独立后端身份认证；`resolved_backend_independently_attested=false` 同时出现在 dispatch observation 与 meta：`dispatch_observation.json:8`、`001-complete-baseline.meta.json:9`。当前 reviewer 没有再委派下级 reviewer；任务中的 no-delegation 要求只是执行边界，不提供任何关于模型可用性或不可用性的证据。review-family scope：Codex/GPT-family Type-A 审计；不声称 Type-B cross-family acquittal。

## Claim Impact

支持：

- B0 工程完成：三折 source-only Signal 模型各 50 epoch / 650 step，总 1950 step；证据见 `TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:5-7`、`independent_replay_output.json:695-778`。
- 内部 OOF 指标：600 query / 1032 gallery 的 mAP 53.1293805608712、Rank-1 63.0、Rank-5 77.0、Rank-10 82.83333333333334；证据见 `independent_replay_output.json:158-168`。
- 标签和排序算术完整：真实 dataset identity/scene label、完整 gallery 排序、per-query AP/Rank 与 summary 一致；证据见 `protocols/msvr310_train_oof_v1.json:13-22`、`independent_replay_output.json:68-173`。
- M0 R1 失败已保留，R2 是工程 pass 且不计为 B0 检索结果；证据见 `TRIFUSION_MSVR310_SIGNAL_SOURCE_M0_2026-09-06.md:16-24`、`28-35`。

需要限定：

- “MSVR310 Signal baseline”只能写成“官方训练集内部身份隔离 OOF source-only baseline, seed42, fixed epoch50”；不能省略 internal/train-only/single-seed/fixed-endpoint 限定。
- “source 文件和远端 artifact 已核验”只能写成“本地文本/JSON/receipt 已核验；远端 checkpoint/retrieval arrays/图像像素未由本审计直接读取”。
- “接近公开 mAP 数字”不能写成官方复现，因为官方 query/gallery 条件不同，结果文本已在 `TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md:79-82` 限定。

不支持：

- TriFusion 新方法有效性、V24/RGBNT201 晋级、跨数据集改进机制成立。
- 官方 MSVR310 test 结果、SOTA、公开榜单比较、多 seed 稳健性、统计显著性。
- 严格逐项复现 Signal 作者训练设置；当前 B64/K8 与同步几何等差异已登记。
- 对本地未提供的 `.pth`、`.pt` 或图像像素做“我已直接检查”的声明。

## Action Items

- 后续任何方法比较必须另立合同：同一 train-internal OOF split、同一 fixed endpoint、同一 query/gallery/scene filter，并在看 held-out 分数前冻结。
- 在本次合同/计划边界内，二进制证据应继续表述为远端 artifact 收据与本地 JSON/SHA 可追溯；计划明确“特征、距离、checkpoint留在远端，JSON逐query数值及文件SHA可归档本地”，且本地代码检查不 import 模型、张量、图片或执行检索：`C:/Users/gb/.trifusion_github_publish_22c3bee/refine-logs/msvr310_signal_v1/EXPERIMENT_PLAN.md:67`、`87`。不得追改本 run17 的 input manifest，也不得把当前审计改称二进制直接复核。
- 如果未来需要 `.pth`、`.pt`、CLIP 权重或图像像素的直接二进制复核，应新建明确授权的 provenance contract 和新 manifest，再做 byte/SHA 与只读解析；这不是本次 metadata-only 跟进或原 B0 审计的补丁项。
- 如果要写论文式结果，只能把本 B0 放在“内部基线/工程基线”段落，并保留 single-seed、train-only、非官方 test、非严格作者复现限定。
- 在真正执行官方 test、多 seed 或消融前，不得宣称 MSVR310 官方性能、泛化性能、稳健性或新方法资格。

## 完整报告输出

- Markdown：`C:/Users/gb/.trifusion_github_publish_22c3bee/EXPERIMENT_AUDIT_MSVR310_SIGNAL_B0.md`
- JSON：`C:/Users/gb/.trifusion_github_publish_22c3bee/EXPERIMENT_AUDIT_MSVR310_SIGNAL_B0.json`
- 独立重放脚本：`C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-06_run17/audit_msvr310_signal_b0_replay.py`
- 独立重放输出：`C:/Users/gb/.trifusion_github_publish_22c3bee/.aris/traces/experiment-audit/2026-09-06_run17/independent_replay_output.json`

本次限定跟进只修正 dispatch/provenance 和 action-item 表述；A-F、科学范围、指标算术和 WARN 判定不变。

最终判定：WARN。B0 内部基线的工程、标签、排序算术和结果落盘可信；方法资格、官方复现、多 seed 稳健性和二进制/图像直接复核不在本证据支持范围内。
