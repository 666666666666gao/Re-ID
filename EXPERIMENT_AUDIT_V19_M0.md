# V19 T0/M0 实验完整性审计

**日期**: 2026-09-05  
**审计员**: GPT-5.5 xhigh，独立只读审计  
**项目**: TriFusion RGBNT201 V19 私有语义尾部 T0/M0  
**总体结论**: **WARN**  
**工程完整性**: **PASS**  
**科学资格**: **WARN，尚未获得 Q1/D1/dev/official 结果**

本次审计只覆盖 V19 T0/M0 快照和关联执行合同。结论是：V19 M0 的工程合同有充分本地证据支持，未发现假 GT、自归一化、M0 数字错配或 M0 代码路径伪造；但该快照 `status=RUNNING`，`folds=[]`，只证明 M0 与前置检查，不证明 Q1、D1、30-dev、official、跨数据集、SOTA、因果归因或等可训练容量比较。源 checkpoint、数据集图像和远端训练环境字节没有被本地审计者直接持有，只能按已记录 SHA/receipt 标为远端断言。

## A. Ground-Truth Provenance: PASS

**证据**

- `configs/RGBNT201/TriFusion-signal-preserving-v19-private-tail-rtx3090.yml:6-13` 指向远端 RGBNT201 数据根、固定 `protocols/rgbnt201_dev_v1.json`、B64/K8、cross-camera sampler。
- `protocols/rgbnt201_dev_v1.json:2-13` 记录协议检查均为 true；`protocols/rgbnt201_dev_v1.json:126-131` 说明 30 个 dev 身份由 train_171 内部合格身份的 salted SHA 规则选出，`uses_test_labels=false`。
- `tools/build_v12_complete_path_oof_targets.py:156-173` 从协议的 `train_ids` 加载 `train_171` 记录并校验 141-fit 与 triplet 数；`tools/build_v12_complete_path_oof_targets.py:29-55` 只对 fit records 重标号，heldout records 保持原身份并计算 overlap。
- `tools/train_signal_preserving_v19.py:62-68` 启动时验证 V12 source checkpoint SHA 和 V12 summary SHA；`tools/train_signal_preserving_v19.py:75-95` strict 加载 Signal 与 expert state，并检查 checkpoint payload 的 fit/heldout 身份集合。
- `modeling/trifusion/aligned_data.py:91-94` dataset item 返回图像、identity、camera、view 和文件名；`tools/run_signal_preserving_v5.py:568-584` 将这些 labels/camera_ids 转为训练 batch。
- `evidence/trifusion_v19_m0_seed42_4b749cd.json:26` 将本快照标为 `real_gt_train_internal_complete_path_oof`；`evidence/trifusion_v19_m0_seed42_4b749cd.json:33-36` 记录 dev/official access 均为 0，D1 未执行，next phase 未 qualified。

**细节**

M0 使用数据集身份标签和 camera 标签进行训练内检查，不从模型输出生成 ground truth。V12 OOF target cache 是历史训练域代理/教师材料，但本次 V19 M0 不把它作为评价 GT；V19 绑定的是 V12 的 Signal/expert checkpoint 和身份 split。Q1 若完成，将使用自定义 full-gallery ReID 评价而非 official test：`tools/train_signal_preserving_v18.py:133-151` 抽取五路特征，`tools/audit_v17_full_gallery.py:14-42` 保留所有 heldout gallery 并只排除无跨 camera 正例的 query。

**行动项**

- Q1 终态审计必须重新核验远端 checkpoint 字节、数据路径、full-gallery query/gallery manifest 和全部 AP/rank 数组。

**Claim Impact**

- “M0 使用真实数据标签和身份隔离训练域 source-only 检查”: **supported**。
- “V19 已有 dev/official/SOTA 性能结果”: **unsupported**。

## B. Score Normalization: PASS

**证据**

- `modeling/trifusion/signal_preserving_v8.py:702-711` 使用 cross-entropy 与 batch-hard triplet；`modeling/trifusion/criterion.py:17-34` 的 triplet 分母来自 batch rows，不来自 prediction max/min。
- `tools/train_signal_preserving_v19.py:151-160` 按 frozen config 的七路 loss 权重求和；权重在 `configs/RGBNT201/TriFusion-signal-preserving-v19-private-tail-rtx3090.yml:39-47` 固定。
- `tools/train_signal_preserving_v19.py:384-391` 以 label smoothing 解析下界计算 overfit excess ratio；对应 receipt 为 `evidence/trifusion_v19_m0_seed42_4b749cd.json:8332-8334`。
- `tools/diagnose_v6_oracle_complementarity.py:91-104` per-query AP 是排序后的 precision at matched ranks；`tools/audit_v17_full_gallery.py:39-41` 将 raw AP / Rank 命中率乘以 100。
- `tools/train_signal_preserving_v18.py:149-150` Q1 评价会先 L2-normalize features 后计算 pairwise distance，这是 ReID 表征归一化，不是用模型自身最大值归一化分数。

**细节**

未发现把指标除以模型自身 max/min/mean 的自归一化。V19/V8 内部的 `F.normalize` 和 residual bank scaling 属于特征构造；M0 overfit ratio 使用解析 label-smoothing floor，仅是训练内工程门，不是检索性能指标。

**行动项**

- Q1 完成后继续按 raw AP、Rank-1/5/10 和逐 query 数组审计，不接受只给归一化汇总分。

**Claim Impact**

- “M0 过拟合比值 0.0595140216 ≤ 0.1”: **supported**，我从 receipt 数组重算得到 `0.0595140216437626`。
- “M0 证明检索 mAP 提升”: **unsupported**。

## C. Existence And Reproducibility: WARN

**证据**

- T0/launch receipt 存在并记录 commit、config SHA、plan SHA、远端 output/log/screen 和启动命令：`evidence/trifusion_v19_t0_and_launch_20260905.json:3-14`；四项单测 receipt 在 `evidence/trifusion_v19_t0_and_launch_20260905.json:16-20`。
- M0 快照存在，顶部记录 `status=RUNNING`、commit、runner/config/plan SHA：`evidence/trifusion_v19_m0_seed42_4b749cd.json:3-9`；其 source file hash map 在 `evidence/trifusion_v19_m0_seed42_4b749cd.json:12-24`。
- M0 pass 和六个布尔门在 `evidence/trifusion_v19_m0_seed42_4b749cd.json:8130-8137`；capacity 两端 8-step 记录在 `evidence/trifusion_v19_m0_seed42_4b749cd.json:8141-8204`；100-step overfit 记录在 `evidence/trifusion_v19_m0_seed42_4b749cd.json:8207-8334`。
- Transfer receipt 说明本地 `trifusion_v19_m0_seed42_4b749cd.json` 是 M0 后、Q1 仍运行的 run_summary 快照，并与远端 SHA 核对：`evidence/trifusion_v19_m0_transfer_receipt_20260905.json:2-9`。
- Tracker 记录 V19-T0 与 V19-M0 为 `DONE_PASS`，Q1 为 `RUNNING`：`refine-logs/v19/EXPERIMENT_TRACKER.md:5-8`；同文件 `refine-logs/v19/EXPERIMENT_TRACKER.md:11-19` 写明 Q1 尚无完整最终检索结果，dev/official 仍为 0。
- Master handoff §35.1 的数字与 receipt 一致：`docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:1951-1978`。

**细节**

本地所有请求文件均存在。独立 SHA 核验结果：plan `2b7674cf...afe7b`、config `89f7335a...97dd5`、runner `5f000d3d...b9c5`、V19 module `69989cce...918e`、M0 JSON `89fd884a...e992`，且 M0 JSON SHA 与 transfer receipt 完全一致。M0 receipt 的 source hash map 与本地文件语义一致；其中 `criterion.py` 和 `protocols/rgbnt201_dev_v1.json` 的 Windows CRLF 原始字节 SHA 与 receipt 不同，但 LF-normalized SHA 分别匹配 receipt 的 `0b2a6370...3740a` 和 `d916e7da...81946`。这说明可复现 ledger 使用的是远端/Linux LF 字节；本地 byte-for-byte 审计需记录该 EOL caveat。

限制：V12 source checkpoint、CLIP weight、RGBNT201 图像和运行中的 Q1 checkpoint 均只在远端路径中声明，未被本地审计者直接读取字节。`tools/train_signal_preserving_v19.py:62-66` 会在远端运行时 assert 这些 SHA，config 也列出 checkpoint SHA：`configs/RGBNT201/TriFusion-signal-preserving-v19-private-tail-rtx3090.yml:49-63`；但本审计没有远端 checkpoint byte possession。

**行动项**

- Q1 结束后保存并传回六个终态 checkpoint、六个 endpoint receipt、run_summary、log、source checkpoint SHA 复核结果和 full-gallery AP/rank 数组。
- 若需要 fresh-clone byte-for-byte 审计，统一记录 LF 字节或同时发布 CRLF/LF 归一化 hash 规则。

**Claim Impact**

- “T0/M0 证据文件存在且 M0 数字匹配 receipt”: **supported**。
- “从本地仓库可直接重算远端 checkpoint/data 的所有字节”: **needs qualifier**。

## D. Dead Code And Contract Coverage: WARN

**证据**

- V19 wrapper 用 `deepcopy` 给三专家复制 tail blocks，并用 `requires_grad_(train_private_tail)` 控制私有尾部训练：`modeling/trifusion/signal_preserving_v19.py:17-25`；storage disjoint 检查在 `modeling/trifusion/signal_preserving_v19.py:75-82`。
- optimizer group 覆盖所有 `requires_grad` 参数且无重复：`modeling/trifusion/signal_preserving_v19.py:85-92` 与 `tools/train_signal_preserving_v19.py:125-134`。
- M0 preflight 检查原 V8 五路输出 parity、batch receipt、pair counts 和 state unchanged：`tools/train_signal_preserving_v19.py:178-217`；三折两端 paired equality 在 `tools/train_signal_preserving_v19.py:356-368`。
- capacity/overfit 真实训练步检查 train/eval 模式、梯度、overflow、frozen state、tail mutation 和 optimizer groups：`tools/train_signal_preserving_v19.py:226-256` 与 `tools/train_signal_preserving_v19.py:394-403`。
- Q1 strict reload 与 full-gallery evaluate 路径已经实现：`tools/train_signal_preserving_v19.py:411-506`；但当前 M0 快照明确为 `folds=[]`：`evidence/trifusion_v19_m0_seed42_4b749cd.json:8127`，progress log 只显示 fold0 frozen endpoint 早期训练到 epoch 11：`evidence/trifusion_v19_m0_and_progress_log_20260905.txt:101-111`。
- T0 单测覆盖小模型的初始输出保持、strict reload、optimizer 分组、private-tail 更新范围和 Signal baseline 不变：`tests/test_trifusion_signal_preserving_v19.py:27-78`。

**细节**

M0 的实际执行路径覆盖了它声称覆盖的工程合同。未发现 M0 度量函数定义后完全不调用、或 receipt 声称执行但代码无路径的情况。WARN 来自范围边界：Q1 的 strict reload、终态 checkpoint 保存、full-gallery evaluate、bootstrap 与科学 gate 代码存在，但在本快照中尚未产生可审计输出；这些不能由 M0 pass 代替。

**行动项**

- 不要把 `m0.passed=true` 写成 Q1 reload/eval 已审计。
- Q1 终态审计需要逐 endpoint 验证 `strict_reload=true`、checkpoint SHA、state SHA、baseline_only 两端一致和 `total_gallery_records=3126 / total_eligible_queries=571`。

**Claim Impact**

- “V19 M0 工程合同执行并通过”: **supported**。
- “V19 Q1 终态 reload/eval 已通过”: **unsupported pending Q1**。

## E. Scope Assessment: WARN

**证据**

- frozen plan 明确承认两端可训练参数量不同，不能把收益单独归因于角色分工或排除容量效应：`refine-logs/v19/EXPERIMENT_PLAN.md:19-22`。
- M0 只定义为训练内工程检查：`refine-logs/v19/EXPERIMENT_PLAN.md:68-86`；Q1 完整科学门必须等三折×两端×20epoch 完成后判断：`refine-logs/v19/EXPERIMENT_PLAN.md:88-112`。
- 后续 D1/dev/official 是条件阶段，当前 runner 不执行：`refine-logs/v19/EXPERIMENT_PLAN.md:116-123`；当前最好仍是 V8 Phase-B，official/SOTA 未实现：`refine-logs/v19/EXPERIMENT_PLAN.md:125-127`。
- M0 receipt 记录 `epochs_per_endpoint=20` 和 `model_selection=none_final_epoch_only`，但同时 `folds=[]`：`evidence/trifusion_v19_m0_seed42_4b749cd.json:28-29` 与 `evidence/trifusion_v19_m0_seed42_4b749cd.json:8127`。
- Master handoff 明确说 M0 快照只覆盖 M0/preflight，不能充当 Q1 终态证据：`docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md:1969-1978`。

**细节**

实际完成范围为：三折×两端 preflight；fold0 两端各 8 个真实 source batch capacity；fold0 实验端固定 batch 100-step overfit。解析 JSON 得到 3 个 preflight folds、6 个 endpoints、48 个 batch receipts、48 个 output hashes、2 个 capacity checks、16 个 capacity steps、1 个 100-step overfit、0 个 Q1 folds。没有完成 retrieval aggregate、bootstrap、D1、30-dev、official、跨数据集训练或多种子。

**行动项**

- 当前只可公开为 “V19 T0/M0 工程门通过，Q1 运行中”。
- 禁止从该快照推导检索提升、统计显著性、跨数据集泛化、SOTA 或 65 mAP gate。

**Claim Impact**

- “Q1 正在运行，最终科学结论待完整终态”: **supported**。
- “V19 已通过主科学门或具备发布主结果”: **unsupported**。

## F. Evaluation Classification: PASS

**分类**

- **M0 工程检查**: `real_gt_train_internal_complete_path_oof`。依据为 `evidence/trifusion_v19_m0_seed42_4b749cd.json:26`，训练 labels/camera 来自 dataset records：`modeling/trifusion/aligned_data.py:91-94`。
- **Q1 计划评价**: 若完成，应归类为 `real_gt_train_internal_complete_path_oof`，使用 train_171 内部 heldout identities 的 identity/camera 真实标签和自定义 full-gallery ReID AP/Rank：`tools/train_signal_preserving_v18.py:133-151`、`tools/audit_v17_full_gallery.py:14-42`。
- **V12 target cache 历史材料**: `synthetic_proxy / teacher_proxy_train_identity_oof` 性质，但不是本次 M0 的评价 GT；V12 summary 仅作为 checkpoint/source provenance，`evidence/trifusion_v12_complete_path_oof_seed42.json:1930-1932` 记录其 PASS 和 target cache SHA。
- **D1/dev/official/cross-dataset**: 本快照中 **not executed**。

**行动项**

- 后续报告应分开标记 train-internal OOF、30-dev、official test 与跨数据集，不得混用。

**Claim Impact**

- “M0 是训练域 real-GT 工程检查”: **supported**。
- “M0 是官方 benchmark 评价”: **unsupported**。

## Verification Ledger

**我独立完成的检查**

- 逐项确认请求列出的 25 个 primary 文件均存在，并计算本地 SHA256。
- 读取并审计 V19 plan、tracker、config、V19 runner、V19 wrapper、V8 builder、V8 criterion path、aligned data、V17/V18/V5/V12 相关 helper、full-gallery AP/rank helper、T0/M0/transfer/V12 evidence 和 master §35/35.1。
- 独立计算 `evidence/trifusion_v19_m0_seed42_4b749cd.json` 的 SHA256 为 `89fd884a8c894de16639c26cfa162a2bbd005dc13c1e38ebe0fe6ce7adafe992`，与 transfer receipt 一致。
- 独立重算 overfit ratio：`(0.5803269147872925 - 0.57838292104621) / (0.6110473871231079 - 0.57838292104621) = 0.0595140216437626`，与 receipt 一致。
- 独立解析 preflight/capacity/overfit 数量：3 folds、6 endpoints、48 batch receipts、48 output hashes、2 capacity checks、16 capacity steps、100 overfit steps、Q1 folds 0。
- 独立比较 M0 source hash map：10 个 listed files 原始字节直接匹配；`criterion.py` 和 `rgbnt201_dev_v1.json` 的 LF-normalized hash 匹配，Windows CRLF 原始字节 hash 不匹配。

**我没有独立完成的检查**

- 没有读取远端 V12 Signal/expert checkpoint、CLIP weight、RGBNT201 图像、远端 Q1 checkpoint 或远端 CUDA 环境字节。
- 没有重新运行 T0 单测、M0 GPU 步、训练、评价、download、SSH 远端命令或任何模型计算。
- 没有重算 Q1 retrieval、bootstrap 或 D1/dev/official 指标；当前 M0 快照没有这些终态输出。

## Final Claim Impact

- **支持**: V19 T0 四项测试 receipt 存在；V19 M0 工程门通过；M0 的梯度覆盖、显存、overflow、frozen-state、tail mutation、paired preflight 和 overfit ratio 数字与 receipt 一致。
- **需要限定**: source checkpoint 和数据字节由远端 receipt/SHA 支撑，非本地 byte possession；两个本地文本文件存在 CRLF/LF hash caveat。
- **不支持**: V19 Q1 终态、检索提升、bootstrap 正下界、D1、30-dev、official test、跨数据集结果、SOTA、等可训练容量比较、排除容量效应或角色分工独立因果归因。
