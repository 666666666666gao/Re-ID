# TriFusion RGB–NIR–TIR ReID 完整交接（2026-09-01）

## 0. 一页结论

本工程是在 DeMo 代码基座上实现的 RGB–NIR–TIR 多模态目标重识别研究分支。最新终态是 V17 DTRED 完整跑完三折×两个endpoint×20 epoch后Q1失败；2026-09-05严格重载六个final checkpoint并补齐全部留出gallery后，fused相对matched weight0仍为-0.328915 mAP。V17没有D1/dev/official结果。当前最高的可部署结果仍是 V8 OOF-margin Router Phase-B：唯一冻结 dev fused=`58.4050 mAP / 59.3939 Rank-1`，比 exact Signal baseline 高 `0.3941 mAP / 1.9394 Rank-1` 并严格超过三个固定专家，但仍比65 mAP门低`6.5950`，因此不支持 official test 或 SOTA。

V6 的三个候选论文级主创新点已经落到核心代码、专项测试和完整 dev 运行中；性能主门仍然失败：

1. **Signal-preserving shared semantic expertization**：完整冻结 Signal，三专家共享其 patch/global 强语义场；`baseline_only` 保持原始 3072D 路径，专家训练不能改写 baseline 参数或输出。
2. **Stagewise bidirectional heterogeneous feature exchange**：CNN、Transformer、Mamba 都是三阶段完整专家；阶段 1/2 后进行双向 HFER，可靠性在阶段 1/2/3 分别刷新，使下一阶段能使用其他专家的互补信息。
3. **Complementarity-activated utility-routed residual bank**：不再把九路贡献压成一个向量，而是保留全部 `expert×modality` 残差；联合可靠性与身份效用只控制追加银行，并把银行按样本无自由倍率地校准到 baseline 能量；最终 `fused` 的 3072D 前缀严格等于 `baseline_only`。

当前最重要状态：

- 云端 RTX 3090 的正式 seed-42 主实验已完成 60 epoch 全 171 身份训练，并在固定终点完成唯一一次官方评估。
- 正式融合结果为 `59.1478 mAP / 63.2775 Rank-1`；CNN 略高，为 `59.1561 / 63.7560`。官方测试访问和评估计数均恰好为 1。
- 相对登记目标 `85.3 mAP / 87.9 Rank-1`，融合结果低 `26.1522 mAP / 24.6225 Rank-1`；`single_seed_target_exceeded=false`，不支持 SOTA 或融合增益主张。
- 后续 V3 task-anchor 与 V4 等能量残差银行均已在固定 141-fit/30-dev 上完整训练 60 epoch。V4 最佳 epoch27 fused 为 `43.4031/42.7879`，仍低于同 checkpoint 的 Mamba `44.0659/43.5152`，且距 65 mAP dev 门 `21.5969`；official access=0。
- V4 只保留了三模态 projected-CLS 的 1536D anchor，不等于 Signal 的完整 3072D 检索特征。Signal 还包含 1536D SIM 交互特征和 camera SIE；上游 `80.3/85.2` 尚未在本服务器复现，不能与 V4 held-out dev 数字直接相减。
- Signal baseline 已完整训练 50/50 epoch并严格重载最佳 checkpoint 确定性复评：`58.0109 mAP / 57.4545 Rank-1 / 69.9394 Rank-5 / 76.6061 Rank-10`；完整 3072D `direct+SIM`、camera SIE=true、official access=0。
- V5 核心、独立 runner/config 和专项测试已经完成。真实 preflight、B32/K4 8-step capacity、固定批 100-step overfit 和完整 60-epoch dev 均执行完成；最佳 epoch51 的 baseline/fused/CNN mAP 分别为 `58.0109/58.0168/58.0181`，fused 未超过 CNN，且距 65 mAP 门仍差 `6.9832`。
- 只读 checkpoint 诊断确认三分支参数实际更新，但 fused 追加残差范数只有 baseline 的 `2.747%`；融合距离与 baseline 距离相关系数为 `1.0`，Top-10 邻居重合率为 `99.9879%`。当前 V5 基本没有改变检索排序，因此不支持融合有效性主张。
- V6 真实 preflight、capacity、overfit 和唯一 seed42、60-epoch dev 已全部完成。最佳 epoch8 的 baseline/fused/CNN mAP 为 `58.0109/58.7321/59.1022`：fused 比 baseline 高 `0.7212`，但低于 CNN `0.3701`，距 65 mAP 门 `6.2679`；official access=0。
- V6 只读诊断确认残差/baseline 范数比已为 `1.0`，fused/baseline 距离相关降至 `0.96875`、Top-10 overlap 为 `95.3939%`，说明 V6 确实改变检索几何。当前首要失败原因是路由失配：最强 CNN 获得最低权重；次要问题是 epoch8 后的身份外泛化回落。
- V6 ground-truth Oracle 只读诊断覆盖 825 个 dev 查询：branch Oracle `63.6089 mAP`，比最强固定 CNN 高 `4.5067`；CNN/Transformer/Mamba 的 leave-one-out 边际 mAP 均为正。因此保留三专家，V7 直接修复共享几何、匹配 Token 残差、层级模态/专家路由、逐槽边际效用和有界样本 α。Oracle 不是部署结果，也仍未达到 65。
- V7 专项回归启动前 `32 passed`；exact Signal preflight、真实 B64/K8 双视图 capacity 和 100-step overfit 均 PASS。唯一正式 dev 已完成 60/60 epoch、2,520 optimizer steps、0 overflow，最佳 epoch1 的 baseline/fused/CNN/Transformer/Mamba mAP 为 `58.0109/58.3293/58.2773/58.3028/58.3476`。fused 只比 baseline 高 `0.3184`，低于 Mamba `0.0183`，距 65 仍差 `6.6707`；official access=0。
- V7 只读终态诊断显示联合 Router 熵 `0.99791`、模态熵 `0.99994`、alpha 几乎固定 `0.198947`、预测与目标 Top-slot 一致率 `14.0625%`；fused/baseline 距离相关 `0.999786`、Top-10 overlap `99.6364%`。但 residual-only Oracle 为 `62.7435 mAP`，比最强 residual 高 `3.6118`，三专家 leave-one-out 均为正。失败点是 learned routing 与 joint optimization，而不是不存在专家互补。
- optimizer0 的 V8 frozen-router 探针已否决“冻结现有专家、只重训 Router”路线：21 个跨摄像头合格 fit 身份、571 个 query 的最佳 residual 专家 100% 为 CNN；身份隔离教师在 dev 仅达到 CNN 多数类先验 `55.27%`，V7 Router 更低，为 `27.39%`。恢复 residual 与 baseline 等能量后，均匀/教师融合达到 `59.6188 mAP / 59.1515 Rank-1`，仍比 65 低 `5.3812`。下一版本必须增强专家表征与分工。
- V8 Phase-A 已完成该表征修正：exact preflight、真实 B64/K8 capacity、100-step overfit 全部 PASS；20 epoch/840 step 训练期间不评估 dev，最终 epoch 只评估一次。固定 fused 为 `58.0972/56.8485`，不能称为部署增益；branch GT Oracle 为 `64.7850/65.9394`，比最强固定输出高 `6.7741 mAP`，CNN/Transformer/Mamba 均有独有胜例与正 leave-one-out 边际。residual-only Oracle 为 `63.4813/66.9091`，比最强 residual 高 `9.6153 mAP`。Oracle 使用真实标签，只是诊断上限。
- 独立 result-to-claim 为 `partial/medium`；V8 专属完整性审计为 `WARN`，GT、指标归一化、活代码与 dev 泄漏检查均 PASS，警告只来自大 checkpoint/history/run identity 仍按 SHA/path 留在远端。下一步仅授权冻结专家、fit-only 的层级 Router 可行性阶段；Router 未证明可部署增益前不得启用 HFER，也不做 official test、消融或多种子。
- V8 Phase-B 已完成：连续 OOF margin 的 expert/modality winner 均不塌缩，但 learned-vs-fixed OOF margin 只高 `0.000314`；三种单模态模糊均使自身质量下降，missing modality 权重严格为 0。冻结 dev fused=`58.4050/59.3939`，超过 baseline 和三个固定专家，但主门仍失败。独立 result-to-claim=`partial/medium`、完整性审计=`WARN`（仅 remote-only 大 artifact 封装警告）。Phase-B 已封存，不启动 HFER、消融、多种子、official test 或 Router 超参数扫描。
- 原正式启动在官方指标写出后的路由校准审计因缺失导入失败；`repair-0002` 仅重算训练集路由审计，`optimizer_steps=0`、`training_reexecuted=false`、`official_test_reexecuted=false`，公开 verifier 返回 PASS。
- 用户最新指令：只做 seed 42；现在优先完成远端 Signal baseline 保底；主实验达到目标以后才考虑消融；所有训练、评估、数据和环境只在云端 GPU，Windows/WSL 仅作传输和文档存档。

## 1. 权威位置

### 1.1 云端工程

```text
Repository : /root/autodl-tmp/trifusion-v2/TriFusion-ReID
Branch     : main
Conda env  : /root/miniconda3/envs/tri_reid
Signal env : /root/miniconda3/envs/signal
Dataset    : /root/autodl-tmp/trifusion-v2/data/RGBNT201
Pretrained : /root/autodl-tmp/trifusion-v2/pretrained/ViT-B-16.pt
Artifacts  : /root/autodl-tmp/trifusion-v2/artifacts
GPU        : NVIDIA GeForce RTX 3090, 24 GiB
```

目标公开仓库：`https://github.com/666666666666gao/Re-ID`

仓库只保存代码、配置、协议、测试、轻量 evidence 和文档。以下内容不得提交：

- RGBNT201 数据集；
- CLIP 或其他预训练权重；
- 训练检查点、恢复状态和完整实验 artifacts；
- SSH 密钥、云主机口令、访问令牌或任何凭据。

### 1.2 本地文档

本文件的本地权威副本：

```text
C:\Users\gb\Desktop\document\TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md
```

原始调研综述仅保留在本地，未提交到公开仓库：

```text
E:\调研综述趋势_2026-08-31_01-59.md
```

## 2. 数据集与固定协议

数据集：RGBNT201，每个样本由 RGB、NI（NIR）和 TI（TIR）三模态配对组成。

已核验的数据统计：

| Split | 身份数 | 三模态 triplets | 图像数 | 摄像头 |
|---|---:|---:|---:|---|
| `train_141` | 141 | 3280 | 9840 | 1, 2, 3, 4 |
| `train_171` | 171 | 3951 | 11853 | 1, 2, 3, 4 |
| `test` | 30 | 836 | 2508 | 1, 2 |

所有 JPEG、身份数、摄像头集合、模态配对和 triplet 数量均通过版本化审计。数据审计文件是：

```text
evidence/rgbnt201_audit_20260831.json
```

协议边界：

- 开发阶段只使用 `train_171` 内部的 141-fit / 30-dev 身份隔离划分。
- CIRC 三折生成器的目标身份与生成器训练身份重叠为 0。
- 正式训练使用 `train_171` 全部 171 个身份。
- 正式模型在 epoch 60 固定，不允许使用官方 test 选 epoch、调阈值或选择模型。
- 官方 test 只允许在固定终点后评估一次。
- 当前不开启 reranking。

## 3. 可复现环境

已验证的云端运行栈：

| 项目 | 版本/配置 |
|---|---|
| Python | 3.10.14 |
| PyTorch | 2.5.1+cu121 |
| CUDA build | 12.1 |
| NumPy | 1.24.4 |
| scikit-learn | 1.3.2 |
| PyYAML | 6.0.2 |
| GPU | RTX 3090 24 GiB |
| Train batch | 32 |
| Instances per ID | 4 |
| Eval batch | 64 |
| AMP init scale | 512 |
| Gradient checkpointing | 开启 |

进入环境：

```bash
cd /root/autodl-tmp/trifusion-v2/TriFusion-ReID
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tri_reid
export PYTHONPATH="$PWD"
```

完整重建说明见 `docs/ENVIRONMENT_REPRODUCTION.md`。该文件包含早期 WSL2 路径；当前正式执行位置以后续章节的云端路径为准。

Signal baseline 使用独立环境：Python 3.10.13、PyTorch 2.1.1+cu118、
torchvision 0.16.1+cu118、CUDA 11.8。完整训练依赖锁见
`environment/signal_requirements-lock.txt`，构建和三项已证实的可视化/构建工具排除说明见
`environment/SIGNAL_BASELINE.md`。远端环境回执位于
`/root/autodl-tmp/trifusion-v2/artifacts/signal_env_cd1b0a6/`。

## 4. 已完成 V1 网络结构（历史）

本节记录已经跑完正式实验的 V1 HFER/CIRC/URGC 结构；它不是当前 V5 候选。V5 的最新实现边界见第 12.6 节。

```text
RGB / NIR / TIR
       │
       ▼
共享 CLIP ViT-B/16 语义主干（只执行一套强预训练编码）
       │
       ├── CNN expert：二维局部纹理、高频边缘与细粒度结构
       ├── Transformer expert：全局身份语义与跨区域关系
       └── Mamba expert：线性复杂度的长程空间序列传播
                    │
                    ▼
        HFER 分阶段双向异构特征交换
                    │
                    ▼
        CIRC 监督的统一可靠性后验
                    │
                    ▼
          URGC 可靠性感知身份融合
                    │
                    ▼
       fused + cnn + transformer + mamba embeddings
```

主配置：

```text
configs/RGBNT201/TriFusion-circ-urgc-postfreeze-final-shared-semantic-rtx3090.yml
```

关键尺寸：共享语义宽度 768；CNN/Mamba 宽度 256；adapter 宽度 192；relay rank 64；最终 embedding 512；参数预算上限 1.2 亿。

### 4.1 HFER

HFER 不是把三路 logits 在末端求平均。它接受三套完整专家状态，构造异构专家共识，并以低秩残差形式把其他专家的信息双向注入当前专家。这样：

- CNN 能借用 Transformer 的全局身份线索和 Mamba 的长程上下文；
- Transformer 能恢复 CNN 保留的局部纹理；
- Mamba 能在序列传播中接收另外两种归纳偏置；
- 每个专家仍有独立 embedding 和辅助身份监督，能够单独评估。

### 4.2 CIRC

CIRC 的监督不是由同一训练身份上的单模型置信度自举。已完成的 postfreeze-final 目标构建使用三个身份不重叠折：

- folds 0、1、2 全部完成；
- generator/target identity overlap 为 0；
- 共覆盖 clean、exposure、blur、modality missing、NIR noise、occlusion、thermal noise 七类条件；
- 所有条件的经验浓度覆盖率均不低于 0.90；
- 构建和评分期间官方 test 访问计数为 0。

当前 CIRC 有一条必须保留的负证据：query/gallery 交换对称性审计失败，`sign agreement = 0.671875 < 0.70`，虽然 `Spearman = 0.743356 > 0.50`。因此正式运行只授权使用 **calibrated directional training input**，不得宣称 query/gallery 对称性成立。

### 4.3 URGC

URGC 使用共同尺度的可靠性后验协调中继和最终融合，避免“交换模块认为 A 可靠、融合模块却认为 B 可靠”的控制冲突。当前正式配置中：

```text
EVIDENCE_WEIGHT = 0.1
PEER_LOGITS = 0.0
PEER_ROLE = 0.0
PRIVATE_DIVERSITY = 0.0
```

RDPT 仍是辅助机制，不属于本次主实验启用的核心贡献。

## 5. 结果账本

### 5.1 上游公开参照

历史阶段用户曾要求不复现 baseline；该约束已被 2026-09-01 19:00 的“先做 baseline 保底”指令覆盖。下列数字在新的本地结果产生前仍只作为冻结的公开参照，不是本工程复现结果：

| 方法/角色 | mAP | Rank-1 | 边界 |
|---|---:|---:|---|
| PEFT-BoA released selected endpoint | 82.7 | 86.1 | 上游日志，使用官方 test 选择 epoch 80 |
| PEFT-BoA fixed epoch 120 | 82.2 | 85.8 | 上游固定终点证据 |
| 本项目登记目标 | 85.3 | 87.9 | 需要同协议正式证据后才能比较 |

不得把选过官方 test epoch 的数字与本项目固定 epoch 60 的数字描述为完全公平复现。

### 5.2 已完成开发结果

`train_171` 内部 141-fit / 30-dev 身份隔离；seed 42；60 epoch；最佳为 epoch 36：

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| fused | 47.4001 | 45.4545 | 60.9697 | 70.3030 |
| CNN | 47.4396 | 46.1818 | 59.8788 | 70.5455 |
| Transformer | 47.6153 | 45.3333 | 61.0909 | 70.6667 |
| Mamba | 46.8994 | 44.8485 | 60.9697 | 70.3030 |

证据：

```text
/root/autodl-tmp/trifusion-v2/artifacts/
  trifusion_shared_semantic_circ_urgc_v3_amp_safe_dev_seed42/run_summary.json
```

这些数字只能证明三分支已基本平衡并且训练链可运行，不能与公开官方 test SOTA 直接相减。

### 5.3 正式主实验

状态：**COMPLETE，经 audit-only repair 验证；官方 test access/evaluation = 1/1**。

`postfreeze-final`；seed 42；epoch 60 固定终点；train 171 身份/3951 记录；query/gallery 各 836；无 reranking：

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| fused | 59.1478 | 63.2775 | 77.2727 | 83.6124 |
| CNN | 59.1561 | 63.7560 | 78.3493 | 83.2536 |
| Transformer | 59.1219 | 62.6794 | 76.9139 | 83.6124 |
| Mamba | 58.8748 | 62.4402 | 77.2727 | 83.0144 |

结果判定：

- fused 相对登记目标低 `26.1522 mAP / 24.6225 Rank-1`；未超过目标。
- CNN 比 fused 高 `0.0083 mAP / 0.4785 Rank-1`；当前结果不支持“融合优于各分支”。
- 这是单数据集、单 seed、无 baseline 复现的正式结果，不能宣称 SOTA、统计显著性或广泛稳健性。
- 用户规定“先超过目标再做消融”；本结果未过门槛，因此不启动消融。

正式输出路径：

```text
/root/autodl-tmp/trifusion-v2/artifacts/
  trifusion_shared_semantic_circ_urgc_directional_final_seed42
```

只增不改的启动账本：

```text
/root/autodl-tmp/trifusion-v2/artifacts/
  trifusion_shared_semantic_circ_urgc_directional_final_seed42_launch_ledger
```

权威结果链：

```text
official_test_metrics.json
official_test_access_guard.json
run_summary.json
fixed_final_receipt.json
launch_ledger/launch-0001/failure_receipt.json
launch_ledger/repair-0002/completion_receipt.json
```

原 `launch-0001` 在唯一官方评估后的路由审计因缺失 `build_rgbnt201_record_eval_loader` 导入而失败，失败回执永久保留。`repair-0001` 完成训练集路由审计后因没有复用定向授权上下文而在汇总门失败，已事务回滚。`repair-0002` 复用原定向授权，只运行训练集路由校准审计并通过；未重训、未执行优化器 step、未重评官方 test。

### 5.4 Signal-preserving V5 held-out-dev 终局

V5 只使用固定 141-fit/30-dev，未访问 official test。seed42、B32/K4、60/60 epoch 共执行 5498 个 optimizer steps，0 AMP overflow；按 fused dev mAP 选择 epoch51 后严格重载，同 checkpoint 五路结果为：

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| fused | 58.0168 | 57.4545 | 69.9394 | 76.6061 |
| CNN | 58.0181 | 57.4545 | 69.9394 | 76.6061 |
| Transformer | 58.0137 | 57.4545 | 69.9394 | 76.6061 |
| Mamba | 58.0135 | 57.4545 | 69.9394 | 76.7273 |

主门为 FAIL：fused 比 baseline 仅高 `0.00587 mAP`，比 CNN 低 `0.00130 mAP`，并比 65 mAP 门低 `6.98324`。Signal state SHA 在训练前、训练后、严格重载后均为 `97234c...5a92`；official access=0。完整结果见 `results/TRIFUSION_RGBNT201_V5_DEV_SEED42_2026-09-01.md`。

只读诊断处理全部 825 个 dev 样本，不训练也不创建 optimizer。fused 残差/baseline 范数比为 `0.027471`，距离 Pearson 相关为 `1.0`，平均绝对距离变化 `0.0002017`，Top-10 邻居重合率 `0.9998788`；路由归一化熵为 `0.9600`。CNN/Transformer/Mamba 残差两两余弦均接近 0，说明专家差异存在，但当前缩放和路由没有让差异实质改变检索几何。

## 6. 正式运行与修复命令（历史记录，禁止重跑）

本实验的官方测试已消费一次。以下启动命令只用于法证复现记录，**不得再次执行同一实验身份**。

### 6.1 启动前检查

必须保证 RTX 3090 至少有 22000 MiB 空闲显存：

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader
```

运行只读预检：

```bash
cd /root/autodl-tmp/trifusion-v2/TriFusion-ReID
PYTHONPATH=. /root/miniconda3/envs/tri_reid/bin/python \
  tools/run_trifusion_directional_final.py \
  --authorization protocols/circ_directional_final_authorization_v1.json \
  --config configs/RGBNT201/TriFusion-circ-urgc-postfreeze-final-shared-semantic-rtx3090.yml \
  --output-dir /root/autodl-tmp/trifusion-v2/artifacts/trifusion_shared_semantic_circ_urgc_directional_final_seed42 \
  --ledger-dir /root/autodl-tmp/trifusion-v2/artifacts/trifusion_shared_semantic_circ_urgc_directional_final_seed42_launch_ledger \
  --preflight-only
```

必须看到：`status=READY`、`launch_allowed=true`、`blockers=[]`、`official_test_access_count=0`、`model_constructed=false`、`training_started=false`。

### 6.2 正式启动

下列命令是 2026-09-01 已执行的历史命令，不是待办操作：

```bash
cd /root/autodl-tmp/trifusion-v2/TriFusion-ReID
screen -dmS circ_directional_final_seed42 bash -lc '
  cd /root/autodl-tmp/trifusion-v2/TriFusion-ReID &&
  PATH=/root/miniconda3/envs/tri_reid/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONPATH=. python tools/run_trifusion_directional_final.py \
    --authorization protocols/circ_directional_final_authorization_v1.json \
    --config configs/RGBNT201/TriFusion-circ-urgc-postfreeze-final-shared-semantic-rtx3090.yml \
    --output-dir /root/autodl-tmp/trifusion-v2/artifacts/trifusion_shared_semantic_circ_urgc_directional_final_seed42 \
    --ledger-dir /root/autodl-tmp/trifusion-v2/artifacts/trifusion_shared_semantic_circ_urgc_directional_final_seed42_launch_ledger \
  >> /root/autodl-tmp/trifusion-v2/artifacts/trifusion_directional_final_seed42.log 2>&1
'
```

不要启动第二个同身份进程。当前 `official_test_access_guard.json` 和 metrics receipt 已完整存在，任何再次正式评估都会违反一次性协议。

### 6.3 监控

```bash
screen -ls
nvidia-smi
tail -n 80 /root/autodl-tmp/trifusion-v2/artifacts/trifusion_directional_final_seed42.log
python -m json.tool \
  /root/autodl-tmp/trifusion-v2/artifacts/trifusion_shared_semantic_circ_urgc_directional_final_seed42/.resume/latest.json
```

### 6.4 修复完成验证

由于原 `launch-0001` 保留失败回执，最终可用链由独立 `repair-0002` 完成回执验证。验证覆盖：

1. `run_summary.json`；
2. `run_identity.json`；
3. `.resume/latest.json`；
4. 当前完整恢复 generation；
5. `fixed_final_receipt.json`；
6. `fixed_final_model.pth`；
7. `official_test_metrics.json`；
8. `official_test_access_guard.json`；
9. `final_worker_result.json`；
10. `router_calibration_receipt.json`。

独立重验：

```bash
cd /root/autodl-tmp/trifusion-v2/TriFusion-ReID
PYTHONPATH=. /root/miniconda3/envs/tri_reid/bin/python \
  tools/repair_trifusion_directional_final_completion.py \
  --verify /root/autodl-tmp/trifusion-v2/artifacts/\
trifusion_shared_semantic_circ_urgc_directional_final_seed42_launch_ledger/\
repair-0002
```

该 verifier 已返回 `status=PASS`，并确认 `official_test_access_count=1`、`official_test_evaluation_count=1`、`official_test_reexecuted=false`、`optimizer_steps=0`。

## 7. 测试状态

正式启动器与修复器均采用 TDD。定向启动器专项原为 `27 passed`；新增最终修复器后，联合专项为 `42 passed`。排除三个用户明确不要运行的外部 baseline 仓库测试后，内部全量回归为：

```text
133 passed, 4 skipped
```

V5 新增诊断工具的专项回归为 `1 passed`；V5 core+runner readiness 联合专项此前为 `10 passed, 3 warnings`，warnings 仅来自 timm 弃用提示。提交前的当前 V5 组合回归命令和终态记录见第 12.6 节。

全量命令：

```bash
cd /root/autodl-tmp/trifusion-v2/TriFusion-ReID
PATH=/root/miniconda3/envs/tri_reid/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
PYTHONPATH=. python -m pytest -q tests \
  --ignore=tests/test_audit_peft_boa_source.py \
  --ignore=tests/test_peft_boa_resumable_runner.py \
  --ignore=tests/test_mfrnet_checkpoint_runner.py
```

## 8. 当前证据与重要哈希

冻结训练器：

```text
tools/run_trifusion_experiment.py
SHA256 50540f112d99b55e761be91eaa36a273444c0318c9929929cb8a62d8cb25897c
```

正式配置：

```text
SHA256 24fc81f984d1f4c6094a22edb0a4969d249467c0a5c75af052e952be1d3478ae
```

postfreeze-final CIRC targets：

```text
SHA256 29b11a562422648c21870ad3a49b06a42101662efac660602ad576ee18cdd7ab
```

正式 epoch-60 checkpoint：

```text
SHA256 ca4a7963e0c5630bd760ee68c973a5f8511a0597e38b7221e1f73526cd09edab
```

官方 metrics 与 access guard：

```text
metrics SHA256 a75d51aa5e17bc11c8c27246fc005fac5c764b4813b147c9706ed2ca5b0eeb85
guard   SHA256 9a162b865b09f6ae13c7cc4513938df3e533703a1305091e68f92b20d65fc405
```

最终 audit-only repair 完成回执：

```text
..._launch_ledger/repair-0002/completion_receipt.json
```

正式定向授权只允许保留失败的对称性结论并使用校准后的方向性训练输入。授权文件：

```text
protocols/circ_directional_final_authorization_v1.json
```

启动器哈希必须以授权文件当前登记值为准，任何字节改动都要先重新测试、重新审查并更新授权，不能临时绕过。

V5 held-out-dev 结果与只读诊断：

```text
run summary SHA256 58fb5ebb30f4a72b02d2377e52d55e20fa070f7a3b2f831d6a52987d32f8c4ab
diagnostic  SHA256 4aeafcfa29219ba51fbb81accf9d8d14528e096fe82026baa922b222e9473555
best ckpt  SHA256 43f4806437545520d91b2fe70349b6036dbb3949e6d6351d79a24c3aa7f539c0
```

## 9. 已知限制与下一步决策

### 9.1 必须保留的限制

- 当前只做一个 seed 42；不能据此给出多种子均值、方差或统计显著性。
- 已完成 Signal 的同 held-out-dev 协议 baseline floor；Signal 上游官方 test `80.3/85.2` 仍未本地复现，必须写成 upstream-reported。
- CIRC query/gallery symmetry 审计失败；禁止对称性主张。
- 正式官方 test 已恰好评估一次；不得再次访问本次 test 做选模、调参或重评。
- 在主结果超过冻结目标前，禁止启动消融实验。
- 本次 fused 未优于 CNN，不能把 HFER/CIRC/URGC 写成已获检索增益的实证结论。
- V5 fused 同样未优于 CNN，且几乎不改变 baseline 排序；不能把 V5 三个候选创新点写成已获性能验证的论文贡献。
- 路由校准是训练目标上的描述性证据；缺少身份留出校准，不能主张因果或身份外泛化校准。
- 路由平均概率在条件、专家和模态之间几乎固定为 `0.24997`；`modality_missing` 的训练目标校准最差（Brier `0.22338`、ECE `0.07178`）。这不是官方 test 的分场景 ReID mAP。

### 9.2 结构风险

只读代码、checkpoint 和日志诊断确认以下主结构风险：

1. 当前融合将九个“专家×模态”贡献直接加权求和为一个 512 维向量；DeMo 参照推理头则保留三模态原始特征和七个 MoE 特征的拼接。路由近似常数时，当前融合退化为信息损失很大的近均匀平均。
2. 三个融合投影在最终 checkpoint 中两两余弦相似度均高于 `0.99992`；CNN/Transformer/Mamba 官方 mAP 最大差只有 `0.2813`。三支接收完全相同的共享 CLIP token，且 `PEER_LOGITS`、`PEER_ROLE`、`PRIVATE_DIVERSITY` 均为 0，缺少防止专家同质化的训练约束。
3. baseline 使用 CLIP 投影后的 CLS 全局特征并保留局部 token；当前共享 tokenizer 将 CLS 广播到 patch 后只输出 patch 场，专家再做均值池化。正式评估使用 BN neck 后的 512 维特征，而 DeMo 配置为 neck 前特征。因此“加载了同一 CLIP”不等于保留了 baseline 的检索表征。
4. epoch 60 的 fused ID/triplet loss 已为 `0.01823/0.00562`，但官方 mAP 仅 `59.1478`，说明主要是身份外泛化失败，不是训练未完成。无 label smoothing、无 camera SIE、前 7 epoch 仅训练路由是次级泛化差异。
5. HFER 的两次交换都使用 stage-1 后验；stage 3 后会为最终融合刷新质量，但第二次交换仍使用旧后验。新主版本应在第二次交换前重新估计质量。
6. 当前 Mamba 专家负责各模态内的空间序列扫描；跨模态传播主要由通用 HFER 完成。若论文要声称“Mamba 特有的跨模态状态传播”，必须增加相应机制和消融，否则应使用更窄的表述。

### 9.3 正式结果后的唯一决策树

```text
V6 held-out-dev 完成且 official access=0
  ├─ fused > 85.3 mAP 且 Rank-1 > 87.9
  │    └─ 才允许设计消融；仍只能称单种子目标超越，不能直接宣称统计 SOTA
  └─ 未通过 dev 门（本次路径：58.7321 mAP，低于 CNN 59.1022）
       └─ 不做消融、多种子或 official test；只做一次 marginal-gain routing main-only 架构修正
```

## 10. 文档索引

本文件是当前统一入口；其他文件保留原始审计细节：

| 文档 | 用途 |
|---|---|
| `docs/SOURCE_INTAKE_2026-08-31.md` | 原始综述需求映射 |
| `docs/RESEARCH_AUDIT_2026-08-31.md` | RGBNT201 协议、公开目标与贡献边界 |
| `docs/NOVELTY_CHECK_2026-08-31.md` | 近年工作碰撞与新颖性查核 |
| `docs/METHOD_SPEC_V1.md` | HFER/CIRC/URGC 方法规格 |
| `docs/IMPLEMENTATION_BLUEPRINT_V1.md` | 模块接口、张量形状和实现蓝图 |
| `docs/TDD_SEAMS.md` | 测试接缝和用户同意边界 |
| `docs/ENVIRONMENT_REPRODUCTION.md` | 环境与历史复现说明 |
| `docs/BASELINE_SELECTION_AND_LICENSE_AUDIT_2026-08-31.md` | baseline 选择、许可证和复现边界 |
| `docs/BASELINE_PROTOCOL_AUDIT_2026-08-31.md` | checkpoint selection 公平性审计 |
| `results/TRIFUSION_RGBNT201_FINAL_SEED42_2026-09-01.md` | 正式原始指标、差距和负结果分析 |
| `results/TRIFUSION_RGBNT201_V5_DEV_SEED42_2026-09-01.md` | V5 五路 dev 终局、门禁与只读诊断 |
| `results/TRIFUSION_RGBNT201_V6_DEV_SEED42_2026-09-01.md` | V6 五路 dev 终局、检索几何、路由失配与 claim gate |
| `EXPERIMENT_AUDIT.md` / `.json` | 独立实验完整性审计 |
| `findings.md` | result-to-claim 否定结论与后续边界 |
| `evidence/README.md` | 版本化 evidence 说明 |

## 11. 交接检查清单

- [x] RGBNT201 完整性与开发/正式协议已核验。
- [x] 云端 Conda/CUDA/RTX3090 环境可运行。
- [x] 共享 CLIP + CNN/Transformer/Mamba 三专家已实现。
- [x] HFER、CIRC、URGC 已实现并有测试。
- [x] CIRC postfreeze 三折生成、干预评分和校准已完成。
- [x] 正式定向授权、失败边界和唯一官方 test gate 已固定。
- [x] 启动器/修复器完成链已通过 `133 passed, 4 skipped`。
- [x] 正式 60 epoch 全 171 身份训练。
- [x] 唯一固定终点官方评估，access/evaluation = 1/1。
- [x] `repair-0002` 完成收据独立重验 PASS。
- [x] 最终 fused/CNN/Transformer/Mamba 指标已回填。
- [x] 结果未超过冻结目标，已锁定“不启动消融”。
- [x] V3 与 V4 各完成一次 seed42、60-epoch、held-out dev 主实验，均未晋级且 official access=0。
- [x] 已确认 V4 的 1536D anchor 不是 Signal 完整 3072D baseline。
- [x] 在远端建立完整 Signal baseline-only 路径、独立环境和可复现训练回执；同协议 50-epoch dev 已完成并确定性复评为 `58.0109/57.4545/69.9394/76.6061`。
- [x] V5 核心已建立同 checkpoint `baseline_only/fused/cnn/transformer/mamba` 五输出、冻结 Signal 路径和非破坏式残差银行；专项测试 `4 passed`。
- [x] V5 独立 runner/config 已完成；真实 baseline parity、8-step capacity 和 100-step overfit 门均 PASS，official access=0。
- [x] 唯一一次 V5 seed42、60-epoch held-out-dev 主训练、严格重载和五路评估完成；主门失败，official access=0。
- [x] V5 只读协同诊断完成：确认三专家有更新但最终检索几何几乎等同 baseline。
- [x] V6 baseline-preserving main-only 架构修正、工程门、60-epoch dev、严格重载和只读诊断全部完成；fused `58.7321` 低于 CNN `59.1022`，official0。
- [ ] 只实现一个基于 V6 证据的 marginal-gain routing V7 main-only 修正；通过同样 dev 门前继续禁止消融、多种子和 official test。

## 12. V3/V4 主方法恢复终态

### 12.1 V3 task-anchor

V3 将三模态 direct CLIP projected-CLS 作为 1536D anchor，并追加质量路由的三专家残差。完整 60-epoch held-out dev 最佳为 epoch14：

| 输出 | mAP | Rank-1 |
|---|---:|---:|
| fused | 42.8978 | 43.8788 |
| CNN | 42.8402 | 44.0000 |
| Transformer | 43.0168 | 44.0000 |
| Mamba | 42.9259 | 43.8788 |

冻结诊断表明 residual-only 有身份信息，但 residual/anchor norm ratio 只有约 `0.216`，路由归一化熵约 `0.9998`，残差在最终距离中的能量过弱。V3 未通过 65 mAP dev 门，official access=0。

### 12.2 V4 等能量非破坏残差银行

V4 commit：

```text
3fbedbb98940c6c9765c07af01f52e40f809ff95
```

V4 将 CNN、Transformer、Mamba 的三模态残差分别保留为 4608D bank，并将整个 bank 的样本级 L2 能量校准为等于 1536D anchor；最终 fused 为 6144D。工程门全部通过：95,197,266 参数，B32/K4 8-step capacity 无 overflow，366/366 梯度覆盖；固定批 100-step loss ratio `0.06677`。

完整 dev 运行身份：

```text
/root/autodl-tmp/trifusion-v2/artifacts/
trifusion_task_anchor_v4_core_dev_seed42_3fbedbb
```

终态：60/60 epoch，`run_summary=PASS`，phase=`complete`，60 次 dev 评估，无 fatal/nonfinite，official test access=0。最佳为 epoch27：

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| fused | 43.4031 | 42.7879 | 58.5455 | 65.5758 |
| CNN | 40.9147 | 39.5152 | 55.5152 | 64.7273 |
| Transformer | 41.6819 | 40.1212 | 56.7273 | 65.4545 |
| Mamba | **44.0659** | **43.5152** | **58.7879** | **66.0606** |

V4 fused 比 V3 fused 提高约 `0.5053 mAP`，但仍比 Mamba 低 `0.6628 mAP / 0.7273 Rank-1`，距 65 mAP dev 门低 `21.5969`。epoch60 fused 回落到 `40.1199/40.0000`，Mamba 为 `41.0375/42.7879`。这是一项完成后的负结果和过拟合信号，不是未训练完。

注意：V4 terminal receipt 没有 V4 anchor-only 指标；V3 的 anchor `42.4787` 不能挪用为 V4 anchor。独立 reviewer 的初稿曾发生这一混淆，已在 trace 中纠正；最终 verdict 不声称 V4 fused 优于 V4 anchor。

关键哈希：

| 工件 | SHA-256 |
|---|---|
| best checkpoint | `47fea7f42a5673e42deb1d67540cca6338af62b028be4d69daedfe309de1e852` |
| run summary | `cd7992d0c0a7a2deb3225f7f3a78b0185cdea1264ea39b39303fdfabc1d4a9af` |
| dev worker result | `dceca4839ec5bafdacf03ebfff53c63dffe508592afd585e13e5719c08856ec9` |
| best dev receipt | `0f0c1781c4352feb4a6339489f14e0e80d8219577a14661207a627c056c5f013` |
| final resume generation | `ec7c0c0554d7fea5b58406524a3a284a3ab59435eeb1cd012dfd47fb6b30151b` |

轻量终态证据：`evidence/trifusion_task_anchor_v4_dev_terminal_seed42.json`。

### 12.3 为什么现在必须先保住 baseline

官方 Signal commit `cd1b0a6` 的推理路径已逐行核对：

```text
ori = concat(RGB_global, NI_global, TI_global)        # 1536D
sim = SIM(rgb_patch, ni_patch, ti_patch, globals)    # 1536D
Signal retrieval = concat(ori, sim)                  # 3072D
```

其 ViT 还在 CLS token 上加入 camera SIE。V4 只有 `ori` 语义的一部分，没有 SIM 和 SIE，因此不是 Signal baseline。上游发布 `80.3 mAP / 85.2 Rank-1` 来自官方 test 路径；服务器独立 `signal` conda 环境的 held-out-dev baseline 已完成，但因数据划分不同，`80.3/85.2` 仍必须标为 upstream-only，不能与本地 dev 数值直接相减。

真正的“baseline 保底”不是简单拼接更多维度，而是以下可验证合同：

1. 同一模型/同一 checkpoint 同时输出 `baseline_only` 与 `fused`；
2. baseline 使用完整 Signal 3072D 路径，且能独立检索；
3. 先训练/验证 baseline，再冻结 baseline 路径训练专家增量，专家梯度不得破坏 baseline；
4. 同一 141/30 dev 协议上 fused 只有不低于 baseline 才能晋级；否则拒绝 fused，论文不得主张融合增益；当前不增加运行时 fallback 逻辑；
5. 只有 dev 主门通过后才全 171 固定训练并进行一次官方评估；仍不做多种子和消融。

### 12.4 Signal baseline floor 终局

唯一 seed42 baseline 使用 Signal commit `cd1b0a6`、B64/K8、50 epoch、141-fit/30-dev 和完整 3072D direct+SIM 检索特征；camera SIE 开启。运行目录：

```text
/root/autodl-tmp/trifusion-v2/artifacts/signal_baseline_dev_seed42_f7d4b30
```

训练完整结束 50/50 epoch，终局 `run_summary.json` 为 PASS。最佳 checkpoint 在 epoch30 更新，runner 训练结束后以 `strict=True` 重新加载并确定性复评：

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| Signal baseline-only | **58.0109** | **57.4545** | **69.9394** | **76.6061** |

完整性边界：

- fit/dev triplets=`3126/825`，dev query/gallery=`825/825`；
- 参数量 `91,077,121`；峰值 allocated/reserved=`11,915.32/13,620 MiB`；
- elapsed=`1678.65s`；日志无 Traceback、OOM、nonfinite、NaN、ERROR 或 Exception；
- retrieval width=`3072`，feature=`concat(direct_3x512,SIM_3x512)`，camera SIE=true；
- official-test access count=`0`；
- `Signalbest.pth` SHA256=`1f5c200cd43fcbc00b8a0494329519eed3e6f062d9a29d43a0ecdd97ff4966c3`；
- 固定 `Signal_50.pth` SHA256=`9f3a74a75fd5e2d1fa2dff0db011dfcd0360bdd76d75ba7b4a140965dcf15b5c`；
- `run_summary.json` SHA256=`ede1d7764a2a5e3bb8c9e63475e3e8fcc942540783910e952f008fe85f9f98b0`；
- `run_identity.json` SHA256=`da83cc7ba40ef63741ca3c147fd4973237373323123749a301193d83285c4093`。

该 baseline 比 V4 fused 高 `14.6078 mAP`，也比 V4 最强 Mamba 高 `13.9450 mAP`，证明先前主方法确实破坏了强基线能力。它仍比冻结的 65 mAP dev 晋级门低 `6.9891`，因此 V5 不能只“回到 baseline”，还必须在同 checkpoint 下使 fused 超过 `58.0109`、超过所有专家并达到至少 65 mAP。该 dev 结果不是 Signal 上游官方 test `80.3/85.2` 的本地复现。

### 12.5 当前 claim gate

独立 result-to-claim 纠正后结论：`claim_supported=no`。高置信度支持“V4 是稳定、完整但失败的 dev 结果”；中等置信度支持“缺少完整 baseline floor 是下一项结构性优先问题”。V4-specific independent integrity audit 尚未完成，因此 V4 完整性标签为 provisional，不能复用只审计旧 V1 的 `EXPERIMENT_AUDIT.json`。

### 12.6 Signal-preserving V5 最新代码状态

当前版本已新增：

```text
modeling/trifusion/signal_preserving_v5.py
modeling/trifusion/signal_preserving_v5_builder.py
configs/RGBNT201/TriFusion-signal-preserving-v5-rtx3090.yml
tools/run_signal_preserving_v5.py
tests/test_trifusion_signal_preserving_v5.py
tests/test_run_signal_preserving_v5.py
tools/diagnose_signal_preserving_v5.py
tests/test_diagnose_signal_preserving_v5.py
```

已实现并测试的合同：

- `FrozenSignalBackbone` 严格冻结上游 Signal 参数，并在父模型进入 train 模式时继续保持 Signal 为 eval；
- 从 Signal 的三模态视觉编码器取得 patch/global 特征，`baseline_only` 严格使用原 3072D `direct+SIM`；
- CNN、Transformer、Mamba 各有三阶段完整专家，阶段 1/2 后执行双向 HFER，阶段 1/2/3 刷新联合可靠性；
- 每个专家残差以对应 direct Signal 模态全局特征为基准，归一化、限幅后按可靠性和身份效用路由，九路残差全部保留；
- 默认宽度为 `baseline=3072`、各 branch=`3840`、residual bank=`2304`、fused=`5376`，且 `fused[:, :3072]` 与 `baseline_only` 精确相同；
- 同一模型和同一 checkpoint 显式输出 `baseline_only`、`fused`、`cnn`、`transformer`、`mamba`；
- 专项测试覆盖 baseline 前缀相等、所有专家和路由均有梯度、一次 optimizer step 后 Signal 状态逐张量不变、两次 relay、冻结实验合同、损失权重、晋级门、过拟合门和学习率；远端联合实测 `10 passed, 3 warnings`，warnings 仅为 timm 导入弃用提示。

真实工程门终态：

| 门 | 结果 | 关键证据 |
|---|---|---|
| preflight | PASS | 全 825/825 dev；上游 Signal 与 V5 baseline 逐批逐元素相等；四指标精确为 `58.0109/57.4545/69.9394/76.6061` |
| capacity | PASS | B32/K4、8/8 step、213/213 可训练梯度张量、0 overflow、峰值 reserved `3542 MiB`、Signal SHA 不变 |
| overfit | PASS | 同一真实 B32/K4 批 100 step；loss `2.81624→0.05921`、ratio `0.02102≤0.10`、0 overflow、Signal SHA 不变 |

capacity 首次运行真实发现 6 个 `private_projection` 张量无梯度。原因是 V5 不启用 peer-teaching/private-diversity，这 6 个张量没有任何训练目标；最小修复是在 V5 builder 中冻结它们。修复后可训练参数为 `5,395,989`、梯度覆盖 213/213，forward 特征和 baseline parity 均未改变。失败收据永久保留在远端 `_capacity_5f1ecf6_v1`，通过收据为 `_capacity_5f1ecf6_v2`。

版本化轻量证据：

```text
evidence/trifusion_signal_preserving_v5_preflight_seed42.json
evidence/trifusion_signal_preserving_v5_capacity_seed42.json
evidence/trifusion_signal_preserving_v5_overfit_seed42.json
```

完整 V5 dev 运行目录：

```text
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_signal_preserving_v5_dev_seed42_18f81c3
```

最佳 epoch51 已严格重载，终局指标和失败门见第 5.4 节。read-only 诊断文件为同目录 `diagnostic.json`；版本化轻量副本为：

```text
evidence/trifusion_signal_preserving_v5_dev_terminal_seed42.json
evidence/trifusion_signal_preserving_v5_diagnostic_seed42.json
```

runner 已按最直接方案处理已证实的包名冲突：Signal 源码拥有顶层 `modeling`，本项目从 `<repo>/modeling` 以顶层 `trifusion` 导入 V5；没有增加兼容层，也没有复制或改写 Signal 包。

V5 晋级合同保持不变，而本次实际未通过：`fused` 高于 baseline/Transformer/Mamba，但低于 CNN 且未达到 65 mAP。独立 result-to-claim 只给出 `partial`：精确保留 Signal 的工程子主张成立，协同增益和三项创新有效性不成立。当前没有运行时 fallback。

## 13. 建议技能与接续顺序

下一执行者应按以下顺序调用技能：

1. 先读 V6 terminal/diagnostic receipt 与 `results/TRIFUSION_RGBNT201_V6_DEV_SEED42_2026-09-01.md`；只实现一个“相对 exact baseline 的边际身份收益路由”main-only 修正；
2. `/tdd` 与 `/run-experiment`：V7 先过最小 TDD、preflight、capacity 和 overfit，再运行唯一 seed42 held-out-dev；
3. `/monitor-experiment`：长任务按 180–300 秒间隔或预计结束前数分钟轮询；
4. `/analyze-results` 与 `/result-to-claim`：完整 dev 后重新判断；
5. `/ablation-planner`：只有新主方法正式超过冻结目标且 `claim_supported=yes` 后才允许使用。

接续时不要重跑 Signal baseline、V5 或 V6，不做多种子，不做消融，不访问 official test。下一步不是调 batch、epoch、学习率、温度或残差倍率，而是利用“CNN 最强但获得最低路由权重”的诊断，让路由目标直接表达各专家相对 exact baseline 的边际身份收益。泛化回落是次级问题；不要把 60/60 epoch 的结果解释为未训练完成。

## 14. Signal-preserving V6 完整终态

V6 是 V5 诊断后的唯一 main-only 修正，不是消融或超参扫描：

1. exact Signal 3072D 前缀和独立 `baseline_only` 输出保持不变；
2. 移除 learned residual scale，把路由后的联合残差银行按样本无自由倍率地校准到 baseline 能量；
3. 为 CNN、Transformer、Mamba 各自增加 residual-only ID/triplet 监督，并从 residual-only batch-hard 身份间隔构造路由效用，避免冻结 baseline 代替专家完成目标。

源码身份：训练启动时 Git commit `e4a6bffcbf77ee6dde301551b8ac7a249af0fed9`。V5/V6 联合专项 `16 passed, 3 warnings`。真实工程门：

| 门 | 结果 | 关键证据 |
|---|---|---|
| preflight | PASS | 825/825 exact Signal parity；baseline `58.0109/57.4545/69.9394/76.6061`；optimizer0/official0 |
| capacity | PASS | B32/K4、8 step、218/218 梯度、0 overflow、峰值 reserved `3554 MiB`、Signal SHA 不变 |
| overfit | PASS | 同一真实批 100 step；`4.06445→0.22984`，ratio `0.05655≤0.10`；official0 |

轻量证据：

```text
evidence/trifusion_signal_preserving_v6_preflight_seed42.json
evidence/trifusion_signal_preserving_v6_capacity_seed42.json
evidence/trifusion_signal_preserving_v6_overfit_seed42.json
evidence/trifusion_signal_preserving_v6_dev_terminal_seed42.json
evidence/trifusion_signal_preserving_v6_diagnostic_seed42.json
```

完整 dev 运行目录：

```text
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_signal_preserving_v6_dev_seed42_e4a6bff
```

按 fused dev mAP 选择 epoch8 并严格重载后的同 checkpoint 指标：

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| fused | 58.7321 | 57.5758 | 69.2121 | 76.7273 |
| CNN | **59.1022** | **59.6364** | **70.3030** | 76.1212 |
| Transformer | 57.7962 | 57.4545 | 68.3636 | 76.0000 |
| Mamba | 58.7298 | 57.6970 | 69.5758 | 76.3636 |

60/60 epoch、5,498 optimizer steps、0 overflow，峰值 reserved `6084 MiB`，Signal state 在训练前后和重载后完全不变，official access=0。主门 FAIL：fused 比 baseline 高 `0.7212 mAP`，但比 CNN 低 `0.3701`，且比 65 mAP 低 `6.2679`。

只读诊断处理全部 825 个 dev 样本且 optimizer0。V6 的 suffix/baseline norm ratio 为 `1.0`；fused/baseline 距离相关 `0.96875`、Top-10 overlap `95.3939%`，证明残差已实际改变排序。三专家残差保持低余弦，但路由熵 `0.97435` 且样本间变化很小；最强 CNN residual-only mAP `56.9267` 却只获约 `0.228–0.245` 权重，低于 Transformer 和 Mamba。result-to-claim 为 `no/high/provisional`：只支持 exact Signal preservation 和 held-out-dev 上 `+0.7212 mAP` 的窄主张，不支持协同优越性、65 mAP、official 或 SOTA。

V6 已完成且失败，不得重跑。下一步只允许一个 V7 main-only routing-alignment 修正；正式晋级合同仍是 fused mAP 至少 65，并严格高于 baseline_only、CNN、Transformer、Mamba。失败前继续禁止 official test、消融和多种子。

## 15. Signal-preserving V7 启动前冻结状态（2026-09-02）

V7 不是在 V6 上扫描倍率或学习率，而是一次由失败证据限定的结构修正：

1. RGB/NIR/TIR 一次采样并共享 flip、padding、crop；几何对齐后才独立 RandomErasing；
2. 残差定义为匹配的 `expert final token - Signal anchor token`，再归一化、池化和投影；
3. 保留两次 HFER 和三次 reliability refresh；
4. 联合权重为 `P(modality|x) * P(expert|modality,x)`，九个槽位总质量为 1，缺失模态严格为 0；
5. Router 目标是每个 `expert×modality` 追加槽相对 exact baseline 的 L2-normalized batch-hard 身份间隔增益；
6. 干净同步视图训练 ReID/边际效用，单模态受控模糊视图独立监督质量；
7. 最终残差强度由样本级 `alpha∈[0,0.5]` 控制，初始化 0.2，不再固定等能量。

V6 epoch8 checkpoint SHA256 为 `32bba88c...ee2e`。V7 迁移结果 unexpected keys 为 0，missing keys 仅为新增 `fusion.alpha_predictor` 的四个 weight/bias；Signal checkpoint 和 3072D baseline 保持原 SHA 与逐元素输出。

Oracle 与启动门：

| 项目 | 结果 |
|---|---:|
| V6 branch Oracle | 63.6089 mAP / 64.1212 Rank-1 |
| Oracle - 最强固定 CNN | +4.5067 mAP / +4.4848 Rank-1 |
| V7 preflight | PASS；baseline 58.0109/57.4545；official0 |
| V7 capacity | PASS；B64/K8；8 step；222/222；reserved 11486 MiB |
| V7 overfit | PASS；100 step；超额损失 ratio 0.08048；official0 |

轻量证据：

```text
evidence/trifusion_signal_preserving_v6_oracle_complementarity_seed42.json
evidence/trifusion_signal_preserving_v7_preflight_seed42.json
evidence/trifusion_signal_preserving_v7_capacity_seed42.json
evidence/trifusion_signal_preserving_v7_overfit_seed42.json
results/TRIFUSION_RGBNT201_V7_READINESS_2026-09-02.md
```

正式只允许一次 seed42、60-epoch、141-fit/30-dev。前 10 epoch 只训练 reliability Router 与 alpha；第 10 epoch 结束后，RGB、NIR、TIR 各自受控模糊都必须使对应模态平均质量严格下降，否则在进入 joint phase 前 fail closed。之后才联合微调全部既有专家/HFER/Router。正式门仍是 fused mAP 至少 65 且严格超过 baseline_only、CNN、Transformer、Mamba；未通过前继续禁止 official test、消融和多种子。

## 16. Signal-preserving V7 终态（2026-09-02）

远端运行目录：

```text
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_signal_preserving_v7_dev_seed42_b0087fa
```

代码身份为 `b0087fafbc25efccadceb97e4050ca04d977d3c3`。运行完成 60/60 epoch、2,520 optimizer steps，耗时 `2419.56 s`，0 overflow，峰值 allocated/reserved 为 `11176.83/12908 MiB`；Signal state SHA 在训练前、训练后和严格重载后完全一致；official access=0。最佳 checkpoint 为 epoch1，SHA256 `8bcdf3583e121dd7a7b0071743b8fd34f93a82cc710bd07e26b53c3693609a2b`。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| fused | 58.3293 | 57.9394 | 70.1818 | 76.7273 |
| CNN | 58.2773 | 57.4545 | 69.9394 | 76.9697 |
| Transformer | 58.3028 | 58.0606 | 70.1818 | 76.6061 |
| Mamba | **58.3476** | 57.8182 | **70.3030** | **76.9697** |

主门 FAIL：fused 比 baseline 高 `0.3184 mAP`，但比 Mamba 低 `0.0183`，并比 65 mAP 门低 `6.6707`。epoch10 质量门虽通过，但 joint 阶段最佳 epoch11 fused 仅 `57.9804`，epoch60 为 `57.7550`，因此不是没训练完，而是联合训练降低身份外泛化。

只读诊断对全部 825 个 dev 查询执行，`optimizer_steps=0`、`official_test_access_count=0`：

- 联合 Router 归一化熵 `0.99791`，模态熵 `0.99994`，样本 alpha 为 `0.198947±0.0000015`；
- 单个确定性 B64 fit batch 上，预测 Top-slot 与逐槽边际身份效用目标一致率为 `14.0625%`；
- fused 与 baseline 距离 Pearson 为 `0.999786`，Top-10 overlap 为 `99.6364%`；
- residual-only CNN/Transformer/Mamba mAP 为 `59.1317/54.8594/57.8991`，ground-truth Oracle 为 `62.7435`，比最强固定 residual 高 `3.6118`，三专家 leave-one-out 边际均为正。

独立 result-to-claim 判定为 `no/high`。只支持“V7 在固定 held-out dev 上精确保留 Signal 并取得 `+0.3184 mAP`”这一窄主张；不支持 fused 优于全部专家、65 mAP、三项创新有效、official 或 SOTA。V7-specific independent integrity audit 尚未补做，完整性边界标记为 provisional/warn。V7 不得重跑，消融、多种子和 official test 继续封闭。下一主版本必须先提出一个由上述诊断直接导出的结构假设并重新通过 train-only 门，不能原样延长训练或扫超参数。

轻量证据：

```text
evidence/trifusion_signal_preserving_v7_dev_terminal_seed42.json
evidence/trifusion_signal_preserving_v7_diagnostic_seed42.json
results/TRIFUSION_RGBNT201_V7_DEV_SEED42_2026-09-02.md
```

## 17. V8 frozen-router 预训练探针终态（2026-09-02）

该探针不是 V8 主模型，而是正式实现前的 fail-fast 资格检查。它加载 V7 epoch1 checkpoint `8bcdf358...09a2b`，冻结全部模型参数；只用 141-fit 中具备跨摄像头正样本的 21 个身份、571 个 query 解析拟合一个 18→3 最小二乘效用教师，然后在完全身份隔离的 30-dev、825 query 上评估。`model_training_executed=false`、`optimizer_steps=0`、`official_test_access_count=0`。

专家赢家预测：

| 项目 | CNN | Transformer | Mamba | 准确率 | 多数类 |
|---|---:|---:|---:|---:|---:|
| fit 真实标签/教师预测 | 100.00% | 0.00% | 0.00% | 100.00% | 100.00% |
| dev 真实赢家 | 55.27% | 17.45% | 27.27% | — | 55.27% |
| fit 教师在 dev 的预测 | 100.00% | 0.00% | 0.00% | 55.27% | 55.27% |
| V7 Router 在 dev 的预测 | 0.00% | 4.61% | 95.39% | 27.39% | 55.27% |

部署型冻结特征结果：

| 输出 | mAP | Rank-1 |
|---|---:|---:|
| exact Signal baseline | 58.0109 | 57.4545 |
| residual-only CNN | 59.1317 | 59.3939 |
| equal-energy current Router | 59.5902 | 58.9091 |
| equal-energy uniform | **59.6188** | **59.1515** |
| equal-energy fit utility teacher | **59.6188** | **59.1515** |

结论是双重的：V7 的约 0.2 residual 能量确实太弱，恢复等能量可比 baseline 高 `1.6079 mAP`；但 fit 域没有专家赢家多样性，教师无法学习逐样本选择，且最佳部署型结果仍比 65 低 `5.3812`。结合 GT residual Oracle `62.7435<65`，Router-only V8 已被正式拒绝，不得启动完整训练。下一主版本必须改变专家表征能力与结构化任务分工，而不是继续扫 Router、alpha、epoch 或学习率。

FP32、关闭 cuDNN benchmark 后的两次探针重放核心 JSON 字节级一致。最终证据：

```text
evidence/trifusion_v8_frozen_router_probe_seed42.json
results/TRIFUSION_RGBNT201_V8_FROZEN_ROUTER_PROBE_2026-09-02.md
```

## 18. V8 pretrained-tail Phase-A 专家形成终态（2026-09-02）

该版本不是在 V7 上继续扫 Router、alpha 或 epoch，而是直接修复专家表示与任务分工：冻结并逐元素保留 Signal 3072D baseline；从 CLIP block8 的 token 序列分叉；CNN、Transformer、Mamba 分别通过相同的冻结 pretrained tail blocks 9/10/11，再在每个 tail stage 后加入结构化残差。CNN 负责横向 part/local detail，Transformer 负责 CLS/global relation，Mamba 同时执行二维空间扫描与对齐 RGB/NI/TI 的跨模态长程扫描。残差严格定义为专家 tail 输出减去同路径冻结 tail reference。Phase-A 关闭 Router 和 HFER，避免专家形成前被联合目标拉回同质化。

源码身份与工程门：

- core/runner 形成提交：`b21db0ae6d1a42add651459242edd10940025dd3`；
- formation-probe runner 提交：`abbf33d0f8ccee897391d910fb0461ffe3184aaf`；
- exact preflight：全 825/825，baseline `58.0109/57.4545/69.9394/76.6061` 逐项一致；
- capacity：真实 B64/K8，8 step，203/203 梯度张量，0 overflow，峰值 reserved `6006 MiB`；
- overfit：100 step，loss `4.1156→0.6125`，扣除 label-smoothing 理论下限后的 ratio=`0.000534≤0.1`；
- 总参数/可训练参数=`100,171,789/9,068,556`，Signal state SHA 在所有门前后不变。

Phase-A 探针只用 seed42，训练 20 epoch/840 optimizer steps，训练期间 dev evaluation=0，最终 checkpoint 才在 30-ID held-out dev 上评估一次；耗时 `933.24s`，0 overflow，峰值 reserved `6214 MiB`，official access=0。未进行 checkpoint 选择。

固定输出：

| 输出 | mAP | Rank-1 | 相对 baseline mAP |
|---|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | — |
| fixed equal-energy fused | 58.0972 | 56.8485 | +0.0863 |
| baseline + CNN residual | 57.6071 | 56.4848 | -0.4037 |
| baseline + Transformer residual | 56.3031 | 55.8788 | -1.7077 |
| baseline + Mamba residual | 56.6277 | 54.4242 | -1.3832 |

query-wise GT Oracle（诊断、非部署）：

| 诊断 | 最强固定 mAP | Oracle mAP | Oracle Rank-1 | 增益 |
|---|---:|---:|---:|---:|
| baseline + expert branch | 58.0109 | 64.7850 | 65.9394 | +6.7741 |
| residual-only experts | 53.8660 | 63.4813 | 66.9091 | +9.6153 |

branch 独有 AP 胜例 CNN/Transformer/Mamba=`201/170/138`，leave-one-out 边际=`+1.2043/+1.9592/+0.8435 mAP`；residual-only 对应为 `257/232/199` 与 `+3.1128/+4.9698/+2.6370`。这支持“三专家现在具有不同查询优势”的窄结论，但固定 fused 仍只比 baseline 高 `0.0863 mAP` 且 Rank-1 更低。branch Oracle 也仍比 65 低 `0.2150`，因此仅硬选择分支不足以达门。

独立 result-to-claim=`partial/medium`；独立 V8 审计=`WARN`。下一步只允许：冻结 Phase-A 专家，不读取 dev Oracle 标签，用 fit-only OOF/CIRC `expert×modality` 效用和受控质量退化训练层级 Router；缺失模态质量必须为零，受损模态质量必须下降。Router 证明可部署 fused 超过 baseline/固定专家后，才允许以低学习率启用 typed HFER。65 mAP 前继续禁止 official test、消融和多种子。

证据与报告：

```text
evidence/trifusion_signal_preserving_v8_expert_formation_preflight_seed42.json
evidence/trifusion_signal_preserving_v8_expert_formation_capacity_seed42.json
evidence/trifusion_signal_preserving_v8_expert_formation_overfit_seed42.json
evidence/trifusion_signal_preserving_v8_expert_formation_probe_seed42.json
results/TRIFUSION_RGBNT201_V8_EXPERT_FORMATION_PHASE_A_2026-09-02.md
EXPERIMENT_AUDIT_V8_PHASE_A.md
```

## 19. V8 OOF-margin Router Phase-B 终态（2026-09-02）

Phase-B 没有更新 Phase-A 专家，也没有读取 dev/official 生成训练目标。原 OOF per-query AP 标签接近饱和后，改用连续 identity margin：最近负样本距离减最远正样本距离。571 个 fit-only OOF query 中，CNN/Transformer/Mamba 独有 slot winner 为 `38/350/183`，RGB/NI/TI 为 `215/59/297`；slot Oracle mean margin=`0.317710`，比最佳固定 slot 高 `0.164303`。Oracle 使用真实身份标签，只是训练域诊断，不是部署结果。

层级 Router 严格实现 `w(e,m)=P(m|x)P(e|m,x)`，并预测 `alpha∈(0,0.5]`。三个身份隔离 Router fold 各训练 100 epoch，随后在全部合格 fit 身份上 refit 100 epoch，共 400 个 Router optimizer step。Phase-A expert state SHA 在训练前后均为 `ecfd7fbc...fb77`；combined checkpoint SHA256 为 `6f95f99a86763580c3bd8592974347825659a5336f9afec43062516d21fbfe02`。

fit-only OOF 门仅窄幅通过：

| Router 诊断 | Learned | Fixed/majority | 差值 |
|---|---:|---:|---:|
| Expected identity margin | 0.1020340 | 0.1017202 | +0.0003137 |
| Top-slot accuracy | 17.8634% | 17.6883% | +0.1751 pp |

这只能证明 Router 训练链有很弱的正向泛化迹象，不能称为强路由能力。质量语义门通过：missing modality 最大权重严格为 0；单独模糊 RGB/NI/TI 后，对应模态平均质量分别从 `0.306154/0.298051/0.395795` 降到 `0.117502/0.102016/0.166562`。

combined checkpoint 只进行了一次冻结 held-out-dev 评估，评估期间 optimizer0、模型状态不变、official access0：

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| baseline_only | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| fused | **58.4050** | **59.3939** | **71.2727** | 76.6061 |
| CNN | 57.6071 | 56.4848 | 70.9091 | **77.5758** |
| Transformer | 56.3031 | 55.8788 | 69.6970 | 76.2424 |
| Mamba | 56.6260 | 54.4242 | 68.8485 | 75.1515 |

fused 比 exact Signal baseline 高 `0.3941 mAP / 1.9394 Rank-1`，并严格超过三个固定专家；这是当前唯一支持的部署结论。但 fused 仍比 65 mAP 门低 `6.5950`，所以 `promotion_gate=false`、`next_phase_authorized=false`。独立 result-to-claim=`partial/medium`：不能把联合增益单独归因为 learned Router，因为当前输出同时使用软融合和样本级 alpha，且 OOF learned-vs-fixed 优势极小。

V8 Phase-B 至此封存为“正向但未晋级”。不得开启 HFER、消融、多种子、official test，也不得扫描 Router/alpha/epoch/LR。若继续冲击 65，下一版本必须是新的表示级主假设，能生成现有固定输出之外的新身份表示，并重新通过 exact Signal parity、真实 B64/K8 capacity、overfit 和 fit-only 互补门。

独立完整性审计为 `WARN`，不是结果逻辑失败：GT 来源、常规 ReID 归一化、实际调用路径、fit/dev 边界和评价类型分类均 PASS。警告来自大型 checkpoint/cache 仍仅保存在远端，本地审计者不能从 fresh clone 直接重算其 SHA。封存时已在远端只读重算 Phase-A checkpoint `d37ca17...b40f` 和 combined checkpoint `6f95f99a...fe02`，均与 receipt 一致；这降低了拷贝错误风险，但不取消 remote-only packaging 警告。

证据与报告：

```text
evidence/trifusion_v8_oof_router_margin_targets_seed42.json
evidence/trifusion_v8_oof_margin_router_phase_b_seed42.json
evidence/trifusion_v8_oof_margin_router_dev_seed42.json
results/TRIFUSION_RGBNT201_V8_OOF_MARGIN_ROUTER_PHASE_B_2026-09-02.md
EXPERIMENT_AUDIT_V8_PHASE_B.md
EXPERIMENT_AUDIT_V8_PHASE_B.json
```

---

本文件记录的是可核验工程状态，不是论文结论。任何后续结果都必须保留单种子、固定终点、官方 test 一次访问以及失败对称性审计这些边界。

## 20. V9 Orthogonal Triadic Relay Synthesis 终态（2026-09-02）

V9 是 V8 Phase-B 之后唯一执行的新表示级主假设。它冻结 exact Signal、
V8 pretrained-tail 三专家与 Phase-B Router；每个专家进行两轮仅来自另外
两支的 receiver-specific peer relay，将消息投影到 receiver 的正交补后注入，
再由三专家与三组 pairwise product 合成新的 1536D synergy。完整 7680D
Phase-B embedding 是 V9 9216D fused 的逐元素精确前缀。

工程门全部通过：

- 公共接缝 RED→GREEN，V9 与相邻 V8 共 12 tests；
- preflight 保持 Signal/Phase-B exact prefix，最大 relay cosine=`2.98e-8`；
- RTX3090 真实 B64/K8 capacity 8 step，59/59 gradients，0 overflow，
  allocated/reserved=`1426.80/2020 MiB`；
- 真实固定批 100-step loss=`3.78850→0.61228`，label-smoothing excess ratio
  `0.000518≤0.1`；所有 train-only 门均 dev0/official0。

唯一正式训练在代码 `b40b171` 下完成 60/60 epoch、2,520 optimizer steps，
耗时 `1334.80s`，0 overflow；训练 loss=`3.45323→0.62362`。checkpoint：

```text
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v9_train_seed42_b40b171/final_model.pth
SHA256 c118ada931451929ec91cc374f9be8c3f518766b4dc02dda7372e525f07c7cfa
```

训练阶段 dev access=0。final checkpoint 只进行一次冻结 30-ID dev 评估，
optimizer0、training=false、official0、checkpoint/Phase-A/Router state SHA
评估前后不变。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| exact Signal baseline | 58.0109 | 57.4545 | 69.9394 | 76.6061 |
| frozen V8 Phase-B | **58.4050** | **59.3939** | **71.2727** | **76.6061** |
| V9 fused | 56.5339 | 57.2121 | 68.3636 | 75.5152 |
| V9 CNN | 55.8825 | 57.3333 | 68.6061 | 75.8788 |
| V9 Transformer | 51.3416 | 49.3333 | 65.8182 | 73.3333 |
| V9 Mamba | 54.6342 | 54.7879 | 68.1212 | 76.2424 |

主门明确 FAIL：fused 比 exact Signal 低 `1.4770 mAP / 0.2424 Rank-1`，
比 Phase-B 低 `1.8711 / 2.1818`，比 65 门低 `8.4661 mAP`。fused 虽高于
三个已经退化的 V9 专家输出，但这不能支持协同增益。dev 上 beta
mean/min/max=`0.498794/0.462330/0.499998`，接近0.5上限；没有消融，故
不能因果宣称 beta 饱和导致失败。

独立 result-to-claim=`no/high`。独立审计=`WARN / warn /
FAIL_TO_PROMOTE`：GT 来源、普通 ReID L2 normalization、实际执行路径和
评价类型均通过；WARN 来自远端 checkpoint 未随仓库发布、审计时两个终态
JSON 尚未追踪/文档滞后，以及 config 比较字段不是 evaluator 的唯一数据源。
这些不改变负指标。

V9 至此封存：不做 official test、消融、多种子、checkpoint 选择或
beta/epoch/LR/residual 扫描。任何后继必须是新的表示级机制，并在访问 dev
前用 fit-only 身份隔离 OOF 检索证明新增表示具有正效用；当追加表示会伤害
Phase-B 时，必须能由训练侧证据抑制。当前没有授权 V10 或新 GPU 作业。

证据：

```text
evidence/trifusion_v9_preflight_seed42.json
evidence/trifusion_v9_capacity_seed42.json
evidence/trifusion_v9_overfit_seed42.json
evidence/trifusion_v9_train_seed42.json
evidence/trifusion_v9_dev_seed42.json
results/TRIFUSION_RGBNT201_V9_DEV_SEED42_2026-09-02.md
EXPERIMENT_AUDIT_V9.md
EXPERIMENT_AUDIT_V9.json
```

## 21. V10-Q0 frozen DINOv2 fit-only 资格终态（2026-09-02）

V10 没有直接实现或训练 CLIP+DINO 三分支，而是先执行预注册的零训练资格门。
云端 DINOv2 ViT-B/14 权重 SHA256 为
`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`；
只删除预训练专用 `mask_token` 后 strict load。输入由现有归一化确定性转换为
ImageNet normalization 和 `252×126`，输出为一个 CLS 加18×9 patch，token
shape=`163×768`。Phase-B、Router 和 DINO state 在执行前后不变。

范围仅为 141-fit 中 21 个跨摄像头身份、571 query；`optimizer_steps=0`、
`training_executed=false`、`dev_access_count=0`、`official_test_access_count=0`。
这是 real-GT fit-only diagnostic，不是 dev、official 或论文主结果。

| 冻结表示 | mAP | Rank-1 |
|---|---:|---:|
| V8 Phase-B | **100.0000** | **100.0000** |
| DINOv2 ViT-B/14 | 7.6284 | 6.1296 |
| fixed equal-block concat | 92.2120 | 95.9720 |

Phase-B/DINO hard Oracle 仍为100/100，Oracle gain=0；unique AP wins 为
`571/0`。固定拼接相对 Phase-B 下降 `7.7880 mAP`。因此 concat≥+1、
Oracle≥+2 和双源独有 AP 胜例四项预注册门全部失败，
`qualification_gate=false`、`next_phase_authorized=false`。JSON 中
`status=PASS` 只表示程序完成，不能解释为资格通过。

独立 result-to-claim=`no/high`。它只支持“当前冻结 DINO 表示在这一饱和
fit 协议下没有可用互补、固定等块拼接有害”；不能外推为 DINOv2 普遍不适合
RGBNT ReID。独立审计=`WARN / warn / FAIL_TO_QUALIFY—STOP_V10_Q0`：
GT/协议、归一化、实际路径、scope 和评价类型均 PASS；WARN 来自审计时 JSON
未追踪以及大二进制权重仍只在远端，不能从本地 fresh clone 重哈希。

V10 至此封存：不实现Q1，不训练、不访问dev，不扫描模态子集、分辨率、
intermediate block、token pooling、concat 权重或训练头。若未来再使用DINO，
必须是新的预注册假设，并先建立非饱和、身份隔离的train-only资格门；不能作为
V10事后挽救。当前未授权V11或新的GPU作业。

证据：

```text
evidence/trifusion_v10_dinov2_fit_qualification_seed42.json
results/TRIFUSION_RGBNT201_V10_DINOV2_FIT_QUALIFICATION_2026-09-02.md
EXPERIMENT_AUDIT_V10_Q0.md
EXPERIMENT_AUDIT_V10_Q0.json
```

## 22. V11-Q0 identity-OOF residual complement 资格终态（2026-09-02）

V11 试图修复 V10 的显式 Phase-B 100 mAP 饱和问题：复用三个已训练的

```text
trifusion_v8_oof_router_targets_seed42_f7b3cfc/fold_{0,1,2}_experts.pth
```

checkpoint，每折只在该 expert adapter 未见过的 held-out 身份上计算距离；
所有检索严格折内完成，跨折仅按 query 聚合 AP/Rank-1。资格指标完全排除
exact Signal 和 Phase-B embedding，只比较 CNN/Transformer/Mamba
residual、三专家 residual bank、固定 DINOv2 和唯一等块拼接。

| 输出 | mAP | Rank-1 |
|---|---:|---:|
| CNN residual | 98.5115 | 98.4238 |
| Transformer residual | 100.0000 | 100.0000 |
| Mamba residual | 99.9416 | 100.0000 |
| residual bank | **100.0000** | **100.0000** |
| DINOv2 | 14.1323 | 9.4571 |
| fixed equal-block concat | 95.8582 | 96.4974 |

fixed concat 比 residual bank 低 `4.1418 mAP`；residual-bank/DINO hard
Oracle 仍为100/100，Oracle gain=0，unique AP wins=`570/0`。三折和571
query协议门通过，但 `non_saturation=false`、总资格门 false。

原因已经由执行证据和独立审计定位：fold expert adapter 本身没有看到 held-out
身份，但每支专家的输入仍来自

```text
/root/autodl-tmp/trifusion-v2/artifacts/signal_baseline_dev_seed42_f7d4b30/Signalbest.pth
```

对应的 frozen Signal token field；该 Signal checkpoint 已在全部141个fit身份
上训练。因此 adapter training 是 OOF，完整特征路径不是 identity-unseen。
100 mAP 是 fit-identity 泄漏/饱和证据，不是部署增益，也不是指标归一化造假。

本次运行 `optimizer_steps=0`、`training_executed=false`、dev0、official0，
冻结状态未改变，峰值 allocated/reserved=`2820.82/3878 MiB`。独立
result-to-claim=`no/high`；独立审计=`WARN / warn /
FAIL_TO_QUALIFY—STOP_V11_Q0`。WARN 来自完整路径隔离不成立以及大二进制
只在远端；GT、普通L2检索、实际执行路径和评价类型本身通过。

V11 至此封存：不实现Q1/Q2，不训练、不访问dev/official，不做消融、多seed
或 DINO 模态/分辨率/block/token/fusion/head 扫描。该结果不能外推为 DINOv2
普遍无效。任何后继必须作为新预注册假设，使完整测量特征路径对 held-out
身份未见并先证明非饱和；同时遵守不复跑 baseline 的用户约束。

证据：

```text
evidence/trifusion_v11_dinov2_oof_residual_complement_seed42.json
evidence/trifusion_v11_dinov2_oof_residual_complement_seed42_provenance.json
results/TRIFUSION_RGBNT201_V11_DINOV2_OOF_RESIDUAL_QUALIFICATION_2026-09-02.md
EXPERIMENT_AUDIT_V11_Q0.md
EXPERIMENT_AUDIT_V11_Q0.json
```

## 23. V12 complete-path identity-OOF teacher 与 Router 终态（2026-09-02）

V12 直接修复 V11 已证明的完整路径身份泄漏。固定三折内，每折从 raw
`ViT-B-16.pt` 初始化一个内部 Signal 教师，只在另外94个fit身份上训练50
epoch；随后 CNN/Transformer/Mamba expert 也只在相同94身份上训练20 epoch。
Signal 和 expert 均只使用 final epoch，不读取 held-out 指标选 checkpoint。
每折 held-out 为47身份，train/heldout overlap 全为0；可评价 query 数依次为
`190/179/202=571`。fold Signal 是方法内部教师，不是 baseline 重跑，也不报告
为部署模型。

### 23.1 工程门与 Q0 资格

公共接缝经历真实 RED→GREEN，远端相邻测试8/8通过。真实 fold0 B64/K8
preflight 完成1步，191个 gradient tensors，loss=`13.70739`，0 overflow，
allocated/reserved=`10701.82/11502 MiB`，dev0/official0。

正式 Q0 完成三折 Signal50+Expert20：

- 总 optimizer steps=`5839`，overflow=0；
- 峰值 allocated/reserved=`11375.61/17530 MiB`；
- 耗时=`3958.52s`；
- dev access=0，official access=0；
- raw CLIP SHA=`5806e77c...416f`；
- target cache SHA=`fdacc405...ac681`。

完整路径 held-out residual-only 聚合结果：

| 输出 | mAP | Rank-1 |
|---|---:|---:|
| CNN residual | 83.7717 | 85.6392 |
| Transformer residual | 86.9549 | 89.4921 |
| Mamba residual | 85.8870 | 88.6165 |
| residual bank | 87.9968 | 90.1926 |
| residual expert hard Oracle | **92.2679** | **95.2715** |

Oracle 比最强固定 expert 高 `5.3130 mAP / 5.7793 Rank-1`。AP 独有胜出
CNN/Transformer/Mamba=`79/118/76`；slot-margin expert winner=
`210/186/175`，RGB/NI/TI winner=`293/119/159`。slot Oracle mean margin
`0.099913`，比最强 fixed slot 高 `0.186130`。所有 fixed 输出均低于99 mAP，
完整路径隔离、非饱和、专家/模态多样性、Oracle、训练计划、运行时和访问门
全部通过，故 Q0 唯一授权了固定 Q1 Router。

上述数值分类为 `real_gt_train_identity_oof` 与
`teacher_proxy_train_identity_oof`；是 train-only 资格诊断，不是 dev、official
或 deployable mAP。

### 23.2 Q1 Router 失败并停止

Q1 保持 V8 Phase-A checkpoint、层级 Router、质量退化、100 epoch、LR、
alpha 和全部门槛不变，只替换成 V12 cache。三折各100 epoch，共300 Router
optimizer steps；Phase-A expert state SHA 前后同为 `ecfd7fbc...fb77`，没有
expert training。

| Router 门 | Learned | Fixed / majority | 结论 |
|---|---:|---:|---|
| OOF expected identity margin | -0.117330 | -0.099975 | FAIL |
| Top-slot accuracy | 12.2592% | 16.8126% | FAIL |

质量语义门通过：missing modality 最大质量为0；扰动 RGB/NI/TI 后各自平均
质量从 `0.325516/0.316737/0.357746` 降至
`0.111743/0.104597/0.144510`。但这不改变身份效用路由的两项失败。

因此 `next_phase_authorized=false`、`final_training=null`、
`combined_checkpoint=null`。Q1 耗时32.36s，峰值 allocated/reserved=
`2459.15/3400 MiB`，dev0、official0。没有运行 V12-R001，也没有新的可部署
指标；当前同协议最好仍是 V8 Phase-B fused `58.4050 mAP / 59.3939 Rank-1`。

### 23.3 Claim、审计与封存边界

独立 result-to-claim=`partial/high`：只支持“完整路径 OOF 产生非饱和且具有
专家/模态多样性的 residual utility 教师”，不支持 Router 增益、65 mAP、
official、SOTA 或泛化。独立完整性审计=
`WARN/warn/Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`；GT、完整路径隔离、普通
ReID L2、实际代码路径、scope 和评价类型均 PASS。WARN 来自大 checkpoint/
cache 仍只在远端，轻量 GitHub clone 不能独立重哈希；终态 provenance wrapper
已补齐项目 commit、config/runner/log/result SHA，但不取消 remote-only 包装限制。

Q0 原始 summary 的本地/远端 SHA 都为 `9105b86a...8a8b69`，Q1 都为
`c42f1148...c9b5`。Q0 目录只有一个按 fold0→fold1→fold2→terminal 顺序生成的
artifact 序列；以原始 summary 和日志为权威，任何较早转述中的不同步数、耗时
或 cache SHA 均作废。

V12 至此封存：不访问dev/official，不做消融、多种子、HFER，且不扫描fold、
epoch、LR、alpha、margin temperature 或门槛。一个与现象一致但未被因果
消融证明的解释是 complete-path fold target 与 all-fit Phase-A Router 输入存在
表示/分布错配；后继必须是新的预注册监督-表示对齐假设，并先在train-only
identity-disjoint门上让learned Router严格超过固定策略。

证据：

```text
evidence/trifusion_v12_complete_path_preflight_seed42.json
evidence/trifusion_v12_complete_path_oof_seed42.json
evidence/trifusion_v12_complete_path_router_seed42.json
evidence/trifusion_v12_complete_path_execution_provenance_seed42.json
results/TRIFUSION_RGBNT201_V12_COMPLETE_PATH_OOF_ROUTER_2026-09-02.md
EXPERIMENT_AUDIT_V12.md
EXPERIMENT_AUDIT_V12.json
```

## 24. V13 deployment-aligned actual-path Router 终态（2026-09-02）

V13 修复 V12 的监督/部署动作失配，但没有通过 Router 泛化门。它将每个
complete-path identity-OOF teacher 样本的 exact Signal baseline、九槽 residual
和 actual-path query-side counterfactual utility，与同一 sample key 的 frozen
all-fit Phase-A deployment `direct_modal/modal_residual` 配对。Q0、Q1 replay 和
未来 dev 均调用同一融合函数：

```text
F(x,w) = L2([z0, 0.2 * ||z0|| * L2(vec(w_e,m * residual_e,m))])
```

Router 不再包含 learned alpha，只输出层级
`P(modality|x) * P(expert|modality,x)`；alpha 固定0.2。公共接缝完成真实
RED→GREEN，远端 commit `46b3e993b732c3afee63af9a56c75a62b3dbae21`
通过19/19 V13及相邻V8/V12测试。

### 24.1 P1 与 Q0

真实 fold0 八样本 preflight 通过，Phase-A state SHA 前后同为
`ecfd7fbc...fb77`，耗时13.91s，峰值 reserved1440MiB，dev0/official0。

Q0 在571个fit-only eligible query上得到：

| Q0 quantity | Value |
|---|---:|
| CNN/Transformer/Mamba unique positive wins | 218 / 196 / 157 |
| RGB/NI/TI unique positive wins | 241 / 109 / 221 |
| Oracle mean utility | 0.0020423282 |
| Best fixed mean utility | 0.0005741757 |
| Oracle-minus-fixed | +0.0014681525 |
| Read-only action-transfer aggregate gain | +0.0008705698 |

action transfer 三折均不劣；target health、reference immutability、专家/模态
diversity、Oracle gain、access boundary 全部通过。Q0 无训练、dev0、official0。
paired target cache SHA 为
`1cc499a1acb7b12336f19de0e74ad4ef452dae8b2aa8299e4a16e2d619e15e27`。

### 24.2 Q1 失败

Q1 使用完全冻结的 all-fit deployment features，三折各100 epoch，共300 Router
optimizer steps。Phase-A state 未改变；耗时34.26s；峰值
allocated/reserved=`2459.15/3400 MiB`；dev0、official0。

| Fold | Utility gain | Top1 gain | Replay AP gain | Replay margin gain | Result |
|---:|---:|---:|---:|---:|---|
| 0 | +0.0000726 | -0.0210526 | +0.0041573 | +0.0005697 | Top1 FAIL |
| 1 | +0.0004395 | +0.0111732 | +0.0011984 | +0.0056176 | PASS |
| 2 | -0.0003723 | +0.0742574 | -0.0039748 | -0.0023345 | utility/AP/margin FAIL |

聚合点估计均略正，但21个身份簇、10,000次 bootstrap 的95%下界全部为负：

```text
expected utility  -0.0004691
Top1              -0.0396049
replay AP         -0.0081192
replay margin     -0.0028545
```

质量门独立通过：受损 RGB/NI/TI 的自身质量从
`0.338163/0.331485/0.330352` 降到 `0.111049/0.119141/0.115227`；missing mass
严格为0。但身份策略和 replay 硬门失败，所以
`next_phase_authorized=false`、`final_training=null`、
`combined_checkpoint=null`。

### 24.3 Claim、审计与后续边界

独立 result-to-claim=`no/high`。完整性审计=`WARN/warn`：Q0 proxy 与 Q1
real-GT replay 分类正确，普通L2评价、实际执行路径、scope/leakage均PASS；WARN
仅来自远端大cache/checkpoint以SHA而非本体进入轻量Git仓库，以及审计时tracker
尚未更新。未发现假GT、自归一化、dev/official泄漏或隐藏final refit。

V13 终态为 `Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`。不运行final refit、dev、
official、消融、多seed，也不扫描fold/epoch/LR/temperature/门槛。当前可部署
最好仍为V8 Phase-B `58.4050 mAP / 59.3939 Rank-1`，距65 mAP为6.5950。
任何后继必须是新的预注册train-only policy-generalization主假设，并先通过
identity-disjoint Q0/Q1式门禁。

证据：

```text
evidence/trifusion_v13_deployment_aligned_preflight_seed42.json
evidence/trifusion_v13_deployment_aligned_q0_seed42.json
evidence/trifusion_v13_deployment_aligned_router_q1_seed42.json
results/TRIFUSION_RGBNT201_V13_DEPLOYMENT_ALIGNED_ROUTER_2026-09-02.md
EXPERIMENT_AUDIT_V13.md
EXPERIMENT_AUDIT_V13.json
RESULT_TO_CLAIM_V13.md
RESULT_TO_CLAIM_V13.json
```

## 25. V14 fold-robust retrieval-regret Router 终态（2026-09-02）

V14 只替换 Router identity objective：删除 V13 近均匀的 utility-KL，在每个
identity-OOF teacher 坐标系内以 cross-camera hardest-positive / nearest-negative
softplus risk 训练，并由两个 source folds 中相对固定策略 regret 最大的一折
控制更新。专家、all-fit deployment input、fixed alpha0.2、quality loss、seed42
和100 epochs/fold全部冻结。source-only minimax fixed slot 不读取 heldout fold；
任何 feature distance 都不跨 OOF generator。

Q0 在 exact paired cache 上 PASS：fold queries=`190/179/202`、identity=`7/7/7`，
minimax fixed slot=2、worst risk=`0.7034838`；14/14 Router 参数张量梯度有限且
非零，optimizer0、dev0、official0、cross-fold distance0、Phase-A SHA前后均为
`ecfd7fbc...fb77`。耗时9.30s，peak reserved636MiB。

唯一 Q1 共300 Router steps，37.79s，peak allocated/reserved=
`2459.15/3400MiB`。结果：

| Held-out fold | Risk gain | AP gain | Margin gain | Result |
|---:|---:|---:|---:|---|
| 0 | +0.0003567 | -0.0005571 | +0.0004422 | AP FAIL |
| 1 | +0.0045235 | +0.0049162 | +0.0091674 | PASS |
| 2 | -0.0016102 | +0.0001532 | -0.0033642 | risk/margin FAIL |

21个identity clusters、10,000次bootstrap的95%下界为risk `-0.0018584`、
AP `-0.0054337`、margin `-0.0039411`，全部失败。质量门通过：clean→corrupt
RGB `0.335696→0.108547`、NI `0.332487→0.117822`、TI
`0.331818→0.116403`；missing mass=0，Phase-A SHA不变，dev0/official0。

因此 `router_oof.gate.passed=false`、`next_phase_authorized=false`、
`final_training=null`、`combined_checkpoint=null`。JSON 的 `status=PASS`只表示
runner执行完成，不代表科学门通过。没有final refit、checkpoint或dev。

独立result-to-claim=`no/high`；integrity=`WARN/warn`，WARN仅因审计时tracker
未更新及execution-PASS语义可能误读，GT、普通L2/risk、实际路径、scope/
leakage和评价分类均PASS。V14封存：不扫描LR/epoch/temperature/loss/fold/
margin/threshold，不做refit、dev、official、消融或多seed。

当前可部署最好仍为V8 Phase-B `58.4050 mAP / 59.3939 Rank-1`，距65 mAP为
6.5950。V14增加的证据是：即便把训练目标直接对齐到fold-local检索几何，
all-fit sample-local Router输入仍不能可靠预测heldout relational utility；后继
必须是新的结构假设，而不是继续调Router loss。

证据：

```text
evidence/trifusion_v14_q0_seed42.json
evidence/trifusion_v14_q1_seed42.json
results/TRIFUSION_RGBNT201_V14_FOLD_ROBUST_ROUTER_2026-09-02.md
EXPERIMENT_AUDIT_V14.md
EXPERIMENT_AUDIT_V14.json
RESULT_TO_CLAIM_V14.md
RESULT_TO_CLAIM_V14.json
```

## 26. V15 Counterfactual Role-Delta Exchange 终态（2026-09-02）

V15 将协作移入冻结 CLIP tail 内部：CNN/Transformer/Mamba 在 tail9 和
tail10 后只交换各自相对输入的 role-delta，再由后续冻结预训练 block 解释；
两级同步无 self-edge，六条有向边以 `0.25*tanh(theta)` 控制且 theta=0 起步。
训练使用同一 tensor 的 exchange-on 与 state-clean no-exchange off comparator，
总损失为 V8 on-path 监督加 matched retrieval regret，权重固定1.0。

M0 在 clean commit `1f2de44f...` 有效 PASS：两 exchange stage 在 B64/K8
8-step 均 live，0 overflow，peak reserved9798MiB；100-step 达到110/110梯度，
loss `4.095560→1.209675`。扣除 label floor `0.578383` 与 matched-regret
floor `0.474426` 后，excess ratio=`0.051554<=0.1`。dev0/official0。

唯一 seed42 Q1 在 clean commit `71152d3848c05177da0af30b0b921c6a3aa9942a`
完成三折各20 epoch、final-only，共1,669 optimizer steps，0 overflow；每折
110/110梯度且 frozen state SHA 不变。结果为：

| Fold | fused gain | CNN gain | Transformer gain | Mamba gain |
|---:|---:|---:|---:|---:|
| 0 | +0.0952 | -0.0258 | -0.3291 | +0.9375 |
| 1 | -0.8311 | -0.0836 | -0.6967 | -0.8020 |
| 2 | +0.1605 | -0.3470 | +0.1904 | +0.6480 |
| aggregate | -0.1721 | -0.1576 | -0.2606 | +0.2898 |

fused bootstrap 95% lower bound=`-0.9503 mAP`。每折 fused 非劣、aggregate
+1 mAP、bootstrap>0、三receiver aggregate>0、每折两个receiver>0等五项
硬门失败。`status=PASS`只表示执行完成；科学`gate.passed=false`、
`next_phase_authorized=false`、`d1_executed=false`。因此没有 all-fit D1，
没有新的 30-dev 指标；当前可部署最好仍为 V8 Phase-B
`58.4050 mAP / 59.3939 Rank-1`，距65为6.5950。

独立 result-to-claim=`no/high`。V15只支持 CRDE 工程可训练、协议干净以及
Mamba 局部受益，不支持稳定三分支协同。V15已封存：不做D1/dev/official、
消融、多seed、checkpoint selection或LR/epoch/regret/edge-scale扫描。任何后继
必须是新的预注册主假设。

证据：

```text
evidence/trifusion_v15_m0_seed42_1f2de44.json
evidence/trifusion_v15_q1_seed42_71152d3.json
results/TRIFUSION_RGBNT201_V15_CRDE_Q1_2026-09-02.md
EXPERIMENT_AUDIT_V15_M0.md
EXPERIMENT_AUDIT_V15_Q1.md
RESULT_TO_CLAIM_V15.md
```

### 26.1 V15 只读交换后验

三折 final checkpoint 在原 identity-OOF heldout train records 上做一次 matched
on/off 重放；optimizer0、training=false、dev0、official0，三fold frozen SHA
不变。12个 stage×有向边中10个跨fold符号一致度仅1/3。虽然edge scale绝对值
只有约0.0002–0.0164，Transformer实际收到的incoming/own-delta能量比达到
0.291–0.428，CNN为0.024–0.251，Mamba为0.132–0.202；注入与自身角色增量
余弦绝大多数接近0。

571 query 汇总改善/伤害/不变：fused=`87/141/343`，CNN=`113/136/322`，
Transformer=`107/143/321`，Mamba=`153/89/329`。fold1 fused只有5个改善、
30个伤害且表示位移最大。该证据说明V15并非单纯scale太小：静态向量注入的
方向跨身份不稳定，增加能量反而可能放大伤害。

后继不得调V15 scale/edge/regret/epoch/checkpoint。新的预注册假设应把协作从
推理期隐藏向量注入转为训练期选择性检索关系互教，只让两个peer一致且优于
exact Signal anchor的关系修正落后expert，并在推理保留三支私有表示。

证据：

```text
evidence/trifusion_v15_crde_postmortem_seed42_27f9a6a.json
results/TRIFUSION_RGBNT201_V15_CRDE_POSTMORTEM_2026-09-02.md
```

## 27. V16 Signal-Anchored Triadic Repair M0 终态（2026-09-02）

V16 不再做 Router 或推理期 hidden exchange，而是在训练期用 exact Signal
选择共同 hard positive/nearest negative；只有另外两支都以至少0.05 margin
超过 Signal 和 receiver 时，才单向修复落后 receiver。另以
`gamma=0.30, epsilon=0.02` 保护高可信 Signal relation。推理结构完全复用 V8
固定 residual bank，新增推理模块/参数为0。

公共 hard-pair、two-peer detach、receiver-only gradient、criterion、builder、
M0/Q1/D1 gate 测试已完成，V8/V15/V16 相邻回归为23/23 PASS。远端 seed42 M0
也证明工程路径可训练：真实B64/K8 capacity 203/203张量有非零有限梯度，0
overflow，peak allocated/reserved=`5715.68/5962 MiB`；100-step fixed batch
loss `0.622885→0.581252`，floor-aware excess ratio=`0.064479<=0.10`；exact
Signal prefix 与 frozen state 均PASS。3090显存不是限制。

但 M0 最终为 `FAIL`：clean runner 的三折 CNN/T/M fixed-initial coverage 为
`3.125/0/3.125%`、`0/0/7.8125%`、`2.778/0/11.111%`。Transformer 三折均为0，
fold1 CNN也为0，违反预注册每 receiver `[0.5%,25%]`。SATR/no-SATR 两端的
初态、trainable names、seed、sampler indices 和前8个增强后 RGB/NI/TI tensor
SHA 全部相等；失败来自 proposal-time threshold probe 没有绑定原始 batch SHA，
其 margin/coverage 无法由正式 runner 重现。

因此 V16 在 M0 后封存：Q1、D1、dev、official 均未执行，不调 relation gap、
worker/RNG、epoch/LR 或 loss weight。当前没有新 retrieval 指标，可部署最好仍是
V8 Phase-B `58.4050 mAP / 59.3939 Rank-1`，距65 mAP为6.5950。

证据：

```text
evidence/trifusion_v16_satr_m0_seed42_20260902.json
results/TRIFUSION_RGBNT201_V16_SATR_M0_2026-09-02.md
refine-logs/v16/threshold-freeze-readonly.md
```


## 28. V17完整训练终态与全gallery核查（2026-09-05接续）

接续时服务器无训练进程、GPU空闲；原M0和Q1已经完整结束，但交接文档只写到
V16。此次以实际run_summary和保存权重为准补齐，未重复训练。

### 28.1 当前架构与原实验

V17冻结完整Signal的3072D direct+SIM（含camera SIE）与V8三个pretrained-tail
专家：CNN局部细节、Transformer全局CLS、Mamba空间/跨模态序列。共享CLIP在
block8后分支，并复用冻结tail9/10/11；不是三套独立backbone。每expert产生
三模态1536D残差。新增唯一TriadicCorrection将残差映射至256D，共享MLP读取
receiver自身、两个peer的Hadamard乘积及三支均值，三个独立output projections
给出修正残差。fused=3072D exact Signal前缀+等能量4608D残差银行；分支为4608D。
无Router、reranking、sample alpha、测试时训练或runtime fallback。

训练目标是普通ID/triplet加source-only同身份最大cosine/异身份最小cosine的
one-sided关系包络以及既定Signal保护；weight0仅把包络系数置0，其他完全相同。
该训练teacher是模型关系代理，不是性能GT；mAP/CMC使用数据集身份和camera。

M0通过：真实B64/K8、22/22梯度、overflow0、capacity peak reserved1808MiB，
100-step excess ratio0.000693508。Q1完整完成三折每端20 epoch，共3360步，
22/22梯度、overflow0、冻结状态不变，耗时35.23分钟，peak reserved5656MiB。
执行来源commit为`535ef2f305668493c0d07095ab17bb66e9997db6`。

原Q1的fused三折增益为-0.600212/-0.302472/-0.124641，aggregate -0.338635 mAP，
bootstrap95%下界-1.120318。科学gate=false，D1未授权/未运行，dev0/official0。

### 28.2 全gallery补评，不再删除无跨摄像头正例身份的干扰样本

发现原Q1先筛选跨摄像头身份，再把同一列表同时作query/gallery；每fold的47个
留出身份中只有7个进入gallery。原协议的相对比较一致，但不覆盖全部留出gallery。

新只读脚本`tools/audit_v17_full_gallery.py`严格重载六个最终checkpoint，复核
整模型final-state SHA、源文件/Signal/expert权重SHA、exact Signal前缀，遍历
三折gallery 1000/1051/1075条，共3126条、141身份。query仍为全部合格的571条；
其余2555条因无同身份跨摄像头正例不进入query分母，但全部保留为gallery干扰项。

注意：这里的固定141-fit来自`train_171`协议文件，共3126条；它不是数据目录
自带`train_141`的3280条。固定30-dev为825条，合计3951条。

| 输出 | matched weight0 mAP / R1 | DTRED mAP / R1 | mAP差值 |
|---|---:|---:|---:|
| exact Signal-only | 77.487603 / 79.334501 | 77.487603 / 79.334501 | 0 |
| fused | 80.614939 / 84.063047 | 80.286024 / 83.537653 | -0.328915 |
| CNN | 79.952578 / 82.837128 | 78.433784 / 81.436077 | -1.518794 |
| Transformer | 78.123409 / 82.311734 | 78.853236 / 82.837128 | +0.729826 |
| Mamba | 78.858043 / 82.311734 | 79.064729 / 81.961471 | +0.206686 |

全gallery fused三折Δ为+0.129220/-0.906481/-0.248032，aggregate bootstrap95%
下界-1.124616；fused AP改善/受损/不变=165/184/222，Rank1修复/破坏=2/5。
CNN AP改善/受损/不变=138/238/195，Rank1修复/破坏=6/14。DTRED比Signal高
2.798421 mAP，但matched weight0高3.127336，所以不能归因关系包络有效。

原限制gallery的DTRED fused 88.364223降为全gallery80.286024，差8.078199。
两种协议的aggregate相对结论都为负，但fold0及Mamba符号发生变化，说明不能
混用两个gallery协议选择有利分支结论。这些数字均为fit内identity-OOF资格结果，
不是30-dev，更不是官方测试；不得与85.3官方目标混排。

首次补评在全部六端推理后因bootstrap数据类未转JSON失败；提交`0888f454`只改
序列化，随后相同权重/数据的完整只读重放成功，耗时127.24秒。两次均optimizer0、
checkpoint writes0、dev0/official0。所有六端strict reload及state/checkpoint
unchanged通过；原始失败日志保留，原Q1 gate不变，没有重训或D1。

### 28.3 诊断、后续计划与边界

训练最终epoch平均负关系violation降低37.26%，正关系violation却增加7.44%；
全gallery检索同时揭示CNN主要受损。这个证据支持包络训练未把负关系约束变为
稳定正负检索排序改善，但还不能把背景、分辨率、学习率或cosine几何冲突判为
已证实的主因。该loss统计是在线epoch均值，不是固定批次终态因果比较。

V17完整训练已封存，不运行D1/dev/official/消融/多seed，也不扫width/loss/LR/
epoch/checkpoint。当前可部署dev最好仍是V8 Phase-B58.4050/59.3939；65门未过。

下一步工作的具体顺序：

1. 后继协议必须固定完整gallery，并报告所有合法query、排除原因、五路输出及
   mAP/Rank1/5/10；不再沿用仅跨摄像头身份组成gallery的旧资格协议。
2. 利用本次保存的全571-query AP/first-match rank，对三个fold和全部分支做
   固定错误普查，重点区分CNN正例排序受损与新增干扰样本导致的错误匹配。
   必须取得图像/token层面的证据后，才决定新的表征结构，不能凭此直接加掩码
   或换backbone，也不能继续扫描已封存Router/hidden exchange/cosine-envelope。
3. 新假设单独预注册固定seed42、完整训练终点及完整gallery评价，再开展一次
   主实验；先验证跨身份机制，随后才是固定30-dev和冻结官方协议。多次用同一
   21身份的结果不能称为新的独立验证。
4. RGBNT100/MSVR310本项目训练仍未开始；当前服务器只安装RGBNT201。跨数据集
   主训练/完整评估、最终官方比较与SOTA证明继续作为原总目标的未完成项。

原始回执、全部逐query结果与详细CMC：

```text
evidence/trifusion_v17_dtred_m0_seed42_535ef2f.json
evidence/trifusion_v17_dtred_q1_seed42_535ef2f.json
evidence/trifusion_v17_full_gallery_fixed_20260905.json
results/TRIFUSION_RGBNT201_V17_DTRED_2026-09-05.md
EXPERIMENT_AUDIT_V17.md
EXPERIMENT_AUDIT_V17.json
```

## 29. 各数据集公开高指标与代码资源增量核验（2026-09-05）

主源表及资源边界见`docs/SOTA_REFRESH_2026-09-05.md`。本次从RoDI作者GitHub
PDF第6页直接复核：RGBNT201 CLIP84.1/87.2、DINOv3 85.3/87.9；RGBNT100
DINOv3 89.0/99.1；MSVR310 DINOv3 71.8/84.8。PMKD的AAAI官方表为RGBNT201
84.7/88.9、RGBNT100 91.6/98.0，因此最高mAP和最高Rank1可能来自不同方法。
RoDI仍只有README/assets，Hyper-ReID仍只有占位README，不能当成可直接
复现的模块。Signal仍是有MIT许可证、可审计并已建立同协议dev底线的代码基座。
ProxyTTT必须标注测试时更新，PRISM的OpenPifPaf/SAM2掩码属于额外资源。

这些是已定位的公开报告参照，不是本机复现或绝对穷尽榜单。没有新的SOTA
声明，也没有因未获取代码而用猜测补齐指标。完整目标仍未达到。
