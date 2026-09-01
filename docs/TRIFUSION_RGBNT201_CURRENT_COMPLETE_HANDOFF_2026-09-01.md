# TriFusion RGB–NIR–TIR ReID 完整交接（2026-09-01）

## 0. 一页结论

本工程是在 DeMo 代码基座上实现的 RGB–NIR–TIR 多模态目标重识别研究分支。当前主方法不是三个互不相干的并行分支，而是：共享 CLIP ViT-B/16 语义主干为 CNN、Transformer、Mamba 三个完整异构专家提供强预训练表示，再通过分阶段双向异构特征交换和统一可靠性后验完成协同融合。

三个论文级主创新点已经落到代码、协议和测试中：

1. **HFER（Heterogeneous Full-Expert Relay）**：三种完整专家在中间层进行低秩、分阶段、双向特征交换，保留局部纹理、全局语义与长程扫描的异构归纳偏置。
2. **CIRC（Cross-fitted Interventional Reliability Calibration）**：使用身份不重叠的三折生成器，在实际模态退化干预下产生跨专家、跨模态共同尺度的可靠性监督。
3. **URGC（Unified Reliability-Guided Collaboration）**：同一可靠性后验控制中继、最终融合和退化条件下的贡献分配，避免每个模块各自学习一套互相矛盾的 gate。

当前最重要状态：

- 云端 RTX 3090 的正式 seed-42 主实验已完成 60 epoch 全 171 身份训练，并在固定终点完成唯一一次官方评估。
- 正式融合结果为 `59.1478 mAP / 63.2775 Rank-1`；CNN 略高，为 `59.1561 / 63.7560`。官方测试访问和评估计数均恰好为 1。
- 相对登记目标 `85.3 mAP / 87.9 Rank-1`，融合结果低 `26.1522 mAP / 24.6225 Rank-1`；`single_seed_target_exceeded=false`，不支持 SOTA 或融合增益主张。
- 后续 V3 task-anchor 与 V4 等能量残差银行均已在固定 141-fit/30-dev 上完整训练 60 epoch。V4 最佳 epoch27 fused 为 `43.4031/42.7879`，仍低于同 checkpoint 的 Mamba `44.0659/43.5152`，且距 65 mAP dev 门 `21.5969`；official access=0。
- V4 只保留了三模态 projected-CLS 的 1536D anchor，不等于 Signal 的完整 3072D 检索特征。Signal 还包含 1536D SIM 交互特征和 camera SIE；上游 `80.3/85.2` 尚未在本服务器复现，不能与 V4 held-out dev 数字直接相减。
- 原正式启动在官方指标写出后的路由校准审计因缺失导入失败；`repair-0002` 仅重算训练集路由审计，`optimizer_steps=0`、`training_reexecuted=false`、`official_test_reexecuted=false`，公开 verifier 返回 PASS。
- 用户最新指令：只做 seed 42；现在优先完成远端 Signal baseline 保底；主实验达到目标以后才考虑消融；所有训练、评估、数据和环境只在云端 GPU，Windows/WSL 仅作传输和文档存档。

## 1. 权威位置

### 1.1 云端工程

```text
Repository : /root/autodl-tmp/trifusion-v2/TriFusion-ReID
Branch     : research/trifusion-v2
Conda env  : /root/miniconda3/envs/tri_reid
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

## 4. 当前网络结构

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

## 9. 已知限制与下一步决策

### 9.1 必须保留的限制

- 当前只做一个 seed 42；不能据此给出多种子均值、方差或统计显著性。
- 没有复现 baseline；baseline 数字必须写成 upstream-reported。
- CIRC query/gallery symmetry 审计失败；禁止对称性主张。
- 正式官方 test 已恰好评估一次；不得再次访问本次 test 做选模、调参或重评。
- 在主结果超过冻结目标前，禁止启动消融实验。
- 本次 fused 未优于 CNN，不能把 HFER/CIRC/URGC 写成已获检索增益的实证结论。
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
正式/修复 verifier PASS（本次已完成）
  ├─ fused > 85.3 mAP 且 Rank-1 > 87.9
  │    └─ 才允许设计消融；仍只能称单种子目标超越，不能直接宣称统计 SOTA
  └─ 未超过目标（本次路径）
       └─ 不做消融或多种子；先建立完整 Signal baseline floor，再做 baseline-preserving 的 train/dev-only 主方法恢复
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
- [ ] 在远端建立完整 Signal baseline-only 路径、环境/权重或可复现训练回执及同协议 dev 指标。
- [ ] 建立同 checkpoint baseline-only/fused 双输出与 fused 晋级门禁后，才允许下一次主训练。

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

其 ViT 还在 CLS token 上加入 camera SIE。V4 只有 `ori` 语义的一部分，没有 SIM 和 SIE，因此不是 Signal baseline。上游发布 `80.3 mAP / 85.2 Rank-1` 来自官方 test 路径；当前服务器没有 Signal checkpoint，也没有单独 `signal` conda 环境，所以仍必须标为 upstream-only。

真正的“baseline 保底”不是简单拼接更多维度，而是以下可验证合同：

1. 同一模型/同一 checkpoint 同时输出 `baseline_only` 与 `fused`；
2. baseline 使用完整 Signal 3072D 路径，且能独立检索；
3. 先训练/验证 baseline，再冻结 baseline 路径训练专家增量，专家梯度不得破坏 baseline；
4. 同一 141/30 dev 协议上 fused 只有不低于 baseline 才能晋级；否则拒绝 fused，论文不得主张融合增益；当前不增加运行时 fallback 逻辑；
5. 只有 dev 主门通过后才全 171 固定训练并进行一次官方评估；仍不做多种子和消融。

### 12.4 当前 claim gate

独立 result-to-claim 纠正后结论：`claim_supported=no`。高置信度支持“V4 是稳定、完整但失败的 dev 结果”；中等置信度支持“缺少完整 baseline floor 是下一项结构性优先问题”。V4-specific independent integrity audit 尚未完成，因此 V4 完整性标签为 provisional，不能复用只审计旧 V1 的 `EXPERIMENT_AUDIT.json`。

---

本文件记录的是可核验工程状态，不是论文结论。任何后续结果都必须保留单种子、固定终点、官方 test 一次访问以及失败对称性审计这些边界。
