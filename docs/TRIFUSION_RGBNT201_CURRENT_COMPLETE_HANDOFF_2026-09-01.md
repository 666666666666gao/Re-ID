# TriFusion RGB–NIR–TIR ReID 完整交接（2026-09-01）

## 0. 一页结论

本工程是在 DeMo 代码基座上实现的 RGB–NIR–TIR 多模态目标重识别研究分支。V17完整训练和完整gallery补评已封存为失败；其全部查询/图像/分区诊断见§30。V18完整三折两端20epoch主实验已结束（§33）：fused增益+0.921504 mAP，bootstrap下界-0.117338，未通过固定晋级条件；无D1/dev/official。MSVR310、RGBNT100均已安装核验（§32）。V17 fused相对matched weight0为-0.328915 mAP，没有D1/dev/official结果。当前最高可部署结果仍是V8 Phase-B：冻结dev fused=`58.4050 mAP / 59.3939 Rank-1`，比exact Signal高`0.3941 mAP / 1.9394 Rank-1`并超过三个专家，但仍比65 mAP门低`6.5950`，不能声称SOTA。

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

### 28.4 全21身份错误普查已完成

对全部571条保存的逐query结果在远端汇总，覆盖全部21个合法身份；没有训练、
推理或checkpoint选择。fused AP总损失最大的身份是000261/000239/000251；
CNN为000217/000261/000220，其中000217的17条query平均AP下降12.418605pp，
新增4个Rank1错误。完整所有身份/所有分支表保存于
`evidence/trifusion_v17_error_census_20260905.json`，不是只汇报这些病例。
下一步数值普查已不需要重复；要核实固定病例的图像/token匹配原因，再决定
新表征假设，不能凭病例编号或这些已消费结果调阈值。

### 28.5 独立审计已完成

GPT-5.5 xhigh直接审阅源码和回执：M0 PASS、原Q1科学FAIL、integrity WARN；补评
为PASS_READONLY_EXECUTION_ONLY、FAIL_NO_ADVANCEMENT。逐query AP/rank重新汇总
与报告一致（mAP最大舍入差5.684e-14、rank rate差0）；六个checkpoint SHA及
模型state SHA与原Q1完全相等。其限制是大数据/权重仍在远端，独立审计者未另行
打开远端权重重哈希；由执行脚本强制检查及回执对齐作证，不应写成第三方权重
独立复现。完整报告`EXPERIMENT_AUDIT_V17.md/json`已归档。D1仍不授权。

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

## 30. V17全部查询、图像与分区诊断终态（2026-09-05）

固定范围文档`docs/V17_FAILURE_DIAGNOSIS_PROTOCOL_2026-09-05.md`先提交，再执行
`tools/diagnose_v17_failure_geometry.py`；源码8f31a4d，129.23秒完成全部六端。
3126 gallery/571 query、五路逐query AP/rank与原完整gallery回执精确相等；
checkpoint及整模型final-state SHA相等、推理前后状态不变。新采集correction
原始向量、teacher/corrected cosine、模态能量和CNN四水平分区3x4x768特征。

全部571条CNN query中，DTRED/weight0 correction范数均值0.288484/0.335074，
teacher cosine0.957932/0.943869。更接近冻结teacher没有改善检索：最近负例距离
变化-0.004551、最近正例变化+0.000103，margin变化-0.004655。14个Rank1新增
错误的最近负例距离平均缩短0.014158；fused新增5个Rank1错误，正例距离增加
0.010235、负例距离缩短0.006438。全部分支和全部query都保留，不只统计这些错误。

CNN模态能量均值变化很小：weight0 RGB/NI/TI=.3424/.3244/.3332，DTRED为
.3423/.3213/.3364，因此没有模态能量崩塌证据。CNN原始四分区的正例最佳对应
97.62%为同一区域，不能把区域错位判为主因。最近正/负例的分区cosine差在
RGB/TI均值为正，而NI四区均值为负；这是条件于冻结CNN所选困难负例的诊断，
不是丢弃NI的依据。

按预注册规则检查九张三模态图：每折CNN最差AP、最好AP、fused最差AP各一张；
无人工替换。病例有明显跨相机亮度/视角变化、模态遮挡，例如fold0负例000009的
NI被车遮挡；不能仅从九张图推断全数据集原因。CNN新增错误9/14的最近负例与
query同相机，fused为4/5；全CNN最近负例同相机比例62.17%→64.62%，仅支持
进一步检查视角/相机变化假设。

一次固定后续诊断比较了现有投影头和直接保留相同四分区原特征的检索几何，
没有训练或part/layer/rank搜索：冻结CNN分支79.319874mAP/82.311734R1，原始
四分区分支72.745261/73.380035，三折均更差。故不直接替换分区头。临时写入器
附加了字面量反斜线n，修正版只改JSON序列化；原文件和源码保留，未重新计算结果。

所有大特征和九张图保存在远端：
`artifacts/trifusion_v17_failure_geometry_8f31a4d/`。
本地诊断图片在`C:/Users/gb/.codex_tmp/trifusion_v17_geometry_20260905/`。
原始diagnostic JSON SHA为`ff1144c82436d5006ff324a6eebe7156debf12eb97dd04629523e08508852759`，
analysis JSON SHA为`111de9901870cd01f6e120de5567ec67d450283912e2c89e3361ec038926e4b0`。
这是已消费fit数据上的开发诊断，没有新的dev/official结果，不改变V17失败判定。

## 31. V18 PVNP主实验已启动（2026-09-05）

完整冻结计划：`docs/V18_PAIRED_VIEW_PROJECTION_PLAN_2026-09-05.md`，SHA
`bb7bb3ca6581e9d6d0bac1a3c0a83888fef77d8a0401650e97776f93e695e6d7`。
实现提交`2a71e20`，远端8项相关测试通过。北京时间10:34:57启动，screen
`9812.v18_pvnp_2a71e20`，日志`artifacts/trifusion_v18_pvnp_seed42_2a71e20.log`，
输出目录`artifacts/trifusion_v18_pvnp_seed42_2a71e20/`。

当前网络保留Signal完整3072D前缀和冻结三专家；新表征由各专家source训练身份的
同身份跨相机均值差分估计一个主方向，在共享低秩修正前后各去除该方向。
固定rank1，无额外训练参数、mask模型、rerank或TTT。方向拟合只用每折94个source
身份；不使用47个heldout身份。两端参数和增强配对，`uncentered`显式关闭投影，
`projected`开启；均使用ID/triplet和原Signal保护项，不使用DTRED envelope。

执行链：全三折source拟合→M0真实B64/K8容量8步和固定batch100步→M0通过后
完整三折×两端×20epoch。预计约40分钟，需以实际进程/日志为准，不能用该估计
宣称完成。仅最终checkpoint统一评价全部3126 gallery、571 query和五路CMC。
Q1需fused增益>=1mAP、各折非负、各专家aggregate非负、bootstrap下界>0且fused
超过baseline及三个分支；失败则封存，通过才执行一次all-fit完整主训练与30-dev。
本节启动时尚无V18训练完成或检索成绩。下次接续先检查同一screen/PID和日志，
不得因为文档仍显示RUNNING而重复启动，也不得因观察超时重训。

### 31.1 M0已通过，完整Q1执行中

三折source拟合均完成，各14个身份相机配对。每专家第一方向解释的差分能量：

| fold | CNN | Transformer | Mamba |
|---:|---:|---:|---:|
| 0 | 17.0265% | 13.3208% | 19.3079% |
| 1 | 16.6534% | 15.1640% | 17.4071% |
| 2 | 17.4534% | 16.2247% | 21.5256% |

M0 PASS：三折两端初始全state/前8批增强配对、exact Signal前缀和source状态
检查通过；真实B64/K8容量8步及固定batch100步均22/22梯度、overflow0、冻结
模型/方向buffer不变，峰值reserved1810MiB，floor-aware excess ratio
0.000693512976。该训练内工程门不证明检索提升。

正式Q1已开始；北京时间10:43观测第一折uncentered训练到16/20epoch，仍按
计划跑完所有六端后统一判定，不因首折结果改变epoch/rank/数据范围。
M0独立终态快照`evidence/trifusion_v18_m0_seed42_2a71e20.json`，源运行summary
快照SHA`4fbf61e540e9a083ea966c5715ce9d23a1dd3ffe41fa69c379c25c5f0db5da8d`。

### 31.2 已完成首个对照端的重训差异核查

首折uncentered已完成20epoch/580步，checkpoint严格重载成功。它与历史V17
weight0的全gallery融合mAP为69.256502/68.912955（+0.343548pp），逐query AP
并不完全相等；冻结Signal五项指标完全相等，epochs/steps/sample_order及前8批
增强回执相等。首epoch loss均值已出现0.000246差异，不能宣称跨次重训位级复现。
公共`_set_seed`同时设定cudnn.deterministic=True及benchmark=True；benchmark
允许不同运行选择不同算法，是数值差异的可能来源，但本次没有证明唯一原因。
当前不改公共种子函数、不重训首端，不把历史对照替代新成对对照。完整六端和
预注册晋级条件保持不变；该首端数字只作重训范围审计，不提前判定V18有效。

## 32. 跨数据集数据来源与传输（2026-09-05）

从[ICPL-ReID作者仓库](https://github.com/lsh-ahu/ICPL-ReID)核到公开数据链接：

- MSVR310：Google Drive文件`1IxI-fGiluPO_Ies6YjDHeTEuVYhFdYwD`，
  `MSVR310.zip`，HEAD核得491186967字节。
- RGBNT100：Google Drive文件`1R4XtbfnwTYyTvaTwrEx-pRCK2tApDWjj`，
  `RGBNT100.zip`，HEAD核得1584573535字节。

服务器直连Drive实测连接超时；Windows网络HEAD均200且可匿名访问。采用
Windows内存流→SSH/SFTP传到`/root/autodl-tmp/trifusion-v2/downloads/`，没有
在Windows保存数据文件。传输进程为本轮本地exec session97866，按MSVR310、
RGBNT100顺序进行；10:46远端MSVR310.zip已有175865856字节，此时尚未传完。
后续先检查原会话和文件，不重复下载；传完须核对流SHA与远端文件SHA、查看
zip目录与解压空间，再落到data目录并核实配对结构。当前不能宣称两个数据集
已安装或训练。数据传输不等于官方检索评价，未新增官方测试指标访问。

### 32.1 MSVR310已安装，RGBNT100传输继续

MSVR310内存中转603.10秒完成，491186967字节，流SHA与远端完整文件SHA一致：
`c6b15d61fdee6c34e6d25e5acbf103586dd46ee76138874652d92c4404f3359f`。
ZIP CRC全部通过，18970个archive entry、解压512942463字节，已原样解压到
`/root/autodl-tmp/trifusion-v2/data/MSVR310`，未更改原始图像。

按现有loader使用的目录逐身份核对vis/ni/th文件名集合完全一致：

| 目录 | 身份目录数 | 完整三模态triplet数 |
|---|---:|---:|
| bounding_box_train | 155 | 1032 |
| query3 | 52 | 591 |
| bounding_box_test | 155 | 1055 |

安装回执`evidence/msvr310_dataset_install_20260905.json`。仅文件与配对结构校验，
训练0、检索评估0。RGBNT100由同一session97866继续中转，预计约30余分钟，
完成回执持续写入本地`.codex_tmp/trifusion_cross_dataset_transfer_20260905.json`；
必须先确认该数据集的完成记录再解压，不以文件存在或旧进度猜测完成。

### 32.2 RGBNT100也已完整安装并通过结构核验

同一内存中转会话完成RGBNT100，耗时1742.49秒、1584573535字节，
SHA256 `9fecdf2978cade2a3d165fc3f63e1d0b8aa3283e31aebaacd0f755187b219a30`。
流SHA与远端完整文件SHA相等，71023个ZIP entry的CRC均通过，原样解压
1603150791字节到`/root/autodl-tmp/trifusion-v2/data/RGBNT100`。

R/N/T原始文件集合逐相对路径相等，共100身份、17250 triplets。现有loader
`data/datasets/RGBNT100.py`使用rgbir拼接图；全部拼接图头尺寸均为768×128，
对应三个256×128模态切片，未重编码或重排原始图片。

| loader目录 | 身份数 | 三模态拼接样本数 |
|---|---:|---:|
| bounding_box_train | 50 | 8675 |
| query | 50 | 1715 |
| bounding_box_test | 50 | 8575 |

训练/测试身份不相交，query身份均在gallery，均覆盖8个camera。
回执`evidence/rgbnt100_dataset_install_20260905.json`；安装检查耗时23.02秒。
压缩包readme声明研究使用及不得再分发；GitHub只提交安装元数据，没有数据文件。
本次MSVR310、RGBNT100训练和检索评估都仍为0，数据安装不能当成跨数据集结果。

## 33. V18完整主实验终态（2026-09-05）

原screen `9812.v18_pvnp_2a71e20`自然结束，M0加完整Q1耗时2477.35秒。
三折×两端×20epoch全部完成，共3360 optimizer steps：每端分别为
fold0 580、fold1 560、fold2 540；0 overflow、无中间检索选择。训练代码commit
`2a71e20`不变。新终态原始汇总9600061字节，SHA256
`8c5f99fcd4ba218ac2925a01123e377415c8443b7ed89de9ec0da5f400415f20`。

### 33.1 全部输出与固定晋级判定

下表是141-fit内部完整路径OOF，不是30-dev/official指标。每折全47个heldout
身份保留在gallery，合计3126条；全部571个合法query参与；2555条只因无
跨camera正例而排除query分母，仍作为gallery干扰。完整30个fold×端×输出
行及全部Rank5/10见结果报告，不择优展示。

| 输出 | uncentered mAP/R1 | projected mAP/R1 | mAP增益 |
|---|---:|---:|---:|
| exact Signal | 77.487603/79.334501 | 77.487603/79.334501 | 0 |
| fused | 80.560497/83.712785 | 81.482001/84.938704 | +0.921504 |
| CNN | 79.298869/81.961471 | 79.548593/82.486865 | +0.249724 |
| Transformer | 78.513897/81.961471 | 79.417463/83.187391 | +0.903566 |
| Mamba | 78.865192/82.837128 | 80.702741/83.362522 | +1.837550 |

三折fused mAP增益为`+1.755786/+0.110911/+0.855081`。三个专家aggregate
非负，projected fused也优于同checkpoint Signal和全部专家；这三项通过。
然而总增益0.921504低于冻结+1.0门，21身份聚类bootstrap10000次、seed42的
95%下界为`-0.117338 mAP`，不满足下界>0。因此最终 **Q1_FAIL**、
`next_phase_qualified=false`、`d1_executed=false`、dev/official访问0。
不放宽门、不改秩/方向估计/epoch/LR后重跑V18，不进入D1。

### 33.2 全部查询变化与复现边界

| 输出 | AP改善/下降/相等 | Rank1修复/新增错误 |
|---|---:|---:|
| Signal | 0/0/571 | 0/0 |
| fused | 224/132/215 | 10/3 |
| CNN | 203/181/187 | 12/9 |
| Transformer | 223/146/202 | 11/4 |
| Mamba | 247/131/193 | 13/10 |

全部21身份的分支与fused增益已逐身份列在结果报告和派生JSON。
fused为15身份改善、6身份下降；000235、000201两身份贡献约85.4459%的
查询加权净增益。这解释了平均收益与身份泛化不稳定并存，不能据此声称已找到
唯一图像因果因素，也没有删去负收益身份再计算晋级指标。

六个终态均重新构建/strict reload，模型state SHA与实际最终训练state完全一致。
训练后远端再次逐文件SHA核对6个checkpoint、6个冻结来源、3个source cache及
8个汇总/方案/代码文件，共23个文件；6份endpoint JSON与完整汇总对象相等。
绑定核对见`evidence/trifusion_v18_postrun_bindings_20260905.json`，未新建optimizer。

两端Signal逐query对象精确相等、样本顺序hash和前8增强batch回执相等。
历史V17 weight0与本次uncentered不是位级重训一致，见§31.2；不拿历史对照
替换本次对照。嵌套build_provenance沿用V17默认描述，V18顶层与实际执行
明确`projection_enabled`及`envelope_enabled=false`，原始记录不改写。

当前可部署最好仍为V8 Phase-B的30-dev `58.4050/59.3939`，exact Signal
为`58.0109/57.4545`；65开发门和官方/跨数据集SOTA目标仍未达到。
独立终态审计已完成：完整性为`WARN/warn`，科学晋级为
`fail_no_advancement / Q1_FAIL`。详细核对范围与限制见§33.5。

证据：

- `evidence/trifusion_v18_q1_seed42_2a71e20.json`
- `evidence/trifusion_v18_complete_comparison_20260905.json`
- `results/TRIFUSION_RGBNT201_V18_PVNP_Q1_2026-09-05.md`
- `docs/V18_PAIRED_VIEW_PROJECTION_PLAN_2026-09-05.md`

### 33.3 六端完整缓存重放与投影几何诊断

新脚本`tools/diagnose_v18_projection_geometry.py`在commit `e5ae63d`运行，
复用已SHA绑定的V17完整gallery冻结Signal/teacher缓存，严格加载六份V18最终
correction state、原projection开关与方向。按原128 batch执行全部头部前向；
六端×五路的全部AP/rank数组及指标字典与本次Q1逐项精确相等，Signal前缀不变，
head state和checkpoint文件不变。耗时7.58秒，optimizer0、checkpoint writes0、
dev0、official0，不是新验证集或新训练结果。

| 输出 | 最近正例距离变化 | 最近负例距离变化 | 最近正负margin变化 | 最近负例同camera比例变化 |
|---|---:|---:|---:|---:|
| fused | -0.006439 | +0.002267 | +0.008706 | 63.5727% → 60.0701% |
| CNN | -0.009467 | -0.000633 | +0.008834 | 63.5727% → 60.2452% |
| Transformer | -0.002700 | +0.001041 | +0.003741 | 66.5499% → 63.2224% |
| Mamba | -0.007448 | +0.005348 | +0.012796 | 63.5727% → 58.3187% |

这些是所有571个query的均值。新增CNN Rank1错误的9个query中，最近负例距离
平均下降0.024874，而最近正例仅下降0.002015；fused新增3个错误中正例距离
上升0.007996、负例下降0.008232。条件组覆盖其全部新增错误，完整原始query行
仍在JSON中。投影方向在各fold/expert的heldout平均能量约0.9987%–1.6889%；
projected输出沿该方向的最大系数绝对值均<1e-6，说明投影确实生效。

可支持的诊断是：平均正负间距和同camera负例比例改善，但少量身份和困难负例
仍受损，特别是CNN负例分离没有像Mamba一样整体改善。不能由此断言camera
是唯一因果因素，或把单轴移除改成多轴/方向重估后继续试V18。
后续优化应把困难负例的身份区分和视角不变性共同作为新表征的要求；具体新
网络与主实验须依据此全量证据另行冻结，维持完全身份隔离的训练内比较。

原始诊断`evidence/trifusion_v18_projection_geometry_20260905.json`为8356224
字节，SHA256 `55865c3b10c55871f9ccda48e84f6872750b3d1fad649d7b0bfff60ce9f9ad4f`；
远端位于`/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v18_projection_geometry_e5ae63d/`。

### 33.4 下一项表征方向的设计状态

候选方向见`docs/V19_PRIVATE_SEMANTIC_TAIL_DESIGN_DRAFT_2026-09-05.md`：
保持exact Signal不变，给三个专家各自复制/训练现有CLIP索引9–11尾部，
使角色模块与其后的语义变换共同适应身份区分。该方向基于实际共享冻结尾部
代码约束与V18困难负例证据；尚不能声称已证明该约束是唯一原因。

远端读取实际fold0 Signal checkpoint的model_state_dict核得每尾部block为
7087872参数、12个tensor，三个expert的9个block合计63790848额外参数。
它不需要新backbone下载，但容量变化显著，须在对照中披露并实测B64/K8。
目前只有设计草案，尚无V19实现、M0或训练；预训练尾部学习率、optimizer分组、
参数梯度合同及显存必须在实现核验后、任何检索结果之前正式冻结。
后续按这一顺序继续执行，不重训V18，不先访问dev/official。

### 33.5 独立终态审计完成

`EXPERIMENT_AUDIT_V18.md`及同名JSON由独立GPT-5.5 xhigh审计者直接读取
原始方案、源码和全量证据后生成。GT来源、指标归一化、实际调用路径、评价
范围和评价类型均PASS；完整性总判定WARN，科学判定仍为Q1_FAIL。

审计逐项复算全部六端五路的query指标、fold/aggregate指标和数据范围，最大
差异分别为7.11e-14和9.95e-14，仅属浮点舍入。三份校准和六份endpoint独立
JSON均与完整汇总内嵌对象一致；投影诊断的全部输出与Q1一致，轴系数最大
绝对值1.6764e-8，低于1e-6。未发现指标归一化、GT来源或选择性遗漏问题。

WARN保留两项实际限制：大权重和source cache只在远端，审计者未直接持有
其完整字节，远端23文件SHA回执不能替代独立读取；审计者本地缺NumPy，未
独立重放PCG随机抽样，因此不能称bootstrap随机过程已独立复现。审计检查了
其源码、保存结果与其余数值关系，但没有移除这项限制。最新SOTA网页不属于
该实验审计的独立核验范围。这些限制不改变固定两项科学条件失败的结论。

审计原文未由主代理改写。MD SHA256为
`79cc4fdb3cc0aaccf59e2c62ead1df7dadb7437702a21946fb6d70a59b957da3`；
JSON为`ca3d73f042166f26876c15debf1915eccbad0c13dcd9664738df48ae151b7af0`。

## 34. SOTA参照与开源资源再次增量核对

已从CVF主PDF核读CoT-ReID Table1/2：RGBNT201 83.3/86.1、
RGBNT100 89.9/99.3、MSVR310 71.7/85.3（mAP/R1）。它使用DINOv3和MLLM
推理文本，须与纯视觉静态方法分列。因而§29记录的RGBNT100 R1=99.1和
MSVR310 R1=84.8不再是本次已核文献的最高Rank1。CoT Table3另有MSVR
72.7/86.3，仍保留表间差异，主比较采用Table1而不择优拼接。

DSGM作者稿与开源仓库也已核到；其主表为RGBNT201 82.6/87.0、
RGBNT100 89.4/98.2、MSVR310 64.6/76.0，依赖GPT-4o文本和SAM2软mask。
PMKD仓库当前仅README；CoT有代码但不提供文本与预训练权重；DSGM有MIT
实现和附mask数据链接。本次没有下载或接入这些模块，没有改变V18冻结方案。
主源链接、CoT PDF SHA与CCL/Hyper-ReID待核边界统一见
`docs/SOTA_REFRESH_2026-09-05.md`，不称已穷尽最新SOTA。

## 35. V19私有语义尾部实现与预注册（2026-09-05）

§33.4的候选现已实现为`modeling/trifusion/signal_preserving_v19.py`，独立runner
为`tools/train_signal_preserving_v19.py`。原Signal和V8源码未修改；wrapper持有
原角色模块，并给三专家各自复制CLIP索引9/10/11，原完整3072D Signal冻结。
两端都持有九份相同初始副本；匹配对照冻结副本，实验端训练副本，两端共同
继续训练V12 source-only角色模块和head。可训练参数差63790848，108tensor，
容量差异必须披露，不把此比较称等可训练容量或角色分工的独立因果证明。

执行前固定方案在`refine-logs/v19/EXPERIMENT_PLAN.md`及时间戳副本；配置
`configs/RGBNT201/TriFusion-signal-preserving-v19-private-tail-rtx3090.yml`。
角色/head LR=3.5e-4，预训练尾部LR=3.5e-6，AdamW wd1e-4、5epoch warmup
+cosine、20epoch、B64/K8、seed42。损失沿用V8，ID权重和0.75，过拟合下界
为0.75H。V19自身关闭CuDNN benchmark以处理此前观察到的重训差异风险；
历史seed helper保持原样，不声称所有CUDA算子自动确定性。

M0要求三折两端原V8五路输出/初始state/增强配对精确一致，私有storage互不
共享；两端8步真实容量与实验端固定100步均需全训练梯度、overflow0、冻结
状态不变，过拟合excess ratio<=0.1。通过后自动完整三折×两端×20epoch，
保留3126gallery/571query五路输出，全部最终checkpoint重新strict reload。
Q1沿用+1mAP、全部fold/专家非负、身份bootstrap下界>0及fused严格胜出门。

上述实现于4b749cd提交并部署，执行前状态保留于该提交；实际运行见下节。

### 35.1 V19 T0/M0完成，完整Q1运行中（2026-09-05 12:26 CST）

实际源提交`4b749cd92735c228a4bdb1cfacb0b2c6cb80cfe9`，配置SHA
`89f7335a0d0995aa23f1a3387e76b4693ecb3721c865e706feaf1b059fd97dd5`，
冻结方案SHA`2b7674cf395ba2c53a7b9fd695b76e99eeecd0445957f188e22c5700bdaafe7b`。
远端T0四项定向测试通过（6.95秒）；12:15:41 CST在screen
`18809.v19_private_tail_4b749cd`启动，GPU启动前24126MiB空闲。
运行目录为`/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v19_private_tail_seed42_4b749cd`，
日志为同路径后接`.log`。禁止重复启动同一实验。

三折初始检查均通过：原V8与两端五路输出精确一致，初始state和增强receipt
配对相等，九个副本storage独立。两端实例总参数同为162590989；对照训练
7841292参数/203tensor，实验端71632140/311，差额63790848/108。

两端8个真实训练batch中全部203/311张量均收到有限非零梯度，overflow0，
峰值reserved分别6478/6814MiB；对照私有尾部不变，实验端尾部改变，全部冻结
state保持一致。新初始化实验端固定100步loss从0.6110473871到0.5803269148；
解析平滑下界0.5783829210，excess ratio=0.0595140216 <=0.1，完整梯度覆盖、
overflow0及冻结state检查全部通过。M0全部工程条件PASS，训练状态不带入Q1。

完整原始M0快照`evidence/trifusion_v19_m0_seed42_4b749cd.json`共257741字节，
SHA`89fd884a8c894de16639c26cfa162a2bbd005dc13c1e38ebe0fe6ce7adafe992`，
已与远端汇总SHA核对；启动/T0 receipt和当时完整日志同存evidence目录。
该快照状态RUNNING，只覆盖M0及preflight，不能充当Q1终态证据。

Q1已自动从源checkpoint重新构建；12:26:41时fold0对照第9/20epoch，
约35秒/epoch，GPU使用6672MiB、利用率100%。据实测将六端训练连同终态
重载/评价估计修正为75–90分钟。必须完成所有端后按原固定门判断，当前没有
完整检索结果。工程独立审计与Q1可并行，科学终态审计等待全部端结果。
两个跨数据集仅安装，不训练；D1/dev/official仍为0，全局目标未达。

### 35.2 文献与跨数据集协议准备（2026-09-05）

另核对NEXT当前arXiv v5、PDRNet完整模态主表、FUSE及DCG主表，详见
`docs/SOTA_REFRESH_2026-09-05.md`，其数值未超过前述已核主要高mAP参照。
NEXT仓库当前公开文本/assets而非完整训练实现；CCL/Hyper-ReID待核边界保留。

服务器上的车辆数据已补充全split标签清单检查：MSVR310全部591 query和
RGBNT100全部1715 query均有合法正例，train/test身份不相交。MSVR310的既有
协议按同身份且同scene/time过滤，516/591 query在改成同camera过滤时正例数
会改变；后续不得套用RGBNT201评价过滤。完整清单/逐query计数/脚本SHA见
`evidence/vehicle_query_protocol_labels_20260905.json`与
`docs/VEHICLE_EVAL_PROTOCOL_READINESS_2026-09-05.md`。
该检查仅读取文件名标签，optimizer0、图像/权重/特征加载0、检索评价0；
不改变V19冻结配置，也不构成跨数据集性能结果。

### 35.3 V19 M0独立审计完成（2026-09-05）

按experiment-audit的独立审计要求，GPT-5.5 xhigh直接读取25项原始路径及
调用依赖，原文报告为`EXPERIMENT_AUDIT_V19_M0.md/json`。工程完整性PASS，
overall/integrity为WARN；A/B/F通过，C/D/E保留证据持有与阶段范围限定。
审计员独立核对M0 JSON/plan/config/runner/module SHA、三折六端48组batch
receipt，并从完整loss数组重算excess ratio=0.0595140216437626。

WARN不改变M0工程通过，也不构成Q1科学通过：被审计的是M0后RUNNING且
folds为空的固定快照，尚无可审计的六端终态/检索/统计bootstrap。远端source
checkpoint、CLIP权重和数据图像未被该本地审计员直接读取，须保留receipt约束。
两份既有本地文件criterion.py与rgbnt201_dev_v1.json存在CRLF/LF原始字节差异，
其LF hash与远端source map一致；历史文件未改写，执行SHA以远端LF字节为准。

审计MD SHA`3d97b4aab6d3d5595fb86a78502cab3d3e08421692a17030a14122da48ffab34`，
JSON SHA`e65c23aef16a5ba2fdc630cd0df1c653d53fb0a354c052004591f20caa75fd23`。
完整两轮request/response/meta及原报告快照保存于本地忽略的
`.aris/traces/experiment-audit/2026-09-05_run03/`。第二轮仅补机器接口所需
integrity_status字段，结论和工程/科学区分不变。

Q1保持原进程和冻结预算；全部完成后独立审计全部端、五路AP/rank、图库范围、
strict reload与远端checkpoint终态SHA。大权重保持远端，本地同步完整JSON与
哈希回执，后续审计仍须如实标明独立持有范围。当前全局目标未达。

### 35.4 V19首折完整配对终态完成（2026-09-05 12:48 CST）

12:48:21远端观测：完整Q1仍RUNNING，fold0两端各20epoch/580优化步已结束；
两端均overflow0、冻结state不变、strict reload及只读评价通过。完整sample
order、前8增强batch、初始state和baseline-only输出对象两端逐项相等。
该折保留全部1000gallery记录/190合法query，独立endpoint receipt位于运行
目录`fold_0_frozen_private_tail_receipt.json`与
`fold_0_trained_private_tail_receipt.json`。这是2/6端的进度记录，不是整体科学结论。

当前进入fold1对照第2/20epoch，实测28batch/epoch、约34秒/epoch；同一screen
继续执行，无重新启动。六端终态预计约13:36–13:51 CST，随后核验全部3126
gallery/571query、五路数组、原固定科学门与源checkpoint终态SHA。
下一次常规进度观察窗口12:52–12:53；按180–300秒或预计完成节点检查。
本轮D1/dev/official访问仍为0，RGBNT100/MSVR310仍无训练和检索结果。

### 35.5 V19前两折完整配对完成，进入最后一折（2026-09-05 13:14 CST）

13:14:21远端实际进程18811与screen18809均存活，run_summary仍为RUNNING，
已有fold0/1的四个完整endpoint receipt。fold1两端各20epoch/560优化步，
overflow0、冻结state不变、strict reload与只读评价通过；完整sample order、
前8增强batch、初始state和baseline-only输出对象两端完全一致。
fold1保留全部1051gallery/179合法query；加上fold0为2051gallery/369query。

四端累计2280优化步，下一步继续完成fold2的两个固定20epoch端点，预期再
1080步，总计3360步。当前fold2对照第3/20epoch，27batch/epoch、约33秒/epoch。
剩余fold2必须保留全部1075gallery/202合法query；整体仍以3126/571为准。
以上仅为完成范围与工程进度，尚无全三折的科学门结论。

按当前速度，最后对照端预计13:24左右结束，随后实验端继续；完整Q1预计
13:36–13:51 CST。下一次有意义的阶段观察放在13:21–13:22，接近对照端
预计完成节点时再检查。源配置/方案/训练预算均未改变，没有重启或附加训练。
全部完成后保存完整终态、核验六端checkpoint及source SHA、全部五路数组与
身份bootstrap，并进行独立终态审计。D1/dev/official访问仍为0，全局目标未达。


## 36. V19完整六端终态：Q1_FAIL，未晋级（2026-09-05）

V19按冻结计划完整完成三折×两端×20epoch、3360优化步，seed42，总耗时
4839.900912秒。13:38:22 CST观察训练进程18811和screen18809结束、GPU空闲。
执行源仍为4b749cd92735c228a4bdb1cfacb0b2c6cb80cfe9；后续文档与只读核验工具
提交不改变实验源。以下终态取代§35的RUNNING进度，不修改历史原始快照。

全部五路、三折两端、21身份及AP/Rank查询变化见
`results/TRIFUSION_RGBNT201_V19_PRIVATE_TAIL_2026-09-05.md`。所有heldout身份与
3126 gallery记录均保留，571合法query来自21跨camera身份；2555记录仅不计入
query分母。仍是复用的real_gt训练内完整路径OOF资格，非dev/official/SOTA结果。

| 输出 | 冻结尾部mAP | 训练尾部mAP | 增益 | 冻结R1 | 训练R1 |
|---|---:|---:|---:|---:|---:|
| Signal baseline | 77.487603 | 77.487603 | 0.000000 | 79.334501 | 79.334501 |
| fused | 80.240792 | 80.496828 | +0.256035 | 83.187391 | 84.238179 |
| CNN | 79.915105 | 80.054797 | +0.139692 | 84.763573 | 84.238179 |
| Transformer | 78.150546 | 79.331729 | +1.181183 | 82.136602 | 83.187391 |
| Mamba | 77.801980 | 77.379156 | -0.422824 | 78.984238 | 79.509632 |

三折fused增益+1.619524/-0.900867/-0.001279，21身份10000次seed42聚类bootstrap
95%下界-1.615129。因此aggregate>=1、三折非负、三专家非负、下界>0四项均FAIL；
只有fused严格胜过同checkpoint baseline与三专家PASS。Q1_FAIL，next_phase_qualified
与d1_executed均false，dev_access_count和official_test_access_count均0。

全部六端overflow0、实际203/311训练tensor完整非零梯度、冻结state不变；两端
完整sample order/前8增强receipt/初始state与baseline输出精确匹配。各fold每端
580/560/540步，最终checkpoint重建strict reload通过。六个源checkpoint终态
SHA未变；远端再次逐字节核验29个源码/配置/计划/来源/CLIP/最终checkpoint文件。
六个独立endpoint JSON与汇总对象逐项相同。核验不加载权重张量或执行新检索。

完整汇总`evidence/trifusion_v19_q1_seed42_4b749cd.json`为1971525字节，SHA
`e0c9c2e0683c934fd65ae594186d89452c9786e203e1f4b1a9b7612505316d59`。
完整日志、六receipt及terminal_file_verification同在evidence；传回文件SHA均匹配。
`tools/audit_v19_terminal_arrays.py`重建全部query mask、五路指标和身份sum/count
bootstrap，最大数值差0.0，见terminal_array_audit回执。这是执行端数组/文件核验，
不冒称独立审计员已持有远端权重重新推理。完整Q1独立审计当前待完成，M0审计
已完成的PASS/WARN及范围限定保持原样。

融合共有180query改善、186下降、205相等，R1修复19/新增错误13；21身份为
9改善/11下降/1相等。新增63790848个可训练私有尾部参数在固定合同下没有给出
稳定身份泛化收益，不能证明共享尾部是唯一原因。V18与V19只各自比较实际匹配
对照，不能把跨版本均值差作单因素归因，也不拼接两个版本有利分支。

V19封存：不执行D1/dev/official，不放宽条件，不做层数/宽度/LR/epoch/seed扫描。
后续先分析全部终态中的可重复身份/分支错误，再形成不同的表征假设并提前固定
验证合同。当前可部署dev最好仍为V8 Phase-B58.4050/59.3939，exact Signal
58.0109/57.4545；dev65门与官方85.3/87.9目标未过，RGBNT100/MSVR310仍无训练
或检索成绩。全局目标保持active，不能因这次完整负结果归档而标记完成。


### 36.1 V19 Q1独立审计完成（2026-09-05）

GPT-5.5 xhigh读取38项路径及实际依赖，完成EXPERIMENT_AUDIT_V19_Q1.md/json。
工程完整性PASS，overall/integrity WARN，科学资格FAIL；实际使用NumPy2.5.2
独立重算三折两端五路AP/Rank、全部标签mask、配对变化和10000次身份bootstrap，
下界-1.6151285618296207与原始汇总精确一致。全部源训练预算/配对/冻结记录一致。

WARN保留远端大权重/数据的独立持有范围，以及criterion.py、rgbnt201_dev_v1.json
本地CRLF/远端LF字节差异；不能把审计算术通过写成Q1科学通过。MD SHA为
1e5602615a23609c0f36bfcebadc8e8e6cef9977a956e3900c0f84eb9b4674bf，JSON SHA为
d29f908d62d52c55e7e2f576297494da1cd4006de22cf6ecc16c94d4a6ba31c4。完整request/
response/meta及原报告归档于.aris/traces/experiment-audit/2026-09-05_run04。
tracker已更新DONE_WARN_Q1_FAIL，D1/dev/official未执行，V19封存不变。

### 36.2 全六端来源拟合与模态表征诊断完成（2026-09-05）

固定协议`docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md`和脚本先于
执行提交，source3edb0f9，14:08:51在screen24879启动，329.784541秒完成。
14:16:20观测进程已结束、GPU空闲。全部六个最终模型严格重建，18756次triplet
只读前向，模型state/权重文件不变；全部原五路heldout AP/Rank与Q1精确相等。
optimizer0、checkpoint writes0、dev0、official0。完整source/heldout结果与全部
3×3模态对表见`results/TRIFUSION_RGBNT201_V19_GENERALIZATION_DIAGNOSIS_2026-09-05.md`。

各fold94-source含14跨camera身份，47-heldout含7；来源的所有七个分类头准确率
和fused检索mAP均100%，并非训练身份拟合失败。对照来源三个专家同模态最近
cosine margin分别+.162041/+.191277/+.155974，heldout为-.052331/-.067698/
-.051115。跨模态margin在source/heldout及两个端均为负；完整统计覆盖所有
有向模态对，不挑选RGB/TI单一方向。这显示来源拟合与身份泛化有明显落差。

但原concat距离只比较对应模态；独立正交旋转能改变跨模态cosine而不改变该
部署距离。因此不能断言模态方向不一致是融合失败的唯一原因。后继可检验
“每专家内部的真实身份跨模态约束是否改善泛化”，不强迫不同专家相互对齐，
且必须另行预注册完整配对训练。尚未启动新训练，无新的开发/官方成绩。

原始诊断47990970字节，SHA0e40093688ed568b7e0584672e4a74098c5fba4e57df06fba4bab1b6405adbe6，
已与远端完整文件SHA核对；日志、启动、传输与全数组算术summary回执同存evidence。
Q1独立审计不覆盖这项后续诊断，其独立审计待进行。公开方法参考见
`docs/GENERALIZATION_MODULE_SOURCE_NOTES_2026-09-05.md`（SupContrast/UPCL/MixStyle），
无移植、下载模型或额外数据训练。全局目标仍active且未达。


### 36.3 V19全六端几何诊断独立审计完成（2026-09-05）

EXPERIMENT_AUDIT_V19_GEOMETRY.md/json已由GPT-5.5 xhigh完成；工程PASS、
overall/integrity WARN、scientific FAIL。独立JSON/NumPy复算覆盖全部60个fold
指标行、20个汇总行、42/14分类头、108有向模态对、24组几何和5路配对距离，
与汇总相同。唯一物理来源3126条、fold来源memberships6252条范围核对通过。
远端大权重/数据仍为receipt持有限制；不能据此改变Q1_FAIL或宣称因果/新验证。

MD SHA8406c80fe0a408a40f1bd8e067c984af88cc3e4a2f0476a77f009d6635bea27f；
JSON SHAb163e3349711cb2ec2df5f27b4eb71be79950d498a368b36617e7ca332155d97。
原文和完整request/response/meta归档在.aris/traces/experiment-audit/
2026-09-05_run05。V19全部主实验与后续诊断已归档，封存失败不再扫描。

## 37. V20每专家跨模态身份监督：执行前冻结（2026-09-05 14:44 CST）

新假设是每专家内部的真实身份跨模态监督能改善身份泛化；不是已证明的原因
修复。原concat检索的独立模态正交旋转不变性解释边界仍保留。参考SupContrast
作者监督对比目标，独立实现18项平均（三专家各六个有向不同模态对）；B64/K8
全部8个同身份目标均为正例，其他身份为负例，温度0.07，实验端权重0.25。

原V8完整Signal3072D及共享冻结CLIP尾部不变；CNN局部、Transformer全局、
Mamba跨模态空间角色与原五路输出保留，fused为3072+4608D。无新推理参数，
不继承V19私有尾部；没有跨专家强制对齐。两端identity_concat/cross_modal_identity
从同一V12 fold权重strict reload，唯一差别是新损失系数0/0.25，原ID/Triplet、
模型容量、初始化、20epoch预算、采样和增强相同，三折两端完整训练与评价。

方案refine-logs/v20/EXPERIMENT_PLAN.md与时间戳副本执行前冻结，
SHA28bfbe5dd324e2600bc4bea06d8bfe4c3b1730409d21409d97a981c2b8a86f8f；
配置configs/RGBNT201/TriFusion-signal-preserving-v20-cross-modal-identity-rtx3090.yml，
SHA87d5a53ceb88d2546b9edf62510c3a112d669c4b60293563aba0a7d78cc94026。
仅seed42远端GPU，B64/K8无梯度累积，20epoch/5warmup/AdamW3.5e-4。
新损失FP32，总熵下界0.75H+0.25log8；M0固定100步超额损失比<=0.1，
梯度全覆盖、overflow0、冻结state不变、两端8步容量门保持。

T0为三项远端CUDA数学测试；M0通过才自动执行三折两端20epoch，完整3126
gallery/571query/21身份、五路AP与Rank1/5/10、全部负收益和身份bootstrap；
固定五个Q1科学门仍须全部通过。失败即封存，不以调温度/权重/预算重试。
当前仅完成代码AST与预注册，尚未启动V20训练，T0/M0/Q1无结果；dev/official0。
下一步先发布并在远端验证T0，再按固定合同启动。训练前预计M0约3–8分钟，
Q1约60–80分钟，按实际时长修正。可部署dev最佳与未达65/SOTA状态均不变。


### 37.1 V20 T0/M0全部通过，Q1已启动（2026-09-05 14:54 CST）

执行source3cea5bfc17e214b1829c020527699d939efa221d。远端三项CUDA单测2.87秒
全部通过，14:48:31CST启动screen v20_cross_modal_3cea5bf，输出目录
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v20_cross_modal_identity_seed42_3cea5bf。
日志为同路径+.log。M0实际228.975805秒完成；14:52:27观测已进入Q1首端构建。

六模型初始state/五路输出/增强/源绑定精确配对，全部原Signal前缀精确。实际
两端均98,800,141总参数、7,841,292可训练参数、203训练tensor，推理新增0。
新损失单独梯度probe三折CNN/T/M非零encoder数量42/54/93、42/54/92、
42/54/91；每专家均可达，不声称该单batch所有tensor都非零。两端各8个不同
batch的正式AMP容量步全部203/203梯度覆盖，overflow0，冻结state保持；
两端峰值reserved均6062MiB。100固定batch步也通过全部梯度与冻结检查。

总损失1.886078119277954→1.100658655166626；基础ID/Triplet
0.6110473871231079→0.580313503742218，新损失5.100122928619385→
2.081380605697632。固定解析总下界1.098243306466169，超额损失比
0.0030658060054957735<=0.1，因此M0_PASS。全100步原始分量完整保留，
不能将M0过拟合通过写成泛化或Q1科学通过。

不可变M0 snapshot340010字节，SHA5fd4922a7a7036f6905c54397809faed18387666b1df18aa39e5429cd10876a0，
原日志snapshot26426字节，SHA5ae3ecf49e9de70caf0154060782fa0becf2caa95e9e7f4245011c23eef8a267。
本地/远端SHA逐项一致，原snapshot状态RUNNING、fold数组为空；T0/启动/传输
回执同存evidence/trifusion_v20_*。GPT-5.5 xhigh独立M0审计已提交25原始路径，
trace run06，尚未返回；不借用V19审计覆盖V20。

Q1必须继续原进程完成三折两端20epoch/3360步及全部五路终态评价，当前尚无
完整科学结果。初估60–80分钟，待稳定epoch时长修正；所有科学门/失败封存
策略不变。仅seed42，无本地模型执行、消融、扫描、D1/dev/official或新SOTA。


### 37.2 V20 M0独立审计完成（2026-09-05）

M0独立审计及字面字段/记录范围校对已完成：EXPERIMENT_AUDIT_V20_M0.md/json，
工程PASS，overall/integrity WARN，scientific_qualification not_evaluated。
熵下界与比例独立复算差0，损失分量最大舍入差5.96e-8。两轮原文及完整
request/response/meta保存于trace run06。最终MD SHA
28964c3a11d900db7300671fdd07772c4217710a751daef078c4df15e709de8a，
JSON SHAb3e717f453ee813ea45b4b1abdb6004ae29337b1a140a6b45c974b54ba06892a。
字面键名、3126源码断言/48正例记录推导、preflight无optimizer状态保存、
容量8不同batch/过拟合100固定batch等描述已精确区分，数值/门槛不变。
本地criterion.py及protocol只在LF规范化后匹配SHA，远端大权重/图像独立持有
范围有限；M0审计不覆盖后续Q1终态。不作无关换行符重构或运行中源码改动。


### 37.3 V20第一折完整配对终态（2026-09-05 15:16 CST）

15:16:12CST确认第一折两端各完成20epoch/580步，共1160步；strict reload完整
state SHA、只读评价、overflow0、冻结state、203训练tensor梯度覆盖及完整
采样序列/前8增强/初始state/baseline输出配对检查均通过。保留1000gallery/
190合法query。完整snapshot889149字节，SHA
1f71ee488494019937a2f8a9d76b7ec29a611ac10f4037ae6ac5ff8f42a0eb0c，
本地与远端一致；见evidence/trifusion_v20_first_paired_fold_20260905.json。

第一折fused相对实际对照-1.087608 mAP；固定“各折均非负”条件已有失败项。
CNN+0.418478、Transformer-4.332104、Mamba-0.534771。整体仍RUNNING，
按合同完成剩余4/6端，不能删分支/改温度权重/缩预算；所有五路Rank1/5/10、
逐query AP/rank及源/终态权重绑定完整保留。无D1/dev/official访问。
预计六端15:56–16:06完成，下一观测窗口15:33–15:34临近第二折配对完成。

| 第一折输出 | 对照mAP | 新损失mAP | mAP增益 | 对照R1 | 新损失R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 68.767642 | 68.767642 | +0.000000 | 69.473684 | 69.473684 |
| fused | 71.494649 | 70.407041 | -1.087608 | 71.578947 | 70.000000 |
| cnn | 71.243655 | 71.662133 | +0.418478 | 71.052632 | 73.684211 |
| transformer | 70.175142 | 65.843037 | -4.332104 | 71.578947 | 65.263158 |
| mamba | 69.964462 | 69.429691 | -0.534771 | 68.421053 | 71.578947 |

首次15:15:19观测见两个Q1_final日志行，summary尚未append该折；随后直接核对
已保存完整fold对象和配对条件后才记录该折完成。原始进度与传输回执同存
evidence/trifusion_v20_*。M0审计不代替完整Q1审计，旧V19失败封存和未达
dev65/官方SOTA状态不变。全局任务继续active。


### 37.4 V20完整六端终态及全数组重算（2026-09-05 16:12 CST）

三折两端各20epoch、120条epoch记录、3360优化步全部完成；运行4236.036166秒
（70.600603分钟，包含M0等整体流程）。原进程26383在16:08:40CST观测已退出，
GPU 1MiB/0%。固定Q1判定FAIL，D1拒绝晋级，dev/official/D1访问均为0。

完整141 heldout身份、3126 gallery、571合法query/21跨camera身份保留。
2555条只从query排除、仍作gallery干扰项；三折gallery/query为
1000/190、1051/179、1075/202，六端全部最终epoch20保存后strict reload，
读取六个模型的全部baseline/fused/CNN/T/M输出，无末折提前停止或结果删选。
配对初始化、完整采样序列、前8增强、绑定、baseline输出均相同；
六端203/203训练tensor梯度覆盖、overflow0、冻结state不变全部通过。

本次实际对照→跨模态身份损失的全量mAP：
baseline77.487603→77.487603，fused80.206258→79.195387（-1.010871），
CNN79.126676→78.116938（-1.009739），Transformer78.475388→73.695598
（-4.779791），Mamba77.780907→79.087275（+1.306367）。
三折fused差-1.087608/-2.539986/+0.416314。fused Rank1从83.012259降至
79.334501：9个原错query修复，30个原对query变错。全部571个query中AP改善189、
下降208、相等174；21身份全表和五路Rank1/5/10见完整结果及配套JSON。

固定五个科学条件中，aggregate>=+1、各折非负、各专家非负、bootstrap下界>0
四项失败；只有候选fused高于同checkpoint的baseline和三专家通过。
21身份聚类、10000次seed42 bootstrap的95%下界为-3.8126559810990917。
跨模态监督未带来完整泛化收益；不能以Mamba单路收益晋级，也不能据此断言
所有跨模态监督必然无效。V20封存，不扫描温度/系数/分支/epoch或另种子重训。

远端32个绑定文件（包括六个最终权重）全字节SHA校验及六receipt与summary
对象相等校验通过；下载原始summary2022853字节，SHA
23c683b92ad3551e9aa07a24470e82c47565ef54b6683e00213ce7ea0bfbf522；
日志65335字节，SHA978a9f98f8c2d38cb59b101c834c8838acab139c580f88e2612bb2585a00d50e。
本地NumPy2.5.2实际重算全部掩码、三折两端五路AP/rank聚合、增益和bootstrap，
最大绝对数值差1.3322676295501878e-15个百分点，训练损失分量检查通过。
此为JSON算术与文件SHA核验，未在本地执行模型/权重张量加载/图像或距离重算。
独立GPT-5.5 xhigh完整Q1审计待完成，M0审计不能替代Q1审计。

全部三折/五路/21身份结果见
results/TRIFUSION_RGBNT201_V20_CROSS_MODAL_IDENTITY_2026-09-05.md，
逐身份完整Rank1/5/10、六checkpoint训练绑定见
evidence/trifusion_v20_complete_comparison_20260905.json。
原始全量receipt、日志、远端文件SHA、传输、数组重算各自保留；前两折快照
evidence/trifusion_v20_two_paired_folds_20260905.json亦保留，SHA
59b7bfdce0d1b0d8fe5ca351e3f7f53c16a79ab78ef5eac40bccb22557c49053。

V20作为已失败的主实验封存。训练内OOF反复开发使用，不能与官方85.3/87.9
并列为相同协议；可部署dev最佳仍V8 58.4050/59.3939，dev65及SOTA未达。
下步完成独立Q1审计后再固定新的主实验假设；当前V21无代码、配置、预注册
或训练。SAM只做作者论文/源码可行性研究，尚未采用，不声称已诊断出尖锐极小值。
全局任务继续active，不能把这次负结果闭环写成用户目标已实现。


### 37.5 V20独立终态审计完成（2026-09-05）

V20独立GPT-5.5 xhigh终态审计完成：engineering_integrity pass、
overall_verdict/integrity_status warn、scientific_qualification fail。
JSON evaluation_type为real_gt_train_internal_complete_path_oof_reused_development_qualification。
全部三fold、六端、五路、21身份、571query掩码/数组、120epoch/3360优化步、
10k身份聚类bootstrap均被独立实际重算。Python3.13.12/NumPy2.5.2，
最大数值差1.3322676295501878e-15个百分点，训练损失分量残差最大
2.128737297546479e-8。Q1_FAIL和禁止D1/dev/official晋级保持。

独立报告MD SHA3fd30e649a2a84d23dba923437035b620d22284390201412326f477fa4e6bfcc，
JSON SHAcb05f019fb895209997635647a517ccea1017e9e72dc96a08460d6dabac881ab；
原文未修改。完整request/response/meta留在本地trace run07，含追加的当前文件
字节检查清单及共同最终回复。不得把M0审计或本执行器核验冒充独立Q1审计。

WARN保留两类实际限制：criterion.py/protocol当前本地原字节与远端不同，
仅LF规范化后匹配；发布前修正Windows生成的数组回执换行导致算术核验脚本、
回执、比较JSON和报告四个派生文件在审计窗口内SHA发生变化。
独立审计重读后的当前哈希链与独立算术一致；训练summary/六receipt/原日志
从未变化。数组回执现SHA4579ee11406a9666d7e254c7b1092cd91e0079a27e1cbdb65621d4bbaae92b9b。
9e17a552发布后的远端/本地两份派生JSON已逐字节匹配。
此外独立审计未持有远端大权重、图像/特征/距离，其范围为源码、回执和数组。
这些限定不会使负科学结果转为通过。

独立审计读取的结果MD为ce71979e495342607dd95e237e65d26318a432c75bf59cf0f1a3c5fd4626a3f6；
本节是该审计完成后的追加记录，数值与原报告保持，不追写审计原文。


## 38. V21 SAM固定训练计算预算主比较（2026-09-05）

V21已按新主假设实现并在执行前冻结方案；目前未运行T0/M0/Q1。
保持原V8完整冻结Signal和共享CLIP尾部、三角色专家和七路ID/Triplet，
不含V20跨模态辅助损失、V19私有尾部或新推理参数。
以作者SAM参数邻域梯度检验身份泛化，不能声称已证明尖锐极小值是唯一原因。
作者论文/固定仓库commit与许可证、当前PyTorch2.5.1 AMP限制以及BN代码
证据见docs/SAM_SOURCE_AND_DESIGN_NOTES_2026-09-05.md。

普通AdamW40epoch/SAM20epoch，rho0/.05，LR.00035、wd.0001、warmup10/5，
仅seed42/B64K8/远端3090。Q1各端3360对前向反传，合计6720对；
实际优化步分别3360/1680，合计5040；两端数据暴露和更新次数不同。
保留原模型98,800,141总/7,841,292训练参数，新增推理参数0。
SAM两遍复用batch，参数copy精确恢复；七BN只保留第一遍统计、计数每步+1；
第二遍之后只unscale一次，所有第一遍及实际更新梯度覆盖均检查。
SAM全20epoch采样SHA必须等于对照前20epoch，不能声称两端完整40/20一致。

T0三项CUDA解析/AMP/BN测试通过后才跑M0：六模型8batch配对，
两端各8实际容量步及SAM固定batch100步。过拟合解析底0.75H，
第100/1更新前超额loss比例<=.1；所有梯度/冻结/BN/参数/overflow门保持。
M0全通过才自动进入三fold两端完整训练及五路终态strict reload评价。
全部3126gallery/571query/21身份、三fold五路与负结果保留，固定五科学门不变。
无中途选epoch或对照20epoch检索，无rho/LR/epoch扫描、消融或多种子。
Q1失败封存，D1/dev/official均锁定。预计M0 4–8分钟、Q1 130–160分钟，
按稳定epoch时长修正，遵守180–300秒或更长的端点里程碑查询。

冻结计划refine-logs/v21/EXPERIMENT_PLAN.md及时间戳副本，配置
configs/RGBNT201/TriFusion-signal-preserving-v21-sam-rtx3090.yml。
源码tools/train_signal_preserving_v21.py和modeling/trifusion/sam_training_v21.py；
T0 tests/test_trifusion_sam_v21.py；全部SHA在
evidence/trifusion_v21_preregistration_20260905.json，当前AST通过但无GPU结果。
下一步发布并远端校验三项T0；通过后仅启动一个固定原始M0/Q1进程。
V20失败封存、可部署dev最佳58.4050/59.3939和未达65/SOTA状态保持，任务active。
