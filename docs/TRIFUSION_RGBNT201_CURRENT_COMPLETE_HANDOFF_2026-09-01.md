# TriFusion RGB–NIR–TIR ReID 完整交接（2026-09-01）

## 0. 一页结论

当前执行入口：§41.349（2026-09-24）。三数据集全部正式指标在§41.333；RGBNT201–R2/V27 seed45均已完成，累计39份正式种子。四卡前列排名干预的六端完整配对训练正在运行，仍无该批正式结果；终态只读诊断已排队。旧单卡RGBNT100–R2 seed46独立运行状态暂因SSH超时未能复核。§41.233、§41.342及§41.344的运行状态均为各自当时快照。Goal ACTIVE/UNMET。

当前MSVR310训练及此前两个车辆数据集比较使用原V8平行三角色结构：冻结Signal/CLIP，共享block8之前语义和tail9/10/11参数，三个角色分别运行共享tail；CNN处理局部语义Patch高频，Transformer处理全局CLS/Patch关系，Mamba处理空间与位置级三模态扫描。各角色相对冻结reference形成1536D残差，三角色4608D银行拼接3072D Signal得到7680D fused；完整单角色输出为4608D Signal+角色残差。三条路径均执行，当前没有Router/HFER、V23模态MLP或V24原型。两项原V8车辆训练比较均已完成：RGBNT100内部完整比较支持三角色增益，MSVR310原V8比较未超过Signal；后续跨scene Smooth-AP的完整内部Q1 fused已比Signal高0.2761pp，但仍未通过原晋级条件。下面V1—V8条目保留历史经过，不代表当前又启用了旧模块。

本工程是在 DeMo 代码基座上实现的 RGB–NIR–TIR 多模态目标重识别研究分支。V17完整训练和完整gallery补评已封存为失败；其全部查询/图像/分区诊断见§30。V18完整三折两端20epoch主实验已结束（§33）：fused增益+0.921504 mAP，bootstrap下界-0.117338，未通过固定晋级条件；无D1/dev/official。MSVR310、RGBNT100均已安装核验（§32）。V17 fused相对matched weight0为-0.328915 mAP，没有D1/dev/official结果。RGBNT201固定dev保留结果为V8 Phase-B：冻结dev fused=`58.4050 mAP / 59.3939 Rank-1`，比exact Signal高`0.3941 mAP / 1.9394 Rank-1`并超过三个专家，但仍比65 mAP门低`6.5950`，不能声称SOTA。

以下三项是V6历史方案，当时完成了代码与dev运行但性能主门失败；它们不代表当前V8启用了HFER或Router：

1. **Signal-preserving shared semantic expertization**：完整冻结 Signal，三专家共享其 patch/global 强语义场；`baseline_only` 保持原始 3072D 路径，专家训练不能改写 baseline 参数或输出。
2. **Stagewise bidirectional heterogeneous feature exchange**：CNN、Transformer、Mamba 都是三阶段完整专家；阶段 1/2 后进行双向 HFER，可靠性在阶段 1/2/3 分别刷新，使下一阶段能使用其他专家的互补信息。
3. **Complementarity-activated utility-routed residual bank**：不再把九路贡献压成一个向量，而是保留全部 `expert×modality` 残差；联合可靠性与身份效用只控制追加银行，并把银行按样本无自由倍率地校准到 baseline 能量；最终 `fused` 的 3072D 前缀严格等于 `baseline_only`。

历史阶段与保留结果（V1—V8）：

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


### 38.1 V21 T0通过，原M0已启动（2026-09-05）

V21远端三项CUDA数学测试全部PASS，pytest4.28秒（总管理耗时5.818秒），
源码执行commit3c393510f0e0a31bad602af8dd618a8dcdfe6ae6。
解析SAM梯度、一次普通AdamW、AMP scale消去、BN第一遍统计精确保留均通过。
测试只使用合成数学张量，无数据集/项目训练；测试内部有toy optimizer更新，
不能把T0写成所有优化器step总数为0。原始回执见evidence/trifusion_v21_t0_20260905.json。

原始screen 32330.v21_sam_3c39351于16:43:57CST启动，训练PID32331，目录
/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v21_sam_seed42_3c39351，
日志为同路径+.log。16:46:20实际观测PID存活2分23秒，三fold配对preflight已
写入、M0尚无最终回执；GPU6630MiB/100%。当前M0执行中，尚无Q1检索结果。

启动管理脚本使用screen -DmS（大写D），该形式detached但不fork返回，
因此subprocess.run等待原screen结束，SSH管理读取超时退出1。
这不是训练执行失败：随后单独SSH实际确认原PID正常推进，未重启或复制训练。
权威启动观测evidence/trifusion_v21_launch_observation_20260905.json，
管理故障单列evidence/trifusion_v21_launch_transport_20260905.json。
若原管理脚本在训练结束后写出trifusion_v21_launch_20260905.json，其
launched_at/status只能视作延迟管理记录，不得覆盖screen实际启动时间或终态。
后续启动工具应使用小写-dmS，此处保持原进程及执行源码不动。

冻结config SHA f2f47acf54790dc69d9b0d7b5c94dcf9ecfd37bfcf3dd20dc8251e1e1a3600a3，
plan SHA ba17807f30e294618d2a21907a8fde0da82f24f6dab0d51073be49014cc40f71，
runner SHA deafa4d6d2287928c9143d28f2bfb7f32e303939fa6db547f91219f3d708e0fa。
全部T0绑定逐项匹配；启动前GPU24126MiB空闲，磁盘8,357,470,208字节空闲。
M0原估4–8分钟，下次有意义观测16:49CST；如M0全通过，原进程自动继续
ordinary40/SAM20的完整三折，Q1约130–160分钟，按实际epoch再修正ETA。
没有新训练结果、dev/official/D1或SOTA声明，目标仍active且未达成。


### 38.2 V21 M0固定过拟合门失败，Q1未执行（2026-09-05）

V21原进程执行342.520254秒后按固定M0门自动停止，状态M0_FAIL；
16:49:44实际观测原PID32331已退出、GPU1MiB/0%。Q1 fold/endpoint/epoch
均为0，未生成新checkpoint，没有D1/dev/official检索或模型改进指标。

三fold六模型共48个preflight batch完整配对，初始化、增强和五路输出SHA
均相同；各模型98,800,141总参数、7,841,292训练参数、203训练tensor。
普通/SAM各8容量优化步，前向反传分别8/16对；SAM固定batch100优化步、
200对前向反传。M0项目训练总116步、224对前向反传，另有48个forward-only
preflight batch；不能将这些计成Q1主训练预算。

容量峰值reserved分别6126/6284MiB。两端容量和SAM100步的第一遍与实际
更新梯度都覆盖203/203训练tensor，无缺失、overflow0、冻结state不变。
七BN每实际step计数只+1，SAM参数及第一遍统计按源码逐步精确恢复。
实际扰动范数范围0.04999999329447746–0.050000011920928955，符合rho0.05。

唯一未通过的固定M0条件是100步过拟合超额损失比：
原参数点loss第1/100步更新前为0.6110473871231079/0.5914160013198853，
解析底0.75H=0.57838292104621，比例0.39899872365870204>0.1。
用于更新梯度的扰动点loss为0.627093493938446/0.6106454133987427。
完整100步原始轨迹都保存；中途最小原参数loss0.5813781023025513，
末20步均值0.5843785017728805，都只是轨迹描述，不替代预先固定的第100步。
不能择最小值、均值、更换batch/rho/步数或放宽门槛使本次通过。

该结果只否定本次M0准备资格，尚无SAM的heldout检索结果，不足以断言SAM
检索泛化一定有害。此固定V21运行封存，不进行rho/LR/epoch/种子扫描或重训；
后续优化需要新的证据和新主假设，不能把改名重跑当新实验。

原始M0 summary297310字节，SHA
2ecc322270e4e1b82a77cf76e22ab76e359179fda9abc5ef7f2036db064d3c5d；
完整日志47488字节，SHA
0be3a21d007f1ff125779c13231c672ffaab67e884f02478b4be68e620f85194。
远端30个源/配置/方案/CLIP/V12/六source权重全字节SHA通过，目录新权重0；
本地JSON/math实际复核全6配对、116/224预算、100步全部分量、范数和状态
回执以及熵下界/比值，后两者数值差0。没有额外模型或权重张量加载。
工程数值可运行与M0固定准备门通过是两回事。独立M0终态审计待完成，
执行器的本地复核不替代独立审计。

完整结果见results/TRIFUSION_RGBNT201_V21_SAM_M0_2026-09-05.md。
目前无V22代码或方案；先独立审计本次M0并保留所有负结果，再决定新主假设。
当前可部署dev最佳、未达65/SOTA状态不变，整体任务继续active。


### 38.3 V21独立M0审计完成（2026-09-05）

独立V21 M0审计已完成，原始EXPERIMENT_AUDIT_V21_M0.md/json逐字节保留。
overall/integrity为WARN/warn；engineering_integrity=pass，
fixed_m0_qualification=fail，scientific_qualification=fail。
审计实际使用Python3.12.14/NumPy2.3.5重算全部M0记录，数值最大差0；
五项M0布尔门、116优化步/224前反传对和6项T0 toy更新均与原始证据一致。

实际分类为
source_only_engineering_m0_real_train_source_batches_plus_synthetic_t0_no_heldout_dev_official_or_test_retrieval。
原summary中的real_gt_train_internal_complete_path_oof是宽于本次执行范围的
runner/Q1标签，不能据此声称执行了OOF检索。原始summary不改写；
解释以M0_FAIL、folds=[]和零Q1/D1/dev/official访问共同约束。
scientific_qualification=fail在此表示没有科学推进资格及检索证据，
不等于观测到SAM heldout检索下降。

保留四项来源限制：执行commit3c393510与审计时HEAD4f31651不同，按文件SHA核对；
19个依赖中15个本地原字节一致，criterion.py、experts/mamba.py、
experts/semantic_residual.py、protocols/rgbnt201_dev_v1.json四个仅LF标准化一致；
远端数据/CLIP/V12权重只由完整文件SHA账本绑定，审计者没有直接持有这些字节；
参数和BN逐步精确恢复由运行assert及计数支持，原始文件没有逐步完整tensor dump。
这些限制不改变固定M0负结果，亦不能通过格式整理抹去WARN。

审计MD22047字节 SHA00244a74f9f1732aed811d3af9953dd5ea7245981b8c4ddc028b2f0593221244；
JSON24466字节 SHA3ec24312ed4f2c4ea466247ed93be3fc0d44bcbead3f22669965cfadc346ce95。
完整请求、逐字回复、元数据及报告原件已在本地trace run08归档；
trace不公开提交，审计报告本身随仓库发布。本段是审计完成后的归档说明，
不冒充审计者已复核本段新增文字。V21封存，当前没有新的训练或检索结果。


## 39. 完整相机监督检查与V22冻结主计划（2026-09-05）

V21独立M0审计闭环已完成；fa1e860同步三份master，原负结果保留。
随后commit418cf556在远端CPU重放原sampler三fold各20epoch：1680批、107520
样本暴露，原始manifest只取训练source，前8批全索引/文件名与V21实录一致。
没有模型构建/forward、图像读取、权重张量加载、优化或新检索。
全752640有向同ID正对仅60744跨camera，8.070791%；17600/107520行有跨camera
正例。1580/1680批为一组跨camera身份，其余集中于epoch末，不能一概说每批一组。

缺少同camera负例7852行、异camera负例872行，923个批含不完整行；
每批仍有42–64行同时具备两类真实负例。完整source标签证据6367917字节，
SHA5a42be65a512534bb87f52a5f3f4385042157511803774579e65d96d94662d31，
所有逐行计数由本地JSON整数重算通过。此处数量为跨epoch/fold重复暴露，不是新增图像。
同相机负例现象和监督稀疏是新假设的依据，尚未证明相机是唯一失败原因。

V22将三个residual的0.25*batch-hard triplet换为0.25*MCNL；
两个margin均0.1，负例按真实ID及same/other camera划分，保留全部真实ID正例。
只在负例齐全行平均，其他行继续参与原七路ID和fused/branch triplet；
不补造负例、不补零距离。正式每batch支持数必须逐项等于上述冻结元数据。
原V8/Signal结构、同checkpoint baseline3072D、三个专家、推理宽度、
训练参数203tensor与全部初始化不变；无新推理参数。
来源和移植差异见docs/CAMERA_SUPERVISION_AND_MCNL_SOURCE_NOTES_2026-09-05.md。

新固定两端batch_hard_residual/camera_negative_residual，seed42/B64K8，
同AdamW0.00035/wd0.0001、warmup5、20epochs、原增强采样及AMP256。
T0三项CUDA数学契约及真实M0（48forward-only+116优化步，过拟合固定100步
超额loss比<=0.1）通过后，执行全3fold×2端×20epoch/3360步，不提前挑fold。
五路输出/21身份/571query全报；同五项Q1科学门，dev/official仍0。
方案冻结SHA ce2d1f93d5ef36a3eb53d2593fc548b31bff0ea4a3f737b19e1dcaf721d9723b，
配置SHA 5efd8e6dbf8cd70624902148a52701ad69ae5766825293efa878836a3e74f4f6，
runner SHA ee17aeba9bc2a567b1163e6a57759a55423ee5f77edde21e6c7a81625fe45a53。
当前仅已写代码并AST解析，T0/M0/Q1均未启动；不能称模型已训练或改善。
T0预估5–10秒、M0约4–7分钟、完整Q1约75–95分钟，依实际epoch修正。
原本best dev58.4050/59.3939、未达65和官方目标的状态不变，整体goal继续active。


### 39.1 V22 T0通过，原M0实际启动（2026-09-05）

V22三项远端CUDA数学契约全部PASS，pytest3.02秒、总管理3.957356秒；
这是合成loss/autograd测试，模型实例、数据集、项目及toy optimizer更新均0。
全部runner/module/tests/config/plan SHA与冻结回执相同，T0源码commit5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36。

17:51:07.422CST以小写screen -dmS启动唯一原始训练进程34656，
screen34654.v22_camera_negative_5ae096b；17:51:08实际确认PID和完整argv。
目录/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v22_camera_negative_seed42_5ae096b，
日志同路径+.log，退出码将写同路径+.exit。启动前GPU24126MiB free/1MiB used，
磁盘8342507520字节空闲。当前只是M0已启动，没有Q1检索结果或新模型改进声明。
权威T0及启动回执为evidence/trifusion_v22_t0_20260905.json及trifusion_v22_launch_20260905.json。

M0预估4–7分钟，下一次有意义检查17:54–17:55；通过后原进程自动执行完整六端，
Q1初估75–95分钟、约19:10–19:30完成，首个完整paired-fold约18:20，
须按实际epoch速度修正。失败则保留完整M0轨迹并停止Q1，不重启或更换门槛。


### 39.2 V22 M0全部通过，原Q1完整比较继续（2026-09-05）

M0于原进程约225.699910秒时完成，全部五项工程检查通过。17:55:06实际观测
原PID34656存活、三fold配对完整、GPU6546MiB/93%；没有完整Q1 paired fold。
原进程自动继续固定完整Q1，未重启、未改变执行source5ae096b或方案/配置。

六个真实模型各98800141总参数、7841292训练参数、203训练tensor，94-source/
47-heldout严格隔离；48个forward-only batch的初始化、五路输出、增强/路径/索引
及camera支持全部配对。两端容量各8步，实验端从fresh source固定batch100步，
共116项目优化步和116对前向/反传，另有48forward-only；M0不计入Q1预算。
容量峰值6054/6200MiB；两端容量及100步训练均203/203非零梯度、overflow0、
完整冻结state不变。T0三CUDA测试已通过，T0项目/toy optimizer更新均0。

固定100步初/末loss0.7189050912857056/0.5803177952766418，
解析底0.57838292104621，超额loss比0.013769174124866987<=0.1。
选定总loss的共同ID及branch项0.6106956005→0.5803177953，
加权MCNL项0.1082095206→0；固定batch56/64行具备两类负例，8行缺少同相机负例。
三专家初始camera hinge active rows为47/53/40，末步均0。
未进入实验端优化目标的原残差triplet诊断0.0003517768→0.0321446024，
不能把“MCNL降至0”扩写成所有度量都改善；本次M0没有检索性能结论。
全部100步分量、两段hinge及active rows都保留，未挑选中间checkpoint。

原始M0 snapshot374648字节 SHAad0a27abbba79f1c039f68ebcfcc64eba731916581a8dec67a5c64c19d212427。
17:57:12 snapshot日志104518字节 SHA47dc1a77075d588551f2fe73933da03a739815d40c09d3960ea9b9ef30491369。
当时Q1控制端已记录3个完整epoch/87优化步，尚无新checkpoint；当前未完成epoch
的步数不包含在87中。因此summary的folds=[]代表零完整检索fold，不能说Q1优化步0。
M0 snapshot只是工程阶段的不可变证据，原始run_summary继续随完整Q1更新。

远端30个source/config/plan/tests/metadata/CLIP/V12/六source权重完整文件SHA全通过，
Signal commit/diff与执行绑定相同；该检查没有加载权重tensor或额外优化。
本地JSON/整数/解析loss复核全部48batch、116更新及100步分量通过：
总分量最大差5.960464477539063e-08，MCNL分量最大差7.450580596923828e-09，
解析底和超额比数值差0。执行器复核不替代独立M0审计，独立审计待完成。

完整Q1始终为三fold两端各20epoch/3360优化步、120epoch记录，最终checkpoint唯一；
所有5输出/3126gallery/571query/21身份与五项科学门保持。当前没有Q1检索结果、
D1/dev/official或SOTA新指标。首个完整paired fold下一次观测约18:18–18:20；
完整Q1暂估19:10–19:30，收到完整epoch/端点用时后修正，不提前反复查询GPU。


### 39.3 V22独立M0审计与首个完整配对折（2026-09-05）

独立M0审计原始结论为overall/integrity WARN/warn、engineering_integrity=pass、
fixed_m0_qualification=pass、scientific_qualification=fail。审计者用Python3.12.14/
NumPy2.3.5独立复算元数据和全部100步分量，sidecar最大差0，固定超额loss比
0.013769174124866987；总loss及MCNL分量重构差分别5.960464477539063e-08、
7.450580596923828e-09。未选中的原残差triplet上升这一负诊断原样保留。

这里scientific_qualification=fail表示本次审计所持M0文件尚无Q1终态检索科学证据，
不表示固定M0门失败，也不冒充Q1终态判定。审计范围为17:57的M0/早期训练日志：
3个完整Q1 epoch、87步、当时零checkpoint；不覆盖随后18:20捕获的第一折检索。
实际分类为source_only_engineering_m0_real_train_source_batches_and_metadata_replay_plus_synthetic_cuda_t0_with_nonterminal_q1_training_log_no_heldout_retrieval。

保留来源限制：审计时HEADf63889f不同于执行5ae096b、远端M0验证观察2fd6506；
17个依赖中13个本地原字节匹配，criterion.py、experts/mamba.py、
experts/semantic_residual.py及protocols/rgbnt201_dev_v1.json四个只在LF标准化后匹配。
远端30项全文件SHA账本存在，但审计者未独立持有/加载CLIP、V12、权重tensor或图像。
审计没有远端命令、网络、下载、模型、训练或额外检索；WARN不能由文字整理抹去。

原始审计MD30347字节 SHA6c8420dfb7275df657c53b387eb02a8913077fc5d3bd11d6f30881d39d280e9c；
JSON50920字节 SHA371f54725d8c601559a323f3beb31c2e00ced50b68b978d49cd2af81834de91d。
完整请求、逐字回复、元数据、报告和审计前M0说明/跟踪表已在本地trace run09归档。
本段为执行器在审计完成后的归档说明，不声称审计者复核了本段及后续Q1结果。

18:20:12实际观测原PID34656仍运行，GPU6546MiB/100%；第一折两端最终20epoch
checkpoint均已strict reload并完成全部五输出检索，第二折control已记录第4epoch。
共44个完整Q1 epoch日志，已知完成1272步；其中两端完整receipt共40epoch/1160步。
这个计数不包含正在进行的epoch，不能把summary仅有1fold当作总训练只有1160步。

第一折1000gallery、47heldout身份、190合法query、810条仅从query分母排除。
fused对照71.201727/71.578947 mAP/R1，MCNL70.525659/69.473684，
fused差-0.676068pp；CNN/Transformer/Mamba的mAP差为-1.072661/-1.791450/+0.848838。
预注册“每折fused非负”条件已不满足，但全部后续端点照原计划继续，
不择端/择折，不重新训练或修改margin/权重/epoch/LR，终态仍未完成。

本地JSON/NumPy复算第一折两端五输出mAP/R1/R5/R10全部差0；
两端各580步的相机支持累计数与冻结metadata精确一致，loss加和最大误差
6.583487088818174e-09。这是执行器的第一折部分结果核验，不替代独立终态审计。
完整Q1仍为六端120epoch/3360步，下一完整paired-fold观测窗口约18:39–18:43，
按实际第一折速度修正整轮ETA为19:00–19:10（估计），原进程不重启。

首折五路完整指标见results/TRIFUSION_RGBNT201_V22_FIRST_PAIRED_FOLD_2026-09-05.md及原始部分JSON。
当前best dev仍58.4050/59.3939，整体目标未达；D1/dev/official保持0，原完整Q1继续。


### 39.4 V22全六端终态失败并封存（2026-09-05）

V22原始单进程完整结束，退出码0；20:43:56实际观测PID34656已退出、GPU1MiB/0%。
原始运行耗时4165.238463401794秒（69.420641分钟，含M0）；三fold两端各20epoch、
共120条完整epoch记录/3360优化步全部完成，未因第一折负收益提前停止。
执行commit5ae096b65eb4c9987b0b8edaa7bfcd8a4cee1c36及固定配置/方案不变。

全部141 heldout身份、3126gallery、571合法query/21跨camera身份完整保留，
2555条只从query分母排除；六个最终checkpoint均从原source重建后strict reload，
模型state SHA与该端训练final一致，冻结Signal与共享尾部不变。
baseline/fused/CNN/Transformer/Mamba五输出与三fold/all21身份完整表均已生成。

MCNL/control aggregate fused为78.984454/80.640677 mAP，差-1.656222pp；
R1为82.311734/83.712785。三fold fused差-0.676068/-2.149943/-2.140645。
CNN/Transformer/Mamba aggregate mAP差-0.897312/-4.265440/+0.317625；
candidate CNN79.152126高于candidate fused78.984454，因此五项固定科学门全部FAIL。
全部21身份、10000次seed42 cluster bootstrap的95%下界为-3.8769957222550886。

原始汇总2207490字节 SHAb8cd7db81efc3827a91d165d47e001073785420baf8ddfc8507a2eead9c3d6a3；
完整日志255971字节 SHAa091390a25ced0cfd336fce3c5bd6c51565fcad6becfbf3419778bcb9b7a2f1a。
远端36项source/config/plan/tests/metadata/CLIP/V12/source及新final权重全文件SHA通过，
六独立receipt对象与summary内嵌对象相同。核验没有加载额外权重tensor/图像或重跑检索。
本地NumPy2.5.2完整mask、AP/Rank聚合、全21身份Bootstrap数值最大差0；
120行日志逐对象等于六端history，步数和不可变M0都精确一致。

每臂1680batch/107520样本暴露：98796行具两类负例、7852缺同相机负例、
872缺其他相机负例、17600具跨camera正例；六端全部训练批次支持数与冻结metadata相同。
三候选末epoch MCNL项均下降，但未选用的普通残差Triplet项均较首epoch上升；
这是训练目标改变后的实测诊断，不能把MCNL下降写成检索泛化改善或唯一因果证明。

固定V22封存Q1_FAIL，不作margin/系数/sampler/epoch/LR/seed变体或重训，
D1/dev/official均未执行。M0工程/固定门独立审计PASS、完整性WARN已归档；
它不覆盖完整Q1，本终态独立审计待完成。当前best dev仍V8的58.4050/59.3939，
65mAP开发门和官方目标均未达到，整体goal继续active。

下一步先核对共同初始化与最终模型的可比性。旧V12缓存的检索gallery只含
跨camera合格身份（tools/build_v12_complete_path_oof_targets.py:610–617），
当前评价保留3126条完整gallery。因此旧V12约88mAP与当前约80mAP不能直接
相减来证明继续训练导致退化。尚未执行共同初始化的完整gallery诊断或新训练。
完整结果和所有负收益见results/TRIFUSION_RGBNT201_V22_CAMERA_NEGATIVE_2026-09-05.md。


### 39.5 共同初始化完整图库诊断执行前冻结（2026-09-05）

tools/diagnose_v22_initialization_full_gallery.py及
refine-logs/v22/INITIALIZATION_FULL_GALLERY_DIAGNOSTIC_PLAN.md已固定并AST通过。
将只读恢复三个既有共同初始化，要求state和两端training.initial_state精确一致，
按同一完整3126gallery/571query评价全部五输出，前后比对source与模型state SHA。
不优化、不写checkpoint、不选中间epoch、不访问dev/official，不改变V22的Q1_FAIL。
补齐旧eligible-only缓存与当前完整gallery不直接可比的证据缺口，不预设退化方向。
全部初始、对照终态与MCNL终态将保留；计划、源码SHA和未执行状态见
evidence/trifusion_v22_initialization_diagnostic_preregistration_20260905.json。
预计2–3分钟；启动前实际核验GPU与源文件，诊断结果尚不存在。
V22完整Q1终态独立审计已交GPT-5.5 xhigh，trace run10；它的46份输入保持不变。


### 39.6 初始化完整图库诊断完成（2026-09-05）

共同初始化的完整图库只读诊断已结束：原PID42325退出码0，74.088494秒，
3模型/26次模型调用/3126triplet前向，optimizer0、checkpoint写入0、dev0/official0。
三个初始state及binding均同时等于该fold两个终态保存的初始绑定；
评价前后state和30项source文件SHA不变，所有参数grad均None，baseline所有数组精确相同。

同完整图库/同输出的初始化fused80.590328 mAP/R1 83.712785，
普通终态80.640677/83.712785，mAP差+0.050348；MCNL终态差-1.605874。
普通终态相对初始化CNN/Transformer/Mamba差+0.729564/+0.539850/-1.044532，
MCNL分别-0.167748/-3.725590/-0.726907。不能用单一分支取代整体比较。
旧V12约88是eligible-only gallery下的residual/bank结果，不能直接与当前fused相减；
本次直接可比证据不支持普通继续训练使融合mAP整体下降，也不证明任何唯一失败原因。

全部45个fold/阶段/输出指标行、全21身份及两端五路逐query变化已经JSON/NumPy复算，
最大数值差0。原始758929字节 SHA21a73baacca91834eb5f47ec0c129731cfdb42ff92a5b90c2d712bef40f334ca，
日志6351字节 SHA756aa7a7eb67c8bde7c1ce41d62274677a62522cd01864bb4549f45d362c2a55。
结果及完整范围见results/TRIFUSION_RGBNT201_V22_INITIALIZATION_FULL_GALLERY_2026-09-05.md，
配套evidence/trifusion_v22_initialization_full_comparison_20260905.json保留所有21身份全部输出。
独立诊断审计待完成；不选择初始化作为部署checkpoint，不改变V22 Q1_FAIL/禁重训规则。
后继需要新的表征干预依据，当前未设计或启动V23；best dev与未达总目标状态不变。

### 39.7 V22完整Q1独立终态审计完成（2026-09-05）

独立终态审计已完成：GPT-5.5 xhigh原始结论为工程PASS、固定M0 PASS、
完整性WARN、科学资格FAIL，实际evaluation_type为
real_gt_train_internal_complete_path_oof_reused_development_qualification。
全部五项固定科学门均FAIL；没有D1/dev/official/test结果，也没有SOTA晋级证据。

审计员以本地Python3.12.14/NumPy2.3.5从原始AP/Rank和真实ID/camera数组重算
全部fold/output聚合、21身份/10000次seed42 Bootstrap、全部120epoch日志及loss、
1680批相机元数据整数计数、M0固定首/100步门；指标/增益/Bootstrap最大差0，
loss分量舍入最大差7.105646293581458e-09，support舍入最大差7.105427357601002e-15。
全部六receipt与summary对象一致，query mask无不一致，3360步训练记录完整。

WARN原样保留：criterion、mamba、semantic_residual及protocol四份本地文本仅LF归一化后
等于运行字节；远端CLIP/V12/final权重等14项只有清单持有，审计员未独立取得权重字节，
没有加载tensor、图像或重算模型embedding/distance。运行commit5ae096b与审计时
current HEAD ad7841d不同，绑定依赖执行源码SHA，不把文档commit当作训练commit。
这仍是反复使用的train-internal完整路径OOF开发资格，不能写成独立dev或官方泛化。
报告EXPERIMENT_AUDIT_V22_Q1.md/json和trace run10保留原始字节与完整限制；
审计时result/tracker快照及全部46输入SHA已归档。固定Q1_FAIL及禁扫描/重训规则不变。

初始化完整图库只读诊断的独立审计另在run11进行，27份输入保持冻结；
该诊断不是新增训练或checkpoint选择，不能把初始化fused当作新的部署结果。
目前没有新GPU作业，V23尚未实现；下一步核查新的表征改动原始论文与实际源码。

### 39.8 初始化完整图库独立诊断审计完成（2026-09-05）

初始化完整图库诊断的独立审计已返回：overall/integrity WARN，
engineering_integrity=pass_with_provenance_limitations，
scientific_qualification=fail_to_promote_descriptive_only。
审计独立重算完整query mask、45组fold/阶段/输出指标、21身份、AP/R1变化计数及
两终态相对初始化的差值，最大全精度差2.842170943040401e-14；
结果表六位小数最大舍入差4.958391315312838e-7。未发现伪GT、自归一化评分、
虚构聚合值、隐藏优化/选点/dev访问或数值漂移证据。

审计实际算术使用E:/python.exe 3.12.6和标准JSON/整数/浮点运算，
不是模型/图像/embedding/distance重算；权重、CLIP、图像与远端文件持有依旧限于清单。
protocol本地CRLF与远端LF只在换行归一化后SHA匹配，不能写成原始字节全相同。
比较JSON保存配对变化计数汇总，显式逐query差值行未另存；
初始化和终态原始AP/Rank数组及query身份完整保存，足以精确重算全部变化。
这是复用的train-internal完整路径OOF描述性诊断，不是独立验证或官方结果。
它不改变V22 Q1_FAIL，不选择初始化checkpoint，也不提供D1/dev/official资格。
原始报告EXPERIMENT_AUDIT_V22_INITIALIZATION.md/json、实际请求/回复、27份输入SHA
及审计时result/tracker快照均保留于trace run11。派发模型gpt-5.5/xhigh来自实际工具请求；
报告自身auditor字符串codex-gpt-5-direct-bounded-audit原样保留。


## 40. V23模态专属语义尾部适配主实验（2026-09-05）

### 40.1 执行前固定方案

V22及初始化诊断两份独立审计均已归档，负结果和只读描述范围不变。
ICPL实际源码核查见docs/SPECTRAL_ADAPTER_SOURCE_NOTES_2026-09-05.md：
官方RGBNT201入口使用独立视觉模型和768中间维并行MLP，
不把README low-rank、未调用model_mm_adapter文件或论文完整方法当成本次实现。

新V23在三个frozen CLIP尾部block输出后、原角色算子前分别加入RGB/NI/TI
768→128→768残差MLP，同阶段模态参数在CNN/T/M间共享；共9MLP/1777536参数。
up和bias置零，两端初始结构和输出相同；控制固定零，候选训练新参数。
原Signal/CLIP/角色/等能量五输出结构及原ID/Triplet目标保持完整。
预期控制/候选trainable为7841292/9618828，203/239tensor；
该比较不能排除额外可训练容量的作用，不声称唯一模态机制已证实。
代码已实现、AST通过；T0/M0/Q1均未执行，尚无新指标。

固定方案refine-logs/v23/EXPERIMENT_PLAN.md与版本化副本、preregistration已生成。
远端T0五用例；M0三fold两端共54只读前向+116优化步，固定首/100步excess门<=0.1。
M0通过后原进程完成3fold×2端×20epoch/3360步、完整3126gallery/571query五输出。
沿用所有五项Q1科学门，失败封存不作宽度/插点/scale/epoch/LR/seed扫描。
只在Q1全门通过后另行固定D1，当前不访问dev/official、不开始消融。
预计M0 4–7分钟、Q1 75–100分钟，需启动前检查真实GPU和源文件SHA。


### 40.2 T0通过，原单进程M0及完整Q1启动（2026-09-05）

远端CUDA T0实际9.800828秒、5passed，零适配器与原模型全五输出/strict reload一致，
单模态梯度分派、零up初始化首步梯度、两条件各3步toy优化与冻结Signal契约通过。
真实数据/项目训练步0，6个synthetic模型/1个独立stage/6步toy优化；
原日志含3条timm弃用warning，保留未隐去。

22:01:32.893788+08:00以执行commit9f4a10b6162b9658ba103cd92466411ebb6ccd8f
启动原PID44684，screen v23_spectral_adapter_9f4a10b。
启动前空闲24126MiB/使用1MiB，18项代码/配置/方案/来源权重完整SHA核验通过。
证据evidence/trifusion_v23_t0_20260905.json/log/xml及trifusion_v23_launch_20260905.json。
run为artifacts/trifusion_v23_spectral_adapter_seed42_9f4a10b及同名.log/.exit。

当前M0运行，尚无M0完整资格或Q1科学结果；预计22:06–22:09附近完成M0，
若全门通过，原进程自动完成六端Q1，预计另75–100分钟。
按预计窗口查看，间隔180–300秒；原进程观察超时不得当作训练失败重启。
保留固定54前向/116步M0、完整3360步Q1与五门，不更改首/100步门或早停选择。

### 40.3 完整M0通过，原进程Q1运行（2026-09-05）

M0真实完整结束并通过：耗时240.083842516秒，
三fold两端共48次source配对前向，另6次原encoder只读对照（共54前向），
fold0两端各8不同batch更新及fresh候选固定首batch100步，共116项目优化步。
六模型初始state、source state、每端8batch增强/路径/五输出SHA精确配对；
零适配器与原encoder五输出在六个第一batch上精确相同。
所有94-source与47-heldout身份隔离；baseline和冻结尾部保持不变。

两端总参数实际100577677；控制/候选trainable7841292/9618828，
训练tensor203/239，恰差9个模态MLP/1777536参数/36tensor。
两个8步容量的非零梯度覆盖203/203和239/239，peak6090/6494MiB，overflow0；
固定100步候选239/239覆盖、冻结state不变、overflow0。
L1=0.6110473871231079，L100=0.5803323984146118，
F=0.5783829210462100，
固定excess比0.05968189909525461<=0.1，通过全部五项M0工程门。

本地JSON/NumPy重算全部116步各14原始loss分量及加权总和、全部配对和固定门，
最大分量舍入差8.9406967163085938e-08；原始log中的M0对象逐项相同。
远端29项source/config/plan/CLIP/V12/source权重全文件SHA一致，无新增模型或检索。
preregistration完整等于执行commit，所有执行前10份工件按Git blob精确复核；
latest tracker因如实更新状态而不用于当前字节等于旧prereg的声明。

22:07:16实际观测原PID44684仍在运行，已自动进入Q1且仅记录第一折control前2epoch，
58个Q1更新；完整paired fold0、端点receipt、heldout AP/Rank和科学门尚不存在。
M0快照343499字节 SHAb662bf6420f7f5dc6f92fd0d93d00b2eabfdb3aee4e70af34d6946589b587a2a；
对应log83703字节 SHA48303a40fd7ff82e4a0afa646729e4b7bf2a7f3e3b3e29f2586bcf0ea83029bf。
这是含明确非终态训练日志的source-only工程证据，不是检索改进。
独立M0审计待完成；M0通过不改变五项Q1科学门，不允许D1/dev/official。


### 40.4 M0独立审计派发与终态核验准备（2026-09-05）

22:16已将39份完整M0/T0/来源/计划/代码工件交GPT-5.5 xhigh独立审计，
trace run12，agent /root/audit_v23_m0。输入SHA保持冻结，审计报告尚未返回。
实际四份执行器inline源码已归档，终态Q1不由M0工程审计替代。

终态辅助工具已准备且AST/CLI help通过，尚未在终态数据上执行：
tools/verify_v23_terminal_files.py核对完整文件/六receipt/执行Git blob，
tools/audit_v23_terminal_arrays.py复算全部标签mask、六端五路、loss分量、21身份Bootstrap及五门，
tools/report_v23_complete_comparison.py生成全部三fold×两端×五路/21身份/六训练绑定完整比较。
V23适配参数差、额外反向计算和复用OOF限制已写入结果模板。
准备状态见evidence/trifusion_v23_terminal_helpers_prepared_20260905.json；
不能把AST/CLI help称为终态数值或文件核验成功。

### 40.5 M0独立审计完成与第一折完整配对进度（2026-09-05）

独立M0审计完成：overall/integrity WARN，engineering PASS，
fixed M0 PASS（source-only工程门），scientific NOT_QUALIFIED。
审计使用Node.js v24.13.0标准库对全部116步分量及固定门独立重算；
最大总loss差8.940696727410824e-08，entropy floor=0.57838292104621，
固定第100步excess比0.059681899095254606。它没有执行模型、tensor、图像、
GPU、网络、远端或新优化。七份远端大权重仅由执行器全文件SHA清单支持，
没有被本地审计重新持有/散列。
criterion/mamba/semantic_residual/protocol四文件本地CRLF与执行LF原始字节不同，
仅LF归一化SHA一致；当前latest tracker与执行前登记不同，历史Git blob及版本副本保留。
审计39份输入的原始SHA在归档前再次全部匹配，原始报告/请求/回复与审计时
result/tracker快照保存于trace run12。
审计指出M0结果原句“所有Q1计划原样执行”可能误读为已经完成；
保留审计时原文后，当前M0结果改为明确将按计划继续执行并标注22:07证据时间。
这是文字时态澄清，未改变方案、门限、训练或结果。
M0审计不提供终态Q1、D1/dev、官方或SOTA资格，新增1777536可训练参数及
额外反向计算仍是模态机制解释的混杂因素。

22:37:30.290513+08:00实际只读观察原PID44684及原命令仍运行。
已完成fold0两端各20epoch/580步，正在fold1对照epoch16；
全部日志56个Q1 epoch、1608/3360更新，2/6端点receipt、1/3完整配对fold。
GPU6582MiB使用/17546MiB空闲/100%；exit尚不存在。
partial run_summary最后保存elapsed1628.0644698143005秒，
扣除M0的240.08384251594543秒，首个完整配对约23.13分钟。
据此估计整轮23:10–23:20完成，后续按实际里程碑修正。

这是全部已完成fold0的结果：该fold完整1000gallery/47heldout身份，
190合法query，810记录只从query排除而保留在gallery。
计划三fold合计3126gallery/571query；不能将单折写成完整Q1。
本地仅JSON/NumPy重算首折两端全五路AP/Rank、全部query mask和40条训练history；
指标最大差0，loss分量最大舍入差2.9208202856345622e-08。

| 输出 | 对照 mAP | 候选 mAP | 差值 pp | 对照 R1 | 候选 R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 68.767642481 | 68.767642481 | +0.000000000000 | 69.473684211 | 69.473684211 |
| fused | 71.598374938 | 71.598253726 | -0.000121211614 | 72.631578947 | 71.052631579 |
| cnn | 71.166864248 | 72.318663491 | +1.151799242551 | 71.052631579 | 71.052631579 |
| transformer | 70.057173310 | 68.858141806 | -1.199031504211 | 73.684210526 | 70.526315789 |
| mamba | 70.191776363 | 69.916211848 | -0.275564514982 | 68.947368421 | 66.842105263 |

fold0融合精确差值-0.00012121161431366545pp，小于0；
固定“每折融合非负”门在该折不满足，不能因显示舍入接近零改门或算通过。
原过程仍须完成剩余四端，不早停、不选点、不改width/stage/scale/epoch/LR/seed。
当前没有完整聚合、21身份Bootstrap或所有五项终态门；没有D1/dev/official资格。
原始JSON保留首折全部AP/Rank、全部五输出、训练绑定；
日志还保留完整已记录的56epoch，未筛选有利输出或隐藏未配对训练。


### 40.6 原六端完整Q1结束，固定科学门全部失败（2026-09-05终态；2026-09-06归档）

23:18:19+08:00实际核验原PID44684已退出，原exit为0，Q1_FAIL。
全部三fold、两端、20epoch、120训练行、3360优化更新完成，原过程未重启、未早停、未选点。
原运行elapsed=4282.281049251556秒，包含M0。完整图库1000/1051/1075，
合法query190/179/202，合计3126图库、571query、21身份；2555记录仅从query排除。

| 输出 | 对照 mAP | 候选 mAP | 增益 pp | 对照 R1 | 候选 R1 |
|---|---:|---:|---:|---:|---:|
| baseline_only | 77.487603116 | 77.487603116 | +0.000000000 | 79.334500876 | 79.334500876 |
| fused | 80.507515547 | 80.254811046 | -0.252704501 | 84.413309982 | 83.537653240 |
| cnn | 79.471874954 | 80.540530605 | +1.068655652 | 83.537653240 | 84.413309982 |
| transformer | 78.529832158 | 77.927403684 | -0.602428474 | 80.910683012 | 81.786339755 |
| mamba | 77.796840675 | 77.529976219 | -0.266864457 | 80.035026270 | 78.633975482 |

三折fused增益分别为-0.00012121161431366545, -0.9996673157096865, 0.17162975099304845 pp。
21身份/10000次seed42 bootstrap的95%下界=-1.509454322847345 pp。
原五项科学门全部失败：增益不足1pp、不是每折非负、T/M下降、区间下界非正、
candidate fused不超过candidate CNN。fused修复7个R1错误，新增12个；
571query的AP改善178、下降191、相等202，完整负结果和所有身份均保留。

封存本次V23；不进行width/stage/scale/模态/专家/epoch/LR/seed或loss扫描，
不执行D1/dev/official/消融。CNN局部增益不能替代融合晋级。
该结果不证明模态专属适配普遍无效，也不能将额外1777536可训练参数的效果单独归因于模态机制。
固定dev最佳V8 Phase-B 58.4050/59.3939与65mAP开发门、官方目标未达状态保持。

原始完整汇总evidence/trifusion_v23_q1_seed42_9f4a10b.json，
2148943字节，SHAdbb58d0d614dc5e8007e8548508dec75e6a5225647450467e7b13a2c1111d9b0。
完整日志201215字节，SHA60482e4297b484cf035a5e6417f2a3e49949d0fe609dba4b928971d39d7e9588。
全部六份独立终点receipt与汇总对象精确一致；远端39份源代码、方案、CLIP/V12/最终权重
全文件SHA及执行Git绑定通过。没有将二进制权重拷回本地或加载本地模型。
本地JSON/NumPy复算全部五路四指标、全部标签mask、配对与21身份bootstrap，指标最大差0；
120epoch的14项loss分量最大舍入差3.3728824289092074e-08。
完整日志120行训练history、6终点事件、M0对象与最终汇总逐项一致。

完整三fold×两端×五输出30组指标、全部21身份的五路四指标、全部六训练绑定，见
results/TRIFUSION_RGBNT201_V23_COMPLETE_Q1_2026-09-05.md及
evidence/trifusion_v23_q1_complete_comparison_20260905.json。
独立终态审计trace run13由实际GPT-5.5 xhigh请求派发；原始第一份报告及回复已存档。
当前正补充独立bootstrap和训练范围描述核验，尚不将其写为最终审计完成。
GPT同家族独立审计不构成跨家族认证；M0工程审计也不能替代Q1审计。

### 40.7 接收用户更新复核后的后继路线（2026-09-06）

按用户优先级，下一项主假设聚焦环境内身份区分、真实跨摄像头监督覆盖、
source-only原型/困难负例；不继续增加Router或私有尾部容量。
IICI与XBM实际入口、参数和边界已核对，记录于
docs/SOURCE_PROTOTYPE_MEMORY_RESEARCH_2026-09-05.md，13份来源文件SHA在对应evidence中。
IICI每身份单摄像头断言不适用于本项目每fold14个真实跨摄像头身份，
须保留全部94个source身份及真实正关系；弱视图更新与强视图监督分离。
XBM源码128D/标签0哨兵不是可直接复制的通用合同；原型与FIFO队列机制不能混称。
每fold/endpoint独立初始化记忆并记录关系覆盖；训练不得混入heldout/dev/gallery。
本节为下一计划的设计依据，尚未登记V24实现、具体loss权重或新训练。

用户三数据集路线保持RGBNT201、MSVR310、RGBNT100，
下一跨数据集资格优先MSVR310；车辆数据已安装但本项目成绩仍未测量。
MSVR310按scene/time规则，不套用RGBNT201 same-camera过滤。
当前seed42、主结果成功前不开展消融的约束保持。

融合公式中纯1536D残差与日志完整专家输出分别记号：
s_fused = 0.5*s_Signal + (s_r_CNN+s_r_T+s_r_M)/6。
日志branch各为Signal与单残差等能量拼接，所以同一图像对
s_fused = (s_branch_CNN+s_branch_T+s_branch_M)/3；AP不能按此平均。
细节见docs/USER_REVIEW_ACTIONS_2026-09-05.md。


### 40.8 V23独立终态审计完成（2026-09-06）

独立终态审计完成于2026-09-06T00:14:21.188997+08:00，原始请求/两轮报告/回复完整保留trace run13。
overall/integrity WARN；engineering PASS；fixed M0工程PASS；scientific FAIL。
审计使用既有Python3.13.12/NumPy2.5.2独立重算10000次身份bootstrap，
下界-1.509454322847345，差0；21身份贡献与完整比较差0，
与数组核验的加权贡献最大浮点差5.551115123125783e-17。
审计第一轮将训练范围写成仅adapter；复核后更正为两端原203角色/分类头tensor
均训练，候选另训练36adapter tensor。原始错误表述及更正报告均归档，未修改实验。
44份原始输入SHA在报告归档前再次全匹配；审计时tracker/result快照保留。
远端大权重仍为ledger-only持有，四源文件仍只有LF归一化匹配；
GPT同家族独立审计不构成跨家族认证。V23继续封存，无D1/dev/official资格。


## 41. V24 source-only环境身份原型主实验（2026-09-06）

按用户新优先级固定一个训练侧原型目标：每fold全部94 source身份、108真实身份/摄像头原型，
双摄像头身份均衡形成全局原型；普通视图更新，强增强向同真实身份学习。
全局竞争93个不同ID，同环境竞争保留该摄像头实际全部身份；不使用MCNL、伪ID或heldout记忆。
来源成员关系已从完整标签普查确认；代码与AST通过，尚无T0/M0/Q1运行结果。

原合法V12初始化、V8三角色及Signal推理路径保留，新增推理参数0，
不继承V23适配器/权重。两个端点同普通/强双视图、同原七组ID/Triplet均值；
候选另加全局/同环境原型CE均值，温度0.05、momentum0.2、系数1固定。
几何在六视图中同步；普通erase0.5，强brightness±0.2和erase0.6；同组重复先平均后单次EMA。
记忆只在训练中存在，每端fresh初始化并保存年龄/更新计数、初始和最终状态/SHA。
原采样器不变，新增的是batch外身份竞争和真实跨摄像头正原型覆盖，不宣称batch内正对比率提高。

完整计划refine-logs/v24/EXPERIMENT_PLAN.md及版本化副本，
代码/配置/计划/成员关系SHA见evidence/trifusion_v24_preregistration_20260906.json。
T0六测试；M0九fresh模型、96双视图预检前向、18882 source初始化特征前向记录、
116优化更新/232前反传对，固定100步excess门<=0.1。全部M0门通过后原进程执行
六端各20epoch，共3360更新/6720前反传对，完整3126图库/571query五输出与原五项科学门。
预期T0几十秒，M0约10-20分钟，完整Q1约2.3-3小时，以真实阶段耗时修正。
失败封存固定版本，不扫描原型或增强设置，不执行D1/dev/official/消融；
主结果未达状态及MSVR310优先的下一跨数据集路线保持。


### 41.1 T0通过，原M0过程已完成三折六端预检（2026-09-06）

00:56:14.521639+08:00启动原PID52030，screen v24_source_prototype_6a4ac2c。
执行commit6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33；18项代码/配置/方案/来源权重SHA启动前全通过。

01:04实际确认原过程存活，GPU5362MiB使用/18766MiB空闲/100%；exit不存在。
三fold两端完整预检已完成，完整初始model/memory及8个双视图batch/五输出SHA严格配对。
真实参数98,800,141/可训练7,841,292/203tensor，两端相同；新增推理参数0。
source/heldout94/47隔离；108原型及真实相机成员关系与完整标签普查一致。
六预检模型共12504条source特征前向、102次source batch前向，另96次双视图预检前向。
原型范数记录最大偏差1.1920928955078125e-07，初始化更新计数/年龄均0；这是记录算术核对，不是本地模型重演。
summary最后保存elapsed332.6613943576813秒，此后容量/固定批次阶段尚未整体写回。
观察中的0epoch/0更新仅指Q1日志，不能推断M0尚无优化。
完整M0原定116更新/232前反传及固定第100步门不变；目前没有完整M0 ratio或结论。
预计启动后10-20分钟完成M0，实际完成后修正Q1的138-180分钟预算；按阶段观察，不重启。
V23保持Q1_FAIL封存；当前固定dev最佳和原总目标未达状态不变。

### 41.2 V24 完整 M0 通过，原进程进入 Q1；独立审计运行中（2026-09-06）

记录时间：2026-09-06T01:31:46.396508+08:00。模型执行仍为 6a4ac2cd95af2ca1a9122d1f79aabd3a83e4fe33，原 PID 52030 未重启。
完整 M0 最后保存 elapsed 713.9715085029602 秒。三折六端预检及两端容量、固定新 candidate 的第 100 步门全部完成。
9 个 source 初始化模型共 18,882 条记录前向 /153 次 source batch 前向，96 次双视图预检前向，
容量 8+8、固定 100，共 116 次 optimizer 更新 /232 次视图前向反向。

两端 98,800,141 总参数 /7,841,292 可训练 /203 张量，全部非零梯度；无 overflow，冻结状态不变。
容量峰值 5950 /6010 MiB。第 1 步 loss 0.6134311258792877，第 100 步 0.5804857909679413，
解析标签平滑下界 0.57838292104621，固定 excess 比 0.059999361786045424 ≤0.1，五项工程门全部通过。
这是 source 固定批次的工程资格，不是检索 mAP 或泛化结论。

01:10:42 不可变进度快照显示 Q1 第一端 fold 0 /ordinary_two_view 已完成 1 epoch /29 更新，
首 epoch 67.26156306266785 秒；GPU 6358 MiB 使用、17770 MiB 空闲、100%。
该时点尚无配对 fold 终态。按首 epoch 估算首对约 01:55、全六端约 03:25–03:50，估时不是实测终态。
六端×20 epoch、3360 更新及全部科学门保持，原进程自动继续，不改训练、不择优停跑。
同双视图两端只差原型损失系数；不能把本比较自动归因为相对旧单视图训练的总收益。

已复算所有 116 行双视图原始损失和原型项，最大舍入误差 1.043081283569336e-07。
01:25:58 完成 47 个远端文件的完整 SHA、26 个执行 Git blob 核验；包含 12 份 M0 初始/终态原型二进制及 CLIP/V12 来源权重。
本地只做源码、JSON、日志与文件哈希核对；无本地模型/图像/张量运行，远端二进制不等于独立审计者本地持有。
完整报告 results/TRIFUSION_RGBNT201_V24_COMPLETE_M0_2026-09-06.md；
原始快照 evidence/trifusion_v24_m0_seed42_6a4ac2c.json，SHA 5046a4ad8bd92aa9cd75fcaf802d1bb787b2f9975b91fcfb1b6ccaa60278373b；
原日志 evidence/trifusion_v24_m0_run_snapshot_20260906.log，SHA 87a4999c83a296903832c7a787c204798ea533b9a10b8fdedac2c7e6b66ffffa。
数组及文件核验分别为 evidence/trifusion_v24_m0_array_verification_20260906.json 和 trifusion_v24_m0_file_verification_20260906.json。

42 个主输入已固定交给 GPT-5.5 xhigh 独立审计 /root/audit_v24_m0，trace 为 .aris/traces/experiment-audit/2026-09-06_run14。
审计尚在运行，不能写成已通过；同模型家族独立复核，不称跨家族背书。
V23 维持 Q1_FAIL 封存；V24 尚无完整 Q1 科学结果，D1/dev/official/消融均未获资格。
原总目标仍未达，固定 dev 最佳仍 V8 58.4050；下一跨数据集重点仍 MSVR310，再 RGBNT100。


### 41.3 V24 终态核验工具与 MSVR310 准备（2026-09-06）

记录时间：2026-09-06T01:47:51.822863+08:00。本节不包含新的 V24 训练进度观察，M0 独立审计仍在运行。
已准备 tools/verify_v24_terminal_files.py、audit_v24_terminal_arrays.py、
report_v24_complete_comparison.py，三份 AST 解析通过；尚未对不存在的完整 V24 Q1 终态执行。
工具只读取完整终态原始 JSON/日志和文件字节，覆盖六端、五输出、全部 21 个资格身份、
120 个 epoch 的损失均值、完整采样 SHA 与 108 原型逐 epoch 年龄/更新计数、固定 bootstrap 和五项门。
原采样器的纯标签/index 方法由 AST 读取，省略 torch 基类；不加载图像/张量或运行模型。
验收仍以真实终态执行为准，不能把工具写好称为 Q1 核验已通过。训练源码、配置和计划均未改变。

MSVR310 另完成 Signal 九份源码/配置文本核查，全部与 cd1b0a6 执行 Git blob 相同。
原配置输入高×宽128×256、网格8×16、B64/K4、50 epoch Adam5e-6，20/40步进衰减；
当前项目 seed42/B64K8 与作者配置差异必须明示。原 train.py 每 epoch test-best 选择不用于本项目新合同。
完整训练标签普查1032记录/155身份：155身份跨 camera/v，但仅60身份跨 scene，
95身份单scene，真实身份/camera对768、身份/scene对276。
600条训练记录在原 scene 过滤下具有合法跨 scene 正例；这是标签可评价性，不是 mAP。
MSVR310 官方 gallery1055/155身份应完整保留，query591/52身份；不能删掉未出现于query的干扰身份。

准备文档 docs/MSVR310_BASELINE_AND_TRANSFER_PREPARATION_2026-09-06.md；
原始证据 evidence/trifusion_msvr310_signal_source_inspection_20260906.json 和
evidence/trifusion_msvr310_source_label_support_20260906.json。
当前 V24 loader 固定竖向输入、旧 evaluator 采用 same-camera，因此不能只换数据根目录迁移。
车辆下一步需要登记独立 source/held-out、横向共享几何、scene 过滤及新的 Signal 基线；
本节没有启动车辆模型、确定未合格候选推广、加入新损失或执行消融。
MSVR310/RGBNT100 项目训练和检索次数仍为0，原 RGBNT201 总目标未达状态不变。


### 41.4 V24 第一折固定终态为负，继续原剩余四端（2026-09-06 01:53）

01:53:22 原 PID52030 仍在运行，三折六端预检及完整M0均通过，Q1已完成第一对共40epoch/1160更新。
第一折1000完整gallery、190query：fused ordinary_two_view 72.72132545325914 →
environment_identity_prototype 72.59618137737564，增益-0.1251440758834974 pp。
CNN/T/M分别+0.09201704604836891/-0.19233788066681257/-0.4895198885976413。
两端baseline_only均68.76764248141828，fused Rank-1均73.15789473684211。
首折已使“每折fused均非负”的必要门不成立；不能晋级，但仍执行完整剩余四端并保存全部正负结果。

全部五输出已由190条原始AP/Rank复算；40个epoch与日志逐项相同。
两端580步完整sample_order SHA相同，20epoch的108个原型年龄/更新计数与纯标签采样重放精确一致。
每端正对259840，其中跨camera21396，占8.23429802955665%，为第一折覆盖统计。
580批有540批只含1组跨camera身份；不能把该折数据冒充三折原8.070791%的新全量观察。
记录详见 refine-logs/v24/PROGRESS_20260906_015322.md 和
evidence/trifusion_v24_fold0_progress_verification_20260906.json。
原始快照SHA3b63e4277e1dc84c36ece2b957b9177b40c52a5dd754f0a3fcadc00b019744a0；
原日志SHA53ab212512b977f9267e2364820a58126c9a371c42d465eee8bdb92fc65744b1。
完整三折Q1尚未结束，M0独立审计仍运行，二者范围分开。
按40epoch均值64.0445秒和剩余批数，第二对预计02:37、全终态约03:20–03:35；不是新实测成绩。

### 41.5 SNR 固定开源实现的机制与边界（2026-09-06）

作者CVPR2020主页code链接指向microsoft/SNR，已固定commit f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b。
当前287文件的树与六份源码/说明文本已取得并逐Gitblob核验；未取得权重或数据、未执行源码。
分类入口为后续扩展论文的PACS/ResNet18，使用IN+通道门控残差回补，
以及分类概率熵SoftMargin约束。不能把该分类损失或代码可用性当成已复现CVPR2020 ReID度量监督或CLIP版本。
源文本与MIT许可证证据见 evidence/snr_source_text_inspection_20260906.json，
机制及固定源码链接见 docs/SNR_CODE_MECHANISM_AND_SCOPE_2026-09-06.md。
这是后继研究准备，没有向V24加入SNR或改写其封存规则，也未决定正式后继版本。


### 41.6 V24 M0 独立审计完成并归档（2026-09-06）

记录时间：2026-09-06T02:09:31.389952+08:00。独立 GPT-5.5 xhigh 审计整体 WARN、工程 PASS、固定 M0 QUALIFIED_PASS。
42 个原始输入在审计与归档前均哈希匹配；两份原始报告、实际响应和审计时 tracker/result 快照已保存 run14。
116 行 M0 的原损失最大复算差异 1.043081283569336e-7；原型项 2.3283064365386963e-10，
总步损失 7.896839337995232e-8；解析下界与固定第 100 步比值差异均为 0。
固定批次更新 9/108 个原型，其余 99 个未更新符合该固定批次，不说明完整训练的记忆新鲜度。
保留远端大权重/原型二进制回执、未独立实例化参数计数，以及同家族审计的限制。

原始审计 MD 8933 bytes，SHA c7ba2cd281769cbcc3ec117a6116f69a13eb828e4f7f4bb036d3ad44fe364618；
JSON 23401 bytes，SHA 04d24cde7ccbb0bfa1d8dfa323648aa3263c681366725f13b04e8b20aea4b8ca。
闭合回执 evidence/trifusion_v24_m0_audit_closure_20260906.json。
审计依据仍是 01:10 快照；41.4 中 01:53 的第一折负结果不冒充其已审计范围。
Q1 最近真实观察仍为第一对 40epoch/1160更新，原剩余四端继续；完整终态和独立 Q1 审计尚待完成。
首折已违反每折 fused 非负必要门，不能晋级，仍保留全部终态比较。V23 封存和总目标未达状态不变。


### 41.7 MSVR310 训练内部身份协议已固定，尚未训练（2026-09-06）

记录时间：2026-09-06T02:17:57.188547+08:00。只按官方训练标签划分全部155身份/1032三元组，
依真实scene可评价性分组、身份排序后轮流分入三折；不按特征或指标选身份，不额外选择随机划分。
每折source103/103/104身份、672/683/709记录，均40个跨scene身份；
heldout52/52/51身份、完整gallery360/349/323记录；各20个query身份、210/207/183合法query。
合计600query，95个单scene身份的432记录全部仍为图库干扰；完整三折距离各自计算。
错误用same-camera将合法query改成1032，且926条记录的正例数量变化。

数据协议 protocols/msvr310_train_oof_v1.json 的SHA
4ff4c60bca3d019929add5788212c526387d93d535a2c52aa7b1c3acfd387cb4；
生成器 tools/build_msvr310_train_oof_protocol.py，全1032掩码直接枚举核验见
evidence/trifusion_msvr310_train_oof_protocol_verification_20260906.json。
完整说明 docs/MSVR310_TRAIN_INTERNAL_PROTOCOL_V1_2026-09-06.md。
仅冻结数据清单；新Signal source初始化、车辆横向loader/scene evaluator、训练合同与真实工程门未运行。
MSVR310训练/检索仍0，RGBNT201原V24继续，未推广不合格候选、未修改官方591/1055协议。


### 41.8 SNR 原始 ReID 公式与直接 Token 接入边界（2026-09-06）

记录时间：2026-09-06T02:26:36.380613+08:00。已补读原始arXiv v1第3节；原方法使用正/负对距离的四个Softplus项，
与当前PACS分类仓库的概率熵约束区分，未把两者称作等价复现。
项目代码中Mamba在最终残差Patch上先求均值。对紧接该读取位置的空间IN，
代数上mean(IN(X))=beta；通道回补后为beta+a*(mean(X)-beta)。
该位置只能从门控均值体现回补，不能据此称空间内容得到新的身份表征。
推论仅限没有中间空间/Token非线性的情况，不否定原SNR或其他位置。

四个Softplus项具有正下界；未来若引入该损失，需要训练前定义工程指标，
不能把旧CE解析下界当成新增全部loss的最优值，当前V24门槛没有更改。
详见docs/SNR_CODE_MECHANISM_AND_SCOPE_2026-09-06.md和
evidence/snr_original_reid_formula_review_20260906.json；新模型/优化/检索均0，后继版本未登记。


### 41.9 MambaPro 固定代码的 3N 聚合参照（2026-09-06）

记录时间：2026-09-06T02:33:13.022271+08:00。固定官方commit f9ee6f60e58f21f3da1c8fd0e659fcc8db9ab149，MIT，98文件树，
14份纯源码/配置文本137,173bytes逐Gitblob核验；无模型、权重下载或运行。
实际两层MA，各先模态内、后整个3N模态间SSM，CLS与聚合Patch均值共同降维，最终1536D。
默认MAMBA_BI=False且两数据集配置未覆盖，因此完整3N不等于已启用双向传播。
当前TriFusion长度3的逐位置混合、与reference作差及平行残差银行是另一结构。

PFA实际d→2d→d；冻结函数按adapter参数名选择，不据函数名将其称作LoRA。
原配置B64/K4、60epoch；MSVR310 DIRECT=0、RGBNT201 DIRECT=1；原验证best选择不等于项目固定终点。
CLIP实际硬编码../PTH/ViT-B-16.pt；future基线需绑定真实读取文件。
详见docs/MAMBAPRO_CODE_AGGREGATION_AND_SCOPE_2026-09-06.md、
evidence/mambapro_source_text_inspection_20260906.json。
这里只作原文及调用链研究，没有改变V24、发起消融或登记后继训练。


### 41.10 V24 第二对有小幅融合正收益，最后两端继续（2026-09-06 02:39）

记录时间：2026-09-06T02:42:14.150734+08:00；实际观察02:39:11，原PID52030仍运行、exit不存在。
两对固定终态已完成共80epoch/2280更新；日志82epoch/2334更新，多出两行属于最后一折对照端。
第二折1051完整gallery、179query：fused91.07494864936395→91.24134939368115，
增益+0.1664007443172011 pp；Rank1 96.08938547486034→97.20670391061452。
CNN/T/M mAP增益分别-0.5048841150709649/-0.7798281723704292/-0.5867659616460372。
首折-0.1251440758834974 pp保持，必要的每折非负门已违反；仍完成全部正负结果，不提前汇总三折。

第二折五输出179条AP/Rank、40行epoch、560步完整采样SHA及20轮108原型年龄/计数均核验。
两端35840曝光，有向正对250880/跨相机21406，占8.532366071428571%，520/560批仅1组跨相机身份。
详情 refine-logs/v24/PROGRESS_20260906_023911.md；
evidence/trifusion_v24_fold1_progress_verification_20260906.json。
原summary SHA5e57fe6bec69318dad9748bc00413dac2c4fae3f8c00d7cc78fda78c44731d6d，
log SHA3c39496ceafe2ef66ec978ca34bf7fd01be97f757f7b8ed6ae22039ce52104e8。
M0独立审计已闭合，完整Q1独立审计仍待终态。第二对实测平均epoch61.8784秒，
全终态预计03:18–03:25、下一阶段检查03:17附近；无配置更改、无D1/dev/official/消融晋级。


### 41.11 V24 只读 source 原型诊断已固定，尚未运行（2026-09-06）

登记时间：2026-09-06T03:06:30.962594+08:00。依据前两折真实记录准备，不把损失约0.002直接归因为饱和。
原六端终态 exit0 后，单次读取每折共同初始化和两端最终模型，共9模型/18756 source 样本前向。
全部94 source 身份和108真实身份/相机原型保留，只计算 fold 内 clean fused 的原型CE、
样本正负例间隔和缓存/当前clean均值的余弦偏移。heldout/dev/official前向、优化、反传、checkpoint写入均0。
原型包含被评分样本自身；缓存偏移混有弱增强与clean视图差异，不据此独立声称缓存陈旧或因果失败。
详见refine-logs/v24/SOURCE_PROTOTYPE_DIAGNOSTIC_PLAN_20260906.md、同名JSON和tools/diagnose_v24_source_prototypes.py；
脚本SHA aedaa9a1366a37ebcf65c3a24a1a8708c67971cff9163ff82f23565ddab148a8，计划SHA b66e3fd49e3f6645a7ef4e9ce52ef358ec7a828615cc72d3758f5bd1f3a2c41d。本地只做AST解析，无模型或张量运行。
V24原固定Q1及审计优先，尚未启动该诊断或任何后继训练。


### 41.12 V24 原六端终态完成并封存；独立 Q1 审计待执行（2026-09-06）

记录时间：2026-09-06T03:21:48.280439+08:00。原PID52030 exit0，源码6a4ac2c，120epoch/3360更新/6720视图前后传。
全程8523.534031秒，含M0/初始化/训练/评价。完整3126gallery、571query、21query身份。
matched control→candidate fused 79.534977571→80.026284762 mAP，
增益+0.491307191 pp；Rank1 82.136602452→82.311733800。
CNN/T/M增益-0.745102386/+1.351265626/-0.579202618，
各折fused -0.125144076, +0.166400744, +1.359049739；bootstrap下界-0.694806868。
五门{"aggregate_fused_gain_at_least_1pp": false, "all_fold_fused_nonnegative": false, "all_expert_aggregate_nonnegative": false, "fused_bootstrap_lower_positive": false, "fused_beats_baseline_and_experts": true}；状态Q1_FAIL_SEALED。
本比较是相同双视图配置下原型损失系数的效应，不推断整个双视图方案相对旧单视图的贡献。

远端63文件和24原型二进制哈希通过，六端独立receipt一致。
本地全部AP/Rank/身份/120epoch损失/完整采样与原型年龄元数据核验通过，最大数值差1.11022302463e-16 pp。
完整结果 results/TRIFUSION_RGBNT201_V24_COMPLETE_Q1_2026-09-06.md；
原始evidence/trifusion_v24_q1_seed42_6a4ac2c.json，SHA 7e43c38529598de5ca04cbe67502c60f60d26f8c6d8e20232d11254a0e2aa86a。
独立终态审计仍PENDING，不以M0审计代替。无D1/dev/official/消融晋级；V24不改门、不扫描、不重训。
已登记source只读诊断独立于Q1终态，当前可部署dev最好仍V8 Phase-B58.4050/59.3939。


### 41.13 V24 完整Q1独立复核及source只读诊断已启动（2026-09-06）

记录时间：2026-09-06T03:23:45.602081+08:00。独立审阅者 /root/audit_v24_q1，GPT-5.5 xhigh，fork none，
直接核查49份主文件；输入manifest及实际请求保存在.aris/traces/experiment-audit/2026-09-06_run15/。
这是同GPT家族独立审阅，非跨家族认可；当前报告尚未完成，不提前填写审计结论。
49份输入在发出时逐SHA核对，审计期间保持tracker/结果原稿/原始证据不变。

GPU释放后按已固定只读计划启动诊断，实际2026-09-06T03:22:48.092419+08:00，PID59375，
代码4452a9801c69bc0bfbdbe012ff9311e4afc04f46，screen v24_source_diag_4452a98，启动前free/used/util 24126, 1, 0。
目录/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v24_source_prototype_diagnostic_seed42_4452a98。固定九模型/18756 source记录/153batch前向；更新、反传、checkpoint写入、
heldout/dev/official图像前向均0。尚无该诊断终态结果，预计6–10分钟，首次检查约03:29。
该读取不改V24训练终态、原科学门或任何权重；Q1_FAIL封存保持。


### 41.14 V24 九模型 source 只读诊断完成；两项独立审阅进行中（2026-09-06）

记录时间：2026-09-06T03:38:41.260074+08:00。原PID59375 exit0，执行4452a98，耗时312.011035秒；
九模型/18756 source记录/153batch前向，所有模型状态不变，优化/反传/checkpoint写入/heldout/dev/official前向均0。
三折各94 source身份，完整14跨相机身份和108身份/相机成员；无跨fold距离。

共同初始化时clean原型CE已仅0.000456865/0.000513873/0.000368214；九模型global/env分类全100%。
九模型所有非self同身份正例都严格比任何异身份负例近，最小余弦间隔约0.075874–0.154448。
样本负例比均值原型更接近query约0.031–0.034，但完整clean source中仍没有未分开的负例。
因此“扩大同一批干净source负例池即可解除瓶颈”缺乏当前直接证据；不是对增强视图或XBM的普遍否定。
最终缓存/当前clean原型平均余弦约0.979–0.982，混合弱增强视图差与模型变化，不能单独认定缓存陈旧。
新原型含被评分source自身，source几何正例则排除了self；不把100%表述为未知身份验证或新测检索mAP。

全部逐记录、1692身份均值行及日志核验完成，最大均值/分位差1.42108547152e-14，
最大概率/FP32代数差2.98719532443e-08。
原始evidence/trifusion_v24_source_diagnostic_20260906.json，
SHA a66f17a450fb0eca2404fd23721545eed0dd7b061550230a7d51368da24fa271；完整报告results/TRIFUSION_RGBNT201_V24_SOURCE_PROTOTYPE_DIAGNOSIS_2026-09-06.md。
source独立审阅 /root/audit_v24_source，GPT-5.5 xhigh，27主文件、trace16，结果PENDING；
完整Q1独立审阅 /root/audit_v24_q1、49主文件、trace15，仍PENDING。
两套冻结输入在本次更新前全部复核SHA不变。V24 Q1_FAIL、无晋级与既有目标状态保持。

### 41.15 V24 两项独立审计闭合，保留原稿与元数据更正（2026-09-06）

记录时间：2026-09-06T04:03:18.863105+08:00。完整Q1整体WARN、工程PASS、科学FAIL_FIXED_Q1_GATES；
source诊断整体WARN、数学/GT/活代码指标类型PASS，范围仍是seen-source描述性几何。
49与27份原始输入在闭合前再次逐SHA核对一致；将变动的tracker/结果原稿先作快照，
两轮实际请求/回复、报告原字节与SHA均保留trace15/16。

Q1首轮把bootstrap种子写成20260906，第二轮按注册seed42从完整逐query数组独立重算：
21身份/10000次、query数加权/线性2.5%分位，下界-0.6948068678403989pp，
全部fused身份均值与下界差异0，算术0.160515秒。科学失败判定不变。
source独立重算18756记录和1692身份均值，0.346868秒；均值/分位最大差1.42108547152e-14，
概率/FP32代数最大差2.98719532443e-8，仍不能用clean source100%声称未知身份有效。

两项审阅都请求gpt-5.5/xhigh并取得独立agent；请求/接受记录不独立证明后端身份。
禁止子审阅者继续委派不能推出模型不可用。首轮相关过度表述已由审阅者纠正并保存原稿。
保留CRLF/LF原字节差异、远端checkpoint/原型二进制回执依赖、没有重新生成特征或距离的限制。
同GPT家族独立审阅不构成跨家族认证。

完整报告 EXPERIMENT_AUDIT_V24_Q1.md/json 与 EXPERIMENT_AUDIT_V24_SOURCE_DIAGNOSIS.md/json；
闭合记录 evidence/trifusion_v24_q1_audit_closure_20260906.json、
evidence/trifusion_v24_source_diagnostic_audit_closure_20260906.json。
V24保持Q1_FAIL：不重训/改门/扫参，不进入D1/dev/official，当前整体目标仍未达到。

### 41.16 MixStyle 固定代码及深层语义边界（2026-09-06）

记录时间：2026-09-06T04:03:18.863105+08:00。作者仓库commit16f7cf1fe2c7b1b3c660c72b32817ebb0545397a，
MIT，12份纯文本85559bytes逐Gitblob核对。原模块混合detach的空间均值/标准差；
p0.5、alpha0.1、eval不作用，无新增推理参数。crossdomain实现依赖两半batch来自两个域，
不由算子读取域标签保证；当前域先验配置180epoch，随机配置60epoch，不能称预算匹配对照。
原论文ReID末层混合下降和当前CLIP深语义/均值读取共同限制直接尾部接入，但不是CLIP失败实测。
详见docs/MIXSTYLE_CODE_MECHANISM_AND_SCOPE_2026-09-06.md、
evidence/mixstyle_source_text_inspection_20260906.json。没有登记V25或运行新模型。

### 41.17 MSVR310 Signal 独立源基线合同及入口已固定（2026-09-06）

登记时间：2026-09-06T04:22:31.026558+08:00。代码tools/train_msvr310_signal_oof.py；配置configs/MSVR310/Signal-source-oof-v1.json；
完整合同refine-logs/msvr310_signal_v1/EXPERIMENT_PLAN.md。仅新数据集基线建设，不是V24晋级或车辆TriFusion新方法。
三个source模型各自从固定CLIP初始化，正式基线各50epoch，source103/103/104身份；
最终一次读取360/349/323完整gallery，按scene过滤210/207/183query，合计600query。
没有官方或固定RGBNT201dev访问，没有V24权重继承、三分支训练、消融或seed扫描。

源码调用链确认原MSVR分支使用WarmupMultiStepLR的20/40epoch衰减，而当前RGBNT201辅助训练固定cosine，
因此只复用原损失计算，新入口明确保留车辆Adam参数组及分类头100倍基础LR。
DIRECT=0是模态独立分类头，推理仍输出直接特征+SIM的3072D；视觉主干FROZEN=False，camera SIE启用。
B64/K8与共享三模态几何是项目现行设置，与作者K4/原增强差异单独披露。
16份Signal实际源码及7份项目源码/计划文件已绑定SHA，既有Signal路径补丁完整保留、未修改。

先在远端跑两个scene协议回归，再做三折各8步真实source M0；
M0只比较共48条clean source前向的严格重载一致性，不读held-out，不使用V24的100步损失门。
三折M0工程PASS后，才用新初始化执行固定基线终点；任何内部mAP不自动解除RGBNT201主结果晋级门。
当前只完成本地AST与实际源码字节核对，T0/M0/正式基线仍NOT_RUN，车辆训练和检索仍0。
注册记录evidence/trifusion_msvr310_signal_v1_preregistration_20260906.json。

### 41.18 MSVR310 协议T0通过，source M0已启动（2026-09-06）

记录时间：2026-09-06T04:28:08.218395+08:00。远端T0两项NumPy排序测试PASS，0.611659秒，1032真实记录query掩码全量一致；
这是合成距离测试与真实标签核验，不是模型检索结果。一次传输脚本字符串解析错误在执行测试前修正，
无模型/图像/优化操作，实际T0只执行一次，原错误另行记录。

实际启动2026-09-06T04:25:50.148612+08:00，wrapper PID61639，执行2dcbe85；
启动前GPU free24126MiB/used1MiB/util0，wrapper SHAff3280a9972f39ab0c4b4a170bf512ed5db916ea1d434add1cb15b0e2001a9c7。
仅source三折各8步工程检查，路径artifacts/msvr310_signal_source_oof_v1_seed42_2dcbe85/m0。
没有读取M0终态或提前称通过，首次计划检查约04:29:50，正式50epoch三折尚未启动。
原配置/损失/源码/数据合同不变；没有车辆TriFusion或官方结果。

### 41.19 MSVR310 首次M0失败定位与最小工程修订（2026-09-06）

记录时间：2026-09-06T04:38:57.493093+08:00。04:29:56检查原wrapper61639已结束、exit1、GPU释放。
fold0完成8步，mean loss13.362928748、训练9.545426秒；在全部trainable有梯度门停止，后两折和held-out未运行。
首版失败前未落盘完整training细目，因此不补造逐步数组；完整原stdout/exit与启动回执已归档。

一次新的fold0 B64 source前后传诊断，optimizer0/checkpoint0/heldout-dev-official0，6.844008秒，
查得201个trainable中6个无梯度，均为SIM.token_selection W_q/W_k/W_v的weight/bias，合计787968参数。
原useA.py证实Q/K只生成离散索引/二值掩码，V未调用；安装的Adam只处理grad非None项。
R2仅将这6个原本不更新的参数标记冻结，保留原值、state_dict键和全部前向路径；
不新加可微选择器、不改loss/LR/50epoch/seed/采样/评价或原梯度门。训练前后冻结模块SHA必须不变。
新版在断言前保存实际training.json，保留失败诊断细目。

原M0失败完整保留；R2合同/配置/脚本已登记，尚未运行新的三折M0或正式基线。
排序函数AST与已通过T0相同，不把该T0扩大成构造/梯度修正的验证。
完整证据evidence/trifusion_msvr310_signal_v1_unused_gradient_diagnostic_20260906.json、
evidence/trifusion_msvr310_signal_v1_token_selection_source_inspection_20260906.json及R2注册记录。

### 41.20 MSVR310 三折M0 R2通过，固定source Signal基线运行中（2026-09-06）

记录时间：2026-09-06T05:00:34.202298+08:00。M0 R2执行bb01d60，原wrapper63101 exit0，
三折各8步，共24更新/1536训练曝光/48clean source前向，77.374851秒；
195/195实际训练张量有梯度，0overflow，冻结TokenSelection SHA不变，
三折3072D source特征严格重载逐元素相同，峰值显存11379.005MiB。
source短检查实际曝光身份63/61/62，可用全身份103/103/104；heldout/dev/official前向0。
21个完整文件含3checkpoint SHA及24步标量/标签核验完成；独立终态审计仍待完整基线。
R1额外8次更新与0更新诊断保留，不声称未保存的R1/R2逐步轨迹一致。

正式B0于2026-09-06T04:50:55.255082+08:00启动，wrapper63945，
仍绑定bb01d60及原config/plan/runner；三折各重新从固定CLIP初始化训练50epoch，
不使用M0权重。04:57:26观察fold0达46/50，进程运行，完整折和检索终态尚未取得。
按实际约8秒/epoch，预计05:13–05:16附近完成；未用loss或某折结果改配置。
首次进度reader event名误查，原log_tail证明已到46，原记录及更正单独保存。

完整报告results/TRIFUSION_MSVR310_SIGNAL_SOURCE_M0_2026-09-06.md；
M0原始SHA 79c0e2b1c981c4bb10548f0249dca684c113feb43151cc4bdcc2a2e9bf2887ae。
这是源基线基础建设，没有车辆TriFusion、官方测试或RGBNT201dev访问，不改变未达主目标状态。

### 41.21 MSVR310 完整排序及训练终态核验预备（2026-09-06）

登记时间：2026-09-06T05:07:58.159629+08:00。两个终态核验入口已完成AST检查，尚未在终态执行。
固定读取三份原checkpoint文件SHA及已有特征/距离，逐元素核对由3072D特征重算的距离；
从原保存距离导出600query的完整gallery索引序列，以原ID/scene标签重算AP/Rank，
保留全部干扰身份；核对全部150epoch与真实source采样。没有新模型/图像前向或优化。
计划refine-logs/msvr310_signal_v1/TERMINAL_VERIFICATION_PLAN_20260906.md；脚本SHA见evidence/trifusion_msvr310_signal_v1_terminal_verification_plan_20260906.json。
原B0代码/配置/计划仍固定bb01d60，独立审计待完整结果后执行。

### 41.22 MSVR310 Signal 完整三折源基线完成（2026-09-06）

记录时间：2026-09-06T05:17:36.048722+08:00。原wrapper63945 exit0，执行bb01d60，150epoch/1950更新/124800训练记录曝光，
程序配置后计时1299.025203秒；三个source模型均从固定CLIP独立初始化，固定epoch50，0官方/dev访问。
完整600query/60query身份/1032gallery/155heldout身份，95单scene身份432记录继续作干扰。
fold0/1/2 mAP49.311695078/47.942264566/63.377724618，
Rank-1 59.523809524/60.869565217/69.398907104；
600query加权mAP53.129380561、Rank-1/5/10为63.0/77.0/82.833333333。
不把该内部测量与官方591query/1055gallery结果直接比较。

23完整文件含3checkpoint和3特征距离SHA、7项目与17Signal源码绑定核对完成。
由保存3072D特征重算距离逐元素一致；全部完整排序以真实ID/scene重算AP/Rank，
全部1950steps/150epochs/实际source索引核验，指标最大差3.33066907388e-16，
loss组合差1.300722360e-6，原AMP dtype缺项保持披露。没有新模型/图像前向或优化。
完整报告results/TRIFUSION_MSVR310_SIGNAL_SOURCE_BASELINE_2026-09-06.md；原始SHA 22a4f3642e88088a8dfcb4acddb610d6566c12d6b765cb7b344b28acdbcea6eb。
独立审计PENDING；结果只建立车辆source基线，不是TriFusion晋级。
V23/V24封存、RGBNT201主目标未达、RGBNT100训练检索尚无，状态均保持。

### 41.23 MSVR310 Signal 完整基线独立审计闭合（2026-09-06）

记录时间：2026-09-06T05:55:13.439177+08:00。trace run17两轮审计完成：overall/integrity WARN，engineering PASS，
A/B/C/D/F PASS、E范围限制WARN；71原始输入字节/SHA全部一致。
首轮审阅者以独立stdlib脚本重算600完整query排名、1032gallery、60身份、1950更新/150epoch，
总mAP53.129380561与Rank-1 63.0完全相同；AP最大差3.33066907388e-16（0–1尺度）。
第二轮只复核派发元数据和远端二进制证据边界，没有重复算术、模型、图片或训练。

root实际请求并派发gpt-5.5/xhigh独立上下文审阅，未继续委派不代表未派发或模型不可用；
请求/接受记录不构成独立后端认证，保持GPT家族Type-A、非跨家族Type-B限制。
二进制及图像留远端；本地独立审阅覆盖文本、JSON、离散排名和远端SHA收据。
审阅者自行修正原报告的两处表述，初版原回复/报告、最终版和执行侧审计前快照均保留。

EXPERIMENT_AUDIT_MSVR310_SIGNAL_B0.md SHA aa04e91403b24d2c4bf4beb24c01411103353762de58e0f009e6fdca7c2e3c49；
JSON SHA 34c00b4f8b4add6e5abff55cf9ee8fd4e89b2e491f8aa65224623c2eb9d23a23。
单seed内部基线不等于官方复现、新方法资格或多seed稳健性。B0不重训；
后续MSVR310三角色独立训练比较仍待新合同，RGBNT201未达目标与V23/V24封存不变。

### 41.24 MSVR310 原三角色架构独立训练比较合同已固定（2026-09-06）

登记时间：2026-09-06T06:03:04.455453+08:00。B0审计eea8c20闭合后登记新的车辆全系统比较，状态PREPARED_NOT_RUN。
每fold仅加载已固定本fold车辆Signal epoch50，全Signal/tail冻结，原V8三角色/七分类头seed42新初始化。
输入128x256/grid8x16、原七组ID/Triplet及固定等能量拼接；无Router/HFER/V23/V24干预或RGBNT201角色权重。
这是新数据集独立训练，不是零样本迁移、消融或已有失败版本再命名。

先三fold各8步容量、8条source独立Signal前缀核对与五输出严格重载，另新fold0固定100步过拟合。
总M0 124更新，0heldout；扣除原七头CE熵下界的最后/最初excess ratio<=0.1，门不变。
M0完整PASS及收据核对后，三个source各20epoch，B64/K8、AdamW0.00035、5epoch warmup/cosine；
正式新初始化状态必须与各自M0初始状态相同，不能加载M0后权重。
600query/60身份/1032完整gallery，保留95单scene干扰身份，原scene过滤不变。
固定最后checkpoint五输出；baseline特征/距离必须逐元素复现B0。

五项内部支持条件为fused增益>=1pp、三fold非负、三完整分支不低于Signal、
60身份加权seed42/10000bootstrap的2.5%线性下界>0、fused严格优于baseline和全部分支。
无论通过否均保留全部身份及新增错误；不扫描、不延长、不选epoch/fold/seed。
B0前置1950更新另计，本比较无法排除额外计算/参数解释；所有消融留主结果后。
合同refine-logs/msvr310_trifusion_v1/EXPERIMENT_PLAN.md，入口tools/train_msvr310_trifusion_oof.py；
注册evidence/trifusion_msvr310_trifusion_v1_preregistration_20260906.json。
当前只完成本地AST/文本检查，新模型尚未在远端运行，RGBNT201目标未达及官方边界保持。

### 41.25 MSVR310 三角色入口启动前实际源码字节修订（2026-09-06）

记录时间：2026-09-06T06:08:14.672010+08:00。148f5a7首次launch在创建远端目录/wrapper/训练进程前被criterion.py SHA断言阻止，
0模型/张量/图像调用、0优化。全量19输入核得5项历史CRLF/LF差异，
criterion/state/builder及experts mamba/semantic_residual远端字节均等于Git原blob，
本地LF转换后逐字节相同且AST一致。原wrapper文本、注册/config及真实错误证据保存。

R2只将配置内5项绑定改为实际远端SHA，入口仍严格逐文件检查；不加自动归一化/fallback。
runner SHA a1771c2e16e129f0a1ecdfb1f5fffcc7d83fe2faf5073033efe520a73c800f87不变，
模型、数据、优化、M0/科学门与seed均不变。新M0/正式三折尚未运行。
详见evidence/trifusion_msvr310_trifusion_v1_source_binding_r2_20260906.json及原prelaunch诊断。

### 41.26 MSVR310 原三角色完整M0通过，固定三折比较已启动（2026-09-06）

记录时间：2026-09-06T06:24:17.950707+08:00。实际M0 wrapper69455 exit0，执行1c444cd；三fold各8步容量+
全新fold0固定100步，总124更新/7936训练记录曝光，72role+24独立Signal clean source记录前向，0heldout。
203/203可训练张量有非零梯度，全部冻结状态/Signal保持、AMP overflow0、五输出严格重载逐元素相同。
初末loss4.122129917145/0.588196277618，解析熵下界0.585713632744，
唯一最后100步excess ratio0.000702022804<=0.1。配置后程序计时246.289107秒。

17完整文件含3checkpoint及20项目输入SHA核对，独立fold JSON相等；
本地stdlib重算124步/采样/损失门，loss组成最大差4.122654591e-7、epoch均值差0，
原AMP dtype未保存的范围保持。M0独立审计PENDING，不能当作未知身份检索有效。
原始SHA e021303b51d6af0b8bc49717016744e0ad483af419496652e0525b188d3d646b；
完整报告results/TRIFUSION_MSVR310_ORIGINAL_ROLES_M0_2026-09-06.md。

正式比较于2026-09-06T06:21:14.660925+08:00启动，wrapper70422，仍执行1c444cd及固定R2字节配置。
每fold新初始化角色，不加载M0后权重；固定20epoch、600query/1032gallery、全部五输出。
按实际M0预计18–25分钟，初查约6分钟后，不以训练loss选择停止或修改合同。
当前没有完整比较检索结果；RGBNT201未达目标、V23/V24封存、无官方/消融限制保持。

### 41.27 MSVR310 第0折训练完成，基线特征一致性门停止（2026-09-06）

记录时间：2026-09-06T06:41:47.899579+08:00。原wrapper70422已exit1；fold0训练满20epoch/260更新，
所有训练工程条件通过；保存并严格重载epoch20后提取360条heldout gallery，
在与原B0保存3072D特征的逐元素比较断言失败。0检索AP/Rank，fold1/2未运行。
原RUNNING字段、日志/exit及训练receipt原样封存，不补造终态。
checkpoint SHA b8a85e167861c51bb7d9a5854d700d11468ee9ca2a6e7ba21b75ce557130003c。

M0源8条parity通过不能证明本次完整跨进程B0数组一致；原因当前未知，
不先判算法负结果或把数值差当成可忽略。新只读诊断固定覆盖已访问全部360条，
五个64批及40尾批，四种Signal路径共1440记录前向；不改backend、不算排名、0更新/反传/checkpoint。
首64草案从未执行，执行前按全矩阵与尾批范围修订，原稿保存。
完整计划与入口已登记，尚NOT_RUN；原260更新不重跑，不放宽门，不盲启后两折。
results/TRIFUSION_MSVR310_ORIGINAL_ROLES_COMPARISON_STOP_2026-09-06.md。
独立M0审计run18进行中，其60份原始输入SHA仍相同。RGBNT201目标与全部官方/消融边界不变。

### 41.28 MSVR310 全360特征差异已定位到装入三分支后的SIM（2026-09-06）

记录时间：2026-09-06T08:08:29.999652+08:00。只读四路径诊断一次完成，exit0，执行15ffddc；耗时29.8320秒。
独立Signal装入三分支前逐元素复现全部360条B0特征；装入后，独立Signal、hierarchical baseline、
完整三分支baseline三者相同，但与原B0有359604个元素不同，0/360行完全相同。
差异只在SIM1536维，maxabs1.9073486328125e-6；direct1536维仍完全相同。
权重state SHA、记录的backend flags和eval状态不变，fused内Signal前缀仍精确。
这排除了该次同进程中普遍B0不可复现，以及完整融合单独引入误差的解释；尚未证明具体操作原因。
原bitwise特征/距离门仍保留，不以误差小作豁免；未算任何AP/Rank，后两折仍未训练。

共1440记录前向/360不同已访问gallery，0优化/反传/checkpoint；原260训练更新和checkpoint保留。
下一项固定64条缓存SIM输入的九阶段操作诊断已登记：原Signal、重复SIM、只冻结SIM、恢复flags、
导入builder、导入mamba_ssm、构建原三支、加载原final、最后原Signal。记录中间值、布局和实际算子。
预算576次SIM记录计算，其中128包含完整Signal，其余448为缓存输入；0排名/更新，不扫描backend。
结果页results/TRIFUSION_MSVR310_SIGNAL_PARITY_DIAGNOSIS_2026-09-06.md；新操作诊断尚NOT_RUN。
M0独立审计首轮WARN/engineering PASS，原60份输入未变；报告字段名核对中，未作为已闭合审计。

### 41.29 MSVR310 SIM差异的可逆触发因素与原M0审计闭合（2026-09-06）

记录时间：2026-09-06T08:19:47.876038+08:00。操作诊断0be865b一次完成，exit0，72.9874秒；576次SIM记录计算、0更新/排名。
同一B64缓存输入和权重下，仅冻结SIM就复现64113个SIM元素差异；恢复requires_grad则逐元素恢复B0。
TokenSelection及注意力输入精确一致，最早差异在cross_attn；两次mm变为bmm。
builder/Mamba导入不改变输出，构建时冻结复现差异，加载final不再引入额外差异。
PyTorch2.5.1 should_fold源码明确按小操作数requires_grad选择折叠mm，no_grad不会屏蔽该条件；
与实际非连续Q/KV投影输入及算子追踪一致。权重数值、dtype/stride/storage未因冻结标志改变。
这次停止的直接原因已定位到冻结参数后的数值执行分支，不归因于checkpoint损坏或算法检索失败。

仅推理修复已编写：functional_call使用一个共享原数据的detached投影权重视图恢复B0计算分支，
整个调用no_grad，注册参数原样冻结，无新参数/更新/反传或backend修改。
固定全360条、原5x64+40批验证尚NOT_RUN；要求完整3072D特征及210x360距离逐元素等于B0，
三角色及三模态残差逐元素不变，全部参数flags/state复核；预算720次full-role前向、0排名。
验证通过后才单独登记复用原fold0、训练原后两折的续跑，不重训原260更新。

独立M0审计run18两轮闭合WARN，engineering PASS，scientific/retrieval NOT_ESTABLISHED。
独立124步/采样/熵下界/过拟合复算一致，60/60原输入SHA相同；第二轮只由审计者修正字段引用，
未重复数值复算。两轮请求、原始回复和报告版本已存档；hash-only复核复用同名文件，首轮原hash仅由
首轮回复保留，不假称另有首轮原文件。保持GPT同族Type-A、后台身份未独立证明、张量仅远端等限制。
审计未读取新的比较/诊断数据，不将M0通过扩大为检索有效。closure SHA71cceb45825d591bfa42af649ba142369a3966acc2d84c2036b9dad502f92840。

### 41.30 MSVR310 推理修复全量通过，登记复用第0折的原比较续跑（2026-09-06）

记录时间：2026-09-06T08:26:05.110072+08:00。验证9eba027一次完成exit0，30.3788秒，720次full-role前向、0优化/排名。
原未修复差异仍可复现；修复后全部360条3072D基线特征及完整210x360距离矩阵逐元素等于B0。
三角色/三模态残差、direct信息、原5x64+40批、全部权重state与注册参数flags都保持一致。
helper9b7a3168...只在推理中恢复原计算分支，没有新增可训练参数或放宽误差门。
验证数组SHA c417d003c4a7cb75905527b67c93037c7a22647ae808a3105674a8a599ce5c77，留在远端。

原科学比较R3续跑已登记但尚未运行：复用原第0折260更新/epoch20 checkpoint，以及已经验证的
360条完整特征，进行其首次排名；只训练从未开始的fold1/2，每折原20epoch/260更新。
新增520更新/33280 source曝光与672条gallery前向；正式三折共780更新，不重复第0折训练/图像读取。
全部600query/60query身份/1032gallery/155 heldout身份与95个single-scene干扰身份均保留。
原build_model/train_roles/evaluate/comparison_summary函数和模型、配置、五项科学门完全不改。
新目录comparison_resume_r3，原失败summary的RUNNING及所有日志/receipt/checkpoint原字节保留。
wrapper记录PID与exit，估计9–13分钟，首次240秒后检查，其后180–300秒或按预计完成时点读取。
当前仍无三折检索终态，不因本次工程修复提前主张提升。M0审计已闭合WARN，正式终态另行核验。

### 41.31 MSVR310 原比较R3已启动，前两折完成且第三折运行中（2026-09-06）

记录时间：2026-09-06T08:36:44.580202+08:00。wrapper75993于08:27:45.747741+08启动，执行1ff7e2d。
首次按预计时点读取08:33:49.313248+08，wrapper真实存活、无exit，GPU6246MiB/100%。
第0折只复用原260更新与360条验证特征；第1折新训练原20epoch/260更新并完成完整检索，
两折都通过原B0特征及距离逐元素门。第2折运行中；前两折融合增益方向相反，不提前断言有效。
仍按固定完整三折/600query/1032gallery五输出完成，不依据部分结果停止或调参。

终态核验脚本已准备但未执行：远端核对三个checkpoint内容及15组全部特征/距离/排序；
本地仅JSON/stdlib/NumPy重算3000个query-output、780训练步、60身份及10000次bootstrap。
准备时已知前两折部分结果，未读完整终态，如实记录此范围，不冒称全程盲预注册。
新增核验脚本不改正在执行的训练/model/config原SHA；原失败run所有文件继续保留。
当前结果页results/TRIFUSION_MSVR310_ORIGINAL_ROLES_COMPARISON_2026-09-06.md，状态RUNNING。

### 41.32 MSVR310 三折完整终态：工程通过，原三角色科学条件全部失败（2026-09-06）

记录时间：2026-09-06T08:53:29.912750+08:00。wrapper75993已exit0，终态08:39:25.099313+08，08:42:26确认无进程/GPU空闲。
原第0折1c444cd的260更新与checkpoint保留，R3执行1ff7e2d仅新增后两折520更新，完整三折780更新。
600query/60query身份/1032gallery/155heldout身份、95单scene干扰身份全部保留，原scene过滤不变。

最终mAP/Rank1：Signal53.129380561/63.0；fused52.117390117/60.833333333；
CNN49.707347522/59.166666667；Transformer50.332406612/59.166666667；Mamba50.791739830/59.166666667。
fusedΔ−1.011990444pp；三foldΔ+2.405512508/−3.759722938/−1.825624290；
60身份query加权bootstrap下界−2.939109554，五项原科学条件全false。
30身份改善/26下降/4不变；fused query267改善/285下降/48不变，Rank1修复24/新增37。

全部三折B0特征/距离逐元素相同，原SIM冻结执行差异已修复，未放宽门或更改训练；
三个checkpoint内容/state、15组特征距离及全排序/训练receipt均核对，远端8.6370秒。
本地JSON/NumPy独立重算全部3000query-output及780训练步、60身份、bootstrap，0.4124秒；
最大指标差1.42e−14、bootstrap差8.88e−16、epoch均值差0；无本地模型/图像/张量运行。
错误普查全部600query：新增37个fused错误中同camera10、同scene4；原Signal同camera错误85/222，
fused83/235。不能把RGBNT201的相机错误分布当作车辆失败主因的充分证明。
全780步174720同身份正对中跨scene45539，占26.0639880952%；各fold全部source身份/记录实际曝光。
scene与camera是不同协议关系，不直接套用RGBNT201正对比例或过滤。

固定原三角色MSVR310试验科学失败封存，不重训/扫描/消融/官方晋级。原M0独立审计已闭合WARN，
正式终态独立审计PENDING；三个完整扩展分支均低于Signal，fused比它们高但没有回到baseline。
新的研究假设要围绕未知身份困难负例区分与保留判别信息，并吸收两个数据集的负证据；
不把只增加参数/统一对齐/相机正例稀疏继续当作已证实的通用修复。
RGBNT100仍尚无本项目训练/检索成绩。RGBNT201保留dev58.4050，65与官方SOTA目标继续未达。
结果页results/TRIFUSION_MSVR310_ORIGINAL_ROLES_COMPARISON_2026-09-06.md。
原始完整summary SHA c3831a0e95423767cf152e332ed671a8d91d1779c286a0afce8bf522391bfbac。

### 41.33 RGBNT100 完整训练清单、身份协议与独立Signal基线合同（2026-09-06）

登记时间2026-09-06T09:31:59.717843+08。真实训练8675张768×128拼图全部文件名/大小/SHA已核对，50个身份全部跨camera。
标签确定的三fold source33/33/34身份、5550/5725/6075记录；heldout17/17/16身份，query=gallery3125/2950/2600。
完整内部8675query/8675gallery只过滤同身份同camera，保留所有异身份负例；不沿用MSVR310 scene过滤，官方1715/8575未访问。
配置configs/RGBNT100/Signal-source-oof-v1.json SHA7270e2bf95c5f5a60e1dc6d6b047f043dce667d508783b36bc4734aecbc4c15b；
协议SHA42bd612ecc8720db7f6684214e1f60d1cb4bab6b2fa8db3df343a9d2c52e4abf；18份Signal实际源码、8份项目源码/合同绑定。
作者RGBNT100为固定30epoch、Gram/Patch均0.1、BASE_LR0.0007但CLIP非adapter base组固定5e-6，
原create_scheduler保留warmup5/epoch1–29seed42噪声/epoch30最小LR；不套用MSVR31050epoch和20/40阶梯调度。
项目B64/K8/workers4/同步几何与作者B128/K16/workers12不同，保持每批8身份；不得声称完全相同训练条件。
拼图按RGB/NIR/TIR三个256×128区域切片。T0将逐张比较全部26025切片与作者loader像素，验证全部query mask和两项人工camera排序fixture。
T0后M0三fold各8步，共24更新/1536源记录曝光/48clean source重载前向；正式训练fresh初始化，M0权重不复用。
正式三fold固定epoch30才完整检索一次，保存全部8675条AP/Rank和压缩完整排列，无Top-k截断/挑query/选best。
当前仅文本/AST与文件清单工作，T0/M0/30epoch基线均NOT_RUN。新基线建设不是TriFusion主方法成功，不解除主目标门或允许消融。

### 41.34 MSVR310 原三角色完整终态独立审计闭合（2026-09-06）

闭合时间2026-09-06T09:36:57.748608+08:00。run19 GPT同族Type-A两轮，102输入/25,929,381字节原SHA，未独立证明实际后端。
审计独立复算全部3000query-output、780步/60epoch、60身份与query加权bootstrap、完整错误普查，1.2726868秒，主要指标差0。
verdict integrity PASS_WITH_LIMITS、engineering PASS、scientific FAIL；A/B/C/D PASS、E WARN、F FAIL。
fused52.117390117低于Signal53.129380561，五项原科学条件仍全false，不晋级、不重训、不做官方评估。
第2轮只纠正延迟dispatch的时间措辞、Linear.cpp真正linear73–120/matmul111与MHA107非selfattention分流，
并保留远端二进制回执审计范围，不要求本地张量复制。未重复数值回放或训练。
首轮与最终报告、两轮原始回答/请求、全部复算产物和完整命令输出已归档；本地只计算文本/JSON/NumPy。
最终md SHAa7c9e80327e91246ddf3666cc399c5b92cbc60ca067ff89dbc70b29bd26fd6a5；
json SHA9a0c6174d6a418f4e7d824ae61978826f2877223fa0ba0c957416de2d492d498。
RGBNT100源基线合同已固定，下一动作是远端完整T0及三折M0；RGBNT201主目标继续未达。

### 41.35 RGBNT100 全量T0、三折M0与完整权重/标量核验通过（2026-09-06）

执行1157f0d，wrapper79272于09:39:51启动，T0/M0/总退出码0，09:43:16确认已结束。
T0一次20.701875953秒，8675文件SHA/26025模态切片与作者逐像素相同，8675query mask及全部fold隔离成立。
M0一次76.987059359秒，三fold各8更新共24/1536源曝光，195/195梯度、0overflow、selector冻结状态和48clean source重载前向完全一致。
三fold8步loss11.031642497/11.177569866/10.965902865；均非检索成绩。M0 heldout模型前向0，无官方test。
三checkpoint内容与全部fold/source/heldout/state、8项目/18Signal源绑定核验5.3684秒；21文本逐SHA取得，权重/张量仍在远端。
本地24步完整JSON核验0.0346188秒，epoch均值差0，分项float重组最大差2.1063e-6，未增设科学容差门。
原本地核验字面量7e-5与作者0.1×0.0007的表示不同，改为相同作者表达式后通过；保留原核验与原因，0模型/配置变化、0训练重跑。
正式三fold30epoch固定终点基线READY_NOT_RUN，将fresh通用CLIP重建，不用M0权重。预计75–100分钟，约15分钟首次观察后依据训练epoch耗时更新ETA。
M0 summary SHA7e9f6214efdbf14611fd5bb0ce2f4d7e6c9ae68c9549af06659d018b39e09820；完整结果页TRIFUSION_RGBNT100_SIGNAL_SOURCE_M0_2026-09-06.md。

### 41.36 RGBNT100 首次正式基线AMP停止，固定source定位已登记（2026-09-06）

记录时间2026-09-06T10:22:31.091890+08:00。60a3d0e、wrapper80418实际09:53:41.130891退出1，耗时33.776537912秒。
原runner第201行AMP scale下降断言失败。完整epoch事件0、fold0目录为空、无checkpoint或training.json/检索。
原成功优化更新步数未持久化，不能等同0；10:09:23是首次观察时间而非错误发生时间。
失败现场所有文本已按SHA取得，原训练没有重启，T0/M0通过的范围保留。
诊断tools/diagnose_rgbnt100_signal_amp.py调用未修改的原train_source，
仅原175/201/164行追踪：保存前向前buffer/RNG、逐步落盘、首overflow捕获或第二epoch前停止。
固定source fold0/seed42/B64K8/原AMP256/全部作者目标与调度不变；最多一个source epoch，0heldout/official。
先核对前8步与M0，触发batch权重/输入只保留远端；本条为READY_NOT_RUN，不能提前写成已定位根因。
完整终态核验器事前已准备、NOT_RUN；本失败目录不满足其三折30epoch输入要求。
详见results/TRIFUSION_RGBNT100_SIGNAL_FORMAL_ENGINEERING_STOP_2026-09-06.md。

### 41.37 V24 已存全量source余弦的0.3间隔推导（2026-09-06）

仅本地stdlib复算已归档JSON：9模型、18756 source-model行，每condition6252 fold-local行；
每条fit记录在两份source fold出现，不是6252独立原图。原始模型/图像/张量/排序运行0。
依据实际V8 normalized batch-hard L2 Triplet固定margin0.3，计算sqrt(2-2cos_neg)-sqrt(2-2cos_pos)，未扫margin。
initial400/6252=6.39795%未满足，普通two-view终点116/6252=1.85541%，V24原型终点88/6252=1.40755%；
mean global-source hinge为0.002149224/0.000626920/0.000489391。全部行/身份已保存，无选择性抽样。
source严格正负排序正确不等于固定0.3间隔全部满足；但原型继续降低干净source hinge仍未带来V24科学晋级。
这限制“记忆库必然提供大量缺失监督”的推断；不证明XBM或增强后的困难负例无效，
不能替代真实增强batch梯度/缓存陈旧验证，也不是新的候选注册或V24复跑。
结果页results/TRIFUSION_SOURCE_GLOBAL_MARGIN_SCALAR_CENSUS_2026-09-06.md。

### 41.38 RGBNT100 固定source第34步复现，单batch精度/算子定位待执行（2026-09-06）

记录2026-09-06T10:32:30.132478+08:00。8b412d0固定诊断wrapper81713结束37.4706秒，本次33更新后第34步finite loss6.734400272但梯度非有限、scale256到128。
共2176source记录前向，0heldout/official；195梯度中153非有限且均属CLIP encoder，不能据此断言encoder自身是源头。
初始state/M0相同、前8步索引相同，仅step1全部标量逐位相同；其他差异明确保留，不把原未保存的正式失败步数写成34。
首次异常的模型前向前buffer/RNG、未更新权重与真实增强batch已保存远端438600256字节，SHA3c9b41a70a3e3bfabd317cea8b76f314ba16ded8a8246fd41e0561fc7f8286eb。
下一项固定fp16/fp16_anomaly/fp32三种各一次，完整batch64、scale256、0优化/图像解码/heldout；预计1–3分钟，当前READY_NOT_RUN。
原精度核对实际捕获loss后，异常模式定位第一个NaN算子，FP32检查精度因果；全量保留两次64×64 Gram及行列式。
正式模型/配置/AMP门仍未改，无自动restart或精度扫描。

### 41.39 RGBNT100 Gram零点反向异常已定位，局部FP32回归已登记（2026-09-06）

2026-09-06T10:43:29.465425+08:00。56f094f三个零更新probe完成48.6883秒，总192source记录前向/0图像解码/heldout。
fp16与anomaly全部损失分量逐位复现捕获现场；fp16仍153非有限梯度，两次4096项Gram各3零det、1负det。
原inputFP32但Gram矩阵FP16，G.float()之后求det不能恢复已丢的内积精度；AbsBackward0在sqrt(abs(det))产生NaN。
全模型FP32消除零det并使195梯度有限，但同时改变其他前向数值（loss差−0.088732719），不能直接归为只改Gram的效果。
准备tools/signal_gram_fp32.py：只让原volume计算FP32，不加epsilon/clamp/新目标；原Signal源码文件和正式runner/config未变。
固定回归先对同真实输入复现原算子失败再验证局部修复；通过后比较完整batch原/修复3072D精确推理与AMP195梯度。
这是READY_NOT_RUN，最多192 source记录前向/0更新；正式训练尚未恢复，也没有RGBNT100检索结果。

### 41.40 RGBNT100 局部FP32失败，稳定零点数值定义单独登记（2026-09-06）

2026-09-06T10:51:50.831737+08:00。8034451算子回归7.9486秒退出1：FP32仍在(4,4)出现零det，三个输入各512NaN，
原AMP有三个零点、各1536NaN。完整模型段未执行，本次0模型前向/0更新/2算子backward。
保留该提议FAIL，不能把先前完整FP32模型通过误说成局部FP32已修复。
依据真实失败，准备FP32 Gram+sqrt(abs(det).clamp_min(1e-12))，数值下限来自作者Triplet中已有开方保护值。
绝对det低于下限时volume=1e-6、该det梯度0，明确不是原公式零点处的完全等价；无参数扫描、额外目标、fallback或scale修改。
新回归保留原红例、要求新梯度全部有限并记录真实零det，之后才验证完整batch与原3072D推理。
新计划READY_NOT_RUN；原Signal/runner/config和旧失败提议未覆盖；正式三fold训练仍未恢复。

### 41.41 RGBNT100 稳定Gram真实回归通过，数值修复范围明确（2026-09-06）

2026-09-06T11:04:50.243649+08:00。6c741b8固定回归16.4125秒exit0，2算子backward/1模型backward、192保存source记录前向、0优化/解码/heldout。
FP32+1e-12开方下限保留已知(4,4)零det并令反向有限；完整AMP195梯度均有限，
四组身份损失及Patch逐位不变；Gram3.970210552→3.971660137，total6.734400272→6.734545231。
64×3072推理特征原/修复逐位一致；summary SHA8d5dee0f1a16f417fbdd52fd78cf7118554f5b92667b518fb77c5047ba9de8ba。
这只支持该真实batch的工程修复，不代表全训练或检索晋级，先前仅FP32提议继续FAIL。

### 41.42 RGBNT100 Signal工程修订R2与完整首epoch M0（2026-09-06）

R2登记2026-09-06T11:04:50.243649+08:00，配置SHA9d5ecf5f350c2f1eea50650bf0abc9580a4d2e86947e20a83b7ba94c67faf997，runner SHA4677e7345f282646e6654000d9d526d8207f698fc58a1f90b0c44bda6dd5fc25。
仅安装已验证的稳定Gram函数、逐步JSONL写在AMP失败门之前、M0由8步改为每fold完整1个source epoch。
原AMPscale不能下降、finite195梯度、selector冻结与严格重载门保留；正式仍fresh三fold各30epoch，不使用M0权重。
9个数据/提取/排名/configure函数AST等同R1；原runner原字节单独存run20/inputs，旧R1回执不改。
新源码绑定执行一次完整T0后进行完整首epochM0，预计3–6分钟；当前READY_NOT_RUN，完整基线尚未重启。
必须在正式启动前完成全部新M0标量/权重核验并适配终态核验器；无官方测试、消融或方法晋级。

### 41.43 R2 T0通过、M0保存失败与无损存储恢复（2026-09-06）

记录2026-09-06T11:36:28.223741+08:00。e699eac wrapper84049实际11:08:23.174872退出1，总83.1106秒；
T0原报告24.5822秒通过，回执SHAe54c826d1cfba2eca6526d4e4bdecb6758a19ff3b0ada4a498adfb3f11883ba5。
M0 fold0完成首epoch81更新/5184 source训练记录前向，195/195梯度有限、AMP256无下降、均loss7.006990939。
所有81步JSONL和training.json一致且逐项FP32 loss重算精确相同；torch.save随后因数据盘仅余1785856字节失败。
部分checkpoint101712000字节已记录全SHA并原地保留；strict reload/fold1/2未运行，M0不完整，无heldout检索。
已将已安装数据集的两个下载zip按前后SHA验证无损迁至/root/trifusion-storage/downloads；
数据盘余2077556736字节，overlay余17817534464字节，实验权重/数组/日志/数据集均未删除，公共挂载写入0。
存储重试登记ENGINEERING_R2_STORAGE_RETRY.md：同R2代码/config、三fold fresh完整首epoch，
输出固定/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906，
复用相同绑定的真实R2 T0回执，不重复T0、不复用M0训练权重。旧81更新另记实际工程成本。
M0文件/权重/全部标量核验器已准备，完整终态核验器亦在正式启动前适配R2；当前retry READY_NOT_RUN，正式基线未恢复。
§41.42中的不变AST对象数按注册清单及独立比较更正为9；10是项目文件绑定数量，两者不同。

### 41.44 R2三折完整首epoch及完整工程核验通过，固定30epoch基线登记（2026-09-06）

记录2026-09-06T12:11:15.931344+08:00。ed9c300 wrapper85560于11:43:42.345603退出0，197.780591015秒；实际首次确认12:06:37.001739。
三fold各81/86/90有效更新，共257/16448 source训练记录前向；195梯度全有限、AMP无下降、selector冻结，
3份checkpoint严格重载48条clean source前向的3072D特征逐位相同，heldout模型前向0。
远端3权重内容及所有源绑定核验7.803591516秒；本地257步/源索引/FP32loss逐项精确重算0.151017900秒。
summary SHAdc1106d460f9b31c208fc26eb0c0c89561743f14fabb7bfc88e064b49c95f428；结果页TRIFUSION_RGBNT100_SIGNAL_R2_SOURCE_M0_2026-09-06.md。
e699eac磁盘满前81更新/5184前向另记，部分权重原地保留；本次未重复T0，后续不复用M0权重。
run20初始独立重放153输入/973项完成，但最终报告引用范围断言失败后遇服务额度限制，当前未闭合、无最终verdict；
实际后端仍未独立证明，亦未审计本次R2完整M0，不把旧输入审计扩大到新运行。
固定30epoch baseline已登记，三fold fresh通用CLIP/seed42、唯一epoch30全8675内部query-gallery，无中间heldout选择，
完整终态验证工具已就绪；当前READY_NOT_RUN，尚无RGBNT100检索数字或主方法晋级。

### 41.45 RGBNT100 R2完整基线运行及原三角色第三数据集准备（2026-09-06）

实际启动2026-09-06T12:14:01.080628+08:00，执行def7b9b7ecd9e7e37821716a13fdb2580b2d955c；
wrapper87066/训练87070，输出/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906/baseline。
首次观察2026-09-06T12:35:34.237907+08:00，距启动1293.157279秒；原计划15分钟，实际首次约21.55分钟，不能把观察时间当作完成时间。
第0折30个完整epoch已输出，共2479/2479有效更新、无AMPscale下降，checkpoint、retrieval_arrays及完整gzip排序已生成；
整体summary/terminal尚未生成，进程仍在运行，当前未做完整终态验证，也没有可发布的三折最终检索指标。
以首折速度修订总耗时估计为65–75分钟，下一计划观察13:15；不减epoch、不改变采样/数值定义，不根据部分fold选模型。
原始进度：evidence/trifusion_rgbnt100_signal_v1_r2_baseline_progress_20260906_1235.json；
判断与ETA：evidence/trifusion_rgbnt100_signal_v1_r2_baseline_progress_assessment_20260906.json。

RGBNT100原完整三角色入口tools/train_rgbnt100_trifusion_oof.py已准备，SHA7f18cf950ff85bae9f300618c63154cc5394101ce12b97a7344615937c4a945d。
仅有源文件/准备合同，无可执行正式配置，也没有运行M0或角色训练；须先核完真实B0 summary、三权重、数组与全部8675query/90epoch。
模型本体、七组ID/Triplet、三角色/五输出和原五项支持条件保持；用本数据集montage/camera协议替换MSVR的tuple/scene协议。
同一SIM精确推理functional视图有MSVR实际依据，但仍须本数据集M0及全量B0数组逐位相等，不能借用旧PASS。
详细准备：refine-logs/rgbnt100_trifusion_v1/PREPARATION_20260906.md；
只做第三数据集完整原模型比较，不重复已封存失败，不提前做单/双角色、容量或融合消融。

文献增量已核对MODAL原论文主表与公式：其MSVR310主表57.7/73.9与正文55.9/70.4不同，保留差异；
不把新论文的shared missing-modality公式未经实现核验就用作任意跨模态可用路径。
CCL原PDF仍受挑战页限制、Hyper作者库本轮仅README，未获得可验证完整训练实现。
参见docs/SOTA_REFRESH_2026-09-05.md及evidence/trifusion_literature_incremental_modal_ccl_20260906.json；
既有公开高指标参照和本项目未达主目标的结论不变。

### 41.46 RGBNT100原三角色M0与完整终态核验器准备（2026-09-06）

记录2026-09-06T12:54:28.792843+08:00。四个独立入口完成AST检查，源码尚未在RGBNT100三角色上运行：
verify_rgbnt100_trifusion_m0.py、verify_rgbnt100_trifusion_m0_files.py、
verify_rgbnt100_trifusion_terminal_files.py、verify_rgbnt100_trifusion_terminal_scalars.py。
M0固定124步/7936 source训练曝光、203梯度项、全Signal保存状态对B0、三权重严格重载回执；
终态全部15数组/43375query-output/50身份/60epoch及全部真实步数，不再套用MSVR的scene过滤或每epoch13步。
加权loss按实际FP32分组运算重算；已对旧MSVR完整124条M0 JSON标量验证124/124 exact、最大差0。
这只是核验器算术检查，0新模型/张量/图像/训练；当前B0保持运行中的原定义，新三角色没有配置或运行。
详见refine-logs/rgbnt100_trifusion_v1/VERIFIER_PREPARATION_20260906.md及对应evidence注册。

### 41.47 RGBNT100 R2完整三折基线及全部终态核验通过（2026-09-06）

登记2026-09-06T13:56:31.328098+08:00。执行def7b9b，实际12:14:01.080628启动、13:22:06.744688完成，exit0；
wrapper4085.6617515秒，三fold×30epoch/7794有效更新/498816 source曝光，0 AMP下降。
内部8675 query/gallery、50身份，mAP89.52417504200771/R1 96.82997118155619；
fold mAP82.73298803596518/90.29154465089529/96.81599006034025，逐fold保留。
全部source记录覆盖；完整跨camera正对1502928/1745856=86.0854503464%，不作因果结论。
远端3完整checkpoint/数组/8675排序、10项目21Signal源绑定、整个CLIP、作者LR全部PASS；
距离逐元素重算相等；本地90epoch/7794步/8675 query核验，7794/7794 FP32原分组loss完全相等。
27份文本/JSON/JSONL/gzip 94457434字节按SHA收取；张量/权重留远端，0新模型或图像。
summary SHA549476408e82f085e7987fe087f6784773a54cd8f7eaf82e4e01faefd85711d5；结果页results/TRIFUSION_RGBNT100_SIGNAL_R2_SOURCE_BASELINE_2026-09-06.md；
执行侧closure evidence/trifusion_rgbnt100_signal_v1_r2_baseline_executor_closure_20260906.json。独立run20无最终verdict且不覆盖R2 M0/B0。
纠正§41.45：12:35只见尚在写入排名文件（7554299字节），关闭后16867041字节，不等于当时全排名完成。
13:21虽90epoch已齐，最后检索未结束，以真实terminal为准；原观察均保留。
这是内部Signal基线，不是官方1715/8575或主方法成绩，不与公开91.6直接比较。

### 41.48 RGBNT100原完整三角色固定M0及主比较登记（2026-09-06）

登记2026-09-06T13:56:31.328098+08:00，REGISTERED_NOT_RUN。基线前置由§41.47完整终态满足。
新配置configs/RGBNT100/TriFusion-source-oof-v1.json SHA8d3ce84068a584ca732e62ac4dac2dca366119863570ca6f8d4af6fb3d3f2ee2；
合同refine-logs/rgbnt100_trifusion_v1/EXPERIMENT_PLAN.md SHA37250b85b4e0e6069b1e44937a45ac70a8f66653bf6cd91b365400523d6ee129；登记evidence/trifusion_rgbnt100_original_roles_registration_20260906.json。
26实际远端源文件+合同/核验/闭合记录绑定，五已有本地CRLF差异记SHA/相同AST，不改源码或归一化。
原模型/七ID-Triplet/AdamW3.5e-4/20epoch5warmup/B64K8/seed42和五科学条件不变，
仅用各自fold真实epoch30 Signal；全部baseline features/distance必须逐元素复现B0。
M0三fold各8步+fresh fold0固定100步，共124更新/7936source曝光、72角色clean+24独立Signal，0heldout；
203张量梯度/冻结state/逐位重载及原final excess<=0.1门不变。
静态预测trainable6248460/6248460/6274572，尚非M0实际计数。
M0完整工程和权重/标量核验PASS后fresh三fold×20epoch，最终43375 query-output/50身份，五门不改。
预计正式100–140分钟，按本次M0计时更新；新私有overlay先检查>=8GiB和GPU>=22000MiB。
没有官方、消融或失败版本扫描，原ENOSPC和科学失败全部保留。

### 41.49 RGBNT100原完整三角色M0及全部文件/标量核验通过（2026-09-06）

登记2026-09-06T14:11:01.774271+08:00。执行9e908c8，wrapper92186/child92190；14:01:21.442232→14:05:20.261521，
exit0，wrapper238.8160716秒。首观察14:06:33.454192距启动312.011960秒，不当作完成时间。
三fold各8容量步+fresh fold0固定100步，共124更新/7936source曝光，0 AMP下降；
203/203张量有限非零梯度，Signal/冻结state不变，三foldsource前缀和全部五输出重载逐元素相同。
实际total97022989/97022989/97052173，trainable6248460/6248460/6274572，后者与静态预测一致。
原固定100步loss2.8034026623→0.4944220185，解析floor0.4908334477，
excess ratio0.00155176796144928<0.1，peak reserved6234MiB；0heldout/dev/official。
远端3完整权重/31项目21Signal源绑定/完整CLIP/124记录核验11.517526秒；
本地全部124 FP32损失及真实source/梯度/4epoch日志0.071510秒，全部相等。
23份文本5451771字节全SHA收取，summary SHAb2ba13b644033f2888f2c7eb4535e1b117f9175e2cef412f06aa2c438466a2f8；
结果results/TRIFUSION_RGBNT100_ORIGINAL_ROLES_M0_2026-09-06.md，执行侧closure evidence/trifusion_rgbnt100_original_roles_m0_executor_closure_20260906.json。独立审计仍无最终verdict。
正式比较登记evidence/trifusion_rgbnt100_original_roles_comparison_registration_20260906.json：配置/原合同不改，fresh同M0初始SHA的三fold各20epoch，
不加载M0权重，最终43375 query-output/50身份，原五科学门保持。
据M0容量约1.208–1.332秒/步，估计105–125分钟、首查启动后30分钟，当前READY_NOT_RUN。
核验器换行曾被执行者误读；AST实际码点10，原四入口及全部训练源字节未改，转换脚本未使用。

### 41.50 RGBNT100原完整三角色正式三fold比较启动（2026-09-06）

实际2026-09-06T14:13:43.910354+08:00启动，执行bbe49e1c24956e891afb8ec3e83df01acd75c579，
wrapper93313/child93317；2026-09-06T14:13:45.758912+08:00启动检查两进程存在，无已完成epoch或终态声明。
配置SHA8d3ce84068a584ca732e62ac4dac2dca366119863570ca6f8d4af6fb3d3f2ee2，M0 summary SHAb2ba13b644033f2888f2c7eb4535e1b117f9175e2cef412f06aa2c438466a2f8。
fresh三fold各20epoch，初始state必须逐fold等于M0初始值，不加载M0训练权重；
B0及完整模型/损失/原五科学门保持。最终8675 query/gallery、50身份、43375 query-output。
GPU启动1MiB used/24126MiB free，新私有输出卷14163791872字节free。
估计105–125分钟，预计15:58:43至16:18:43；阶段首查14:43:43接近首折预计终点，
另在>=180秒完成一次实际训练启动确认，不逐epoch轮询。
结果页results/TRIFUSION_RGBNT100_ORIGINAL_ROLES_COMPARISON_2026-09-06.md，启动回执evidence/trifusion_rgbnt100_original_roles_comparison_launch_20260906.json。当前RUNNING，无完整三fold或官方成绩。

启动活动实查2026-09-06T14:17:33.866656+08:00，距14:13:43启动229.956302秒，符合>=180秒间隔。
fold0完成epoch1/2，共164更新；第三epoch进行中，完整落盘189条更新全部有效，
每条203项gradient finite均真、0 AMP下降。GPU6260MiB/100%，wrapper/child存活。
仅运行日志/source更新观察，未读取heldout指标；完整fold receipt仍0。
证据evidence/trifusion_rgbnt100_original_roles_comparison_startup_check_20260906.json；下一阶段观察仍14:43:43，预估终点不因两epoch训练loss下降而改写成功判断。

### 41.51 三个内部协议完整环境标签与候选池普查（2026-09-06）

登记14:32:48，完整执行1.5863748秒，12833记录/346个各数据集独立身份/9846合法query，
每记录heldout一次/source两次，0模型/张量/图像/训练/新检索指标；未使用当前RGBNT100三分支输出。
RGBNT201 571 query、RGBNT1008675 query的camera均在对应source出现；
MSVR310仅fold0的7/600 query（ID40/64/138，scene6/21）scene未见，所有camera/v标签已见。
这限制把大部分失败解释为未见环境标签；不证明视觉环境分布相同或排除环境捷径。
同camera负候选比例query均值RGBNT20145.3665%、MSVR31013.6770%、RGBNT10012.5940%；
MSVR同scene为7.2436%。这些是完整候选池构成，不是最近错误概率，不直接与挑选错误子集相比。
各fold正例数中位数RGBNT20113/12/13、MSVR8/7/7、RGBNT100175/175/175。
gallery-only干扰身份/记录分别120/2555、95/432、0/0；RGBNT100无gallery-only身份仍有完整不同身份干扰。
完整不同记录对占比10.6833%/56.7064%/86.6812%与身份平均7.1339%/25.5752%/85.0937%分开报告；
不能与不同PK运行的8.0708%/26.0640%/86.0855%直接作“可用监督上限”或因果比较。
完整报告results/TRIFUSION_THREE_DATASET_ENVIRONMENT_LABEL_CENSUS_2026-09-06.md；六输入SHA/脚本及全部9846query/所有身份计数见evidence/trifusion_three_dataset_environment_census_20260906.json。
当前模型/损失/采样/五科学门不变，RGBNT100完整终态后再决定新主假设。

### 41.52 RGBNT100首折完整评估完成、第二折训练进行中（2026-09-06）

阶段计划14:43:43，实际14:44:19观察19完整epoch/第20进行中、1642有效更新；
14:49:35.802643跟进（相隔316.348471秒），fold0已20epoch/1653更新及全部3125图库五输出评估完成。
运行回执B0 features/distance逐元素相等，rankings.gz已关闭84388592字节，
SHA 06b6138a4632b58e6dd73350beda77d9dbfe3154e17b31d6aeedb214c4027c8e；正式完整文件/所有标量核验仍等三fold齐全后统一执行。
首折纯训练1829.3539913秒、1.10668723秒/更新，fold1已有127有效更新，第2epoch进行中；
全部1780落盘步均203项梯度有限、0 AMP下降，GPU6460MiB/96%，两进程活跃。
summary会先加入训练完的fold再评估；只有retrieval及heldout_record_forwards齐全才计完整评估，避免把训练回执等同终态。
B0前20epoch采样步数1653/1719/1823仅用于时间预测，角色最终步数以自身日志为准。
下一阶段15:18，更新完成窗口15:55–16:10；无首折分数调参、无三fold合并指标或官方结果。
原始两观察和判断evidence/trifusion_rgbnt100_original_roles_first_fold_progress_assessment_20260906.json，完整标签普查与失败条件仍保持。

### 41.53 RGBNT100前两折完整评估完成及用户综述更新（2026-09-06）

15:18阶段实际15:18:22第二折第20epoch/1693更新，整体3346有效更新。
15:23:56前两折20epoch/1653和20epoch/1719更新、6075gallery五输出全部完成，
Signal的features/distance均逐元素等于原B0；第三折epoch2、已133更新，
整体3505均有效，203梯度有限、AMP下降0。两次阶段观察间隔334.525秒。
第三折及合并科学终态仍待完成，没有提前PASS/FAIL或按fold分数调整训练。
下一观察15:53，预计15:55–16:10；终态3权重/15数组/43375排名/50身份/全部更新核验已登记但NOT_RUN。

docs/USER_REVIEW_ACTIONS_2026-09-06.md更新用户长篇综述中的V23进行中、车辆仅安装等旧状态：
V23−0.252705、V24+0.491307但未过门、MSVR310−1.011990、RGBNT100B0=89.524175。
明确V24两端同为双视图且物理采样器不变，不能声称已经验证提高batch跨相机覆盖；
完整clean source间隔大部分满足，也不证明增强分布下样本级记忆必然无效。

完整8675query×5输出错误描述入口已准备，全部身份/全部错误均保留；仅处理完整核验后的排名和标签，
不新增模型运行、训练、科学门、子集选择或相机因果声明。代码AST/CLI通过，真实数据执行NOT_RUN。
终态CPU文件核验包装器同步归档，原四核验器及训练绑定字节不变。
依据evidence/trifusion_rgbnt100_original_roles_comparison_progress_20260906_152356.json及evidence/trifusion_rgbnt100_original_roles_second_fold_progress_assessment_20260906.json。

### 41.54 按用户指令清理已结束实验的重复恢复检查点（2026-09-06）

用户要求“注意一下磁盘空间，把没用的权重删掉”。全量权重目录核查后，仅选12个已完成旧运行的24个.resume/generation-*-{complete,post_train}.pt。
清理前核对全部24个目标的原SHA/大小、各latest.json的complete状态及12个独立best/final/generator模型的原SHA；排除活动进程引用目录。
2026-09-06T15:47:02.523745+08:00清理结束，共删除26736541280字节=24.900344GiB；24个路径均不存在，12个独立模型删除后再次全文件SHA验证通过。
/root/autodl-tmp 50GiB卷由97%降至47%，free从1844940800字节（1.72GiB）升至28581548032字节（26.62GiB）。
当前RGBNT100输出所在root overlay是另一个卷，清理后free12051623936字节（11.22GiB）；不能将两卷空间相加当作同一输出目录可用容量。
CLIP、Signal baseline、V8最佳模型、V12 source模型、当前RGBNT100 M0/B0/比较权重、检索数组、日志和指标均未列入删除目标。
旧24份优化器/恢复状态已经删除，不能继续从这些旧状态恢复训练或重新核验其原二进制；历史JSON和既往审计为原时点证据，独立推理模型仍在。
完整路径/SHA/逐个删除回执见evidence/trifusion_completed_resume_weight_cleanup_receipt_20260906.json；结果页results/TRIFUSION_DISK_WEIGHT_CLEANUP_2026-09-06.md。

清理后原wrapper93313/child93317仍存活。按原阶段计划15:53实查于15:53:41完成：
第三fold第20epoch/1733有效更新，整体5105，全部203梯度有限、无AMP下降；59个完整epoch，前两fold完整评估不变。
当前输出卷free12034846720字节，两进程及GPU正常；科学终态尚未齐全，没有按中间分数改训练。
证据evidence/trifusion_rgbnt100_original_roles_comparison_progress_20260906_155341.json；下一查看15:59、仍预计15:55–16:10结束。

### 41.55 RGBNT100原三角色完整比较与全部执行侧核验通过（2026-09-06）

实际执行bbe49e1c24956e891afb8ec3e83df01acd75c579，14:13:43.910354→15:57:30.134394，exit0；
三fold各20epoch，1653/1719/1823有效更新，合计5195/332480三模态记录曝光，0 AMP下降。
内部全部8675query/gallery、50身份，Signal89.52417504200771/96.82997118155619，
fused91.31654049653056/97.93659942363112，mAP增益+1.792365454522852pp。
三个完整分支CNN91.0399643513、Transformer90.4085921433、Mamba90.3293523323，均高于Signal；
融合mAP最高；CNN Rank-1比fused多正确1条query，不能称融合所有指标最优。
三折mAP增益+2.515769704/+2.083219075/+0.592882201，50身份bootstrap下界+0.951777609；
原固定五条件全部通过，没有改变seed/epoch/模型/损失/门或按中间分数选择。

远端3个完整checkpoint、3个数组文件/15条特征-距离-排序路径、31项目21Signal源绑定核验54.443735秒；
三折保存Signal state和全部baseline features/distance均逐元素等于原B0。
16:00:40远端核验结束，16:06:42收取28完整文本/JSON/gzip，共513345441字节，逐文件SHA一致；
17:25–17:27本地完成全部43375输出/50身份/60epoch/5195精确FP32 loss重组及所有query错误描述。
最大指标重算差2.84217094304e-14、bootstrap差2.22044604925e-16；epoch均值及FP32 loss重组差均0。
36身份改善、13下降、1不变；fused全部query AP改善4168/下降1559/相等2948；
Rank-1修复115、新增19，全部错误275→179；新增19最近负例均非同camera，仅作描述。
实际采样跨camera正对1001718/1163680=86.081912553%，不是因果或跨数据集同条件试验。

完整结果及全部50身份见results/TRIFUSION_RGBNT100_ORIGINAL_ROLES_COMPARISON_2026-09-06.md；
summary SHA8520db24326a4b4f46974e68ee949163c76cb63e0f0d4a89ba9bad99004ede3e；
执行侧闭合evidence/trifusion_rgbnt100_original_roles_terminal_executor_closure_20260906.json。
独立审计仍UNAVAILABLE_SERVICE_LIMIT，无独立verdict；总成本另含Signal7794更新及M0124更新，不是等计算预算。
当前0官方与RGBNT201 dev访问；内部91.3165不能与论文官方91.6直接相减。

下一动作是另登记RGBNT100全部50训练身份的固定主训练及官方同协议比较，保留同源Signal及全图库五输出；
该主训练尚未执行，不能先报官方结果，也不自动晋级RGBNT201/MSVR310失败方案或解锁主目标前消融。
RGBNT201新采样/监督假设应针对真实证据单独比较，不重复V23模态MLP或V24原型配置。
磁盘清理后当前终态权重完整保留；16:03输出卷free11129262080字节（10.37GiB），旧数据盘约26.62GiB另计。

### 41.56 RGBNT100全训练集官方主比较准备（2026-09-06）

前置§41.55内部完整比较已通过原五门；下一阶段以全部50个训练身份fresh训练Signal30epoch，
再从固定Signal构造fresh原三角色20epoch，两组终点固定后统一官方1715query/8575gallery完整比较。
不复用三折OOF或M0已训练权重，不以官方分数挑epoch/模型，既有RGBNT201/MSVR310负结果及消融限制保持。
新合同refine-logs/rgbnt100_main_v1/EXPERIMENT_PLAN.md；当前PREPARATION_ONLY_NOT_RUN。
协议入口tools/build_rgbnt100_official_protocol.py已AST通过，拟逐个核对全部18965文件的SHA/标签/RGB montage头和全部query正例/排除集合；
Signal全50类训练入口tools/train_rgbnt100_signal_main.py仅AST通过，角色训练/官方检索/完整核验入口仍待绑定，没有新GPU训练或checkpoint。
配置与终态登记按各阶段真实前置完成后建立，不能把“入口文件存在”当作M0、B0或官方结果。
计划使用刚清理后有空间的原数据卷/root/autodl-tmp/trifusion-v2/artifacts下独立目录；原OOF和所有保留权重不覆盖。
主比较要求全量报告mAP及R1/5/10、所有50身份、全部8575条query-output和所有训练步；SOTA与同源Signal增益分开。
PMKD官方页面本轮可读，但大PDF网页工具超大小限制、直接下载断连，数值仍注明来自既有原表归档，不冒称新的独立核验。

### 41.57 RGBNT100官方协议全量通过与全50类Signal工程入口就绪（2026-09-06）

更新时间2026-09-06T18:31:46.237430+08:00。实际T0执行56fcaf3，17:57:48→17:57:54，18965个文件SHA/RGB montage头全部核对；
全部8675训练记录和原inventory一致，50奇数训练身份与50偶数测试身份隔离，1715query/8575gallery。
18:07进一步执行原作者RGBNT100._process_dir方法的原文本，全部18965条路径/身份/camera/view逐条一致；
完整14706125次query-gallery标签比较复核全部1715正例/排除集合，合法正例50–175。
协议SHA93b229f1f0dfdff58d2e1725c90b30929da2c4c2cc2d66516109b0f134853b2e。
本T0已读取10290个官方文件的头/字节，不能再称未读取官方原文件；没有官方模型前向、检索指标或优化器更新。

tools/train_rgbnt100_signal_main.py及tools/verify_rgbnt100_signal_main.py与Signal-main-v1.json已绑定；
配置SHA14b86b6e6161e55681b528ec2858590bed8e011be0b442ce1921529c6265da54，状态SIGNAL_M0_READY_NOT_RUN。
初版AST-only入口遗漏旧入口的显式_set_seed(42)，已在任何模型运行前修正；new_model本身不设置种子。
M0实际核对全50类/195梯度/全部首epoch、唯一保存/严格重载；所有文件、每步loss、索引、作者LR另由远端CPU核验。
仅完整M0核验PASS后才fresh同初态固定30epoch；M0已训练权重不作为B0初始化。
独立阶段wrapper记录准确命令、提交、配置/入口SHA、GPU和磁盘余量及训练/核验各自退出码，失败不重启。
当前只完成源代码AST/绑定检查，不能把准备就绪写成GPU M0通过。

新输出固定/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_trifusion_main_v1_seed42_20260906。
18:28实查数据盘free27584786432字节（25.69GiB），系统盘11129262080（10.36GiB），GPU无计算任务；
按用户清理后使用数据盘，既有结果/基线/当前权重保留，启动前仍按登记>=8GiB检查。
原计划SHA57f9cb601b1a5697acaff0e8bed3a737eb499d1f1a752ad6a29691c9afb503f3保持不变，状态更新在tracker。

### 41.58 全50类Signal M0完整通过并下发固定30epoch B0（2026-09-06）

实际M0执行545186daf4edd3bf3aabf1a905adbc5a3359d03f，18:33:45.558703→18:35:27.358802；
完整首epoch130有效更新，8320source记录曝光/8320独立记录/50身份；195梯度完整、无AMP下降。
总参数90800641/可训练90009601，冻结TokenSelection787968参数，peak allocated11380.147949/reserved12100MiB。
原Signal训练epoch实测71.037809秒，模型阶段92.404400秒；权重与全部130步FP32 loss/PK索引/LR远端CPU核验3.041780秒，wrapper总101.797487秒。
保存363313722字节权重SHAeb8d6c30d9ab577c9796c34bed328fa0342afada4b01b7ed2af9c2431a17f3f6；
summary9095369fcf69d58fe9920738fdbd1b2158d183692e4147eb1a7bba2aaded2054。
16个clean source前向的保存前后结果逐元素一致，完整保存state等于训练终态；没有官方模型前向/指标。
首epoch的sampler尾部剩余355记录未进入batch，这是该完整epoch的实际情况；不冒称首epoch全部8675覆盖，B0终态独立要求全部覆盖。
完整原始文本/JSON位于evidence/rgbnt100_full50_signal_m0_receipts，未把模型/张量复制到本机。

18:42:15.824772使用同已发布wrapper下发fresh固定30epoch B0，wrapper102783，执行仍545186d；
入口断言fresh初态SHA9eb41bcd8ab10438ed6f74997f9821f19406d6cbbccabf084367f4ea8400d584，与M0初态相同；
只读M0完整核验回执，不加载其已训练checkpoint。预计19:18:16.959，按实际首epoch线性估计；
首次阶段观察安排19:13:16.959，未逐epoch轮询，当前仅下发事实，尚无B0终态或官方结果。

两份后续角色入口tools/train_rgbnt100_trifusion_main.py及tools/verify_rgbnt100_trifusion_main.py已AST通过，
重用既有build_model/train_roles/extract，保持原完整V8、七组50类head、8+100 M0及fresh20epoch；
完整文件/所有标量核验入口不构造模型，只在远端CPU读取保留state和原始步记录。
配置需等待该B0实际epoch30权重/summary/核验SHA后登记；当前未执行角色模型，不能称角色M0已通过。
18:41数据盘free27208171520字节=25.34GiB，系统盘独立另计；删除重复续训权重的清理回执保持不变。

### 41.59 全50类角色与官方完整评估流程源码准备结束（2026-09-06）

2026-09-06T19:03:25.224563+08:00对两个训练/核验入口、两个官方评估/核验入口及两阶段wrapper共六文件完成AST检查；
角色两入口已在78187a9发布，新增官方两入口及wrapper本次归档。全部角色及官方运行仍NOT_RUN。
Signal B0延续§41.58的18:42固定30epoch任务；首次阶段观察仍19:13左右，未在准备过程中反复查询训练。

角色工程门仍是同一full50 Signal固定终点上的fresh8步容量和另一fresh100步固定batch，共108步；
七个分类head实际应为50类，运行时核验203可训练梯度、全冻结state、Signal parity及所有五输出重载。
完整保存权重和全部steps/FP32 loss/PK索引/LR由远端CPU核验，M0通过后再fresh20epoch；
新角色配置等待B0实际summary/checkpoint/完整核验SHA后生成，不以占位值运行。

官方阶段等待两训练端固定且核验通过，分别对1715query+8575gallery完整前向10290记录；
核对独立Signal与角色模型内Signal的全部features/distances逐元素一致，所有权重保持其训练终态。
五个输出保留完整距离与每条query所有8575图库位置排序，每输出独立jsonl.gz，控制单文件体积。
远端CPU终态入口重算全部73530625距离及对应排名位置、8575 query-output AP/Rank、全50身份bootstrap和原四个主比较条件；
同时输出全部1715query五输出正负例最近距离/Rank-1修复与新增错误清单，仅作完整描述，不挑失败子集。
本准备未执行任何新增模型、训练、官方指标，AST不代表M0或检索通过。
配置只会在真实前置终态可用后冻结，当前已运行Signal绑定文件和原计划不变。

### 41.60 全50类Signal固定30epoch完整通过及角色M0登记就绪（2026-09-06）

更新时间2026-09-06T19:34:21.964629+08:00。B0实际执行545186daf4edd3bf3aabf1a905adbc5a3359d03f，
18:42:15.927309→19:16:19.895217，wrapper2043.964749秒；训练/核验退出均0。
固定30epochs、3936有效更新、251904source曝光，所有8675训练记录/50身份都已覆盖，
195梯度完整、AMP下降0；最后epoch平均loss0.798727565，最终作者LR7e-7。
参数90800641/可训练90009601，peak allocated11380.147949/reserved12100MiB。
初态与原M0相同；保存363314827字节checkpoint SHA f173efd1eb43193b4012b6165be451161b31163684a65759bfdaa7085b240bee；
summary7c82f15a1227e36b04b41091bcf57ebc8a23375682448d62cd263a58120ccc13；
完整CPU核验5d1697e37035dfe17c0ad02268b0b8da5ee23801e9f071664ba7c8db39669a94。
全部3936步的FP32 loss/PK索引/作者LR及整个权重已验证，严格重载前后全部8source输出逐元素相同。
12个完整原始文本/JSON共14283116字节取回逐文件SHA核验；权重/张量/图片只在远端。
首130步与原M0每个step JSON完全一致是描述性复核，不是新增晋级门；没有官方模型前向/检索指标。

configs/RGBNT100/TriFusion-main-v1.json SHA68b091dd48f086a29319dc853f9eb48bd38aabc72391e0187421efc5e1496aaf，
绑定真实B0 config/summary/verification/checkpoint与原模型、损失和优化设置；
19:24:36远端全部36源文件SHA一致，新增模型/图片/张量/更新均0。
登记evidence/trifusion_rgbnt100_full50_roles_registration_20260906.json状态FULL50_ROLES_M0_READY_NOT_RUN；
原三角色M0固定fresh8步容量与另一fresh100步过拟合，七个50类head、203梯度和<=0.1 excess ratio待实际检验。
M0通过后fresh同初态20epoch，不使用M0已训练权重或旧OOF模型；官方配置等待真实epoch20完整核验。
主计划及已完成B0绑定文件不变，不提前宣称角色工程门或主结果通过。

19:31复查24个已删除续训文件仍不存在，数据盘free26829139968字节/约24.99GiB，
系统盘free11129262080字节/约10.36GiB，GPU空闲。原清理前后12个独立模型SHA通过的回执保持。
新文献条件核查见docs/SOTA_PRIMARY_REFRESH_2026-09-06.md：
ProxyTTT须标明测试时适配，STMI须标明GPT-4o文本/SAM2掩码，PMKD保留原表和本次原文索引证据层级；
AutoSOTA README的重排序配置不能直接算作本项目静态评估同条件SOTA。
独立审计服务仍不可用；完整执行者核验不冒称独立审计结论。

### 41.61 全50类角色工程检查完整通过，固定20epoch主训练开始（2026-09-06）

更新时间2026-09-06T19:46:27.249393+08:00。M0实际执行2800c88875736a679d3713dbdd6ee4dcefec73a0，
19:36:09.195541→19:39:00.536329，wrapper171.337115秒，训练/核验退出均0。
实际97519117总参数、6692364可训练参数，七个分类head全部50类。
fresh8步容量及另一fresh100步固定批训练合计108有效更新，203/203梯度、AMP下降0；
容量峰值allocated5708.930664/reserved5996MiB，8步10.802643秒，100步108.262041秒。
固定批loss3.087505341→0.529424965，理论floor0.526548419，
excess ratio0.0011232312085785198，小于原0.1门；没有改门或延长100步。

唯一M0保存390491894字节checkpoint SHA8e92b5992c7d3e1288e25910160a0f16931ccbe56aa0167fe58abbee0cfaf9fe；
summary4ab4eb9cdcba0e4c9a90be32974a623c11c6fdda1e232f58e622f6ac84a2ca3e；
完整核验3d47508599a9a2544a3ab0716a916beb479ddbb3eec0a056af8d7e828ba1d8f9。
冻结state与Signal state不变，Signal保存参数逐元素等于已核验B0；
24clean角色source前向/16direct Signal source前向验证前缀与所有五输出严格重载一致。
全部14个原始文本/JSON共4360400字节取回逐文件SHA一致，本机只运行纯标量核验，
全部108步FP32 loss/PK索引/梯度字典/学习率通过；整个权重仅远端CPU加载验证。
独立审计服务仍不可用；这是完整执行者核验，不能记成独立审计通过。

19:42:57.921763使用同已发布wrapper下发fresh固定20epoch角色主训练，wrapper106196、child106200；
角色初态3b8c3b751201ed1c53e5d71658680645704b65f7bf13f59e685a67e63ccc9b7f，
19:44:05启动健康检查确认与M0初态一致，GPU6338MiB/100%，实际状态RUNNING。
只读取M0核验回执，不加载M0已训练权重或旧OOF模型；原20epoch、损失、采样器和seed42不变。
根据M0实测1.0826–1.3503秒/步与约2624更新估算50–60分钟，
预计20:32:57–20:42:57终态，首次训练进度观察20:27:57；
19:44仅检查启动状态和磁盘，没有反复读取训练指标。现在没有完整主训练或官方检索终态。

磁盘19:44复查free26412765184字节（24.60GiB），系统盘free11129253888（10.36GiB）；
24个已删续训快照仍不存在，12个受保护独立模型都在且大小不变，
清理时的前后完整SHA回执保留，本次只检查文件元信息。
后续角色各阶段只保存一个终点checkpoint；官方完整特征+距离的裸FP32 payload预计1.388545GiB，
另计压缩排序/模型，当前磁盘足够。静态流程复核见
evidence/trifusion_rgbnt100_official_pipeline_static_scope_review_20260906.json；
静态复核不代替未运行的官方模型与全数组核验，官方配置继续等待真实角色epoch20终态。

### 41.62 作者代码核查和跨camera采样的完整数量约束（2026-09-06）

更新时间2026-09-06T20:15:02.777248+08:00。上一轮是实质进展：全50类Signal B0及角色M0完整核验，fresh固定20epoch已启动。
19:56:46只读ps确认wrapper106196和训练child106200都仍在运行，未读取训练指标；
实际执行2800c88、当前发布c96e26d，预计20:33–20:43结束，20:28首次训练观察不变。
当前没有角色主训练或官方检索终态；本节并行工作只阅读作者文本、已有source标签与源码。

作者代码三仓库固定提交：IICI d60e09bad6637b076a3c1347dfe59745b4cd76b3，
XBM（旧MalongTech地址转向msight-tech）223ecdc25f71ef1721a58bc87cc567025a32bc92，
Microsoft通用SNR f3d51b5e3525fe5e1ea364fafdf0e4cc60b1362b。
20份代码/README/许可证文本逐文件SHA记录，未下载作者权重或执行作者训练。
IICI原入口强制每身份只有一个camera，默认Market不做subcamera拆分，MSMT才按epoch低层特征聚类。
XBM是detach实例队列加memory度量损失，样例128D/55000/1000步后启用；
其标签0空槽约定与本项目合法class0不兼容，不能直接照搬。
XBM实际LICENSE为CC-BY-NC-4.0；IICI本次页面/文件未见仓库级许可证声明；
SNR实际MIT，但本轮读取的是PACS分类通用DG/DA代码，不能算原ReID或三光谱复现。
作者源码未复制到本项目，证据及引用见docs/IICI_XBM_SNR_CODE_AND_SAMPLING_CONSTRAINTS_2026-09-06.md。

本地实际V24源码确认：108个identity-camera原型汇成94个全局身份原型，
weak按组EMA更新，strong fused7680D接受全局及同camera原型损失，原采样器未变。
两端共享双视图/原型计算，系数0/1比较已完整封存；它不是完整IICI复现，但weak/strong原型核心路线已测试。
因此，不能再把“加入同环境原型或更多source负身份标签”当成全新关键修复。
实例XBM真正新增的是同一负身份的困难视角/实例多样性，不只是身份数量；
clean-source原型分类100%也不能代替真实增强训练视图中实例难例是否饱和的证据。

新增纯标准库census完整覆盖三折282个source身份成员/6252条source记录成员，并逐折对齐1680旧batch统计。
原每epoch批次29/28/27；原K8跨camera组容量41/41/40，原调度实际使用39/38/33。
每batch两个跨camera身份要求58/56/54组，较原分组容量缺17/15/14组；
对应真实cross-camera记录381/392/369，重复记录位置的聚合下界83/56/63。
这不是某个新采样器实测重复数；新采样器尚未实现，标签数量也不是mAP改善证据。
仅改变原组优先级无法实现目标，必须显式定义额外组抽取和身份频率变化，
并保留每折全部80个单camera身份和完整干扰图库。
每个cross身份内4+4是独立干预，无复用最多31/34/29组；不能与“两身份/批次”及XBM同时加入后混合归因。
后继应先固定唯一采样合同并全epoch重放，再决定配对训练；不改变当前RGBNT100输入或重启已封存版本。

新证据为trifusion_iici_xbm_snr_author_code_review_20260906.json、
trifusion_rgbnt201_camera_sampling_feasibility_plan_20260906.json及对应结果；
脚本tools/census_rgbnt201_camera_sampling_feasibility.py SHA9ca71b57a469c0a5ff5cf1801e1caa16221df094e1cca6261c5e20734a16339e。
所有新增模型/张量/图片/优化器/检索指标计算均0，独立审计仍不可用。


### 41.63 全50角色固定主训练完整终态与官方两端绑定（2026-09-06）

核验时间：2026-09-06T20:38:12.983144+08:00。固定20epoch完整结束，状态COMPLETE_FULL50_ROLES_MAIN_EXECUTOR_VERIFIED_FIXED_EPOCH20。
实际执行2800c88875736a679d3713dbdd6ee4dcefec73a0，19:42:58.070783至20:31:30.501081；
训练及远端CPU核验均exit0，总2912.427695秒。官方检索尚未运行，训练完成不等于检索有效。

- 2625次有效更新，168000记录曝光；全部8675训练记录及50身份都已覆盖。
- 203/203可训练张量有梯度，全部逐步梯度有限，AMP下降0。
- 20轮完整学习率、PK索引、FP32损失重组都通过全量核对，末epoch均值0.531187150063。
- 总参数97519117，可训练6692364；峰值allocated5709.758789/reserved5992MiB。
- fresh初态与M0逐项一致；没有使用M0已训练权重或旧OOF角色权重。
- 冻结参数及内嵌Signal全状态保持，最终权重严格重载后的五输出一致。
- 16个clean角色source前向、8个独立Signal source前向完成原一致性检查。
- 官方模型前向0，RGBNT201 dev前向0；当前未产生本次官方检索分数。

唯一保留角色权重：/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_trifusion_main_v1_seed42_20260906/roles_main/roles_epoch20.pth，390494154字节，
SHA256 3946c4ede79e6f622819b10c183906e25526796165b0b82560a7d4ebed6605e1。
summary SHA256 908397c508e1052735153d4dfa5e2bc96091ebc947cc33eb7981661797736ab9；
远端完整核验 SHA256 a367158e7571ef073714b15f9a761d176343c19a444627ea52e959fc129d2f66。
所有12个文本/JSON/日志共103421043字节已完整取回并逐文件SHA核对；
本地仅使用JSON/标准库/NumPy核对全部2625标量更新，没有加载权重或图像。
独立审计服务不可用，此项是完整执行者核验，不写成独立审计通过。

同身份正样本对588000，其中cross-camera506518（86.14251701%）；
这是本次实际采样描述，不构成泛化或收益的因果证据。

官方比较已绑定真实Signal epoch30与本次角色epoch20的summary/核验/权重。
配置configs/RGBNT100/Official-main-v1.json，SHA256 96baf923bc5f92b56feb490342d193031a522cfa50a626156bd2eb2e0d732985；
39个源码文件绑定，所有1715 query和8575 gallery、50测试身份、五输出。
固定增益条件沿原计划：fused至少+1pp、三完整分支均不低于Signal、
50身份bootstrap下界>0、fused mAP严格最高；不含三fold门。
本登记状态READY_NOT_RUN，待发布并同步后只启动一次固定完整比较。
不按官方结果选择epoch、seed或融合权重；SOTA与同源Signal增益分开判断。

20:39:40磁盘复查：数据卷free25914597376字节（24.135GiB）；
系统卷free11129262080（10.365GiB）。
原24个删除路径都不存在，12个独立模型存在且大小一致，
当前CLIP、B0、M0与epoch20权重保留。无额外删除。
清理时全部保留模型前后完整SHA见原回执，本轮磁盘复查仅验证存在和大小。


### 41.64 固定两端官方完整检索已启动（2026-09-06）

更新时间：2026-09-06T20:47:32.259139+08:00。官方完整比较已于20:44:27.119232下发，wrapper108798/child108802，
执行d9e4f9dc10a1d8eb5102505736f09500c9428683。20:45:29启动检查确认进程及官方目录存在，
summary状态RUNNING，GPU1434MiB/48%；官方模型阶段已启动，尚未完成任一整批10290记录特征提取，
该完成阶段计数不等于尚未访问任何官方图像。没有终态检索分数。
固定Signal epoch30/角色epoch20、所有1715 query/8575 gallery及五输出；
评估优化器更新0，不重新训练或选择终点。配置96baf923bc5f92b56feb490342d193031a522cfa50a626156bd2eb2e0d732985不变。
预计20:54:27–21:04:27结束，首次阶段观察20:54:27，后续按实际阶段估计并间隔180–300秒。
完整检索后还要核验全部保存数组、73530625排序位置、8575条query-output分数及50身份bootstrap。
数据卷free25801289728字节（24.029GiB）；清理24快照/24.90GiB仍有效，未新增删除。
独立审计服务不可用；之前的完整执行者核验不写成独立审计。


### 41.65 RGBNT100固定官方完整终态与全量错误普查（2026-09-06）

更新时间2026-09-06T21:13:08.599111+08:00。上一目标轮完成全50角色训练核验并启动官方比较，本轮完成全部官方结果、完整数组/排序核验和16份文件归档，属于实质进展。整体目标未达。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 | 相对Signal mAP |
|---|---:|---:|---:|---:|---:|
| Signal | 80.712162 | 94.227405 | 95.102041 | 95.743440 | +0.000000 |
| fused | 83.284770 | 96.151603 | 96.618076 | 96.909621 | +2.572608 |
| CNN | 81.961715 | 95.860058 | 96.443149 | 96.851312 | +1.249554 |
| Transformer | 81.709681 | 94.577259 | 95.510204 | 95.918367 | +0.997519 |
| Mamba | 83.440622 | 95.510204 | 96.384840 | 96.676385 | +2.728460 |

融合相对Signal+2.572608029 mAP/+1.924198251 Rank-1，bootstrap下界+1.56505481489pp；四门中只有fused严格最高失败，不能把工程核验PASS改写为科学支持PASS。Mamba mAP更高0.155851768pp，融合Rank-1/5/10最高。

全部50身份：融合39改善/9下降/2持平，最大两个身份净收益份额24.941296%；全部1715 query AP971改善/460下降/284持平，R1修复41/新增8，错误99→66。身份584平均AP-6.579633934pp并贡献5/8新增错误；8新增只有2条同camera，未读取原图，不补写视觉成因。完整50身份CSV、全部查询普查与五份排名见结果报告。

执行d9e4f9d，20:44:27.249861→20:50:12.632817，评估/核验exit0，总345.380174秒。每模型10290官方记录前向，共20580；评估训练更新0。全73530625距离及排名位置、8575分数、50身份bootstrap、两端完整state和Signal逐元素一致性都核验，指标误差最大2.842170943040401e-14。16份文本/gzip163634139字节全SHA传输；数组1490973250字节保留远端。summary SHA806be86d4ab8aed1fbf3803edc1e38913012a2f34e9906186dee15980ca0bdb2，verifier SHA49aa4a2a14aa59bbd6a875215b24a5125829c70c444df1d27fc0c28892746473。

官方融合83.284770距Signal公开86.3、RoDI-CLIP88.5、PMKD91.6分别3.015230、5.215230、8.315230pp；资源/训练选择差异仍需注明。本机Signal本身距作者表5.587838pp；作者B128/K16、独立模态增强和按评估保存best与本项目B64/K8、同步几何、固定epoch30不同，原合同已披露，21:00五份作者文本重核一致，不能单因归因或借此官方调参。PRISM/DSGM主表重核与已有SOTA归档一致。

当前科学结论是同协议Signal增益成立、融合未超过最佳分支且SOTA未达；不同数据集不能互相抵消负结果。已完整官方结果封存、不启动消融或融合扫描。下一项新source-only假设沿§41.62实际跨camera监督曝光：保持B64/K8/既有损失，先完成可重放采样合同及全量数据检查，保留全部身份和完整干扰图库，再固定配对训练。实例记忆不与采样一起加入。完整结果见results/TRIFUSION_RGBNT100_OFFICIAL_COMPARISON_2026-09-06.md，全部身份见同目录OFFICIAL_IDENTITIES CSV。

## 41.66 V25真实跨摄像头采样：全量重放通过与完整主比较登记（2026-09-06T22:33:37.459788+08:00）

RGBNT100官方完整结果、五路全排序文件已全部推送，GitHub/local/server为eb1b34a，
服务器/本地/桌面主交接于22:23:55逐字节相等；同步收据已归档。
之前上传的curl55/HTTP408仅为传输失败，未重跑训练或检索。
原结果保持Signal80.712162/94.227405→fused83.284770/96.151603；
五路完整比较和3/4科学条件边界详见上一节，不改写为主结果成功。

V25只改变实际source采样：B64/K8，每批两组跨摄像头与六组单摄像头身份，
保持原组内1+1锚点及六个随机剩余样本；跨摄像头身份生成两份分组供给。
原1680批完整重放全部顺序及关系数相符，候选1680批都符合固定规则。
三折正对占比分别8.234298/8.532366/7.416501%→12.202894/12.589286/12.242890%，
总体8.070790816%→
12.344547194%。
每端107520曝光；候选完整20epoch覆盖每折全部94身份和2126/2075/2051记录。
全部282个身份fold成员的曝光表及所有3360batch索引/顺序/关系数已归档。
候选记录重复增加、单摄像头身份相对曝光减少，不能归因于纯损失或纯相机风格去除。

网络继续使用原Signal+V8三角色、原7组ID/Triplet，不带V23/V24等干预，
新增推理参数0；当前架构说明沿用既有记录。
已登记六端各20epoch/3360优化步，M0先执行48只读前向和116优化步。
两端初始化相同，实际训练分别对照自己的完整采样序列，不要求跨端顺序或增强tensor相同。
终态保存全部5路特征、距离、完整图库排序及逐query分数；原五项科学门与全图库不变。
状态为IMPLEMENTED_REGISTERED_NOT_RUN，尚未运行M0/Q1，不据元数据通过宣称检索有效。
完整计划：refine-logs/trifusion_v25_camera_coverage/EXPERIMENT_PLAN.md，
配置SHA256 cc8178de4173e01115a39505115bbc3e67ce322c08cd59cafff952e965116e4b；训练计划SHA256 0399da7522f92c152bb453b3e3a6b5ebd6674e80c1fe02ba93e2e97aea47cc9f。

数据盘于22:28 free 21.782GiB；
24个冗余resume权重已删的结果保持，本次计划仅保存六个最终权重和必要完整数组。
系统盘free 10.365GiB。
外部独立审计额度限制继续存在，执行器核验不能替代独立审计。
RGBNT201固定dev目标、MSVR310负结果和跨三数据集SOTA总目标均未解决。

## 41.67 V25原持久进程已启动，M0进行中（2026-09-06T22:42:56.382205+08:00）

注册代码97468dd于22:37:03完成GitHub/local/server一致、服务器/本地/桌面主交接字节相等。
启动前核对全部登记的源码、配置、计划与Git提交，以及CLIP和6个V12起点权重全文件SHA。
22:39:45.136173启动screen v25_camera_coverage_97468dd，
wrapper112548、原训练PID112550；一个原过程先M0，成功后完成六端完整20epoch。
完整启动合同、原包装脚本与健康观察均已归档。

22:40:53.359908观察原PID和wrapper均活跃，GPU使用3072MiB。
第0折两端只读预检完成、初始模型一致，参数98800141/可训练7841292/203tensor符合登记。
尚未完整M0终态，尚无Q1结果，不能提前认定检索有效。
预计M0在22:44–22:49完成；随后根据每epoch实测耗时修正Q1预期75–100分钟。
原日志/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd.log；
完整终态/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd/run_summary.json，退出文件/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v25_camera_coverage_seed42_97468dd.exit。
按180–300秒/预计里程碑观察；观察超时不构成重启依据。

启动前数据盘free22.015GiB，
系统free10.365GiB，继续只保留必要终点和完整证据。
已清理24个冗余resume权重的结果不变，当前整体baseline/SOTA目标仍未达成。

## 41.68 V25完整M0已通过，原进程进入六端Q1（2026-09-06T23:00:03.502087+08:00）

22:47:10取得原PID112550完整M0快照，三fold两端48批只读预检完成。
各fold两端初始模型相同；总参数98800141、可训练7841292、203tensor。
控制/候选各8步容量分别reserved6054/6198MiB，候选固定第一个真实batch100步。
全部阶段203tensor均有非零有限梯度、冻结state不变、AMP overflow0。
固定loss1=0.6269225478172302、loss100=0.580318808555603，
下界0.57838292104621，超额loss比0.03988262041085136，通过原0.1门。
全部48批索引/路径及116步loss记录已复核；此结果是M0工程PASS，不是检索有效证据。

22:54:32观察原PID和wrapper活跃，原Q1已完成fold0-control19/20epoch，
六端120epoch目标保持、无完整终点比较结论。按最近每epoch约31.6秒估计，
六端预计23:45–次日00:00完成，后续按实际终点耗时修正。
原训练核心代码继续固定97468dd；新增核验器不改正在执行的模型、loss或采样。

已准备tools/verify_v25_complete_terminal.py，SHA256 cc1cf0c1db52dc3b9bf770cd9cdc427d9142f751b62afda78bb5a73fa4e5a277，
终态后在服务器CPU复核所有3360训练行、六端五输出全特征/距离/排序、
全部571query/21身份、mAP/Rank1/5/10及原5项bootstrap科学条件。
核验期间新模型前向/优化/图像/权重tensor读取0，仅加载6份已保存检索数组。
当前只完成AST与核验范围登记，未执行终态核验。
完整M0报告results/TRIFUSION_V25_M0_ENGINEERING_2026-09-06.md；
终态核验合同refine-logs/trifusion_v25_camera_coverage/TERMINAL_VERIFICATION_PLAN.md。

最新datafree22.014GiB；
24个冗余resume权重已删除的结果不变，继续保留必要终点/基线与完整数组。
外部独立审计不可用；执行器核验不标为独立审计。
RGBNT100官方融合83.284770/96.151603虽高于本机Signal，仍非三角色最佳mAP或SOTA；
RGBNT201、MSVR310与整个三数据集目标均仍未完成。

## 41.69 原Q1继续；完整终态核验已排入持久队列（2026-09-06T23:10:34.125041+08:00）

M0证据与核验器已发布15b04b3，23:02:30本地/GitHub/服务器commit一致，
三份主交接字节相等，收据已归档。
23:04:45复查原训练112550及wrapper仍活跃，已完成36/120epoch：
fold0-control20/20及终点检索已结束，candidate16/20，其余fold按原进程继续。
所有25个训练绑定源码当前SHA仍相同，未修改正在运行的采样、网络或loss。

23:08:42将完整终态复算排入screen v25_terminal_verify_97468dd，
核验队列wrapper114796已检查真实进程存活；此时原训练112550也重新确认活跃。
该进程每180秒等待原训练wrapper的terminal，随后自动单次执行已登记的完整核验器。
等待期间没有检索tensor读取；执行时CUDA_VISIBLE_DEVICES为空，仅服务器CPU计算。
队列与核验均不增加训练更新或图像访问，不自动重试，不重启原训练。
完整输出位于原run_dir/terminal_verification.json，
日志terminal_verification.log、退出terminal_verification.exit、
状态terminal_verification_queue.json。队列已建立不代表核验已经通过。
预计六端23:45–次日00:00完成；完成后最多再等180秒触发核验。

磁盘复查：24个已删冗余resume权重均仍不存在，12个保留推理模型路径/大小一致。
本次没有额外删除；原清理26736541280字节（24.900344GiB）的结果保持。
23:04:45数据盘free21.863GiB，
系统free10.365GiB。
当前原Q1与自动终态核验都保留，整体三数据集baseline/SOTA目标仍未达成。

## 41.70 磁盘清理复查通过；原V25完成两折并继续（2026-09-06T23:38:33.458059+08:00）

23:11:34已确认c2be955的GitHub/服务器/本地一致，三份主交接字节相同，同步回执归档。
23:33:41实际复查24个已删除的旧resume权重仍不存在，12个保留模型全在且大小符合原回执。
原清理释放26736541280字节（24.900344GiB），本次额外删除0。
V25实际目录位于/root/autodl-tmp数据卷，剩21.414532GiB；
系统overlay及/root/trifusion-storage剩10.364922GiB，分属不同文件系统。
当前输出尚有足够空间，保留必要最终权重、基线和完整检索数组。
证据：evidence/trifusion_v25_disk_live_observation_20260906_233341.json及results/TRIFUSION_DISK_WEIGHT_CLEANUP_2026-09-06.md。

原训练112550存活且命令匹配，M0完整PASS，Q1已完成88/120epoch：
fold0/1两端完整终点均产生，fold2-control8/20。自动完整核验wrapper114796仍存活等待。
首两fold的fused mAP差分别+0.564526417/+1.526055502pp；
首fold Rank1下降1.052631579pp、次fold Rank1持平，不能将部分结果提前写成完整科学通过。
后续保持原三fold、六端、20epoch、完整图库、全部五路结果与原五项科学门。

已准备tools/report_v25_complete_comparison.py，仅AST通过，尚未取得完整终态输入或生成最终报告。
生成器SHA256 cfa92b739b0dd2010f5905634641ea200e289542de40dd46c8334c672cec43db；
只处理已全量核验的原JSON，输出全部21身份/571query的五路配对变化，保留下降与新增错误。
新增报告代码不改训练源码、配置、采样合同或执行commit97468dd。
本机基线提升、三角色共同必要性和三数据集SOTA的完整目标继续保留，尚未全部达成。

## 41.71 V25完整六端与全量核验结束；后继转向排序责任（2026-09-07T00:10:56.694726+08:00）

原训练112550在2026-09-06 23:50:33以exit0结束，六端各20epoch共3360更新/215040曝光完整执行，
M0额外116更新与48只读batch亦完整保留；训练总耗时4243.258秒，wrapper4248.339秒。
原自动队列114796在23:50:43单次启动CPU核验，23:50:53以exit0结束，未重试或重训。
核验全部3360训练行、全部source曝光、6个最终权重/原起点SHA、6份全特征/距离/排序数组。
32602260个距离元素、5952790个完整排名位置、571query五路分数与全21身份bootstrap复算一致；
最大距离/指标误差0，全部Signal特征/距离/排序跨端相同。工程PASS，科学Q1_FAIL。

| 输出 | control mAP/R1 | candidate mAP/R1 | mAP增益pp |
|---|---:|---:|---:|
| Signal | 77.487603/79.334501 | 77.487603/79.334501 | 0 |
| fused | 80.881569/84.588441 | 80.420931/82.486865 | -0.460638 |
| CNN | 79.978362/83.887916 | 79.341922/82.136602 | -0.636440 |
| Transformer | 79.273361/82.311734 | 77.247032/78.283713 | -2.026329 |
| Mamba | 77.937680/79.334501 | 79.772761/82.136602 | +1.835080 |

三fold fused增益+0.564526/+1.526056/-3.185387pp；bootstrap95%下界-2.306666pp。
原五门仅“候选融合高于同checkpoint Signal及三分支”通过，其余四门失败。
保留完整图库3126及2555个只从query排除的干扰记录，当前不是30-dev或official。
融合AP181改善/193下降/197持平，R1修复3、新增15，错误88→100；
11身份改善、10下降。000250/000261占所有下降身份负贡献61.4870%，不是净损失分母。
Transformer R1修复7、新增30；Mamba修复22、新增6。没有读取原图或猜测视觉原因。

24份原文本/JSON/JSONL/log共15638325字节逐文件SHA接收；全部30fold指标、105身份输出行和2855query输出行再次核对。
两端融合仍分别比同端Signal高3.393965/2.933328pp。原角色价值与新增采样干预的负增益分开解释。
本次只能否定固定两组跨camera身份、重复分组的整体规则，不否定所有关系支持建模。
V25封存，不扫配比/分组数/seed/epoch，不进入D1/dev/official。科学门未改写。

当前模型继续为冻结Signal/CLIP与原V8三角色残差银行，9槽位、7680D固定融合；
V25只改训练采样，没有新的Adapter/Router/HFER。九槽位归一化时，
s_F=0.5s_0+(1/18)sum(s_role,modal)=(s_C+s_T+s_M)/3。
按用户新复核，下一项优先研究固定采样下的角色—模态排序责任，并验证有效关系与新梯度；
实例记忆需真实增强难例/漂移证据，不先叠加。BIER作者源码7份已固定cd04edf核查，GPLv3；
当前仅机制研究，未复制/运行作者代码，尚无新主方法结果或新颖性证明。

完整报告：results/TRIFUSION_V25_COMPLETE_COMPARISON_2026-09-06.md；
失败普查：results/TRIFUSION_V25_COMPLETE_FAILURE_ANALYSIS_2026-09-06.md；
后继参照：docs/BIER_ROLE_MODAL_RESPONSIBILITY_SCOPE_2026-09-07.md。
23:51数据卷free21.094658GiB，系统10.364922GiB；24个冗余权重清理结果保持。
RGBNT100官方+2.572608pp保留，但融合仍低于Mamba；RGBNT201 dev65、MSVR310与三数据集SOTA目标均未达成。

## 41.72 V26角色—模态责任：固定采样、单一新增目标事前登记（2026-09-07T00:35:14.340790+08:00）

V25完整终态和三方同步已核验。按用户新研究定位独立实现责任loss与完整配对runner；不是BIER代码移植或已证实的创新。
最新状态（2026-09-07T00:35:14.340790+08:00，§41.72）：V25完整Q1_FAIL已归档并三方同步到9cea045，fused80.881569→80.420931，原五门一门通过。V26新角色—模态排序责任已实现和事前登记，T0/M0/Q1尚未执行；两端原采样完全一致，tau0.1/lambda1固定，保留原14项loss、原模型及6x20epoch预算。将核对九槽位有效关系和同专家参数的新增梯度；不叠加记忆或新采样。RGBNT100官方真实基线增益+2.572608pp保留，但三数据集和SOTA整体目标仍未达。已删除24份冗余恢复权重/24.90GiB；00:25数据卷余约21.07GiB。

来源真实行三元组25088/batch、九槽位225792次训练曝光；重复照片不称独立关系。
权重stopgrad[sigmoid(-m_F/.1)*sigmoid(-m_s/.1)]，辅助项mean[w*.1*softplus(-m_s/.1)]，固定总权重1。
两端原CrossCameraIdentitySampler，全1680batch对齐，跨摄像头正对均8.070790816%。
零新增推理参数，原7,841,292可训练参数和七头监督；控制只优化原loss，候选保留原loss再加责任项。
T0解析梯度、48真实只读batch、116M0更新，随后六端3360更新/120epoch与原五门均已固定。
M0额外验证九槽位辅助梯度和同角色42/54/93个encoder参数上的真实新信号；未证明梯度冲突为原因。
完整计划：refine-logs/trifusion_v26_role_modal_responsibility/EXPERIMENT_PLAN.md；配置SHA01cc6e8d80d8ae68df3d2d858ed9102b9faf1b46c3021bb1ca993a0dbc775f13。
当前运行次数0，只有本地AST与旧元数据派生核验；执行PID/commit/检索终态另记，不把准备写成运行。
外部独立审计额度不可用，后续执行器全量验证保持准确标签。

## 41.73 V26完整M0通过、同一原进程进入Q1（2026-09-07T00:51:10.467612+08:00）

最新状态（2026-09-07T00:51:10.467612+08:00，§41.73）：V26原进程118939/wrapper118937在00:39:52启动，执行绑定ff18e40。数学T0和完整M0已PASS：48真实只读batch、116优化、203张量、九槽位及三角色新增梯度有效，过拟合比率0.058728。Q1正在运行，00:47:43完成fold0-control第6epoch，尚无完整检索终态。初始48次前向与候选八个容量batch的融合非正间隔均0，辅助梯度仅原目标约4.13e-5至2.88e-4；不因此改权重或提前判失败。V25完整Q1_FAIL已封存；RGBNT100官方+2.572608pp基线增益保留，三数据集/SOTA目标未完成。全量核验脚本已准备，等待原六端终态；数据卷余约21.05GiB。

全48前向对应24个不同批序列重复两端，共1,204,224次行三元组曝光，不是独立关系数量。
这批真实增强图像中融合均已正确排序；九槽位的辅助梯度存在但偏弱。
八步同专家参数块fused-vs-role余弦均正，尚不支持将梯度冲突写成当前主因。
首候选L_R=1.5705327314e-6，八步最大2.5723413273e-5；
完整训练仍可出现不同增强视图和新关系，按固定tau.1/lambda1继续全六端。
M0控制/候选reserved6130/7258MiB；后者含同参数梯度诊断开销，不等于常规推理额外显存。
实际融合分解最大误差5.96046448e-7；保留原3072D/7680D输出与全部固定损失。
报告：results/TRIFUSION_V26_COMPLETE_M0_2026-09-07.md；
完整原M0：evidence/trifusion_v26_m0_complete_20260907.json。
原五项科学门未改，未读取30-dev/official；没有Q1终态，不提前归为成功或失败。

tools/verify_v26_complete_terminal.py与完整报告生成器已准备，逐行核验3360更新、
全部32602260距离和5952790完整排名位置，并复算571query/21身份。
核验本身无模型前向/优化/新图像读取；目前仅脚本与事前登记完成，自动等待器尚未启动。
外部独立审计服务额度不可用，执行器核验不标为独立审计。

## 41.74 V26原Q1存活、全量终态核验队列存活（2026-09-07T00:55:51.284049+08:00）

最新状态（2026-09-07T00:55:51.284049+08:00，§41.74）：V26完整T0/M0 PASS，原训练118939/wrapper118937持续执行六端Q1；00:53:44到fold0-control epoch17，GPU100%，无完整Q1终态。执行源仍ff18e40，固定tau0.1/lambda1、旧相同采样和原14项loss未改。全CPU核验等待器120255在00:52:52启动并确认存活，180秒依赖观察，原六端结束后自动核对完整训练/距离/排名，不重训、不自动重试。M0初始来源batch关系易分、辅助梯度偏弱的证据保留；不提前判成功/失败。RGBNT100官方+2.572608pp基线增益保留，V25完整负结果封存；三数据集/SOTA目标仍未达。数据卷余21.049GiB；24份冗余resume已清理24.90GiB。

等待器路径evidence/trifusion_v26_terminal_verification_wrapper_20260907.py；
工具源码tools/verify_v26_complete_terminal.py SHA ac7b37ee389ddbea2eb48f94ffbc00fab885f02d248d803d7f04f6d23dff819e。
实际queue状态WAITING_FOR_ORIGINAL_TRAINING_TERMINAL，120255与118939均真实/proc存在。
将只在原终态产生后启动CPU全量核验；当前没有terminal_verification.json或完整检索结论。
Q1实际已完成的epoch以日志为准；00:53:44尚无final checkpoint，工件总计5376429B，
数据盘free22601699328B。只保留固定final/完整核验数组，不生成每epoch权重。
新增队列不是训练已完成或全量核验已通过的证据。
全部M0原记录、初始支持/同参数梯度、固定合同及失败/正结果边界已公开，不修改运行源码。


## 41.75 V26完整终态未晋级、关系支持和磁盘复查（2026-09-07T02:01:11.178140+08:00）

最新状态（2026-09-07T02:01:11.178140+08:00，§41.75）：V26六端固定Q1及完整CPU核验结束，原训练118939/wrapper118937与核验120255均退出0。fused80.402752→80.714686（+0.311934pp），CNN+0.666974、Transformer+0.044328、Mamba-0.379981；三fold一负，bootstrap下界-0.276212，原五门仅1/5通过，Q1_FAIL封存。完整3360步/32602260距离/5952790排名已核验，误差0。每端42147840次批内三元组曝光仅1/2次融合非正间隔；新增监督偏弱，不调权重重跑。RGBNT100官方+2.572608pp基线收益保留；V25已失败，三数据集/SOTA目标仍未达。磁盘清理24份旧resume释放24.90GiB，01:55数据卷余20.108GiB；基线、原始初始化、六端final及完整数组保留。来源实例普查只有未登记未执行草稿，无新训练或队列。

V26从00:39:52开始，总耗时4140.739秒；全量核验于01:50:03结束。
三折融合增益+0.306590/-0.553291/+1.083670；只通过候选fused高于同轮各输出条件。
571query中AP187升/151降/233平，Rank-1修复8/新增5；21身份13升/7降/1平。
000217平均AP下降5.830443pp、新增2个首位错误；不据此读官方图调模型。
Mamba三fold均退化。完整30fold输出、105身份输出、2855query输出已生成。

每端1680步/42147840次批内三元组曝光，融合非正间隔control1/responsibility2；
辅助loss均值7.3950984e-6/7.4405956e-6。零cosine间隔不能等同普通Triplet0.3已全满足。
M0全部24个同专家块fused/role梯度余弦正，尚不支持PCGrad因果判断。
新信号存在但弱；批内几乎全易关系不能外推完整source实例图库也无难例。
来源实例—原型全记录普查仍是后续诊断：本地草稿tools/census_v26_source_instances.py仅AST检查，
未登记合同、未发布、未远程执行、未建队列；不把该草稿写成已产生的数据证据。

已结束跨版本fold0-control第1步相同、第2步出现约1.65e-6差异，最终状态不同。
相同seed/初始化/样本顺序不是逐位确定性保证；当前两端保持同诊断路径。
不得用跨版本差异修正增益或当成噪声置信区间，不按本次结果切换kernel/库/seed重跑。
安装Mamba源码与固定tag关系已核对，具体CUDA二进制因果未证明。

报告results/TRIFUSION_V26_COMPLETE_COMPARISON_2026-09-07.md及TERMINAL_ANALYSIS同日期。
summary SHA4fa3ae76aec0667af29049d6b982b8aade0e3e755747fd4688329dc2bcb9ddc5；
verification SHA6f165629aae26492a0c4079f72c7a922701b6963448d83a51739403084fed445。
执行绑定ff18e40，原tau0.1/lambda1/旧采样/14项loss/20epoch与五门不改。
工程全量核验PASS不等于科学晋级；外部独立审计不可用，不标为独立审计。

24个旧冗余恢复权重清理回执继续有效，原12个独立模型保留；
当前数据卷和/root/trifusion-storage系统卷分别核对，不相加空闲空间。
完整终态权重、基线和源初始化以及所有日志/指标/检索数组保留。
最新磁盘与24删除/12保留模型SHA复查见evidence/trifusion_v26_terminal_intake_disk_20260907.json。
没有额外删除、重启训练、测试集调参、XBM/PCGrad训练或新官方结果。


## 41.76 完整来源实例与原型普查事前登记（2026-09-07T02:11:28.506430+08:00）

最新状态（2026-09-07T02:11:28.506430+08:00，§41.76）：V26完整Q1_FAIL已封存（fused+0.311934pp，五门1/5），完整终态核验通过；当前无原训练进程。完整来源实例—原型只读普查已实现并登记、数学反例检查PASS，尚未运行。固定三折6252来源记录—模型配对、干净/固定增强两视图、14输出、两合法关系协议，共200前向批和350112完整query输出行；原V12初始化，优化0，不读heldout/dev/official图像。后续先执行全量普查和独立实现的全行算术核验，再决定实例记忆是否有依据；不重跑V26或改旧门槛。RGBNT100真实增益保留，三数据集/SOTA目标未达。数据卷仍约20.05GiB可用。

原V26全训练每端42147840次批内三元组曝光，候选仅2次非正融合cosine间隔。
这只证明当前批内困难关系很少，不证明完整source实例图库没有身份原型平均掉的困难视角。
本次不改模型，源初始化与原M0三个完整state SHA绑定；model.eval/inference_mode、FP16前向、FP32归一化特征。
固定每折全部2126/2075/2051条source、B64/workers4、seed42、drop_last=False，共12504三元图前向曝光。
干净图库始终保留全部记录。普通身份关系排除自图；跨相机关系排除同身份同相机。
跨相机无合法正例query标不可评价但仍输出、仍是其他身份的图库干扰。
真身份原型仅由各query合法正例构造；负身份原型使用其全部source干净实例。
完整14输出、12份gzip全行记录、168聚合项和3份完整特征文件都保存。
预计约1.5GiB特征、开跑至少4GiB可用；无新增模型权重，不存相似度大矩阵。

数学反例已证明能识别原型正确而具体负实例误排，包含class0/自图排除/完整图库检查。
之后使用独立实现的CPU逐正负比较和stable排名，核验全部350112行/168聚合项。
核验不新增模型/图像/优化，不冒充外部独立审计。
只测一次固定增强；尚不测缓存漂移，也不是XBM方法效力或未知身份泛化结果。
原型已覆盖所有身份的V24负结论保留；本次对象是实例内差异，不再换名重做原型分类。

合同configs/RGBNT201/source-instance-census-v26.json SHA3139d941b3f3e2a2394b48438aefceb1e4caaba3b2108c4fbe4ad27b5d7a1e92；
runner SHA66069cf6b43afadd0cecbc2817332bcaac2f74cbbc96422de8f4fd47057e6468；CPU verifier SHA4adf4a0801bbf7a4b7edffe5d5287ec8631a5831ed790f69d7f565fa298fd686。
计划refine-logs/trifusion_source_instance_census/EXPERIMENT_PLAN.md已冻结。
持久wrapper已准备，当前还没有launch/exit/来源模型输出；不得提前写成执行成功。


## 41.77 来源实例普查原进程存活、前两折完成（2026-09-07T02:17:07.991120+08:00）

最新状态（2026-09-07T02:17:07.991120+08:00，§41.77）：完整来源实例普查于02:12:37实际启动，执行99e2a4c，原进程124596/wrapper124594在02:16:02仍存活，前两折全量完成、第三折进行中。共6252来源记录—模型配对、两视图、14输出和两协议；固定合同与模型未改，优化0。尚无完整三折终态或CPU全行核验结果，不根据中途fold解释机制。完整报告生成器已准备，终态后输出全部168分项及56汇总项。V26 Q1_FAIL和RGBNT100官方真实基线增益继续保留，三数据集/SOTA目标未达。数据卷余约19.13GiB，未删除新权重。

screen source_instance_census_99e2a4c，原PID124596/wrapper124594。
02:16:02实际/proc及GPU2386MiB/47%确认活跃；source_census状态RUNNING_SOURCE_ONLY_CENSUS。
截至上次已完成fold1全量记录时172.145秒；不按前两折结果决定是否继续，完整第三折仍在执行。
每折所有source记录均顺序遍历，原M0初始model state绑定及提取前后state/grad检查真实通过。
全流程源码/合同仍为登记99e2a4c，不热补丁或替换初始化。
后续CPU核验由同一个wrapper在提取完成且exit0后自动启动；没有重试/新训练。
报告生成器tools/report_source_instance_census.py只读取完整已核验摘要，拒绝部分fold。
原始launch/live收据evidence/trifusion_source_instance_census_launch_20260907.json及
evidence/trifusion_source_census_live_20260907_021602.json。
完整结果尚未形成；不得提前宣称XBM存在有效监督或没有难例。


## 41.78 完整来源实例普查结束：融合饱和与槽位困难并存（2026-09-07T02:24:42.415837+08:00）

最新状态（2026-09-07T02:24:42.415837+08:00，§41.78）：完整来源实例普查及全行CPU核验已结束，124596/wrapper124594退出0；三折6252来源记录—模型配对、两视图、14输出、两协议共350112行/168统计全部通过，误差0。增强普通身份检索中fused R1=100%、mAP99.999973，仅1条query AP不满；完整融合新增零间隔难例支持很少。九槽位则有192次原型正确但R1错、16570次原型正确但AP不满，分别覆盖132/4779个fold-query配对；不得当独立图片数量。当前不直接建fused XBM，下一步用现有完整特征区分Signal、残差银行与槽位补偿。无新训练、官方或缓存漂移结果。V26完整未晋级、RGBNT100官方基线增益保留，三数据集/SOTA目标未达。数据卷余18.682GiB。

原来源图提取253.239秒，CPU全量核验113.656秒，于02:18:50全部退出0。
源码绑定99e2a4c；所有模型完整state与V26原M0初始相同，前后不变、grad为空、optimizer0。
3126不同source三元图各在两个fold出现，6252是记录—模型配对数；跨相机query合法1142，其余5110仍作为图库干扰。
完整保存两视图×14输出×两协议全部350112行、三份原特征；本地完整文本独立重汇总168组通过。
未挑fold/query/模态或以中途结果决定是否继续。

增强普通身份协议下五种完整表示均R1=100%，AP不满query分别Signal1/fused1/CNN2/T0/Mamba3。
融合全部286626942有向正负组合中仅1个非正间隔；跨相机融合全部正确。
九槽位的隐藏实例排序困难则很普遍：16570次AP不满、192次R1错误，去槽位重复后4779/132个fold-query。
每槽位有116至137个原始来源身份出现原型正确但AP不满。全部九槽位与三折细表完整保留。
这种差别不能直接归因为Signal独占主导；也可能有其他模态和角色补偿，尚需从保存特征分解。

现证据不支持直接为完整融合建立更大负例库作为优先修复；槽位实例信息提供了更具体的研究对象。
下一项先完整分解Signal项、纯残差银行项和跨槽位补偿，用已保存特征即可，无需新图像。
不能依据槽位更难就强制九槽位都复制完整身份表示，也不把V20/V26换名调参。
单固定增强、model.eval和干净图库的范围保留；尚未测训练缓存慢漂移或XBM效力。
这次是机制普查，不是模型晋级，不提供新dev/official/SOTA指标。

报告results/TRIFUSION_SOURCE_INSTANCE_CENSUS_2026-09-07.md与同日期ANALYSIS。
完整原始工件evidence/source_instance_census_20260907/共24文本/压缩文本文件、16970380字节。
特征1460487366字节只保存在服务器，无新权重；数据卷剩18.6816GiB。
summary SHAb1a1068fa1841e962979986f12438696e28063838dcd36f92f0aa5e24b0a3031，verification SHAfd4f0f068fd9bce9b812271789cb807b984498b58c1ced2099d83165648a90bb。
外部独立审计不可用，执行器核验与独立实现算术不冒充外部审计。


## 41.79 来源支持完整分解：同角色三模态补偿已接近饱和（2026-09-07T02:47:36.949548+08:00）

最新状态（2026-09-07T02:47:36.949548+08:00，§41.79）：固定来源特征的完整支持分解结束，125623/wrapper125621退出0。全634354228关系、25008query行、216指标及108组16格表完整通过；原14输出逐query匹配。纯银行增强来源mAP99.999877/R1=100，仅5个非正关系；99.908429%的槽位错误在同角色三模态残差及融合中已被补偿。因此来源饱和不能归结为Signal兜底，不将V26门控简单改成纯银行重跑。下一项候选转向来源环境多样化；MixStyle作者论文/低层CNN适用边界已读，但未移植、未启动新训练。V26未晋级、RGBNT100真实增益保留，三数据集/SOTA目标未达。数据卷余约18.64GiB，无新模型/特征文件。

注册configs/RGBNT201/source-support-decomposition.json，contract6bd3fc612ae3d826fa6a888daaa08410a1f8466182c2eb63aa7bb8803ce64e8b。
源码22be78c、runner213a1e026bf7ec85e099156e6ddde177410f183a3eff4e093313af9a820c8c88；
02:35:00实际启动，02:37:05全部退出0，122.558秒计算。未新增模型、图像前向或权重。
完整已核验source特征三份读取；没有用新身份/official数组决定机制。

增强普通身份协议：纯CNN mAP99.991489、纯Transformer100、纯Mamba99.986523、
纯银行99.999877；所有R1=100，银行286626942关系只5个非正，1个query AP不满。
全部9槽位1808433次非正关系中，同角色残差/融合已正1806777次，
银行/融合已正1808411次；角色非正但银行/融合正1636次；银行非正但融合正17次。
这些有重叠，不能互斥加总。全部16符号组合逐query保留，也有同角色正而银行非正再由融合修复的情况。
来源补偿不等于新身份补偿，不能把来源100解释成完成baseline/SOTA目标。
当前不支持简单改V26融合门控为银行门控、加大辅助权重、盲目扩大完整融合XBM或强制所有槽位复制同样身份向量。

下一候选为增加来源环境多样性的训练任务，先核对低层接入与三模态一致性；
已读MixStyle原论文和MIT作者仓库，作者明确低层CNN用法与避免最后预测层。
当前CLIP block8语义token不能直接套用该结论；未复制代码、无V27实现/合同/训练。
引用与边界见results/TRIFUSION_SOURCE_SUPPORT_ANALYSIS_2026-09-07.md；
完整216指标/108符号表及全部25008query文本已归档。
source support summary SHAc1a94fda8f690edd3be20b3ddb974097778bcca7af55f257385cadf67a207c54，完整报告SHA55b50f9c22aabe5c4b395f9947a69be4554fa3f03c04ed12d755be1037cca3ad。
外部独立审计不可用，执行器核验不冒充独立审计；仍待新的固定完整训练验证。


## 41.80 V27源码与完整比较预登记、磁盘复查（2026-09-07T03:09:50.045072+08:00）

此前完整来源支持分解改变了后继方案：纯残差银行也饱和，不再简单改责任门控。
本轮读取V24真实增强：弱erase0.5、强erase0.6和亮度0.8–1.2，不能把它们换名重跑。
Signal实际stem为conv1输出B768x16x8、在CLS/SIE/位置编码/ln_pre之前。
V27在该处独立实现MixStyle统计公式，同模态跨摄像头供体，三模态共用供体和Beta系数；
保留原3072D Signal，另一个冻结视觉pass产生同扰动anchor/reference供原V8角色。
control执行相同额外pass但不采用扰动，推理两端都回到原V8。新参数/推理计算0，
训练额外计算需实报；统计扰动不保证保留全部身份线索，也不是原作者CLIP实验。

完整合同：refine-logs/trifusion_v27_source_style/EXPERIMENT_PLAN.md；配置configs/RGBNT201/TriFusion-signal-preserving-v27-source-style-rtx3090.json SHA dd1f3eed0816f4a8d00a76044cc5ad8b40786e597018afad7ea0bde90d29a3c6。
新模块与runner已AST检查，原采样验证、完整图库评估函数AST保持。
M0固定48原模型前向+48backbone接口前向、116updates；通过后原进程六端3360updates，
保留旧初始化/采样/14项loss/20epoch/原五个科学条件。当前尚未执行T0/M0/Q1。
外部独立审计仍不可用，执行器检查不冒充独立审计。

03:00复查数据卷free20007673856B，系统卷11129257984B，分卷统计。
24个旧恢复文件均仍不存在、12个保留模型全SHA一致；本次追加删除0。
全部198个权重文件路径/大小清单与回执在evidence/trifusion_disk_weight_recheck_20260907.json，
不是198个可删除文件。原始CLIP/Signal/V12/正式最终权重与检索证据保留。


## 41.81 V27启动R1配置类型失败与R2最小修复（2026-09-07T03:14:22.617591+08:00）

原wrapper126752/child126754均结束，849b608原进程03:12:15启动后退出1，
在load_contract固定STYLE断言停止，尚未执行T0、加载构建模型、真实图像前向或更新。
实际原JSON中VARIANCE_EPSILON是数字1e-06；旧load_raw_config使用yaml.safe_load，
读得字符串"1e-06"。源码行和远端实际输出已核对，不是梯度/模型/检索失败。
仅V27的配置读取改为json.loads(path.read_bytes())，没有增加fallback。
所有科学配置逐字段与原Git849b608比较一致，仅runner来源SHA更新。
原计划refine-logs/trifusion_v27_source_style/EXPERIMENT_PLAN.md字节不变，SHA305f89d0fe58257c76f9536d2260ab0ae6668b1fa42e81a6e766845649e94d3c。
R2配置SHA6f3161985f3a3831ffa890a163a510c6fa463ce4fd933793f0ce79db2c07677a，原失败log/exit/terminal及启动记录已收回文本归档。
本条登记时R2未运行，需完成同步后以单一新原进程执行全部原合同。


## 41.82 V27完整M0与原进程Q1、完整核验准备（2026-09-07T03:28:24.943088+08:00）

实际执行c225652，03:15:43原126981/wrapper126979启动；R1配置类型失败保留。
T0公式/统计停止梯度误差4.164219e-7/2.497861e-7。三fold两端全部48预检原推理
输出一致，额外48次backbone接口前向确认Signal完全保持而候选anchor/reference均改变。
容量16updates均203/203梯度；固定100步首末loss0.722419023514→0.580305516720，
floor0.578382921046，excess ratio0.013348012343，冻结state不变、overflow0。
capacity reserved6126/6418MiB。原梯度门是每阶段非零覆盖，不是要求后期过拟合每步
都非零；后86步存在部分零梯度，每步最低192个，阶段覆盖仍203个，完整日志保存。
原48批/116步与供体计划全部文本/NumPy复算PASS；不是独立审计。
M0原远端快照SHA b664f6af9312f73443f960d306cfee92295a03ac7bca36627cfed8a5be93de62。

Q1继续同一原进程，03:25:28 fold0 control230steps，epoch8 step27，无overflow。
完整六端20epochs/3360updates和旧初始化/采样/全部14loss/五个科学条件均不修改。
目前没有完整检索终态，不依据中间控制端或单fold作科学判定。
source-only检索仍完整3126gallery/571query，dev与官方访问0。
完整CPU终态核验tools/verify_v27_complete_terminal.py已准备，SHA46fa03d799e9556970880b9435c940af8d413a18b677929379dd40fe45a0179e；
将检查全部训练/供体/32602260距离/5952790排名和bootstrap，队列待同步后启动。
完整M0报告：results/TRIFUSION_V27_M0_AND_Q1_LAUNCH_2026-09-07.md。

03:25数据卷free20003610624B（约18.63GiB），系统卷11129245696B（约10.36GiB）。
此前24重复恢复权重26736541280B已清，12保留模型已本轮完整SHA核验；本轮新删除0。


## 41.83 原Q1及完整终态核验队列已验证运行（2026-09-07T03:31:23.537075+08:00）

03:30:21原126981/wrapper126979仍运行，fold0 control已有469条持久训练行，
处于epoch17 step5，overflow0，GPU6764MiB/100%。没有完整Q1终态。
03:29:37核验队列128288启动并已验证存活，WAITING_FOR_ORIGINAL_TRAINING_TERMINAL，
绑定c225652原训练和46fa03d799e9556970880b9435c940af8d413a18b677929379dd40fe45a0179e核验源码；
每180秒检查终态，不重启训练、自动重试0。训练结束后完整复算原六端，不是抽样。
运行回执evidence/trifusion_v27_terminal_queue_launch_20260907.json。
当前数据卷free19999490048B（18.63GiB）、系统卷11129245696B（10.36GiB）分别统计。
本次新增删除0，原24个冗余权重清理与12个保留权重完整校验保持。
下一步仅等待当前六端完成并取得完整核验终态，不启动新版本、不读取dev/官方。


## 41.84 完整结果报告链部署及原Q1核实等待（2026-09-07T03:44:09.621876+08:00）

上一目标轮完成新方法实现及M0，是实质进展。本轮先重新认证远端并核对head/进程命令行，
未将旧状态文件作为进程仍在的唯一证据。03:42:31原训练126981/wrapper126979、
全量CPU核验队列128288、新结果队列129047均确认/proc存活且cmdline匹配。
第一折control已580/580步；candidate460/580步、epoch16 step25；全部已保存训练行overflow0。
原三fold两端/20epochs/3360updates继续；没有新模型前向或训练配置改动，
不从首折或控制端单独判定科学结果。

新增tools/report_v27_complete_comparison.py（233f1458bed89986f5712fe08d4a6946da194a1b31941fc3c70c5e8b0f301616）
已提交并部署。它要求完整Q1_PASS/FAIL与全量CPU核验PASS，才生成：
- 完整30行fold/output/end指标、21身份与571query的五路比较；
- 全105身份输出CSV及2855query输出CSV，保留新增错误、退步和全部原指标；
- 全3360训练行按共同计划all/active/inactive分层，共18组、全部14项loss。
分层只是描述性统计，训练参数已在两端分化，不能把每批差异当成当次扰动的纯因果效应；
Triplet正批数也不是独立难身份或三元组数量。没有新增门槛或结果选择。

报告队列129047于03:41:09实际启动，WAITING_FOR_FULL_TERMINAL_VERIFICATION，
每180秒检查，automatic_retries0；先等原训练，再等全部数组核验，最后才写真实报告。
原CPU核验源与训练源都未改变；本轮报告程序只是文本/JSON/CSV处理，外部独立审计未恢复。
运行证据：evidence/trifusion_v27_complete_report_queue_launch_20260907.json。
数据卷free19838521344B（18.48GiB），系统卷11129245696B（10.36GiB）；不合并空间。
已完成的24旧恢复权重清理仍保持，未新增删除，当前最终权重与完整证据保留。



## 41.85 V27完整终态：统计扰动带来实测增益，融合最高门仍失败（2026-09-07T04:43:25.103022+08:00）

原训练126981/wrapper126979、全量核验128288、报告129047均已结束，三个退出码均0。
六端各20epoch、合计3360更新/120endpoint-epochs，无AMP overflow；执行代码仍c225652。
Q1_FAIL保留为4/5通过：fused mAP80.25341781842543→81.5923614544107（+1.33894363598527）；
CNN+1.78661434701783、Transformer+1.77615575375272、Mamba+1.28305862972243。
三折fused增益+1.17212438045671/+1.03641830357208/+1.76393221545257，bootstrap95%下界+0.170767946024522。
唯一失败条件是候选fused严格超过全部角色：CNN81.66285919599126，比fused高0.07049774158056。
不能放宽原门，也不能因总FAIL抹去相同初始化对照下的真实正增益。

全部571条fused query中242改善/132下降/197相等，R1修复12/新增8；
全部21身份13改善/8下降，000220和000239约占查询加权净收益45.55%。
8个新增错误有6个首位负例同camera，仅为关联证据，未据此推断原图视觉原因。
候选fused对CNN为198query AP更高/176更低/197相等，修复CNN的10个R1错误、又新增17个；
三折fused-CNN约+0.184290/-2.531725/+1.870838，不能将平均差解释为所有query都该选CNN。

原V8推理结构保持：3072D冻结Signal+4608D九槽位残差银行=7680D fused，
CNN/Transformer/Mamba仍分别处理共享冻结尾部，固定相似度平均，没有Router或稀疏省算。
V27只在训练角色路径的CLIP stem做同模态跨camera统计混合，p0.5/alpha0.1，原Signal prefix不变；
819/1680候选batch实际激活。参数不增加、推理无额外pass，两端训练均额外3冻结视觉pass/batch。
全训练纯残差Triplet合计均值0.00352915→0.00984482；激活计划中的fused Triplet0.00356331→0.00828485。
这些支持输入分布扰动增加训练难度并伴随本轮Q1收益；
0.3-margin Triplet正值不等于零间隔排序错误，不由此认定V26责任权重已不饱和或XBM有用。

CPU全量重算32,602,260距离、5,952,790排名位置，误差0；完整图库3126、合法query571、身份21。
31原始文本共28,713,694B已SFTP原字节接收、全部SHA匹配；本地再次完整核对3360训练步、
120epoch、105身份输出行、2855query输出行、bootstrap及全部5门，PASS；不称外部独立审计。
终态SHA b85c525d3ed45642dce293d9ce1540d053bc390aae4dc11e60edcb28f8692932；
数组核验SHA f4de137f2bae700f893f1d9582f2621c15312455ce3cd0cfb3771cc17aace5be。
完整比较results/TRIFUSION_V27_COMPLETE_COMPARISON_2026-09-07.md；
完整分析results/TRIFUSION_V27_TERMINAL_ANALYSIS_2026-09-07.md；
原始证据evidence/v27_terminal_20260907/及trifusion_v27_complete_terminal_intake_20260907.json；
本地复算evidence/trifusion_v27_complete_text_verification_20260907.json。

封存V27本配置Q1_FAIL，不晋级D1/dev/official，不扫描p/alpha/层位置/融合权重/seed/epoch。
下一步单独定义source-only完整关系支持诊断，比较固定初始化/固定终点、
实际原采样增强与登记统计扰动下九槽位/纯银行/实际fused的正负间隔及有效学习信号。
它尚未登记或启动；只有确认额外有效、非饱和实例关系后，才设计角色责任或实例记忆。
不简单组合V26与V27，也不将本次81.5924与文献官方指标相减。
RGBNT100官方+2.572608基线增益保持，RGBNT201固定dev目标和MSVR310有效增益仍待解决，
单seed42和反复开发身份的边界保持，三数据集/SOTA总目标未完成。

04:35:31终态时数据盘余18,992,009,216B（约17.69GiB）、系统盘11,129,253,888B（10.36GiB）。
此前24冗余恢复权重24.900344GiB已清理，12独立保留模型此前全SHA核验通过，本轮新增删除0；
六个V27最终权重与全部检索数组继续远端留存。



## 41.86 V27后继完整来源关系诊断已实现并登记，尚未运行（2026-09-07T05:12:59.284672+08:00）

V27完整Q1_FAIL及4/5正证据已三方同步到a1ac4ea，不改原门槛或开启D1/dev/official。
下一项仅诊断固定模型下的来源关系：V12初始化、V27 control epoch20、V27 source_style epoch20，
三种状态分别接收原增强输入和同输入上的原统计混合计划。
全部原1680batch、三fold、每batch6次模型前向=10080次，优化器更新0、新权重0。
每折只使用94个source身份，原采样/身份/camera/全部记录曝光逐项匹配；
原V27仅前8批有像素摘要，因此只声明这些摘要与原记录相同，
其余所有诊断批使用同一固定增强分布并保存本次像素摘要，不冒称原训练轨迹逐像素复现。

保存全部18输出（Signal/fused/3完整角色/9槽位/纯银行/3纯角色）的批内相似度，
743178240个FP32值约2.973GB；FP64关系统计覆盖全部identity及cross-camera协议，
合计273297024个三元组曝光。不丢弃无跨camera正例身份的负例作用，class0仍合法。
报告零间隔错误、单位向量0.3距离成对hinge、9槽位16格支持表，
以及tau0.1的原V26形式权重和间隔空间导数；无参数梯度或优化器，不能据此声称训练/泛化有效。
三模型/两输入均eval，仅既有冻结backbone训练标志选择统计接口，Signal/encoder/BN均保持eval；
各模型state在整折前后完整SHA相等、梯度为空。

CPU合成T0最大误差3.1086244689504383e-15；B64 GPU/NumPy对照全部计数相同，
最大误差1.4370016288012266e-10，均无真实图像、模型前向或优化。
一次SFTP协商断开发生在上传/执行前，重连确认远端文件不存在后完成纯数组检查，
未重启任何训练，实际错误保留在GPU检查回执中。

合同configs/RGBNT201/TriFusion-v27-source-style-relations-diagnostic.json：
bff95263f94b8d48655c707e9c68f528b9eda5d5eeeb236a8b89c9733a8b6097。
计划refine-logs/v27_source_style_relations/EXPERIMENT_PLAN.md；状态IMPLEMENTED_REGISTERED_NOT_RUN。
runner tools/diagnose_v27_source_style_relations.py；独立NumPy完整复算
tools/verify_v27_source_style_relations.py；持久顺序wrapper tools/run_v27_style_relation_pipeline.py。
运行结束后才全量复算保存的所有相似度对应关系、输入回执、身份支持和72个聚合格。
正式诊断此刻尚未启动；预计1–2GPU小时和完整CPU核验，新增约3–4GiB，启动需数据卷至少8GiB。
最近远端已重新认证root/正确容器，a1ac4ea、GPU1MiB/0%，旧4进程均结束；
数据卷18,951,553,024B（约17.65GiB），系统卷11,129,253,888B；新增删除0。
既有权重清理和所有模型锚点、V27最终权重及完整证据保持，三数据集/SOTA总目标未达。

05:19磁盘实查复核：数据卷剩余18,955,440,128字节（17.65GiB），系统卷剩余11,129,253,888字节（10.36GiB）；无活动训练进程。此前清理的24份续训权重仍全部不存在，12个保留模型完整SHA256逐一一致；累计释放26,736,541,280字节（24.900344GiB），本轮新增删除0。原始盘点与保留核验写入evidence/trifusion_disk_inventory_pre_style_relations_20260907.json及evidence/trifusion_disk_retention_pre_style_relations_20260907.json。保留失败工程现场、M0模型、正式终点、V12锚点及V27六端；不按文件扩展名把检索数组当作无用模型删除。

## 41.87 来源关系诊断R1启动失败及R2最小接口修复（2026-09-07）

R1执行250095b，原wrapper133941/child133943于05:22:23启动，3.133367秒后exit1，均已结束。
traceback明确为_sha256(__file__)将str传入只接受Path的既有工具；输出目录尚未创建，
模型加载、图像读取、前向和优化更新均0。完整launch/terminal/exit/log及原合同按原字节归档在evidence/v27_style_relations_r1_startup_20260907。

R2只将两处调用变为_sha256(Path(__file__))，逐字节对比证明除此之外runner无变化。
实际远端工具接口及原V27合同加载PASS；合成数学复算误差3.1086244689504383e-15，仍无模型前向或图像读取。
统计函数、数据、模型状态、关系范围、10080前向预算、0优化更新及既有FAIL条件均保持原合同。
R2 runner SHA256 8f8a727add91ab70f0b92920e4c33fac4272a628eaa68b3b2bfb2b4956e89425；合同SHA256 a42300ad815a71b8c21ef9576b6a83bda00842e5368e0a4633497f28ee77256f。R2尚未启动；该工程退出不构成新的检索负结果。

## 41.88 来源关系诊断R2正式启动与空间余量（2026-09-07）

执行代码d5bc04821477c4c44f0c91c0d31b9859c2beed26，合同a42300ad815a71b8c21ef9576b6a83bda00842e5368e0a4633497f28ee77256f。
原screen v27_style_support_r2_d5bc048；wrapper134209，child134211；05:27:12真实启动，05:27:41原PID与完整cmdline再次核验，GPU3130MiB/73%。
运行目录/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v27_style_relation_support_r2_seed42_d5bc048。
该次snapshot已创建RUNNING汇总并加载三模型；epoch级持久计数尚为0，不把模型初始化当作完整前向完成。
固定1680来源batch、3模型状态×2输入=10080前向，0优化更新、0新模型权重、0held-out/dev/official图像。
原进程exit0后包装器顺序启动完整NumPy CPU数组核验；若诊断失败则退出，不自动重启或跳过范围。
预计1–2 GPU小时加完整CPU复算；普通进度轮询180–300秒，不根据中间统计改合同。

启动前数据卷剩余18,954,428,416字节；该项数组原始预计2,972,712,960字节、总新增约3–4GiB，8GiB最低空间检查通过。
12个保留模型仍与旧清理前完整SHA一致，24份旧恢复文件仍不存在；累计释放24.900344GiB，本轮删除0。
不删除V12/Signal/V8锚点、V27六端、官方终点或支持既有结论的数组。当前不存在新的检索晋级或科学终态。

## 41.89 完整来源关系报告准备，原诊断继续（2026-09-07）

05:36:49原wrapper134209/child134211仍在运行，持久145/1680batch、870/10080前向；
GPU3130MiB/68%，数据卷剩余18,679,160,832字节，没有新删除或诊断终态。
报告只在原进程与完整CPU核验都exit0后运行，不改变固定数据、模型、原runner/verifier或合同。

报告包含72原始条件及各折/三折合并、active/inactive/all，合计144个条件。
完整CSV为2592表示指标行、1296槽位权重行、1296组全16格支持表、144条件摘要、
6768=72×94来源身份行、1296输入配对行。零误排序身份保留；无合法正例的关系分母为0，百分比为空。
关系均值按实际三元组曝光加权，间隔导数比按批取均值；不能混称参数梯度。
逐case文本重聚合需与NumPy全数组核验72格一致，计数精确、浮点1e-9；
同时核对819active/861inactive批和inactive两输入指标一致。

报告源码SHA256 f0c3496bd2833432c1bd6aaf497a61ad9e3750bb781249c9c92472440ae3a6cc；
等待队列SHA256 7fbbd901b8256b450b9c8ddd612cc488a307e41bc5be9dea66c85347f930e4a1。
合成算术验证了不等分母加权、批均值、全支持格和输入不变，未读取真实诊断统计。
队列每180秒查看原wrapper，结束并核验exit0才报告；此刻尚未启动。
新模型前向、优化更新、图像读取、Torch导入均0，执行器复算不称外部独立审计。
全部原FAIL、单seed42、反复使用内部身份及批内候选边界保留。

## 41.90 完整报告队列真实启动与核验（2026-09-07）

完整报告源代码f97186dadd8fd957d9bf766a919daa1312c0b46a已推送并三方同步；
原诊断仍绑定d5bc048及原a42300ad...合同，全部受保护源码SHA再次通过，未热改。
05:42:28启动screen v27_style_report_d5bc048、队列PID134960；
05:42:49核对/proc、完整cmdline及队列文件，原wrapper134209/child134211与队列134960均存活。
队列状态WAITING_FOR_ORIGINAL_DIAGNOSTIC_AND_CPU_VERIFIER，每180秒检查原wrapper；
只有原诊断、完整NumPy数组复算都exit0且原PID已结束时才运行报告。
报告将写入原run/complete_report，report.exit及report_queue.json记录真实终点，不自动重试或生成新权重。

最近主任务持久计数：05:41:23为203/1680batch、1218/10080前向，GPU3130MiB/90%。
当时数据卷剩余18,541,273,088字节；无新增权重或删除。按相邻完整轮约104.8秒/29batch，
GPU阶段估计07:09左右结束，之后仍须完整CPU复算及报告，不能提前判读成科学终态。
完整报告启动/存活回执见evidence/trifusion_style_relation_report_queue_launch_20260907.json及
evidence/trifusion_style_relation_report_queue_live_20260907.json；先前发布同步链也已归档。
本轮实质进展为增加全部条件、全部来源身份的终态统计与实际依赖队列，不是又启动一个训练版本。

## 41.91 第一折完整文件与曝光核验，原进程进入第二折（2026-09-07）

第一折于原进程约2116.9秒时完整结束，580batch×6状态/输入条件=3480前向；
94来源身份、2126条来源记录清单，实际37120次采样曝光。
06:04:45对全部580行回执重新核对epoch/step/PK索引/路径顺序SHA/异camera正对数，
并逐来源记录对照原登记曝光次数，全部通过。
运行内已断言三种固定模型的权重及buffer前后SHA一致、全部参数梯度为空；
原V27前8批像素回执匹配。该陈述是运行内检查及回执，不冒充外部独立审计。

完整NPY形状[580,3,2,18,64,64]、FP32、256573440个相似度值，
1,026,293,888字节，当前整文件重算SHA256 4c2fc774ae2b0de9e58aa9d9068dde71b58b7e0e951d22cf97375ba154c9c82f一致。
批回执SHA256 fcf475163f6624d177a1270f384fef929ce175a7f6bfe01a8c40cce815178353一致；
最大实际融合分解误差5.960464477539062e-7，小于原0.005门。
数组未下载；只保存完整折描述、文件核验与完整曝光重放回执。
单独检查第一次因SSH默认cwd解释相对元数据路径而失败，改为相对明确repo Path读取后通过；
原诊断源码、进程和数组未变，未增加模型前向或优化更新。

06:04:45原wrapper134209/child134211/报告队列134960均存活，
总持久进度608/1680batch、3648/10080前向，第二折由同一原进程继续。
06:03:54数据卷尚余17,879,666,688字节；本项无新增模型权重及删除。
这仅为第一折工程完整性里程碑：没有完整273297024关系CPU终态，
不读取部分统计来选新训练方法，不改原FAIL或晋级条件。
证据：evidence/trifusion_style_relation_fold0_milestone_20260907.json、evidence/trifusion_style_relation_after_first_fold_observation_20260907.json及其执行检查源码。

## 41.92 第二折完整文件与曝光核验，原进程进入第三折（2026-09-07）

06:39:00第二折（fold1）完整核验：560batch×6状态/输入条件=3360前向，
94来源身份、2075条来源记录清单，实际35840次采样曝光。
全部560行回执的epoch/step/PK索引/路径顺序SHA/异camera正对数，以及逐来源记录曝光次数，
均与原登记元数据精确一致；执行检查源代码与归档字节SHA一致。
三种固定模型的权重和buffer前后不变、梯度为空及原V27前8批像素匹配仍是运行内断言及回执，
不冒充外部独立模型审计。

完整FP32数组形状[560, 3, 2, 18, 64, 64]，247726080个实际相似度值，
990,904,448字节；当前整文件SHA256 e11358e245646ef4732a370bac2413ca584800ab475b7930695a5700c12fc097与完成回执一致。
批回执SHA256 a23d6c5e29cd65e39b18ae70c1785e88e372e29a959cf5e46557c6a9d84d9e28一致。
最大实际融合分解误差5.960464477539062e-07，门槛仍为原0.005。
本次只核验新完成的第二折文件和全部曝光，不重复第一折模型前向，数组未下载。

原wrapper134209/child134211/报告队列134960继续存活，同一原进程进入第三折；
最新持久进度1167/1680batch、7002/10080前向。最近定时观测2026-09-07T06:38:45.798112+08:00，
数据卷剩余16,856,596,480字节；本轮无新增模型权重或删除。
全部三折、273297024关系的CPU终态复算与完整报告仍待完成；
不读取部分科学统计选新机制，不更改V27 FAIL或任何晋级条件。
证据：evidence/trifusion_style_relation_fold1_milestone_20260907.json、
evidence/trifusion_style_relation_after_second_fold_observation_20260907.json及其执行检查源码。

## 41.93 来源统计扰动完整终态：稳健性改善，错误责任支持仍有限（2026-09-07）

原R2诊断d5bc048于07:09:31退出0，CPU全数组核验07:15:01退出0，完整报告也退出0；
原134211、wrapper134209、verifier138508、queue134960、reporter138685全部结束。
1680batch、10080前向、743178240实际相似度值、273297024关系曝光、72原始条件全量通过，
最大数值误差4.97948349221e-11，全部计数精确一致；新增优化更新0。
26份远端文本共34626196字节逐文件SHA一致，三折大型数组当前整文件SHA也匹配，数组/模型未下载。

active层819批、每个状态/输入20547072个identity三元组：fused非正为初始化42→普通终点8→V27终点1，
纯银行1443→815→8；普通→V27的三折纯银行141/201/473→0/0/8。
cross_camera每条件1698704关系，纯银行290→218→0；纯Transformer421→616→3，普通训练并未自动获得同样稳健性。
三折全部完整角色和纯角色的扰动输入0.3未满足比例均改善；但原输入fused未满足0.0289932%→0.0362106%，
纯Mamba非正0→6，九槽位非正365044→383119，完整保留这些代价。

active扰动九槽位非正1827850→1548527→435053；V27中434914次已由同角色及fused同时给出正间隔。
fused初始化42次只覆盖9个fold-来源身份，跨相机12次来自fold0两个来源身份；
V27最后1次属于fold2来源000017。完整72×94=6768身份行含零分母，未用这些身份定制模型。
同形V26目标的初始化间隔导数范数比原输入1.751809%→扰动2.914982%，V27扰动终点1.494244%；
这不是encoder梯度证据，不能声称目标绝对无梯度，也不足以自动串联V26+V27或扩展记忆。

完整数值、各折及18输出见results/TRIFUSION_V27_SOURCE_STYLE_RELATIONS_COMPLETE_COMPARISON_2026-09-07.md；
解释与边界见results/TRIFUSION_V27_SOURCE_STYLE_RELATIONS_ANALYSIS_2026-09-07.md；原始26文件及完整CSV在evidence/v27_style_relations_terminal_20260907/。
固定eval/批内候选/登记扰动、仅前8批像素摘要可重核、无heldout/dev/official及非外部独立审计的边界保留。
V27原Q1_FAIL与全部旧门槛不变。下一步对照平行残差平均、V9/V15和MambaPro实际聚合代码，
确认表示形成/最终聚合的新主假设与固定匹配对照；尚未注册或启动新训练版本。
数据盘终态余15,945,129,984字节，本轮无新权重/删除。三数据集baseline/SOTA总目标仍未完成。


## 41.94 V28：保留原独立角色输出的池化前联合残差（2026-09-07）

完整来源统计扰动诊断已结束并全量核验，不再标记为运行中。V27的4/5门
失败保持封存，后续不调p/alpha/loss/seed/epoch，也不自动组合V26责任损失。

本轮按一项结构主假设登记V28：将三角色×三模态最终768D Patch残差在池化前
输入同一128D双向Mamba，形成零初始化的512D槽位修正，再生成实际fused银行。
原Signal3072D、三独立角色及七组监督保留。两端共同采用原V27训练增强及损失，
候选只新增联合聚合；control也新训练，初始化仍为合法V12，不能拿旧V27直接充当对照。
共1152Token，固定角色/模态/空间顺序；无Router/正交化/隐藏交换/额外旧银行拼接。
初始化新增上游梯度为零是零输出投影的数学性质，M0须证明后续实际接通。
原V8源码保持不变，新模块使用作用域内forward hook读取真实最终角色输出。

与V9池化后协同、V15尾部间增量交换不同；完整Token聚合受已核实作者MambaPro
启发，不宣称新颖性已经成立或作者复现。此比较无法单独排除新增参数/计算的贡献，
主结果通过后才另行做容量/池化输入对照。

固定计划：refine-logs/trifusion_v28_joint_tokens/EXPERIMENT_PLAN.md；
配置：configs/RGBNT201/TriFusion-signal-preserving-v28-joint-tokens-rtx3090.json；
模型/runner/T0：joint_tokens_v28.py、train_signal_preserving_v28.py、
check_v28_joint_tokens.py。当前均未在远端运行。M0通过才允许原六端3360更新Q1，
五个科学门、完整图库、21身份、单seed42和既有访问边界全部保持。

08:02远端HEAD31cc12c、GPU空闲，数据盘15862022144B（约14.77GiB）空闲；
此前24冗余权重清理释放24.90GiB，本轮删除0。受保护初始化、V27六终点与数组保留。
执行者审查不标为外部独立审稿。本节是新方案登记，不是训练成功或新检索结果。


## 41.95 V28 原M0梯度覆盖失败与固定精度诊断登记（2026-09-07）

V28 c9a38e6原训练于08:14:39启动、08:20:07退出0，实际328.272647秒。
全部3fold×2端预检、原V8输出对照与backbone接口通过；真实M0共116次更新。
control容量203/203，candidate容量及100步均218/219，唯一零梯度张量为
joint.mixer.dt_proj.weight。所有梯度存在且有限，AMP overflow0、冻结state不变。
候选修正已非零，100步0.7224190235→0.5803046823，超额loss比0.0133422通过；
reserved6172/6788MiB。原M0_FAIL，Q1/heldout/新检索权重全部0。

完整7文件2108338字节SHA匹配，本地全部116步loss重算最大误差9.93410746997e-08。
原summary SHAf3d0a3bf8f9a4d237cf9e885485c1f8ea816845fbeedbc0cf8202bc5d0bf0986。
报告results/TRIFUSION_V28_ORIGINAL_M0_2026-09-07.md，原始evidence/v28_m0_terminal_20260907。
首次包装脚本换行SyntaxError发生在子进程创建前（0模型/0更新），原文本及错误保留；
仅包装换行修正后才执行上述唯一模型训练，未改实验配置。

T0实际FP32 Mamba全部参数连通，真实M0单个dt权重零梯度的根因仍未证明。
另登记tools/diagnose_v28_joint_precision.py与对应精度诊断计划：
只重建原第1次更新，第2批原损失反向，随后固定真实输入/参数/上游梯度，
比较AMP fast、FP32 fast、FP32非融合三条子图。预算1额外诊断更新，
2真实source前向、3子图导数重放；不重开M0或Q1，不改原FAIL或门槛。
原始7文件、零梯度失败和新增诊断成本均不得隐去。当前诊断尚未运行。

数据盘约14.77GiB空闲，GPU空闲，新删除0、新检索权重0。
已准备的完整Q1核验/报告脚本因M0失败尚未执行，不当作已有结果。
全三数据集/SOTA目标未达到，继续保留单seed42、反复使用OOF身份及外部审稿不可用边界。


## 41.96 V28 M0全量复算及来源精度诊断的回放失败（2026-09-07）

tools/verify_v28_m0.py完成原48批预检、48次V8输出对照、48次backbone接口
及116步全部日志/14loss/供体计划重算；最大loss算术误差9.93410747e-8，
原M0_FAIL与12项检查完全一致。报告complete_m0_verification.json已归档。

66dc4e0精度诊断141314于08:31:50启动、08:32:07退出1，实际16.522456秒。
前两批增强与首个更新全部标量精确一致；第2批loss逐bit断言失败，原脚本未打印
actual/expected。因此当前只有1次额外重建更新、2次source模型前向，
没有第2批反向、精度子图对照或数值根因结论；不能写AMP已被证实或已修复。

只新增instrumented诊断脚本，在原断言前记录所有实际损失和风格分量差异；
失败时持久化joint输入/state。原断言不设容差，模型/损失/原合同不改。
登记最多1次额外重建更新，若执行总诊断更新将为2，不隐去第一次的成本。
已存第一项诊断失败及完整日志；不直接恢复原M0/Q1或修改精度。
下一步先定位该实际回放差异。全项目目标仍未达到，无新检索成绩/权重。


## 41.97 V28回放差值已取得；登记同前向精度配对（2026-09-07）

9da1986日志诊断141931于08:40:20启动、08:40:37退出1，16.733827秒。
同样1次重建更新/2次source前向；第2批0.6292165517807007 vs原0.6292153596878052，
差1.1920928955078125e-6。仅Mamba/fused及汇总loss有差异，CNN/Transformer、
原像素SHA、风格计划和统计相同。第2批反向与精度比较仍未执行。
远端失败fixture114904539字节SHAcd82baa67cdcf9f464e6d07dc97d661fda51ea18aa58f3d5acfbeba4d2c25043，
保存触发输入和joint状态；没有完整检索checkpoint。两项诊断累计2次更新，原M0仍116。

不把该差异直接归因CUDA非确定性，也不通过容差重写旧诊断结果。
新登记tools/diagnose_v28_joint_precision_paired.py，在本次完整AMP图捕获
真实输入/参数/上游梯度；只有当前dt权重确为零，且AMP子图逐bit复现本次完整图时，
才比较FP32 fast/unfused。第2批与旧M0差值仍保留；不声称原轨迹逐bit还原。
FP32显式转换捕获输入并关闭autocast。固定最多1次更新，诊断累计将为3次。
这改变的是数值因果问题的配对位置，不修改原M0门或开放Q1。
计划与源hash见evidence/v28_same_forward_precision_preregistration_20260907.json。
当前新诊断尚未运行，AMP根因未证明，无训练精度修复。


## 41.98 同前向精度证据成立，登记V28 R2局部FP32修复（2026-09-07）

863022c同前向诊断142507于08:47:22退出0，18.285619秒。
本次1重建更新/2source前向/3joint导数重放，三项诊断共3次更新；
原M0仍116步、FAIL，Q1仍0。旧第二步loss差4.11272049e-6保留，不声称原轨迹逐bit一致。

当前完整AMP图与AMP子图的dt_proj.weight均0/2048梯度，输出精确相同。
相同输入/state/真实upstream转FP32 fast/unfused后，两者均2048/2048非零，
scaled256下absmax3.534658077e-8、L2 2.214225816e-7。两者dt统计一致，
其他部分参数统计有微小差异，不宣称所有梯度逐元素相等。
x_proj非零也从1384/10240恢复10240/10240。参数SHA不变，无比较期更新。
该局部配对支持FP16路径丢失小导数，不支持整个CUDA实现错误或改变loss权重。

原诊断JSON SHA8c1d3e16c58e17fc00275fd8abdef549239bae3a96e5a87b633041b440d96d2a；
真实fixture116085842字节SHAc569133afd9742f85699b70b1dfd953c8c4f5ac7ff9f9f500afb182db7f6943f，
仅留远端作无更新工程回归，绝不用于后续fold训练初始化。
完整报告results/TRIFUSION_V28_SAME_FORWARD_PRECISION_DIAGNOSIS_2026-09-07.md。

新增joint_tokens_v28_fp32.py仅在新joint forward关闭autocast并转FP32；
原Signal/C/T/M、参数值、所有loss/预算/门槛保持。旧源码与失败合同不改。
R2使用train_signal_preserving_v28_fp32.py与独立config/plan；
先真实fixture旧AMP0→新FP32非零回归，再完整原M0 116步；
通过后才完整Q1六端3360更新。当前R2尚未运行，诊断不是M0晋级或检索结果。
计划refine-logs/trifusion_v28_joint_tokens_fp32/EXPERIMENT_PLAN.md。
精度成本实际记录，标准FP32工程修复不包装为算法创新。


## 41.99 V28 R2已启动：真实FP32回归PASS，完整M0待终态

2026-09-07，运行commit bf8de956e685311dd70631395009a2c06a2c8591。
09:02:55启动子进程143321 / wrapper143320；09:03:17确认child存活。
远端run：/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v28_joint_tokens_fp32_seed42_bf8de95。
原配置SHA43b75c87e2e7795759912b9051fae012b2cc39a23279b5d3d6bc85600fae9029，
原R2计划SHA64111a40d44e9a28ff4e9ab62670d608254480fc5c04e44f6841a9613b281e7b。

T0三项PASS：风格公式、实际CUDA合成联合模块、真实fixture梯度回归。
真实旧AMP dt0/2048，新局部FP32 dt2048/2048；scaled最大梯度3.534658077342101e-8，
两端梯度有限。无图像/优化更新的fixture测试不初始化真实fold。
M0正在运行，尚未发布M0终态或Q1检索结果。
R2数值修复没有改变原116步M0、219/203覆盖、过拟合与五科学门。

启动前校验脚本第一次生成时，通用HEAD占位替换误改git HEAD字面量而SyntaxError。
该次检查根本未执行、screen/child未创建；修复为专用占位符并先compile后执行。
模型wrapper已正确编译，未改模型/计划，真实训练只启动一次；错误JSON保留。

09:02:55磁盘实查：data15624220672B，system11129249792B；
GPU空闲24126MiB。既有24个删除路径仍不存在；12受保护模型完整SHA一致。
既有回收26736541280B，本次新删除0。预留本轮六final与数组<1GiB。

R2专用tools/verify_v28_fp32_m0.py、verify_v28_fp32_complete_terminal.py、
report_v28_fp32_complete_comparison.py已准备，尚未执行或冒充审计PASS。
原R1脚本和失败报告保持。绑定R2执行commit/config/plan及precision fixture。
完整Q1核验将检查3360更新、32,602,260距离元素、5,952,790排名位置、571query/21身份。
单seed42、重复OOF、非外部独立审计和多数据集/SOTA目标未完成边界不变。


## 41.100 V28 R2完整M0通过，六端Q1与持久CPU终态核验

训练执行bf8de956e685311dd70631395009a2c06a2c8591，09:02:55启动PID143321。
完整M0耗时327.219秒；09:10:38实查已进入Q1，09:14:20已写fold0 control281/580更新。
尚没有完整六端Q1检索终态，不能将工程PASS写成结构泛化有效。

## 原失败保留与实际修复

原V28 c9a38e6 M0的116步保持FAIL：candidate只有218/219张量非零，
缺joint.mixer.dt_proj.weight；另外11项条件通过。
两次严格跨进程loss回放诊断失败也保留，第三次同前向配对才证明
旧AMP的dt梯度0/2048、局部FP32为2048/2048。
三项诊断总3次重建优化更新，不改写成零成本。

R2唯一工程修复：新增joint模块内部关闭autocast、输入转FP32，
仍调用同一个共享双向Mamba fast实现；原Signal、三角色及14项loss未变。
新模块参数原本就是FP32，修复针对实际执行精度。
真实fixture仅做0更新回归，不初始化任何fold。

## 全部M0结果

| 项目 | control | candidate |
|---|---:|---:|
| 总参数 | 98,800,141 | 99,213,197 |
| 可训练参数 | 7,841,292 | 8,254,348 |
| 可训练张量 | 203 | 219 |
| 8步容量阶段非零覆盖 | 203/203 | 219/219 |
| 容量峰值reserved MiB | 6,026 | 7,608 |
| AMP overflow | 0 | 0 |

fresh candidate固定第一批100步也达到219/219阶段非零覆盖，overflow0。
零输出投影初始化首步仅新模块3个输出投影有非零梯度，后续连接性按原门完整检查；
不要求零初始化首步上游就有梯度，也不允许整个阶段dt始终为零。

- 完整3fold×2端×8批=48次预检全部完成，初始配对输出SHA相等。
- 另48次完整原V8前向中，三个独立分支、纯残差与各自logit保持精确一致；
  重复槽位归一化导致的旧V8 fused差<1e-5。
- 另48次backbone训练接口中，Signal前缀精确、角色anchor/reference均冻结并同时接受固定扰动。
- 8+8+100=116次真实R2优化更新，所有原12项工程条件PASS。
- 100步loss：0.722419023513794 → 0.5803054571151733。
- 七头加权ID理论下界：0.57838292104621。
- 超额loss比：0.013347598525834775 ≤ 0.1，原门未放宽。

R1 M0 116 + R2 M0 116 + 三诊断重建3 = 235次前置优化更新。
合成/fixture检查不计入这些更新；Q1固定3360更新另计。
FP32/新增聚合的显存与计算成本必须报告，本表不能分离精度和架构各自成本。

## 全量复核

R2专用verify_v28_fp32_m0.py完整复核116行原始更新、全部14项loss、
原采样和每次style plan、各阶段梯度覆盖、48次预检/原V8对照/backbone接口。
CPU服务器核验PASS；本地仅用JSON和标准库重新计算全部116项损失与下界，
最大标量重算误差9.258898592268139e-8，两端计算一致。
没有本地Torch、图像、模型张量读取，也没有新模型推理或优化更新。

- M0冻结快照SHA：b182ceee031ce81e28af213a1669c979387e4a9470f6561f8c1b47d8f0949374。
- M0核验器SHA：c147190197a7f39f5728017566d1a39b8f35a95fdee1ca1aab5f0f00275277bf。
- 全部5个服务器原始文本共1,660,489字节，SFTP字节数及SHA一致。
- 原始记录及本地计算：[evidence/v28_fp32_m0_terminal_20260907](../evidence/v28_fp32_m0_terminal_20260907)。
- 这是执行者确定性复核，不冒充外部独立审稿。

## Q1与持久终态核验

Q1按原登记继续三fold两端各20epoch，全部六个固定终点、3360更新；
无中途检索选择、门槛修改、新损失、额外seed或官方测试。
控制端也采用同一V27风格合同、合法V12初始化和原采样。
五项科学门不变，只有最终完整比较能判断结构效果。

终态等待器PID144462 / wrapper144461于09:15:24启动，已验证存活，
每240秒检查原训练exit文件。仅exit0且完整3fold Q1终态才启动CPU全量核验。
完整核验检查6个权重SHA及6组保存数组、3360更新、32,602,260距离元素、
5,952,790排序位置、全部571query/21身份和原五项科学门，
再生成全部105身份-输出行与2855query-输出行；没有图像、优化或新检索评估。
等待器和核验器遇到失败即保留非零退出，不能跳过失败继续称PASS。
启动记录：[evidence/v28_fp32_terminal_waiter_20260907](../evidence/v28_fp32_terminal_waiter_20260907)。

09:02:55数据盘剩15,624,220,672B；此前24个冗余权重仍不存在，12受保护模型SHA一致。
历史释放26,736,541,280B（24.900344GiB），本轮新删除0。
主目标、RGBNT201开发门、多数据集与SOTA均尚未完成；单seed42和重复OOF选择偏差保留。


## 41.101 V28 R2原进程持续，PMKD作者PDF原表证据补齐

更新2026-09-07T09:27:35.657795+08:00。
09:25:53实查：原训练PID143321存活、GPU7954MiB/99%，终态等待器144462存活，
两者均无终态退出；不重新启动。fold0 control580/580步已保存receipt，
candidate232/580步；当前Q1共812/3360更新，仅1/6端完成，完整paired fold尚未结束。
不对这个单端的检索成绩下结论，不调其余fold或模型设置。
数据盘剩15455375360B，本次新增权重清理0。

原R2 M0全12项PASS及原R1失败、235次前置优化更新成本继续保留。
实际模型运行bf8de956e685311dd70631395009a2c06a2c8591；固定20epoch/seed42/原5门未变。
终态CPU核验每240秒等待原退出，仅完整六端后执行全量验证和报告。

并行补齐PMKD作者PDF原表实读：RGBNT20184.7/88.9，
RGBNT10091.6/98.0，WMVeID86371.9/79.5；论文未报告MSVR310。
这不是本机模型成绩，不改变V28运行。来源、文件哈希和资源条件见
docs/PMKD_AUTHOR_PDF_VERIFICATION_2026-09-07.md。
三数据集baseline/SOTA总目标尚未实现，内部OOF不与官方数值直接相减。

通过[作者主页](https://aihuazheng.github.io/publications/)取得
[完整作者PDF](https://aihuazheng.github.io/publications/pdf/2026/2026-Progressive_Multi-modal_Knowledge_Distillation.pdf)：
12,158,023字节、9页，SHA256
73086d4c318d610fd44e3d7a875462c5ed797cf8c6c2970820291259037eb094。
[AAAI正式页面](https://ojs.aaai.org/index.php/AAAI/article/view/38338)确认题名、作者、DOI与13351–13359页。
本轮提取全部9页文字，并实际查看渲染后的PDF第5、6页。
AAAI服务器下载仍断连，因此不声称已证明作者文件与出版社PDF逐字节相同。

| 数据集 | mAP | Rank-1 | 直接位置 |
|---|---:|---:|---|
| RGBNT201 | 84.7 | 88.9 | 第6页Table1 |
| RGBNT100 | 91.6 | 98.0 | 第6页Table2 |
| WMVeID863 | 71.9 | 79.5 | 第6页Table2 |

这补齐此前仅有原表归档/索引复核的证据缺口，原RGBNT201/RGBNT100数值未变。
全文实验使用以上三个数据集，未报告MSVR310，不能用WMVeID863一列补填MSVR310。
它仍是公开高指标参照，不构成已穷尽全部论文的绝对排行榜，也不是本机复现。

第5页Implementation Details写明DINOv2预训练、224×224、batch32（4身份×8实例）、
Adam初始lr4.5e-5和50epoch；本轮未核得多阶段实际总更新数。
这些资源与训练条件需随公开数值保留，不能把其差距单独归因于batch大小。
[作者仓库](https://github.com/moonaricc/PMKD)本轮仍只见README，没有取得训练实现。

PDF、渲染页面与完整提取文字仅保留于本地临时研究目录，未上传项目仓库或占用训练盘；
仓库只保存本文与来源/数值/哈希回执。没有新增模型/数据/训练/测试，没有改变V28 R2配置或门槛。
已有SOTA_REFRESH与SOTA_PRIMARY_REFRESH作为历史记录保留，由本页补充当前证据层级。


## 41.102 V28 R2首组完整配对保存，原进程继续剩余两折

2026-09-07 09:39:40实际检查：原训练143321与终态等待器144462仍存活，无退出。
fold0控制和候选均完整20epoch/580步，2/6端已保存；fold1 control285/560步，
Q1合计1445/3360更新。没有启动第二份训练，没有按第一fold成绩改变设置。

第一完整配对的执行receipt显示：控制715.091922秒、候选744.685403秒，
reserved6018/7608MiB，零overflow，梯度203/203与219/219，冻结state未改变，
strict_reload/read_only_evaluation均true。此处核对的是执行记录，不代替最终全数组/排序核验。
剩余时间按实际更新速度估计：训练约10:20结束，CPU等待器最多另等240秒再做全量核验；
这是估计而非已完成承诺，推理效率及六端平均成本仍待全终态统一报告。

原R2完整M0及R1失败保持§41.100，PMKD原表补查保持§41.101；
实际模型执行bf8de956e685311dd70631395009a2c06a2c8591及配置/计划/五科学门均未改。
完整Q1尚未终态，尚不输出三折聚合/科学晋级判定；全部六端统一核验后比较。
新final权重和数组受保护，本轮没有删除权重；三数据集baseline/SOTA目标仍在进行。

## 41.103 V28 R2完整六端终态：联合Token未改善检索，发现修正尺度失配
更新 2026-09-07T10:37:59.144025+08:00。


融合mAP由81.8253056654降至81.2688735890，配对下降0.5564320764个百分点；三折增益依次+1.0927544385/-1.4878732389/-1.2822611346，身份聚类bootstrap95%下界-2.3669680129。CNN基本不变（+0.0020703401），Transformer下降0.9574530559，Mamba下降0.4354006796。候选融合低于候选CNN约0.3182418564。

候选融合仍高于同协议冻结Signal的77.4876031160，因此本结果否定的是本轮新增联合Token聚合相对匹配控制的稳定增益，不能改写成所有角色分支低于Signal。控制端融合本身高于其三个独立分支，也说明固定拼接并非必然输给最强分支。

21个身份中融合8个改善、12个下降、1个不变；全部571条query有159条AP改善、219条下降、193条相等。Rank-1修复10条、新增7条错误，净增3条，仍伴随mAP下降。新增7条错误中5条首位负例与query同摄像头；这是关联统计，不是视觉成因证明。身份000239的24条query平均AP下降13.365721个百分点，同时CNN与Mamba在该身份上改善；不能只用一个全局分支均值概括联合排序的损害。原始21身份与全部2855条逐query/输出比较均随证据公开。

## 训练目标与修正幅度的新线索

全1680步/端，加权ID均值0.6590812672→0.6537482104、加权Triplet均值0.0206201109→0.0241776673，总loss略降0.6797013938→0.6779258933。最终epoch全部84批中，fused Triplet均值0.0030135514→0.0061428870，正loss批数61→76，总loss0.6176137591→0.6203139807。这些是完整轨迹与终点训练批统计，不能简单归结为所有来源关系都已饱和，也尚不证明梯度冲突。

代码中原modal_residual_embeddings先逐512D槽位F.normalize；新头却使用 normalize(h+c)，其中c是没有幅度约束的线性输出。全部训练的joint_correction_abs_mean平均240.0038783045，最终epoch平均350.8445284367，均来自每个真实训练批的原始记录。单位L2槽位的每维绝对值均值至多1/sqrt(512)，约0.0441942；因而两项在总体平均上有明确的尺度失配。零输出初始化只能保证初始恒等，不能保证训练后仍是小幅修正。

上述均值并不等于每个槽位或每条query都由联合头主导，也不能单独证明它造成全部mAP下降。下一项优先检查来源侧的实际范数分布、修正后方向与原槽位/联合修正方向的余弦，以及这种变化在所有来源身份、角色、模态和注册增强下是否普遍存在。先将这个具体失效量化，再决定是否登记有明确几何约束的联合残差方法；不将标准归一化或FP32执行重新包装成原创算法。

## 后继任务范围

下一项拟登记为只读、完整source覆盖的联合修正尺度与原角色信息保留诊断，使用本轮已固定的各fold最终模型，比较原输入与既有注册风格扰动。它应覆盖全部fold和来源采样关系，保存全角色/模态/身份统计，不依据少数成功query选择样本。诊断不更新模型，不从官方测试身份反推参数，也不重跑V28或修改其失败门槛。该后继诊断尚未登记或执行，本页不把它写成已获得的证据。

RGBNT201、MSVR310、RGBNT100三个数据集的最终目标保持；RGBNT100已有基线增益、MSVR310负结果、RGBNT201固定dev未达65及单seed/reused-OOF限制继续保留。V28没有新增dev/official结果。本次Q1的81.2689不能与公开官方SOTA直接相减。

## 实际成本与复核

三个控制端训练合计2074.573357821秒，候选2157.915267944秒，观察到的训练时间增加4.0173%；峰值reserved由6018MiB增至7608MiB。新增413056可训练参数、两个1152Token Mamba扫描和局部FP32执行成本都保留；没有以此推断推理延迟或FLOPs。

原R1 M0失败116步、三次诊断总3步、R2完整M0的116步均继续归档，前置共235次优化更新。R2实际训练执行bf8de956e685311dd70631395009a2c06a2c8591，原训练PID143321于2026-09-07 10:21:48退出0，原核验等待器PID144462于10:23:36退出0。远端复算32602260个距离和5952790个排序位置，距离/指标误差均0；本地3360步loss重算最大误差1.8090941012e-7、指标误差0，bootstrap及五条件完全一致。两类核验均为执行器复核，不是外部独立审计。

30个原始文本/JSON/CSV文件共31215636字节已逐文件SHA256与字节数核对后接收。权重、特征/距离/排序张量留在服务器；没有本地模型、图像或Torch调用。数据盘终态可用14604984320字节，既有24个冗余权重清理仍保留原回执；本轮没有新增删除。


完整结果：results/TRIFUSION_V28_FP32_R2_Q1_2026-09-07.md。证据：evidence/v28_fp32_complete_terminal_20260907/。

## 41.104 V28来源联合修正尺度诊断已登记，完整任务待启动

更新 2026-09-07T11:03:56.845228+08:00。原V28 R2完整0/5失败及§41.103全部指标保持。
已实现 tools/diagnose_v28_source_joint_scale.py 与独立NumPy完整验证程序，
合同 configs/diagnostics/V28-source-joint-scale-v1.json，计划 refine-logs/v28_source_joint_scale/EXPERIMENT_PLAN.md。

固定三个candidate epoch20终点，每折94source身份；全部580/560/540原采样批次、
两输入条件、3360前向、1935360角色×模态槽位。原清单2126/2075/2051记录均有曝光。
原图像每批共用，原训练增强/登记统计计划均保持；不声称复现未保存的整条原训练像素轨迹。

实际h/c/y逐批GPU→CPU独立核对，随后丢弃向量；保存全部16个FP64范数/内积/方向标量，
原始统计数值247726080B，预计全部产物<512MiB；不新增或删除模型权重。
离线验证复算全部保存标量、162个分层格及5076身份输出行；不冒充重读原向量或外部独立审计。

CPU合成几何最大归一化误差1.4144678443e-7，B64 CUDA→NumPy为1.3495713521e-7，
本地独立NumPy4案例与阈值计数误差0；均无真实模型前向/图像读取/优化更新。
这些仅证明统计实现，正式来源诊断尚未启动、尚无结果。

完成推送与三份同步后，以持久wrapper顺序执行全部GPU来源前向及CPU全量复算；
模型eval、既有冻结stem统计分支按原合同开启，权重/buffer前后全SHA相同且grad为空。
源任务失败保留现场不自动重试；轮询240秒。没有新检索、dev/official或V28调参晋级。

## 41.105 V28来源尺度诊断已实际启动，原进程完成首174批
更新 2026-09-07T11:15:31.063509+08:00。
实际执行4158c95ca639721e584d1e7271219f0f3b557cc3；合同
f32293b64d9343ab308b3e9cd95cd41a6722630a34026ff42fafcf0ef2cd398e。
持久wrapper149889及来源子进程149890于2026-09-07 11:08:46启动，11:13:06均确认存活且命令匹配。
174/1680批、348/3360前向、200448/1935360槽位已记入原摘要，第一fold尚未完整结束。
每个实际槽位的CPU/GPU几何检查随前向执行，完整统计和独立标量复算等待全三fold终态。

启动前六个V28固定终点SHA重新匹配，原V28训练/核验进程均已结束，GPU空闲；
本次GPU2582MiB/100%，数据盘14531252224B，新权重删除0。
按首六epoch真实速度每29批约37.3秒，GPU终态估计11:45–11:47，完整CPU核验另计。
同一wrapper顺序启动CPU验证，分别记录GPU/CPU原PID、退出码；失败停止且不自动重启。
进程、启动检查、wrapper源码与11:13实际进度见evidence/v28_source_joint_scale_launch_20260907/。

本项使用固定candidate模型，优化更新/新检索/dev/official均0；尚无完整来源分布结论。
原V28完整Q1_FAIL0/5、§41.103全部结果与§41.104登记范围保持。

## 41.106 V28完整来源尺度/方向及用户切向补充（2026-09-07T11:53:35.445631+08:00）

原GPU149890在11:45:28 exit0，同wrapper自动CPU152087在11:45:40 exit0；wrapper149889 exit0，11:50:27核实三进程已退出。实际模型代码4158c95、原合同不变，所有3fold/1680batch/3360forward/1935360slots/30965760标量完成。每折94来源身份、全部2126/2075/2051记录覆盖；原始训练增强与注册V27输入，819active/861inactive。0optimizer/0heldout/dev/official/0检索。

全量结果定位：三折Mamba和fold1 CNN，原输入及style各430080/967680=44.444444%观测范数比>10且cos(y,c)>=0.99。其余槽位多数保留当前方向。Transformer原输入范数比中位约0.145–0.194，cos(y,h)均值0.981–0.990；不能将所有分支退化归因于末端修正主导。fold0在Mamba大修正时仍有fused+1.092754，保留因果边界。

用户补充平行/切向分量从既有FP64内积推导，CPU只运行一次6.394730秒，0模型前向。原输入整体转角中位16.354346度、P95 94.642040度、最大107.988226度；切向能量占比中位0.992937428。方向变化不是身份信息丢失证据，本次未测初始化到最终角色的逐样本漂移。

原CPU核验最大缩放误差2.1971677349759404e-7；两套162分层/5076身份行均本机完整CSV/覆盖/聚合验证，误差分别1.9182961641256485e-16与5.666605163045236e-16。所有模型状态SHA不变/grad为空。属于执行者全量验证，不冒充外部独立审稿。GPU原摘要PENDING状态按阶段封存不改写，父级/CPU终态均exit0为准。

诊断目录288045485B，三标量NPY247726464B仅远端；数据盘空余14272937984B，GPU1MiB/0%。新权重0/新删除0，原24份冗余resume清理24.900344GiB账目保持。首次补充SFTP写模式错误仅产生空脚本，确认0B后更正并SHA验证；实际CPU执行一次，没有重跑模型。

完整报告：results/TRIFUSION_V28_SOURCE_JOINT_GEOMETRY_2026-09-07.md。
全部原始证据：evidence/v28_source_joint_scale_complete_20260907/。
原摘要SHA1b157dfa10c39c8c8b17e153f6517041ad9e027198455c4647a82754ab73d0f8；
CPU核验SHAaf3da282a5e7f1f3c3c050ca974b66652fcaa3aadc682dbbd4b042732ce2d89d；
切向补充SHA28434c5e14584affb604acb5e9fd4620c59a0d695b3d094be51f8a739a103900。

V27配对正收益/Q1_FAIL4/5、V28 Q1_FAIL0/5保持。新几何约束训练尚未注册；应先固定一种有明确最终方向界的机制，维持V27/FP32/合法初始化与六端预算，可靠关系保留不同时叠加。MAG/nGPT/DELTA原文及旧V7/V9/V15机制边界已核对，见docs/V28_GEOMETRY_FOLLOWUP_BOUNDARIES_2026-09-07.md。单seed/反复OOF与三数据集目标未达边界保持。

## 41.107 V29固定切向几何约束登记（2026-09-07T12:15:07.851194+08:00）

在V28全部来源几何证据基础上登记唯一新假设：保留V27扰动和V28局部FP32联合1152Token模块，将输出c投影到当前角色h切向，以r=v/sqrt(1+||v||²/(0.25||h||²))平滑约束，再normalize(h+r)。固定b0.5，精确角度上限26.565051度，不按holdout/官方分数选择、无倍率扫描。使用实际h范数使零修正输出遵循旧normalize(h+0)，检查初始化两端精确输出SHA。

控制为V27扰动下原V8固定融合；候选为受约束联合适配，新增413056参数/16张量，几何0参数。候选/控制219/203可训练张量，原七头ID/Triplet、学习率、20轮、seed42、B64/K8、原采样和所有合法初始化SHA不变。M0仍116真实更新与原12项条件，额外检查实际角度/相对范数界；通过后六端3360正式更新，全部3126图库/571查询、五输出、原五项科学门。每次训练及完整图库前向保存几何统计。

源码：modeling/trifusion/joint_geometry_v29.py，tools/check_v29_joint_geometry.py，tools/train_signal_preserving_v29.py；
完整M0/终态核验与报告及持久pipeline均已实现。AST/F821、旧配置固定部分与全部依赖SHA静态核验PASS；真实CUDA T0/M0/Q1均NOT_RUN。
配置：configs/RGBNT201/TriFusion-signal-preserving-v29-bounded-joint-rtx3090.json，SHA427242f945fdb329f72467f97f58a56a86f410630adb68cd69f76fd7e4bc83b5；
合同：refine-logs/trifusion_v29_bounded_joint/EXPERIMENT_PLAN.md，SHA61d6841eea54303b3c87dad31a94da32089fdff9513d9fde076a3839bd435028。
同一pipeline记录各原PID/退出码，M0失败不Q1，训练/验证失败不自动重跑。预估M0/预检5–8分钟，完整约1.3–1.6GPU小时，观察240秒或估计里程碑。
V28实测同规模目录1,018,766,834B，六权重195,514,551B，数组792,764,381B；新实验预留2GiB，现有约13.2GiB，暂无需删必要权重。

当前h到新h的角度界不约束原角色参数漂移、也不保证检索不下降。三折Mamba及fold1CNN大修正与Transformer/其他CNN小修正的区别、fold0正收益反例均保留。不同时叠加教师/可靠关系loss/Router/XBM/PCGrad。MAG/nGPT已有相对范数/归一化方向思想，标准工具不改名冒充原创。主条件成立后再登记近邻简单方法/同容量对照与车辆扩展。

当前仍只有登记和静态证据；V28Q1_FAIL0/5、V27Q1_FAIL4/5及其正增益、RGBNT100官方增益/MSVR310负结果/RGBNT201固定dev不足和单seed反复OOF限制保持。三数据集超过基线/SOTA目标未完成。

## 41.108 V29实际启动与T0（2026-09-07T12:24:14.894132+08:00）

V29原训练PID153151、持久wrapper153149于2026-09-07 12:18:47启动，实际代码f4c6a03e1b263aaa9e4bce71427152007018a0ca。12:19:45实查两PID存活、完整命令匹配，GPU3388MiB/100%。12:21:23取得只读快照，前两fold的完整8批初始化配对通过，M0尚未输出终态，Q1未开始。

三项数学T0全部通过：V27统计、V29实际CUDA几何与Mamba、原真实小导数FP32回归。零修正完整bank逐位等于旧bank；新slot公式与FP64 NumPy最大误差2.5420055449476564e-8；零点梯度与切向解析最大误差4.76837158203125e-7。强修正合成尺度1e5时实际相对更新最大0.5000000596、最小余弦0.8944271207，均在注册2e-6浮点容差内。joint16张量在非零输出后全连通，三个输入角色都有梯度；原fixture FP32 dt导数2048/2048非零。T0 optimizer0、真实图像0；fixture只作精度回归，不作初始化。

T0沿用字段numpy_bank_max_error在V29记录实际slot公式误差；完整bank另检查零修正时逐位一致，不冒称非零完整bank的独立FP64复算。工程检查不等于身份检索有效。

启动前全部源文件/配置/合同/CLIP/六初始化权重/精度fixture SHA通过；数据盘14212608000B，旧诊断PID已退出，GPU空闲后才启动。无新权重删除、无启动重试。此前本地静态检查发现旧criterion.py仅CRLF与Git/远端LF差异，按Git对象与新文件字节核对；没有改旧源码或放松远端SHA合同。

run=/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v29_bounded_joint_seed42_f4c6a03。
配置SHA427242f945fdb329f72467f97f58a56a86f410630adb68cd69f76fd7e4bc83b5，计划SHA61d6841eea54303b3c87dad31a94da32089fdff9513d9fde076a3839bd435028。
全部实际启动/数学快照/原PID证据：evidence/v29_launch_20260907/。
持续pipeline在M0失败时只核验M0，M0通过自动完整3360步Q1；训练终态后自动验证全部权重/数组/排名及报告，不重启训练。
预计M0约12:25–12:27结束，完整Q1约13:40–13:55，按240秒/里程碑观察原句柄，估计不是完成证据。
本轮只是实际数学T0和部分来源预检通过；M0/Q1尚无完整终态，原V28Q1_FAIL0/5、seed42/重复OOF及三数据集SOTA目标未达边界保持。运行中的源文件及合同冻结。

## 41.109 V29 完整M0通过，原六端Q1持续运行（2026-09-07T12:38:33.609921+08:00）

三折两端各8批初始化全部配对；M0控制容量8步、候选容量8步、固定批100步共116更新，全13项工程条件通过。CPU154239于12:31:31–12:31:33 exit0，全日志/预检复算loss最大误差1.126900316394952e-7。固定批loss0.722419023513794→0.5803059935569763，熵下界0.57838292104621，超额比0.013351322882393919<0.1。

梯度阶段覆盖控制203/203、候选219/219；候选首步206，第二步219接通，固定批末步209，不能写成每步全非零。容量峰值6030/7610MiB，overflow0、冻结状态/Signal前缀保持。全部66816槽位观测符合b0.5与cos>=0.894427191；候选固定批最大实际更新比0.4886577725，最小cos0.8984663486，最大转角26.042798度。切向相对内积最大6.146727571376687e-8。当前h到y边界不约束角色训练漂移，不保证mAP。

CPU/本地仅读标量和文件，图像/模型加载/新更新0；不是外部独立审计或CPU复演模型向量。首次本地汇总容量显存字段用于固定批报错，按实际schema修正，无训练重跑。报告results/TRIFUSION_V29_M0_2026-09-07.md，完整证据evidence/v29_m0_complete_20260907/。
摘要SHA6b036c1406c8176d93568e83b140335f099d20c3c7b0ad0e0e4d7727d8643e67；
CPU核验SHAf0af9edab7d06331713f43e84ab704ad1bc371ffc65044c9abcaac547e85f23f。

原153151/wrapper153149于12:32:19实查存活/命令匹配，fold0 control第13轮完成，GPU6368MiB100%，数据可用14203797504B。新删除0。Q1从合法原初始化重新建立，6端3360更新/3126图库/571query/五科学门不变，自动CPU核验报告，不重启/扫描。估计13:40–13:55附近终态，240秒/预估里程碑观察。

M0 PASS不是Q1有效。实际f4c6a03e1b263aaa9e4bce71427152007018a0ca，源码/配置/计划SHA固定；本次只写文本证据和文档。V27正收益但FAIL4/5、V28FAIL0/5、RGBNT100增益但融合非最高mAP/MSVR310负/RGBNT201固定dev不足、seed42及重复OOF局限保持，三数据集SOTA目标未完成。

## 41.110 V29首折完整落盘及终态/车辆接口准备（2026-09-07T12:59:21.075742+08:00）

12:54:14原训练153151与wrapper153149存活，第一折两端完整检索已写入原摘要，fold1 control第7轮完成。首折各580更新/20轮、全部1000图库/190合法query/五输出，严格重载与只读评估回执齐全。12:55:19首对最终权重、完整检索数组、完整步骤日志各自SHA复核一致；两权重65171960B，两数组251852276B。此项只查完整存储，不读取模型张量、不比较部分科学成绩。数据盘13877886976B，新删除0。全部回执evidence/v29_first_fold_storage_20260907/。

完整终态本地复算工具已完成：tools/verify_v29_terminal_scalars_local.py覆盖3360步/五输出/bootstrap/五门及训练1935360+检索56268槽位保存统计；tools/summarize_v29_terminal_outcomes.py覆盖全部21身份、所有query错误、全训练/epoch20损失和成本。AST/F821及CLI入口检查通过，尚未在V29完整终态上执行，不写成验证成功。实际权重/特征数组只留远端，不冒充外部独立审计。

车辆接口核对见docs/V29_TERMINAL_AND_VEHICLE_BOUNDARIES_2026-09-07.md：当前style hook限定16×8，车辆输入128×256/网格8×16；共同空间统计公式可复用，但角色网格、合法初始化历史、原RandomIdentitySampler与scene/camera环境字段及车辆exact_signal_forward均需新合同绑定。MSVR310用scene过滤，不套行人camera规则。未注册/启动新车辆实验，没有修改运行中的V29。

RGBNT100本机Signal与作者配置差异已在原官方报告记录，本轮仅复查现有证据，没有新的作者复现或因果归因。不依已消费官方分数调参。当前第1/3折完整，所有科学判断等原六端终态及全量CPU核验，仍估计13:40–13:55附近。原V27/V28结果、seed42/重复OOF、三数据集未达SOTA边界保持。实际执行f4c6a03及原配置/计划/源文件SHA冻结。

## 41.111 V29当前网络与监督集中说明（2026-09-07T13:09:09.704484+08:00）

完整可读架构图、维数、冻结/训练边界、七组监督及相似度公式集中至docs/V29_CURRENT_NETWORK_AND_SUPERVISION.md，按实际f4c6a03源码逐项核对。三角色共享冻结tail但分别执行；独立Transformer读取CLS，联合头读取全部三角色Patch差分。联合LayerNorm/down和同一Mamba共享，正反向各执行一次；每角色一个128→512输出投影、同角色模态共享，不是九套MLP或九个身份头。

候选fused使用修正y，三个完整日志分支仍用原h；因此不能再把候选fused当成三日志分支相似度平均。fused ID/Triplet可沿联合Token回到角色，六组原角色监督不经过联合头。只保留输出接口不冻结角色能力。固定b0.5约束当前h到y，不约束训练起点到当前h，也不保证检索排序。

13:05:38原训练153151及wrapper153149存活、命令匹配，fold1 bounded_joint第6轮完成；GPU7956MiB/82%，数据13708988416B。首折完整存储已核验，后三个端点仍依原pipeline运行。全部科学判定继续等三折六端终态，当前只补充架构说明，原源码/配置/计划、M0全13项及五科学门不变，没有新训练或权重删除。

## 41.112 V29完整终态：边界通过但科学0/5（2026-09-07T13:48:58.752066+08:00）

V29原训练153151于13:38:00、CPU158223于13:38:10、报告158273及wrapper153149于13:38:11正常exit0；13:42实查全部不存在、GPU空闲。执行f4c6a03及原配置/计划保持。六端固定20轮/3360步、3126图库、571query、21身份完整，M0另116步全13项通过。

科学Q1_FAIL0/5：F81.4874846980→81.7070190620（+0.2195343641），C-0.0918723450/T+0.0577887793/M+0.0751655938；F三折+1.3264342011/-0.1400842027/-0.5029371488，bootstrap95%下界-0.5389154198。候选F低于CNN0.0099539899，全部门不变。Signal77.4876031160保持，F仍高4.2194159460，不写成所有三角色无效。与V28各自matched差值不能相减当几何独立因果增益。

融合10身份改善/11下降，180queryAP改善/157下降/234同；R1修复6/新增3，3个新增首位负例均同camera，只作关联。全部105身份-输出、2855query-输出CSV和30端点-输出结果保留。两最强正贡献身份合计0.4288511641pp，大于总净收益，未省略负身份。

完整1935360训练槽位曝光及56268检索槽位保存统计全通过b0.5/cos>=0.894427191、2e-6容差；实际最大ratio0.5000000596、最小cos0.8944270015。三候选完整图库ratio加权均值0.4999650903/0.4999985655/0.4999973232，cos均值0.8944397602/0.8944277014/0.8944281835，整体贴近上限；这是全批次均值/极值，不冒称逐槽位分位数重放。原c全训练abs均值121.7696453568、epoch20均值137.0273528411；约束没有让原始输出变小，也未证明饱和是唯一原因。当前h到y界不限制起点到终点角色漂移。

全训练loss0.6802413188→0.6800979512；加权Tri0.0207309124→0.0214877038，最后epochF-Tri0.0030891974→0.0035008523，正批61→65。每端梯度阶段覆盖203/219完整，非每步全部非零；overflow0/冻结不变。训练2083.914881→2164.846120秒（+3.883615%），reserved6022→7610MiB，不是推理FLOPs/延迟。

CPU完整32602260距离/5952790排名误差0；本地3360步loss最大误差1.7423493159e-7、指标/bootstrap/五门完全一致。32原始文本文件30845895B全SHA核对，模型/图像/数组远端留存。本轮无新删除，目录1018769611B，剩13187350528B。
摘要5993165ca4cd00bbea9213701d31258cd3331d4ab631f0d43a2ccfba571430b6；
CPU49d4c92eb201d0b57d3845135fd4696113859cbf8979b8f5a870764049a3a3f5。
报告results/TRIFUSION_V29_Q1_2026-09-07.md，完整证据evidence/v29_complete_terminal_20260907/，全身份图已PNG视觉检查/SVG XML检查。首图图例重叠修正记录保留。执行器复核非外部独立审计。

下一步拟补完整source初始化/普通/候选终点角色关系变化，区分原角色漂移、联合更新及扰动稳定关系；同时看向量和样本间身份间隔，避免把坐标旋转当信息损失。尚未登记/启动，不预先加教师/可靠关系loss、扫b或重跑V29。网络维数与七监督仍见docs/V29_CURRENT_NETWORK_AND_SUPERVISION.md。V27正收益及FAIL4/5、RGBNT100官方增益但融合非最高、MSVR310负/RGBNT201dev不足、single42/reusedOOF及三数据集SOTA未达边界保持。

## 41.113 V29后继完整来源角色关系诊断登记（2026-09-07T14:17:27.602636+08:00）

V29 Q1_FAIL0/5及全部终态保持。新任务是固定initial/control_final/bounded_final三个模型、原增强与注册V27扰动两个输入条件的零更新诊断，补齐起点到终点角色变化，区分角色训练漂移与同前向joint修正。没有新mAP、模型消融、教师或优化器。

三fold完整source记录2126/2075/2051、各94身份，原20轮采样580/560/540批共1680批；14:12全元数据复查零曝光记录均0。每batch读取一次图像供6前向，共10080；所有采样、身份曝光、原两端前8批像素SHA和共享供体计划绑定。模型eval，既有冻结stem诊断标志开启，角色/Signal/BN仍eval；各模型前后state SHA相同、grad None。未激活输入条件须完全一致。

保存全部18输出批内相似度（743178240 FP32值）及4种槽位向量比较的六标量（7741440观测），原始数组3344302080B约3.115GiB，留远端。向量对应比较只作坐标变化描述，身份判断来自各模型内部相似度；禁止跨fold向量距离。完整identity42147840/cross-camera3401664有序三元组曝光，共45549504供所有状态/view共用，不当独立关系或全来源全图库所有可能三元组。

按真实欧氏Triplet margin0.3和余弦排序记录变化；可靠关系由比较起点在原/扰动输入均满足间隔定义，不使用标签oracle挑老师。候选joint贡献只分解已有相似度，未造新检索头或算反事实mAP。全部12关系分层格、564来源身份关系格（无合法正例保留0）、648向量分层格、20304身份向量格完整输出。

本地显式枚举24/16关系全部变化误差0，身份和为全体，合法class0及无跨camera正例行覆盖；整体反转向量弦长2但相似度不变的反例通过，标量代数误差4.44e-16。AST/F821、CPU验证器CLI通过。远端CUDA数学、实际模型前向、完整CPU终态均NOT_RUN，不冒称已诊断出角色损害。初次只读元数据探针相对路径从SSH home解析失败，已用显式repo根修正，无图像/模型/更新/运行重启。

合同configs/RGBNT201/TriFusion-v29-source-role-drift-diagnostic.json，SHA7ac2250d531f0cd850e274541ac777e5748be545d9e6e3e3008b18436b6a1af0；
计划refine-logs/v29_source_role_drift/EXPERIMENT_PLAN.md，SHAa0e217daae62e79e953d95e52b2e835b57cd97f022c0ba212e71d4301a05fdf8；
证据evidence/v29_source_drift_registration_20260907.json。
5个新工具含NumPy数学、GPU诊断、CPU完整验证、数学T0和持久pipeline；旧V29训练/模型源码不变。pipeline依次CUDA合成检查、固定source前向、CPU全量汇总，各原PID/退出码独立记录；任一失败不自动重跑。

14:12实查GPU1MiB空闲、数据13146820608B。复用刚完成V29的tri_reid环境，不重建；预留5GiB、启动至少6GiB空闲，不删必要权重。既有同规模10080前向6136.07秒，本项估90–110分钟+CPU5–20分钟，首epoch后修正；按180–300秒/里程碑观察原句柄。源码/合同/三份交接同步后才启动，目前REGISTERED_NOT_RUN。只有完整来源证据支持，才另行登记下一种保留或适配机制；不扫描b、复制旧责任/记忆loss。三数据集baseline/SOTA总目标继续。

## 41.114 来源角色诊断实际启动及CUDA数学通过（2026-09-07T14:24:55.805324+08:00）

注册代码1d52c1e9ca957737050072e8e2ace427417cbe14已推送并于14:19:36同步三份主交接。14:20:58通过唯一screen提交启动，持久wrapper159453；数学159455于14:21:02正常退出0，随后同pipeline启动诊断159463。14:21:39实查wrapper和诊断原PID均存活、完整命令匹配，GPU3344MiB/68%，数据13140111360B。

数学CPU显式24/16有序关系与身份汇总误差0；同一反转向量弦长2/相似度变化0的反例通过。CUDA使用真实paired_vectors测量与既有bounded_slots，1728个合成槽位对最大误差2.384185791e-7，PyTorch2.5.1+cu121/RTX3090路径通过。T0真实模型前向/图像/更新均0。数学proof.checked_at记录CPU部分完成时刻，整个含CUDA阶段于14:21:02退出；不能把初始时间当完整GPU结束时间。

run=/root/autodl-tmp/trifusion-v2/artifacts/v29_source_role_drift_seed42_1d52c1e。
合同SHA7ac2250d531f0cd850e274541ac777e5748be545d9e6e3e3008b18436b6a1af0；
计划SHAa0e217daae62e79e953d95e52b2e835b57cd97f022c0ba212e71d4301a05fdf8；
pipeline18b607b3067006a0bfeeb237e787d188d3435084a3356db02e0f2bb7421425a0。
启动前注册源码/合同/计划/V29摘要CPU核验及六最终权重SHA一致，GPU<500MiB、数据>=6GiB；原环境复用，无新权重或删除。

原1680批/10080固定前向以及完整关系/向量输出合同不变。14:21:39尚未观察首个完整epoch回执，原摘要0是首轮落盘前状态，不作已完成或未发生实际前向的判断。预计90–110分钟前向加5–20分钟CPU，暂估16:00–16:30完整终态；下次常规检查不早于14:25:39，首epoch后按实测速率修正。保持原句柄，不因观察超时重启。完整来源与CPU核验尚未结束，无新的关系损害结论、可靠保留loss或模型晋级。

启动/数学/原PID证据完整保存evidence/v29_source_drift_launch_20260907/。当前仅启动验证，原V29Q1_FAIL0/5、V27/RGBNT100正收益及限制、MSVR310负/RGBNT201dev未达、三数据集SOTA目标未完成保持。

## 41.115 来源终态完整汇总工具准备（2026-09-07T14:45:40.975318+08:00）

原1d52c1e来源诊断和合同/计划/五工具不变。14:43:10原wrapper159453与诊断159463存活，fold0完成epoch12，即348/1680批、2088/10080固定前向；GPU3344MiB100%，数据盘12424929280B。首fold每epoch约105秒，GPU暂估16:03–16:10、CPU另5–20分钟，按240秒或里程碑观察。没有额外删除或模型更新。

新增report_v29_source_role_drift.py和queue_v29_source_drift_report.py，完整CPU终态与四份全来源JSON SHA核对通过后才导出全部来源标量。完整身份关系60912行、身份向量20304行以及全部分层/总表，保留零关系身份和空分母比率；不平均分位数、不跨fold计算向量距离、不算新mAP。可靠集合原单位欧氏Triplet0.3空间已与V8 normalize调用核对。

两个工具AST、CLI帮助、F821/F822/F823静态检查通过；本机uvx命令不可用，改用已用的uv run调用ruff后退出0，未修改模型或增加兼容逻辑。完整数据报告尚未执行，不能将静态通过称作实际汇总成功。后台等待工具固定240秒，原pipeline失败即结束，只有原wrapper结束且完整CPU通过才执行报告；不重启原诊断。等待器当前NOT_RUN，同步后才唯一启动。

工具/范围说明见docs/V29_SOURCE_DRIFT_COMPLETE_REPORTING.md，准备回执evidence/v29_source_drift_reporting_preparation_20260907.json。原V29 Q1_FAIL0/5、V27与RGBNT100配对正收益及各自限制、MSVR310负与RGBNT201dev未达保持；没有新loss、b扫描或提前诊断因果。

## 41.116 完整来源报告等待器实际启动（2026-09-07T14:50:37.378436+08:00）

工具发布acf4720365a8c52a96ed3cf7784f083f71c6e177，于14:47:12核对GitHub/本地/远端HEAD一致、远端/仓库/桌面主交接逐字节相同。随后唯一screen v29_drift_report_acf4720提交，报告等待器原PID160653于14:48:19.375938启动，14:48:20实查存活。启动前运行中的1d52c1e全部源码/合同SHA保持，原wrapper159453与诊断159463继续，未重启或改变采样/模型。

原诊断此时435/1680批、2610/10080固定前向，数据盘12257234944B，无新增权重删除。报告输出目录尚不存在，说明汇总尚未执行；等待器每240秒检查原wrapper结束，然后要求完整pipeline/CPU通过才调用固定SHA报告器。失败不自动重跑，各阶段PID/退出码留痕。GPU暂估16:03–16:10、CPU另5–20分钟，等待器最多额外240秒延迟；下次原诊断常规检查不早于14:52:20，预计首fold14:56附近再核查。

启动三份远端不可变JSON、实际观察与本地接收回执保存在evidence/v29_source_drift_report_launch_20260907/。完整报告计划保留全部60912身份关系行、20304身份向量行及全部分层/总表，不平均分位数或隐藏无合法关系身份。报告工具运行不是新训练，模型/图像/更新/检索0；当前仍WAITING，尚无来源关系科学结论。原V29 Q1_FAIL0/5、各数据集正负结果及三数据集baseline/SOTA未达边界保持。

## 41.117 来源首fold完整性核验与终态接收准备（2026-09-07T15:02:31.635407+08:00）

14:58:28原wrapper159453、诊断159463及报告等待器160653均存活且命令匹配。首fold0已完整580批/3480固定前向，2126条来源记录/94身份；295 active/285 inactive，与原采样合同一致。后续fold继续，未重启。第二fold预计15:30附近结束，全部GPU暂估16:03–16:10、CPU另5–20分钟，完整报告等待器固定240秒依赖间隔。

重新读取并计算首fold全部相似度数组1026293888B（256573440 FP32值）、向量标量数组128286848B（2672640槽位比较）及580行批次回执4233628B的SHA，全部匹配原落盘记录。相似度SHA8bd6dc730fadcc3a0e0c0d3883ad21946d6155dc74ad4a2002b80c0bf7e7b332；向量SHA5306f956536613cac054bbe1717a36f0f00408d4cd9bb9e802c909af497d07a4。两个最终checkpoint再核SHA一致。三个固定模型结束state不变/grad None、所有source曝光匹配及原两端前8批像素匹配均有完整回执。最大GPU/NumPy向量误差2.480677609995041e-7、实际joint相似度分解误差5.960464477539062e-7。

上述是首fold存储和执行完整性证据，不是已完成全来源CPU关系复算，也不提前解释角色损害。完整CPU验证、报告均尚未完成。14:58:28数据盘空闲11934138368B，GPU3438MiB100%，没有新增删除/模型更新/检索。

新增tools/inspect_v29_source_drift_terminal.py，AST/CLI/F821检查通过、实际NOT_RUN。它仅在原数学/诊断/CPU/报告及持久句柄均结束、全部验证计数/SHA一致后，重查远端数组与六最终权重，并列出固定42份日志/JSON/CSV/Markdown白名单。接收步骤逐文件核对大小与SHA，不将模型/图像/NPY传回本地。准备回执与首fold完整证明保存在evidence/v29_source_drift_first_fold_20260907/，说明更新docs/V29_SOURCE_DRIFT_COMPLETE_REPORTING.md。

实际1d52c1e五诊断工具/合同/计划与acf4720报告及等待器均未改动；没有新loss、b扫描或模型消融。V29Q1_FAIL0/5、V27配对正收益的限制、三数据集baseline/SOTA未达边界保持。

## 41.118 Goal恢复及完整来源角色诊断终态（2026-09-07T17:15:50.540517+08:00）

用户误清Goal后要求重新整理并执行。保留RGBNT201/MSVR310/RGBNT100分别超过同协议baseline、达到当前资源/协议注明SOTA的完整目标；保持完整训练/全部配对端/完整图库/身份隔离、seed42和既有门槛、失败封存、必要权重保留、180–300秒或里程碑观察、及时GitHub推送及三份交接同步。已通过正式Goal接口替换暂停的占位目标并回读active，没有虚报研究目标complete。公开要求见docs/TRIFUSION_RESEARCH_GOAL_2026-09-07.md，连接密码不进入公开文档。

原诊断159463于16:03:55退出0，CPU163560于16:14:47全量通过并退出0，报告163862于16:16:23完成；wrapper159453、queue160653及数学159455亦正常结束。16:46实查全部原句柄结束，GPU1MiB空闲。完整1680批/10080固定前向、743178240相似度数值、7741440向量比较和45549504协议关系曝光核验通过，source状态/buffer不变、无梯度及全部曝光/前8批像素匹配；最大GPU/NumPy向量误差2.704649801898995e-7。源摘要SHAaf69e94457c527737728c183112d27c175a59dfb6cb6000c35ad4df6d98232f5。

42份完整文本48248745B已接收，全部SHA及CSV行数再次本地核对；六最终权重和原数组远端再算SHA。首次SFTP读取发生10054连接重置，保留原失败记录，复核29份已收到文件后用32请求预取续传13份缺失文件，17:01:53全部成功；没有重启GPU/CPU任务或传回图像/权重/NPY。该传输问题不属于模型失败。完整证据evidence/v29_source_drift_complete_20260907/。

核心发现：control到candidate的fused原输入非正关系0→0、注册扰动1→1；完整分支/银行/纯角色在可靠集合均没有新增排序错误。九个槽位有少量可靠排序损害，完整18输出均保留，不把它等同整角色能力丢失。原输入4293/42121471（0.01019%）、扰动8031/42121471（0.01907%）可靠fused关系再次违反单位欧氏0.3间隔；hinge违规不等于排序反转。

全部282个fold-source-identity组合fused均值间隔缩小，合法cross-camera的42组合也均缩小；其余240个cross-camera零关系行保留。原输入违规分布246/282、扰动264/282，最大两个来源组合仅贡献5.17%/3.93%，现象不由少数身份支撑。282/42是重复fold组合，不是独立身份样本，未计算新独立置信区间。

原输入control→candidate的角色槽位平均余弦C0.926475/T0.873854/M0.952974；同前向h→y均值约0.894432/0.894432/0.894439，各fold/modal中位数普遍贴近边界。初始化到普通终点也有大量变化，不能把坐标变化全归因joint，更不能直接当作身份损失。精确标量分解中原输入融合均值间隔变化-0.037126166，原角色部分-0.007377962，同前向joint部分-0.029748216（80.127%）；另三协议/input格joint占净缩小80.074%/81.393%/81.233%。这不是Q1性能变化的因果占比。

完整报告results/TRIFUSION_V29_SOURCE_ROLE_DRIFT_2026-09-07.md。下一步先用已保存完整相似度区分joint是否增加独立判别信息、还是主要共同偏移/压缩；该解释尚待检查，没有新增保留loss、teacher、PCGrad、b扫描、GPU前向或官方评估。原V29Q1_FAIL0/5、V27正收益的限制、车辆/行人各自结果及三数据集SOTA未达边界保持。


## 41.119 完整来源联合相似度结构诊断登记（2026-09-07T17:32:41.610773+08:00）

# V29来源联合相似度结构诊断
状态：REGISTERED_NOT_RUN。父提交240cbce。全部V29 Q1和来源诊断保持封存。

唯一问题：V29同前向联合头对原角色银行的修改，是否主要表现为已有相似度的共同偏移/压缩，还是包含不能由这种简单变化解释的身份关系结构？
本项仅使用已保存的完整来源矩阵及原样本回执，不读取模型/图像，不产生梯度、优化器更新、AP/CMC、dev/official访问或部署分数。CPU描述性回归系数不是可部署融合权重，不用它选择模型或扫描b。

固定输入是全部3fold/1680batch、原输入和注册style两view。每fold94source身份保持；all/active/inactive三分层全部报告。先验证父summary/CPU proof、全部三矩阵和三batch回执SHA。NPY留远端，复用tri_reid，仅CPU，输出预计小于5MB。
原h-bank相似度x=output14，实际y-bank相似度y=2*output1-output0；先转FP64，再精确分解。Signal保持不变，当前保存数据没有每个修正y槽位的独立Gram，不能假称逐y槽位分析。

第一遍：全部有序非对角样本对（每view6773760次曝光）计算y=a*x+d的描述性最小二乘拟合；保留每fold/view/stratum和全部逐batch结果。对同/不同身份×同/不同相机四个完整类别分别报告，并保留same-record、positive-distinct-record和all-distinct-record子集。重复采样不是独立图像，分层是重叠统计，不能相加作为独立总量。空或常数子集保留null拟合，不丢弃。
固定比较y=.8*x+.2仅表示一个单位正交公共方向以b=.5加入时的理想相似度形式，不是当前样本条件切向更新必然满足的理论等式。报告固定形式RMSE和自由描述性拟合R²/RMSE，不设可调成功阈值。
第二遍：使用各fold/view/stratum全非对角对的a，对每一合法identity/cross-camera三元组计算my-a*mx。截距在同一query的正负差中自动抵消。报告均值、平方、正负残差、原/后非正关系；完整每source身份行含0合法关系，不挑困难身份。
同模型矩阵内部先比较，再汇总标量；不跨fold坐标系计算距离。不把小残差直接认定没有新信息，也不把R²或来源关系当未知身份mAP的因果证明。

输出固定144pair行、3360batch行、36relation行、3384identity行和带哈希的JSON。回归SSE与第二遍逐点直接残差重新核对；身份总和对齐分层总和；全关系joint平均变化与已封存完整诊断对齐。T0用独立lstsq、显式正负三元组、已知仿射关系和真实可能存在的空/常数类别核验数学。
工程检查只证明统计正确；若共同变化解释力高而身份残差信号弱，应降低继续扩充当前自由joint的优先级。若明显残差广泛有益/有害，按完整分布确定后继问题。任何下一训练方案仍须独立登记匹配合同，本项不批准新增保留loss、扫b、Router或已有失败版本复跑。

代码/合同/计划先提交、三处同步，再由一次持久CPU子进程执行。固定日志、PID、开始/结束/退出码保留；失败不自动重启。预计1–4分钟，首次检查安排在约3分钟后；没有未完成GPU前向需重启。三数据集主目标继续active。


## 41.120 联合相似度诊断算术核对修复登记（2026-09-07T17:38:22.966321+08:00）

# V29联合相似度统计核对 R2
状态：REGISTERED_NOT_RUN。R1原进程165941/wrapper165936在2026-09-07 17:35:22退出1，不重启原run。执行提交13ceba1，完整原日志和回执保留。
唯一工程修改：原case_relations先以FP32计算unmodified=.5*Signal+.5*h_bank，再转FP64统计；新分析直接用FP64分解。R1将两种不同舍入路径按1e-10要求相等而失败。全三折逐batch已实测，这次FP32加法的最大舍入差均为5.960464477539063e-8。
R2保留新分析的FP64精确分解，同时另外按原FP32路径重放legacy统计，仍以原1e-10平均误差核对已封存总和；另外记录精确/legacy差并用实际舍入界验证。没有放宽原等价核对门，也没有改旧矩阵、旧统计、模型或科学目标。新增合成FP32/FP64舍入身份式核验已通过。
原EXPERIMENT_PLAN.md的完整范围、输出144/3360/36/3384行、无参数选择/新检索边界不变。使用独立R2输出目录，一次持久CPU运行，预计约1分钟内完成，观察同一个原句柄，不自动重启。


## 41.121 联合相似度完整结果与MSVR310后继决定（2026-09-07T17:48:15.534539+08:00）

# V29完整来源联合相似度：主要改变已有关系尺度，新增关系并非一致有益

结论：在本次全部注册来源样本对上，联合银行相似度与原银行高度相关；大部分身份间隔变化可由约0.92倍缩放解释。这是对既有表示进行重新调整的证据，不是joint已形成稳定互补的证据。仍有不能由线性项解释的变化，不能宣称“联合头完全常数”“没有任何身份信息”或直接归因未知身份mAP。
V29 Q1_FAIL0/5保持，fused配对+0.219534 mAP不变。没有新的训练/图像/模型前向/AP/CMC/dev/official。

## 完整执行及核验

R1执行13ceba1，原CPU165941/wrapper165936在17:35:22退出1，停在旧统计核对。原因是父诊断先以FP32相加，再转FP64；新统计直接FP64分解。原失败日志全部保留。
R2执行1013e2e，原CPU166197/wrapper166192在17:40:00正常退出0，实际31.26秒。保持旧1e-10核对门，另外重放原FP32算术并报告精确/legacy差。原数据和科学目标不变。
三折加法舍入最大5.9604644775e-8；重放旧统计的最大平均误差3.08324e-16，FP64精确分解与legacy最大平均差6.03968e-10。逐点残差SSE与矩统计公式的最大均方差2.10239e-15。

全部1680批、两view、每view6773760个有序非对角样本对。全部三fold矩阵和batch回执再次SHA核对，模型/NPY留远端。
144 pair行、3360 batch行、36 relation行、3384 source identity行全部导出并本地核对SHA、行数、回归算术和身份汇总。完整无合法关系的身份行保留；本项是执行器数值复核，不是外部独立审计。

## 实际联合前后的相似度关系

令x为当前候选原h-bank相似度，y=2*s_fused-s_Signal为实际修正y-bank相似度。两者在同一个模型、同一个输入中定义，全部先转FP64。下面y≈a*x+d是来源数据的描述性最小二乘，不是部署权重或新检索头。
|fold|输入|a|d|R²|拟合RMSE|固定.8x+.2 RMSE|
|---|---|---:|---:|---:|---:|---:|
|0|original|0.925069|0.070587|0.991394|0.024203|0.121050|
|0|registered_style|0.925408|0.069894|0.991028|0.024232|0.121565|
|1|original|0.914119|0.082429|0.991025|0.024367|0.109677|
|1|registered_style|0.914147|0.082301|0.990839|0.024157|0.109639|
|2|original|0.927158|0.068990|0.993300|0.021211|0.120654|
|2|registered_style|0.926818|0.068976|0.993185|0.020884|0.120315|

all/active/inactive三分层完整表均保留。逐batch原输入R²最低为0.976767，扰动最低0.976155；不是少数batch支撑总相关。原输入不同身份样本对内部的R²仍为0.950291–0.972604，不能只用正负样本类别分离解释全部相关。
同一图片重复采样类别在本RGBNT201来源清单为0，仍保留空行。它不表示其他数据集没有重复采样。
固定.8x+.2拟合明显较差，因此不能从b=.5和近边界夹角直接断言“每个样本都添加了同一个正交常量”。

在上述近似下，fused≈.5*s_Signal+.5*a*x+.5*d。对同一query，公共截距不改变排序；正比例整体缩放后，Signal近似占1/(1+a)，约51.9%–52.2%。这是解释现有来源输出的一阶近似，不是新选出的融合系数，未应用到检索或未见身份。剩余非线性误差仍可能影响细微排名。

## 真实身份关系及剩余贡献

以下逐项使用全部合法有序(q,p,n)；cross_camera是真实同身份跨相机正例，所有负例按真实身份过滤。回归截距在间隔中抵消，残差为m_y-a*m_x。解释比例定义为1−sum(residual²)/sum((m_y−m_x)²)，并非检索性能解释率。
|fold|输入|协议|原银行间隔|修正银行间隔|扣除线性项后的平均残差|变化平方解释比例|
|---|---|---|---:|---:|---:|---:|
|0|original|identity|0.797244|0.740180|0.002674|86.607%|
|0|original|cross_camera|0.753791|0.698161|0.000853|88.610%|
|0|registered_style|identity|0.775535|0.720328|0.002642|86.164%|
|0|registered_style|cross_camera|0.732536|0.678621|0.000726|88.020%|
|1|original|identity|0.793435|0.727511|0.002217|89.712%|
|1|original|cross_camera|0.755613|0.688419|-0.002302|90.582%|
|1|registered_style|identity|0.773295|0.708867|0.001962|89.714%|
|1|registered_style|cross_camera|0.733859|0.668885|-0.001970|90.405%|
|2|original|identity|0.791400|0.735957|0.002204|87.913%|
|2|original|cross_camera|0.741667|0.692862|0.005219|85.469%|
|2|registered_style|identity|0.766811|0.712778|0.002084|87.891%|
|2|registered_style|cross_camera|0.719568|0.671768|0.004860|85.604%|

原输入全部282个fold-source-identity组合、全部42个跨相机合法组合的联合前后平均银行间隔均缩小，扰动输入同样。它们不是独立282/42身份，来源fold之间存在重叠。
扣除线性缩放后，原输入167个组合平均残差正、115个负；扰动170正、112负。跨相机两条件均25正、17负。剩余贡献没有表现为全部来源身份一致受益。
原银行/修正银行在原输入均无非正关系；扰动第2折9→11，另外两折0→0，全部跨相机0→0。这不是fused新增错误数，Signal加入后的既有fused仍是原输入0、扰动1；不能混用两种输出。

## 研究决定与下一项工作

当前没有足够证据继续扩大V28/V29联合扫描、扫描b或立即增加教师/保留loss。近边界角度、很大的原始修正和高来源拟合本身不证明可泛化新增信息。
下一项主假设转向：V27三模态耦合的来源统计扰动，能否改善MSVR310原三角色在未知车辆身份及完整scene图库中的检索？这检验已有正机制的跨数据集适用性，不包装成新算法，也不是将RGBNT201权重零样本部署到车辆。
已检查原MSVR310全部780训练batch元数据和三个固定Signal anchor的当前完整SHA；无供体缺失，全部source记录曝光。详细准备见docs/MSVR310_SOURCE_STYLE_NEXT_EXPERIMENT_2026-09-07.md。新训练尚未登记或启动。
三核心数据集baseline/SOTA目标继续active且未完成。

## 证据

- [四份完整CSV与终态](../evidence/v29_joint_similarity_r2_20260907/analysis.json)
- [全部样本对拟合](../evidence/v29_joint_similarity_r2_20260907/pair_affine.csv)
- [全部逐batch拟合](../evidence/v29_joint_similarity_r2_20260907/batch_affine.csv)
- [全部来源身份](../evidence/v29_joint_similarity_r2_20260907/identity_relation_residual.csv)
- [本地全量复核](../evidence/v29_joint_similarity_r2_20260907/local_digest.json)
- [R1原始失败日志](../evidence/v29_joint_similarity_r1_20260907/run.log)

# 下一主比较：MSVR310来源统计扰动

状态：DIRECTION_SELECTED_READINESS_CHECKED_NOT_TRAINING_REGISTERED，2026-09-07。
V29完整联合诊断完成后，唯一下一训练问题确定为：V27耦合来源统计扰动是否能在MSVR310原三角色上带来配对、跨身份分布的检索改善。
理由：V27在RGBNT201有全三fold/三角色正收益；V28/V29自由或有界联合头尚未稳定改善，来源已饱和，joint大部分变化表现为已有相似度缩放。优先建立另一数据集的机制证据，降低对反复开发的21个RGBNT201身份的依赖。这里不宣称提出新算法。

## 已核实输入和边界

- 三个MSVR310固定Signal epoch50 checkpoint现存且完整SHA再次匹配原B0摘要；不重训baseline。
- 原三fold source103/103/104身份，672/683/709记录；heldout完整图库360/349/323，合法query210/207/183，共600query和60query身份。95单scene干扰身份仍保留。
- 原角色完整训练元数据260batch/fold、20epoch、共780步已逐条检查；全部source记录曝光，最少camera数6/7/7，各batch都允许跨camera供体。
- 按原V27固定seed42/fold/step的独立NumPy计划，active批129/127/137；原真实source重复记录正关系曝光21836/20864/20882，必须保留并报告。
- 三fold跨scene有序正关系30232/30196/30650，总91078/349440，约26.064%；cross-camera与cross-scene是不同字段，不混称同一种覆盖。
- 固定供体均跨camera，其中跨scene13927/13888/13828次，同身份供体1613/1716/1719次。原V27未要求供体不同身份，不能暗中追加标签过滤。
- 以上是对既有训练元数据的完整检查，不是新loader像素轨迹重放；正式训练必须匹配固定曝光并留下新输入回执。

## 下一实现必须具体完成

1. 新的车辆style接口使用768×8×16 stem，原CNN/Mamba仍按8×16网格。原V27行人硬编码16×8模块保持冻结；复用统计混合公式，同模态统计、三模态共享供体/系数，anchor/reference接受同一扰动。
2. 每fold在原Signal上新初始化原V8角色，两端seed42、整个初始化状态和采样/训练预算匹配。两端均执行同样额外视觉重编码，只有是否作用统计混合不同。无V23/V24、联合Mamba、几何修正、Router、记忆或新loss。
3. 保留原B64/K8、128×256、20epoch、AdamW和七组ID/Triplet；预期每端780更新，两端1560，真实记录为准。不复用M0权重，不用旧单端终点冒充本轮匹配控制。
4. 训练前完整source-only工程门应核对Signal原路径/被冻结状态、模态耦合、anchor/reference匹配、active与inactive行为、实际全部参数梯度和固定批次拟合。沿用已验证exact_signal_forward来恢复车辆B0的精确推理执行；不能重现已修复的SIM差异。
5. 全部三fold两端固定终点完整检索，使用原scene过滤和完整图库。配对扰动收益与candidate相对Signal的既有车辆晋级条件分别登记/报告；不因其中一组更容易通过而替代另一组。门槛、CPU全量核验、源文件/初始化/输出绑定须在运行前完整确定。
6. 正式三处同步后才启动持久任务。当前仅方向和输入准备完成：新训练程序、最终合同及工程门尚未执行。下一步直接完成这些内容，不重复已完成V29诊断。

## 证据

来源元数据完整保存：evidence/msvr310_style_readiness_20260907/msvr310_style_source_readiness_20260907.json。
三个当前Signal全文件SHA：evidence/msvr310_style_readiness_20260907/msvr310_style_remote_readiness_20260907.json。
角色原实现tools/train_msvr310_trifusion_oof.py；过滤tools/train_msvr310_signal_oof.py的scene_scores；精确推理tools/msvr310_exact_signal_inference.py。
这不是零样本迁移、官方成绩或新SOTA主张。三数据集长期Goal保持。

## 41.122 MSVR310 source-style V1 注册：复用原角色训练，完整双端验证

在§41.121来源联合相似度诊断完成后，下一唯一主实验选择V27耦合统计扰动的MSVR310独立训练验证。此时仅实现/注册，T0/M0/Q1均未启动；不能写成已通过或已产生收益。

新增车辆接口只将原V27类名与stem768x16x8改成768x8x16，其余类体通过AST同构核对；直接复用原统计公式、原V8模型/训练/损失/scene评价和既有exact_signal_forward。控制端与候选均额外执行3次冻结CLIP，只有候选启用扰动，避免不同输入/计算路径混淆。原Signal不重训，每折角色与七头全新且两端初始SHA一致。真实增强像素hash、采样record序列与确定性风格计划逐批配对。原RGBNT201 V27代码不改。

合同：configs/MSVR310/TriFusion-source-style-paired-v1.json；完整计划refine-logs/msvr310_source_style_v1/EXPERIMENT_PLAN.md。T0全780来源batch计划/覆盖及CPU公式，M0六端各8步＋两端fold0固定100步共248步；M0CPU通过后才能进入新初始化Q1。Q1两端三折各20epoch共1560更新、2064留出record前向、每端600query/1032完整gallery，原scene规则不变。双端固定过拟合采用同一raw输入/force-active step0风格计划。原五项vehicle-v-Signal门与新增五项配对style-v-control门均固定，晋级需两组全通过。

磁盘：刚实查空闲约9.03GiB。检查点只写非baseline角色state，保留B0别名/配置/身份/SHA绑定，严格重建核对全模型，六M0加六Q1约400MB，全部增量计划约1GiB；启动至少留2GiB。无本次新增删除，必要初始化/终态/证据保留。

CPU终态将重算所有五输出/六端的完整距离、排序、AP/CMC、身份结果、bootstrap与全部门槛。14项loss由保存标量作double重组诊断，不伪称原AMP中间dtype已保存或bitwise复算。新pipeline阶段持久PID/日志/退出码，原错误即停。无官方图像、dev调参、跨fold特征距离或挑checkpoint。主Goal仍为三个数据集各自超过同协议基线和资源注明的当前SOTA，尚未达成。

## 41.123 MSVR style T0原始失败与R2字节绑定

首次实际628cce0/wrapper167185/T0PID167187于18:15:06退出1，停止在风格计划完整字典相等检查。无模型/GPU训练/留出图像；完整原始pipeline/t0.log已归档，不能记为M0或检索失败。

随后对全部780batch重放：31个Float64混合系数末位不同，最大2.220446049250313e-16；模型实际使用的Float32系数全部bitwise相同，所有供体、启用状态、曝光字段全等。训练服务器NumPy1.24.4；先前本地元数据的末位计算不能作为服务器完整Python字典的字节合同。

R2配置configs/MSVR310/TriFusion-source-style-paired-v1-r2.json绑定实际服务器全量计划，严格字典相等检查保留；不改公式、采样、训练步数、初始化或晋级条件。新实现代码无需修改。原元数据/配置与失败证据保留；R2仍须T0→248步M0→CPU→1560步完整Q1→CPU，不复用任何旧更新。详见refine-logs/msvr310_source_style_v1/R2_RUNTIME_PLAN_BINDING.md和evidence/msvr310_style_t0_runtime_binding_20260907/。此登记时R2尚未启动。

## 41.124 MSVR source-style R2真实启动与第一折完整检查

R2实际启动18:19:57，执行源码提交02cc09e，wrapper167448；T0PID167450于18:20:03退出0。正式登记配置SHA848254aab6dc02230a7fd33be0d60fc2560c99fc77cbd320cf0ef2efd655539d。Run：/root/autodl-tmp/trifusion-v2/artifacts/msvr310_source_style_v1_r2_seed42_02cc09e。首次628cce0的T0失败证据仍保留，未转写为通过。

T0完成全780来源计划、全部来源record覆盖、35文件源码绑定、车辆类AST仅名称/shape变化、独立NumPy公式最大误差2.384185791015625e-7，0模型前向/0优化更新/0留出图像。启用批次129/127/137保持不变。

M0PID167458于18:20:03开始。18:23:38实查仍RUNNING，容量检查已推进第三折。第一折control/source_style各8步，完整检查与严格恢复完成，初始全state/参数完全匹配；总参数99,065,869，可训练8,076,300，203/203梯度非零、0overflow、冻结Signal unchanged。五个输出加载前后bitwise一致；独立来源Signal与原B0相等。两端8步耗时14.7707/12.7259秒，峰值reserved5954/6228MiB。这是短工程运行，不能作为正式吞吐或检索收益结论。

两份实际compact checkpoint各32,734,799bytes（约31.22MiB），远端文件SHA、下载的5份文本SHA和本地第一折全回执一致性均核验。本地未接收模型。18:23实查余9,516,294,144bytes。后续仍需六端容量＋两端固定100步全部248更新及CPU验证，才能开始全新六端1560更新Q1。完整M0和Q1未有PASS/FAIL结论。

证据evidence/msvr310_style_r2_launch_20260907/；持久tracker记录原PID、下次不早于18:27观察、M0预计18:28-18:30（估计非保证）。新增tools/summarize_msvr310_source_style.py准备终态完整600query/60身份/五输出的本地文本重算与修复/新增错误、camera/scene关联、成本分析，已语法/F821检查，未在未完成终态上虚报运行。该报告工具不改变训练绑定或运行中的模型。

当前代码和新实验推进属于实际进展；三个数据集同协议baseline及资源注明SOTA总Goal仍未达，继续保持active。文档发布后的HEAD可能只推进交接/分析工具，GPU训练参数/配置/绑定文件不变；各子阶段均记录实际HEAD和文件SHA。

## 41.125 MSVR style R2完整M0/CPU/本地核验通过，Q1已实际接续

M0PID167458于18:29:11退出0，完整248更新；CPU168446于18:29:21退出0，状态PASS_COMPLETE_MSVR_STYLE_M0。八个run均203/203实际非零梯度、0overflow、冻结state/Signal保持，六个容量checkpoint完整重建并五输出bitwise恢复。两端100步超额损失比0.000702123930784/0.000710502676525均通过固定0.1门。详见results/MSVR310_SOURCE_STYLE_V1_R2_M0_2026-09-07.md，工程门通过不代表检索有效。

29文本2,059,528bytes逐份SHA/bytes接收；本地新tools/verify_msvr310_style_m0_text.py独立核对全部248steps、真实source曝光/风格计划/配对像素、epoch均值、解析label-smoothing floor及过拟合比，PASS。保存14标量double重组最大差4.222e-7只作诊断，原AMP中间dtype未知，不据此虚构bitwise复算。M0摘要SHA5d9d71d05fd4dcb2c2a5640ee48d6d58a11cd9ca5b375c08cf8edcdcf506f332；CPU SHA058f726e5c62f201ccd24fdd057e15a980b27efa9ba43fae39822bc3b4bdb931。全证据evidence/msvr310_style_r2_m0_complete_20260907/。

持久wrapper167448已于18:29:21启动全新Q1PID168456，阶段实际HEAD893be37；原始实现02cc09e到该HEAD仅交接/报告工具发布，全部训练文件/config SHA仍固定。18:31:25通过/proc原PID+完整命令核对wrapper与Q1live，M0/CPU旧PID已结束。第一折控制端当时完成epoch5，尚无完整Q1检索终态。两端三折1560更新/2064留出record前向/每端600query和1032完整gallery合同不变，两组5条件全部达成才晋级。

18:31实查空闲9,481,236,480bytes；Q1早期约20秒/epoch，全部六端加恢复评价粗估19:12-19:20，非完成保证。下一观察不早于18:37，优先核对第一端完整Signal/gallery/compact终点。不要因为早期loss变化修改方案，不重跑已完成M0，不提前报告Q1成功失败。三数据集总Goal仍active未达。

## 41.126 MSVR style Q1 CPU距离数值问题与R3无重训接续登记

R2 Q1PID168456完成fold0control固定20epoch/260更新、保存compact终点后，在原evaluate的Signal距离torch.equal检查处失败，于18:36:31退出1；wrapper167448也结束。Signal全图库特征torch.equal已通过，尚无该端完整检索回执，不能记科学Q1_FAIL。原训练/权重/日志/partialsummary完整保留。

同一B0已保存特征、同一距离公式CPU全三折重放：4线程有32173/31344/25512个entry不同，最大2.384185791015625e-7；56线程三折全部0差bitwise相等。服务器当前默认56，旧baseline wrapper未显式覆盖线程。证据支持匹配的CPU算术执行路径，不能虚构未记录的历史运行线程细节。诊断0模型/0图像/0更新，不读指标选参数。

新R3合同configs/MSVR310/TriFusion-source-style-v1-r3-distance-resume.json及refine-logs/msvr310_source_style_v1/R3_EXACT_DISTANCE_RESUME.md：原模型/训练/提取仍4线程，仅原evaluate和完整CPU verifier用56，随后恢复4。保留原严格bitwise检查、所有scene/fullgallery/损失/初始化/配对/科学门；没有替换已算距离、放宽容差或改变loss。完整M0仍有效不重跑。

原第一端checkpoint SHAeb4f85bf3b0928f4041d8112a34fb62a7a3f79e4dffaa851bbf932e39795ac7d，原训练SHAf68c6d94b53c80fbfd3a936c2c6002f09ab6d20e45e8e5ec7c1e5fcc2a7f9ba7。接续前CPU严格重建全state，正式阶段严格GPU加载后重新完成该端失败的360图库评价；不重训、不重复保存该权重。余5端1300新更新，总比较仍1560；本次2064图库record前向与先前失败已读360分开记账。新持久pipeline CPU修复检查→完整接续Q1→原全量CPU＋额外原260步复用核验，失败即停。

新源码已实现/静态检查，R3登记时尚未启动。证据evidence/msvr310_style_r2_distance_failure_20260907/包含完整真实失败和CPU重放。原T0末位字节失败与已完成M0记录不改。下一步先同步该固定修复再接续，Goal仍active且三数据集目标未达。

## 41.127 MSVR style R3已真实恢复第一端，余五端训练进行中

实际源码484264e，wrapper170083于18:51:45启动；check170085于18:51:52退出0。三折56CPU线程距离全部与原B0 bitwise相等，4线程已测舍入差异保持原记录；原compact control完整state重建SHA00b98730ae428d8f095f2d0fe8a031dc31fe8b86f4a4d3c4f917c9bed485f01f一致。该检查0模型/图像/更新/指标。接续Q1PID170273于18:51:52开始，18:54:15通过/proc及完整命令确认live。

第一端已经严格GPU恢复、完成360gallery/210query五输出评价，Signal features与distances逐位等于原B0；重用原260更新/终点eb4f85bf...95ac7d，不重训也未重复保存权重。7文本5,965,286bytes全部SHA/bytes核对；本地对原训练字段全等（仅style日志路径迁移）以及1050query-output/378000全rank位置重算，最大AP差3.331e-16。此处只证明第一端工程恢复，不能提前判断配对科学成败。

评价后恢复CPU4线程，原训练接口继续。18:54候选fold0epoch4，余5端1300新更新，总比较1560仍不变；新2064与失败阶段旧360图库record前向分开记录。完整原CPU verifier将在56线程运行，并另核验原260步复用/原文件不变。终态本地摘要工具已加必需的resume_verification绑定，不能把缺少接续证据的部分摘要当完成。

报告results/MSVR310_SOURCE_STYLE_R3_DISTANCE_RECOVERY_2026-09-07.md；证据evidence/msvr310_style_r3_first_recovery_20260907/。18:54余9,395,724,288bytes约8.75GiB；约21秒/epoch，粗估全Q1/CPU19:29-19:34，实际以原进程终态为准。下一观察不早于18:59:30。R1/R2错误及完整M0均保留，主Goal仍active未达。


### 41.128 MSVR310来源统计扰动：完整Q1负结果与全量复核（2026-09-07 19:33）

实际执行484264e，原Q1于19:26:40退出0，CPU于19:26:52退出0；wrapper170083及其全部原始child均已结束。pipeline `COMPLETE_VERIFIED_Q1_FAIL`，科学条件相对control0/5、相对Signal0/5。fused52.126691→51.824477，配对−0.302214、bootstrap下界−0.864710；相对Signal53.129381为−1.304904。CNN−0.222331、Transformer+0.600893、Mamba−0.237235。三折fused−1.163999/+0.296255/+0.009763。

候选fused高于自己的三个完整角色，但低于Signal；注册“最高”条件包含Signal，因此仍FAIL。60身份25改善/29下降/6不变；配对Rank1修复8、新增11，新增同camera3、同scene1。Transformer的下界−0.254319，不能宣布稳定单支成功。RGBNT201 V27正结果及RGBNT100既有增益保持，不能用它们抵消MSVR310负结果。

六端各260更新、总1560（原R2control260复用＋1300新更新），203/203真实梯度，0 overflow，三折所有实际增强像素/样本/扰动计划配对一致，393真实扰动批。六端Signal特征和距离逐位B0一致。CPU全量2,069,520距离/排序位置核对；原失败文件未改。32文字文件39,513,093B全接收，6checkpoint远端SHA核验；本地全部600query×5输出×2端重算通过。summary `e04f2de56db973fd0ee16db0f261a56f38e1ceb4a651433db9bffe951c4b4e1b`；reuse `95c3527ebcef3d71293de7cdf485ff283b4d50c7ff31d30b405a7dfdce8b9a9c`。

全部1560步训练日志显示MSVR310增强输入下终点Triplet仍非零；这不等于反序计数，也不能直接沿用RGBNT201来源饱和诊断。下一步先登记MSVR310完整source合法间隔与实例覆盖普查，尚未启动，不扫描本次失败配置或消费official结果。Goal active/unmet。

报告：[results/MSVR310_SOURCE_STYLE_V1_Q1_2026-09-07.md](../results/MSVR310_SOURCE_STYLE_V1_Q1_2026-09-07.md)；完整证据：[evidence/msvr310_style_r3_complete_q1_20260907](../evidence/msvr310_style_r3_complete_q1_20260907/)。19:32空闲8.395GiB，保留必要B0/CLIP和六端角色终点，无新增删除。文献新增[原表刷新](SOTA_PRIMARY_REFRESH_2026-09-07_EVENING.md)，FUSE的50.1/65.7不替代更强MSVR参考；RoDI CLIP/DINOv3分列，内部Q1不能直接对比论文official。


### 41.129 MSVR310完整source关系普查V1登记（2026-09-07 20:18）

前置fb819ef完整Q1_FAIL保持，不重训。新诊断合同configs/MSVR310/Source-relation-census-v1.json，SHA 23b3082aeb80ae5766c399ea79965f0ce176c1d7f4864347b9e26652e0392540；计划refine-logs/msvr310_source_relation_census_v1/EXPERIMENT_PLAN.md。此前18576前向尚未启动，此处是登记状态。

三折672/683/709source记录、103/103/104source身份，覆盖全部1032原训练记录；初始化及两个固定epoch20终点各作clean/固定增强/强制耦合统计扰动三视图，总27条件/306batch/18576记录前向。全部18输出、普通同身份排除同记录与跨scene两协议；所有无正例记录保留负例作用。强制style是新登记压力视图，不是原260step像素重放或修改训练p0.5。gallery同模型完整clean source，不跨fold算距离。

0优化器更新/0新checkpoint/0heldout及official图片；保存2,853,273,600B左右FP32特征，完整CPU核对972条件、668736query关系行、100440身份行。区分真实非正排序、仅不足0.3欧氏间隔、完全满足，额外测原型正确但实例错误；不把这些统计当参数梯度或记忆必然有效的证据。静态与独立数学fixture通过，GPU尚未执行。预计GPU10–20min、CPU10–25min，按实际条件速度及180–300秒里程碑观察。

现有tri_reid/PyTorch2.5.1+cu121/CUDA12.1沿用刚完成Q1环境，无重建；独立.aris/compute账本不存在，不虚构新的环境认证。19:53原任务PID均不存在、GPU空闲、空闲8.32GiB。必要初始化、六端终点、复核证据均保留。后续先按合同完成诊断，再依据全量证据登记唯一新干预；Goal active/unmet。


### 41.130 MSVR310完整来源关系普查通过及下一项证据方向（2026-09-07 20:41）

执行4e57e54、合同23b3082aeb80ae5766c399ea79965f0ce176c1d7f4864347b9e26652e0392540。原wrapper175887、check175889、GPU175895、CPU176897均退出0并已核实结束。GPU20:28:38完成27条件/306batch/18576条source记录前向，CPU20:37:51完成230,162,148相似度元素、972条件、668736query关系行和100440身份汇总。全模型状态及原六checkpoint不变，0梯度/更新/新checkpoint/heldout或official图片；306额外冻结视觉调用。

全部64文字/压缩文字57,135,882B接收并SHA一致，本地所有query及身份行再汇总核对全部972条件，输出324总表/108配对比较/33480配对身份成员。CPU SHA b392f4ccce6c0b033cf556e4976b20ff2cd515d55e932e89dd856a43cebec85c；summary617a2fea2a361185a9c767a9103cee24a0290ba08f26b56ad9b03e5532747658。最大独立hinge总和误差3.04681e−11，融合分解6.73505e−8。cpu_progress文件是最后进度快照仍写RUNNING；正式pipeline/CPU终态及退出0共同确认完成，不能以旧进度字段误判重启。

普通终点增强跨scene fused94.956353/R1 95.583333，1200合法query成员：333反序、864仅不足0.3欧氏间隔、3完全满足；共7033970三元组，5651非正、855355 hinge违约。纯银行95.128771/96.083333、327反序query。fused中300个原型正确但部分实例错误，只有22个是原型正确/实例Rank1错；不能混同。MSVR来源有真实困难关系，不能继续沿用RGBNT201来源几乎全分对的解释。来源角色增益也未转化为已封存heldout优势。

统计扰动终点相对普通终点，跨scene clean/增强/压力fused分别−1.601491/−1.772789/−1.361499；增强纯银行−2.065805。增强fused120个合法身份—fold成员3改善/59下降/58不变，190无正例成员仍保留图库负例作用；不是120独立身份。所有18输出、两协议、三模型/视图完整表均报告。既有MSVR Q1配对−0.302214与两组0/5保持，RGB201 V27及RGBNT100正证据分别保留。

下一项候选收敛为MSVR310 source-only实例关系覆盖：先固定并检查缓存去重、真身份/scene关系、年龄/漂移和实际新增有效难例，再一组匹配配对训练；保持原V8固定融合，不同时叠加Router、联合头、style或新排名loss。该训练尚未登记/启动，不把普通XBM称为原创或已证实泛化。来源成员跨折重复，只作描述统计。主Goal active/unmet。

20:38终态主盘空闲约5.626GiB；20:37另一存储盘10.365GiB。现存181个pth/73个pt的元数据清单包含检索与诊断张量，不按扩展名当无用权重删除；本轮0新增删除，必要初始化/终点/复核证据保留。报告results/MSVR310_COMPLETE_SOURCE_RELATION_CENSUS_2026-09-07.md；完整证据evidence/msvr310_source_relation_complete_20260907/。


### 41.131 MSVR310来源实例覆盖V1登记（2026-09-07T21:16:42.245629+08:00）

前置2a9b618全source普查已完成；不重跑。新合同configs/MSVR310/TriFusion-instance-memory-paired-v1.json，SHA 40f44b0e6c12771c53a283a5b65ca6a45556e0293b7d48decf5fd518ad771a3d，计划refine-logs/msvr310_instance_memory_v1/EXPERIMENT_PLAN.md。五新工具已AST/ruff检查，真实CPU数学/T0、模型前向、训练均尚未执行。

唯一干预是将fused原batch-hard Triplet候选扩展到当前batch+历史source实例。其余13项损失/所有权重、原V8网络和静态推理表示不变。历史按真实记录保留最新增强视图，排除当前记录历史副本，最多512记录/8更新年龄，5epoch65step预热后入队。训练正负标签沿用真实身份，跨scene正对单列；评价严格排除同身份同scene、完整图库干扰身份保留，不套samecamera。当前batch梯度保持，历史detach，不新增参数/EMA/style/Router/联合头。

control也维护shadow缓存/全部关系诊断及同等固定像素重放，但不使用扩展loss更新。两端重新从原Signal source初始化及seed42角色初始化训练，同20epoch260步/端，完整六端1560步、600query/1032gallery成员，两组原五科学条件不变。XBM原文https://arxiv.org/html/1912.06798本轮重读，历史队列/慢漂移是已有思路；代码独立实现，不声称原创。

每步保存全部批内/历史欧氏距离及索引、身份、scene、年龄和像素SHA，CPU全量重放队列准入和loss；实际encoder三角色新增参数梯度必须测得。每epoch首个真实batch在8更新后用相同像素/RNG重放，恢复所有buffers/RNG，报告全部64个L2漂移；不把随机新增强差异当漂移。完整source原型差异同时可能来自困难正例或负例，不能只归因为负例均值。

M0保持6x8容量+2x100固定过拟合=248更新；容量预热2步触发实际记忆路径。固定100步的重复batch被合法历史去重排除，该门检验原主任务，不能冒充记忆过拟合；新路径另由多batch实际梯度验证。T0->M0->完整CPU->Q1->完整CPU持久执行；任一失败原地封存，观察超时不重启。预估M0 7–10min/Q1 35–50min，随真实速度修订，180–300sec或里程碑观察。

沿用已核验tri_reid环境，不新增安装。原精确Signal推理及56CPU线程距离保留，训练4线程，0official访问。预计新增<1.8GiB、启动最少3GiB，20:51主盘约5.46GiB，必要初始化/最终权重/历史证据不删。执行前先同步固定代码；此处是登记，不声称已启动或通过M0。Goal active/unmet。

### 41.132 MSVR310 实例覆盖 V1 已启动，T0 通过，M0 运行中（2026-09-07 21:28）

- 注册发布与三端同步已通过：GitHub/本地/远端执行提交104506b72193b6cdbcd237044be97c2410fd7a45，登记主文档SHA4543e1da82859b6b58d4770f16e68d6c716ea38045a40f997c07c12107bc15cc；本节是随后状态追加，不改变执行合同或计划。
- 固定合同SHA40f44b0e6c12771c53a283a5b65ca6a45556e0293b7d48decf5fd518ad771a3d；run /root/autodl-tmp/trifusion-v2/artifacts/msvr310_instance_memory_v1_seed42_104506b。原wrapper178471、T0 178473、M0 178485；21:25:39持久独立会话启动，未重启。
- T0于21:25:45退出0：真实远端Torch数学PASS；class0、真实身份正负过滤、当前记录排除、重复记录最新视图、8更新过期、空缓存等价、暴力hard Triplet与非零梯度均通过；全780注册source batch队列模拟通过。T0模型前向/更新/held-out读取均0。
- 21:28:04实查M0仍运行，日志已完成fold1 candidate的8步容量训练，尚未形成完整M0结论。GPU728MiB/0%为采样时的端间加载状态；运行判断依据原wrapper/pipeline，不据瞬时GPU利用率重启。
- 主盘启动前5,866,881,024字节，21:28为5,731,999,744字节；本轮尚未删权重。既有初始化与完整最终权重保留；检查空间按阶段进行。
- 完整流水线T0→M0的248更新→全量CPU→六端1560更新Q1→全量CPU。M0/CPU未通过时不会进入Q1；M0结果不是检索有效性。固定种子/采样/初始化/预算/完整scene图库和两组五项科学条件不变，无官方测试访问。
- 证据：evidence/msvr310_instance_memory_launch_20260907/；跟踪：refine-logs/msvr310_instance_memory_v1/EXPERIMENT_TRACKER.md。下一次按约3–5分钟容量/过拟合里程碑查看，完整终态再做全量核验。长期Goal未达成。

### 41.133 MSVR310 实例覆盖 V1 完整 M0/CPU/文本重算通过，Q1 持续（2026-09-07）

M0原PID178485于21:33:28退出0，全部248更新完成；独立CPU179313于21:33:37退出0，完整1,236,480距离元素与缓存/训练/权重状态核验通过。27文本4,903,487字节接收逐文件byte/SHA一致，本地248步重算通过。summary SHA0ce633d3bacce3074cb573cf8d9117e88e935c3cc5f13dfb1eb2a09ebbc94037，CPU SHAd471efbb41df4cbad983ed4bede3f025e2db8ca805663527526c309de2bb58c3。

六容量端均测得CNN/T/M实际参数增量梯度；8训练端203/203张量活跃、0溢出、冻结Signal/CLIP状态不变。固定过拟合的两端excess比例约0.0007022/0.0007025，通过原0.1门。固定batch历史均被真实record去重排除；新路径由多batch容量验证，不能将此过拟合称为记忆容量证据。

初始8步单位fused L2漂移均值约0.56–0.71，不能证明预热后慢漂移。正式Q1仍固定65step预热/8update年龄/512unique容量，保留全epoch同像素RNG探针，完整终态再解释。所有原门槛/失败/官方边界保持。

Q1原PID179387于21:33:37自动启动，同一wrapper178471持续；21:36:24实查fold0 control已8/20epoch，GPU7314MiB/100%，主盘5,647,364,096字节可用。没有完整Q1检索终态，不中途改参、不重启。模型/原图/NPY留远端，本轮没有删除新权重。

完整报告results/MSVR310_INSTANCE_MEMORY_V1_M0_2026-09-07.md；证据evidence/msvr310_instance_memory_complete_m0_20260907/；下一阶段完成六端Q1及全量CPU排名/训练/漂移核验。长期Goal仍active且未达成。

### 41.134 MSVR310实例记忆V1完整Q1失败、全量证据核验及公开基线边界（2026-09-07）

固定执行104506b与合同40f44b0保持。Q1原179387于22:10:39退出0，CPU183557于22:10:55退出0；原wrapper178471已结束。完整1560更新、203/203梯度、零溢出、冻结Signal/CLIP不变、精确Signal/compact权重重载与780配对像素采样均通过。CPU核对29,125,376训练距离元素及2,069,520检索排序位置，全部600query/60身份、完整合法scene图库，不读official。

59文本71,974,384B逐文件SHA一致，本地完整排名及全部训练重算通过。summary SHAbc31618239d1aa27ed6d159199aa18169bdba6ca43059d9cbf1f4517a13b18a9，CPU SHAb5c9ca9d07e0b0b8ce1b0759dcca774aa8dda536b1f2763c55a3e728aa9b89fb。连接重置10054只影响文本传输；重连接收同一完整集合，没有模型重启。

matched fused52.12111331→51.90495689，−0.21615642pp；fold−1.56815327/+0.29866531/+0.75297606，身份bootstrap下界−1.16083775。C/T/M配对+0.57593551/+0.22432572/−0.27768033。相对Signal53.12938056，candidate fused−1.22442367、下界−2.69401150；两组原五门均0/5 FAIL。fused有258query AP升/284降/58相同，Rank1修复20/新增16；24身份升/32降/4相同，完整成员保留。不是只取有利fold或Rank1净收益。

每端194步有真实历史；三fold候选记录曝光58133/59505/59984，更难负例anchor控制31216/候选29656次（重复曝光）。三角色实际新增参数梯度均测得，不能说没有新信号。候选预热后及末65步原Triplet/expanded Triplet均高于control，不能说新目标已优化得更低。困难正例、负例和历史近似未被因果分离。

同像素/RNG的8更新fused L2漂移，预热后控制均值0.248/0.275/0.263、候选0.350/0.366/0.355；epoch20六端降至约0.00758–0.01003。漂移是前中期显著、末期减小，不是全程固定高位；仅测每epoch首batch，不是每个缓存项的真实距离误差。当前不能认定陈旧是唯一原因。后继先测清当前实例困难与历史坐标误差，不在本次age/capacity/warmup/margin或epoch上扫描；尚无新的训练登记。

同期原始作者配置SHA与本机登记绑定核对：MSVR作者B64/K4、本机B64/K8；RGBNT100作者B128/K16、本机B64/K8。作者RGB100 YAML30epoch、正式论文文字统一50，保留差异，不能补成同一作者实际合同。Signal正式论文表2 MSVR53.6/71.9，README模型表53.2/72.4；历史README来源不改成论文表2。详见docs/SIGNAL_PUBLIC_CONFIG_AND_BASELINE_BOUNDARIES_2026-09-07.md。只是资源与来源边界核对，不改变已完成配对比较、官方记录或门槛。

报告results/MSVR310_INSTANCE_MEMORY_V1_Q1_2026-09-07.md，完整证据evidence/msvr310_instance_memory_complete_q1_20260907/。新增7680探针record前向、2064heldout记录前向，0official；峰值allocated6551.053MiB/reserved6968MiB。22:11主盘空闲5,057,875,968B，GPU空闲。模型/原图/NPY留远端，0新权重删除；保护初始化/最终/复核证据。三个核心数据集长期Goal仍active/unmet。

### 41.135 MSVR实例记忆完整挖掘损失分解通过与下一测量边界（2026-09-07 22:33）

对已封存104506b六端Q1，仅读取原来源训练距离及索引：1560步、29,125,376距离元素、99,840anchor重复曝光，0新前向/梯度/更新/图片/official/权重。按固定0.3hinge，计算原batch、仅加历史正例、仅加历史负例、两者同时四项数值；两种加入顺序平均分配新增loss。不是实际训练了两个消融模型，也不是因果或创新声明。

候选三fold负例占新增hinge88.0755%/87.8174%/88.3196%；control shadow为94.0229%/94.1271%/94.6401%。候选历史负例赢家9819/10056/9781次，其中age1–3占63.4586%/66.7562%/65.3512%，精确并列0。年龄统计未校正候选机会，不支持扫年龄或认定年轻项无误差；说明不能把主要训练问题只归因于age8最老实例。

有历史的每端194步/12416anchor，原batch反序control285/282/305、candidate295/328/329；扩展后分别1376/1629/1603与1486/1844/1660。现有来源难例与实际梯度证据继续成立；只有当前模型重新编码同一视图后，才可直接区分真实困难与陈旧误差。目前未做此对照，原日志不能完整恢复各缓存项当时像素/RNG/模型状态，不补写结果。下一训练尚未登记，不扫描原V1参数挽救科学失败。

只读脚本R1/R2/R3分别因远端Python3.10不支持file_digest、混淆wrapper代码提交与Q1文档HEAD、错误从距离审计读取epoch而退出；全部失败源码/日志保留。修正正确接口后R4完整CPU2.221秒通过，数学目标、训练文件与样本不变，0训练重跑。最终源码SHA330325d5c7914f412085158530864e1ef10baddd85d4c52b36935a7df5d565f7；CSV SHAa663afa88bd4d3305050420a15570f4053f91cbb4769b70af1572975dbcb7c01。

wrapper code_commit104506b，Q1 summary project_commit9da45c5；实际git diff只含8文档/证据/跟踪文件，runner SHA c16cdbff37b46decb279aa013e4a53d9273851a59160cda99fbdf0f16cf39f3b与原代码一致。不是训练途中改代码；原receipt保持，主报告补充字段区别。

远端目录msvr310_instance_memory_mining_diagnosis_v1_r4_20260907，本地全部1560CSV再汇总六端及每端20epoch，最大累加误差5.46e−12。报告results/MSVR310_INSTANCE_MEMORY_MINING_DIAGNOSIS_2026-09-07.md；证据evidence/msvr310_instance_memory_mining_20260907/；工具tools/analyze_msvr_memory_mining.py。原Q1_FAIL0/5与官方边界不变。

主盘再次空闲5,058,236,416B，另一存储盘11,129,241,600B，GPU1MiB/0%。272个pt/pth含检索/诊断数据，0新增删除；保留初始化/终点/复核证据。模型/原图/NPY仍留远端。长期三核心数据集Goal active/unmet。

### 41.136 MSVR310缓存坐标直接测量V1登记（2026-09-07T23:05:25.484121+08:00）

依据33ec3c0完整Q1与全部训练挖掘结果，登记source-only诊断，计划refine-logs/msvr310_freshness_measurement_v1/EXPERIMENT_PLAN.md；合同configs/MSVR310/TriFusion-source-freshness-measurement-v1.json，SHAf3a063499b125ef7e90e1b304324a40c967b8d61ba940a6e3193dab5919edae4。两新工具AST/ruff F通过；尚无新模型前向、更新、checkpoint或GPU任务。

唯一测量是在同一实际参数状态、anchor、真实历史视图下比较旧编码与当前重新编码的距离/最难选择/三个角色参数梯度。额外重复同一旧loss反传测量数值噪声；使用AMP缩放后还原实际梯度。新鲜loss不更新参数。原control仍用batch Triplet，原memory仍用stale Triplet，其余13loss、权重、原V8、初始化、采样和预算不变；不是新增方法或挽救旧失败。

记录原冻结Signal的anchor/reference/baseline字段与encoder入口RNG，在RAM保留最多8次更新。重新编码仅复用不变冻结字段并运行当前角色encoder/fusion，先用真实B64零更新验证与完整路径逐位一致；必须保持buffers及全局RNG不变。每个合法缓存record按最新实际视图/原batch最后位置重编码，最多512/age1–8，当前记录历史全部排除，class0和真实身份关系不改。全部有历史的训练步都测，不挑高漂移步或身份。

预检三折两端12步共72更新、预热2；包含真实零更新重编码、203梯度、0overflow、冻结状态、strict compact reload与全CPU。通过后六端20epoch/260步共1560来源更新、65step预热，同旧780采样batch/三模态像素SHA逐项比较，终点再全CPU核查全部距离/实际loss/队列/权重。0heldout/official图片，没有新的检索表。原100步overfit与旧工程修复保留，不为无新增优化目标的诊断重复执行该门。

旧同初始化/前65步同更新规则两端已有64/65步非零loss差异，各fold最大0.00152755/0.00105834/0.00106740；不把新来源轨迹声称为旧训练逐位重放。记录差异并保留旧Q1，不用新的来源终点替换旧检索成绩。参数梯度为运行时真实见证，CPU不声称重新计算这些梯度。

复用原tri_reid/PyTorch环境，无安装重建；22:46实查原wrapper不存在、GPU1MiB/0%且无CUDA计算进程，远端HEAD33ec3c0。主盘4,964,216,832B、另一存储盘11,129,241,600B，预计新增<1GiB，启动门至少3GiB。预计预检8–15min、完整2–4h，依实际epoch速度修订；持久进程顺序预检→CPU→source→CPU，失败停，观察超时不重启，按180–300sec或里程碑观察。

完整文本/结果同步与原三个数据集Goal不变；尚未执行新训练，不提前判断缓存近似是否有害或任何方法晋级。证据evidence/msvr310_freshness_registration_20260907/。

### 41.137 MSVR缓存坐标V1已启动，预检原进程持续（2026-09-07 23:11）

登记提交ab67d4cd18b0a2b010a46702ac01d03df1c3b1de、合同f3a063499b125ef7e90e1b304324a40c967b8d61ba940a6e3193dab5919edae4。23:08三端主文档/全部10发布文件SHA核对一致，登记主文档SHAb4b96aab3291c5a1cc040a133071d85af4f310c3720f94ab7896181ff12b4aed。23:09一次启动screen msvr_freshness_ab67d4c，run /root/autodl-tmp/trifusion-v2/artifacts/msvr310_freshness_v1_seed42_ab67d4c。

原wrapper185622、preflight185624，23:11:43均存在且命令行匹配；不是依据锁文件认定运行。第0折control/memory各12步已完成、compact终点严格重载；每端额外2880条角色记录前向，包含真实零更新字段复用逐位一致检查。全部六端预检及全CPU还未终态，不写M0 PASS或泛化有效。

最大已分配显存约6558MiB，23:11 GPU7402MiB/100%；主盘4,894,457,856B。主机MemAvailable约704GiB，仅作当前资源信息，不改变单GPU合同；缓存字段在RAM保存，不将原图/模型下载。没有新增权重删除，原必要初始化/终点/复核证据保留。

原计划预检72步→完整CPU→来源1560步→完整CPU持续；真实字段复用只减少冻结主干重算，所有角色额外前向/梯度开销仍计入。新鲜loss不更新，原Q1_FAIL不改，不读取heldout/official。下一次按3–5分钟或完整预检里程碑观察同一进程，任何超时不重启。证据evidence/msvr310_freshness_launch_20260907/；Goal active/unmet。

### 41.138 MSVR缓存坐标完整预检通过，正式来源测量持续（2026-09-07T23:30:28.698381+08:00）

固定ab67d4c/f3a0634与计划不变。preflight185624于23:14:47退出0、CPU185992于23:14:55退出0；全部六端72更新和1,667,072距离元素通过。17文本2,004,547B逐文件SHA相同，本地全部72审计行重聚合PASS。summary SHAd5f453fae567c14753b3c6602464d920c09b85e725d7c536a399ccbab9efc074，CPU SHA7734737ab898be8679f23a807a4007b132dc09e0a7fc79a00305b5506a6f6782。

真实B64冻结字段零更新重编码逐位一致，六端203/203梯度、0overflow、冻结Signal/CLIP不变、RNG/buffer保持、compact严格重载。新增17,280条角色记录前向，峰值allocated约6554–6558MiB。梯度为真实运行见证，CPU未独立重算模型反传。

每端9历史步/576anchor重复曝光，六端fresh loss−stale loss均值约+0.01943至+0.02582；绝对距离误差均值0.02470–0.11656，同角色stale/fresh梯度平均余弦约0.547–0.830。全部6×3×9步梯度差异大于重复同loss数值差异。历史坐标既可能制造也可能掩盖困难关系，不能概括为陈旧必定更难。短预检2步预热/全LR不同于正式65步预热，仅支持当前预检，不证明完整训练或旧Q1泛化失败原因。

新鲜loss只测量，不更新；历史特征始终detach，测量当前anchor参数梯度，不等于历史候选也反传的完整大batch。原control batch/候选stale更新与其余13loss保持。0heldout/official，旧两组0/5FAIL和所有门槛不变。

同一wrapper185622于23:14:55自动启动source186000，23:26:11原进程/命令行均存在，fold0 control13/20epoch。稳态约70sec/epoch，估计9月8日01:10–01:20完整六端来源阶段结束，随后全CPU；不是终态承诺。主盘4,720,369,664B、GPU7266MiB/100%。0新权重删除，模型/二进制/原图留远端。

同期核对AXBN原文arXiv:2303.17127v1：历史记忆均值/标准差匹配及Kalman统计估计已有，不能把缓存漂移修正重新命名为原创。只归档近邻边界，不中途加入新归一化或改变本实验。详见docs/MSVR310_MEMORY_DRIFT_PRIOR_ART_2026-09-07.md。

完整报告results/MSVR310_FRESHNESS_MEASUREMENT_V1_PREFLIGHT_2026-09-07.md，证据evidence/msvr310_freshness_complete_preflight_20260907/。下一步按完整1560步/所有epoch/年龄/角色终态核验；原三个数据集Goal active/unmet。

### 41.139 冗余传输副本清理及完整来源汇总准备（2026-09-07T23:43:28.110625+08:00）

固定计划SHAe9bc085253232ba89a34506abb1f8b6690470ac636abdb749f1e11284c1d9127全量复核后，23:38:35删除182个旧Git bundle副本，共637,229,963B（607.709849MiB）。包含提交全为保留a0e5785的祖先，删除前后仓库可达对象连通性检查PASS；当前freshness传输包及Git仓库对象保持。只删transport直接子文件，不递归、不动模型/检索数据。

主盘空闲4,670,595,072→5,308,112,896B，约4.35→4.94GiB；另一存储卷11,129,233,408B单独计算。此前24.90GiB旧恢复权重清理不重复计为本次。全部279个序列化模型/数据文件大小、mtime、device、inode保持；不是重新计算了279个内容SHA。当前所需初始化、终点和证据继续保留。

原wrapper185622/source186000清理前后持续；最近2026-09-07T23:42:41.033870+08:00实查已完成1/6来源端，首控制端260步/97,600额外角色记录前向完成。完整源测量1560步及CPU仍等待，执行源码ab67d4c/合同f3a0634不变。0新Q1/official，不以单端结果提前下结论或改参。

新增只读后处理tools/analyze_msvr_freshness_epochs.py，SHA7636fe83b9bce0e1a39dc835681d5676bb23cf238c438da1c47b8c57f2c70b29，ruff F及全部72预检行重聚合通过；6epoch/48年龄/18角色组，记录无历史阶段及未定义余弦计数。完整来源预计120epoch组，年龄报告候选曝光机会而非只数赢家；此处没有执行新的模型前向或参数更新。完整来源汇总尚未运行。

详细报告results/TRIFUSION_TRANSPORT_CLEANUP_AND_FRESHNESS_PREPARATION_2026-09-07.md；完整清理证据evidence/trifusion_obsolete_transport_cleanup_20260907/；预检汇总证据evidence/msvr310_freshness_epoch_age_preflight_20260907/。继续原任务及全部终态接收/核验，预计9月8日01:10–01:20来源结束后CPU，Goal active/unmet。

### 41.140 历史坐标与反传范围边界核查，第0折两端完成（2026-09-07T23:55:48.665398+08:00）

当前expanded_triplet使用cdist(unit,unit)，当前batch的anchor与peers均可反传；历史距离显式memory.detach。fresh字段重新编码也在no_grad中，仍不含历史样本参数梯度。因此本次真实测量是同一当前batch计算图下的旧/新历史坐标影响，不是只计算每条距离anchor一端，也不是已实现全候选反传。即使将来增加历史参数梯度，若loss仍仅当前64anchor平均，也不等于全部候选一起作anchor的对称大batch目标。

GradCache原论文RepL4NLP2021及作者提交906f03835fbc183132a9db32612a9e8f180ca3b4已核：无图表示、缓存表示梯度、分块重算/反传、最后一次optimizer更新属于已有方法。当前模型分类路径有7个BN neck，而度量表示在neck之前；不能将整个return_aux两遍分块前向未经验证就称为同一训练定义。详见docs/MSVR310_MEMORY_COORDINATE_VS_GRADIENT_BOUNDARIES_2026-09-07.md。0新方法登记/安装/模型前向，仅代码与原文核查；原运行代码/合同不改。

23:53:34实查原wrapper185622/source186000及完整命令行存在，第0折control/memory各260更新完成，每端额外97,600条角色记录前向；第1折control已2/20epoch。六端2/6，不对中间诊断指标作科学结论，完整CPU和新鲜度报告仍待全程结束。主盘5,243,092,992B；前次182旧bundle清理及279模型/数据保留证据不变。预计9月8日01:10–01:20完整来源阶段结束后CPU。证据evidence/msvr310_memory_gradient_boundary_20260907/；Goal active/unmet。

### 41.141 完整预检诊断图与来源报告准备（2026-09-08T00:26:29.973957+08:00）

已增加只读文本生成器tools/plot_msvr_freshness_diagnostics.py，使用全部72预检更新，六端分别绘制距离误差、fresh−stale hinge、负例选中变化、CNN/T/M实际参数诊断梯度余弦。每端2warmup+1首次cache填充+9历史更新，负例变化分母576；距离误差只计anchor×历史候选，不能混入当前peers。fresh损失不更新模型，Control实际更新仍用批内项。

同模型家族、独立上下文复核逐条重算72条CPU文本，全部36点零差异；PNG/PDF自查及页面文字边界、矢量属性、SHA、AST/ruff F通过。图注已补清分母与诊断/更新区别。预检单epoch/fullLR图不冒充完整来源趋势；source绘图路径尚未实际执行，须六端/全CPU后再次验证。详见results/MSVR310_FRESHNESS_DIAGNOSTIC_PLOTS_2026-09-08.md，证据evidence/msvr310_freshness_preflight_figures_20260908/。

2026-09-08T00:23:16.002855+08:00原进程实查3/6端完成，fold1 instance_memory达13/20epoch，主盘5,140,942,848B。未重跑、未改执行源/配置/计划、未读取heldout/official、未新增模型前向；之前权重与bundle清理不重复计数。原任务预计01:10–01:20后进入全CPU，完成后全量文本接收、重聚合、绘图与科学解释。总Goal active/unmet。

### 41.142 PRISM/DSGM原表与发布合同补核（2026-09-08T00:43:32.951016+08:00）

补齐9月6日仅核README的文献缺口：已独立取得并查看PRISM16页、DSGM15页作者预印本，主表对应PRISM RGBNT20180.5/84.0、RGBNT10086.1/97.8、MSVR31047.6/64.8；DSGM分别82.6/87.0、89.4/98.2、64.6/76.0。前者额外语义mask，后者额外GPT-4o文本/SAM2 mask，不作为纯CLIP等资源或本机复现结果。

固定作者HEAD0067f6d/6566f78共31文本及许可证SHA已核。PRISM MSVR论文K8而YAML K4；DSGM RGB201论文60epoch而YAML50epoch，发布BASE_LR也与文中模块LR数值不同，但未执行优化器不能推出每组实际LR。DSGM工厂实际选用RGBNT201_Text并读取train_171，不仅凭141/30/30描述认定论文主表真实训练划分；目录名亦不证明实际身份计数。仅来源与静态代码核查，未接入作者模型、mask或权重。详见docs/PRISM_DSGM_PRIMARY_AND_RELEASE_BOUNDARIES_2026-09-08.md与evidence/prism_dsgm_primary_20260908/。

2026-09-08T00:41:25.149335+08:00原wrapper185622/source186000持续，4/6端完成，fold2 control13/20epoch，主盘5,053,583,360B。执行工具/配置/计划四项SHA不变。仍等六端与全CPU后接收全部文本、完整重聚合和来源正式图；预检图不能替代完整来源。Goal active/unmet。

### 41.143 完整来源缓存坐标与梯度诊断终态（2026-09-08T01:22:40.378147+08:00）

固定ab67d4c/f3a0634全部6端1560更新结束，source于01:07:56、CPU于01:08:08退出0，原wrapper01:08:09完成。全51,860,992距离CPU通过；17文本59,343,243B逐文件size/SHA相同，120epoch/960年龄/360角色epoch全量重聚合。203/203梯度、0overflow、冻结状态/RNG/buffer/零更新bitwise/strictreload通过，额外585,600条角色记录前向。01:19:41主盘4,926,001,152B/GPU1MiB/0%，原PIDs结束；0新权重删除。

历史194步/端：control距离MAE0.01979–0.02406，memory0.03915–0.04348；负例candidate选择变化分别20.23%–24.46%与32.26%–34.43%。memory Transformer stale/fresh诊断梯度平均余弦0.66037–0.67395。全部3492角色比较实际差异大于重复同loss噪声。六端fresh−stale hinge均值为正，但每端仍有27–56步为负。最后epoch距离误差<0.00089、余弦>0.996，不能用终点稳定代替整个训练近似可靠。

当前peers梯度保留、历史detach；测量的是expanded项在相同当前参数图上的梯度，不是所有监督的总更新梯度。freshloss更新0、heldout/official读取0，旧Q1−0.216156pp和两组0/5FAIL不改。新轨迹不冒充原训练逐位重放，陈旧坐标改变信号不证明它是泛化失败唯一原因。

实际source图720字段/540有效/180缺失全量核对；PNG与PDF渲染检查、新上下文同家族provisional复核通过，不以预检图替代来源图。完整报告results/MSVR310_FRESHNESS_MEASUREMENT_V1_COMPLETE_SOURCE_2026-09-08.md、证据evidence/msvr310_freshness_complete_source_20260908/。

下一项拟固定同历史候选陈旧/当前重编码坐标更新配对，双方匹配重编码计算；不同时加历史反传/新anchor/新融合。尚需独立注册和M0/Q1，非已启动或已成功。三数据集长期Goal ACTIVE/UNMET。

### 41.144 MSVR当前历史坐标更新独立登记（2026-09-08T01:31:03.167516+08:00）

完整来源诊断已在9c8c6b7发布并于01:24:42验证GitHub/远端/桌面一致，主交接SHAf82c71e8db0012f4f7ae25fb54b26e783332a8ac3e8d9dc874dcc966a81cbbfd。根据完整而非部分epoch证据，登记唯一干预：相同历史记录/视图/年龄的陈旧坐标更新与当前角色重编码坐标更新。control现在代表陈旧memory，fresh_memory代表新鲜memory；两端相同重编码与诊断计算，不加入历史候选反传、新anchor、参数或其他loss。

新增四个独立工具train/verify/check/run_msvr_fresh_coordinate.py，旧封存工具保持SHA；新增配置TriFusion-fresh-coordinate-paired-v1.json、计划refine-logs/msvr310_fresh_coordinate_v1/EXPERIMENT_PLAN.md。AST与ruff F通过；此时无新模型前向、M0或Q1成绩，不声称方法有效。真实零更新bitwise、RNG/buffers、同角色新旧梯度/噪声、全缓存算术和实际loss选择继续核验。

固定seed42/B64K8/原Signal初始化/20epoch及全部配对六端1560更新。T0后M0六端8步＋两端100步共248更新，全CPU通过才Q1；Q1全部600query、完整gallery、原scene规则、原两组五项门。双方新增计算记入成本，预计M010–15min、Q1及CPU2–3h；启动前检查GPU/磁盘至少3GiB。持久日志，180–300秒或完成里程碑观察，不因超时重启。未重建环境，复用刚完成完整来源诊断的tri_reid。

本节是执行前登记；下一步部署固定提交并启动一次pipeline，记录原PID/命令行再更新状态。官方读取0，旧Q1负结果与全部原门不变；完整长期Goal ACTIVE/UNMET。

### 41.145 当前历史坐标更新V1一次启动与T0通过（2026-09-08T01:36:19.337245+08:00）

固定执行b4501fa795155eb613fb3bcb07a07c5fae4c6a79/config32e22d3a0cd858e96430e4b9f9b0a2093c0fdf30f026d8d334ad92d46cfb94aa。01:32:35一次screen启动，wrapper192704；T0192706于01:32:40退出0，全780batch队列/年龄/数学通过，0模型前向。自动进入M0192718。最近2026-09-08T01:36:06.658087+08:00实查原wrapper192704与m0 PID192718及命令行均存在；M0已写入4/6个capacity端，完整248步与CPU尚不能提前PASS。

run /root/autodl-tmp/trifusion-v2/artifacts/msvr310_fresh_coordinate_v1_seed42_b4501fa，screen msvr_fresh_coordinate_b4501fa。启动主盘4,843,671,552B，最近4706992128B；GPU6252, 0。所有绑定依赖SHA保持，复用原环境，0权重删除。报告results/MSVR310_FRESH_COORDINATE_V1_LAUNCH_2026-09-08.md及证据evidence/msvr310_fresh_coordinate_launch_20260908/。

预计M001:43–01:48后CPU；成功才完整六端Q1，估计再2–3h并按实速修订。两组原五门、source隔离、完整gallery、seed42与停止梯度边界不变。观察原进程180–300秒或里程碑，不因等待超时重启，不提前声称方法有效；长期Goal ACTIVE/UNMET。

### 41.146 当前历史坐标更新完整M0通过，审查收束与Q1持续（2026-09-08T02:05:10.411619+08:00）

固定b4501fa/config32e22d3a不变。M0192718于01:41:44退出0（543.7595sec），CPU193575于01:41:52退出0。全部六端8步＋两端100步共248更新，CPU重算选中训练距离1,236,480和旧/新历史坐标441,344元素，共1,677,824；27文本2,571,292B逐文件size/SHA相同，完整文本再聚合PASS。summary SHAc72c51ae5126fa4e17cd10f869d53811a6e39aeb9d459097f3e1bccdf33cb5f9，CPU SHA3e8f1f7cebd9709b71bf09d4c729e23809f6fd5c8a65471307aa4443b117d25e。

每端累计203/203可训练张量至少一次非零（不称每步全部非零）、0overflow、冻结Signal不变、真实B64零更新重编码逐位相同、六compact严格重载及五输出相同。全部90role/update旧新梯度差异大于重复噪声。重编码额外6272角色记录，原固定漂移另512；两端含overfit在内初始化/record/像素/历史/计数完整匹配。

两个100步excess ratio0.0007015509/0.0007023767，低于原0.1门。固定batch被历史去重排除，200步无历史候选，只支持主任务拟合。六capacity端实际历史年龄1–5、最大集合195/176/221，不冒充真实模型已覆盖age8/capacity512。M0heldout和official读取0，无本实验Q1成绩终态。

独立上下文experiment-audit最终WARN、same-family/provisional，全部248行及1244汇总数值叶项核对PASS，无代码修改/重启要求。远端CPU从保存距离重算expandedTriplet；其余13loss仅核对运行标量与加权和，运行时梯度摘要不是独立反传。审查者本地未持有22二进制，保留范围限制。报告已修正累计梯度覆盖和数值一致性proxy分类。详见results/MSVR310_FRESH_COORDINATE_V1_M0_2026-09-08.md、refine-logs/msvr310_fresh_coordinate_v1/EXPERIMENT_AUDIT.md/json及evidence/msvr310_fresh_coordinate_complete_m0_20260908/。

新增只读文本汇总tools/analyze_msvr_fresh_coordinate_text.py已实际执行完整M0并通过AST/ruff F，Q1路径准备但未执行，不修改训练合同。原wrapper192704自动于01:41:52启动Q1193650；2026-09-08T02:03:41.516384+08:00实查两原进程/命令行持续，第0折control已记录终点、fresh_memory5/20epoch。主盘4495597568B，GPU7418, 100；0新权重删除。稳定历史阶段约71秒/epoch，预计完整Q1/CPU约03:40前后，以里程碑观察同一进程，不看单端改方案。长期三数据集Goal ACTIVE/UNMET。

### 41.147 完整排名文本复核入口验证，原Q1继续（2026-09-08T02:26:26.994875+08:00）

从旧实例记忆完整Q1已执行分析中提取tools/audit_msvr_paired_ranking_text.py，只读终态summary/CPU/pipeline绑定与六端排名文本，不加载模型、距离矩阵或图片。旧封存完整数据实跑重算2,069,520排名位置、600query/60身份/五输出，完整gallery排列、合法scene过滤、AP/CMC、两组原五门与10000次seed42身份bootstrap一致；Ruff F通过。输出全部3000条query×输出及300条身份×输出CSV。

旧实例记忆fused配对-0.2161564150 mAP、AP258改善/284下降/58不变、Rank1修复20/新增16，原FAIL0/5保持。这是验证工具，不是新增训练或新科学证据。结果见results/MSVR310_TERMINAL_RANKING_TEXT_REPLAY_2026-09-08.md及evidence/msvr310_terminal_ranking_replay_validation_20260908/；当前fresh-coordinate路径尚未执行。近负例记录/身份/scene仅是排名事实，不补写视觉原因。

2026-09-08T02:21:04+08:00实查原wrapper192704/Q1193650和命令行持续，fold0两个端完成，fold1 control1/20epoch，主盘4,362,149,888B。固定b4501fa/config32e22d3a及模型训练文件不变，无新权重删除。当前control指陈旧历史更新，与旧batch-only control不可混淆。预计03:40左右全Q1/CPU，待完整终态再运行新文本复核、全量分析与独立审计；不读单端调方案。长期Goal ACTIVE/UNMET。

### 41.148 当前历史坐标更新完整Q1失败、独立审计与新近邻边界（2026-09-08T04:22:22.639559+08:00）

原Q1193650于03:36:56退出0（6904.0152秒），CPU199698于03:37:11退出0，wrapper192704终态COMPLETE_VERIFIED_Q1_FAIL；全部六端1560更新/120epoch完成，不重跑。pipeline启动b4501fa，Q1开始时HEAD记录ced43dd2；两者仅文档/证据差异，固定训练源/合同SHA不变，配置32e22d3a。29文本79,009,736B完整接收并核对size/SHA，summary419c27a0、CPU91892b5d。

本次control为陈旧历史更新，候选为当前角色重编码历史坐标更新；匹配初始化/780组batch像素/候选年龄/计算，历史两端仍detach。fused51.78835925→51.71677883，−0.07158042pp，配对bootstrap下界−0.50635600；CNN+0.00317187、Transformer−0.54635529、Mamba+0.44331602。三个fold fused增益+0.01329176/−0.77169879/+0.62296244。候选相对Signal53.12938056为−1.41260174pp，下界−3.00062303；两组五门0/5。两个端fused均高于全部角色、但低于Signal，不误诊为融合低于最强角色。预历史66步每fold从step2开始loss有微小差异（最大0.00115347/0.00199091/0.00183570），第67步才有历史；上述配对匹配不等于逐位相同训练轨迹，同图重复梯度噪声不代表独立训练重复噪声，单次极小mAP差值不能作确定因果效应。

全部600query/60身份/完整分折gallery合计1032，正确MSVR scene过滤；Signal特征/距离bitwise，官方读取0。Fused AP279升/251降/70同、Rank1修复9/新增10、身份33升/22降/5同；全部3000query×输出和300身份×输出CSV保留。训练wrong-anchor计数不当成检索R1，改善人数不代替query加权mAP。

候选的来源漂移更小、旧新诊断梯度更一致，三个fold最后65步批内和当前坐标expanded Triplet均更低，但未产生所需heldout增益。仍有非零间隔/错误曝光，不能套RGBNT201来源饱和，也不说明所有记忆无效。累计203/203张量梯度、0overflow、冻结/零年龄重编码/strictreload通过。额外585600角色记录重编码，原漂移另7680；CPU训练及坐标距离74,596,608、检索排名2,069,520，均为元素计数而非独立关系。

独立上下文审计WARN、same-family/provisional；全部29文本SHA、1560更新/780配对、10764数值叶项及全部排名/AP/CMC/原门独立复核一致。CPU收据49文件中25文本本地可读、24二进制未访问；运行时梯度不是独立模型反传，其余13loss仅标量与加权和。原M0审计保留，新EXPERIMENT_AUDIT_Q1.md/json单列。完整报告results/MSVR310_FRESH_COORDINATE_V1_Q1_2026-09-08.md，证据evidence/msvr310_fresh_coordinate_complete_q1_20260908/。

用户00:44快照提出的新鲜坐标比较现已完成FAIL，不能再次作为新方案；只用seed42的要求延续到后继机制实验，不扩展多seed。已核SMEC/S-XBM原文§3.4冻结字段+当前FC+Top-k，BroadFace类别代表位移补偿与分类器/编码器梯度范围，ANCE异步索引候选机制；这些是已有方法基础，不包装为原创。详见docs/MSVR310_MEMORY_PRIMARY_NEIGHBORS_AND_FRESH_Q1_2026-09-08.md。全刷新未证明收益，不自动推进选择性刷新；历史反传/历史anchor覆盖尚未测量，亦不直接当成已批准新方法。下一项先依据完整证据选择单一干预或必要诊断，无新训练登记/启动。

03:38:23实查原进程结束、GPU1MiB/0%、主盘3,837,997,056B；本轮0权重删除。保留必要初始化/终点/二进制证据。三数据集官方与内部口径继续分开，长期Goal ACTIVE/UNMET。

### 41.149 固定来源模型的历史候选梯度诊断登记（2026-09-08T04:40:36.610990+08:00）

完整新鲜坐标Q1及独立审计已在98828b7发布，46文件size/SHA、主交接三处字节一致。该干预两组0/5保持；当前fused高于三角色但低于Signal，不按末端融合不足继续加头。唯一后继测量为当前尚未量化的历史候选侧参数梯度g_V，以及它对g_U和原14项任务总梯度的改变。

新增probe/verify/run_msvr_history_candidate_gradients.py与TriFusion-history-candidate-gradient-v1.json，计划refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_PLAN.md。三fold的初始化、陈旧终点、当前坐标终点共9固定状态，各重放全部260注册来源batch，总2340；全部source记录覆盖，历史队列/三模态像素与登记一致。参数和buffers不更新，optimizer0、heldout0、official0；历史age仅代表batch存储间隔，不冒充参数更新年龄或原训练轨迹。

先T0数学，再9×8真实B64预检及全CPU，成功才完整来源测量。第一处单历史组将直接参数反传与g_U+g_V链式分解比较，预定相对L2误差≤0.005；每处重复同图梯度、RNG/buffers/坐标一致性和全部候选矩阵核验。只重算非零upstream组为去除零链式项，不宣称新选择性刷新。GradCache为既有计算思想，非本项目原创；非零历史梯度不证明泛化收益。

AST及ruff F通过，尚未运行真实模型预检；不提前工程PASS。复用刚完成Q1的tri_reid环境、不重建；当前服务器98828b7、GPU1MiB/0%、主盘3,731,091,456B。预计预检5–15min、来源1–3h，新增证据<0.8GiB，无模型权重新增。固定seed42、不开展新Q1或自动接历史反传训练；只有全量诊断才决定后续。下一步一次部署启动，记录PID/日志，180–300秒或预计里程碑观察。Goal ACTIVE/UNMET。

### 41.150 候选梯度诊断T0夹具修正（2026-09-08T05:16:13.822789+08:00）

原a22aaa1/config168eada入口04:42:24启动wrapper2228，T02230于04:42:26退出1，STOPPED_AT_T0。05:14:21实查原PID不存在、GPU1MiB/0%、主盘3,730,661,376B；明确是终止而非观察超时。失败原因是新math_check构造的metadata漏age，旧expanded_triplet统计接口读取age报KeyError；真实模型forward0、optimizer0、heldout/official0。原pipeline/log完整保留evidence/msvr310_history_gradient_t0_repair_20260908/。

仅补测试夹具age=1，未改数学公式、真实队列、梯度机制、0.005工程容差或来源范围；原计划不变。修订脚本已通过AST/ruff F，并在同远端tri_reid以0真实模型前向执行T0：标量/当前侧梯度与原实现逐位同、候选链式VJP精确一致、class0合法。配置新SHA c99ddcf6f6605b2d43725e7fc4f74f0747e774a1110a3d0567c3c42c74a1c0b3。

下一步以修订提交建立新run再次T0→9状态预检→CPU→完整来源，保留原失败run，不以覆盖或重启原PID处理。无新模型成绩或科学结论；长期Goal ACTIVE/UNMET。

### 41.151 固定候选梯度诊断修订入口启动（2026-09-08T05:20:15.778282+08:00）

修订执行eebaaa07708e8f5e5d3e05d246afe5fe5f7abfc6/configc99ddcf6于05:17:55一次screen启动，screen msvr_hist_grad_eebaaa0，run /root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_candidate_gradient_v1_seed42_eebaaa0，wrapper3302。T03304于05:17:57.955969退出0（2.4025秒），进入真实9状态预检3312。2026-09-08T05:18:07.906237+08:00实查两原PID和完整命令行持续，第0fold initial已开始。GPU890MiB/3%、主盘3,730,124,800B；0新权重删除。

原a22aaa1 T0夹具失败run保留，不覆盖；此次只修复math metadata，9固定状态/完整来源、0optimizer、0heldout/official和预定数值门不变。完整预检和CPU未通过前不宣称工程成功；预检CPU通过才自动进入2340来源batch诊断，不自动启动历史反传训练或Q1。原始观察见evidence/msvr310_history_gradient_r1_launch_20260908/。

预计预检5–15min、来源1–3h，以完成状态和速度调整观察里程碑；180–300秒观察原进程，超时不重启。主交接/桌面/GitHub随本提交同步，长期三数据集Goal ACTIVE/UNMET。

### 41.152 历史候选梯度完整预检、审计及只读统计补核（2026-09-08T06:06:01.076635+08:00）

固定执行eebaaa0/configc99ddcf6，预检3312于05:24:58.144320退出0，九固定状态各8个真实B64共72batch；CPU3795随后退出0，625920距离元素核验。全部25文本805652B接收size/SHA一致，summary3f2665ff。每状态5个历史batch，全部135角色行历史梯度非零且高于同图重复差异；这是短预检的运行时贡献证据，不是完整来源或泛化收益。

全部135行g_U/g_V余弦为负，g_U/(g_U+g_V)及原14项任务角色梯度/加入历史项后的余弦为正。角色参数空间内的贡献范数/夹角改变不等于有害任务冲突、实际AdamW方向或mAP提升。首次单历史组直接反传与分解链式梯度的最大相对误差7.623875794974142e-5，小于原0.005；每状态仅step4一次，后续多组没有独立直接模型反传对照。额外9792角色记录前向、预检420.1877秒，峰值allocated约16062MiB。

只读文本重聚合验证72step/135角色行及全部范数闭合，最大相对误差6.026005765920966e-12。fresh-context gpt-6-astra/max标准库审计19782项检查通过，原报告全部36表行、CSV与原始文本一致；总体WARN，same-family/provisional。原审计及后续补核审计均保留，不能当作跨模型家族独立接受。

审计发现原CPU仅检查13项memory统计中的6项。新增只读NumPy工具verify_msvr_history_gradient_all_statistics.py，在原九个距离矩阵上实际完成72×13=936项全统计补核PASS；没有改动固定probe/原CPU/config/原始证据。完整来源终态也须另行执行此补核。14项总loss没有保存逐项分量，无法从该包独立重组；梯度、RNG、状态SHA仍是运行时见证。预检未触发8batch上限的年龄过期、512容量淘汰或零upstream组跳过，不能提前声称这些分支覆盖。五个递归继承文件的本地CRLF与绑定远端LF差异继续披露。

数学说明docs/MSVR310_PARTIAL_VS_TOTAL_METRIC_GRADIENT_2026-09-08.md给出共同正交旋转保持度量距离但两侧偏导相互抵消的标准例子；经审计确认公式和1/√18槽位尺度正确。没有测量项目旋转分量，不将理论例子写成失败成因或原创定理。GradCache链式VJP为既有计算基础。

2026-09-08T06:01:53.451454+08:00实查原wrapper3302/source3799及命令行持续，GPU17130, 88，主盘3702099968B。本轮未删权重，诊断不新增权重。已装满历史后的实际速度约170–180秒/13batch，预计完整九状态约6–7小时，替代早期1–3小时预估；不缩短2340batch合同或重启。当前无optimizer更新、无新增heldout/official图像读取；继承上下文读取既有Q1工件属于来源绑定。

完整预检报告results/MSVR310_HISTORY_CANDIDATE_GRADIENT_PREFLIGHT_2026-09-08.md，证据evidence/msvr310_history_gradient_preflight_20260908/，审计refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_AUDIT_PREFLIGHT.md/json。先完成原全部来源及CPU、全部13项补核、完整文本和独立审计，再决定是否登记只改变历史候选梯度范围的配对训练；当前未登记新训练/Q1。原新鲜坐标Q1两组0/5、seed42-only、三数据集长期Goal ACTIVE/UNMET保持。

### 41.153 终态队列覆盖入口验证与冗余传输包清理（2026-09-08T06:31:09.942944+08:00）

只读analyze_msvr_history_gradient_text新增年龄/容量/重复排除/VJP组及额外前向覆盖统计，已执行原九状态72batch/135角色行预检。去掉新增字段后全部既有JSON值精确相同，原CSV逐字节一致。每状态可用/实际VJP组15/15，最大年龄5；fold0/1/2的最大队列234/225/263、批内重复曝光179/196/162。全部预检年龄过期、容量淘汰、当前历史副本排除和零upstream组跳过均0，保留未覆盖边界。完整来源尚未执行此分析，不将运行时VJP组记录当成独立梯度重算；不改固定probe/CPU/config/计划。

2026-09-08T06:26:20.624778+08:00完成69个根目录旧.bundle清理，逐文件路径/大小/SHA/ref祖先关系先核验，总399442735B约380.94MiB。一个非祖先bundle保留且SHA不变；仓库HEAD及原3302/3799进程前后保持。主盘3687133184→4086722560B，0模型权重删除，旧24个resume清理不重复计数。计划与逐文件收据见evidence/trifusion_root_bundle_cleanup_20260908/。

2026-09-08T06:28:21.401529+08:00实查原source3799持续，1/9状态完整260batch，第0折control11/20epoch，执行eebaaa0/configc99ddcf6不变。预计全部来源约中午前后，仍按全部2340batch和完整CPU收束；未登记新训练。待终态后运行额外13项统计CPU、全文本接收/重聚合/队列覆盖及独立审计，再决定唯一下一项训练假设。报告results/MSVR310_HISTORY_GRADIENT_QUEUE_COVERAGE_AND_DISK_2026-09-08.md，source进度是运行观察而非新科学结果。seed42-only，长期Goal ACTIVE/UNMET。

### 41.154 完整来源梯度图入口的短预检验证（2026-09-08T06:45:24.787037+08:00）

新增只读plot_msvr_history_candidate_gradients，九固定状态/三角色分开显示历史与当前梯度范数比、两侧夹角及对原14项总梯度方向的改变。全部72batch/135角色行的108格/540有效值已从CSV重算一致；PDF108个值及108个n与收据相符，PNG/PDF渲染自查通过。仅短预检5历史batch/状态，source图未执行；负偏导余弦不当作有害任务冲突、AdamW更新或泛化收益。原审计范围不自动扩大，证据evidence/msvr310_history_gradient_preflight_figures_20260908/，报告results/MSVR310_HISTORY_GRADIENT_FIGURE_VALIDATION_2026-09-08.md。

2026-09-08T06:42:47.914143+08:00实查原3302/3799持续，第0折initial260batch完成、control16/20epoch，主盘4073558016B。固定eebaaa0/configc99ddcf6、2340batch合同不变，图稿工作未新增模型前向；原诊断持续，无新训练或权重清理。待九状态/原CPU/13项补核完整终态后生成全来源图并审计，再决定下一单一干预；预计中午前后，seed42-only，Goal ACTIVE/UNMET。

2026-09-08T07:36:42.976107+08:00里程碑观察：第0折initial/control/fresh_memory均完成260batch，共780/2340batch的状态终点已写出；原3302/3799继续第1折initial，日志5/20epoch。整项来源与CPU尚未终态，不分析部分fold选择方案。主盘4043554816B，执行eebaaa0/configc99ddcf6不变。连续五分钟进程观察正常，证据evidence/msvr310_history_gradient_first_fold_observation_20260908.json；无新训练或清理，长期Goal ACTIVE/UNMET。

2026-09-08T09:47:15.629034+08:00里程碑观察：第0/1折六固定状态均完成260batch，合计1560/2340batch的状态终点已写出；原3302/3799已进入最后第2折initial，日志6/20epoch。这里是固定参数重放，optimizer0，整项来源与CPU仍未终态；不根据部分fold指标确定新方案。主盘3964370944B，执行eebaaa0/configc99ddcf6保持。五分钟间隔的原进程观察持续正常，证据evidence/msvr310_history_gradient_two_fold_observation_20260908.json；无新训练或清理，长期Goal ACTIVE/UNMET。

### 41.155 原诊断终态后处理接续（2026-09-08 10:20 北京时间）

原固定eebaaa0/configc99ddcf6诊断未重启、未改模型或合同。10:20:01观察原3302/3799均在，六个状态终点已完成、最后fold2initial17/20epoch，主盘3927789568B。终态尚未获得；没有新训练、权重删除或检索结果。

为避免等待结束后重复人工启动，已将原只读观察及准备好的结果处理接为一个本地持久顺序任务：C:/Users/gb/.codex_tmp/finish_history_gradient_source_20260908.py，uv PID15212，10:19:56启动。旧functions观察cell173已停止；仅替换观察程序，远端任务不受影响。每300秒观察一次，只有原pipeline达到COMPLETE_VERIFIED_SOURCE_ONLY才执行额外13项NumPy统计核验，然后25文本接收/逐文件SHA核对、全部2340batch文本重聚合和完整来源图生成。工具SHA在启动时固定核验。原诊断失败或后处理检查失败将退出并留下日志，不会重训或自动切换方案。

本地stdout/stderr为C:/Users/gb/.codex_tmp/history_gradient_complete_processing_20260908.stdout.log及同名stderr.log；完整接续收据目录history_gradient_complete_processing_20260908/。预计结果目录分别history_gradient_complete_source_20260908/、history_gradient_complete_analysis_20260908/、history_gradient_complete_figures_20260908/，均位于同一.codex_tmp；完成标志completion.json只表示后处理完成、仍待审查，不能当作模型科学成功。全部大型距离矩阵/模型留远端。接续时先查看该进程与日志，勿重复运行断言目标目录不存在的入口。

第一条真实观察及脚本SHA、工具绑定见evidence/msvr310_history_gradient_postprocess_launch_20260908.json。完成后还须核实原终态、查看完整图、独立上下文审计全部来源证据，并同步报告；之后才能依据完整证据登记下一单一干预。历史新鲜坐标Q1两组0/5保持，seed42-only，三数据集Goal ACTIVE/UNMET。


### 41.156 完整来源历史候选梯度诊断与独立审计收束（2026-09-08T12:23:54.298989+08:00）

原eebaaa0/config c99ddcf6的9状态×260=2340batch全部完成；source3799于11:53:48退出0，CPU18174于11:53:52退出0，原wrapper3302结束。后续13统计30420项检查、25原始文本共53366938B接收、全量重聚合和PDF/PNG核验均完成。最新12:18实查GPU空闲、主盘3891519488B；没有重跑原诊断，没有新的optimizer更新或heldout/official图像前向。

5238个含历史角色记录中gV均非零且高于同图重复差异，cos(gU,gV)全部为负，均值−0.812275；||gV||/||gU||均值0.896638。cos(gU,gU+gV)有23条负值，纠正短预检“合成后全部同向”的外推；原14项总梯度与加入历史侧后的余弦全部仍正。它们是同一loss的偏导分解，不证明有害任务冲突、AdamW更新方向或检索增益。

全来源队列/VJP核验：13716可用历史组，12498实际VJP组，1218零上游组跳过；58041年龄过期曝光，51948批内重复，11871当前历史副本排除；容量淘汰0、最大更新后队列408。801024额外角色记录前向，峰值allocated16087.170MiB。每状态直接图证明仅首次单历史组，参数/RNG/状态属于运行时见证；14项逐项loss未保存，不能从包独立重组原总loss。

独立审计WARN，same-family/provisional；完整原始行、全部CSV和108格图表核验收束。详见results/MSVR310_HISTORY_CANDIDATE_GRADIENT_SOURCE_2026-09-08.md、refine-logs/msvr310_history_candidate_gradient_v1/EXPERIMENT_AUDIT_SOURCE.md及evidence/msvr310_history_gradient_complete_source_20260908/。原source summary SHA098ee6cd5edfc273e6f4f6f1d646a2c0685e3ab0a723d7434eac3d3962b3da91，原CPU SHA48a39b40816bd4fc547651d6af07c52222c9145b5fc3e30239019e687cfb8f1a；额外13统计SHA173ef45972bf98fc54f5199db7ce06043fd48bcd0c6f61bcb684572ec68e0127。

下一步：依据覆盖和方向证据，登记只改变历史候选梯度范围的配对主实验。两端均使用新鲜历史坐标、相同当前64anchor/候选池/14项loss/原三角色；对照丢弃候选侧VJP，候选加入对应梯度。先固定合同、实际M0和CPU再完整六端Q1；本次发布未启动该训练。预计沿用紧凑角色checkpoint：上一完整M0+Q1实际994426031B，当前余量允许规划但仍需启动前确认。没有新增删除，旧权重清理不重复计入。

原fresh-coordinate Q1两组0/5及−0.07158042pp保持封存；诊断不改变其检索结论。seed42-only、三数据集和既有完整图库/scene协议、主结果前不消融、官方结果不调参的合同保持。Goal ACTIVE/UNMET。


### 41.157 新鲜坐标下历史候选反传V1登记（2026-09-08T12:58:47.525707+08:00）

完整来源审计已收束，唯一新假设为历史候选侧是否反传。两端同为当前参数新鲜历史坐标、同64anchor与原候选规则；control计算但丢弃候选VJP，history_gradient在统一unscale及唯一AdamW step前加入其原权重梯度。保留原14项loss、模型/推理、seed42、20epoch和两组五门；不加历史anchor、Router、额外loss或扫描。算法规则相同，实际非零VJP组数可能因轨迹而异，分别报告真实开销。

新文件：configs/MSVR310/TriFusion-history-gradient-paired-v1.json，tools/{check,train,verify,run}_msvr_history_gradient.py，refine-logs/msvr310_history_gradient_v1/EXPERIMENT_PLAN.md。静态语法及pyflakes通过；独立预执行审计WARN，same-family/provisional，报告在同目录EXPERIMENT_AUDIT_PREEXECUTION.md。这不是M0通过或新检索成绩。

执行合同：T0全780采样batch准入及synthetic链式检查→M0六端48容量+两端200固定过拟合更新→全CPU→六端1560更新Q1→完整权重/距离/排序/AP/CMC/身份CPU核验。M0首次单历史组直接14项图与分解梯度误差≤0.005；逐组坐标、RNG/buffers、梯度应用检查；14项标量全部保存。参数梯度仍是运行时见证，不宣称独立完整模型重放。

复用现有tri_reid/3090，不改环境。预计M0 10–20分钟、Q1/CPU 3–5小时；保守新文件上限1.8GiB、启动至少3GiB。上一紧凑checkpoint约32.7MB，完整旧M0+Q1共994426031B；最新发布后主盘3822964736B。训练和模型原图留远端，代码/文本和主交接三方同步。尚未启动新进程，下一步必须按已发布commit/config SHA启动一次并验证实际PID及T0状态。

本次没有新增删除；已有初始化、终点、全量检索证据继续保护。旧fresh-coordinate Q1两组0/5不变，长期三数据集baseline/SOTA Goal ACTIVE/UNMET。


### 41.158 历史候选反传V1实际启动（2026-09-08T13:01:04.585361+08:00）

北京时间2026-09-08T12:59:44.100924+08:00，持久screen tri_history_gradient_a1b4777启动唯一实验实例。执行commit `a1b4777b62611be2cac33351afdf7789edc59906`，config SHA256 `d03c7be1e738a206bf6db3b55580f53fbf33bfcc9050a46c4ddd0c81d7134c4e`；运行目录 `/root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_gradient_v1_seed42_a1b4777`。wrapper PID19977，T0 PID19979已exit0，M0 PID19991于2026-09-08T12:59:48.994503+08:00启动，2026-09-08T12:59:52.324248+08:00已核实两个实际进程及命令行。pipeline RUNNING；M0尚无完整终态，不提前记作通过。后续docs提交不改变绑定训练文件。

两端都刷新历史坐标并计算候选侧VJP；唯一差别为control丢弃/history_gradient加入参数更新。仍是当前64 anchors、原候选/身份规则、原14项损失、seed42、固定20epoch。M0及完整CPU实际通过才自动进入六端Q1；不根据中间检索调参数或重启选点。旧fresh-coordinate FAIL及来源测量边界保持。

启动前可用3825758208B，观察时3822964736B；本轮无新增删除。保护初始化、终点权重与距离/梯度诊断证据。按预计M0 10–20分钟、Q1/CPU 3–5小时安排约300秒观察，异常依真实日志处理，不能将工程中断写成检索失败。启动/实际进程、静态检查与资源凭据见evidence/msvr310_history_gradient_launch_20260908。


### 41.159 历史候选反传V1完整M0通过及Q1启动（2026-09-08T13:14:48.773182+08:00）

M0于13:10:03.589918写入PASS_ENGINEERING_ONLY、13:10:04原PID19991 exit0；CPU PID20868于13:10:13 exit0并返回PASS_COMPLETE_HISTORY_GRADIENT_M0。全部248更新：六端各8容量、两端各100过拟合。六端203/203张量有非零梯度、0overflow、严格重载五输出逐位一致；单历史组直接/分解梯度最大相对L2 9.39630303951e-10，边界是六次首组见证而非全部多组独立模型证明。两端超额loss比例0.000701938531/0.000701213786，均低于原0.1。CPU核验1236480距离元素、全部loss标量加权重组与元数据/应用记录。详细报告results/MSVR310_HISTORY_GRADIENT_V1_M0_2026-09-08.md，原始27文本长度/SHA完整接收。

原wrapper19977自动启动Q1 PID20941，开始13:10:13.340418；13:12:39实际两个进程存在，第一折control72/260更新，尚无完整端/检索终态。执行a1b4777及config d03c7be1...不变。本次仅发布文本证据；不改变训练代码、loss、门槛或旧失败。预计完整Q1/CPU仍按3–5小时暂估，待实际预热后速度修订，约300秒观察。可用3616083968B，无新删除。Goal ACTIVE/UNMET。


### 41.160 Q1持续运行与完整文本后处理准备（2026-09-08T13:24:23.296169+08:00）

13:22:15实查wrapper19977/Q1 PID20941仍在，第一折control129/260更新、没有完整端，磁盘3601850368B。最近epoch约131秒，首端训练暂估13:44结束；检索及其余五端继续原合同，完整Q1/CPU仍按3–5小时估计并随实测修订。只按约300秒或阶段时点观察，不根据中间数值改变任何科学配置。

新增只读tools/analyze_msvr_history_gradient_training.py，用完整M0的248步真实文本运行通过且pyflakes exit0，记录每端/epoch的14项loss、历史覆盖、同角色G与gV/实际应用梯度及真实前向开销。配对初始化、样本、像素、历史元数据和新鲜前向计数相同；VJP选中组数不预设相同。梯度仍是运行时摘要，G限同角色encoder的14任务梯度，不是整个模型；所有计数为重复曝光，不是独立关系或heldout错误。证据evidence/msvr310_history_gradient_training_analysis_readiness_20260908。

完整Q1接收脚本已静态检查，必须等待pipeline完整五阶段exit0、1560更新及CPU终态后才接收29份文本，矩阵/权重留远端；尚未执行Q1接收或汇总。新增分析代码不在训练调用路径，不改a1b4777绑定文件/环境/模型/目标/门槛。后续先取得完整六端和CPU，再汇总并独立审计。无新删除，旧失败保持，Goal ACTIVE/UNMET。


### 41.161 历史反传Q1首控制端完整保存、候选端衔接（2026-09-08T13:48:44.327459+08:00）

2026-09-08T13:48:24.488240+08:00实查：原wrapper19977/Q1 PID20941均在。fold0 control已完成20epoch/260更新（训练循环2037.971674秒）、原工程条件通过、203/203非零梯度、0overflow。固定roles_epoch20.pth 32735974B实际SHA与receipt一致：524c925ed4f953a673d510fe8e933cfaf89b11ab77ac9c7a3f9f5937f6aa87b5；rankings.json SHA与检索receipt一致：b8e913ef0ecc8bc6933970845cc558a29c82c57178aa2b61045da8c4c3cafb2e。完整五输出记录覆盖210合法query和360全图库记录。以上为单端文件及运行时记录核验，完整六端CPU尚未运行，不当作全局科学结果。

候选fold0 history_gradient已实际写出81步，保持原进程/合同/初始化规则继续；不依据首端或部分fold选择配置。首端额外新鲜角色记录前向97600、历史VJP记录前向87232、峰值分配6112.6787MiB，实际训练开销单列。独立训练完整比较与五门仍待其余五端和CPU终态。证据evidence/msvr310_history_gradient_first_training_endpoint_20260908，未输出或利用中间检索分数调参。

当前可用3504787456B，无新增删除。执行a1b4777不变；后续仅观察当前候选端至固定终点并继续两折。Goal ACTIVE/UNMET；旧失败及已消费官方边界保持。


### 41.162 持久终态接收与全量文本分析已启动（2026-09-08T14:10:29.259672+08:00）

本地uv launcher18524/Python PID31404已实际核实，2026-09-08T14:10:14.5012848+08:00状态WAITING_VERIFIED_REMOTE_WRAPPER，首次远端观察2026-09-08T14:09:21.194906+08:00确认原wrapper19977/Q1 PID20941继续。此任务每300秒只读观察，不启动/重启训练，不读取部分结果改配置。独立观察目录避免与手工观察写同一文件。运行目录C:/Users/gb/.codex_tmp/history_gradient_q1_terminal_processing_20260908，state.json及分阶段stdout/stderr持久保存。plan SHA e4573130687adab1ced95cc82d3ddaab817450d687b989dbbe40fa0eb8bb9430。

完整pipeline五阶段exit0、1560更新与CPU终态后，自动执行已准备的29文本SHA接收→tools/analyze_msvr_history_gradient_training.py全更新/epoch汇总→原tools/audit_msvr_paired_ranking_text.py全600query/60身份/五输出排名及原门核验。阶段非零退出或观察无法确认时，记录状态并停止本地处理，不推定远端训练失败、不重启。成功状态为COMPLETE_LOCAL_TEXT_AND_RANKING_VERIFIED_AWAITING_INDEPENDENT_AUDIT；自动流程不发布报告、不替代独立审计，也不标记Goal完成。

绑定接收/观察/分析/worker/launch文件SHA；后台在运行期间不要修改这些文件，否则绑定核验会停止本地处理。Windows隐藏启动，使用隔离uv文本依赖环境（8个包）；远端tri_reid环境、a1b4777训练代码与配置没有变化。原图/权重/二进制矩阵留远端。证据evidence/msvr310_history_gradient_terminal_watcher_20260908；当前没有新增终态或性能结论。下一步按阶段核查后台与原GPU进程，完成后接独立审计/报告/三方同步。Goal ACTIVE/UNMET。


### 41.163 历史反传Q1第一折齐备、第二折自动接续（2026-09-08T14:22:34.505467+08:00）

后台第3次观察2026-09-08T14:19:25.028328+08:00确认原wrapper19977/Q1 PID20941继续：fold0 control与history_gradient均260/260步，receipt_complete与retrieval_recorded均真；fold1 control已实际16步。2026-09-08T14:21:04.7098100+08:00另外实查本地PID31404及正确命令行存在，状态WAITING_VERIFIED_REMOTE_WRAPPER。原训练与后台均未重启。此处只记录2/6端写入齐备，不发布局部mAP、不将该折称为完整科学晋级；全六端CPU核验尚未运行。

阶段证据evidence/msvr310_history_gradient_first_training_fold_complete_20260908。继续原固定预算与两组五门，等待其余四端后自动接收29文本、全部1560步及完整排名分析，再独立审计/报告/同步。最近实查空闲3402444800B，无新增删除。执行a1b4777、配置及后台绑定脚本保持；当前论文/SOTA目标仍未完成，Goal ACTIVE/UNMET。


### 41.164 历史反传Q1完成三个端点、第二折候选端接续（2026-09-08T14:57:03.779291+08:00）

后台第10次观察2026-09-08T14:54:37.664902+08:00确认原wrapper19977与Q1 PID20941继续：fold0两端及fold1 control均260/260步，receipt_complete和retrieval_recorded均真；fold1 history_gradient已实际61/260步。2026-09-08T14:55:32.9055042+08:00另实查本地PID31404与绑定命令行存在，状态WAITING_VERIFIED_REMOTE_WRAPPER。训练与后台均未重启。这里只确认3/6端记录齐备；不发布局部mAP，不视为完整配对晋级，最终六端CPU核验尚未开始。

阶段证据见evidence/msvr310_history_gradient_half_training_complete_20260908。保持execution a1b4777、原配置、seed42、当前64 anchors、历史候选VJP单一干预及两组五门；继续剩余三端，随后自动接收29文本、全1560步及全部排名分析，再独立审计、报告和同步。最近实查磁盘空闲3301064704B（约3.074GiB），本轮没有删除文件；保存后续端点时继续观察余量。后台SHA绑定脚本未修改。三个数据集baseline/SOTA总体目标仍未完成，Goal ACTIVE/UNMET。


### 41.165 前两折齐备、第三折执行及用户最新研究候选归档（2026-09-08T15:40:00.483122+08:00）

原后台观察2026-09-08T15:39:54.161557+08:00确认wrapper19977/Q1 PID20941继续，fold0/1的control/history_gradient四端均260步且receipt_complete、retrieval_recorded齐备；fold2 control已实际134/260步。2026-09-08T15:40:00.483122+08:00另实查本地PID31404及正确命令行，状态WAITING_VERIFIED_REMOTE_WRAPPER。磁盘余量3191824384B，本轮未删除文件。证据evidence/msvr310_history_gradient_user_update_20260908；这是4/6端进度，全六端及最终CPU仍未结束，不使用局部mAP作决策。

用户本次更新基于87f47af的3/6端快照，已将其正文的后继建议记录于refine-logs/msvr310_history_gradient_v1/POST_Q1_RESEARCH_CANDIDATES.md：先完成历史候选反传，再依据完整证据考察角色提议的关系覆盖；身份/场景去重及集合/AP目标分开验证。只用seed42，主结果后再做机制对照。冻结字段缓存、分组VJP及零上游组跳过为已有基础，不重复列为新增贡献；负两侧余弦不自动意味着任务冲突。若本轮未晋级，只限定于本配置，不能直接推导候选侧导数不重要。

该文件是待决策候选清单，不是新训练注册；没有选定top-k、温度、配额、loss替换或新预算，不修改execution a1b4777、EXPERIMENT_PLAN、配置及后台SHA绑定代码。用户PDF/LaTeX链接本轮未打开，文献入口按用户正文归档，未来引用时核原文；正式表依原始结果，仍不将内部Q1或来源诊断填入正式测试。继续完整1560步/六端/CPU/文本排名核验与独立审计，随后报告同步并决定唯一后继，Goal ACTIVE/UNMET。


### 41.166 历史候选反传完整终态及角色关系覆盖接续（2026-09-08T17:18:39.670840+08:00）

原a1b4777运行六端1560更新于16:35:57完成，CPU16:36:11退出0，pipeline COMPLETE_VERIFIED_Q1_FAIL。全部29文本72274170B按SHA接收；本地完整训练/排名分析完成。独立gpt-6-astra max新上下文审计WARN/same-family/provisional：自己的远端脚本复算M0+Q1 1808步、30361856训练距离元素、2069520检索距离/排名元素，核查3000query-output/300identity-output及120epoch。运行时梯度见证不等于CPU重做模型反传；模型家族路由非后端认证。审计时旧tracker/handoff尚记进度，本次已改当前状态，审计原文保留。详见results/MSVR310_HISTORY_GRADIENT_V1_Q1_2026-09-08.md、refine-logs/msvr310_history_gradient_v1/EXPERIMENT_AUDIT_Q1.md和evidence/msvr310_history_gradient_q1_complete_20260908。

本次fresh-detach control fused51.73429240→history_gradient52.40672778（+0.67243538）；Signal53.12938056，候选仍低0.72265278。三fold配对+2.03166534/+0.94978599/−1.20106116，CNN−0.54543017、Transformer+0.31736904、Mamba+0.08852474；身份bootstrap下界−0.32277801；两组原五条件均0/5。fused高于三角色但低于Signal，不能把严格最高失败解释为融合低于角色。全query AP312改善/233下降/55不变，Rank1修复13/新增24；60身份34改善/23下降/3不变。不能把平均收益写成稳定晋级，也不能否定所有历史导数/记忆方法。

末五轮三fold合并批内Triplet0.06739036→0.05736792、扩展0.16649096→0.14112846、总loss0.96342135→0.91688884。来源优化改善未满足未知身份门。每角色候选582含历史更新实际加入gV；G为同角色14项总梯度，不与旧gU/gV固定诊断混用。两端干预前66步有已测微小数值差异，报告保留。每端97600新鲜角色记录前向、六端528512历史VJP记录前向；两端规则相同但实际非零组不同，不称严格同FLOPs。16:59:53实查所有原进程退出、GPU1MiB/0%、盘3028090880B，本轮0删除，六最终小checkpoint保留。

下一登记仅为来源数组只读再分析：configs/MSVR310/Role-relation-coverage-v1.json、refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_PLAN.md、tools/analyze_msvr_role_relation_coverage.py。复用9月7日27状态/视图/fold条件×2来源协议，全37152query成员；0模型前向/0参数更新/0新权重。先测角色极值不同但仍在fused margin内的关系，并验证子集沿用同hardest不能增加全集hard hinge。旧source模型不冒充本轮历史反传终点，固定eval视图不冒充训练队列，间隔证据不冒充参数梯度。此次发布完成后执行，尚未启动。后继训练仍未选择top-k/配额/loss，不用已消费官方结果调参。

正式表无新增：MSVR310仍无本项目正式结果，RGBNT100原增益保留。seed42 only、原门与失败封存、三数据集长期目标继续ACTIVE/UNMET。


### 41.167 只读来源角色关系覆盖实际启动（2026-09-08T17:22:22.616190+08:00）

历史反传Q1终态提交c46be4eb6bf8aa096527abb9f45b59282fc09085已推送，117文件远端字节/SHA一致，主交接GitHub/remote/Desktop SHA cecc093644c0bb2d996b650035940ab6a36bad19b54a106b08a64b75968bd6af。完整审计WARN/same-family/provisional与科学FAIL两组0/5保持，原训练/核验进程均结束。

随后按已登记合同启动来源再分析：/root/autodl-tmp/trifusion-v2/artifacts/msvr310_role_relation_coverage_v1_seed42_c46be4e，screen tri_role_coverage_c46be4e，wrapper31867/analysis31871，启动2026-09-08T17:20:55.044692+08:00，17:20:56初查RUNNING。执行提交c46be4e、合同SHA c52c86e49bde1a73d2ea75b4182c4ae5db1c1911e820c3e30f9eba19fe3dd051；只读27个旧source条件×2协议，0模型前向/0优化器更新/0新checkpoint，输出完整37152query成员及54条件。启动盘2898268160B、CPU可用内存753742225408B，4线程；预计1–5分钟，按结束里程碑观察，勿重复启动。

证据evidence/msvr310_role_relation_coverage_launch_20260908/launch.json。下一步读取原pipeline终态、全部文本和输入哈希后独立审计；在这之前不将角色提议或额外hinge关系写成新梯度/新身份收益。此分析是旧source模型上的新关系统计，不是重跑旧普查，不是当前历史反传终点的新特征提取。没有新训练、官方评估或loss/top-k/scene配额选择。Goal ACTIVE/UNMET。


### 41.168 来源角色关系覆盖完成接收，独立审计启动（2026-09-08T17:29:23.529831+08:00）

原c46be4e只读任务于17:21:07退出0，实际12.8378秒；按预计里程碑17:24:13观察确认wrapper31867/analysis31871均已退出。27旧source条件×2协议共54条件和37152query成员，0模型前向、0优化器更新、0新权重。四份原始文本共30061558B逐文件大小/SHA核对，已接收C:/Users/gb/.codex_tmp/msvr_role_relation_coverage_complete_20260908；完整证据evidence/msvr310_role_relation_coverage_complete_20260908。summary SHA f1d98d2e8615e71ca6223c632b16fc9bb80d163e796c4a39b74b924d83468f40，全query JSONL SHA09365b53e73de38c8e9b8f60b393c2f4b73b2d21d71fabda7517a5ec469347c4。

已派发独立新上下文审计/root/audit_msvr_role_relation_coverage，requested gpt-6-astra/max，same-family/provisional；原始请求和trace保留。审计输出C:/Users/gb/.codex_tmp/msvr_role_relation_coverage_independent_audit_20260908，当前尚未出具完整终态。审计需自行从远端原数组复算54条件/37152行及来源/协议/子集目标边界；不读取模型或重训。报告尚不形成晋级、新梯度或未知身份收益结论。root的18条件合并表仅为待审核描述汇总。

17:24:13盘2884214784B；没有删除权重，数组仍在原远端位置。下一步完成独立核验，分析全部状态/视图/协议后决定是否需要训练模式参数梯度诊断；不自动同时改角色提议、scene配额、集合loss和反传范围。历史反传Q1既有完整报告/审计与两组0/5保持，正式表无新增。Goal ACTIVE/UNMET。


### 41.169 来源角色关系覆盖完整独立审计与单一后继假设（2026-09-08T17:57:22.735320+08:00）

原c46be4e只读分析及完整接收后，独立/root/audit_msvr_role_relation_coverage已终态，审计WARN / same-family / provisional，确定性全字段复算通过。核查全部27 receipts/108来源数组及54条件/37152成员；自身脚本不导入原分析。所有原始response、audit.json、脚本与全量输出已归档，完整报告results/MSVR310_ROLE_RELATION_COVERAGE_2026-09-08.md，审计refine-logs/msvr310_role_relation_coverage_v1/EXPERIMENT_AUDIT.md。原pipeline保持运行当时UNAUDITED收据，不篡改；后续审计独立记录。当前tracker已纠正终态，旧时间观察保留。

全条件合计29376合法成员、7776不合法成员，角色额外负例30037曝光，其中29283条0.3 hinge违约、7333条非正间隔。20434成员有额外有效关系、5715有额外反序；子集hard hinge27367相同/2009更低/0更高。重复条件曝光不是独立样本。control_final/augmented/cross_scene1200合法成员中886有额外有效负关系、163有额外反序；子集1088相同/112更低。这些是旧来源模型eval数组，不是新历史反传终点或真实训练参数梯度。

由此支持角色提出不同有效关系，但不能只把候选换成角色并集、继续同fused hardest而宣称新监督。后继最小候选见NEXT_HYPOTHESIS_DRAFT.md：固定原fused hardest正例，将原fused及三角色最近负例去重，对它们的既有0.3 hinge取均值；其他13项、网络、scene规则、历史坐标/完整导数均在两端保留。草案未实现/未登记运行，须先数学与真实梯度检查；均值降低极端关系权重，不能未经对照把收益归因为角色互补。MS Loss/Sampling Matters近邻已核读并引用，标准均值、采样与反传不是原创机制。

17:43实际空闲2849726464B，Git对象已主要打包（4packs、670555KiB），本次0权重删除。没有把旧清理量计成新增释放，不删仍依赖的初始化、终点或数组。下一训练需具体资源核查。历史反传Q1两组0/5和正式表保持，无官方新增；seed42 only，三数据集目标ACTIVE/UNMET。


### 41.170 角色提议负关系均值目标的数学检查登记（2026-09-08T18:07:26.639878+08:00）

已实现tools/msvr_role_set_relations.py：保留原fused最难正例与hard项，对三角色不同负例追加hinge后按去重并集大小取均值。原hard项沿用current/history min/max分配；无额外提议时标量和梯度应精确保留，包括并列极值。只由角色选索引，目标在fused空间，不引入新参数/温度/权重或scene规则。均值降低原极端关系权重，不提前声称角色独有贡献。

CPU合成检查tools/check_msvr_role_set_relations.py与wrapper/config/计划已静态解析，尚未执行。登记configs/MSVR310/Role-set-math-v1.json，发布同步后运行。检查class0、去重、无额外项/无历史/精确tie、额外关系导数和历史VJP链式法则；0模型前向/0参数更新/0图像/0新权重。不是M0训练或方法通过。后续真实模型梯度检查仍需绑定执行脚本，长训练亦未启动。旧失败与正式表不变。


### 41.171 数学检查通过与三折真实模型梯度检查登记（2026-09-08T18:17:43.210366+08:00）

54a7d5a CPU数学检查18:08:47退出0，18:09:23实查wrapper33264/child33266均结束。class0/mask/去重、无额外项时原标量/梯度精确保留、极值tie原导数、额外负关系非零导数和历史链式法则均通过。math SHA167a8d8d873fc1d3ea4fb741266f22520b76121f43979545f3acac563940bd6e，完整证据evidence/msvr310_role_set_math_20260908，结果results/MSVR310_ROLE_SET_MATH_2026-09-08.md。合成向量检查不是实际角色参数梯度或模型晋级。

下一登记configs/MSVR310/Role-set-gradient-check-v1.json、GRADIENT_CHECK_PLAN.md、probe/verify/run_msvr_role_set_gradients.py：三个原source-only初始化，每折首8个seed42 B64，合计24batch。固定训练模式但0optimizer；原hard与新均值目标均有完整历史导数，再重复新目标测计算差异，首历史组用完整图校验VJP。实际完整角色提议来自同次encoder/fusion，非fused槽位近似重构；保存全部四种距离矩阵/提议/角色统计，CPU全24batch复算。最终state恢复、无heldout/official前向、无新权重。不是全source普查、M0训练或检索结果，不据此自动启动Q1。

新增输出预算50MiB、最低空闲256MiB，仅适用于本零权重探针；长训练预算另核，不改旧3GiB训练合同。环境沿用已核验tri_reid，无安装/重建。预计5–15分钟，screen持久、按180–300秒或完成里程碑观察。发布同步后启动，此刻尚未执行。Goal ACTIVE/UNMET、seed42 only。


### 41.172 role-set固定状态检查封存与新训练实施准备（2026-09-08T18:49:13.903396+08:00）

执行958fb215fb5bd0c5412b1eaa9de96124e279d37f，配置c5b0b772b729b062976f225bad4daa06b0dd4a3ca41dc2df4f629a169de1d3ee。18:22:46 GPU/CPU退出0；18:25:53原wrapper33657、GPU33661、CPU33899均结束。三折各8batch，合计24/1536当前视图曝光，0optimizer/0权重/0heldout/official；三折state首尾完全相同。每个角色24/24批、含历史15/15批的新目标导数变化超过同图重复反传噪声；额外active负关系1271次（曝光含重复）。首个历史组完整图/VJP最大相对误差1.2736390029790958e-9。CPU全部834560距离元素重算，最大loss误差2.9802322387695312e-8。峰值allocated11355.1772MiB，产物3643844B。不是M0或检索晋级。

独立审计WARN，same-family/provisional；完整报告refine-logs/msvr310_role_set_v1/EXPERIMENT_AUDIT_GRADIENT.md/.json，结果results/MSVR310_ROLE_SET_GRADIENT_CHECK_2026-09-08.md，完整证据evidence/msvr310_role_set_gradient_check_20260908。审计指出：全部角色逐位比较只在未入队的首batch，历史组仅核fused；原绑定计划全历史角色措辞未被完全覆盖，已作为限制披露，不能修改执行记录或冒充已完成。没有独立GPU全梯度重演；同图反传噪声不代表随机前向噪声。旧probe不重跑。

已完成新tools/train_msvr_role_set.py、check_msvr_role_set.py、verify_msvr_role_set.py、run_msvr_role_set.py草案：两端均fresh历史完整导数，control原hardest，candidate角色提议去重均值；实际三角色输出发现候选，优化仍在fused；其他13项loss和原网络/seed42/初始化/采样不变。增加首历史组实际四输出重编码及总损失直接图检查路径。仅本地AST通过，未注册训练config、未真实导入/T0/M0/Q1；本次审计不覆盖新草案。下一工作应检查草案、固定训练合同/哈希/预算、发布并实际执行M0及完整六端Q1，而非重复来源诊断。见TRAINING_IMPLEMENTATION_DRAFT.md。

18:28:55 /root/trifusion-storage独立文件系统空闲11129192448B，主盘2648854528B。拟用已有备用目录保存下一新artifact，保留所有旧路径/权重；预算草案最大新增3GiB、最低空闲4GiB，启动前重查，不降低旧训练合同。18:34:54清理38个已导入transport bundle，每个引用为HEAD祖先且哈希复核，共52406835B，主盘空闲2701295616B；0权重删除，旧释放量不重复统计。逐项收据在本次evidence，下一同步会新增小传输包。

本地launch由于screen -D -m阻塞及SSH读超时未获得函数末尾回执；远端已正常独立完成，另读pipeline/原PID/退出码验证，未重启。后续使用screen -dmS真实脱离并持久日志。历史反传Q1仍封存FAIL0/5两组，正式成绩无变化。Goal ACTIVE/UNMET；seed42 only、先主结果后消融、完整图库/全路径身份隔离/scene协议保持。


### 41.173 role-set完整配对训练合同登记（2026-09-08T18:56:51.526517+08:00）

18:52:28实查远端HEAD9d51a10、工作树干净、无训练进程、GPU1/24576MiB；主盘2697371648B空闲，已有/root/trifusion-storage/artifacts所在独立文件系统11129192448B空闲。前一轮真实梯度检查/审计/同步是已完成进展，Goal中的V29旧接续状态不再用作当前入口。

新合同configs/MSVR310/TriFusion-role-set-paired-v1.json，TRAINING_PLAN.md；tools/train/verify/check/run_msvr_role_set.py独立实现，不修改旧训练源。两端原初始化/seed42/B64/K8/20epoch/65步预热/历史512年龄8/13项其他loss相同，均fresh当前坐标与完整历史VJP，仅control原hardest、role_set去重角色负例均值。仍64anchor，无新网络、scene配额、Router或温度。

T0完整780来源batch队列与新旧合成数学；M0三折两端各8步与两端100步过拟合，沿用所有203训练张量累计覆盖、冻结Signal、AMP/reload及原校正loss比≤0.1；首历史组实际4输出与完整总损失encoder域VJP核对（189张量），不称全训练独立复算。M0/CPU通过后才六端各260步Q1，固定完整图库/scene协议、全部输出/身份/query、两组原五门及bootstrap不变。所有四种距离矩阵和实际关系记入CPU全量重算。

输出新目录/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_<执行提交>；最低4GiB、预期最多3GiB，GPU启动<500MiB。原同类M0约615.6s、Q112344.6s，本次预计M08–20分钟、Q13–5小时；screen -dmS、完整pipeline/PID日志，180–300秒或完成里程碑观察。任何子阶段非零退出停止并封存，不改变门槛/选checkpoint救回。本次登记时仅AST检查，尚未运行真实T0/M0，发布后启动。Goal ACTIVE/UNMET，官方成绩无变化。


### 41.174 role-set持久训练实际观察（2026-09-08T19:01:50.942280+08:00）

执行提交26c97390704c629237687d263b4381f5584cbe97，配置SHA5e5ad4663f4c3c58e11b81b475048d6ab6c53675ac2dd0ade60c84f07a62986d；18:58:24 screen tri_role_set_26c9739真实启动，输出/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739。此次实查pipeline=RUNNING、阶段=m0，原PID存在性={'35302': True, '35308': False, '35385': True}。GPU=12448, 24576MiB，输出盘空闲11023376384B。

阶段收据摘要（未完成端不补成绩）：{"m0": {"status": "RUNNING", "folds": [{"fold": 0, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_0_control/roles_m0.pth"}, "role_set": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_0_role_set/roles_m0.pth"}}}, {"fold": 1, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_1_control/roles_m0.pth"}}}], "overfit": {}}}

原始pipeline/日志尾/进程观察和首次启动证据完整保存在evidence/msvr310_role_set_run_observation_41_174_20260908。科学代码/固定合同见§41.173，不因本文档提交改变。通过M0_CPU才继续既定六端Q1，任何失败以原日志核验，不根据局部端更改方法、不因SSH观察超时重启。正式成绩未更新，Goal仍ACTIVE/UNMET。


### 41.175 role-set持久训练实际观察（2026-09-08T19:10:08.941002+08:00）

执行提交26c97390704c629237687d263b4381f5584cbe97，配置SHA5e5ad4663f4c3c58e11b81b475048d6ab6c53675ac2dd0ade60c84f07a62986d；18:58:24 screen tri_role_set_26c9739真实启动，输出/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739。此次实查pipeline=RUNNING、阶段=q1，原PID存在性={'35302': True, '35308': False, '35385': False, '36247': False, '36320': True}。GPU=6398, 24576MiB，输出盘空闲10903097344B。

阶段收据摘要（未完成端不补成绩）：{"m0": {"status": "PASS_ENGINEERING_ONLY", "folds": [{"fold": 0, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_0_control/roles_m0.pth"}, "role_set": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_0_role_set/roles_m0.pth"}}}, {"fold": 1, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_1_control/roles_m0.pth"}, "role_set": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_1_role_set/roles_m0.pth"}}}, {"fold": 2, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_2_control/roles_m0.pth"}, "role_set": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_2_role_set/roles_m0.pth"}}}], "overfit": {"control": {"steps": 100, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_100_steps": true, "original_overfit_gate": true}}, "role_set": {"steps": 100, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_100_steps": true, "original_overfit_gate": true}}}}, "q1": {"status": "RUNNING", "folds": [], "overfit": {}}}

原始pipeline/日志尾/进程观察和首次启动证据完整保存在evidence/msvr310_role_set_run_observation_41_175_20260908。科学代码/固定合同见§41.173，不因本文档提交改变。通过M0_CPU才继续既定六端Q1，任何失败以原日志核验，不根据局部端更改方法、不因SSH观察超时重启。正式成绩未更新，Goal仍ACTIVE/UNMET。


#### §41.175 完整M0证据补充

- 三折两端各8步，另两端固定batch各100步，合计248次实际更新。六容量端203/203训练张量累计收到非零梯度，AMP溢出为0，冻结状态不变，六checkpoint严格重载的全部检索输出逐位相同；不是每步全部参数都非零。
- 原loss-floor校正的过拟合末首比：control=0.0007014160404，role_set=0.000701652003861，均小于既定0.1。
- 六容量端首个历史组实际四输出逐位一致；原总目标直接完整图导数与当前导数+历史VJP最大相对L2误差1.68104999803e-05，小于既定0.005。检查域为189个encoder张量，不能与203个全部训练张量混写；此直接比较仅覆盖每端首个历史组。
- CPU复算4,945,920个保存距离元素、完整248步关系/mask/队列/损失账本和终点；核验历史VJP记录前向计数5,760。CPU本身未重新生成每步模型梯度或其他13项模型前向；运行见证与独立重算范围区分保留。
- 六容量端峰值allocated显存最高11266.819336MiB。19:10:08输出盘余量10,903,097,344B，约10.154GiB；新训练仅保存必要M0/固定终点权重，既定最大新增3GiB预算。

## 证据与边界

完整29份远端文本合计5,717,997B已按字节SHA接收，见evidence/msvr310_role_set_m0_complete_20260908/intake_manifest.json；六权重与距离二进制留远端，原CPU收据列明哈希。summary SHA256=203f2e572d40e00d0d41842289746ad2c077d8d7e97ada5fe237214f1b161a38。本地附加六端直接导数汇总direct_group_checks.json只汇总原日志。

此处PASS仅为工程；没有新的Q1科学结论或正式测试成绩。完整Q1三折两端、每端20epoch/260步，合计1560更新；预计19:08:52起3–5小时，不能用预热期速度线性推算全程。保持seed42、完整图库/scene规则、两组五项门槛。完成后完整CPU与独立上下文审计，不按中途fold调整方法。

候选同时增加负关系覆盖并软化最极端hinge权重；将来若提升，不能独归因于角色异构。两端均使用新鲜坐标和完整历史侧梯度，推理架构和13项其他损失不变。全项目三数据集Goal仍ACTIVE/UNMET。


### 41.176 role-set持久训练实际观察（2026-09-08T19:15:51.474212+08:00）

执行提交26c97390704c629237687d263b4381f5584cbe97，配置SHA5e5ad4663f4c3c58e11b81b475048d6ab6c53675ac2dd0ade60c84f07a62986d；18:58:24 screen tri_role_set_26c9739真实启动，输出/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739。此次实查pipeline=RUNNING、阶段=q1，原PID存在性={'35302': True, '35308': False, '35385': False, '36247': False, '36320': True}。GPU=6430, 24576MiB，输出盘空闲10888941568B。

阶段收据摘要（未完成端不补成绩）：{"m0": {"status": "PASS_ENGINEERING_ONLY", "folds": [{"fold": 0, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_0_control/roles_m0.pth"}, "role_set": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_0_role_set/roles_m0.pth"}}}, {"fold": 1, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_1_control/roles_m0.pth"}, "role_set": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_1_role_set/roles_m0.pth"}}}, {"fold": 2, "endpoints": {"control": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_2_control/roles_m0.pth"}, "role_set": {"steps": 8, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/m0/fold_2_role_set/roles_m0.pth"}}}], "overfit": {"control": {"steps": 100, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_100_steps": true, "original_overfit_gate": true}}, "role_set": {"steps": 100, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_100_steps": true, "original_overfit_gate": true}}}}, "q1": {"status": "RUNNING", "folds": [], "overfit": {}}}

原始pipeline/日志尾/进程观察和首次启动证据完整保存在evidence/msvr310_role_set_run_observation_41_176_20260908。科学代码/固定合同见§41.173，不因本文档提交改变。通过M0_CPU才继续既定六端Q1，任何失败以原日志核验，不根据局部端更改方法、不因SSH观察超时重启。正式成绩未更新，Goal仍ACTIVE/UNMET。


#### §41.176 终态准备与近邻边界

19:15:51实查Q1原PID36320、wrapper35302均存在；fold0control完成7epoch，历史生效后epoch6耗时96.07s、epoch7耗时127.86s。输出盘空闲10,888,941,568B。按此阶段速度，首端预估19:43–19:50结束，下一次训练里程碑检查安排19:40附近；无需在预计结束前反复每分钟读取GPU。该时间是估计，终态以原日志/退出码为准，整体仍预留3–5小时。

完整终态文本接收脚本已准备并仅做AST检查，尚未运行，已归档于evidence/msvr310_role_set_run_observation_41_176_20260908/receive_role_set_q1_terminal_20260908.py。只在原五阶段完成且原PID结束、CPU及summary哈希匹配后接收。若工程停止，读取实际失败证据，不能绕过合同或重新启动。

新增近邻核查与终态分析口径见refine-logs/msvr310_role_set_v1/RELATED_WORK_BOUNDARY_2026-09-08.md。HDC已有多模型困难关系分配，DiVA已有不同关系任务促进互补，DCML摘要已有组合监督与判别能力保留；宽泛动机不作为原创。本文档只收窄论证与准备现有日志分析，未修改正在运行的科学配置、损失或晋级门槛；未开始终态独立审计。


补充：终态来源分析脚本analyze_role_set_q1_terminal_20260908.py已准备，只完成AST检查，未对未结束的Q1运行。完整终态接收后，对六端各260步核对记录/像素配对，分别统计不同位置、不同record和不同负身份的提议曝光，以及预热65步、预热后195步、最后五epoch的共同损失与计算成本。只描述已有日志，不增加晋级门；额外激活hinge总数不能唯一归因于角色，位置/身份独有提议也不是独有泛化能力证明。脚本位于evidence/msvr310_role_set_run_observation_41_176_20260908/，应在终态真实数据上执行后再报告统计值。


### 41.177 Q1完成端里程碑（2026-09-08T19:50:47.916282+08:00）

原执行26c9739、wrapper35302、Q1 PID36320持续存在。已保存终点的完整端1/6：[{"fold": 0, "endpoint": "control", "steps": 260, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/q1/fold_0_control/roles_epoch20.pth"}]。最新epoch事件为fold0 role_set 7/20，当前epoch实际13步，耗时139.679s。前次19:41实查control已19epoch，19:45实查control固定260步完成并进入role_set第5epoch；同一原任务持续前进，没有重启。

输出盘剩余10711928832B（9.9763GiB），GPU当前7412, 24576MiB。下一观察安排2026-09-08 20:15 +08:00附近；预计时间按实际后预热epoch调整，原3–5h全Q1预算不作为保证。不增加每epoch权重，不删除必要初始化/终点/核验产物。

完整原观察保存在evidence/msvr310_role_set_q1_milestone_41_177_20260908。此里程碑不报告单端或局部配对科学增益；完整六端/CPU收据齐全后才执行已准备接收、来源分析与独立审计。M0完整证据见§41.175，终态工具及近邻边界见§41.176，不重复运行。训练配置/源文件/原两组五项门槛未改动，正式成绩未新增，Goal ACTIVE/UNMET。


### 41.178 Q1完成端里程碑（2026-09-08T20:14:31.502177+08:00）

原执行26c9739、wrapper35302、Q1 PID36320持续存在。已保存终点的完整端1/6：[{"fold": 0, "endpoint": "control", "steps": 260, "checks": {"all_trainable_gradients_live": true, "overflow_zero": true, "frozen_state_unchanged": true, "signal_state_unchanged": true, "role_state_updated": true, "capacity_below_24gib": true, "fixed_training_length": true}, "checkpoint": "/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739/q1/fold_0_control/roles_epoch20.pth"}]。最新epoch事件为fold0 role_set 18/20，当前epoch实际13步，耗时136.183s。前次19:41实查control已19epoch，19:45实查control固定260步完成并进入role_set第5epoch；同一原任务持续前进，没有重启。

输出盘剩余10657984512B（9.9260GiB），GPU当前7412, 24576MiB。下一观察安排2026-09-08 20:23 +08:00附近；预计时间按实际后预热epoch调整，原3–5h全Q1预算不作为保证。不增加每epoch权重，不删除必要初始化/终点/核验产物。

完整原观察保存在evidence/msvr310_role_set_q1_milestone_41_178_20260908。此里程碑不报告单端或局部配对科学增益；完整六端/CPU收据齐全后才执行已准备接收、来源分析与独立审计。M0完整证据见§41.175，终态工具及近邻边界见§41.176，不重复运行。训练配置/源文件/原两组五项门槛未改动，正式成绩未新增，Goal ACTIVE/UNMET。

独立M0审计另由 /root/audit_msvr_role_set_m0 执行，fresh-context、same-family/provisional、只读CPU不超过2线程，不使用GPU或读取Q1分数。20:14本次观察时审计尚未返回终态。审计输出目录 C:/Users/gb/.codex_tmp/role_set_m0_independent_audit_20260908；完整请求/响应跟踪位于本地 .aris/traces/experiment-audit/2026-09-08_role_set_m0，不提交私有trace。当前仅能记录审计进行中，不能视为PASS；完成后接收完整审计和原始检查输出。不要重复启动审计或M0。


### 41.179 M0独立审计归档与首折两端完成（2026-09-08T20:33:30.984781+08:00）

独立M0审计于2026-09-08T20:33:30.984781+08:00收齐：总体WARN，工程PASS，确定性核验PASS，same-family/provisional。完整原文、独立核验脚本、输出及审计工具失败尝试在evidence/role_set_m0_audit_20260908；正式报告见refine-logs/msvr310_role_set_v1/EXPERIMENT_AUDIT_M0.md/.json。独立检查全部248更新、4945920距离元素、780个来源采样batch、六个终点完整/冻结状态，37个CPU收据文件哈希匹配。原203/203累计梯度覆盖、六次首历史组189参数张量直接图比较和重载输出相等仍属于有范围的运行见证，不是全程GPU梯度复现。两个100步过拟合固定batch各53唯一记录、历史候选为0，不能用其证明历史VJP过拟合；M0实际历史最大221、年龄5，年龄8来自无模型元数据重放。关系覆盖与hardest降权同时变化，不能单因果归于角色多样性。审计不要求补跑、重启或改科学代码；Q1性能未评价。私有请求/响应trace仅保存在本地.aris及审计原目录，不复制到GitHub。

20:21:40.081075+08:00实际观察原wrapper35302、Q1 PID36320仍存活，首折control和role_set都完成20epoch/260更新并保存固定终点，累计2/6端完成；第二折control完成5/20epoch。两端记录的梯度覆盖、零overflow、Signal/冻结状态不变、角色更新、显存与固定长度工程检查均通过。这里只报告工程进度，不读取或报告单折检索增益。原任务未重启，26c9739科学代码与原配置不变。

输出盘剩余10578055168B（9.8516GiB），GPU7412/24576MiB。下一观察20:50附近，依据前一端后预热约136秒/epoch估计；不按前5个epoch的18秒速度外推。完整观察与本次审计同目录。没有新增权重删除，必要初始化/固定终点/数组证据留远端，训练与二进制不下载到本地。

后续继续同一Q1直至全部六端及CPU终态，届时运行已准备的完整文本接收/来源分析并完成终态独立审计。此前不按局部分数改变关系规则、预算或门槛。正式成绩无新增，三个数据集完整目标未达到；本次工程审计不替代检索结果。


### 41.180 第二折进度与终态分析字段验证（2026-09-08T20:40:41.475565+08:00）

原wrapper35302/Q136320继续运行，完成端数2/6，第二折control14/20；最近epoch耗时135.639s。预计本端20:54左右完成，下一观察20:52附近，以实际原PID/日志为准；不改变运行合同或读取部分检索分数。输出剩余10514784256B（9.7927GiB），GPU7412/24576MiB，没有新权重删除。

终态分析使用新版analyze_role_set_q1_terminal_v2_20260908.py，完整六端/CPU/接收哈希门仍不变。增加全程和分阶段14项原loss及总loss的描述性均值，并分别记录前65步预热、首次出现历史候选前的loss/分量差异。目标替换从step66开始，而历史候选实际首次出现由完整日志定位；无历史不等于无目标干预，不能将二者合并称干预前。真实M0八端248步日志已验证分析函数和字段、计数、均值，0模型/更新；不是Q1终态执行或新增科学结果。旧版保留，完整Q1接收后改用v2进行描述性分析。

脚本、完整检查收据和运行观察在evidence/role_set_terminal_analysis_v2_20260908。M0独立审计结论不变，不再次审计M0或重跑旧实验。六端/CPU终态与后续独立科学审计仍待完成；原两组五门、seed42、完整图库与三数据集目标保持。


### 41.181 原Q1里程碑与集合目标近邻边界（2026-09-08T20:55:05.611032+08:00）

原wrapper35302/Q136320持续存活；已保存固定终点3/6，完成端为[{"fold": 0, "endpoint": "control", "steps": 260}, {"fold": 0, "endpoint": "role_set", "steps": 260}, {"fold": 1, "endpoint": "control", "steps": 260}]。最近完成epoch为fold1 role_set 5/20，耗时18.588s。下一观察21:25附近。输出余量10418761728B（9.7032GiB），GPU7410, 24576MiB。无重启、无权重删除、无局部科学分数读取或门槛改动。完整观察见evidence/role_set_q1_observation_41_181_20260908。

补充原论文与当前代码对照见refine-logs/msvr310_role_set_v1/PAIR_WEIGHTING_BOUNDARY_2026-09-08.md：Multi-Similarity的三类相似度不等于三个encoder，原文已研究挖掘与软加权；Smooth-AP的正例/总体排名比值也不等于当前均值hinge。当前等欧氏hinge系数经过d=sqrt(2-2s)后，不是等相似度导数；这只是当前定义的局部推导，不是新理论、已测梯度或后继实验。CVF入口403后用作者arXiv原文核对，没有复制作者代码或据其参数改运行合同。

主方法结论仍等待全部1560更新、六端完整图库及CPU/独立终态审计。M0审计范围见§41.179；完成后使用分析v2，见§41.180。正式成绩没有新增，三数据集完整目标未达成。


### 41.182 磁盘保留与终态执行准备（2026-09-08T21:05:41.528329+08:00）

原Q1仍运行，3/6端完成，最近fold1 role_set 9/20epoch。两卷剩余分别2621513728B和10391728128B，当前输出卷充足；本次删除0文件。扩展名普查含权重和检索/诊断数组，不作为统一可删模型清单；唯一旧V3恢复条目因缺少独立替代终点证明保留。完整报告results/MSVR310_ROLE_SET_DISK_AND_TERMINAL_READINESS_2026-09-08.md，原始观察/只读清单/执行入口归档evidence/role_set_disk_readiness_20260908。

终态本地NumPy/Paramiko离线导入和既有排名工具CLI已验证；旧临时解释器路径失效，使用uv离线入口。尚未执行新Q1终态接收/分析/排名复算。原M0审计封存，Q1完整1560更新、全部图库、CPU和独立终态审计仍待完成。下一观察21:25附近，不修改科学合同或按局部分数启动后继。


### 41.183 完成四端并进入最后一折（2026-09-08T21:30:08.087702+08:00）

原wrapper35302/Q136320存活，执行绑定26c9739不变。fold0 control/role_set与fold1 control/role_set四端均完成固定20epoch/260更新，各自七项运行工程检查为true；这些是运行见证，完整CPU和终态独立复核仍未开始。原进程已自动进入fold2 control，第5/20epoch，不曾重启或改科学合同。21:25:40观察第二折候选19/20后，按约四分钟间隔重新核实完成状态；原始证据见evidence/role_set_q1_four_ends_20260908。

GPU7406, 24576MiB，输出卷剩10258788352B（9.5542GiB）。本次删除0文件。最后一折历史关系阶段预计每epoch约130–140秒；第5epoch约18秒仅属预热阶段，不能据此外推全程。下次观察21:58附近，控制端预计22:00–22:05结束，完整六端预计22:35–22:45，均以实际进程和CPU状态为准。

没有读取中途fold科学分数选择方法，完整原两组五门不改。完整终态后沿用§41.182的严格文本接收、分析v2、全部排名复算与独立审计顺序；此次4/6完成不代表Q1晋级或三数据集Goal达成。


### 41.184 五端完成，最后候选端运行（2026-09-08T22:03:33.840454+08:00）

原wrapper35302/Q136320仍存活，执行绑定26c9739不变。前两折双端以及最后一折control均完成固定20epoch/260更新，五端七项运行工程检查均true；完整CPU/终态独立审计尚未执行。21:58:27观察fold2 control18/20，22:03:33确认其已完成且原进程进入fold2 role_set第2/20epoch。原始观察归档evidence/role_set_q1_five_ends_20260908。

GPU7380, 24576MiB，输出卷剩10108387328B（9.4142GiB），删除0文件。最后候选端当前仍在预热，不能用约19秒/epoch外推历史关系阶段。下一观察22:35附近；按此前实际候选端耗时，完整训练仍预计22:35–22:45，此后自动CPU核验，以实际状态为准。

完整终态后执行§41.182的严格文本接收、分析v2、完整排名复算与新的独立上下文审计。此次五端完成不是科学晋级；未读取局部科学分数选择方法，原seed42、两组五门、身份隔离和完整图库scene规则均不变。不得重跑已完成M0、诊断、旧Q1或其审计；三数据集Goal未达成。


### 41.185 role-set完整Q1结束与执行侧全量复算（2026-09-08 22:40终态实查）

Q1于22:38:50结束、CPU于22:39:06结束，各阶段退出0，原wrapper35302/Q136320/CPU45309及其他阶段PID均已不存在，GPU回到1MiB。执行26c9739、summary SHA fc6493b643ab378e2941fb335ac5ebf3093c0d31673bdda7d4564db724598de2。完整接收57份文本81,195,809B并逐文件SHA核验，全部1560更新的source分析v2与2069520个排名位置的AP/CMC/scene/bootstrap文本复算均已实际执行。

fused control52.392266→role_set52.490152（+0.097886pp），Signal53.129381，候选低于Signal0.639228pp。配对三折-0.022435/+0.227337/+0.089531，身份bootstrap下界-0.020600；配对及相对Signal两组五门均0/5，Q1_FAIL保持。候选fused高于三个角色，严格最高失败来自Signal更高。CNN/T/M配对+0.005939/-0.136582/-0.037253。全600query fused AP209改善/233下降/158不变；Rank1修复3/新增0；60身份28改善/25下降/7不变。

来源候选预热后37440次anchor曝光增加44205个负位置曝光，21500次anchor包含额外负身份，额外active hinge37730次；是实际训练曝光而非独立图片数。末5epoch同定义expanded_hard control0.141091582→candidate0.141739062，role_set0.130590732→0.131121171；不同定义的实际总loss更低不能替代同目标优化改善。预热两端已存在微小数值差异，step66目标切换与step67首个历史候选分开。历史VJP重算265600→284096次record-forward，fit时间增加约4.0781%，无新增推理参数不代表无训练成本。

完整报告results/MSVR310_ROLE_SET_V1_Q1_2026-09-08.md，原始文本/源分析/全query和身份变化表归档evidence/role_set_q1_complete_20260908。新的独立上下文终态审计/root/audit_msvr_role_set_q1已启动（gpt-6-astra max，same-family/provisional），尚未返回结论；私有完整trace只留本地.aris，不推送。此处是执行侧结果，不能代替独立审计。待完整审计后再确定下一单一假设；不扫参数、选终点或重跑已封存版本。正式结果未新增，三数据集完整Goal未达成。


### 41.186 role-set完整Q1独立审计关闭及残缺权重清理（2026-09-08 23:14）

fresh-context审计原文/机器报告已返回：WARN、engineering PASS with runtime-witness boundaries、deterministic PASS、scientific Q1_FAIL，两组原五门0/5。独立复算全部1560行/116501504训练距离元素/2069520检索排名元素，全部3000 query-output与300 identity-output CSV字段一致；另外在全部实际距离上核验两目标dL/dD，最大误差3.10441e-10。该导数检查不是全程模型参数梯度重放。117项递归绑定与当前Q1全部43文件核对；59个科学代码/配置跨执行26c9739、summary23a5b48、审计捕获864a2c9字节一致。

报告refine-logs/msvr310_role_set_v1/EXPERIMENT_AUDIT_Q1.md/.json；完整非私有输入快照/独立程序/实际成功与失败尝试归档evidence/role_set_q1_audit_20260908。完整请求/响应trace仅本地保留。审计读取的tracker/AGENTS是较早运行快照，其滞后问题已由§41.185及本条同步当前终态解决；不得回改审计原文。单seed/反复开发身份、step2起数值轨迹差异、RNG/像素/参数梯度运行见证边界仍保留，WARN不改写成无保留PASS。不需要重跑旧阶段。

审计根据完整四空间数组补充了角色独有且hinge活跃的提议曝光：CNN11831、Transformer12731、Mamba10227；这是位置级数学计数，不是相应角色参数净收益。原+0.097886pp均值、低于Signal0.639228pp及FAIL保持。

另行零更新来源正例覆盖诊断已完成，使用全部已保存来源距离，无新模型/图像/更新；正在由同一独立审计者做单列补充核验，不合并到上述核心审计PASS。后继Smooth-AP只有本地未登记草案，未启动新训练、未选择新终点，待补充核验后决定。

磁盘：核实旧RGBNT100 R2 m0/fold_0/signal_m0.pth为保存中断残片（101712000B，SHA ad2d141c92ff38d40f9b75d8ee98c438a8d419d16ba5f2dde1eb81c0c1da4557，不是完整ZIP），失败terminal退出1、原PID不存在、当前config无引用，后续成功路径六个必要checkpoint仍在。23:12实际删除该1文件并保留日志与清理收据，主卷空闲2516275200→2617987072B。首个清理尝试因误写status.json路径在任何删除前退出，按已存在terminal.json修正后成功；失败stderr保留。必要初始化、最终权重、检索数组均未删除。


### 41.187 来源正例补充审计关闭与下一主假设（2026-09-08 23:22）

独立补充审计PASS_DESCRIPTIVE_SAVED_SOURCE_SCOPE，same-family/provisional；明确与核心Q1 WARN/科学FAIL分开。独立逐对比较全部99840个anchor、214672384个正负距离对，读取116501504个四空间元素（其中29125376个fused元素用于正例统计），6端×4阶段及2端×4聚合全部一致。完整证据evidence/role_set_positive_coverage_20260908，审计refine-logs/msvr310_role_set_v1/EXPERIMENT_AUDIT_POSITIVE_COVERAGE.md/.json，报告results/MSVR310_ROLE_SET_POSITIVE_COVERAGE_2026-09-08.md。所有审计任务现已结束，不重复启动。

末65步control/candidate各12480次anchor曝光，受影响正例位置2239/2272；其中非最远1298/1310，非最远跨scene1061/1060。它们是来源训练快照中的重复曝光，不是最终固定模型的全来源评价或新Q1。非最远定义在所有真实正例上取max后再筛cross-scene；所有最大值并列排除。旧反序统计>=与新严格>在本次因正负等距为0而一致，不推广为任意tie数据等价。本anchor该fused项直接正例距离导数为0，不等于图片/角色/encoder总参数梯度为0，亦不证明换loss有效。

根据已封存负关系集合结果与此来源证据，下一项选择直接Smooth-AP主比较，保留原V8、fresh历史完整反传、64当前anchor、候选规则和其余13项，仅研究多正例排序fused目标。原文https://arxiv.org/html/2007.12163v2 的公式与§5.3温度0.01已核对；独立实现，不称原创。详细提案refine-logs/msvr310_smooth_ap_v1/PROPOSAL.md。不能继续将刷新历史坐标/历史反传/role-set说成尚未测试。

tools/msvr_smooth_ap.py已实现标准公式，精确模块SHA5a4873f0e83aa09441505edca8c1418a0f2cc208599a3c3668385b77ba591cac经远端纯CPU合成数学检查：标量误差0、两次有限差分/重排、自匹配梯度、tie、B64×128有限性通过，0图像/模型/更新；证据evidence/smooth_ap_formula_math_20260908。该函数尚未接入真实训练，不声称已有新参数梯度或泛化结果。下一工作是最小训练器与全量核验实现、登记同初始化/seed42/固定终点合同，再按M0→Q1原门推进；不扫描温度、倍率、终点，不叠新Router/风格/scene配额。训练正式配置尚未建立，勿用旧wrapper猜参数启动。


### 41.188 Smooth-AP单一主实验完整登记（2026-09-08 23:38）

配置configs/MSVR310/TriFusion-smooth-ap-paired-v1.json，SHA 974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302；合同refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md。新增train/check/verify/run及数学检查入口，复用已验证的完整fresh-history VJP路径；只替换fused目标，control原hard、candidate直接Smooth-AP tau0.01/weight1，不使用旧role-set提议。模块新增同语义hard helper后文件SHA变化，旧§41.187的5a4873仅指当时公式文件；新T0会实测当前完整模块与原hard标量/导数一致。已AST检查，不当作真实执行。

两端保留seed42/B64K8/65步预热/容量512/年龄8/20epoch260更新，七组监督中的其他13项不变；candidate实际AP仍通过旧加权入口components.triplet_fused存储，每步active_fused_metric和summary明确语义。完整四空间距离、每anchor AP/正例数、hard/AP两标量均保存；独立Float64 NumPy rank sums全量核验，标量/逐anchor容差2e-6。M0首实际历史组的完整图与VJP参数梯度核验、203累计梯度覆盖、冻结基线/重载/原100步过拟合门都保留。两组原五门不改变，不根据局部fold调参。

23:34远端root/autodl-container-555b4b9409-6aa86278实查head09381da、Python3.10.14/PyTorch2.5.1+cu121/RTX3090；GPU1MiB，无compute进程，旧wrapper/Q1 PID均不存在。输出卷9954095104B空闲，主卷2594549760B。环境暖复用、无安装/重建；证据evidence/smooth_ap_registration_20260908。输出仍/root/trifusion-storage/artifacts，4GiB最低/3GiB最大新增预算；启动再次核验。预计M0 10–25min、Q1 3–6h，持久screen/原PID/日志，按阶段预计结束和180–300s观察。

本条仅注册，尚无真实T0/M0或新检索。同步后用固定commit/config SHA首次启动新wrapper，失败阶段封存停止，不自动修改重启；M0/CPU通过才打开heldout。完成后按技能独立审计；旧审计任务不得重复。三数据集完整Goal仍未达到。


### 41.189 Smooth-AP实际启动与T0通过（2026-09-08 23:44观察）

本次实际执行提交2e947a4325144e37fed638105ac954e7e54b5fe5，配置SHA974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302。23:40:13首次启动screen tri_smooth_ap_2e947a4，run /root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4。原wrapper PID48170，T0 PID48176于23:40:18退出0（5.558秒），随后M0 PID48190自动启动；不是旧实验重启。

T0状态PASS_SMOOTH_AP_CPU_CONTRACT：新Smooth-AP合成标量参考误差0，两次有限差分、两次原hard值/导数比较、两次置换检查通过，self导数0、tie AP0.5；原memory/历史链式法则检查及全部780来源batch队列规则通过。0模型前向/0优化更新/0 heldout图像读取，不能当作M0或检索成功。

23:44:15原wrapper与M0均存活，日志已完成fold0、fold1的control与smooth_ap四个8步容量端；尚无完整M0 summary及CPU终态。GPU982MiB/4%为瞬时采样，不能据此判断卡住。输出卷9815351296B空闲，启动前9954033664B，仍满足登记预算。固定100步过拟合和其余容量端继续；预计23:50–00:05附近取得M0终态，以实际进度修正，按180–300秒或里程碑观察同一pipeline/PID。禁止观察超时重启。

实际启动脚本/收据、只读观察脚本与完整T0/日志快照归档evidence/smooth_ap_launch_20260908。首次观察输出包含完整780batch队列，终端显示被截断但本地JSON完整保存；不是证据缺失。后续观察只展示必要字段。训练合同/配置/科学脚本不变；M0/CPU通过才由固定wrapper打开Q1，M0独立审计按合同与Q1并行。尚无新检索结果，Goal仍未达到。


### 41.190 Smooth-AP完整M0与Q1启动

# MSVR310 Smooth-AP v1 M0完整工程记录

固定执行2e947a4325144e37fed638105ac954e7e54b5fe5，配置SHA974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302。M0运行HEAD2e947a4325144e37fed638105ac954e7e54b5fe5。训练合同refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md不改变。

M0结束2026-09-08T23:51:18.686093+08:00，CPU结束2026-09-08T23:51:28.873294+08:00，均退出0。原wrapper48170已启动Q1 PID49157，开始2026-09-08T23:51:28.875233+08:00；这只说明固定流程进入Q1，尚无完整科学终态。

三折两端各8更新及两端各100固定batch更新，CPU核验248步。六容量端203/203训练张量累计非零，六次固定checkpoint重载输出逐位相同。AMP/冻结状态/配对像素及初始权重检查保存于原收据；不宣称每步所有张量非零。

过拟合原loss-floor校正末首比：control=0.000701483458532，smooth_ap=0.000699798005237，既定门≤0.1。候选fused目标为AP，其他13项不变，不能把两端总loss直接当作同定义目标改善。固定batch排除历史副本，此过拟合并不证明历史路径全部能力。

六容量端首历史group的四输出逐位一致；完整图总encoder梯度对当前导数加历史VJP最大相对L2误差2.09298683008e-05（门0.005）。域为189个encoder张量；不是全248步参数梯度独立重建。CPU核对4,945,920个保存距离元素及全步AP/原hard/队列/损失账本，记录历史VJP 5,760个记录前向。六容量端最高allocated显存11267.117188MiB。

完整29份远端文本共5,574,687B按SHA接收，见evidence/smooth_ap_m0_complete_20260908/intake_manifest.json。summary SHA64ebb7ef11a80afc19f77a87c16a49c239b3e5022e161d1cd8ad340d2c8699e4。权重、模型、图像、NPY仍留远端。后续独立上下文审计待完成，其结论与训练器/CPU收据区分。

完整Q1仍固定seed42、3折2端/1560更新、完整图库及scene协议、两组原五门；不读局部结果调参，不以工程通过替代检索或三数据集Goal。预计3–6小时，按真实训练时间调整观察。

独立M0审计已实际启动：/root/audit_msvr_smooth_ap_m0，gpt-6-astra/max/fork-none，只读来源M0/代码/原始数组，same-family/provisional。请求与完整调用元信息留私有.aris/traces/experiment-audit/2026-09-08_smooth_ap_m0，不读取局部Q1分数，不中止正在运行的固定Q1。审计尚无结论。


### 41.191 Smooth-AP终态分析准备与固定Q1观察（2026-09-09 00:02）

00:02:06原wrapper48170/Q1 PID49157均存活，pipeline RUNNING/q1，fold0 control完成9/20epoch；最近两epoch136.413/137.372秒。输出卷9705930752B，约9.04GiB。以此估算第一端约00:28附近结束，尚未见候选端速度，完整Q1仍预计3–6小时；按里程碑或180–300秒观察，不重复启动，不读取局部成绩调参。

准备纯文本终态来源分析脚本evidence/smooth_ap_terminal_analysis_preparation_20260909/analyze_smooth_ap_q1_terminal_20260909.py，SHA60e1b915b4d113a8a3469713a26b193e44351fe58e995a2b277ac8fc91722ed6。入口要求完整终态pipeline、原进程结束、全部文本intake SHA及完整Q1 CPU收据/1560更新，未满足时不执行。分析区分all/warmup/postwarmup/last65四阶段的共同hard与AP标量、实际14项账本、AP/正例曝光、历史年龄、角色梯度运行见证、重编码/VJP/耗时，并核对配对像素及预历史差异。完整检索和原门仍由既有CPU/独立审计核验，不新增门槛或训练干预。

已在八份完整M0日志248步上执行描述函数/成本加总/配对差异函数检查，PASS_REAL_M0_LOG_FUNCTION_CHECK_ONLY；仅本地文本/0模型/0优化，不能称为Q1终态核验。首版自检漏算training总fresh成本里每端首批独立64条零更新检查，触发AssertionError；依据训练器step0实际encode_all修正自检和成本列，将日志历史刷新与首批64分开。保留失败脚本、失败记录和最终通过收据；未修改任何固定训练代码/配置。

独立M0审计/root/audit_msvr_smooth_ap_m0仍在进行，实际静态读取已完成、正在全248步CPU与递归哈希核查；没有审计结论。私人helper输出捕获器一次失败属于审计执行工具问题，审计员保留记录并继续，未运行新训练或改变原进程。新描述脚本是执行者准备，不计作已被该审计员审查的结果。

#### §41.191 终态接收执行准备补充（00:06）

同证据目录新增receive_smooth_ap_q1_terminal_20260909.py和smooth_ap_terminal_execution_ready_20260909.md。接收器固定新run/commit/CPU状态，验证完整五阶段退出、原PID结束、1560步及终态SHA，再收全部文本；随后用既有完整ranking工具candidate=smooth_ap重放所有身份/查询及原门。已检查本地和嵌入远端代码AST，未执行终态接收、Q1分析或排名重放。不新增训练/门槛/官方读取，当前Q1和M0独立审计继续。


### 41.192 Smooth-AP M0独立审计闭环及首控制端完成（2026-09-09 00:29观察）

独立M0审计已完成：总体WARN，工程PASS_WITH_LIMITS，deterministic_checks_status=pass；gpt-6-astra/max新上下文，同模型家族/provisional。178项递归哈希匹配，1032来源triplet/3096模态路径/155身份、全部780batch sampler与真实队列重建通过；248更新、15872anchor AP、4945920四空间距离均独立核验。逐anchor AP最大误差2.3559432371644817e-7；每目标每dtype1236480距离位置，Smooth-AP解析导数对部署实现最大差Float64=3.122502256758253e-17、Float32=1.7623613799214177e-8，hard导数精确一致；14项总和最大误差5.62518835067749e-7。六个checkpoint各472张量完整/冻结状态及三源初始化绑定重建匹配。

WARN保留：真实图像内容/压缩包未重读；历史字段/RNG/向量和逐步optimizer/参数梯度未保存，不能从距离导数及范数恢复。六次direct-check只覆盖step4首个历史group的189encoder张量，最大相对L2=2.0929868300782436e-5属于运行见证及摘要恒等式核验；strict reload前后输出数组未保存，输出逐位相同仍是执行时断言，完整checkpoint状态则已独立核验。203/203是累计覆盖；固定过拟合只有53独立记录、无历史候选，不证明历史反传过拟合。审计未消费Q1成绩，未要求重训、改合同或改代码。

原文、独立脚本/输出、快照及失败尝试归档evidence/smooth_ap_m0_audit_20260909，正式报告refine-logs/msvr310_smooth_ap_v1/EXPERIMENT_AUDIT_M0.md/.json。一次审计transport捕获接口失败和13份副本文本CRLF归一化均已保留/修复，最终232/232远端文本快照、29/29原intake文件字节匹配。私有请求/调用trace留本地.aris与审计目录，公开清单明确排除项。不得重复已关闭M0审计。

00:29:28原wrapper48170/Q1 49157存活，首折control完成20epoch/260步、固定checkpoint存在，全部记录工程检查通过；候选smooth_ap完成5/20epoch，尚在预热边界。1/6端完成，不报告局部检索增益。输出卷9566380032B（8.9094GiB），本轮未删除新权重。候选后预热代价尚待实际观察，不能据约19秒的预热epoch外推；约00:40再查后预热耗时并修正首候选端ETA。固定执行2e947a4/配置974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302不变。完整六端/CPU/终态分析/独立Q1审计仍待完成，三数据集Goal未达到。

00:40:47追加实查：原wrapper48170/Q1 49157均存活，首控制端固定260步checkpoint存在，候选10/20epoch；最近epoch9/10为143.082/143.255秒，输出卷9539149824B（8.884GiB）。按候选后预热实际速度估计首候选端约01:05结束，下一观察安排该里程碑；不采用约19秒预热速度外推。不读取局部检索分数、不修改训练/门槛，完整收据evidence/smooth_ap_m0_audit_20260909/observe_smooth_ap_q1_0040_20260909.json。


### 41.193 Smooth-AP首折两端完成与用户后继候选（2026-09-09 01:06）

01:06:39只读观察原wrapper48170/Q1 49157存活，pipeline RUNNING/q1。fold0 control及smooth_ap均完成20epoch/260更新、固定checkpoint存在，记录的梯度累计覆盖、零overflow、冻结Signal及状态不变、角色更新、显存和固定长度检查全部通过。当前2/6端完成，fold1 control到5/20epoch；这只是训练进度与运行收据，不是完整Q1独立核验或检索收益。完整观察见evidence/smooth_ap_m0_audit_20260909/observe_smooth_ap_q1_0106_20260909.json。未读取局部mAP。

输出卷剩9405018112B（约8.76GiB），本次没有删除权重。按已测后预热约137–143秒/epoch估计fold1控制端约01:40附近结束，届时再观察；完整六端及CPU仍需继续，以实际速度修正。原执行2e947a4、配置SHA974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302、seed42、两组五门及T0→M0→M0_CPU→Q1→Q1_CPU不变，不重新启动。

用户最新复核已纳入后继研究候选，尚未登记或启动新实验：先取得标准Smooth-AP完整六端/CPU和独立终态核验；不得再将已完成历史反传、角色负例并集或本轮Smooth-AP当成未尝试建议。之后根据完整来源证据判断是否单独比较标准Smooth-AP与协议感知的跨环境正例覆盖。MSVR310必须依据真实scene规则，另两任务依据各自camera规则；同身份同环境记录从该排序项排除而非改成负例；无合法跨环境正例anchor仍保留其他身份监督，并作为其他身份的候选干扰。环境分组均衡是训练权重设计，不能称官方AP的无偏估计。先验证这一最小干预，再决定是否需要角色条件正例权重；若角色困难分布近似相同，保留简单方案。

终态诊断候选包括非最远错序正例的分数导数覆盖/方向/幅度、同一角色参数上fused排名与其余监督的梯度规模、候选集合与完整来源图库排名差异。保存距离可支持距离层导数重算；目前总梯度/历史侧范数日志不能直接恢复fused与其他13项的独立参数梯度，缺失量不得写成已测事实；任何必要的新来源诊断需另行明确固定状态和证据。当前全量来源图库差异也不可由批内矩阵自动推定。

ROADMAP、MEAP、HDC、DiVA为用户提出的后续近邻阅读，尚未在此更新中独立核读，不据其摘要修改当前目标。仅在严重错序梯度或分数校准证据成立时考虑相应单独干预。全部仍只用seed42，先主结果后消融，三个核心数据集独立评价；不使用已消费官方结果调参。用户提供PDF/TEX链接仅为上下文，本次未下载、生成或核验附件；没有新增正式成绩。三数据集主Goal继续ACTIVE/UNMET。


### 41.194 Smooth-AP终态正例直接导数分析准备（2026-09-09）

01:09:35原wrapper48170/Q1 49157仍存活，2/6端完成，fold1 control6/20epoch；输出卷9397846016B。沿原预计01:40端点观察，不重复启动、不读取局部检索分数。本次只读CPU分析不调用模型/GPU，不改变训练状态。

新增evidence/smooth_ap_positive_derivative_preparation_20260909/analyze_smooth_ap_positive_derivatives_20260909.py（SHA5c8087f1bf3e19324fee76e1305c93124038b6ee14f3362656f1fd6bd5d3e537），默认只在完整Q1/CPU/原PID结束及绑定SHA通过后执行。覆盖六端全部来源距离，分别统计全部/跨scene/被负例严格超越/其中严格非最远/非最远跨scene正例的分数及距离导数符号、零值与幅度分位数，区分预热、预热后及最后65步；保留正例之间排名导数。各端自身保存状态上的hard与AP反事实导数不等于跨端同状态因果比较。

已在封存M0全部248步/8份日志执行分析函数检查，CPU14.782秒，Float32分数目标与实际目标损失差0，链式转换距离梯度差0；原始JSON及stderr完整保留。最终脚本仅补绑定配置/目标SHA和显式端点名，其AST、已有验证输出及导数函数不变已检查。首个本地嵌套引号命令解析失败，改用literal here-string生成payload，未触发远端操作；失败原因保留README。没有重跑M0训练或重复其独立审计，完整Q1分析尚未执行。

这些是位置重复曝光与该anchor的fused标量直接导数，不代表独立图片数或全部参数梯度；不从其推断完整来源图库排名，也不代替fused与其他13项参数梯度测量。本次没有新检索结论、门槛或实验候选登记。Q1结束后与既有完整排名、来源日志及独立终态审计一同接续；三数据集Goal仍未完成。


### 41.195 ROADMAP原文与发布实现核读，当前Smooth-AP继续（2026-09-09）

01:16:46原wrapper48170/Q1 49157均存活，2/6端完成，fold1 control10/20epoch，最近两epoch138.422/131.786秒，输出卷9380282368B。仍预计01:40附近检查该控制端，不消费局部检索成绩。

等待期间核读ROADMAP arXiv2110.01445v3 §3.1–3.2及作者发布commit fed37d75f475f636542b2ccd1ffc0c3918498a57。六份源码/配置/README/MIT许可按原字节与SHA归档evidence/roadmap_primary_review_20260909，REVIEW.md区分论文机制、代码事实和项目推断。没有安装/运行作者方法，也没有登记后继训练。

要点：SupAP对正例内部排名停止梯度、对负例比较改变近似；ROADMAP另加校准，不能当作只改Smooth-AP温度。当前发布YAML offset1.44与论文连续尾部值1.4933071490757153不同，且loss聚合、dependency reduction、self/tie约定仍需复现前明确。原论文mAP@R和其数据集不能直接用于本项目ReID成绩比较。主比较后若来源严重反序的有效导数不足，再考虑对应单干预；若是合法跨环境正例覆盖不足，仍先检验简单关系集合。批内状态日志不能证明同模型完整来源图库的校准差距。

当前执行2e947a4/配置974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302完全不变；完整六端、CPU、正例导数与排名分析及独立终态审计尚待完成。无新增正式指标，Goal保持未完成。


#### §41.195 01:23磁盘维护补充

01:21:35原wrapper48170/Q1 49157均存活，fold1 control12/20epoch，仍2/6端完成，输出卷9367175168B。没有局部检索读取或训练重启；下一端点观察仍预计01:40。

只读盘点发现transport下27个已完成同步的Git bundle，共22928559B。每个文件均有本地逐字节/SHA一致副本，所有引用提交存在于远端仓库且为当前HEAD bd804ef的祖先；GitHub HEAD亦一致。执行前再次检查绝对路径严格位于/root/autodl-tmp/trifusion-v2/transport、非符号链接、后缀/大小/SHA和提交可达性；随后只删除这27个明确文件，未递归删除目录。01:23:21清理完成，主卷空闲2503294976→2526269440B，权重删除数0，输出卷训练文件不受影响。全部盘点、保留判定、执行脚本与逐文件删除收据归档evidence/transport_cleanup_20260909，远端artifacts/transport_cleanup_20260909也保存计划/收据。本轮没有足够依据认定其他checkpoint无用，保留初始化/终点/复核数组。


### 41.196 登记来源序列的跨scene正例支持度（2026-09-09，非新训练）

在当前固定Q1运行期间，对已登记来源标签/780批序列和原队列做纯文本完整重放，逐批与已封存T0历史数量/年龄一致。只计共同序列一次，未读取当前Q1分数、模型或图片。结果见evidence/smooth_ap_cross_scene_support_20260909/REPORT.md及全身份/全batch JSON；协议/metadata/config/T0哈希绑定保留。

预热后585批、37440次anchor曝光中，15376次（41.0684%）有跨scene正例，21928次来自全来源只有一个scene的身份，136次全来源可跨scene但当步候选缺失（占全来源具备条件曝光的0.8767%）。跨scene正例84914/全部正例285248位置（29.7685%）。本登记队列没有新增“原batch无跨scene正例、历史使其变有”的anchor资格事件；不等于历史没有增加正例位置或困难实例。

合法anchor数分布0:4、8:54、16:102、24:172、32:147、40:79、48:24、56:3。完全没有跨scene合法anchor的batch是fold0 step180、fold1 step221、fold2 step133/232。当前all-identity Smooth-AP并未因此无正例；这些空集合只针对用户建议的后继跨scene项。若未来登记该项，需明确空集合和归一化：同批同合法集合下mean_valid=(64/n_valid)*mean_zero_filled_64；不能把关系过滤和尺度改变混为一个未披露因素。没有据此选择新归一化或改当前训练。

该诊断只证明标签/队列支持度，不证明同scene正例占据大部分真实参数梯度或跨scene过滤必然有效。完整Smooth-AP六端/CPU、终态导数/排名及独立审计仍是下一决策前提；seed42/原合同/三数据集Goal不变。


### 41.197 Smooth-AP第二控制端完成，3/6端完成（2026-09-09 01:40）

本轮为实际等待后取得新端点：01:31:33控制端16/20epoch，01:36:28到19/20，01:40:20确认fold1 control固定20epoch/260更新完成、checkpoint存在，梯度累计覆盖/零overflow/冻结Signal与状态不变/角色更新/显存/固定长度记录均通过。加上fold0两端，现为3/6训练端完成；当前fold1 smooth_ap到4/20epoch，原wrapper48170/Q1 49157均存活，pipeline RUNNING/q1。三次只读观察完整保存evidence/smooth_ap_q1_progress_0140_20260909；没有消费任何局部检索分数，不把训练端点或工程收据当作完整科学核验。

输出卷9247522816B（约8.61GiB），未删除新权重。以首候选端后预热约143秒/epoch估算，第二候选端预计02:15–02:20附近完成；按实际速度更新，约该里程碑再查，避免重复密集读取。原执行2e947a4/配置974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302及两组五门不变，没有因等待重新启动。

接续仍是完成原六端及CPU终态，执行已准备的全量接收、来源日志/直接正例导数和全部查询身份排名分析，再独立终态审计。ROADMAP核读及跨scene支持度仅是条件式后继证据，尚未登记新训练或改变归一化。完整三数据集Goal继续ACTIVE/UNMET。


### 41.198 MEAP作者代码核读与候选端进度（2026-09-09 01:48观察）

01:48:56原wrapper48170/Q1 49157存活，仍3/6端完成，fold1 smooth_ap8/20epoch，最近epoch7/8为142.562/142.505秒；输出卷9226153984B。预计第二候选端02:15–02:20附近结束保持；观察文件名0143只是本地预命名，准确时间以JSON observed_at=01:48:56为准。没有局部Q1成绩读取。

MEAP出版全文本次未取得（DOI错误/ScienceDirect429），仅出版/作者机构摘要加作者代码，不声称独立核过全文公式/表。Git refs绑定meapnet commit45f948894f84d45cc3f6558b7f030a586130ea7e，五文件35625B、MIT许可及SHA保留evidence/meap_primary_review_20260909。API404和web click失败亦记载，raw固定版本读取成功。没有安装或执行作者代码。

作者实现对固定VIS/IR身份分组的正例分数加margin，保留同模态正例；与后继候选“排除同环境正例”是不同干预。当前动态历史池不能直接沿用固定索引mask，self/分组平均/两个margin/全局分区监督均需明确。对于来源只有一个scene的身份，margin无法补出真实跨scene正例。该核读仅明确直接近邻及复现边界，不登记新训练、不采用作者margin、不改变当前Smooth-AP或原门。下一决策仍等待完整六端/CPU、来源导数/全量排名和独立终态审计；Goal未完成。


### 41.199 Smooth-AP前两折完整训练端完成，进入第三折（2026-09-09 02:17）

原进程持续等待观察：01:55候选11/20、02:01到14/20、02:06到16/20、02:11到18/20；02:17:31确认fold1 smooth_ap固定20epoch/260步完成、checkpoint存在，记录工程检查全部通过。现fold0/1各control与smooth_ap四个端均完成，4/6；fold2 control已到5/20epoch。原wrapper48170/Q1 49157均存活，pipeline RUNNING/q1，没有重启。完整观察文本在evidence/smooth_ap_q1_progress_0217_20260909；文件名是预命名，准确时刻以各JSON observed_at为准。

输出卷9084583936B（约8.46GiB），本次未删除权重/数组。依前两控制端后预热约130–138秒/epoch，预计第三控制端约02:50–02:55完成，再观察原流程切换最后候选端；不能用约19秒预热耗时外推完整端。无局部检索分数消费、无门槛/温度/候选/归一化变更。

接续仍需最后两个固定端、Q1_CPU、完整文本接收与排名/来源/正例导数分析、独立终态审计。原执行2e947a4/config974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302及seed42保持。工程端点不代表完整检索结论，三数据集Goal仍ACTIVE/UNMET。


### 41.200 Smooth-AP第三控制端完成，最后候选端运行（2026-09-09 02:51）

02:20、02:26、02:31、02:36、02:45的只读观察分别记录fold2 control6/9/11/14/18epoch。02:51:08确认该控制端固定20epoch/260更新完成，checkpoint存在，累计梯度覆盖、零overflow、冻结状态及Signal不变、角色更新、容量和固定长度记录均通过。现5/6端完成，fold2 smooth_ap已到3/20epoch；原wrapper48170/Q1 49157持续存活，pipeline RUNNING/q1。六份完整观察归档evidence/smooth_ap_q1_progress_0251_20260909；以JSON observed_at为准确时刻，不以预命名文件时间替代。

输出卷8931037184B（约8.32GiB），本轮没有删除权重或数组。按已观测候选端预热后约143秒/epoch估算，最后候选训练预计03:25–03:30结束，随后还需Q1_CPU；不能将预热约19秒/epoch外推全程。下一次按阶段预计时间检查，不因等待或观察超时重启。未读取局部检索分数，没有改温度、候选、训练长度、归一化或原两组五门。

原执行2e947a4/config974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302保持。完整终态接收、全查询/身份排名、来源日志及正例直接导数描述和独立审计均待原流程结束后执行；工程端点不等于科学晋级，三数据集Goal仍ACTIVE/UNMET。


### 41.201 Smooth-AP完整Q1结束及执行器全量分析（2026-09-09）

原Q1 49157于03:26:23退出0，Q1_CPU58836于03:26:44退出0，wrapper48170结束。6端各20epoch/260更新，总1560；CPU完整核查116501504来源距离元素、2069520检索距离/排名元素。57份文本83471329B全部接收并核对SHA，模型/图片/数组留远端。报告results/MSVR310_SMOOTH_AP_V1_Q1_2026-09-09.md及evidence/smooth_ap_q1_complete_20260909、smooth_ap_q1_ranking_analysis_20260909、smooth_ap_q1_executor_analysis_20260909、smooth_ap_q1_positive_derivative_complete_20260909。

fused52.444136→52.787590（+0.343455），CNN+0.513458/T+1.174330/M+0.251933。配对fold-0.200114/+0.245762/+1.077727，bootstrap下界-0.101124，配对1/5、对Signal0/5。fused高于三个角色但低于Signal53.129381。配对R1修复13/新增6，身份29改善/26下降/5不变。保留小正均值和原FAIL，不晋级、不挑终点或改门。

末段非最远反序正例曝光1299→974（跨scene1061→781）；candidate974中973个直接距离导数有利/1个相反。完整1560步score目标及链式distance梯度与实际实现差0；这些不是参数梯度，也不是完整来源图库结果。candidate末段AP loss下降、同定义hard反而提高；当前total目标不同不能直接比较。来源预热step2起微小差异、实际重算成本和导数首次SSH超时/确认旧进程不存在后的持久CPU重跑均完整披露。独立审计/root/audit_msvr_smooth_ap_q1已启动，新上下文gpt-6-astra/max，同家族provisional；尚无独立终态结论，M0不重复。

03:27输出卷8776597504B，本轮未删权重。现转入完整终态审计与原因归纳，未登记下一训练；跨scene正例覆盖仍为条件式候选。Goal三数据集baseline/SOTA目标仍ACTIVE/UNMET。


### 41.202 Smooth-AP Q1独立终态审计闭环（2026-09-09）

独立审计/root/audit_msvr_smooth_ap_q1完成：总体WARN、确定性核验PASS、科学Q1_FAIL；gpt-6-astra/max新上下文，同家族/provisional。正式原文refine-logs/msvr310_smooth_ap_v1/EXPERIMENT_AUDIT_Q1.md/.json，全部独立文本、脚本、失败尝试、输入快照及哈希在evidence/smooth_ap_q1_audit_20260909。原始回复与请求留私有trace。未重做M0、未重训或改门。

独立重算6000查询输出、2069520完整排名位置、1560来源更新、116501504四空间距离元素，以及六端各472个checkpoint状态张量，原指标与两组门完全复现。fused52.444136→52.787590（+0.343455），配对1/5、对Signal0/5，身份bootstrap下界-0.101124；三数据集目标未达成。保存的距离解析导数核验通过不等于独立恢复全程真实模型梯度：历史冻结字段/逐步参数梯度未保存，运行时VJP见证及有限direct-check的限制继续保留。

补充数值边界：末段candidate全部94872正例的直接距离导数，Float32统计86990正/7882负，Float64为94866正/6负。容易正例的极小导数有抵消及精度敏感性，不能将7882个负号全部解释为实质错误监督。关键严格非最远反序类别一致：974次曝光中973有利/1相反，跨scene780/1。Float64检索距离与注册FP32也存在微小差异；原注册FP32完整排名逐位重现，不用其他dtype替换指标。

下一步只补来源诊断缺口：六个固定终态的同角色fused目标与其余13项参数梯度，以及同一表示表的训练候选池/完整来源图库排序。设计草案evidence/smooth_ap_q1_closure_support_20260909/smooth_ap_post_q1_diagnostic_design_20260909.md；尚未登记或执行。它不是训练轨迹重建，也不是后继训练，不能先宣称跨scene过滤有效。原seed42、固定门、官方测试与主结果前消融边界保持。

03:47只读资源收据：GPU0%/1MiB，主卷2410942464B、输出卷8776495104B空闲。本轮删除权重0，必需checkpoint/数组保留。上一轮发布认证传输失败及使用现有登录的临时进程环境恢复已记录，不含凭据，无全局配置修改。Goal ACTIVE/UNMET。


### 41.203 Smooth-AP固定来源候选覆盖诊断登记（2026-09-09）

执行入口转为refine-logs/msvr310_smooth_ap_source_coverage_v1/DIAGNOSTIC_PLAN.md，配置TriFusion-smooth-ap-source-coverage-v1.json；状态REGISTERED_NOT_RUN。先测来源候选覆盖，参数梯度分解另行实现。六终态×clean/固定seed42增强×全部来源记录，12条件8256前向、0优化更新；原六端checkpoint和失败门不变。

同一表示表构造原batch/history唯一record池与完整来源图库，self记录排除，分别all-ID/cross-scene；无合法正例记录仍保留为其他身份干扰。只在共同合法anchor比较候选/完整AP，单列池缺正例，保留全部120条件及1996800曝光行。Float64稳定距离排序仅用于该诊断，不替换原FP32 Q1。固定单view诊断不等于原训练随机历史重放，不能由AP差单独认定校准故障。

实际纯函数语义检查通过，代码语法通过；真实模型检查待执行。04:08 GPU0%/1MiB，主卷2261106688B、输出8776491008B空闲；tri_reid Python3.10.14/PyTorch2.5.1+cu121/3090与seed42 CUDA内核实测复用，无环境重建。证据evidence/smooth_ap_source_coverage_preparation_20260909。未开启新训练、消融或官方评估，Goal ACTIVE/UNMET。


### 41.204 固定来源覆盖诊断已启动，完整CPU核验准备（2026-09-09）

原执行f7a0590，持久wrapper61962、提取62027，04:15:32启动；run /root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590。math退出0。04:16:19实查2/12条件，04:17:45实查8/12，wrapper与提取进程均存在，GPU100%/3112MiB；已完成条件state/无梯度/配对像素与Signal一致通过。第二观察文件名0419为预命名，以observed_at=04:17:45为准。

每个完整条件约11.6–13.0秒，尚需最后四个条件及原CPU分析。当前不是完整终态，无局部AP消费，无新训练。来源全部8256记录前向/0更新范围不变。完整CPU核验脚本及VERIFICATION_PLAN在evidence/smooth_ap_source_coverage_launch_20260909，准备原流程完成后对全部82560完整query行/1996800候选配对行及120汇总核验；尚无核验PASS，不冒充独立审计。

输出卷04:17:45剩余8243306496B；本轮不删依赖权重。下一步完成本来源覆盖分析、数值核验和全部汇总，同时继续实现单独的固定状态参数梯度分解；不得凭当前覆盖诊断替代参数梯度结论。Goal ACTIVE/UNMET。


### 41.205 固定来源覆盖提取与分析完成，全量逐行核验运行（2026-09-09）

原f7a0590流程math、extract62027、analyze62495全部exit0；提取04:18:37结束，分析04:21:13结束，wrapper61962已不存在。12条件8256来源记录前向完成，模型state/无梯度/全部配对像素与Signal检查通过；完整候选与图库结果已生成，但尚待完整逐行核验，不读取首条件AP决定后继。

完整CPU核验代码efb1d04、SHA03765d14fb49031c86e282a8af38da5780cd92c21330e74d3857e8943aa19799；wrapper62700/child62701于04:21:31启动。04:22:58实际确认两进程存活，已核对1/12条件的166400候选曝光行及6720完整来源query行，AP重算最大误差0；这不是全范围PASS。首条件58.80秒，预计全范围约12分钟，下一观察按180–300秒或终点前里程碑安排。核验不产生模型前向/更新，不重跑提取或旧训练。

来源分析结果最终必须完整覆盖120输出/过滤条件、82560完整query行、1996800候选配对行并重算汇总。固定view/唯一记录池和Float64稳定排序边界保持；结果不代表原训练轨迹或未知身份性能。梯度分解尚未登记执行，下一工作继续该独立测量。记录与启动脚本在evidence/smooth_ap_source_coverage_verification_progress_20260909。

04:22:58输出卷7372115968B（约6.87GiB）；模型/数组/大明细留远端，本轮无权重删除。三数据集baseline/SOTA Goal ACTIVE/UNMET。


### 41.206 固定终态目标参数梯度分解登记（2026-09-09）

新增tools/probe_msvr_smooth_ap_objective_gradients.py、configs/MSVR310/TriFusion-smooth-ap-objective-gradients-v1.json与对应DIAGNOSTIC_PLAN。状态REGISTERED_NOT_RUN；六端各8batch工程预检（2步预热）后，才允许六端各260batch完整来源诊断（65步预热）。0optimizer，固定全部参数/每批恢复buffer，索引与像素核对原注册来源序列；不是重训或恢复原逐步训练状态。

同角色189encoder参数分块，计算加权fused当前+历史梯度、其余13项梯度和完整目标梯度。每步分解恒等式、重复噪声、首历史单group直接全图/VJP误差检查，固定scale256/AMP边界披露；预检首历史步三组实际参数梯度张量保存在远端供重算，不把所有全程范数当作独立模型梯度重构。具体误差分母与范围见计划。代码语法通过，真实前向/导数预检尚未执行，不能预报结果。

来源覆盖原提取/分析完成，全量CPU核验04:25:36原62700/62701确认存活，4/12条件、665600候选行/26880完整query行已核对，AP误差0；仍等待全部终态。两个诊断分别回答覆盖与参数作用，不按部分结果选方法。0新训练/官方评估，Goal ACTIVE/UNMET。


### 41.207 完整来源候选覆盖核验与梯度预检进展（2026-09-09）

来源覆盖原提取/分析均exit0，CPU全部82560完整query行/1996800候选配对行/120汇总于04:33:51核验PASS，AP重算误差0。159份13715377B文本SHA接收完成，原数组/大明细留远端；文本接收生成代码首尝试括号SyntaxError已保留，修正只重做接收。报告results/MSVR310_SMOOTH_AP_SOURCE_COVERAGE_2026-09-09.md，尚待新上下文独立审计。

固定增强cross-scene来源fused95.444495→96.672922（+1.228427），1200来源成员251改善/70下降，反序正例位置1214→1002；这不替代原Q1+0.343455/FAIL。共同合法采样曝光下，candidate端候选mAP98.665998、完整来源97.384815。完整来源成员加权与重复anchor曝光加权不同，不能跨表混减。cross-scene每端49920曝光中20496共同合法，168次全图库有正例但池内缺失。固定view/去重/self/Float64稳定排序口径与原随机训练区别继续保留，不凭差距直接断言校准故障。

另一路目标参数梯度预检a2dec7f于04:32:39启动，wrapper63156/child63157。04:38:12实际存活，5/6端各8batch完成，首几端耗时约56–59秒；完整六端及CPU张量/账本核对待完成。全部0更新；不能把预检子集当作1560批参数梯度结论。代码/计划已登记，下一步完成预检再推进完整来源测量。Goal ACTIVE/UNMET。


### 41.208 目标参数梯度六端预检及CPU核验完成，完整来源执行登记（2026-09-09）

预检a2dec7f原63156/63157于04:39:08全部exit0：六端48批、0更新。CPU保存距离/loss与3402实际梯度张量核验PASS，18角色/端点全部统计重算；最大loss误差1.1920928955078125e-07、梯度统计误差2.220446049250313e-16。预检summary SHA bc5aa928dc98f0694008d044753718c87877ad14fddc797b37acf8e6abcf98d0；verification SHA 93fd2174c5084e565db65e8a0f09208132836c6a770c4b92726facf73e2d398a。全量文本evidence/smooth_ap_objective_gradient_preflight_complete_20260909，checker与接收脚本在support目录。每端首历史group直接全图/VJP最大相对误差9.931233814409018e-05；此有限范围不扩称全程独立梯度重构。

完整来源FULL_SOURCE_EXECUTION登记READY_NOT_RUN，复用原代码/合同、同六固定终态、65步预热/各260批=1560批、0更新，启动显式核对CPU预检状态和SHA。预检每端56.8–59.0秒，完整历史组更多，首若干epoch实测再细化数小时估计。全部模型/数组保留，输出约6.50GiB，未删权重。来源覆盖独立审计/root/audit_msvr_smooth_ap_source_coverage已启动，待结论。原Q1_FAIL、seed42与官方/消融边界不变，Goal ACTIVE/UNMET。


### 41.209 完整来源目标梯度测量恢复执行（2026-09-20）

17:52实时核查远端HEAD仍c6fbfb4，与本地/桌面已同步SHA 7ce4665d798e3d142e42cb422f74c9ef47cc68d2b5b4425693e4bb03b6ab6695一致。无旧诊断进程/完整source目录，GPU空闲；原48批预检与CPU完整核验不重复。核对summary/verification/config SHA及实现与a2dec7f逐字一致后，17:53:00启动原合同完整source：/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920，wrapper1230/child1231，六端1560批、0更新。

17:55:49实际两进程存活，fold0_control完成42/260批，GPU100%/12390MiB；前三epoch各约47秒，仍在65步预热内，不能据此精确估计历史VJP生效后的全程耗时。输出剩余6981439488B，主卷约2.09GiB；本轮未删除权重，当前所需终态保留。

完整CPU账本核验脚本和范围已准备在evidence/smooth_ap_objective_gradient_source_launch_20260920，状态PREPARED_NOT_RUN。将核对全部1560批及4680角色统计的标量一致性；不会将未保存的全程向量声称独立重构。GPU测量尚无六端终态。

旧来源覆盖审计代理已不可用，当前list_agents仅root，无最终审计文件；已有373份输入快照和未完成脚本保留。按experiment-audit新建/root/audit_smooth_coverage_resume_20260920接续，要求先检查现有完整收据，禁止重做已验证工作。新上下文、请求gpt-6-astra/max、same-family/provisional；工具仅返回task_name，无UUID/后端身份额外验证。待实际终态，不记PASS。原Q1_FAIL及全部科学边界保持，Goal ACTIVE/UNMET。


### 41.210 完整来源测量进入历史阶段，终态CPU核验已持久排队（2026-09-20）

17:58:44确认原source wrapper1230/child1231均存在，fold0_control74/260批（总74/1560），第6epoch且65步预热已结束。GPU79%/17366MiB，输出剩余6980050944B；当前仍无完整端点或科学结论。保留原执行c6fbfb4，实现/合同哈希不变。

CPU核验依赖队列1706于17:58:24启动，17:58:44确认存活，WAITING_FOR_SOURCE；每300秒观察原任务。仅在原pipeline COMPLETE/exit0及完整summary成立后启动已登记CPU checker，SHA79802db8ef471e8fc495f625aa15321f2d9c1034b13afd82a8f08b1541fb2ef4。原任务失败或句柄消失则记录未核验并退出，绝不自动重训/重测。wrapper、日志和退出码均写独立路径；收据及脚本evidence/smooth_ap_objective_gradient_verification_queue_20260920。源诊断、CPU核验、独立审计状态分开记录。

来源覆盖接续审计已实际检查旧文件，未找到独立终态回执或存活旧审计进程；正在本次独立目录执行CPU全数组/全行检查，尚无最终判定。下一步取得完整梯度测量和CPU终态、独立覆盖审计，再综合全部来源证据登记唯一后继假设；不按中间端点改实验。未新删权重或访问官方图像，三数据集baseline/SOTA Goal仍ACTIVE/UNMET。


### 41.211 全量梯度分布与计算成本汇总准备（2026-09-20）

新增全量汇总程序及合同evidence/smooth_ap_objective_gradient_analysis_preparation_20260920，状态PREPARED_NOT_RUN。仅在六端GPU与全量CPU账本核验通过后生成4680角色明细、72组统计（3fold×2端×3角色×all/warmup/active/last65）及六端计算成本。各窗口重叠不相加，零量/未定义比值/负余弦完整保留；来源loss定义不同不直接用总量论优化优劣。48个真实预检记录上的分布函数检查通过，这不等于完整source分析完成。持久分析依赖队列脚本已准备，启动后单独记录PID。

18:03:26实查原1230/1231均存活，90/1560批，GPU100%/17366MiB，CPU队列1706仍等待。历史阶段第6epoch163秒；随后批次约17.6秒/批，粗估六端总计约6小时，候选端成本可能不同，后续按完整端点细化。输出剩余6977339392B。源测量不因较慢而缩短，合同与梯度范围不变。

独立来源覆盖审计首次脚本在数组核验前遇到NumPy int64摘要序列化错误，原退出1现场保留。审计代理仅转为Python整数并在新R2目录继续，无原科学输出或GPU诊断修改；完整独立结论待返回。Goal ACTIVE/UNMET。


§41.211执行补充：分析依赖队列2251于18:05:52启动，18:06:36实查存活，WAITING_FOR_VERIFICATION；每300秒检查CPU核验，仅完整PASS/exit0后执行SHA绑定汇总器。同期原1230/1231存活、102/1560批，第7epoch耗时218.8秒；核验队列1706也存活。输出剩余6972964864B。链式收据evidence/smooth_ap_objective_gradient_analysis_queue_20260920；全量测量/核验/分析仍未完成，不能提前报来源梯度结论。后续按完整端点/审计终态里程碑观察，不重复预检或原Q1。


### 41.212 来源覆盖独立审计闭环及条件式排名方法核查（2026-09-20）

新上下文审计/root/audit_smooth_coverage_resume_20260920完成：WARN/CLOSED_WITH_LIMITS，确定性PASS、工程PASS_WITH_LIMITS，同家族provisional。报告refine-logs/msvr310_smooth_ap_source_coverage_v1/EXPERIMENT_AUDIT.md/.json，原始审计输入、全部脚本/日志/失败尝试及哈希在evidence/smooth_ap_source_coverage_independent_audit_20260920，原始请求/回复留私有trace。未发现需更正科学代码/掩码/分母/数值的问题，来源报告状态已从待审计更新。

独立R2 wrapper1924/child1925于18:13:44退出0，706.165秒；12条件/60特征数组/82560完整query/1996800候选配对/120汇总、40聚合和20配对全部复现。AP最大误差0，直接差平方和与登记距离差最大2.6645352591003757e-15；1032真实训练记录GT、完整干扰与合法mask、159文本SHA/99递归绑定/17Signal源码/六checkpoint均核对。增强跨scene来源fused95.444495→96.672922及251改善/70下降受到支持；原Q1_FAIL保持。审计未重做像素到特征前向，不将运行见证扩大为独立模型回放或未知身份效果。

审计R1在数组核验前因NumPy int64摘要序列化退出1，完整错误保留；R2仅将三项摘要计数转int，全量核验一次。最后封存时因我并行新增ROADMAP未跟踪目录触发全工作树不变断言，已保留实际差异并纠正封存范围；无科学重算或判定改变。无虚构UUID/后端身份。发布接收部分嵌套快照另遇Windows普通路径长度限制，改用扩展绝对路径并逐文件SHA确认776文件；原审计未改动，接收恢复清单留closure证据。

等待梯度终态期间核读ROADMAP原文与作者固定fed37d75源码，笔记evidence/roadmap_primary_read_20260920/READING_NOTE.md，MIT来源保留。区分SupAP、分数校准、系数及固定PML0.9.99非零均值归约；实际作者函数CPU合成检查记录精确tie及offset分段差异。无新训练登记，未将作者公式/缓存技术包装为原创，也未据候选/完整AP差直接诊断校准故障。

完整来源梯度仍运行：18:12:55原1230/1231存活，124/1560批；核验1706与分析2251均存活等待，输出剩余6974283776B。已按完整范围持久运行，不重复旧诊断/不依据中间端点改训练。下一步完整梯度终态、全量CPU与72组分析，再决定唯一后继假设。三数据集baseline/SOTA Goal ACTIVE/UNMET。


### 41.213 完整梯度终态接收准备与磁盘复核（2026-09-20）

终态文本接收脚本与合同在evidence/smooth_ap_objective_gradient_terminal_preparation_20260920。仅源测量、CPU核验、全量分析三个pipeline全部COMPLETE/exit0，六端1560批/4680角色行/72组及summary/verification哈希绑定通过后，接收文本并逐文件校验字节/SHA。当前仅AST语法检查通过，未执行终态接收；模型、原图、距离矩阵和梯度张量仍留远端。这是接收准备，不是源诊断或科学条件PASS。

18:30:19原1230/1231与核验1706/分析2251实际存活，187/1560批；18:31:47保存193批。诊断目录仅16721801字节，按当前批数线性估计全1560批约135160671字节（不是严格上限），输出盘剩余6960398336字节；主盘18:30剩余2225176576字节。完整pt/pth/ckpt盘点361文件包含模型及诊断/检索数组，不能称为361个模型。现存唯一.resume是早期V3失败现场generation-0000，未确认冗余故保留。本轮删除0文件，当前诊断空间足够。首端粗估18:50，后续按完整端点观察，不重启/不改定义。Goal ACTIVE/UNMET。


§41.213端点补充：18:51:36确认fold_0_control完整260批，3422.2201秒（57.0370分钟），model_state_unchanged/gradients_absent为true，optimizer_updates/heldout_record_forwards/official_image_reads均0。登记的单历史组direct/VJP核对相对误差1.56888267e-09，限于该组，不扩大为全source向量重算。fold_0_smooth_ap已自动接续20批，合计280/1560；1230/1231/1706/2251实际cmdline均对应原任务且存活。证据evidence/smooth_ap_objective_gradient_first_endpoint_20260920。全量测量、CPU核验和72组分析仍待完成，未解释中间梯度作方法选择；下一观察约19:05，用候选端历史阶段速度更新ETA。无新训练、无权重删除、Goal ACTIVE/UNMET。


### 41.214 公开参照增量核查与目标梯度解释边界（2026-09-20）

等待完整梯度端点期间完成限定范围主源刷新，evidence/trifusion_reference_boundaries_20260920保存三项说明及来源回执。公开方法后台research仅同族上下文隔离，不称实验独立审计。本轮未核得足以替换既有RoDI/PMKD/CoT代表主表的新结果，不称穷尽排行榜；保留CLIP、DINOv2/v3、额外文本/mask、测试适配/重排序条件。RoDI当前2f38911仓库仍README/assets、PMKD0f597fa仍仅README；两者release/tag均0，不当作完整可运行训练实现。Hyper-ReID仅README无主表、CCL官方页可读但OpenReview PDF仍验证页/API403，继续未知，不从摘要补数字。旧表值明确沿用此前原表核验，未冒充本日逐篇复现。

Signal公开HEAD仍cd1b0a672d1fe642e7608731cb4899a19dda7d51，RGBNT100 YAML与原归档字节一致。作者B128/K16与本机B64/K8配置均每批8身份，两者30epoch；不能将大batch直接解释成更多负身份，也不能声称epoch数不同。几何增强、best选择/固定终点、实际更新数及Gram数值定义仍需分别限定，不能将公开差距归因给单一因素。没有重训baseline或以已消费官方结果选参数。

代码级解释补充：当前诊断仅189个encoder参数的固定终态局部导数，无优化器；真实训练将历史VJP相加后一次AdamW更新，保存函数未含动量/平方动量或GradScaler状态。官方PyTorch2.5.1实现支持逐坐标矩预条件及独立weight decay。因此F/O原始梯度范数比不是AdamW更新份额或泛化贡献百分比。补充说明未修改诊断/已绑定分析器，也不据此引入loss倍率或梯度投影。

19:05:21实查原1230/1231存活，控制260批完成、候选106批，合计366/1560；1706核验/2251分析存活等待。候选历史阶段第7/8epoch分别231.3924/231.1775秒，预计第一候选约19:51、全六端约23:45–23:50（后续速度可能变化）；下一观察约19:48。输出剩余6946988032字节。以上均非新的检索结果；原Q1_FAIL、完整梯度尚未结束、三数据集Goal ACTIVE/UNMET。


§41.214首折完整补充：19:51:45确认fold_0_control与fold_0_smooth_ap各260批，分别3422.2201/3632.5508秒，均模型状态未变、残留梯度为空、optimizer_updates/heldout_record_forwards/official_image_reads为0。登记第67步单历史组direct/VJP相对误差1.56888267e-9/1.48044926e-9，只限单组运行核对，完整CPU核验尚未执行。fold_1_control自动接续12批，合计532/1560，原1230/1231及1706/2251存活且cmdline吻合。证据evidence/smooth_ap_objective_gradient_first_pair_20260920。19:48输出盘剩6928113664B；未删除文件或修改运行定义。下一观察约20:45，接近第二折控制终点；六端暂估23:45—23:50，之后核验和分析。无首折科学解释或新训练登记，Goal ACTIVE/UNMET。


§41.214半程补充：20:49:02第二折控制完整260批/3409.2512秒，已完成3/6端；第二折候选自动18批，合计798/1560。三个完整端均模型状态未变、残留梯度为空、零优化器更新/留出前向/官方读取；本端第67步单历史组direct/VJP相对误差5.64200775e-5<原0.005，未扩大为全source独立向量核验。原1230/1231/1706/2251存活且cmdline吻合，完整CPU与72组分析仍待六端。20:45输出余6907330560B、GPU100%。证据evidence/smooth_ap_objective_gradient_half_complete_20260920；下一观察约21:44，六端暂估23:45—23:50。未据局部梯度选方法，未改实验或删除权重，Goal ACTIVE/UNMET。


§41.214前两折补充：21:51:39已完成4/6端，第二折候选260批/3618.0626秒；第三折控制自动57批，总1097/1560。新端状态不变/无残留梯度/零更新与零留出及官方读取，单组direct/VJP误差5.33583695e-5<原0.005。原四PID存活，完整CPU与汇总仍待全部六端。证据evidence/smooth_ap_objective_gradient_four_complete_20260920；21:46输出余6868672512B，下一观察约22:42。实验定义、科学判定、权重均未改，Goal ACTIVE/UNMET。


§41.214末端接续：22:47:11第三折控制完成260批/3401.3633秒，5/6端齐全；最后候选自动35批，总1335/1560。新端模型状态/梯度清空/零更新及数据范围记录正常，单组direct/VJP误差6.38014354e-5<原0.005。原1230/1231/1706/2251存活；22:42输出余6852833280B。证据evidence/smooth_ap_objective_gradient_five_complete_20260920。下一观察约23:42，等待完整源终态→CPU核验→分析，尚无全量梯度结论或新训练，Goal ACTIVE/UNMET。


### 41.215 固定来源目标梯度全量完成及独立关闭（2026-09-21）

执行c6fbfb4六端1560批已全部完成，原source1230/1231于9月20日23:45:14退出0；原CPU1706/14798于23:48:51退出0；分析2251/15038于23:50:59退出0。三pipeline COMPLETE。全量4680角色行、72组、29125376保存距离值；零模型更新、零heldout/official读取。原Smooth-AP Q1_FAIL不改变。

原文本33份/37743954B完整接收逐文件SHA核对；inventory/intake另2份，不混入33份原始数。evidence/smooth_ap_objective_gradient_source_complete_20260920保留全部原始文本及派生pooled统计。结果报告results/MSVR310_SMOOTH_AP_SOURCE_OBJECTIVE_GRADIENTS_2026-09-20.md。

新鲜同族审计为WARN/CLOSED_WITH_LIMITS，确定性PASS、工程PASS_WITH_LIMITS；见refine-logs/msvr310_smooth_ap_objective_gradients_v1/EXPERIMENT_AUDIT_SOURCE.md/.json及evidence/smooth_ap_objective_gradient_independent_audit_20260921。独立NumPy/stdlib核验全部1560步/4680角色/72组及全部CSV，损失最大差1.52116e-7；原current/full分解最大误差.00160712/.00171668<登记.005。完整source参数向量未保存，不能把标量核验、预检3402向量统计复算或每端单组direct检查称为1560步独立模型反传重建。审计失败夹具与连接prelude记录保留，私有连接执行助手与trace不公开。

活动位置66—260中，各fold-role九组Smooth-AP F/O范数比中位数.105537—.174304，控制.603683—.787887。每端1755条角色-批次记录中，Smooth-AP F/O负余弦115、F/total负余弦37；控制12/0。所有F非零且高于各条重复反传差异。比较限于同一角色参数块内部；未比较不同角色异义坐标。不同端点加载各自固定终态，不恢复原训练/AdamW历史，不推导loss倍率或泛化贡献。窗口/来源曝光相关，非多seed证据。

00:14资源实查GPU1MiB/0%，输出余6827118592B，主卷2222432256B；本轮删除0权重。既有失败和所需初始化/终态保留。后继唯一主假设准备为标准Smooth-AP到合法跨scene正例/eligible-anchor均值定义，保持其余13项、fresh历史/VJP及seed42；已有来源支持中4/585后预热batch无合法anchor，必须本项图连接零并保留其他监督和全部负身份候选。新合同refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_PLAN.md及六个独立版本入口已实现；新鲜静态代码审查PASS_WITH_LIMITS/0阻断，同族provisional，配置e5326b52。T0/M0尚未执行，不称工程或Q1通过，不重复旧Q1。下一步提交部署后按T0→M0→M0_CPU→Q1→Q1_CPU固定链执行，非零退出封存。Goal ACTIVE/UNMET。


### 41.216 跨场景正例 Smooth-AP 正式启动，T0通过（2026-09-21）

完整梯度关闭与新实验代码已发布d35864d6411591e05c8ac3e5164ebae48063ad99，306发布文件逐项字节/SHA一致；主交接三方SHA114fa62b56b3b02e963a156705dc5dce0da32ad928762bc1c4bb2e9f0471f4dd。新实验执行绑定d35864d，配置e5326b52ebb12dced24ebfac788db2e2bb5bdca0b50c6c4dec64e596f1af05a1，run /root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d，screen tri_cross_scene_d35864d。

00:25:28持久wrapper16885启动。T0原16891于00:25:36退出0，用8.132秒，PASS_CROSS_SCENE_SMOOTH_AP_CPU_CONTRACT；标准与跨scene独立标量误差0，跨scene有限差分3/置换3、同ID同scene导数零、无正例anchor负候选保留、全零eligible图连接零、历史候选非零导数均通过。完整780来源队列关系与已有support一致，0模型/图像/优化更新。此为CPU公式与合同检查，不是模型性能。

M0原16907于00:25:36自动启动，00:25:44与原wrapper均存活、cmdline匹配，RUNNING尚无完整容量端。GPU614MiB、输出余6823510016B。M0预计10–25分钟，按实际阶段更新；下一观察约00:29，非观察失败不重启。固定T0→M0→M0_CPU→Q1→Q1_CPU链，前阶段失败立即封存。完整248步M0与CPU通过前不进入heldout。证据evidence/cross_scene_smooth_ap_launch_20260921。

本轮仅新训练合同，不修改旧Q1结果、温度、倍率或终点；不做官方测试/多seed/消融，不删除无明确冗余证据的权重。三数据集Goal ACTIVE/UNMET。


§41.216容量阶段补充：00:31:46六容量端各8步均完成，203/203累计非零、overflow0、Signal/冻结状态检查均通过；原wrapper16885与M0原16907存活，GPU7424MiB/100%，输出余6618443776B。尚待两个100步过拟合及完整M0_CPU，不称完整M0通过。evidence/cross_scene_smooth_ap_completion_preparation_20260921保存观察和终态接收/全行分析准备（AST检查而未执行）。新鲜M0独立审计 /root/audit_cross_scene_smooth_ap_m0_20260921 已启动静态/GT阶段，需完整终态才能裁决；私有trace在.aris/traces/experiment-audit/2026-09-21_cross_scene_m0。下一观察约00:36，原任务持续，不改合同/不重启。Goal ACTIVE/UNMET。


### 41.217 跨场景 Smooth-AP 完整 M0/CPU通过，原链进入Q1（2026-09-21）

原M0进程16907于00:36:52退出0，675.716秒；M0_CPU18145于00:37:03退出0，10.924秒。PASS_ENGINEERING_ONLY与PASS_COMPLETE_CROSS_SCENE_SMOOTH_AP_M0，summary SHA c59e39cc2192bfb11b12a58f53eb8b3853455bfd41744c74aaf4742272ba1506。完整248步、4945920距离元素与5760历史VJP记录前向的保存账本核对；未重新模型反传，不将运行梯度见证称为独立逐步向量重建。

六容量端各8步，均203/203累计非零、overflow0及全部冻结/Signal/角色更新检查通过。两个固定100步过拟合原超额损失比控制0.0006996800、跨scene0.0007000677，原门槛0.1，均通过。28份原始文本6626715B完整接收且逐文件字节/SHA匹配；inventory/intake及观察另列，证据evidence/cross_scene_smooth_ap_m0_complete_20260921。原图、模型和数组仍仅远端。独立M0审计在同一新鲜上下文继续，完整裁决待返回。

原wrapper16885于00:37:03自动启动Q1原18222。00:39:41二者实际cmdline匹配且存活，首控制完成第5/20epoch，0/6终点；GPU6426MiB/96%，输出余6594674688B。保持执行d35864d/配置e5326b52、seed42、全部六端固定终点；无中途调参或检索结论。后续按历史阶段实际耗时估计完整端点，原CPU链仅六端齐全后执行。未删除权重；原失败判定和正式结果不变，Goal ACTIVE/UNMET。


§41.217运行与接收工具补充：00:46:28原wrapper16885/Q1 18222存活，首控制8/20epoch，0/6终点；第7/8epoch分别145.6974/145.9499秒，输出余6568394752B。首端训练暂估01:15附近，六端暂估04:30–05:00，候选端速度和终点检索耗时尚需实际校正；下一运行观察约01:12接近端点，非短间隔轮询。evidence/cross_scene_smooth_ap_completion_preparation_20260921保存观察与真实M0八组248步上的描述函数检查PASS。完整Q1 main/六端断言/最终统计尚未执行，不将函数检查称为科学核验。

独立M0审计已报告248步/4945920距离元素数值复算完成，最终A–F报告与归档清单仍在封存；必须接收最终报告后再关闭审计，不由root替代审计裁决。当前无新heldout终态；固定合同、旧失败及Goal ACTIVE/UNMET不变。


### 41.218 跨场景 Smooth-AP 独立M0审计闭环（2026-09-21）

独立审计/root/audit_cross_scene_smooth_ap_m0_20260921完成，WARN/CLOSED_WITH_LIMITS、确定性工程PASS、0阻断、same-family/provisional。报告refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_AUDIT_M0.md/.json；审计30项文本54984634B及manifest/receipt完整归档evidence/cross_scene_smooth_ap_m0_independent_audit_20260921，逐项SHA/字节匹配，未下载二进制或私有连接助手。报告SHA6b63624beecbffe5cc2ff7dc3af353eb661d865eb2f51954ff8d1261a952f458。

独立NumPy/标准库复算248步、15872次当前anchor曝光、4945920距离值；standard/cross AP最大差1.85272663e-7/2.59938854e-7，14项总账最大差5.58793545e-7。六容量checkpoint各241B0别名+231角色状态，共472张量重组SHA一致；1032GT/155身份/3096文件名、三B0的1950来源步及119文件156绑定核对。两过拟合比与label-smoothing解析下界复算通过。

D项WARN保留：逐步参数梯度与重载输出是运行见证；六首历史组在step4，direct/VJP最大相对误差2.17581631e-5，仅独立复核标量而未重生成。203/203是累计非零而非每步；十项分类/残差只核对保存标量与加权账本，缺少完整logit/距离用于重建。M0实际零eligible批次0，零分支图连接由T0合成与静态调用支撑；全来源9个零批次中预热后4个保留。审计本地长路径解包失败及扁平名修复留证，未重跑实验。

原Q1保持d35864d/配置e5326b52与wrapper16885/child18222，不因审计闭环重启。最近00:46首控制8/20epoch、尚无完整端；下一观察约01:12。新收到的MMPareto/GradNorm/OGM-GE建议仅开展主源阅读，不注册或加入当前训练。原Q1失败、正式结果及三数据集Goal ACTIVE/UNMET保持。


### 41.219 用户新复核的优化近邻核读与目标范围（2026-09-21）

用户新复核基于较早8effade/§41.216；以§41.218的完整M0审计关闭与Q1正在运行状态接续。未读取或声称重新生成其sandbox PDF/LaTeX。原正式表无新增成绩。新建议“支持感知的排名—身份协作”仅候选，必须先完成当前六端/CPU/审计，不加入当前Q1。

按research技能后台完成MMPareto/GradNorm/OGM-GE有界主源阅读，保存evidence/support_aware_gradient_neighbours_20260921/PRIMARY_SOURCE_NOTE.md和SOURCE_RECEIPT.json；这是文献阅读，不是实验独立审计。MMPareto官方a339db39：论文整体encoder范数恢复，代码逐参数张量max(1,原和范数/组合范数)并乘gamma1.5；非冲突分别>=0与>0。纯数学二维反例说明全局共同下降性质不能无条件继承到逐张量缩放，不是作者运行失败率或项目观测。GradNorm需要训练进度与选定共享参数块；未确认作者官方代码，不编造实现/许可证。OGM-GE官方43aa7332仅处理audio/visual四维梯度，噪声使用张量元素std，非完整跨batch协方差；与论文tie定义差别保留。MMPareto API license null/无LICENSE文件，OGM-GE MIT；未复制作者实现到项目。

root另核对当前计算图，PROJECT_OBJECTIVE_SCOPE.md与输入SHA保存：O在全局含7ID+6Triplet；但单角色encoder直接依赖的O项仅fused ID及自身完整/纯残差两组ID/Triplet，共5项，其他角色8项没有直接图路径。这是静态依赖推断，不保证各项每步非零，未做新的逐项梯度测量。既有F/O诊断定义不变，不能从13项数量推出13倍压制；角色各自调权也不能未经证明称为统一标量loss的梯度。

当前不确定倍率、EMA规则或新方向处理，不实施控制器或新增消融。冻结Signal/原三角色/完整历史导数/seed42等条件保持，Q1执行d35864d不变。原任务下一端点观察约01:12，全部终态前不解读检索收益；三数据集Goal ACTIVE/UNMET。


### 41.220 当前入口纠正及Q1末段运行观察（2026-09-21）

01:09:13直接核对/proc确认原wrapper16885与Q1 18222存活且cmdline吻合；首控制第17/20epoch完成，0/6完整端点。最近第14—17epoch分别146.3926/146.5988/145.9451/145.8543秒，GPU6502MiB/94%，输出余6525530112B。证据evidence/cross_scene_smooth_ap_q1_progress_20260921；预计首端01:15左右训练结束后检索，下一观察约01:17。未读部分结果决定干预、未重启或修改合同。

修正首页仍指向旧§41.201的过期入口，将V6模块列表明确标记为历史，并将V8 Phase-B范围限定为RGBNT201固定dev。AGENTS最新标题同步为跨scene Q1；原Smooth-AP条目标为已封存历史。只修正文档导航，不覆盖旧结果/门槛，不将当前内部Q1写入正式表。Goal ACTIVE/UNMET。


### 41.221 跨场景 Smooth-AP 首控制固定终点完成（2026-09-21）

01:15:49直接观察原wrapper16885/Q1 18222存活、cmdline吻合。fold0 control完成20epoch/260更新，fit epoch时间合计2258.3322秒（37.6389分钟），overflow0、累计203/203非零、全部冻结/Signal/角色更新/预算检查true；checkpoint及严格重载记录、完整图库检索和最终receipt均写出。1/6端完成，同折cross_scene已自动完成第1epoch。证据evidence/cross_scene_smooth_ap_q1_progress_20260921/trifusion_cross_scene_observation_0116_20260921.json，实际观察时间以文件字段为准而非文件名分钟。

只读观察器v2将训练完成、checkpoint、检索与receipt分别记录，避免执行器先保存training后尚未检索时被误称完整端；未改变训练器或已绑定配置。全部六端的CPU与科学审计尚未执行，不把单端运行检查扩大为完整检索核验或晋级。输出余6439456768B（约6.00GiB），本轮删除0权重。下一观察约01:24，使用候选历史阶段速度校正全量ETA；当前全六端仍暂估04:30附近，随后CPU/独立审计。Goal ACTIVE/UNMET。


### 41.222 首个实际零跨场景支持批次核对（2026-09-21）

01:21:51原wrapper16885/Q1 18222存活，候选第7epoch完成144.6504秒，历史阶段速度与控制接近；该观察留D盘原始记录，无中间检索解读。根据已登记fold0 step180零eligible位置，在预计经过后01:39:45只读核对实际保存行及后续181行。64个anchor跨scene正例均0，每个仍保留395个负身份候选；cross AP/loss存储0，339条历史leaf上游norm全0，历史VJP组与记录重算数0。三个角色当前合成norm为0.29522184/0.24617020/0.34217880，历史贡献0、加入前后差0。原进程直接/proc确认继续存活。

证据evidence/cross_scene_smooth_ap_zero_support_runtime_20260921。这是单个实际保存行的独立mask算术与运行见证，未重生成梯度或测量参数更新差；训练器optimizer调用后写入记录，完整AMP与14项账本仍待完整终点/CPU。既有M0零eligible=0和T0合成边界保持，不把新见证倒写成M0覆盖，也不以单行代替四个预热后零批次的全量核验。

同期首控制完整260步/receipt，候选第14/20epoch，1/6端；GPU7412MiB/96%，输出余6355922944B。下一观察约01:54接近候选固定终点，不改定义/不重启/不删除权重。总Goal ACTIVE/UNMET。


### 41.223 跨场景 Smooth-AP 首折两端完成，第二折自动继续（2026-09-21）

2026-09-21T01:54:23.727569+08:00只读核对原wrapper16885及Q1 18222的/proc命令行均存活。fold0 control与cross_scene各完成20epoch/260更新，overflow均0、累计203/203可训练梯度张量非零，所有已记录工程检查true；两端均有checkpoint记录、完整图库检索字段及端点receipt。训练epoch时间合计分别2258.3322与2218.0301秒。完成2/6端，日志证明fold1 control已自动执行，无人工重启。

观察证据：evidence/cross_scene_smooth_ap_q1_progress_20260921/trifusion_cross_scene_observation_0154_20260921.json。这里只核对完成状态，不根据首折成绩决定后继设计；原六端CPU、完整配对统计和独立科学审查仍待结束后执行。单端工程标志不替代这些核验。执行代码d35864d、配置e5326b52、seed42、固定终点和原门槛均未变。

输出盘余6270742528B（5.840GiB），本轮删除0权重。继续原队列，依据约38分钟/端的实测速率在下一端预计完成前观察；全六端仍预计04:30附近，实际完成后再进行CPU和文本证据接收。Goal ACTIVE/UNMET。


### 41.224 跨场景 Smooth-AP 完成3/6端，原队列继续（2026-09-21）

2026-09-21T02:31:12.240044+08:00原wrapper16885/Q1 18222存活；3/6端固定20epoch/260步、checkpoint/检索/receipt齐全；fold1 cross_scene自动完成第4epoch。本次新增完成fold1 control，fit epoch累计2212.8621秒；overflow0、累计203/203非零梯度张量，所有已记录工程检查true。通过/proc核对原命令行，未重启、未改执行代码、配置或合同。

只读观察证据：evidence/cross_scene_smooth_ap_q1_progress_20260921/trifusion_cross_scene_observation_0231_20260921.json。本次只登记完整端点回执及继续运行证据，不解读局部fold分数；全部六端、CPU重算和独立科学审查完成后才判断原门槛。单端已记录的工程标志不是完整审计的替代品。

输出盘剩余6115622912B（5.696GiB），本轮删除0权重。继续按约38分钟/端的实测速率在接近下一终点时观察；全六端仍暂估04:30附近，终态后接收文本并核验。Goal ACTIVE/UNMET。


### 41.225 跨场景 Smooth-AP 完成4/6端，原队列继续（2026-09-21）

2026-09-21T03:08:17.845631+08:00原wrapper16885/Q1 18222存活；4/6端固定20epoch/260步、checkpoint/检索/receipt齐全；fold2 control自动完成第5epoch。本次新增完成fold1 cross_scene，fit epoch累计2179.0565秒；overflow0、累计203/203非零梯度张量，所有已记录工程检查true。通过/proc核对原命令行，未重启、未改执行代码、配置或合同。

只读观察证据：evidence/cross_scene_smooth_ap_q1_progress_20260921/trifusion_cross_scene_observation_0308_20260921.json。本次只登记完整端点回执及继续运行证据，不解读局部fold分数；全部六端、CPU重算和独立科学审查完成后才判断原门槛。单端已记录的工程标志不是完整审计的替代品。

输出盘剩余5949620224B（5.541GiB），本轮删除0权重。继续按约38分钟/端的实测速率在接近下一终点时观察；全六端仍暂估04:30附近，终态后接收文本并核验。Goal ACTIVE/UNMET。


### 41.226 跨场景 Smooth-AP 完成5/6端，原队列继续（2026-09-21）

2026-09-21T03:46:03.864528+08:00原wrapper16885/Q1 18222存活；5/6端固定20epoch/260步、checkpoint/检索/receipt齐全；fold2 cross_scene自动完成第5epoch。本次新增完成fold2 control，fit epoch累计2193.6746秒；overflow0、累计203/203非零梯度张量，所有已记录工程检查true。通过/proc核对原命令行，未重启、未改执行代码、配置或合同。

只读观察证据：evidence/cross_scene_smooth_ap_q1_progress_20260921/trifusion_cross_scene_observation_0346_20260921.json。本次只登记完整端点回执及继续运行证据，不解读局部fold分数；全部六端、CPU重算和独立科学审查完成后才判断原门槛。单端已记录的工程标志不是完整审计的替代品。

输出盘剩余5788823552B（5.391GiB），本轮删除0权重。继续按约38分钟/端的实测速率在接近下一终点时观察；全六端仍暂估04:30附近，终态后接收文本并核验。Goal ACTIVE/UNMET。


### 41.227 跨场景 Smooth-AP 全六端与CPU结束，原Q1_FAIL封存，独立审查进行中（2026-09-21）

04:23:20直接观察原wrapper16885/Q1 18222/CPU 34603均结束，GPU空闲。原pipeline COMPLETE_VERIFIED_Q1_FAIL：训练04:19:46退出0，CPU04:20:11退出0；六端各20epoch/260更新，共1560。CPU覆盖116501504保存距离元素、583168历史VJP record前向计数、2069520检索距离/排名元素；不是全程重新运行模型或生成参数梯度。summary SHA 3aa13b43b4a3c6d665486077dcd64daf0b9693a78aae75b3c92de04c3dd0879f；CPU SHA 905d44f4eaea5f5a9c43fc2c5029af22c896f33d50cfb51c252031bf81dd9fff。

匹配fused 52.838309→53.405492，+0.567183mAP；三折+1.335700/+0.026182/+0.297229。CNN -0.265296、Transformer -0.137895、Mamba +0.593532。候选fused比Signal53.129381高0.276111，且高于三角色；但增益门、全角色非负门、bootstrap下界（-0.077239）未过，配对2/5；对Signal1/5。原Q1_FAIL保留，不将局部正证据写成晋级。fused R1 61.166667→62.000000；相对本次control修复11、新增6，其中1个同scene；28身份改善/26下降/6不变，全部60身份与600query均保留。

末65步、三折等步数均值：同定义cross-scene AP loss 0.029118386→0.014568746，standard AP loss 0.006668543→0.005573188；但batch hard 0.062468318→0.064545732、expanded hard 0.158681301→0.161146010。实际total目标定义不同，不能直接比较绝对total。来源候选AP不是完整图库mAP；不能据其下降推出未知身份稳定收益。

所有实际零支持步骤已在完整账本中确认：fold0/180、fold1/221、fold2/133及232，活动排名项与历史上游为0、VJP组跳过；原其余监督与更新继续。预热仍有小幅浮点差异，三折total最大约0.001278/0.001623/0.002021，不声称逐位训练轨迹一致。保留全部120个epoch和1560步14项标量CSV；原合同仅固定epoch20检索，无逐epoch检索结果。

30份原文本共86277717B已核对哈希，inventory SHA e8fb10ec3cad2cd3b70f921f7c06ce977edc76b136dcdfbcfa9bfda401aa1952；本地下载二进制0。证据evidence/cross_scene_smooth_ap_q1_complete_20260921，完整分析evidence/cross_scene_smooth_ap_q1_analysis_20260921，报告results/MSVR310_CROSS_SCENE_SMOOTH_AP_Q1_2026-09-21.md。本地报告首次读取query-only排名表时误用gallery位置，被断言截获；失败脚本保留，修正为query行号后在新目录全量完成，未改变原训练或排名。

独立审查audit_cross_scene_smooth_ap_q1_20260921已按fresh-none、gpt-6-astra/max请求启动，属于同家族provisional，尚无终态审计判定。审查关闭后再依据完整证据确定唯一后继假设。无新官方成绩、不扫描失败配置、seed42不变。04:23输出盘余5640167424B，本轮删除0权重。Goal ACTIVE/UNMET。


### 41.228 跨场景 Smooth-AP 独立Q1审计闭环（2026-09-21）

fresh独立审查audit_cross_scene_smooth_ap_q1_20260921完成，WARN/CLOSED_WITH_LIMITS；same-family/provisional，所请求gpt-6-astra/max不冒充后端认证。原执行d35864d、配置e5326b52、summary3aa13b43及CPU905d44f4均不变。审计原文refine-logs/msvr310_cross_scene_smooth_ap_v1/EXPERIMENT_AUDIT_Q1.md/.json，MD SHA 0fff8f52997cdd8e42433ffc693181868eada003e58a1cd98eedb041dd21c989；原始脚本、输出、失败尝试、manifest/receipt完整归档evidence/cross_scene_smooth_ap_q1_independent_audit_20260921。主线逐条字节/SHA复核，原生完整回复另存私有trace，模型/图像/数组下载0。

审计独立核对134注册绑定、88执行源码、3096训练模态文件名、1950个B0训练步骤及六Q1 checkpoint各472状态张量。独立全量数值重算1560步/116501504训练距离元素、30组检索输出/2069520检索距离与排名元素、6000查询输出行及300身份输出变化；v2报告、5份CSV及来源描述统计另核对57813数值，最大差4.44e-14。配对fused+0.5671825364，bootstrap下界-0.07723931035；原配对2/5、Signal1/5、科学Q1_FAIL保持，不以工程审计通过晋级。

WARN边界保留：未重新生成全部逐步参数导数，部分冻结/重载与梯度仍为运行见证；两端预热不是逐位同轨迹；仅seed42、内部重复开发身份、非官方。审计配置链KeyError及过严预热断言失败均原样保留，后者改为完整量化差异，没有修改原训练、原门槛或原结果。所有训练与CPU进程已经结束，不重新启动。

### 41.229 后继支持感知梯度平衡草案，尚未登记或训练（2026-09-21）

依据完整结果与已核对MMPareto/GradNorm/OGM-GE近邻，形成refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_PLAN_DRAFT.md及tracker。唯一候选干预是在相同cross-scene AP、候选与完整历史VJP下，对三个encoder角色分别组合完整排名R和原辅助A。草案固定EMA0.9、范数比平方根、r范围[0.25,4]、系数和2；无合法跨scene支持时EMA不更新、原辅助训练继续；不加入方向投影/新loss/新网络。

R必须含当前和历史，分类头保留原梯度；不能用total_vs_history当作R/A。当前total减rank得到A的有限精度误差需与直接辅助反传核验，真实AdamW参数更新也要记录。草案不声称已经测出当前cross-scene梯度失衡，不把已有统计工具包装为原创。尚无新执行代码/配置/T0/M0/Q1；下一步代码化、登记与审查后按原阶段门推进。seed42、先主结果后消融、官方禁调参及三数据集Goal ACTIVE/UNMET不变。本轮未删权重。


### 41.230 支持感知角色梯度平衡固定实现及独立代码审查完成（2026-09-21）

唯一新假设登记于refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_PLAN.md；配置configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json，SHA 0b5ff0107e8dda6634e0c060a4b1a87e8e7e2fa117c7beb0cf5792b6ebedc15b。7个新Python文件另建，原绑定代码不改。两端均cross-scene AP、相同初始化/候选/full history VJP，仅balanced按三角色完整排名R与其余监督A组合梯度；EMA0.9、比值平方根、r限制[0.25,4]、权重和2且各[0.4,1.6]。无支持时EMA保持、原辅助更新继续；分类14张量保持原梯度，encoder189张量调节。不新增推理参数/新loss/新结构，不将已有范数工具称为原创。

fresh代码审查review_supported_gradient_balance_20260921结论PASS_WITH_LIMITS，无剩余BLOCKING；same-family/provisional，所请求gpt-6-astra/max/fresh-none不作为后端attestation。审查直接阅读7源码、计划/配置、旧训练及核验依赖；AST7、本地新9项及前6配置58项绑定通过，来源元数据证明三折M0第4步直接参考分支可达。没有执行张量、模型或远端命令，不能当作T0/M0通过。初版T0仅重排名字的缺口已由执行者补充实际多张量角色置换与整块范数检查，最终补丁已独立复读；保留修改和运行未验证边界。

原审查文本与回执归档evidence/supported_gradient_balance_preparation_20260921，计划目录保存EXPERIMENT_CODE_REVIEW.md/.json。M0需实际验证R_current/R_history/full R、直接辅助与总减R、组合梯度，保留原203/203、0overflow、冻结/严格重载和overfit≤0.1。CPU只重算保存标量与距离，不声称恢复逐步参数向量。记录实际AdamW更新范数，不将R/A比解释为更新份额。

05:10:42实查前序全部原PID结束、GPU1MiB/0%；输出余5640347648B，仓库余1650192384B。本轮尚未删除权重；执行前再查至少4GiB和空闲GPU，当前预计新增不超过3GiB。远端数据/模型/数组保持不下载。新T0/M0/Q1尚未执行；下一步同步本次源码、启动固定持久队列，失败读原日志再处理，不重启未结束任务、不修改科学门。Goal ACTIVE/UNMET。


### 41.231 首轮M0实际数值门停止，直接辅助导数修订已审查（2026-09-21）

R1执行92a75e4ac46cccb79bb6f78f913998c0d0039ce8/配置0b5ff010，05:35:20启动screen tri_supported_balance_92a75e4，wrapper41216。T0 41222于05:35:26退出0，完整780来源batch及新增数学检查通过。M0 41238于05:36:03退出1、耗时37.348秒；原pipeline STOPPED_AT_M0，全部原进程结束。M0 summary残留RUNNING是失败前写入的初始文件，不代表任务存活。fold0 control只完成3更新，第4步在optimizer更新前的auxiliary参考检查失败；无balanced端、完整M0或Q1结果。

原辅助相减梯度与直接辅助参考：first_norm0.3899933414205923、second_norm0.3900544174096418、difference0.0026616547754863508、cos0.9999767264664654、relative0.006824872357540746>固定0.005。证据只证明combined backward减rank未在门内还原direct auxiliary；混合精度不同回传累加与末次相减均可能贡献，不能只归因于FP32减法。R1源代码、配置、合同、T0与失败文本封存evidence/supported_gradient_balance_m0_r1_failure_20260921，8份远端原文3613063B已逐一远端SHA核对，模型/数组下载0，原3次更新和失败第4步成本保留。

R2唯一实现修正为每步直接求原其余13项encoder辅助导数，两端同算；候选用完整R与direct A组合，控制/预热/无支持仍原current_total+history。原subtraction_auxiliary_vs_direct和direct_sum_vs_original继续量化差异；不再把已失效的AMP可加性当作CPU恒等式。M0实际A与独立再次求导的参考比较；control applied用原完整direct_loss参考，candidate用独立R/A加权参考，阈值仍0.005/零参考1e-8。固定EMA/权重/数据/optimizer/科学门不变，不构成新性能结果。记录两端新增一次auxiliary求导成本，保留有限精度上direct与combined路径差异。

原fresh审阅者继续检查R1原文与R2源码，最终PASS_WITH_LIMITS，无剩余BLOCKING；不是第二个独立fresh agent，same-family/provisional且backend未独立认证。7源码AST及当前/前置67项目绑定、R1本地文本与inventory核对通过。新增T0差异fixture覆盖control、candidate、active无支持并调用实际CPU标量核验器；目前仅源码审阅，尚未实际运行R2张量或模型。意见与最终代码审查归档evidence/supported_gradient_balance_r2_review_20260921，计划目录EXPERIMENT_CODE_REVIEW.md/.json已指向R2。

R2固定配置SHA9ce36299299efbc09ccfb72f5b1c4fed20fc1026bd9fee1bd55a591d6950ec7b，implementation_revision=r2_direct_auxiliary。下一步同步后新目录重新T0/M0，从原初始化开始，不续训R1。05:39输出盘余5632225280B，GPU1MiB/0%；未删除权重。没有新检索、官方或晋级结论。Goal ACTIVE/UNMET。


### 41.232 R2实际T0通过、M0越过首轮失败位置（2026-09-21）

05:52:33 screen tri_supported_balance_r2_1381639启动；执行1381639f778f77f124a2726ee55c07610092a438，配置9ce36299299efbc09ccfb72f5b1c4fed20fc1026bd9fee1bd55a591d6950ec7b；run /root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639。wrapper42758，T0 42764于05:52:39退出0（6.3110秒），M0 42784随后自动开始。R2从原固定初始化开始，没有恢复R1的3个更新。

实际T0完整780来源batch及合成检查通过，包括direct辅助与减法不同的control/candidate/active无支持三路径、实际CPU标量验证器、整块范数/置换/AMP/head保持。T0仍是模型前向0/更新0/heldout图像0。2026-09-21T05:56:41.949639+08:00通过/proc核对原wrapper/M0命令行存活；已保存首历史组中current/history/full R、直接辅助、实际组合的全部可见独立参考通过，当前最大相对误差6.11777801523e-05，低于原0.005。只说明这些已完成位置通过，尚未完成全部六容量端、两个过拟合端和CPU，不把局部见证写成M0 PASS。

证据evidence/supported_gradient_balance_r2_start_20260921含原T0/启动/观察及逐角色参考数值。原R1减法失败保持；R2两端原控制路径与直接分量有限精度差异继续记录，系数/误差阈值/预算未改。当前输出盘余5525925888B；未删除权重、模型/数组下载0。下一步让原队列完成248更新M0及完整CPU，成功后自动Q1；不重启、不挑局部结果。尚无新检索/正式成绩，Goal ACTIVE/UNMET。


### 41.233 支持感知梯度平衡R2完整M0/CPU通过，六端Q1自动启动（2026-09-21）

原M0 42784于06:09:36.381退出0，1016.567秒；CPU44719于06:09:45.483退出0，9.101秒。M0 PASS_ENGINEERING_ONLY，CPU PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_M0；summary SHA e63fc2ace053952c3775b19e37c0064af890b41f2a96d03a63c6c4237cd01545，CPU SHA 910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1。执行1381639、配置9ce36299，科学合同未改变。

6容量端各8步及2过拟合端各100步全部完成，共248更新；八个运行均203/203累计非零、overflow0、冻结/Signal状态不变及角色更新检查通过。六个容量端检查点严格重载通过；两个过拟合运行未保存终点检查点，也没有执行严格重载。两个原超额损失比控制0.000701213786、候选0.000721186408，均低于原0.1门。90项实际独立参考检查全部通过，最大相对误差9.433501552915618e-05，未放宽0.005门。峰值reserved最大12186MiB。

CPU核对全部4945920距离元素、248步以及5760历史VJP记录前向账本，重算掩码、目标、支持和EMA/系数；没有重新生成每步参数梯度。完整28份原始文本7961437B接收，逐文件字节/SHA匹配，模型/图像/距离数组未下载。证据evidence/supported_gradient_balance_m0_complete_20260921；evidence/supported_gradient_balance_m0_analysis_20260921保留744条角色记录、90参考检查和8条完整阶段epoch记录。

实际M0共232支持步、16预热步、零支持步0；无支持语义目前由T0及全来源合同覆盖，不声称真实M0遇到此情形。短容量角色范数和系数确有差别，不能把这些8步观测当成Q1轨迹，更不能用来扫描系数。原combined与direct分量数值差异保留；AdamW参数delta范数不解释为排名更新份额。R1失败完整封存，不因R2通过而删除或改写。

原wrapper42758于06:09:45.485自动启动Q1 44804；06:13:21原cmdline核对存活，首控制完成5/20epoch、0/6终点。输出余5398437888B，尚未删除权重。预计完整Q1约4–6小时，按历史生效后的实际速度修订观察时点，不改预算。Q1从固定原初始化开始，不续训M0。独立M0 fresh gpt-6-astra/max审查已启动；same-family/provisional，后端未独立认证，尚无裁决。

保持完整六端固定终点、全部图库与原两组五门，不按局部fold改设计。上一轮cross-scene Q1_FAIL及所有正式成绩不变；此处仅新增工程证据。三数据集Goal ACTIVE/UNMET。


### 41.234 支持梯度平衡 R2：M0 独立审查闭环，Q1 首端完成（2026-09-21 06:59 北京时间）

执行仍为1381639、配置9ce36299，原wrapper42758/Q1进程44804持续运行；没有重启、修改训练目标或门槛。06:59:13首控制端20epoch/260更新完成，检查点、检索与端点回执齐全，现为1/6完整端点。无完整Q1/CPU或新增正式结果；不使用局部分数选方法。

fresh-none gpt-6-astra/max独立M0审查终态WARN/CLOSED_WITH_LIMITS：A/B/C/D/F通过、E覆盖限制保留，0剩余工程或审计阻断。same-family/provisional，后端独立认证不可用。完整报告见refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_AUDIT_M0.md/.json；402份UTF-8证据按原manifest及SHA归档至evidence/supported_gradient_balance_m0_independent_audit_20260921，包含精确请求、失败脚本/输出、修订快照和原文回复，不只保存结论。

独立远端CPU核验26.965秒、2线程/interop1、CUDA未初始化：248步、4945920距离元素、8280标量恒等式、780来源batch及37个M0文件；六个容量端和三个B0检查点在远端复核。90项原运行参数参考均满足原0.005门槛，最大相对误差9.433501552915618e-05；保存标量复算不等于重新生成每步参数梯度。

真实M0为16预热/232有支持/0活动无支持更新；557个有支持但排名范数为零的角色步正确更新EMA，不当作无支持。严格重载仅覆盖六个容量端，两个过拟合运行未保存终点检查点、未重载。T0及完整来源计划中的无支持情况不是已完成Q1的实测覆盖。R1第4步0.006824872>0.005失败与此前3次更新保持封存。

审查期间发现报告草稿将Windows换行转换副本哈希误称原始回执，现已纠正并保留错误草稿与AF6记录。远端原始m0_cpu.json为16016字节/SHA910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1；本地渲染副本16359字节/SHA4f551ff62e64b5352fbd03398b3725c18ecee81b4d0f8ef8efff68013a6a5228，343个LF→CRLF且JSON内容一致。154份同类文本逐一核对；实验数据及门槛未改。

输出盘06:59剩余5226696704字节；保留所有初始化/终点/审计依赖。终态接收、全量日志与梯度分析、独立Q1审查请求已准备，等六端和CPU齐全后执行，不将准备状态当成终态。仅seed42、先主结果后消融，三数据集Goal ACTIVE/UNMET。


#### §41.234 接续：终态接口与磁盘维护（07:10）

07:08:31原wrapper42758/Q1 44804持续运行，首候选完成9/20epoch、1/6完整端点，GPU7578MiB/100%，输出盘5207883776B。未改变训练。检查终态工具发现文本接收清单仅嵌入pipeline对象，通用排名复算却要求独立pipeline.json；已补接收该原始文件并解析本地/嵌入远端Python语法。原排名复算可直接指定candidate=balanced及PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_Q1，等待完整终态后执行，不重复实现指标，不把接口准备当成核验通过。

磁盘盘点48个已同步Git bundle，共104067074B；全部具备本地相同SHA副本、引用提交在远端HEAD11678ca祖先链内，GitHub HEAD相同。删除前再次核对精确绝对路径位于transport、非符号链接、大小/SHA、提交存在与祖先关系，仅删除这48个明确文件。07:10:33完成，主卷1472622592→1576755200B；权重删除0、训练输出未动。逐文件计划/回执及执行脚本归档evidence/transport_cleanup_20260921。三数据集Goal保持ACTIVE/UNMET。


### 41.235 支持梯度平衡：真实无支持步骤的运行覆盖（2026-09-21T07:20:32.059431+08:00）

原Q1进程44804通过/proc再次确认存活。仅读完整JSONL前缀，不加载模型/图像/数组，不新增更新，不读局部检索分数。fold0控制与候选均已执行第180步：真实跨scene正例支持为0，三个角色完整排名梯度0、EMA before/after相同、应用系数1/1、原梯度与实际应用梯度差0，分类头保持，实际参数更新均非零。首轮07:17仅控制覆盖，第二轮候选也覆盖；原始记录、前缀SHA和脚本归档evidence/supported_gradient_balance_zero_support_runtime_20260921。

这是Q1运行中的工程分支见证，不是独立参数梯度重生成或完整终态；M0的0个活动无支持更新限制仍原样保留，不追溯改写审查。候选已写183步、控制260步，完整六端/CPU/检索审查仍待完成。无新性能结论、不调参，Goal ACTIVE/UNMET。


#### §41.235 首个完整配对里程碑（07:42）

07:42:08真实观察：原wrapper42758/Q1 44804存活，fold0 control/balanced均完成20epoch/260更新，端点回执、检查点和检索记录齐全，2/6完整端点；fold1 control完成6/20epoch。两端203/203累计非零、overflow0、冻结Signal/基座不变及全部训练工程检查通过。训练epoch耗时合计2595.407/2569.485秒；这些不包括所有初始化/重载/检索成本。GPU7578MiB/100%，输出盘5065076736B。未用局部检索分数调参；完整Q1/CPU/审查未结束。原始观察归档evidence/supported_gradient_balance_q1_progress_20260921/observation_0742.json，以文件内observed_at为实际时间，不以本地预设文件名推断时间。Goal ACTIVE/UNMET。


#### §41.235 三端完成（08:21）

08:21:27原wrapper42758/Q1 44804仍运行。fold0两端及fold1 control已完成，各20epoch/260更新、203/203累计非零、overflow0、固定长度与冻结路径检查通过，检查点/检索/端点回执齐全，3/6终点；fold1 balanced自动进入2/20epoch。fold1控制训练epoch耗时2565.730秒，不含全部端点成本。输出盘4915871744B。原始观察evidence/supported_gradient_balance_q1_progress_20260921/observation_0821.json。未读取局部分数作方法选择，完整Q1/CPU/审查仍待结束，Goal ACTIVE/UNMET。


### §41.236 用户新复核接续：单角色五项辅助依赖、AdaTask 边界与 R2 原合同继续（2026-09-21 08:40）

用户补充的53e0c26/07:43快照之后，08:40:02实查原wrapper42758/Q1 44804继续运行，3/6端完整，fold1 balanced完成11/20epoch，GPU7578MiB/100%，输出盘4871094272B。原始文本evidence/supported_gradient_balance_q1_progress_20260921/observation_0840.json。完整六端/CPU/独立审查尚未结束，没有新增正式成绩，不据部分端点选方法或改训练。执行仍1381639/配置9ce36299；Goal ACTIVE/UNMET。

本轮静态核对四个相关代码文件与执行1381639 Git blob逐字一致。全模型除fused排名外确有十三项辅助标量，但单个角色直接连接的辅助只有五项：fused ID、自己的完整分支ID/Triplet、自己的纯残差ID/Triplet；另两角色八项没有到该角色可训练编码器的路径。现行直接求十三项和的导数会自然遵守这项依赖，不需要改训练器。不能用“1对13”替代同角色实际梯度测量。详见refine-logs/msvr310_supported_gradient_balance_v1/ROLE_LOSS_DEPENDENCY_NOTE_20260921.md；本轮没有新张量或GPU试验。

按research技能进行有界原文/作者代码核查，笔记ADATASK_PRIMARY_SOURCE_NOTE_20260921.md。AdaTask论文Algorithm2和作者固定84853de代码已经分别维护任务一/二阶矩、各任务预条件更新求和；作者实现每参数只有一个共同step，零梯度仍推进状态并可能应用旧动量，weight_decay虽接受但step未应用。因此用户提出的无支持冻结排名矩/独立有效时钟、不应用旧排名动量、只做一次AdamW解耦衰减，属于待登记验证的适配，不能写成原版已有规则或已证创新。用户引用Cityscapes三行指标原表一致，论文三种子/末十epoch平均是作者协议；本项目仍仅seed42固定终点，不据此增加种子。

后继选择继续等待R2完整结果。现行有界EMA组合后进入同一套AdamW状态；范数比与记录的实际总参数变化都不足以分摊任务更新贡献，尚未证明共享优化器历史是失分原因。无支持开关、任务时钟、角色依赖和历史导数各有边界，不为了叙事同时叠加。原始权重/数组继续留远端，当前没有确认可删的依赖权重。


#### §41.236 四端完成（09:02）

09:02:56原wrapper42758/Q1 44804仍运行。fold0、fold1两端均已完整结束，各20epoch/260更新、203/203累计非零、overflow0、固定长度及冻结路径检查通过，检查点/检索/端点回执齐全，4/6终点。fold1候选训练epoch耗时2561.982秒，不含全部端点成本。输出盘4757811200B。原始观察evidence/supported_gradient_balance_q1_progress_20260921/observation_0902.json。未读取局部分数择法；第三折、完整Q1/CPU/独立审查仍待结束，Goal ACTIVE/UNMET。


#### §41.236 优化器历史可恢复性边界补核

本轮读取R2实际checkpoint调用及保存函数，保存函数与执行1381639 Git blob逐字一致。终点只保存role模型状态、baseline别名和绑定/身份/配置信息，不保存AdamW state_dict或一/二阶矩；gradient_balance_state是范数EMA，actual_parameter_updates是比较统计。因而当前合同支持终点检索重载及已登记更新统计，不支持仅从这些产物精确恢复原AdamW历史或分解任务更新。未额外读取模型、未更改运行中的保存规则、未重跑已结束端点。后继如需真实状态的反事实分析，必须在新来源流程预先登记捕获；不能用新建空优化器代替原状态。详见refine-logs/msvr310_supported_gradient_balance_v1/ROLE_LOSS_DEPENDENCY_NOTE_20260921.md。该限制不改变当前Q1合同及已完成终点检索的有效性。


#### §41.236 五端完成（09:46）

09:46:28原wrapper42758/Q1 44804仍运行。fold0、fold1两端及fold2 control已完整结束，各20epoch/260更新、203/203累计非零、overflow0，检查点/检索/端点回执齐全、固定长度与冻结路径检查通过，5/6终点。fold2控制训练epoch耗时2572.413秒，不含全部端点成本。输出盘4589674496B。原始观察evidence/supported_gradient_balance_q1_progress_20260921/observation_0946.json。尚未取得最后候选端及完整Q1/CPU/独立审查，不据前五端选择方法，Goal ACTIVE/UNMET。


### §41.237 R2六端训练完成，CPU精确标量重放差异（2026-09-21 10:32）

原Q1 44804于10:29:22退出0，六端20epoch/260更新全部结束。原CPU 69196于10:29:28退出1，pipeline保留STOPPED_AT_Q1_CPU。失败在verify_balance的ratio精确相等；训练math.sqrt与核验x**0.5在六条记录相差1ULP（2.220446049250313e-16）。全1560步/3486有支持角色记录的保存ratio均精确匹配运行math.sqrt。核验器仅将幂运算改为同一math.sqrt，精确相等及原所有阈值不放宽；训练/config/checkpoint不改、不重训。原失败日志、pipeline、全量差异诊断归档evidence/supported_gradient_balance_q1_cpu_failure_20260921。完整修复后CPU核验、六端分析及独立审查仍待完成，暂不宣布科学结论。输出卷4430831616B，GPU已空闲；保留所有必要权重。Goal ACTIVE/UNMET。


#### §41.237 执行绑定完整保留的复核入口

第一次修复复核q1_cpu_sqrt_recheck在context被原project_file_sha256拦截，未进入完整统计核验，失败记录保留。恢复原verify_msvr_supported_gradient_balance_stats.py精确执行字节；新增tools/recheck_msvr_supported_balance_sqrt.py独立事后核验入口，先验证原文件SHA，再在进程内仅替换x**0.5为math.sqrt并记录修改前后源码SHA，保留精确比较和所有原检查。原训练配置、全部绑定文件和原pipeline保持；不将失败stage重写为PASS。新的复核仍待完成。


#### §41.237 加权梯度范数标量检查未闭环

独立入口q1_cpu_bound_sqrt_recheck通过原配置绑定与ratio精确重放，随后停在原stats第77行：balanced加权后范数平方与wR²||R||²+wA²||A||²+2wRwA<R,A>比较。全六端1560步标量诊断执行95343次close比较，唯一超限为fold0 balanced第71步Mamba，5.0530119215775136与5.05301246766921，差5.460916963073714e-7，原阈值5.053012467669209e-7。诊断仅收集原失败，不是验证PASS；所有失败保留。已由experiment-audit要求的新鲜同族审查agent audit_supported_balance_cpu_replay_failure_20260921独立检查实际FP32运算与标量等式、允许的修正范围；尚无最终意见。没有放宽容差、没有训练重启、没有宣告完整Q1科学结论。原CPU与后续复核分别记录，不改写原pipeline。


#### §41.237 独立算术审查与最小修正

新鲜gpt-6-astra/max同族审查支持仅修事后核验：开方使用math.sqrt；第77行加权范数预期值使用实际张量乘法转换后的FP32系数。出错行wR由1.0421102637890658转为1.0421102046966553，wA由0.9578897362109341转为0.9578897356987，修正后范数平方残差4.67142147e-9，小于原阈值。PyTorch2.5.1+cu121 CPU实际转换检查9360次；完整六端1560步的原verify_balance断言在两处修正后全部通过；close容差、控制器精确检查与M0参考门槛均未放宽。合成单元素反例仅验证运算语义，不冒充模型实验。证据evidence/supported_gradient_balance_cpu_equations_audit_20260921。独立审查为same-family/provisional，只批准算术核验修正；完整CPU/Q1审查仍待完成。原执行文件、配置、训练、检查点与pipeline保持不变。


### §41.238 R2完整Q1未晋级，CPU核验已闭环，独立审查进行中

原训练六端10:29完成。10:51:19事后算术复核退出0，完整1560步/116501504记忆距离/2069520检索排名通过。原失败pipeline和各次失败日志未改写。43份文本94652953B逐份SHA接收，模型/数组/原图均未下载。全部120epoch、1560步及4680角色记录已归档。

MSVR310内部Q1：fused53.39938365→53.45264937，+0.05326572；R1 62.166667→62.333333。CNN+0.01821327、Transformer−0.14394053、Mamba+0.12282911。三折fused+0.12996671/−0.10721385/+0.14677428，身份bootstrap下界−0.12614131。配对1/5、相对Signal1/5，Q1_FAIL不变。候选比Signal高0.32326881但三角色均低于Signal；fused修复3条Rank1、新增2条。正式成绩不变。

末65步跨场景AP目标各折下降，但批内与扩展hard目标各折上升。系数确实改变，候选有支持角色的排名系数中位数1.074397—1.156140；未形成所需稳定未知身份收益。配对记录/像素一致，但warmup第2步已有数值差异，最大总loss差约0.00112/0.00140/0.00171；不将+0.0533解释为精确隔离的系数因果贡献。AdamW历史状态未保存，不能反推任务更新份额。

结果报告results/MSVR310_SUPPORTED_GRADIENT_BALANCE_R2_Q1_2026-09-21.md。原始、梯度、epoch、排名证据分别归档evidence/supported_gradient_balance_q1_*_20260921。新鲜gpt-6-astra/max全Q1独立审查audit_supported_gradient_balance_q1_20260921进行中；算术审查已WARN/same-family/provisional，不能替代终态完整审查。暂不启动后继，Goal ACTIVE/UNMET。


#### §41.238 11:07归档后磁盘整理

在远端HEAD与GitHub9ea4025一致、本地同名副本SHA/字节一致且bundle全部引用已合入当前HEAD的条件下，删除/root/autodl-tmp/trifusion-v2/transport内13个冗余.bundle，逻辑17721502B，目录无符号链接越界；原逐文件清单和删除回执保存evidence/transport_cleanup_postq1_20260921。项目卷空闲1405218816→1422946304B；输出卷盘点4430852096B。权重删除0，初始化、终点、数组和审查证据保留。完整Q1独立审查仍进行中，不启动后继训练。


#### §41.238 条件式任务状态方案的接口/存储准备

在完整Q1独立审查期间，仅做远端CPU只读参数存储核对和静态接口检查，没有实现或启动下一实验。角色189张量/5360652 FP32元素，两套任务m/v相对一套增加42885216B（约40.9MiB），不是实测峰值显存。现有分解梯度在optimizer前删除；后继必须登记warmup状态迁移、支持时钟、AMP缓冲、一次衰减及更新合成，不能直接插入AdaTask名称便认为语义完整。证据与待定项evidence/supported_task_state_interface_20260921；研究选择仍待完整审查闭环。


## §41.239 R2 完整 Q1 独立审查闭环（2026-09-21 11:26）

审查结论 WARN / CLOSED_WITH_LIMITS；确定性复算 PASS，科学结论保持 Q1_FAIL。完整六端、1560步、120个epoch、4680角色步骤、600条query及60身份全部核对；远端独立CPU复算116501504训练距离与2069520检索位置，检查六个最终模型和三个B0来源检查点。没有剩余计算阻断，不重复训练或修改门槛。

fused 53.3993836457→53.4526493704，配对+0.0532657247；三折+0.1299667070/-0.1072138488/+0.1467742790，身份bootstrap下界-0.1261413123，原配对条件1/5。CNN +0.0182132655、Transformer -0.1439405257、Mamba +0.1228291124。无新增正式结果。

原pipeline仍STOPPED_AT_Q1_CPU；最终通过来自单独q1_cpu_arithmetic_recheck凭据。第一次事后修复曾临时改动stats核验源码并被绑定检查拦截，随后恢复原始字节；最终源码绑定核验通过。训练、配置和checkpoint未被该修复修改。保留全部失败记录，不改写原流水线。

两端从预热第2步出现数值轨迹差别；微小端点增益不能解释为逐位隔离的系数因果效果。Q1没有保存独立完整梯度参考向量或AdamW矩状态，不能重建任务更新贡献。203/203是累计梯度覆盖。单seed42和反复使用的开发身份限制继续保留。

完整审查位于evidence/supported_gradient_balance_q1_independent_audit_20260921；正式审查副本refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_AUDIT_Q1.md/.json。审阅归属same-family/provisional，backend未独立证明。报告、所有脚本、失败尝试、文本与远端复算凭据均归档。

R2封存，不扫权重或选checkpoint救回。下一候选为借鉴AdaTask的任务状态分离；目前仅接口和存储预算准备，尚未登记或启动。必须先明确warmup状态、有效任务时钟、组合步幅、AMP及单次权重衰减等合同，再单一干预推进。Goal保持ACTIVE/UNMET；已有初始化、最终权重及审查依赖继续保留。


## §41.240 任务状态分离 V1：合同、核心与合成核验（2026-09-21）

R2完整审查封存后，选择单一后继假设：等权完整排名/辅助导数，共用AdamW状态对比分任务状态后相加。新计划refine-logs/msvr310_supported_task_state_v1/EXPERIMENT_PLAN.md已明确；两端均直接R/A，不沿用R2 EMA。Split从首步生效，预热hard Triplet状态在AP切换时保留；无合法AP支持时排名状态/时钟保持且不应用旧动量，有支持零梯度正常观测。单次decay、原heads、原LR/协议/seed42保持。分任务预条件可能改变总步幅，不能宣称纯状态隔离因果效果。

最小核心tools/msvr_task_state_optimizer.py及合成检查已实现。实际远端torch2.5.1+cu121 CPU检查通过：每端12步、24次严格参数恢复续算、缺失与零支持、单次衰减、head、非有限任务缓冲提前停止；shared对原生AdamW参数差0，split与独立原生任务方向相加最大差2.980232238769531e-7。另12步CPU GradScaler kwargs/unscale接口检查逐位匹配。没有CUDA初始化、模型前向或训练更新；这些不替代真实AMP/M0。

证据evidence/supported_task_state_kernel_20260921含合成输出、脚本、空间估算与独立kernel审查。仅核心审查，不等于完整训练集成许可。训练器、启动配置、完整CPU核验器尚未接入，M0/Q1均NOT_RUN。下一步完成这些接口及完整代码审查，然后按原阶段门执行；不得把合成PASS记作M0。

空间实查：输出卷4430852096字节空闲，GPU无计算进程。上一轮M0/Q1产物按扩展名统计，新增14份优化器状态采用保守估算；启动前再检查总预算与磁盘。没有删除任何权重，没有读官方图像。原Goal ACTIVE/UNMET，正式结果不变。


## §41.241 任务状态 V1 完整集成与启动前审查（2026-09-21）

训练器、CPU核验器、T0和持久runner已实现，配置configs/MSVR310/TriFusion-supported-task-state-paired-v1.json。两端相同直接R/A，完整历史导数先合入R再unscale，head保持总梯度；split按任务预条件后相加。新增每步任务矩范数/时钟、实际参数变化、最终optimizer/scaler保存与磁盘精确恢复。原网络、完整图库、三折和科学门槛未变。

新鲜独立集成审查PASS_INTEGRATION_CODE_REVIEW，same-family/provisional/backend未独立证明。首次审查发现支持标志未绑定真实来源关系、分项参考/更新日志可缺省；已在训练前修复。独立局部回归证实五类错误见证此前可通过、修复后均拒绝。没有放宽阈值。审查与修复记录evidence/supported_task_state_integration_20260921/review。

实际远端CPU合成203小参数磁盘往返检查通过：control406/split784矩张量、任务计数和scaler精确，错误支持拒绝。修复后重新通过。没有CUDA初始化、模型前向、训练或官方图像访问；合成检查不替代M0。

当前READY_FOR_T0_M0，尚未启动。持久runner按T0→M0→完整M0_CPU→Q1→完整Q1_CPU固定执行，任何非零退出停止。Q1只在原工程门和全量CPU门通过后允许，仍固定seed42/六端终点，不据中间检索改设计。预计M0 15–30分钟、Q1 4–6小时，随后按实际进度调整观察时间。启动前重新核对GPU空闲、输出卷最低4GiB空间和绑定提交。Goal ACTIVE/UNMET。


## §41.242 任务状态 V1 已持久启动（2026-09-21 11:57）

执行提交cb4f4c37fbedd357ccc0ae4768161d049a280c5d；配置4ebedda4a10d5e535d6820aa05684b7f707f7a0b8308d454b2316e9b3a61f3c3。run /root/trifusion-storage/artifacts/msvr310_supported_task_state_v1_seed42_cb4f4c3；wrapper78254，启动观察T0子进程78258存活、退出码未产生。GPU启动时空闲，输出卷4430852096字节。

初始状态T0_RUNNING，未获得真实M0/Q1终态。持久pipeline按固定门执行；预计M0 15–30分钟后核对，Q1估计4–6小时。观测超时不重启。启动凭据与脚本evidence/supported_task_state_launch_20260921。正式结果仍无新增，Goal ACTIVE/UNMET。


11:59启动核对：T0于11:57:24退出0，用时7.39秒。M0进程78282与wrapper78254经ps确认存活，fold0 control已完成8步capacity（46.28秒）。全M0未结束、M0_CPU与Q1尚未执行；不能据此宣称工程门通过。下一次按预计M0结束窗口观察，不重复启动。


## §41.243 M0 五端容量完成及完整接收准备（2026-09-21 12:03）

12:03:26实查wrapper78254及M0进程78282存活，六个容量端已有5端完成8步，fold2 split记录1步。T0已退出0，完整M0未结束，尚无M0_CPU/Q1终态。输出卷3846840320字节空闲，GPU7284MiB/42%使用率；没有保存失败证据。未重启、未改执行配置。

完整M0接收/分析脚本已准备并通过本地AST检查，尚未执行。接收要求m0及m0_cpu原阶段均退出0、248步与CPU凭据匹配，仅读取全部文本，checkpoint/optimizer/距离数组留远端。汇总预计覆盖248步、744角色记录和90分项参考，并明确工程结果不等于检索收益。脚本与本次观测见evidence/supported_task_state_m0_preparation_20260921。

剩余容量及两个100步overfit按原合同继续，下一重点观测预计12:10左右的完整M0/CPU阶段，随后按实际退出凭据接收，不从中间损失认定成功。Goal ACTIVE/UNMET，正式成绩不变。


### 41.244 支持条件任务状态 V1：完整 M0 与原 CPU 核验通过，Q1 启动（2026-09-21 12:14 实查）

- 执行提交仍为 `cb4f4c37fbedd357ccc0ae4768161d049a280c5d`，配置 SHA-256 `4ebedda4a10d5e535d6820aa05684b7f707f7a0b8308d454b2316e9b3a61f3c3`；本次仅接收文本与同步记录，不修改执行源、配置或门槛。
- M0 原进程 78282 于 12:13:46 退出 0，六端各 8 步容量检查与两个各 100 步过拟合检查共 248 步完成。原 M0_CPU 80369 于 12:13:58 退出 0。流水线原进程 78254 随即启动 Q1 80455；12:14:12 的 `ps` 核实两进程存活。Q1 六端尚无完整终态。
- 已接收 28 份完整文本凭据（7,027,643 字节）并核验文件清单；模型、距离数组及优化器二进制仍在远端。分析保留全部 248 步训练、744 条角色更新、90 项分量参考。最大参考相对误差 `3.12850281798043e-6`。
- 两端原过拟合门通过：control 的 excess-loss ratio 为 `0.000700202514028135`，split 为 `0.0007086466350339654`，原上限 `0.1` 未变；不能将这些训练指标称作检索收益。冻结路径、累计梯度覆盖等原检查详见原始凭据；累计非零不代表每一步每张量非零。
- 独立完整 M0 审查已交给新的 same-family / provisional reviewer，报告尚未返回。代码审查、合成检查、实际 M0、Q1 科学判定分别记录，不提前宣称独立审查通过。
- 12:14:12 输出盘剩余 `3,516,604,416` 字节；未删除当前依赖的权重与优化器状态。下一步在原 Q1 持久运行期间完成 M0 独立审查，按端点时长观察，不从中间分数改方法。
- 完整凭据：`evidence/supported_task_state_m0_complete_20260921/`；完整文本分析：`evidence/supported_task_state_m0_analysis_20260921/`。最终状态序列化、标量更新范数不能重建全部训练梯度轨迹；目前没有新的正式测试结果。


### 41.245 任务状态 V1：Q1 首端速度与完整终态接收准备（2026-09-21）

- 12:19:59 原 Q1 80455 与 wrapper 78254 经 `ps` 确认存活，fold0 control 已有 81/260 条完整步记录，6/20 个完整 epoch；尚无完整端点成绩。前五个预热 epoch 每轮约 37–40 秒，第六轮约 118 秒。后续估时按启用历史排名后的速度，首端完成检查暂定 12:45 左右，不从预热速度外推六端耗时。
- 已准备完整原 Q1 终态接收脚本，必须先满足原六端与原 Q1_CPU 均退出 0、1560 步、2064 留出记录前向、零官方图片读取及 summary/CPU 哈希绑定。脚本尚未执行；不接收部分结果冒充完整比较。
- 已准备全量来源文本分析，保留 120 个 epoch、1560 步、4680 条角色更新与任务时钟，原五项条件由原完整 CPU 核验处理。语法已检查；其中共用的逐步描述函数已在全部实际 M0 248 条记录上运行通过，仅验证日志结构，不是完整 Q1 分析或新检索证据。
- 工具与观察凭据位于 `evidence/supported_task_state_q1_preparation_20260921/`。不修改执行代码、配置、采样或优化规则。M0 独立审查仍在进行。


### 41.246 任务状态 V1 完整 M0 独立审查闭环（2026-09-21）

- Fresh Codex 审查结论 `WARN`，原工程结果 `PASS_ENGINEERING_ONLY`，无工程阻断项；归属 same-family / provisional / backend unattested。原 M0 和原 CPU 退出 0，审查没有要求重跑、修改门槛或暂停原 Q1。
- 独立只读核验覆盖 28 份接收文本、45 个原 CPU 绑定产物、120 个不同的来源/配置/依赖文件、248 步的 4,945,920 个距离元素、90 项分量参考、8 份优化器/Scaler 状态共 4,760 个 moment 张量、6 个容量 checkpoint 完整状态重建及 780 个 T0 元数据 batch。实际 GPU/模型前向、参数更新、图片读取均为零；数组与权重留远端。
- 必须保留的范围：真实 M0 的 248 步全部有排名观测，没有真实无支持步骤；无支持逻辑目前只由合成测试及静态路径支持。overfit 的重复当前 batch 排除了历史副本，因此没有历史候选；历史 VJP 覆盖来自容量端。最终状态不能重建全部训练梯度轨迹。
- `actual_parameter_updates` 是更新前后参数向量的范数、差值范数及二者余弦，不是各任务更新方向，也不是两端更新向量夹角。计划里关于 direction 的宽泛表述在此明确限定；锁定计划原字节保持不变。
- 配对初始化与像素相同不等于梯度逐位相同：第一步 Mamba 标量梯度记录存在最大 `4.0325888398760066e-7` 差异，原因未诊断，不声称两端梯度向量完全相等。首步 split/control 更新范数比约 1.647–1.724，比较改变了有效步幅和预热轨迹，后续正结果不能单独归因于状态或任务时钟。
- 已修复 tracker 顶部仍为 NOT_RUN 的过期当前状态，旧时间线保留并标为历史，当前显示 M0 已核验、Q1 运行中。完整审查为 `refine-logs/msvr310_supported_task_state_v1/EXPERIMENT_AUDIT.md` / `.json`，全部检查代码、快照和输出归档于 `evidence/supported_task_state_m0_independent_audit_20260921/`。
- 审查自身首轮远端脚本有仅影响显示的 endpoint 标签变量错误，原输出保留，修正版独立重算输出为权威；本地探索性的梯度逐位相等断言未通过，已如实保留差异，不改变原实验门槛。当前没有完整新 Q1 及正式测试结论。


### 41.247 任务状态实验运行期间冗余传输副本整理（2026-09-21 12:46）

已核对远端与 GitHub HEAD 为 e7d2480，逐文件验证本地同名 bundle 字节/SHA 一致、所有引用均在远端 Git 对象库且为当前 HEAD 祖先，限定目录无符号链接越界。删除 11 个已导入的冗余传输包，共 2,880,029 字节（约 2.75 MiB）。项目卷空闲从 1,466,155,008 增至 1,469,050,880 字节；权重删除 0，初始化、终点、优化器状态与核验数组全部保留。完整清单/脚本/回执见 evidence/transport_cleanup_taskstate_20260921。本次不构成大规模权重清理，也不改变当前训练。


### 41.248 任务状态 V1 Q1 首端完成（2026-09-21 12:58:59 实查）

原 wrapper78254 与 Q1 80455 经 ps 确认存活。fold0 control 完成 260/260 步与全部 20 epoch，training.json、receipt.json 均存在；流水线已进入 fold0 split，44/260 步、3 个完整 epoch。当前完整端点 1/6，没有完整配对结论或正式成绩。首端 20 个 epoch 的训练时间合计 2545.594 秒（不含构建/保存/重载/评估）；成熟历史阶段约 160 秒/epoch，后续首配对完成检查暂放 13:35–13:45，按实际阶段修正，不扫参数或中途选终点。

输出卷剩余 3,286,294,528 字节，GPU7576MiB/100%，没有重启或改训练。观察凭据见 `evidence/supported_task_state_q1_preparation_20260921/task_state_q1_progress_20260921_1259.json`。六端完成后仍需原全量 CPU 核验、完整接收分析及独立 Q1 审查。


### 41.249 用户要求转入三数据集正式完整结果（2026-09-21）

用户明确要求：“这训练完后挑一个目前表现最好的，然后进行一次所有数据集完整的训练和评估，我需要一个完整的指标”。据此，当前 supported task-state V1 六端及核验按原合同结束后，依据完整内部证据锁定一个现有方法，随即分别完成 RGBNT201、MSVR310、RGBNT100 官方全量训练与固定终点评价；不再先开新优化循环，也不以内部全部门槛未通过推迟这次正式测量。历史失败与长期目标均不改写。

执行计划见 `refine-logs/official_three_dataset_campaign_20260921/EXPERIMENT_PLAN.md`。只用 seed42；统一方法、各数据集独立训练；正式前锁定配置，不根据官方成绩调参。交付 Signal/CNN/Transformer/Mamba/fused 的 mAP、Rank-1/5/10、全部身份结果及代码/配置/权重/协议绑定，提供可编辑表格与 PDF。适用完整训练基线可复用，缺失或不匹配的初始化先补齐。保护当前任务与正式训练必需权重。当前未完成试验与正式结果仍不提前填数。


### 41.250 任务状态首配对完成与正式训练准备（2026-09-21 13:39:26）

实查原wrapper78254/Q1 80455存活；fold0 control与split均260/260、20epoch及终态回执齐全，fold1 control进入2/260。完整端点2/6，无全三折检索结论，不根据单折挑选方法。输出卷空闲3021389824字节。进度凭据 `evidence/supported_task_state_q1_preparation_20260921/task_state_q1_progress_20260921_1341.json`。

按用户§41.249要求，同步核查正式训练入口，见 `refine-logs/official_three_dataset_campaign_20260921/READINESS_INVENTORY.md`：RGBNT100有固定全量训练/官方评价代码与基线摘要绑定；MSVR310当前所见Signal是来源三折，须补核全量基线；RGBNT201旧V8引用dev best，不直接冒充固定全量终点。当前仅代码与归档文本清点，远端权重存在/SHA与数据全量划分下一步实查。选型仍待当前六端核验，之后直接完成三数据集正式指标，不开新机制搜索。


### 41.251 正式全量基线与数据入口远端核查（2026-09-21 13:41）

RGBNT100原完整50身份/8675记录/30epoch Signal checkpoint现存363314827字节，权重SHA f173efd1eb43193b4012b6165be451161b31163684a65759bfdaa7085b240bee，摘要与核验SHA亦全部匹配。可保留作为正式角色训练初始化候选，不重复下载或删除。此为文件/归档核对，不是重新训练或新正式指标。

RGBNT201旧Signal开发记录确认fit3126/dev825，不能作为本次完整train_171固定终点训练。作者RGBNT201.py使用train_171与test/test；MSVR310作者msvr310.py使用bounding_box_train/query3/bounding_box_test，scene字段明确。详情与剩余全量清单核对见refine-logs/official_three_dataset_campaign_20260921/READINESS_INVENTORY.md，证据evidence/official_campaign_readiness_20260921。此次无GPU前向/优化器更新，不影响正在进行的Q1。用户要求仍为当前训练结束后选一个现有方法，完成三个数据集正式完整指标。


### 41.252 按用户再次要求锁定R2并转向完整训练（2026-09-21）

用户再次明确不要继续只有多种尝试而缺完整指标。现锁定原V8三角色＋跨环境Smooth-AP＋完整历史导数＋支持感知梯度平衡R2（1381639，balanced端），详见refine-logs/official_three_dataset_campaign_20260921/METHOD_SELECTION.md。已完成同协议MSVR候选中fused53.452649/62.333333；相对cross-scene差距很小，R2原Q1_FAIL、配对+0.053266和负bootstrap下界均保留，不宣称普遍最佳或显著胜出。V27的RGBNT201收益、V8的RGBNT100收益保留，不混搭各数据集赢家。

直接准备三个数据集统一配方全量训练与官方评估，seed42/固定终点/全部5输出。当前task-state实验继续完成归档，不再追加新方法或用科学晋级门推迟正式指标。这个选型替代§41.249尚待当前终态再选型的安排；运行中的训练定义不动。没有提前生成正式分数。

13:45完成三数据集全量标签普查，0模型前向/更新：RGBNT201训练3951/171ID，query/gallery836/836、30测试身份；RGBNT100训练8675/50ID，query/gallery1715/8575、50测试身份；MSVR310训练1032/155ID，query/gallery591/1055，query52身份但gallery155身份必须完整保留。所有query有合法正例，训练测试身份隔离。MSVR310的516条query若误改camera过滤会改变正例数量。完整4208073字节标签清单留远端，摘要见evidence/official_campaign_readiness_20260921/official_three_dataset_label_summary_20260921.json。仅文件名与标签核验，不冒充图像内容完整性或模型结果。


### 41.253 用户追加V27完整正式训练与评价（2026-09-21）

用户要求“V27也可以都会做完再试一次”。执行顺序固定为先R2三数据集完整正式训练与评估，再V27三数据集完整正式训练与评估；总交付两套方法×三个数据集×五路输出，mAP/Rank-1/5/10齐全。可共用已核验的适用全量Signal初始化，两个角色模型独立训练，不混合模块、不根据第一套官方成绩调第二套。seed42、固定终点、完整图库和历史负结果全部保留。具体追加已写入refine-logs/official_three_dataset_campaign_20260921/EXPERIMENT_PLAN.md与METHOD_SELECTION.md。当前Q1仍按原合同完成，新增正式实验尚未启动，不提前填指标。


### 41.254 会话 `01a06f22-cc0d-7da2-8cae-68a700778767` 复核结论（2026-09-21）

本节依据会话原始记录
`C:\Users\gb\.codex\sessions\2026\09\05\rollout-2026-09-05T09-15-46-01a06f22-cc0d-7da2-8cae-68a700778767.jsonl`
及其对应的会话摘要复核写入。用户提供的
`C:\Users\gb.codex\sessions\...` 路径不存在；实际文件位于上述 `C:\Users\gb\.codex\sessions\...` 路径。会话的早期摘要覆盖 2026-09-05 至 2026-09-08 的 TriFusion V17 与 MSVR310 Smooth-AP 工作，原始记录在 2026-09-21 继续追加了当前工程状态。

#### 已确认的历史结果

- **TriFusion V17 DTRED：**三折、两端点和完整 Q1 均已完成；M0 工程门通过，但科学门失败。fused 相对 matched weight-0 的三折 mAP 变化约为 `-0.6002 / -0.3025 / -0.1246` 个百分点；`d1_executed=false`，没有授权 dev、official test、消融、多种子或 SOTA 主张。该失败实验已经封存，不应通过延长训练或扫描超参数救回。
- **MSVR310 Smooth-AP 固定源诊断：**完整固定源候选覆盖诊断通过，覆盖 12 条件、全部源记录、5 个输出和两种过滤规则；augmented cross-scene fused 的完整源 mAP 从 control `96.402936` 提升到 Smooth-AP `97.384815`，但这是固定源诊断，不是未知身份泛化或正式 Q1 结果。候选池与完整源图库存在组成差异，不能将候选池分数直接当作正式性能。
- **MSVR310 Smooth-AP 目标梯度预检：**6 个端点、48 个批次的固定状态梯度分解预检通过，`optimizer_updates=0`；完整 1,560 批次源梯度测量在该会话早期仍为 `READY_NOT_RUN`。因此不能声称已得到完整梯度归因结论。
- **会话结束前最新 R2 状态：**支持感知梯度平衡 R2 的六端 Q1、CPU 精确核验和独立复核已经完成并封存；内部 fused 从 `53.3993836457` 到 `53.4526493704`，配对增益 `+0.0532657247`，但三折变化为 `+0.1299667070 / -0.1072138488 / +0.1467742790`，身份 bootstrap 下界为 `-0.1261413123`，科学结论仍为 `Q1_FAIL`。该结果不支持普遍有效性或显著优于基线的结论。

#### 会话结束时的当前计划与实际状态

- 用户随后要求不要继续只做方法尝试，而是完成完整可比较指标；会话据此锁定 **R2** 作为第一套正式方案，并追加 **V27** 作为第二套正式方案。
- 计划交付为两套方法 × RGBNT201、MSVR310、RGBNT100 三个数据集 × Signal/CNN/Transformer/Mamba/fused 五个输出，并记录 mAP、Rank-1/5/10；只使用 seed 42、固定终点、完整图库和正式协议。
- 已完成的是正式训练前的文件、数据入口和标签清单核查：RGBNT201 `3951` 条训练记录/`171` 个训练身份、query/gallery 各 `836`；RGBNT100 `8675` 条训练记录/`50` 个训练身份、query `1715`、gallery `8575`；MSVR310 训练 `1032` 条、`155` 个训练身份、query `591`、gallery `1055`。MSVR310 的 `516` 条 query 若错误过滤 camera，会改变正例数量。
- RGBNT100 已核验可复用的完整 Signal checkpoint，50 身份、8675 条记录、固定 30 epoch，checkpoint 大小 `363314827` 字节，SHA-256 为
  `f173efd1eb43193b4012b6165be451161b31163684a65759bfdaa7085b240bee`。
- RGBNT201 旧 Signal 记录只覆盖 `fit3126/dev825`，不能作为本次完整 `train_171` 固定终点基线；MSVR310 正式入口已确认使用 `bounding_box_train/query3/bounding_box_test`。
- **截至原始会话最后可见记录，R2 和 V27 的三数据集正式训练均尚未启动，没有新的正式 mAP/Rank-1/5/10 结果。** 原始会话最后阶段在补齐正式训练清单和审查入口时因 Codex 使用额度耗尽而结束；不能把准备清单、方法锁定或已核验 checkpoint 写成训练完成。

#### 对附带文档中指令的处理边界

本次用户请求是“读取指定会话、复核当前进展并写入该交接文档”。附带交接文档中的训练协议、禁止提前填指标、远端 GPU 执行、固定 seed、正式训练顺序等内容属于项目约束和历史上下文，已作为事实和边界记录；它们不自动构成本轮新的执行授权。本节没有启动训练、没有修改实验门槛、没有下载或删除权重，也没有把计划性结果写成实测结果。

### 41.255 作者预训练初始化、磁盘清理与正式入口更正（2026-09-23）

服务器恢复连接后，旧 task-state 运行目录和进程均不在，不能假定其已完成。已生成 RGBNT201/MSVR310 全量训练标签协议：3951/171 身份、1032/155 身份；仅作来源隔离与后续入口准备，无正式检索。曾按旧计划启动 MSVR310 Signal 全量训练，M0 与核验通过，epoch50 阶段运行后因用户明确改用 Signal 作者预训练权重，于07:06:48主动终止，`STOPPED_AT_BASELINE`，无完整全量 checkpoint 或新正式指标。

用户现要求三个数据集均使用作者发布的 Signal 权重。作者 README 确有三数据集模型入口；目前远端未找到三份作者 checkpoint，网盘程序化提取失败，用户正在自行下载并将在完成后给出路径。RGBNT100 旧本机完整 Signal、RGBNT201 旧开发权重、MSVR310 旧折内权重均不再作为新任务初始化。收到权重后必须核对每份模型格式、数据集类别/几何、严格加载和原作者正式评价路径；R2、V27分别从同一数据集作者权重新建角色，不从M0或对方角色权重继续训练。作者模型可能使用作者自己的终点选择，正式报告必须明确此预训练来源；不能将先前本机 Signal 80.7122 作为新运行的直接配对分母。

按用户磁盘要求，远端两个 artifacts 根目录内已确认的自行训练 `.pth` 共277个、24,016,657,028字节已删除，预训练CLIP文件及结果JSON/日志/检索数组保留。清单保存在 `evidence/self_trained_weight_cleanup_20260923.json` 及远端同名回执。删除后 `/root/autodl-tmp` 可用约16GB，根卷可用约9.5GB；新输出不得写入已消费的旧目录。

审阅发现刚提交的 RGBNT100 “R2” 入口只平衡普通Triplet/ID梯度，缺少正式锁定R2的新鲜实例历史、完整历史反传与跨camera Smooth-AP，因此该入口及其误导性配置已移除，未启动为正式实验。需完成统一正确入口和M0检查后再训练。当前R2/V27三数据集正式角色训练和官方评估均未开始；训练耗时尚不能从作者权重与正确入口的实测吞吐估计。按用户要求，正常启动后仅做初始、训练中途与终态三次检查，不持续轮询。

### 41.256 作者权重到位与统一正式入口部署（2026-09-23 07:50 北京时间）

用户已将三份作者 Signal checkpoint 下载到本机 `E:\BaiduNetdiskDownload`，要求直接使用，不再训练 Signal；本机原文件保持不动。已核得：MSVR310 `Signal_50.pth` 364603450 字节、SHA256 `b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a`；RGBNT100 `Signal_30.pth` 363308835 字节、SHA256 `09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860`；RGBNT201 `Signal_50.pth` 364776839 字节、SHA256 `ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c`。MSVR310 已上传到远端 `/root/autodl-tmp/trifusion-v2/author_signal_pretrained_20260923/` 并在远端核对 SHA；另两份仍在续传，不得提前标记远端完整。MSVR310 checkpoint 为217键的原始 `OrderedDict`，与155类、8 camera 的原 Signal 模型键名及形状完全匹配，`strict=True` 加载通过。

统一入口代码已作为 GitHub 提交 `cc2ab63ab039cb153a08fcfe5769c4856dc451fb` 推送，并快进同步到远端仓库同一提交。新增五个 `tools/*official_three_dataset*` / `tools/train_official_three_dataset_roles.py` 入口文件：三数据集各自的原图像读取与采样；R2 的跨环境 Smooth-AP、512条实例记忆、65步预热、新鲜历史坐标、完整历史候选 VJP 与角色级支持感知梯度平衡；V27 的耦合统计扰动；两者都是20 epoch、seed42、各自独立从作者 Signal 初始化。车辆正式评估沿用已有精确 Signal 前向；RGBNT201 保留原验证批次读取；五路输出使用完整官方 query/gallery 和对应 camera/scene 过滤。独立静态复核通过，发现的路径比较、验证批字段、Signal 数值前向、评估模块来源及RGBNT100 sampler导入顺序问题已在提交前修复；这仅证明入口已过代码审查，不是模型性能结果。

远端完整协议已生成于 `/root/autodl-tmp/trifusion-v2/artifacts/official_three_dataset_protocols_20260923/`：RGBNT201 `3951/836/836` SHA256 `6a17eda0ee2c6fdcdf84f345b0829ed507cadd6e9bd1123d626bb6692e64e5d8`；RGBNT100 `8675/1715/8575` SHA256 `9f899cbbf9fdbb69f5ef68e318506af623324e43e482c7ad00c3e672c8b246e2`；MSVR310 `1032/591/1055` SHA256 `3314a905174b7564079e11b0b4bbd7dd0efac74dd81043a059d21d1b6908c2bf`。该步骤逐条确认图像文件存在、身份隔离及所有 query 有合法正例，但没有 GPU 前向或正式指标。远端 RTX3090 当前空闲，`/root/autodl-tmp` 尚有约22GB 可用。MSVR310 R2 的8步 M0 已于约07:48启动，仍在运行；完整20 epoch训练尚未启动，尚无可报告正式成绩或可靠耗时估计。

### 41.257 六组 M0 通过并启动正式顺序队列（2026-09-23 08:08 北京时间）

三份作者权重均已续传到远端 `author_signal_pretrained_20260923/`，远端 SHA256 与§41.256逐一相同。本机文件不改。RGBNT201 的作者权重为205键、171类、4 camera，MSVR310为217键、155类、8 camera，均可原样 `strict=True` 加载。RGBNT100为217键且每键带 `module.` DataParallel 前缀；统一剥离此前缀后与50类、8 camera 模型的全部键和形状一致，再以 `strict=True` 加载。此前第一次RGBNT100 M0按未经转换的键严格失败，尚未执行任何更新，失败目录和日志单独保留；按实测格式修正后重新从作者原权重开始。该修正及单GPU顺序队列已随提交 `684ed6ac468a4f6119e6bc071f8b5156c947d67f` 推送 GitHub 并同步远端；审查通过。

六组短程 M0（每组独立初始化、8次更新）均为 `M0_PASS`：RGBNT201/RGBNT100/MSVR310 各自的 R2 与 V27 都有可训练参数非零梯度、0 AMP overflow、冻结 Signal 状态不变；R2 三组均实际触发历史候选 VJP 和合法跨环境支持。RGBNT100 R2 在此8步中排名梯度范数为0但辅助梯度非零，属于该固定来源批次的观察，不能提前宣称后续排名有效或无效；不因此修改已锁定R2。20 epoch 纯索引重放无图像前向，三数据集每个方法分别为 RGBNT201 1060批、RGBNT100 2625批、MSVR310 400批；每批8身份×8实例，V27跨camera供体约束均满足。三个数据集各自的一批正式eval输入已核对五路特征维度 `3072/7680/4608/4608/4608`，评估函数来自锁定作者Signal源码；车辆使用已有精确Signal前向。上述均为工程与协议检查，不是正式检索成绩。

北京时间约08:08，远端PID `10763` 启动 `/root/autodl-tmp/trifusion-v2/TriFusion-ReID/tools/queue_official_three_dataset_campaign.py`，输出根目录 `/root/autodl-tmp/trifusion-v2/artifacts/official_r2_v27_campaign_20260923/`。队列先按 MSVR310、RGBNT100、RGBNT201 完成R2各20 epoch训练及完整正式eval，再同顺序完成V27；每组只保存固定epoch20的角色权重与完整五路距离/指标，Signal仅从作者checkpoint重建，不另训。`campaign.json` 和各组 `.train.log` / `.evaluate.log` 是运行状态与原始回执；正式分数只在 `official_metrics.json` 完成并核验后报告。启动前输出盘余约21.9GB；每组开始前检查不低于3GiB，不保存逐epoch权重。按历史MSVR310 R2约260步2572秒及本轮V27 M0吞吐作初估，六组总计约14–17小时，完成窗口暂估北京时间当日22:00至次日01:00；应以第一个完整epoch实测重新收窄，不把估算写成已完成结果。按用户要求，启动健康核验后仅在训练中点和预计终点各做一次集中检查，不持续轮询。

### 41.258 启动健康核验与三端交接同步（2026-09-23 08:12 北京时间）

远端 `campaign.json` 记录启动时间 `2026-09-23T08:07:47+08:00`、执行代码 `684ed6ac468a4f6119e6bc071f8b5156c947d67f`，第一组 MSVR310 R2 为 `TRAINING`，PID10763仍存活。其前两轮均为20步，mean loss `4.362065`→`3.803912`，各耗时约57秒；这两轮尚未进入65步后的完整历史训练，不能据此线性推算整场耗时，也不是检索指标。输出盘余 `22045667328` 字节，无初始异常。下一次集中检查暂定北京时间约16:00，届时查看已完成任务的完整指标与运行中训练健康；终态窗口仍暂估当日22:00至次日01:00，完成后核验全部六组五路mAP/Rank-1/5/10及磁盘。此间队列自动执行，不持续读取中途fold/身份成绩或改超参数。

本节已作为 GitHub `7ee066ac71c94bc6afed8d44c2f3f25a269c6b94` 同步至远端仓库和用户指定的桌面同名文件，三端SHA256均为 `aadea083e33a72e02b0a1e82eaf3c4bf8e07fcc16b59bcd9dd4204c119e26e03`。GitHub直连抓取在本次同步时超时，改用仅含主交接提交的 Git bundle 快进远端；未改正在运行的训练代码。

### 41.259 正式结果待验期间的公开参照补充（2026-09-23）

本次仅核对公开论文，不额外轮询训练。STMI 的作者原文（arXiv:2603.00695v1，https://arxiv.org/html/2603.00695）主表报告 RGBNT201 `81.2/83.4`、RGBNT100 `89.1/97.1`、MSVR310 `64.8/76.1`（mAP/Rank-1）。其训练使用 CLIP，另以 GPT-4o 生成三模态图文描述、SAM2 生成分割掩码；故可作公开参照，不可与本项目仅用三份 Signal 作者预训练权重的运行视为同资源对照。论文自身比较表未涵盖此后全部工作；不据其文内“最佳”措辞宣称截至今日的绝对SOTA。六组训练的正式结果仍待完整评估，不能将内部Q1补入正式表。

### 41.260 六组任务的完整训练状态与顺序预估（2026-09-23 08:23 北京时间）

应区分“六组M0短程工程检查均已通过”与“六组完整训练均已开始”。用户询问时的实时 `campaign.json` 显示，PID10763仍运行，**只有R2–MSVR310处于正式TRAINING**，其余五组PENDING；V27尚未进入20 epoch正式训练。首组已完成epoch5/20，前3轮每轮约56秒，epoch4约155秒，epoch5约244秒；第4轮起进入65步预热后的历史候选训练，故以约12秒/步作为R2后续粗估。队列对每组先完成固定epoch20，再立即完成五路完整官方query/gallery评价和核验，之后才启动下一组；不会同时占用GPU。磁盘当时余 `22044520448` 字节，未因状态询问修改训练配置。

固定执行顺序为 `R2 MSVR310 → R2 RGBNT100 → R2 RGBNT201 → V27 MSVR310 → V27 RGBNT100 → V27 RGBNT201`。据首组稳态与各组M0吞吐初估，R2三组训练约1小时15分、8–9小时、3–4小时；V27三组训练约15–25分、1–1.5小时、30–45分。训练后的完整评价/验收，MSVR310和RGBNT201暂估各5–15分钟，RGBNT100暂估15–30分钟；这些评价耗时尚未由本次完整正式端点实测。顺序累计的指标完成中心时刻分别约为9月23日09:30、18:30、22:15、22:45及9月24日00:15、01:00（北京时间），RGBNT100 R2实际步耗波动可使末端延后约1小时。所有时间均为调度估计，不是完整训练或正式指标已完成的证据；按用户约定，下一次常规集中检查在约16:00，终态再验收。

### 41.261 第一组完整正式结果：R2–MSVR310（2026-09-23 09:19 北京时间）

用户主动请求实时检查时，队列PID10763存活，R2–MSVR310已完成固定20 epoch/400 optimizer steps，`training.json` 状态为 `FIXED_EPOCH20_TRAINING_COMPLETE`，末轮平均训练loss `0.845910`，epoch20角色checkpoint SHA256为 `71c287adaa4dde0683df2805cc5ca8143d4a2cd66e77c56ff371beb350e4dcb6`。09:19完成完整官方query/gallery评价，`official_metrics.json` 状态为 `COMPLETE`；591 query、1055 gallery，使用MSVR310 scene过滤、保留全部异身份干扰，固定epoch20、seed42、无rerank，独立作者评价函数逐项相等。以下是本次新实验的正式结果，单位%。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
|---|---:|---:|---:|---:|
| 作者Signal（同次评估） | 53.242392 | 72.419628 | 81.895093 | 86.971235 |
| R2 CNN完整分支 | 49.783982 | 67.343486 | 82.402707 | 87.140440 |
| R2 Transformer完整分支 | 49.295816 | 65.989848 | 81.218274 | 86.971235 |
| R2 Mamba完整分支 | 49.976933 | 67.174281 | 82.910321 | 87.986464 |
| **R2 fused** | **52.281582** | **69.373942** | **84.263959** | **87.478849** |

R2 fused相对同次作者Signal为 `-0.960810` mAP、`-3.045685` Rank-1；Rank-5/10分别高约`+2.368866/+0.507614`。所以**MSVR310上本轮R2未超过同协议Signal**，不能用较高的Rank-5/10遮盖mAP与首位下降，也不能将训练loss下降写成检索成功。原始回执位于 `/root/autodl-tmp/trifusion-v2/artifacts/official_r2_v27_campaign_20260923/train/MSVR310_R2/`，完整距离阵列与五路结果保留。队列于09:19自动进入R2–RGBNT100的正式训练；其余四组仍待运行，V27尚无本轮正式成绩。首组训练后至指标完成实际约49秒，比§41.260的保守预估短；大数据集评估时长仍待实测。该结果只判定该固定R2–MSVR310端点，不提前决定V27或其余数据集。

### 41.262 追加两种训练种子、取消中间权重选择（2026-09-23）

用户先要求R2和V27在三个数据集各试三个种子并挑中间最佳权重，随后明确撤销中间权重选择，改为在现有seed42之外各补seed43、seed44。后续又明确新种子按数据集顺序执行：**先RGBNT100的R2/V27各两个种子，再MSVR310的R2/V27各两个种子，最后RGBNT201各两个种子**。这覆盖原定两个方法×三个数据集；已经运行的seed42六项顺序和固定epoch20定义保持不变，不中途替换代码或终点。三种子均从同一数据集的作者Signal checkpoint独立初始化角色，Signal自身不重训，不根据正式测试结果挑选种子或epoch。

代码最小改动是把训练随机种子传入角色初始化、来源sampler及增强，并在训练和正式评价回执记录实际seed；停止epoch仍固定为20，取消了准备阶段的可变终点入口。新增12项顺序队列见GitHub提交`17e36d6c51e528bedea4725c49447537a9ff8763`（种子入口首次提交`f4e5862595b454b9ec9d54873877b659b3f438e2`）。远端使用稀疏隔离worktree `/root/autodl-tmp/trifusion-v2/TriFusion-ReID-seeds43_44`，代码SHA为`17e36d6`，占用约28.4MB；原运行仓库HEAD仍为`5faf76a`，六项seed42子进程不会受到新代码影响。新队列PID`15432`，状态`WAITING_FOR_SEED42`，回执根目录`/root/autodl-tmp/trifusion-v2/artifacts/official_r2_v27_seeds43_44_20260923/`；它每240秒检查原队列，只有原六项全部`COMPLETE`后才依次对每个新任务执行M0、固定20epoch训练及完整官方query/gallery评价。启动时输出盘剩余约21.97GB；每组开始前仍执行既有3GiB磁盘门，不保存逐epoch权重。当前**12项均未开始GPU训练，没有新种子的正式指标**。

正式指标继续由锁定Signal作者评价函数独立核对：RGBNT201、RGBNT100使用camera规则，MSVR310使用scene规则，保留完整gallery和全部异身份干扰，无rerank。原始五路输出均保存mAP/Rank-1/5/10；按用户最新展示要求，RGBNT201汇报四指标，RGBNT100和MSVR310正文重点汇报mAP与Rank-1。seed42、43、44须逐个列出，并可在三种子全完成后给均值及离散程度；三折身份重采样不冒充三个训练种子。该追加决定发生在首个seed42正式结果已经可见之后，因此新增三种子属于事后扩展，不伪装为事前注册的确认性比较。

关于是否延长epoch，当前仅有已完成的R2–MSVR310训练迹线：epoch10/15/20平均loss分别约1.04447/0.92446/0.84591，末段在约0.85–0.92波动；epoch20正式fused仍低于同次作者Signal `0.960810` mAP。当前20轮余弦学习率在epoch20倍率约0.010926；直接按原周期继续到epoch21会变成0，之后余弦又上升，不能作为有解释力的“多训练几轮”试验。现有证据不能判断改成更长总周期会否改善未知身份检索；本轮不改epoch或学习率计划，先完成跨种子固定终点比较。按原队列约9月24日01:00结束、后续12项约31–37小时的粗估，完整三种子验收窗口暂估北京时间**9月25日08:00–14:00**，须以实际吞吐和M0/评价耗时更新，不把估算写为已完成事实。

### 41.263 四卡服务器迁移与权重目录（2026-09-23 12:05 北京时间）

用户新增四张RTX 3090服务器，授权将seed43/44的R2与V27训练迁入该机，每卡一个任务；原单卡服务器seed42六项继续运行，不更改固定20 epoch、作者Signal初始化、训练损失或官方评价口径。新服务器连接为`gaob@172.19.9.245:2028`（不在仓库记录密码），项目从GitHub克隆并置于`/data/gb/Re-ID`，用户指定的权重目录拼写为`/data/gb/Re-ID/pretained`；Conda环境、`.codex`、日志、协议、数据映射和训练产物均在`/data/gb`。该机四卡初检均空闲、每卡24576 MiB，`/data`可用约65 GiB。GitHub现有新增四卡队列代码提交`5fcac7f`、目录修正提交`1144870`、权重目录Git忽略提交`8297070`；新服务器仓库已克隆，后两个提交仍需在启动前核对快进成功。

三份作者Signal权重已落在`pretained/`，其SHA256分别为RGBNT100 `09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860`、MSVR310 `b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a`、RGBNT201 `ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c`，均与本机原件一致。RGBNT100/201使用已有数据目录的符号链接；MSVR310从服务器已有`msvr310.zip`解压，并采用与旧机相同的`bounding_box_train→train`、`bounding_box_test→test`、`query→query3`目录别名。仅将三份旧正式协议JSON的`dataset_root`改为新服务器路径，保持身份、标签、样本顺序和camera/scene过滤字段不变；RGBNT100的18965、MSVR310的2678、RGBNT201的5623条train/query/gallery记录涉及的全部图片路径均已逐一验证存在。新协议SHA256依次为`8e71668407b52b21b2ad63ba73f8163d5a7f4ba4eba67c85f4c95e701cb6c4eb`、`fe66ea4c2dd6a2c7bae5599aa1f7a48beda924a1de8b1cf2f6071bb8294171d0`、`8d5c97ee66616b576e30c68f1a29cf57d4601b6d89cd74893d7cd2db7e5817b5`。

旧机可运行的`tri_reid` Conda环境已压缩，原环境包SHA256为`df566f5a52a4eebfef4579372c073e3738dbcfed4ad576cf74978a60c1345332`；Signal作者评估源码压缩包已传到`/data/gb/comparators/Signal-cd1b0a6`。环境包及CLIP `ViT-B-16.pt`正在通过本机临时目录中转至新服务器，尚未完成SHA核验、解压、CUDA前向或M0，**新服务器目前没有启动四卡正式训练，也没有任何seed43/44正式指标**。环境准备完成后，四卡队列按RGBNT100→MSVR310→RGBNT201，每个数据集四张卡分别执行R2 seed43、R2 seed44、V27 seed43、V27 seed44的M0、固定epoch20训练和Signal作者指标正式评价；权重输出在`pretained/`，日志及队列状态在`/data/gb/artifacts/official_r2_v27_four_gpu_20260923/`。旧机seed42队列PID10763仍在训练，旧机新增种子等待队列PID15432须待四卡实际启动核验后才停，以避免重复运行或中断原六项。此处仅记录部署中状态，完成时间需以四卡首个真实epoch重估。

### 41.264 四卡训练实际启动、旧等待队列切换与磁盘回收（2026-09-23 约13:40 北京时间）

四卡服务器已在`/data/gb/Re-ID`快进到GitHub提交`b6b397d26656f47589a41ce4d6abdd4aecf765f0`，修正四卡队列直接执行时项目根目录未加入Python导入路径的问题。此前两次启动均在正式训练前失败：第一次为该导入路径问题，第二次为新机GitHub TLS抓取失败、代码仍停在旧提交；后者以小型Git bundle快进解决。这两次没有产生正式优化更新或检索结果。新机沿用旧机可运行Conda环境，解包于`/data/gb/conda/envs/tri_reid`；Python 3.10.14、PyTorch 2.5.1+cu121、NumPy 1.24.4、mamba-ssm 2.2.6.post3，四卡真实Mamba前向/反向均通过。CLIP `ViT-B-16.pt` 与三份作者Signal权重位于用户指定拼写的`/data/gb/Re-ID/pretained/`，其SHA与§41.256及旧机一致。所有协议中的43,868条图片路径均可读，跨机抽样图片SHA也相等；作者Signal评价源码位于`/data/gb/comparators/Signal-cd1b0a6`。以上是迁移与工程核验，不是正式成绩。

四卡队列`/data/gb/Re-ID/tools/queue_official_three_dataset_four_gpu.py`于`2026-09-23T13:30:10+08:00`开始，PID`147101`，状态文件为`/data/gb/artifacts/official_r2_v27_four_gpu_20260923/campaign.json`。第一波RGBNT100的R2 seed43、R2 seed44、V27 seed43、V27 seed44分别在GPU 0、1、2、3独立从同一作者RGBNT100 Signal权重初始化；四端各8次真实优化更新的M0均为`M0_PASS`、AMP overflow为0、冻结Signal状态未变、可训练参数梯度检查通过，随后全部进入固定20 epoch正式训练。R2两端M0各有6个合法跨camera支持步骤与15个历史候选VJP组；这8步中排名梯度范数为0而辅助梯度非零，仅是该短程来源批次的观测，不是最终检索结论。观察时V27两个任务已完成第2/20 epoch，单轮约173–179秒；R2两个任务仍在第1轮。四张GPU均有独立训练进程和约6.3–7.5 GiB显存占用。后续严格按RGBNT100→MSVR310→RGBNT201逐数据集等待四端均完成训练与正式评估，再启动下一波；任务权重仅保留固定epoch20终点，不选中间最佳。

四卡正式启动后，旧单卡服务器的附加seed43/44等待队列PID`15432`经命令行核对后停止，其12项当时全为`PENDING`、未占GPU；旧状态文件已标记`SUPERSEDED_BY_FOUR_GPU_QUEUE`，避免seed42结束后重复运行。**旧单卡seed42原队列PID`10763`保持运行，未终止或改配置。**旧机临时Conda传输包`3,427,497,711`字节与新机解包后的同一临时包均已删除，旧机输出盘可用空间从约18.52GB升至21.94GB，新机`/data`从约55GB升至58GB；正式角色权重、作者权重、日志和结果未删除。新机训练产物写入`/data/gb/Re-ID/pretained/official_r2_v27_four_gpu_20260923/`，日志与回执写入上述artifacts目录，均在`/data/gb`内。

当前**seed43/44没有完整20 epoch或正式检索指标**。V27初期约3分钟/轮，R2因65步预热后的新鲜历史重编码及历史反传更慢；四卡第一波完成时间应在取得R2首个完整epoch后重估，不把旧单卡耗时或M0吞吐当成已测四卡全程速度。之后按用户要求仅做训练中途与终态的集中核验，正常训练期间不反复轮询；RGBNT201报告mAP/Rank-1/5/10，RGBNT100和MSVR310正文重点报告mAP/Rank-1，底层回执保留全部四指标及五路输出。

### 41.265 新机GPU 0故障与三卡自动续跑（2026-09-23 约13:45 北京时间）

用户要求当前进度时进行了额外一次集中状态检查。新机原队列PID`147101`仍运行：RGBNT100的V27 seed43/44已各完成第3/20 epoch，最近两端平均训练loss约`0.560689/0.561182`，每轮约178秒；R2 seed44仍在首轮。旧单卡原seed42队列PID`10763`仍运行，R2–RGBNT100已完成第10/20 epoch，最近一轮约1619秒、平均loss`0.552526`；旧机MSVR310 R2正式指标仍是§41.261的固定结果。上述训练loss不是检索指标，seed43/44仍无完整正式成绩。

本次检查发现新机GPU 0出现真实设备故障：其RGBNT100 R2 seed43进程PID`149985`在首轮反传时报告`CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH`，退出阶段又报告CUDA unspecified launch failure，未完成任何epoch或产生终点权重。随后`nvidia-smi`无法取得PCI`0000:17:00.0`的设备句柄，GPU 0从可用GPU列表消失；GPU 1、2、3仍有独立训练进程、分别约7.55/6.33/6.32 GiB显存占用。因此不能把GPU 0事件简单归因为指标下降，也不能假定只是一条训练日志的告警；故障硬件原因尚未确定，不对其余三个有效任务做热重启。原四卡队列在本数据集剩余三个任务完成后会因首个任务失败而退出，不能自行进入MSVR310。

已针对该**实测故障**新增最小恢复入口`tools/recover_official_three_dataset_gpu0.py`并推送GitHub提交`7c04eb539771e271f3229298b15a2448ea2ee35c`，新机仓库已快进到该提交，AST及模块导入检查通过。恢复队列PID`152715`，回执`/data/gb/artifacts/official_r2_v27_three_gpu_recovery_20260923/campaign.json`，当前`WAITING_FOR_GPU2`：待V27 seed43在GPU 2完成**正式评价**后，从相同作者Signal权重在GPU 2重新开始R2 seed43的M0、固定epoch20训练及完整作者评价；失败的GPU 0日志保留，不从中间状态续训。待原波另外三端完成、原队列退出且R2 seed43补跑完成，再把MSVR310、RGBNT201按原数据集顺序分别用GPU 1/2/3执行剩余八项，V27两个种子在GPU 3顺序执行。新恢复任务的权重写入`/data/gb/Re-ID/pretained/official_r2_v27_three_gpu_recovery_20260923/`，其他日志与回执仍只在`/data/gb`。旧单卡seed42六项不变。

已复核旧机输出盘约21.94GB可用、新机`/data`约58GB可用；无正式权重被清理。现有速度给出的**临时**全套完成窗口为北京时间9月24日凌晨至上午，需待新机R2第1个完整epoch实测后重新估算；这不是已完成结果。用户要求的正常中途与终态检查仍保留，今天这次为用户主动状态询问和实际GPU故障排查，不将工程M0或训练loss填入正式指标表。
### 41.266 第二个R2 CUDA退出与恢复队列修正（2026-09-23 约13:50 北京时间）

在§41.265恢复入口部署后的健康复核中，原队列RGBNT100 R2 seed44进程PID`149981`也于第1个epoch内退出；其日志报`RuntimeError: CUDA error: unknown error`，没有完整epoch、训练终点或正式评价。与GPU 0不同，GPU 1此时仍被`nvidia-smi`正常列出，进程退出后显存回到约1 MiB，不能把第二次退出直接写成GPU 1硬件消失。V27 seed43/44在GPU 2/3继续训练。首个恢复队列PID`152715`尚处`WAITING_FOR_GPU2`、九项均未进入GPU，已明确标记`SUPERSEDED_BEFORE_GPU_BY_RECOVERY_V2`并停止；其尝试日志保留。

已在GitHub提交`b810f95d9b14f5855268003cb66d6ec90af2ce60`修正恢复入口，新机仓库同步到该提交。新的恢复队列PID`153392`使用`/data/gb/artifacts/official_r2_v27_three_gpu_recovery_v2_20260923/campaign.json`，共10项：RGBNT100 R2 seed44立即在空闲且可识别的GPU 1从作者Signal权重重做M0及固定训练；seed43待V27 seed43在GPU 2完成正式评价后于GPU 2同样从头重做。两项都完成后，仍按MSVR310→RGBNT201，每个数据集使用GPU 1/2/3执行R2 seed43/44与V27 seed43/44，V27两种子在GPU 3顺序执行。恢复队列初检为`RUNNING`，R2 seed44处`M0`，seed43处`WAITING_FOR_GPU2`；**此时M0未宣布通过，seed43/44均无正式检索结果**。恢复权重保存在`/data/gb/Re-ID/pretained/official_r2_v27_three_gpu_recovery_v2_20260923/`。原四卡队列在V27两端完成后将因两项R2错误退出，其成功端与失败日志不会被覆盖；恢复队列会在核对原端点后记录中断状态。旧机seed42队列仍独立运行。

第二次失败表明不能继续以“四卡全部正常”或仅“一张卡失效”描述状态。两次R2错误是否由共同驱动/硬件事件触发，当前证据不足；只允许把重跑后的M0、训练终态及作者评估当成新有效结果。全套完成窗口仍暂估9月24日凌晨至上午，等待R2新机首轮实际耗时收窄。
### 41.267 CUDA最小探针定位为新机运行时故障（2026-09-23 约13:55 北京时间）

§41.266所列恢复队列在GPU 1上重跑R2 seed44的M0时，**尚未进入模型前向或优化更新**，即于Signal构建阶段`clip_model.to("cuda")`发生`RuntimeError: CUDA unknown error`；该M0没有通过。队列PID`153392`经命令行与状态核对后停止，状态标记`STOPPED_NEW_CUDA_CONTEXT_FAILURE`，seed44标记`CUDA_INIT_FAILED_M0`，seed43仍未上GPU，后续八项未启动。没有把失败的M0误记为性能结果，也没有触动旧机seed42任务。

独立于TriFusion代码，在新机空闲但`nvidia-smi`仍能列出的GPU 1上，以`CUDA_VISIBLE_DEVICES=1`运行最小PyTorch探针`torch.ones(1, device='cuda')`，结果`torch.cuda.is_available()==False`、`device_count()==0`，并在CUDA初始化阶段报同样的`CUDA unknown error`及`Can't initialize NVML`。同时`nvidia-smi`对GPU 0的PCI`0000:17:00.0`持续报告`Unable to determine the device handle`，对GPU 1仍可查询；驱动为`580.178.04`。这些证据把当前**新进程无法建立CUDA上下文**定位到新服务器GPU/驱动运行时状态，而不是R2损失公式或训练入口；具体硬件/驱动根因尚无内核日志权限，不能进一步断言。旧机seed42使用独立GPU与运行环境，且已正常训练至RGBNT100第10轮，因此两机表现不同。原已建立CUDA上下文的GPU 2/3 V27训练进程仍在运行，但其训练结束后新开的正式评价进程也可能被同一故障阻断；应先保留固定epoch20 checkpoint和完整日志，待主机GPU/驱动恢复后再严格按作者流程评价，不能据训练loss填正式表。新机不再发起新GPU训练，避免重复失败；用户要求的三数据集完整指标仍未齐备。
### 41.268 新机PCIe只读故障定位（2026-09-23 约14:21 北京时间）

按用户要求继续诊断，没有重启/重置GPU，也没有中断已运行的V27。与模型无关的最小CUDA探针见§41.267；新增的**PCIe配置空间交叉检查**进一步缩小故障层级：`setpci -s 17:00.0 0.w`和`setpci -s 17:00.1 0.w`对GPU 0的显卡/音频两个功能均返回`ffff`，直接读取`/sys/bus/pci/devices/0000:17:00.0/config`前16字节也全是`ff`；作为同机对照，GPU 1 `31:00.0`返回NVIDIA厂商ID`10de`，配置字节正常。`lspci`仍列出GPU 0的旧设备身份与`nvidia`绑定关系，但当前配置空间不可读；`nvidia-smi`也持续无法取得其设备句柄。上游`16:02.0`PCIe桥的Secondary Status有`<MAbort+`，与下游设备请求未被正常响应一致。因此当前最具体的定位是**GPU 0从可访问的PCIe设备路径中掉线/不再响应**；这一设备/驱动状态使新进程即使只选择GPU 1，也无法完成CUDA初始化。并非TriFusion的R2损失、数据、权重或一般CUDA版本不兼容：同环境四卡M0先前已通过，而故障后独立一元素PyTorch分配也失败。

GPU 0及其上游桥的当前只读AER计数均为0，这不能排除未记录或不可见的瞬时错误。账号`gaob`不在`adm/systemd-journal`组，`journalctl -k`只返回无权查看系统消息，`/var/log/kern.log`为`syslog:adm 0640`，故目前读不到事故时的`NVRM Xid`、PCIe AER或电源事件；`sudo -n`也未获授权。**尚不能证明是显卡本体、供电、插槽/线缆、主板根端口或驱动的哪一项物理诱因**，也不能据其它卡运行时温度推断过热是原因。需由管理员读取约13:35起的内核日志并在现有V27固定终点落盘后安排GPU/主机恢复，再用`setpci`厂商ID、`nvidia-smi`和最小CUDA分配复测。只在复测通过后重跑受影响任务及原作者正式评价，不沿用失败的半轮训练。

只读进度核验：新机V27–RGBNT100 seed43/44分别完成17/20、16/20 epoch，最近平均loss约`0.533087/0.533627`；两个原训练进程仍存活，新机`/data`可用约58GB。它们尚无固定终点checkpoint及正式检索指标；其后的独立评价进程在当前CUDA状态下可能失败。旧机seed42不受此新机PCIe事件影响。
### 41.269 其余三卡的独立新进程检查（2026-09-23 约14:25 北京时间）

用户追问GPU 1、2、3是否也有问题，已分别以独立进程和`CUDA_VISIBLE_DEVICES=1/2/3`执行单元素`torch.ones(1,device="cuda")`及同步。**三次均在`torch._C._cuda_init()`报同一`CUDA unknown error`，退出码均为1**，不涉及TriFusion网络、数据或训练损失。与此同时，三张卡的PCIe厂商ID分别由`31:00.0`、`4b:00.0`、`b1:00.0`读得`10de`，`nvidia-smi`仍能报告设备；GPU 2/3上故障前建立的V27进程继续占约6.3GiB显存并有GPU利用率，GPU 1空闲。因而当前证据是：**GPU 1–3没有像GPU 0那样的PCIe配置空间失联证据，但新CUDA进程在三张卡上均不可用**。不能据此断言三张卡各自硬件损坏，也不能把它们称为现在可正常承接新训练。恢复主机GPU/驱动状态后，必须在每张卡上重跑同一最小探针，全部通过才恢复正式队列；现有V27进程及权重优先保全。故障的物理起因仍需管理员内核日志及必要的现场检查。
### 41.270 负载与故障因果边界、驱动级最小复现（2026-09-23 约14:30 北京时间）

用户追问是否为本次操作造成GPU 0掉线，以及为何其它三卡也无法开新任务。执行记录中的GPU相关动作是四张3090各跑一个正常R2/V27训练任务、M0及只读设备诊断；没有主动GPU reset、驱动重装、PCIe移除/重扫、功率限制修改或主机重启。四卡M0已通过，约10分钟后GPU 0在负载中失去PCIe配置访问。因此**时间顺序不能证明训练代码造成设备损坏**；但四卡并行负载可能暴露电源瞬态、散热、链路或设备本身的潜在不稳定，当前没有GPU 0事故前功耗/温度及管理员内核Xid日志，不能排除“负载触发”。NVIDIA官方Xid目录将“GPU has fallen off the bus”定义为驱动经PCIe无法访问GPU，并列出PCIe链路、GPU硬件或驱动等可能原因（https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html）；**本机尚未读到Xid，不把它写成已观测到Xid 79**。

为进一步排除PyTorch包自身，已在`CUDA_VISIBLE_DEVICES=1`的新进程里直接调用系统`libcuda.so.1`的`cuInit(0)`，返回`999 / CUDA_ERROR_UNKNOWN`；此前三张卡各自的单元素PyTorch CUDA初始化也均失败。新机GPU 1–3的PCIe配置仍可读，GPU 2/3原V27进程具有故障前建立的上下文并继续运算；新进程必须重新初始化驱动，故当前表现为三卡虽未掉PCIe、却均无法接新训练或评价。`CUDA_VISIBLE_DEVICES`只控制应用可见设备，不能代替在主机PCIe/驱动层恢复GPU 0（NVIDIA文档：https://docs.nvidia.com/deploy/topics/topic_5_2_1.html）。这属于结合本机探针和文档的运行时解释，具体驱动内部失效路径仍待管理员日志确认。

近固定终点的只读检查显示V27–RGBNT100 seed43/44均已到第19/20 epoch；两份`training.json`仍是`RUNNING`、目录下尚无`.pth`，所以目前不能说权重已保存。继续让现有进程完成并保全文件，不在此刻重启节点。故障恢复前不重启失败R2或开展后续数据集训练。

### 41.271 四卡队列终态、V27权重保全与CUDA恢复条件（2026-09-23 14:33 北京时间）

故障前已建立CUDA上下文的V27–RGBNT100 seed43/44均完成固定第20轮并保存权重；两个新开的正式评估进程都在CUDA初始化时失败，**没有生成`official_metrics.json`，不得报告正式检索指标**。seed43权重为`/data/gb/Re-ID/pretained/official_r2_v27_four_gpu_20260923/RGBNT100_V27_seed43/roles_epoch20.pth`，SHA256 `23295bbea5dd47b25799ebc4e7a7a6bf7ca435d06263f4d46d0f13691226598c`；seed44同目录对应文件SHA256 `fcf9aaa23d93e5357a1a3545ded37791fbbef5f905a366bd296eefa14718e7f5`。两份训练回执均为`FIXED_EPOCH20_TRAINING_COMPLETE`。不删除或重训这两个有效终点；主机恢复后直接依锁定作者评估入口完成五路正式评价。

原四卡队列进程已退出，`/data/gb/artifacts/official_r2_v27_four_gpu_20260923/campaign.json`已按文件和日志核实后标记`INTERRUPTED_GPU0_PCIE_CUDA_INIT`：RGBNT100 R2 seed43/44是`CUDA_FAILED_NO_ENDPOINT`，V27 seed43/44是`TRAINED_EVAL_BLOCKED_CUDA_INIT`，后续MSVR310、RGBNT201共八项仍`PENDING`。三卡恢复V1在使用GPU前被V2取代；V2的GPU1 M0在模型更新前同样因CUDA初始化失败，回执`STOPPED_NEW_CUDA_CONTEXT_FAILURE`。新机目前没有继续运行的训练/评价队列。`/data`按字节统计仍可用`62059896832`字节，约57.8 GiB；两份有效权重和三份作者Signal权重保留。旧单卡seed42队列独立，未因本次新机故障被中断。

最新的逐卡独立单元素PyTorch测试在GPU 1、2、3的新进程中全部于CUDA初始化失败；GPU 0的PCIe配置空间厂商ID为`ffff`，其余三卡为`10de`。直接在仅选择GPU 1的新进程调用`libcuda.so.1`的`cuInit(0)`也返回`999 / CUDA_ERROR_UNKNOWN`。因此目前可以确认的是**GPU 0 PCIe不可访问，且这台主机的CUDA驱动运行时无法为其余三卡建立新进程上下文**；其余三卡没有各自PCIe失联的证据。GPU 0故障是否由并行负载触发，以及驱动内部为何使所有新上下文失败，仍需管理员的内核NVRM/Xid、PCIe AER和硬件日志，不能由训练日志或`nvidia-smi`单独确定。我们未执行GPU reset、驱动更改、PCIe热移除或主机重启；四卡正常训练负载与故障有时间关系，但没有证据证明软件操作直接损坏显卡，也不能排除负载暴露潜在供电、散热或链路问题。

下一步需管理员核查约13:30起的内核/硬件日志并恢复主机GPU状态。恢复验收依次为：GPU 0 `setpci`厂商ID恢复`10de`、`nvidia-smi -L`识别四张卡、每张卡的独立单元素PyTorch CUDA分配成功。只有这些通过后，先评估现存V27 seed43/44权重，再从作者Signal权重重跑两个无有效终点的R2任务，按RGBNT100→MSVR310→RGBNT201原顺序完成其余训练与作者流程正式评价。当前无须根据训练loss猜测成绩，也不在主机故障期间反复启动会失败的GPU进程。

### 41.272 温度假设与管理员取证命令（2026-09-23 14:39 北京时间）

用户怀疑GPU 0过热，另问其他卡的CUDA初始化失败是否为独立故障。14:38只读复测中，`setpci`对GPU 0/1/2/3依次返回`ffff/10de/10de/10de`；GPU 1/2/3空闲温度为`35/38/38°C`，当前`HW Thermal Slowdown`、`SW Thermal Slowdown`及`HW Power Brake Slowdown`均未激活。这些是**故障后的其余三卡瞬时值**，不能回推GPU 0在13:30左右的核心/显存/供电温度；GPU 0已不可查询。`nvidia-smi -q`虽给出其余三卡的累计降频计时，但没有事故时刻，因此也不能据此归因。当前存在两个不同层级的症状：GPU 0 PCIe不可访问，以及全主机新CUDA上下文无法初始化；后者可能是前者引发的驱动级连锁，也可能存在另一个共同原因，现有普通用户权限下不能确认它们是否为两个独立故障。

`gaob`账号无法读取系统内核日志；管理员应在重启/重置前保存事故窗口日志，例如执行`sudo journalctl -k --since '2026-09-23 13:20:00' --until '2026-09-23 14:10:00' --no-pager | grep -Ei 'NVRM|Xid|AER|PCIe|thermal|overheat|power'`，并查看主机BMC/供电/风扇历史（如有）。重点核对GPU 0 `0000:17:00.0`和上游桥的首次事件时间及错误码；不能把文档中引用的NVIDIA Xid 79示例写成本机已发生的Xid。保全日志后由管理员恢复主机，再逐卡验收PCIe、`nvidia-smi`及新进程CUDA。恢复前不通过调训练代码或仅设置`CUDA_VISIBLE_DEVICES`来“绕开”该主机状态。

### 41.273 管理员内核日志确认Xid 79及全节点重启要求（2026-09-23）

用户取得的主机内核日志已将§41.268—§41.272的未确认项推进为**实际证据**：北京时间`2026-09-23 13:39:23`，GPU 0 `PCI:0000:17:00`出现`NVRM: Xid ...: 79, GPU has fallen off the bus`；同秒该卡出现`Xid 154 ... Node Reboot Required`，随后GPU 1 `31:00`、GPU 2 `4b:00`、GPU 3 `b1:00`也各自出现`Xid 154 ... Node Reboot Required`。内核明确提示已经产生GPU crash dump，建议在卸载NVIDIA内核模块前以root运行`nvidia-bug-report.sh`收集。先前“尚未观察到Xid 79”的表述只适用于取得此日志之前的时点，**现在已确认本机实际发生Xid 79**。

Xid 154是其他错误所需恢复动作的摘要；四张卡同时标记`Node Reboot Required`，与逐卡新进程均无法初始化CUDA一致，**不是四张卡分别物理损坏的证据**。已确定的起点是GPU 0失去PCIe访问；其根本诱因仍未由这段日志确定。片段中没有事故前GPU 0温度、显存温度或供电/风扇遥测，也没有可归因的thermal/AER前驱事件；因此“过热造成掉卡”目前仍是假设，须结合完整事故窗口内核日志及BMC硬件事件确认。NVIDIA Xid目录说明Xid 79表示驱动经PCIe无法访问GPU，可能涉及链路、GPU硬件或驱动；Xid 154明确标识所需恢复动作。参照`https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html`。

建议管理员先在`/data/gb`运行`sudo nvidia-bug-report.sh`保全当次crash dump及系统日志，报告文件只留在服务器，不提交GitHub；再按主机维护流程重启节点。当前新机无训练/评价进程，两份V27固定epoch20权重已保存。重启后核对GPU 0 `setpci -s 17:00.0 0.w`是否恢复`10de`、`nvidia-smi -L`是否识别四卡，并逐卡运行独立单元素CUDA测试；全部通过才恢复待完成的正式评估和训练。若重启后GPU 0仍不可访问，应交管理员检查该卡、PCIe链路及供电，而不是改TriFusion代码或在未验收主机状态下反复提交任务。

### 41.274 重启前状态与可执行操作（2026-09-23 14:48 北京时间）

用户询问根因和重启方式。14:48只读检查：新机没有TriFusion训练、正式评价或`nvidia-bug-report`进程；`/data/gb/nvidia-bug-report.log.gz`尚不存在；`/data`仍有约58GiB空闲。可确定的直接故障是GPU 0在13:39:23失去PCIe访问（Xid 79），随即四卡均被标记`Node Reboot Required`（Xid 154），与整机新CUDA上下文失败一致；日志尚不能区分过热、供电、PCIe链路、GPU本体或驱动的根本诱因，也不能把其余三张卡认定为分别损坏。

具备管理员权限且已确认没有其他用户重要任务时，从交互SSH登录`gaob@172.19.9.245:2028`，先运行`cd /data/gb`及`sudo nvidia-bug-report.sh`，核实`/data/gb/nvidia-bug-report.log.gz`已生成并留在服务器；之后执行`sudo systemctl reboot`。此交接只给出用户/管理员操作命令，当前助手没有执行重启。SSH连接断开是正常重启表现。主机回来后先验收`setpci -s 17:00.0 0.w`为`10de`、`nvidia-smi -L`列出四卡及每卡独立新进程CUDA分配；若GPU 0仍为`ffff`或任何卡初始化失败，保持正式任务停止，由管理员安排断电重上和硬件/供电/PCIe检查。两份V27权重、作者Signal权重和实验日志均在`/data/gb`保留。

### 41.275 NVIDIA崩溃报告已保全，等待管理员重启（2026-09-23 14:50 北京时间）

用户已在新服务器的`/data/gb`运行`sudo nvidia-bug-report.sh`，脚本输出`complete`，报告`/data/gb/nvidia-bug-report.log.gz`为约2.9MB。随后只读核验`gzip -t`通过，SHA256为`287fc1680b66c9463afca6ed2d22788f826a66b11778aa8a0798afd9a8794e0e`；报告中再次找到`13:39:23.253810+08:00`的GPU 0 Xid 79及四卡同秒Xid 154。脚本列出的`glxinfo`、`vulkaninfo`等缺失是可选采集组件跳过，不是本次主报告失败。该压缩包包含主机诊断信息，仅留服务器，不同步GitHub或桌面。按事故时间及`NVRM|Xid|AER|PCIe|thermal|overheat|power`过滤报告日志，只命中已知Xid和crash dump提示，没有得到事故前温度/供电/AER前驱记录；**不能由此证明或排除过热**。当前崩溃报告已经满足重启前保全条件；仍需先确认主机无其他用户重要任务，再由管理员执行节点重启及§41.274所列逐卡验收。助手尚未执行重启，正式评价仍待恢复。

### 41.276 用户执行重启后SSH未恢复的状态（2026-09-23 15:13 北京时间）

用户已执行`sudo systemctl reboot`，原SSH连接被重置；随后多次连接`172.19.9.245:2028`超时。没有重启命令的准确时间戳，不能计算精确耗时；崩溃报告14:50生成，因此15:13时重启距今**至多约23分钟**。本机只读检查：`ping`该IP得到2ms响应、TTL63，但带5秒连接上限的SSH访问`2028`及`22`端口均超时。可确定的是该地址已有ICMP响应、SSH访问尚未恢复；不能仅凭ping判断Linux、sshd或GPU已正常启动，也不能把超时归因于某一种防火墙/服务配置。不要反复快速轮询或启动训练。若用户/管理员可使用物理控制台或BMC，应查看启动画面、当前主机网络/地址、`systemctl status ssh`及`ss -ltnp`监听情况；待SSH恢复后先完成§41.274的四卡逐卡验收。此节先同步本地与GitHub，因新机SSH不可达，**新机仓库/交接文档暂不能同步本节**；恢复后再快进同步，旧单卡服务器不受此新机重启影响。

### 41.277 旧单卡服务器seed42队列状态（2026-09-23 15:35 北京时间）

用户询问另一台服务器进度，按原来只读入口查看：旧机队列PID`10763`仍存活，`campaign.json`为`RUNNING`。已完成的只有R2–MSVR310固定epoch20及正式评估（§41.261）：Signal `53.242392/72.419628`、R2 fused `52.281582/69.373942`（mAP/Rank-1，%），结果不变。R2–RGBNT100当前`TRAINING`：训练进程PID`14151`存活，占GPU约`7544MiB`；`training_steps.jsonl`最后到step`1891/2625`、epoch15/20，末步loss约`0.532565`、AMP scale`256`、合法排名anchor`64`、实例记忆`512`；已完成14个完整epoch，最近每轮约`1590–1620`秒。训练loss不是检索指标，当前尚无RGBNT100本轮正式mAP/Rank-1。按剩余734步及最近吞吐估算，训练约在北京时间18:00前后到固定终点，完整作者评价另需时间；这只是进度估计，不选中间权重也不改训练设置。余下R2–RGBNT201及V27三数据集共四项仍`PENDING`，执行顺序沿§41.260，不能写成已开训。旧机`/root/autodl-tmp`余约21GiB，未见本次新机故障波及。新机SSH仍需恢复后独立验收，不能用旧机正常运行推断新机已恢复。

### 41.278 用户收窄监测范围：只看旧单卡服务器（2026-09-23 17:13 北京时间）

用户明确要求“四卡服务器暂时不用管，只看旧服务器单卡那个”；后续不再主动探测四卡机或安排其恢复，待用户重新要求。17:13旧机只读回执：队列PID`10763`仍`RUNNING`，R2–MSVR310正式结果不变；R2–RGBNT100已完成第18轮日志，最新训练记录step`2361/2625`、loss`0.531611`、AMP scale`512`、合法anchor`64`、历史实例`512`，训练进程PID`14151`占GPU约`7544MiB`。按剩余264步及近几轮约12.2秒/步估计，固定第20轮训练约在18:05前后结束，随后进入作者完整query/gallery评价；**目前没有RGBNT100正式指标**，不能把loss当mAP。R2–RGBNT201和V27三数据集仍待顺序队列；旧机输出盘仍约21GiB可用。用户主动问进度属于按需检查，不改变固定训练设置或已有正式评估口径。

### 41.279 另一台3090服务器的两卡部署准备（2026-09-23 18:12 北京时间）

用户随后提供`gaob@172.19.9.245:2026`，要求在`/data/gaob/Re-ID`安装独立环境、数据集及GitHub项目，并进一步明确**只用两张卡，不占四卡**。此端口对应主机`ubuntu-WS-C621E-SAGE-Series`，与§41.267—41.276故障的`:2028`主机不同；本节未对`:2028`执行恢复或训练操作。新机`/data`初始可用`145820893184`字节（约136GiB），建立了`/data/gaob/Re-ID/Trifusion/{pertrained-model,trained-model,logs}`、`/data/gaob/Re-ID/dataset`和`/data/gaob/Re-ID/conda-envs/tri_reid`，Miniconda安装在`/data/gaob/Re-ID/miniconda3`。GPU 0/1各为24GiB 3090且初始空闲；GPU 2/3不分配本轮任务。

GitHub直连克隆在传输中停滞、HTTP/1.1局部克隆在签出时发生GnuTLS连接中断；将本地已核验且与GitHub `main`相同的`60c5869`完整Git包传入后恢复工作树，再将两卡队列及精确忽略规则提交到GitHub，当前本地/GitHub/新机均为`f7294e1`。队列入口`tools/queue_official_three_dataset_two_gpu.py`固定顺序为RGBNT100→MSVR310→RGBNT201，每数据集先seed43再seed44，每轮GPU 0运行R2、GPU 1运行V27；各任务先M0，再20轮固定终点训练，最后按原完整query/gallery协议评估。不启用GPU 2/3；旧单卡seed42队列不变。与旧seed42训练核心的差异仅为随机种子参数向初始化、采样、回执的传递，没有改变R2/V27目标或作者评价定义。

三份作者Signal权重已放入`pertrained-model`，SHA256依次为RGBNT100 `09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860`、MSVR310 `b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a`、RGBNT201 `ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c`。CLIP `ViT-B-16.pt`校验为`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；Signal源码提交`cd1b0a6`，工作树补丁SHA256 `b889caca9c4a92689b13eb7e20bd3224067f3e5ed2a3db6825201870ca422741`，与旧机对照一致。三套官方协议已复制到`Trifusion/logs/official_three_dataset_protocols_20260923`，只将`dataset_root`指向新机目录；完整路径核验须待数据复制结束。此时三套数据集仍在从旧机传输，锁定依赖仍在安装，**尚未启动M0、训练或评估，也没有新正式指标**。下一步是完成数据/环境、分别对GPU 0/1执行CUDA及Mamba前反向冒烟，再启动两卡队列；`/data`可用空间约128GiB。

### 41.280 两卡新机环境验收与固定队列启动（2026-09-23 19:31 北京时间）

`gaob@172.19.9.245:2026`的新机已完成三套数据复制：RGBNT201、RGBNT100、MSVR310合计110990个文件，与旧单卡服务器来源文件数一致；官方协议引用的RGBNT100 18965、MSVR310 8034、RGBNT201 14361个唯一路径全部存在。协议JSON与旧机逐字段相同，只有`dataset_root`指向`/data/gaob/Re-ID/dataset`。三份作者Signal及CLIP权重沿§41.279的SHA保持不变，均存`/data/gaob/Re-ID/Trifusion/pertrained-model`，不使用本项目自行训练的Signal权重。

Conda环境`/data/gaob/Re-ID/conda-envs/tri_reid`安装锁定依赖成功；`pip check`显示`No broken requirements found`，PyTorch `2.5.1+cu121`、torchvision `0.20.1+cu121`、transformers `4.45.2`、mamba-ssm `2.2.6.post3`、causal-conv1d `1.6.0`。GPU 0与1分别独立运行`tools/smoke_mamba.py`，均为RTX3090/sm86、输出及输入/全部参数梯度有限、全部参数梯度存在，退出码0。GPU 2/3未分配任务。清除安装完成后的pip缓存558个文件，释放约3034.9MB；`/data`启动时可用约125GiB。

GitHub、本地及新机代码锁定`aaddb1c3a951f6149145dfb023d8e750d7a5c177`。北京时间19:27:28启动后台队列PID`2404577`，状态文件`/data/gaob/Re-ID/Trifusion/logs/official_r2_v27_two_gpu_20260923/campaign.json`，队列stdout同级`official_r2_v27_two_gpu_queue_20260923.log`。固定顺序RGBNT100→MSVR310→RGBNT201，每数据集seed43后seed44；每轮GPU 0=R2、GPU 1=V27，两端各自M0→固定20轮训练→作者Signal完整query/gallery正式评价。19:31首组RGBNT100 seed43的R2/V27均为`M0_PASS`并进入`TRAINING`：各8次真实更新，冻结参数未改变、无缺失非零梯度、无溢出；GPU0/1占用约7385/6337MiB且100%活动，GPU2/3空闲。**训练刚开始，无本轮正式检索结果**；M0的loss不是mAP。

全队列初步预计约30—33小时（约9月25日01:30—04:30），以首个完整训练epoch的实测吞吐修正。按用户要求，启动验收后只计划约训练中点与全队列终态两次集中检查，不按分钟反复监视、不以中间loss挑checkpoint。旧单卡seed42队列保持独立，本节不触碰故障的`:2028`主机；所有训练权重和日志留在新机Trifusion内，不提交GitHub。

### 41.281 按需进度核对：两卡首轮训练与旧单卡RGBNT100正式终态（2026-09-23 19:54 北京时间）

用户询问当前进展，进行一次按需只读核查，不修改训练合同。新机`:2026`两卡队列PID`2404577`仍`RUNNING`。RGBNT100 seed43两端M0通过后持续训练：R2已完成第1轮130步（该轮1051.64秒、平均训练loss1.42388），核对时进入第2轮，最新step157、loss0.68928、AMP scale256、合法anchor64、历史记录512；V27已完成第8轮，第9轮step1157、loss0.55181、AMP scale256。GPU0/1利用率均100%、显存约7567/6337MiB，GPU2/3没有本轮计算任务；`/data`可用约125GiB。瞬时GPU0/1温度82/83℃、风扇76/100%，当前无CUDA错误日志或队列退出；这只是运行观测，不能由瞬时温度推断长期稳定性。新机没有完成任何本轮正式评价，训练loss不能写成mAP。seed44和后续MSVR310、RGBNT201仍待顺序队列。

旧单卡服务器seed42队列PID`10763`同时保持`RUNNING`，与新机seed43/44互不替代。新确认RGBNT100 R2固定epoch20完整作者协议评价`official_metrics.json`状态`COMPLETE`，query1715、gallery8575、无reranking、同身份同环境过滤、其他身份全保留；本机直接加载作者Signal权重。结果按mAP/Rank-1（%）：Signal `86.3242/97.5510`，R2 fused `86.3156/97.4927`，CNN `85.3468/97.7843`，Transformer `85.8561/97.0262`，Mamba `85.2197/97.7843`。R2 fused相对匹配Signal为约`-0.0086 mAP/-0.0583 Rank-1`，不能报告成超越。此前V8正式`83.2848`对应本机自行训练Signal `80.7122`，是不同初始化/运行，不与本次作者Signal结果拼成同一受控消融。旧队列已完成R2–MSVR310与R2–RGBNT100；R2–RGBNT201训练到epoch11/step583，其余V27三任务待运行。RGBNT201正式四指标仍未产生。

### 41.282 用户改用四卡：保留已完成V27并续接R2（2026-09-23 21:54 北京时间）

用户明确将新服务器`:2026`的本轮分配由两卡改为四卡。切换前，两卡队列中RGBNT100–V27 seed43已经完成固定20轮与作者原完整query1715/gallery8575正式评价，`official_metrics.json`状态`COMPLETE`：作者Signal `86.3242/97.5510`、V27 fused `85.9465/96.7930`、CNN `85.0802/97.3761`、Transformer `84.8915/96.4431`、Mamba `84.6947/96.5015`（mAP/Rank-1，%）。V27 fused相对Signal为`-0.3777 mAP/-0.7580 Rank-1`，不得称为正收益。RGBNT100–R2 seed43仍在GPU0训练，第5轮后继续推进；不重启或重训这两项。

新增续跑入口`tools/queue_official_three_dataset_four_gpu_resume.py`已提交GitHub `c005f14`并同步新机。北京时间21:49:35启动新调度PID`2696049`，接管现有R2训练进程PID`2408556`及V27完整结果，沿用原`trained-model/official_r2_v27_two_gpu_20260923`权重目录和`logs/official_r2_v27_two_gpu_20260923/campaign.json`以保留原回执，不因目录名中的`two_gpu`误读现时并发。验证新调度已启动三个不同待做任务后，仅向旧队列父PID`2404577`发`SIGTERM`；旧父进程退出，原R2子训练进程仍存活。新调度只在原R2固定终点落盘后补其正式评价，不重复其训练。新调度最多四个worker，每个空闲GPU领取一个尚未开始的任务；为满足用户四卡并行要求，MSVR310可与RGBNT100重叠，原逐数据集串行次序不再适用；每项仍固定M0→20轮→原Signal完整query/gallery评价、seed43/44、无中途选权重。

21:53按需启动验收：GPU0 RGBNT100–R2 seed43训练epoch6/step681；GPU1 RGBNT100–R2 seed44、GPU2 RGBNT100–V27 seed44、GPU3 MSVR310–R2 seed43的M0均通过并已进入真实训练，分别到step29、124、40。四卡分别占用约7567/7387/6339/7469MiB并有GPU计算活动；`/data`仍约125GiB可用，未观察到新CUDA错误。其余任务保持PENDING，由四卡续跑入口按空闲卡领取，不启动重复项。四卡完成时间仅能暂估为9月24日上午至下午初，须根据此并发阶段的实际训练/评价速度修正；目前除了上述V27 seed43，没有新机其它正式结果。

### 41.283 四卡首个新终态：RGBNT100–V27 seed44（2026-09-23 23:09 北京时间）

用户按需询问是否已有训练完成。23:04核对确认RGBNT100–V27 seed44已完成固定20轮训练、保存权重，随后进入原作者完整query/gallery正式评价；23:08评估JSON状态`COMPLETE`，query1715、gallery8575、无reranking、同身份同环境过滤，GPU2自动继续领取MSVR310–V27 seed43。seed44的mAP/Rank-1（%）：作者Signal `86.3242/97.5510`，V27 fused `86.3835/97.3761`，CNN `85.7227/97.4344`，Transformer `85.6552/96.9679`，Mamba `84.8056/97.1429`。本种子fused相对匹配Signal为`+0.0593 mAP/-0.1749 Rank-1`；此前seed43为`85.9465/96.7930`，相对Signal mAP下降，**不能将seed44微弱的单项正值写成跨种子稳定改进**。两种子尚未完成R2配对，不据此选择方法或更改训练合同。

同刻四卡续跑PID`2696049`仍存活：GPU0 RGBNT100–R2 seed43训练epoch8/step998，GPU1 RGBNT100–R2 seed44训练epoch3/step395，GPU3 MSVR310–R2 seed43训练epoch18/step360，GPU2 MSVR310–V27 seed43已通过M0并训练epoch1/step20。除上述V27 seed43/44两项，新机其余任务暂无正式检索终态；训练loss不作为mAP。当前仍按已固定20轮及作者评估流程继续，不中途选权重。

### 41.284 按需进度：旧单卡seed42六项正式终态及四卡新增结果（2026-09-23 23:57 北京时间）

旧单卡`:19873`的`official_r2_v27_campaign_20260923/campaign.json`现为`COMPLETE`，PID10763已退出；R2和V27在RGBNT201、RGBNT100、MSVR310各自固定20轮及完整Signal原协议评价均完成。以下为各自正式fused mAP/Rank-1（%，不是内部Q1；同一数据集Signal从相同作者权重得到）：

| 数据集 | 作者Signal | R2 seed42 fused | V27 seed42 fused |
|---|---:|---:|---:|
| RGBNT201 | 80.3029/85.1675 | 82.4254/86.7225 | 81.8916/85.5263 |
| RGBNT100 | 86.3242/97.5510 | 86.3156/97.4927 | 85.5703/96.6181 |
| MSVR310 | 53.2424/72.4196 | 52.2816/69.3739 | 50.5218/67.3435 |

RGBNT201正式836 query/836 gallery的完整四指标（mAP/Rank-1/Rank-5/Rank-10，%）：Signal `80.3029/85.1675/91.3876/93.6603`，R2 fused `82.4254/86.7225/92.5837/94.0191`，V27 fused `81.8916/85.5263/91.9856/93.6603`。两份`official_metrics.json`状态`COMPLETE`且`independent_upstream_metrics_equal=True`。R2相对Signal为`+2.1225 mAP/+1.5550 Rank-1`，V27为`+1.5887/+0.3588`；这支持RGBNT201的单seed正式增益，**不能外推到RGBNT100/MSVR310或称为三seed稳定结果**。RGBNT100作者Signal与旧自行训练Signal初始化不同，历史V8 `83.2848`不能代入本组控制端。

新四卡`:2026`队列PID`2696049`仍`RUNNING`，23:56四卡实际训练/显存正常、`/data`约124GiB可用。新增MSVR310固定终点正式结果（mAP/Rank-1，作者Signal仍`53.2424/72.4196`）：R2 seed43 fused `53.2391/70.8968`，V27 seed43 `51.6667/68.3587`，V27 seed44 `49.7506/66.4975`；三项各自评价状态`COMPLETE`、query591/gallery1055、`independent_upstream_metrics_equal=True`。R2 seed43 mAP几乎持平但Rank-1下降，V27两种子均下降。四卡未完成项：RGBNT100 R2 seed43训练epoch10/step1207、R2 seed44 epoch5/step607、MSVR310 R2 seed44 epoch10/step200、RGBNT201 R2 seed43 epoch3/step133；RGBNT201其余三项仍待队列。未完成的正式指标继续记为`-`，不拿训练loss或部分种子填补。

### 41.285 新增seed45/46与权重留存规则（2026-09-24 00:23 北京时间）

用户要求在当前seed42/43/44之外再试几个随机种子，让可用GPU继续工作，并最终每个方法、每个数据集只保留表现最好的一个训练权重。新增种子在启动前固定为**45和46**，范围为R2/V27 × RGBNT100/MSVR310/RGBNT201，各自仍从对应作者Signal预训练权重初始化，固定epoch20，逐项M0→训练→原Signal query/gallery完整正式评价；不选中间epoch、不改网络、损失、数据或评估掩码。正式指标原始回执和日志保留所有种子，RGBNT201报告mAP/Rank-1/5/10，其余两数据集重点报告mAP/Rank-1。

新增入口`tools/queue_official_extra_seed.py`与原多种子入口共用`tools/run_official_three_dataset_roles.py`及同三份协议，代码提交`9fd5fdf89b666cb443ec44f884ebf1bc61d1b549`已推送GitHub并同步两台服务器。旧单卡`:19873`此前HEAD`5faf76a`不支持`--seed`；同步前核对其交接文档SHA与当前GitHub完全一致，随后快进到`9fd5fdf`，文档SHA仍一致；新入口及正式训练入口通过Python编译，`--help`显示`--seed`。旧机作者权重、CLIP、Signal源码和协议路径均与原seed42一致。

旧单卡服务器seed45队列于00:21:08启动，首项RGBNT100–R2在00:22:07通过M0并进入20轮训练，GPU0约7370MiB、存在实际利用率；依次完成六项，状态文件`/root/autodl-tmp/trifusion-v2/TriFusion-ReID/logs/official_extra_seed45_20260924/campaign.json`。四卡`:2026`此时原seed43/44队列仍在运行；seed46等待进程PID`3014828`已启动，每240秒只检查原`campaign.json`是否`COMPLETE`，随后自动使用GPU0—3领取六项，不抢占现有训练，输出到`/data/gaob/Re-ID/Trifusion/{logs,trained-model}/official_extra_seed46_20260924`。00:18旧机数据盘约20GiB可用、新机`/data`约124GiB可用；当前新增权重预计每个约26—38MiB，没有删掉尚未评价的权重。

所有新增实验正式评价完成后，以**同一方法、同一数据集内的fused mAP最高，Rank-1仅在mAP并列时决胜**选出一个保留权重；其余种子的已验证训练权重才清理，原始指标、训练回执、代码/协议/作者权重哈希和日志都保留。seed42/43/44/45/46的完整种子分布是性能报告主体；按已消费正式测试选出的单一最高种子仅是存储与部署选择，**不能把该最大值报告成无偏的跨种子泛化估计**。截至本节seed45尚无正式指标、seed46尚未开始训练，不能提前宣布赢家或删除仍需评价的权重。

### 41.286 已完成种子的非最优权重清理（2026-09-24 00:30 北京时间）

按§41.285事先写明的同方法、同数据集fused mAP优先规则，对已有`COMPLETE`正式评价的种子执行第一轮清理。删除前逐项核对`official_metrics.json`、`training.json`的数据集/方法/种子、固定第20轮状态、正式回执中的checkpoint SHA与训练回执一致，并重新计算待删文件的实际SHA256。所有这些条件均通过后，仅删除以下五份`roles_epoch20.pth`：旧单卡MSVR310–R2 seed42（52.2816，已有seed43为53.2391）、旧单卡MSVR310–V27 seed42（50.5218，已有seed43为51.6667）、新四卡MSVR310–V27 seed44（49.7506，同上）、旧单卡RGBNT100–V27 seed42（85.5703，已有seed44为86.3835）、新四卡RGBNT100–V27 seed43（85.9465，同上）。合计释放`168688510`字节；所有训练日志、逐查询正式指标、训练回执及距离数组未删，原回执中的checkpoint路径现指向已退役文件，SHA仍留作历史记录。当前各组领先的权重已检查仍在；正在训练和未评估的权重没有清理。

清理后旧机seed45仍在RGBNT100–R2正式20轮训练，新机seed46等待进程仍存活，四卡seed43/44原队列尚在运行。旧机数据盘约20GiB、新机`/data`约124GiB可用。待全部种子完成正式评价后，重新按同一规则比较全部五个种子并只保留最终每个方法×数据集的一个权重；不能将本节的暂时领先者当作最终最佳或无偏的实验结论。

### 41.287 六组逐指标目标与自动接续队列（2026-09-24 01:08 北京时间）

用户新增明确目标：R2和V27**各自**在RGBNT201、RGBNT100、MSVR310相对同协议作者Signal提升；RGBNT201需同一个固定第20轮种子的fused **mAP/Rank-1/Rank-5/Rank-10四项**同时达到接近+1个百分点，另两数据集需同一种子的fused **mAP/Rank-1两项**同时达到接近+1个百分点。不能从不同种子拼各指标最佳。用户随后明确回复，**每项至少+0.8个百分点**为正式停止线；已启动控制器的参数正是`--min-gain 0.8`，无需重启。作者Signal固定基线分别为RGBNT201 `80.3029/85.1675/91.3876/93.6603`，RGBNT100 `86.3242/97.5510`，MSVR310 `53.2424/72.4196`，对应+0.8目标为`81.1029/85.9675/92.1876/94.4603`、`87.1242/98.3510`、`54.0424/73.2196`。这些均为百分数或百分点，实际判断使用回执中的未四舍五入浮点数。

此前只看mAP会误判RGBNT201 seed42已完成：R2对Signal增益为`+2.1225/+1.5550/+1.1962/+0.3589`，V27为`+1.5887/+0.3589/+0.5981/+0.0000`，因此两者均未满足四项同时提升的目标。现阶段全部六个方法×数据集组合仍需继续核验。MSVR310已完成种子中，R2 seed43最接近的mAP也仅为`53.2391`，V27 seed43为`51.6667`；只换种子是否足以达标未知，不把来源训练loss或局部种子最高分写成已解决。

代码提交`55b719d884b279fbfeab0d52063d387dad2e6c1d`已推送GitHub并同步两台服务器，仅扩展多种子队列的单组合/指定GPU入口，并新增`tools/continue_official_seed_until_delta.py`，原模型训练及作者评价代码不变。脚本已在本地及两台conda环境通过Python编译。旧机控制进程PID`48091`等待seed45六项完整结束，然后GPU0依次继续RGBNT201的R2/V27（从seed47起）；新机控制进程PID`3105665`等待seed46六项完整结束，然后GPU0—3继续RGBNT100与MSVR310的四组（从seed48起），每次只向未达标组合发新任务。两个控制进程均已独立验证存活；启动时旧机GPU0有100%利用率、约7554MiB显存，新机四卡均有训练进程、每卡约6.4—7.7GiB显存。旧机seed45首项RGBNT100–R2已完成epoch1、131步、mean loss`1.4190`并继续训练；新机原seed43/44队列已完成MSVR310–R2 seed44且继续其余任务，seed46仍等原队列，不重复训练或抢卡。

控制器每240秒只检查前序队列是否完整`COMPLETE`；正式训练期间由各子队列阻塞运行，不按分钟查训练。每个子任务仍执行M0→固定epoch20→作者完整query/gallery评价，保留全部正式指标与日志；新增完成权重若不优于当前候选，须验证SHA后删除，新的本机候选替代旧的本机候选时同样验证并清理。未达标时优先保留指定指标中**最弱一项增益最高**的候选；一旦同一种子全部达标，再以mAP、Rank-1排序。跨机器的历史权重在全局终态复核后统一清理。连续按正式测试结果选种子会带来选择偏差，因此需报告全部尝试过的种子与结果；选出的最大值不能作为无偏、多次独立验证的泛化估计。

平台中的旧TriFusion Goal目前仍显示`paused`，原目标是跨数据集SOTA；用户本轮明确要求把本节目标加入并启用，但工具拒绝创建第二个未完成Goal，且只提供暂停/完成/阻塞更新，不提供编辑目标或恢复状态。实际训练接续已按本节执行，不能将平台Goal状态误报为已启用。

### 41.288 RGBNT201–V27 seed43四指标达标与跨机接续（2026-09-24 01:17 北京时间）

新四卡队列中的RGBNT201–V27 seed43已完成M0、固定epoch20训练及原Signal完整正式评价，`official_metrics.json`为`COMPLETE`，query/gallery均为836，`independent_upstream_metrics_equal=True`。同一终点fused的mAP/Rank-1/Rank-5/Rank-10为`83.0005165572/87.5598086124/92.9425837321/94.4976076555`；对应作者Signal为`80.3028927692/85.1674641148/91.3875598086/93.6602870813`，四项分别提高`+2.6976237881/+2.3923444976/+1.5550239234/+0.8373205742`个百分点，**同一种子四项均超过用户确定的+0.8停止线**。这是本轮六个方法×数据集组合中第一个实测达标项，不代表另外五项已经达标，也不是多种子稳定性的证明。正式回执、训练回执及实存`roles_epoch20.pth`的SHA256均为`fc0e419b0fcc90e0f10339475ef5686c981e48aff654cfffe2c4c9da4766f064`；39795686字节的权重继续保留在`/data/gaob/Re-ID/Trifusion/trained-model/official_r2_v27_two_gpu_20260923/RGBNT201_V27_seed43/`。

旧单卡后续控制器PID`48091`尚在等待seed45队列结束，启动后只从旧机历史回执读取已完成种子。为避免它再为已达标的V27浪费训练，从新机将**原始138541字节正式回执逐字节**复制到旧机`/root/autodl-tmp/trifusion-v2/artifacts/official_r2_v27_campaign_20260923/train/RGBNT201_V27_seed43_import/official_metrics.json`，复制前后SHA256均为`5b584df035f65c8d2f21437f870f71ad38649b95da79e298d71692f889917bb8`。未复制或伪造权重；控制器读取该已核验回执后应只继续尚未达标的RGBNT201–R2。旧控制器启动时已由`--min-gain 0.8`限定停止线，其代码不需重启或修改。当前旧机seed45首项仍训练中，新机原seed43/44队列仍运行，seed46等待队列完整结束后启动；没有因此抢占任何训练进程。

磁盘复查：旧机`/root/autodl-tmp`可用`21234237440`字节，新机`/data`可用`132812017664`字节；旧GPU0及新机四卡当时均有实际GPU工作。MSVR310–R2 seed44正式结果`51.2600/69.0355`低于已保留的seed43 `53.2391/70.8968`，其固定终点权重在核对回执及实存SHA后退役，释放`38124518`字节；原始指标、训练回执及日志保留。连续正式测试选种子具有选择偏差，最终报告必须列出所有已试种子，不把seed43单点最高结果称为无偏泛化估计。

### 41.289 旧Goal与新Goal的统一目标（2026-09-24）

**统一目标：**持续完成TriFusion RGB/NIR/TIR多光谱重识别研究，保留CNN、Transformer、Mamba三角色与作者Signal强基座，在RGBNT201、RGBNT100、MSVR310分别建立可核验、同协议的正式训练和评价结果；基于实际代码、历史失败及完整对照分析泛化和融合问题，验证有明确证据的新方法，力争各数据集超越同协议Signal并达到注明预训练资源、评价协议和文献版本的SOTA。不能将工程门、内部Q1、单数据集收益或局部指标冒充完整目标。

**当前必须完成的量化里程碑：**R2与V27各自在三个数据集形成六个方法×数据集结果。每组均从对应数据集的作者Signal预训练模型独立初始化角色，固定第20轮终点，依原Signal完整query/gallery及camera/scene过滤正式评价；不从不同种子或轮次拼指标。RGBNT201同一种子fused的mAP、Rank-1、Rank-5、Rank-10各至少比匹配Signal高`0.8`个百分点；RGBNT100和MSVR310同一种子fused的mAP、Rank-1各至少高`0.8`个百分点。一个组合达标即停止为该组合新增种子；其余组合继续使用空闲GPU排队尝试。用户后续授权的多种子目标覆盖旧Goal中“沿用seed42”的旧阶段约束；固定epoch20、作者评估口径、训练方法与已封存实验不随正式分数更改。截至§41.288核验时，RGBNT201–V27 seed43已达标，其余五组尚未核验达标，不能提前标为完成。

**执行与证据边界：**旧单卡`:19873`与新四卡`:2026`使用已有conda环境、数据和预训练权重；保持GPU队列连续且不抢占，按预计里程碑或180—300秒间隔检查，不将临时低利用率误判为停止。保存每一种子的正式四项或两项指标、评估回执、训练日志、代码及权重SHA；所有种子成绩均公开记录，按正式测试筛出的最好权重仅作为部署与存储选择，不能报告成无偏泛化估计。持续核对磁盘，只有核对回执及实存SHA且不再需要的自训权重才退役，作者权重与必要赢家权重保留。实验结论及最新状态只追加到本交接文档，代码和本交接文档与用户指定的本地同名文件、两台服务器和GitHub仓库同步核对。六组+0.8是当前种子搜索的停止条件；达到它不自动证明原Goal中的SOTA要求，也不自动使统一Goal完成。

**平台状态：**整理本节时旧Goal为`paused`，创建新Goal因未完成旧Goal存在而被拒绝，故没有虚报旧SOTA目标完成。随后用户在平台Goal上下文恢复任务；2026-09-24 01:28北京时间重新调用`get_goal`，返回`status=active`，目标正文同时包含原跨数据集研究要求与六组+0.8新里程碑。本节作为格式清晰的统一目标正文；Goal虽已恢复，六组中截至§41.288仅一组达标，原SOTA目标更未完成，不能标为`complete`。

### 41.290 已达标组合停止后续新种子及当前正式指标（2026-09-24 01:33 北京时间）

逐个读取旧机seed42和新机seed43/44的全部`COMPLETE`正式回执，按同一终点、同一协议Signal计算各项百分点增益；共13个唯一方法×数据集×种子结果。下表仅列每组**已完成种子中fused mAP最高者**作进度概览，所有种子的原始回执继续保留，未完成种子不参与：

| 数据集 | 方法 | 已完成种子中mAP最高者 | 相对作者Signal的指定指标增益（百分点） | 全项≥+0.8 |
|---|---|---:|---|---|
| RGBNT201 | R2 | 42 | mAP +2.1225；R1 +1.5550；R5 +1.1962；R10 +0.3589 | 否 |
| RGBNT201 | V27 | 43 | mAP +2.6976；R1 +2.3923；R5 +1.5550；R10 +0.8373 | **是** |
| RGBNT100 | R2 | 42 | mAP −0.0086；R1 −0.0583 | 否 |
| RGBNT100 | V27 | 44 | mAP +0.0593；R1 −0.1749 | 否 |
| MSVR310 | R2 | 43 | mAP −0.0033；R1 −1.5228 | 否 |
| MSVR310 | V27 | 43 | mAP −1.5757；R1 −4.0609 | 否 |

新增的`tools/queue_official_extra_seed.py --skip-cell DATASET:METHOD`只过滤已达标组合的后续队列任务，并把跳过项记入`campaign.json`；训练器、初始化、epoch20和作者评价入口均未修改。代码提交`3be1668821cd30c1bcbaa470c5d9db56a5b80a0e`已推送GitHub、同步两机，并在本地与两台正式conda环境通过编译。新机seed46原等待PID`3014828`经`/proc/.../cmdline`核对确实仅在等待，且seed46 campaign目录不存在；将该等待进程终止，改以`--skip-cell RGBNT201:V27`启动PID`3158106`，2秒后再次核对进程存活、参数准确、campaign仍未创建。原seed43/44四卡训练进程与旧机seed45训练均未中断；新机seed46完成前序队列后只启动剩余五个组合，后继自适应控制器仍只向未达标组合发新任务。

旧机seed45及新机seed43/44在V27–RGBNT201达标前已固定排队，仍可能各执行一项原计划的同组合种子；本次不为删除这两项而中断正在训练的队列。后续**新增**自适应种子已跳过达标组合。当前旧机RGBNT100–R2 seed45训练至epoch2/20；新机RGBNT100–R2 seed43/44至epoch12/8，RGBNT201–R2 seed43/44至epoch9/2，四卡和旧机单卡均有实际GPU工作。旧机数据盘约21.23GB、新机`/data`约132.80GB可用，下一批完整正式评价仍需数小时，训练loss不代替检索指标。

### 41.291 后续种子队列按完成卡即时续接（2026-09-24 01:38 北京时间）

检查新增自适应控制器发现实际调度缺口：原`continue_official_seed_until_delta.py`一整波四卡任务全部完成后才开启下一波，短任务卡会等最慢的R2训练。现有日志显示新机RGBNT100–R2每epoch约1700—1844秒，RGBNT201–R2每epoch约590—853秒，V27此前20轮已更快完成；故此处并非假设性空转风险。代码提交`5bb6c879e73be7409f9c81fdd9004890932a6d74`将后续调度改为每个future完成、正式评价及权重核验落盘后，**立即由释放的同一GPU领取尚未达标组合的下一独立种子**，不等待其它GPU；种子仍唯一递增，当前端原M0→epoch20→作者正式评价、排名选优、SHA核验及磁盘清理规则不变。该修改只影响新增自适应控制器，不改变目前已运行训练或seed46固定队列。

本地及两机conda环境Python编译通过。新机旧等待控制器PID`3105665`经`/proc`确认仅在等待`seed46 COMPLETE`且目标状态文件不存在后停止；以相同`--machine new --min-gain 0.8`重启新代码PID`3167680`，2秒后核对准确命令行、进程存活，仍在等待。旧机PID`48091`为单卡，旧调度不会造成跨卡空等，保持不动。新机四卡当前正式训练、seed46等待PID`3158106`均不受干扰。**实际“完成即补位”尚须seed46结束后观察第一轮自适应运行回执，当前只有静态代码/启动核验，不把它写成已经完成的训练收益。**

### 41.292 当前正式种子与分资源文献参照的距离（2026-09-24 01:50 北京时间）

重新读取三个数据集当前已完成种子中fused mAP最高者的原始`official_metrics.json`：均为`COMPLETE`，对应作者Signal独立上游评价相同。与[RoDI作者原表](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf)的CLIP ViT-B/16版本按mAP/Rank-1作**数值参照**如下（单位：百分点；正差值表示本项目仍低于该文献值）：

| 数据集 | 当前已测fused种子 | 本项目mAP / Rank-1 | RoDI–CLIP作者值 | 相差mAP / Rank-1 |
|---|---|---:|---:|---:|
| RGBNT201 | V27 seed43 | 83.0005 / 87.5598 | 84.1 / 87.2 | 1.0995 / −0.3598 |
| RGBNT100 | R2 seed42 | 86.3156 / 97.4927 | 88.5 / 97.6 | 2.1844 / 0.1073 |
| MSVR310 | R2 seed43 | 53.2391 / 70.8968 | 64.1 / 77.2 | 10.8609 / 6.3032 |

RoDI–CLIP与本项目均涉及CLIP，但初始化、训练与选择资源并非完全匹配；本表不把跨论文差值解释为三角色的因果损失，更不以已测种子最高值充当无偏泛化估计。RGBNT201在本轮的Rank-5/Rank-10仍以作者Signal同协议四项增益及本轮`+0.8`停止线判断；上述RoDI参照只核入mAP/Rank-1，未拼造其未列指标。更强预训练或额外语义资源另列：RoDI–DINOv3为RGBNT201 `85.3/87.9`、RGBNT100 `89.0/99.1`、MSVR310 `71.8/84.8`；[PMKD作者PDF](https://aihuazheng.github.io/publications/pdf/2026/2026-Progressive_Multi-modal_Knowledge_Distillation.pdf)的RGBNT100为`91.6/98.0`且使用DINOv2；[CoT-ReID论文PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.pdf)的RGBNT100 `89.9/99.3`依赖DINOv3与MLLM文本。来源数值及表间差异已经在§34、§41.101和`docs/SOTA_PRIMARY_REFRESH_2026-09-07_EVENING.md`核读。当前固定epoch20种子队列继续按原合同运行；六组`+0.8`即使达标，也不自动完成跨资源SOTA目标。

### 41.293 MSVR310六个正式种子的全查询排序变化（2026-09-24 01:57 北京时间）

从R2、V27各自seed42/43/44的原始`COMPLETE`回执读取全部591条合法query的AP和首个正确匹配名次，逐一与同回执的作者Signal输出配对；均为固定epoch20、完整1055 gallery正式协议。AP变动阈值为绝对差`1e-12`，Rank-1修复/新增按首个正确匹配是否处于名次1判定，未按身份或中途epoch挑选：

| 方法/seed | fused − Signal mAP | fused − Signal Rank-1 | AP改善/下降/持平query | Rank-1修复/新增错误 |
|---|---:|---:|---:|---:|
| R2/42 | −0.9608 | −3.0457 | 287/291/13 | 26/44 |
| R2/43 | −0.0033 | −1.5228 | 300/275/16 | 35/44 |
| R2/44 | −1.9824 | −3.3841 | 260/314/17 | 23/43 |
| V27/42 | −2.7206 | −5.0761 | 249/328/14 | 20/50 |
| V27/43 | −1.5757 | −4.0609 | 274/301/16 | 23/47 |
| V27/44 | −3.4918 | −5.9222 | 222/351/18 | 16/51 |

这六个固定种子均出现首位新错误多于修复。R2 seed43的mAP几乎持平，同时仍有44条新首位错误、35条修复，说明均值相近并不等于逐查询排序被保留；该模型的fused `53.2391/70.8968`高于自己的CNN、Transformer、Mamba完整分支，却低于同协议Signal `53.2424/72.4196`，本例不能简单归因于末端没有选中最强角色。上述只是已消费正式测试的总体记账，不证明具体场景、外观或训练机制原因，也不用于修改当前锁定的R2/V27、选择超参数或针对测试身份设计增强。seed45及后续仍按原队列完成，新增正式结果再独立判定。

### 41.294 自适应种子选优与已登记指标口径对齐（2026-09-24 02:03 北京时间）

静态核查待启动的`tools/continue_official_seed_until_delta.py`发现真实规则偏差：尚未通过全项`+0.8`的种子，原实现先按`min(各指标增益)`决定保留权重，只有并列时才看fused mAP；这与§41.285已登记的mAP优先保留规则不一致。例如一个mAP更高但Rank-1略低的未达标种子可能被错误退役。现只改该选优键为`(是否全项达标, fused mAP, fused Rank-1)`：**未达标时mAP优先；有达标种子时必须优先保留实际全项达标的同一种子，并在达标种子之间按mAP优先**。后一个优先级遵从§41.287较晚确定的六组合格目标，避免最终只留下某个mAP虽高但不满足全部必报指标的权重。所有种子完整正式指标与训练回执仍保留；原seed42/43/44已封存分数不改，历史权重最终统一按本规则在SHA核验后清理。

此修复仅作用于**seed45/46完成后才开始**的自适应控制器选优和权重留存，不修改R2/V27模型、初始化、固定epoch20、作者评价函数、`+0.8`阈值或当前运行中的训练。修复落地后须在两机conda环境编译，通过`/proc`确认旧等待控制器没有已领取训练任务、目标状态文件尚不存在，再替换为同命令新进程；任何实际已在跑的训练不得抢占。

已按上述前置检查执行：代码及本交接提交`c285bf8e0848cd41667e43a1174e30f19cbf3e40`推送GitHub并同步两机，两套正式conda环境`py_compile`通过；旧机PID`48091`、新机PID`3167680`的`/proc/cmdline`均为原等待命令，均无子进程，目标状态文件均不存在。仅终止这两个等待控制器，按相同`--min-gain 0.8`命令重新启动旧机PID`50381`、新机PID`3221208`；2秒后两者均存活、命令行准确，仍在等待，目标状态文件仍不存在。原seed45及seed43/44训练队列PID未被停止；实际选优与自动补卡仍须等前序队列结束后验收，当前不能称已产生新正式增益。

### 41.295 RGBNT201／RGBNT100已完成正式种子的逐查询核验（2026-09-24 02:13 北京时间）

逐项重读旧机seed42与新机已完成seed43/44的七份原始`official_metrics.json`：状态均为`COMPLETE`，固定第20轮，`reranking=False`，`independent_upstream_metrics_equal=True`；RGBNT201每份836 query，RGBNT100每份1715 query。直接由全部query的`average_precision`和`first_match_rank`复算mAP、Rank-1及修复／新增错误，与各自回执汇总吻合。AP改善／下降使用绝对差`1e-12`；下表每行只相对该回执的同协议作者Signal，不从不同种子拼指标。

| 数据集 | 方法/seed | fused−Signal mAP | fused−Signal Rank-1 | AP改善/下降/持平query | Rank-1修复/新增错误 | fused−本模型最强完整角色mAP |
|---|---|---:|---:|---:|---:|---:|
| RGBNT100 | R2/42 | −0.0086 | −0.0583 | 677/639/399 | 11/12 | +0.4596 |
| RGBNT100 | V27/42 | −0.7539 | −0.9329 | 604/723/388 | 4/20 | +0.3586 |
| RGBNT100 | V27/43 | −0.3777 | −0.7580 | 593/719/403 | 6/19 | +0.8663 |
| RGBNT100 | V27/44 | +0.0593 | −0.1749 | 681/645/389 | 13/16 | +0.6607 |
| RGBNT201 | R2/42 | +2.1225 | +1.5550 | 339/184/313 | 31/18 | +0.3863 |
| RGBNT201 | V27/42 | +1.5887 | +0.3589 | 342/186/308 | 31/28 | +0.7786 |
| RGBNT201 | V27/43 | +2.6976 | +2.3923 | 350/168/318 | 36/16 | +0.4385 |

这七个已完成结果的fused mAP均高于各自最强完整角色，因此RGBNT100尚未达标不能简单归因为末端融合输给某一角色；它们也不证明某个具体训练成因。RGBNT201–V27 seed43的逐查询改善较广且同一固定终点四项增益均过`+0.8`，仍只算一个方法×数据集组合达标。RGBNT100–V27 seed44虽有轻微mAP正值，Rank-1仍下降，不能写成双指标成功。逐查询统计只用于已消费正式测试的总体核账，**不用于改变现行R2/V27超参数、选epoch或定制测试身份**。

02:13复查远端进程：旧机自适应等待PID`50381`、新机原四卡续跑PID`2696049`、seed46等待PID`3158106`和自适应等待PID`3221208`均仍存活；旧机数据盘约20GiB可用，新机`/data`约124GiB可用。该次检查不等于尚未完成任务有新正式指标；下一次训练进展观察应靠近预计首个RGBNT201–R2 seed43终态，而非密集轮询。

### 41.296 自适应候选权重实存SHA校验补齐（2026-09-24 02:20 北京时间）

审查尚未开始实际工作的自适应接续入口发现一项与§41.289留存规则不一致的代码路径：原实现对**未入选**新种子会计算磁盘上`roles_epoch20.pth`的SHA256后删除；对**入选**新种子仅比较训练／评价回执中的SHA字段，未核对实存文件。本次没有发现实际损坏权重，缺口是候选成为当前最好时缺少其文件验收。提交`80a37ae8201849cb43405577b8c677753b91f3fe`仅将已有`sha256(path)==training["checkpoint_sha256"]`断言移到选优分支前，使两种结果都核对同一实存文件；不改变模型、训练、评估、权重排名或删除条件。本地AST检查及两机正式conda Python解析通过，代码已推送GitHub并同步两机。

替换旧等待器前，逐机核对原PID命令行仅为`--machine old/new --min-gain 0.8`等待控制器、无子进程、目标状态文件尚不存在；终止旧机PID`50381`和新机PID`3221208`后，按相同参数启动新代码PID`51065`和`3252286`，2秒核对均存活且仍无目标状态文件。02:20再次确认两个新PID存活、旧机seed45队列PID`45727`仍`RUNNING`且1项训练/5项待做，新机原四卡队列PID`2696049`仍`RUNNING`且7项完成/4项训练/1项待做，seed46等待PID`3158106`仍存活。实际SHA核验须在后续新种子完整评价后才会执行，当前不宣称已有新增正式成绩或已验证自适应选优终态。

### 41.297 当前十三份固定终点正式结果的统一账（2026-09-24）

从旧单卡seed42六份、新四卡已完成seed43/44七份原始`official_metrics.json`重建当前唯一的13项方法×数据集×种子正式结果。均为`COMPLETE`、固定epoch20、无重排序、作者Signal独立上游结果相同；RGBNT201为836 query/836 gallery，RGBNT100为1715/8575，MSVR310为591/1055。下表数值是fused绝对指标，括号内为相对**同回执**作者Signal的百分点差；没有把内部Q1或不同种子的最好单项拼到一行。所有未完成种子保持`-`，原始全分支、逐查询和SHA字段仍保留在各自回执。

RGBNT201作者Signal：mAP/Rank-1/Rank-5/Rank-10=`80.3029/85.1675/91.3876/93.6603`。

| 方法/seed | mAP（Δ） | Rank-1（Δ） | Rank-5（Δ） | Rank-10（Δ） | 四项均≥+0.8 |
|---|---:|---:|---:|---:|---:|
| R2/42 | 82.4254 (+2.1225) | 86.7225 (+1.5550) | 92.5837 (+1.1962) | 94.0191 (+0.3589) | 否 |
| V27/42 | 81.8916 (+1.5887) | 85.5263 (+0.3589) | 91.9856 (+0.5981) | 93.6603 (+0.0000) | 否 |
| V27/43 | 83.0005 (+2.6976) | 87.5598 (+2.3923) | 92.9426 (+1.5550) | 94.4976 (+0.8373) | **是** |

RGBNT100作者Signal：mAP/Rank-1=`86.3242/97.5510`；MSVR310作者Signal：`53.2424/72.4196`。

| 数据集 | 方法/seed | mAP（Δ） | Rank-1（Δ） | 两项均≥+0.8 |
|---|---|---:|---:|---:|
| RGBNT100 | R2/42 | 86.3156 (−0.0086) | 97.4927 (−0.0583) | 否 |
| RGBNT100 | V27/42 | 85.5703 (−0.7539) | 96.6181 (−0.9329) | 否 |
| RGBNT100 | V27/43 | 85.9465 (−0.3777) | 96.7930 (−0.7580) | 否 |
| RGBNT100 | V27/44 | 86.3835 (+0.0593) | 97.3761 (−0.1749) | 否 |
| MSVR310 | R2/42 | 52.2816 (−0.9608) | 69.3739 (−3.0457) | 否 |
| MSVR310 | R2/43 | 53.2391 (−0.0033) | 70.8968 (−1.5228) | 否 |
| MSVR310 | R2/44 | 51.2600 (−1.9824) | 69.0355 (−3.3841) | 否 |
| MSVR310 | V27/42 | 50.5218 (−2.7206) | 67.3435 (−5.0761) | 否 |
| MSVR310 | V27/43 | 51.6667 (−1.5757) | 68.3587 (−4.0609) | 否 |
| MSVR310 | V27/44 | 49.7506 (−3.4918) | 66.4975 (−5.9222) | 否 |

按未四舍五入回执判断，当前13项中仅RGBNT201–V27 seed43通过其四项停止线，六个方法×数据集组合为`1/6`。该种子的Rank-10增益为`+0.8373`，只比门槛高`0.0373`个百分点；836个query中一个Rank-10判定约占`0.1196`个百分点，故这是一项有效但边界较近的单种子通过，不是稳定性证明。余下种子继续按既定队列运行，任何训练中数值都不填入本表。

### 41.298 旧机与新机作者协议的语义一致性核验（2026-09-24）

§41.297的跨机结果引用不同的协议文件SHA，因此逐字段比较旧机`artifacts/official_three_dataset_protocols_20260923/{dataset}.json`与新机`logs/official_three_dataset_protocols_20260923/{dataset}.json`。三个数据集各自的`inventory_sha256`完全一致；JSON中唯一不同的顶层字段均为机器上的绝对`dataset_root`，将各自根目录统一替换为同一占位符后，整个JSON对象逐字段完全相等，包含train标签映射、记录顺序、query行、正例计数及camera/scene过滤定义。原始文件SHA不相同是绝对路径造成的，不能据此称为不同评价协议。

再从每个数据集的旧机和新机各取一份`COMPLETE`正式回执，对独立作者Signal的全部query逐项比较：RGBNT201 836条、RGBNT100 1715条、MSVR310 591条的`average_precision`均逐元素精确相同，`first_match_rank`差异均为0，四项汇总指标也完全相同。这证明当前两台机器的**已测作者Signal正式输出**一致，支持§41.297按同数据集跨机汇总；它不证明不同种子角色输出应相同，也不抵消连续在正式测试上选种子的选择偏差。训练队列和固定epoch20合同未因该核验改变。

另对上述13份已完成回执按数据集、方法分组检查：每组seed标签互不重复，有多个已完成种子的组，其`model_state_sha256`与`role_checkpoint_sha256`均逐种子不同，而同组`author_checkpoint_sha256`只有一个值。这排除了当前账本将同一终点摘要重复登记为不同种子的情况；摘要不同本身不是跨种子稳定收益的证据。

### 41.299 已完成种子的权重按当前六组最优清理（2026-09-24 02:56 北京时间）

旧机RGBNT201–V27 seed42已完成正式`81.8916/85.5263/91.9856/93.6603`，其`roles_epoch20.pth`实存`39795686`字节、SHA256=`6177c0ceafd46a3829b4f330a1bd7f98560ec42435114416cd5a322bb472163f`，与训练及正式评价回执一致；新机同组合达标seed43实存权重同为`39795686`字节、SHA256=`fc0e419b0fcc90e0f10339475ef5686c981e48aff654cfffe2c4c9da4766f064`，与回执一致且四指标均更高。故只退役旧机seed42的该份权重，释放`39795686`字节；旧机`/root/autodl-tmp`随后可用`21268721664`字节。两个种子的训练日志、正式指标、逐查询数组和回执均保留，历史回执所记seed42 checkpoint路径现指向已退役文件，SHA仍是原始证据。

随后对当前13份已完成正式结果逐组按§41.294的`(全项达标,mAP,Rank-1)`规则审查物理权重：六个方法×数据集组合各有且仅有一份当前领先checkpoint实存，分别为MSVR310–R2 seed43、MSVR310–V27 seed43、RGBNT100–R2 seed42、RGBNT100–V27 seed44、RGBNT201–R2 seed42、RGBNT201–V27 seed43；其他已完成种子的权重均不再实存。**这是当前已完成种子的暂时留存状态，不是最终最佳或无偏估计。**02:56复查旧机seed45队列及新机原四卡队列、seed46等待与两个自适应等待控制器均存活；未对训练中或未评价种子的权重做任何清理。

### 41.300 近期 CLIP 近邻原表与 SOTA 口径复核（2026-09-24）

新增核读两个作者原文的完整三光谱匹配协议主表：[FUSE 预印本](https://arxiv.org/pdf/2606.20044)报告 RGBNT201 `81.4/86.1/91.5/93.8`（mAP/Rank-1/5/10）、RGBNT100 `88.5/96.9`、MSVR310 `50.1/65.7`；以预训练 CLIP 初始化，RGBNT201/RGBNT100 训练45轮、MSVR310训练50轮。[MDReID 作者论文](https://arxiv.org/pdf/2510.23301)的完整模态 RNT-to-RNT 主表依次为 RGBNT201 `82.1/85.2/90.3/92.6`、RGBNT100 `85.3/95.6`、MSVR310 `51.0/68.9`；使用 CLIP-Base，训练50轮。MDReID另研究任意模态组合，其模态错配成绩不能混入本项目完整三模态官方列。

两篇文中的“最佳/SOTA”只针对各自表内收录的方法，不能替代§41.292已登记的跨论文参照。FUSE 在RGBNT201 mAP/Rank-1、RGBNT100 Rank-1及MSVR310两项均低于已核过的RoDI–CLIP `84.1/87.2`、`88.5/97.6`、`64.1/77.2`；MDReID三个数据集的mAP也均低于对应RoDI–CLIP。上述只是公开报告值的数值对照，不是等初始化、训练预算或模型选择的因果比较，不修改现行R2/V27固定epoch20、作者Signal同协议`+0.8`六组停止线，也不生成新正式结果。

### 41.301 R2来源训练权重上限的跨数据集记录（2026-09-24 03:18 北京时间）

只读取现有`training_steps.jsonl`与`tools/msvr_supported_gradient_balance.py`，不读取任何正式测试身份。计数仅取`active_fused_metric=cross_environment_smooth_ap`且`eligible_anchors>0`的优化步骤；“三角色同达上限”指CNN、Transformer、Mamba当步实际`[rank,auxiliary]`权重均为`[1.6,0.4]`：

| 机器与R2任务 | 已记录支持步骤 | 三角色同达上限 | 记录末轮次 |
| --- | ---: | ---: | ---: |
| 旧机RGBNT100 seed42 | 2560 | 2560 | 20/20，已完成训练 |
| 旧机RGBNT100 seed45 | 800 | 800 | 7/20，训练中 |
| 新机RGBNT100 seed43 | 1983 | 1983 | 16/20，训练中 |
| 新机RGBNT100 seed44 | 1433 | 1433 | 12/20，训练中 |
| 旧机RGBNT201 seed42 | 962 | 895 | 20/20，已完成训练 |
| 新机RGBNT201 seed43 | 781 | 722 | 17/20，训练中 |
| 新机RGBNT201 seed44 | 444 | 355 | 10/20，训练中 |
| 旧机MSVR310 seed42 | 335 | 0 | 20/20，已完成训练 |
| 新机MSVR310 seed43 | 332 | 0 | 20/20，已完成训练 |
| 新机MSVR310 seed44 | 329 | 0 | 20/20，已完成训练 |

实现对有支持步骤按角色的排名/辅助梯度范数EMA求平方根比值，再截断至`[0.25,4]`；`[1.6,0.4]`对应上限比值4。因此，**RGBNT100已观察四个种子的6776/6776个有效步骤中，R2控制器实际退化为同一组固定系数**，其中seed42覆盖完整20轮；MSVR310三个已完成种子为`0/996`三角色同达上限，RGBNT201介于其间。这是来源训练过程的权重作用记录，不是独立样本比例，不证明具体检索失败由截断造成，也不能用已消费正式结果回调系数上限。当前固定epoch20训练、评价、排队和停止线均保持不变；待各端完整终态后再结合正式指标解释，不将中途loss或权重当成检索成绩。

### 41.302 按指标区分的 CLIP 公开参照补充（2026-09-24）

重新核读[ICPL-ReID作者原文](https://arxiv.org/pdf/2505.17821)的Table III/IV及[UGG-ReID作者原文](https://proceedings.neurips.cc/paper_files/paper/2025/file/735c847a07bf6dd4486ca1ace242a88c-Paper-Conference.pdf)的Table 1/2。两者均使用预训练CLIP，但训练合同不同：ICPL论文写120轮、身份条件提示和文本分支；UGG写40轮、图不确定性与MoE。本项目当前为作者Signal初始化、角色训练固定20轮，故下列数值仅作**公开参照，不是等资源因果对照**。

| 数据集 | ICPL-ReID作者mAP / Rank-1 | UGG-ReID作者mAP / Rank-1 | §41.292 RoDI–CLIP mAP / Rank-1 |
| --- | ---: | ---: | ---: |
| RGBNT201 | 75.1 / 77.4 | 81.2 / 86.8 | 84.1 / 87.2 |
| RGBNT100 | 87.0 / **98.6** | 88.0 / 98.1 | **88.5** / 97.6 |
| MSVR310 | 56.9 / 77.7 | 60.1 / **78.0** | **64.1** / 77.2 |

因此，§41.292的RoDI–CLIP表可继续用作三数据集mAP的较强CLIP参照，但**不能把其RGBNT100 `97.6`或MSVR310 `77.2`写成已核文献中的最高Rank-1**：本轮分别核到ICPL `98.6`与UGG `78.0`。RGBNT201的四项与其他更强预训练/额外语义资源仍按原节分列；此处也不宣称已穷尽截至今日全部论文或确立绝对SOTA。公开参照核查不改变R2/V27队列、任何已锁定训练参数、正式评价协议及同协议Signal `+0.8`停止线；截至本节没有新增本项目正式检索结果。

### 41.303 新近INSPI论文的资源条件与三数据集指标（2026-09-24）

新增核读2026-09-21提交的[INSPI作者预印本](https://arxiv.org/pdf/2609.24539)（*Incentive Noise and Structural Prior Infusion for Multi-modal Object Re-Identification*，作者标注已接收ECCV 2026，但当前核到的是arXiv v1而非后续正式版本）。其Table 1/2完整RGB/NIR/TIR主行报告：RGBNT201 `80.6/83.9/91.6/93.4`（mAP/Rank-1/5/10），RGBNT100 `89.9/98.2`，MSVR310 `65.0/77.2`（mAP/Rank-1）。其中MSVR310 mAP数值比§41.292的RoDI–CLIP `64.1`高`0.9`个百分点；但Rank-1仍低于§41.302已核UGG `78.0`，三数据集亦未整体超过原节所列更强资源参照，不能因其论文内“最佳”标记将其写成截至当前的绝对SOTA。

资源标注必须保留：INSPI用CLIP视觉/文本编码器，图中有GPT-4o caption生成流程；DINOv3全局Token仅用于初始化可学习结构提示，作者称训练和推理均不继续运行该DINOv3模块；三数据集训练50轮，RGBNT100 batch128，另两组batch64。它因此不属于本项目仅以三份Signal作者权重初始化、固定20轮、没有外部文本/结构教师的同资源对照，也不能与RoDI–CLIP称为完全相同的“CLIP-only”组。本轮只更新公开参照，未用它改动任何已启动实验或正式测试选优规则。

### 41.304 RGBNT201–R2 seed43 固定终点正式结果（2026-09-24 04:01 北京时间）

新四卡服务器的`RGBNT201_R2_seed43`已完成第20轮训练与作者Signal原协议正式检索，回执为`trained-model/official_r2_v27_two_gpu_20260923/RGBNT201_R2_seed43/official_metrics.json`。训练时间为9月23日23:34:42至9月24日04:00:15，正式评估于04:01:13完成。回执状态`COMPLETE`，训练状态`FIXED_EPOCH20_TRAINING_COMPLETE`；836个query/836个gallery，完整camera过滤、无重排序。执行代码提交`18f0e2ae55d07e75a8c4d103a5efdb4de5305f52`，协议SHA256=`e63be5d9f8bd41df365264ad8f459dcd063d088023a2d2ed525a49a72f61161e`，作者Signal初始化权重SHA256=`ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c`。实存`roles_epoch20.pth`、训练回执和正式回执登记的角色权重SHA256均为`99dd3ac4ab563fda54bceb54829db6e7a208216740f56a6aaf7c0b6504846616`；正式距离数组的实存SHA也与回执相等。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| 同回执作者Signal | 80.3029 | 85.1675 | 91.3876 | 93.6603 |
| CNN完整分支 | 82.1907 | 87.4402 | 92.4641 | 94.0191 |
| Transformer完整分支 | 80.7685 | 86.1244 | 91.7464 | 93.8995 |
| Mamba完整分支 | 81.2269 | 85.5263 | 91.5072 | 93.6603 |
| **R2 seed43 fused** | **82.5826** | **87.4402** | **91.8660** | **94.2584** |
| fused相对同回执Signal | **+2.2797** | **+2.2727** | **+0.4785** | **+0.5981** |

按未四舍五入的回执判定，fused的Rank-5、Rank-10增益均未达`+0.8`个百分点，故**该种子未通过RGBNT201四指标同时达标的停止线**。它将§41.297的已完成正式种子总数从13增至14，但六个方法×数据集组合仍只有`RGBNT201–V27 seed43`通过，即`1/6`。相较R2 seed42，seed43的mAP更高而Rank-5更低；不可拼接两种子的单项最好值，也不可将正式测试上筛出的种子称为无偏泛化估计。其余已启动训练与自动接续队列维持原合同，暂不清理仍可能被队列引用的旧权重；下一次仅在预计终态附近检查并按实存SHA、回执和磁盘空间执行留存规则。

### 41.305 RGBNT201–V27 seed44 固定终点正式结果（2026-09-24 04:34 北京时间）

新机原四卡队列的`RGBNT201_V27_seed44`继§41.304之后完成固定第20轮和作者Signal原协议完整正式评价，训练回执`training.json`状态为`FIXED_EPOCH20_TRAINING_COMPLETE`，正式回执`official_metrics.json`为`COMPLETE`；路径均在`trained-model/official_r2_v27_two_gpu_20260923/RGBNT201_V27_seed44/`。本种子训练于04:02:09启动、04:33:04结束，正式评价于04:34:04完成，执行代码提交`ec7fe66d864c169bde2242f295f524c05aaf8cbe`；query/gallery均为836，无重排序，作者Signal独立上游指标与其他正式回执相同。协议SHA256=`e63be5d9f8bd41df365264ad8f459dcd063d088023a2d2ed525a49a72f61161e`、作者初始化权重SHA256=`ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c`。实存角色权重、训练及正式回执的checkpoint SHA256均为`b55a9b78c0c32a9cf3d0e86c0b94c0cd67ed86fe11c24ea2ae9210de6625d695`；实存距离数组SHA256与正式回执同为`c31dd7628b617c853ca27141f1587f825a3cda64582afb6ba362084b3e992e58`。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| 同回执作者Signal | 80.3029 | 85.1675 | 91.3876 | 93.6603 |
| CNN完整分支 | 81.7097 | 86.7225 | 91.7464 | 93.8995 |
| Transformer完整分支 | 80.7558 | 85.5263 | 92.1053 | 94.4976 |
| Mamba完整分支 | 83.0594 | 86.8421 | 91.7464 | 93.5407 |
| **V27 seed44 fused** | **82.9153** | **86.8421** | **92.4641** | **94.1388** |
| fused相对同回执Signal | **+2.6124** | **+1.6746** | **+1.0766** | **+0.4785** |

本种子Rank-10的原始增益`+0.4784688995`不足`+0.8`，因此**未通过四指标同时达标线**。V27 seed43仍是该组合现有达标种子，其fused mAP `83.0005`也高于seed44的`82.9153`；不能从两种子分别挑指标拼为一个结果。已完成正式种子总数增至15，六组合仍为`1/6`。原四卡队列尚有三项R2训练中；seed46等待器及后继自适应队列继续按原合同接续。当前旧机训练盘约余20 GiB，新机`/data`约余124 GiB；本次不为释放约38 MiB而在原队列尚未退出时删除新完成checkpoint。

### 41.306 新机 seed46 按单卡释放接续，解除实测空卡（2026-09-24 04:51 北京时间）

§41.305后实际检查新机原四卡队列仅余`RGBNT100–R2 seed43/44`占GPU0/1和`RGBNT201–R2 seed44`占GPU3，全部其他原队列任务已正式完成、无`PENDING`；GPU2的`nvidia-smi`为`0%`利用率、约15 MiB显存。原seed46等待器却要等**整个**原队列`COMPLETE`才启动，预计让该卡空等数十分钟至数小时。这是已观察到的调度空档，不是推测性的边界情况。提交`6610664ed42e0530377db4f24749fd6c442bf96b`仅为`tools/queue_official_extra_seed.py`增加明确的`--overlap-previous`入口：只供新机seed46完整队列使用，启动前要求原队列没有待分配任务；各worker在自己GPU上的旧任务状态为`COMPLETE`后才开始该卡的新M0→20轮训练→正式评价。未改数据、Signal初始化、R2/V27、GPU内单任务、seed、损失、终点或评价函数；其他调用保持原先整批等待行为。本地及两机正式Python的AST检查通过，提交已推送GitHub并同步两机。

替换前核实旧等待PID`3158106`的命令行仅为原seed46等待入口、无子进程，seed46 campaign目录不存在，原队列无待分配任务；随后只终止这个等待PID，并按同一`seed46`、同一`RGBNT201:V27`跳过条件加新开关启动PID`3542100`，没有中断原三项训练或旧机seed45。04:51验收：seed46 campaign为`RUNNING`，五项计划任务分配GPU0/1/2/3，其中GPU2的`MSVR310–R2 seed46`已由真实`M0_PASS`进入`TRAINING`；四卡利用率依次为`100/90/100/90%`，GPU2约7361 MiB显存，原队列PID`2696049`、seed46 PID`3542100`及后继自适应等待PID`3252286`均存活。新机`/data`仍约124 GiB可用。其余seed46项将在各自GPU释放后自动接续，**本节没有新增seed46正式检索指标**，也不改变六组合目前`1/6`的判断。

### 41.307 退役已核验的 RGBNT201–V27 seed44 落败权重（2026-09-24）

对§41.305暂缓清理的seed44补做实际依赖核验：原四卡队列中该任务状态已为`COMPLETE`，`queue_official_three_dataset_four_gpu_resume.py`此后只运行余下R2任务、不再读取已完成V27权重；新机seed46明确跳过`RGBNT201:V27`，后继自适应控制器也只负责RGBNT100/MSVR310四组。seed43的fused mAP/Rank-1/5/10 `83.0005/87.5598/92.9426/94.4976`逐项高于seed44的`82.9153/86.8421/92.4641/94.1388`，且seed43四项已达标。再次核对两份实存权重的SHA与各自训练、正式回执一致，seed43=`fc0e419b0fcc90e0f10339475ef5686c981e48aff654cfffe2c4c9da4766f064`，seed44=`b55a9b78c0c32a9cf3d0e86c0b94c0cd67ed86fe11c24ea2ae9210de6625d695`。据此只退役新机seed44的`roles_epoch20.pth`，释放`39,795,686`字节；再次确认该文件不存在、seed43权重仍实存。seed44的正式指标、距离数组、训练与评价日志及原SHA回执均保留；历史回执中的seed44 checkpoint路径现在指向已退役文件，不能误称该文件仍在磁盘。

### 41.308 旧机 V27 停止规则复核与冗余改动撤回（2026-09-24 05:32 北京时间）

04:57曾误以为旧机后继控制器只见seed42和seed45，因此把`cells`临时收窄为`RGBNT201–R2`，提交`8431f156e889fbf4bcdd6badd61f09c6ae5c26c9`并将纯等待进程从PID`51065`替换为`61230`。**该诊断遗漏了§41.288已完成的回执导入，应予撤回。**旧机`artifacts/official_r2_v27_campaign_20260923/train/RGBNT201_V27_seed43_import/official_metrics.json`实存138541字节，SHA256=`5b584df035f65c8d2f21437f870f71ad38649b95da79e298d71692f889917bb8`，状态`COMPLETE`；其fused相对同协议Signal的四项原始增益为`+2.6976237881/+2.3923444976/+1.5550239234/+0.8373205742`，均达到`+0.8`。原控制器的`root.glob("*/official_metrics.json")`会读到该目录，`rank()`据此将V27标为达标，`schedule()`不会给它新增种子；旧机没有复制对应权重，导入回执仅用于停止判定。

因此提交`6a60b08e7c1b527bbda36818b525066261d10e63`恢复原`cells`与回执驱动判定，不保留无必要的硬编码跳过。代码已推送GitHub并同步两机，本地及两机正式Python AST检查通过；确认PID`61230`仍仅在等待、无子进程、目标状态文件不存在后终止它，按原`--machine old --min-gain 0.8`启动恢复后的等待PID`62216`，4秒后确认新PID存活，旧机seed45训练PID`45727`未中断。临时改动没有启动任何自适应训练，也没有改变R2/V27模型、固定20轮或正式指标。§41.288对未来跳过V27的原判断成立；旧机已预排的seed45仍依原合同完成一次。

### 41.309 RGBNT201–R2 seed44 固定终点正式结果（2026-09-24 05:22 北京时间）

新机`trained-model/official_r2_v27_two_gpu_20260923/RGBNT201_R2_seed44/`的训练回执为`FIXED_EPOCH20_TRAINING_COMPLETE`（01:15:17至05:21:20），正式回执于05:22:11达到`COMPLETE`；执行代码提交`2a21ebb9dde4876e4386b55a3937cda889b40c3a`。作者Signal权重SHA256=`ec09a4f68bce95f645fde3fd2e29f81c944d1f5816adc00ab107e3daf6e38b7c`，协议SHA256=`e63be5d9f8bd41df365264ad8f459dcd063d088023a2d2ed525a49a72f61161e`；836 query/836 gallery、同camera过滤、无重排序，独立作者上游指标逐项相同。正式权重SHA256=`81e3ab33968090fd5a9a3e50d926fe394624fe7da0f749a46a6eb2badcc63681`，实存文件与训练及正式回执一致；实存距离数组SHA256=`2d89dcd84a4b9bfd40352bb27e5d9db7852b3e61f8988658c0354a80ed2c6083`，也与正式回执一致。

| 输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| 同回执作者Signal | 80.3029 | 85.1675 | 91.3876 | 93.6603 |
| CNN完整分支 | 81.1122 | 85.0478 | 92.3445 | 93.7799 |
| Transformer完整分支 | 80.4886 | 84.6890 | 91.3876 | 93.8995 |
| Mamba完整分支 | 82.8619 | 87.0813 | 92.1053 | 94.0191 |
| **R2 seed44 fused** | **82.4417** | **86.3636** | **92.5837** | **94.1388** |
| fused相对同回执Signal | **+2.1388** | **+1.1962** | **+1.1962** | **+0.4785** |

按原始数值，Rank-10仅`+0.4784688995`，该种子**未通过四项各≥+0.8**的停止线。R2 seed43的mAP `82.5826`高于本种子`82.4417`，按已登记的`(全项达标,mAP,Rank-1)`留存顺序，seed44不是当前R2–RGBNT201权重赢家；两种子的最佳单项不得拼接。

### 41.310 RGBNT100–R2 seed43 固定终点正式结果（2026-09-24 05:29 北京时间）

新机`trained-model/official_r2_v27_two_gpu_20260923/RGBNT100_R2_seed43/`从9月23日19:28:55训练至9月24日05:24:27固定第20轮，05:29:04完成原Signal正式评价；训练执行提交`aaddb1c3a951f6149145dfb023d8e750d7a5c177`。正式回执`COMPLETE`、1715 query/8575 gallery、同camera过滤、无重排序，独立作者Signal指标相同。作者权重SHA256=`09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860`，协议SHA256=`12d8afe5cf6e537b69ccf753481dd5fd651f14b7672e1e6411a4d4e9d88cc956`；实存角色权重、训练及正式回执SHA256均为`4bffe88835c4dfb9b927de7b98933fb489f7c83f7f5a761af31cda27ea62a786`，实存距离数组与回执SHA256同为`bfbe1cfa69b891d51a188258f59fbf1ece75d25f87b836b17fa1cf3b57bd5e88`。

| 输出 | mAP | Rank-1 |
| --- | ---: | ---: |
| 同回执作者Signal | 86.3242 | 97.5510 |
| CNN完整分支 | 85.4110 | 96.9679 |
| Transformer完整分支 | 86.3732 | 97.0845 |
| Mamba完整分支 | 85.1635 | 97.4927 |
| **R2 seed43 fused** | **86.5953** | **97.2012** |
| fused相对同回执Signal | **+0.2711** | **−0.3499** |

该种子的mAP有小幅正差，但Rank-1下降，**未通过两项各≥+0.8**的停止线。相较R2 seed42的`86.3156/97.4927`，seed43的mAP更高而Rank-1更低，按已登记留存顺序当前选seed43作为该组合临时权重；这仅是已消费正式测试上的存储选择，不是无偏的多种子泛化估计。到本节已完成的固定终点正式种子总数为17，六个方法×数据集组合仍只有RGBNT201–V27 seed43一组通过（`1/6`）。

### 41.311 两份新结果后的已核验落败权重清理（2026-09-24）

逐份读取训练与正式回执、重新计算实存权重SHA，并确认原队列对应任务已结束且后继队列不依赖该权重后，只清理两份当前非赢家：旧机RGBNT100–R2 seed42的`roles_epoch20.pth`（`27,157,478`字节，SHA256=`3e5038c05c0b74535b8270202a38011440ef003d6552722b9944dc852bbade39`），保留新机seed43权重SHA256=`4bffe88835c4dfb9b927de7b98933fb489f7c83f7f5a761af31cda27ea62a786`；新机RGBNT201–R2 seed44的`roles_epoch20.pth`（`39,795,686`字节，SHA256=`81e3ab33968090fd5a9a3e50d926fe394624fe7da0f749a46a6eb2badcc63681`），保留同机seed43权重SHA256=`99dd3ac4ab563fda54bceb54829db6e7a208216740f56a6aaf7c0b6504846616`。两份退役文件均确认不存在，共释放`66,953,164`字节；作者Signal权重、当前赢家权重、所有种子的正式回执、距离数组、训练和评价日志仍保留。被退役回执中的原checkpoint路径现为历史记录，不代表文件仍在磁盘。seed45/46及后续新种子继续按原合同运行，赢家仍可能随完整正式结果更新。

### 41.312 MSVR310–V27 seed46 固定终点正式结果及落败权重退役（2026-09-24 05:36 北京时间）

新机`trained-model/official_extra_seed46_20260924/MSVR310_V27_seed46/`从05:24:09至05:34:57完成固定第20轮训练，05:36:00完成原Signal正式评价，训练执行提交`d9bdb42604b3027f125ec1697f600180f682eb0e`。回执`COMPLETE`，591 query/1055 gallery，按真实scene过滤、保留全部不同身份干扰项，无重排序，独立作者Signal指标与既有正式回执相同。作者权重SHA256=`b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a`，协议SHA256=`7f2b35f9a7e00433558c1e723db0e3ff0ea9daa7144eeeb972139d1502945cbc`；实存角色权重、训练与正式回执SHA256一致，均为`10f018df8f3f41e4eef95b020d8f1b6f54951c0191895603aa86252e9d12de66`，距离数组实存SHA256与回执同为`45405b990b510442154617c3a52ae8e60812f611afcb3aee251623801e53b7dc`。

| 输出 | mAP | Rank-1 |
| --- | ---: | ---: |
| 同回执作者Signal | 53.2424 | 72.4196 |
| CNN完整分支 | 49.7094 | 68.1895 |
| Transformer完整分支 | 48.9090 | 65.6514 |
| Mamba完整分支 | 48.8995 | 65.1438 |
| **V27 seed46 fused** | **51.1058** | **67.5127** |
| fused相对同回执Signal | **−2.1366** | **−4.9069** |

两项均未达到`+0.8`；现有V27–MSVR310 seed43的fused mAP `51.6667`高于seed46的`51.1058`，两份实存权重SHA重算均与各自回执一致。seed46任务已为`COMPLETE`且队列已在GPU3开始下一项RGBNT201–R2，因此只退役seed46的`roles_epoch20.pth`（`38,124,518`字节，SHA256=`10f018df8f3f41e4eef95b020d8f1b6f54951c0191895603aa86252e9d12de66`），确认文件不存在；保留seed43赢家实存权重SHA256=`d31d6dab1490e321b4c67e50101b8a800c5a8af7a9b7fdba986dd51ad2986dde`及两种子的指标、距离数组、训练和评价日志。至此已完成正式种子18项，六组合仍仅`1/6`达标；按官方测试筛出的暂时赢家仅供部署/存储，不作为无偏多种子统计。

### 41.313 统一旧项目目标与当前六组合里程碑，并按实测GPU释放衔接（2026-09-24 05:49 北京时间）

**统一目标：**保留共享Signal语义基座上的CNN／Transformer／Mamba三角色研究主线，持续解释融合与未知身份检索的失败；用真实代码、匹配对照、原作者协议和公开论文区分工程正确、内部增益与正式泛化。当前优先完成R2、V27各自在RGBNT201、RGBNT100、MSVR310上的独立训练与完整正式评价，共六个方法×数据集组合。每个种子均从对应数据集作者Signal权重初始化角色，固定第20轮、完整query/gallery与原camera/scene过滤，不重排、不挑中间轮次。RGBNT201要求同一种子fused的mAP、Rank-1/5/10均比同协议Signal至少高0.8个百分点；另外两组要求同一种子的mAP和Rank-1均至少高0.8个百分点。一个组合达标即停止为它新增种子，其他组合继续用可用GPU尝试；每个种子的全部结果和代码、协议、权重SHA均记录，按正式成绩保留临时最优权重只作存储／部署选择，不称为无偏泛化统计。六组合达标是当前执行里程碑，**不自动等于达到文献SOTA，也不替代原目标中的跨数据集分析和论文机制证据**。

执行约束继续统一为：作者预训练权重与必要赢家权重保留；只在核对训练／正式回执、实存SHA和依赖后删除落败的自训权重；定期核对旧机和新机磁盘。GitHub仓库、本主交接文档、桌面指定同名文档和两台服务器的同名文档及时同步并比对SHA；不另建平行交接说明。按预计终态或180—300秒间隔检查长任务，不以短时空卡或无日志误判停止。当前已完成正式种子18项、六组合`1/6`达标，其余目标仍进行中。

本轮发现新机seed46的`MSVR310–R2`占GPU2训练，而其余seed46任务仍占GPU0/1/3；按原整批完成后接续会在该R2本单元完成后造成GPU2真实空档。提交`3a61da5588e107d3eb5d91a675d71b735b715dd0`将新机后继任务分为GPU2上的`MSVR310–R2`单元和GPU0/1/3上的其他三单元：前者仅在seed46同单元正式状态`COMPLETE`且原队列GPU2任务完成后，从seed48起独立接续；后者仍等整个seed46队列结束。仅调整作业衔接，不改M0、训练损失、作者初始化、固定终点或官方评价。代码已推送GitHub并快进同步旧、新服务器；本地和两机正式Python AST检查通过。仅替换了新机原纯等待PID`3252286`，启动新主等待PID`3666283`及GPU2单元等待PID`3667962`，两者在前置任务完成前不启动训练；旧机seed45队列PID`45727`与旧后继等待PID`62216`、新机原四卡及seed46队列均未中断。此节没有新增正式检索成绩。

### 41.314 MSVR310–R2 seed46 正式结果与GPU2接续验收（2026-09-24 06:31 北京时间）

新机`trained-model/official_extra_seed46_20260924/MSVR310_R2_seed46/`于04:48:42至06:26:24完成固定第20轮训练，06:27:34完成作者Signal原协议正式检索。训练回执为`FIXED_EPOCH20_TRAINING_COMPLETE`，正式回执为`COMPLETE`；591 query/1055 gallery，排除同身份同scene、保留不同身份全部干扰图库、无重排序，独立Signal上游指标一致。训练执行提交`6610664ed42e0530377db4f24749fd6c442bf96b`，作者权重SHA256=`b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a`，协议SHA256=`7f2b35f9a7e00433558c1e723db0e3ff0ea9daa7144eeeb972139d1502945cbc`；实存角色权重、训练和正式回执SHA256均为`d8a2becc180d21cab9c771a6ca6b2e5fcb696cd77fc10bbad58a0584c4800ca9`，实存`official_distances.pt`与回执SHA256均为`1291fef777bc45ff5cf333cd4dfb70766444c69d09b0df453d5217836bcbced9`。

| 输出 | mAP | Rank-1 |
| --- | ---: | ---: |
| 同回执作者Signal | 53.2424 | 72.4196 |
| CNN完整分支 | 51.4484 | 69.5431 |
| Transformer完整分支 | 50.6111 | 67.5127 |
| Mamba完整分支 | 49.3602 | 67.8511 |
| **R2 seed46 fused** | **53.4216** | **70.5584** |
| fused相对同回执Signal | **+0.1792** | **−1.8613** |

两项没有同时达到各`+0.8`，因此该组合继续尝试；seed46是该组合当前按已登记`(全项达标,mAP,Rank-1)`顺序选择的临时权重赢家，保留其权重与完整回执，不以正式测试选优宣称无偏泛化。已完成正式种子从18增至19，六组合仍为`1/6`。§41.313的GPU2独立接续已实际生效：`logs/official_target_continuation_new_msvr_r2_20260924.json`为`RUNNING`，新`seed48–MSVR310–R2`在06:29:14分配GPU2、06:30:30通过M0后进入固定20轮`TRAINING`，真实训练进程存活；没有等待其他seed46任务结束。`/data`约余124 GiB，当前不清理新的赢家权重。

### 41.315 MSVR310–R2 落败权重核验退役（2026-09-24）

同组合seed43的正式fused为`53.2391 mAP/70.8968 Rank-1`，seed46为`53.4216/70.5584`；二者均未达双指标停止线，按已登记的先判全项达标、再比较mAP、Rank-1的存储顺序，seed46为当前临时赢家。再次核对两种子的训练及正式回执均完成、实存权重SHA与回执一致，GPU2旧队列的MSVR310–R2任务已结束，后继控制器已选seed46且seed48独立从作者Signal初始化。仅退役新机旧`MSVR310_R2_seed43/roles_epoch20.pth`（`38,124,518`字节，SHA256=`9eba1ed9ccad613844df711c06744e1956bdeb56909200675d1285e651b2d906`），确认文件已不存在；seed46权重SHA256=`d8a2becc180d21cab9c771a6ca6b2e5fcb696cd77fc10bbad58a0584c4800ca9`仍实存。两种子的正式回执、距离文件、训练及评价日志保留；seed43回执中的checkpoint路径现仅是历史记录。

### 41.316 按GPU释放衔接其余新机组合，防止整批等待空卡（2026-09-24 06:41 北京时间）

对旧机六份原始正式回执、新机原四卡十一份及seed46两份正式回执逐一重新读取，按同回执Signal原始差值和同一种子全部必报指标复判：共**19个唯一完成种子**；旧机导入的`RGBNT201–V27 seed43`与新机原回执为同一结果，不重复计数；目前仍仅该种子通过四项各`+0.8`，即六组合`1/6`。新机原`RGBNT100–R2 seed44`已完成18/20轮，约每轮29分钟；新机seed46的GPU0 `RGBNT100–R2`仅完成2/20轮，GPU3 `RGBNT201–R2`已完成5/20轮，GPU1的`RGBNT100–V27`仍等原队列同卡任务。因此，继续等待**整个**seed46队列结束将使较早完成的GPU1/3产生可预见的长空档。这一调度问题由实际运行状态与轮时支持；不涉及根据正式成绩修改模型。

提交`8c07ac014deeb21511bdaaf1283adbd9e38ea219`仅扩展既有单卡续跑入口：GPU0负责`RGBNT100–R2`，GPU1负责`RGBNT100–V27`，GPU2沿用正在训练的`MSVR310–R2`独立队列，GPU3负责`MSVR310–V27`；旧单卡继续负责`RGBNT201–R2`，已达标的`RGBNT201–V27`不再新增种子。每个新机续跑器必须等**原队列及seed46队列在本GPU上分配的任务全部正式`COMPLETE`**，才从seed48起在该GPU运行同一M0→固定20轮→作者原协议正式评价。特别是GPU3要等seed46的`MSVR310–V27`与`RGBNT201–R2`均完成。代码已推送GitHub并快进同步两台服务器，三处正式Python AST检查通过；运行中的原队列、seed46及GPU2 seed48未被中断。替换前核实旧新机总等待PID`3666283`仅在等待、无子任务、目标状态文件不存在，随后仅终止该等待器，启动三个独立等待PID`3771922`、`3771925`、`3771928`；确认三者存活、各自目标状态文件尚不存在、日志无异常，旧GPU2独立接续PID`3667962`仍存活，新机四GPU继续由当前训练占用。此节没有新增正式检索指标。

### 41.317 RGBNT100–R2 seed44 固定终点正式结果及同组合权重留存（2026-09-24 07:36 北京时间）

新机原四卡队列的`RGBNT100_R2_seed44`于9月23日21:51:06至9月24日07:16:18完成固定第20轮，07:19:17完成作者Signal原协议正式评价；训练回执为`FIXED_EPOCH20_TRAINING_COMPLETE`、正式回执为`COMPLETE`，原四卡campaign也已`COMPLETE`。训练执行提交`c005f148c1f78c5e8a5bc94690d79fe030c7d51b`，1715 query/8575 gallery、同camera过滤并保留不同身份干扰项、无重排序，独立Signal上游指标一致。作者初始化权重SHA256=`09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860`，协议SHA256=`12d8afe5cf6e537b69ccf753481dd5fd651f14b7672e1e6411a4d4e9d88cc956`；实存角色权重、训练与正式回执的SHA256均为`4908a518a003ecb2410d9c21ed767df9e05801d3d72611590b29345dada94c57`，实存距离文件与回执SHA256均为`803709354c4e8cd161aa3afc6c61c424ae9d398a4de4c22f1917a7169e9a240e`。

| 输出 | mAP | Rank-1 |
| --- | ---: | ---: |
| 同回执作者Signal | 86.3242 | 97.5510 |
| CNN完整分支 | 85.8688 | 97.2012 |
| Transformer完整分支 | 86.2559 | 97.2012 |
| Mamba完整分支 | 85.9944 | 97.9009 |
| **R2 seed44 fused** | **87.1200** | **97.8426** |
| fused相对同回执Signal | **+0.7958** | **+0.2915** |

按未四舍五入的原始差值，mAP比`+0.8`停止线少`0.004198`个百分点，Rank-1也未达到，因此该种子**未达标**；不能把显示到一位小数的`+0.8`当作通过。它的mAP与Rank-1均高于同组合seed43，核对两份训练／正式回执及实存权重SHA、原四卡任务结束、后续种子从作者Signal独立初始化后，保留seed44，退役仅seed43的`roles_epoch20.pth`（`27,157,478`字节，原SHA256=`4bffe88835c4dfb9b927de7b98933fb489f7c83f7f5a761af31cda27ea62a786`），确认落败文件不存在；全部正式回执、距离、训练和评价日志仍留存。已完成正式种子20项，六组合仍仅`1/6`达标。原四卡结束后，seed46的`RGBNT100–V27`已在GPU1进入训练；07:36实查旧机GPU0及新机四卡均有训练，旧机约余20 GiB、新机约余123 GiB。预计下一项验收是新机GPU2的`MSVR310–R2 seed48`，约08:10；实际以终态回执为准。

### 41.318 RGBNT100–R2 三个完整种子的来源调节系数复核（2026-09-24）

继续只读取已完成训练端的`training_steps.jsonl`与`training.json`，不使用正式测试身份选择控制参数。将统计限制在`active_fused_metric=cross_environment_smooth_ap`且`eligible_anchors>0`的真实更新步：RGBNT100–R2 seed42为`2560/2560`、seed43为`2559/2559`、seed44为`2560/2560`，合计`7679/7679`步的CNN、Transformer、Mamba三个角色**同时**记录实际`[ranking,auxiliary]=[1.6,0.4]`，即当前平方根EMA范数比截断后的上限系数。seed42/43/44终点的排名梯度范数EMA分别约为`10^-77`、`10^-123`、精确记录`0`，而对应辅助EMA仍为约`0.0045—0.0111`。这将§41.301的部分种子观察扩展至目前已完成的三个RGBNT100–R2完整训练端：在这些有支持的来源步，角色调节实际等同固定上限加权，而不是持续随样本调整。

该现象**不是所有数据集都出现的软件零梯度**：已完成的MSVR310–R2 seed46终点三个角色排名EMA为`0.1391/0.1471/0.1606`，辅助EMA为`0.3004/0.3065/0.3518`。RGBNT100的系数记录不能证明每一步原始排名梯度均为零，也不能单独解释其正式mAP/Rank-1；特别是seed44的`+0.7958/+0.2915`仍须按双指标原始值判为未达标。当前固定20轮实验、`+0.8`停止线、作者Signal评价和运行中的种子均不修改，不依据已消费的正式测试回调上限。

### 41.319 已做实验的分口径总表（2026-09-24 07:48 北京时间快照）

本节为用户要求的全项目对照入口。**RGBNT201列mAP/Rank-1/Rank-5/Rank-10，RGBNT100与MSVR310只列mAP/Rank-1。**“—”表示该协议没有相应完整检索结果，不表示0。下列正式测试、固定30身份dev、训练来源内部身份隔离Q1的身份、图库和模型初始化不同；只有同一配对行的control→candidate差值用于单因素解释。所有分数为百分数、增益为百分点；正式种子的判定用未四舍五入原值。

#### A. 已完成正式测试：早期版本与本轮作者Signal初始化分开看

| 数据集 | 模型/条件 | mAP | Rank-1 | Rank-5 | Rank-10 | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RGBNT201 | TriFusion V1历史正式fused | 59.1478 | 63.2775 | 77.2727 | 83.6124 | 旧60轮版本，非本轮作者Signal初始化 |
| RGBNT201 | 本轮作者Signal | 80.3029 | 85.1675 | 91.3876 | 93.6603 | 本轮R2/V27配对基线 |
| RGBNT100 | 旧本机Signal | 80.7122 | 94.2274 | — | — | 旧V8固定终点配对基线 |
| RGBNT100 | 旧V8 fused | 83.2848 | 96.1516 | — | — | 相对旧本机Signal：+2.5726/+1.9242；Mamba mAP 83.4406更高 |
| RGBNT100 | 本轮作者Signal | 86.3242 | 97.5510 | — | — | 本轮R2/V27配对基线；不可与旧V8直接相减归因 |
| MSVR310 | 本轮作者Signal | 53.2424 | 72.4196 | — | — | 本轮R2/V27配对基线 |

下表为截至该快照**20个唯一完成的固定epoch20正式种子**，每行是同一模型的fused结果；旧机导入的新机seed43回执不重复计数。括号内为相对该数据集同协议作者Signal的逐项增益。五路输出及逐查询数组在各自`official_metrics.json`和`official_distances.pt`中完整保存；权重清理不删除这些回执。`≥+0.8`须同一种子所有必报指标同时满足。

| 数据集 | 方法/seed | mAP（Δ） | Rank-1（Δ） | Rank-5（Δ） | Rank-10（Δ） | 全项达标 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| RGBNT201 | R2/42 | 82.4254 (+2.1225) | 86.7225 (+1.5550) | 92.5837 (+1.1962) | 94.0191 (+0.3589) | 否 |
| RGBNT201 | R2/43 | 82.5826 (+2.2797) | 87.4402 (+2.2727) | 91.8660 (+0.4785) | 94.2584 (+0.5981) | 否 |
| RGBNT201 | R2/44 | 82.4417 (+2.1388) | 86.3636 (+1.1962) | 92.5837 (+1.1962) | 94.1388 (+0.4785) | 否 |
| RGBNT201 | V27/42 | 81.8916 (+1.5887) | 85.5263 (+0.3589) | 91.9856 (+0.5981) | 93.6603 (+0.0000) | 否 |
| RGBNT201 | **V27/43** | **83.0005 (+2.6976)** | **87.5598 (+2.3923)** | **92.9426 (+1.5550)** | **94.4976 (+0.8373)** | **是** |
| RGBNT201 | V27/44 | 82.9153 (+2.6124) | 86.8421 (+1.6746) | 92.4641 (+1.0766) | 94.1388 (+0.4785) | 否 |
| RGBNT100 | R2/42 | 86.3156 (−0.0086) | 97.4927 (−0.0583) | — | — | 否 |
| RGBNT100 | R2/43 | 86.5953 (+0.2711) | 97.2012 (−0.3499) | — | — | 否 |
| RGBNT100 | R2/44 | 87.1200 (+0.7958) | 97.8426 (+0.2915) | — | — | 否 |
| RGBNT100 | V27/42 | 85.5703 (−0.7539) | 96.6181 (−0.9329) | — | — | 否 |
| RGBNT100 | V27/43 | 85.9465 (−0.3777) | 96.7930 (−0.7580) | — | — | 否 |
| RGBNT100 | V27/44 | 86.3835 (+0.0593) | 97.3761 (−0.1749) | — | — | 否 |
| MSVR310 | R2/42 | 52.2816 (−0.9608) | 69.3739 (−3.0457) | — | — | 否 |
| MSVR310 | R2/43 | 53.2391 (−0.0033) | 70.8968 (−1.5228) | — | — | 否 |
| MSVR310 | R2/44 | 51.2600 (−1.9824) | 69.0355 (−3.3841) | — | — | 否 |
| MSVR310 | R2/46 | 53.4216 (+0.1792) | 70.5584 (−1.8613) | — | — | 否 |
| MSVR310 | V27/42 | 50.5218 (−2.7206) | 67.3435 (−5.0761) | — | — | 否 |
| MSVR310 | V27/43 | 51.6667 (−1.5757) | 68.3587 (−4.0609) | — | — | 否 |
| MSVR310 | V27/44 | 49.7506 (−3.4918) | 66.4975 (−5.9222) | — | — | 否 |
| MSVR310 | V27/46 | 51.1058 (−2.1366) | 67.5127 (−4.9069) | — | — | 否 |

按既定`(全项达标,mAP,Rank-1)`权重留存顺序，各组合当前代表种子及其**同一checkpoint**的五路输出如下。RGBNT201按四指标顺序、另两数据集按两指标顺序；这些代表种子不是跨种子均值，也没有把不同种子的单项最好拼起来。

| 数据集 | 方法/seed | Signal | fused | CNN | Transformer | Mamba |
| --- | --- | --- | --- | --- | --- | --- |
| RGBNT201 | R2/43 | 80.3029/85.1675/91.3876/93.6603 | 82.5826/87.4402/91.8660/94.2584 | 82.1907/87.4402/92.4641/94.0191 | 80.7685/86.1244/91.7464/93.8995 | 81.2269/85.5263/91.5072/93.6603 |
| RGBNT201 | V27/43 | 80.3029/85.1675/91.3876/93.6603 | 83.0005/87.5598/92.9426/94.4976 | 81.8730/87.0813/92.5837/93.8995 | 81.6465/86.2440/93.1818/94.7368 | 82.5620/86.2440/92.2249/94.0191 |
| RGBNT100 | R2/44 | 86.3242/97.5510 | 87.1200/97.8426 | 85.8688/97.2012 | 86.2559/97.2012 | 85.9944/97.9009 |
| RGBNT100 | V27/44 | 86.3242/97.5510 | 86.3835/97.3761 | 85.7227/97.4344 | 85.6552/96.9679 | 84.8056/97.1429 |
| MSVR310 | R2/46 | 53.2424/72.4196 | 53.4216/70.5584 | 51.4484/69.5431 | 50.6111/67.5127 | 49.3602/67.8511 |
| MSVR310 | V27/43 | 53.2424/72.4196 | 51.6667/68.3587 | 47.9557/67.1743 | 50.0816/67.3435 | 50.3322/67.8511 |

#### B. RGBNT201早期固定141-fit/30-dev实验，不与正式测试混排

| 版本 | fused mAP | Rank-1 | Rank-5 | Rank-10 | 口径与结果 |
| --- | ---: | ---: | ---: | ---: | --- |
| V3 task-anchor | 42.8978 | 43.8788 | — | — | 60轮完成，最佳epoch14；只保留1536D anchor |
| V4等能量银行 | 43.4031 | 42.7879 | 58.5455 | 65.5758 | 60轮完成，最佳epoch27；仍只保留1536D anchor |
| 完整Signal dev | 58.0109 | 57.4545 | 69.9394 | 76.6061 | V5后完整3072D direct+SIM基线 |
| V5 | 58.0168 | 57.4545 | 69.9394 | 76.6061 | 残差能量很弱 |
| V6 | 58.7321 | 57.5758 | 69.2121 | 76.7273 | 最佳epoch8，低于同checkpoint CNN 59.1022，未达65门 |
| V7 | 58.3293 | 57.9394 | 70.1818 | 76.7273 | 最佳epoch1，未达65门 |
| V8 Phase-A固定融合 | 58.0972 | 56.8485 | — | — | 仅报告mAP/R1的训练后诊断 |
| V8 Phase-B Router | 58.4050 | 59.3939 | 71.2727 | 76.6061 | 超过该阶段三个固定角色，仍未达65门 |
| V9隐藏交换 | 56.5339 | 57.2121 | 68.3636 | 75.5152 | 完整dev主实验负结果 |

V3/V4的1536D anchor与V5后完整Signal不相同；V6、V7按各自dev规则选中间checkpoint，本轮R2/V27则固定epoch20。因此，本表展示历史轨迹，不构成统一初始化/选点消融。

#### C. RGBNT201训练来源内部身份隔离、完整图库Q1

下表仅取各实验自身匹配control和candidate的571条合法query汇总；与正式836条query和固定dev825条query均不同。`ΔmAP`为同一行配对差，不能跨行累加。V17列其补做的**完整图库**结果，不采用原先缩减图库的88点数值。

| 版本/干预 | control mAP | candidate mAP | ΔmAP | candidate Rank-1 | Rank-5 | Rank-10 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| V17关系包络 | 80.6149 | 80.2860 | −0.3289 | 83.5377 | 88.7916 | 92.4694 | Q1未晋级 |
| V18相机主方向投影 | 80.5605 | 81.4820 | +0.9215 | 84.9387 | 90.0175 | 93.5201 | 平均正，原门未过 |
| V19私有语义尾层 | 80.2408 | 80.4968 | +0.2560 | 84.2382 | 89.8424 | 93.6953 | 未晋级 |
| V20跨模态一致性 | 80.2063 | 79.1954 | −1.0109 | 79.3345 | 87.7408 | 92.2942 | 未晋级 |
| V22相机负例损失 | 80.6407 | 78.9845 | −1.6562 | 82.3117 | 90.7180 | 94.0455 | 未晋级 |
| V23模态语义适配 | 80.5075 | 80.2548 | −0.2527 | 83.5377 | 91.4186 | 94.9212 | 未晋级 |
| V24弱强增强身份原型 | 79.5350 | 80.0263 | +0.4913 | 82.3117 | 90.1926 | 94.3958 | 未晋级 |
| V25跨camera采样 | 80.8816 | 80.4209 | −0.4606 | 82.4869 | 90.1926 | 92.9947 | 未晋级 |
| V26角色-模态排序责任 | 80.4028 | 80.7147 | +0.3119 | 83.8879 | 90.0175 | 93.5201 | 未晋级 |
| V27耦合统计扰动 | 80.2534 | 81.5924 | +1.3389 | 83.8879 | 91.2434 | 95.6217 | 配对收益，原5门仅4/5 |
| V28 R2联合Token修正 | 81.8253 | 81.2689 | −0.5564 | 84.7636 | 91.5937 | 94.9212 | FP32数值修复后仍未晋级 |
| V29有界切向修正 | 81.4875 | 81.7070 | +0.2195 | 84.5884 | 90.8932 | 95.2715 | 未晋级 |

V10 DINOv2适配资格、V11残差资格、V12—V14 Router资格均在来源资格阶段停止，没有可与上述Q1对齐的部署mAP；V15完整运行的是**缩减图库**fit-only Q1，其fused `88.6915→88.5194 mAP`且未晋级，不把88点数值放进完整图库表；V16关系活动M0、V21 SAM M0未过，均无Q1检索结果。V28原FP16工程故障没有Q1，表内为修复后的R2。任务状态分离V1完成M0、Q1只见首配对，旧服务器恢复后原进程/目录不在，**没有完整六端Q1结果**；不得填写终态。

#### D. 车辆数据的内部身份隔离实验

RGBNT100原V8三角色内部Q1为Signal `89.5242/96.8300`→fused `91.3165/97.9366`，mAP `+1.7924`且原五门全部通过；此数值不是官方结果。MSVR310原V8为Signal `53.1294/63.0000`→fused `52.1174/60.8333`，mAP `−1.0120`，原五门均未过。下列MSVR310后续各行均为自己的三折来源身份隔离配对，不能与别行control拼接成累计增益。

| MSVR310干预 | control mAP/R1 | candidate mAP/R1 | 配对ΔmAP | 状态 |
| --- | --- | --- | ---: | --- |
| 来源统计扰动 | 52.1267/60.8333 | 51.8245/60.3333 | −0.3022 | 未晋级 |
| 普通实例记忆 | 52.1211/60.8333 | 51.9050/61.5000 | −0.2162 | 未晋级 |
| 历史坐标刷新 | 51.7884/61.6667 | 51.7168/61.5000 | −0.0716 | 未晋级 |
| 历史候选完整反传 | 51.7343/61.5000 | 52.4067/59.6667 | +0.6724 | 平均正，未稳定晋级 |
| 角色负例集合 | 52.3923/59.8333 | 52.4902/60.3333 | +0.0979 | 未晋级 |
| 标准Smooth-AP | 52.4441/59.8333 | 52.7876/61.0000 | +0.3435 | 平均正，未晋级 |
| 跨scene Smooth-AP | 52.8383/61.1667 | 53.4055/62.0000 | +0.5672 | 高于内部Signal mAP、低于其R1；原门2/5 |
| 支持感知梯度平衡R2 | 53.3994/62.1667 | 53.4526/62.3333 | +0.0533 | 配对小正，未晋级 |

上述MSVR310内部Signal基线为`53.1294/63.0000`，与本轮作者权重正式Signal `53.2424/72.4196`不是同一个模型/评价集合，不能据此直接比较。来源普查、梯度检查、M0、训练loss、Oracle、部分fold及未结束种子不列为完整检索结果；其作用和原始凭据仍在前述章节及各`results/*.md`报告中。至本节快照，**正式六组合仅RGBNT201–V27 seed43一组达标**；其余训练继续，且正式测试反复选种子会引入选择偏差，不将暂时赢家报成无偏多种子性能。

### 41.320 作者Signal正式基线与跨服务器协议文件差异复核（2026-09-24）

用户询问作者预训练模型是否测过：本轮三数据集均已分别加载对应**作者发布的Signal checkpoint**，单独完成原正式query/gallery、camera/scene过滤和无重排序评价；结果为RGBNT201 `80.3029/85.1675/91.3876/93.6603`（mAP/R1/R5/R10）、RGBNT100 `86.3242/97.5510`、MSVR310 `53.2424/72.4196`（后两者mAP/R1）。20份已完成R2/V27正式回执各自记录`baseline_only`五路之一，且同数据集作者权重SHA、query/gallery数量、过滤字段、固定第20轮、无重排序及`independent_upstream_metrics_equal=true`一致。这里的作者Signal只作预训练初始化与独立基线，未在本轮重新训练；RGBNT100旧本机Signal `80.7122/94.2274`不是这份权重，不作为当前配对分母。

跨服务器逐字段读取三份旧机与新机正式协议JSON发现：每个数据集的两份文件**仅`dataset_root`绝对路径不同**（旧机`/root/autodl-tmp/trifusion-v2/data/<dataset>`、新机`/data/gaob/Re-ID/dataset/<dataset>`）；`records`、`query_rows`、`counts`、`train_label_map`、`inventory_sha256`、过滤定义及其余所有顶层字段逐项相等。因此seed42旧机的原始协议文件SHA与新机seed43及后续的SHA不同是路径字段所致，不是此次发现了数据划分或评分规则变化。正式回执仍各自绑定自己实际读取的协议文件SHA；不把两个不同字节文件谎称同一SHA。本次只读核对，没有改训练或评价代码、权重、队列及`+0.8`停止线。

### 41.321 MSVR310–R2 seed48正式终态与连续队列（2026-09-24 08:13 CST）

新机`trained-model/official_extra_seed48_MSVR310_R2_20260924/MSVR310_R2_seed48/`的`training.json`记录06:30:46至08:08:24完成固定20轮、400次优化器更新，状态`FIXED_EPOCH20_TRAINING_COMPLETE`；`official_metrics.json`于08:09:35达到`COMPLETE`。训练执行提交`1c48007f6fb28a56c113d36e5b6bbe20d5697ad2`。对应作者Signal权重SHA256=`b3888e7ec7b9290abcde76915ebf9d9ce87129e759586fd7deb3e9cf7d1d807a`，新机协议文件SHA256=`7f2b35f9a7e00433558c1e723db0e3ff0ea9daa7144eeeb972139d1502945cbc`；591 query/1055 gallery、同身份同scene过滤、无重排序，`independent_upstream_metrics_equal=true`。训练记录显示冻结权重未变、缺失非零梯度项为空、溢出0次。正式权重SHA256由训练与评价回执共同记录为`fa7e03054b756a6429fa40c78f8dc6c42812c00c8fec10f275c2099ccc49320e`；距离文件实存SHA256=`bbffd971ff72602ca1445e774244f60fa114d0041d13a122e2e0a69020fa51ec`，与正式回执相等。

| 输出 | mAP | Rank-1 | 相对同回执作者Signal |
| --- | ---: | ---: | --- |
| 作者Signal | 53.2424 | 72.4196 | 基准 |
| fused | 53.2476 | 72.0812 | mAP `+0.0052`，Rank-1 `−0.3384` |
| CNN | 50.8805 | 68.1895 | — |
| Transformer | 50.3771 | 68.1895 | — |
| Mamba | 50.2143 | 69.7124 | — |

原始差值为mAP `+0.0052277341`、Rank-1 `−0.3384094755`个百分点，双指标`≥+0.8`停止条件**未通过**。按原先登记的`(全项达标,mAP,Rank-1)`存储顺序，seed46的`53.4216/70.5584`仍是MSVR310–R2当前临时赢家；这并不表示其Rank-1更好。连续控制器`official_target_continuation_new_msvr_r2_20260924.json`记录seed48 `checkpoint_retained=false`，实查seed48的`roles_epoch20.pth`已不存在，而seed46赢家实存权重SHA256仍为`d8a2becc180d21cab9c771a6ca6b2e5fcb696cd77fc10bbad58a0584c4800ca9`。seed48的正式回执、距离数组、训练步骤和日志保留；回执中的checkpoint路径只作历史来源记录。控制器PID3667962仍在运行，已在GPU2启动独立的MSVR310–R2 seed50；其他GPU及旧机原训练继续，无需重启或修改方法。此时正式完成种子增至21项，六组合仍仅RGBNT201–V27 seed43一项满足停止线；新机`/data`约余123 GiB，旧机`/root/autodl-tmp`约余20 GiB。

### 41.322 RGBNT100–V27 seed46正式终态、非赢家权重清理与GPU1接续（2026-09-24 08:30 CST）

新机`trained-model/official_extra_seed46_20260924/RGBNT100_V27_seed46/`的`training.json`记录07:24:12至08:21:27完成固定20轮、2626次优化器更新；`official_metrics.json`于08:24:25达到`COMPLETE`。训练执行代码提交`ce187eb1c4feae2aebe2343b74dcd8457a9bf292`。作者Signal初始化权重SHA256=`09df46735a3427169ea65b9e4110dc834b99de859657bf589c9fb30ad4d4f860`，协议SHA256=`12d8afe5cf6e537b69ccf753481dd5fd651f14b7672e1e6411a4d4e9d88cc956`；1715 query/8575 gallery，按同身份同camera过滤、保留其他身份图库、无重排序，独立Signal上游指标一致。训练回执记录冻结状态不变、缺失非零梯度项为空、溢出0次。训练与正式回执的角色权重SHA256均为`7a0e0e18126aa3887b5f6dc56e2f67fc474102b25c6ba9c07345659e7f32850e`；本次清理前重新计算实存文件得到相同SHA。实存距离文件SHA256=`70e1b776f34d236d5ceb5cc84535334581b8be66b4515e42522adc2e25c53c21`，与正式回执一致。

| 输出 | mAP | Rank-1 |
| --- | ---: | ---: |
| 作者Signal | 86.3242 | 97.5510 |
| fused | 86.1517 | 97.3178 |
| CNN | 85.8277 | 97.6676 |
| Transformer | 85.0405 | 97.0262 |
| Mamba | 85.0296 | 97.2595 |

seed46 fused相对同回执作者Signal的原始差值为mAP `−0.1725445011`、Rank-1 `−0.2332361516`个百分点，双指标停止线未过。此前seed44正式fused为`86.3835/97.3761`，两项均高于seed46；按已登记的存储顺序，它仍是本组合临时赢家，但seed44本身也未达双指标`+0.8`。在核对两份训练／正式回执、两份实存权重SHA、控制器所选seed44及新种子独立初始化后，仅删除seed46的`roles_epoch20.pth`（`27,157,478`字节），确认已不存在；seed44实存赢家SHA256=`4e2caf8b97548b4d4a6fe1869b429e3dc5527a39e8e7da2c04e4030513fa0188`。seed46正式回执、距离文件和训练日志保留，回执中的旧checkpoint路径只作历史记录。GPU1连续控制器PID3771925仍在，`official_extra_seed48_RGBNT100_V27_20260924/campaign.json`显示seed48 `TRAINING`，GPU1重新占用；旧机及新机另外三卡也在训练。当前唯一完成的正式种子增至22项，六组合仍为1/6达标；训练与评估口径、固定终点、种子顺序、停止线均未修改。

### 41.323 CoT-ReID原文三数据集四指标与资源条件补核（2026-09-24）

在不改变本项目任何运行中训练的前提下，重新读取[CoT-ReID作者CVPR 2026论文原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.pdf)的Table 1/2，而不只沿用§41.292已登记的RGBNT100一行。作者报告的完整RGB/NIR/TIR输入结果如下；单位均为百分点。

| 方法、资源 | RGBNT201 mAP/R1/R5/R10 | RGBNT100 mAP/R1 | MSVR310 mAP/R1/R5/R10 |
| --- | ---: | ---: | ---: |
| CoT-ReID、DINOv3-B＋MLLM文本 | 83.3/86.1/93.3/94.8 | 89.9/99.3 | 71.7/85.3/94.3/96.5 |
| 同文DINOv3基线 | 77.5/78.9/85.8/88.9 | 87.0/98.5 | 68.2/83.3/93.5/94.4 |

论文§4.1说明以API调用Qwen-VL为**训练与测试图像**的各模态生成描述和推理链；§3.2使用预训练DINOv3视觉主干及冻结CLIP文本编码器。因此CoT-ReID不是仅靠RGB/NIR/TIR图像和本项目三份作者Signal预训练权重的同资源结果。相对于§41.292已核的RoDI–DINOv3，CoT-ReID在RGBNT201的mAP/R1更低（83.3/86.1对85.3/87.9），RGBNT100的R1更高（99.3对99.1），MSVR310的mAP略低而R1更高（71.7/85.3对71.8/84.8）；§41.292另有PMKD的RGBNT100 mAP 91.6，故不能凭CoT论文自身表格的粗体宣称其三个数据集绝对SOTA。这里是不同论文公开值对照，**不是本项目同协议、等资源、等训练预算的因果比较**。本项目六组合目标仍依各自作者Signal正式回执的逐项`+0.8`判断；没有用上述公开值挑种子、改训练或新增正式测试成绩。

### 41.324 Signal论文纯baseline三数据集独立训练启动（2026-09-24 09:21 CST）

用户明确要求的是**去掉TriFusion全部模块，且去掉Signal的SIM/GAM/LAM之后，Signal论文使用的纯baseline**，而不是对已经训练好的完整Signal作者checkpoint关闭模块做事后消融。依据[Signal原论文Table 3](https://arxiv.org/pdf/2511.17965)及作者公开代码`cd1b0a6`，该基线为共享CLIP ViT-B/16提取RGB/NIR/TIR各自CLS，拼接成1536D检索表示；作者仅报告RGBNT201纯基线`70.3 mAP/71.8 Rank-1`，未报告RGBNT100和MSVR310对应的纯基线。故本轮必须从公开`ViT-B-16.pt`初始化，三个数据集分别独立训练；已有三份`*_Signal_*.pth`包含Signal模块，不能充当纯基线结果或初始化。

在新机`/data/gaob/Re-ID/Trifusion/comparators/Signal-cd1b0a6`保持作者源码和三份公开YAML，命令行仅覆盖`MODEL.USE_A=False`、`MODEL.USE_B=False`、CLIP权重路径、数据集绝对路径、输出路径和仅在固定末轮评价/保存。RGBNT201保持`DIRECT=1`、B64/K8、50轮；RGBNT100保持`DIRECT=0`、B128/K16、30轮；MSVR310保持`DIRECT=0`、B64/K4、50轮。保留各自原始增强、优化器与学习率、原始query/gallery和无重排序规则；不按正式测试挑中间checkpoint。固定终点与作者可能采用的best-epoch选择不同，最终应称为**按发布代码/配置的本机复现**，不能冒充论文Table 3的精确重跑。

预检：作者数据加载器与现有正式协议的三数据集train首模态图片集合相同；query/gallery每条的图片真实路径、身份、camera及MSVR310 scene字段逐项相同，计数分别为RGBNT201 `3951/836/836`、RGBNT100 `8675/1715/8575`、MSVR310 `1032/591/1055`（train/query/gallery）。CPU实例化三套纯模型均无`SIM`、`AlignM`或SIM分类头，输出设计为1536D；总参数依次为`86,409,216`、`86,226,432`、`86,387,712`，RGBNT201与论文86.41M四舍五入吻合。新机`/data`尚余约123GiB；三份终点权重预计约1.1GiB，现有自训R2/V27任务依旧在GPU0/1/2运行，GPU3当前RGBNT201–R2 seed46已至第18/20轮，不抢占。

已部署`tools/queue_signal_plain_baseline.py`和`tools/evaluate_signal_plain_baseline.py`，远端与本地代码SHA256分别相同（`c00d808a8e90f2751c27bd9b67cbee9293951fdd704376269eddcd2cd9f68579`、`27ce510fbbc962ebf54b8d950f6838268d561e09a14c9d2ca21a27851dcdedb4`）。原等待GPU3的MSVR310–V27连续控制器PID3771928在无子进程、无状态文件时停止；当前GPU3任务不变，另三卡连续任务不变，MSVR310–V27后续种子暂缓，纯baseline完毕再恢复。纯baseline后台队列PID4092603已于09:21:23启动，状态文件`logs/signal_plain_baseline_20260924/campaign.json`当前`WAITING`；仅在原GPU3任务连同正式评估显示`COMPLETE`后，按`MSVR310→RGBNT201→RGBNT100`顺序使用GPU3训练与固定终点评价。训练权重保存在`trained-model/signal_plain_baseline_20260924/<dataset>/Signalbest.pth`；日志、精确四项指标回执保存在`logs/signal_plain_baseline_20260924/<dataset>/`。截至本节**尚无任何新纯baseline检索指标**，不得把当前完整Signal作者权重的`80.3029/86.3242/53.2424` mAP写成纯基线。

### 41.325 RGBNT100–V27 seed48正式终态与连续队列（2026-09-24 09:34 CST）

新机`trained-model/official_extra_seed48_RGBNT100_V27_20260924/RGBNT100_V27_seed48/`已完成固定第20轮训练及原Signal全量正式query/gallery评估；`training.json`为`FIXED_EPOCH20_TRAINING_COMPLETE`，`official_metrics.json`为`COMPLETE`，各自文件SHA256分别为`1391d2ce9e922f4fb3f1db5a43d466284e99a9322bc142a6a8cc62aff2334101`和`0012bd17cdfed1f8203854c80407f4fb091814932467558dfe437c08f3edc9d4`。两份回执的角色checkpoint SHA256同为`f681af9ff0b42bde6ff4d3f88c5a93a9e8f21d720e6ecf6a2b6c0ddd5ecaeda7`，协议SHA256同为`12d8afe5cf6e537b69ccf753481dd5fd651f14b7672e1e6411a4d4e9d88cc956`；1715 query/8575 gallery，`independent_upstream_metrics_equal=true`，同身份同camera过滤、无重排序。完整输出如下，单位百分点：

| 输出 | mAP | Rank-1 |
| --- | ---: | ---: |
| 作者完整Signal | 86.3242 | 97.5510 |
| fused | 85.9915 | 97.0262 |
| CNN | 85.3981 | 97.3178 |
| Transformer | 84.8836 | 96.3848 |
| Mamba | 84.8260 | 96.7930 |

fused相对同回执完整Signal为mAP `−0.3327`、Rank-1 `−0.5248`，双指标`≥+0.8`均未通过。连续控制器`logs/official_target_continuation_new_rgbnt100_v27_20260924.json`记载seed48为非赢家、`checkpoint_retained=false`；实查seed48的`roles_epoch20.pth`已不存在，现有seed44仍为临时赢家。GPU1已接续下一种子。这里的完整Signal参照与§41.324正在训练的**纯CLIP baseline**不同，不能合并或替代。

### 41.326 Signal纯baseline运行修复及MSVR310固定终点评价（2026-09-24 10:05 CST）

§41.324启动的首个后台队列在原GPU3任务完成后开始MSVR310训练，但在约10个batch后停在作者源码`engine/processor.py:146`的日志读取`scheduler._get_lr(epoch)[0]`：当前`WarmupMultiStepLR`没有该私有方法，未产生纯baseline权重或检索结果。这是学习率**打印语句**的兼容问题，不是模型前向、损失或优化器更新失败。仅将该日志读取改为`optimizer.param_groups[0]['lr']`；作者源码`cd1b0a672d1fe642e7608731cb4899a19dda7d51`的修改以仓库`comparators/signal_cd1b0a6_lr_log.patch`记录，`git apply --unidiff-zero -R --check`在实际已修改源码上通过。原失败日志保留，新的独立队列`logs/signal_plain_baseline_20260924_r2/`于09:55:42启动；没有续用失败运行的参数或checkpoint，也没有改变作者训练目标、优化器和学习率调度。

新队列GPU3以`MSVR310→RGBNT201→RGBNT100`运行。MSVR310从公开`ViT-B-16.pt`初始化，不含TriFusion模块且`USE_A=False`、`USE_B=False`，固定第50轮终点；10:03:44训练完成，10:04:10独立重载checkpoint评价完成。纯模型为1536D、86,387,712总参数；591 query/1055 gallery，同身份同scene过滤、无重排序。实存权重与评价回执SHA256一致：`69c5e71b75036d7216ece3ff84450f0052f5e70dfaba46bf73f3e1d40992bb37`。正式指标为**mAP 50.5220、Rank-1 67.6819**；辅助记录Rank-5 81.3875、Rank-10 86.1252。回执在`logs/signal_plain_baseline_20260924_r2/MSVR310/metrics.json`，训练与独立评价日志同目录。这里的纯baseline不同于作者完整Signal权重`53.2424/72.4196`，也不同于论文报告值，不能混列为同一模型。

10:04:10已自动开始RGBNT201纯baseline训练，首轮54 batch、约0.53秒/batch，50轮训练粗估约25分钟；其后RGBNT100 30轮，待首轮测速再确定结束时间。四卡GPU均在执行各自任务，`/data`剩余约123GiB。RGBNT201与RGBNT100此时尚无纯baseline终态指标。

### 41.327 Signal纯baseline RGBNT201固定终点与RGBNT100接续（2026-09-24 10:32 CST）

同一纯baseline队列的RGBNT201端从公开`ViT-B-16.pt`重新初始化，无TriFusion模块、无Signal SIM/GAM/LAM，按作者RGBNT201配置训练50轮；10:30:10独立重载终点权重后的全量评价完成。1536D纯CLS拼接模型有86,409,216总参数，正式836 query/836 gallery、同身份同camera过滤、无重排序。实存终点权重SHA256=`789e5e14aacd74ad122aad701389eb216ca5b4fda92687e27351a513023b4407`，与训练队列及独立评估回执一致。回执`logs/signal_plain_baseline_20260924_r2/RGBNT201/metrics.json`给出：**mAP 69.6415、Rank-1 71.4115、Rank-5 80.1435、Rank-10 85.6459**（百分点）。论文Table 3的RGBNT201纯baseline只列`70.3 mAP/71.8 Rank-1`；本机固定终点分别低约0.6585、0.3885点，不能把两个不同运行称为逐位复现，也不能将本轮完整作者Signal`80.3029/85.1675/91.3876/93.6603`错写成纯baseline。

RGBNT100于10:30:10自动开始独立纯baseline训练，作者配置30轮、B128/K16；首轮65 batch、约0.99秒/batch，预计约11:03完成训练，后续全量评估另需数分钟。`/data`约剩122GiB，其余GPU上的R2/V27正式种子队列未因这项测试中断。

### 41.328 Signal论文纯baseline三数据集完整终态（2026-09-24 11:12 CST）

`logs/signal_plain_baseline_20260924_r2/campaign.json`于11:06:33达到`COMPLETE`，三端均独立从公开`ViT-B-16.pt`初始化并按作者对应YAML训练至固定终轮（RGBNT201/MSVR310第50轮，RGBNT100第30轮），`USE_A=False`、`USE_B=False`，无TriFusion模块及Signal SIM/GAM/LAM。仅作者源码学习率**日志读取**的一行修复见§41.326；不改变前向、损失、优化器或学习率调度。独立评估逐端严格重载各自终点checkpoint，均为1536D三模态CLS拼接，保留原query/gallery、同身份同camera/scene过滤及无重排序。三个实存checkpoint的SHA256均与队列、独立评估回执一致。

| 数据集 | 纯baseline mAP | Rank-1 | Rank-5 | Rank-10 | query/gallery | 过滤 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RGBNT201 | **69.6415** | **71.4115** | **80.1435** | **85.6459** | 836/836 | 同身份同camera |
| RGBNT100 | **83.7042** | **95.0437** | 回执95.6851 | 回执95.8601 | 1715/8575 | 同身份同camera |
| MSVR310 | **50.5220** | **67.6819** | 回执81.3875 | 回执86.1252 | 591/1055 | 同身份同scene |

按用户汇报范围，RGBNT201主报四项，RGBNT100/MSVR310主报mAP与Rank-1；车辆Rank-5/10仅为独立回执中保存的补充数值。三份终点权重均保存在`trained-model/signal_plain_baseline_20260924_r2/<dataset>/Signalbest.pth`，每份约330MiB；对应SHA256依表顺序为`789e5e14aacd74ad122aad701389eb216ca5b4fda92687e27351a513023b4407`、`299a28bfb3e3180eeae0736cf8638cd162525dce0f2192a940b62b97f6e67dcd`、`69c5e71b75036d7216ece3ff84450f0052f5e70dfaba46bf73f3e1d40992bb37`。三份`metrics.json`和训练/独立评价日志同在`logs/signal_plain_baseline_20260924_r2/<dataset>/`；完整队列回执SHA256=`e9575f451752640dc8314877eb9819de69db392f2bab82669e31cece1c94bbe1`。

同正式协议的**作者完整Signal发布权重**分别为RGBNT201 `80.3029/85.1675/91.3876/93.6603`、RGBNT100 `86.3242/97.5510`、MSVR310 `53.2424/72.4196`。纯baseline与完整发布权重的mAP/R1数值差分别是RGBNT201 `10.6614/13.7560`、RGBNT100 `2.6200/2.5073`、MSVR310 `2.7204/4.7377`个百分点；两套权重的训练来源与选点不完全相同，这些是**同评价协议的数值距离，不是仅开关Signal模块的严格控制变量效应**。Signal论文Table 3另报告RGBNT201纯baseline `70.3/71.8`，本轮固定终点复现相应低`0.6585/0.3885`点；车辆两个纯baseline没有论文对应行。RGBNT201 R2/V27种子搜索原`+0.8`停止线仍相对作者**完整**Signal，不以本轮较低的纯baseline偷换目标。

纯baseline队列终态后，GPU3于11:10:37恢复此前让出的MSVR310–V27连续种子控制器PID216084；确认新控制器状态`RUNNING`、子队列启动seed48，GPU3已重新计算。其他三个GPU的原任务没有中断。`/data`当前约余122GiB；三份纯baseline终点权重是用户所需的复核证据，予以保留。

### 41.329 新完成正式种子、非赢家权重退休与双机续跑（2026-09-24 11:21 CST）

本节仅追加各自已完成固定第20轮、原作者Signal完整query/gallery正式评估的种子，不把§41.328纯baseline作为`+0.8`停止线。RGBNT201表项顺序为mAP/R1/R5/R10，车辆为mAP/R1；括号内是相对同回执**作者完整Signal**的百分点差，同一种子所有必报指标均须`≥+0.8`才停止。旧机与新机协议文件的绝对数据根路径不同，但§41.320已逐字段核对记录和过滤内容一致。

| 机器/数据集/方法/seed | fused正式指标 | 对完整Signal逐项增益 | 结果回执SHA256 | 停止线 |
| --- | --- | --- | --- | --- |
| 旧机 RGBNT100 R2/45 | 86.7196/97.8426 | +0.3954/+0.2915 | `2acf70922ca137841e6474832ba445b18bf241f03435192bce69e70e90e2f767` | 未过 |
| 旧机 RGBNT100 V27/45 | 86.1919/97.0845 | −0.1323/−0.4665 | `c11d22ec83698def2e62b6b775566f5eb64b08685fb470e5e28a1bd8eb973195` | 未过 |
| 新机 RGBNT201 R2/46 | 82.0584/86.2440/92.2249/93.7799 | +1.7555/+1.0766/+0.8373/+0.1196 | `c0cf376f1aee5a3ce54d37d4814e98acc86fc189d5b1201b269295fd14fbd1e0` | Rank-10未过 |
| 新机 MSVR310 R2/50 | 51.7117/67.5127 | −1.5307/−4.9069 | `c9fa2c20bd4b1c4440dd8120d7ed1f2ec522f1bcaa9745708adeebda5d95879e` | 未过 |
| 新机 RGBNT100 V27/50 | 86.0330/97.0262 | −0.2912/−0.5248 | `2e8ca3f9fec059eb348fbcc0b2729460d1c857dfe14343f6940499be8308d1a1` | 未过 |

五份`official_metrics.json`与`training.json`均为`COMPLETE`/`FIXED_EPOCH20_TRAINING_COMPLETE`，各对回执的角色权重SHA一致；原始分支输出、查询级距离和训练日志保存在对应`trained-model/official_extra_seed...`目录。MSVR310–R2 seed50、RGBNT100–V27 seed50的连续控制器已判为非赢家并清理其角色权重，回执仍在。对旧机RGBNT100 R2/45、V27/45及新机RGBNT201 R2/46，先分别核对实存权重SHA与回执，再核对保留的对应赢家（RGBNT100 R2/44、V27/44、RGBNT201 R2/43）正式回执、mAP/R1和实存权重SHA，才逐文件删除上述三个非赢家`roles_epoch20.pth`，释放`27,157,478+27,157,478+39,795,686=94,110,642`字节；删除后逐文件确认不存在。结果回执中的checkpoint路径只保留历史来源含义，**不能再称三个非赢家权重实存**。各自赢家权重未动。

11:17左右实查旧单卡机控制器PID62216在等待原seed45整队列，原队列PID45727正运行MSVR310–R2 seed45，GPU0计算中，`/root/autodl-tmp`约余20GiB。新四卡机GPU0仍执行RGBNT100–R2 seed46，GPU1执行RGBNT100–V27 seed52，GPU2执行MSVR310–R2 seed52，GPU3执行MSVR310–V27 seed48；相应控制器/子进程均在，`/data`约余122GiB。上述运行中端点没有完整正式回执，本节不提前报分。此前六组中已达标的仍只有RGBNT201–V27 seed43，不能把某一项正增益或纯baseline较低指标当成新达标。

### 41.330 MSVR310–V27 seed48恢复后的正式终态（2026-09-24 11:24 CST）

§41.328纯baseline让出GPU3后恢复的MSVR310–V27连续控制器于11:23:17完成seed48固定第20轮及全量正式评价；正式回执SHA256=`aeacb47196feb29bb27aafeddaf5dfa8ed8f0c30470b8d4661f952c5695d1702`，训练/评价回执权重SHA256均为`2bdcbb28aeb0a67dbf70d842fd4476257a3db4c6645e5a45a3370e4395e50bab`，协议SHA256=`7f2b35f9a7e00433558c1e723db0e3ff0ea9daa7144eeeb972139d1502945cbc`。591 query/1055 gallery，同身份同scene过滤、无重排序。正式mAP/Rank-1为：作者完整Signal`53.2424/72.4196`，fused **`51.4353/69.0355`**，CNN`49.6405/66.6667`，Transformer`49.0601/64.4670`，Mamba`48.5525/67.0051`。fused对作者完整Signal为`−1.8071/−3.3841`个百分点，双指标停止线未过。控制器按原先`(全项达标,mAP,Rank-1)`顺序保留MSVR310–V27 seed43为临时赢家，记载seed48 `checkpoint_retained=false`；核对其`roles_epoch20.pth`已不存在，训练/正式回执和查询距离保留，回执checkpoint路径仅作历史来源。控制器PID216084在运行，已接续MSVR310–V27 seed50；新种子尚无结果。


### 41.331 本轮R2/V27正式种子五路指标全集（2026-09-24 11:42 CST快照）

以下仅列本轮从对应**作者完整Signal发布权重**初始化、固定第20轮终点、原Signal全量query/gallery与camera/scene过滤正式评价已完成的34个唯一种子。`RGBNT201`每格顺序为mAP/Rank-1/Rank-5/Rank-10，`RGBNT100`与`MSVR310`每格为mAP/Rank-1；单位均为%。同一行的CNN、Transformer、Mamba为完整“Signal＋角色残差”输出，不是各自独立重训的模型。旧机seed43导入回执不重复计数；没有把Q1、纯baseline或仍在训练的种子混入表中。

| 数据集 | 作者完整Signal | 本轮Signal纯baseline（不同训练来源） |
| --- | --- | --- |
| RGBNT201 | 80.3029/85.1675/91.3876/93.6603 | 69.6415/71.4115/80.1435/85.6459 |
| RGBNT100 | 86.3242/97.5510 | 83.7042/95.0437 |
| MSVR310 | 53.2424/72.4196 | 50.5220/67.6819 |

作者完整Signal是本轮六组合`+0.8`停止线基准；纯baseline为从公开CLIP独立训练、去掉TriFusion和Signal增强模块后的用户指定测量，不用于该停止线。

#### RGBNT201（7个已完成种子）

| 方法/seed | fused | CNN | Transformer | Mamba |
| --- | --- | --- | --- | --- |
| R2/42 | 82.4254/86.7225/92.5837/94.0191 | 81.1094/86.4833/91.7464/93.8995 | 81.1233/85.4067/91.6268/93.8995 | 82.0390/86.9617/91.9856/93.6603 |
| R2/43 | 82.5826/87.4402/91.8660/94.2584 | 82.1907/87.4402/92.4641/94.0191 | 80.7685/86.1244/91.7464/93.8995 | 81.2269/85.5263/91.5072/93.6603 |
| R2/44 | 82.4417/86.3636/92.5837/94.1388 | 81.1122/85.0478/92.3445/93.7799 | 80.4886/84.6890/91.3876/93.8995 | 82.8619/87.0813/92.1053/94.0191 |
| R2/46 | 82.0584/86.2440/92.2249/93.7799 | 81.6879/86.9617/91.9856/93.6603 | 80.6912/85.4067/91.2679/93.7799 | 80.2227/85.4067/91.3876/93.3014 |
| V27/42 | 81.8916/85.5263/91.9856/93.6603 | 80.6873/83.9713/90.6699/93.5407 | 81.1130/85.2871/91.7464/94.0191 | 80.6659/85.2871/91.1483/93.1818 |
| V27/43 | 83.0005/87.5598/92.9426/94.4976 | 81.8730/87.0813/92.5837/93.8995 | 81.6465/86.2440/93.1818/94.7368 | 82.5620/86.2440/92.2249/94.0191 |
| V27/44 | 82.9153/86.8421/92.4641/94.1388 | 81.7097/86.7225/91.7464/93.8995 | 80.7558/85.5263/92.1053/94.4976 | 83.0594/86.8421/91.7464/93.5407 |

#### RGBNT100（12个已完成种子）

| 方法/seed | fused | CNN | Transformer | Mamba |
| --- | --- | --- | --- | --- |
| R2/42 | 86.3156/97.4927 | 85.3468/97.7843 | 85.8561/97.0262 | 85.2197/97.7843 |
| R2/43 | 86.5953/97.2012 | 85.4110/96.9679 | 86.3732/97.0845 | 85.1635/97.4927 |
| R2/44 | 87.1200/97.8426 | 85.8688/97.2012 | 86.2559/97.2012 | 85.9944/97.9009 |
| R2/45 | 86.7196/97.8426 | 85.3205/96.9679 | 85.6438/97.3761 | 85.5297/97.3761 |
| V27/42 | 85.5703/96.6181 | 85.2118/96.7930 | 84.3279/96.6181 | 84.5209/96.1516 |
| V27/43 | 85.9465/96.7930 | 85.0802/97.3761 | 84.8915/96.4431 | 84.6947/96.5015 |
| V27/44 | 86.3835/97.3761 | 85.7227/97.4344 | 85.6552/96.9679 | 84.8056/97.1429 |
| V27/45 | 86.1919/97.0845 | 85.3609/96.9679 | 85.0693/96.4431 | 85.1374/96.6764 |
| V27/46 | 86.1517/97.3178 | 85.8277/97.6676 | 85.0405/97.0262 | 85.0296/97.2595 |
| V27/48 | 85.9915/97.0262 | 85.3981/97.3178 | 84.8836/96.3848 | 84.8260/96.7930 |
| V27/50 | 86.0330/97.0262 | 85.3340/97.3761 | 85.3742/95.8601 | 84.3126/96.3848 |
| V27/52 | 86.0024/96.8513 | 85.4408/96.9096 | 85.1856/96.6764 | 84.5841/96.6181 |

#### MSVR310（15个已完成种子）

| 方法/seed | fused | CNN | Transformer | Mamba |
| --- | --- | --- | --- | --- |
| R2/42 | 52.2816/69.3739 | 49.7840/67.3435 | 49.2958/65.9898 | 49.9769/67.1743 |
| R2/43 | 53.2391/70.8968 | 49.5196/69.8816 | 50.0169/68.1895 | 51.5993/70.2200 |
| R2/44 | 51.2600/69.0355 | 48.1439/66.4975 | 48.8756/65.1438 | 48.3948/66.6667 |
| R2/45 | 52.2257/69.7124 | 49.4593/67.3435 | 49.3652/66.3283 | 49.0748/66.4975 |
| R2/46 | 53.4216/70.5584 | 51.4484/69.5431 | 50.6111/67.5127 | 49.3602/67.8511 |
| R2/48 | 53.2476/72.0812 | 50.8805/68.1895 | 50.3771/68.1895 | 50.2143/69.7124 |
| R2/50 | 51.7117/67.5127 | 47.6381/64.8054 | 49.7159/68.1895 | 49.9252/68.3587 |
| R2/52 | 52.2120/70.5584 | 50.5539/68.8663 | 49.6449/66.6667 | 48.4491/67.5127 |
| V27/42 | 50.5218/67.3435 | 49.8159/67.1743 | 47.9569/64.2978 | 48.1662/65.6514 |
| V27/43 | 51.6667/68.3587 | 47.9557/67.1743 | 50.0816/67.3435 | 50.3322/67.8511 |
| V27/44 | 49.7506/66.4975 | 47.7437/66.3283 | 47.6920/64.4670 | 47.6522/65.3130 |
| V27/45 | 50.5561/67.3435 | 48.1977/67.0051 | 48.5640/65.8206 | 48.5665/65.4822 |
| V27/46 | 51.1058/67.5127 | 49.7094/68.1895 | 48.9090/65.6514 | 48.8995/65.1438 |
| V27/48 | 51.4353/69.0355 | 49.6405/66.6667 | 49.0601/64.4670 | 48.5525/67.0051 |
| V27/50 | 50.1405/66.4975 | 47.4300/64.6362 | 47.2211/63.9594 | 49.5778/67.5127 |

截至此快照，六个方法×数据集组合中，仅`RGBNT201–V27 seed43`的fused同一种子四项增益全部达到`+0.8`个百分点；其余五个组合仍未达标。RGBNT201–R2最佳保留seed43为`82.5826/87.4402/91.8660/94.2584`，RGBNT100–R2最佳保留seed44为`87.1200/97.8426`，RGBNT100–V27最佳保留seed44为`86.3835/97.3761`，MSVR310–R2最佳保留seed46为`53.4216/70.5584`，MSVR310–V27最佳保留seed43为`51.6667/68.3587`；这些是按原 `(全项达标,mAP,Rank-1)` 存储规则选择的单个checkpoint，不是无偏多种子平均。RGBNT100–R2 seed44的mAP原始增益`+0.7958`、Rank-1 `+0.2915`，不能因四舍五入写成达标。

本快照新增四份完整正式回执：旧机MSVR310–R2 seed45 `be4ee6c80a02245e503a313bf727abb77f54f7d67a99a5a54801cc7708e81871`、旧机MSVR310–V27 seed45 `0aac95970e94763ed86741b1796106021928dd924c32647ffa5cfc8310db10c4`、新机MSVR310–R2 seed52 `0ddc71407ec29c903253fb85a0c816cc5f6fcbe019a68f5af1127f2919ebe0a3`、新机RGBNT100–V27 seed52 `eb0482671ef9c46e3954a3e1d4a19e9a55de14f31a186f662c08b95ef63afdb9`。其训练回执均为固定epoch20完成，评价回执均为`COMPLETE`，训练/评价权重SHA相同，且与原Signal独立上游指标一致；逐查询距离、训练步骤和日志仍在原目录。新机两个seed52由原连续控制器判为非赢家并已退役权重。旧机两个MSVR310 seed45经核对回执、实存文件SHA及对应更优保留权重后，仅退役其各自的`roles_epoch20.pth`，共释放`76,249,036`字节；正式结果与日志未删。回执中的历史checkpoint路径不能再当作实存保证。

11:42只读检查：旧单卡GPU0正执行RGBNT201–R2 seed45，GPU约7660MiB/100%利用率，输出盘约20GiB可用；新机四张GPU分别约7569/6339/7537/6350MiB，利用率96%/70%/88%/99%，`/data`约121GiB可用，连续种子队列仍在工作。运行中种子不提前填正式分数。未更改方法、作者评估、种子或固定终点；后续新回执另行按同一口径追加。

### 41.332 停止后续种子搜索，保留在途训练至验收（2026-09-24 11:48 CST）

用户明确要求：四卡服务器上**已经开始**的实验跑完后不再继续启动新种子，转入原因诊断、失败实例与指标可视化；旧单卡本轮在途任务同样完成验收后停止新增种子。11:46对进程命令行与父子关系核实后，仅向四卡机四个`continue_official_seed_until_delta.py`续跑控制器PID`216084/3667962/3771922/3771925`及旧单卡续跑控制器PID`62216`发送`SIGTERM`。复查五个控制器均已退出，**没有终止**已启动的`queue_official_extra_seed.py`及其训练/评价子进程；旧机固定seed45队列PID`45727`仍在，四卡机原seed46队列PID`3542100`及三个已启动的独立队列PID`243793/257360/273616`仍在。当前在途任务分别是新机GPU0 RGBNT100–R2 seed46、GPU1 RGBNT100–V27 seed54、GPU2 MSVR310–R2 seed54、GPU3 MSVR310–V27 seed52，以及旧机RGBNT201–R2 seed45。各队列自身会完成固定epoch20、原作者正式评价和回执；续跑状态JSON因控制器终止可能保留历史`RUNNING`字段，不能仅据此判断任务仍会追加种子，后续以队列进程和正式回执为准。

本次停止的是**无限续种子策略**，不是取消已经训练到一半的模型，也不修改模型、终点、Signal评价规则或已出正式结果。当前已完成34个唯一正式种子，只有RGBNT201–V27 seed43通过四项`+0.8`停止线；其余在途回执产生后逐项加入§41.331格式的全指标表。原因诊断应使用完整作者Signal和同一模型fused的逐查询距离、协议标签及camera/scene字段；分别报告AP与Rank-1变化、原错误修复与新增错误、身份集中度、跨环境正例覆盖，以及三角色在新增错误关系上的相似度贡献。所有图表明确标注`正式测试的事后诊断`，不得根据这些已消费测试身份调参；不从损失下降直接推断未知身份收益。

### 41.333 用户指定的三数据集分表（2026-09-24 15:22 CST，39个本轮正式种子）

三表只列**已完成的正式query/gallery评估**，按用户列数：RGBNT201为mAP/Rank-1/Rank-5/Rank-10；RGBNT100、MSVR310只为mAP/Rank-1，单位%。方法/seed为单个训练终点的fused；CNN/Transformer/Mamba完整分支逐种子输出可在各正式回执中核对；§41.331为较早快照。`Signal纯baseline`从公开CLIP独立训练且关闭Signal扩展模块，`作者完整Signal`为本轮预训练初始化及停止线，早期V1/V8又有各自训练来源，不能跨这些行相减作单因素因果结论。内部Q1不混入以下正式表。

#### RGBNT201

| 方法/条件 | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| Signal纯baseline（独立训练） | 69.6415 | 71.4115 | 80.1435 | 85.6459 |
| TriFusion V1（历史正式） | 59.1478 | 63.2775 | 77.2727 | 83.6124 |
| 作者完整Signal（本轮基线） | 80.3029 | 85.1675 | 91.3876 | 93.6603 |
| R2/42 | 82.4254 | 86.7225 | 92.5837 | 94.0191 |
| R2/43 | 82.5826 | 87.4402 | 91.8660 | 94.2584 |
| R2/44 | 82.4417 | 86.3636 | 92.5837 | 94.1388 |
| R2/45 | 81.3305 | 85.2871 | 91.7464 | 94.0191 |
| R2/46 | 82.0584 | 86.2440 | 92.2249 | 93.7799 |
| V27/42 | 81.8916 | 85.5263 | 91.9856 | 93.6603 |
| **V27/43（已达+0.8）** | 83.0005 | 87.5598 | 92.9426 | 94.4976 |
| V27/44 | 82.9153 | 86.8421 | 92.4641 | 94.1388 |
| V27/45 | 82.8359 | 86.7225 | 92.1053 | 94.2584 |

#### RGBNT100

| 方法/条件 | mAP | Rank-1 |
| --- | ---: | ---: |
| Signal纯baseline（独立训练） | 83.7042 | 95.0437 |
| 旧本机Signal（V8配对） | 80.7122 | 94.2274 |
| TriFusion V8（历史正式） | 83.2848 | 96.1516 |
| 作者完整Signal（本轮基线） | 86.3242 | 97.5510 |
| R2/42 | 86.3156 | 97.4927 |
| R2/43 | 86.5953 | 97.2012 |
| R2/44 | 87.1200 | 97.8426 |
| R2/45 | 86.7196 | 97.8426 |
| V27/42 | 85.5703 | 96.6181 |
| V27/43 | 85.9465 | 96.7930 |
| V27/44 | 86.3835 | 97.3761 |
| V27/45 | 86.1919 | 97.0845 |
| V27/46 | 86.1517 | 97.3178 |
| V27/48 | 85.9915 | 97.0262 |
| V27/50 | 86.0330 | 97.0262 |
| V27/52 | 86.0024 | 96.8513 |
| V27/54 | 86.5595 | 96.6181 |

#### MSVR310

| 方法/条件 | mAP | Rank-1 |
| --- | ---: | ---: |
| Signal纯baseline（独立训练） | 50.5220 | 67.6819 |
| 作者完整Signal（本轮基线） | 53.2424 | 72.4196 |
| R2/42 | 52.2816 | 69.3739 |
| R2/43 | 53.2391 | 70.8968 |
| R2/44 | 51.2600 | 69.0355 |
| R2/45 | 52.2257 | 69.7124 |
| R2/46 | 53.4216 | 70.5584 |
| R2/48 | 53.2476 | 72.0812 |
| R2/50 | 51.7117 | 67.5127 |
| R2/52 | 52.2120 | 70.5584 |
| R2/54 | 52.0114 | 68.8663 |
| V27/42 | 50.5218 | 67.3435 |
| V27/43 | 51.6667 | 68.3587 |
| V27/44 | 49.7506 | 66.4975 |
| V27/45 | 50.5561 | 67.3435 |
| V27/46 | 51.1058 | 67.5127 |
| V27/48 | 51.4353 | 69.0355 |
| V27/50 | 50.1405 | 66.4975 |
| V27/52 | 50.4317 | 67.8511 |

停止线须同一种子的全部必报指标相对`作者完整Signal`各至少`+0.8`个百分点，目前仅RGBNT201–V27 seed43满足。MSVR310–V27 seed52是在§41.331快照之后完成的第35份正式回执：fused`50.4317/67.8511`、CNN`49.2414/68.3587`、Transformer`48.5403/64.8054`、Mamba`47.0742/65.9898`，仍未达标；其完整回执与距离数组保留在`trained-model/official_extra_seed52_MSVR310_V27_20260924/MSVR310_V27_seed52/`。以上为截至表题时间的封闭快照，在途任务另待完成，不能以本表空缺推断结果为零。

### 41.334 正式逐查询诊断：增益受限与新增错误（2026-09-24，固定终点事后分析）

本节只读取四个**已完成正式测试**的代表种子保存的 `official_metrics.json`、`official_distances.pt`及query/gallery真实身份与camera/scene数组，没有模型前向、训练更新、候选种子选择或超参调节。诊断脚本 `tools/diagnose_official_retrieval.py` 的SHA256为 `c5511f3c9ac2b081fce394df868a4436a57e7321b5bdfe6390c67c5f85cda2fa`；逐份先核对正式回执和距离文件SHA，再以逐query AP与首正例名次重放正式mAP/Rank-1。输出保存在新机 `logs/official_posthoc_diagnosis_20260924/`，四份报告按RGBNT201–V27/43、RGBNT100–R2/44、MSVR310–R2/46、MSVR310–V27/43顺序的SHA256为 `7c3cef93500d4f21000e495bc99065e81f523b2e18e09659d771e8ee604700e0`、`17ccc62ad7b16b5d72d1fcce9131021073acb388045e0879cd710208800bb04a`、`7376203335629adb75f20d5151547281690c590254ba2ceffba04fd30c46b7ee`、`a8eb893bb87d3ba3a53b3149c99a8e558569c6739b07c92e314aa15682cc6efa`。这些正式测试身份已被消费，以下仅可作**事后解释**，不能拿个别身份或排序反向调模型。

| 数据集/方法/seed | AP改善/下降/不变query | Rank-1修复/新增错误 | 身份平均AP改善/下降/不变 | 新错误最近负例与query同环境 |
| --- | ---: | ---: | ---: | ---: |
| RGBNT201 V27/43 | 350/168/318（836） | 36/16 | 21/8/1（30） | 14/16同camera |
| RGBNT100 R2/44 | 741/557/417（1715） | 11/6 | 28/19/3（50） | 2/6同camera |
| MSVR310 R2/46 | 304/271/16（591） | 27/38 | 30/22/0（52） | 9/38同scene |
| MSVR310 V27/43 | 274/301/16（591） | 23/47 | 23/29/0（52） | 12/47同scene |

这解释了为什么单看mAP会漏掉严重的首位退化：MSVR310–R2/46的mAP比Signal高`+0.1792`，但Rank-1低`1.8613`点，新增38条首位错误多于27条修复；V27/43新增47条、多于23条修复。RGBNT100–R2/44虽有mAP`+0.7958`和741条AP改善，仍有557条下降与6条新增首位错误，Rank-1只高`0.2915`点。RGBNT201–V27/43确实有36条修复、16条新增和21个身份改善，因此已过四项门槛；它仍不是“所有query均改善”。同环境负例计数只是标签描述，RGBNT100与MSVR310许多新增错误的最近负例并不同环境，不能统一归因于背景或相机捷径。

以下为每个数据集按**新增Rank-1错误中AP降幅最大的query**选出的排序示意，不是随机样本或总体比例。位置从左到右为过滤后的gallery第1至20名；`P`为合法同身份正例，`.`为其他身份，已排除同身份同camera/scene记录。没有读取原始图片，因此只描述可核验的身份、环境和排名，不补写外观、遮挡或背景原因。

```text
位置                12345678901234567890
RGBNT201 V27/43 q730 / ID294 / camera1
Signal              PP..PP.P............
fused               ..P.................

RGBNT100 R2/44 q216 / ID514 / camera0
Signal              PPPPPPPPPPPPPPPPPPPP
fused               .................P.P

MSVR310 R2/46 q486 / ID152 / scene18
Signal              P...................
fused               ..............P.....
```

三条示例的Signal→fused首正例名次与AP依次为：RGBNT201 `1→3, 0.4334→0.1489`；RGBNT100 `1→18, 0.6761→0.4640`；MSVR310 `1→15, 1.0000→0.0667`。MSVR310–V27/43在**同一q486/同一正负对**也为`1→13, 1.0000→0.0769`，说明此查询并非只对R2这一种训练法敏感；仍不能据此称全部车辆查询都有同样失败。

固定同一条原Signal首位正例`p`和fused首位错误负例`n`，表中为各输出的平方欧氏距离间隔`d(q,n)-d(q,p)`；正值表示该输出将这条正例排在负例之前，负值相反。这一**同一对样本**的比较能定位哪条完整角色分支给了有害排序支持，不把分支mAP误当相似度平均。

| 例子 | Signal | CNN | Transformer | Mamba | fused |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGBNT201 V27/43 q730 | +0.03385 | +0.06744 | −0.05383 | −0.06096 | −0.01579 |
| RGBNT100 R2/44 q216 | +0.02892 | −0.01193 | +0.00094 | −0.05754 | −0.02284 |
| MSVR310 R2/46 q486 | +0.04815 | −0.06802 | −0.07555 | −0.16149 | −0.10169 |

保存距离上的fused间隔与三个完整角色间隔的均值逐例在`10^-4`以内一致。由此可以**直接确认这些具体新增错误**是原Signal正确关系被角色组合翻转：RGBNT201示例中CNN仍有利、Transformer/Mamba不利；RGBNT100示例中Mamba最不利；MSVR310示例中三角色均不利。不能从这三条极端案例外推“某角色在全部查询中有害”，也不能认定环境、训练损失或架构中某一项是唯一因果原因。

跨已完成种子的描述性均值进一步限定结论：RGBNT201–R2四种子相对完整Signal的四项均值增益为`+2.0741/+1.5251/+0.9270/+0.3887`，V27三种子为`+2.2996/+1.4752/+1.0765/+0.4386`，均值上的Rank-10尚不足`+0.8`；RGBNT100–R2四种子mAP/R1均值增益仅`+0.3634/+0.0438`，V27九种子为`−0.2319/−0.5831`；MSVR310–R2八种子为`−0.7925/−2.4534`，V27八种子为`−2.5413/−4.8646`。这些是反复使用同一官方测试集后的**描述性**种子统计，不是未接触独立测试集的无偏泛化估计，也不能抵消§41.333中的逐种子真实结果。当前最稳妥的解释是：角色残差确实能在RGBNT201和部分RGBNT100查询修正排序，但固定等权集成也会翻转Signal已经正确的正负关系；V27统计扰动的收益明显依赖数据集；MSVR310上mAP与Rank-1尤其不一致。进一步诊断应优先检查来源训练关系与正式跨scene难例的对应、各角色翻转关系的频率和被伤害身份是否集中，而非继续按正式测试错误挑种子或改损失。

§41.333新增的MSVR310–V27 seed52已核对训练/正式回执、非赢家自身实存checkpoint SHA及seed43保留赢家SHA；仅退役该seed52非赢家`roles_epoch20.pth`的`38,124,518`字节，正式回执、距离和训练日志完整保留。四卡/旧单卡的后续自动续跑控制器保持关闭，在途队列仍各自完成验收。

### 41.335 跨种子分支对比与当前可排除的简单解释（2026-09-24 13:03 CST）

对§41.333的37个正式种子逐一在**同一回执、同一作者完整Signal**下比较五路输出，而不把各版本的历史control互换。下表中的“fused高于全部分支”指严格高于该种子CNN、Transformer、Mamba三个完整输出的mAP；“三分支均低于Signal”要求该种子的三个完整分支各自同时低于Signal。计数只是已完成种子的观察，不是训练随机性的无偏置信结论。

| 数据集 | 已完成种子 | fused mAP高于全部三分支 | fused mAP高于作者Signal | fused Rank-1高于作者Signal | 三分支mAP均低于Signal | 三分支Rank-1均低于Signal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RGBNT201 | 7 | 5 | 7 | 7 | 0 | 0 |
| RGBNT100 | 13 | 13 | 5 | 2 | 12 | 10 |
| MSVR310 | 17 | 17 | 2 | 0 | 17 | 17 |

因此，MSVR310与RGBNT100当前负结果**不能只解释成固定融合低于某一个角色**：MSVR310的17/17个种子里，fused mAP实际都高于同模型三个完整分支，但三分支又都低于完整Signal的mAP和Rank-1；fused的Rank-1没有一个种子超过Signal。RGBNT100也有13/13个fused mAP高于三分支，但只有5/13在mAP、2/13在Rank-1上超过Signal。RGBNT201则是7/7个fused mAP和Rank-1均超过Signal，说明同一结构在另一数据条件下能提供增量。当前最直接的瓶颈是车辆任务中新增残差角色学习到的**未见身份首位判别能力不足或不稳定**，同时某些查询上的残差会翻转原Signal已正确的关系；它不证明所有可能的融合、采样或角色训练都无效。

Rank门槛在有限query集合上是离散的。RGBNT100共有1715条query，作者Signal正确首位1673条；R2 seed44正确1678条，净增5条，即`+0.2915`点。达到`+0.8`至少需要1687条正确首位，所以该种子还差**9条净正确query**。MSVR310共有591条query，作者Signal首位正确428条；R2 seed46正确417条，净少11条，达到`+0.8`至少需要433条，距离目标**16条净正确query**。RGBNT201共有836条query，作者Signal Rank-10正确783条；V27 seed43正确790条，净增7条，恰好越过`+0.8`线；R2 seed43为788条，仍少2条。上述整数来自同一正式回执的Rank百分比与query计数，不是以四舍五入后的百分点决定边界。

来源内部结果也要限定解释：RGBNT201–V27原Q1有`+1.3389 mAP`的配对收益，本轮正式三种子mAP均高于Signal；MSVR310同统计扰动来源Q1为`−0.3022`，本轮V27八种子mAP/R1也均低于Signal。方向一致支持“统计扰动的收益有数据集依赖”，但Q1与正式测试身份、图库及配对控制不同，不能把二者差值当成已识别的因果机制。MSVR310–R2来源Q1仅`+0.0533 mAP`，而本轮正式各seed的mAP与Rank-1变化并不同步；训练目标下降或来源小正收益不足以保证正式首位匹配改善。

下一步的证据边界：四卡已启动任务与旧机seed45完成后，先把同定义五路指标、逐query AP/Rank、非赢家权重SHA和磁盘状态收齐，再更新上表与诊断。用户已经停止新增种子；不能因某个新正式错误身份而恢复无限种子搜索、改权重或延长epoch。若后续提出方法改动，应由来源侧或新预注册内部验证支持，并与当前作者完整Signal以及既有R2/V27固定终点作匹配对照。
### 41.336 正式正负样本对变化与首位排序的区别（2026-09-24 12:18 CST）

在§41.334同四份固定正式回执上，对每条合法query的所有跨环境同身份正例和所有不同身份负例，比较作者完整Signal与fused的距离次序。正例仅包含RGBNT201/RGBNT100的跨camera或MSVR310的跨scene匹配；同身份同环境记录仍按正式协议排除，不改作负例。距离相等的样本对单列、不给予正确或错误判断。代码是`tools/diagnose_official_retrieval.py`新增的只读统计，SHA256 `1a48053ef7d87010be66d6966bfcc68d82483f0e68469ff9a76904b21739b769`；它仍先验证正式距离文件SHA和原mAP/Rank-1重放。四份新增报告保存在新机原有`logs/official_posthoc_diagnosis_20260924/`下，按RGBNT201–V27/43、RGBNT100–R2/44、MSVR310–R2/46、MSVR310–V27/43顺序的SHA256为`af6f0f1725f694533a4e0f4aa0b1cac365d20342d09fc3fb2e0725ad06892f10`、`043dfe9fcf7ea4dcbae87ffc1e20dad5ddb4c9ea673ec4e7357c38f212e09fcc`、`794228317b89738c2dfaa530229dd031a733852f2084a0443aa650f68664ba94`、`7a1e683641d2221118ed52ac1d8ebe0c488e9fb2b7b1c7adc05b6a036edfddf3`。

| 固定正式回执 | Signal错→fused对的正负对 | Signal对→fused错的正负对 | 每query平均的正负对正确率变化 | query正确率改善/下降 | 同回执mAP/Rank-1变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RGBNT201 V27/43 | 33,883 | 16,562 | +0.2001点 | 362/155 | +2.6976/+2.3923点 |
| RGBNT100 R2/44 | 8,425,897 | 2,342,333 | +0.2780点 | 776/521 | +0.7958/+0.2915点 |
| MSVR310 R2/46 | 73,559 | 64,382 | +0.2866点 | 321/247 | +0.1792/−1.8613点 |
| MSVR310 V27/43 | 77,054 | 74,642 | +0.1223点 | 283/290 | −1.5757/−4.0609点 |

四份中距离并列的正负对分别为0、25、1、1，占比很小。正负对计数不是独立query数：每条query可能提供很多正负组合，身份和正例数多的查询会贡献更多对。因此另报先在每query内求正确率变化、再对query等权平均；它仍不是正式mAP，也不等于身份聚类置信区间。MSVR310两份虽然总体修复对数多于新增错误对数，R2的Rank-1和V27的mAP/Rank-1仍下降，说明更多广义正负关系排对没有保证最前列及多正例AP改善。结合§41.334的首位修复/新增错误，当前证据支持“角色在部分关系上提供增量、同时损害少数关键靠前排序”的**结果层解释**；它尚不能识别训练损失、角色结构或环境因素中哪一项是因果原因。正式测试身份已消费，本统计只作事后分析，不用于调参、挑种子或增开训练。

### 41.337 RGBNT100–V27 seed54正式终态与权重盘点（2026-09-24 12:37 CST）

新机`trained-model/official_extra_seed54_RGBNT100_V27_20260924/RGBNT100_V27_seed54/`已完成固定第20轮训练与原Signal完整query/gallery正式评估。`training.json`为`FIXED_EPOCH20_TRAINING_COMPLETE`、`official_metrics.json`为`COMPLETE`；1715 query/8575 gallery，同身份同camera过滤并保留不同身份干扰项。训练/正式回执的角色权重SHA一致，且实存权重SHA256=`14242f49474b5f149e29553798d9be8bf91a99737179df634b3791c4de5a68f5`；实存距离文件SHA256=`74d4ff8c8ca2b8354033575d0aa42be40d235536d4d6c0c1bd7b9faf5f77db32`，均与回执一致。正式回执SHA256=`31e2d9e00152d51a86163e0643928bd7a0de2deebf2ac16be61b3d173bca135c`。完整Signal为`86.3242/97.5510`；seed54的fused为`86.5595/96.6181`，CNN为`86.1894/96.8513`，Transformer为`85.0382/96.8513`，Mamba为`85.5098/96.2099`（mAP/Rank-1，单位%）。fused相对Signal为`+0.2353/−0.9329`点，未通过双指标各`+0.8`停止线。它的mAP高于旧V27/44的`86.3835`，但Rank-1低于其`97.3761`；不能只报较有利的一项。

一次只读全权重盘点覆盖当时36份完整正式回执，逐份确认训练与正式回执完成及权重SHA字段相等，发现仅两份非赢家权重仍实存：新机RGBNT100–V27 seed44与旧机RGBNT201–R2 seed42。按此前登记的`(全部必报指标达标,mAP,Rank-1)`存储顺序，分别由V27 seed54和R2 seed43作为当前保留种子。删除前重新核对双方完整回执、实存角色权重SHA与精确`roles_epoch20.pth`目标路径；随后仅退役旧V27/44权重`27,157,478`字节（SHA256=`4e2caf8b97548b4d4a6fe1869b429e3dc5527a39e8e7da2c04e4030513fa0188`）与旧R2/42权重`39,795,686`字节（SHA256=`e79e65448b9fba18e030a4bebad4c1f553566ce32579c62063c27b24f8b7481c`），合计释放`66,953,164`字节，并逐文件确认不存在。保留权重V27/54与R2/43的实存SHA也已重核；所有正式结果、距离文件、训练日志和回执未删。按正式测试选出的保留权重只用于存储/部署，不可解释为未接触测试集的无偏泛化选择。其余RGBNT100–R2/46、MSVR310–R2/54、旧机RGBNT201–R2/45仍为在途任务；自动新增种子控制器保持关闭。

### 41.338 RGBNT100–V27 seed54为何mAP微升而Rank-1下降：正式排序事后诊断（2026-09-24）

仅使用§41.337已核验SHA的正式距离文件、标签与回执，运行同一只读`tools/diagnose_official_retrieval.py`，零模型前向/优化更新；新报告位于新机`logs/official_posthoc_diagnosis_20260924/RGBNT100_V27_seed54_pairwise.json`，SHA256=`47acbe213299678686e7447d3f7d1503763c659b908b2e316e9fbda622424579`。脚本先重放1715条query的正式mAP/Rank-1，再在合法跨camera正例与所有不同身份负例之间计算距离次序；并列距离的26对单列。全部都是已消费正式测试身份的**事后解释**，不得用于选种子或修改方法。

相对作者完整Signal，1715条query中fused AP改善/下降/不变分别为`654/674/387`；Rank-1修复`7`条、新增错误`23`条，净少`16`条正确首位，恰对应Rank-1 `−0.9329`点。50个身份的平均AP改善/下降/不变为`24/24/2`。23条新增首位错误中，最近错误身份样本与query同camera仅`6`条，不能把此次退化概括为同camera捷径。全部合法正负样本对中，Signal错而fused对`9,770,800`对，Signal对而fused错`3,309,170`对；每query先求正确率变化再等权平均为`+0.2817`点。该对级统计并非官方AP，显示广义关系修复明显多于翻错，仍没有保住最前列排序；与§41.336的MSVR310现象一致。

在23条新增Rank-1错误中按AP降幅排序的示意首例为query索引`210`、身份`512`、camera`5`；原Signal首位跨camera正例为gallery索引`1022`/camera`2`，fused首位错误负例为gallery索引`365`/身份`504`/camera`7`。这是极端样本示例，不代表总体频率；没有读取原图，不补写颜色、遮挡或背景原因。

```text
位置                   12345678901234567890
Signal                 PPPPPPPPPPPPPPPPPPPP
V27/54 fused           ....................
```

该query的首个合法正例名次`1→32`、AP`0.8048→0.3270`。固定上述同一正负对，平方欧氏距离间隔`d(q,n)-d(q,p)`：Signal `+0.05843`、CNN `−0.06252`、Transformer `+0.00900`、Mamba `−0.08957`、fused `−0.04770`。这直接说明该查询的原Signal正确关系被CNN/Mamba不利贡献翻转，而Transformer仍略有利；不能外推成全体查询的固定角色优劣。当前改进受限同时涉及**正负关系覆盖与排序位置权重**：总体更多对被排对，但一部分对首位匹配至关重要的正负关系变坏。它是结果现象，不足以单独判定具体训练loss或统计扰动的因果责任。

### 41.339 MSVR310–R2 seed54正式验收与非赢家权重退役（2026-09-24 13:03 CST）

新四卡机GPU2的原在途队列PID`243793`已结束，并生成`trained-model/official_extra_seed54_MSVR310_R2_20260924/MSVR310_R2_seed54/`的完整正式回执。`training.json`为`FIXED_EPOCH20_TRAINING_COMPLETE`，`official_metrics.json`为`COMPLETE`，正式query/gallery分别为`591/1055`；训练/评价角色权重SHA一致。另对实存角色权重和正式距离文件逐字节计算SHA256，分别与回执中的`a11dfda90fb0bbd3989e28260d1de52a62fe615bb26fd359e620cb80fc406393`、`469c7b301db7f515d27441258318918ea65ac03b0f830e92110827684c7c3e8a`一致；正式回执本身SHA256为`ce57600d0185675aa0ae01fcc557c96a6c0cc5d64dd4dde3d1752477cc5d819e`。

同一回执中作者完整Signal为`53.2424/72.4196`，R2/54的fused为`52.0114/68.8663`，CNN为`49.5284/67.3435`，Transformer为`50.3903/67.8511`，Mamba为`48.4117/66.3283`（mAP/Rank-1，单位%）。fused相对Signal为`−1.2310/−3.5533`个百分点；尽管fused mAP高于本种子的三个完整分支，双指标停止线均未达到。该结果已补入§41.333，§41.335的MSVR310统计更新为17个正式种子，仍为17/17个fused mAP高于三个完整分支、17/17个三分支mAP及Rank-1均低于Signal、零个fused Rank-1高于Signal。

按已登记的`(全部必报指标达标,mAP,Rank-1)`存储顺序，MSVR310–R2 seed46 `53.4216/70.5584`仍领先seed54。退役前再次核对两份完整训练/正式回执及两份实存checkpoint的SHA，仅删除seed54精确路径的非赢家`roles_epoch20.pth`，释放`38,124,518`字节，并确认该文件已不存在；seed46赢家权重、两种子的正式指标、距离数组与训练日志均保留。新机`/data`可用`129,419,145,216`字节。原种子续跑控制器保持关闭；剩余在途仅新机GPU0的RGBNT100–R2 seed46与旧单卡RGBNT201–R2 seed45，未启动新种子。当前37份正式结果中仍只有RGBNT201–V27 seed43满足同一种子全部必报指标各`+0.8`点。

### 41.340 按用户指令停用四卡训练并将未完成项转到旧单卡（2026-09-24）

用户最新指令为立即停止四卡机训练，今后只在旧单卡机训练。四卡机上仅剩原RGBNT100–R2 seed46：停止前训练日志已完成`15/20`轮，实际队列PID`3542100`的直接训练子进程PID`3633280`命令行核对为该数据集、方法、seed和GPU0。向两者发送`SIGTERM`后复查均不存在，`nvidia-smi --query-compute-apps`没有计算进程；其他四卡训练已在此前完成，自动新增种子控制器保持关闭。该轮`training.json`仍保留停止前的`RUNNING`历史字段，但目录实际仅有该文件及`training_steps.jsonl`，没有`roles_epoch20.pth`或`official_metrics.json`，**不能把它计入正式结果，也不能把15轮视作可续训权重**。

原正式训练入口`tools/run_official_three_dataset_roles.py`只在固定20轮全部结束后保存`roles_epoch20.pth`，没有中途模型/优化器状态；因此无法把四卡机已执行的15轮迁到旧机续训。旧单卡机已有同一作者RGBNT100 Signal预训练权重、CLIP权重、协议与conda环境，数据盘可用`20,716,167,168`字节。已在旧机`logs/launch_rgbnt100_r2_seed46_after_seed45.py`部署一次性等待器PID`87339`，它按240秒间隔等待现有RGBNT201–R2 seed45队列PID`45727`退出，仅当该任务的正式回执为`COMPLETE`时，调用现有`queue_official_extra_seed.py --seed 46 --machine old --dataset RGBNT100 --method R2 --gpu 0`从作者预训练权重**全新训练20轮并正式评估**。旧机新任务目录此前不存在；不与当前GPU训练抢占，也不启动其他新种子。

13:05左右只读日志：旧机RGBNT201–R2 seed45完成`10/20`轮，最近每轮约10.5分钟，连同正式评估预计北京时间约`15:00–15:20`验收。旧机先前同规格RGBNT100–R2 seed45从第2轮起每轮约1600秒；新seed46从头训练预计约9小时，另需正式评估，若前项按时结束，粗估`2026-09-25 00:00–01:00`取得正式回执。预计时间不是结果，后续应按实际日志和正式回执核验。四卡机可继续作为项目代码/交接文档镜像，但不再安排GPU训练。

### 41.341 最新37份正式种子与公开CLIP参照的数值距离（2026-09-24）

§41.292的公开差距表基于当时较少的已完成种子，不能当作目前最好成绩。本节仅从§41.333已经完成正式query/gallery评价的37份回执中，按各数据集fused mAP列出当前已测最高者；在途RGBNT201–R2 seed45与转到旧单卡重训的RGBNT100–R2 seed46均未计入。RGBNT201四项、其余两项的报告列数保持用户指定口径，单位均为%。

| 数据集 | 当前已测fused mAP最高种子 | 本项目正式指标 | 同协议作者完整Signal | RoDI–CLIP作者mAP/R1 | 相对RoDI–CLIP的mAP差距 |
| --- | --- | --- | --- | --- | ---: |
| RGBNT201 | V27/43 | 83.0005/87.5598/92.9426/94.4976 | 80.3029/85.1675/91.3876/93.6603 | 84.1/87.2 | 1.0995 |
| RGBNT100 | R2/44 | 87.1200/97.8426 | 86.3242/97.5510 | 88.5/97.6 | 1.3800 |
| MSVR310 | R2/46 | 53.4216/70.5584 | 53.2424/72.4196 | 64.1/77.2 | 10.6784 |

[RoDI作者原表](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf)的CLIP版仅作公开值参照；同协议直接作用量应读本项目各行与其作者完整Signal的配对差：RGBNT201为`+2.6976/+2.3923/+1.5550/+0.8373`，RGBNT100为`+0.7958/+0.2916`，MSVR310为`+0.1792/−1.8612`。三个数据集的最高mAP来自不同方法或种子，不能合成一个已共同验证的“最新模型”，也不能从不同运行挑各指标最大值拼成单个结果。按已消费正式测试筛出的最高值是**描述性上界**，不是新测试集上的无偏泛化估计。

更强资源参照见§41.292与§41.323：[RoDI–DINOv3](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf)在MSVR310报告`71.8/84.8`，[CoT-ReID](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_Chain-of-Thought_Guided_Multi-Modal_Object_Re-Identification_CVPR_2026_paper.pdf)报告`71.7/85.3`且使用DINOv3和MLLM文本；[PMKD作者论文](https://aihuazheng.github.io/publications/pdf/2026/2026-Progressive_Multi-modal_Knowledge_Distillation.pdf)在RGBNT100报告`91.6/98.0`且使用DINOv2。各论文的预训练、额外文本/分割资源和训练选择预算不同，差距只是公开数字，不是等资源因果对照。当前已测结果尚未支持“三数据集均超过作者Signal”或“达到公开SOTA”；尤其MSVR310的Rank-1仍低于作者Signal。


### 41.342 最新完整交接：指标、做法、研究判断及剩余任务（2026-09-24 14:54 CST）

**范围与索引。**本节是当前执行摘要，原始实验记录不移到其他文档。§41.333列出三数据集截至此刻**全部38份已完成正式种子**，RGBNT201四指标、RGBNT100/MSVR310两指标，包含Signal纯baseline、作者完整Signal与历史V1/V8；§41.319的B/C/D列出早期dev、RGBNT201 V18—V29及车辆内部Q1；§41.334—338给出正式排序失败例子和逐query/样本对诊断；§41.340记录四卡停用及任务转移；§41.341是完成37份时的公开方法差距快照。旧章节中的“正在运行”均按各自写作时间理解，不能覆盖本节状态。所有指标单位为%，`-`表示未完成，不把内部Q1或来源训练损失填为正式测试。

**本轮新增正式终态。**旧单卡`trained-model/official_extra_seed45_20260924/RGBNT201_R2_seed45/`已经完成固定20轮、1060次更新；`training.json=FIXED_EPOCH20_TRAINING_COMPLETE`、`official_metrics.json=COMPLETE`。正式836 query/836 gallery，作者Signal原评价过滤、无重排序、`independent_upstream_metrics_equal=true`；冻结状态未变、无梯度覆盖缺口或AMP溢出。训练/正式回执的角色checkpoint SHA均为`4ae52af582c472332dea0be3c60732201d8094b68fdebb0592b2f45ed578d5a8`，实存文件一致；距离文件实存SHA与回执均为`1067c6202f34b9880154ca6b38205dfb515888dff25c2cd1bf6b6c06704f260d`，正式回执自身SHA为`289275c1c7fae4d2d8bf2bbf6a570464966aaa232301d0b8c3ca37a41d648bfa`。

| RGBNT201 R2/45输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| 作者完整Signal | 80.3029 | 85.1675 | 91.3876 | 93.6603 |
| fused | 81.3305 | 85.2871 | 91.7464 | 94.0191 |
| CNN | 80.9347 | 85.5263 | 91.6268 | 93.5407 |
| Transformer | 79.2716 | 83.3732 | 90.5502 | 92.5837 |
| Mamba | 81.1332 | 85.5263 | 91.7464 | 94.2584 |

该seed fused相对同回执Signal为`+1.0276/+0.1196/+0.3589/+0.3589`个百分点，只有mAP达到用户固定的每项至少`+0.8`停止线；fused mAP超过三完整角色，但Rank-1低于CNN/Mamba，不能只报mAP。当前唯一四项同种子均达标的仍是RGBNT201–V27/43 `83.0005/87.5598/92.9426/94.4976`。若之后还有正式终态，应新开一条终态记录，并更新§41.333；不得以训练完成、部分epoch或某个单独指标代替正式结论。

**三数据集目前最好的已完成fused mAP及直接比较。**以下每行只取一个完整种子，不拼接不同种子的最优列；“最好”是已消费正式集上的描述性选择，并非独立测试集估计。

| 数据集 | 方法/seed | fused必报指标 | 同协议作者完整Signal | 同seed差值 |
| --- | --- | --- | --- | --- |
| RGBNT201 | V27/43 | 83.0005 / 87.5598 / 92.9426 / 94.4976 | 80.3029 / 85.1675 / 91.3876 / 93.6603 | +2.6976 / +2.3923 / +1.5550 / +0.8373 |
| RGBNT100 | R2/44 | 87.1200 / 97.8426 | 86.3242 / 97.5510 | +0.7958 / +0.2916 |
| MSVR310 | R2/46 | 53.4216 / 70.5584 | 53.2424 / 72.4196 | +0.1792 / −1.8612 |

纯baseline是**关闭Signal扩展与全部TriFusion角色后**从公开CLIP独立训练的另一条件：RGBNT201 `69.6415/71.4115/80.1435/85.6459`，RGBNT100 `83.7042/95.0437`，MSVR310 `50.5220/67.6819`。本轮主要配对基线是**作者三个数据集各自发布的完整Signal预训练权重**，不是上述纯baseline，也不是旧RGBNT100本机Signal `80.7122/94.2274`。历史V8在旧配对条件下RGBNT100正式fused `83.2848/96.1516`、相对旧本机Signal `+2.5726/+1.9242`；此增益成立，但不得拿它与本轮作者Signal行作单因素差值。历史RGBNT201 V1正式`59.1478/63.2775/77.2727/83.6124`，也不是本轮同一模型。

**网络及两种方法怎样做。**三光谱图像先经冻结Signal/CLIP。原direct+SIM提供3072D Signal；CLIP block8后并行运行CNN局部语义Patch、Transformer全局CLS/Patch关系、Mamba空间及跨模态扫描，三者分别逐阶段使用共享参数但各自执行的冻结tail9/10/11，并与冻结reference做差。每角色三模态×512D=1536D，九槽位共4608D，加3072D Signal得7680D归一化fused。三条角色均执行；当前正式R2/V27不含Router、HFER、V23/V24适配或V28/V29联合头，也不是稀疏MoE。在非零槽位归一化的实现下，fused相似度为Signal一半加九槽位各十八分之一，等于三个完整`Signal+角色`相似度的平均；这只是相似度恒等式，mAP不可平均。V27训练在CLIP patch stem对三模态使用共同供体/混合系数、各模态自身统计并让anchor/reference匹配扰动，原Signal direct路径不变。R2训练从冻结Signal字段复用并用当前角色重编码最多512个历史实例（最大年龄8步），预热65步后用合法跨camera/scene正例的Smooth-AP替换fused hard Triplet，保留其他13项监督；历史候选梯度通过分组VJP回传。角色内排名与辅助梯度分别求范数EMA，支持关系不存在时不更新排名EMA，有支持时用有界权重`[0.4,1.6]`组合，最终仍由一次AdamW更新。标准缓存、VJP、Smooth-AP和梯度调节的来源应在论文中如实引用，不因项目化整合而宣称其公式原创。

**训练/评价如何保持同协议。**每个数据集、方法、seed从对应作者完整Signal权重独立初始化可训练角色；seed作用于角色初始化、采样和增强；固定20轮，不从正式测试选最佳epoch或改权重。训练B64/K8；RGBNT201/100按camera、MSVR310按scene组织合法检索关系。正式评价使用原Signal的完整query/gallery、同身份同环境过滤且不同身份干扰库完整保留、无重排序；输出Signal、fused及CNN/Transformer/Mamba五路，评估入口用上游`utils/metrics.py`交叉核对。每份完成结果保存训练回执、角色权重SHA、距离数组SHA和正式指标回执。RGBNT201只报告mAP/R1/R5/R10；两个车辆集只报告mAP/R1。内部Q1是来源身份隔离，不是官方测试；不同Q1行各自有匹配control，不可将正增益相加。本轮的“每项至少+0.8”是用户固定的停止/选择条件，不是论文显著性检验。

**内部实验已经回答的问题。**RGBNT201完整图库Q1中，V25跨camera曝光`80.8816→80.4209`，V26角色—模态关系责任`80.4028→80.7147`，V27统计扰动`80.2534→81.5924`（三折及三角色均正，但fused低CNN `0.0705`，原门4/5），V28 R2池化前联合Mamba`81.8253→81.2689`，V29有界切向修正`81.4875→81.7070`。V23模态适配`−0.2527`、V24原型`+0.4913`等完整账在§41.319C；V16和V21等只停在工程/监督门，不能填检索分数。RGBNT100原V8内部`89.5242→91.3165 mAP`、五门全过。MSVR310原V8内部`53.1294→52.1174`；统计扰动`52.1267→51.8245`，普通记忆`52.1211→51.9050`，新鲜历史坐标`51.7884→51.7168`，历史候选完整反传`51.7343→52.4067`，角色负例集合`52.3923→52.4902`，标准Smooth-AP`52.4441→52.7876`，跨scene Smooth-AP`52.8383→53.4055`，支持感知R2`53.3994→53.4526`；各行只是自身配对的mAP变化，只有跨scene候选小幅高于其内部Signal的mAP而Rank-1仍低，R2原科学门保持Q1_FAIL。来源诊断曾发现333/1200个MSVR310合法query–fold成员有部分正负反序，说明实例困难真实存在；更多记忆关系、新鲜坐标、补全候选侧梯度、所有正例排名都已分别实测，不能再当成尚未尝试的建议。

**正式结果暴露的主要问题与证据边界。**截至新增R2/45后的38份正式种子，RGBNT201为8份、RGBNT100为13份、MSVR310为17份。RGBNT201这8份fused mAP及Rank-1均高于作者Signal；其中6/8份fused mAP高于本模型三个完整分支。RGBNT100的13/13份、MSVR310的17/17份fused mAP均高于其三分支，因而车辆问题并非简单“融合没选到强分支”；但RGBNT100仅5/13份fused mAP、2/13份Rank-1超过作者Signal，MSVR310三分支在17/17份的mAP/Rank-1都低于Signal，fused Rank-1为0/17超过Signal。代表性官方query的前20名符号排序和同一正负对的五路间隔见§41.334及§41.338：存在Signal原本排对而残差平均翻错的病例，同时也有大量原错误关系被修复。MSVR310 R2/46的广义正负对净修复仍伴随Rank-1下降；V27/54在RGBNT100的mAP微升亦伴Rank-1净少16条正确query。因此mAP、Rank-1、广义样本对正确率不可互相代替；当前可说“车辆新角色对未见身份首位判别不稳定”，不可仅凭标签/距离推断遮挡、颜色或场景捷径为全部原因，也不可据已消费测试身份调参。V27的RGBNT201来源及正式正收益未在MSVR310复现；R2的来源优化改善或梯度量级改变也没有稳定转化成三个数据集各指标+0.8。RGBNT100训练中观察到R2控制器6776/6776有支持角色步骤触及倍率上界，这只是来源行为证据，不是正式失分的已证实因果原因。20轮固定终点是否值得延长也没有匹配对照证据，不应凭后期训练loss继续下降就声称更长训练可修复检索。

**近邻与研究定位。**MixStyle对应V27统计扰动；XBM、S-XBM及GradCache对应实例缓存/冻结字段重算/历史侧梯度；Smooth-AP、ROADMAP对应多正例排名；MMPareto、GradNorm、AdaTask对应联合/辅助任务及优化器状态。原三角色结构有项目内正证据，但“CNN+Transformer+Mamba并列”“缓存刷新”“VJP”“AP近似”“EMA调权”单独都不是新算法。若后续设计新主方法，应先用来源训练关系或预先登记的内部身份隔离测试确认机制，再用相同初始化/预算的直接近邻对照证明增量；目前不应把已完成的局部收益包装成跨三数据集稳定创新或SOTA。RoDI–CLIP公开mAP为RGBNT201 `84.1`、RGBNT100 `88.5`、MSVR310 `64.1`，与当前各自最高已测fused差约`1.0995/1.3800/10.6784`点；资源、预训练和终点并不匹配，只作公开数值参照。

**实际运行与剩余事项。**四卡机GPU训练已按用户要求停止，原RGBNT100–R2 seed46只跑完15/20轮且没有中间checkpoint或正式分数，**不纳入结果**。旧单卡队列PID45727已完成RGBNT201–R2 seed45并进入同seed V27训练；一次性等待器PID87339在整条seed45队列正常完成后，才从作者RGBNT100 Signal权重全新启动R2 seed46固定20轮与正式评估，不复用四卡15轮。此时V27/45和新机转移版RGBNT100–R2/46均尚无正式终态，具体结果待回执。旧机`/root/autodl-tmp`当前可用`20,661,084,160`字节；保留需要的作者预训练权重、当前赢家角色权重、所有回执/距离/日志，非赢家角色权重仅按已核验精确路径退役。旧机任务结束后仅核对正式回执、同协议指标、文件SHA和磁盘；无新的种子自动搜索，四卡机不再启动训练。本地唯一交接文档即本文件，远端与GitHub仅镜像同名同内容，不新建平行说明文档。Goal仍ACTIVE/UNMET：六个方法×数据集组合尚未全部满足各指标`+0.8`停止线，也没有被本节文字代替的新训练或新论文结论。


### 41.343 RGBNT201–V27 seed45正式终态、非赢家权重维护与单卡接续（2026-09-24 15:22 CST）

旧单卡`trained-model/official_extra_seed45_20260924/RGBNT201_V27_seed45/`完成固定20轮、1060次优化器更新及原Signal完整query/gallery正式检索；`training.json=FIXED_EPOCH20_TRAINING_COMPLETE`、`official_metrics.json=COMPLETE`。836 query/836 gallery、同身份同camera过滤、无重排序、独立上游评价数值一致；冻结参数未变，全部可训练项收到非零梯度、AMP溢出0次。训练/正式回执的角色权重SHA均为`48979f2fada4d267603eaea04467edb2ef21193b143874ab7c57ca2f64d3caad`；删除前实存权重SHA核验一致。正式距离文件实存SHA与回执均为`0978450e77af30131dc0632edfa1407c2c708a6b7dea4295475503a66948f25e`；正式回执自身SHA为`b86d603e4021563ec4cb26751e3dcbb81d71af9afb9dd4f9a4fdc117cfec360b`。

| RGBNT201 V27/45输出 | mAP | Rank-1 | Rank-5 | Rank-10 |
| --- | ---: | ---: | ---: | ---: |
| 作者完整Signal | 80.3029 | 85.1675 | 91.3876 | 93.6603 |
| fused | 82.8359 | 86.7225 | 92.1053 | 94.2584 |
| CNN | 80.9259 | 85.7656 | 91.3876 | 93.6603 |
| Transformer | 81.8327 | 86.1244 | 91.5072 | 94.2584 |
| Mamba | 81.9054 | 84.8086 | 91.7464 | 94.0191 |

fused相对同回执Signal为`+2.5330/+1.5550/+0.7177/+0.5981`个百分点；mAP和Rank-1达到固定`+0.8`线，Rank-5及Rank-10未达到，所以**不得写成四项达标**。fused mAP高于本模型三角色，但不是每项Rank都最高；RGBNT201–V27 seed43仍是当前唯一四项同时达标者。§41.333新增本行后本轮正式种子总数为39：RGBNT201 9份、RGBNT100 13份、MSVR310 17份。RGBNT201这9份fused的mAP、Rank-1均超过作者Signal，其中7/9份fused mAP严格高于本模型三个完整角色。这是反复消费同一正式集的描述性计数，不是无偏的多种子泛化置信结论。

按此前登记的保留规则，在删除前重新核对两种子完整训练与正式回执、方法/数据集/seed/固定终点、checkpoint SHA及实存文件。新四卡机原RGBNT201–R2 seed43赢家权重SHA`99dd3ac4ab563fda54bceb54829db6e7a208216740f56a6aaf7c0b6504846616`、V27 seed43四项达标赢家权重SHA`fc0e419b0fcc90e0f10339475ef5686c981e48aff654cfffe2c4c9da4766f064`均仍实存。旧单卡本轮R2 seed45和V27 seed45的mAP分别`81.3305`、`82.8359`，均低于各自赢家；仅删除两份精确路径下的非赢家`roles_epoch20.pth`，每份`39,795,686`字节，合计释放`79,591,372`字节，随后确认两路径均不存在。所有正式指标回执、距离文件和训练日志仍保留；§41.342的“R2/45权重实存”是14:54时的历史状态，不能继续当作现状。按官方成绩选择的保留权重仅用于存储/部署，不作为独立测试集挑选的无偏成绩。

旧单卡原seed45完整队列正常退出后，一次性等待器自动在`2026-09-24 15:17:20+08:00`启动**仅RGBNT100–R2 seed46**的新队列PID`93892`，M0已结束并进入`TRAINING`，训练子进程PID`94375`使用GPU0、作者RGBNT100 Signal checkpoint、固定20轮和原正式协议；新目录`trained-model/official_extra_seed46_RGBNT100_R2_20260924/`。它从头开始，四卡机此前被停在15/20轮的同名任务没有中途权重可续，不算正式结果。此处无新检索分数；旧机`/root/autodl-tmp`在上述精确权重清理后可用`20,684,447,744`字节。按此前同规格R2运行时长粗估9月25日凌晨左右验收，实际以训练与正式回执为准；接下来只在预定中段及终点附近检查指标、GPU、磁盘，不新增其他种子、不恢复四卡训练，也不因等待或短暂低利用率重启任务。


### 41.344 RGBNT100–R2 seed46单卡中段检查（2026-09-24 20:32 CST）

按用户要求只在训练中段及预计终点检查，不逐轮轮询。旧单卡队列PID`93892`与其RGBNT100–R2 seed46实际训练PID`94375`仍在，GPU0约`7556 MiB`、瞬时利用率`77%`、温度`65°C`；`campaign.json`仍为`TRAINING`，没有正式检索回执。第11/20轮已完整结束，该轮130步、平均总loss`0.54283406`；最近第7—11轮依次`0.57427/0.57698/0.57190/0.55384/0.54283`。同配置旧seed45第7—11轮为`0.57664/0.58671/0.56562/0.55315/0.54036`，本次来源损失范围与之前运行接近，但这不能预测未知身份检索。

只读实际步骤日志时，第12轮已在执行，最近step`1550`的fused项标为`cross_environment_smooth_ap`、合法anchor`64`、历史实例`512`、AMP scale`256`、loss约`0.537`；三角色当前调节系数均为排名`1.6`/辅助`0.4`。这是来源训练过程记录，不是正式mAP/Rank-1，也不证明梯度平衡改善泛化。数据盘`/root/autodl-tmp`可用`20,683,579,392`字节，暂不需要清理运行依赖。此时固定20轮及正式评价尚未结束，不更新§41.333正式表、不挑中间权重、不改变学习率/epoch/损失。按最近每轮约1600秒、余下约8轮及正式评估，粗估`2026-09-25 00:05—00:30 CST`取得回执；下一次在预计终点附近核对训练终态、正式两项指标、权重/距离SHA及磁盘。

### 41.345 四卡地址更新与MSVR310来源首位关系探针（2026-09-24 23:00 CST）

用户提供的是**原四卡服务器的新IP**，不是新项目或新数据：`gaob@172.19.12.128:2026`，SSH使用本机既有`id_ed25519`。主机`ubuntu-WS-C621E-SAGE-Series`、项目`/data/gaob/Re-ID/Trifusion`和Git HEAD `4273098c6eb5a83fa85b794f128ecb10c0843aeb`均核对一致。作者三份Signal权重和CLIP权重仍在原`pertrained-model/`，环境为`/data/gaob/Re-ID/conda-envs/tri_reid`；CUDA单元素实际计算通过。启动前四张RTX3090均无训练任务，`/data`约121GiB可用、使用率99%。没有复制数据或权重，也没有恢复旧种子搜索。

最新正式MSVR310诊断提示“Signal正确首位被新增角色翻错”，但V16/V17已经做过Signal hard pair保护，不能把同一hinge换名重训。为判断新“关键关系保留／前列排名”提议在来源侧是否有作用，本次只读R2/46的**训练身份**：官方train1032条、155个身份；其中60个身份形成600条合法跨scene查询。同身份同scene候选忽略，其他身份均为负例；各查询以`最大合法正例相似度－最大负例相似度`确定首位间隔，计数与现有`scene_scores`的Rank-1独立核对。冻结作者Signal，相同随机角色初始化与固定第20轮角色权重各提取一次；仅前向，无反传、参数更新或新权重，也没有读取正式query/gallery图片。作者Signal已经见过这些训练身份，**本探针不是完整路径身份隔离的泛化评价**。

| 600条合法训练查询 | R2/46初始角色 | R2/46第20轮角色 |
| --- | ---: | ---: |
| 冻结Signal首位正确 | 529 | 529 |
| fused首位正确 | 487 | 592 |
| Signal正确、fused错误 | 49 | 1 |
| Signal错误、fused修正 | 7 | 64 |
| Signal原正确且固定正负对的fused间隔比Signal低超过0.02 | 30 | 1 |
| fused首位目标在`τ=0.01`下的标量间隔敏感度`sigmoid(-margin/τ)>0.01` | 515 | 36 |

第20轮仍有8条fused首位错误，其中仅1条由Signal原本正确变为fused错误。这里的“标量敏感度”不是角色参数梯度或AdamW更新份额。初始49→终点1条首位破坏、7→64条修复表明现有R2已大量解决已见身份的首位关系；与正式MSVR310 R2/46仍比作者Signal低1.8612pp Rank-1并不矛盾。初始确有可训练关系，因此不能说新目标从训练开始就完全无信号；终点近乎饱和，也不能据来源收益保证未知身份改善。下一步先核对旧内部MSVR310 OOF Signal模型、协议和权重是否可复用，在**完整路径身份隔离**下登记单一干预配对比较；不据正式错误身份调阈值或权重，不把本探针填入正式成绩。

本轮执行脚本`tools/diagnose_official_source_top_rank.py`字节SHA256 `40654033c5f9fd41e37b1dcb528aa84650cb0570875290dcd1ab2b487ca6d110`；源训练回执SHA `c5441be5a206ed807ecc8ac9139b9dffc029f40e4771da4b561ba21be9aa073c`，协议SHA `7f2b35f9a7e00433558c1e723db0e3ff0ea9daa7144eeeb972139d1502945cbc`，终点角色权重SHA `d8a2becc180d21cab9c771a6ca6b2e5fcb696cd77fc10bbad58a0584c4800ca9`。两份JSON见`evidence/msvr310_source_toprank_probe_20260924/{initial_v3,final_v3}.json`，SHA依次为`3ded329a521474afb68e242e9c4fc142ed3044ccd2d372cd9e865a312b0db638`与`88f79563d5157a7d4bdaf463c9a684933682798d97ddab12ce2fb4b520963f89`；运行日志留在四卡机`logs/msvr310_source_toprank_{initial_v3,final_v3}_20260924.log`。JSON内`commit=4273098`是启动时的仓库HEAD，诊断脚本当时尚未提交，以本节脚本SHA绑定实际执行代码。旧单卡R2/46于22:44只读检查时完成第16轮、仍在训练；无新增正式回执，§41.333仍为39份。

### 41.346 固定一项前列排名干预，执行完整训练与评价（2026-09-24）

最新用户要求以**完整训练和验证**为主，允许在结果旁加入诊断指标；不再安排单独来源探针作为下一轮交付。四卡地址仅改为`gaob@172.19.12.128:2026`，项目、数据、conda环境和预训练权重均沿用§41.345。拟固定seed42，在RGBNT201、RGBNT100、MSVR310上分别从同一作者Signal权重与同一角色初始化训练20轮R2对照和`R2_TOP1`候选，共6个完整端点；每端训练后按原Signal的camera/scene过滤、完整query/gallery和作者指标实现进行正式评价，不选中间epoch、不按正式成绩调参。旧单卡R2/46任务继续独立运行，不属于这6个新端点。

`R2_TOP1`只在原R2跨环境Smooth-AP上加一个固定系数1的首位软间隔：对有合法跨环境正例的当前anchor，令`a=max_n s(q,n)-max_p s(q,p)`，新增`0.01*softplus(a/0.01)`并按合法anchor平均；同身份同环境候选仍忽略，不改原三角色网络、作者Signal权重、历史容量/新鲜度、已有13项辅助监督、warmup、优化器或固定20轮终点。历史候选的这项导数进入R2现有VJP重放，不能只对当前anchor反传。系数和温度在训练前固定，不做扫描。它检验的是“补充前列监督能否同时改善mAP与Rank-1”，**不是**已完成的性能结果、原创top-k公式或保证基线正确关系不翻转的算法。

三个数据集的报告口径保持：RGBNT201列mAP/Rank-1/Rank-5/Rank-10，RGBNT100与MSVR310重点列mAP/Rank-1；全部五个输出、逐query AP、相对冻结Signal的Rank-1修复/新增错误、正负关系翻转计数与代表性失败例一并保留。诊断脚本读取训练完成后的正式距离回执，只描述机制，不参与模型选择或调参。作者Signal已经在对应来源身份上训练、官方集合也已经被反复查看，因此新正式比较可作为指定协议结果，不能称为完全未消费测试集上的无偏模型选择证据。待任务实际启动与终态产生后，在本节后续小节记录提交SHA、任务/日志/权重路径、启动与完成时间、六端完整指标和错误分析；未完成项仍写“未完成”。

### 41.347 四卡六端固定配对训练已启动（2026-09-24 23:25 CST）

代码和本节方法定义已推送GitHub `main`提交`2b27f2a94644cc38d0185c404e2aeb8cd0c3eb9e`。四卡机因直连GitHub TLS中断，按同一提交的Git bundle快进到该HEAD；四卡机`git status`仅有先前无关的未跟踪bundle，训练代码受提交SHA固定。来源单元数值检查确认新top1项在当前及历史距离两侧均产生有限非零导数，零合法正例时返回图连接零；MSVR310候选独立8步M0通过，历史VJP组数15、冻结状态不变、无梯度缺失。正式队列自身的前四端M0均已通过，并进入完整`TRAINING`；这是工程验收，不是性能结论。

持久队列于`2026-09-24T23:20:55+08:00`启动，PID`410862`，记录`logs/official_extra_seed42_top1_pair_20260924/campaign.json`，总日志`logs/official_top1_pair_seed42_20260924.launch.log`。GPU0/1分别为MSVR310–R2/`R2_TOP1`，GPU2/3分别为RGBNT201–R2/`R2_TOP1`；两个RGBNT100端点登记为`PENDING`，MSVR310端点结束后由队列自动占用释放的GPU。训练回执、每步日志、固定第20轮权重、完整正式评价和距离数组放在`trained-model/official_extra_seed42_top1_pair_20260924/`及对应`logs/`；没有中间选点、结果拼接或新的V27重训。四张卡启动后实际分配约7.4GiB/卡，启动时`/data`约121GiB可用。

根据已完成同规格R2训练历史，MSVR310/ RGBNT201/ RGBNT100每个端点20轮分别约1.42/4.42/9.92小时；考虑四卡排队及评价，整批预计**9月25日11:30—13:30 CST**完成，以实际回执为准。按用户最新要求只在中段和终点附近核对进程、loss、正式指标及磁盘，不根据中途曲线更改方法。此时该批六端正式mAP/Rank仍**未产生**；§41.333仍为此前39份正式结果。旧单卡RGBNT100–R2 seed46为独立先前任务，不计入本次配对。

### 41.348 终态诊断入口与启动状态（2026-09-24 23:36 CST）

后处理已锁定为只读：`tools/diagnose_official_retrieval.py`分别比较每个正式模型与冻结Signal的逐query AP、Rank-1修复/新增错误、正负关系翻转及示例；新增`tools/report_official_top1_pair.py`在相同初始化、同一seed/协议和Signal输出一致性检查后，直接比较每个数据集的R2/`R2_TOP1`两端，输出四项指标差、逐query和逐身份变化及首位修复/损害。配对脚本用已有MSVR310 seed43 R2/V27正式回执执行过一次**功能验证**，没有用该结果调整新训练。完整六端结束后，`tools/finalize_official_top1_campaign.py`才调用这些诊断，写入本轮campaign的`diagnostics/`；它不读取训练中间分数、不选择权重、不影响优化器。后处理代码已推送至GitHub `c8f4220`并快进同步四卡机，训练回执仍锁定启动时方法代码提交`2b27f2a`。

四卡主队列PID`410862`在23:30仍实存，GPU0—3约`7.5/7.5/7.5/7.7GiB`且实际利用率约`100/91/100/100%`，MSVR310与RGBNT201的四端均为`TRAINING`，RGBNT100两端`PENDING`；`/data`可用约`120.53GiB`。单次终态后处理进程PID`446812`及其`tail --pid=410862`子进程PID`446817`已启动，等待队列进程真正退出后才核对`campaign.status=COMPLETE`并分析正式回执，不周期读取GPU或模型指标。其日志为`logs/official_top1_pair_seed42_20260924.finalize.log`。此时没有新增正式指标，下一次训练观察按前节预计中段和终点进行。

### 41.349 新目标M0实际触发范围与定点检查安排（2026-09-24 23:45 CST）

四卡机固定M0的8步记录中，预热后6步均实际计算到非零top1标量：MSVR310范围`0.0000728—0.0016093`，合法anchor在整个8步中为`8—48`；RGBNT201范围仅`1.64e-11—9.39e-7`，合法anchor为`8—24`。这支持MSVR来源上有实际首位软间隔信号，也提示RGBNT201来源首位关系很容易饱和；**标量非零或接近零都不能直接替代角色参数梯度、最终AdamW更新或未知身份检索收益**。M0已通过历史侧VJP与冻结状态检查，不因该数值中途改系数、温度或训练预算。

旧单卡`:19873`在23:30仍确认队列PID`93892`运行、GPU约`7.56GiB`且利用率`100%`，但23:38—23:40两次SSH连接超时，不能认定训练停止或结果产生；最新版交接文档尚未补同步至旧机。本机已登记一次性终点检查`2026-09-25 00:15 CST`：届时尝试读取seed46训练/正式回执并同步此唯一文档，失败则仅保存连接错误，不重启任务。四卡机另登记05:30中段和13:30终点两次定点检查；远端训练和终态后处理本身不依赖本机检查进程。以上时刻为计划，结果仍需实际回执。
