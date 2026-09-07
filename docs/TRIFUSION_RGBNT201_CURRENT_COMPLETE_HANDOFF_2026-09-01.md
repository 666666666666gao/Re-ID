# TriFusion RGB–NIR–TIR ReID 完整交接（2026-09-01）

## 0. 一页结论

最新状态（2026-09-07T11:53:35.445631+08:00）：V28完整来源几何诊断及切向补充全部完成；三折Mamba及fold1 CNN占44.44%槽位曝光，出现大修正主导。其余槽位多数保留当前方向，不能解释为全部身份信息丢失。Q1_FAIL0/5不变；下一项独立几何约束待注册，三数据集目标继续。

当前用于已完成MSVR310比较及新登记RGBNT100比较的是原V8平行三角色结构：冻结Signal/CLIP，共享block8之前语义和tail9/10/11参数，三个角色分别运行共享tail；CNN处理局部语义Patch高频，Transformer处理全局CLS/Patch关系，Mamba处理空间与位置级三模态扫描。各角色相对冻结reference形成1536D残差，三角色4608D银行拼接3072D Signal得到7680D fused；完整单角色输出为4608D Signal+角色残差。三条路径均执行，当前没有Router/HFER、V23模态MLP或V24原型。两项车辆训练比较均已完成；RGBNT100内部完整比较支持三角色增益，MSVR310尚未超过Signal。下面V1—V8条目保留历史经过，不代表当前又启用了旧模块。

本工程是在 DeMo 代码基座上实现的 RGB–NIR–TIR 多模态目标重识别研究分支。V17完整训练和完整gallery补评已封存为失败；其全部查询/图像/分区诊断见§30。V18完整三折两端20epoch主实验已结束（§33）：fused增益+0.921504 mAP，bootstrap下界-0.117338，未通过固定晋级条件；无D1/dev/official。MSVR310、RGBNT100均已安装核验（§32）。V17 fused相对matched weight0为-0.328915 mAP，没有D1/dev/official结果。当前最高可部署结果仍是V8 Phase-B：冻结dev fused=`58.4050 mAP / 59.3939 Rank-1`，比exact Signal高`0.3941 mAP / 1.9394 Rank-1`并超过三个专家，但仍比65 mAP门低`6.5950`，不能声称SOTA。

V6 的三个候选论文级主创新点已经落到核心代码、专项测试和完整 dev 运行中；性能主门仍然失败：

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
