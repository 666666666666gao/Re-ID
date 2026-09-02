# TriFusion V12 Complete-Path OOF Router 终态

V12 的完整路径 identity-OOF 教师资格门通过，但由该教师监督的既有层级 Router 没有通过 train-only 泛化门。因此本版本在访问 held-out dev 前停止，没有生成 combined checkpoint，也没有产生新的可部署 mAP。

## 执行边界

- 数据和 GPU：仅远端 RTX 3090；RGBNT201 固定 141-fit 内部身份折。
- 随机种子：42；无多种子。
- Q0：三折，每折从 raw CLIP 初始化训练内部 Signal 教师 50 epoch，再训练专家 20 epoch；final epoch only，无 held-out checkpoint selection。
- Q1：完全复用 V8 Phase-A checkpoint、Router 结构、100 epoch、LR、alpha、质量退化和门槛；唯一变化是 OOF margin cache。
- Q0/Q1 的 dev access 和 official-test access 均为 0；无消融、无超参数扫描、无 baseline 重跑。

## Q0：完整路径 OOF 教师

三个 fold 的 Signal/expert fit identity 均为 94，held-out identity 均为 47；两类训练身份与 held-out 的交集均为 0。可评价 query 数依次为 190、179、202，总计 571。每折均完成 Signal 50 epoch 和 expert 20 epoch，使用 final epoch，AMP overflow 为 0。

### 折内 residual-only 检索

| 输出 | mAP | Rank-1 |
|---|---:|---:|
| CNN residual | 83.7717 | 85.6392 |
| Transformer residual | 86.9549 | 89.4921 |
| Mamba residual | 85.8870 | 88.6165 |
| residual bank | 87.9968 | 90.1926 |
| residual expert hard Oracle | **92.2679** | **95.2715** |

Oracle 比最强固定专家高 `5.3130 mAP / 5.7793 Rank-1`。CNN/Transformer/Mamba 的独有 AP 胜出为 `79/118/76`；按 slot identity margin 统计的专家独有胜出为 `210/186/175`，RGB/NI/TI 独有胜出为 `293/119/159`。slot Oracle mean margin 为 `0.099913`，比最强固定 slot 高 `0.186130`。

所有固定输出均低于预注册的 99 mAP 饱和线，三专家、三模态都有独有效用，hard Oracle 增益超过 1.0 mAP。Q0 的隔离、非饱和、多样性、Oracle、训练计划、运行时和访问门全部通过，`next_phase_authorized=true`。

这些数值是 real-GT、fit-only、identity-OOF 的 residual 诊断，不是 held-out-dev 或 official 指标，也不是可部署结果。fold Signal 是内部教师，不是新的 baseline 复现。

## Q1：冻结专家的层级 Router

三折 Router 各训练固定 100 epoch，共 300 optimizer steps。训练期间 Phase-A expert state SHA 前后同为 `ecfd7fbc...fb77`；没有专家训练。三个 fold 的总损失均明显下降，但 OOF 门如下：

| Router 门 | Learned | Fixed / majority | 结论 |
|---|---:|---:|---|
| expected identity margin | -0.117330 | -0.099975 | FAIL，低 0.017355 |
| top-slot accuracy | 12.2592% | 16.8126% | FAIL，低 4.5534 pp |

质量语义门通过：missing-modality 最大质量严格为 0；单独扰动 RGB/NI/TI 后，各自模态平均质量从 `0.325516/0.316737/0.357746` 降到 `0.111743/0.104597/0.144510`。这证明质量分支能响应受控退化，但不能抵消身份效用路由的 OOF 泛化失败。

Q1 的工程执行状态为 PASS，但科学晋级门为 FAIL：

- `next_phase_authorized=false`；
- `combined_checkpoint=null`；
- `final_training=null`；
- dev access=0，official access=0；
- 不运行 V12-R001 dev、消融、多种子或 official test。

## 结果解释

Q0 证明了一个重要但有限的结论：当完整特征路径对 held-out identity 真正未见时，三类 residual expert 不再出现 V11 的 100 mAP 饱和，并具有显著的 query-level Oracle 互补空间。

Q1 同时否定了更强主张：仅把这套 complete-path OOF utility target 接到既有 all-fit V8 Phase-A Router 输入上，并不能把互补空间转化为可部署路由。一个与证据一致但尚未被因果消融证明的解释，是 fold-specific complete-path teacher target 与 all-fit Phase-A quality feature 之间存在表示坐标/分布错配；当前结果不能据此断言唯一根因。

因此，V12 不支持融合增益、65 mAP、official 或 SOTA 主张。当前同协议可部署最佳仍是 V8 Phase-B fused `58.4050 mAP / 59.3939 Rank-1`，exact Signal floor 为 `58.0109 / 57.4545`。

## 完整性与证据

- Q0：5,839 optimizer steps，0 overflow，峰值 allocated/reserved `11,375.61/17,530 MiB`，耗时 `3,958.52s`。
- Q1：300 optimizer steps，峰值 allocated/reserved `2,459.15/3,400 MiB`，耗时 `32.36s`。
- Q0/Q1 原始 summary 的本地与远端 SHA 分别一致为 `9105b86a...8a8b69`、`c42f1148...c9b5`。
- 以原始 summary 与远端日志为唯一权威来源；任何较早转述中不同的步数、耗时或 cache SHA 均被原始文件取代。

证据文件：

```text
evidence/trifusion_v12_complete_path_preflight_seed42.json
evidence/trifusion_v12_complete_path_oof_seed42.json
evidence/trifusion_v12_complete_path_router_seed42.json
evidence/trifusion_v12_complete_path_execution_provenance_seed42.json
```

V12 至此封存。若提出后继，必须作为新的预注册表示/监督对齐假设，先在 train-only identity-disjoint 门上证明 learned Router 严格超过固定策略；不能对 V12 的 fold、epoch、LR、alpha、margin 温度或门槛进行事后扫描。

独立 result-to-claim 为 `partial/high`：只支持 Q0 窄资格主张。独立完整性审计为 `WARN/warn/Q0_QUALIFIED_Q1_FAILED_DO_NOT_PROMOTE`；GT、完整路径隔离、常规检索归一化、实际执行路径、scope 和评价分类均 PASS，WARN 仅来自远端大 checkpoint/cache 不能由轻量本地 clone 重哈希。
