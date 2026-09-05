# MSVR310 基线与迁移准备（2026-09-06）

状态：源码及完整训练标签准备完成，尚未启动本项目的 MSVR310 训练或检索。
V24 RGBNT201 原六端 Q1 继续运行，不更改其源码、数据、终点或门槛。
本文件整理下一跨数据集实验的确定约束，不是已经注册或执行的车辆训练合同。

## 新取得的直接证据

本次从服务器读取了 Signal commit cd1b0a672d1fe642e7608731cb4899a19dda7d51 的九份源码/配置纯文本；
九份文件均与该 commit 的 Git blob 完全相同。
仓库的既有其他补丁仍由整体 diff SHA b889caca9c4a92689b13eb7e20bd3224067f3e5ed2a3db6825201870ca422741 约束。
路径、全文件 SHA、字节数与本地文本副本位置见
evidence/trifusion_msvr310_signal_source_inspection_20260906.json。
未下载 checkpoint、读取图像、运行模型或检索。

Signal 的 configs/MSVR310/Signal.yml 确定了以下公开实现设置：

| 项目 | Signal MSVR310 原配置 | 当前 RGBNT201 主线 |
|---|---|---|
| 输入高×宽 | 128×256 | 256×128 |
| CLIP stride / Patch 网格 | 16×16 /8×16 | 16×16 /16×8 |
| batch /每身份张数 | B64/K4，16 个身份 | B64/K8，8 个身份 |
| Signal 训练 | 50 epoch，Adam，LR 5e-6 | 不能直接套用当前专家 AdamW 继续训练设置 |
| 调度 | 第 20/40 epoch 衰减，gamma 0.1，无 warmup | 当前专家用 5-epoch warmup/cosine |
| 评价 batch | 64 | RGBNT201 当前完整图库评估为 128 |
| 检索输出 | direct + SIM，保留 camera SIE | 同源完整 3072D Signal 前缀 |
| 保存/评价 | 原代码每 epoch 官方测试并保存 test-best | 项目要求固定终点，不按官方测试选择 |

源码证据：configs/MSVR310/Signal.yml:1–54；modeling/make_model.py:35–36、223–228；
train.py:87–91；engine/processor.py:162–187。
本项目当前 seed42、真实 B64/K8 约束仍有效；因此将来的项目配对基线必须明确标注这些与公开 K4 配置的差异，
不能称其数值为完全照搬作者条件的严格复现。当前没有发起 K4/K8 扫描或额外 seed。

## 相机标签与真正评价环境不同

已经归档的完整 MSVR310 训练清单有 1032 条三元组、155 个身份、camera/v 标签 0–7。
本次对全部 155 个身份做标签普查：

- 155 个身份均有多个 camera/v 标签；真实身份/camera 对 768 个。
- 只有 60 个身份跨多个 scene；95 个身份只在一个 scene。
- 真实身份/scene 对 276 个；scene 值共 30 个，不应假设原始编码连续。
- 若以整个训练集作内部 gallery 并应用同身份同 scene 过滤，600 条记录具有跨 scene 正例。
  这只是标签可评价性统计，不是训练集 mAP 或检索结果。

完整逐身份相机/scene列表及正关系整数计数见
evidence/trifusion_msvr310_source_label_support_20260906.json。
清单来自 evidence/vehicle_query_protocol_labels_20260905.json；本次没有重新载入远端图片。

MSVR310 loader 把文件名 [11] 的 v 字段作为 camera， [6:9] 的 s 字段作为 scene
（data/datasets/msvr310.py:83–91）。
实际 evaluator 排除同身份同 scene，调用链为 engine/processor.py:52–55 →
utils/metrics.py:195–218 → eval_func_msrv:68。
不能把“所有训练身份跨 camera”写成“所有训练身份都提供跨评价环境正例”；
也不能把 RGBNT201 的 camera 含义及相机原型策略自动当成已验证的 MSVR310 scene 策略。

官方 query3 的 591 条 query 来自 52 身份；gallery 1055 条仍保留全部 155 身份，
不能按 query 身份删掉其余干扰身份。原标签审计已确认有 516 条 query 在误用 same-camera
过滤时正例数改变；仍须使用原 scene 过滤。三数据集的成绩继续分别报告。

## 最小实现路线与开跑前仍需完成的工作

1. 从现有 MSVR310 清单建立训练专用 records，保留真实 identity、camera、scene 和三模态文件路径。
   原作者 loader 固定了 root/512/data/MSVR310，项目安装位置为 data/MSVR310；
   新入口必须直接绑定实际目录，不能把路径适配成功当成模型结果。
2. 为内部身份隔离比较先登记 source/held-out 列表及原 scene 过滤。
   对只在单个 scene 出现的 held-out 身份，保留全部 gallery，仅不让无合法正例的记录进入 query 分母。
   列表只能依据标签和固定随机规则产生，不能依据特征、AP 或模型偏好挑选身份。
3. 在每个需要的 source 集上建立新的 Signal 基线，训练和 checkpoint 选择只用相应 source；
   没有已核验的本项目 MSVR310 Signal checkpoint，不能把 RGBNT201 权重或论文 53.2/72.4 冒充车辆复现。
   按固定 epoch 保存，再做明确限定的内部验证；官方测试不用于逐 epoch 选择。
4. 车辆三分支入口显式传入 8×16 grid 与 128×256 共享几何变换。
   现有角色实现可接收 grid 参数，但当前 V24 dual-view loader 固定默认 256×128，
   当前 V18/V24 RGBNT201 evaluator 又按 camera 过滤，因此现有 runner 不能只换 DATASET_ROOT 就用于 MSVR310。
5. CNN 四个水平分区在车辆输入上仍只是空间汇聚，不能称为人体语义部位；
   实际角色、训练预算、两端初始化、完整 Signal baseline-only 与 fused 输出都须登记并验证。
6. 完成车型入口的真实 source 工程检查、完整配对训练合同和全部终态证据后，才报告跨数据集性能。
   本文件未决定把一个未通过 Q1 的 RGBNT201 候选继续推广，也未加入新 scene loss、Router 或结构消融。

优先级仍为 MSVR310 在前、RGBNT100 在后。MSVR310 的数据、源代码和协议准备不等于本项目已测出成绩；
截至本次准备记录，项目 MSVR310 训练次数和检索评价次数仍为 0。


## 后续数据协议准备（2026-09-06T02:17:57.188547+08:00）

训练内部三折清单已固定为 protocols/msvr310_train_oof_v1.json：
全 155 身份/1032 记录，各折 20 个跨 scene 留出身份，总 600 合法 query，
95 个单 scene 身份的 432 记录完整保留为图库干扰。
精确清单、标签掩码核验与边界见 docs/MSVR310_TRAIN_INTERNAL_PROTOCOL_V1_2026-09-06.md。
当前仅数据合同完成；车辆 loader/evaluator 运行入口、训练合同和真实工程门仍待完成，训练/检索仍为 0。
