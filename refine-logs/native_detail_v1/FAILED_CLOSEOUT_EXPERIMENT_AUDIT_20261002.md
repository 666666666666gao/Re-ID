**总体结论：WARN；未发现阻塞性实验完整性错误。**  
`review_independence=same-family`，`acceptance_status=provisional`。原六端验收仍为 **FAILED**；本次失败收尾完成不能改写该结论，也不证明 N1 有效。

以下路径缩写：

- `R`：`C:/Users/gb/.trifusion_github_publish_22c3bee`
- `T`：`R/logs/native_detail_failed_terminal_20261002/raw`
- `S`：`C:/Users/gb/.codex_tmp/clean_clip_audit_source721_20261002`

本次只读取文件并执行独立的 JSON、哈希及计数算术；未修改文件，未 SSH，未调用模型、GPU、优化器、评分器或报告程序。

**A．GT 来源与评估范围：PASS（固定 protocol 与封存源码范围）。**

GT 来自数据集记录中的身份及环境标签。逐条按作者解析规则检查了三个 protocol 的身份、camera/view 字段；全部一致。RGBNT 的 `scene` 是 camera 的副本，实际评分使用 camera；MSVR 使用文件名中的 `s###` 标签。训练身份与 query 身份无交集。

- 作者解析依据：`S/comparators/Signal-cd1b0a6/data/datasets/RGBNT201.py:79`、`RGBNT100.py:76`、`msvr310.py:81`。
- 实际评估从 protocol 提取身份、camera、scene：`S/tools/run_correspondence_roles.py:85–99`；逐 split 遍历全部记录且检查输出数量：`:66–75`。
- RGBNT 同身份同相机过滤：`R/tools/train_rgbnt100_signal_oof.py:253–267`；MSVR 同身份同时间标签过滤：`R/tools/train_msvr310_signal_oof.py:223–237`，与作者 `S/comparators/Signal-cd1b0a6/utils/metrics.py:68` 一致。
- protocol 数量依据：`S/logs/official_three_dataset_protocols_20260923/RGBNT201.json:78309`、`RGBNT100.json:218978`、`MSVR310.json:38538`。

| 数据集 | Query | Gallery | Query 身份 | Gallery 身份 |
|---|---:|---:|---:|---:|
| RGBNT201 | 836 | 836 | 30 | 30 |
| RGBNT100 | 1,715 | 8,575 | 50 | 50 |
| MSVR310 | 591 | 1,055 | 52 | 155 |

MSVR 的 **103 个 gallery-only 身份、464 条干扰记录**保留。所有 query 经对应环境过滤后均有有效正例；protocol 保存的正例/排除数量与独立文本计数一致。未发现模型输出充当 GT。

**B．预测统计归一化检索成绩：PASS。**

AP 分母来自 GT 正例数，CMC 分母来自有效 query 数；未发现按模型预测最大值、最小值或均值缩放最终成绩。依据：作者 `utils/metrics.py:93–106`、`:153–168`；独立 scorer 的 `train_rgbnt100_signal_oof.py:263–267`、`train_msvr310_signal_oof.py:233–237`。

`S/tools/run_official_three_dataset_roles.py:230–238` 的 L2 特征归一化及 `R/modeling/trifusion/native_detail_roles.py:45` 的 attention softmax 属于模型/距离计算，不是成绩归一化。正式回执均为 `reranking:false`，例如 `T/trained-model/native_detail_20261002_v1_clean_clip_high_RGBNT201_seed42_full/official_metrics.json:31`。

**C．文件、数字、终态和失败记录：PASS。**

独立核验结果：

- 终态 INTAKE 的 **63 个文件、3,942,802 字节**全部匹配文件大小及 SHA256。
- closeout INTAKE 的 **8 个文件**全部匹配。
- 原 manifest 的 **238 份源码/config/protocol**全部匹配。
- closeout 的 **55 个本地可读输入绑定**全部匹配；其中六份旧控制正式回执从实际旧归档读取，与旧 SUMMARY、accepted matrix 和新报告一致。
- 六端均有连续 epoch 1–50；逐条检查了全部正式 JSONL 的 epoch、batch 顺序、有限 loss、每轮步数及均值，并与原 `train.log` 和 `training.json` 交叉核对。

| 端点 | 正式 step | 选定 epoch | 正式 mAP | 正式 R1 |
|---|---:|---:|---:|---:|
| RGBNT201 high | 2,649 | 10 | 68.30145965 | 69.25837398 |
| RGBNT100 high | 6,559 | 15 | 79.38738569 | 95.21865845 |
| MSVR310 high | 1,000 | 16 | 51.15014620 | 68.02030206 |
| RGBNT201 low | 2,649 | 13 | 68.13286702 | 71.05262876 |
| RGBNT100 low | 6,559 | 16 | 79.17484541 | 93.76093149 |
| MSVR310 low | 1,000 | 不可用 | 不可用 | 不可用 |

合计 **300 epoch、20,416 step**，与 `R/results/native_detail_failed_closeout_20261002/SUMMARY.json:8–17` 一致。五份正式回执的选点及指标均位于对应 `T/trained-model/native_detail_20261002_v1_clean_clip_{variant}_{dataset}_seed42_full/official_metrics.json:19–32`。六份 `training.json:659` 记录 epoch50，`:691–692` 记录训练选点及完成时间。

原失败证据完整：

- 父 campaign 仍 FAILED：`T/logs/native_detail_20261002_v1/campaign.json:2`；MSVR-low FAILED：`:464–485`。
- 子端 train exit0、evaluate exit1：`T/logs/native_detail_20261002_v1/native_detail_20261002_v1_clean_clip_low_MSVR310/campaign.json:74`、`:108`。
- 原 traceback 精确指向未修改的 `<1e-5` 断言：同目录 `evaluate.log:21–23`；源码 `R/tools/run_visual_update_control.py:241`。
- 原成功报告调用次数仍为0：`T/logs/native_detail_analysis_waiter_20261002.json:2–10`。
- 新收尾实际 exit0：`R/logs/native_detail_failed_closeout_execution_20261002.json:2–12`，输出日志 `R/logs/native_detail_failed_closeout_20261002.log:1`。

还比较了先前726/728归档：原失败 traceback、已完成的 low-MSVR training.json 和 waiter JSON 与终态副本逐字节一致。未发现修改原失败、放宽门槛或用诊断回执替换正式成绩。失败端正式指标保持 null：`SUMMARY.json:187–190`；MSVR high−low 比较保持 unavailable：`:2539–2545`。

**D．实际调用路径与后期轨迹：PASS。**

实际路径成立：

`run_native_detail.py:59` → `run_clean_clip_joint.py:137` → `run_visual_update_control.py:188–193` 的逐轮官方评估；最终 reload 后再次调用 `official_metrics`：`run_visual_update_control.py:235–248`。该函数实际同时调用作者 scorer 与独立 scorer 并校验差异：`S/tools/run_correspondence_roles.py:79–107`。五个成功 `evaluate.log:16` 均含正式回执；失败端实际执行到指标一致性断言。

完整轨迹没有被最佳点替代。epoch50 相对训练选定最佳点的 mAP 变化依次为：

- RGBNT201 high −4.336946、low −3.707399；
- RGBNT100 high −1.374205、low −0.439949；
- MSVR310 high −2.656688、low −3.234307，后者仅为未验收训练诊断。

依据：`SUMMARY.json:29`、`:59`、`:89`、`:119`、`:149`、`:179`，均与原50轮历史相符。可以报告后期检索下降，不能据此确定唯一失败原因。

**E．科学范围、归因和成本：WARN，现有报告已作必要限定。**

- **单 seed、官方选点：** seed42、每端50轮、按最高官方 mAP 选点且并列取较晚轮，其他CMC随同一权重。依据 `run_visual_update_control.py:189–193`、`:236–241`；原计划 `EXPERIMENT_PLAN.md:17–25`。官方集参与研发与选点，不能作无偏测试或多 seed 稳定性证据。
- **Bootstrap：** `analyze_correspondence_distances.py:57–75` 重采样固定模型的身份均值，2,000次；不是训练种子方差。八个有效比较的身份数、query 数、宏平均、修复/新增错误与指标差的算术一致。
- **N1-A 不是单因素实验：**原 CNN attention 读取128语义 patch（`role_global_tokens.py:35–39`），新路径读取512候选并替换 value 来源、增加93,248参数（`native_detail_roles.py:16–22`、`:30–46`）。新报告在 `REPORT.md:30–32` 正确披露。
- **N1-B：** high/low 全部初始状态 hash 相同、参数量相同、512候选相同；实际输入计算仍不同。构造比较源码 `prepare_native_detail.py:53–68`；实际证据 `T/logs/native_detail_preflight_20261002_v1/preflight.json:274–285`、`:314–325`、`:354–365`、`:394–405`、`:434–445`、`:474–485`。不能声称 low 插值恢复了原始细节。
- **成本：**六端训练回执区间之和为 **23,205.861537秒（6.446073小时）**，包含逐轮评估，包含失败端训练；父 campaign wall time **10,805.494915秒**。依据 `SUMMARY.json:2945–2959` 与六份原 training 时间戳。训练区间在构造之后开始（`run_visual_update_control.py:116–138`），不包含 M0、构造、最终 reload、旧控制或单独失败诊断，不能称为总 GPU 成本。

事前门槛未改变：原计划 `EXPERIMENT_PLAN.md:25` 与收尾源码 `analyze_native_detail_failed_campaign.py:128–135` 一致。

| 比较 | RGBNT201 | RGBNT100 | MSVR310 |
|---|---|---|---|
| N1-A：high−原roles | FAIL | FAIL | FAIL |
| N1-B：high−low | FAIL | PASS | UNAVAILABLE |

N1-A 的 mAP 差分别为 **−1.574936、−1.535674、−0.854056**。N1-B 在RGBNT201虽有 +0.168593 mAP，但R1下降 −1.794258，已不满足整体门槛。不能把MSVR缺失解释为通过。依据 `REPORT.md:16–24`、`SUMMARY.json:13–17`。

**F．评估类型与诊断隔离：PASS。**

五端正式检索及旧控制属于 `real_gt`。MSVR-low 的逐轮轨迹和固定checkpoint探针也使用真实GT，但属于 **未获正式验收的诊断**；初始化/M0属于工程验证。

原诊断记录训练 mAP `51.3448542116076` 与 reload mAP `51.344551531014595` 相差 `−0.00030268059300198047`，超过原 `1e-5` 门槛：`R/logs/native_detail_milestone726_20261002/RELOAD_CPU_DIAGNOSIS.json:8–37`。两次探针只证明同一进程内输入/特征/距离重复一致，其边界明确：`R/logs/native_detail_reload_probe726_20261002/INTAKE.json:98–109`。这些分数未进入正式结果表或MSVR配对比较。

**阻塞性错误及最小修复：无。** 保留现有失败记录、门槛及范围说明即可，无需增加 fallback、异常包装或兼容逻辑。

允许的主张是：该固定配方与seed下完成六端训练，五端正式评估有效，一端原严格reload验收失败；N1-A失败，N1-B整体条件未满足。仍未证明原生细节机制有效、稳定增益、单独value来源或各角色的因果必要性、N2/N3有效、失败的唯一数值原因，以及baseline/SOTA目标达成。

审计证据上限：18个checkpoint/distance二进制仍在远端，本次没有重读或重新评分；已核验其封存绑定、实际执行代码和原始文本回执。没有重新检查原始图像字节或作者历史使用的完整文件清单。因此保留 `same-family/provisional`。

关键输入 SHA256：

- SUMMARY：`819ed74b86864786bfc25fac439a1bc8022cdf473e3ab26f021b46648ef04458`
- REPORT：`70b9c8e7bb815fbcfb1751e5ed62365ca1f187248d279a28bdaa87fc1dfb0584`
- 终态 INTAKE：`65a914bac8df65549040f54ec2b4994c8d1d614fbba5def5b6f3bb03bf58f283`
- 原 campaign：`d3eea571eb8d81afb35d7b177041b6df83517303e8ae300422d35dcf9ac85c89`
- 原 manifest：`1ccb8b8d2db2a53c7879f5de88d19e11254b333a6192d1490a4327c118bf57e7`
- 收尾源码：`341ab5dac881bc761584d5c388d0317cd125700c4249ef460b99e5914ec5a35b`
