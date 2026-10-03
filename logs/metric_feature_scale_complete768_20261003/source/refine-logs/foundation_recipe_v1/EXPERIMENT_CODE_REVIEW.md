# F1 基础配方实现独立源码审查

日期：2026-10-02。

- Reviewer：`/root/review_foundation_recipe_735`；fresh independent context。
- 委派配置：`gpt-6-astra` / `max`。
- `review_independence: same-family`；`acceptance_status: provisional`。
- 审查依据：`EXPERIMENT_PLAN.md`、实际实现及其真实调用链、固定 Signal `cd1b0a672d1fe642e7608731cb4899a19dda7d51` 作者源码。
- 范围：研究语义、数据与评价、初始化/保存/严格重载、六端依赖与资源队列、终态报告、2025 私有部署脚本的凭据边界与持久化。
- 本审查未部署、SSH 连接、构造模型、执行神经前向、调用训练/评价器或启动 GPU。只做源码阅读、标准库 JSON/路径语义核对、既有 SHA 绑定核对与 AST 检查；本文件是 reviewer 唯一新增文件。

## Verdict

**PASS：没有剩余 BLOCKING 问题，可以进入计划规定的初始化与六端 M0。** 这不是 M0、容量、数值稳定性、正式严格重载或科学结果的通过回执。完整50轮仍必须等待全部六端真实 M0 通过；任何实际失败保留原始日志，不重试救分。

**BLOCKING：无。NON-BLOCKING：无待修代码问题。** 审查发现的最终评价耗时缺口已由主 agent 做最小修正，随后重新读取修订位置并核对 AST；见下文。

## 研究语义核对

1. **两条件均为真正无模块基础系统。** `PlainFoundation` 只保留 `initialized.backbone.signal`；current 另外保留既有 `neck/classifier`。适配器属于被丢弃的 `CrossLayerAdaptedCLIP`，角色、readout、teacher 等均未注册到最终模型。作者 `USE_A/USE_B` 在构造前关闭，且检查不存在 `SIM/AlignM`。原构造过程会创建角色以复用相同当前头初值，但没有调用其 forward，符合计划明示边界。
2. **公共起点及头的区别处理正确。** clean 构造只读取公共 `ViT-B-16.pt`，逐值校验全部152个视觉 state 项；位置编码按固定作者在 CUDA 上的双线性 resize 过程复核。camera 是作者构造器的新截断正态参数。没有调用继承 runner 中加载已训练 ReID checkpoint 的 build 分支。队列要求同数据集 author/current 的公共视觉、camera、当前头初始化摘要一致；author 真正使用自身原生头，不能称所有模型参数初始化相同。当前 plan 已准确披露。
3. **原生头与原始特征正确。** Signal `DIRECT=1` 的 RGBNT201 返回一个1536维未归一化特征及一个头；两个车辆配置 `DIRECT=0` 返回三个512维未归一化特征及三个头。F1-author 逐头调用作者 loss 并相加，与作者 `engine/processor.py` 的 sign=1 分支一致。作者 BN bias 和无 bias classifier 的冻结状态保留，实际使用的 BN/classifier weight 恢复训练；其他未用头冻结。F1-current 是整体 L2 归一化1536维特征加当前一组头。
4. **作者 loss/optimizer/scheduler 未被当前配方替换。** 直接导入固定源码 `make_loss`、`make_optimizer`、两个 scheduler。三 YAML 的 ID 权重0.25、平滑 CE、未归一化 soft-margin triplet、Adam/bias规则及 MSVR classifier 100倍 LR 均生效。201/100 使用作者带噪声 cosine，MSVR 使用20/40里程碑；每轮训练前 `scheduler.step(epoch)` 与作者入口一致。100的30改50已显式披露。每步保存实际各组 LR，不将名义视觉5e-6误称为全程恒定实际 LR。
5. **当前配方正确复用。** current 为平滑 CE + 归一化特征 margin0.3 batch-hard hinge、AdamW、视觉5e-6和其余3.5e-4、weight decay1e-4、既有5轮 warmup/cosine。optimizer 恰好覆盖所有 requires-grad 参数；公共视觉 FP32存储、camera 和实际头参加更新。原视觉无 prompt/adapter，也没有启用会因 train/eval 切换引入额外随机前向的 dropout；原适配全局控制的 signal.eval 与本次 current.train 不构成已发现的额外有效模块差异。
6. **增强/采样与记录键正确。** author 最终版本使用固定作者 `ImageDataset`、`RandomIdentitySampler`、`train_collate_fn` 及该作者文件自己定义的 `RandomErasing`。独立的逐模态 transform、Resize/flip/pad/crop/normalize/pixel erase 顺序与作者一致。B/K 分别为201=64/8、100=128/16、MSVR=64/4；current 使用既有共享几何三个loader，均64/8。所有实际 collate 键都是 `RGB/NI/TI`；batch转换提供真实 `camera_ids`，Signal调用确实消费该值。四workers及共同AMP/scale256按计划覆盖，且原入口差异已披露。
7. **不是单因子归因。** 两套 head/loss/optimizer/schedule/B/K/augmentation 同时不同，50轮不代表相同步骤、曝光或算力。报告只比较配方包；不把其收益归给 CNN/Transformer/Mamba，不宣称作者未修改精确复现、显著性或 SOTA。历史 shared-global 被单列为容量不同的历史条件，无重训入口。

## 数据、评价和保存核对

已用标准库逐条读完三个既有 protocol JSON，验证所有 index、训练label映射、文件名身份/camera解析及 MSVR scene/view解析。所有查询都有原协议允许的异环境同身份正例，训练/测试身份不交叉。

| Dataset | Train / query / gallery | Train identities | Camera values | Gallery-only distractor identities |
|---|---:|---:|---|---:|
| RGBNT201 | 3951 / 836 / 836 | 171 | 0–3 | 0 |
| RGBNT100 | 8675 / 1715 / 8575 | 50 | 0–7 | 0 |
| MSVR310 | 1032 / 591 / 1055 | 155 | 0–7 | 103 |

- `records_for` 只对训练使用label，query/gallery使用真实identity；100传单张montage路径，作者读取器按0/256/512分割三个256×128模态，另两集传三条路径。
- 评价遍历完整query和完整gallery，不按query身份削减MSVR图库。距离只使用归一化模型特征；监督/评分标签来自protocol真实identity、camera/scene，没有另一个模型输出作为GT。
- RGBNT201/RGBNT100过滤同identity且同camera；MSVR过滤同identity且同scene（时间段），保留其他所有负例。与固定作者 `eval_func` / `eval_func_msrv` 逐字段调用核对；原作者文件路径也在运行时确认。独立 scorer 与作者四指标要求误差小于1e-5个百分点，无re-ranking。
- 训练每轮评价，`>=` 更新best；终评用 `(mAP, epoch)` 确认并列取较晚轮。所有CMC取同份mAP-best权重，未拼列。
- F1的 `save` 保存完整 `model.state_dict()`，包含视觉、camera、头、BN buffers和保留的未用状态，不复用旧runner省略Signal的checkpoint函数。`load_state_dict(strict=True)` 与数据集、配方、初始化文件、protocol绑定同时检查。
- M0保存全state后，从公共初值重新构造并严格加载，检查同一真实audit batch输出；八步有限loss/梯度、所有训练参数出现非零梯度、视觉/camera变化、冻结参数未变均有实际执行检查。正式训练调用新build，没有读M0 checkpoint。
- 每次best更新同时保存该轮真实距离；终评另存距离并对原best四指标执行1e-5个百分点严格核对。历史N1式重载失败不会通过重试择优消失。
- 这次审查没有实际F1 checkpoint、距离或神经输出，以上是实现覆盖的检查，不能写成已发生的通过结果。

## 队列、终态报告与封存边界

- 队列先准备六份初始化见证，再运行全部六M0；`run_phase(m0)`完成且逐端 `verify_m0`后才调用full阶段。full输出目录与M0目录不同，最多使用GPU0–3空闲卡，每卡一个活动worker。
- 复用的 `run_phase` 通过真实process返回码推进，不把预计时间当终态；内部 `POLL_SECONDS=240`。失败后不再安排pending任务，已运行任务正常收尾；不减B、不梯度累积、不重训、不换种子，没有科学分数提前停负端的逻辑。
- worker使用argv形式 `Popen`，train/evaluate按顺序执行；所有状态、PID、command和日志落盘。实际持久化由私有部署helper的 `start_new_session=True`、DEVNULL stdin及独立文件stdout实现，子进程继承这个脱离SSH会话的运行条件。
- `PREDECESSOR` 最终指向新登记的 `SEALED_SOURCE243.json`，不会依赖2025 archive中不存在的旧runtime manifest。原243条source字典保留，读取校验原240条不变；仅三个protocol允许 `dataset_root` 迁移至2025，且其余JSON语义必须逐值相同。原协议副本在 `.aris/compute/source2026_protocols`，原比较器仍须保留Git元数据给既有 `_configure_signal_source` 验证commit。
- 此处只核对既有SHA机制的真实调用，没有建议新hash方案。实际核对了23个作者文件、6个主复用文件与继承243 seal一致；三个原protocol还与历史clean summary的initializer protocol SHA一致。2025完整source_map运行仍是实际部署/M0前置检查，不能由本地副本核对替代。
- 所有六端严格终评通过后才生成accepted matrix并唯一调用CPU reporter；CPU环境将CUDA_VISIBLE_DEVICES清空，线程数1。report先核对12个stage返回0及完整六条件，复验50轮轨迹/step数，再对真实保存距离做完整成对诊断。`receipt_sha256` 已提供给复用 `compare()` 的实际要求。
- repaired/new Rank-1 errors、逐query AP、identity宏平均及固定模型identity bootstrap均由CPU `compare`计算；与训练seed显著性区分。历史shared-global仅从封存summary读入，核对公共视觉、camera和当前头初值后列差值，不重放历史神经评价。
- report的训练完成状态和 `report_exit_code` 分开保存；发布完整终态须同时有唯一report退出0和SUMMARY，不能只看controller训练状态COMPLETE。
- 新训练产物均写新campaign、新trained-model及新report目录。源清单和部署文件列表未包含修改或重跑已封存EV1/clean/其他旧研究的动作。

## 2025 私有部署helper审查

实际读取：`C:/Users/gb/.codex_tmp/deploy_foundation2025_20261002.py`，全文90行，未执行。

- 固定用户指定2025端点和 `/data2/gb/Re-ID/Trifusion`；读取既有known_hosts，未设置AutoAddPolicy，使用既有密钥文件认证。未读出或打印私钥内容。
- SSH `exec_command` 必经shell，但完整argv通过 `shlex.join` 引用；内嵌代码只使用固定本地路径/文件列表/期望摘要。远端训练启动再用argv形式Popen，不拼接训练shell命令。
- 部署先确认九个新增artifact路径不存在，再SFTP传输并逐字节校验；不覆盖封存源。launch对campaign/report/log使用新路径，只有一个controller启动点，没有隐藏observer/report waiter或重试启动点。
- 本地receipt输出目录实际已存在。部署helper审查是凭据/调用边界检查，不声称远端已部署、空卡或在线持久化已实测。

## 审查期间已修正项目

1. 主 agent 自查修正旧runtime manifest在2025 export中缺失的问题，改为已有243 source字典的新只读封存文件；已读取最终source_map并核对其语义。
2. 主 agent 自查把最初timm RandomErasing导入改为作者make_dataloader.py自身的实现；已重读最终loader与作者完整实现，不再存在这个原配方差异。
3. Reviewer指出原报告未直接记录计划要求的最终评价耗时。主 agent 加入 `final_evaluation_seconds`，计时边界明确为fresh construction、strict reload与full-gallery scoring，并写入SUMMARY每行。重新读取最终 `run_foundation_recipe.py:287–313` 和 `report_foundation_recipe.py:38–52`，修正成立。它不改变初始化、训练或选择逻辑。

## 实际阅读清单

项目根：`C:/Users/gb/.trifusion_github_publish_22c3bee`。作者根：`C:/Users/gb/.codex_tmp/clean_clip_audit_source721_20261002/comparators/Signal-cd1b0a6`。

全文阅读：

- `refine-logs/foundation_recipe_v1/EXPERIMENT_PLAN.md`。
- `tools/run_foundation_recipe.py`、`tools/queue_foundation_recipe.py`、`tools/report_foundation_recipe.py`；已读审查期间最终修改。
- `tools/run_clean_clip_joint.py`、`tools/run_visual_update_control.py`、`tools/run_correspondence_roles.py`、`tools/official_three_dataset_data.py`、`tools/queue_correspondence_refinement.py`。
- `tools/official_three_dataset_model.py`、`tools/analyze_correspondence_distances.py`、`modeling/trifusion/role_global_tokens.py`、`modeling/trifusion/criterion.py`。
- 作者 `modeling/make_model.py`、`modeling/meta_arch.py`、`layers/make_loss.py`、`layers/triplet_loss.py`、`layers/softmax_loss.py`、`solver/make_optimizer.py`、`solver/scheduler_factory.py`、`solver/scheduler.py`、`solver/cosine_lr.py`、`solver/lr_scheduler310.py`。
- 作者 `config/defaults.py`、三个 `configs/{RGBNT201,RGBNT100,MSVR310}/Signal.yml`。
- 作者 `data/datasets/make_dataloader.py`、`bases.py`、`sampler.py`及三个数据集parser。
- 上述私有2025部署helper。

所需调用链的定向阅读（没有把这些大文件声称为全文审完）：

- `tools/run_signal_preserving_v5.py:1–98,250–270,518–586,1618–1630`：实际复用seed/state摘要/batch/LR函数。
- `tools/build_v12_complete_path_oof_targets.py:244–261,362–384`：原模型构造与eval transform。
- `tools/run_signal_baseline_dev.py:1–66`：来源commit与import路径。
- `tools/train_rgbnt100_signal_oof.py:70–154,248–324`；`tools/train_msvr310_signal_oof.py:67–126,218–290`：实际loader/montage/scorer。
- `tools/train_signal_preserving_v18.py:1–64`、`tools/run_official_three_dataset_roles.py:1–80,228–244`：eval batch、protocol、距离。
- `modeling/trifusion/correspondence_roles.py:1–93,216–285`、`modeling/trifusion/aligned_data.py:1–97,227–258`：适配模块归属、冻结/当前头构造、共享几何。
- `tools/queue_correspondence_roles.py:1–100`：ROOT派生路径常量，旧baseline权重命令不是F1实际调用路径。
- 作者 `train.py:58–107`、`engine/processor.py:1–193`：loss汇合和scheduler调用顺序。
- 作者 `modeling/make_model_clipreid.py:1–100,159–197`、`modeling/clip/model.py:168–239,388–503,651–734`：公共CLIP装载、视觉前向、禁用prompt/adapter路径和位置编码。
- 作者 `utils/metrics.py:1–186`：实际两个完整scorer。
- JSON：`SEALED_SOURCE243.json`、三个完整原protocol（逐条检查全部记录）、封存clean summary（完整解析，检查三行global_only初始化/指标/schema）。

## 本次确定性检查与最终文件版本

- 三个F1 entry最终版本AST解析通过；没有导入训练模块执行。
- 三protocol所有记录parser/label/index/有效正例检查通过；实际counts、camera集合和完整图库如上。
- 三protocol原字节摘要同时匹配继承243 seal与历史global_only initializer；历史summary摘要与report常量完全一致。
- 23个实际检查的固定作者文件与继承seal匹配；6个主要复用文件与继承seal匹配。

| 最终审查文件 | Bytes | SHA256 |
|---|---:|---|
| `tools/run_foundation_recipe.py` | 18242 | `c18042509dc33ff66eee1a6b04a723ad55685bd5d9daf218c3779703ffe20758` |
| `tools/queue_foundation_recipe.py` | 12850 | `3afe051358aa25d99cb3900b4c6fdb13d5d79d9dcee76d01018fccd27e27d302` |
| `tools/report_foundation_recipe.py` | 6442 | `02aa7ef9f97feb87908639fbf616787b4cf56f1d7da228eb22755f482a7e072e` |
| `EXPERIMENT_PLAN.md` | 6562 | `9ba24f07c014b0238cb690b55f3883094f5c5953adeae18f32ef74a2dfffed87` |
| `SEALED_SOURCE243.json` | 30496 | `3834440f9f924d1cba869088469e06678b27c779994bb0232bbc930ec61ef329` |

封存clean summary SHA256：`784e3938a0203ea043543804c6e2933b43877405d7ca624116d59e59d35538fd`。继承EV1 manifest的已记录SHA256：`8555b297cdbe2a017fff3cea68430a26a135ed9afc0ffd0dfe1e9c7bed5d9842`；本审查没有重新取得原远端runtime manifest。

没有新F1检索结果；总体目标仍为ACTIVE_UNMET。下一项允许工作的证据是六端真实M0，而不是本源码审查的PASS文字。
