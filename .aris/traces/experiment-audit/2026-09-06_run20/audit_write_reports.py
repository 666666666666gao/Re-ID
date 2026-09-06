"""Emit this reviewer's reports and perform the final immutable-input hash audit."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")
T = Path(__file__).resolve().parent
ROOT = T.parents[3]
BASE = "EXPERIMENT_AUDIT_RGBNT100_GRAM_ENGINEERING"
load = lambda name: json.loads((T / name).read_text(encoding="utf-8-sig"))
manifest = load("input_manifest.json")
entries = manifest["entries"]
request = load("001-rgbnt100-gram-engineering.request.json")
replay = load("audit_replay_summary.json")
stage_replay = load("audit_stage_replay.json")
m0 = load("audit_M0_replay.json")
capture = load("audit_capture_replay.json")
gradients = load("audit_gradient_replay.json")
protocol = load("audit_protocol_replay.json")
sha = lambda b: hashlib.sha256(b).hexdigest()
texts = {i: Path(e["path"]).read_text(encoding="utf-8-sig") for i, e in enumerate(entries)}
evidence_used = []
def ev(i, start, end=None):
    end = start if end is None else end
    assert 0 < start <= end <= len(texts[i].splitlines())
    item = {"source_relative_path": entries[i]["source_relative_path"], "line_start": start, "line_end": end}
    evidence_used.append(item)
    return item
def cite(*items):
    return "；".join("`" + x["source_relative_path"] + ":" + str(x["line_start"]) + ("-" + str(x["line_end"]) if x["line_end"] != x["line_start"] else "") + "`" for x in items)

reviewer = {"requested_model": "gpt-6-astra", "requested_reasoning_effort": "max", "identity_basis": "API-requested identity from the frozen request; independently attested backend identity unavailable", "review_family": "GPT-family", "review_type": "Type-A advisory", "cross_family_review": False, "independent_artifact_review": True}
corrections = [
 {"id": "C1", "severity": "WARN", "required": True, "finding": "最新 tracker 和结果页将 R2 仍写成 READY_NOT_RUN，但冻结输入内已有 11:06:58.956344 的 wrapper 启动观察；没有 R2 T0/M0 的结果或终态。", "correction": "在最新状态处写 LAUNCH_OBSERVED_RESULTS_NOT_IN_SNAPSHOT，标出启动观察时间和快照时间；T0/M0 各自是否完成仍为 UNKNOWN_FROM_SNAPSHOT。保留 11:00 注册表的历史 READY_NOT_RUN，不将启动回执改写为工程通过。", "evidence": [ev(139,15,16), ev(140,89), ev(130,1,9), ev(131,15,35)]},
 {"id": "C2", "severity": "WARN", "required": True, "finding": "结果页称 configure/数据/提取/排名共 10 个函数或类 AST 相同；注册表明确列出 9 个，本审计逐一比较也为 9 个且全部相同。", "correction": "将结果页的 10 改成 9，或明确另外列出第 10 个对象并提供相应比较。当前 9 个为 configure、records_for、clean_triplet、read_montage、RGBNT100AlignedDataset、loader_for、extract、camera_scores、evaluate_gallery。", "evidence": [ev(140,87), ev(132,11,21)]},
 {"id": "C3", "severity": "INFO", "required": False, "finding": "M0 结果页“wrapper 三阶段退出均 0”的措辞容易混淆阶段数与退出文件数。原 wrapper 实际执行 T0、M0 两个子进程，另写自身 engineering_exit；checkpoint 文件核验另行发生。", "correction": "可将该句澄清为 T0、M0 和 wrapper 的三个退出状态文件均为 0；另列 09:44:59 的 checkpoint 文件核验。没有发现退出码冲突。", "evidence": [ev(141,4), ev(63,15,36), ev(123,2,3)]}
]

checklist = [
 {"id": "A", "verdict": "WARN", "title": "数据、参考来源、身份隔离与访问范围", "findings": [
 "全部 8,675 条文件名元数据、50 个身份、camera 0–7、view=-1、identity-camera 计数和三折标签映射独立重算通过。全部身份有跨 camera 正例；source 身份为 33/33/34，记录 5,550/5,725/6,075；heldout 身份 17/17/16，query=gallery 为 3,125/2,950/2,600。每条记录恰好在一个 fold 的 gallery/query 中、在两个 fold 的 source 中。每一条合法正例数及同 camera 不同身份负例保留规则均核对，不抽样。",
 "身份真值取自数据集文件名，source 分类标签是每折身份的连续重编号，camera 由文件名减 1。Gram 的 targets 则是同一 batch 的配对位置 0…63；其目标是模态实例对齐，不是新的身份真值。Patch MSE 比较模型产生的模态特征。身份监督与自监督对齐损失的来源不同。",
 "T0 实际覆盖整个官方训练目录，含各折后来作为内部 heldout 的图像；这是协议/切片核验，不能写成从未读取内部 heldout 图像。训练和 M0 的已保存采样索引全部在本折 source 内，heldout 模型前向为 0。官方测试和 RGBNT201 dev 未出现在这条执行路径的训练/评估调用中；零访问结论来自代码、记录与回执，没有系统级访问审计。",
 "参考来源仍有边界：清单引用的原始 inventory 和 protocol builder 不在本次不可变输入中。协议结构和全部记录可独立重算，真实图像字节、完整来源链和预训练 checkpoint 内容只能依据远端 T0/文件回执；本审计没有读取它们。"
 ], "evidence": [ev(136,5,24), ev(26,63,84), ev(149,69,123), ev(21,56,104), ev(152,33,75), ev(76,10,45), ev(138,11,14), ev(19,62,70)]},
 {"id": "B", "verdict": "PASS", "title": "损失、数值定义、归一化与指标合法性（限已保存证据）", "findings": [
 "执行分支为 sign=3、together_CLS_Patch：四个 ID/Triplet 分量相加，再依次加 0.1×Gram 和 0.1×Patch。每个身份分量调用作者的 label-smoothed ID CE 与 Triplet 加权组合，ID 权重 0.25、Triplet 权重 1。全部 114 个包含重复嵌入副本的保存损失对象，在按代码顺序模拟 float32 加法/乘法舍入后与记录 loss 精确相同；独立训练尝试为 M0 24 次和捕获 34 次，共 58 次。不能凭这些标量重新计算缺失的 logits/feature 上的 CE、Triplet、Gram 矩阵或真实梯度。",
 "两项 camera 排名人工 fixture 的合法正例位置分别为 [2] 和 [1,3]；AP 为 0.5 和 0.8333333333333333，后一 expected 5/6 的浮点表示为 0.8333333333333334。它们是合成协议单元检查，没有构成 RGBNT100 检索成绩。clean 输入的 [0.5]*3 归一化、Gram 特征的单位范数和预定检索特征 L2 归一化是输入/表示变换，没有发现拿模型输出当身份真值或用自除分数制造成功指标。",
 "原 volume 实现是 3×3 Gram 的 sqrt(abs(det(G.float())))；其 docstring 误写 4×4/字幕输入，不应按注释误读已执行运算。FP16 normalized 输入虽为 float32，保存记录中的 Gram 是 float16，后续转 float32 求 det 无法恢复此前舍入。原 AMP 每次 64×64 中 3 个零、1 个负 determinant；异常栈定位到该复合表达式中的 AbsBackward0 返回 NaN，不等于独立证明 torch.det 原语有缺陷。",
 "全模型 FP32 对照使上游特征也变化：loss 6.645667552947998，零 determinant 为 0、全部 195 份梯度有限。它不能单独证明局部 Gram FP32 充分。固定已保存真实输入的局部 FP32 算子仍有一个零 determinant，三个 64×512 输入各有 512 个 NaN，保持失败。",
 "稳定 helper 明确改为 FP32 Gram 加 sqrt(abs(det).clamp_min(1e-12))。对 |det|<1e-12，输出约 1e-6 且对 determinant 的导数为 0；对 |det|>1e-12，导数为 sign(det)/(2 sqrt(|det|))；阈值点不作一般可微性声明。原 sqrt(abs(det)) 在零点的复合导数无正常有限定义，零点无限 sqrt 反传与 abs 的零点处理相乘可产生 NaN，这与异常栈和算子对照一致；不是本地原语重放证据。floor 是实质数值定义变化，不能称原训练函数完全等价。作者 Triplet 的 1e-12 是另一种量（平方距离）的保护值，沿用该常数本身不能证明所有 batch 稳定。"
 ], "evidence": [ev(142,201,241), ev(7,37,56), ev(138,32,35), ev(152,76,93), ev(149,81,87), ev(24,14,60), ev(29,21,75), ev(33,18,52), ev(34,9,70), ev(82,87,139), ev(148,5,23), ev(20,24,30)]},
 {"id": "C", "verdict": "WARN", "title": "文件、哈希、执行修订、阶段状态与数字一致性", "findings": [
 "153 个输入文件共 5,345,509 字节，初始及最终 SHA-256/字节数均与不可变 manifest 完全一致。manifest SHA 为 b83e0121166effe4e6e84940ade7ab593610846f93634f9bca1b195a9c837e69。55 个可取得的配置源码绑定全部精确匹配；两条缺失绑定是同一 protocol builder 在 R1/R2 的重复引用。89 个可取得 intake 文件引用匹配；另 3 个 checkpoint 二进制不在输入中，未假装本地核验。",
 "历史 runner 必须按 R1 archive SHA 514a2c86634b61ef8b6de32b6ff1990cdf18e208a91673fb192534b52e5a1357 解析；当前 R2 runner SHA 为 4677e7345f282646e6654000d9d526d8207f698fc58a1f90b0c44bda6dd5fc25。不能用 R2 字节去判 R1 回执失配。R1 有 18 个 Signal 和 8 个项目选择绑定；R2 为 21 和 10。可取得文本绑定全部核对，Git commit/diff 值是执行器记录及源码断言的来源声明，本地没有远端完整 Git 工作区可独立证明其全部状态。",
 "T0/M0 的逐折 receipt、training 和 summary 嵌入副本精确一致；所有 24 步均值、采样、LR、585 个参数组及全部可取得文件回执数字一致。M0 双精度重组最大差为 2.1062791333292807e-6。捕获的双精度“先相加再减 loss”误差为 1.1775642629885397e-6，原次级报告 1.1775642632990552e-6 可由“从 loss 逐项相减”精确重现，属于表达式括号差异，不是模型损失冲突；原式 float32 则全部精确。",
 "必须更正 C1、C2 两处最新文档。注册表的历史 READY_NOT_RUN 可保留，最新状态必须反映 R2 wrapper 已被观察到启动，但快照没有其 T0/M0 结果。R1 标量 verifier 的 .00007 与 .1*.0007 差异已由保存的两个版本确认，实际更改只有两处字面表达式，未修改实验门或重跑模型。",
 "各失败均有真实终态，未被后续通过覆盖：原 B0 exit=1、捕获 exit=1、原 FP16 probe exit=0 但数值失败、anomaly exit=1、全模型 FP32 exit=0、局部 Gram FP32 exit=1、stable floor exit=0。程序退出成功和梯度通过是不同层次。"
 ], "evidence": [ev(19,62,80), ev(18,64,86), ev(132,3,25), ev(123,25,126), ev(126,5,88), ev(105,3,14), ev(127,51,63), ev(128,51,63), ev(55,2,4), ev(48,2,4), ev(38,1,58), ev(84,18,23), ev(94,18,23)]},
 {"id": "D", "verdict": "PASS", "title": "真实调用路径、停止条件、参数/梯度与保存语义", "findings": [
 "本路径从自定义 RGBNT100 runner 调用 source 配置、模型构造、四头 loss、seed/state-hash helper。build_v12_complete_path_oof_targets 仅复用 _build_signal_teacher 和 _signal_training_loss；run_signal_baseline_dev 仅复用 source 配置；run_signal_preserving_v5 仅复用 seed 和 module hash。那些文件中完整的 V5–V12、oracle、dev 选择、专家或其他数据集训练代码没有在本路径执行。上游 train.py/processor.py 的 standalone runtime 也不是此 runner 的执行入口。",
 "M0 是每折 8 步的部分首 epoch。R1 在训练分量 finite 检查、backward、unscale、scaler.step/update 后检查 AMP scale，成功才将行加入内存；完整 fold 返回后才写 training/checkpoint。原 B0 在该断言失败，保存完整 epoch 为 0，成功更新数不可恢复；不能填写成 0 或后续诊断的 33。诊断在原 R1 第 164/175/201 行做边界、前向前状态与 AMP 停止前记录，34 行 JSONL 保留失败步；其初态和采样与 M0 对齐，但只有第 1 步全标量相同，不能以固定 seed 宣称轨迹逐位复现。",
 "捕获与原 FP16 probe 的全部 195 份梯度计数逐项一致，覆盖 89,957,377 个元素，其中 153 份含 NaN、合计 86,141,184 个 NaN 元素，Inf 为 0，非有限仅在 CLIP encoder。全 FP32 和 stable full-batch 的 195 份梯度覆盖相同参数元素数且均为有限。梯度张量计数不是“153 个 NaN 元素”。",
 "TokenSelection 的 Q/K 仅经 top-k 整数位置生成二值 mask，W_v 定义但未调用；冻结其六个参数不移除前向选择。冻结 selector 共 787,968 个参数，加四个 BN bias 的 3,072 个参数，解释 total-trainable=791,040。三折总参数为 90,748,417/90,748,417/90,751,489，可训练为 89,957,377/89,957,377/89,960,449；第三折额外一个分类 ID 增加 3,072 个分类权重。center criterion/其优化器虽被构造，但此损失分支不使用 center loss，也无 center optimizer step。",
 "局部 FP32 回归在第 102 行 finite gate 失败，模型构造在第 108 行之后，完整模型段未执行。stable 回归有 2 次算子 backward、2×64 条推理比较加 64 条训练前向、1 次完整模型 backward、0 optimizer update。3072D 相等是固定权重下保存 batch 的输出 parity。上游推理路径仍执行 AlignM 后丢弃其损失，因此不能把相等扩大成“推理时完全不调用 Gram”、无额外开销或未来训练权重一致。",
 "R2 只在 new_model 绑定 stable helper、把 source 步骤 JSONL 放在实际观察到的 AMP stop 之前，并将 M0 改成完整首 epoch。loss finite 和 backward 仍在 JSONL 之前，故保存保证应限定为经过该写点的尝试，尤其本次已观测的有限 loss/梯度溢出；无需为没有观察到的新失败增加防御分支。R2 尚无结果回执，源码门存在并不等于已通过。"
 ], "evidence": [ev(149,31,44), ev(149,126,138), ev(149,308,310), ev(142,201,258), ev(17,163,227), ev(143,111,142), ev(42,17,34), ev(44,1,34), ev(30,84,108), ev(90,1,19), ev(8,49,84), ev(13,4,8), ev(13,43,45), ev(150,98,108), ev(151,102,175), ev(9,214,228), ev(149,194,217)]},
 {"id": "E", "verdict": "WARN", "title": "范围、不确定性、可复现性与证据限度", "findings": [
 "本审计为本地只读文本/JSON/AST/stdlib 重放。共解析 67 个 JSON 文件、34 行 JSONL 和 54 个 Python AST，遍历 130,938 个数值叶节点；全部 114 个 loss 对象、804 行梯度计数及 585 个 optimizer group 均纳入，没有挑选通过样本。最终共 994 项显式检查通过。scripts/commands/stdout/output JSON 保存于 run20；首次辅助脚本 schema 错误、intake 字段名修正和双精度括号核对的失败输出也保留。",
 "未导入 torch、未装载任何 checkpoint/Gram/feature/gradient tensor、未解码图像、未训练、未运行模型或真实检索、未使用 SSH、未读 home memory。远端保存 fixture 的 SHA、bytes=438,600,256、三个 M0 checkpoint 内容和 bitwise feature parity 是可核对代码/一致回执，仍不是本地独立重算。CUDA primitive backward 实现及完整传递依赖源码未收齐；对 CE/Triplet 等叶子算子的张量数值不能额外认证。",
 "原始 inventory 与 builder 缺失；选定源码哈希绑定不等于完整依赖封闭包。比如 source make_loss 引入 softmax_loss/center_loss、模型引入 CLIP/DAS 依赖，输入没有全部这些文件。若描述本次审计，必须使用“所列可取得源码与回执一致”，不能写“完整实际源码及所有运行数据已独立复现”。本审计不要求为当前有限工程结论补写兼容层、fallback 或额外训练。",
 "作者配置 B128/K16/workers12，本项目 B64/K8/workers4，且共享几何增强；这些差异已登记，不能称完全复现作者训练条件。基础 LR=5e-6 指原参数组 initial_lr，epoch1 实际 LR 已受 warmup/noise 改变；本地仅重放保存值的调度方程，未运行 torch RNG。全 FP32 对照峰值显存 19,953.01025390625 MiB，FP16 为 10,700.3798828125 MiB，也不是同精度条件的效用比较。",
 "9 个 R1/R2 数据与提取/排名 AST 对象相同只能证明这些定义相同，不能把不同训练数值定义的最终学习轨迹视为等价。已有 M0/capture 标量轨迹差异保持未解释；stable 仅一个保存失败 batch。快照 11:09:14.614294 中 R2 只有启动观察，完成整个首 epoch、三折、30 epoch、长期稳定性及检索性能均未得到证明。"
 ], "evidence": [ev(138,11,14), ev(19,62,70), ev(7,7,10), ev(42,1514,1524), ev(123,102,135), ev(137,14,16), ev(138,47,50), ev(141,22,31), ev(132,11,31), ev(130,1,9), ev(92,169,175), ev(134,3,14)]},
 {"id": "F", "verdict": "PASS", "title": "评估分类及可支持的工程/科学结论", "findings": [
 "数据层面：内部身份 OOF 数据协议和合法 query/gallery 定义结构通过；远端 T0 对全量训练拼图的回执一致。数值工程层面：R1 短 M0 通过、正式 B0 溢出停止、独立捕获与原精度复现通过定位、局部 FP32 失败、固定 floor 的单 batch 算子和模型回归通过。各层证据不同。",
 "检索层面：没有任何完成的 RGBNT100 真实 mAP/Rank 结果。T0 两个 AP 是合成 fixture；stable 的 64×3072 bitwise equality 是特征输出检查；source loss、有效更新、参数/梯度计数均不是检索收益。未来计划的 8,675 query/gallery 是官方训练集内部三折 OOF，不能称官方测试。",
 "此证据支持固定保存失败 batch 的显式数值修复及 R2 工程准备/启动观察，不能支持 R2 完整 M0 已通过、固定 30 epoch 训练完成、长期数值稳定性、TriFusion 方法收益、消融收益或官方目标达成。未发现将工程失败冒充科学负结果、偷偷重跑已有科学失败版本或用局部通过代替真实检索指标的已执行证据。完整终态 verifier 仅准备，源码不在本次清单且没有执行结果。"
 ], "evidence": [ev(76,48,65), ev(71,2,10), ev(108,4,17), ev(82,87,139), ev(92,153,175), ev(136,9,16), ev(139,19,21), ev(140,79,89), ev(134,3,14)]}
]

hash_rows = []
for e in entries:
    b = Path(e["path"]).read_bytes()
    hash_rows.append({"source_relative_path": e["source_relative_path"], "bytes": len(b), "sha256": sha(b), "size_match": len(b) == e["bytes"], "hash_match": sha(b) == e["sha256"]})
hash_check = {"checked_at": datetime.now().astimezone().isoformat(), "manifest_sha256": sha((T / "input_manifest.json").read_bytes()), "files": len(hash_rows), "bytes": sum(r["bytes"] for r in hash_rows), "all_hashes_and_sizes_match": all(r["size_match"] and r["hash_match"] for r in hash_rows), "entries": hash_rows}
assert hash_check["manifest_sha256"] == request["input_manifest_sha256"]
assert hash_check["all_hashes_and_sizes_match"]
assert replay["failed_checks"] == [] and stage_replay["all_checks_pass"]
(T / "audit_final_hash_recheck.json").write_text(json.dumps(hash_check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
now = datetime.now().astimezone().isoformat()
report = {"schema": "independent-experiment-integrity-audit-v1", "created_at": now, "input_snapshot_at": request["created_at"], "overall_verdict": "WARN", "summary": "完整可取得标量和身份协议一致；两处最新文档需更正。稳定化仅单 batch 工程通过，R2 只有启动观察，无完成的真实检索结果。", "reviewer": reviewer, "input_manifest": {"path": str(T / "input_manifest.json"), "sha256": hash_check["manifest_sha256"], "files": len(entries), "bytes": hash_check["bytes"]}, "checklist": checklist, "required_corrections": [c for c in corrections if c["required"]], "optional_clarifications": [c for c in corrections if not c["required"]], "stage_ledger": stage_replay["stage_ledger"], "replay": {**replay, "stage_checks": len(stage_replay["checks"]), "total_explicit_checks_passed": replay["passed_checks"] + len(stage_replay["checks"]), "stage_checks_pass": True, "complete_numeric_outputs": ["audit_protocol_replay.json", "audit_loss_replay.json", "audit_M0_replay.json", "audit_capture_replay.json", "audit_gradient_replay.json", "audit_numerical_chain_replay.json", "audit_source_bindings.json", "audit_intake_hash_replay.json", "audit_AST_comparison.json", "audit_stage_replay.json"], "trace_directory": str(T)}, "final_input_hash_recheck": {k: v for k, v in hash_check.items() if k != "entries"}, "prohibited_operations_performed": [], "scientific_retrieval_result_available": False, "R2_completion_available_in_snapshot": False}

md = ["# RGBNT100 Signal / Gram 工程链独立完整性审计", "", "**总体结论：WARN。** 可取得的完整身份协议、保存标量、梯度计数、参数组和历史文件绑定一致；R2 最新状态与 AST 对象数量有两处应更正。原正式运行的工程失败、局部 FP32 方案失败及后续单 batch 稳定化通过均被保留。当前证据没有 RGBNT100 真实检索结果，也没有 R2 T0/M0 完成回执。", "", f"输入快照：{request['created_at']}。审计生成：{now}。本结论仅针对这份不可变快照，不推断其后运行状态。", "", "审阅者身份记录：API 请求为 **gpt-6-astra / max**；**GPT-family Type-A advisory**。这是请求身份，不是独立验证的后端模型身份，也不是跨模型家族审查。审阅依据为逐文件原始输入与独立重放，未将执行器总结当作判定依据。", "", "## 实际完成范围", "", "| 层次 | 判定 | 可支持范围 |", "|---|---|---|", "| 数据 | 结构 PASS；物理内容为远端回执 | 全 8,675 记录、三折身份/标签/正例/负例规则逐项重算 |", "| 数值工程 | 各阶段分别判定 | 原 M0 通过；B0 停止；捕获 34 次/33 更新；局部 FP32 失败；floor 单 batch 通过 |", "| R2 | 启动观察 | 11:06:58 观察到 wrapper 84049；T0/M0 各自终态不在快照 |", "| 检索/科学结论 | 尚无结果 | 合成 AP、loss、gradient finite 与特征 parity 均不构成真实 mAP/Rank 收益 |", "", "## 阶段与成本核对", "", "以下秒数是原记录的 perf_counter 时长；wrapper、子阶段、之后的文件核验不可当成互斥训练成本相加。墙钟时间另在 audit_stage_replay.json 做算术比较。", "", "| 阶段 / 执行 commit | 实际状态 | 已保存成本 / 范围 |", "|---|---|---|", "| R1 T0 / 1157f0d | PASS 数据协议 | 20.701875953 s；17,350 解码；0 模型/优化/真实检索 |", "| R1 M0 / 1157f0d | PASS_ENGINEERING_ONLY | 76.987059359 s；3×8 更新，1,536 训练记录前向 + 48 clean source parity 前向 |", "| M0 文件核验 / 1157f0d | 回执 PASS | 5.368404388 s；3 checkpoint 内容核验回执；本地未加载 tensor |", "| 原 B0 / 60a3d0e | exit 1：AMP stop | 33.776537912 s；0 完整 epoch；成功更新/前向数 UNKNOWN_NOT_PERSISTED |", "| source 捕获 / 8b412d0 | exit 1：第 34 次 AMP stop | 37.470637565 s wrapper；34 尝试、33 更新、2,176 训练记录前向；独立轨迹 |", "| 三模式保存 batch / 56f094f | fp16 数值失败；anomaly exit 1；full FP32 finite | 48.688285528 s wrapper；3×64 前向、0 更新 |", "| 局部 Gram FP32 / 8034451 | FAIL_OPERATOR_FINITE_GATE | 7.948580876 s wrapper；2 算子 backward、0 模型/更新 |", "| stable Gram floor / 6c741b8 | PASS 单个真实保存 batch | 16.412545100 s wrapper；2 算子 backward + 1 模型 backward，192 记录前向、0 更新 |", "| R2 T0→M0 / e699eac | LAUNCH_OBSERVED_RESULTS_NOT_IN_SNAPSHOT | 仅 11:06:58.956344 启动观察；不填写未知完成数 |", "| 完整终态核验器 | PREPARED_NOT_EXECUTED | source 本体不在输入；不是已完成的 90 epoch/8,675 query 核验 |", "", "阶段引用见下列 A–F 项与结构化 JSON 的 stage_ledger；完整状态由日志、脚本分支、退出文件和回执交叉判定。", "", "## A–F 检查表", "", "| 项 | 判定 |", "|---|---|"]
md.extend(f"| {c['id']} {c['title']} | {c['verdict']} |" for c in checklist)
for c in checklist:
    md.extend(["", f"### {c['id']}. {c['title']} — {c['verdict']}", ""])
    for paragraph in c["findings"]:
        md.extend([paragraph, ""])
    md.append("原始文件依据：" + cite(*c["evidence"]) + "。")

md.extend(["", "## 可重放数值细节", "", "| M0 fold | 更新 | 记录前向 / 唯一记录 | 覆盖 source ID | 同 ID 对 / 跨 camera 对 | loss 均值 |", "|---|---:|---:|---:|---:|---:|"])
for f in m0["folds"]:
    md.append(f"| {f['fold']} | {len(f['steps'])} | {f['records_forwarded']} / {f['unique_records']} | {f['unique_identities']} | {f['same_identity_pairs']} / {f['cross_camera_pairs']} | {f['mean_loss']!r} |")
md.extend(["", "每个 batch 都是 8 个身份×8 条记录。三个 fold 的 optimizer 初始 LR 为 5e-6、0.0007、0.0014；warmup 构造后的组 LR 均为 0.1×0.0007。epoch1 的实际三值为 2.0041138350963593e-5、6.891338801383971e-5、0.00011813723659515383；由保存值反推共同 noise=-0.6484010815620422、倍率 0.35159891843795776，按作者 v+v×noise 公式可精确重建。本地没有重放 torch 随机数。", "", "捕获与 M0 的前八步差异完整保留：", "", "| step | 全记录标量精确相同 | 采样索引精确相同 | 捕获 loss − M0 loss |", "|---|---|---|---:|"])
for r in capture["first8_comparison"]:
    md.append(f"| {r['step']} | {str(r['all_recorded_values_exactly_match_m0'])} | {str(r['sample_indices_exact'])} | {r['loss_difference']!r} |")
md.extend(["", "稳定化模型 loss 从 6.734400272369385 变为 6.7345452308654785，差 0.00014495849609375；Gram 从 3.970210552215576 变为 3.9716601371765137，差 0.0014495849609375，其 0.1 加权差与总差在本记录中相等。四个身份分量与 Patch 精确保持，真实 raw-zero 仍为 1，并未把 zero determinant 计数改成 0。最低 volume 的 float32 值为 9.999999974752427e-7。对应完整 gradient-count JSON 的全部 195 行都被纳入。", "", "这些是对保存数值、运算次序和代码语义的独立重放；没有加载远端 Gram 数组来重新求 determinant、算 CE 或求导。", "", "## 具体更正", ""])
for correction in corrections:
    md.extend([f"**{correction['id']} — {'必须更正' if correction['required'] else '可澄清措辞'}：** {correction['finding']}", "", correction["correction"], "", "依据：" + cite(*correction["evidence"]) + "。", ""])
md.extend(["无需因这两项文档问题重跑已通过 M0、扫描 floor/LR/seed、重训已有科学失败版本或补入 fallback。若后续要声称 R2 完整工程通过，应以该修订真实的完整 T0/M0 步骤、回执及终态另行判定；当前快照不具备这项结论。", "", "## 重放与最终哈希记录", "", "- 所有输入 bytes/text 均读取，全部 Python 源 AST 解析；长 JSON 逐记录遍历/重算。`audit_inventory.json` 保存 153 文件的逐项来源、字节、行数和初始 SHA。", "- `audit_read.py` 和 `audit_read_*.stdout.txt` / `audit_compact_*.stdout.txt` / `audit_diff_*.stdout.txt` 保存带原始行号的审查材料；`audit_commands.txt` 为辅助脚本参数日志，首批记录的解释见 `audit_operational_notes.md`。", "- `audit_replay.py` → `audit_replay.stdout.txt` / `audit_replay_summary.json` / 全部逐项 replay JSON：973 项显式检查通过。首次 intake schema 适配错误的脚本和 stdout 另存。", "- `audit_stage_replay.py` → `audit_stage_replay.stdout.txt` / `audit_stage_replay.json`：21 项附加阶段、次级数字及日志一致性核对通过；双精度括号调查前后的脚本/输出都保留。", "- `audit_final_hash_recheck.json`：最终 153/153 SHA 和字节数匹配，manifest SHA 同上。`audit_write_reports.py` 构造本报告，校验每条引用的行范围存在。", "", f"审计输出目录：`{T.as_posix()}`。本地禁止操作计数均为 0；没有修改任何输入或实验配置。"])

report_path = ROOT / (BASE + ".json")
markdown_path = ROOT / (BASE + ".md")
report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
markdown_path.write_text("\n".join(md) + "\n", encoding="utf-8")
(T / "audit_report_evidence_index.json").write_text(json.dumps(evidence_used, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
assert json.loads(report_path.read_text(encoding="utf-8"))["overall_verdict"] == "WARN"
assert all(sha(Path(e["path"]).read_bytes()) == e["sha256"] for e in entries)
out = {"markdown": str(markdown_path), "json": str(report_path), "verdict": "WARN", "required_corrections": len(report["required_corrections"]), "checks_passed": report["replay"]["total_explicit_checks_passed"], "hash_recheck": hash_check["all_hashes_and_sizes_match"], "markdown_sha256": sha(markdown_path.read_bytes()), "json_sha256": sha(report_path.read_bytes())}
(T / "audit_report_write.stdout.txt").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, indent=2))
