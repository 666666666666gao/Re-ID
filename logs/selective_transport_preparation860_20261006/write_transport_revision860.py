from pathlib import Path
from datetime import datetime
import json, hashlib

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
folder = root/'refine-logs/selective_role_transport_v1'
original = (folder/'round-0-initial-proposal.md').read_text(encoding='utf-8')
anchor = (folder/'PROBLEM_ANCHOR.md').read_text(encoding='utf-8').rstrip()
review = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selective_transport_method_review860')
trace = Path('C:/Users/gb/.aris/traces/research-refine/2026-10-05_run860')
trace.mkdir(parents=True, exist_ok=True)
for name in ('REVIEW_ROUND1.json', 'round-1-review.md'):
    (trace/name).write_bytes((review/name).read_bytes())
(trace/'round-0-initial-proposal.md').write_bytes((folder/'round-0-initial-proposal.md').read_bytes())
(trace/'PROBLEM_ANCHOR.md').write_bytes((folder/'PROBLEM_ANCHOR.md').read_bytes())
(trace/'PROVENANCE.json').write_text(json.dumps({
    'requested_reviewer':'gpt-6-astra/max', 'actual_backend_verified':False,
    'review_independence':'same-family', 'acceptance_status':'provisional',
    'calibration':'none', 'max_rounds':2, 'agent':'/root/review_selective_transport860',
    'round1_original_prompt_verbatim_saved':False,
    'boundary':'Original prompt was not persisted verbatim before compaction; preserve actual input files and actual full reviewer output, do not reconstruct and label a fake verbatim prompt. Round2 request is saved verbatim before followup.'
},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

text = '# 按槽位分配跨光谱传输质量：完整修订候选，尚未实现或训练\n\n## Problem Anchor\n\n'+anchor+'\n\n'+'''## 当前证据与唯一干预

§859六端全部50轮、首次strict与15配对完成。patch−mean mAP在201/100/MSVR分别−0.338730/−0.059234/+0.000801，0/15项目门；不重开候选重建。源码Transformer按B×3独立运行，旧Mamba把16×3槽位交错为48-token序列，未显式分配跨模态传输权限。新的问题是：按接收槽位分配跨光谱消息质量，是否比同一模态对内统一衰减更有检索效用。

原RAWsemantic作为底座，不带原生CNN、容量MLP、候选重建、文本或新辅助loss。只保留一条候选及其直接活动控制。额外DINO/文本属于不同资源路线，当前没有匹配初始化及训练对照，不同时加入。总体三数据集与资源合格强参照/SOTA目标保持，不把本轮窄机制检查当最终Goal。

## 对第一轮评审的处理

R1为REVISE 6.55/10，CALIBRATION:none、same-family/provisional，实际后端未独立核实。原partial与forced在同一参数状态满足P=diag(r)Q，主要差别是消息质量衰减，因此不能证明局部选择或可靠对应。

本稿以mean(r)·Q替换forced，仍只有六个训练端；删除与出口线性等价的message_value，只保留两矩阵；将“区域”限定为学习槽位，不宣称真实部件/几何对应；写明固定null只是简化先验。固定null并非因为forced中学习标量必然闲置，dustbin仍可能经列缩放影响条件Q。下面是完整修订，不追认原R1通过。

## 精确数据流与参数

现有role sampler输出C0,T0,M0均为B×3×16×128。CNN、Transformer及其直接读出照旧。Mamba输入仍是M0+anchor+旧Transformer→Mamba桥接，按B*3×16×128排列，经原mamba_norm、同一原Mamba正反扫描取均值，恢复私有序列H=B×3×16×128。单次SSM不跨光谱传播；global上下文仍来自三模态，不能称整个系统模态独立。

新增Transport仅matching_projection与message_output两个bias-free128×128矩阵，共32768参数/2张量；前者普通初始化，后者是唯一零出口。无新MLP、FFN、持久分类头、归一化参数或辅助损失。两个臂的所有参数实际执行计算，累计8步活动按现有M0合同检查。

Transformer输出T经functional LayerNorm和共享matching_projection得到Z。每对m<n计算Cmn=Zm Zn^T/sqrt(128)。只为三个无向模态对求分配，反向使用同一log分配转置，不增加方向预测器。此处slot sampler未使用positions；所有索引是学习槽位，不是物理空间真值。

## 软分配与直接控制

分数16×16扩为17×17，末行、末列、角固定null logit=1。采用FP32 log空间Sinkhorn固定100迭代。真实行/列目标质量1，null目标16，总质量32；归一化log边际分别真实−log32、null log16−log32，输出logP加log32。两臂相同算法、迭代数、参数与初始state，无硬阈值/测试统计/几何标签。数学来自SuperGlue部分分配，未调用其预训练/GNN/原几何监督，不能称完整复现。

对接收m、来源n，令Lmn为真实16×16 log块（反向由转置取得），Qmn=softmax(Lmn, dim=-1)，r_mn=exp(logsumexp(Lmn,dim=-1))。

候选slot_mass：Wmn=exp(Lmn)=diag(r_mn)Qmn，保留每槽位分配的真实质量。

控制uniform_mass：Wmn=mean_k(r_mn[k])·Qmn，保留同一模态对平均真实质量，仅去掉接收槽位之间质量差异。反向必须重新计算转置块的行质量与均值，不能直接转置最终W。

同一个P下，两臂总真实矩阵质量相同，条件Q相同；差别是局部质量分配。它不严格匹配消息向量范数：权重与不同H相乘后仍有方向/抵消/范数差。两臂独立训练后参数和P可不同，不能声称最终实测质量或能量始终相同。该控制隔离slot级许可相对样本/模态对全局衰减，不证明P是正确几何或语义对应。

Jm=0.5·sum_{n≠m} Wmn Hn，M=原output_norms[2](Hm+message_output(Jm))。value不再另投影；原CNN/T直接旁路、structured readout1536、原learnable gain、h=g+gain*c和f=Normalize(h)不变。私有只指H自身旁路不经过peer权重，不宣称可辨识语义解耦。

## 初始化、训练、推理

新两臂完整state相同，初始化raw/L2/global/head和BN行为逐项匹配。旧视觉、共享adapter、camera、作者head、原role参数起点按登记公共初始化一致。3×16扫描与旧48扫描不同，因此不要求新臂初始化fused等于旧RAWsemantic；零出口只消除新peer消息，不保证等价旧RAW48或训练性能保底。

保持RAW职责分离：global作者raw ID/Triplet更新共享路径及正常头；fused使用sg(g)及角色输入detach，角色通过当前作者head的detached参数/clone BN buffer获得原主任务梯度，没有第二套持久训练head。seed42、fresh50、201 B64/K8、100 B128/K16、MSVR B64/K4、作者采样/增强/优化器与日程不变。原FP32/AMP边界、前6视觉块GPU1/后6与head GPU0保持；不做旧parity/kernel修复，不缩batch，不增加loss或按成绩救LR/gain/margin/seed。

推理仅在一个对象三光谱内部计算，模型/BN统计固定，一次1536维L2向量可离线建库。不逐query-gallery联合匹配，不用测试标签，不更新图库统计。

## 数学与活动检查

CPU组件在预先构造的有限分数（明确同槽高分、离对角低分，以及全低分无对应条件）验证：FP32有限输出；真实边际与null边际残差≤1e-3；行真实质量≤1+1e-3；同P两臂总真实质量相等至atol/rtol1e-5；转置方向与接收方行质量正确；全低分条件确实减少真实质量而不是重新归一化为1。100次迭代不因结果调整。若检查失败，记录实现或合同错误，不以当前输出重新定义容差。

8次toy更新验证两矩阵累计非零有限梯度/实际参数变化、零出口初始化输出为0、组件strict state重载一致。两矩阵都零初始化会切断路径，本稿只零出口。完整真实8M0使用原作者batch，按现有逐端合同检查优化器恰好一次覆盖、梯度/更新/BN、初始新两臂输出和重载。每个失败端封存，不延长M0、缩batch或放宽旧门。

每步记录r的mean/min/max、条件Q行熵、接收列质量集中度、peer/self及最终修正/global范数；全部训练曲线保留。只有身份loss时，这些是latent task gate统计，不是校准可靠性或部件正确率。全部拒绝、统一质量、slot内容重复均可能发生；不预先加额外loss救，也不根据官方成绩改null/迭代/倍率。

## 唯一训练与评价块

三数据集×slot_mass/uniform_mass共六端，分别匹配初始化→真实8M0→fresh50→首次strict→一次保存距离报告。原sealed RAWsemantic与独立global六个匹配控制直接复用，不重训。原RAW187依赖及354执行源保持。

主要两个配对每数据集：slot_mass−uniform_mass、slot_mass−原RAWsemantic，均mAP≥+0.5且R1不下降；同时完整展示两臂对独立global和RAWsemantic的结果、全部合法query/identity、首位修复/新增错误、50轮行为和真实成本。15配对整表一次报告，不跨epoch拼CMC。项目门不等于显著性或完整流程种子证据。

slot_mass−uniform_mass是局部质量分配的窄检查；与旧RAWsemantic比较还包含48→3×16和32768额外参数，不能全部归因局部门控。若主结果成立，另登记self-only、普通cross-attention等必要结构消融与完整流程多种子；不在本轮添加第三臂或所有论文消融。

若两臂均不超过RAWsemantic/global，传播重写不晋级；若uniform相当或更好，不主张局部质量分配必要；单集正不推广三集。若差别只来自全局衰减，不包装成可靠对应贡献。已消费官方集仍用于mAP-best与研发选择，不称未触碰测试。

## 研究价值与实际存储

候选主张是“根据学习槽位分配跨光谱交流质量，比同一模态对统一衰减更能保留最终检索判别性”。不是首次Sinkhorn/共享私有/三个算子，也不是可靠语义对应已获证明。它必须有三集净增益和后续必要性证据才有论文价值；若不能形成，不强行凑第二模块。

历史六端NN+逐轮评价约6小时，新增100次17×17分配需实测；估计6.5–8小时，用持久远端队列及预计端完成前/180–300秒观察。存储预算保持7*384MiB+2GiB=4,966,055,936B，不降低余量。已核实清理7个闭合冗余best释放2,482,212,968B，free后4,775,702,528B，仍略不足；原354执行源/187控制SHA均前后相同。只再资格核查和清理已闭合无用自训权重，作者/公共CLIP/保留赢家与现控制不动。未达固定存储门不启动队列。

该稿无新实现/M0/训练成绩。方法复核最多2轮，不为评分堆组件；即使接口可实施也不预写论文READY。总体Goal仍ACTIVE/UNMET，需要完整三数据集、强参照、多种子与资源合格SOTA结果。
'''
output = folder/'round-2-revised-proposal.md'
assert not output.exists()
output.write_text(text,encoding='utf-8')
assert anchor in text
r = json.loads((review/'REVIEW_ROUND1.json').read_text(encoding='utf-8'))
state = {'phase':'revision','round':2,'agent_id':'/root/review_selective_transport860',
         'last_score':r['weighted_composite'],'last_verdict':r['verdict'],'status':'awaiting_round2',
         'timestamp':datetime.now().astimezone().isoformat(),'max_rounds':2,
         'review_independence':'same-family','acceptance_status':'provisional',
         'proposal_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
(folder/'REFINE_STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(state,ensure_ascii=False))
