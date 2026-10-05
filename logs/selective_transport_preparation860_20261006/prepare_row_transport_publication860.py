from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private=Path('C:/Users/gb/.codex_tmp')
base=private/'independent_evidence_draft'
proof=private/'foundation_recipe_v1_20261002'
previous=json.loads((proof/'four_copy859_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
assert subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo)==b''
component=json.loads((base/'row_transport_component860/COMPONENT.json').read_text())
assert component['status']=='CPU_SYNTHETIC_COMPONENT_PASS'
assert json.loads((base/'selective_transport_component860/EXIT.json').read_text())['exit_code']==1
cleanup=json.loads((base/'closed_role_and_msvr_control_retirement860/RETIREMENT.json').read_text())
assert cleanup['removed_count']==7 and cleanup['removed_bytes']==2482212968
inventory=json.loads((base/'remaining_closed_weight_inventory860/stdout.json').read_text())
assert inventory['free_bytes']>=4966055936
own_roots=('modeling/trifusion/row_mass_role_transport.py','modeling/trifusion/selective_role_transport.py',
 'tools/check_row_mass_role_transport.py','tools/check_selective_role_transport.py',
 'tools/queue_row_mass_role_transport.py','tools/queue_selective_role_transport.py',
 'tools/report_row_mass_role_transport.py','tools/report_selective_role_transport.py',
 'tools/run_row_mass_role_transport.py','tools/run_selective_role_transport.py',
 'refine-logs/selective_role_transport_v1/')
status=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=repo).decode()
foreign=[s[3:] for s in status.split('\0') if s and not s[3:].startswith(own_roots)]
protected={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in foreign}
assert len(protected)==10
folder=repo/'refine-logs/row_mass_role_transport_v2';assert not folder.exists();folder.mkdir()
anchor=(repo/'refine-logs/selective_role_transport_v1/PROBLEM_ANCHOR.md').read_text(encoding='utf-8')
plan='# 按行未匹配质量：六端活动对照，正式训练前登记\n\n## Problem Anchor\n\n'+anchor+'\n'+'''## 科学问题与旧版本边界

本次检验接收槽位之间的跨光谱传输质量分配，是否比同一模态对内统一衰减更有最终检索效用。原RAWsemantic底座、公开CLIP/new camera/作者head、RAW职责训练、seed42、50轮、作者batch与1536维L2部署不改。没有原生/容量MLP/候选重建、外部资源、新辅助loss或测试更新。

原Sinkhorn方案方法复核两轮：6.55→7.25，论文REVISE、same-family/provisional、CALIBRATION:none，实际后端未独立核实；不是论文READY，也不是模型结果。其首次CPU数学组件失败：全低分−8条件100次更新最大边际残差0.0058288574>预设0.001。原源码/失败/诊断保持，不增迭代、改null、放宽原容差或追认为通过，未启动M0/NN。以下是新的直接按行归一化方案，非原OT完整复现；旧评分不冒充新算法独立复核。

## 唯一新机制与控制

保持CNN/T读取、直接读出和原role state；Mamba按每模态16槽独立正反扫描同一原参数，输出H=B×3×16×128。与旧RAW48交错扫描不同，因此不宣称初始化fused等于旧RAW48。global原三光谱context保留，不称整个系统模态独立。

两个bias-free128×128矩阵：matching_projection普通nn.Linear默认初始化，message_output唯一零出口，共32768参数/2张量。新增构造置于CPU torch.random.fork_rng(devices=[])，原参数按完整旧role state复制，保持旧视觉/camera/head/RNG起点；新两臂state相同。没有另一个value投影、FFN、分类头或loss。

Z=matching_projection(functional LayerNorm(T))；每个无向模态对计算Cmn=Zm Zn^T/sqrt128。接收方向分别对C及C^T追加一个固定logit1的null列并FP32 log_softmax；得到16×17按行分配。无Sinkhorn/迭代、列容量约束或双向转置互易性主张。每槽真实质量r=sum真实16列，条件Q=softmax(真实log块)，不使用hard threshold或部件真值。

slot_mass候选W=diag(r)Q；uniform_mass控制W=mean_k(r_k)Q，rbar保留梯度。对同一score两者总真实矩阵质量相同、条件Q相同，区别为质量在接收槽位之间的分配。实际H向量能量不严格匹配；独立训练后的P、总质量及特征可以不同。它是任务驱动latent slot gate，不是已校准几何/语义可靠对应。

Jm=0.5·sumpeer(Wmn Hn)，M=旧output_norms[2](Hm+message_output(Jm))；自有H旁路保留，CNN/T直接读出不变。1536structured readout、原learnable gain、h=g+gain*c不改。原槽位采样忽略positions，不能命名物理部件。

## 固定训练与六端

内部semantic标签表示slot_mass，native标签表示uniform_mass，两者都没有原生图像分支。三数据集×两臂按201→MSVR→100，各真实prepare及完整batch初始化对照→8M0→独立fresh50→首次strict→一次保存距离的15配对报告。角色输入detach/global作者任务与角色任务所有权分离保持；正常作者head训练global，角色用detached当前head参数/clone BN buffer；不增加持久第二套head。201 B64/K8、100 B128/K16、MSVR B64/K4；Adam作者分组及原日程/增强，前6视觉块GPU1/后6及headGPU0，原AMP/FP32边界保持。

复用原RAW9控制/187件依赖，不重训练旧控制。每数据集主要比较slot_mass−uniform_mass和slot_mass−RAWsemantic，mAP≥+0.5且R1不下降，同时报告两臂对独立global、原semantic；共15配对，不跨epoch拼CMC。与旧RAW比较同时包含3×16和额外容量，不能把全部作用归因局部门控。三集不一致不推广。项目门不是显著性，官方集已消费，单seed/固定模型bootstrap不是训练稳定性。

M0固定8次真实完整batch，累计所有参数有限非零梯度/实际更新、优化器恰好一次、作者BN8次、完整state严格重载。失败端封存，不缩batch/延长M0/改门。成功端自己的fresh50才启动；不添加全六端M0前置屏障或旧反向parity修复。只保留一个mAP-best；该端首次strict接受后sealed文本/数组/receipt并退役M0二进制。

## 数学、诊断与成本

新的按行组件在同一预先有限强对角/全低分/随机分数检查所有17列行质量1（atol1e-5）、真实质量[0,1]、两臂同score总质量相同（atol/rtol1e-5）、反向独立接收行、两矩阵toy8累计活动/更新、零出口及strict重载。该检查不改变原Sinkhorn失败的边际定义或容差。

逐step保存真实质量mean/min/max、条件Q行熵、Q列集中度=sum行Q/16的最大列份额、peer/self/新增出口范数，以及既有实际修正/global范数。诊断计算detach，控制rbar本身不detach。不因官方分数搜索null/gain/margin/LR/seed或增加loss。若近于统一质量/所有拒绝/槽位重复，将作为原结果解释，不救分。

仅26物理GPU0/1、单进程一对，其他项目/卡不动；无功率温度动作，25原I/O pending保持。沿用既有conda，不安装/重建环境、无W&B/消息发送。六best+最多一个M0 probe+2GiB预算=4,966,055,936B；每执行阶段2GiB余量不变。预计原作者batch六端NN+逐轮评价约6小时，安排6.5–8小时估计，实际时长另报。持久队列按预计端结束前/240秒查看，观察超时不重启。

当前仅组件通过、尚无真实初始化/M0/正式成绩；新方案没有独立论文READY评分。成功后仍需self-only/普通cross-attention等必要性消融、完整流程多种子与资源合格强参照/SOTA；整体Goal ACTIVE/UNMET。
'''
(folder/'EXPERIMENT_PLAN.md').write_text(plan,encoding='utf-8')
spec='''# 实施前锁定与源码自查

直接按行softmax，固定null=1；不运行已失败100迭代OT数学检查。原失败源独立保存。两臂均调用同一match矩阵、同一row-null和同一出口；只有r在接收槽位间保留或平均的差别，rbar可微。matching_projection默认Kaiming均匀；出口唯一置零；CPUfork RNG恢复。score双向转置后分别按行归一化，不转置最终W。Q列集中度是conditional Q分布，不声称OT列边际。所有新增参数真实参与forward/训练，旧权重在fork构造后逐项拷贝。

RAW职责与作者head/BN/optimizer由封存入口直接复用；每个新worker全新进程，不把monkeypatch串入旧控制或原已完成NN。M0诊断覆盖两矩阵累计梯度/更新。源绑定保持旧公共CLIP/camera/shared/author/batch/cfg/head/梯度政策/placement；新初始forward只要求两臂一致和旧global/公共state一致，不要求不同Mamba序列的fused与旧RAW一致。正式验收复用原receipt/全gallery检索且只接受首次strict。

方法两轮评审针对旧Sinkhorn完整稿；新直接softmax是根植同一slot_mass/mean-mass代数控制的简化，未独立获论文评分。根审阅没有将“可实现”冒充“机制有效”。组件数学/活动通过不能替代真实8M0或六端检索。无额外fallback/try/except/兼容层/无关重构。
'''
(folder/'IMPLEMENTATION_SPEC.md').write_text(spec,encoding='utf-8')
archive_relative='logs/selective_transport_preparation860_20261006'
archive=repo/archive_relative;assert not archive.exists();archive.mkdir()
packets=['closed_role_and_msvr_control_retirement860','selective_transport_component860',
 'transport_marginal_observation860','row_transport_component860',
 'ev1_closed_weight_qualification860','remaining_closed_weight_inventory860']
for name in packets:shutil.copytree(base/name,archive/name)
for name in ['write_transport_revision860.py','create_transport_panel860.py','create_row_transport_panel860.py',
             'retire_closed_role_weights860.py','prepare_row_transport_publication860.py']:
 shutil.copyfile(private/name,archive/name)
at=datetime.now().astimezone().isoformat()
section=f'''

## 41.860 下一干预收束为槽位消息质量；旧OT组件失败封存，新按行分配组件通过

{at}。前§859区域候选重建300轮/15配对已闭合且不晋级，本次没有重试它、读取detach、joint-L2或额外原生读取。围绕后继“选择性协作”形成窄问题：按接收槽位分配跨光谱消息质量，是否优于同一模态对统一衰减，并能增加原RAWsemantic之外的最终检索收益。原RAW/作者配方/1536L2保持，Mamba只改为各模态16槽私有扫描再peer交流；不是同时实现三个主模块。

方法复核按research-refine做两轮，原partial/forced6.55→slot/mean-mass7.25，均论文REVISE；same-family/provisional、CALIBRATION:none，实际后端未独立核实。原forced控制主要混入幅度作用，修订保留同score总矩阵质量并只改变槽位分配；不严格匹配向量能量、独立训练后预算，不能证明真实对应。完整R1/R2及固定anchor在refine-logs/selective_role_transport_v1，未自动升为9或论文READY。

原100次Sinkhorn/17×17/null1版本的首次CPU组件检查退出1：全低分−8构造条件，最大行边际残差0.005828857421875>预登记0.001；独立CPU观察同一原函数确认，该条件真实行质量最大0.01080610789、列残差约1.91e-6。强对角与随机条件最大残差约1.91e-6。原失败保留，不增加迭代、改变null或放宽原容差，也没有真实M0/NN分数；诊断通过不补签原失败。

新的直接按行方案将每方向16×16 score加一个null列，FP32 log_softmax得16×17行分配；取消Sinkhorn与列容量约束，反向对C转置另归一化，不称双向OT/可靠几何匹配或SuperGlue完整复现。W_slot=diag(r)Q，W_uniform=mean(r)Q，rbar保留梯度。仅matching_projection和唯一零出口message_output两个128×128矩阵，共32,768活动参数/2张量；原CNN/T与自身Mamba路径保留，不加value头、FFN、分类头、loss或资源。

新CPU组件首次通过：强对角/全低分/随机17列最大行残差分别5.96e-8/1.19e-7/2.38e-7，真实质量均值0.999089/0.001971/0.899337，两臂同score总质量匹配；两矩阵各8次toy更新累计有限非零梯度/实际变化、零出口及strict组件重载通过。此时没有真实完整模型初始化、M0或新检索成绩。新直接softmax方案未独立获论文评分，旧7.25不能冒充其论文READY；数学/活动证据不能代替机制结果。

唯一六端计划见refine-logs/row_mass_role_transport_v2/EXPERIMENT_PLAN.md：内部semantic=slot_mass、native=uniform_mass，两者都不读取原生图像；201→MSVR→100，各两臂真实prepare/完整batch匹配→8M0→fresh50→首次strict→一次15配对。seed42，201B64/K8、100B128/K16、MSVRB64/K4，作者RAW职责、head/BN所有权、optimizer/日程、原AMP和first6/last6两卡位置不变。原RAW9/187依赖直接复用，主对照slot−uniform与slot−RAWsemantic各mAP≥0.5/R1不降，独立global单列；跨旧RAW还含Mamba序列/容量差异，不能全部归因门控。固定50轮一mAP-best、完整合法gallery与camera/scene规则；不按官方分数搜null/gain/LR/margin/seed救。

23:xx清理实做7份已闭合无用best，释放2,482,212,968B：当前region201patch、MSVR两臂、100两臂，以及F2/F3重复normalizedMSVR；每份50轮/原receipt/实存weight与距离SHA核验、保留winner四项均支配，region201mean/F3metricRAWpositive/RAW187与作者公共权重保留。原354源/187控制删除前后SHA均一致；7份binary重载路径退役，历史数组/训练/回执未变，不能继续执行依赖这些旧best的直接重放。保护表187键实际为远端绝对路径；原脚本relative排除判断不充分，但显式7目标均不在保护表且全表前后SHA通过。后续资格检查同时检查绝对和相对键，原已执行源码不回改。

随后额外EV1combined201资格检查在发现best已退役时退出1，无新删除/NN；只读存量清单显示free5,553,836,032B，已高于未改变的4,966,055,936B启动预算。它相对前一free变化的外部原因未确定，不归于本次删除。正式启动仍实时检查，两卡之外项目不动。所有清理journal/组件失败/观察/新组件与helpers在{archive_relative}，无模型/NPY回传。

当前只26物理GPU0/1、无功率温度动作、25原I/O pending；不重建环境、无消息/W&B。初始全模型与逐端M0通过后才启动其fresh50，旧反向parity失败不追认修复、不作为本轮额外屏障。预计六端6.5–8小时，持久队列按端里程碑/240秒观察，超时不重启。整体Goal ACTIVE/UNMET；机制有效、三集净增益、必要性消融、完整流程多种子及资源合格强参照/SOTA仍缺。
'''
doc_name='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
doc=repo/doc_name;assert hashlib.sha256(doc.read_bytes()).hexdigest()==previous['doc_sha256']
original=doc.read_text(encoding='utf-8');assert '## 41.860 ' not in original
doc.write_text(original+section,encoding='utf-8',newline='')
shutil.copyfile(doc,Path('C:/Users/gb/Desktop/document')/doc.name)
goal=repo/'refine-logs/CURRENT_GOAL.md'
old=goal.read_text(encoding='utf-8');parts=old.split('\n',3)
parts[2]=f'更新：{at} §41.860。旧区域重建六端/15配对0晋级，不重试；后继旧OT组件FAIL封存，新row-null槽位质量vs模态对统一质量组件PASS，六端固定RAW/作者/seed42/fresh50计划待真实初始化与逐端8M0。只26 GPU0/1，无功率温度动作；25原I/O pending。整体Goal ACTIVE/UNMET；尚无新正式成绩。旧阶段历史状态不恢复。'
goal.write_text('\n'.join(parts),encoding='utf-8')
extra_names=['modeling/trifusion/selective_role_transport.py','modeling/trifusion/row_mass_role_transport.py',
 'tools/run_selective_role_transport.py','tools/run_row_mass_role_transport.py',
 'tools/check_selective_role_transport.py','tools/check_row_mass_role_transport.py',
 'tools/queue_selective_role_transport.py','tools/queue_row_mass_role_transport.py',
 'tools/report_selective_role_transport.py','tools/report_row_mass_role_transport.py',
 'refine-logs/row_mass_role_transport_v2/EXPERIMENT_PLAN.md',
 'refine-logs/row_mass_role_transport_v2/IMPLEMENTATION_SPEC.md']
old_sources=json.loads((repo/'refine-logs/region_evidence_reconstruction_v1/SOURCE_SCOPE.json').read_text())['source_sha256']
assert len(old_sources)==354
assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest()==d for n,d in old_sources.items())
sources={**old_sources,**{n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in extra_names}}
(folder/'SOURCE_SCOPE.json').write_text(json.dumps(dict(schema='trifusion-row-mass-role-transport-v2',source_sha256=sources,
 boundary='354 original sources unchanged plus12 fixed new implementation/plan files. Original failed OT source frozen and reused only for class/state helpers; active assignment is row softmax.'),indent=2)+'\n')
owned=[*extra_names,'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json',doc_name,'refine-logs/CURRENT_GOAL.md']
for directory in (archive,repo/'refine-logs/selective_role_transport_v1'):
 owned.extend(p.relative_to(repo).as_posix() for p in directory.rglob('*') if p.is_file())
owned=sorted(set(owned))
assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest()==d for n,d in protected.items())
publication=dict(previous_head=previous['head'],files=owned,protected_files=protected,
 immutable_archive_roots=[archive_relative,'refine-logs/selective_role_transport_v1'],
 remote_sparse_roots=[archive_relative,'refine-logs/selective_role_transport_v1','refine-logs/row_mass_role_transport_v2'],
 doc_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),source_count=len(sources),
 boundary='Preparation only, old OT FAIL retained; row-null component PASS, real M0/fresh50 not yet run. 2025 no probe.')
path=proof/'publication860_local.json';assert not path.exists();path.write_text(json.dumps(publication,indent=2)+'\n')
print(json.dumps(dict(owned_files=len(owned),protected_files=len(protected),source_count=len(sources),doc_sha256=publication['doc_sha256']),indent=2))
