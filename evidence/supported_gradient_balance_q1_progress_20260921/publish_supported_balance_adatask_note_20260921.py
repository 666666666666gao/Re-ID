from pathlib import Path
import hashlib,json,subprocess

p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
t=Path('D:/Program Files/UserCache/gb/codex/tmp')
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='5723365cb0f004de2efeb45f3a760401e16a9e653c0ffcaab727bc98ddb583e5'
folder='refine-logs/msvr310_supported_gradient_balance_v1'
paths=[folder+'/ROLE_LOSS_DEPENDENCY_NOTE_20260921.md',folder+'/ADATASK_PRIMARY_SOURCE_NOTE_20260921.md']
for name in paths:assert (p/name).is_file()
observation=t/'trifusion_supported_balance_r2_observation_0840_20260921.json'
o=json.loads(observation.read_bytes())
assert o['observed_at'].startswith('2026-09-21T08:40:02')
assert len(o['q1']['endpoints'])==3
raw='evidence/supported_gradient_balance_q1_progress_20260921/observation_0840.json'
(p/raw).write_bytes(observation.read_bytes());paths.append(raw)
b+='''

### §41.236 用户新复核接续：单角色五项辅助依赖、AdaTask 边界与 R2 原合同继续（2026-09-21 08:40）

用户补充的53e0c26/07:43快照之后，08:40:02实查原wrapper42758/Q1 44804继续运行，3/6端完整，fold1 balanced完成11/20epoch，GPU7578MiB/100%，输出盘4871094272B。原始文本evidence/supported_gradient_balance_q1_progress_20260921/observation_0840.json。完整六端/CPU/独立审查尚未结束，没有新增正式成绩，不据部分端点选方法或改训练。执行仍1381639/配置9ce36299；Goal ACTIVE/UNMET。

本轮静态核对四个相关代码文件与执行1381639 Git blob逐字一致。全模型除fused排名外确有十三项辅助标量，但单个角色直接连接的辅助只有五项：fused ID、自己的完整分支ID/Triplet、自己的纯残差ID/Triplet；另两角色八项没有到该角色可训练编码器的路径。现行直接求十三项和的导数会自然遵守这项依赖，不需要改训练器。不能用“1对13”替代同角色实际梯度测量。详见refine-logs/msvr310_supported_gradient_balance_v1/ROLE_LOSS_DEPENDENCY_NOTE_20260921.md；本轮没有新张量或GPU试验。

按research技能进行有界原文/作者代码核查，笔记ADATASK_PRIMARY_SOURCE_NOTE_20260921.md。AdaTask论文Algorithm2和作者固定84853de代码已经分别维护任务一/二阶矩、各任务预条件更新求和；作者实现每参数只有一个共同step，零梯度仍推进状态并可能应用旧动量，weight_decay虽接受但step未应用。因此用户提出的无支持冻结排名矩/独立有效时钟、不应用旧排名动量、只做一次AdamW解耦衰减，属于待登记验证的适配，不能写成原版已有规则或已证创新。用户引用Cityscapes三行指标原表一致，论文三种子/末十epoch平均是作者协议；本项目仍仅seed42固定终点，不据此增加种子。

后继选择继续等待R2完整结果。现行有界EMA组合后进入同一套AdamW状态；范数比与记录的实际总参数变化都不足以分摊任务更新贡献，尚未证明共享优化器历史是失分原因。无支持开关、任务时钟、角色依赖和历史导数各有边界，不为了叙事同时叠加。原始权重/数组继续留远端，当前没有确认可删的依赖权重。
'''.encode()
(p/master).write_bytes(b);paths.append(master)
tracker=folder+'/EXPERIMENT_TRACKER.md';b=(p/tracker).read_bytes()
old='08:21:27原44804持续执行；fold0两端及fold1控制完整，3/6终点；fold1候选2/20epoch'
new='08:40:02原44804持续执行；fold0两端及fold1控制完整，3/6终点；fold1候选11/20epoch'
assert b.count(old.encode())==1
(p/tracker).write_bytes(b.replace(old.encode(),new.encode()));paths.append(tracker)
agents='AGENTS.md';b=(p/agents).read_bytes()
old='08:21原wrapper42758/Q1 44804正常，fold0两端及fold1控制完整结束、3/6终点，fold1候选2/20epoch'
new='08:40原wrapper42758/Q1 44804正常，fold0两端及fold1控制完整结束、3/6终点，fold1候选11/20epoch'
assert b.count(old.encode())==1
b=b.replace(old.encode(),new.encode()).replace('当前执行入口：§41.235'.encode(),'当前执行入口：§41.236'.encode(),1)
(p/agents).write_bytes(b);paths.append(agents)
rel='evidence/supported_gradient_balance_q1_progress_20260921/'+Path(__file__).name
(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_adatask_publication_20260921.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),master_sha256=m['new_master_sha256'])))
