from pathlib import Path
import hashlib,json,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp')
folder='evidence/supported_gradient_balance_q1_cpu_failure_20260921'
(p/folder).mkdir()
paths=['tools/verify_msvr_supported_gradient_balance_stats.py']
for name in ['trifusion_supported_balance_r2_observation_1031_20260921.json','trifusion_supported_balance_q1_cpu_failure_20260921.json','trifusion_supported_balance_ratio_diagnosis_20260921.json','diagnose_supported_balance_ratio_20260921.py']:
    rel=folder+'/'+name;(p/rel).write_bytes((t/name).read_bytes());paths.append(rel)
note='''# Q1 CPU verification failure and arithmetic repair

2026-09-21 10:29:22: all six endpoints completed; original Q1 exited 0. Original CPU verifier exited 1 at 10:29:28 with exact ratio equality assertion. Original pipeline remains STOPPED_AT_Q1_CPU; no restart or training update.

Full saved scalar scan: six files, 1560 steps, 3486 supported role rows. Every recorded ratio exactly matches runtime math.sqrt. Six rows differ from verifier exponentiation x**0.5 by one ULP, 2.220446049250313e-16. The minimal verifier repair uses math.sqrt while retaining exact equality and all existing thresholds. No change to training, config, checkpoint, scientific gates or results. Full repaired verification and independent audit remain pending. No scientific qualification asserted.
'''
rel=folder+'/README.md';(p/rel).write_text(note,encoding='utf-8');paths.append(rel)
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md'
b=(p/tracker).read_bytes();old='09:46:28原44804持续执行；fold0、fold1两端及fold2控制完整，5/6终点；最后候选端及终态核验待完成'
new='10:29六端训练完整退出0；CPU ratio精确比较失败，已定位sqrt与幂运算1ULP差异；核验器最小修复，完整重核及审查待完成'
assert b.count(old.encode())==1;(p/tracker).write_bytes(b.replace(old.encode(),new.encode()));paths.append(tracker)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='7e56868b4daa5970552516847d7eefd0e3d92c86601bc4481af8bf97f59ebf7f'
b+='''

### §41.237 R2六端训练完成，CPU精确标量重放差异（2026-09-21 10:32）

原Q1 44804于10:29:22退出0，六端20epoch/260更新全部结束。原CPU 69196于10:29:28退出1，pipeline保留STOPPED_AT_Q1_CPU。失败在verify_balance的ratio精确相等；训练math.sqrt与核验x**0.5在六条记录相差1ULP（2.220446049250313e-16）。全1560步/3486有支持角色记录的保存ratio均精确匹配运行math.sqrt。核验器仅将幂运算改为同一math.sqrt，精确相等及原所有阈值不放宽；训练/config/checkpoint不改、不重训。原失败日志、pipeline、全量差异诊断归档evidence/supported_gradient_balance_q1_cpu_failure_20260921。完整修复后CPU核验、六端分析及独立审查仍待完成，暂不宣布科学结论。输出卷4430831616B，GPU已空闲；保留所有必要权重。Goal ACTIVE/UNMET。
'''.encode();(p/master).write_bytes(b);paths.append(master)
agents='AGENTS.md';b=(p/agents).read_bytes();old='09:46原wrapper42758/Q1 44804正常，fold0、fold1两端及fold2控制完整结束、5/6终点，最后候选端及终态核验待完成'
new='10:29六端训练完整结束，原CPU因sqrt与幂运算1ULP精确比较差异停止；仅修核验器，待完整重核和独立审查，不重训'
assert b.count(old.encode())==1;(p/agents).write_bytes(b.replace(old.encode(),new.encode()));paths.append(agents)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_cpu_sqrt_fix_publication_20260921.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(json.dumps(m))
