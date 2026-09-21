from pathlib import Path
import hashlib,json,subprocess

t=Path('D:/Program Files/UserCache/gb/codex/tmp');p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
source=t/'trifusion_supported_balance_r2_observation_0903_20260921.json';o=json.loads(source.read_bytes())
assert o['observed_at'].startswith('2026-09-21T09:02:56') and len(o['q1']['endpoints'])==4
for r in o['q1']['endpoints']:
    assert r['updates']==260 and r['epochs']==20 and r['nonzero']==203 and r['overflow']==0
    assert r['endpoint_receipt_present'] and r['checkpoint_recorded'] and r['retrieval_recorded'] and all(r['checks'].values())
folder='evidence/supported_gradient_balance_q1_progress_20260921'
rel=folder+'/observation_0902.json';(p/rel).write_bytes(source.read_bytes());paths=[rel]
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md';b=(p/tracker).read_bytes()
old='08:40:02原44804持续执行；fold0两端及fold1控制完整，3/6终点；fold1候选11/20epoch'
new='09:02:56原44804持续执行；fold0、fold1两端检查点/检索/端点回执齐全，4/6终点；第三折及终态核验待完成'
assert b.count(old.encode())==1;(p/tracker).write_bytes(b.replace(old.encode(),new.encode()));paths.append(tracker)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='7b72239458c44654f47f9e5a699890cca413e3a5b6551b543c3d2fd70483e07f'
b+='''

#### §41.236 四端完成（09:02）

09:02:56原wrapper42758/Q1 44804仍运行。fold0、fold1两端均已完整结束，各20epoch/260更新、203/203累计非零、overflow0、固定长度及冻结路径检查通过，检查点/检索/端点回执齐全，4/6终点。fold1候选训练epoch耗时2561.982秒，不含全部端点成本。输出盘4757811200B。原始观察evidence/supported_gradient_balance_q1_progress_20260921/observation_0902.json。未读取局部分数择法；第三折、完整Q1/CPU/独立审查仍待结束，Goal ACTIVE/UNMET。
'''.encode();(p/master).write_bytes(b);paths.append(master)
agents='AGENTS.md';b=(p/agents).read_bytes()
old='08:40原wrapper42758/Q1 44804正常，fold0两端及fold1控制完整结束、3/6终点，fold1候选11/20epoch'
new='09:02原wrapper42758/Q1 44804正常，fold0、fold1两端完整结束、4/6终点，第三折及终态核验待完成'
assert b.count(old.encode())==1;b=b.replace(old.encode(),new.encode()).replace(b'balance R2 Q1 three endpoints complete',b'balance R2 Q1 four endpoints complete',1)
(p/agents).write_bytes(b);paths.append(agents)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
j=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_four_endpoints_publication_20260921.json').write_bytes((json.dumps(j,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),master_sha256=j['new_master_sha256'])))
