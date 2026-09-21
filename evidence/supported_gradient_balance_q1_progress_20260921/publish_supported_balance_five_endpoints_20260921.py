from pathlib import Path
import hashlib,json,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp')
source=t/'trifusion_supported_balance_r2_observation_0947_20260921.json';o=json.loads(source.read_bytes())
assert o['observed_at'].startswith('2026-09-21T09:46:28') and len(o['q1']['endpoints'])==5
for r in o['q1']['endpoints']:
    assert r['updates']==260 and r['epochs']==20 and r['nonzero']==203 and r['overflow']==0
    assert r['endpoint_receipt_present'] and r['checkpoint_recorded'] and r['retrieval_recorded'] and all(r['checks'].values())
folder='evidence/supported_gradient_balance_q1_progress_20260921';raw=folder+'/observation_0946.json'
(p/raw).write_bytes(source.read_bytes());paths=[raw]
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md';b=(p/tracker).read_bytes()
old='09:02:56原44804持续执行；fold0、fold1两端检查点/检索/端点回执齐全，4/6终点；第三折及终态核验待完成'
new='09:46:28原44804持续执行；fold0、fold1两端及fold2控制完整，5/6终点；最后候选端及终态核验待完成'
assert b.count(old.encode())==1;(p/tracker).write_bytes(b.replace(old.encode(),new.encode()));paths.append(tracker)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='f790e8861352102ae30e0134e184bfbe49935a31cf21ff81c9ae9ed3300fb1ca'
b+='''

#### §41.236 五端完成（09:46）

09:46:28原wrapper42758/Q1 44804仍运行。fold0、fold1两端及fold2 control已完整结束，各20epoch/260更新、203/203累计非零、overflow0，检查点/检索/端点回执齐全、固定长度与冻结路径检查通过，5/6终点。fold2控制训练epoch耗时2572.413秒，不含全部端点成本。输出盘4589674496B。原始观察evidence/supported_gradient_balance_q1_progress_20260921/observation_0946.json。尚未取得最后候选端及完整Q1/CPU/独立审查，不据前五端选择方法，Goal ACTIVE/UNMET。
'''.encode();(p/master).write_bytes(b);paths.append(master)
agents='AGENTS.md';b=(p/agents).read_bytes()
old='09:02原wrapper42758/Q1 44804正常，fold0、fold1两端完整结束、4/6终点，第三折及终态核验待完成'
new='09:46原wrapper42758/Q1 44804正常，fold0、fold1两端及fold2控制完整结束、5/6终点，最后候选端及终态核验待完成'
assert b.count(old.encode())==1;b=b.replace(old.encode(),new.encode()).replace(b'balance R2 Q1 four endpoints complete',b'balance R2 Q1 five endpoints complete',1)
(p/agents).write_bytes(b);paths.append(agents)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_five_endpoints_publication_20260921.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),master_sha256=m['new_master_sha256'])))
