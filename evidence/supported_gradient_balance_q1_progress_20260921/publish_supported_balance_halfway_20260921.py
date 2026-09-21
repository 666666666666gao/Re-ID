from pathlib import Path
import hashlib,json,subprocess
t=Path('D:/Program Files/UserCache/gb/codex/tmp');p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
source=t/'trifusion_supported_balance_r2_observation_0821_20260921.json';o=json.loads(source.read_bytes())
assert o['observed_at'].startswith('2026-09-21T08:21:27') and len(o['q1']['endpoints'])==3
for r in o['q1']['endpoints']:
    assert r['updates']==260 and r['epochs']==20 and r['nonzero']==203 and r['overflow']==0
    assert r['endpoint_receipt_present'] and r['checkpoint_recorded'] and r['retrieval_recorded'] and all(r['checks'].values())
folder='evidence/supported_gradient_balance_q1_progress_20260921'
rel=folder+'/observation_0821.json';(p/rel).write_bytes(source.read_bytes());paths=[rel]
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md';b=(p/tracker).read_bytes()
old='| R2 Q1 | RUNNING | 07:42:08原44804持续执行；fold0两端20epoch/260更新及检查点/检索回执齐全，2/6终点；fold1控制6/20epoch |'
new='| R2 Q1 | RUNNING | 08:21:27原44804持续执行；fold0两端及fold1控制完整，3/6终点；fold1候选2/20epoch |'
assert b.count(old.encode())==1;(p/tracker).write_bytes(b.replace(old.encode(),new.encode()));paths.append(tracker)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='ff6955cc9a818a4cc9704f17ae0c14a997002a8d34229395e327a21b463608e5'
b+='''

#### §41.235 三端完成（08:21）

08:21:27原wrapper42758/Q1 44804仍运行。fold0两端及fold1 control已完成，各20epoch/260更新、203/203累计非零、overflow0、固定长度与冻结路径检查通过，检查点/检索/端点回执齐全，3/6终点；fold1 balanced自动进入2/20epoch。fold1控制训练epoch耗时2565.730秒，不含全部端点成本。输出盘4915871744B。原始观察evidence/supported_gradient_balance_q1_progress_20260921/observation_0821.json。未读取局部分数作方法选择，完整Q1/CPU/审查仍待结束，Goal ACTIVE/UNMET。
'''.encode();(p/master).write_bytes(b);paths.append(master)
agents='AGENTS.md';b=(p/agents).read_bytes()
old='07:42原wrapper42758/Q1 44804正常，fold0两端完整结束、2/6终点，fold1控制6/20epoch'
new='08:21原wrapper42758/Q1 44804正常，fold0两端及fold1控制完整结束、3/6终点，fold1候选2/20epoch'
assert b.count(old.encode())==1;b=b.replace(old.encode(),new.encode());b=b.replace(b'balance R2 Q1 first pair complete',b'balance R2 Q1 three endpoints complete',1)
(p/agents).write_bytes(b);paths.append(agents)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
j=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_halfway_publication_20260921.json').write_bytes((json.dumps(j,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),master_sha256=j['new_master_sha256'])))
