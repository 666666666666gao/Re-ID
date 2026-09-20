from pathlib import Path
import hashlib,json,subprocess
t=Path('D:/Program Files/UserCache/gb/codex/tmp');p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
source=t/'trifusion_supported_balance_r2_observation_0728_20260921.json'
o=json.loads(source.read_bytes())
assert o['observed_at'].startswith('2026-09-21T07:42:08')
assert o['original_processes']['42758'] and o['original_processes']['44804']
assert len(o['q1']['endpoints'])==2
for r in o['q1']['endpoints']:
    assert r['fold']==0 and r['epochs']==20 and r['updates']==260 and r['nonzero']==203 and r['overflow']==0
    assert r['endpoint_receipt_present'] and r['checkpoint_recorded'] and r['retrieval_recorded'] and all(r['checks'].values())
folder='evidence/supported_gradient_balance_q1_progress_20260921';(p/folder).mkdir(exist_ok=True)
rel=folder+'/observation_0742.json';(p/rel).write_bytes(source.read_bytes());paths=[rel]
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md'
b=(p/tracker).read_bytes();old='| R2 Q1 | RUNNING | 06:59:13原44804持续执行；首控制20epoch/260更新及检查点/检索回执齐全，1/6终点 |'
new='| R2 Q1 | RUNNING | 07:42:08原44804持续执行；fold0两端20epoch/260更新及检查点/检索回执齐全，2/6终点；fold1控制6/20epoch |'
assert b.count(old.encode())==1;(p/tracker).write_bytes(b.replace(old.encode(),new.encode()));paths.append(tracker)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='41c063f0f7bf95b2eff69b8f81f5e7b87b2f7a491d27babd41c0fc7761ea59b0'
b+='''

#### §41.235 首个完整配对里程碑（07:42）

07:42:08真实观察：原wrapper42758/Q1 44804存活，fold0 control/balanced均完成20epoch/260更新，端点回执、检查点和检索记录齐全，2/6完整端点；fold1 control完成6/20epoch。两端203/203累计非零、overflow0、冻结Signal/基座不变及全部训练工程检查通过。训练epoch耗时合计2595.407/2569.485秒；这些不包括所有初始化/重载/检索成本。GPU7578MiB/100%，输出盘5065076736B。未用局部检索分数调参；完整Q1/CPU/审查未结束。原始观察归档evidence/supported_gradient_balance_q1_progress_20260921/observation_0742.json，以文件内observed_at为实际时间，不以本地预设文件名推断时间。Goal ACTIVE/UNMET。
'''.encode();(p/master).write_bytes(b);paths.append(master)
agents='AGENTS.md';b=(p/agents).read_bytes();start=b.index(b'## Latest verified state');end=b.index(b'## Archived standard Smooth-AP',start)
entry='''## Latest verified state (2026-09-21 balance R2 Q1 first pair complete)

- 当前执行入口：§41.235。支持梯度平衡R2执行1381639/配置9ce36299，M0/CPU及独立审查闭环，限制保留。07:42原wrapper42758/Q1 44804正常，fold0两端完整结束、2/6终点，fold1控制6/20epoch；真实无支持步骤180两端均覆盖。run /root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639。不改源码/合同，不使用中间分数选方法，不因观察超时重启。Goal ACTIVE/UNMET。
'''
(p/agents).write_bytes(b[:start]+entry.encode()+b[end:]);paths.append(agents)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
j=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_first_pair_publication_20260921.json').write_bytes((json.dumps(j,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),master_sha256=j['new_master_sha256'])))
