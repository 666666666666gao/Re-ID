from pathlib import Path
import hashlib,json,subprocess
t=Path('D:/Program Files/UserCache/gb/codex/tmp');p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
observations=['supported_balance_zero_support_0717_20260921.json','supported_balance_zero_support_0721_20260921.json']
latest=json.loads((t/observations[-1]).read_bytes())
assert latest['status']=='PASS_OBSERVED_ZERO_SUPPORT_RUNTIME_WITNESSES' and latest['live_q1_handles']
found={e['endpoint']:e for e in latest['endpoints']}
for name in ('fold_0_control','fold_0_balanced'):
    r=found[name]['active_zero_support_records'];assert len(r)==1 and r[0]['step']==180
    for v in r[0]['roles'].values():assert v['ema_unchanged'] and v['rank_norm']==v['original_to_applied_difference']==0 and v['actual_parameter_update']>0
archive='evidence/supported_gradient_balance_zero_support_runtime_20260921';(p/archive).mkdir()
names=observations+['observe_supported_balance_zero_support_20260921.py',Path(__file__).name]
paths=[]
for name in names:
    (p/archive/name).write_bytes((t/name).read_bytes());paths.append(archive+'/'+name)
readme='''# Partial Q1 runtime coverage of unsupported updates

Read-only inspection of complete JSONL prefixes from the unchanged run. This is not terminal Q1, a new experiment, gradient regeneration or a retrieval result. Both fold-0 endpoints record their registered active zero-support update at step 180: full ranking gradient zero, EMA unchanged, coefficients 1/1, original-to-applied gradient difference zero, preserved heads, and nonzero actual parameter updates. The raw saved record is included. No models, images or arrays were loaded; no optimizer update was added. This supplements Q1 engineering coverage while the sealed M0 limitation (zero actual unsupported updates) remains true.

The original live Q1 handle was checked through /proc at each observation. The final incomplete JSONL fragment, if any, is excluded and its byte count reported; every included record ends with the writer's newline. Full six-endpoint CPU checks and independent terminal audit remain required. No partial retrieval metric was read or used for tuning.
'''
(p/archive/'README.md').write_bytes(readme.encode());paths.append(archive+'/README.md')
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='982516d37fc44832c516209a1a7e7b1e190a087c4e3ad6884cd860c36065ad1e'
b+=f'''

### 41.235 支持梯度平衡：真实无支持步骤的运行覆盖（{latest['observed_at']}）

原Q1进程44804通过/proc再次确认存活。仅读完整JSONL前缀，不加载模型/图像/数组，不新增更新，不读局部检索分数。fold0控制与候选均已执行第180步：真实跨scene正例支持为0，三个角色完整排名梯度0、EMA before/after相同、应用系数1/1、原梯度与实际应用梯度差0，分类头保持，实际参数更新均非零。首轮07:17仅控制覆盖，第二轮候选也覆盖；原始记录、前缀SHA和脚本归档evidence/supported_gradient_balance_zero_support_runtime_20260921。

这是Q1运行中的工程分支见证，不是独立参数梯度重生成或完整终态；M0的0个活动无支持更新限制仍原样保留，不追溯改写审查。候选已写{found['fold_0_balanced']['complete_steps']}步、控制260步，完整六端/CPU/检索审查仍待完成。无新性能结论、不调参，Goal ACTIVE/UNMET。
'''.encode()
(p/master).write_bytes(b);paths.append(master)
j=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
(t/'trifusion_supported_balance_zero_support_publication_20260921.json').write_bytes((json.dumps(j,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),observed_at=latest['observed_at'],candidate_steps=found['fold_0_balanced']['complete_steps'],master_sha256=j['new_master_sha256'])))
