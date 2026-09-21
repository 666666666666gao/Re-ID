from pathlib import Path
import json,hashlib,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp');folder='evidence/supported_task_state_interface_20260921';(p/folder).mkdir();paths=[]
for name in ['inspect_supported_balance_optimizer_storage_20260921.py','trifusion_supported_balance_optimizer_storage_20260921.json']:
    rel=folder+'/'+name;(p/rel).write_bytes((t/name).read_bytes());paths.append(rel)
text='''# Task-state interface preparation — NOT AN EXPERIMENT CONTRACT

2026-09-21. Full R2 Q1 audit still in progress. No optimizer implementation, training, deployment, sweep or new method qualification. This note scopes a user-suggested conditional successor; it does not select or register one.

Current source tools/train_msvr_supported_gradient_balance.py: role parameter selection74–80, AdamW83–86, full current/history rank and direct auxiliary combination267–268, gradient deletion288, unscale290 and one step295. Encoder189 tensors; heads14. Existing full R/A are AMP-scaled, with scale256 in this run. They are deleted before optimizer step. A task-state implementation must explicitly retain/unscale those buffers if used; reading p.grad afterward cannot reconstruct them.

Read-only remote checkpoint4fee97298b318c5183ce5cf60880382480487546bf0c561976fd24a511adfdb9:

| Role | Tensors | FP32 elements |
|---|---:|---:|
| CNN |42|2229120|
| Transformer |54|1388928|
| Mamba |93|1742604|
| Total |189|5360652|

One FP32 m/v pair costs42,885,216B; two task pairs85,770,432B; incremental moments40.898529MiB. This arithmetic excludes gradients, activations, heads, counters and allocator. It is not measured peak GPU memory. No GPU initialized, no model forward/update, no checkpoint downloaded.

Before any future implementation, a registered single hypothesis must settle:

1. Control optimizer and whether R2 EMA coefficients are retained; distinguish task-state separation from simultaneous changes to loss/weights/support.
2. Warmup state continuity versus state reset, and rank task meaning when fused objective switches after65steps. Duplicating common moments is not automatically a valid task decomposition.
3. Full current+history rank must enter rank moments once; direct auxiliary enters its own moments. Heads retain original total gradient; frozen Signal has no state.
4. Unsupported step versus supported zero gradient: whether rank clock/moments hold and rank momentum is not applied; this is a candidate definition, not demonstrated benefit.
5. Task effective clocks versus global LR schedule; sum/mean of preconditioned updates; weight coefficients before/after moments; epsilon placement and once-only decoupled decay.
6. Actual training uses GradScaler. All active gradient buffers must have explicit finite/scale handling and parameters/moments must advance consistently. Current execution had overflow0; no need for speculative compatibility paths.
7. Save actual optimizer state if claiming task-history diagnostics. Current R2 checkpoints have no AdamW m/v, so they cannot reconstruct past task updates. Any new stateful diagnostic must be labeled as newly observed, not a recovered old trajectory.

AdaTask attribution and primary sources: refine-logs/msvr310_supported_gradient_balance_v1/ADATASK_PRIMARY_SOURCE_NOTE_20260921.md. Separate task moments are established prior art. Missing-support clock treatment is only a candidate adaptation and not a novelty claim. This note adds no retrieval result.
'''
rel=folder+'/INTERFACE_PREPARATION.md';(p/rel).write_text(text,encoding='utf-8');paths.append(rel)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
b+='''

#### §41.238 条件式任务状态方案的接口/存储准备

在完整Q1独立审查期间，仅做远端CPU只读参数存储核对和静态接口检查，没有实现或启动下一实验。角色189张量/5360652 FP32元素，两套任务m/v相对一套增加42885216B（约40.9MiB），不是实测峰值显存。现有分解梯度在optimizer前删除；后继必须登记warmup状态迁移、支持时钟、AMP缓冲、一次衰减及更新合成，不能直接插入AdaTask名称便认为语义完整。证据与待定项evidence/supported_task_state_interface_20260921；研究选择仍待完整审查闭环。
'''.encode();(p/master).write_bytes(b);paths.append(master)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
(t/'trifusion_task_state_interface_publication_20260921.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(paths))))
