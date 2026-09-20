from pathlib import Path
import hashlib, json, subprocess

tmp=Path('D:/Program Files/UserCache/gb/codex/tmp')
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
src=tmp/'trifusion_supported_balance_m0_independent_audit_20260921'
manifest=json.loads((src/'artifact_manifest.json').read_bytes())
receipt=json.loads((src/'audit_receipt.json').read_bytes())
assert receipt['input_hashes']['m0_cpu_receipt']=='910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1'
assert receipt['closure_status']=='CLOSED_WITH_LIMITS' and not receipt['remaining_audit_blockers']
archive='evidence/supported_gradient_balance_m0_independent_audit_20260921'
assert not (repo/archive).exists()
paths=[]
def extended(p):
    return Path('\\\\?\\'+str(p.resolve()))
for row in manifest['files']:
    name=row['relative_path']
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    b=extended(src/name).read_bytes()
    assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],name
    b.decode('utf-8')
    target=extended(repo/archive/name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
    paths.append(archive+'/'+name)
for name in ('artifact_manifest.json','artifact_manifest.sha256'):
    (repo/archive/name).write_bytes((src/name).read_bytes());paths.append(archive+'/'+name)
for ext in ('md','json'):
    rel='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_AUDIT_M0.'+ext
    (repo/rel).write_bytes((src/('EXPERIMENT_AUDIT.'+ext)).read_bytes());paths.append(rel)
trace=repo/'.aris/traces/experiment-audit/2026-09-21_supported_balance_m0'
(trace/'001-response.md').write_bytes((src/'final_response.md').read_bytes())
meta=json.loads((trace/'001-meta.json').read_bytes())
meta.update(status='CLOSED_WITH_LIMITS',verdict='WARN',archive=archive,report_sha256=hashlib.sha256((src/'EXPERIMENT_AUDIT.md').read_bytes()).hexdigest(),response_sha256=hashlib.sha256((src/'final_response.md').read_bytes()).hexdigest())
(trace/'001-meta.json').write_bytes((json.dumps(meta,ensure_ascii=False,indent=2)+'\n').encode())
obs=tmp/'trifusion_supported_balance_r2_observation_0659_20260921.json'
o=json.loads(obs.read_bytes())
assert len(o['q1']['endpoints'])==1
ep=o['q1']['endpoints'][0]
assert ep['updates']==260 and ep['checkpoint_recorded'] and ep['retrieval_recorded'] and ep['endpoint_receipt_present']
rel=archive+'/root_observation_0659.json';(repo/rel).write_bytes(obs.read_bytes());paths.append(rel)
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md'
b=(repo/tracker).read_bytes()
old='| R2 M0独立审查 | RUNNING | fresh gpt-6-astra/max，same-family/provisional，待完整裁决 |'
new='| R2 M0独立审查 | WARN/CLOSED_WITH_LIMITS | 独立CPU全量复算通过，0工程/审计阻断；same-family/provisional；实际M0无支持更新0 |'
assert b.count(old.encode())==1;b=b.replace(old.encode(),new.encode())
old='| R2 Q1 | RUNNING | 原44804于06:09:45自动启动；06:13:21首控制5/20epoch，0/6终点 |'
new='| R2 Q1 | RUNNING | 06:59:13原44804持续执行；首控制20epoch/260更新及检查点/检索回执齐全，1/6终点 |'
assert b.count(old.encode())==1;b=b.replace(old.encode(),new.encode())
(repo/tracker).write_bytes(b);paths.append(tracker)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
b=(repo/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='63df3e03a80a346351c34f6029d14490d75f84c94ca6960d213530d946f81b8d'
append='''

### 41.234 支持梯度平衡 R2：M0 独立审查闭环，Q1 首端完成（2026-09-21 06:59 北京时间）

执行仍为1381639、配置9ce36299，原wrapper42758/Q1进程44804持续运行；没有重启、修改训练目标或门槛。06:59:13首控制端20epoch/260更新完成，检查点、检索与端点回执齐全，现为1/6完整端点。无完整Q1/CPU或新增正式结果；不使用局部分数选方法。

fresh-none gpt-6-astra/max独立M0审查终态WARN/CLOSED_WITH_LIMITS：A/B/C/D/F通过、E覆盖限制保留，0剩余工程或审计阻断。same-family/provisional，后端独立认证不可用。完整报告见refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_AUDIT_M0.md/.json；402份UTF-8证据按原manifest及SHA归档至evidence/supported_gradient_balance_m0_independent_audit_20260921，包含精确请求、失败脚本/输出、修订快照和原文回复，不只保存结论。

独立远端CPU核验26.965秒、2线程/interop1、CUDA未初始化：248步、4945920距离元素、8280标量恒等式、780来源batch及37个M0文件；六个容量端和三个B0检查点在远端复核。90项原运行参数参考均满足原0.005门槛，最大相对误差9.433501552915618e-05；保存标量复算不等于重新生成每步参数梯度。

真实M0为16预热/232有支持/0活动无支持更新；557个有支持但排名范数为零的角色步正确更新EMA，不当作无支持。严格重载仅覆盖六个容量端，两个过拟合运行未保存终点检查点、未重载。T0及完整来源计划中的无支持情况不是已完成Q1的实测覆盖。R1第4步0.006824872>0.005失败与此前3次更新保持封存。

审查期间发现报告草稿将Windows换行转换副本哈希误称原始回执，现已纠正并保留错误草稿与AF6记录。远端原始m0_cpu.json为16016字节/SHA910c5c8493059cded0adb02bcbdb5ab0b701b40567b645f3a1ca37014feb21b1；本地渲染副本16359字节/SHA4f551ff62e64b5352fbd03398b3725c18ecee81b4d0f8ef8efff68013a6a5228，343个LF→CRLF且JSON内容一致。154份同类文本逐一核对；实验数据及门槛未改。

输出盘06:59剩余5226696704字节；保留所有初始化/终点/审计依赖。终态接收、全量日志与梯度分析、独立Q1审查请求已准备，等六端和CPU齐全后执行，不将准备状态当成终态。仅seed42、先主结果后消融，三数据集Goal ACTIVE/UNMET。
'''
b+=append.encode();(repo/master).write_bytes(b);paths.append(master)
agents='AGENTS.md';a=(repo/agents).read_bytes()
start=a.index(b'## Latest verified state');end=a.index(b'## Archived standard Smooth-AP',start)
entry='''## Latest verified state (2026-09-21 balance R2 M0 audit closed; Q1 1/6)

- 当前执行入口：§41.234。支持梯度平衡R2执行1381639/配置9ce36299，M0/CPU完整通过，独立M0审查WARN/CLOSED_WITH_LIMITS、0阻断；覆盖限制见报告。原wrapper42758/Q1 44804，06:59首控制20epoch/260更新、检查点/检索回执齐全，1/6终点。run /root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639。R1失败保留；不修改运行源码/科学门，不因观察超时重启；Goal ACTIVE/UNMET。
'''
(repo/agents).write_bytes(a[:start]+entry.encode()+a[end:]);paths.append(agents)
script_rel=archive+'/archive_supported_balance_m0_audit_20260921.py'
(repo/script_rel).write_bytes(Path(__file__).read_bytes());paths.append(script_rel)
publication=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
out=tmp/'trifusion_supported_balance_m0_audit_publication_20260921.json'
assert not out.exists();out.write_bytes((json.dumps(publication,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(files=len(paths),archive_bytes=manifest['total_bytes'],master_sha256=publication['new_master_sha256'],publication=str(out))))
