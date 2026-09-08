import json, hashlib, math, pathlib, shutil, datetime

A = pathlib.Path(__file__).resolve().parent
I = pathlib.Path('C:/Users/gb/.codex_tmp/role_set_q1_complete_20260908')
R = pathlib.Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def shape(x):
    if isinstance(x,dict): return {k:shape(v) for k,v in x.items()}
    if isinstance(x,list): return {'length':len(x),'first':shape(x[0]) if x else None}
    return x
def inspect(x, counter):
    counter[type(x).__name__]=counter.get(type(x).__name__,0)+1
    if isinstance(x,float): assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values(): inspect(v,counter)
    elif isinstance(x,list):
        for v in x: inspect(v,counter)
inv=json.loads((I/'remote_terminal_inventory.json').read_text())
registry=[]
for f in inv['files']:
    p=I/f['path']; raw=p.read_bytes()
    row={'path':str(p),'bytes':len(raw),'sha256':sha(p),'registered_bytes':f['bytes'],'registered_sha256':f['sha256']}
    assert row['bytes']==f['bytes'] and row['sha256']==f['sha256'],f['path']
    txt=raw.decode('utf-8'); row['lines']=len(txt.splitlines())
    counter={}
    if p.suffix=='.json':
        obj=json.loads(txt); inspect(obj,counter); row['keys']=list(obj) if isinstance(obj,dict) else None
    elif p.suffix=='.jsonl':
        rows=[json.loads(s) for s in txt.splitlines()]; row['row_count']=len(rows)
        for obj in rows: inspect(obj,counter)
        row['keys']=list(rows[0]) if rows else []
    else: row['text']=txt
    row['full_parse_type_counts']=counter; registry.append(row)
save(A/'local_intake_inspection.json',{'generated_at':datetime.datetime.now().astimezone().isoformat(),'files':registry,'total_bytes':sum(r['bytes'] for r in registry),'status':'ALL_57_REGISTERED_TEXT_HASHES_AND_COMPLETE_PARSE_MATCH','m0_scope':'intake parse and provenance only; no rerun or re-audit of M0'})
s=json.loads((I/'q1/summary.json').read_text())
save(A/'q1_summary_extract.json',{k:v for k,v in s.items() if k not in ('folds','overfit')})
save(A/'q1_endpoint_schema.json',shape(s['folds'][0]))
save(A/'q1_cpu_schema.json',json.loads((I/'q1_cpu.json').read_text()))
paths=[pathlib.Path('C:/Users/gb/.codex_tmp/role_set_q1_audit_request_20260908.md'),pathlib.Path('C:/Users/gb/.codex_tmp/analyze_role_set_q1_terminal_v2_20260908.py')]
names=['train_msvr_role_set.py','verify_msvr_role_set.py','run_msvr_role_set.py','check_msvr_role_set.py','msvr_role_set_relations.py','msvr_instance_memory.py','msvr_freshness_probe.py','train_msvr_history_gradient.py','probe_msvr_history_candidate_gradients.py','train_msvr310_trifusion_oof.py','train_msvr310_signal_oof.py','audit_msvr_paired_ranking_text.py','train_msvr_instance_memory.py','train_msvr310_source_style.py','probe_msvr_role_set_gradients.py','run_signal_preserving_v5.py']
paths += [R/'tools'/n for n in names]
paths += [R/'refine-logs/msvr310_role_set_v1'/n for n in ['TRAINING_PLAN.md','EXPERIMENT_TRACKER.md']]
config=R/'configs/MSVR310/TriFusion-role-set-paired-v1.json'
visited=set()
def recurse(p):
    if p in visited: return
    visited.add(p);paths.append(p)
    if p.suffix!='.json': return
    d=json.loads(p.read_text())
    for k,v in d.items():
        if isinstance(v,str) and v.endswith(('.json','.yaml','.yml','.md','.py')) and (R/v).is_file(): recurse(R/v)
    for name in d.get('project_file_sha256',{}):
        q=R/name
        if q.is_file(): recurse(q)
recurse(config)
hashes={str(I/f['path']):f['sha256'] for f in inv['files']}
for p in dict.fromkeys(paths):
    hashes[str(p)]=sha(p)
    target=A/'snapshots'/(p.relative_to(R) if p.is_relative_to(R) else pathlib.Path('request_and_analysis')/p.name)
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
save(A/'audited_input_hashes.json',hashes)
save(A/'recursive_local_configs.json',{'files':[str(p) for p in visited if p.suffix=='.json']})
print(json.dumps({'status':'PASS_LOCAL_INTAKE_HASH_PARSE','files':len(registry),'bytes':sum(r['bytes'] for r in registry),'summary_keys':list(s),'snapshot_files':len(set(paths))}))
