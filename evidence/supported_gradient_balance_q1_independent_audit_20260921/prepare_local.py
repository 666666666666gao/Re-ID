from pathlib import Path
import json, hashlib, shutil

OUT=Path(__file__).parent
REPO=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
INTAKE=OUT.parent/'trifusion_supported_balance_q1_complete_20260921'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
inventory=json.loads((INTAKE/'inventory.json').read_bytes())
checked=[]
for r in inventory['files']:
    p=INTAKE/r['path']
    assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],r['path']
    checked.append(r)
todo=[REPO/'configs/MSVR310/TriFusion-supported-gradient-balance-paired-v1.json']
seen=set(); config_records={}; bindings=[]
while todo:
    p=todo.pop()
    if p in seen: continue
    seen.add(p); x=json.loads(p.read_bytes()); config_records[str(p.relative_to(REPO))]=x
    for k in ('project_file_sha256','project_source_file_sha256'):
        for name,h in x.get(k,{}).items():
            q=REPO/name
            bindings.append(dict(config=str(p.relative_to(REPO)),path=name,expected=h,actual=sha(q)))
            seen.add(q) if q.suffix!='.json' or not str(name).startswith('configs/') else None
            if q.suffix=='.json' and str(name).startswith('configs/'): todo.append(q)
    for k in ('previous_config','base_config','coordinate_config'):
        if k in x: todo.append(REPO/x[k])
    if 'BASELINE' in x: todo.append(REPO/x['BASELINE']['CONFIG'])
extras=['AGENTS.md','tools/recheck_msvr_supported_balance_sqrt.py','tools/msvr_cross_scene_smooth_ap.py','tools/audit_msvr_paired_ranking_text.py','tools/train_msvr310_trifusion_oof.py','tools/train_msvr310_signal_oof.py','tools/train_msvr310_source_style.py','tools/train_msvr_instance_memory.py','tools/probe_msvr_history_candidate_gradients.py','tools/probe_msvr_role_set_gradients.py','tools/msvr_freshness_probe.py','tools/msvr_instance_memory.py','tools/run_signal_preserving_v5.py','modeling/trifusion/signal_preserving_v8.py','tools/msvr_role_set_relations.py','tools/msvr_smooth_ap.py','tools/verify_msvr310_source_style.py','refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md','refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_AUDIT_M0.md','refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_AUDIT_M0.json','evidence/smooth_ap_cross_scene_support_20260909/smooth_ap_cross_scene_support_20260909.json']
seen.update(REPO/s for s in extras)
for x in config_records.values():
    if 'protocol' in x:seen.add(REPO/x['protocol'])
    if 'SOURCE_METADATA' in x:seen.add(REPO/x['SOURCE_METADATA']['PATH'])
snapshots=[]
for p in sorted(seen):
    assert p.is_file(),str(p)
    target=OUT/'snapshots/repo'/p.relative_to(REPO);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists(): assert sha(target)==sha(p)
    else: shutil.copyfile(p,target)
    snapshots.append(dict(source=str(p),snapshot=str(target.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
save(OUT/'input_manifest.json',dict(local_intake=checked,snapshots=snapshots,bindings=bindings,config_paths=list(config_records)))
summary=json.loads((INTAKE/'q1/summary.json').read_bytes())
proof=json.loads((INTAKE/'q1_cpu_arithmetic_recheck/verification.json').read_bytes())
save(OUT/'schema_inspection.json',dict(configs=config_records,summary_keys=list(summary),verification=proof,folds=[dict(fold=r['fold'],endpoint_keys=list(r['endpoints']['control']),training_keys=list(r['endpoints']['control']['training']),retrieval_keys=list(r['endpoints']['control']['retrieval'])) for r in summary['folds']]))
print(json.dumps(dict(status='PASS_LOCAL_INTAKE_HASHES',files=len(checked),snapshots=len(snapshots),binding_rows=len(bindings),local_binding_mismatches=[b for b in bindings if b['expected']!=b['actual']],config_paths=list(config_records),summary_sha256=sha(INTAKE/'q1/summary.json'),verification_counts={k:v for k,v in proof.items() if k.startswith('checked_')}),indent=2))
