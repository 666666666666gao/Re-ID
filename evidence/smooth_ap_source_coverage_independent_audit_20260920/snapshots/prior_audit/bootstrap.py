from pathlib import Path
import hashlib, json, shutil, datetime

root = Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
out = Path(__file__).parent
declared = '''AGENTS.md
configs/MSVR310/TriFusion-smooth-ap-source-coverage-v1.json
refine-logs/msvr310_smooth_ap_source_coverage_v1/DIAGNOSTIC_PLAN.md
tools/diagnose_msvr_smooth_ap_coverage.py
tools/diagnose_msvr_source_relations.py
tools/msvr310_exact_signal_inference.py
tools/train_msvr_smooth_ap.py
tools/train_msvr_instance_memory.py
tools/train_msvr310_trifusion_oof.py
tools/train_msvr310_signal_oof.py
tools/msvr_instance_memory.py
tools/msvr_freshness_probe.py
protocols/msvr310_train_oof_v1.json
results/MSVR310_SMOOTH_AP_SOURCE_COVERAGE_2026-09-09.md
docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'''.splitlines()
dirs = ['smooth_ap_source_coverage_'+s+'_20260909' for s in ['preparation','launch','verification_progress','complete','analysis','completion_support']]
files = [(root/p, 'repo/'+p) for p in declared]
for d in dirs:
    files += [(p, 'repo/'+p.relative_to(root).as_posix()) for p in sorted((root/'evidence'/d).rglob('*')) if p.is_file()]
tmp = Path(r'C:\Users\gb\.codex_tmp\smooth_ap_source_coverage_complete_20260909')
files += [(p, 'executor_complete/'+p.relative_to(tmp).as_posix()) for p in sorted(tmp.rglob('*')) if p.is_file()]
skillbase=Path(r'C:\Users\gb\.codex\skills')
for rel in ['experiment-audit/SKILL.md','shared-references/local-codex-policy.md','shared-references/review-tracing.md','shared-references/experiment-integrity.md','shared-references/reviewer-independence.md']:
    files.append((skillbase/rel,'audit_policy/'+rel))
inventory=[]
for p,rel in files:
    if not p.exists():
        inventory.append({'source':str(p),'snapshot':rel,'exists':False}); continue
    if p.suffix.lower() not in ['.py','.md','.json','.jsonl','.csv','.txt','.log','.sh','.ps1','.yaml','.yml','.toml']:
        inventory.append({'source':str(p),'snapshot':rel,'exists':True,'excluded_binary':True,'bytes':p.stat().st_size}); continue
    data=p.read_bytes()
    target=out/'snapshots'/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes(data)
    inventory.append({'source':str(p),'snapshot':str(target.relative_to(out)),'exists':True,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'lines':len(data.splitlines())})
(out/'local_input_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
print(json.dumps({'snapshots':sum('sha256' in i for i in inventory),'missing':[i for i in inventory if not i['exists']],'excluded_binary':[i for i in inventory if i.get('excluded_binary')],'total_bytes':sum(i.get('bytes',0) for i in inventory),'largest':sorted(inventory,key=lambda i:i.get('bytes',0),reverse=True)[:8]},indent=2))
