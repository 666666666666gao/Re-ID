import json,hashlib,datetime
from pathlib import Path
O=Path(__file__).resolve().parent
p=json.loads((O/'remote_provenance.stdout').read_bytes())
for path,content in p['texts'].items():
    dest=O/'snapshots'/'remote'/path.lstrip('/');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content,encoding='utf-8',newline='')
(O/'provenance_summary.json').write_text(json.dumps({k:v for k,v in p.items() if k not in ('texts','files')},indent=2),encoding='utf-8')
inv=json.loads((O/'remote_inventory.stdout').read_bytes());arith=json.loads((O/'arithmetic_summary.json').read_bytes())
hashes={path:'sha256:'+entry['sha256'] for path,entry in inv['files'].items()}
hashes.update({path:'sha256:'+entry['sha256'] for path,entry in p['files'].items()})
for path,entry in json.loads((O/'local_input_manifest.json').read_bytes()).items():hashes[path]='sha256:'+entry['sha256']
(O/'audited_input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
pipeline=json.loads(inv['texts']['/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4/pipeline.json'])
summary=json.loads(inv['texts']['/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4/m0/summary.json'])
stepdetails=[]
for f in summary['folds']:
    for end,row in f['endpoints'].items():
        t=row['training'];w=t['historical_parameter_gradient_witness']
        stepdetails.append({'fold':f['fold'],'endpoint':end,'first_history_witness_step':w['step'],'historical_role_norms':{e:v['total_vs_history']['second_norm'] for e,v in w['roles'].items()}})
cost={'optimizer_updates':248,'current_anchor_exposures':sum(r['current_anchor_exposures'] for r in arith['endpoints']),'extra_fresh_role_record_forwards':sum(r['fresh_record_forwards'] for r in arith['endpoints']),'extra_history_vjp_record_forwards':sum(r['history_vjp_forwards'] for r in arith['endpoints']),'extra_direct_record_forwards':sum(r['direct_record_forwards'] for r in arith['endpoints']),'historical_candidate_exposures':sum(r['history_candidates'] for r in arith['endpoints']),'fit_seconds':sum(r['actual_fit_seconds'] for r in arith['endpoints']),'max_allocated_mib':max(r['peak_allocated_mib'] for r in arith['endpoints']),'max_reserved_mib':max(r['peak_reserved_mib'] for r in arith['endpoints']),'stages':[r for r in pipeline['stages'] if r['stage'] in ('t0','m0','m0_cpu')],'witnesses':stepdetails,'remote_process_snapshot':inv['processes']}
(O/'cost_and_runtime_summary.json').write_text(json.dumps(cost,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in p.items() if k not in ('texts','files','files_changed_since_execution_commit')},indent=2))
print('changed_paths',json.dumps(p['files_changed_since_execution_commit'],indent=2))
print('audited_input_hashes',len(hashes));print('cost',json.dumps({k:v for k,v in cost.items() if k not in ('stages','witnesses','remote_process_snapshot')},indent=2))
