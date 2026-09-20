from pathlib import Path
import json,datetime
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
s=json.loads((root/'summary.json').read_bytes())
keys=['directory','steps','elapsed_seconds','model_state_unchanged','gradients_absent','optimizer_updates','heldout_record_forwards','official_image_reads','scale','peak_cuda_memory_bytes','peak_memory_bytes','direct_history_proof']
result={'observed_at':datetime.datetime.now().astimezone().isoformat(),'status':s['status'],'conditions':[{k:v[k] for k in keys if k in v} for v in s['conditions']]}
result['steps']={p.parent.name:len(p.read_text().splitlines()) for p in root.glob('fold_*/steps.jsonl')}
result['processes']={str(pid):{'exists':Path('/proc',str(pid)).exists(),'cmdline':Path('/proc',str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode() if Path('/proc',str(pid)).exists() else None} for pid in [1230,1231,1706,2251]}
print(json.dumps(result))
