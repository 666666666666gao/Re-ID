from pathlib import Path
import json,shutil,datetime
run=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
files=[(p,p.stat().st_size) for p in run.rglob('*') if p.is_file()]
steps=sum(len(p.read_text().splitlines()) for p in run.glob('fold_*/steps.jsonl'))
r={'observed_at':datetime.datetime.now().astimezone().isoformat(),'root':str(run),'files':len(files),'bytes':sum(n for p,n in files),'steps':steps,'free_bytes':shutil.disk_usage(run).free,'largest':[{'path':str(p.relative_to(run)),'bytes':n} for p,n in sorted(files,key=lambda x:x[1],reverse=True)[:8]]}
r['linear_1560_batch_size_estimate']=r['bytes']/steps*1560
r['estimate_scope']='Current diagnostic disk growth only; extrapolation, not a guaranteed upper bound; no new checkpoints expected.'
old=Path('/root/autodl-tmp/trifusion-v2/artifacts/trifusion_shared_semantic_circ_urgc_v3_dev_seed42')
r['old_failed_run_files']=[{'path':str(p.relative_to(old)),'bytes':p.stat().st_size} for p in old.rglob('*') if p.is_file()]
r['deleted_files']=0
print(json.dumps(r))
