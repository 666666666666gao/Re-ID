from pathlib import Path
import json,datetime,subprocess,shutil
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_source_seed42_c6fbfb4_20260920')
pipeline=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
r=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),pipeline=pipeline,processes={},conditions=[],gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),output_free=shutil.disk_usage(root).free)
for key in ['pid','child_pid']:
 pid=pipeline[key];p=Path('/proc',str(pid));r['processes'][key]=dict(pid=pid,exists=p.exists(),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode() if p.exists() else None)
for p in sorted(root.glob('fold_*/steps.jsonl')):
 rows=p.read_text().splitlines();last=json.loads(rows[-1]) if rows else {};r['conditions'].append(dict(directory=p.parent.name,steps=len(rows),last_step=last.get('step'),epoch=last.get('epoch')))
r['epoch_events']=[json.loads(l) for l in Path(str(root)+'_source.log').read_text().splitlines() if l.startswith('{"event":')][-8:]
q=json.loads(Path(str(root)+'_verification_pipeline.json').read_bytes());r['verification_queue']=q;r['verification_queue_proc_exists']=Path('/proc',str(q['pid'])).exists()
a=json.loads(Path(str(root)+'_analysis_pipeline.json').read_bytes());r['analysis_queue']=a;r['analysis_queue_proc_exists']=Path('/proc',str(a['pid'])).exists()
print(json.dumps(r))
