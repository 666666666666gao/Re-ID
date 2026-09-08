from pathlib import Path
import datetime,json,shutil,subprocess
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_objective_gradients_preflight_seed42_a2dec7f')
state=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
result=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),pipeline=state,
    live={str(p):Path('/proc',str(p)).exists() for p in [state['pid'],state['child_pid']]},
    free_bytes=shutil.disk_usage(root.parent).free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip(),
    log_tail=Path(str(root)+'_preflight.log').read_text()[-3000:])
if (root/'summary.json').exists():
    j=json.loads((root/'summary.json').read_bytes());result['summary']=dict(status=j['status'],conditions=len(j['conditions']))
print(json.dumps(result))
