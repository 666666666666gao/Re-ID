from pathlib import Path
import datetime,json,shutil,subprocess
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
status=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
live={str(p):Path('/proc',str(p)).exists() for p in [status['pid']]+[r['pid'] for r in status['stages']]}
result=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),pipeline=status,proc_exists=live,
    free_bytes=shutil.disk_usage(root.parent).free,
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip())
for name in ('summary.json','coverage.json'):
    if (root/name).exists():
        j=json.loads((root/name).read_bytes());result[name]=dict(status=j['status'],conditions=len(j['conditions']),
            latest=j['conditions'][-1]['directory'] if j['conditions'] else None)
for stage in status['stages']:
    p=Path(str(root)+'_'+stage['stage']+'.log')
    result[stage['stage']+'_log_tail']=p.read_text()[-2000:]
print(json.dumps(result))
