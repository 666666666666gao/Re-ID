from pathlib import Path
import datetime,json,shutil
root=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_source_coverage_seed42_f7a0590')
original=json.loads(Path(str(root)+'_pipeline.json').read_bytes())
verification=json.loads(Path(str(root)+'_verify_pipeline.json').read_bytes())
result=dict(observed_at=datetime.datetime.now().astimezone().isoformat(),original=original,verification=verification,
    live={str(p):Path('/proc',str(p)).exists() for p in [original['pid'],verification['pid']]+([verification['child_pid']] if 'child_pid' in verification else [])},
    free_bytes=shutil.disk_usage(root.parent).free)
path=root/'coverage_verification.json'
if path.exists():
    j=json.loads(path.read_bytes());result['verified_progress']={k:j[k] for k in ('status','candidate_rows','full_rows','max_ap_error')}
    result['verified_progress']['conditions']=j['conditions']
result['log_tail']=Path(str(root)+'_verify.log').read_text()[-1800:] if Path(str(root)+'_verify.log').exists() else None
print(json.dumps(result))
