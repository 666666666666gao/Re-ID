from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/semantic_capacity_control_v1_20261005_856'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
accepted=json.loads((campaign/'acceptance/MSVR310_native.json').read_text())
assert accepted['status']=='FULL50_FIRST_STRICT_AND_M0_VERIFIED_BEFORE_RETIREMENT'
assert all(sha(Path(n))==d for n,d in accepted['artifact_sha256'].items())
assert not Path(accepted['probe']['path']).exists()
training=json.loads((Path(accepted['row']['run_dir'])/'training.json').read_text())
assert len(training['history'])==50
seal=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
controls=json.loads(seal.read_text())
assert len(controls['artifact_sha256'])==187
files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [
 campaign/'acceptance/MSVR310_native.json',campaign/'probe_retirement.jsonl',
 campaign/'initialization/MSVR310_native.json',campaign/'initial_forward_pair_MSVR310.json',
 *campaign.glob('MSVR310*.log'),*campaign.glob('prepare_MSVR310.log'),
 *campaign.glob('initial_forward_pair_MSVR310.log'),
 *[Path(n) for n in accepted['artifact_sha256'] if Path(n).suffix in ('.json','.jsonl','.csv','.log')]]}
print(json.dumps(dict(status='MSVR310_FULL50_FIRST_STRICT_ACCEPTED_INTAKE',at=datetime.now().astimezone().isoformat(),
 acceptance=accepted,files=files,formal_steps=sum(r['steps'] for r in training['history']),
 best_to_last_map_drop=accepted['row']['metrics']['mAP']-training['history'][-1]['official_fused']['mAP'],
 controls=[r for r in controls['rows'] if r['dataset']=='MSVR310'],disk_free_bytes=shutil.disk_usage(root).free,
 boundary='Read-only immutable accepted evidence. No NNforward, control retraining, full campaign report, current-progress/hardware query, source edit or remote checkout.')))
