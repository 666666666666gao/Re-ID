from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
assert not (w/'SUPERSEDED_MSVR42_RETENTION.json').exists()
old=next(row for row in json.loads((w/'CLOSED_R2_ARTIFACTS.json').read_bytes())['rows'] if row['dataset']=='MSVR310')
new=json.loads((w/'SEED43_MSVR_TERMINAL.json').read_bytes())
cold=json.loads((w/'SEED43_MSVR_GOLD_COLD_VERIFIED.json').read_bytes())
assert cold['all_four_strictly_above_signal'] and cold['actual_metrics']['mAP']>old['metrics']['fused']['mAP']
artifact=old['artifacts']['role_best']
assert artifact['present'] and artifact['actual_sha256']==artifact['expected_sha256']
target=Path('D:/Program Files/UserCache/gb/codex_tmp/trifusion_resume_20261014/closed_legacy_r2_weights/MSVR310_R2_seed42_E24_roles_best_map.pth').resolve()
assert target.drive=='D:' and not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
options=['-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes']
copy=subprocess.run(['scp','-q',*options,'-P','2026','gaob@172.19.12.138:'+artifact['path'],str(target)],capture_output=True)
(w/'SUPERSEDED_MSVR42_COPY.json').write_text(json.dumps(dict(exit_code=copy.returncode,stderr=copy.stderr.decode()))+'\n',encoding='utf-8')
assert copy.returncode==0,copy.stderr.decode()
actual=hashlib.sha256(target.read_bytes()).hexdigest()
assert target.stat().st_size==artifact['bytes'] and actual==artifact['actual_sha256']
request=dict(old=artifact,old_metrics=old['metrics']['fused'],new=new['artifacts'],gold=cold,
    archive_path=str(target),archive_bytes=target.stat().st_size,archive_sha256=actual)
(w/'SUPERSEDED_MSVR42_ARCHIVE_VERIFIED.json').write_text(json.dumps(request,indent=2)+'\n',encoding='utf-8')
code='request='+repr(request)+'\n'+r'''
import hashlib,json,shutil
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=root/'trained-model/official_extra_seed42_MSVR310_R2_horizon50_v1_bestmap_e50_20260926/MSVR310_R2_seed42/roles_best_map.pth'
target=Path(request['old']['path']).resolve();assert target==expected.resolve()
assert target.is_relative_to((root/'trained-model').resolve())
campaign=json.loads((root/'logs/official_extra_seed42_MSVR310_R2_horizon50_v1_bestmap_e50_20260926/campaign.json').read_text())
assert campaign['status']=='COMPLETE' and campaign['seed']==42
assert json.loads((root/'logs/near_signal_seed43_launch970/MSVR310_EXIT.json').read_text())['exit_code']==0
gold_exit=json.loads((root/'logs/seed43_msvr_gold978/EXIT.json').read_text());assert gold_exit['exit_code']==0
assert not (Path('/proc')/str(gold_exit['pid'])).exists()
gold_path=root/'results/seed43_msvr_gold978/SUMMARY.json';assert sha(gold_path)==request['gold']['actual_summary_sha256']
gold=json.loads(gold_path.read_text());assert gold['all_reported_metrics_strictly_above_signal']
protected={item['path']:item['sha256'] for item in request['new'].values()}
protected[gold['distance_path']]=request['gold']['actual_distance_sha256']
assert all(sha(Path(name))==digest for name,digest in protected.items())
assert str(target) not in protected
assert target.stat().st_size==request['archive_bytes']==request['old']['bytes']
assert sha(target)==request['archive_sha256']==request['old']['actual_sha256']
before=shutil.disk_usage(root).free;target.unlink()
assert not target.exists() and all(sha(Path(name))==digest for name,digest in protected.items())
record=dict(status='EXACT_D_OLDER_MSVR42_ARCHIVED_REMOTE_DUPLICATE_RETIRED',
    retired_remote_path=str(target),retired_bytes=request['archive_bytes'],archive_path=request['archive_path'],
    archive_sha256=request['archive_sha256'],old_metrics=request['old_metrics'],selected_actual_metrics=request['gold']['actual_metrics'],
    protected_selected_artifact_sha256=protected,remote_free_before=before,remote_free_after=shutil.disk_usage(root).free,
    at=datetime.now().astimezone().isoformat(),nn_calls=0,optimizer_updates=0,running_training_queries=0,gpu_queries=0,
    restore_obligation='Restore this exact D archive to the original path before replaying original seed42 R2 weight consumers or the old residual-budget verifier. Original metrics/distances/training/history and selected seed43 best remain on server. No active RGBNT100 file was touched.')
(root/'logs/seed43_msvr_gold978/SUPERSEDED_SEED42_RETENTION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
run=subprocess.run(['ssh',*options,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code.encode(),capture_output=True)
(w/'SUPERSEDED_MSVR42_RETENTION.stdout').write_bytes(run.stdout);(w/'SUPERSEDED_MSVR42_RETENTION.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);(w/'SUPERSEDED_MSVR42_RETENTION.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))
