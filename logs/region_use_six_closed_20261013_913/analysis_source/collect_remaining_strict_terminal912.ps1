$ErrorActionPreference='Stop'
$script=@"
import hashlib,json,os,subprocess,zipfile
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/region_use_remaining_strict_20261013_912/region_use_remaining_20261013_911'
admin=root/'logs/region_use_remaining_strict_admin912'
state=json.loads((campaign/'campaign.json').read_text())
terminal=json.loads((admin/'EXIT.json').read_text())
assert state['status']=='TRAINING_AND_STRICT_EVALUATION_COMPLETE' and terminal['exit_code']==0
assert len(state['jobs'])==1
job=state['jobs'][0]
assert (job['dataset'],job['comparison'],job['status'])==('RGBNT100','ambiguity','COMPLETE')
reporter=root/'tools/report_region_use_endpoint.py'
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(reporter)=='807746c6ab295fc39de40bd3a91028c57dc7d346f89e1351d803d5ac401a314d'
out=root/'logs/region_use_remaining_endpoint_20261013_912/RGBNT100_ambiguity'
receipt=admin/'CPU_REPORT_912.json'
assert not receipt.exists() and not out.exists()
cmd=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(reporter),
 '--campaign',str(campaign),'--dataset','RGBNT100','--comparison','ambiguity','--output-dir',str(out)]
value=dict(status='RUNNING',started_at=datetime.now().astimezone().isoformat(),
 command=cmd,reporter_sha256=sha(reporter),model_inference=0,optimizer_updates=0)
receipt.write_text(json.dumps(value,indent=2)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
log_path=admin/'CPU_REPORT_912.log'
with log_path.open('x') as log:
 run=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
value.update(exit_code=run.returncode,completed_at=datetime.now().astimezone().isoformat(),
 status='COMPLETE' if run.returncode==0 else 'FAILED')
receipt.write_text(json.dumps(value,indent=2)+'\n')
if run.returncode:
 print(json.dumps(value));print(log_path.read_text());raise SystemExit(run.returncode)
summary=json.loads((out/'SUMMARY.json').read_text())
assert summary['status']=='ONE_ORIGINAL_ENDPOINT_CPU_DIAGNOSIS_COMPLETE'
run_dir=Path(summary['candidate']['run_dir'])
files=[campaign/'campaign.json',admin/'LAUNCH.json',admin/'EXIT.json',
 root/'logs/region_use_remaining_admin911/NEAR_END_NODE_912.json',receipt,log_path,out/'SUMMARY.json',out/'REPORT.md',
 root/'logs/region_use_remaining_20261013_911/acceptance/RGBNT100_ambiguity_initialization.json',
 root/'logs/region_use_remaining_20261013_911/acceptance/RGBNT100_ambiguity_m0.json']
files += [root/'logs/region_use_remaining_20261013_911/campaign.json',root/'logs/region_use_remaining_admin911/EXIT.json',root/'logs/region_use_remaining_admin911/TERMINAL_STORAGE_912.json',root/'logs/region_use_remaining_admin911/CLEANUP_912.json']
files += [run_dir/name for name in ('training.json','training_steps.jsonl','training_batch_order.jsonl','official_metrics.json')]
manifest=dict(created_at=datetime.now().astimezone().isoformat(),
 files={str(p.relative_to(root)):sha(p) for p in files},
 pinned_binaries={str(run_dir/'best_map.pth'):summary['candidate']['checkpoint_sha256'],
 str(run_dir/'official_distances.pt'):summary['candidate']['distance_sha256']},
 boundary='Only original remaining911 accepted endpoint; five previous reports not rerun. CPU saved distances only; no NN, new weight, or checkpoint reselection.')
archive=admin/'SIXTH_RESULTS_912.zip'
assert not archive.exists()
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(root)))
 z.writestr('INTAKE_MANIFEST_912.json',json.dumps(manifest,indent=2)+'\n')
meta=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,files=len(files))
meta_path=admin/'SIXTH_RESULTS_912.meta.json'
assert not meta_path.exists();meta_path.write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(dict(status=value['status'],archive=meta,metrics=summary['metrics'],
 best_epoch=summary['candidate']['best_epoch'],last_map=summary['history'][-1]['official_fused']['mAP'],
 pairs=[dict(control=p['control'],delta=p['delta_metrics'],repairs=p['rank1_repairs'],
 new_errors=p['rank1_new_errors']) for p in summary['pairs']])),flush=True)
"@
$script | & ssh -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i C:/Users/gb/.ssh/id_ed25519 -p 2026 gaob@172.19.12.138 /usr/bin/python3 -
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
& scp -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i C:/Users/gb/.ssh/id_ed25519 -P 2026 gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/logs/region_use_remaining_strict_admin912/SIXTH_RESULTS_912.meta.json C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/SIXTH_RESULTS_912.meta.json
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
& scp -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i C:/Users/gb/.ssh/id_ed25519 -P 2026 gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/logs/region_use_remaining_strict_admin912/SIXTH_RESULTS_912.zip C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/SIXTH_RESULTS_912.zip
exit $LASTEXITCODE
