$ErrorActionPreference='Stop'
& scp -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i C:/Users/gb/.ssh/id_ed25519 -P 2026 C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903/READONLY_DEPLOYMENT_918.zip gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/logs/region_use_remaining_strict_admin912/READONLY_DEPLOYMENT_918.zip
if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}
$code=@'
import hashlib,json,os,shutil,subprocess,zipfile,time
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
admin=root/'logs/region_use_fixed_best_diagnosis_admin918'
output=root/'logs/region_use_fixed_best_diagnosis_20261013_918'
assert not admin.exists() and not output.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=root/'logs/region_use_remaining_strict_admin912/READONLY_DEPLOYMENT_918.zip'
with zipfile.ZipFile(archive) as z:
 pins=json.loads(z.read('FILE_DIGESTS.json'))
 for name,h in pins.items():
  target=root/name;assert target.resolve().is_relative_to(root.resolve())
  assert not target.exists()
  value=z.read(name);assert hashlib.sha256(value).hexdigest()==h
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(value)
 assert all(sha(root/name)==h for name,h in pins.items())
seal_path=root/'refine-logs/region_use_fixed_best_diagnosis918/INPUT_SEAL.json'
seal=json.loads(seal_path.read_text())
assert len(seal['source_sha256'])==465 and len(seal['artifact_sha256'])==90
for name,h in seal['source_sha256'].items():assert sha(root/name)==h,name
for name,h in seal['artifact_sha256'].items():assert sha(Path(name))==h,name
gpu=subprocess.check_output(['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert len(gpu.splitlines())==2 and all(int(line.split(',')[1])<500 for line in gpu.splitlines())
free=shutil.disk_usage(root).free;assert free>=seal['storage_required_bytes']
python='/data/gaob/Re-ID/conda-envs/tri_reid/bin/python'
versions=json.loads(subprocess.check_output([python,'-B','-c','import json;from importlib.metadata import version;print(json.dumps({p:version(p) for p in ("torch","numpy","mamba-ssm","timm")}))'],text=True))
assert versions['torch']=='2.5.1+cu121' and versions['numpy']=='1.24.4'
assert versions['mamba-ssm']=='2.2.6.post3' and versions['timm']=='1.0.15'
admission=dict(status='READ_ONLY_FIXED_BEST_ADMISSION_PASSED',at=datetime.now().astimezone().isoformat(),
 input_seal_sha256=sha(seal_path),source_checks=465,artifact_checks=90,gpu01_memory=gpu,free_bytes=free,
 required_bytes=seal['storage_required_bytes'],environment_versions=versions,environment_rebuild=False,training=False,
 basis='Existing user failure-diagnostic authorization; finish-current-and-stop preserved for training. No unreceived scope response represented as approval.')
(admin/'ADMISSION.json').write_text(json.dumps(admission,indent=2)+'\n')
supervisor=admin/'supervisor.py'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',PYTHONDONTWRITEBYTECODE='1')
with (admin/'supervisor.log').open('x') as log:
 process=subprocess.Popen(['/usr/bin/python3',str(supervisor)],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(admin/'DEPLOY_LAUNCH.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(),supervisor_pid=process.pid,
 supervisor_sha256=sha(supervisor),files=pins,admission_sha256=sha(admin/'ADMISSION.json')),indent=2)+'\n')
time.sleep(1)
assert Path('/proc/'+str(process.pid)).exists()
print(json.dumps(dict(status='READ_ONLY_DIAGNOSIS918_SUPERVISOR_LIVE',supervisor_pid=process.pid,
 at=datetime.now().astimezone().isoformat(),admission=admission,root=root.as_posix(),output=output.as_posix())))
archive.unlink()
'@
$result = $code | & ssh -o ProxyCommand=none -o ProxyJump=none -o BatchMode=yes -i C:/Users/gb/.ssh/id_ed25519 -p 2026 gaob@172.19.12.138 /usr/bin/python3 - 2>&1
$status=$LASTEXITCODE
$work='C:/Users/gb/.codex_tmp/independent_evidence_draft/evidence_comparison903'
[System.IO.File]::WriteAllText("$work/READONLY_DEPLOYMENT_918.log",($result -join "`n")+"`n",[System.Text.UTF8Encoding]::new($false))
Write-Output $result
exit $status
