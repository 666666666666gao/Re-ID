import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
assert not (w/'SEED44_RGBNT100_ADMISSION.json').exists()
row=next(row for row in json.loads((w/'CLOSED_R2_INTAKE.json').read_bytes())['rows'] if row['dataset']=='RGBNT100' and row['seed']==42)
pins=json.loads((w/'seed43_m0/logs/near_signal_seed43_launch970/SOURCE_SEAL.json').read_bytes())
request=dict(metrics_path=row['metrics_path'],metrics_sha256=row['metrics_sha256'],pins=pins)
code='request='+repr(request)+'\n'+r'''
from datetime import datetime
import hashlib,json,shutil,subprocess
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(sha(root/name)==digest for name,digest in request['pins'].items())
path=Path(request['metrics_path']);assert sha(path)==request['metrics_sha256']
receipt=json.loads(path.read_text());assert receipt['status']=='COMPLETE' and receipt['seed']==42 and receipt['training_epochs']==50
signal=receipt['outputs']['baseline_only']['metrics'];fused=receipt['outputs']['fused']['metrics']
assert fused['mAP']>signal['mAP'] and fused['Rank-1']==signal['Rank-1']
assert not all(fused[key]>value for key,value in signal.items())
assert json.loads((root/'logs/seed43_msvr_gold978/EXIT.json').read_text())['exit_code']==0
gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
index,memory=map(int,gpu.strip().split(','));assert index==1 and memory<500,(index,memory)
free=shutil.disk_usage(root).free;assert free>3*1024**3,free
assert not (root/'logs/near_signal_rgbnt100_seed44_launch982').exists()
print(json.dumps(dict(status='READY_FOR_ONE_NEAR_ONLY_SEED44',at=datetime.now().astimezone().isoformat(),
    dataset='RGBNT100',gpu=1,gpu_memory_used_mib=memory,free_disk_bytes=free,signal_metrics=signal,original_seed42_metrics=fused,
    source_seal_count=len(request['pins']),qualification_receipt_sha256=sha(path),
    boundary='Only free GPU1 memory admission and closed seed42 receipt/code checked. No GPU0/current seed43 epoch or metric/temperature/power query; no inference or launch in this preflight.')))
'''
run=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code.encode(),capture_output=True)
(w/'SEED44_RGBNT100_ADMISSION.stdout').write_bytes(run.stdout);(w/'SEED44_RGBNT100_ADMISSION.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);(w/'SEED44_RGBNT100_ADMISSION.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))
