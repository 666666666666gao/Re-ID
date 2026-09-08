from pathlib import Path
import json,os,shutil,subprocess
from datetime import datetime
r=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
p=json.loads((r/'pipeline.json').read_bytes())
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),pipeline=p,processes={str(pid):Path('/proc',str(pid)).exists() for pid in [p['wrapper_pid'],*[s['original_pid'] for s in p['stages']]]},free_bytes=shutil.disk_usage(r).free,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True),logs={f.name:f.read_text()[-7000:] for f in r.glob('*.log')},receipts={f.name:json.loads(f.read_bytes()) for f in r.glob('*.json') if f.name!='pipeline.json'}),ensure_ascii=False))
