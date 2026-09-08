from pathlib import Path
import json,hashlib,time
p=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_positive_derivatives_q1_20260909')
r=json.loads((p/'running.json').read_bytes())
print(json.dumps(dict(observed_at=time.time(),running=r,live={str(pid):Path('/proc',str(pid)).exists() for pid in [r['wrapper_pid'],r['pid']]},exit=json.loads((p/'exit.json').read_bytes()) if (p/'exit.json').exists() else None,files={x.name:dict(bytes=x.stat().st_size,sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in p.iterdir() if x.is_file()})))
