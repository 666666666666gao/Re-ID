from pathlib import Path
import json
p=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/q1_cpu_bound_sqrt_recheck')
r={n:json.loads((p/n).read_text()) if (p/n).exists() else None for n in ['receipt.json','verification.json']}
r['log_tail']=(p/'verification.log').read_text()[-5000:]
r['worker_live']=Path('/proc/70304/cmdline').read_bytes().decode().replace('\x00',' ') if Path('/proc/70304/cmdline').exists() else None
print(json.dumps(r))
