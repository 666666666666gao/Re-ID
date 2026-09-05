from pathlib import Path
from datetime import datetime
import json,subprocess,hashlib
base=Path("/root/autodl-tmp/trifusion-v2/artifacts/trifusion_v23_spectral_adapter_seed42_9f4a10b")
raw=(base/"run_summary.json").read_bytes()
summary=json.loads(raw)
log=Path(str(base)+".log").read_bytes()
exitfile=Path(str(base)+".exit")
pid=44684
rows=[]
for line in log.decode().splitlines():
    if line.startswith('{"fold":') and '"epoch":' in line:rows.append(json.loads(line))
observation={"observed_at":datetime.now().astimezone().isoformat(),"original_pid":pid,
"original_process_live":Path(f"/proc/{pid}/cmdline").exists(),"status":summary["status"],
"completed_preflight_folds":len(summary["preflight"]),"m0_available":"m0" in summary,
"m0":{k:summary["m0"][k] for k in ["passed","checks"]} if "m0" in summary else None,
"completed_q1_folds":len(summary["folds"]),"q1_epoch_rows":len(rows),"latest_epoch":rows[-1] if rows else None,
"gpu":subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip(),
"exit_code":int(exitfile.read_text()) if exitfile.exists() else None,"run_summary_sha256":hashlib.sha256(raw).hexdigest(),
"run_summary_bytes":len(raw),"log_sha256":hashlib.sha256(log).hexdigest(),"log_bytes":len(log)}
out=Path(str(base)+".observation_2207.json")
assert not out.exists()
out.write_bytes((json.dumps(observation,indent=2)+"\n").encode())
if "m0" in summary:
    snap=Path(str(base)+".m0_snapshot_2207.json");assert not snap.exists();snap.write_bytes(raw)
    Path(str(base)+".m0_log_2207.log").write_bytes(log)
print(json.dumps(observation))
if exitfile.exists() and int(exitfile.read_text())!=0:print(log.decode()[-5000:])
