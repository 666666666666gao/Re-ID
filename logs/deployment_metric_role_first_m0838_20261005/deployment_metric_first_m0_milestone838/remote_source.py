from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');campaign=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_v1_20261005_837');parent=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_role_launch_20261005_837')
pid=3606472;proc=Path('/proc/'+str(pid))
alive=False
if proc.exists():
 stat=(proc/'stat').read_text();alive=int(stat[stat.rfind(')')+2:].split()[19])==39300650
state=json.loads((campaign/'campaign.json').read_text()) if (campaign/'campaign.json').exists() else None
files={}
for name in ['manifest.json','campaign.json','initialization/RGBNT201_semantic.json','prepare_RGBNT201_semantic.log','RGBNT201_semantic_m0.log']:
 path=campaign/name
 if path.is_file():files[str(path.relative_to(root))]=dict(text=path.read_text(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
for path in [parent/'EXIT.json',parent/'console.log',root/'trained-model/deployment_metric_role_v1_20261005_837_m0_semantic_RGBNT201/training.json']:
 if path.is_file():files[str(path.relative_to(root))]=dict(text=path.read_text(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),supervisor_alive_matching_startticks=alive,campaign_state=state,files=files,disk_free_bytes=shutil.disk_usage(root).free,power_temperature_queried=False)))
