from datetime import datetime
from pathlib import Path
import json
root=Path('/data/gaob/Re-ID/Trifusion/logs/deployment_metric_storage_five_report_launch_20261005_850')
path=root/'EXIT.json'
exit_record=json.loads(path.read_text()) if path.exists() else None
proc=Path('/proc/'+str(413519));alive=False
if proc.exists():
 stat=(proc/'stat').read_text();alive=int(stat[stat.rfind(')')+2:].split()[19])==41782336
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),supervisor_alive=alive,exit_record=exit_record,
 report_tail=(root/'report.log').read_text()[-2500:])))
