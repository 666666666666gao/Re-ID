from pathlib import Path
import json,shutil
from datetime import datetime
r=Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4');p=json.loads((r/'pipeline.json').read_bytes());s=p['stages'][-1]
q=json.loads((r/'q1/summary.json').read_bytes());events=[json.loads(x) for x in (r/'q1.log').read_text().splitlines() if x.startswith('{"event":')]
ends=[dict(fold=f['fold'],endpoint=e,steps=v['training']['optimizer_steps'],checkpoint_exists=Path(v['checkpoint']).exists(),checks=v['engineering_checks']) for f in q['folds'] for e,v in f['endpoints'].items() if 'checkpoint' in v]
print(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),status=p['status'],stage=s['stage'],wrapper_pid=p['wrapper_pid'],wrapper_live=Path('/proc',str(p['wrapper_pid'])).exists(),stage_pid=s['original_pid'],stage_live=Path('/proc',str(s['original_pid'])).exists(),complete_endpoints=ends,latest_events=events[-2:],free_bytes=shutil.disk_usage(r).free),ensure_ascii=False))
