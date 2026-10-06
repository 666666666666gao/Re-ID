from pathlib import Path
from datetime import datetime
import json,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_signal_selection_reference as panel
campaign=root/'logs/signal_selection_reference_v1_20261006_870'
state=json.loads((campaign/'campaign.json').read_text());job=next(j for j in state['jobs'] if (j['dataset'],j['selection'],j['phase'])==('RGBNT100','masked','full'))
assert job['status']=='COMPLETE' and job['exit_code']==0 and len(job['steps'])==2 and all(s['exit_code']==0 for s in job['steps'])
row=panel.accepted_row(campaign,'RGBNT100','masked');assert row==job['result'] and row['formal_steps']==3129
folder=Path(row['run_dir']);t=json.loads((folder/'training.json').read_text());r=json.loads((folder/'official_metrics.json').read_text())
probe=root/'trained-model/signal_selection_reference_v1_20261006_868_m0_masked_RGBNT100/m0_reload_probe.pth'
retired=next(r for r in map(json.loads,(campaign/'retired_m0.jsonl').read_text().splitlines()) if r['path']==str(probe))
assert not probe.exists() and retired['sha256']=='615c754c4a04abdc2e937fc151a6fe2c979213b526a8bd59a88357a445b033cf' and retired['formal_receipt_sha256']==row['receipt_sha256']
files=[folder/n for n in ('training.json','official_metrics.json','training_steps.jsonl','training_batch_metadata.jsonl')]
files.extend(campaign/n for n in ('manifest.json','campaign.json','retired_m0.jsonl'))
result=dict(status='FIRST_STRICT_MASKED_RGBNT100_ACCEPTED',at=datetime.now().astimezone().isoformat(),row=row,
 best_to_last_map_drop=row['metrics']['mAP']-t['history'][-1]['official_fused']['mAP'],train_cli_seconds=(datetime.fromisoformat(job['steps'][0]['completed_at'])-datetime.fromisoformat(job['steps'][0]['started_at'])).total_seconds(),
 eval_cli_seconds=(datetime.fromisoformat(job['steps'][1]['completed_at'])-datetime.fromisoformat(job['steps'][1]['started_at'])).total_seconds(),retirement=retired,
 files={str(p.relative_to(root)):panel.sha(p) for p in files},boundary='Read-only closed endpoint intake; no NN/evaluator/report replay; same mAP-best allCMC. All_patch matched pair pending; no three-dataset selection orSOTA claim.')
print(json.dumps(result))
