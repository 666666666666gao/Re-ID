
from datetime import datetime
from pathlib import Path
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
models=root/'trained-model'
rows=[]
for weight in sorted(models.rglob('m0_reload_probe.pth')):
    training=weight.parent/'training.json'
    item={'weight':str(weight),'bytes':weight.stat().st_size,'run_name':weight.parent.name,'training_receipt_exists':training.is_file()}
    if training.is_file():
        value=json.loads(training.read_text())
        item.update(schema=value.get('schema'),status=value.get('status'),dataset=value.get('dataset'),recipe=value.get('recipe'),variant=value.get('variant'),completed_at=value.get('completed_at'),
            probe_recorded_sha256=value.get('m0',{}).get('reload_probe_sha256'),initializer=value.get('initializer'))
    rows.append(item)
reports=[]
for folder in sorted((root/'results').iterdir()):
    path=folder/'SUMMARY.json'
    if path.is_file() and any(name in folder.name for name in ('foundation','feature_scale','semantic_native','role_input','native_research')):
        value=json.loads(path.read_text())
        reports.append({'folder':str(folder),'schema':value.get('schema'),'status':value.get('status'),'accepted':value.get('accepted'),
            'formal_epochs':value.get('formal_epochs'),'rows':[{key:item.get(key) for key in ('dataset','variant','recipe','run_dir','m0_dir','status','checkpoint_sha256','receipt_sha256')} for item in value.get('rows',[])]})
print(json.dumps({'status':'READ_ONLY_M0_WEIGHT_AND_CLOSED_REPORT_INVENTORY','at':datetime.now().astimezone().isoformat(),
 'weights':rows,'reports':reports,'total_probe_bytes':sum(row['bytes'] for row in rows),'disk_free_bytes':shutil.disk_usage(root).free,
 'boundary':'Read-only directory/receipt inventory. No weight deletion, hash changes, model import/execution, training, evaluation, GPU or power/temperature query. Current six-endpoint probes remain required until their report closes.'}))
