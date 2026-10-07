from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');rows=[]
for proc in Path('/proc').iterdir():
    if proc.name.isdigit() and (proc/'cmdline').is_file():
        assert not any(w.startswith(b'/data/gaob/Re-ID/Trifusion/tools/') for w in (proc/'cmdline').read_bytes().split(bytes([0])))
for folder in (root/'trained-model').iterdir():
    if folder.is_dir() and folder.name.startswith(('foundation_recipe','training_feature_scale','metric_feature_scale')):
        weights=[dict(name=p.name,bytes=p.stat().st_size) for p in folder.glob('*.pth') if p.is_file()]
        if weights:
            receipt=json.loads((folder/'official_metrics.json').read_text()) if (folder/'official_metrics.json').is_file() else None
            rows.append(dict(folder=str(folder),weights=weights,metrics=receipt['metrics'] if receipt else None))
print(json.dumps(dict(status='READONLY_CLOSED_FOUNDATION_WEIGHT_INVENTORY',at=datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,rows=rows)))
