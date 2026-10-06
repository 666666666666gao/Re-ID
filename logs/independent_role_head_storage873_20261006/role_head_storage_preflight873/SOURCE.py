from pathlib import Path
from datetime import datetime
import json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
for name in ('logs/independent_role_heads_launch_20261006_872','logs/independent_role_heads_v1_20261006_872','results/independent_role_heads_v1_20261006_872'):
    assert not (root/name).exists(),name
missing=[]
for p in (root/'trained-model').glob('*/best_map.pth'):
    if p.stat().st_size<200*1024**2:continue
    if not (p.parent/'official_metrics.json').exists():
        t=json.loads((p.parent/'training.json').read_text())
        missing.append(dict(path=str(p),bytes=p.stat().st_size,nlink=p.stat().st_nlink,
                            status=t['status'],dataset=t['dataset'],epochs=t['epochs']))
keeper=root/'trained-model/metric_feature_scale_20261003_v1_full_metric_raw_MSVR310'
target=root/'trained-model/signal_selection_reference_v1_20261006_867_full_global_only_MSVR310'
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,
    keeper_receipt=json.loads((keeper/'official_metrics.json').read_text()),
    target_receipt=json.loads((target/'official_metrics.json').read_text()),unaccepted_large_bests=missing,
    own_campaign_not_created=True)))
