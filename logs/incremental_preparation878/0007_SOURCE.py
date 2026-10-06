from pathlib import Path
from datetime import datetime
import json,hashlib
root=Path('/data/gaob/Re-ID/Trifusion');rows=[]
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
    p=root/'logs/training_feature_scale_protocols_20261002'/f'{dataset}.json'
    protocol=json.loads(p.read_text());mapping={};conflicts=[]
    for split,records in protocol['records'].items():
        for record in records:
            key=Path(record['paths'][0]).name
            env=record['scene'] if dataset=='MSVR310' else record['camera']
            if key in mapping and mapping[key]!=env:conflicts.append(key)
            mapping[key]=env
    log=root/'trained-model'/f'global_task_role_v1_20261004_824_full_semantic_{dataset}'/'training_batch_order.jsonl'
    with log.open() as f:first=json.loads(next(f))
    raw_paths=first['paths'];matched=[Path(n).name in mapping for n in raw_paths]
    rows.append(dict(dataset=dataset,protocol_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        basename_environment_entries=len(mapping),basename_environment_conflicts=conflicts,
        first_raw_path=raw_paths[0],first_protocol_path=protocol['records']['train'][0]['paths'][0],
        first_batch_all_paths_resolved=all(matched),logged_camera_proxy=first['cameras'][:8],
        actual_environments=[mapping[Path(n).name] for n in raw_paths[:8]] if all(matched) else None))
print(json.dumps(dict(at=datetime.now().astimezone().isoformat(),rows=rows,
    boundary='Read-only record metadata binding witness; no image/model/optimizer/GPU query. Only first saved source batch names for binding, not a repeat of full support census or evidence of loss activity.')))
