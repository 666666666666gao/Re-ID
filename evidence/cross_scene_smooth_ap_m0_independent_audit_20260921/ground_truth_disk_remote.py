import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
paths=[repo/'evidence/vehicle_query_protocol_labels_20260905.json',
       repo/'tools/audit_vehicle_query_protocol_labels.py',
       repo/'tools/build_msvr310_train_oof_protocol.py',
       repo/'evidence/msvr310_dataset_install_20260905.json']
records=json.loads((repo/'protocols/msvr310_train_oof_v1.json').read_bytes())
labels=json.loads(paths[0].read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert labels['script_sha256']==sha(paths[1])
assert records['builder_sha256']==sha(paths[2])
data_root=Path(records['dataset_root_at_inventory'])
inventory={}
for position,modality in enumerate(('vis','ni','th')):
    actual=sorted(str(p.relative_to(data_root)) for p in (data_root/'bounding_box_train').glob('*/'+modality+'/*.jpg'))
    expected=sorted(r['paths'][position] for r in records['records'])
    assert actual==expected
    inventory[modality]=dict(files=len(actual),all_training_names_exact=True)
print(json.dumps(dict(status='PASS_ACTUAL_TRAIN_DIRECTORY_LABEL_PROVENANCE',checked_at=datetime.now(timezone.utc).isoformat(),
    inventory=inventory,source_image_directory_entries=3096,image_content_reads=0,official_directory_reads=0,
    files={str(p):dict(bytes=p.stat().st_size,sha256=sha(p),text=p.read_bytes().decode('utf-8')) for p in paths},
    scope='The current training directory filenames equal the frozen label manifest; no image content is opened, and the historical install/CRC receipt is not rerun.'),ensure_ascii=False))
