from pathlib import Path
from datetime import datetime
import hashlib
import json

private=Path('C:/Users/gb/.codex_tmp')
output=private/'training_feature_scale_completed_pairs753'
assert not output.exists()
rows=[]
for dataset,normalized,raw in (('RGBNT201','first_full749','fourth_full753'),('MSVR310','second_full751','third_full752')):
    roots=[private/f'training_feature_scale_{label}' for label in (normalized,raw)]
    receipts=[json.loads((root/'INTAKE.json').read_bytes()) for root in roots]
    assert all(row['dataset']==dataset and row['training_epochs']==50 for row in receipts)
    training=[json.loads((root/'trained_model/training.json').read_bytes()) for root in roots]
    initial=[]
    for record in training:
        fields=dict(record['initializer'])
        fields.pop('recipe')
        fields.pop('training_feature_scaling')
        initial.append(fields)
    assert initial[0]==initial[1]
    paths=[root/'trained_model/training_batch_order.jsonl' for root in roots]
    raw_bytes=[path.read_bytes() for path in paths]
    assert raw_bytes[0]==raw_bytes[1],dataset
    assert len(raw_bytes[0].splitlines())==receipts[0]['formal_steps']==receipts[1]['formal_steps']
    rows.append({'dataset':dataset,'initial_fields_equal_except_recipe_and_scaling':True,
        'full50_actual_batch_order_bytes_equal':True,'steps':receipts[0]['formal_steps'],
        'batch_order_sha256':hashlib.sha256(raw_bytes[0]).hexdigest(),
        'primary_intakes':[str(root/'INTAKE.json') for root in roots],
        'metrics':{row['variant']:row['metrics'] for row in receipts},
        'delta_raw_minus_normalized':{name:receipts[1]['metrics'][name]-receipts[0]['metrics'][name] for name in receipts[0]['metrics']}})
record={'status':'TWO_COMPLETED_FULL50_PAIRS_EXACT_BATCH_ORDER_CHECKED',
    'checked_at':datetime.now().astimezone().isoformat(),'rows':rows,
    'boundary':'CPU checks of already completed exact text only, not original final six-arm report or independent semantic audit. No pixel augmentation equivalence, new-seed stability, whole F1 causality, role contribution or SOTA claim.'}
output.mkdir()
(output/'PAIRED_ORDER_CHECK.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
print(json.dumps(record,indent=2))
