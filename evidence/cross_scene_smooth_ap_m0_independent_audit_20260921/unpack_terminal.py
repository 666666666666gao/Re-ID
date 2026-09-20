import json
from pathlib import Path
root = Path(__file__).resolve().parent
data = json.loads((root/'terminal_intake.stdout').read_text(encoding='utf-8-sig'))
for name,row in data['files'].items():
    if 'text' in row:
        p=root/'terminal_text'/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(row['text'].encode('utf-8'))
summary=json.loads(data['files']['m0/summary.json']['text'])
cpu=json.loads(data['files']['m0_cpu.json']['text'])
print(json.dumps(dict(collected_at=data['collected_at'],status=summary['status'],optimizer_steps=summary['optimizer_steps'],
                     cpu_status=cpu['status'],cpu_steps=cpu['checked_training_steps'],
                     overfit={e:r['gate'] for e,r in summary['overfit'].items()}),indent=2))
deps=json.loads((root/'static_dependencies.stdout').read_text(encoding='utf-8-sig'))
path='/root/autodl-tmp/trifusion-v2/artifacts/msvr310_signal_source_oof_v1_seed42_bb01d60/baseline/summary.json'
base=json.loads(deps['files'][path]['text'])
print('BASELINE_KEYS',list(base))
print('BASELINE_FOLD_KEYS',list(base['folds'][0]))
print('BASELINE_TRAIN_KEYS',list(base['folds'][0]['training']))
print('BASELINE_FIRST_STEP',json.dumps(base['folds'][0]['training']['steps'][0]))
