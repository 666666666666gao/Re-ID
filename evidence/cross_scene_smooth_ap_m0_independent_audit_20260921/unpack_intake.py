import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root/'intake_01.stdout').read_text(encoding='utf-8-sig'))
for group in ('files', 'artifacts'):
    for name, row in data[group].items():
        if 'text' in row:
            path = root/'snapshots'/group/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(row['text'].encode('utf-8'))
issues = []
for r in data['bindings']:
    actual = data['files'][r['child']].get('sha256')
    if actual != r['expected']:
        issues.append(dict(**r, actual=actual))
local = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
local_differences = []
for name, row in data['files'].items():
    path = local/name
    if path.is_file():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row.get('sha256'):
            local_differences.append(dict(path=name, local=actual, remote=row.get('sha256')))
report = dict(collected_at=data['collected_at'], head=data['head'], file_count=len(data['files']),
              binding_count=len(data['bindings']), binding_mismatches=issues, local_differences=local_differences,
              artifacts={n:{k:v for k,v in r.items() if k!='text'} for n,r in data['artifacts'].items()})
(root/'intake_01_receipt.json').write_text(json.dumps(report, indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}, indent=2))
print('ARTIFACT_NAMES', sorted(data['artifacts']))
for name in ('pipeline.json','m0/summary.json','m0_cpu.json'):
    if name in data['artifacts']:
        item = json.loads(data['artifacts'][name]['text'])
        if name == 'm0/summary.json':
            item = {k:v for k,v in item.items() if k not in ('folds','overfit')}
        print(name, json.dumps(item, indent=2))
