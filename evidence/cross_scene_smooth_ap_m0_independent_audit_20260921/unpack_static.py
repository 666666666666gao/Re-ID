import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parent
data = json.loads((root/'static_dependencies.stdout').read_text(encoding='utf-8-sig'))
snapshots = {}
for index, (name, row) in enumerate(data['files'].items()):
    if 'text' in row:
        p = root/'snapshots/supplemental'/f'{index:03d}_{Path(name).name}'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(row['text'], encoding='utf-8', newline='')
        snapshots[name] = str(p.relative_to(root))
problems = [dict(**r,actual=data['files'][r['child']]['sha256']) for r in data['bindings']
            if r['expected'] != data['files'][r['child']]['sha256']]
baseline_problems = [r for r in data['baseline_bindings'] if r['expected'] != r['actual']]
result = dict(collected_at=data['collected_at'], unique_files=len(data['files']),
              binding_count=len(data['bindings']), binding_mismatches=problems,
              baseline_bindings=data['baseline_bindings'], baseline_mismatches=baseline_problems,
              text_snapshots=snapshots,
              text_snapshot_note='Raw hashes are remote file bytes; this supplemental text capture normalizes line endings.')
(root/'static_bindings_receipt.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='text_snapshots'},indent=2))
