from pathlib import Path
import json
root=Path(__file__).parent
d=json.loads((root/'independent_recompute_02.json').read_bytes())
print(json.dumps({k:v for k,v in d.items() if k not in ('training','checkpoint_checks','reference_checks')},indent=2))
p=root/'check_local_bindings_claims.py';q=root/'check_local_bindings_claims_02.py'
assert not q.exists()
q.write_text(p.read_text(encoding='utf-8').replace("item['local_relative_path']","item['path']"),encoding='utf-8')
