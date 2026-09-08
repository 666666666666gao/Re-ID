"""Verify final audit file structure, references and trace identity."""
import ast
import hashlib
import json
from pathlib import Path
import re
OUT=Path(__file__).resolve().parent
ROOT=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
report=(OUT/'EXPERIMENT_AUDIT.md').read_text(encoding='utf-8')
bad=[];refs=[]
for name,start,end in re.findall(r'([A-Za-z0-9_./-]+\.(?:py|json|jsonl|md|log)):(\d+)(?:-(\d+))?',report):
    p=OUT/name
    if not p.is_file():p=ROOT/name
    if not p.is_file():bad.append(dict(reference=f'{name}:{start}-{end}',reason='missing'));continue
    n=len(p.read_text(encoding='utf-8-sig').splitlines())
    if not (1<=int(start)<=int(end or start)<=n):bad.append(dict(reference=f'{name}:{start}-{end}',lines=n))
    refs.append(dict(path=str(p),line_start=int(start),line_end=int(end or start),file_lines=n))
data=json.loads((OUT/'EXPERIMENT_AUDIT.json').read_bytes())
assert data['verdict']=='WARN' and data['engineering_status']=='PASS' and data['deterministic_verification']['status']=='PASS'
assert data['review_independence']=='same-family' and data['acceptance_status']=='provisional'
assert data['verdict_id']=='sha256:'+hashlib.sha256((OUT/'EXPERIMENT_AUDIT.md').read_bytes()).hexdigest()
assert (OUT/'reviewer_full_response.md').read_bytes()==(OUT/'trace/001-independent-audit.response.md').read_bytes()
scripts=[]
for p in OUT.glob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));scripts.append(p.name)
result=dict(status='PASS' if not bad else 'FAIL_REFERENCE_RANGE',reference_count=len(refs),bad_references=bad,ast_valid_scripts=scripts,audited_input_hashes=len(data['audited_input_hashes']),reviewer_response_trace_identical=True,report_sha256=data['verdict_id'],output_files={p.name:p.stat().st_size for p in OUT.iterdir() if p.is_file()})
(OUT/'artifact_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='output_files'},indent=2))
