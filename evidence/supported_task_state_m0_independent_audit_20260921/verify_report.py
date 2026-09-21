import ast
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
def js(name):return json.loads((root/name).read_bytes())
report=js('EXPERIMENT_AUDIT.json');local=js('local_checks.json');remote=js('remote_checks.json')
assert report['verdict']=='WARN' and report['blockers']==[]
assert report['review_independence']=='same-family' and report['acceptance_status']=='provisional'
assert report['backend_attribution_status']=='unattested'
assert len(report['checks'])==6 and len(remote['endpoints'])==8
assert report['deterministic_checks']['real_model_optimizer_steps']==sum(r['steps'] for r in remote['endpoints'])==248
assert len(js('reference_checks.json'))==90
assert all(all(row['checks'].values()) for row in js('original_m0_gate_checks.json'))
for expected,row in zip([f'fold_{f}_{a}' for f in range(3) for a in ('control','split')]+['overfit_control','overfit_split'],remote['endpoints'],strict=True):
    assert row['endpoint']==expected and Path(row['directory']).name==expected
assert (root/'trace/001-m0-audit.response.md').read_bytes()==(root/'EXPERIMENT_AUDIT.md').read_bytes()
for path in root.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
for p,digest in report['audited_input_hashes'].items():
    path=Path(p)
    if p.startswith('C:/Users/gb/.trifusion_github_publish_22c3bee') or p.startswith('C:\\Users\\gb\\.trifusion_github_publish_22c3bee'):
        relative=path.relative_to(Path('C:/Users/gb/.trifusion_github_publish_22c3bee'))
        snapshot=root/'snapshots'/relative
        if snapshot.exists():assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==digest.removeprefix('sha256:'),str(snapshot)
    elif path.is_file():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest.removeprefix('sha256:'),str(path)
validation=dict(status='PASS_REPORT_CONSISTENCY',scope='JSON/report/trace agreement, exact endpoint labels, counts, original gates, local/snapshot hash bindings and helper AST syntax; not additional model execution.',report_sha256=hashlib.sha256((root/'EXPERIMENT_AUDIT.md').read_bytes()).hexdigest(),json_sha256=hashlib.sha256((root/'EXPERIMENT_AUDIT.json').read_bytes()).hexdigest())
(root/'report_validation.json').write_text(json.dumps(validation,indent=2)+'\n',encoding='utf-8')
manifest={str(p.relative_to(root)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p.name!='output_manifest.json'}
(root/'output_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(validation,indent=2));print(json.dumps({'output_files':len(manifest)},indent=2))
