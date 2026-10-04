"""Prepare the exact same dependency guards for six closed prompt probes."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

private = Path('C:/Users/gb/.codex_tmp')
repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
old_source = (private / 'retire_closed_visual_start_m0_fixed_posix832.py').read_text(encoding='utf-8')
tree = ast.parse(old_source)
remote = next(ast.literal_eval(node.value) for node in tree.body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'code' for t in node.targets))
old_block = """old=root/'logs/visual_start_roles_20261001_v1'
state=json.loads((old/'campaign.json').read_text())
analysis=json.loads((old/'analysis_waiter_status.json').read_text())
assert state['status']=='COMPLETE' and len(state['jobs'])==6
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert analysis['status']=='CPU_REPORT_COMPLETE' and analysis['exit_code']==0
assert analysis['complete_parent_jobs']==analysis['expected_jobs']==6"""
new_block = """old=root/'logs/prompt_role_state_20260930'
state=json.loads((old/'campaign.json').read_text())
matrix=json.loads(report.read_text())
assert state['status']=='COMPLETE' and len(state['jobs'])==6
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert matrix['verified_complete']==matrix['expected_endpoints']==6
assert len(matrix['rows'])==6 and all(r['status']=='VERIFIED_COMPLETE' for r in matrix['rows'])
for target in targets:
 child=json.loads((Path(target['campaign_dir'])/'campaign.json').read_text())
 assert child['status']=='COMPLETE' and len(child['jobs'])==3
 assert [j['mode'] for j in child['jobs']]==['m0','train','evaluate']
 assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in child['jobs'])"""
assert old_block in remote
remote = remote.replace(old_block, new_block)
for old, new in (
    ("results/visual_start_roles_complete_20261001/SUMMARY.json", "logs/prompt_role_state_20260930/accepted_matrix.json"),
    ("visual_start_roles_20261001_v1_visual_start_", "prompt_role_state_20260930_prompt_"),
    ("logs/visual_start_m0_retirement832_20261004", "logs/prompt_m0_retirement834_20261005"),
    ("SIX_CLOSED_VISUAL_START_M0_PROBES_RETIRED", "SIX_CLOSED_PROMPT_M0_PROBES_RETIRED"),
    ("CPU report completed 2026-10-01", "complete-gallery verification completed 2026-09-30"),
):
    assert old in remote, old
    remote = remote.replace(old, new)
header = '''"""Retire only the six exact successful, closed prompt M0 reload probes."""
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
raw = Path('C:/Users/gb/.codex_tmp/prompt_role_state_20260930')
base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = base / 'closed_prompt_m0_retirement834'
assert not packet.exists()
summary_path = repo / 'logs/prompt_role_state_20260930/accepted_matrix.json'
summary = json.loads(summary_path.read_bytes())
assert summary['verified_complete'] == summary['expected_endpoints'] == len(summary['rows']) == 6
assert {(r['dataset'], r['variant']) for r in summary['rows']} == {
    (d, v) for d in ('RGBNT201', 'RGBNT100', 'MSVR310') for v in ('reset_roles', 'carry_roles')}
targets = []
expected_artifacts = {}
def bind(remote_path, local_path):
    assert local_path.is_file()
    expected_artifacts[str(remote_path)] = hashlib.sha256(local_path.read_bytes()).hexdigest()
root = PurePosixPath('/data/gaob/Re-ID/Trifusion')
for row in summary['rows']:
    assert row['status'] == 'VERIFIED_COMPLETE'
    full = PurePosixPath(row['run_dir'])
    m0 = full.with_name(full.name.removesuffix('_full') + '_m0')
    receipt_path = raw / 'trained-model' / m0.name / 'training.json'
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt['status'] == 'M0_PASS' and receipt['m0']['reload_max_abs_difference'] == 0
    assert receipt['m0']['nonzero_gradient_parameters'] == receipt['m0']['trainable_parameters']
    targets.append(dict(path=str(m0 / 'm0_reload_probe.pth'), sha256=receipt['m0']['reload_probe_sha256'],
                        receipt=str(m0 / 'training.json'), full_dir=str(full), campaign_dir=row['campaign_dir'],
                        checkpoint_sha256=row['checkpoint_sha256'], distance_sha256=row['distance_sha256'],
                        receipt_sha256=row['receipt_sha256']))
    bind(m0 / 'training.json', receipt_path)
    for name in ('training.json', 'official_metrics.json'):
        bind(full / name, raw / 'trained-model' / full.name / name)
    child = PurePosixPath(row['campaign_dir']) / 'campaign.json'
    bind(child, repo / child.relative_to(root).as_posix())
for name in ('campaign.json', 'accepted_matrix.json', 'paired_RGBNT201.json', 'paired_RGBNT100.json', 'paired_MSVR310.json'):
    p = root / 'logs/prompt_role_state_20260930' / name
    bind(p, repo / p.relative_to(root).as_posix())
p = root / 'results/PROMPT_ROLE_STATE_FULL_2026-09-30.md'
bind(p, repo / p.relative_to(root).as_posix())
'''
footer = old_source[old_source.index('for marker, value in '):]
footer = footer.replace('remote_retire_closed_visual_start_m0.py', 'remote_retire_closed_prompt_m0.py')
source = header + '\ncode = ' + repr(remote) + '\n' + footer
ast.parse(source)
target = private / 'retire_closed_prompt_m0_834.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
print(json.dumps(dict(status='EXACT_CLOSED_PROMPT_RETIREMENT_HELPER_PREPARED',
    helper=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest(), targets=6)))
