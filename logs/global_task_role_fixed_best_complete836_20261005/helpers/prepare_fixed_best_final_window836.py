import ast,json
from pathlib import Path

private=Path('C:/Users/gb/.codex_tmp')
base=private/'independent_evidence_draft'
observation=json.loads((base/'global_task_role_fixed_best_milestone835/stdout.json').read_bytes())
assert observation['status']=='RUNNING' and observation['exit'] is None
assert len(observation['jobs'])==5 and sum(j['status']=='COMPLETE' for j in observation['jobs'])==4
assert observation['jobs'][-1]['dataset']=='RGBNT100' and observation['jobs'][-1]['variant']=='semantic'
source=(private/'wait_global_task_role_fixed_best835.py').read_text(encoding='utf-8')
for old,new in [('global_task_role_fixed_best_milestone835','global_task_role_fixed_best_final_window836'),('timedelta(minutes=9)','timedelta(minutes=14)'),('estimated8–12min end','revised final window after observed4/6complete')]:
    assert old in source
    source=source.replace(old,new)
ast.parse(source)
target=private/'wait_global_task_role_fixed_best_final_window836.py'
assert not target.exists()
target.write_text(source,encoding='utf-8')
print('Final window:01:41:32; original launch unchanged, one observation only.')
