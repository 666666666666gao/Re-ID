"""Schedule the next existing full50 job from a measured matched-control duration."""
from pathlib import Path
import ast
import hashlib
import json

private = Path('C:/Users/gb/.codex_tmp')
source = (private / 'wait_deployment_metric_first_full838.py').read_text(encoding='utf-8')
old = "current=json.loads((base/'deployment_metric_first_m0_intake838/SUMMARY.json').read_bytes())"
new = "current=dict(active_command=json.loads((base/'deployment_metric_RGBNT201_semantic_full839/CAMPAIGN_SNAPSHOT.json').read_bytes())['campaign']['active_command'])\nassert current['active_command']['mode']=='train' and current['active_command']['status']=='RUNNING'\nassert current['active_command']['command'][current['active_command']['command'].index('--variant')+1]=='native'"
replacements = (
    ("packet=base/'deployment_metric_first_full_milestone838'", "packet=base/'deployment_metric_native_full_milestone839'"),
    (old, new),
    ("previous=json.loads((base/'global_task_role_first_full826/received/trained-model/global_task_role_v1_20261004_824_full_semantic_RGBNT201/training.json').read_bytes())",
     "previous=json.loads((base/'global_task_role_native_full827/stdout.json').read_bytes())['files']['trained-model/global_task_role_v1_20261004_824_full_native_RGBNT201/training.json'];previous=json.loads(previous)"),
    ("('RGBNT201','semantic','full')", "('RGBNT201','native','full')"),
    ('remote_first_full_observation838.py', 'remote_native_full_observation839.py'))
for old, new in replacements:
    assert old in source, old
    source = source.replace(old, new)
ast.parse(source)
target = private / 'wait_deployment_metric_native_full839.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
print(json.dumps(dict(status='NEXT_NATIVE_FULL_OBSERVER_PREPARED_NOT_STARTED', helper=str(target), sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
