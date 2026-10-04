"""Receive current MSVR semantic M0 and schedule its original full50 milestone."""
from pathlib import Path
import ast

private=Path('C:/Users/gb/.codex_tmp')
source=(private/'collect_deployment_metric_first_m0838.py').read_text(encoding='utf-8')
start=source.index("milestone=base/")
end=source.index("packet=base/",start)
source=source[:start]+source[end:]
for old,new in (('deployment_metric_first_m0_intake838','deployment_metric_msvr_semantic_m0_intake841'),('RGBNT201','MSVR310'),('remote_first_m0_intake838','remote_msvr_m0_intake841')):
    assert old in source
    source=source.replace(old,new)
ast.parse(source)
target=private/'collect_deployment_metric_msvr_m0841.py'
assert not target.exists()
target.write_text(source,encoding='utf-8')

observer=(private/'wait_deployment_metric_native_full839.py').read_text(encoding='utf-8')
old="current=dict(active_command=json.loads((base/'deployment_metric_RGBNT201_semantic_full839/CAMPAIGN_SNAPSHOT.json').read_bytes())['campaign']['active_command'])"
assert old in observer
observer=observer.replace(old,"current=json.loads((base/'deployment_metric_msvr_semantic_m0_intake841/SUMMARY.json').read_bytes())")
for old,new in (
    ('deployment_metric_native_full_milestone839','deployment_metric_msvr_semantic_full_milestone841'),
    ("=='native'","=='semantic'"),
    ('global_task_role_native_full827','global_task_role_msvr_semantic_full828'),
    ('global_task_role_v1_20261004_824_full_native_RGBNT201','global_task_role_v1_20261004_824_full_semantic_MSVR310'),
    ("('RGBNT201','native','full')","('MSVR310','semantic','full')"),
    ('remote_native_full_observation839','remote_msvr_semantic_full_observation841')):
    assert old in observer
    observer=observer.replace(old,new)
ast.parse(observer)
target=private/'wait_deployment_metric_msvr_semantic_full841.py'
assert not target.exists()
target.write_text(observer,encoding='utf-8')
print('Prepared one current M0 intake and one measured full50 observer; no model launch.')
