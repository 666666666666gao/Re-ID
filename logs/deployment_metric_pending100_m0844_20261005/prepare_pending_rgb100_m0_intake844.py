"""Prepare one actual M0 intake and a measured original full50 observer."""
from pathlib import Path
import ast
private=Path('C:/Users/gb/.codex_tmp')
source=(private/'collect_deployment_metric_first_m0838.py').read_text(encoding='utf-8')
for old,new in (('deployment_metric_first_m0_milestone838','pending_rgb100_first_m0_milestone843'),
    ('deployment_metric_first_m0_intake838','pending_rgb100_semantic_m0_intake844'),
    ('deployment_metric_role_v1_20261005_837','deployment_metric_role_pending100_20261005_842'),
    ('deployment_metric_role_launch_20261005_837','deployment_metric_pending100_launch_20261005_843'),
    ('RGBNT201','RGBNT100'),('remote_first_m0_intake838','remote_pending100_m0_intake844')):
    assert old in source,old
    source=source.replace(old,new)
ast.parse(source);target=private/'collect_pending_rgb100_m0844.py';assert not target.exists();target.write_text(source,encoding='utf-8')
observer=(private/'wait_deployment_metric_native_full839.py').read_text(encoding='utf-8')
old="current=dict(active_command=json.loads((base/'deployment_metric_RGBNT201_semantic_full839/CAMPAIGN_SNAPSHOT.json').read_bytes())['campaign']['active_command'])"
assert old in observer
observer=observer.replace(old,"current=json.loads((base/'pending_rgb100_semantic_m0_intake844/SUMMARY.json').read_bytes())")
for old,new in (('deployment_metric_launch837','pending_rgb100_launch843'),
    ('deployment_metric_native_full_milestone839','pending_rgb100_semantic_full_milestone844'),
    ("=='native'","=='semantic'"),('global_task_role_native_full827','global_task_role_rgb100_semantic_full829'),
    ('global_task_role_v1_20261004_824_full_native_RGBNT201','global_task_role_v1_20261004_824_full_semantic_RGBNT100'),
    ("('RGBNT201','native','full')","('RGBNT100','semantic','full')"),('remote_native_full_observation839','remote_pending100_full_observation844')):
    assert old in observer,old
    observer=observer.replace(old,new)
ast.parse(observer);target=private/'wait_pending_rgb100_semantic_full844.py';assert not target.exists();target.write_text(observer,encoding='utf-8')
print('Prepared one actual pending100 M0 intake and one measured full50 observer.')
