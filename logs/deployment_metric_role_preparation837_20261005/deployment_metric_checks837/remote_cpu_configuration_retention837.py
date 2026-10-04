from pathlib import Path
import argparse,hashlib,json,sys,tempfile
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root));sys.path.insert(0,str(root/'modeling'))
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==d for n,d in {'modeling/trifusion/deployment_metric_role.py': 'fa60927aa22c1af6055a0b8bb229bcd07eda41a864e8a549a53a6cb27d1e0564', 'tools/run_deployment_metric_role.py': '42b061a7426d032675d53e1c576fb5955d7be50c20b2b2a9559e15b7979a907b', 'tools/check_deployment_metric_role.py': '0beb0b82a9fd94ba47c2899015c3d11d3a094bff3c90771659c5441d92771e26', 'tools/queue_deployment_metric_role.py': '1ad600fd4fb48c976bee6d1dabdb9745a6b3da782325485c7f9798d6c425390b', 'tools/report_deployment_metric_role.py': '6fdd3d4a46cc15bb5d64020580fa86033be60b2fa46531040dae5c2cecbf0d34', 'refine-logs/deployment_metric_role_v1/EXPERIMENT_PLAN.md': '110f9d25866755316664ee582d558b636854296a8b318bd66cfdde77b3064ac7'}.items())
from tools import run_deployment_metric_role as run
run.configure()
inner=run.previous.previous.base.entry.entry
assert inner.foundation.build_core is run.build_core
assert inner.foundation.loss_values is run.loss_values
assert inner.AuthorHeadEvidence is run.previous.GlobalTaskRoleHeads
assert inner.foundation.condition is run.condition
assert inner.SCHEMA==inner.foundation.SCHEMA==run.SCHEMA
condition=inner.foundation.condition(argparse.Namespace(variant='semantic',baseline_sha256='cpu-placeholder',initialization=root/'tools/run_deployment_metric_role.py'))
assert condition['role_metric_policy']==run.METRIC_POLICY
assert condition['objective_gradient_policy']==run.previous.POLICY
print(json.dumps(dict(status='CPU_CONFIGURATION_CHAIN_PASS',condition=condition)))
from tools import queue_deployment_metric_role as panel
panel.configure()
assert panel.base.SCHEMA==panel.SCHEMA==run.SCHEMA
assert panel.base.command is panel.command and panel.base.source_map is panel.source_map
# Exercise the original strict verifier on synthetic receipts. Only initializer
# lookup/root placement are fixture seams; original verify/m0 logic is unchanged.
with tempfile.TemporaryDirectory(prefix='trifusion_metric_retention_') as tmp:
 panel.ROOT=Path(tmp);campaign=panel.ROOT/'logs/campaign';campaign.mkdir(parents=True)
 panel.base.output_dir=lambda c,p,d,v:panel.ROOT/f'trained-model/{c.name}_{p}_{v}_{d}'
 binding=dict(trainable_parameter_tensors=3,protocol_sha256='synthetic',role_metric_policy=run.METRIC_POLICY)
 panel.base.expected_binding=lambda c,d,v:binding
 d,v='RGBNT201','semantic'
 m0=panel.base.output_dir(campaign,'m0',d,v);full=panel.base.output_dir(campaign,'full',d,v)
 m0.mkdir(parents=True);full.mkdir(parents=True)
 probe=m0/'m0_reload_probe.pth';probe.write_bytes(b'synthetic verified probe')
 (full/'best_map.pth').write_bytes(b'synthetic formal best')
 (full/'official_distances.pt').write_bytes(b'synthetic independent distances')
 (full/'best_epoch_distances.pt').write_bytes(b'synthetic best distances')
 m0value=dict(schema=panel.SCHEMA,status='M0_PASS',initializer=binding,dataset=d,recipe=v,
  history=[dict(steps=8)],m0=dict(nonzero_gradient_parameters=3,trainable_parameters=3,reload_max_abs_difference=0,reload_probe_sha256=panel.base.sha(probe)),
  frozen_parameters_unchanged=True,visual_parameters_changed=True,fresh_camera_parameters_changed=True)
 panel.base.queue.write(m0/'training.json',m0value)
 metrics=dict(mAP=50.,**{'Rank-1':60.})
 condition={'variant':v,'role_metric_policy':run.METRIC_POLICY}
 training=dict(schema=panel.SCHEMA,status='BEST_OFFICIAL_MAP_TRAINING_COMPLETE',initializer=binding,condition=condition,
  history=[dict(epoch=i,official_fused=metrics) for i in range(1,51)],best_epoch=50)
 panel.base.queue.write(full/'training.json',training)
 result=dict(schema=panel.SCHEMA,status='COMPLETE',condition=condition,selected_epoch=50,metrics=metrics,training_epochs=50,seed=42,
  protocol_sha256='synthetic',checkpoint_sha256=panel.base.sha(full/'best_map.pth'),distance_sha256=panel.base.sha(full/'official_distances.pt'),
  training_best_distance_sha256=panel.base.sha(full/'best_epoch_distances.pt'),independent_upstream_metrics_equal=True,reranking=False)
 panel.base.queue.write(full/'official_metrics.json',result)
 row=panel.accept_and_retire_probe(campaign,d,v)
 assert not probe.exists() and (full/'best_map.pth').exists()
 assert panel.accepted_row(campaign,d,v)==row
 # A changed formally accepted file must be rejected rather than silently reused.
 (full/'training.json').write_text('changed')
 rejected=False
 try:panel.accepted_row(campaign,d,v)
 except AssertionError:rejected=True
 assert rejected
 print(json.dumps(dict(status='CPU_PROBE_RETENTION_CONTRACT_PASS',original_full_and_m0_verifiers_used=True,only_probe_retired=True,changed_accepted_artifact_rejected=True,boundary='Synthetic receipt/file witness only, not real M0 or formal model verification.')))
