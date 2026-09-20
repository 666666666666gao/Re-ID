"""Local text-only census; no tensor/array/model inputs."""
from pathlib import Path
from collections import Counter
import json, hashlib
ROOT=Path(__file__).parent
REPO='/root/autodl-tmp/trifusion-v2/TriFusion-ReID'
RUN='/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639'
def read_remote(path):
    return json.loads((ROOT/'remote_text'/path.lstrip('/')).read_text(encoding='utf-8'))
raw=json.loads((ROOT/'remote_intake_02.json').read_text(encoding='utf-8'))
base=read_remote(REPO+'/configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
signal=read_remote(REPO+'/'+base['BASELINE']['CONFIG'])
protocol=read_remote(REPO+'/'+signal['protocol'])
records=protocol['records']
summary=read_remote(RUN+'/m0/summary.json')
runrows=[]
for folder in sorted((ROOT/'remote_text'/RUN.lstrip('/')/'m0').iterdir()):
    if not folder.is_dir(): continue
    steps=[json.loads(x) for x in (folder/'memory_steps.jsonl').read_text(encoding='utf-8').splitlines()]
    tr=json.loads((folder/'training.json').read_text(encoding='utf-8'))
    indices={i for row in steps for i in row['record_indices']}
    runrows.append(dict(endpoint=folder.name,steps=len(steps),source_unique_records=len(indices),
        source_unique_identities=len({records[i]['identity'] for i in indices}),
        source_scenes=sorted({records[i]['scene'] for i in indices}),
        peak_reserved_mib=tr['peak_reserved_mib'],
        extra_fresh_role_record_forwards=tr['extra_fresh_role_record_forwards'],
        extra_history_vjp_record_forwards=tr['extra_history_vjp_record_forwards'],
        extra_direct_check_record_forwards=tr['extra_direct_check_record_forwards'],
        current_rank_backward_calls=tr['current_rank_backward_calls'],
        current_auxiliary_backward_calls=tr['current_auxiliary_backward_calls'],
        direct_component_backward_calls=tr['direct_component_backward_calls'],
        optimizer_steps=tr['optimizer_steps'],nonzero_gradient_tensors=tr['nonzero_gradient_tensors'],
        trainable_tensors=tr['trainable_tensors'],overflow_events=tr['overflow_events']))
assert all(x['trainable_tensors']==x['nonzero_gradient_tensors']==203 and x['overflow_events']==0 for x in runrows)
counts=Counter()
for row in runrows:
    for key in ('extra_fresh_role_record_forwards','extra_history_vjp_record_forwards','extra_direct_check_record_forwards',
                'current_rank_backward_calls','current_auxiliary_backward_calls','direct_component_backward_calls','optimizer_steps'):
        counts[key]+=row[key]
result=dict(status='PASS_LOCAL_TEXT_SCOPE_CENSUS',
    registered_binding_rows=len(raw['registered_bindings']),
    unique_registered_paths=len({x['path'] for x in raw['registered_bindings']}),
    all_registered_bindings_match=all(x['match'] for x in raw['registered_bindings']),
    execution_commit_blob_rows=len(raw['executed_bindings']),
    all_executed_bindings_match=all(x['match'] for x in raw['executed_bindings']),
    remote_text_inputs=len(raw['text_inputs']),remote_m0_inventory_files=len(raw['m0_inventory']),
    observed_head=raw['observed_head'],
    protocol_records=len(records),protocol_identities=len({x['identity'] for x in records}),
    protocol_scenes=sorted({x['scene'] for x in records}),protocol_cameras=sorted({x['camera'] for x in records}),
    endpoints=runrows,recorded_training_costs=dict(counts),
    costs_limit='Recorded M0 counters; excludes unchanged current forward/backward, initialization, preflight and strict reload; no wall-clock attribution by component.',
    remote_pipeline=read_remote(RUN+'/pipeline.json'))
(ROOT/'audit_scope_census.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='remote_pipeline'},ensure_ascii=False,indent=2))
