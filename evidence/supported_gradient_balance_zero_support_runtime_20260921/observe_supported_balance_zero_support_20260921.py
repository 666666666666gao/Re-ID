"""Inspect complete JSONL prefix only; no metrics, model loads or updates."""
from pathlib import Path
from datetime import datetime
import hashlib,json

root=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639')
pipeline=json.loads((root/'pipeline.json').read_bytes())
handles={str(s['original_pid']):Path('/proc',str(s['original_pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode() for s in pipeline['stages'] if s['stage']=='q1' and Path('/proc',str(s['original_pid']),'cmdline').exists()}
endpoints=[]
for p in sorted((root/'q1').glob('*/memory_steps.jsonl')):
    raw=p.read_bytes()
    complete=raw[:raw.rfind(b'\n')+1]
    rows=[json.loads(line) for line in complete.splitlines()]
    zero=[]
    for row in rows:
        if not row['replacement_active'] or row['support']['eligible_anchors']!=0:continue
        assert row['relation_objective']['cross_scene_loss']==0
        roles={}
        for role,b in row['gradient_balance'].items():
            assert not b['supported'] and b['before']==b['after']
            assert b['applied_rank_weight']==b['applied_auxiliary_weight']==1
            assert b['rank_vs_auxiliary']['first_norm']==0
            assert b['current_rank_vs_history']['first_norm']==b['current_rank_vs_history']['second_norm']==0
            assert b['original_sum_vs_applied']['difference_norm']==0
            roles[role]=dict(ema_unchanged=True,applied_rank_weight=b['applied_rank_weight'],applied_auxiliary_weight=b['applied_auxiliary_weight'],rank_norm=b['rank_vs_auxiliary']['first_norm'],auxiliary_norm=b['rank_vs_auxiliary']['second_norm'],original_to_applied_difference=b['original_sum_vs_applied']['difference_norm'],actual_parameter_update=row['actual_parameter_updates'][role]['difference_norm'])
        assert row['classification_head_gradients_unchanged']
        zero.append(dict(step=row['step'],support=row['support'],roles=roles,raw_record=row))
    endpoints.append(dict(endpoint=p.parent.name,complete_steps=len(rows),last_complete_step=rows[-1]['step'] if rows else None,complete_prefix_bytes=len(complete),complete_prefix_sha256=hashlib.sha256(complete).hexdigest(),incomplete_trailing_bytes=len(raw)-len(complete),active_zero_support_records=zero))
print(json.dumps(dict(status='PASS_OBSERVED_ZERO_SUPPORT_RUNTIME_WITNESSES',observed_at=datetime.now().astimezone().isoformat(),run=str(root),live_q1_handles=handles,endpoints=endpoints,model_forwards=0,optimizer_updates=0,retrieval_reads=0,scope='Only fully written source JSONL records observed so far; saved runtime witnesses, not independent gradient regeneration or full Q1 qualification.')))
