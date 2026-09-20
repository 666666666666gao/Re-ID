from pathlib import Path
import json,hashlib,datetime,shutil
root=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
path=root/'q1/fold_0_cross_scene/memory_steps.jsonl'
with path.open() as stream:
    prefix=[next(stream) for _ in range(181)]
row=json.loads(prefix[179]);following=json.loads(prefix[180])
assert row['step']==180 and following['step']==181
assert row['replacement_active'] and row['warmup_steps']==65
assert row['coordinate_rule']=='fresh' and row['history_candidate_vjp_applied']
assert row['current_anchor_count']==64 and row['history_anchor_count']==0
ids=row['identities'];scenes=row['scenes']
cids=ids+[x['identity'] for x in row['memory']]
cscenes=scenes+[x['scene'] for x in row['memory']]
counts=[sum(y==candidate and scene!=cs for candidate,cs in zip(cids,cscenes,strict=True)) for y,scene in zip(ids,scenes,strict=True)]
negative_counts=[sum(y!=candidate for candidate in cids) for y in ids]
rel=row['relation_objective']
assert counts==[0]*64==rel['cross_scene_positive_counts']
assert rel['cross_scene_eligible_anchors']==0 and rel['cross_scene_loss']==0
assert rel['cross_scene_per_anchor_ap']==[0]*64 and all(x>0 for x in negative_counts)
assert len(row['historical_leaf_upstream_norms'])==len(row['memory'])>0
assert all(x==0 for x in row['historical_leaf_upstream_norms'])
assert row['history_vjp_record_forwards']==0 and row['history_vjp_groups']==[]
for role in ('cnn','transformer','mamba'):
    g=row['roles'][role]
    assert g['total_vs_history']['first_norm']>0 and g['total_vs_history']['second_norm']==0
    assert g['total_vs_both']['difference_norm']==0
    assert row['applied_gradients'][role]['difference_norm']==0
assert row['history_rng_buffers_preserved'] and row['final_gradient_addition_bitwise']
pipeline=json.loads((root/'pipeline.json').read_bytes())
pids=[pipeline['wrapper_pid'],next(x['original_pid'] for x in pipeline['stages'] if x['stage']=='q1')]
processes={str(pid):(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode() for pid in pids}
result=dict(status='PASS_SAVED_FIRST_ZERO_SUPPORT_ROW',observed_at=datetime.datetime.now().astimezone().isoformat(),root=str(root),
            source_file=str(path),step=180,later_saved_step=181,source_row_sha256=hashlib.sha256(prefix[179].encode()).hexdigest(),
            eligible_anchors=0,negative_count_range=[min(negative_counts),max(negative_counts)],history_records=len(row['memory']),
            current_nonzero_gradient_norms={e:row['roles'][e]['total_vs_history']['first_norm'] for e in ('cnn','transformer','mamba')},
            history_vjp_record_forwards=row['history_vjp_record_forwards'],original_processes=processes,free_bytes=shutil.disk_usage(root.parent).free,
            source_row=row,later_row_sha256=hashlib.sha256(prefix[180].encode()).hexdigest(),
            model_forwards=0,optimizer_updates=0,
            scope='Read-only saved actual Q1 source row. Independent label-mask arithmetic; gradients and optimizer call progress are runtime witnesses, not regenerated vectors or independently measured parameter deltas. Full training scalar ledger and complete Q1 CPU verification remain pending.')
print(json.dumps(result))
