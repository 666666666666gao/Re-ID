"""Descriptive source-log analysis after complete Q1 intake; no training or new gates."""
from pathlib import Path
from collections import Counter
import hashlib,json,sys

root=Path(sys.argv[1]);output=Path(sys.argv[2]);assert not output.exists()
inventory=json.loads((root/'remote_terminal_inventory.json').read_bytes())
assert inventory['pipeline']['status'] in ('COMPLETE_VERIFIED_Q1_PASS','COMPLETE_VERIFIED_Q1_FAIL')
assert not any(inventory['original_pid_exists'].values())
intake=json.loads((root/'intake_complete.json').read_bytes())
assert intake['status']=='RECEIVED_ALL_REGISTERED_TERMINAL_TEXT'
assert intake['inventory_sha256']==hashlib.sha256((root/'remote_terminal_inventory.json').read_bytes()).hexdigest()
for item in inventory['files']:
    f=root/item['path']
    assert f.stat().st_size==item['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],item['path']
summary=json.loads((root/'q1/summary.json').read_bytes())
cpu=json.loads((root/'q1_cpu.json').read_bytes())
assert cpu['status']=='PASS_COMPLETE_ROLE_SET_Q1' and cpu['checked_training_steps']==1560
assert cpu['summary_sha256']==hashlib.sha256((root/'q1/summary.json').read_bytes()).hexdigest()

def analyze(rows,steps):
    counts=Counter();hist_position=Counter();hist_record=Counter();hist_identity=Counter()
    losses=Counter();roles={e:Counter() for e in ('cnn','transformer','mamba')}
    for row,tr in zip(rows,steps,strict=True):
        assert row['step']==tr['step'] and row['current_anchor_count']==64 and row['history_anchor_count']==0
        assert row['proposal_order']==['fused','cnn','transformer','mamba']
        relation=row['relation_objective'];props=relation['proposals']
        assert len(props)==64
        records=row['record_indices']+[x['record_index'] for x in row['memory']]
        identities=row['identities']+[x['identity'] for x in row['memory']]
        counts['steps']+=1;counts['anchor_exposures']+=64
        counts['memory_record_exposures']+=len(row['memory'])
        counts['extra_active_hinge_exposures']+=sum(relation['extra_active_counts'])
        counts['anchors_with_extra_active_hinge']+=sum(x>0 for x in relation['extra_active_counts'])
        losses['batch_hard_sum']+=row['statistics']['current_triplet']
        losses['expanded_hard_sum']+=relation['hard_loss']
        losses['role_set_sum']+=relation['role_set_loss']
        for i,ps in enumerate(props):
            assert len(ps)==4 and all(identities[k]!=row['identities'][i] for k in ps)
            unique=set(ps);rs={records[k] for k in ps};ids={identities[k] for k in ps}
            assert len(unique)==relation['negative_counts'][i]
            assert 0<=relation['extra_active_counts'][i]<=len(unique)-1
            hist_position[len(unique)]+=1;hist_record[len(rs)]+=1;hist_identity[len(ids)]+=1
            counts['selected_position_exposures']+=len(unique)
            counts['extra_position_exposures']+=len(unique)-1
            counts['extra_record_exposures']+=len(rs)-1
            counts['extra_negative_identity_exposures']+=len(ids)-1
            counts['anchors_with_extra_position']+=len(unique)>1
            counts['anchors_with_extra_record']+=len(rs)>1
            counts['anchors_with_extra_negative_identity']+=len(ids)>1
            counts['position_duplicates_of_selected_records']+=len(unique)-len(rs)
            counts['selected_record_redundancy_within_identities']+=len(rs)-len(ids)
            for k in unique-{ps[0]}:
                same=identities[k]==identities[ps[0]]
                counts['extra_positions_same_negative_identity_as_fused']+=same
                counts['extra_positions_other_negative_identity_than_fused']+=not same
                counts['extra_historical_positions']+=k>=64
            for j,e in enumerate(('cnn','transformer','mamba'),1):
                k=ps[j];other=ps[:j]+ps[j+1:]
                roles[e]['position_differs_from_fused']+=k!=ps[0]
                roles[e]['record_differs_from_fused']+=records[k]!=records[ps[0]]
                roles[e]['identity_differs_from_fused']+=identities[k]!=identities[ps[0]]
                roles[e]['position_exclusive_among_four_proposals']+=k not in other
                roles[e]['identity_exclusive_among_four_proposals']+=identities[k] not in [identities[x] for x in other]
    assert counts['steps']>0
    assert counts['extra_position_exposures']==counts['extra_positions_same_negative_identity_as_fused']+counts['extra_positions_other_negative_identity_than_fused']
    return dict(counts=dict(counts),unique_positions_histogram=dict(hist_position),unique_records_histogram=dict(hist_record),unique_negative_identities_histogram=dict(hist_identity),role_proposal_exposures={e:dict(v) for e,v in roles.items()},mean_losses={k.removesuffix('_sum'):v/counts['steps'] for k,v in losses.items()})

ends=[];paired=[]
for fold in range(3):
    fold_rows={}
    for end in ('control','role_set'):
        p=root/'q1'/f'fold_{fold}_{end}'
        tr=json.loads((p/'training.json').read_bytes())
        rows=[json.loads(x) for x in (p/'memory_steps.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(rows)==len(tr['steps'])==tr['optimizer_steps']==260
        assert all(r['warmup_steps']==65 and r['replacement_active']==(r['zero_based_step']>=65) for r in rows)
        phases={}
        for label,a,b in (('warmup_steps_1_65',0,65),('post_warmup_steps_66_260',65,260),('last_five_epochs_steps_196_260',195,260)):
            phases[label]=analyze(rows[a:b],tr['steps'][a:b])
        ends.append(dict(fold=fold,endpoint=end,phases=phases,cost=dict(fresh_role_record_forwards=tr['extra_fresh_role_record_forwards'],historical_vjp_record_forwards=tr['extra_history_vjp_record_forwards'],fit_epoch_seconds=sum(x['elapsed_seconds'] for x in tr['history']),peak_allocated_mib=tr['peak_allocated_mib'])))
        fold_rows[end]=(rows,tr['steps'])
    first,first_steps=fold_rows['control'];second,second_steps=fold_rows['role_set']
    assert all((a['record_indices'],a['pixel_sha256'])==(b['record_indices'],b['pixel_sha256']) for a,b in zip(first,second,strict=True))
    paired.append(dict(fold=fold,paired_source_records_and_pixels_exact=True,warmup_max_abs_total_loss_difference=max(abs(a['loss']-b['loss']) for a,b in zip(first_steps[:65],second_steps[:65],strict=True)),warmup_total_loss_exact_steps=sum(a['loss']==b['loss'] for a,b in zip(first_steps[:65],second_steps[:65],strict=True))))
result=dict(status='COMPLETE_DESCRIPTIVE_SOURCE_LOG_ANALYSIS',input_inventory_sha256=intake['inventory_sha256'],q1_summary_sha256=cpu['summary_sha256'],endpoints=ends,paired_warmup=paired,scientific_status=summary['status'],scope='All six completed source training logs; exposure counts are repeated training relations, not independent samples. Active-hinge totals cannot identify which role contributed each active relation. Loss comparisons preserve their scalar definitions. Does not alter or replace CPU verification, registered gates, or independent integrity audit.',model_forwards=0,optimizer_updates=0)
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=result['status'],endpoints=len(ends),steps=1560,output=str(output))))
