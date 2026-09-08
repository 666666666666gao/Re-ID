from pathlib import Path
from collections import Counter
from datetime import datetime
import hashlib,json,time
import numpy as np

started=time.monotonic()
root=Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
protocol_path=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID/protocols/msvr310_train_oof_v1.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(root/'q1_cpu.json')=='b19622644d45007560e9f930ea52dffea921044f456f8f82d4fc9935d5d55726'
cpu=json.loads((root/'q1_cpu.json').read_bytes());protocol=json.loads(protocol_path.read_bytes())
assert cpu['status']=='PASS_COMPLETE_ROLE_SET_Q1' and cpu['checked_training_steps']==1560
inputs={str(root/'q1_cpu.json'):sha(root/'q1_cpu.json'),str(protocol_path):sha(protocol_path)}
ends=[];checked_elements=checked_steps=direct_checks=0
for fold in range(3):
    allowed=set(protocol['folds'][fold]['source_record_indices'])
    for end in ('control','role_set'):
        directory=root/'q1'/f'fold_{fold}_{end}'
        for name in ('training.json','memory_steps.jsonl','memory_distances.f32'):
            path=directory/name;proof=cpu['files'][str(path)]
            assert path.stat().st_size==proof['bytes'] and sha(path)==proof['sha256']
            inputs[str(path)]=proof['sha256']
        rows=[json.loads(x) for x in (directory/'memory_steps.jsonl').read_text().splitlines()]
        assert len(rows)==260
        phases={k:Counter() for k in ('all','warmup','post_warmup','last65')}
        with (directory/'memory_distances.f32').open('rb') as stream:
            for index,row in enumerate(rows):
                assert row['step']==index+1 and row['current_anchor_count']==64 and row['history_anchor_count']==0
                assert stream.tell()==row['distance_offset_bytes']
                records=row['record_indices']+[x['record_index'] for x in row['memory']]
                identities=row['identities']+[x['identity'] for x in row['memory']]
                scenes=row['scenes']+[x['scene'] for x in row['memory']]
                assert set(records)<=allowed
                assert identities==[protocol['records'][i]['identity'] for i in records]
                assert scenes==[protocol['records'][i]['scene'] for i in records]
                values=np.fromfile(stream,dtype=np.float32,count=row['distance_float_count'])
                assert values.size==4*64*len(records) and np.isfinite(values).all()
                distance=values.reshape(4,64,len(records))[0]
                checked_elements+=values.size;checked_steps+=1
                ids=np.asarray(identities);ss=np.asarray(scenes);rr=np.asarray(records)
                count=Counter(steps=1,anchor_exposures=64)
                for i in range(64):
                    pm=ids==ids[i];pm[i]=False;nm=ids!=ids[i]
                    pos=np.flatnonzero(pm);neg=np.flatnonzero(nm);assert pos.size and neg.size
                    pd=distance[i,pos];nd=np.sort(distance[i,neg]);hp=pd.max()
                    inv=np.searchsorted(nd,pd,side='left')
                    ties=np.searchsorted(nd,pd,side='right')-inv
                    affected=inv>0;nonmax=pd<hp;cross=ss[pos]!=ss[i]
                    if index==0:
                        assert np.array_equal(inv,(pd[:,None]>distance[i,neg][None,:]).sum(1))
                        direct_checks+=1
                    count['anchors_with_any_strict_inversion']+=int(affected.any())
                    count['anchors_with_cross_scene_positive']+=int(cross.any())
                    count['anchors_with_cross_scene_strict_inversion']+=int((affected&cross).any())
                    count['anchors_with_nonmax_strict_inversion']+=int((affected&nonmax).any())
                    count['anchors_with_nonmax_cross_scene_strict_inversion']+=int((affected&nonmax&cross).any())
                    count['anchors_all_hardest_positives_same_scene_with_cross_available']+=int(cross.any() and not cross[~nonmax].any())
                    for prefix,mask in (('all_positive',np.ones(pos.size,dtype=bool)),('cross_scene_positive',cross)):
                        count[prefix+'_positions']+=int(mask.sum())
                        count[prefix+'_distinct_record_exposures']+=len(set(rr[pos[mask]].tolist()))
                        count[prefix+'_strict_inversion_pairs']+=int(inv[mask].sum())
                        count[prefix+'_negative_distance_tie_pairs']+=int(ties[mask].sum())
                        count[prefix+'_affected_positions']+=int((affected&mask).sum())
                        count[prefix+'_affected_nonmax_positions']+=int((affected&mask&nonmax).sum())
                        count[prefix+'_affected_max_positions']+=int((affected&mask&~nonmax).sum())
                        count[prefix+'_affected_historical_positions']+=int((affected&mask&(pos>=64)).sum())
                        count[prefix+'_nonmax_strict_inversion_pairs']+=int(inv[mask&nonmax].sum())
                    count['same_record_positive_position_exposures']+=int((rr[pos]==rr[i]).sum())
                assert count['anchors_with_any_strict_inversion']==row['statistics']['expanded_wrong_order_anchors']
                for prefix in ('all_positive','cross_scene_positive'):
                    assert count[prefix+'_affected_positions']==count[prefix+'_affected_nonmax_positions']+count[prefix+'_affected_max_positions']
                phases['all'].update(count);phases['warmup' if index<65 else 'post_warmup'].update(count)
                if index>=195:phases['last65'].update(count)
            assert not stream.read(1)
        assert phases['all']['steps']==260 and phases['post_warmup']['steps']==195 and phases['last65']['steps']==65
        ends.append(dict(fold=fold,endpoint=end,phases={k:dict(v) for k,v in phases.items()}))
assert checked_steps==1560 and checked_elements==cpu['checked_memory_distance_elements'] and direct_checks==384
print(json.dumps(dict(status='COMPLETE_SAVED_SOURCE_POSITIVE_COVERAGE',completed_at=datetime.now().astimezone().isoformat(),inputs=inputs,endpoints=ends,checked_steps=checked_steps,checked_distance_elements=checked_elements,independent_first_batch_anchor_checks=direct_checks,model_forwards=0,optimizer_updates=0,official_reads=0,heldout_rankings_read=0,remote_files_written=0,elapsed_seconds=time.monotonic()-started,scope='Descriptive strict inversion counts from all saved source training distances; nonmaximal means below all-positive maximum, including all maximum ties as protected. This does not establish missing total parameter gradients, label noise, causal failure, or benefit of another loss.')))
