"""Independent CPU checks of complete fixed-state probe distances and metadata."""
import argparse
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, config):
    spec=json.loads(config.read_bytes())
    summary=json.loads((root/'summary.json').read_bytes())
    assert summary['status']=='PASS_COMPLETE_FIXED_STATE_PROBE'
    assert summary['config_sha256']==sha(config)
    assert summary['optimizer_updates']==summary['heldout_record_forwards']==summary['official_image_reads']==0
    protocol=json.loads(Path(spec['protocol']).read_bytes())
    total_rows=total_values=0;states=[]
    for f,fold in zip(summary['folds'],protocol['folds'],strict=True):
        assert f['fold']==fold['fold']
        matching=[]
        for name,receipt in f['states'].items():
            directory=root/f"fold_{f['fold']}_{name}"
            assert receipt==json.loads((directory/'receipt.json').read_bytes())
            for filename,expected in receipt['files'].items():
                path=directory/filename
                assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256']
            rows=[json.loads(s) for s in (directory/'steps.jsonl').read_text().splitlines()]
            expected_steps=8 if summary['mode']=='preflight' else 260
            assert len(rows)==receipt['batches']==expected_steps
            data=np.fromfile(directory/'distances.f32',dtype=np.float32)
            queue=OrderedDict();seen=set();offset=0;nonzero=0;role_rows=[]
            warmup=2 if summary['mode']=='preflight' else 65
            for step,row in enumerate(rows):
                assert row['step']==step+1 and row['optimizer_updates']==0
                assert row['historical_coordinate_mode']=='fixed_current_parameters'
                assert row['age_semantics']=='batch_recency_not_parameter_updates'
                current=row['record_indices'];seen.update(current)
                assert set(current)<=set(fold['source_record_indices'])
                assert [protocol['records'][i]['identity'] for i in current]==row['identities']
                assert [protocol['records'][i]['scene'] for i in current]==row['scenes']
                for index in [i for i,stored in queue.items() if step-stored>8]:del queue[index]
                members=[i for i in queue if i not in set(current)]
                expected=[dict(record_index=i,identity=protocol['records'][i]['identity'],scene=protocol['records'][i]['scene'],
                               age=step-queue[i],stored_step=queue[i]) for i in members]
                assert row['memory']==expected
                count=64*64+64*len(members)
                assert row['distance_offset_bytes']==offset*4 and row['distance_float_count']==count
                dc=data[offset:offset+4096].reshape(64,64);dh=data[offset+4096:offset+count].reshape(64,len(members));offset+=count
                assert np.isfinite(dc).all() and np.isfinite(dh).all()
                ids=np.array(row['identities']);mid=np.array([r['identity'] for r in expected])
                positive=(ids[:,None]==ids[None,:]) & ~np.eye(64,dtype=bool);negative=ids[:,None]!=ids[None,:]
                hp=np.where(positive,dc,-np.inf).max(1);hn=np.where(negative,dc,np.inf).min(1)
                basic=np.maximum(hp-hn+.3,0).mean()
                if members:
                    mp=ids[:,None]==mid[None,:]
                    ph=np.where(mp,dh,-np.inf).max(1);nh=np.where(~mp,dh,np.inf).min(1)
                    uh=np.maximum(hp,ph);un=np.minimum(hn,nh)
                    assert row['statistics']['harder_positive_anchors']==int((ph>hp).sum())
                    assert row['statistics']['harder_negative_anchors']==int((nh<hn).sum())
                    assert set(row['roles'])=={'cnn','transformer','mamba'}
                    role_rows.append(row['roles'])
                else:
                    uh,un=hp,hn
                    assert row['roles']=={} and row['extra_role_record_forwards']==0
                assert abs(float(basic)-row['statistics']['current_triplet'])<2e-6
                assert abs(float(np.maximum(uh-un+.3,0).mean())-row['statistics']['expanded_triplet'])<2e-6
                assert int((uh>=un).sum())==row['statistics']['expanded_wrong_order_anchors']
                assert len(members)==row['statistics']['memory_records']
                assert set(row['candidate_vjp_groups'])<=set(r['stored_step'] for r in expected)
                nonzero+=row['history_nonzero_gradient_records']
                if step>=warmup:
                    for index in current:
                        queue.pop(index,None);queue[index]=step
                    while len(queue)>512:queue.popitem(last=False)
            assert offset==len(data) and sorted(seen)==receipt['observed_source_records']
            if summary['mode']=='source':assert seen==set(fold['source_record_indices'])
            assert receipt['initial_state_sha256']==receipt['final_state_sha256']
            assert receipt['extra_role_record_forwards']==sum(r['extra_role_record_forwards'] for r in rows)+64
            assert receipt['candidate_gradient_chain_rule']['relative_l2_error']<=.005
            assert receipt['history_batches']==len(role_rows)==(5 if summary['mode']=='preflight' else 194)
            aggregate={}
            for role in ('cnn','transformer','mamba'):
                points=[r[role] for r in role_rows]
                historical=[r['current_vs_history']['second_norm'] for r in points]
                noise=[r['history_repeat_noise']['difference_norm'] for r in points]
                aggregate[role]=dict(rows=len(points),history_nonzero=sum(x>0 for x in historical),
                    above_repeated_noise=sum(x>n for x,n in zip(historical,noise,strict=True)),
                    current_history_cosines=[r['current_vs_history']['cosine'] for r in points],
                    total_change_cosines=[r['total_vs_both']['cosine'] for r in points],
                    history_parameter_norm=historical,current_parameter_norm=[r['current_vs_history']['first_norm'] for r in points])
            states.append(dict(fold=f['fold'],state=name,batches=len(rows),history_nonzero_record_exposures=nonzero,roles=aggregate,
                               extra_role_record_forwards=receipt['extra_role_record_forwards'],peak_allocated_mib=receipt['peak_allocated_mib']))
            matching.append([(r['record_indices'],r['pixel_sha256'],r['memory']) for r in rows])
            total_rows+=len(rows);total_values+=len(data)
        assert matching[0]==matching[1]==matching[2]
    assert total_rows==(72 if summary['mode']=='preflight' else 2340)
    result=dict(status='PASS_COMPLETE_FIXED_STATE_PROBE_CPU',verified_at=datetime.now().astimezone().isoformat(),
                summary_sha256=sha(root/'summary.json'),batches=total_rows,distance_elements=total_values,states=states,
                scope='All saved matrices and identities rechecked; parameter gradients and state hashes remain runtime witnesses.',
                model_forwards=0,optimizer_updates=0,scientific_qualification=None)
    (root/'cpu_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],batches=total_rows,distance_elements=total_values)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();verify(args.input_dir,args.config)
