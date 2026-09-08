"""CPU replay of every recorded relation in the fixed-state engineering probe."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.train_msvr310_signal_oof import sha256,write_json


def run(args):
    spec=json.loads(args.config.read_bytes())
    summary=json.loads((args.run_dir/'summary.json').read_bytes())
    assert summary['status']=='COMPLETE_ROLE_SET_PARAMETER_GRADIENT_CHECK'
    assert summary['config_sha256']==sha256(args.config) and len(summary['folds'])==3
    assert sha256(spec['protocol'])==spec['protocol_sha256']
    protocol=json.loads(Path(spec['protocol']).read_bytes())
    rows_count=elements=0;max_error=0.;support=[]
    for result,fold in zip(summary['folds'],protocol['folds'],strict=True):
        assert result['fold']==fold['fold']
        directory=args.run_dir/f"fold_{fold['fold']}"
        receipt=json.loads((directory/'receipt.json').read_bytes())
        assert receipt=={k:v for k,v in result.items() if k!='initialization'}
        assert receipt['initial_state_sha256']==receipt['final_state_sha256']
        assert receipt['optimizer_updates']==receipt['checkpoint_writes']==0
        for name,item in receipt['files'].items():
            assert (directory/name).stat().st_size==item['bytes'] and sha256(directory/name)==item['sha256']
        rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        assert len(rows)==8
        observed=[]
        with (directory/'distances.f32').open('rb') as stream:
            for row in rows:
                ids=np.array([protocol['records'][i]['identity'] for i in row['record_indices']])
                assert ids.tolist()==row['identities'] and set(row['record_indices'])<=set(fold['source_record_indices'])
                memory=row['memory'];m=len(memory)
                assert all(r['record_index'] in fold['source_record_indices'] for r in memory)
                assert all(r['identity']==protocol['records'][r['record_index']]['identity'] for r in memory)
                assert not {r['record_index'] for r in memory}&set(row['record_indices'])
                assert stream.tell()==row['distance_offset_bytes']
                n=4*64*(64+m);assert n==row['distance_float_count']
                arrays=np.fromfile(stream,dtype=np.float32,count=n).reshape(4,64,64+m)
                assert np.isfinite(arrays).all() and (arrays>=0).all()
                mids=np.array([r['identity'] for r in memory]);allids=np.concatenate((ids,mids))
                positive=ids[:,None]==allids[None,:];positive[np.arange(64),np.arange(64)]=False
                negative=ids[:,None]!=allids[None,:]
                distance=arrays[0]
                hp=np.where(positive,distance,-np.inf).max(1)
                hn=np.where(negative,distance,np.inf).min(1)
                proposals=np.stack([np.where(negative,d,np.inf).argmin(1) for d in arrays],axis=1)
                assert proposals.tolist()==row['proposals']
                losses=[];active=[];counts=[]
                for i,values in enumerate(proposals):
                    unique=sorted(set(values.tolist()));extra=[j for j in unique if j!=values[0]]
                    terms=np.maximum(np.float32(0),hp[i]-distance[i,extra]+np.float32(.3))
                    original=max(np.float32(0),hp[i]-hn[i]+np.float32(.3))
                    losses.append((original+terms.sum())/len(unique));counts.append(len(unique));active.append(int((terms>0).sum()))
                hard=float(np.maximum(0,hp-hn+np.float32(.3)).mean());candidate=float(np.mean(losses))
                assert counts==row['negative_counts'] and active==row['extra_active_counts']
                error=max(abs(hard-row['hard_loss']),abs(candidate-row['candidate_loss']))
                max_error=max(max_error,error);assert error<2e-6
                for stats in row['roles'].values():
                    for value in stats.values():
                        a,b,d=value['first_norm'],value['second_norm'],value['difference_norm'];cos=value['cosine']
                        assert all(np.isfinite(v) and v>=0 for v in (a,b,d))
                        assert (cos is None)==(a==0 or b==0)
                        if cos is not None:
                            assert abs(cos)<=1.00001
                            assert abs(d*d-(a*a+b*b-2*a*b*cos))<=1e-7*max(1,a*a+b*b)
                observed.extend(row['record_indices']);rows_count+=1;elements+=n
            assert stream.read()==b''
        assert sorted(set(observed))==receipt['unique_source_records']
        for field,only_history in [('changed_gradient_batches_above_repeat_noise',False),('changed_gradient_history_batches_above_repeat_noise',True)]:
            calculated={e:sum((bool(r['memory']) or not only_history) and r['roles'][e]['hard_vs_role_set']['difference_norm']>r['roles'][e]['role_set_repeat_noise']['difference_norm'] for r in rows) for e in ('cnn','transformer','mamba')}
            assert calculated==receipt[field]
        assert sum(sum(r['extra_active_counts']) for r in rows)==receipt['active_extra_negative_exposures']
        assert receipt['direct_history_chain_rule']['relative_l2_error']<=.005
        support.append(dict(fold=fold['fold'],history_gradient_support=receipt['changed_gradient_history_batches_above_repeat_noise'],active_extra_negative_exposures=receipt['active_extra_negative_exposures']))
    assert rows_count==24 and summary['optimizer_updates']==summary['heldout_record_forwards']==summary['official_image_reads']==0
    write_json(args.output,dict(status='PASS_COMPLETE_ROLE_SET_GRADIENT_PROBE_TEXT_AND_ARRAYS',summary_sha256=sha256(args.run_dir/'summary.json'),batches=rows_count,distance_elements=elements,max_loss_error=max_error,support=support,
                               scope='All 24 recorded batches. Parameter gradients and state preservation are runtime witnesses, not CPU model re-execution. No training/retrieval qualification.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
