"""Complete CPU source ranking and full-triplet census from frozen saved features."""
import argparse
import csv
from datetime import datetime
import gzip
import json
from pathlib import Path
import time

import numpy as np

from tools.msvr_source_relation_math import WIDTHS, STATES, VIEWS, PROTOCOLS, source_rows, unit, math_check
from tools.diagnose_msvr_source_relations import contract, ROOT
from tools.train_msvr310_signal_oof import sha256, write_json


def aggregate(rows):
    eligible = [r for r in rows if r['eligible']]
    triplets = sum(r['triplets'] for r in rows)
    result = dict(query_records=len(rows),eligible_queries=len(eligible),
                  ineligible_queries=len(rows)-len(eligible),triplets=triplets)
    for key in ('nonpositive_triplets','hinge_positive_triplets','wrong_order','margin_only','margin_satisfied',
                'prototype_correct_instance_rank1_wrong','prototype_correct_instance_some_positive_wrong'):
        result[key] = sum(r[key] for r in eligible)
    result['hinge_sum'] = sum(r['hinge_sum'] for r in eligible)
    if eligible:
        result.update(mAP=100*float(np.mean([r['average_precision'] for r in eligible])),
                      rank1=100*float(np.mean([r['first_match_rank']==1 for r in eligible])),
                      mean_hinge=result['hinge_sum']/triplets,
                      mean_hardest_hinge=float(np.mean([r['batch_hard_hinge'] for r in eligible])),
                      mean_worst_positive_margin=float(np.mean([r['worst_positive_margin'] for r in eligible])),
                      mean_best_positive_margin=float(np.mean([r['best_positive_margin'] for r in eligible])))
        assert result['wrong_order']+result['margin_only']+result['margin_satisfied']==len(eligible)
    return result


def independent_row_check(scores, ids, scenes, protocol, row):
    q = row['query_position']
    positives = np.array([j for j in range(len(ids)) if ids[j]==ids[q] and j!=q
                          and (protocol!='cross_scene' or scenes[j]!=scenes[q])],dtype=np.int64)
    negatives = np.flatnonzero(ids!=ids[q])
    assert len(positives)==row['positive_records'] and len(negatives)==row['negative_records']
    if not len(positives):
        assert not row['eligible'] and row['triplets']==0
        return 0
    pp,nn = scores[positives],scores[negatives]
    margins = pp[:,None]-nn[None]
    hinge = np.maximum(0.,np.sqrt(np.maximum(0.,2-2*pp[:,None]))
                       -np.sqrt(np.maximum(0.,2-2*nn[None]))+.3)
    assert row['nonpositive_triplets']==int((margins<=0).sum())
    assert row['hinge_positive_triplets']==int((hinge>0).sum())
    assert row['triplets']==hinge.size
    error = abs(row['hinge_sum']-float(hinge.sum()))
    assert error<1e-8
    legal=np.concatenate((positives,negatives))
    # Tie-break by original record position, independently of the full argsort.
    order=sorted(legal.tolist(),key=lambda j:(-float(scores[j]),j))
    ranks=np.array([k+1 for k,j in enumerate(order) if ids[j]==ids[q]])
    ap=float(np.mean(np.arange(1,len(ranks)+1)/ranks))
    assert abs(ap-row['average_precision'])<1e-14 and ranks[0]==row['first_match_rank']
    return error


def run(args):
    started=time.perf_counter()
    spec,q1=contract(args.contract)
    summary=json.loads((args.root/'summary.json').read_bytes())
    assert summary['status']=='COMPLETE_FROZEN_SOURCE_EXTRACTION'
    assert summary['contract_sha256']==sha256(args.contract)
    assert len(summary['conditions'])==27 and summary['source_record_forwards']==18576
    assert {(c['fold'],c['state'],c['view']) for c in summary['conditions']}=={
        (f,s,v) for f in range(3) for s in STATES for v in VIEWS}
    protocol=json.loads((ROOT/spec['protocol']).read_bytes())
    check=math_check()
    fields=('fold','state','view','output','protocol','query_position','record_index','identity','scene',
            'eligible','positive_records','negative_records','triplets','best_positive_margin',
            'worst_positive_margin','mean_triplet_margin','nonpositive_triplets','hinge_positive_triplets',
            'hinge_sum','hinge_mean','batch_hard_hinge','average_precision','first_match_rank',
            'prototype_margin','prototype_correct_instance_rank1_wrong',
            'prototype_correct_instance_some_positive_wrong','wrong_order','margin_only','margin_satisfied',
            'nearest_negative_position','hardest_positive_position')
    query_path=args.root/'all_source_query_relations.csv.gz'
    identity_path=args.root/'all_source_identity_relations.jsonl.gz'
    assert not query_path.exists() and not identity_path.exists()
    complete=[];query_rows=0;identity_rows=0;checked_pair_entries=0;maximum_error=0.;decomposition_error=0.
    inputs={};saved_baselines={}
    with gzip.open(query_path,'wt',encoding='utf-8',newline='') as stream, \
         gzip.open(identity_path,'wt',encoding='utf-8') as idstream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for condition in summary['conditions']:
            folder=args.root/condition['directory']
            fold=protocol['folds'][condition['fold']]
            records=[protocol['records'][j] for j in fold['source_record_indices']]
            ids=np.array([r['identity'] for r in records]);scenes=np.array([r['scene'] for r in records])
            assert condition['record_indices']==fold['source_record_indices']
            assert condition['model_state_unchanged'] and condition['gradients_absent']
            for name,proof in condition['files'].items():
                assert (folder/name).stat().st_size==proof['bytes'] and sha256(folder/name)==proof['sha256'],name
            input_rows=[json.loads(line) for line in (folder/'inputs.jsonl').read_text().splitlines()]
            assert [j for row in input_rows for j in row['record_indices']]==fold['source_record_indices']
            pixel_list=[r['pixel_sha256'] for r in input_rows]
            key=(condition['fold'],'clean' if condition['view']=='clean' else 'augmented')
            if condition['state']=='initial' and condition['view']!='coupled_style':
                inputs[key]=pixel_list
            assert inputs[key]==pixel_list
            for bi,row in enumerate(input_rows):
                indices=row['record_indices'];plan=row['style_plan']
                if condition['view']=='coupled_style':
                    assert plan['forced_active'] and plan['active'] and row['statistics']['style_active']==1
                    assert len(plan['donors'])==len(indices)
                    cameras=np.array([protocol['records'][j]['camera'] for j in indices])
                    rng=np.random.default_rng(np.random.SeedSequence([42,condition['fold'],bi]))
                    rng.random()
                    choices=rng.random((len(indices),len(indices)))
                    choices[cameras[:,None]==cameras[None,:]]=np.inf
                    assert plan['donors']==choices.argmin(axis=1).tolist()
                    assert plan['coefficients']==rng.beta(.1,.1,size=len(indices)).tolist()
                    assert all(protocol['records'][indices[j]]['camera']!=protocol['records'][indices[d]]['camera']
                               for j,d in enumerate(plan['donors']))
                else:
                    assert plan is None and row['statistics']['style_active']==0
            clean=args.root/f'fold_{condition["fold"]}_{condition["state"]}_clean'
            matrices={}
            for output,width in WIDTHS.items():
                x=np.load(folder/f'{output}.npy',mmap_mode='r',allow_pickle=False)
                g=np.load(clean/f'{output}.npy',mmap_mode='r',allow_pickle=False)
                assert x.shape==g.shape==(len(records),width) and x.dtype==g.dtype==np.float32
                assert np.isfinite(x).all() and np.isfinite(g).all()
                qf,gf=unit(x),unit(g)
                scores=qf@gf.T
                matrices[output]=scores
                assert np.isfinite(scores).all()
                checked_pair_entries+=scores.size
                if output=='baseline_only':
                    key=(condition['fold'],condition['view'])
                    if condition['state']=='initial':saved_baselines[key]=np.array(x)
                    assert np.array_equal(saved_baselines[key],x)
                    if condition['view']=='coupled_style':
                        assert np.array_equal(saved_baselines[(condition['fold'],'augmented')],x)
                for relation in PROTOCOLS:
                    rows=list(source_rows(x,g,ids,scenes,relation))
                    assert len(rows)==len(records)
                    context={k:condition[k] for k in ('fold','state','view')}
                    context.update(output=output,protocol=relation)
                    for row in rows:
                        maximum_error=max(maximum_error,independent_row_check(scores[row['query_position']],ids,scenes,relation,row))
                        writer.writerow({**context,**row,'record_index':records[row['query_position']]['index']})
                        query_rows+=1
                    complete.append({**context,**aggregate(rows)})
                    for identity in fold['source_ids']:
                        selected=[r for r in rows if r['identity']==identity]
                        assert selected
                        idstream.write(json.dumps(dict(**context,identity=identity,**aggregate(selected)))+'\n')
                        identity_rows+=1
                del x,g,qf,gf,scores
            role_names=('cnn','transformer','mamba')
            slots=[matrices[f'{e}_{m}_residual'] for e in role_names for m in ('RGB','NI','TI')]
            error=max(float(np.abs(matrices['fused']-.5*matrices['baseline_only']-sum(slots)/18).max()),
                      float(np.abs(matrices['fused']-sum(matrices[e] for e in role_names)/3).max()),
                      float(np.abs(matrices['pure_bank']-sum(matrices[f'pure_{e}'] for e in role_names)/3).max()))
            assert error<2e-6
            decomposition_error=max(decomposition_error,error)
            del matrices
            write_json(args.root/'cpu_progress.json',dict(status='RUNNING',complete_conditions=len(complete),
                query_rows=query_rows,identity_rows=identity_rows,last_condition=condition['directory']))
            print(json.dumps(dict(event='source_cpu_condition_complete',condition=condition['directory'],
                relation_conditions=len(complete),query_rows=query_rows,elapsed_seconds=time.perf_counter()-started)),flush=True)
    assert len(complete)==972 and query_rows==668736 and identity_rows==100440
    assert checked_pair_entries==sum(n*n for n in (672,683,709))*9*18
    contract(args.contract)
    result=dict(status='PASS_COMPLETE_SOURCE_RELATION_CENSUS',verified_at=datetime.now().astimezone().isoformat(),
                contract_sha256=sha256(args.contract),extraction_summary_sha256=sha256(args.root/'summary.json'),
                all972_conditions=complete,all_query_rows=query_rows,all_identity_rows=identity_rows,
                checked_pair_entries=checked_pair_entries,maximum_independent_hinge_sum_error=maximum_error,
                maximum_fixed_similarity_decomposition_error=decomposition_error,
                mathematical_witness=check,all_source_records_and_ineligible_gallery_records_retained=True,
                model_forwards=0,optimizer_updates=0,heldout_image_reads=0,official_image_reads=0,
                text_files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in (query_path,identity_path)},
                elapsed_seconds=time.perf_counter()-started,
                boundaries=['Seen-source frozen evaluation, not heldout/official performance.',
                    'All source records under newly registered diagnostic views; not original260step pixel replay.',
                    'Coupled style forced active for this fixed stress view only; sealedtrainingp0.5 unchanged.',
                    'Rank/hinge margins are not parameter gradients; prototype gaps do not establish memory benefit.',
                    'No across-fold feature distances; repeated source memberships are not independent identities.'])
    write_json(args.root/'cpu_verification.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args())
