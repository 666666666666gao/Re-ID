"""Fixed-terminal source-only candidate coverage, using one view per record."""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np

from tools.train_msvr310_signal_oof import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ENDS = ('control', 'smooth_ap')
VIEWS = ('clean', 'augmented')


def contract(path):
    spec = json.loads(path.read_bytes())
    assert spec['schema'] == 'msvr310-smooth-ap-source-coverage-v1'
    assert spec['seed'] == 42 and spec['optimizer_updates'] == 0
    for name, digest in spec['project_file_sha256'].items():
        assert sha256(ROOT/name) == digest, name
    assert sha256(spec['q1_summary']) == spec['q1_summary_sha256']
    q1 = json.loads(Path(spec['q1_summary']).read_bytes())
    assert q1['status'] == 'Q1_FAIL' and q1['optimizer_steps'] == 1560
    audit = json.loads((ROOT/spec['q1_audit']).read_bytes())
    assert audit['deterministic_checks_status'] == 'pass'
    assert audit['scientific_qualification'] == 'Q1_FAIL'
    for fold in q1['folds']:
        for end in ENDS:
            row = fold['endpoints'][end]
            assert sha256(row['checkpoint']) == row['checkpoint_sha256']
    return spec, q1


def rank_row(distances, identities, scenes, anchor, candidates, cross_scene):
    """All negatives retained; same-ID/same-scene excluded only in cross mode."""
    candidates = np.asarray(candidates, dtype=np.int64)
    assert len(candidates) == len(set(candidates.tolist()))
    candidates = candidates[candidates != anchor]
    same = identities[candidates] == identities[anchor]
    if cross_scene:
        candidates = candidates[~(same & (scenes[candidates] == scenes[anchor]))]
    positive = identities[candidates] == identities[anchor]
    count = int(positive.sum())
    negatives = len(candidates)-count
    if not count:
        return dict(eligible=False, positives=0, negatives=negatives, ap=None,
                    rank1=None, inverted_positives=None)
    order = np.argsort(distances[candidates], kind='stable')
    hits = positive[order]
    positions = np.flatnonzero(hits)+1
    minimum_negative = distances[candidates[~positive]].min() if negatives else np.inf
    return dict(eligible=True, positives=count, negatives=negatives,
                ap=float(np.mean(np.arange(1,count+1)/positions)), rank1=int(hits[0]),
                inverted_positives=int((distances[candidates[positive]] > minimum_negative).sum()))


def math_check():
    ids=np.array([0,0,0,1,2]);scenes=np.array([0,0,1,0,0])
    distances=np.array([0.,.1,.4,.2,.8])
    full=rank_row(distances,ids,scenes,0,range(5),True)
    assert full['positives']==1 and full['negatives']==2 and full['ap']==.5
    missing=rank_row(distances,ids,scenes,0,[0,1,3,4],True)
    assert not missing['eligible'] and missing['negatives']==2
    all_ids=rank_row(distances,ids,scenes,0,range(5),False)
    assert all_ids['positives']==2 and abs(all_ids['ap']-5/6)<1e-15
    distractor=rank_row(distances,ids,scenes,3,range(5),True)
    assert not distractor['eligible'] and distractor['negatives']==4
    return dict(status='PASS',checks=['same-record self excluded','same-ID same-scene removed not negative',
        'no-positive records remain distractors','AP uses every legal positive'])


def memory_rows(qrow):
    path=Path(qrow['checkpoint']).parent/'memory_steps.jsonl'
    expected=qrow['training']['audit_files']['memory_steps.jsonl']
    assert sha256(path)==expected['sha256']
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows)==260 and [r['step'] for r in rows]==list(range(1,261))
    return rows


def extract(args):
    import torch
    import torch.nn.functional as F
    from tools.train_msvr_smooth_ap import context, reload_model
    from tools.train_msvr310_signal_oof import records_for
    from tools.train_msvr310_trifusion_oof import OUTPUT_WIDTHS, output_mapping
    from tools.diagnose_msvr_source_relations import source_loader
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed, _training_batch

    spec,q1=contract(args.contract)
    assert shutil.disk_usage(args.root.parent).free >= 2*1024**3
    args.root.mkdir()
    torch.set_num_threads(4)
    _,(config,base,cfg,environment,protocol,baseline,meta)=context(ROOT/spec['training_config'])
    result=dict(status='RUNNING',contract_sha256=sha256(args.contract),math=math_check(),
        started_at=datetime.now().astimezone().isoformat(),conditions=[],source_record_forwards=0,
        optimizer_updates=0,heldout_record_forwards=0,official_image_reads=0,environment=environment)
    write_json(args.root/'summary.json',result)
    for fold,b0,finished in zip(protocol['folds'],baseline['folds'],q1['folds'],strict=True):
        indices=fold['source_record_indices']
        assert len(indices)==[672,683,709][fold['fold']]
        records=records_for(base,protocol,fold,True)
        expected_pixels={};baseline_values={}
        for end in ENDS:
            row=finished['endpoints'][end];memory_rows(row)
            model=reload_model(config,cfg,fold,b0,row['checkpoint'],row['initialization'],
                row['training']['final_state_sha256'],sha256(ROOT/spec['training_config']))
            model.eval();fixed=_module_state_sha256(model)
            for view in VIEWS:
                started=time.perf_counter();_set_seed(42)
                directory=args.root/f'fold_{fold["fold"]}_{end}_{view}';directory.mkdir()
                parts={name:[] for name in OUTPUT_WIDTHS};pixels=[];seen=[]
                with (directory/'inputs.jsonl').open('x',encoding='utf-8') as stream:
                    for bi,raw in enumerate(source_loader(records,view=='augmented')):
                        selected=indices[bi*64:(bi+1)*64]
                        assert list(raw[-1])==[Path(protocol['records'][i]['paths'][0]).name for i in selected]
                        assert raw[1].tolist()==[fold['source_label_map'][str(protocol['records'][i]['identity'])] for i in selected]
                        batch,_=_training_batch(raw)
                        from tools.train_msvr310_source_style import pixels as pixel_hash
                        digest=pixel_hash(batch);pixels.append(digest);seen.extend(selected)
                        output=output_mapping(exact_signal_forward(model,batch))
                        for name,width in OUTPUT_WIDTHS.items():
                            x=output[name].float()
                            assert x.shape==(len(selected),width) and bool(torch.isfinite(x).all())
                            assert bool((x.norm(dim=1)>0).all())
                            parts[name].append(F.normalize(x,dim=1).cpu().numpy())
                        stream.write(json.dumps(dict(batch=bi,record_indices=selected,pixel_sha256=digest))+'\n')
                        stream.flush();result['source_record_forwards']+=len(selected)
                assert seen==indices
                arrays={name:np.concatenate(values) for name,values in parts.items()}
                if end=='control':
                    expected_pixels[view]=pixels;baseline_values[view]=arrays['baseline_only'].copy()
                assert pixels==expected_pixels[view]
                assert np.array_equal(arrays['baseline_only'],baseline_values[view])
                for name,x in arrays.items():np.save(directory/f'{name}.npy',x,allow_pickle=False)
                assert _module_state_sha256(model)==fixed and all(p.grad is None for p in model.parameters())
                receipt=dict(fold=fold['fold'],endpoint=end,view=view,record_indices=indices,
                    state_sha256=fixed,model_state_unchanged=True,gradients_absent=True,
                    paired_pixels_exact=True,paired_baseline_exact=True,elapsed_seconds=time.perf_counter()-started,
                    files={p.name:dict(sha256=sha256(p),bytes=p.stat().st_size) for p in directory.iterdir()})
                write_json(directory/'receipt.json',receipt)
                result['conditions'].append(dict(directory=directory.name,**receipt))
                write_json(args.root/'summary.json',result)
                print(json.dumps(dict(event='source_coverage_extracted',condition=directory.name,
                    complete=len(result['conditions']),elapsed_seconds=receipt['elapsed_seconds'])),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    assert len(result['conditions'])==12 and result['source_record_forwards']==8256
    contract(args.contract)
    result.update(status='COMPLETE_SOURCE_EXTRACTION',ended_at=datetime.now().astimezone().isoformat())
    write_json(args.root/'summary.json',result)


def analyze(args):
    spec,q1=contract(args.contract)
    summary=json.loads((args.root/'summary.json').read_bytes())
    assert summary['status']=='COMPLETE_SOURCE_EXTRACTION'
    assert summary['contract_sha256']==sha256(args.contract)
    protocol=json.loads((ROOT/spec['protocol']).read_bytes())
    result=dict(status='RUNNING',math=math_check(),conditions=[],dtype='float64',
        ties='stable ascending global source record order',single_view_unique_record_pool=True,
        optimizer_updates=0,model_forwards=0)
    for cond in summary['conditions']:
        directory=args.root/cond['directory'];indices=cond['record_indices']
        for name,entry in cond['files'].items():assert sha256(directory/name)==entry['sha256']
        ids=np.array([protocol['records'][i]['identity'] for i in indices])
        scenes=np.array([protocol['records'][i]['scene'] for i in indices])
        local={v:i for i,v in enumerate(indices)}
        logs=memory_rows(q1['folds'][cond['fold']]['endpoints'][cond['endpoint']])
        metrics=[]
        for path in sorted(directory.glob('*.npy')):
            x=np.load(path,allow_pickle=False).astype(np.float64)
            d=np.maximum((x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T,0)
            for cross in (False,True):
                full=[rank_row(d[i],ids,scenes,i,range(len(ids)),cross) for i in range(len(ids))]
                name=path.stem+('_cross_scene' if cross else '_all_identity')
                write_json(directory/(name+'_full.json'),[dict(record_index=indices[i],**r) for i,r in enumerate(full)])
                common=[];missing=0;exposures=0
                with (directory/(name+'_candidates.jsonl')).open('x',encoding='utf-8') as stream:
                    for batch in logs:
                        pool=sorted({local[i] for i in batch['record_indices']} |
                                    {local[r['record_index']] for r in batch['memory']})
                        for position,record in enumerate(batch['record_indices']):
                            i=local[record];candidate=rank_row(d[i],ids,scenes,i,pool,cross)
                            pair=dict(step=batch['step'],anchor_position=position,record_index=record,
                                      candidate=candidate,full=full[i])
                            stream.write(json.dumps(pair)+'\n');exposures+=1
                            if candidate['eligible']:
                                assert full[i]['eligible'];common.append((candidate['ap'],full[i]['ap']))
                            elif full[i]['eligible']:missing+=1
                paired=np.asarray(common,dtype=np.float64).reshape(-1,2)
                legal=[r for r in full if r['eligible']]
                metrics.append(dict(output=path.stem,cross_scene=cross,full_eligible=len(legal),
                    full_source_map=100*np.mean([r['ap'] for r in legal]),
                    anchor_exposures=exposures,common_eligible=len(common),pool_missing_positive=missing,
                    common_candidate_map=100*paired[:,0].mean(),common_full_map=100*paired[:,1].mean()))
        result['conditions'].append(dict(directory=cond['directory'],metrics=metrics))
        write_json(args.root/'coverage.json',result)
        print(json.dumps(dict(event='source_coverage_analyzed',condition=cond['directory'],complete=len(result['conditions']))),flush=True)
    assert len(result['conditions'])==12
    result['status']='COMPLETE_SOURCE_COVERAGE';write_json(args.root/'coverage.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--mode',choices=('extract','analyze','math'),required=True)
    args=parser.parse_args()
    if args.mode=='math':print(json.dumps(math_check()))
    else:(extract if args.mode=='extract' else analyze)(args)
