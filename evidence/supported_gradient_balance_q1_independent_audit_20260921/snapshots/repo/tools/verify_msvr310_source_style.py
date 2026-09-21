#!/usr/bin/env python3
"""Full saved-checkpoint, exposure and ranking verification; CPU only."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.train_msvr310_source_style import context,ENDPOINTS,OUTPUT_WIDTHS,EXPERTS
from tools.train_msvr310_signal_oof import sha256


def state_sha(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode());digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def bootstrap(differences,identities):
    unique=np.unique(identities)
    sums=np.array([differences[identities==i].sum() for i in unique])
    counts=np.array([(identities==i).sum() for i in unique])
    rng=np.random.default_rng(42)
    selected=rng.integers(0,len(unique),size=(10000,len(unique)))
    return float(np.quantile(sums[selected].sum(1)/counts[selected].sum(1),.025,method='linear'))


def verify(args):
    started=time.time()
    config,base,_cfg,_environment,protocol,baseline,metadata=context(args.config)
    directory=args.run_dir
    summary=json.loads((directory/'summary.json').read_bytes())
    m0=summary['mode']=='m0'
    assert summary['status'] in (('PASS_ENGINEERING_ONLY',) if m0 else ('Q1_PASS','Q1_FAIL'))
    assert summary['config_sha256']==sha256(args.config)
    assert summary['optimizer_steps']==(248 if m0 else 1560)
    assert summary['heldout_record_forwards']==(0 if m0 else 2064)
    files={}
    def record(path):
        files[str(path)]=dict(bytes=Path(path).stat().st_size,sha256=sha256(path))
    record(directory/'summary.json')
    steps_checked=0;max_loss_error=0.;distance_entries=0;ranking_positions=0
    all_aps={e:{name:[] for name in OUTPUT_WIDTHS} for e in ENDPOINTS}
    all_first={e:{name:[] for name in OUTPUT_WIDTHS} for e in ENDPOINTS}
    fold_maps={e:[] for e in ENDPOINTS}
    identities=[]
    def training(row,styles,md,mode):
        nonlocal steps_checked,max_loss_error
        tr=row['training']
        assert all(row['checks' if mode=='overfit' else 'engineering_checks'].values())
        assert tr['missing_nonzero_gradients']==[] and tr['overflow_events']==0
        assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
        assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
        assert tr['initial_state_sha256']==row['initialization']['initial_state_sha256']
        assert len(styles)==tr['optimizer_steps']==(100 if mode=='overfit' else (8 if m0 else 260))
        for k,(step,style) in enumerate(zip(tr['steps'],styles,strict=True)):
            wanted=md['batches'][0 if mode=='overfit' else k]
            assert step['step']==style['step']==k+1
            assert step['sampled_record_indices']==wanted['record_indices']
            expected=wanted['style_plan']
            if mode=='overfit':expected={**expected,'active':True,'forced_active':True}
            assert style['plan']==expected
            assert style['statistics']['additional_visual_passes']==3
            assert set(style['pixel_sha256'])=={'RGB','NI','TI'}
            c=step['components'];w=config['LOSS']
            value=w['ID_FUSED']*c['id_fused']+w['TRIPLET_FUSED']*c['triplet_fused']
            value+=sum(w['ID_BRANCH']*c['id_'+e]+w['TRIPLET_BRANCH']*c['triplet_'+e]
                       +w['ID_RESIDUAL']*c['id_residual_'+e]+w['TRIPLET_RESIDUAL']*c['triplet_residual_'+e] for e in EXPERTS)
            assert np.isfinite(list(c.values())).all() and np.isfinite(step['loss'])
            max_loss_error=max(max_loss_error,abs(value-step['loss']))
            assert step['amp_scale_after']>=step['amp_scale_before']
            steps_checked+=1
        for epoch in tr['history']:
            rows=[r for r in tr['steps'] if r['epoch']==epoch['epoch']]
            assert len(rows)==epoch['optimizer_steps']
            assert float(np.mean([r['loss'] for r in rows]))==epoch['mean_loss']
        assert tr['additional_visual_passes']==3*tr['optimizer_steps']
        assert tr['active_style_steps']==sum(r['statistics']['style_active'] for r in styles)
    for fold,actual,b0,md in zip(protocol['folds'],summary['folds'],baseline['folds'],metadata['folds'],strict=True):
        assert fold['fold']==actual['fold']
        base_payload=torch.load(b0['checkpoint'],map_location='cpu',weights_only=True)
        original=base_payload['model_state_dict']
        paired=[]
        for end in ENDPOINTS:
            row=actual['endpoints'][end]
            local=directory/f"fold_{fold['fold']}_{end}"
            assert row==json.loads((local/'receipt.json').read_bytes())
            assert row['training']==json.loads((local/'training.json').read_bytes())
            tr=row['training'];styles=[json.loads(s) for s in Path(tr['style_steps_path']).read_text().splitlines()]
            assert sha256(tr['style_steps_path'])==tr['style_steps_sha256']
            training(row,styles,md,'capacity' if m0 else 'comparison')
            assert all(r['statistics']['style_active']==int(end=='source_style' and r['plan']['active']) for r in styles)
            paired.append([{k:r[k] for k in ('plan','pixel_sha256')} for r in styles])
            p=Path(row['checkpoint']);assert sha256(p)==row['checkpoint_sha256']
            payload=torch.load(p,map_location='cpu',weights_only=True)
            assert payload['binding']==row['initialization']
            assert payload['binding']['signal_checkpoint_sha256']==b0['checkpoint_sha256']
            assert payload['config_sha256']==summary['config_sha256'] and payload['fold']==fold['fold']
            assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids']
            state={k:original[source] for k,source in payload['baseline_aliases'].items()}
            state.update(payload['role_state_dict'])
            assert state_sha(state)==tr['final_state_sha256']==row['strict_reload_state_sha256']
            for path in (p,local/'receipt.json',local/'training.json',Path(tr['style_steps_path'])):record(path)
            if m0:
                assert row['strict_reload_all_outputs_bitwise_equal']
                pre=row['preflight'];assert pre['standalone_source_signal_bitwise_equal'] and pre['original_state_unchanged']
                assert [(r['enabled'],r['active']) for r in pre['field_cases']]==[(False,True),(True,True),(True,False)]
                assert [r['style_active'] for r in pre['field_cases']]==[0,1,0]
                continue
            retrieval=row['retrieval']
            array_path=local/'retrieval_arrays.pt';rank_path=local/'rankings.json'
            assert sha256(array_path)==retrieval['retrieval_arrays_sha256']
            assert sha256(rank_path)==retrieval['rankings_sha256']
            arrays=torch.load(array_path,map_location='cpu',weights_only=True)
            b0_arrays=torch.load(Path(b0['checkpoint']).parent/'retrieval_arrays.pt',map_location='cpu',weights_only=True)
            ranks=json.loads(rank_path.read_bytes())
            assert arrays['gallery_record_indices']==fold['gallery_record_indices']
            positions=[q['gallery_position'] for q in fold['query_rows']]
            assert arrays['query_gallery_positions']==positions
            rows=[protocol['records'][i] for i in fold['gallery_record_indices']]
            ids=np.array([r['identity'] for r in rows]);scenes=np.array([r['scene'] for r in rows])
            if end=='control':identities.extend(ids[positions].tolist())
            assert torch.equal(arrays['features']['baseline_only'],b0_arrays['features'])
            assert torch.equal(arrays['distances']['baseline_only'],b0_arrays['distances'])
            assert torch.equal(arrays['features']['fused'][:,:3072],arrays['features']['baseline_only'])
            for name,width in OUTPUT_WIDTHS.items():
                feature=arrays['features'][name];distance=arrays['distances'][name]
                assert tuple(feature.shape)==(len(rows),width) and torch.isfinite(feature).all()
                unit=torch.nn.functional.normalize(feature.float(),dim=1);q=unit[positions]
                calculated=q.square().sum(1,keepdim=True)+unit.square().sum(1)[None]
                calculated.addmm_(q,unit.T,beta=1,alpha=-2)
                assert torch.equal(calculated,distance)
                order=np.argsort(distance.numpy(),axis=1)
                assert order.tolist()==ranks[name]
                ap=[];first=[]
                for qi,gallery_order in enumerate(order):
                    qpos=positions[qi]
                    legal=[int(g) for g in gallery_order if not(ids[g]==ids[qpos] and scenes[g]==scenes[qpos])]
                    positive=[rank+1 for rank,g in enumerate(legal) if ids[g]==ids[qpos]]
                    assert positive
                    ap.append(sum((i+1)/rank for i,rank in enumerate(positive))/len(positive))
                    first.append(positive[0])
                score=retrieval['outputs'][name]
                assert np.allclose(ap,score['average_precision'],atol=1e-14,rtol=0)
                assert first==score['first_match_rank']
                all_aps[end][name].extend(ap)
                all_first[end][name].extend(first)
                assert abs(float(np.mean(ap)*100)-score['metrics']['mAP'])<1e-10
                for k in (1,5,10):assert abs(float(np.mean(np.array(first)<=k)*100)-score['metrics'][f'Rank-{k}'])<1e-10
                distance_entries+=distance.numel();ranking_positions+=order.size
            fold_maps[end].append({name:float(np.mean(retrieval['outputs'][name]['average_precision'])*100)
                                   for name in OUTPUT_WIDTHS})
            record(array_path);record(rank_path)
        assert paired[0]==paired[1] and actual['all_paired_inputs_and_style_plans_exact']
        assert actual['endpoints']['control']['initialization']==actual['endpoints']['source_style']['initialization']
    if m0:
        paired=[]
        for end,row in summary['overfit'].items():
            tr=row['training'];path=Path(tr['style_steps_path'])
            styles=[json.loads(s) for s in path.read_text().splitlines()]
            assert sha256(path)==tr['style_steps_sha256']
            training(row,styles,metadata['folds'][0],'overfit')
            assert all(r['statistics']['style_active']==int(end=='source_style') for r in styles)
            paired.append([{k:r[k] for k in ('plan','pixel_sha256')} for r in styles])
            assert all(r==paired[-1][0] for r in paired[-1])
            gate=row['gate'];floor=gate['minimum_loss']
            from tools.run_signal_preserving_v5 import overfit_loss_floor
            assert floor==overfit_loss_floor(config,num_classes=len(protocol['folds'][0]['source_ids']))
            ratio=(tr['steps'][-1]['loss']-floor)/(tr['steps'][0]['loss']-floor)
            assert abs(ratio-gate['loss_ratio'])<1e-12 and ratio<=.1
            record(path);record(path.parent/'training.json')
        assert paired[0]==paired[1] and steps_checked==248
    else:
        assert steps_checked==1560 and len(identities)==600 and len(set(identities))==60
        identities=np.array(identities);c=summary['comparison']
        maps={end:{name:float(np.mean(all_aps[end][name])*100) for name in OUTPUT_WIDTHS} for end in ENDPOINTS}
        for end in ENDPOINTS:
            for name in OUTPUT_WIDTHS:
                assert abs(maps[end][name]-c['endpoints'][end]['metrics'][name]['mAP'])<1e-10
                for k in (1,5,10):
                    assert abs(float(np.mean(np.array(all_first[end][name])<=k)*100)-c['endpoints'][end]['metrics'][name][f'Rank-{k}'])<1e-10
                for row in c['endpoints'][end]['per_identity']:
                    chosen=identities==row['identity'];assert int(chosen.sum())==row['query_count']
                    assert abs(float(np.array(all_aps[end][name])[chosen].mean()*100)-row['map_by_output'][name])<1e-10
            gain=(np.array(all_aps[end]['fused'])-np.array(all_aps[end]['baseline_only']))*100
            lower=bootstrap(gain,identities)
            assert abs(lower-c['endpoints'][end]['identity_bootstrap']['lower_bound_pp'])<1e-10
            gains={name:maps[end][name]-maps[end]['baseline_only'] for name in OUTPUT_WIDTHS}
            fold_gains=[m['fused']-m['baseline_only'] for m in fold_maps[end]]
            assert np.allclose(fold_gains,c['endpoints'][end]['fold_fused_gains_pp'],rtol=0,atol=1e-10)
            checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,
                        all_fold_fused_gains_nonnegative=all(g>=0 for g in fold_gains),
                        all_full_branches_not_below_signal=all(gains[e]>=0 for e in EXPERTS),
                        identity_bootstrap_lower_positive=lower>0,
                        fused_strictly_best=all(maps[end]['fused']>maps[end][e] for e in ('baseline_only',*EXPERTS)))
            assert checks==c['endpoints'][end]['scientific_checks']
            assert all(checks.values())==c['endpoints'][end]['scientific_passed']
            for name in EXPERTS+('fused',):
                delta=(np.array(all_aps[end][name])-np.array(all_aps[end]['baseline_only']))*100
                first=np.array(all_first[end][name]);ref=np.array(all_first[end]['baseline_only'])
                changes=dict(ap_improved=int((delta>0).sum()),ap_declined=int((delta<0).sum()),
                             ap_unchanged=int((delta==0).sum()),rank1_repaired=int(((ref>1)&(first==1)).sum()),
                             rank1_new_errors=int(((ref==1)&(first>1)).sum()))
                assert changes==c['endpoints'][end]['query_changes'][name]
        paired_gain=(np.array(all_aps['source_style']['fused'])-np.array(all_aps['control']['fused']))*100
        paired_lower=bootstrap(paired_gain,identities)
        assert abs(paired_lower-c['paired_bootstrap_lower_pp'])<1e-10
        for name in OUTPUT_WIDTHS:
            d=(np.array(all_aps['source_style'][name])-np.array(all_aps['control'][name]))*100
            assert abs(float(d.mean())-c['matched_gains_mAP'][name])<1e-10
            for row in c['paired_per_identity']:
                chosen=identities==row['identity'];assert int(chosen.sum())==row['query_count']
                assert abs(float(d[chosen].mean())-row['gains_mAP'][name])<1e-10
        gains={name:float(((np.array(all_aps['source_style'][name])-np.array(all_aps['control'][name]))*100).mean()) for name in OUTPUT_WIDTHS}
        fold_gains=[b['fused']-a['fused'] for a,b in zip(fold_maps['control'],fold_maps['source_style'],strict=True)]
        assert np.allclose(fold_gains,c['fold_fused_gains_mAP'],rtol=0,atol=1e-10)
        checks=dict(fused_gain_at_least_1pp=gains['fused']>=1,
                    all_fold_fused_nonnegative=all(g>=0 for g in fold_gains),
                    all_role_gains_nonnegative=all(gains[e]>=0 for e in EXPERTS),
                    paired_identity_bootstrap_lower_positive=paired_lower>0,
                    candidate_fused_strictly_best=all(maps['source_style']['fused']>maps['source_style'][e] for e in ('baseline_only',*EXPERTS)))
        assert checks==c['paired_checks']
        assert c['paired_pass']==all(checks.values())
        assert c['vehicle_baseline_pass']==all(c['endpoints']['source_style']['scientific_checks'].values())
        assert c['next_phase_qualified']==(c['paired_pass'] and c['vehicle_baseline_pass'])
        assert summary['status']==('Q1_PASS' if c['next_phase_qualified'] else 'Q1_FAIL')
    result=dict(status='PASS_COMPLETE_MSVR_STYLE_M0' if m0 else 'PASS_COMPLETE_MSVR_STYLE_Q1',
                verified_at=datetime.now().astimezone().isoformat(),summary_sha256=sha256(directory/'summary.json'),
                verifier_sha256=sha256(__file__),checked_training_steps=steps_checked,checked_distance_entries=distance_entries,
                checked_full_ranking_positions=ranking_positions,maximum_saved_loss_double_reassembly_error=max_loss_error,
                loss_reassembly_scope='Diagnostic double arithmetic from saved components; original AMP intermediate dtypes not saved.',
                files=files,model_forwards=0,optimizer_updates=0,elapsed_seconds=time.time()-started)
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args())
