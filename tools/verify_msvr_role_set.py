"""Draft full role-set training/retrieval replay; not yet runtime validated."""
import argparse
from collections import OrderedDict
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch

from tools.train_msvr_role_set import context, ENDPOINTS, OUTPUT_WIDTHS, EXPERTS
from tools.train_msvr310_signal_oof import sha256, write_json
from tools.verify_msvr310_source_style import state_sha, bootstrap


def training(directory,tr,fold,protocol,md,spec,config,endpoint,mode):
    audits=[json.loads(s) for s in (directory/'memory_steps.jsonl').read_text().splitlines()]
    for name,proof in tr['audit_files'].items():
        assert sha256(directory/name)==proof['sha256'] and (directory/name).stat().st_size==proof['bytes']
    assert len(audits)==tr['optimizer_steps']==(260 if mode=='comparison' else 100 if mode=='overfit' else 8)
    assert tr['missing_nonzero_gradients']==[] and tr['nonzero_gradient_tensors']==tr['trainable_tensors']==203
    assert tr['overflow_events']==0
    assert tr['frozen_state_before_sha256']==tr['frozen_state_after_sha256']
    assert tr['signal_state_before_sha256']==tr['signal_state_after_sha256']
    assert tr['fresh_history_both_endpoints'] and tr['history_anchor_count']==0
    assert tr['history_candidate_vjp_applied']
    cache=OrderedDict();matrix_elements=max_error=candidates=harder=active=0
    fresh_forwards=64;vjp_forwards=direct_checks=0
    warmup=spec['memory']['warmup_steps' if mode=='comparison' else mode+'_warmup_steps']
    with (directory/'memory_distances.f32').open('rb') as stream:
        for index,(row,audit) in enumerate(zip(tr['steps'],audits,strict=True)):
            assert row['step']==audit['step']==index+1 and audit['zero_based_step']==index
            indices=row['sampled_record_indices']
            assert indices==audit['record_indices']==md['batches'][0 if mode=='overfit' else index]['record_indices']
            assert set(indices)<=set(fold['source_record_indices'])
            assert audit['identities']==[protocol['records'][i]['identity'] for i in indices]
            assert audit['scenes']==[protocol['records'][i]['scene'] for i in indices]
            for i in [i for i,step in cache.items() if index-step>8]:del cache[i]
            wanted=[dict(record_index=i,identity=protocol['records'][i]['identity'],scene=protocol['records'][i]['scene'],
                         age=index-step,stored_step=step) for i,step in cache.items() if i not in set(indices)]
            assert wanted==audit['memory']
            assert audit['warmup_steps']==warmup and audit['replacement_active']==(index>=warmup)
            assert audit['coordinate_rule']=='fresh' and audit['current_anchor_count']==64 and audit['history_anchor_count']==0
            assert audit['history_candidate_vjp_applied']
            m=len(wanted)
            assert stream.tell()==audit['distance_offset_bytes']
            raw=np.fromfile(stream,dtype=np.float32,count=4*64*(64+m))
            assert len(raw)==audit['distance_float_count']==4*64*(64+m)
            assert np.isfinite(raw).all() and (raw>=0).all()
            distance=raw.reshape(4,64,64+m)
            dc,dm=distance[0,:,:64],distance[0,:,64:]
            count=64*len({r['stored_step'] for r in wanted})
            assert count==audit['fresh_role_record_forwards'];fresh_forwards+=count
            ids=np.array(audit['identities']);scenes=np.array(audit['scenes'])
            positive=(ids[:,None]==ids[None,:]) & ~np.eye(64,dtype=bool)
            negative=ids[:,None]!=ids[None,:]
            hp=np.where(positive,dc,-np.inf).max(1);hn=np.where(negative,dc,np.inf).min(1)
            mids=np.array([r['identity'] for r in wanted]);msc=np.array([r['scene'] for r in wanted])
            mp=ids[:,None]==mids[None,:];mn=~mp
            if m:
                mph=np.where(mp,dm,-np.inf).max(1);mnh=np.where(mn,dm,np.inf).min(1)
                uhp=np.maximum(hp,mph);uhn=np.minimum(hn,mnh)
                harder_p=int((mph>hp).sum());harder_n=int((mnh<hn).sum())
            else:uhp,uhn=hp,hn;harder_p=harder_n=0
            basic=float(np.maximum(0,hp-hn+np.float32(.3)).mean())
            pooled=float(np.maximum(0,uhp-uhn+np.float32(.3)).mean())
            negative_mask=np.concatenate((negative,mn),axis=1)
            proposals=np.stack([np.where(negative_mask,d,np.inf).argmin(1) for d in distance],axis=1)
            selected=np.zeros_like(negative_mask)
            for anchor in range(64):selected[anchor,proposals[anchor]]=True
            counts=selected.sum(1)
            extra=selected.copy();extra[np.arange(64),proposals[:,0]]=False
            hinges=np.maximum(0,uhp[:,None]-distance[0]+np.float32(.3))
            hard_terms=np.maximum(0,uhp-uhn+np.float32(.3))
            role_set=float(((hard_terms+(hinges*extra).sum(1))/counts).mean())
            relation=audit['relation_objective']
            assert audit['proposal_order']==['fused',*EXPERTS]
            assert proposals.tolist()==relation['proposals'] and counts.tolist()==relation['negative_counts']
            assert ((hinges>0)&extra).sum(1).tolist()==relation['extra_active_counts']
            target=pooled if endpoint=='control' else role_set
            expected=dict(memory_records=m,memory_positive_pairs=int(mp.sum()),memory_negative_pairs=int(mn.sum()),
                          memory_cross_scene_positive_pairs=int((mp & (scenes[:,None]!=msc[None,:])).sum()),
                          memory_negative_violations_against_batch_hard_positive=int((mn & (dm<hp[:,None]+np.float32(.3))).sum()),
                          harder_positive_anchors=harder_p,harder_negative_anchors=harder_n,
                          current_wrong_order_anchors=int((hp>=hn).sum()),expanded_wrong_order_anchors=int((uhp>=uhn).sum()),
                          expanded_hinge_positive_anchors=int((uhp-uhn+np.float32(.3)>0).sum()),
                          maximum_memory_age=max((r['age'] for r in wanted),default=0))
            assert all(expected[k]==audit['statistics'][k] for k in expected)
            for actual,computed in ((audit['original_triplet'],basic),(audit['statistics']['current_triplet'],basic),
                                    (audit['statistics']['expanded_triplet'],pooled),
                                    (relation['hard_loss'],pooled),(relation['role_set_loss'],role_set),
                                    (row['components']['triplet_fused'],target if index>=warmup else basic)):
                error=abs(actual-computed);max_error=max(max_error,error);assert error<2e-6
            c=row['components'];w=config['LOSS']
            assert set(c)=={'id_fused','triplet_fused',*[key+'_'+e for e in EXPERTS for key in ('id','triplet','id_residual','triplet_residual')]}
            total=w['ID_FUSED']*c['id_fused']+w['TRIPLET_FUSED']*c['triplet_fused']
            total+=sum(w['ID_BRANCH']*c['id_'+e]+w['TRIPLET_BRANCH']*c['triplet_'+e]
                       +w['ID_RESIDUAL']*c['id_residual_'+e]+w['TRIPLET_RESIDUAL']*c['triplet_residual_'+e] for e in EXPERTS)
            assert abs(total-row['loss'])<1e-5 and row['amp_scale_after']>=row['amp_scale_before']
            norms=np.array(audit['historical_leaf_upstream_norms'])
            assert len(norms)==m and np.isfinite(norms).all() and (norms>=0).all()
            groups=sorted({r['stored_step'] for i,r in enumerate(wanted) if norms[i]>0})
            assert groups==audit['history_vjp_groups'] and 64*len(groups)==audit['history_vjp_record_forwards']
            vjp_forwards+=64*len(groups)
            assert audit['gradient_weight']==w['TRIPLET_FUSED']==1
            assert all(audit[k] for k in ('selected_reencoding_bitwise','history_rng_buffers_preserved',
                                          'history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise'))
            assert set(audit['roles'])==set(audit['applied_gradients'])==set(EXPERTS)
            for role in EXPERTS:
                pair=audit['roles'][role]['total_vs_history'];both=audit['roles'][role]['total_vs_both'];applied=audit['applied_gradients'][role]
                for g in (pair,both,applied):
                    a,b,d=g['first_norm'],g['second_norm'],g['difference_norm'];cos=g['cosine']
                    assert all(np.isfinite(x) and x>=0 for x in (a,b,d))
                    assert (cos is None)==(a==0 or b==0)
                    if cos is not None:
                        assert np.isfinite(cos) and abs(cos)<=1.00001
                        assert abs(d*d-(a*a+b*b-2*a*b*cos))<=1e-7*max(1,a*a+b*b)
                assert abs(both['difference_norm']-pair['second_norm'])<=1e-6*max(1,pair['second_norm'])
                wanted_gradient=both
                for key in wanted_gradient:
                    a,b=applied[key],wanted_gradient[key]
                    assert a==b if a is None or b is None else abs(a-b)<=1e-6*max(1,abs(a),abs(b))
            direct=audit['direct_single_group_check']
            if direct:
                assert mode=='capacity' and len({r['stored_step'] for r in wanted})==1
                assert direct['all_four_reencoded_outputs_bitwise_equal']
                assert direct['relative_l2_error']<=.005
                assert abs(direct['relative_l2_error']-direct['difference_norm']/max(direct['first_norm'],1e-12))<1e-12
                direct_checks+=1
            if index>=warmup:
                for i in indices:cache.pop(i,None);cache[i]=index
                while len(cache)>512:cache.popitem(last=False)
            candidates+=m;harder+=harder_p+harder_n;active+=int(m>0);matrix_elements+=len(raw)
        assert stream.read()==b''
    assert fresh_forwards==tr['extra_fresh_role_record_forwards'] and vjp_forwards==tr['extra_history_vjp_record_forwards']
    assert direct_checks==(1 if mode=='capacity' else 0) and 64*direct_checks==tr['extra_direct_check_record_forwards']
    for row in tr['history']:
        selected=[r for r in tr['steps'] if r['epoch']==row['epoch']]
        assert len(selected)==row['optimizer_steps']
        assert float(np.mean([r['loss'] for r in selected]))==row['mean_loss']
    if mode!='overfit':
        witness=tr['historical_parameter_gradient_witness']
        assert witness['roles']==audits[witness['step']-1]['roles']
        assert all(x['total_vs_history']['second_norm']>0 for x in witness['roles'].values())
        assert candidates>0 and harder>0 and vjp_forwards>0
    else:assert candidates==0 and vjp_forwards==0
    return dict(steps=len(audits),matrix_elements=matrix_elements,max_loss_error=max_error,
                extra_fresh_role_record_forwards=fresh_forwards,extra_history_vjp_record_forwards=vjp_forwards,
                historical_candidates=candidates,harder_anchor_exposures=harder,active_historical_steps=active,
                direct_single_group_checks=direct_checks,all_record_age_identity_masks_exact=True),[(r['record_indices'],r['pixel_sha256']) for r in audits]


def run(args):
    started = time.perf_counter()
    spec, (config, _base, _cfg, _env, protocol, baseline, metadata) = context(args.config)
    directory = args.run_dir
    summary = json.loads((directory/'summary.json').read_bytes())
    m0 = summary['mode'] == 'm0'
    torch.set_num_threads(4 if m0 else 56)
    assert summary['status'] in (('PASS_ENGINEERING_ONLY',) if m0 else ('Q1_PASS', 'Q1_FAIL'))
    assert summary['config_sha256'] == sha256(args.config) and summary['optimizer_steps'] == (248 if m0 else 1560)
    assert summary['heldout_record_forwards'] == (0 if m0 else 2064) and summary['official_image_reads'] == 0
    files = {}
    def record(path): files[str(path)] = dict(bytes=Path(path).stat().st_size, sha256=sha256(path))
    record(directory/'summary.json')
    training_checks = []
    distance_elements = 0
    aps = {e: {o: [] for o in OUTPUT_WIDTHS} for e in ENDPOINTS}
    firsts = {e: {o: [] for o in OUTPUT_WIDTHS} for e in ENDPOINTS}
    fold_maps = {e: [] for e in ENDPOINTS}
    query_ids = []
    for fold, row, b0, md in zip(protocol['folds'], summary['folds'], baseline['folds'], metadata['folds'], strict=True):
        assert fold['fold'] == row['fold'] and not set(fold['source_ids']) & set(fold['heldout_ids'])
        original = torch.load(b0['checkpoint'], map_location='cpu', weights_only=True)['model_state_dict']
        paired = []
        for end in ENDPOINTS:
            item = row['endpoints'][end];local = directory/f"fold_{fold['fold']}_{end}"
            assert item == json.loads((local/'receipt.json').read_bytes())
            tr = item['training'];assert tr == json.loads((local/'training.json').read_bytes())
            assert all(item['engineering_checks'].values())
            check, pixels = training(local, tr, fold, protocol, md, spec, config, end, 'capacity' if m0 else 'comparison')
            training_checks.append(dict(fold=fold['fold'], endpoint=end, **check));paired.append(pixels)
            assert tr['initial_state_sha256'] == item['initialization']['initial_state_sha256']
            path = Path(item['checkpoint']);assert sha256(path) == item['checkpoint_sha256']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['config_sha256'] == sha256(args.config) and payload['binding'] == item['initialization']
            assert payload['source_ids'] == fold['source_ids'] and payload['heldout_ids'] == fold['heldout_ids']
            state = {k: original[n] for k, n in payload['baseline_aliases'].items()};state.update(payload['role_state_dict'])
            assert state_sha(state) == tr['final_state_sha256'] == item['strict_reload_state_sha256']
            for p in (path, local/'receipt.json', local/'training.json', local/'memory_steps.jsonl', local/'memory_distances.f32'): record(p)
            if m0:
                assert all(item['preflight'][k] for k in ('standalone_signal_bitwise_equal', 'zero_update_replay_bitwise_equal', 'all_model_state_unchanged'))
                assert item['strict_reload_all_outputs_bitwise_equal']
                continue
            retrieval = item['retrieval']
            arrays = torch.load(local/'retrieval_arrays.pt', map_location='cpu', weights_only=True)
            ref = torch.load(Path(b0['checkpoint']).parent/'retrieval_arrays.pt', map_location='cpu', weights_only=True)
            ranks = json.loads((local/'rankings.json').read_bytes())
            assert sha256(local/'retrieval_arrays.pt') == retrieval['retrieval_arrays_sha256']
            assert sha256(local/'rankings.json') == retrieval['rankings_sha256']
            positions = [q['gallery_position'] for q in fold['query_rows']]
            assert arrays['query_gallery_positions'] == positions and arrays['gallery_record_indices'] == fold['gallery_record_indices']
            gallery = [protocol['records'][i] for i in fold['gallery_record_indices']]
            ids = np.array([r['identity'] for r in gallery]);scenes = np.array([r['scene'] for r in gallery])
            if end == 'control': query_ids.extend(ids[positions])
            assert torch.equal(arrays['features']['baseline_only'], ref['features'])
            assert torch.equal(arrays['distances']['baseline_only'], ref['distances'])
            assert torch.equal(arrays['features']['fused'][:, :3072], arrays['features']['baseline_only'])
            maps = {}
            for name, width in OUTPUT_WIDTHS.items():
                values = arrays['features'][name];distance = arrays['distances'][name]
                assert values.shape == (len(gallery), width) and torch.isfinite(values).all()
                unit = torch.nn.functional.normalize(values.float(), dim=1);q = unit[positions]
                actual = q.square().sum(1, keepdim=True)+unit.square().sum(1)[None]
                actual.addmm_(q, unit.T, beta=1, alpha=-2)
                assert torch.equal(actual, distance)
                order = np.argsort(distance.numpy(), axis=1);assert order.tolist() == ranks[name]
                ap, first = [], []
                for qi, indices in enumerate(order):
                    qpos = positions[qi]
                    legal = [g for g in indices if not (ids[g] == ids[qpos] and scenes[g] == scenes[qpos])]
                    positive = [rank+1 for rank, g in enumerate(legal) if ids[g] == ids[qpos]]
                    assert positive
                    ap.append(sum((k+1)/rank for k, rank in enumerate(positive))/len(positive));first.append(positive[0])
                score = retrieval['outputs'][name]
                assert np.allclose(ap, score['average_precision'], rtol=0, atol=1e-14) and first == score['first_match_rank']
                maps[name] = float(np.mean(ap)*100)
                assert abs(maps[name]-score['metrics']['mAP']) < 1e-10
                for k in (1, 5, 10): assert abs(float(np.mean(np.array(first) <= k)*100)-score['metrics'][f'Rank-{k}']) < 1e-10
                aps[end][name].extend(ap);firsts[end][name].extend(first);distance_elements += distance.numel()
            fold_maps[end].append(maps)
            record(local/'retrieval_arrays.pt');record(local/'rankings.json')
        assert paired[0] == paired[1] and row['all_paired_source_pixels_exact']
        assert row['endpoints']['control']['initialization'] == row['endpoints']['role_set']['initialization']
    if m0:
        pair = []
        for end in ENDPOINTS:
            row = summary['overfit'][end];tr = row['training'];local = directory/('overfit_'+end)
            check, pixels = training(local, tr, protocol['folds'][0], protocol, metadata['folds'][0], spec, config, end, 'overfit')
            training_checks.append(dict(fold=0, endpoint=end, mode='overfit', **check));pair.append(pixels)
            assert all(row['checks'].values())
            from tools.run_signal_preserving_v5 import overfit_loss_floor
            floor = overfit_loss_floor(config, num_classes=len(protocol['folds'][0]['source_ids']))
            ratio = (tr['steps'][-1]['loss']-floor)/(tr['steps'][0]['loss']-floor)
            assert row['gate']['minimum_loss'] == floor and abs(ratio-row['gate']['loss_ratio']) < 1e-12 and ratio <= .1
            for p in local.iterdir(): record(p)
        assert pair[0] == pair[1]
    else:
        assert len(query_ids) == 600 and len(set(query_ids)) == 60
        ids = np.array(query_ids);comparison = summary['comparison']
        maps = {e: {o: float(np.mean(aps[e][o])*100) for o in OUTPUT_WIDTHS} for e in ENDPOINTS}
        for end in ENDPOINTS:
            recorded = comparison['endpoints'][end]
            gains = {o: maps[end][o]-maps[end]['baseline_only'] for o in OUTPUT_WIDTHS}
            fg = [r['fused']-r['baseline_only'] for r in fold_maps[end]]
            lower = bootstrap((np.array(aps[end]['fused'])-np.array(aps[end]['baseline_only']))*100, ids)
            checks = dict(fused_gain_at_least_1pp=gains['fused'] >= 1,
                          all_fold_fused_gains_nonnegative=all(g >= 0 for g in fg),
                          all_full_branches_not_below_signal=all(gains[e] >= 0 for e in EXPERTS),
                          identity_bootstrap_lower_positive=lower > 0,
                          fused_strictly_best=all(maps[end]['fused'] > maps[end][e] for e in ('baseline_only', *EXPERTS)))
            assert checks == recorded['scientific_checks'] and all(checks.values()) == recorded['scientific_passed']
            assert abs(lower-recorded['identity_bootstrap']['lower_bound_pp']) < 1e-10
            assert np.allclose(fg, recorded['fold_fused_gains_pp'], rtol=0, atol=1e-10)
            for output in OUTPUT_WIDTHS:
                assert abs(maps[end][output]-recorded['metrics'][output]['mAP']) < 1e-10
                assert abs(gains[output]-recorded['gains_over_signal_pp'][output]) < 1e-10
                for k in (1, 5, 10): assert abs(float(np.mean(np.array(firsts[end][output]) <= k)*100)-recorded['metrics'][output][f'Rank-{k}']) < 1e-10
                for r in recorded['per_identity']:
                    mask = ids == r['identity'];assert int(mask.sum()) == r['query_count']
                    assert abs(float(np.array(aps[end][output])[mask].mean()*100)-r['map_by_output'][output]) < 1e-10
                if output != 'baseline_only':
                    delta = (np.array(aps[end][output])-np.array(aps[end]['baseline_only']))*100
                    a = np.array(firsts[end][output]);b = np.array(firsts[end]['baseline_only'])
                    changed = dict(ap_improved=int((delta > 0).sum()), ap_declined=int((delta < 0).sum()),
                                   ap_unchanged=int((delta == 0).sum()), rank1_repaired=int(((b > 1) & (a == 1)).sum()),
                                   rank1_new_errors=int(((b == 1) & (a > 1)).sum()))
                    assert changed == recorded['query_changes'][output]
        gain = {o: float(((np.array(aps['role_set'][o])-np.array(aps['control'][o]))*100).mean()) for o in OUTPUT_WIDTHS}
        lower = bootstrap((np.array(aps['role_set']['fused'])-np.array(aps['control']['fused']))*100, ids)
        fg = [b['fused']-a['fused'] for a, b in zip(fold_maps['control'], fold_maps['role_set'], strict=True)]
        candidate = maps['role_set']
        checks = dict(fused_gain_at_least_1pp=gain['fused'] >= 1, all_fold_fused_nonnegative=all(g >= 0 for g in fg),
                      all_role_gains_nonnegative=all(gain[e] >= 0 for e in EXPERTS), paired_identity_bootstrap_lower_positive=lower > 0,
                      candidate_fused_strictly_best=all(candidate['fused'] > candidate[e] for e in ('baseline_only', *EXPERTS)))
        assert checks == comparison['paired_checks'] and abs(lower-comparison['paired_bootstrap_lower_pp']) < 1e-10
        assert np.allclose(fg, comparison['fold_fused_gains_mAP'], rtol=0, atol=1e-10)
        assert comparison['paired_pass'] == all(checks.values())
        assert comparison['vehicle_baseline_pass'] == comparison['endpoints']['role_set']['scientific_passed']
        assert all(abs(gain[o]-comparison['matched_gains_mAP'][o]) < 1e-10 for o in OUTPUT_WIDTHS)
        for r in comparison['paired_per_identity']:
            mask = ids == r['identity'];assert int(mask.sum()) == r['query_count']
            for o in OUTPUT_WIDTHS:
                d = (np.array(aps['role_set'][o])-np.array(aps['control'][o]))*100
                assert abs(float(d[mask].mean())-r['gains_mAP'][o]) < 1e-10
        passed = all(checks.values()) and comparison['endpoints']['role_set']['scientific_passed']
        assert comparison['next_phase_qualified'] == passed and summary['status'] == ('Q1_PASS' if passed else 'Q1_FAIL')
    assert sum(r['steps'] for r in training_checks) == (248 if m0 else 1560)
    result = dict(status='PASS_COMPLETE_ROLE_SET_M0' if m0 else 'PASS_COMPLETE_ROLE_SET_Q1',
                  verified_at=datetime.now().astimezone().isoformat(), summary_sha256=sha256(directory/'summary.json'),
                  training_checks=training_checks, checked_training_steps=sum(r['steps'] for r in training_checks),
                  checked_memory_distance_elements=sum(r['matrix_elements'] for r in training_checks),
                  checked_history_vjp_record_forwards=sum(r['extra_history_vjp_record_forwards'] for r in training_checks),
                  checked_retrieval_distance_and_rank_elements=distance_elements, files=files,
                  model_forwards=0, optimizer_updates=0, elapsed_seconds=time.perf_counter()-started,
                  scope='Complete saved distance/mask/loss and final checkpoint/retrieval replay; per-step feature and parameter gradients are runtime witnesses, not independently regenerated training.')
    assert not args.output.exists();write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'training_checks')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
