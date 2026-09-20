"""Read-only CPU audit. No project imports; torch only deserializes checkpoint tensors."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import hashlib
import json
import math
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch

started=time.monotonic()
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
root=Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')
run=root/'m0'
def read(p): return json.loads(Path(p).read_bytes())
def sha(p):
    value=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()
files={}
def receipt(p, expected=None):
    actual=sha(p)
    if expected is not None:assert actual==expected,(str(p),'sha256')
    files[str(p)]=dict(bytes=Path(p).stat().st_size,sha256=actual)
    return actual
def tensor_state_sha(state):
    h=hashlib.sha256()
    for name in sorted(state):
        value=state[name].detach().cpu().contiguous().numpy()
        assert np.isfinite(value).all(),name
        h.update(name.encode('utf-8'));h.update(value.tobytes())
    return h.hexdigest()

spec_path=repo/'configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json'
spec=read(spec_path);config_hash=receipt(spec_path)
base=read(repo/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')
protocol_path=repo/base['DATA']['PROTOCOL'];protocol=read(protocol_path)
receipt(protocol_path,base['DATA']['PROTOCOL_SHA256'])
metadata_path=repo/base['SOURCE_METADATA']['PATH'];md=read(metadata_path)
receipt(metadata_path,base['SOURCE_METADATA']['SHA256'])
baseline=read(base['BASELINE']['SUMMARY']);receipt(base['BASELINE']['SUMMARY'],base['BASELINE']['SUMMARY_SHA256'])
summary=read(run/'summary.json');receipt(run/'summary.json')
cpu=read(root/'m0_cpu.json');receipt(root/'m0_cpu.json')
pipeline=read(root/'pipeline.json')
assert summary['status']=='PASS_ENGINEERING_ONLY' and summary['mode']=='m0'
assert summary['config_sha256']==config_hash and summary['optimizer_steps']==248
assert summary['heldout_record_forwards']==summary['official_image_reads']==0
assert summary['seed']==42 and summary['history_anchors']==0
assert cpu['status']=='PASS_COMPLETE_CROSS_SCENE_SMOOTH_AP_M0'
assert cpu['summary_sha256']==sha(run/'summary.json') and cpu['checked_training_steps']==248
stage_receipts={x['stage']:x for x in pipeline['stages'] if x['stage'] in ('t0','m0','m0_cpu')}
assert set(stage_receipts)=={'t0','m0','m0_cpu'}
assert all(x['exit_code']==0 for x in stage_receipts.values())
assert summary['project_commit']==pipeline['code_commit']
assert pipeline['config_sha256']==config_hash

roles=('cnn','transformer','mamba')
weights={'id_fused':base['LOSS']['ID_FUSED'],'triplet_fused':base['LOSS']['TRIPLET_FUSED']}
for role in roles:
    for prefix,key in (('id','ID_BRANCH'),('triplet','TRIPLET_BRANCH'),('id_residual','ID_RESIDUAL'),('triplet_residual','TRIPLET_RESIDUAL')):
        weights[prefix+'_'+role]=base['LOSS'][key]

def approximate_ap(distances, current_ids, candidate_ids, current_scenes, candidate_scenes, cross_scene):
    ap=[];positive_counts=[];ignored_counts=[];same_scene_negative_counts=[]
    square=np.asarray(distances,dtype=np.float64)**2
    for anchor in range(len(current_ids)):
        same=candidate_ids==current_ids[anchor]
        if cross_scene:
            positives=same & (candidate_scenes!=current_scenes[anchor])
            allowed=~(same & (candidate_scenes==current_scenes[anchor]))
        else:
            positives=same.copy();positives[anchor]=False
            allowed=np.ones(len(candidate_ids),dtype=bool);allowed[anchor]=False
        positive_positions=np.flatnonzero(positives)
        positive_counts.append(int(positives.sum()))
        ignored_counts.append(int((~allowed).sum()))
        same_scene_negative_counts.append(int(((~same)&(candidate_scenes==current_scenes[anchor])&allowed).sum()))
        precisions=[]
        for target in positive_positions:
            # s_candidate - s_target = (d_target^2 - d_candidate^2)/2.
            contribution=.5+.5*np.tanh((square[anchor,target]-square[anchor])/(4*.01))
            other=allowed.copy();other[target]=False
            numerator=1+contribution[other & positives].sum(dtype=np.float64)
            denominator=1+contribution[other].sum(dtype=np.float64)
            precisions.append(float(numerator/denominator))
        ap.append(math.fsum(precisions)/len(precisions) if precisions else 0.)
    ap=np.array(ap);counts=np.array(positive_counts);eligible=counts>0
    loss=1-float(np.mean(ap[eligible])) if eligible.any() else 0.
    return loss,ap,counts,ignored_counts,same_scene_negative_counts

max_errors=Counter();all_rows=[];run_checks=[];pair_signatures={};direct_checks=[];zero_rows=[]
def error(name,observed,expected,tolerance):
    amount=float(np.max(np.abs(np.asarray(observed,dtype=float)-np.asarray(expected,dtype=float))))
    max_errors[name]=max(max_errors[name],amount)
    assert amount<=tolerance,(name,amount,tolerance)
    return amount
def vector_norm_stats(g):
    a,b,d=(g[k] for k in ('first_norm','second_norm','difference_norm'))
    assert all(math.isfinite(x) and x>=0 for x in (a,b,d))
    cos=g['cosine']
    assert (cos is None)==(a==0 or b==0)
    if cos is not None:
        assert math.isfinite(cos) and abs(cos)<=1.00001
        error('gradient_norm_identity',d*d,a*a+b*b-2*a*b*cos,1e-7*max(1,a*a+b*b))

def audit_training(directory, training, fold, sequence, endpoint, mode):
    expected_steps=8 if mode=='capacity' else 100
    audits=[json.loads(line) for line in (directory/'memory_steps.jsonl').read_text().splitlines()]
    assert read(directory/'training.json')==training
    receipt(directory/'training.json')
    assert len(training['steps'])==len(audits)==training['optimizer_steps']==expected_steps
    assert training['epochs']==1 and training['mode']==mode
    assert training['nonzero_gradient_tensors']==training['trainable_tensors']==203
    assert training['missing_nonzero_gradients']==[] and training['overflow_events']==0
    assert training['initial_state_sha256']!=training['final_state_sha256']
    assert training['frozen_state_before_sha256']==training['frozen_state_after_sha256']
    assert training['signal_state_before_sha256']==training['signal_state_after_sha256']
    assert training['history_anchor_count']==0 and training['fresh_history_both_endpoints']
    assert training['history_candidate_vjp_applied'] and training['cache_not_in_checkpoint']
    for name,binding in training['audit_files'].items():
        receipt(directory/name,binding['sha256']);assert (directory/name).stat().st_size==binding['bytes']
    queue=OrderedDict();signatures=[];fresh_forward=64;vjp_forward=0;direct_count=0;matrix_values=0;history_count=0;active_count=0
    with (directory/'memory_distances.f32').open('rb') as matrices:
        for step,(row,saved) in enumerate(zip(training['steps'],audits,strict=True)):
            assert row['step']==saved['step']==step+1 and saved['zero_based_step']==step
            assert row['epoch']==1
            indices=sequence['batches'][0 if mode=='overfit' else step]['record_indices']
            assert indices==row['sampled_record_indices']==saved['record_indices']
            assert set(indices)<=set(fold['source_record_indices'])
            identity=np.array([protocol['records'][i]['identity'] for i in indices])
            scenes=np.array([protocol['records'][i]['scene'] for i in indices])
            assert identity.tolist()==saved['identities'] and scenes.tolist()==saved['scenes']
            assert sorted(Counter(identity.tolist()).values())==[8]*8
            for i in list(queue):
                if step-queue[i]>8:del queue[i]
            expected_memory=[dict(record_index=i,identity=protocol['records'][i]['identity'],scene=protocol['records'][i]['scene'],
                                  age=step-stored,stored_step=stored) for i,stored in queue.items() if i not in indices]
            assert expected_memory==saved['memory']
            historical=len(expected_memory);history_count+=historical
            candidate_ids=np.concatenate((identity,np.array([r['identity'] for r in expected_memory],dtype=int)))
            candidate_scenes=np.concatenate((scenes,np.array([r['scene'] for r in expected_memory],dtype=int)))
            active=step>=2;active_count+=int(active)
            assert saved['warmup_steps']==2 and saved['replacement_active']==active
            assert row['active_fused_metric']==(('smooth_ap' if endpoint=='control' else 'cross_scene_smooth_ap') if active else 'hard_triplet')
            assert saved['current_anchor_count']==64 and saved['history_anchor_count']==0 and saved['coordinate_rule']=='fresh'
            assert saved['history_candidate_vjp_applied'] and saved['gradient_weight']==1
            assert saved['saved_space_order']==['fused',*roles]
            assert matrices.tell()==saved['distance_offset_bytes']
            nvalues=4*64*(64+historical)
            values=np.fromfile(matrices,dtype='<f4',count=nvalues)
            assert len(values)==saved['distance_float_count']==nvalues
            assert np.isfinite(values).all() and (values>=0).all()
            distances=values.reshape((4,64,64+historical));matrix_values+=len(values)
            current=distances[0,:,:64]
            positives=(identity[:,None]==candidate_ids[None,:]);positives[np.arange(64),np.arange(64)]=False
            negatives=identity[:,None]!=candidate_ids[None,:]
            hp=np.max(np.where(positives[:,:64],current,-np.inf),axis=1)
            hn=np.min(np.where(negatives[:,:64],current,np.inf),axis=1)
            batch_hard=float(np.maximum(0,hp.astype(float)-hn.astype(float)+.3).mean())
            pooled_hp=np.max(np.where(positives,distances[0],-np.inf),axis=1)
            pooled_hn=np.min(np.where(negatives,distances[0],np.inf),axis=1)
            hard=float(np.maximum(0,pooled_hp.astype(float)-pooled_hn.astype(float)+.3).mean())
            standard,ap,counts,_,_=approximate_ap(distances[0],identity,candidate_ids,scenes,candidate_scenes,False)
            cross,cross_ap,cross_counts,ignored,scene_negatives=approximate_ap(distances[0],identity,candidate_ids,scenes,candidate_scenes,True)
            relation=saved['relation_objective'];eligible=int((cross_counts>0).sum())
            assert relation['temperature']==.01 and relation['positive_counts']==counts.tolist()
            assert relation['cross_scene_positive_counts']==cross_counts.tolist()
            assert relation['cross_scene_eligible_anchors']==eligible
            assert relation['cross_scene_reduction']=='mean_eligible_anchors' and relation['ineligible_ap_storage_value']==0
            standard_error=error('standard_anchor_ap',relation['per_anchor_smoothed_ap'],ap,2e-6)
            cross_error=error('cross_anchor_ap',relation['cross_scene_per_anchor_ap'],cross_ap,2e-6)
            error('standard_objective',relation['smooth_ap_loss'],standard,2e-6)
            error('cross_objective',relation['cross_scene_loss'],cross,2e-6)
            error('hard_objective',relation['hard_loss'],hard,2e-6)
            error('basic_hard_objective',saved['original_triplet'],batch_hard,2e-6)
            error('active_fused_objective',row['components']['triplet_fused'],(standard if endpoint=='control' else cross) if active else batch_hard,2e-6)
            if not eligible:
                zero_rows.append(dict(run=directory.name,step=step+1,active=active,cross_loss=cross))
                assert cross==0 and np.count_nonzero(cross_ap)==0
            components=row['components'];assert set(components)==set(weights)
            assert all(math.isfinite(v) and v>=0 for v in components.values())
            total=math.fsum(components[k]*weights[k] for k in components)
            total_error=error('fourteen_term_ledger',row['loss'],total,1e-5)
            for role_index,role in enumerate(roles,1):
                d=distances[role_index,:,:64].astype(float)
                ph=np.max(np.where(positives[:,:64],d,-np.inf),axis=1)
                nh=np.min(np.where(negatives[:,:64],d,np.inf),axis=1)
                error('branch_current_hard',components['triplet_'+role],np.maximum(0,ph-nh+.3).mean(),2e-6)
            statistics=saved['statistics']
            hist_distance=distances[0,:,64:];mp=positives[:,64:];mn=negatives[:,64:]
            computed=dict(memory_records=historical,memory_positive_pairs=int(mp.sum()),memory_negative_pairs=int(mn.sum()),
                          memory_cross_scene_positive_pairs=int((mp & (scenes[:,None]!=candidate_scenes[None,64:])).sum()),
                          memory_negative_violations_against_batch_hard_positive=int((mn & (hist_distance<hp[:,None]+np.float32(.3))).sum()),
                          harder_positive_anchors=int((pooled_hp>hp).sum()),harder_negative_anchors=int((pooled_hn<hn).sum()),
                          current_wrong_order_anchors=int((hp>=hn).sum()),expanded_wrong_order_anchors=int((pooled_hp>=pooled_hn).sum()),
                          expanded_hinge_positive_anchors=int((pooled_hp-pooled_hn+np.float32(.3)>0).sum()),
                          maximum_memory_age=max([r['age'] for r in expected_memory],default=0))
            assert all(statistics[k]==v for k,v in computed.items())
            error('statistics_basic',statistics['current_triplet'],batch_hard,2e-6)
            error('statistics_hard',statistics['expanded_triplet'],hard,2e-6)
            norms=saved['historical_leaf_upstream_norms']
            assert len(norms)==historical and all(math.isfinite(n) and n>=0 for n in norms)
            groups=sorted({r['stored_step'] for r,n in zip(expected_memory,norms,strict=True) if n>0})
            assert saved['history_vjp_groups']==groups and saved['history_vjp_record_forwards']==len(groups)*64
            vjp_forward+=len(groups)*64
            fresh=64*len({r['stored_step'] for r in expected_memory})
            assert saved['fresh_role_record_forwards']==fresh;fresh_forward+=fresh
            for flag in ('selected_reencoding_bitwise','history_rng_buffers_preserved','history_vjp_leaves_current_grad_unchanged','final_gradient_addition_bitwise'):
                assert saved[flag]
            assert set(saved['roles'])==set(saved['applied_gradients'])==set(roles)
            for role in roles:
                pair=saved['roles'][role]['total_vs_history'];both=saved['roles'][role]['total_vs_both'];applied=saved['applied_gradients'][role]
                for record in (pair,both,applied):vector_norm_stats(record)
                error('history_addition_norm',both['difference_norm'],pair['second_norm'],1e-6*max(1,pair['second_norm']))
                for key in both:
                    if both[key] is None:assert applied[key] is None
                    else:error('applied_gradient_statistics',applied[key],both[key],1e-6*max(1,abs(both[key]),abs(applied[key])))
            direct=saved['direct_single_group_check']
            if direct:
                assert mode=='capacity' and len({r['stored_step'] for r in expected_memory})==1
                vector_norm_stats(direct)
                assert direct['all_four_reencoded_outputs_bitwise_equal']
                expected=direct['difference_norm']/max(direct['first_norm'],1e-12)
                error('direct_relative_l2_scalar',direct['relative_l2_error'],expected,1e-12)
                assert expected<=.005;direct_count+=1
                direct_checks.append(dict(run=directory.name,step=step+1,relative_l2_error=expected))
            assert row['amp_scale_after']>=row['amp_scale_before']>0
            signatures.append((indices,saved['pixel_sha256']))
            all_rows.append(dict(run=directory.name,fold=fold['fold'],endpoint=endpoint,mode=mode,step=step+1,
                                 historical_records=historical,eligible_anchors=eligible,ignored_same_identity_scene_positions=sum(ignored),
                                 retained_same_scene_negative_positions=sum(scene_negatives),
                                 standard_ap_max_error=standard_error,cross_ap_max_error=cross_error,total_ledger_error=total_error,
                                 independent_hard_loss=hard,independent_standard_loss=standard,independent_cross_scene_loss=cross))
            if active:
                for i in indices:queue.pop(i,None);queue[i]=step
                while len(queue)>512:queue.popitem(last=False)
        assert matrices.read()==b''
    assert fresh_forward==training['extra_fresh_role_record_forwards']
    assert vjp_forward==training['extra_history_vjp_record_forwards']
    assert direct_count==(1 if mode=='capacity' else 0)
    assert direct_count*64==training['extra_direct_check_record_forwards']
    assert training['history'][0]['optimizer_steps']==expected_steps
    error('epoch_mean_loss',training['history'][0]['mean_loss'],np.mean([r['loss'] for r in training['steps']]),1e-14)
    assert training['history'][0]['learning_rate']==base['OPTIMIZATION']['NEW_MODULE_LR']
    if mode=='capacity':
        witness=training['historical_parameter_gradient_witness'];assert witness['roles']==audits[witness['step']-1]['roles']
        assert all(r['total_vs_history']['second_norm']>0 for r in witness['roles'].values())
        assert history_count>0 and vjp_forward>0
    else:assert history_count==vjp_forward==0
    run_checks.append(dict(run=directory.name,steps=expected_steps,distance_elements=matrix_values,historical_records=history_count,
                           active_objective_steps=active_count,fresh_role_record_forwards=fresh_forward,vjp_record_forwards=vjp_forward,direct_checks=direct_count))
    pair_signatures[directory.name]=signatures

checkpoints=[];baseline_source_steps=0
for fold,fold_summary,baseline_fold,sequence in zip(protocol['folds'],summary['folds'],baseline['folds'],md['folds'],strict=True):
    fi=fold['fold'];assert fi==fold_summary['fold']==baseline_fold['fold']==sequence['fold']
    assert not set(fold['source_ids']) & set(fold['heldout_ids'])
    assert baseline_fold['source_ids']==fold['source_ids'] and baseline_fold['heldout_ids']==fold['heldout_ids']
    for row in baseline_fold['training']['steps']:
        assert set(row['sampled_record_indices'])<=set(fold['source_record_indices'])
        baseline_source_steps+=1
    assert baseline_fold['training']['epochs']==50 and baseline_fold['training']['optimizer_steps']==650
    bpath=Path(baseline_fold['checkpoint']);receipt(bpath,baseline_fold['checkpoint_sha256'])
    bpayload=torch.load(bpath,map_location='cpu',weights_only=True)
    assert bpayload['source_ids']==fold['source_ids'] and bpayload['heldout_ids']==fold['heldout_ids'] and bpayload['fold']==fi
    bstate=bpayload['model_state_dict'];bstate_hash=tensor_state_sha(bstate)
    assert bstate_hash==baseline_fold['training']['final_state_sha256']
    for endpoint in ('control','cross_scene'):
        row=fold_summary['endpoints'][endpoint];directory=run/f'fold_{fi}_{endpoint}'
        assert read(directory/'receipt.json')==row;receipt(directory/'receipt.json')
        assert all(row['engineering_checks'].values())
        binding=row['initialization'];training=row['training']
        assert binding['source_ids']==fold['source_ids'] and binding['heldout_ids']==fold['heldout_ids']
        assert not binding['role_weights_loaded'] and binding['role_initialization_seed']==42
        assert binding['signal_checkpoint_sha256']==baseline_fold['checkpoint_sha256']
        assert binding['signal_state_sha256']==training['signal_state_before_sha256']==training['signal_state_after_sha256']==bstate_hash
        assert training['initial_state_sha256']==binding['initial_state_sha256']
        assert len(binding['trainable_names'])==len(set(binding['trainable_names']))==203
        audit_training(directory,training,fold,sequence,endpoint,'capacity')
        cp=Path(row['checkpoint']);receipt(cp,row['checkpoint_sha256'])
        payload=torch.load(cp,map_location='cpu',weights_only=True)
        assert payload['binding']==binding and payload['config_sha256']==config_hash
        assert payload['source_ids']==fold['source_ids'] and payload['heldout_ids']==fold['heldout_ids'] and payload['fold']==fi
        assert all(k.startswith('baseline.') for k in payload['baseline_aliases'])
        assert all(not k.startswith('baseline.') for k in payload['role_state_dict'])
        assert set(binding['trainable_names'])<=set(payload['role_state_dict'])
        state={name:bstate[source_name] for name,source_name in payload['baseline_aliases'].items()}
        state.update(payload['role_state_dict'])
        final_hash=tensor_state_sha(state)
        assert final_hash==training['final_state_sha256']==row['strict_reload_state_sha256']
        assert row['strict_reload_all_outputs_bitwise_equal']
        assert all(row['preflight'][k] for k in ('standalone_signal_bitwise_equal','zero_update_replay_bitwise_equal','all_model_state_unchanged'))
        checkpoints.append(dict(fold=fi,endpoint=endpoint,path=str(cp),sha256=files[str(cp)]['sha256'],
                                 final_state_sha256=final_hash,state_tensors=len(state),baseline_alias_tensors=len(payload['baseline_aliases']),
                                 role_state_tensors=len(payload['role_state_dict']),all_finite=True,source_binding_exact=True))
        del payload,state
    assert pair_signatures[f'fold_{fi}_control']==pair_signatures[f'fold_{fi}_cross_scene']
    assert fold_summary['all_paired_source_pixels_exact']
    assert fold_summary['endpoints']['control']['initialization']==fold_summary['endpoints']['cross_scene']['initialization']
    del bpayload,bstate

overfit=[]
for endpoint in ('control','cross_scene'):
    row=summary['overfit'][endpoint];training=row['training'];fold=protocol['folds'][0]
    assert row['initialization']==summary['folds'][0]['endpoints'][endpoint]['initialization']
    audit_training(run/('overfit_'+endpoint),training,fold,md['folds'][0],endpoint,'overfit')
    assert all(row['checks'].values())
    classes=len(fold['source_ids']);alpha=base['LOSS']['LABEL_SMOOTHING']
    probabilities=[1-alpha+alpha/classes]+[alpha/classes]*(classes-1)
    entropy=-math.fsum(p*math.log(p) for p in probabilities)
    floor=entropy*sum(w for key,w in weights.items() if key.startswith('id_'))
    first=training['steps'][0]['loss'];last=training['steps'][-1]['loss']
    ratio=(last-floor)/(first-floor)
    gate=row['gate'];error('overfit_floor',gate['minimum_loss'],floor,1e-14)
    error('overfit_ratio',gate['loss_ratio'],ratio,1e-14)
    error('overfit_initial_excess',gate['initial_excess_loss'],first-floor,1e-14)
    error('overfit_final_excess',gate['final_excess_loss'],last-floor,1e-14)
    assert gate['initial_loss']==first and gate['final_loss']==last and gate['passed'] and ratio<=gate['maximum_loss_ratio']==.1
    overfit.append(dict(endpoint=endpoint,initial_loss=first,final_loss=last,analytic_floor=floor,corrected_loss_ratio=ratio))
assert pair_signatures['overfit_control']==pair_signatures['overfit_cross_scene']
assert len(all_rows)==sum(r['steps'] for r in run_checks)==248
assert sum(r['distance_elements'] for r in run_checks)==cpu['checked_memory_distance_elements']
assert sum(r['vjp_record_forwards'] for r in run_checks)==cpu['checked_history_vjp_record_forwards']
assert baseline_source_steps==1950
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_INDEPENDENT_COMPLETE_M0_CPU_RECONSTRUCTION',
    completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
    python=sys.executable,numpy_version=np.__version__,torch_deserialization_version=torch.__version__,
    summary_sha256=sha(run/'summary.json'),m0_cpu_sha256=sha(root/'m0_cpu.json'),
    checked_steps=248,checked_current_anchor_exposures=248*64,
    checked_distance_elements=sum(r['distance_elements'] for r in run_checks),
    checked_baseline_source_steps=baseline_source_steps,checked_checkpoints=len(checkpoints),
    max_errors=dict(max_errors),runs=run_checks,direct_runtime_witnesses=direct_checks,
    zero_eligible_m0_rows=zero_rows,overfit=overfit,checkpoints=checkpoints,stage_receipts=stage_receipts,
    independent_rows=all_rows,files=files,model_forwards=0,backward_calls=0,optimizer_updates=0,official_image_reads=0,
    limitations=['Only saved distances determine independent losses; per-step feature generation is not regenerated.',
                 'Fourteen recorded scalar terms are recombined; logits/residual distances are not stored for ten other per-term recomputations.',
                 'Gradient norms, direct/VJP checks, frozen-start states and strict-reload outputs are runtime witnesses; no parameter gradient tensors or initial role checkpoint are reconstructed.',
                 'Six capacity role checkpoints and three source Signal checkpoints are CPU-deserialized; no model is instantiated.',
                 'M0 engineering evidence makes no retrieval or Q1 performance claim.']),ensure_ascii=False))
