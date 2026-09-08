"""Paired role-set trainer governed by the registered fixed training contract.

Both endpoints retain current-coordinate historical candidate derivatives.
Only the fused negative-relation objective changes; M0 precedes heldout access.
"""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from tools import train_msvr310_source_style as prior
from tools import train_msvr_instance_memory as old
from tools.msvr_freshness_probe import ViewFields
from tools.train_msvr310_trifusion_oof import (
    build_model, frozen_state_sha, engineering_checks, output_mapping, evaluate,
    OUTPUT_WIDTHS, EXPERTS,
)
from tools.train_msvr310_signal_oof import loader_for, records_for, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ('control', 'role_set')
extract, preflight, reload_model = old.extract, old.preflight, old.reload_model


def context(path):
    from tools.train_msvr_history_gradient import context as previous_context
    spec=json.loads(path.read_bytes())
    assert spec['schema']=='msvr310-role-set-paired-v1' and spec['seed']==42
    for name,digest in spec['project_file_sha256'].items():assert sha256(ROOT/name)==digest,name
    for name,digest in spec['fixed_file_sha256'].items():assert sha256(name)==digest,name
    audit=json.loads((ROOT/spec['gradient_audit']).read_bytes())
    assert audit['verdict'] in ('PASS','WARN') and audit['deterministic_checks_status']=='pass'
    gradient=json.loads(Path(spec['gradient_summary']).read_bytes())
    cpu=json.loads(Path(spec['gradient_cpu']).read_bytes())
    assert gradient['status']=='COMPLETE_ROLE_SET_PARAMETER_GRADIENT_CHECK'
    assert cpu['status']=='PASS_COMPLETE_ROLE_SET_GRADIENT_PROBE_TEXT_AND_ARRAYS'
    assert cpu['summary_sha256']==sha256(spec['gradient_summary'])
    previous,base=previous_context(ROOT/spec['previous_config'])
    assert spec['memory']==previous['memory'] and spec['training_epochs']==20
    return spec,base


def encode_graph_all(model,item):
    """Replay one historical group with its original role-entry RNG."""
    import torch
    import torch.nn.functional as F
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.set_rng_state(item['cpu_rng']);torch.cuda.set_rng_state(item['cuda_rng'])
        with torch.autocast('cuda',dtype=torch.float16):
            representations=model.encoder(item['anchor'].cuda(),item['reference'].cuda())
            output=model.fusion(item['baseline'].cuda(),representations)
        return {k:F.normalize(v.float(),dim=1) for k,v in
                {'fused':output.fused_embedding,**output.branch_embeddings}.items()}


def fit(model, records, fold, protocol, config, spec, md, *, endpoint, mode, directory):
    import torch
    import torch.nn.functional as F
    from tools.msvr_role_set_relations import relation_objectives, fused_distances
    from tools.probe_msvr_role_set_gradients import refresh_all, encode_all
    from tools.msvr_instance_memory import InstanceMemory, expanded_triplet
    from tools.run_signal_preserving_v5 import (
        _module_state_sha256, _set_seed, _training_batch, learning_rate_multiplier, weighted_training_loss,
    )
    from tools.probe_msvr_history_candidate_gradients import encode_graph, compare
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    _set_seed(42);model.train()
    initial = _module_state_sha256(model)
    frozen = frozen_state_sha(model)
    signal = _module_state_sha256(model.baseline.signal)
    names = {n for n,p in model.named_parameters() if p.requires_grad}
    selected = [(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    parameters = [p for _,p in selected]
    assert not any(p.requires_grad for p in model.fusion.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config['OPTIMIZATION']['NEW_MODULE_LR'],
                                  weight_decay=config['OPTIMIZATION']['WEIGHT_DECAY'])
    scaler = torch.amp.GradScaler('cuda', init_scale=256.)
    criterion = ExpertFormationV8Criterion(triplet_margin=.3,label_smoothing=.1).cuda()
    source = [protocol['records'][i] for i in fold['source_record_indices']]
    memory = InstanceMemory(source,capacity=512,maximum_age=8)
    fields = ViewFields()
    warmup = spec['memory']['warmup_steps' if mode=='comparison' else mode+'_warmup_steps']
    index_by_name = {Path(r[0][0]).name:i for r,i in zip(records,fold['source_record_indices'],strict=True)}
    loader = loader_for(records,True)
    fixed = next(iter(loader)) if mode=='overfit' else None
    steps,history,live = [],[],set()
    fresh_forwards = vjp_forwards = direct_forwards = overflow = 0
    witness = {}
    weight = config['LOSS']['TRIPLET_FUSED']
    assert weight == 1.
    torch.cuda.reset_peak_memory_stats()
    with (directory/'memory_steps.jsonl').open('x',encoding='utf-8') as stream, \
         (directory/'memory_distances.f32').open('xb') as matrices:
        for epoch in range(1,(20 if mode=='comparison' else 1)+1):
            started=time.perf_counter()
            lr=config['OPTIMIZATION']['NEW_MODULE_LR']
            if mode=='comparison':lr*=learning_rate_multiplier(epoch,max_epochs=20,warmup_epochs=5)
            for group in optimizer.param_groups:group['lr']=lr
            batches=(fixed for _ in range(100)) if mode=='overfit' else itertools.islice(loader,8) if mode=='capacity' else loader
            epoch_rows=[]
            for raw in batches:
                step=len(steps)
                indices=[index_by_name[n] for n in raw[-1]]
                assert indices==md['batches'][0 if mode=='overfit' else step]['record_indices']
                assert sorted(torch.unique(raw[1],return_counts=True)[1].tolist())==[8]*8
                batch,labels=_training_batch(raw)
                ids=[protocol['records'][i]['identity'] for i in indices]
                scenes=[protocol['records'][i]['scene'] for i in indices]
                pixel_sha=prior.pixels(batch)
                optimizer.zero_grad(set_to_none=True)
                stored,metadata=memory.read(step,indices,torch.empty((0,7680),device='cuda'))
                fresh_values,count=refresh_all(model,fields,metadata,step)
                fresh=fresh_values['fused']
                fresh_forwards+=count
                with torch.autocast('cuda',dtype=torch.float16):
                    output,item=fields.capture(model,batch,indices)
                    output_mapping(output)
                    components=criterion(output,labels)
                pooled,basic,unit,dc,dm,stats=expanded_triplet(output.fused_embedding,ids,scenes,fresh,metadata)
                assert torch.equal(basic,components['triplet_fused'])
                role_distances=[]
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    for expert in EXPERTS:
                        value=F.normalize(output.branch_embeddings[expert].float(),dim=1)
                        role_distances.append(torch.cdist(value,torch.cat((value,fresh_values[expert]),dim=0)))
                mids=[r['identity'] for r in metadata]
                hard,role_set,selection=relation_objectives(dc,dm,ids,mids,role_distances)
                assert torch.equal(hard,pooled)
                selected_objective=hard if endpoint=='control' else role_set
                relation_saved=dict(hard_loss=float(hard.detach()),role_set_loss=float(role_set.detach()),
                    proposals=selection['proposals'].cpu().tolist(),negative_counts=selection['counts'].cpu().tolist(),
                    extra_active_counts=selection['extra_active'].sum(1).cpu().tolist())
                if step==0:
                    buffers={n:v.clone() for n,v in model.named_buffers()}
                    zero=encode_all(model,item);fresh_forwards+=64
                    assert torch.equal(unit,zero['fused'])
                    assert all(torch.equal(F.normalize(output.branch_embeddings[e].float(),dim=1),zero[e]) for e in EXPERTS)
                    assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers())
                    del zero,buffers
                active=step>=warmup
                upstream=None
                if metadata:
                    assert active
                    leaf=fresh.detach().requires_grad_(True)
                    # Current coordinates are fixed only for the candidate partial.
                    uc,uh=fused_distances(output.fused_embedding.detach(),leaf)
                    old_partial,new_partial,_=relation_objectives(uc,uh,ids,mids,role_distances)
                    candidate_loss=old_partial if endpoint=='control' else new_partial
                    assert torch.equal(candidate_loss.detach(),selected_objective.detach())
                    upstream=torch.autograd.grad(candidate_loss,leaf)[0].detach()
                    del candidate_loss,leaf,uc,uh,old_partial,new_partial
                original_triplet=float(basic.detach())
                if active:components['triplet_fused']=selected_objective
                with torch.autocast('cuda',dtype=torch.float16):loss=weighted_training_loss(components,config)
                assert torch.isfinite(loss)
                components_saved={k:float(v.detach()) for k,v in components.items()}
                loss_saved=float(loss.detach())
                direct=None
                scale=scaler.get_scale()
                if mode=='capacity' and metadata and direct_forwards==0:
                    keys={r['stored_step'] for r in metadata};assert len(keys)==1
                    old_item=fields.fields[next(iter(keys))]
                    direct_outputs=encode_graph_all(model,old_item)
                    encoded=direct_outputs['fused']
                    positions=[old_item['positions'][r['record_index']] for r in metadata]
                    assert torch.equal(encoded[positions].detach(),fresh)
                    assert all(torch.equal(direct_outputs[e][positions].detach(),fresh_values[e]) for e in EXPERTS)
                    ec,eh=fused_distances(output.fused_embedding,encoded[positions])
                    full_old,full_new,_=relation_objectives(ec,eh,ids,mids,role_distances)
                    full=full_old if endpoint=='control' else full_new
                    assert torch.equal(full.detach(),selected_objective.detach())
                    # Replace only this scalar's gradient, preserving all13 other terms.
                    direct_loss=loss+weight*(full-selected_objective)
                    values=torch.autograd.grad(direct_loss*scale,parameters,retain_graph=True,allow_unused=True)
                    direct=[torch.zeros_like(p,dtype=torch.float32) if g is None else g.float()/scale
                            for p,g in zip(parameters,values,strict=True)]
                    direct_forwards+=64
                    del encoded,full,direct_loss,values,ec,eh,full_old,full_new,direct_outputs
                offset=matrices.tell()
                for distance in (torch.cat((dc,dm),dim=1),*role_distances):
                    matrices.write(distance.detach().cpu().contiguous().numpy().tobytes())
                del distance
                # Release the current graph before history replay; update is still once.
                scaler.scale(loss).backward()
                current=[p.grad.detach().float().clone() for p in parameters]
                current_unit=unit.detach()
                del output,components,loss,pooled,basic,unit,dc,dm,hard,role_set,selected_objective,selection,role_distances
                history_gradient=[torch.zeros_like(p,dtype=torch.float32) for p in parameters]
                buffers={n:v.clone() for n,v in model.named_buffers()}
                cpu_rng,cuda_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
                groups=[]
                if metadata:
                    for key in sorted({r['stored_step'] for r in metadata}):
                        rows=[i for i,r in enumerate(metadata) if r['stored_step']==key]
                        if not bool(upstream[rows].abs().sum()>0):continue
                        history_item=fields.fields[key]
                        encoded=encode_graph(model,history_item)
                        positions=[history_item['positions'][metadata[i]['record_index']] for i in rows]
                        assert torch.equal(encoded[positions].detach(),fresh[rows])
                        coefficients=torch.zeros_like(encoded)
                        coefficients[positions]=upstream[rows]
                        surrogate=(encoded*coefficients).sum()*scale
                        values=torch.autograd.grad(surrogate,parameters,allow_unused=True)
                        for target,value in zip(history_gradient,values,strict=True):
                            if value is not None:target.add_(value.float())
                        groups.append(key);vjp_forwards+=64
                        del encoded,coefficients,surrogate,values
                assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state())
                assert all(torch.equal(v,buffers[n]) for n,v in model.named_buffers())
                assert all(torch.equal(p.grad,a) for p,a in zip(parameters,current,strict=True))
                role_stats={}
                for expert in EXPERTS:
                    indexes=[i for i,(n,_) in enumerate(selected) if n.startswith('encoder.'+expert+'_')]
                    base=[current[i]/scale for i in indexes]
                    extra=[history_gradient[i]/scale for i in indexes]
                    role_stats[expert]=dict(total_vs_history=compare(base,extra),
                                           total_vs_both=compare(base,[a+weight*b for a,b in zip(base,extra,strict=True)]))
                if metadata and not witness and all(r['total_vs_history']['second_norm']>0 for r in role_stats.values()):
                    witness=dict(step=step+1,roles=role_stats)
                direct_check={}
                if direct is not None:
                    decomposed=[(a+weight*b)/scale for a,b in zip(current,history_gradient,strict=True)]
                    direct_check=compare(direct,decomposed)
                    direct_check['relative_l2_error']=direct_check['difference_norm']/max(direct_check['first_norm'],1e-12)
                    direct_check['all_four_reencoded_outputs_bitwise_equal']=True
                    assert direct_check['relative_l2_error']<=.005
                    del direct,decomposed
                for p,g in zip(parameters,history_gradient,strict=True):p.grad.add_(g,alpha=weight)
                applied={}
                for expert in EXPERTS:
                    indexes=[i for i,(n,_) in enumerate(selected) if n.startswith('encoder.'+expert+'_')]
                    expected=[current[i]+weight*history_gradient[i] for i in indexes]
                    actual=[parameters[i].grad for i in indexes]
                    assert all(torch.equal(a,b) for a,b in zip(expected,actual,strict=True))
                    applied[expert]=compare([current[i]/scale for i in indexes],[g/scale for g in actual])
                scaler.unscale_(optimizer)
                for name,p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        assert torch.isfinite(p.grad).all(),name
                        if p.grad.abs().sum()>0:live.add(name)
                scaler.step(optimizer);scaler.update()
                overflow+=int(scaler.get_scale()<scale)
                row=dict(step=step+1,epoch=epoch,loss=loss_saved,components=components_saved,
                         sampled_record_indices=indices,amp_scale_before=scale,amp_scale_after=scaler.get_scale())
                audit=dict(step=step+1,zero_based_step=step,record_indices=indices,identities=ids,scenes=scenes,
                           pixel_sha256=pixel_sha,memory=metadata,replacement_active=active,warmup_steps=warmup,
                           original_triplet=original_triplet,statistics=stats,distance_offset_bytes=offset,
                           distance_float_count=4*64*(64+len(metadata)),
                           relation_objective=relation_saved,proposal_order=['fused',*EXPERTS],coordinate_rule='fresh',current_anchor_count=64,
                           history_candidate_vjp_applied=True,history_anchor_count=0,
                           fresh_role_record_forwards=count,history_vjp_groups=groups,history_vjp_record_forwards=64*len(groups),
                           historical_leaf_upstream_norms=[] if upstream is None else upstream.norm(dim=1).cpu().tolist(),
                           gradient_weight=weight,roles=role_stats,applied_gradients=applied,
                           direct_single_group_check=direct_check,
                           selected_reencoding_bitwise=True,history_rng_buffers_preserved=True,
                           history_vjp_leaves_current_grad_unchanged=True,final_gradient_addition_bitwise=True)
                stream.write(json.dumps(audit)+'\n');stream.flush()
                if active:memory.update(step,indices,current_unit);fields.fields[step]=item
                steps.append(row);epoch_rows.append(row)
                del stored,fresh,fresh_values,current_unit,current,history_gradient,buffers,upstream,item
            history.append(dict(epoch=epoch,optimizer_steps=len(epoch_rows),learning_rate=lr,
                                mean_loss=float(np.mean([r['loss'] for r in epoch_rows])),elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(dict(event='role_set_epoch',endpoint=endpoint,fold=fold['fold'],mode=mode,**history[-1])),flush=True)
    if mode!='overfit':assert witness
    report=dict(mode=mode,epochs=len(history),optimizer_steps=len(steps),initial_state_sha256=initial,
                final_state_sha256=_module_state_sha256(model),frozen_state_before_sha256=frozen,
                frozen_state_after_sha256=frozen_state_sha(model),signal_state_before_sha256=signal,
                signal_state_after_sha256=_module_state_sha256(model.baseline.signal),trainable_tensors=len(names),
                nonzero_gradient_tensors=len(live),missing_nonzero_gradients=sorted(names-live),overflow_events=overflow,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,history=history,steps=steps,
                historical_parameter_gradient_witness=witness,extra_fresh_role_record_forwards=fresh_forwards,
                extra_history_vjp_record_forwards=vjp_forwards,history_candidate_vjp_applied=True,
                extra_direct_check_record_forwards=direct_forwards,
                fresh_history_both_endpoints=True,history_anchor_count=0,cache_not_in_checkpoint=True,
                audit_files={n:dict(bytes=(directory/n).stat().st_size,sha256=sha256(directory/n))
                             for n in ('memory_steps.jsonl','memory_distances.f32')})
    write_json(directory/'training.json',report)
    return report


def paired_summary(folds):
    translated = [dict(fold=f['fold'], endpoints=dict(control=f['endpoints']['control'],
                  instance_memory=f['endpoints']['role_set'])) for f in folds]
    result = old.paired_summary(translated)
    result['endpoints']['role_set'] = result['endpoints'].pop('instance_memory')
    return result


def run(args):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256, evaluate_overfit_gate, overfit_loss_floor
    torch.set_num_threads(4)
    spec, (config, base, cfg, environment, protocol, baseline, meta) = context(args.config)
    m0 = args.mode == 'm0'
    old = None
    if not m0:
        old = json.loads(args.m0_receipt.read_bytes())
        proof = json.loads(args.m0_verification.read_bytes())
        assert old['status'] == 'PASS_ENGINEERING_ONLY' and old['config_sha256'] == sha256(args.config)
        assert proof['status'] == 'PASS_COMPLETE_ROLE_SET_M0' and proof['summary_sha256'] == sha256(args.m0_receipt)
    args.output_dir.mkdir()
    summary = dict(status='RUNNING', mode=args.mode, config_sha256=sha256(args.config),
                   project_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   runner_sha256=sha256(__file__), environment=environment, seed=42, folds=[], overfit={},
                   heldout_record_forwards=0, official_image_reads=0, training_cpu_threads=4,
                   update_rules=dict(control='fresh_history_both_sides_hardest', role_set='fresh_history_both_sides_role_set_mean'),
                   same_candidate_reencoding_and_vjp_algorithm=True, history_anchors=0,
                   role_proposal_indices_detached=True,other_thirteen_loss_terms_unchanged=True,
                   evaluation_cpu_threads=56, started_at=datetime.now().astimezone().isoformat())
    def save(): write_json(args.output_dir/'summary.json', summary)
    save()
    for fold, b0, md in zip(protocol['folds'], baseline['folds'], meta['folds'], strict=True):
        result = dict(fold=fold['fold'], endpoints={})
        summary['folds'].append(result)
        records = records_for(base, protocol, fold, True)
        paired_pixels = []
        for end in ENDPOINTS:
            directory = args.output_dir/f"fold_{fold['fold']}_{end}"
            directory.mkdir()
            model, binding = build_model(config, cfg, fold, b0)
            if not m0:
                assert binding == old['folds'][fold['fold']]['endpoints'][end]['initialization']
            pre = preflight(model, records, cfg, fold, b0) if m0 else None
            tr = fit(model, records, fold, protocol, config, spec, md, endpoint=end,
                     mode='capacity' if m0 else 'comparison', directory=directory)
            checks = engineering_checks(tr)
            checks['fixed_training_length'] = tr['optimizer_steps'] == (8 if m0 else 260)
            row = dict(fold=fold['fold'], initialization=binding, preflight=pre, training=tr, engineering_checks=checks)
            result['endpoints'][end] = row;save()
            assert all(checks.values()), checks
            path = directory/('roles_m0.pth' if m0 else 'roles_epoch20.pth')
            cksha = prior.checkpoint(model, binding, fold, sha256(args.config), path)
            before = extract(model, records[:8]) if m0 else None
            del model;torch.cuda.empty_cache()
            model = reload_model(config, cfg, fold, b0, path, binding, tr['final_state_sha256'], sha256(args.config))
            row.update(checkpoint=str(path), checkpoint_sha256=cksha, strict_reload_state_sha256=_module_state_sha256(model))
            if m0:
                after = extract(model, records[:8])
                assert all(torch.equal(before[k], after[k]) for k in OUTPUT_WIDTHS)
                row['strict_reload_all_outputs_bitwise_equal'] = True
            else:
                gallery = records_for(base, protocol, fold, False)
                features = extract(model, gallery)
                torch.set_num_threads(56)
                row['retrieval'] = evaluate(features, protocol, fold, directory, b0)
                torch.set_num_threads(4)
                summary['heldout_record_forwards'] += len(gallery)
            assert _module_state_sha256(model) == tr['final_state_sha256'] and sha256(path) == cksha
            audits = [json.loads(line) for line in (directory/'memory_steps.jsonl').read_text().splitlines()]
            paired_pixels.append([(r['record_indices'], r['pixel_sha256']) for r in audits])
            write_json(directory/'receipt.json', row);save()
            del model;torch.cuda.empty_cache()
        assert paired_pixels[0] == paired_pixels[1]
        assert result['endpoints']['control']['initialization'] == result['endpoints']['role_set']['initialization']
        result['all_paired_source_pixels_exact'] = True;save()
    if m0:
        fold = protocol['folds'][0]
        records = records_for(base, protocol, fold, True)
        for end in ENDPOINTS:
            directory = args.output_dir/('overfit_'+end);directory.mkdir()
            model, binding = build_model(config, cfg, fold, baseline['folds'][0])
            tr = fit(model, records, fold, protocol, config, spec, meta['folds'][0],
                     endpoint=end, mode='overfit', directory=directory)
            gate = evaluate_overfit_gate([r['loss'] for r in tr['steps']], max_ratio=.1,
                                         minimum_loss=overfit_loss_floor(config, num_classes=len(fold['source_ids'])))
            checks = engineering_checks(tr)
            checks.update(fixed_100_steps=tr['optimizer_steps']==100, original_overfit_gate=gate['passed'])
            summary['overfit'][end] = dict(initialization=binding, training=tr, gate=gate, checks=checks)
            save();assert all(checks.values()), checks
            del model;torch.cuda.empty_cache()
        summary.update(status='PASS_ENGINEERING_ONLY', optimizer_steps=248)
    else:
        result = paired_summary(summary['folds'])
        summary.update(status='Q1_PASS' if result['next_phase_qualified'] else 'Q1_FAIL',
                       optimizer_steps=1560, comparison=result,
                       m0_receipt_sha256=sha256(args.m0_receipt), m0_verification_sha256=sha256(args.m0_verification))
        assert summary['heldout_record_forwards'] == 2064
    summary['completed_at'] = datetime.now().astimezone().isoformat();save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('m0', 'q1'), required=True)
    parser.add_argument('--m0-receipt', type=Path)
    parser.add_argument('--m0-verification', type=Path)
    run(parser.parse_args())
