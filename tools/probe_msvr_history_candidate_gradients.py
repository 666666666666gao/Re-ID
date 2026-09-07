"""Source-only candidate-side VJPs at fixed, already qualified model states."""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from tools import train_msvr_fresh_coordinate as prior
from tools.msvr_freshness_probe import ViewFields
from tools.msvr_instance_memory import InstanceMemory, expanded_triplet
from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed, _training_batch, weighted_training_loss
from tools.train_msvr310_signal_oof import loader_for, records_for, sha256, write_json
from tools.train_msvr310_trifusion_oof import build_model, EXPERTS

ROOT = Path(__file__).resolve().parents[1]
STATES = ('initial', 'control', 'fresh_memory')
SCALE = 256.


def differentiable_history_loss(features, identities, history, metadata):
    """Same current anchors and real-ID mining; history is allowed to differentiate."""
    with torch.autocast(features.device.type, enabled=False):
        unit = F.normalize(features.float(), dim=1)
        ids = torch.as_tensor(identities, device=unit.device)
        mids = torch.as_tensor([r['identity'] for r in metadata], device=unit.device)
        dc, dh = torch.cdist(unit, unit), torch.cdist(unit, history.float())
        positive = (ids[:, None] == ids[None, :]) & ~torch.eye(len(ids), device=unit.device, dtype=torch.bool)
        negative = ids[:, None] != ids[None, :]
        hp = dc.masked_fill(~positive, -torch.inf).max(1).values
        hn = dc.masked_fill(~negative, torch.inf).min(1).values
        mp = ids[:, None] == mids[None, :]
        hp = torch.maximum(hp, dh.masked_fill(~mp, -torch.inf).max(1).values)
        hn = torch.minimum(hn, dh.masked_fill(mp, torch.inf).min(1).values)
        return F.relu(hp-hn+.3).mean()


def encode_graph(model, item):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.set_rng_state(item['cpu_rng'])
        torch.cuda.set_rng_state(item['cuda_rng'])
        with torch.autocast('cuda', dtype=torch.float16):
            representation = model.encoder(item['anchor'].cuda(), item['reference'].cuda())
            fused = model.fusion(item['baseline'].cuda(), representation)
            return F.normalize(fused.fused_embedding.float(), dim=1)


def gradients(loss, parameters):
    values = torch.autograd.grad(loss*SCALE, parameters, retain_graph=True, allow_unused=True)
    return [torch.zeros_like(p, dtype=torch.float32) if g is None else g.float()/SCALE
            for p, g in zip(parameters, values, strict=True)]


def compare(x, y):
    xx = sum(float(a.double().square().sum()) for a in x)
    yy = sum(float(a.double().square().sum()) for a in y)
    xy = sum(float((a.double()*b.double()).sum()) for a,b in zip(x,y,strict=True))
    difference = sum(float((a.double()-b.double()).square().sum()) for a,b in zip(x,y,strict=True))**.5
    assert all(np.isfinite(v) for v in (xx, yy, xy, difference))
    return dict(first_norm=xx**.5, second_norm=yy**.5, difference_norm=difference,
                cosine=xy/(xx*yy)**.5 if xx > 0 and yy > 0 else None)


def summarize_roles(names, current, repeated, historical, historical_repeat, total, weight):
    result = {}
    for expert in EXPERTS:
        index = [i for i,n in enumerate(names) if n.startswith('encoder.'+expert+'_')]
        u, v = [current[i] for i in index], [historical[i] for i in index]
        full = [a+b for a,b in zip(u,v,strict=True)]
        base_total = [total[i] for i in index]
        full_total = [a+weight*b for a,b in zip(base_total,v,strict=True)]
        result[expert] = dict(current_vs_history=compare(u,v), current_vs_both=compare(u,full),
                             total_vs_both=compare(base_total,full_total),
                             current_repeat_noise=compare(u,[repeated[i] for i in index]),
                             history_repeat_noise=compare(v,[historical_repeat[i] for i in index]))
    return result


def candidate_vjp(model, fields, metadata, history, upstream, parameters):
    accumulated = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    duplicate = [torch.zeros_like(p, dtype=torch.float32) for p in parameters]
    record_forwards = 0
    selected_groups = []
    for key in sorted({r['stored_step'] for r in metadata}):
        rows = [i for i,r in enumerate(metadata) if r['stored_step'] == key]
        if not bool(upstream[rows].abs().sum() > 0):
            continue
        item = fields.fields[key]
        unit = encode_graph(model, item)
        positions = [item['positions'][metadata[i]['record_index']] for i in rows]
        assert torch.equal(unit[positions].detach(), history[rows])
        coefficients = torch.zeros_like(unit)
        coefficients[positions] = upstream[rows]
        loss = (unit*coefficients.detach()).sum()
        first, second = gradients(loss, parameters), gradients(loss, parameters)
        for target, noise, a, b in zip(accumulated,duplicate,first,second,strict=True):
            target.add_(a);noise.add_(b)
        selected_groups.append(key)
        record_forwards += len(unit)
        del unit, coefficients, loss, first, second
    return accumulated, duplicate, record_forwards, selected_groups


def math_check():
    torch.manual_seed(42)
    current = torch.randn(8,16,requires_grad=True)
    ids = [0,0,1,1,2,2,3,3]
    scenes = [0,1]*4
    raw = torch.randn(4,16)
    transform = torch.randn(16,16,requires_grad=True)
    history = F.normalize(raw@transform,dim=1)
    metadata = [dict(identity=i, scene=1) for i in (0,1,4,5)]
    old = expanded_triplet(current,ids,scenes,history.detach(),metadata)[0]
    new = differentiable_history_loss(current,ids,history,metadata)
    assert torch.equal(old,new)
    first = torch.autograd.grad(old,current,retain_graph=True)[0]
    second = torch.autograd.grad(new,current,retain_graph=True)[0]
    assert torch.equal(first,second)
    direct = torch.autograd.grad(new,transform,retain_graph=True)[0]
    leaf = history.detach().requires_grad_(True)
    partial = torch.autograd.grad(differentiable_history_loss(current.detach(),ids,leaf,metadata),leaf)[0]
    reconstructed = torch.autograd.grad(history,transform,grad_outputs=partial)[0]
    assert torch.equal(direct,reconstructed) and bool(direct.abs().sum() > 0)
    return dict(status='PASS_SYNTHETIC_CHAIN_RULE', seed=42, legal_class_zero=True,
                selected_loss_and_current_gradient_equal=True, candidate_vjp_exact=True,
                optimizer_updates=0, model_forwards=0)


def context(path):
    spec = json.loads(path.read_bytes())
    assert spec['schema'] == 'msvr310-fixed-state-history-gradient-v1' and spec['seed'] == 42
    for name,digest in spec['project_file_sha256'].items():
        assert sha256(ROOT/name) == digest,name
    for name,digest in spec['fixed_file_sha256'].items():
        assert sha256(name) == digest,name
    saved = json.loads(Path(spec['q1_summary']).read_bytes())
    assert saved['status'] == 'Q1_FAIL'
    cpu = json.loads(Path(spec['q1_cpu']).read_bytes())
    assert cpu['summary_sha256'] == sha256(spec['q1_summary'])
    assert cpu['status'] == 'PASS_COMPLETE_FRESH_COORDINATE_Q1'
    return spec, prior.context(ROOT/spec['coordinate_config']), saved


def fit_probe(model, records, fold, protocol, config, md, saved, *, mode, directory):
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    _set_seed(42);model.train()
    initial = _module_state_sha256(model)
    buffers = {n:v.clone() for n,v in model.named_buffers()}
    selected = [(n,p) for n,p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    names, parameters = [n for n,_ in selected], [p for _,p in selected]
    criterion = ExpertFormationV8Criterion(triplet_margin=.3,label_smoothing=.1).cuda()
    weight = float(config['LOSS']['TRIPLET_FUSED'])
    source = [protocol['records'][i] for i in fold['source_record_indices']]
    memory, fields = InstanceMemory(source,capacity=512,maximum_age=8), ViewFields()
    lookup = {Path(r[0][0]).name:i for r,i in zip(records,fold['source_record_indices'],strict=True)}
    loader = loader_for(records,True)
    steps, seen, proof = [], set(), None
    warmup = 2 if mode == 'preflight' else 65
    reference = [json.loads(s) for s in (Path(saved['checkpoint']).parent/'memory_steps.jsonl').read_text().splitlines()]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (directory/'steps.jsonl').open('x') as log, (directory/'distances.f32').open('xb') as matrices:
        for epoch in range(1,(1 if mode == 'preflight' else 20)+1):
            batches = itertools.islice(loader,8) if mode == 'preflight' else loader
            for raw in batches:
                step=len(steps)
                indices=[lookup[n] for n in raw[-1]];seen.update(indices)
                assert indices==md['batches'][step]['record_indices']
                batch,labels=_training_batch(raw)
                pixels=prior.prior.pixels(batch)
                assert pixels==reference[step]['pixel_sha256']
                ids=[protocol['records'][i]['identity'] for i in indices]
                scenes=[protocol['records'][i]['scene'] for i in indices]
                history,metadata=memory.read(step,indices,torch.empty((0,7680),device='cuda'))
                for k in [k for k in fields.fields if step-k > 8]:
                    del fields.fields[k]
                with torch.autocast('cuda',dtype=torch.float16):
                    output,item=fields.capture(model,batch,indices)
                    components=criterion(output,labels)
                unit=F.normalize(output.fused_embedding.float(),dim=1)
                if step==0:
                    assert torch.equal(unit.detach(),fields.encode(model,item))
                pooled,basic,_,dc,dh,stats=expanded_triplet(output.fused_embedding,ids,scenes,history,metadata)
                assert torch.equal(basic,components['triplet_fused'])
                components['triplet_fused']=pooled
                total_loss=weighted_training_loss(components,config)
                role_stats={};forward_count=0;groups=[];nonzero_records=0
                if metadata:
                    leaf=history.detach().requires_grad_(True)
                    new_loss=differentiable_history_loss(output.fused_embedding,ids,leaf,metadata)
                    assert torch.equal(pooled,new_loss)
                    upstream=torch.autograd.grad(new_loss*SCALE,leaf,retain_graph=True)[0].float()/SCALE
                    assert torch.isfinite(upstream).all()
                    nonzero_records=int((upstream.abs().sum(1)>0).sum())
                    current,repeated,total=gradients(pooled,parameters),gradients(pooled,parameters),gradients(total_loss,parameters)
                    before_cpu,before_cuda=torch.get_rng_state(),torch.cuda.get_rng_state()
                    historical,history_repeat,forward_count,groups=candidate_vjp(model,fields,metadata,history,upstream,parameters)
                    if proof is None:
                        assert len({r['stored_step'] for r in metadata})==1
                        key=metadata[0]['stored_step'];encoded=encode_graph(model,fields.fields[key])
                        positions=[fields.fields[key]['positions'][r['record_index']] for r in metadata]
                        direct_loss=differentiable_history_loss(output.fused_embedding,ids,encoded[positions],metadata)
                        assert torch.equal(direct_loss,pooled)
                        direct=gradients(direct_loss,parameters)
                        combined=[a+b for a,b in zip(current,historical,strict=True)]
                        agreement=compare(direct,combined)
                        denominator=max(agreement['first_norm'],agreement['second_norm'])
                        relative=agreement['difference_norm']/denominator if denominator>0 else 0.
                        assert relative<=.005,agreement
                        proof=dict(step=step+1,agreement=agreement,relative_l2_error=relative,tolerance=.005,
                                   direct_graph_record_forwards=64,history_groups=1)
                        forward_count+=64
                        del encoded,direct_loss,direct,combined
                    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
                    role_stats=summarize_roles(names,current,repeated,historical,history_repeat,total,weight)
                    del leaf,new_loss,upstream,current,repeated,total,historical,history_repeat
                offset=matrices.tell()
                for values in (dc,dh):matrices.write(values.detach().cpu().contiguous().numpy().tobytes())
                row=dict(step=step+1,epoch=epoch,record_indices=indices,pixel_sha256=pixels,identities=ids,scenes=scenes,
                         memory=metadata,statistics=stats,roles=role_stats,history_nonzero_gradient_records=nonzero_records,
                         candidate_vjp_groups=groups,extra_role_record_forwards=forward_count,
                         distance_offset_bytes=offset,distance_float_count=dc.numel()+dh.numel(),
                         loss=float(total_loss.detach()),historical_coordinate_mode='fixed_current_parameters',
                         optimizer_updates=0,age_semantics='batch_recency_not_parameter_updates')
                log.write(json.dumps(row)+'\n');log.flush();steps.append(row)
                if step>=warmup:
                    memory.update(step,indices,unit);fields.fields[step]=item
                for name,value in model.named_buffers():value.copy_(buffers[name])
                assert all(p.grad is None for p in model.parameters())
                del output,components,unit,pooled,basic,dc,dh,total_loss,history,item
            print(json.dumps(dict(event='candidate_gradient_epoch',epoch=epoch,batches=len(steps),
                                  elapsed_seconds=time.perf_counter()-started)),flush=True)
    assert len(steps)==(8 if mode=='preflight' else 260)
    assert initial==_module_state_sha256(model)
    assert proof is not None
    if mode=='source':assert seen==set(fold['source_record_indices'])
    report=dict(status='PASS_FIXED_STATE_SOURCE_PROBE',batches=len(steps),optimizer_updates=0,
                initial_state_sha256=initial,final_state_sha256=_module_state_sha256(model),
                observed_source_records=sorted(seen),history_batches=sum(bool(r['memory']) for r in steps),
                candidate_gradient_chain_rule=proof,extra_role_record_forwards=sum(r['extra_role_record_forwards'] for r in steps)+64,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,elapsed_seconds=time.perf_counter()-started,
                files={n:dict(bytes=(directory/n).stat().st_size,sha256=sha256(directory/n)) for n in ('steps.jsonl','distances.f32')},
                parameter_gradients='runtime_witness_only',heldout_record_forwards=0,official_image_reads=0)
    write_json(directory/'receipt.json',report)
    return report


def run(args):
    torch.set_num_threads(4)
    spec,(coordinate,(config,base,cfg,environment,protocol,baseline,meta)),saved=context(args.config)
    args.output_dir.mkdir()
    result=dict(status='RUNNING',mode=args.mode,config_sha256=sha256(args.config),
                project_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                environment=environment,seed=42,optimizer_updates=0,heldout_record_forwards=0,official_image_reads=0,
                started_at=datetime.now().astimezone().isoformat(),folds=[])
    write_json(args.output_dir/'summary.json',result)
    for fold,b0,md,old_fold in zip(protocol['folds'],baseline['folds'],meta['folds'],saved['folds'],strict=True):
        entry=dict(fold=fold['fold'],states={});result['folds'].append(entry)
        for state in STATES:
            old=old_fold['endpoints']['control' if state=='initial' else state]
            records=records_for(base,protocol,fold,True)
            if state=='initial':model,binding=build_model(config,cfg,fold,b0)
            else:
                model=prior.reload_model(config,cfg,fold,b0,Path(old['checkpoint']),old['initialization'],
                                         old['training']['final_state_sha256'],sha256(ROOT/spec['coordinate_config']))
                binding=old['initialization']
            assert binding==old['initialization']
            directory=args.output_dir/f"fold_{fold['fold']}_{state}";directory.mkdir()
            print(json.dumps(dict(event='candidate_gradient_state',fold=fold['fold'],state=state)),flush=True)
            entry['states'][state]=fit_probe(model,records,fold,protocol,config,md,old,mode=args.mode,directory=directory)
            write_json(args.output_dir/'summary.json',result)
            del model;torch.cuda.empty_cache()
    result.update(status='PASS_COMPLETE_FIXED_STATE_PROBE',completed_at=datetime.now().astimezone().isoformat())
    write_json(args.output_dir/'summary.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=('math','preflight','source'),required=True)
    args=parser.parse_args()
    if args.mode=='math':write_json(args.output_dir,math_check())
    else:run(args)
