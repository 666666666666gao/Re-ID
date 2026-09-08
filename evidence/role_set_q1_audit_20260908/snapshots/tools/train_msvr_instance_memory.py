"""One fixed intervention: expand fused Triplet candidates with source instances."""
import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from tools import train_msvr310_source_style as prior
from tools.train_msvr310_trifusion_oof import (
    build_model, frozen_state_sha, engineering_checks, output_mapping, evaluate,
    OUTPUT_WIDTHS, EXPERTS,
)
from tools.train_msvr310_signal_oof import loader_for, records_for, sha256, write_json, new_model

ROOT = Path(__file__).resolve().parents[1]
ENDPOINTS = ('control', 'instance_memory')


def context(path):
    spec = json.loads(path.read_bytes())
    assert spec['memory'] == dict(capacity=512, maximum_age=8, warmup_steps=65,
                                 capacity_warmup_steps=2, overfit_warmup_steps=2)
    assert spec['seed'] == 42 and spec['training_epochs'] == 20
    for name, expected in spec['project_file_sha256'].items():
        assert sha256(ROOT/name) == expected, name
    for name, expected in spec['fixed_file_sha256'].items():
        assert sha256(name) == expected, name
    assert sha256(ROOT/spec['base_config']) == spec['base_config_sha256']
    source = json.loads(Path(spec['source_census_verification']).read_bytes())
    assert source['status'] == 'PASS_COMPLETE_SOURCE_RELATION_CENSUS'
    assert len(source['all972_conditions']) == 972
    return spec, prior.context(ROOT/spec['base_config'])


def extract(model, records):
    import torch
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    from tools.run_signal_preserving_v5 import _training_batch
    model.eval()
    parts = {name: [] for name in OUTPUT_WIDTHS}
    for raw in loader_for(records, False):
        batch, _ = _training_batch(raw)
        values = output_mapping(exact_signal_forward(model, batch))
        for name, value in values.items():
            parts[name].append(value.float().cpu())
    return {name: torch.cat(value) for name, value in parts.items()}


def preflight(model, records, cfg, fold, b0):
    import torch
    from tools.msvr310_exact_signal_inference import exact_signal_forward
    from tools.msvr_instance_memory import replay_drift
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed, _training_batch
    before = _module_state_sha256(model)
    model.eval()
    raw = next(iter(loader_for(records[:8], False)))
    batch, _ = _training_batch(raw)
    direct = new_model(cfg, fold)
    direct.load_state_dict(torch.load(b0['checkpoint'], map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    direct.eval()
    with torch.no_grad():
        target = direct(batch['images'], cam_label=batch['camera_ids'], view_label=raw[3].cuda(),
                        training=False, sge=cfg.MODEL.stageName)
        assert torch.equal(target, exact_signal_forward(model, batch).baseline_embedding)
    del direct, target
    buffers = {n: v.clone() for n, v in model.named_buffers()}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        _set_seed(42)
        raw = next(iter(loader_for(records, True)))
        batch, _ = _training_batch(raw)
        model.train()
        index_by_name = {Path(r[0][0]).name: i for r, i in zip(records, fold['source_record_indices'], strict=True)}
        probe = dict(batch=batch, indices=[index_by_name[n] for n in raw[-1]],
                     cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        with torch.autocast('cuda', dtype=torch.float16):
            output = model(batch, return_aux=True)
            probe['feature'] = torch.nn.functional.normalize(output.fused_embedding.float(), dim=1).detach()
        del output
        drift = replay_drift(model, probe)
        assert drift['maximum'] == 0, drift
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    model.eval()
    assert _module_state_sha256(model) == before
    return dict(standalone_signal_bitwise_equal=True, zero_update_replay_bitwise_equal=True,
                all_model_state_unchanged=True, source_role_record_forwards=136,
                standalone_signal_record_forwards=8)


def reload_model(config, cfg, fold, b0, path, binding, final_state, contract_sha):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256
    model, actual = build_model(config, cfg, fold, b0)
    assert actual == binding
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['binding'] == binding and payload['config_sha256'] == contract_sha
    assert payload['fold'] == fold['fold'] and payload['source_ids'] == fold['source_ids']
    assert payload['heldout_ids'] == fold['heldout_ids']
    state = model.state_dict()
    assert set(payload['role_state_dict']) == {n for n in state if not n.startswith('baseline.')}
    source = model.baseline.signal.state_dict()
    assert set(payload['baseline_aliases']) == {n for n in state if n.startswith('baseline.')}
    assert all(torch.equal(state[n], source[v]) for n, v in payload['baseline_aliases'].items())
    state.update(payload['role_state_dict'])
    model.load_state_dict(state, strict=True)
    assert _module_state_sha256(model) == final_state
    return model


def fit(model, records, fold, protocol, config, spec, md, *, endpoint, mode, directory):
    import torch
    from tools.msvr_instance_memory import InstanceMemory, expanded_triplet, replay_drift
    from tools.run_signal_preserving_v5 import (
        _module_state_sha256, _set_seed, _training_batch, learning_rate_multiplier, weighted_training_loss,
    )
    from trifusion.signal_preserving_v8 import ExpertFormationV8Criterion
    _set_seed(42)
    model.train()
    initial = _module_state_sha256(model)
    frozen = frozen_state_sha(model)
    signal = _module_state_sha256(model.baseline.signal)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config['OPTIMIZATION']['NEW_MODULE_LR'],
                                  weight_decay=config['OPTIMIZATION']['WEIGHT_DECAY'])
    scaler = torch.amp.GradScaler('cuda', init_scale=256.)
    criterion = ExpertFormationV8Criterion(triplet_margin=.3, label_smoothing=.1).cuda()
    source = [protocol['records'][i] for i in fold['source_record_indices']]
    memory = InstanceMemory(source, capacity=spec['memory']['capacity'], maximum_age=spec['memory']['maximum_age'])
    warmup = spec['memory']['warmup_steps' if mode == 'comparison' else mode+'_warmup_steps']
    index_by_name = {Path(r[0][0]).name: i for r, i in zip(records, fold['source_record_indices'], strict=True)}
    loader = loader_for(records, True)
    fixed = next(iter(loader)) if mode == 'overfit' else None
    steps, history, drifts, witness, live = [], [], [], {}, set()
    overflow = 0
    torch.cuda.reset_peak_memory_stats()
    with (directory/'memory_steps.jsonl').open('x', encoding='utf-8') as stream, \
         (directory/'memory_distances.f32').open('xb') as matrices:
        for epoch in range(1, (20 if mode == 'comparison' else 1)+1):
            started = time.perf_counter()
            lr = config['OPTIMIZATION']['NEW_MODULE_LR']
            if mode == 'comparison':
                lr *= learning_rate_multiplier(epoch, max_epochs=20, warmup_epochs=5)
            for group in optimizer.param_groups:
                group['lr'] = lr
            batches = (fixed for _ in range(100)) if mode == 'overfit' else itertools.islice(loader, 8) if mode == 'capacity' else loader
            epoch_rows = []
            for bi, raw in enumerate(batches):
                step = len(steps)
                indices = [index_by_name[n] for n in raw[-1]]
                assert indices == md['batches'][0 if mode == 'overfit' else step]['record_indices']
                assert sorted(torch.unique(raw[1], return_counts=True)[1].tolist()) == [8]*8
                batch, labels = _training_batch(raw)
                identities = [protocol['records'][i]['identity'] for i in indices]
                scenes = [protocol['records'][i]['scene'] for i in indices]
                pixel_sha = prior.pixels(batch)
                if bi == 0:
                    probe = dict(batch=batch, indices=indices, cpu_rng=torch.get_rng_state(),
                                 cuda_rng=torch.cuda.get_rng_state(), first_step=step)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    output = model(batch, return_aux=True)
                    output_mapping(output)
                    components = criterion(output, labels)
                current_unit = torch.nn.functional.normalize(output.fused_embedding.float(), dim=1)
                cached, metadata = memory.read(step, indices, current_unit.detach())
                pooled, basic, unit, dc, dm, stats = expanded_triplet(output.fused_embedding, identities, scenes, cached, metadata)
                assert torch.equal(basic, components['triplet_fused']), (float(basic), float(components['triplet_fused']))
                active = step >= warmup and endpoint == 'instance_memory'
                delta = pooled-components['triplet_fused']
                embedding_gradient = torch.autograd.grad(delta, output.fused_embedding, retain_graph=True)[0]
                role_gradient = float(embedding_gradient[:, 3072:].norm())
                if not witness and step >= warmup and stats['expanded_triplet']-stats['current_triplet'] > 1e-7:
                    selected = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
                    gradients = torch.autograd.grad(delta, [p for _, p in selected], retain_graph=True, allow_unused=True)
                    norms = {e: sum(float(g.float().square().sum()) for (n, _), g in zip(selected, gradients, strict=True)
                                    if g is not None and n.startswith('encoder.'+e+'_'))**.5 for e in EXPERTS}
                    assert all(x > 0 for x in norms.values())
                    witness = dict(step=step+1, actual_parameter_gradient_l2=norms,
                                   delta_loss=float(delta.detach()), role_embedding_gradient_l2=role_gradient)
                original_triplet = float(components['triplet_fused'].detach())
                if active:
                    components['triplet_fused'] = pooled
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = weighted_training_loss(components, config)
                assert torch.isfinite(loss)
                if bi == 0:
                    probe['feature'] = unit.detach().clone()
                scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                        if parameter.grad.abs().sum() > 0:
                            live.add(name)
                scaler.step(optimizer)
                scaler.update()
                overflow += int(scaler.get_scale() < scale)
                offset = matrices.tell()
                matrices.write(dc.detach().cpu().contiguous().numpy().tobytes())
                matrices.write(dm.detach().cpu().contiguous().numpy().tobytes())
                row = dict(step=step+1, epoch=epoch, loss=float(loss.detach()),
                           components={k: float(v.detach()) for k, v in components.items()},
                           sampled_record_indices=indices, amp_scale_before=scale, amp_scale_after=scaler.get_scale())
                audit = dict(step=step+1, zero_based_step=step, record_indices=indices, identities=identities,
                             scenes=scenes, pixel_sha256=pixel_sha, memory=metadata, replacement_active=active,
                             warmup_steps=warmup, original_triplet=original_triplet, statistics=stats,
                             role_embedding_delta_gradient_l2=role_gradient,
                             distance_offset_bytes=offset, distance_float_count=dc.numel()+dm.numel(),
                             current_features_finite=bool(torch.isfinite(unit).all()),
                             historical_features_detached=not cached.requires_grad)
                stream.write(json.dumps(audit)+'\n');stream.flush()
                if step >= warmup:
                    memory.update(step, indices, unit)
                steps.append(row);epoch_rows.append(row)
                if bi == 7:
                    drift = replay_drift(model, probe)
                    drift.update(epoch=epoch, stored_zero_based_step=probe['first_step'],
                                 measured_after_updates=len(steps), age_updates=len(steps)-probe['first_step'])
                    assert drift['age_updates'] == 8
                    drifts.append(drift)
                    del probe
            history.append(dict(epoch=epoch, optimizer_steps=len(epoch_rows), learning_rate=lr,
                                mean_loss=float(np.mean([r['loss'] for r in epoch_rows])),
                                elapsed_seconds=time.perf_counter()-started))
            print(json.dumps(dict(event='instance_memory_epoch', endpoint=endpoint, fold=fold['fold'], mode=mode, **history[-1])), flush=True)
    if mode != 'overfit':
        assert witness, 'No measured new parameter gradient from historical instances'
    report = dict(mode=mode, epochs=len(history), optimizer_steps=len(steps), initial_state_sha256=initial,
                  final_state_sha256=_module_state_sha256(model), frozen_state_before_sha256=frozen,
                  frozen_state_after_sha256=frozen_state_sha(model), signal_state_before_sha256=signal,
                  signal_state_after_sha256=_module_state_sha256(model.baseline.signal),
                  trainable_tensors=len(names), nonzero_gradient_tensors=len(live),
                  missing_nonzero_gradients=sorted(names-live), overflow_events=overflow,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2, history=history, steps=steps,
                  memory_parameter_gradient_witness=witness, fixed_pixel_drift=drifts,
                  extra_probe_record_forwards=64*len(drifts), cache_not_in_checkpoint=True,
                  audit_files={n: dict(bytes=(directory/n).stat().st_size, sha256=sha256(directory/n))
                               for n in ('memory_steps.jsonl', 'memory_distances.f32')})
    write_json(directory/'training.json', report)
    return report


def paired_summary(folds):
    # Reuse the exact sealed five-gate arithmetic, translating only endpoint names.
    translated = [dict(fold=f['fold'], endpoints=dict(control=f['endpoints']['control'],
                    source_style=f['endpoints']['instance_memory'])) for f in folds]
    result = prior.paired_summary(translated)
    result['endpoints']['instance_memory'] = result['endpoints'].pop('source_style')
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
        assert proof['status'] == 'PASS_COMPLETE_INSTANCE_MEMORY_M0' and proof['summary_sha256'] == sha256(args.m0_receipt)
    args.output_dir.mkdir()
    summary = dict(status='RUNNING', mode=args.mode, config_sha256=sha256(args.config),
                   project_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   runner_sha256=sha256(__file__), environment=environment, seed=42, folds=[], overfit={},
                   heldout_record_forwards=0, official_image_reads=0, training_cpu_threads=4,
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
        assert result['endpoints']['control']['initialization'] == result['endpoints']['instance_memory']['initialization']
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
