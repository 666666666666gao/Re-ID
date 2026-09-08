"""Read-only remote CPU source-distance analysis; no model or retrieval calls."""
from pathlib import Path
from collections import defaultdict
import hashlib, importlib.util, json, time
import numpy as np
import torch

MODE = 'q1'
ROOT = Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
REPO = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
torch.set_num_threads(2)
torch.set_num_interop_threads(2)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def score_loss(score, ids):
    """Expose scores as independent leaves, retaining positive-positive derivatives."""
    aps = []
    columns = torch.arange(len(ids))
    for i in range(score.shape[0]):
        positive = ids == ids[i]
        positive[i] = False
        pos = columns[positive]
        assert len(pos) > 0
        ranks = torch.sigmoid((score[i][None, :] - score[i, pos, None]) / .01)
        valid = (columns[None, :] != i) & (columns[None, :] != pos[:, None])
        ranks = ranks.masked_fill(~valid, 0)
        numerator = 1 + (ranks * positive[None, :]).sum(1)
        denominator = 1 + ranks.sum(1)
        aps.append((numerator / denominator).mean())
    return 1 - torch.stack(aps).mean()


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    assert np.isfinite(a).all()
    out = dict(count=len(a))
    if len(a):
        out.update(negative=int((a < 0).sum()), zero=int((a == 0).sum()),
                   positive=int((a > 0).sum()), signed_mean=float(a.mean()),
                   absolute_sum=float(np.abs(a).sum()),
                   abs_quantiles=dict(zip(['min','p25','p50','p75','p90','p99','max'],
                       np.quantile(np.abs(a), [0,.25,.5,.75,.9,.99,1]).tolist())))
    return out


def main():
    started = time.perf_counter()
    assert MODE in ('m0', 'q1')
    summary_path = ROOT / MODE / 'summary.json'
    cpu_path = ROOT / (MODE + '_cpu.json')
    summary = json.loads(summary_path.read_bytes())
    cpu = json.loads(cpu_path.read_bytes())
    assert summary['config_sha256'] == '974328fee25985b19aa36c84f57a557f9120993fecb98b2a902a1b8de8475302'
    assert cpu['summary_sha256'] == sha(summary_path)
    assert cpu['status'] == ('PASS_COMPLETE_SMOOTH_AP_Q1' if MODE == 'q1' else 'PASS_COMPLETE_SMOOTH_AP_M0')
    assert cpu['checked_training_steps'] == (1560 if MODE == 'q1' else 248)
    if MODE == 'q1':
        pipe = json.loads((ROOT / 'pipeline.json').read_bytes())
        assert pipe['status'] in ('COMPLETE_VERIFIED_Q1_PASS','COMPLETE_VERIFIED_Q1_FAIL')
        assert pipe['code_commit'] == '2e947a4325144e37fed638105ac954e7e54b5fe5'
        assert len(pipe['stages']) == 5 and all(s['exit_code'] == 0 for s in pipe['stages'])
        assert not any(Path('/proc', str(pid)).exists() for pid in [pipe['wrapper_pid'], *[s['original_pid'] for s in pipe['stages']]])
        assert pipe['terminal_summary_sha256'] == sha(summary_path)
        assert pipe['terminal_cpu_sha256'] == sha(cpu_path)
    obj_path = REPO / 'tools/msvr_smooth_ap.py'
    assert sha(obj_path) == '9d7c01830493233183b2cc9366ec8742caab1f782208870badc8dc93a94b3b68'
    spec = importlib.util.spec_from_file_location('fixed_objective', obj_path)
    objective = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(objective)
    endpoints = []
    total_steps = 0
    names = [f'fold_{fold}_{end}' for fold in range(3) for end in ('control','smooth_ap')]
    if MODE == 'm0':
        names += ['overfit_control','overfit_smooth_ap']
    for name in names:
        folder = ROOT / MODE / name
        training = json.loads((folder / 'training.json').read_bytes())
        rows = [json.loads(x) for x in (folder / 'memory_steps.jsonl').read_text().splitlines()]
        assert len(rows) == len(training['steps']) == training['optimizer_steps']
        for name in ('memory_steps.jsonl','memory_distances.f32'):
            proof = training['audit_files'][name]
            assert sha(folder / name) == proof['sha256'] and (folder / name).stat().st_size == proof['bytes']
        buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        max_loss_error = max_chain_error = 0.
        with (folder / 'memory_distances.f32').open('rb') as stream:
            for row, tr in zip(rows, training['steps'], strict=True):
                assert row['step'] == tr['step']
                assert row['saved_space_order'] == ['fused','cnn','transformer','mamba']
                assert stream.tell() == row['distance_offset_bytes']
                n = 64 + len(row['memory'])
                arrays = np.fromfile(stream, '<f4', row['distance_float_count']).reshape(4,64,n)
                d = torch.tensor(arrays[0], requires_grad=True)
                ids_list = row['identities'] + [m['identity'] for m in row['memory']]
                scenes = row['scenes'] + [m['scene'] for m in row['memory']]
                ids = torch.tensor(ids_list)
                hard, smooth, _ = objective.paired_objectives(d[:,:64], d[:,64:], ids_list[:64], ids_list[64:])
                hard_gd = torch.autograd.grad(hard,d,retain_graph=True)[0].numpy()
                smooth_gd = torch.autograd.grad(smooth,d)[0].numpy()
                score = (1 - d.detach().square()/2).requires_grad_()
                reference = score_loss(score,ids)
                gs = torch.autograd.grad(reference,score)[0].numpy()
                loss_error = abs(float(reference.detach()) - float(smooth.detach()))
                chain_error = float(np.max(np.abs(smooth_gd + arrays[0] * gs)))
                assert loss_error < 2e-6 and chain_error < 2e-6
                max_loss_error = max(max_loss_error,loss_error)
                max_chain_error = max(max_chain_error,chain_error)
                active_ap = tr['active_fused_metric'] == 'smooth_ap'
                phases = ['all', 'post_warmup' if row['replacement_active'] else 'warmup']
                if MODE == 'q1' and row['step'] >= 196:
                    phases.append('last65')
                for i in range(64):
                    positive = np.asarray(ids_list) == ids_list[i]
                    positive[i] = False
                    negative = np.asarray(ids_list) != ids_list[i]
                    cross = np.asarray(scenes) != scenes[i]
                    inverted = positive & (arrays[0,i] > arrays[0,i,negative].min())
                    nonmax = positive & (arrays[0,i] < arrays[0,i,positive].max())
                    categories = dict(all_positive=positive, cross_positive=positive & cross,
                        inverted_positive=inverted, inverted_nonmax=inverted & nonmax,
                        inverted_nonmax_cross=inverted & nonmax & cross)
                    for phase in phases:
                        for category, mask in categories.items():
                            b = buckets[phase][category]
                            for key, values in [('smooth_score_derivative',gs[i]),
                                                ('smooth_distance_derivative',smooth_gd[i]),
                                                ('hard_distance_derivative',hard_gd[i]),
                                                ('active_distance_derivative',smooth_gd[i] if active_ap else hard_gd[i])]:
                                b[key].extend(values[mask].tolist())
                total_steps += 1
            assert stream.read(1) == b''
        endpoints.append(dict(endpoint=folder.name,steps=len(rows),
            distance_file_sha256=training['audit_files']['memory_distances.f32']['sha256'],
            max_score_loss_error=max_loss_error,max_distance_chain_error=max_chain_error,
            phases={phase:{cat:{key:stats(v) for key,v in b.items()} for cat,b in cats.items()} for phase,cats in buckets.items()}))
    assert total_steps == (1560 if MODE == 'q1' else 248)
    result = dict(status='COMPLETE_SOURCE_DISTANCE_DERIVATIVE_DESCRIPTION' if MODE=='q1' else 'PASS_REAL_M0_ANALYSIS_FUNCTION_CHECK_ONLY',
        mode=MODE,steps=total_steps,endpoints=endpoints,summary_sha256=sha(summary_path),
        objective_sha256=sha(obj_path),elapsed_seconds=time.perf_counter()-started,
        scope='Float32 independent score-leaf derivative matched to deployed distance objective on every saved step. Repeated source training position exposures, not unique images or parameter-gradient coverage. Strict negative outranking and strict nonmaximum-positive distance; max-distance ties excluded from nonmax. Positive score derivative below zero or distance derivative above zero favors the positive for this term only. Counterfactual hard/AP derivatives use each endpoint own saved state, not a matched-state causal comparison. No heldout scores, no model/image/GPU calls, no optimizer updates, no remote files written. No complete-source-gallery ranking or fused-versus-other-loss parameter gradients inferred.')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
