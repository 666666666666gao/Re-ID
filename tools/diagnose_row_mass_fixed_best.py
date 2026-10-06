"""Post-terminal, fixed-best deployment diagnosis; no training or weight writes."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import queue_row_mass_role_transport as panel
from tools import run_row_mass_role_transport as entry
from tools.train_msvr310_signal_oof import scene_scores
from tools.train_rgbnt100_signal_oof import camera_scores
from trifusion.row_mass_role_transport import row_log_assignment
from trifusion.selective_role_transport import message_weights


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_args(command):
    assert command[1] == '-B' and Path(command[2]) == ROOT/'tools/run_row_mass_role_transport.py'
    parser = argparse.ArgumentParser(add_help=False)
    for name in ('dataset', 'variant', 'mode'):
        parser.add_argument('--' + name, required=True)
    for name in ('protocol', 'signal-source', 'clip-weight', 'initialization', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args(command[3:])
    assert args.mode == 'evaluate' and args.seed == 42 and args.epochs == 50
    args.recipe = args.variant
    args.baseline_sha256 = sha(args.clip_weight)
    return args


class Capture(nn.Module):
    """Use the same feature-only call as the author's default inference wrapper."""
    def __init__(self, model, mode, reference=None):
        super().__init__()
        self.model, self.mode = model, mode
        # A previous capture holds snapshots, not another network submodule.
        object.__setattr__(self, 'reference', reference)
        self.features, self.statistics, self.scores = [], [], []
        self.peer_norms = []
        self.batch_index = 0

    def before_transport(self, module, inputs):
        semantic, private = inputs
        assert semantic.shape == private.shape and semantic.shape[1:] == (3, 16, 128)
        with torch.autocast(semantic.device.type, enabled=False):
            projected = module.matching_projection(F.layer_norm(semantic.float(), (128,)))
            undirected, statistics = [], []
            for m in range(3):
                for n in range(m + 1, 3):
                    scores = projected[:, m] @ projected[:, n].transpose(-1, -2) * 128 ** -0.5
                    undirected.append(scores.cpu())
                    for oriented in (scores, scores.transpose(-1, -2)):
                        real = row_log_assignment(oriented)[..., :16]
                        slot, r, q = message_weights(real, 'slot_mass')
                        uniform, r2, q2 = message_weights(real, 'uniform_mass')
                        assert torch.allclose(r, r2, atol=1e-5, rtol=1e-5)
                        assert torch.allclose(q, q2, atol=1e-5, rtol=1e-5)
                        assert torch.allclose(slot.sum((-1, -2)), uniform.sum((-1, -2)), atol=1e-5, rtol=1e-5)
                        distance = (slot - uniform).abs().sum((-1, -2))
                        deviation = (r - r.mean(-1, keepdim=True)).abs().sum(-1)
                        assert torch.allclose(distance, deviation, atol=1e-5, rtol=1e-5)
                        statistics.append(torch.stack((r.mean(-1), r.amin(-1), r.amax(-1),
                            r.std(-1, unbiased=False), deviation,
                            -(q * real.log_softmax(-1)).sum(-1).mean(-1),
                            (q.sum(-2)/16).amax(-1), slot.sum(-2).amax(-1),
                            uniform.sum(-2).amax(-1)), -1).cpu())
            scores = torch.stack(undirected, 1)
            if self.reference is not None:
                assert torch.allclose(scores, self.reference.scores[self.batch_index], atol=1e-5, rtol=1e-5)
            else:
                self.scores.append(scores)
            self.statistics.append(torch.stack(statistics, 1))

    def after_transport(self, _module, inputs, output):
        private = inputs[1].float()
        computed_write = output-private
        effective_write = torch.zeros_like(computed_write) if self.mode=='self_only' else computed_write
        self.peer_norms.append(torch.stack((private.norm(dim=-1).mean(-1),
            computed_write.norm(dim=-1).mean(-1), effective_write.norm(dim=-1).mean(-1),
            output.norm(dim=-1).mean(-1)), -1).cpu())
        if self.mode=='self_only':
            return private

    def forward(self, batch):
        output = self.model.evidence_model.forward_features(batch)
        g, c, h, f = (output[k].float() for k in ('shared_global', 'correction', 'raw_fused', 'fused'))
        assert all(torch.isfinite(value).all() for value in (g, c, h, f))
        assert torch.allclose(h, g + self.model.evidence_model.readout_gain * c, atol=1e-5, rtol=1e-5)
        norms = torch.stack((g.norm(dim=1), c.norm(dim=1), (h-g).norm(dim=1), h.norm(dim=1),
                            torch.rad2deg(torch.acos(F.cosine_similarity(g, f).clamp(-1, 1)))), -1).cpu()
        assert (norms[:, 0] > 0).all()
        if self.reference is None:
            self.features.append(dict(g=g.cpu(), c=c.cpu(), h=h.cpu(),
                global_l2=F.normalize(g, dim=1).cpu(), correction_l2=F.normalize(c, dim=1).cpu(), norms=norms))
        else:
            assert torch.allclose(g.cpu(), self.reference.features[self.batch_index]['g'], atol=1e-5, rtol=1e-5)
            self.features.append(dict(norms=norms))
        self.batch_index += 1
        return output['fused']


def derived_scores(features, metadata, dataset, protocol):
    count = protocol['counts']['query']
    distances = entry.inner.runner.distance_matrix(features[:count], features[count:]).numpy()
    scorer = scene_scores if dataset == 'MSVR310' else camera_scores
    environment = 'scenes' if dataset == 'MSVR310' else 'cameras'
    return scorer(distances, metadata['query_ids'], metadata['gallery_ids'],
                  metadata['query_' + environment], metadata['gallery_' + environment])


def run_endpoint(row, job, output):
    args = original_args(next(s['command'] for s in job['steps'] if s['mode'] == 'evaluate'))
    protocol = entry.inner.runner.read_protocol(args.protocol, args.dataset)
    training = json.loads((args.output_dir/'training.json').read_text())
    assert [r['epoch'] for r in training['history']] == list(range(1, 51))
    assert sha(args.output_dir/'best_map.pth') == row['checkpoint_sha256']
    model, _cfg, binding = entry.inner.foundation.build(args, protocol)
    assert binding == row['initializer'] == training['initializer']
    transport = model.evidence_model.roles.transport
    initialized_projection = transport.matching_projection.weight.detach().cpu().clone()
    payload = entry.inner.foundation.load(args.output_dir/'best_map.pth', model, args)
    assert payload['epoch'] == row['best_epoch']
    selected_history = next(r for r in training['history'] if r['epoch']==row['best_epoch'])
    assert payload['metrics']==selected_history['official_fused']
    assert all(abs(payload['metrics'][k]-row['metrics'][k])<1e-5 for k in row['metrics'])
    model.eval()
    assert not model.training and not model.signal.training
    before = entry.inner.runner._module_state_sha256(model)
    original_mode = transport.mass_mode
    gain = float(model.evidence_model.readout_gain.detach())
    weight_stats = dict(initial_matching_norm=float(initialized_projection.norm()),
        best_matching_norm=float(transport.matching_projection.weight.detach().norm()),
        matching_delta_norm=float((transport.matching_projection.weight.detach().cpu()-initialized_projection).norm()),
        best_message_output_norm=float(transport.message_output.weight.detach().norm()), readout_gain=gain)
    output.mkdir()
    modes, reference = [], None
    for mode in ('original', 'opposite', 'self_only'):
        folder = output/mode
        folder.mkdir()
        transport.mass_mode = ('uniform_mass' if original_mode == 'slot_mass' else 'slot_mass') if mode == 'opposite' else original_mode
        capture = Capture(model, mode, reference)
        pre = transport.register_forward_pre_hook(capture.before_transport)
        post = transport.register_forward_hook(capture.after_transport)
        os.chdir(folder)
        for device in (0, 1):
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        metrics = entry.inner.runner.official_metrics(capture, protocol, args.signal_source,
                                                      save_distances=folder/'distances.pt')
        for device in (0, 1):
            torch.cuda.synchronize(device)
        seconds = time.perf_counter()-start
        pre.remove()
        post.remove()
        assert capture.batch_index == len(capture.statistics)
        assert capture.batch_index == len(capture.peer_norms)
        if reference is not None:
            assert capture.batch_index == reference.batch_index
        assert entry.inner.runner._module_state_sha256(model) == before
        metadata = torch.load(folder/'distances.pt', map_location='cpu', weights_only=False)
        count = protocol['counts']['query'] + protocol['counts']['gallery']
        statistics = torch.cat(capture.statistics)
        norms = torch.cat([r['norms'] for r in capture.features])
        peer_norms = torch.cat(capture.peer_norms)
        assert statistics.shape == (count, 6, 9) and norms.shape == (count, 5)
        assert peer_norms.shape == (count, 3, 4) and torch.isfinite(peer_norms).all()
        arrays = dict(statistics=statistics, norms=norms,
            statistics_columns=['real_mass_mean','real_mass_min','real_mass_max','real_mass_std',
                'slot_uniform_weight_l1','conditional_entropy','conditional_max_column_share',
                'slot_max_column_absolute_mass','uniform_max_column_absolute_mass'],
            directed_modal_pairs=[[0,1],[1,0],[0,2],[2,0],[1,2],[2,1]],
            peer_norms=peer_norms,
            peer_norm_columns=['private_slot_norm_mean','computed_peer_write_slot_norm_mean',
                'effective_peer_write_slot_norm_mean','computed_transport_output_slot_norm_mean'],
            norm_columns=['global','raw_correction','scaled_correction','raw_fused','global_fused_angle_degrees'])
        if mode=='self_only':
            assert torch.count_nonzero(peer_norms[:,:,2])==0
        if mode == 'original':
            assert all(abs(metrics[k]-row['metrics'][k]) < 1e-5 for k in metrics)
            reference = capture
            arrays.update({key:torch.cat([r[key] for r in capture.features]) for key in ('g','c','h')})
            own_global = torch.cat([r['global_l2'] for r in capture.features])
            own_correction = torch.cat([r['correction_l2'] for r in capture.features])
            write(folder/'own_global.json', derived_scores(own_global, metadata, args.dataset, protocol))
            write(folder/'own_correction.json', derived_scores(own_correction, metadata, args.dataset, protocol))
        else:
            original_meta = torch.load(output/'original/distances.pt', map_location='cpu', weights_only=False)
            assert all(np.array_equal(metadata[k], original_meta[k]) for k in metadata if k != 'fused')
        torch.save(arrays, folder/'diagnostic_arrays.pt')
        result = dict(mode=mode, transport_mass_mode=transport.mass_mode, metrics=metrics,
            elapsed_seconds_including_diagnostics_and_scoring=seconds, state_sha256=before,
            distance_sha256=sha(folder/'distances.pt'), arrays_sha256=sha(folder/'diagnostic_arrays.pt'),
            effective_peer_write='zero_by_exit_hook' if mode == 'self_only' else 'computed_message_output',
            boundary='Fixed parameters, full official query/gallery. self_only is an output intervention, not a retrained role deletion. Raw last_diagnostics are not effective self_only writes.')
        write(folder/'RESULT.json', result)
        modes.append(result)
    transport.mass_mode = original_mode
    assert entry.inner.runner._module_state_sha256(model) == before
    assert sha(args.output_dir/'best_map.pth') == row['checkpoint_sha256']
    result = dict(dataset=args.dataset, variant=args.variant, best_epoch=row['best_epoch'],
        weight_statistics=weight_stats, original_mass_mode=original_mode, restored_mass_mode=transport.mass_mode,
        state_sha256=before, modes=modes)
    write(output/'ENDPOINT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.seal, args.output = args.seal.resolve(), args.output.resolve()
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1'
    seal = json.loads(args.seal.read_text())
    assert seal['schema'] == 'fixed-best-row-transport-diagnosis-v1'
    assert all(sha(ROOT/name) == digest for name,digest in seal['diagnostic_source_sha256'].items())
    launch = Path(seal['original_launch'])
    assert json.loads((launch/'EXIT.json').read_text())['exit_code'] == 0
    proc = Path('/proc')/str(seal['original_supervisor_pid'])/'stat'
    assert not (proc.exists() and int(proc.read_text().split(') ')[1].split()[19]) == seal['original_supervisor_start_ticks'])
    campaign = Path(seal['original_campaign'])
    state = json.loads((campaign/'campaign.json').read_text())
    assert state['status'] == 'COMPLETE' and state['report_invocations'] == 1 and state['report_exit_code'] == 0
    assert len(state['jobs']) == 12 and all(j['status'] == 'COMPLETE' and j['exit_code'] == 0 for j in state['jobs'])
    report = json.loads((Path(seal['original_report'])/'SUMMARY.json').read_text())
    assert sha(Path(seal['original_report'])/'SUMMARY.json') == seal['original_report_sha256']
    assert report['status'] == 'COMPLETE' and report['accepted'] == 6 and len(report['pairs']) == 15
    panel.configure()
    assert panel.source_map() == seal['original_source_sha256']
    panel.previous.require_controls()
    matrix = json.loads((campaign/'accepted_matrix.json').read_text())
    assert matrix['accepted'] == matrix['expected'] == len(matrix['rows']) == 6
    assert not args.output.exists()
    args.output.mkdir()
    write(args.output/'INPUT_SEAL.json', seal)
    entry.configure()
    results = []
    for row in matrix['rows']:
        assert panel.previous.accepted_row(campaign, row['dataset'], row['variant']) == row
        job = next(j for j in state['jobs'] if (j['phase'],j['dataset'],j['variant']) == ('full',row['dataset'],row['variant']))
        results.append(run_endpoint(row, job, args.output/(row['dataset']+'_'+row['variant'])))
        write(args.output/'PROGRESS.json', dict(status='RUNNING', completed=len(results), expected=6, rows=results))
        torch.cuda.empty_cache()
    assert panel.source_map() == seal['original_source_sha256']
    assert all(sha(ROOT/name) == digest for name,digest in seal['diagnostic_source_sha256'].items())
    result = dict(schema=seal['schema'], status='COMPLETE', completed_at=datetime.now().astimezone().isoformat(),
        accepted_models=6, deployment_modes=18, rows=results,
        boundary='Single-seed fixed-best diagnosis on consumed benchmarks. No optimization, epoch reselection, correspondence truth, trained role-deletion ablation, training-seed stability or SOTA acceptance.')
    write(args.output/'SUMMARY.json', result)
    print(json.dumps(dict(status='COMPLETE', accepted_models=6, deployment_modes=18)), flush=True)


if __name__ == '__main__':
    main()
