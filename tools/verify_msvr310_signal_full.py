"""Verify the fixed full155 MSVR310 Signal training files without image access."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_rgbnt100_signal_terminal_files import sha, state_sha
from tools.verify_rgbnt100_signal_terminal_scalars import f32


def run(args):
    import torch

    assert not args.output.exists()
    summary = json.loads(args.summary.read_bytes())
    config = json.loads(args.config.read_bytes())
    protocol_path = Path(config['protocol'])
    protocol = json.loads(protocol_path.read_bytes())
    baseline_path = ROOT / config['baseline_config']
    baseline = json.loads(baseline_path.read_bytes())
    m0 = summary['mode'] == 'm0'
    assert summary['status'] == ('PASS_FULL155_SIGNAL_ENGINEERING' if m0 else
                                 'COMPLETE_FULL155_SIGNAL_FIXED_EPOCH50')
    assert summary['schema'] == config['schema'] == 'msvr310-signal-full155-fixed-v1'
    assert summary['seed'] == 42 and summary['fold'] == 'full_train'
    assert summary['source_records'] == 1032 and summary['source_identities'] == 155
    assert summary['source_ids'] == protocol['source_ids']
    assert set(summary['source_ids']).isdisjoint(protocol['official_test_ids'])
    assert summary['official_model_record_forwards'] == 0
    assert not summary['official_metrics_read'] and not summary['m0_weights_reused']
    assert summary['config_sha256'] == sha(args.config)
    assert summary['protocol_sha256'] == config['protocol_sha256'] == sha(protocol_path)
    assert summary['baseline_config_sha256'] == config['baseline_config_sha256'] == sha(baseline_path)
    assert summary['runner_sha256'] == sha(ROOT / 'tools/train_msvr310_signal_full.py')
    for name, expected in config['source_files_sha256'].items():
        assert sha(ROOT / name) == expected, name
    assert summary['source_files_sha256'] == config['source_files_sha256']
    assert baseline['dataset_root'] == protocol['dataset_root']
    assert all(summary['checks'].values()) and summary['strict_reload_source_features_bitwise_equal']

    training_path = args.summary.parent / 'training.json'
    training = json.loads(training_path.read_bytes())
    assert summary['training'] == training
    epochs = 1 if m0 else 50
    assert training['epochs'] == len(training['history']) == epochs
    assert training['optimizer_steps'] == len(training['steps'])
    assert training['trainable_tensors'] == training['gradient_tensors']
    assert not training['trainable_without_gradient'] and training['overflow_events'] == 0
    assert training['frozen_token_selection_parameters'] == 787968
    seen = set()
    epoch_losses = {epoch: [] for epoch in range(1, epochs + 1)}
    for number, step in enumerate(training['steps'], 1):
        assert step['step'] == number and step['epoch'] in epoch_losses
        assert step['amp_scale_after'] >= step['amp_scale_before']
        indices = step['sampled_record_indices']
        assert len(indices) == 64 and all(0 <= index < 1032 for index in indices)
        counts = Counter(protocol['records'][index]['identity'] for index in indices)
        assert sorted(counts.values()) == [8] * 8
        components = step['id_triplet_head_losses']
        assert len(components) == 4
        loss = 0.0
        for value in components:
            loss = f32(loss + value)
        loss = f32(loss + f32(f32(.2) * step['gram_loss']))
        loss = f32(loss + f32(f32(.01) * step['patch_loss']))
        assert math.isfinite(step['loss']) and abs(loss - step['loss']) < 1e-5
        seen.update(indices)
        epoch_losses[step['epoch']].append(step['loss'])
    for number, epoch in enumerate(training['history'], 1):
        assert epoch['epoch'] == number
        assert epoch['optimizer_steps'] == len(epoch_losses[number]) > 0
        assert abs(epoch['mean_loss'] - statistics.mean(epoch_losses[number])) < 1e-10
    assert summary['unique_source_records_seen'] == len(seen)
    assert summary['unique_source_identities_seen'] == len({protocol['records'][i]['identity'] for i in seen})
    if m0:
        assert training['optimizer_steps'] == 8
    else:
        assert training['optimizer_steps'] == 1000
        assert all(len(epoch_losses[epoch]) == 20 for epoch in epoch_losses)
        assert seen == set(protocol['source_record_indices'])
        m0_summary = json.loads(args.m0_receipt.read_bytes())
        assert summary['m0_receipt_sha256'] == sha(args.m0_receipt)
        assert summary['m0_verification_sha256'] == sha(args.m0_verification)
        assert m0_summary['status'] == 'PASS_FULL155_SIGNAL_ENGINEERING'
        assert m0_summary['training']['initial_state_sha256'] == training['initial_state_sha256']

    checkpoint = Path(summary['checkpoint'])
    assert sha(checkpoint) == summary['checkpoint_sha256']
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert payload['source_ids'] == summary['source_ids'] and payload['fold'] == 'full_train'
    assert payload['config_sha256'] == summary['config_sha256']
    assert payload['protocol_sha256'] == summary['protocol_sha256']
    state = payload['model_state_dict']
    assert state_sha(state) == training['final_state_sha256'] == summary['strict_reload_state_sha256']
    assert training['initial_state_sha256'] != training['final_state_sha256']
    assert all(torch.isfinite(value).all().item() for value in state.values() if value.is_floating_point())
    selector = {name.removeprefix('SIM.token_selection.'): value for name, value in state.items()
                if name.startswith('SIM.token_selection.')}
    assert state_sha(selector) == training['frozen_token_selection_final_sha256']
    assert training['frozen_token_selection_initial_sha256'] == training['frozen_token_selection_final_sha256']
    result = dict(status='PASS_FULL155_SIGNAL_FILES_AND_TRAINING', mode=summary['mode'],
                  verified_at=datetime.now().astimezone().isoformat(), summary_sha256=sha(args.summary),
                  verifier_sha256=sha(__file__), optimizer_steps=len(training['steps']), epochs=epochs,
                  checked_record_exposures=len(training['steps']) * 64,
                  unique_source_records_seen=len(seen), checkpoint_sha256=sha(checkpoint),
                  checkpoint_state_sha256=state_sha(state), image_forwards=0, optimizer_updates=0)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('summary', 'config', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--m0-receipt', type=Path)
    parser.add_argument('--m0-verification', type=Path)
    args = parser.parse_args()
    assert args.m0_receipt is None or args.m0_verification is not None
    run(args)
