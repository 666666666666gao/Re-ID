"""Train full155-ID MSVR310 Signal at a fixed endpoint without official retrieval."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_msvr310_signal_oof import (
    configure, extract, loader_for, new_model, records_for, sha256, train_source, write_json,
)


def run(args):
    import torch
    from tools.run_signal_preserving_v5 import _module_state_sha256, _set_seed

    config = json.loads(args.config.read_bytes())
    assert config['schema'] == 'msvr310-signal-full155-fixed-v1'
    assert config['seed'] == 42 and config['epochs'] == 50
    for name, expected in config['source_files_sha256'].items():
        assert sha256(ROOT / name) == expected, name
    protocol_path = Path(config['protocol'])
    assert sha256(protocol_path) == config['protocol_sha256']
    protocol = json.loads(protocol_path.read_bytes())
    assert protocol['schema'] == 'trifusion-official-full-train-v1'
    assert protocol['dataset'] == 'MSVR310' and protocol['fold'] == 'full_train'
    assert protocol['source_count'] == len(protocol['records']) == 1032
    assert protocol['identity_count'] == len(protocol['source_ids']) == 155
    assert set(protocol['source_ids']).isdisjoint(protocol['official_test_ids'])
    baseline_path = ROOT / config['baseline_config']
    assert sha256(baseline_path) == config['baseline_config_sha256']
    baseline = json.loads(baseline_path.read_bytes())
    cfg, binding = configure(baseline)
    records = records_for(baseline, protocol, protocol, True)
    m0 = args.mode == 'm0'
    if not m0:
        preflight = json.loads(args.m0_receipt.read_bytes())
        verified = json.loads(args.m0_verification.read_bytes())
        assert preflight['status'] == 'PASS_FULL155_SIGNAL_ENGINEERING'
        assert preflight['config_sha256'] == sha256(args.config)
        assert preflight['runner_sha256'] == sha256(__file__)
        assert verified['status'] == 'PASS_FULL155_SIGNAL_FILES_AND_TRAINING'
        assert verified['summary_sha256'] == sha256(args.m0_receipt)
        assert verified['mode'] == 'm0'
    assert not args.output_dir.exists()
    args.output_dir.mkdir(parents=True)
    _set_seed(42)
    model = new_model(cfg, protocol)
    assert model.num_classes == 155
    initial = _module_state_sha256(model)
    if not m0:
        assert initial == preflight['training']['initial_state_sha256']
    start = time.perf_counter()
    summary = dict(schema=config['schema'], status='RUNNING', mode=args.mode,
                   started_at=datetime.now().astimezone().isoformat(), seed=42,
                   config_sha256=sha256(args.config), protocol_sha256=sha256(protocol_path),
                   runner_sha256=sha256(__file__), baseline_config_sha256=sha256(baseline_path),
                   execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   source_ids=protocol['source_ids'], source_records=1032, source_identities=155,
                   source_files_sha256=config['source_files_sha256'], fold='full_train',
                   official_model_record_forwards=0, official_metrics_read=False,
                   m0_weights_reused=False, **binding)
    write_json(args.output_dir / 'summary.json', summary)
    (args.output_dir / 'effective_config.yml').write_text(cfg.dump(), encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    training = train_source(model, loader_for(records, True), cfg,
                            protocol['source_record_indices'], records, preflight=m0)
    training['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024**2
    training['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 1024**2
    write_json(args.output_dir / 'training.json', training)
    seen = {i for row in training['steps'] for i in row['sampled_record_indices']}
    seen_ids = {protocol['records'][i]['identity'] for i in seen}
    checks = dict(fixed_epochs=training['epochs'] == (1 if m0 else 50),
                  no_missing_gradients=not training['trainable_without_gradient'],
                  overflow_zero=training['overflow_events'] == 0,
                  source_parameters_updated=training['initial_state_sha256'] == initial != training['final_state_sha256'],
                  selector_unchanged=training['frozen_token_selection_initial_sha256'] == training['frozen_token_selection_final_sha256'],
                  capacity_below24GiB=training['peak_reserved_mib'] < 24*1024)
    if m0:
        checks['eight_updates'] = training['optimizer_steps'] == 8
    else:
        checks['all_source_records_seen'] = seen == set(protocol['source_record_indices'])
        checks['all_source_identities_seen'] = seen_ids == set(protocol['source_ids'])
    summary.update(training=training, checks=checks, unique_source_records_seen=len(seen),
                   unique_source_identities_seen=len(seen_ids), status='TRAINED_CHECKS_PENDING')
    write_json(args.output_dir / 'summary.json', summary)
    assert all(checks.values()), checks
    checkpoint = args.output_dir / ('signal_m0.pth' if m0 else 'signal_epoch50.pth')
    torch.save(dict(model_state_dict=model.state_dict(), fold='full_train',
                    source_ids=protocol['source_ids'], config_sha256=sha256(args.config),
                    protocol_sha256=sha256(protocol_path)), checkpoint)
    before = extract(model, records[:8], cfg)
    assert _module_state_sha256(model) == training['final_state_sha256']
    del model
    torch.cuda.empty_cache()
    model = new_model(cfg, protocol)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    after = extract(model, records[:8], cfg)
    assert torch.equal(before, after) and tuple(after.shape) == (8, 3072)
    assert _module_state_sha256(model) == training['final_state_sha256']
    summary.update(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                   strict_reload_state_sha256=_module_state_sha256(model),
                   strict_reload_source_features_bitwise_equal=True, clean_source_record_forwards=16,
                   completed_at=datetime.now().astimezone().isoformat(), elapsed_seconds=time.perf_counter()-start,
                   status='PASS_FULL155_SIGNAL_ENGINEERING' if m0 else 'COMPLETE_FULL155_SIGNAL_FIXED_EPOCH50')
    if not m0:
        summary.update(m0_receipt_sha256=sha256(args.m0_receipt), m0_verification_sha256=sha256(args.m0_verification))
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({k: summary[k] for k in ('status', 'checkpoint_sha256', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('m0', 'baseline'), required=True)
    parser.add_argument('--m0-receipt', type=Path)
    parser.add_argument('--m0-verification', type=Path)
    args = parser.parse_args()
    assert args.mode == 'm0' or (args.m0_receipt is not None and args.m0_verification is not None)
    run(args)
