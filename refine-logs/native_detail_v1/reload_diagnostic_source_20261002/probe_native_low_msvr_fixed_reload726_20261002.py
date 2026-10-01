"""Two read-only full eval forwards of one stored checkpoint; no official acceptance."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(ROOT))
from tools import run_native_detail as native

campaign = ROOT / 'logs/native_detail_20261002_v1'
manifest = json.loads((campaign / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 238
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in manifest['source_sha256'].items())
output = ROOT / 'logs/native_detail_reload_probe726_20261002'
assert not output.exists()
output.mkdir()
args = argparse.Namespace(dataset='MSVR310', variant='low', readout='roles', visual_update='low_lr',
    mode='evaluate', seed=42, epochs=50,
    protocol=ROOT / 'logs/official_three_dataset_protocols_20260923/MSVR310.json',
    signal_source=ROOT / 'comparators/Signal-cd1b0a6', clip_weight=ROOT / 'pertrained-model/ViT-B-16.pt',
    initialization=ROOT / 'logs/native_detail_preflight_20261002_v1/MSVR310_low.json',
    output_dir=ROOT / 'trained-model/native_detail_20261002_v1_clean_clip_low_MSVR310_seed42_full')
control = native.clean.control
runner = control.runner
args.baseline_sha256 = runner.sha256(args.clip_weight)
protocol = runner.read_protocol(args.protocol, args.dataset)
control.SCHEMA, control.condition, control.build = native.SCHEMA, native.condition, native.build
model, _cfg, _config, binding = native.build(args, protocol)
checkpoint = args.output_dir / 'best_map.pth'
checkpoint_before = runner.sha256(checkpoint)
original_distances = args.output_dir / 'official_distances.pt'
distance_before = runner.sha256(original_distances)
payload = control.load_checkpoint(checkpoint, model, args)
model.eval()
state_before = runner._module_state_sha256(model)
original_extract = runner.extract
feature_digests = []
input_digests = []

def audited_extract(current, current_protocol, split):
    original_eval_batch = runner._eval_batch
    images_digest = hashlib.sha256()

    def audited_batch(raw, dataset):
        batch = original_eval_batch(raw, dataset)
        for name, image in sorted(batch['images'].items()):
            images_digest.update(name.encode())
            images_digest.update(image.detach().cpu().contiguous().numpy().tobytes())
        images_digest.update(b'camera_ids')
        images_digest.update(batch['camera_ids'].detach().cpu().contiguous().numpy().tobytes())
        return batch

    runner._eval_batch = audited_batch
    features = original_extract(current, current_protocol, split)
    runner._eval_batch = original_eval_batch
    feature_digests.append({'split': split, 'sha256': hashlib.sha256(features.contiguous().numpy().tobytes()).hexdigest()})
    input_digests.append({'split': split, 'sha256': images_digest.hexdigest()})
    torch.save(features, output / f'features_{len(feature_digests)}_{split}.pt')
    return features

runner.extract = audited_extract
rows = []
for index in (1, 2):
    started = time.perf_counter()
    metrics = runner.official_metrics(model, protocol, args.signal_source, save_distances=output / f'distances_{index}.pt')
    rows.append({'index': index, 'metrics': metrics, 'elapsed_seconds': time.perf_counter() - started,
                 'delta_from_training': {name: metrics[name] - payload['metrics'][name] for name in metrics}})
assert runner._module_state_sha256(model) == state_before
assert runner.sha256(checkpoint) == checkpoint_before and runner.sha256(original_distances) == distance_before
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in manifest['source_sha256'].items())
first = torch.load(output / 'distances_1.pt', map_location='cpu', weights_only=False)['fused']
second = torch.load(output / 'distances_2.pt', map_location='cpu', weights_only=False)['fused']
original = torch.load(original_distances, map_location='cpu', weights_only=False)['fused']
assert torch.isfinite(first).all() and torch.isfinite(second).all()
result = {'status': 'FIXED_CHECKPOINT_DIAGNOSTIC_COMPLETE', 'created_at': datetime.now().astimezone().isoformat(),
    'dataset': 'MSVR310', 'variant': 'low', 'checkpoint_sha256': checkpoint_before,
    'original_failed_distance_sha256': distance_before, 'selected_epoch': payload['epoch'],
    'rows': rows, 'input_digests': input_digests, 'feature_digests': feature_digests,
    'fresh_passes_same_inputs': input_digests[:2] == input_digests[2:],
    'fresh_passes_same_features': feature_digests[:2] == feature_digests[2:],
    'fresh_distance_max_abs_difference': float((first - second).abs().max()),
    'pass1_minus_original_failed_distance_max_abs': float((first - original).abs().max()),
    'pass2_minus_original_failed_distance_max_abs': float((second - original).abs().max()),
    'torch_num_threads': torch.get_num_threads(),
    'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark,
    'model_state_unchanged': True, 'original_artifacts_unchanged': True, 'sources_238_unchanged': True,
    'boundary': 'Two diagnostic forwards of one loaded model in one process; checks within-process repeatability of images/camera inputs, features and distances, not cross-process cache equivalence. No training, optimizer, changed threshold, result selection or official acceptance; original failed evaluation remains failed.'}
(output / 'SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
