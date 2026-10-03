"""Real author training batch and common-state parity before any updates."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_independent_native_evidence as entry
from tools import queue_independent_native_evidence as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--dataset', choices=panel.DATASETS, required=True)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    entry.configure()
    protocol_path = panel.PROTOCOLS / f'{args.dataset}.json'
    protocol = entry.runner.read_protocol(protocol_path, args.dataset)
    states, predictions, inputs, configurations = {}, {}, {}, {}
    for variant in panel.VARIANTS:
        values = argparse.Namespace(dataset=args.dataset, variant=variant, recipe=variant,
            mode='prepare', seed=42, epochs=50, protocol=protocol_path,
            signal_source=panel.SOURCE, clip_weight=panel.WEIGHTS / 'ViT-B-16.pt',
            initialization=campaign / 'initialization' / f'{args.dataset}_{variant}.json',
            output_dir=ROOT / 'trained-model/native_pair_unused',
            baseline_sha256=entry.runner.sha256(panel.WEIGHTS / 'ViT-B-16.pt'))
        model, cfg, binding = entry.build_core(values, protocol)
        assert binding == panel.expected_binding(campaign, args.dataset, variant)
        states[variant] = {name: value.detach().cpu().clone()
                           for name, value in model.state_dict().items()
                           if '.detail_reader.' not in name}
        raw = next(iter(entry.original_train_loader(values, protocol, cfg)))
        batch, labels = entry.runner._training_batch(raw)
        assert len(labels) == cfg.SOLVER.IMS_PER_BATCH
        inputs[variant] = {'images': {name: entry.clean.tensor_digest(value)
                                     for name, value in batch['images'].items()},
                          'labels': labels.tolist(), 'cameras': batch['camera_ids'].tolist(),
                          'paths': list(raw[4])}
        configurations[variant] = cfg.dump()
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            output = model(batch, return_aux=True)
        predictions[variant] = {'raw_fused': output['raw_fused'].cpu(),
            'fused': output['fused'].cpu(), 'shared_global': output['shared_global'].cpu(),
            'heads': [(score.cpu(), feature.cpu()) for score, feature in output['heads']]}
        del model, output, batch, raw
        torch.cuda.empty_cache()
    assert len(set(configurations.values())) == 1
    assert inputs['global_only'] == inputs['semantic'] == inputs['native']
    a, b = states['semantic'], states['native']
    assert set(a) == set(b) and all(torch.equal(a[name], b[name]) for name in a)
    common = states['global_only']
    assert set(common).issubset(a) and all(torch.equal(common[name], a[name]) for name in common)
    assert all(torch.equal(predictions['global_only']['shared_global'], predictions[variant]['shared_global'])
               for variant in ('semantic', 'native'))
    for name in ('raw_fused', 'fused', 'shared_global'):
        assert torch.equal(predictions['semantic'][name], predictions['native'][name]), name
    assert all(torch.equal(x, y) for pair_a, pair_b in zip(predictions['semantic']['heads'], predictions['native']['heads'])
               for x, y in zip(pair_a, pair_b))
    path = campaign / f'initial_forward_pair_{args.dataset}.json'
    assert not path.exists()
    path.write_text(json.dumps({'schema': entry.SCHEMA, 'status': 'INITIAL_FORWARD_PAIR_PASS',
        'dataset': args.dataset, 'batch': inputs['semantic'],
        'all_common_state_exact': True, 'semantic_native_all_original_state_exact': True,
        'native_zero_exit_initial_prediction_exact': True,
        'initial_semantic_raw_prediction_sha256': entry.clean.tensor_digest(predictions['semantic']['raw_fused']),
        'initial_global_raw_prediction_sha256': entry.clean.tensor_digest(predictions['global_only']['raw_fused']),
        'boundary': 'Real full author training batch, AMP eval forward; no updates or official scoring. Global-only has shared adaptation; role/native effects still require independent training.'}, indent=2) + '\n')
    print(path, flush=True)


if __name__ == '__main__':
    main()
