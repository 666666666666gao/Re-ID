"""Check identical initial deployment outputs on two real training records."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_training_feature_scale as scale
from tools import queue_training_feature_scale as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--dataset', choices=panel.DATASETS, required=True)
    args = parser.parse_args()
    panel.configure()
    scale.configure()
    campaign = args.campaign.resolve()
    protocol_path = panel.PROTOCOLS / f'{args.dataset}.json'
    protocol = scale.foundation.runner.read_protocol(protocol_path, args.dataset)
    bindings = [panel.expected_binding(campaign, args.dataset, variant) for variant in panel.RECIPES]
    assert bindings[0]['initial_model_state_sha256'] == bindings[1]['initial_model_state_sha256']
    predictions = []
    records = scale.foundation.runner.records_for(protocol, 'train')[:2]
    for variant in panel.RECIPES:
        values = argparse.Namespace(dataset=args.dataset, recipe=variant, seed=42, epochs=50,
            protocol=protocol_path, signal_source=panel.SOURCE, clip_weight=panel.WEIGHTS/'ViT-B-16.pt',
            initialization=campaign/'initialization'/f'{args.dataset}_{variant}.json',
            output_dir=ROOT/'trained-model/f2_pair_unused',
            baseline_sha256=scale.foundation.runner.sha256(panel.WEIGHTS/'ViT-B-16.pt'))
        model, _cfg, binding = scale.build_core(values, protocol)
        assert binding == panel.expected_binding(campaign, args.dataset, variant)
        loader = scale.foundation.runner.loader_for(protocol, records, training=False, method='PLAIN_V8')
        raw = next(iter(loader))
        batch = scale.foundation.runner._eval_batch(raw, args.dataset)
        model.eval()
        with torch.inference_mode():
            predictions.append(model(batch).cpu())
        del model
        torch.cuda.empty_cache()
    assert torch.equal(predictions[0], predictions[1])
    output = campaign/f'initial_forward_pair_{args.dataset}.json'
    assert not output.exists()
    output.write_text(json.dumps({'schema':panel.SCHEMA, 'status':'INITIAL_FORWARD_PAIR_PASS',
        'dataset':args.dataset, 'records':records, 'initial_model_state_sha256':bindings[0]['initial_model_state_sha256'],
        'prediction_sha256':scale.foundation.clean.tensor_digest(predictions[0]),
        'max_abs_difference':float((predictions[0]-predictions[1]).abs().max()),
        'boundary':'Two real train records, eval transform, no optimizer or official scoring.'},indent=2)+'\n')
    print(output, flush=True)


if __name__ == '__main__':
    main()
