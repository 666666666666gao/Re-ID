"""Standalone CPU engineering witness; synthetic inputs, not ReID M0/results."""
import argparse
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    spec = importlib.util.spec_from_file_location('independent_image_evidence', args.source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = module.ImageNativeEvidenceReader().cpu()
    named = dict(reader.named_parameters())
    assert sum(p.numel() for p in named.values()) == 159424
    assert len(named) == 15 and all(p.requires_grad for p in named.values())
    fixture_rows = []
    fixtures = []
    for height, width in ((256, 128), (128, 256)):
        images = torch.randn(2, 3, 3, height, width)
        queries = torch.randn(2, 3, 16, 128)
        semantic = torch.randn_like(queries)
        target = torch.randn_like(queries)
        original_images, original_queries = images.clone(), queries.clone()
        detail = reader(images, queries)
        assert detail.shape == semantic.shape and bool(torch.isfinite(detail).all())
        assert torch.count_nonzero(detail).item() == 0
        assert torch.equal(semantic + detail, semantic)
        fixtures.append((images, queries, semantic, target, original_images, original_queries))
        fixture_rows.append({'input_shape': list(images.shape), 'output_shape': list(detail.shape),
                             'initial_detail_exact_zero': True, 'synthetic_addition_exact_equal': True})
    optimizer = torch.optim.AdamW(reader.parameters(), lr=3.5e-4, weight_decay=1e-4)
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(optimizer_ids) == len(set(optimizer_ids))
    assert set(optimizer_ids) == {id(p) for p in named.values()}
    support = {name: False for name in named}
    steps = []
    for index in range(8):
        images, queries, semantic, target, _, _ = fixtures[index % len(fixtures)]
        optimizer.zero_grad(set_to_none=True)
        output = semantic + reader(images, queries)
        loss = (output - target).square().mean()
        assert bool(torch.isfinite(loss))
        loss.backward()
        row = {}
        for name, parameter in named.items():
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
            nonzero = bool(torch.count_nonzero(parameter.grad).item())
            row[name] = nonzero
            support[name] = support[name] or nonzero
        if index == 0:
            assert row['output.weight']
            assert not any(value for name, value in row.items() if name != 'output.weight')
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in named.values())
        steps.append({'step': index + 1, 'toy_loss': float(loss.detach()), 'nonzero_gradients': row})
    assert all(support.values()), support
    buffer = io.BytesIO()
    torch.save(reader.state_dict(), buffer)
    buffer.seek(0)
    restored = module.ImageNativeEvidenceReader().cpu()
    restored.load_state_dict(torch.load(buffer, map_location='cpu', weights_only=True), strict=True)
    for images, queries, _, _, original_images, original_queries in fixtures:
        assert torch.equal(images, original_images) and torch.equal(queries, original_queries)
        with torch.no_grad():
            assert torch.equal(reader(images, queries), restored(images, queries))
    receipt = {'status': 'PASS_STANDALONE_CPU_ENGINEERING_ONLY',
               'observed_at': datetime.now().astimezone().isoformat(),
               'torch_version': torch.__version__, 'device': 'cpu', 'threads': 4,
               'seed': 42, 'synthetic_batch_size': 2, 'parameter_count': 159424,
               'trainable_tensors': len(named), 'fixtures': fixture_rows,
               'optimizer_covers_component_exactly_once': True,
               'toy_updates': len(steps), 'steps': steps, 'cumulative_gradient_support': support,
               'strict_component_state_reload_outputs_exact': True, 'inputs_unchanged': True,
               'scope': 'Synthetic standalone component and semantic-addition arithmetic only; '
                        'not full role integration, actual dataset inputs, paired initialization, '
                        'production optimizer, AMP/CUDA M0, formal training or retrieval evidence.'}
    args.output.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('status', 'observed_at', 'parameter_count',
                     'trainable_tensors', 'toy_updates', 'scope')}, indent=2))


if __name__ == '__main__':
    main()
