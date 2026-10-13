"""Frozen CLIP patch source diagnosis; retain the original global encoding."""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_joint_spectral_delta as producer
from tools import diagnose_native_research_best as base
from trifusion.native_clip_patch_evidence import query_self_attention, project_attention_patches

runner = producer.inner.runner
SCHEMA = 'trifusion-frozen-native-clip-patch-v1'
FEATURES = ('native_projected', 'attention_only_qk', 'attention_only_qq')


def require_inputs(seal):
    assert seal['schema'] == SCHEMA
    for name, digest in seal['source_sha256'].items():
        assert runner.sha256(ROOT / name) == digest, name
    for name, digest in seal['artifact_sha256'].items():
        assert runner.sha256(Path(name)) == digest, name
    assert json.loads(Path(seal['original_campaign']).read_text())['status'] == 'COMPLETE'
    assert json.loads(Path(seal['original_exit']).read_text())['exit_code'] == 0


def extract(model, protocol, split):
    backbone = model.evidence_model.backbone
    visual = backbone.signal.clip_vision_encoder.base
    attention = visual.transformer.resblocks[-1].attn
    assert attention.embed_dim == 768 and attention.num_heads == 12
    assert attention.dropout == 0 and visual.proj.shape == (768, 512)
    current = {}
    captured = {name: [] for name in FEATURES}
    native_cls, globals_by_batch = [], []
    all_features = {name: [] for name in FEATURES}

    def input_hook(_module, inputs):
        current['input'] = inputs[0]

    def attention_hook(_module, _inputs, output):
        current['qk'] = output[0]

    def visual_hook(_module, _inputs, output):
        assert output.shape[1:] == (129, 512)
        captured['native_projected'].append(output[:, 1:].float().cpu())
        native_cls.append(output[:, 0].float().cpu())
        qk = project_attention_patches(visual, current['qk'])
        qq = project_attention_patches(visual, query_self_attention(attention, current['input']))
        for name, value in (('attention_only_qk', qk), ('attention_only_qq', qq)):
            assert value.shape == output[:, 1:].shape
            captured[name].append(value.float().cpu())
        current.clear()

    handles = [attention.register_forward_pre_hook(input_hook),
               attention.register_forward_hook(attention_hook), visual.register_forward_hook(visual_hook)]
    with torch.inference_mode():
        for raw in runner.loader_for(protocol, runner.records_for(protocol, split), training=False, method='PLAIN_V8'):
            assert not current and not native_cls and all(not value for value in captured.values())
            batch = runner._eval_batch(raw, 'RGBNT201')
            stages, global_feature = backbone(batch['images'], batch['camera_ids'])
            del stages
            assert len(native_cls) == 3 and all(len(value) == 3 for value in captured.values())
            cls = torch.stack(native_cls, dim=1).flatten(1)
            g = global_feature.float().cpu()
            assert torch.allclose(cls, g, atol=1e-6, rtol=1e-6)
            globals_by_batch.append(g)
            for name in FEATURES:
                all_features[name].append(torch.stack(captured[name], dim=1))
                captured[name].clear()
            native_cls.clear()
    for handle in handles:
        handle.remove()
    g = torch.cat(globals_by_batch)
    features = {name: torch.cat(value) for name, value in all_features.items()}
    assert g.shape == (protocol['counts'][split], 1536)
    assert all(value.shape == (len(g), 3, 128, 512) and bool(torch.isfinite(value).all()) for value in features.values())
    return g, features


def local_distances(query, gallery):
    """Identical token normalization; full-gallery mean and symmetric MaxSim."""
    assert query.shape[1:] == gallery.shape[1:] == (3, 128, 512)
    q = F.normalize(query.to('cuda:0'), dim=-1)
    g = F.normalize(gallery.to('cuda:0'), dim=-1)
    mean = torch.zeros(len(q), len(g), device=q.device)
    maximum = torch.zeros_like(mean)
    for modality in range(3):
        candidates = g[:, modality].reshape(-1, 512).T
        for start in range(0, len(q), 4):
            values = q[start:start + 4, modality]
            scores = (values.flatten(0, 1) @ candidates).reshape(len(values), 128, len(g), 128).permute(0, 2, 1, 3)
            mean[start:start + len(values)] += scores.mean(dim=(2, 3)) / 3
            maximum[start:start + len(values)] += .5 * (scores.amax(3).mean(2) + scores.amax(2).mean(2)) / 3
    assert bool(torch.isfinite(mean).all()) and bool(torch.isfinite(maximum).all())
    return dict(mean_pairs=(1. - mean).cpu(), symmetric_maxsim=(1. - maximum).cpu())


def tensor_summary(value):
    assert value.is_contiguous() and value.dtype == torch.float32 and value.device.type == 'cpu'
    mean = value.mean(0)
    return dict(shape=list(value.shape), bytes=value.numel() * value.element_size(),
                sha256=hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest(),
                mean_token_norm=float(value.norm(dim=-1).mean()),
                mean_token_sample_energy=float(value.square().sum(-1).mean()),
                mean_token_common_energy=float(mean.square().sum(-1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.seal, args.output_dir = args.seal.resolve(), args.output_dir.resolve()
    assert str(ROOT) == '/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert not args.output_dir.exists()
    torch.set_num_threads(2)
    seal = json.loads(args.seal.read_text())
    require_inputs(seal)
    row = seal['row']
    original = Path(row['run_dir'])
    values = argparse.Namespace(dataset='RGBNT201', variant='semantic', recipe='semantic', mode='evaluate',
        seed=42, epochs=50, protocol=Path(seal['protocol']), signal_source=Path(seal['signal_source']),
        clip_weight=Path(seal['clip_weight']), initialization=Path(row['initialization']), output_dir=original)
    values.baseline_sha256 = runner.sha256(values.clip_weight)
    producer.MODE = 'joint_delta'
    producer.configure()
    protocol = runner.read_protocol(values.protocol, 'RGBNT201')
    model, _cfg, binding = producer.inner.foundation.build(values, protocol)
    assert binding == row['initializer']
    payload = producer.inner.foundation.load(original / 'best_map.pth', model, values)
    assert payload['epoch'] == row['best_epoch'] == 8
    model.eval()
    before = runner._module_state_sha256(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    start = time.perf_counter()
    query_global, query = extract(model, protocol, 'query')
    gallery_global, gallery = extract(model, protocol, 'gallery')
    extraction_seconds = time.perf_counter() - start
    after = runner._module_state_sha256(model)
    assert before == after and all(parameter.grad is None for parameter in model.parameters())
    metadata = base.metadata(protocol)
    old = torch.load(Path(seal['prior_global_distance']), map_location='cpu', weights_only=False)
    assert all(np.array_equal(metadata[key], old[key]) for key in metadata)
    global_distance = runner.distance_matrix(query_global, gallery_global)
    difference = float((global_distance - old['global']).abs().max())
    assert difference <= 1e-5
    global_score = base.score(global_distance, metadata, 'RGBNT201', values.signal_source)
    assert all(abs(global_score['metrics'][key] - seal['global_metrics'][key]) < 1e-5 for key in seal['global_metrics'])
    args.output_dir.mkdir()
    scores, features = {}, {}
    torch.save(dict(metadata, global_distance=global_distance), args.output_dir / 'global_distance.pt')
    for name in FEATURES:
        started = time.perf_counter()
        distances = local_distances(query[name], gallery[name])
        seconds = time.perf_counter() - started
        scored = {kind: base.score(value, metadata, 'RGBNT201', values.signal_source) for kind, value in distances.items()}
        scores[name] = dict(scores=scored, comparison=base.paired(scored['mean_pairs'], scored['symmetric_maxsim'], metadata['query_ids']),
                            matrix_comparison_seconds=seconds)
        features[name] = dict(query=tensor_summary(query[name]), gallery=tensor_summary(gallery[name]))
        torch.save(dict(metadata, **distances), args.output_dir / (name + '_distances.pt'))
    require_inputs(seal)
    result = dict(schema=SCHEMA, status='COMPLETE_FROZEN_PATCH_SOURCE_DIAGNOSIS', scores=scores, features=features,
        global_score=global_score, global_distance_max_abs_difference=difference,
        input_checkpoint_sha256=row['checkpoint_sha256'], model_state_before_sha256=before,
        model_state_after_sha256=after, optimizer_updates=0, extraction_seconds=extraction_seconds,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        completed_at=datetime.now().astimezone().isoformat(),
        artifacts={path.name:dict(bytes=path.stat().st_size,sha256=runner.sha256(path)) for path in args.output_dir.glob('*.pt')},
        boundary='One frozen best backbone query/gallery pass; original global path untouched. Native projected128 patches versus auxiliary actual-QK-output and QQ-output without final residual/FFN; the latter also bypasses the last external role-adapter snapshots. Larger512-width/128-token bags differ from prior128-width/16-slot evidence, so no unique bottleneck/processing cause is established. Borrowed dense-reading principle, not full ClearCLIP segmentation reproduction. Scores are local-only diagnostics, no global fusion coefficient, fitting, test centering, new formal method score or teacher assumption. Feature tensors are hashed in memory, not saved; inputs/weights/source and resulting distances are retained for replay.')
    base.write(args.output_dir / 'SUMMARY.json', result)
    lines = ['# 原CLIP patch与辅助dense读取：固定权重', '', '| 来源 | mean mAP/R1 | MaxSim mAP/R1 | ΔmAP | 修复/新增错误 |', '|---|---:|---:|---:|---:|']
    for name, row in scores.items():
        a, b = [row['scores'][kind]['metrics'] for kind in ('mean_pairs', 'symmetric_maxsim')]
        pair = row['comparison']
        lines.append(f'| {name} | {a["mAP"]:.4f}/{a["Rank-1"]:.4f} | {b["mAP"]:.4f}/{b["Rank-1"]:.4f} | {pair["delta_metrics"]["mAP"]:+.4f} | {pair["rank1_repairs"]}/{pair["rank1_new_errors"]} |')
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n\n' + result['boundary'] + '\n')
    print(json.dumps(dict(status=result['status'],global_distance_max_abs_difference=difference,
        local_scores={name:{kind:value['metrics'] for kind,value in row['scores'].items()} for name,row in scores.items()})))


if __name__ == '__main__':
    main()
