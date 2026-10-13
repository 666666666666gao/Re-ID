"""Fixed-data mean versus symmetric MaxSim; not new formal model results."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import diagnose_native_research_best as base
from tools import queue_joint_spectral_delta as panel

SCHEMA = 'trifusion-saved-region-comparison-v1'


def region_scores(query, gallery):
    assert query.device.type == gallery.device.type == 'cpu'
    assert query.shape[1:] == gallery.shape[1:] == (3, 3, 16, 128)
    assert bool(torch.isfinite(query).all()) and bool(torch.isfinite(gallery).all())
    query = F.normalize(query.float().flatten(1, 2), dim=-1)
    gallery = F.normalize(gallery.float().flatten(1, 2), dim=-1)
    mean = torch.zeros(query.shape[0], gallery.shape[0])
    maximum = torch.zeros_like(mean)
    for group in range(9):
        candidate = gallery[:, group].reshape(-1, 128).T
        for start in range(0, len(query), 16):
            q = query[start:start + 16, group]
            similarities = (q.reshape(-1, 128) @ candidate).reshape(
                len(q), 16, len(gallery), 16).permute(0, 2, 1, 3)
            mean[start:start + len(q)] += similarities.mean(dim=(2, 3)) / 9
            symmetric = .5 * (similarities.amax(dim=3).mean(dim=2)
                              + similarities.amax(dim=2).mean(dim=2))
            maximum[start:start + len(q)] += symmetric / 9
    assert bool(torch.isfinite(mean).all()) and bool(torch.isfinite(maximum).all())
    return dict(mean_pairs=mean, symmetric_maxsim=maximum)


def require_inputs(seal):
    assert seal['schema'] == SCHEMA and seal['conditions'] == list(panel.MODES)
    assert [row['aggregation'] for row in seal['rows']] == list(panel.MODES)
    for name, digest in seal['source_sha256'].items():
        assert panel.base.sha(ROOT / name) == digest, name
    for name, digest in seal['artifact_sha256'].items():
        assert panel.base.sha(Path(name)) == digest, name
    assert json.loads(Path(seal['original_campaign']).read_text())['status'] == 'COMPLETE'
    assert json.loads(Path(seal['original_controller_exit']).read_text())['exit_code'] == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_dir.exists()
    torch.set_num_threads(2)
    seal = json.loads(args.seal.read_text())
    require_inputs(seal)
    results = []
    args.output_dir.mkdir()
    for mode, row in zip(panel.MODES, seal['rows']):
        assert row['aggregation'] == mode
        run = Path(row['diagnosis_dir'])
        receipt = json.loads((run / 'DIAGNOSIS.json').read_text())
        assert receipt['status'] == 'COMPLETE' and receipt['aggregation'] == mode
        assert receipt['model_state_before_sha256'] == receipt['model_state_after_sha256']
        assert receipt['optimizer_updates'] == 0
        for name, proof in receipt['artifacts'].items():
            assert panel.base.sha(run / name) == proof['sha256']
        query = torch.load(run / 'query_features.pt', map_location='cpu', weights_only=False)
        gallery = torch.load(run / 'gallery_features.pt', map_location='cpu', weights_only=False)
        metadata = torch.load(run / 'diagnostic_distances.pt', map_location='cpu', weights_only=False)
        original = base.camera_scores(metadata['fused'].numpy(), metadata['query_ids'], metadata['gallery_ids'],
                                      metadata['query_cameras'], metadata['gallery_cameras'])
        assert all(abs(original['metrics'][key] - receipt['scores']['fused']['metrics'][key]) < 1e-5
                   for key in original['metrics'])
        for origin in ('processed_regions', 'reference_regions'):
            q, g = query['spectral'][origin], gallery['spectral'][origin]
            assert len(q) == len(metadata['query_ids']) and len(g) == len(metadata['gallery_ids'])
            started = time.perf_counter()
            similarities = region_scores(q, g)
            seconds = time.perf_counter() - started
            distances = {name: 1. - value for name, value in similarities.items()}
            scores = {name: base.camera_scores(value.numpy(), metadata['query_ids'], metadata['gallery_ids'],
                                               metadata['query_cameras'], metadata['gallery_cameras'])
                      for name, value in distances.items()}
            pair = base.paired(scores['mean_pairs'], scores['symmetric_maxsim'], metadata['query_ids'])
            name = mode + '_' + origin + '_distances.pt'
            torch.save(distances, args.output_dir / name)
            results.append(dict(aggregation=mode, origin=origin, scores=scores, comparison=pair,
                input_checkpoint_sha256=receipt['input_checkpoint_sha256'], cpu_comparison_seconds=seconds,
                distance_file=name, distance_sha256=panel.base.sha(args.output_dir / name),
                same_tokens_and_normalization=True, learned_or_selected_fusion_weight=False))
    require_inputs(seal)
    result = dict(schema=SCHEMA, status='COMPLETE_COUNTERFACTUAL_SAVED_REGION_DIAGNOSIS',
        completed_at=datetime.now().astimezone().isoformat(), rows=results, neural_forwards=0,
        optimizer_updates=0, formal_results_replaced=False,
        boundary='Mean of all pairwise cosine versus symmetric mean of row/column maxima on identical normalized tokens, within each of9 role/modality groups. Local-only scores; no global fusion coefficient, training, teacher, model selection, reranking subset or test centering. New diagnostic comparison rules use more stored vectors and pairwise computation; they are not original1536-vector formal results or complete ColBERT reproduction. Neither sign proves all late comparisons useful/useless or identifies compression/geometry as a causal failure.')
    base.write(args.output_dir / 'SUMMARY.json', result)
    lines = ['# 固定区域证据：均值与对称MaxSim比较', '',
             '| 条件 | 来源 | mean mAP/R1 | MaxSim mAP/R1 | ΔmAP | 修复/新增错误 |', '|---|---|---:|---:|---:|---:|']
    for row in results:
        a, b = [row['scores'][name]['metrics'] for name in ('mean_pairs', 'symmetric_maxsim')]
        pair = row['comparison']
        lines.append(f'| {row["aggregation"]} | {row["origin"]} | {a["mAP"]:.4f}/{a["Rank-1"]:.4f} | {b["mAP"]:.4f}/{b["Rank-1"]:.4f} | {pair["delta_metrics"]["mAP"]:+.4f} | {pair["rank1_repairs"]}/{pair["rank1_new_errors"]} |')
    lines += ['', result['boundary']]
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=result['status'],comparisons=len(results),neural_forwards=0)))


if __name__ == '__main__':
    main()
