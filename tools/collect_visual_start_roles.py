"""Verify six complete visual-start comparisons using the unchanged role evaluator."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import collect_role_global_tokens as original
from tools.collect_correspondence_roles import read, sha
from tools.queue_correspondence_refinement import child_campaign

CONDITIONS = ('public_visual', 'reid_visual')
DATASETS = ('RGBNT100', 'RGBNT201', 'MSVR310')


def verify(run, m0_run, dataset, variant, manifest):
    inputs_path = Path(manifest['inputs_path'])
    assert sha(inputs_path) == manifest['inputs_sha256']
    inputs = read(inputs_path)
    checkpoint = inputs['datasets'][dataset][variant]
    assert sha(Path(checkpoint['path'])) == checkpoint['sha256']
    # The reused verifier operates on a single fixed STATIC architecture.
    original.BASELINES = {**original.BASELINES, dataset: (checkpoint['path'], checkpoint['sha256'])}
    result = original.verify(run, m0_run, dataset, 'static', manifest)
    binding = read(run / 'training.json')['initializer']
    assert binding['visual_start'] == variant
    assert binding['visual_start_inputs_sha256'] == manifest['inputs_sha256']
    assert binding['visual_start_entry_sha256'] == manifest['source_sha256']['tools/run_visual_start_roles.py']
    result.update(visual_start=variant, visual_start_inputs_sha256=manifest['inputs_sha256'])
    return result


def collect(campaign):
    manifest = read(campaign / 'manifest.json')
    assert manifest['schema'] == 'trifusion-visual-start-panel-v1'
    assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
    rows = []
    for variant in CONDITIONS:
        for dataset in DATASETS:
            folder = child_campaign(campaign, 'visual_start', dataset, variant)
            child = read(folder / 'campaign.json')
            assert child['status'] == 'COMPLETE'
            assert [stage['mode'] for stage in child['jobs']] == ['m0', 'train', 'evaluate']
            assert all(stage['status'] == 'COMPLETE' and stage['exit_code'] == 0 for stage in child['jobs'])
            run, m0_run = Path(child['jobs'][2]['output_dir']), Path(child['jobs'][0]['output_dir'])
            assert child['jobs'][1]['output_dir'] == str(run)
            result = verify(run, m0_run, dataset, variant, manifest)
            rows.append({'dataset': dataset, 'variant': variant, 'run_dir': str(run),
                         'campaign_dir': str(folder), **result})
    return {'schema': 'trifusion-visual-start-verification-v1', 'expected_endpoints': 6,
            'verified_complete': len(rows), 'rows': rows,
            'boundary': 'Different frozen visual states; matched new trainable initialization is bound by the preflight witness.'}
