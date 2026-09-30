"""Archive six accepted FP32 endpoints; keep model weights and arrays remote."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    campaign, target, archive = (path.resolve() for path in (args.campaign, args.target, args.archive))
    assert all(path.is_relative_to(ROOT.resolve()) for path in (campaign, target, archive))
    state = json.loads((campaign / 'campaign.json').read_text())
    matrix_path = campaign / 'accepted_matrix.json'
    matrix = json.loads(matrix_path.read_text())
    observer_path = ROOT / 'logs/slot_competition_fp32_observer_20261001.json'
    observer = json.loads(observer_path.read_text())
    manifest = json.loads((campaign / 'manifest.json').read_text())
    assert state['status'] == observer['status'] == 'COMPLETE'
    assert matrix['schema'] == 'trifusion-slot-competition-fp32-roles-verification-v2'
    assert matrix['verified_complete'] == matrix['expected_endpoints'] == len(matrix['rows']) == observer['accepted_count'] == 6
    assert observer['accepted_matrix_sha256'] == sha(matrix_path)
    assert {(row['dataset'], row['variant']) for row in matrix['rows']} == {
        (dataset, variant) for dataset in ('RGBNT201', 'RGBNT100', 'MSVR310')
        for variant in ('independent', 'competitive')}
    assert len(manifest['source_sha256']) == 213
    assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
    assert not target.exists() and not archive.exists()
    target.mkdir()
    files = {}

    def copy(source, name):
        assert source.resolve().is_relative_to(ROOT.resolve())
        destination = target / name
        assert destination.resolve().is_relative_to(target)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert not destination.exists()
        shutil.copyfile(source, destination)
        digest = sha(source)
        assert sha(destination) == digest
        files[name] = {'source': str(source), 'sha256': digest, 'bytes': source.stat().st_size}

    for name in ('campaign.json', 'accepted_matrix.json', 'manifest.json', 'parent_matrix.json'):
        copy(campaign / name, name)
    for name in ('slot_competition_fp32_launch_20261001_r2.json',
                 'slot_competition_fp32_first_wave_20261001_r2.json',
                 'slot_competition_fp32_observer_launch_20261001.json',
                 'slot_competition_fp32_observer_20261001.json',
                 'slot_competition_fp32_observer_20261001.log',
                 'slot_competition_fp32_controller_20261001_r2.log'):
        copy(ROOT / 'logs' / name, name)
    for name in ('EXPERIMENT_PLAN.md', 'FP32_PLAN_20261001.md', 'FP32_CPU_CHECK_20261001.json'):
        copy(ROOT / 'refine-logs/slot_competition_roles_v1' / name, 'contract/' + name)
    copy(Path(__file__).resolve(), 'intake_source.py')
    for row in matrix['rows']:
        assert row['status'] == 'VERIFIED_COMPLETE'
        folder, run = Path(row['campaign_dir']), Path(row['run_dir'])
        child = json.loads((folder / 'campaign.json').read_text())
        assert child['status'] == 'COMPLETE'
        assert [job['mode'] for job in child['jobs']] == ['m0', 'train', 'evaluate']
        assert all(job['status'] == 'COMPLETE' and job['exit_code'] == 0 for job in child['jobs'])
        m0 = Path(child['jobs'][0]['output_dir'])
        assert child['jobs'][1]['output_dir'] == child['jobs'][2]['output_dir'] == str(run)
        assert sha(run / 'best_map.pth') == row['checkpoint_sha256']
        assert sha(run / 'official_distances.pt') == row['distance_sha256']
        assert sha(run / 'official_metrics.json') == row['receipt_sha256']
        m0_receipt = json.loads((m0 / 'training.json').read_text())
        assert sha(m0 / 'm0_reload_probe.pth') == m0_receipt['m0']['reload_probe_sha256']
        label = row['dataset'] + '_' + row['variant']
        copy(folder / 'campaign.json', label + '/campaign.json')
        for name in ('m0.log', 'train.log', 'evaluate.log'):
            copy(folder / name, label + '/' + name)
        for name in ('training.json', 'training_steps.jsonl', 'official_metrics.json'):
            copy(run / name, label + '/' + name)
        for name in ('training.json', 'training_steps.jsonl'):
            copy(m0 / name, label + '/m0/' + name)

    receipt = {'copied_at': datetime.now().astimezone().isoformat(), 'files': files,
               'accepted': 6, 'accepted_matrix_sha256': sha(matrix_path),
               'scope': 'Original six-end full50/M0/best/strict-reload text receipts only. '
                        'Weights and distances remain remote; no rerun or metric mutation.'}
    (target / 'INTAKE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(target, arcname=target.relative_to(ROOT))
    print(json.dumps({'status': 'COMPLETE_TEXT_INTAKE', 'copied_files': len(files),
                      'accepted': 6, 'archive': str(archive), 'archive_sha256': sha(archive),
                      'archive_bytes': archive.stat().st_size,
                      'intake_sha256': sha(target / 'INTAKE.json')}, indent=2))


if __name__ == '__main__':
    main()
