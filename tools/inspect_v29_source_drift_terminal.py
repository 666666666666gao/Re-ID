#!/usr/bin/env python3
"""Inspect the complete original V29 source pipeline and enumerate text-only evidence."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def inspect(args):
    run = args.run_dir.resolve()
    assert run == Path('/root/autodl-tmp/trifusion-v2/artifacts/v29_source_role_drift_seed42_1d52c1e')
    prefix = str(run)
    launcher = read(prefix + '_launcher.json')
    assert launcher['wrapper_pid'] == 159453
    assert launcher['repository_commit'] == '1d52c1e9ca957737050072e8e2ace427417cbe14'
    pipeline = read(prefix + '_pipeline_exit.json')
    assert pipeline['exit_code'] == 0 and pipeline['stage'] == 'COMPLETE_VERIFIED_SOURCE_ROLE_DRIFT'
    queue_launch = read(prefix + '_report_queue_launch.json')
    assert queue_launch['queue_pid'] == 160653 and queue_launch['upstream_wrapper_pid'] == 159453
    assert queue_launch['poll_seconds'] == 240
    queue_exit = read(prefix + '_report_queue_exit.json')
    assert queue_exit['exit_code'] == 0 and queue_exit['status'] == 'COMPLETE_SCALAR_REPORT'
    processes = {'source_wrapper': launcher['wrapper_pid'], 'report_queue': queue_launch['queue_pid']}
    for stage in ('math', 'diagnostic', 'verification', 'report'):
        start, end = read(prefix + '_' + stage + '_launch.json'), read(prefix + '_' + stage + '_exit.json')
        assert start['original_pid'] == end['original_pid'] and end['exit_code'] == 0
        processes[stage] = start['original_pid']
    assert processes['diagnostic'] == 159463 and processes['math'] == 159455
    assert all(not Path('/proc', str(pid)).exists() for pid in processes.values())
    summary_path = run / 'source_role_drift.json'
    summary = read(summary_path)
    assert summary['status'] == 'COMPLETE_FIXED_SOURCE_ROLE_DRIFT_PENDING_CPU_VERIFICATION'
    assert summary['completed_batches'] == 1680 and summary['model_forwards'] == 10080
    assert summary['new_optimizer_updates'] == 0 and not summary['retrieval_evaluation']
    assert len(summary['folds']) == 3
    verification_path = run / 'complete_source_drift_verification.json'
    verification = read(verification_path)
    assert verification['status'] == 'PASS_COMPLETE_V29_SOURCE_ROLE_DRIFT_AND_RELATIONS'
    assert verification['source_summary_sha256'] == sha(summary_path)
    expected_counts = {
        'checked_batches': 1680, 'checked_original_model_forwards': 10080,
        'checked_similarity_values': 743178240, 'checked_vector_observations': 7741440,
        'checked_protocol_triplets_per_state_view': 45549504,
        'relation_group_count': 12, 'relation_identity_count': 564,
        'vector_group_count': 648, 'vector_identity_count': 20304,
    }
    for key, value in expected_counts.items():
        assert verification[key] == value
    exports_dir = run / 'complete_scalar_report'
    export_path = exports_dir / 'export_receipt.json'
    export = read(export_path)
    assert export['status'] == 'PASS_COMPLETE_SOURCE_DRIFT_SCALAR_EXPORT'
    assert export['source_summary_sha256'] == sha(summary_path)
    assert export['source_verification_sha256'] == sha(verification_path)
    assert export['source_files'] == verification['files']
    assert export['reporter_sha256'] == queue_launch['reporter_sha256']
    assert export['math_module_sha256'] == queue_launch['math_sha256']
    assert export['all_identity_relation_sums_rechecked']
    assert export['new_model_forwards'] == export['optimizer_updates'] == export['image_reads'] == 0
    assert export['torch_imports'] == export['retrieval_evaluations'] == 0
    expected_exports = {
        'relation_groups.csv': 1296, 'identity_relations.csv': 60912,
        'joint_groups.csv': 24, 'joint_identity_relations.csv': 1128,
        'vector_groups.csv': 648, 'identity_vectors.csv': 20304,
        'relation_totals.csv': 216, 'joint_totals.csv': 4,
    }
    assert set(export['exports']) == set(expected_exports) | {'README.md'}
    for name, count in expected_exports.items():
        assert export['exports'][name]['rows'] == count

    # Fixed whitelist: matrices, vectors, model weights and images stay on the server.
    suffixes = (
        '_launcher.json', '_prelaunch.json', '_submission.json',
        '_math_launch.json', '_math_exit.json', '_math.json', '_math.log',
        '_diagnostic_launch.json', '_diagnostic_exit.json', '_diagnostic.log',
        '_verification_launch.json', '_verification_exit.json', '_verification.log',
        '_pipeline_exit.json', '_screen.log',
        '_report_prelaunch.json', '_report_submission.json', '_report_queue_launch.json',
        '_report_queue_exit.json', '_report_launch.json', '_report_exit.json',
        '_report.log', '_report_queue_screen.log',
    )
    text_files = [(Path(prefix + suffix), Path(prefix + suffix).name) for suffix in suffixes]
    text_files += [(summary_path, summary_path.name), (verification_path, verification_path.name)]
    for name, item in verification['files'].items():
        path = run / name
        assert path.resolve().parent == run and path.suffix == '.json'
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        text_files.append((path, name))
    retained = []
    for fold in summary['folds']:
        assert fold['all_model_states_unchanged'] and fold['all_gradients_absent']
        for kind in ('similarities', 'vectors', 'receipts'):
            item = fold[kind]
            path = Path(item['path']).resolve()
            assert path.parent == run
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
            retained.append({'kind': kind, 'fold': fold['fold'], **item})
        receipt_path = Path(fold['receipts']['path'])
        assert receipt_path.name == f"fold_{fold['fold']}_batch_receipts.jsonl"
        text_files.append((receipt_path, receipt_path.name))
        for binding in fold['bindings'][1:]:
            assert sha(binding['checkpoint']) == binding['checkpoint_sha256']
    for name, item in export['exports'].items():
        path = exports_dir / name
        assert path.resolve().parent == exports_dir and path.suffix in ('.csv', '.md')
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
        text_files.append((path, 'complete_scalar_report/' + name))
    text_files.append((export_path, 'complete_scalar_report/' + export_path.name))
    assert len(text_files) == 42 and len({name for _, name in text_files}) == 42
    files = {}
    for path, relative in text_files:
        assert path.suffix in ('.json', '.jsonl', '.log', '.md', '.csv')
        files[relative] = {'remote': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}
    result = {
        'status': 'PASS_COMPLETE_ORIGINAL_SOURCE_PIPELINE_AND_TEXT_EVIDENCE',
        'checked_at': datetime.now().astimezone().isoformat(),
        'execution_commit': launcher['repository_commit'], 'all_original_processes_ended': processes,
        'counts': expected_counts, 'source_summary_sha256': sha(summary_path),
        'verification_sha256': sha(verification_path), 'export_receipt_sha256': sha(export_path),
        'inspector_sha256': sha(__file__), 'text_files': files,
        'total_text_bytes': sum(item['bytes'] for item in files.values()),
        'retained_remote_artifacts_rehashed': retained,
        'six_final_checkpoints_rehashed': True,
        'new_model_forwards': 0, 'optimizer_updates': 0, 'retrieval_evaluations': 0,
        'independent_external_review': False,
    }
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    inspect(parser.parse_args())
