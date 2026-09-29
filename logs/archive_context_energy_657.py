from pathlib import Path
import hashlib
import json
import subprocess

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
matrix = json.loads((root / 'logs/context_identity_accepted_667_20260929.json').read_bytes())
rows = {(row['dataset'], row['variant']): row for row in matrix['rows'] if row['status'] == 'VERIFIED_COMPLETE'}
scp = ['scp', '-P', '2026', '-i', 'C:/Users/gb/.ssh/id_ed25519', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ProxyCommand=none']
reports = []
for dataset in ('RGBNT201', 'MSVR310'):
    folder = root / f'logs/context_energy_{dataset}_657_20260929'
    folder.mkdir()
    subprocess.run(scp + [f'gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/.git/{folder.name}/*.json', str(folder)], check=True)
    summary = json.loads((folder / 'summary.json').read_bytes())
    assert summary['status'] == 'FIVE_ACCEPTED_BEST_ENERGY_DIAGNOSTICS_COMPLETE'
    assert summary['source_sha256'] == sha(root / 'tools/diagnose_correspondence_context_energy.py')
    assert summary['matrix_sha256'] == sha(root / 'logs/context_identity_accepted_667_20260929.json')
    assert len(summary['rows']) == 5 and len(list(folder.glob('*.json'))) == 6
    for cell in summary['rows']:
        row = rows[(dataset, cell['variant'])]
        assert cell == json.loads((folder / f"{cell['variant']}.json").read_bytes())
        for name in ('checkpoint_sha256', 'receipt_sha256', 'distance_sha256', 'best_epoch'):
            assert cell[name] == row[name]
        assert cell['maximum_forward_reconstruction_error'] < 1e-5
        assert max(cell['maximum_saved_distance_difference'].values()) < 1e-5
    reports.append({'dataset': dataset, 'at': summary['at'], 'files': {p.name: sha(p) for p in folder.glob('*.json')}})
    print(json.dumps({'dataset': dataset, 'at': summary['at'], 'rows': [
        {'variant': cell['variant'], 'gain': cell['readout_gain'],
         'query_scaled_correction_ratio': cell['statistics']['query']['scaled_correction_to_global_norm']['mean'],
         'query_global_correction_cosine': cell['statistics']['query']['global_correction_cosine']['mean'],
         'query_global_fused_l2': cell['statistics']['query']['global_fused_l2']['mean'],
         'maximum_distance_difference': max(cell['maximum_saved_distance_difference'].values())}
        for cell in summary['rows']]}))
subprocess.run(scp + ['gaob@172.19.12.138:/data/gaob/Re-ID/Trifusion/.git/context_energy_launch_657_20260929.json', str(root / 'logs/context_energy_launch_657_20260929.json')], check=True)
output = root / 'logs/context_energy_archive_657_20260929.json'
assert not output.exists()
output.write_text(json.dumps({'status': 'TEN_ACCEPTED_BEST_ENERGY_JSON_RECEIPTS_ARCHIVED', 'reports': reports}, indent=2) + '\n', encoding='utf-8')
