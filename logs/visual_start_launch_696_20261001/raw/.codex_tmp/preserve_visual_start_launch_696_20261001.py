"""Preserve actual launch and production-M0 evidence without selecting interim scores."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/data/gaob/Re-ID/Trifusion')
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
DEST = ROOT / '.codex_tmp/visual_start_launch_696_20261001'
TAR = DEST.with_suffix('.tar.gz')
assert not DEST.exists() and not TAR.exists()
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
           for name, digest in manifest['source_sha256'].items())
state = json.loads((CAMPAIGN / 'campaign.json').read_text())
files = [CAMPAIGN / 'manifest.json', CAMPAIGN / 'campaign.json',
         ROOT / 'logs/visual_start_roles_launch_20261001_v1.json',
         ROOT / 'logs/visual_start_roles_wrapper_20261001_v1.json',
         ROOT / 'logs/visual_start_roles_helpers_launch_20261001.json',
         ROOT / 'logs/visual_start_roles_observer_1145_wrapper_20261001.json',
         ROOT / 'logs/visual_start_roles_analysis_waiter_wrapper_20261001.json',
         ROOT / 'logs/visual_start_roles_observer_1145_20261001.json',
         ROOT / 'logs/visual_start_roles_controller_20261001_v1.log',
         ROOT / 'logs/role_global_tokens_milestone_sync695_20261001.json',
         CAMPAIGN / 'analysis_waiter_status.json',
         Path(manifest['inputs_path']), Path(manifest['initialization_witness_path']),
         ROOT / 'tools/report_visual_start_roles_complete.py']
for filename in ('launch_visual_start_roles_20261001.py', 'launch_visual_start_observers_20261001.py',
                 'observe_visual_start_roles_1145_20261001.py', 'wait_visual_start_complete_analysis_20261001.py',
                 'preserve_visual_start_launch_696_20261001.py', 'sync_role_global_tokens_695_20261001.py'):
    files.append(ROOT / '.codex_tmp' / filename)
rows = []
for row in state['jobs']:
    item = dict(row)
    child = CAMPAIGN / (CAMPAIGN.name + '_visual_start_' + row['variant'] + '_' + row['dataset'])
    if (child / 'campaign.json').exists():
        child_state = json.loads((child / 'campaign.json').read_text())
        item['child'] = child_state
        files.append(child / 'campaign.json')
        for stage in child_state['jobs']:
            files.append(child / (stage['mode'] + '.log'))
            if stage['mode'] == 'm0' and stage['status'] == 'COMPLETE':
                run = Path(stage['output_dir'])
                m0 = json.loads((run / 'training.json').read_text())
                assert m0['status'] == 'M0_PASS' and stage['exit_code'] == 0
                assert m0['m0']['nonzero_gradient_parameters'] == m0['m0']['trainable_parameters'] == 128
                assert m0['m0']['frozen_signal_unchanged'] and m0['m0']['reload_max_abs_difference'] == 0
                item['m0'] = m0
                files.extend((run / 'training.json', run / 'training_steps.jsonl'))
            if stage['status'] == 'RUNNING':
                process = Path('/proc') / str(stage['pid'])
                item['running_process_exists'] = process.exists()
                if process.exists():
                    item['running_command'] = (process / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
    rows.append(item)
record = {'observed_at': datetime.now().astimezone().isoformat(), 'campaign': state, 'rows': rows,
          'source_count_verified': len(manifest['source_sha256']),
          'disk_free_bytes': shutil.disk_usage(ROOT).free,
          'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits']).decode(),
          'cpu_report_status': 'PREPARED_AND_WAITING_NOT_EXECUTED',
          'scope': 'Real production M0 and live full50 processes; no interim score selection or partial official endpoint.'}
DEST.mkdir()
entries = []
for path in dict.fromkeys(files):
    content = path.read_bytes()
    target = DEST / 'raw' / path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    entries.append({'path': target.relative_to(DEST).as_posix(), 'bytes': len(content),
                    'sha256': hashlib.sha256(content).hexdigest()})
(DEST / 'SNAPSHOT.json').write_text(json.dumps(record, indent=2) + '\n')
(DEST / 'INTAKE.json').write_text(json.dumps({'captured_at': record['observed_at'], 'raw_files': entries}, indent=2) + '\n')
with tarfile.open(TAR, 'w:gz') as archive:
    archive.add(DEST, arcname='.')
print(json.dumps({'observed_at': record['observed_at'], 'archive': str(TAR),
                  'archive_bytes': TAR.stat().st_size, 'archive_sha256': hashlib.sha256(TAR.read_bytes()).hexdigest(),
                  'raw_files': len(entries), 'source_count_verified': record['source_count_verified'],
                  'gpu': record['gpu'], 'disk_free_bytes': record['disk_free_bytes'],
                  'status': state['status'], 'm0_pass': sum('m0' in row for row in rows),
                  'running': sum(row['status'] == 'RUNNING' for row in rows),
                  'pending': sum(row['status'] == 'PENDING' for row in rows)}))
