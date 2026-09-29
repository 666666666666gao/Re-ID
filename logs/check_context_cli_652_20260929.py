from datetime import datetime
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path('/data/gaob/Re-ID/Trifusion')
sys.path.insert(0, str(root))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
sources = json.loads((root / '.git/artifact_sha_652.json').read_text())
new_sources = [name for name in sources if name.endswith('.py') and name.startswith(('modeling/', 'tools/'))]
assert len(new_sources) == 5
for name in new_sources:
    assert sha(root / name) == sources[name]
    ast.parse((root / name).read_text(), filename=name)
queue = importlib.import_module('tools.queue_correspondence_context_identity')
collector = importlib.import_module('tools.collect_correspondence_context_identity')
assert len(queue.CONDITIONS) == 5 and collector.CONDITIONS == queue.CONDITIONS
commands = []
for variant, (query, auxiliary) in queue.CONDITIONS.items():
    for dataset in queue.queue.DATASETS:
        for mode in ('m0', 'train', 'evaluate'):
            command = queue.command(dataset, variant, mode, root / '.git/unused_context_cli_probe')
            assert command[command.index('--query-mode') + 1] == query
            assert command[command.index('--auxiliary-target') + 1] == auxiliary
            assert '--no-m3' in command and '--m1' in command and '--m2' in command
            assert command[command.index('--epochs') + 1] == '50'
            assert command[command.index('--seed') + 1] == '42'
            commands.append({'variant': variant, 'dataset': dataset, 'mode': mode})
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = ''
entries = [('tools/run_correspondence_context_identity.py', ['--query-mode', 'static', '--auxiliary-target', 'none']),
           ('tools/queue_correspondence_context_identity.py', []),
           ('tools/collect_correspondence_context_identity.py', []),
           ('tools/audit_correspondence_context_identity_losses.py', [])]
for name, flags in entries:
    result = subprocess.run([sys.executable, '-B', str(root / name), *flags, '--help'],
                            cwd=root, env=env, capture_output=True, text=True, check=True)
    assert 'usage:' in result.stdout
out = root / '.git/context_identity_cli_652_20260929.json'
assert not out.exists()
report = {'status': 'REAL_ENV_CPU_IMPORT_AST_AND_REGISTERED_COMMAND_PASS',
          'at': datetime.now().astimezone().isoformat(),
          'source_sha256': {name: sources[name] for name in new_sources},
          'command_contracts': commands, 'help_entries': [name for name, _ in entries],
          'boundary': 'Real environment imports and argument wiring only; no model/data construction, M0, GPU or retrieval.'}
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'at': report['at'], 'command_contracts': len(commands)}))
