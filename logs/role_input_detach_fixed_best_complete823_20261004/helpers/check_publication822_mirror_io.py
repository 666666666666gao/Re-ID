from datetime import datetime
from pathlib import Path
import json
import shlex
import paramiko

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/publication822_mirror_io_check')
assert not packet.exists()
packet.mkdir()
rows = []
for port, root in ((2026, '/data/gaob/Re-ID/Trifusion'), (2025, '/data2/gb/Re-ID/Trifusion')):
    client = paramiko.SSHClient()
    client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
    client.connect('172.19.12.138', port=port, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
    commands = [
        ['stat', '-c', '%n %F %s', root, root + '/.aris/compute', root + '/.aris/compute/TRIFUSION_EIGHT_GPU_POOL_20261002.md'],
        ['sha256sum', root + '/docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'],
        ['sha256sum', root + '/.aris/compute/TRIFUSION_EIGHT_GPU_POOL_20261002.md'],
    ]
    if port == 2026:
        commands += [['git', '-C', root, 'rev-parse', 'HEAD']]
    for command in commands:
        _, out, err = client.exec_command(shlex.join(command))
        out.channel.settimeout(60)
        data, error = out.read(), err.read()
        rows.append({'port': port, 'command': command, 'exit_code': out.channel.recv_exit_status(),
                     'stdout': data.decode(), 'stderr': error.decode()})
    client.close()
record = {'at': datetime.now().astimezone().isoformat(), 'rows': rows,
          'boundary': 'Read-only metadata/text checks after the actual 2025 mirror I/O error; no GPU queries or model/conda calls, no publication replay.'}
(packet / 'CHECK.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
