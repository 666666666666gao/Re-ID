"""Observe the existing fixed diagnostic, then run its prepared result processing."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import shlex
import subprocess
import sys
import time

TMP = Path('C:/Users/gb/.codex_tmp')
REPO = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
RUN = '/root/autodl-tmp/trifusion-v2/artifacts/msvr310_history_candidate_gradient_v1_seed42_eebaaa0'
REMOTE_REPO = '/root/autodl-tmp/trifusion-v2/TriFusion-ReID'
REMOTE_PYTHON = '/root/miniconda3/envs/tri_reid/bin/python'
OUTPUT = TMP / 'history_gradient_complete_processing_20260908'
INTAKE = TMP / 'history_gradient_complete_source_20260908'
ANALYSIS = TMP / 'history_gradient_complete_analysis_20260908'
FIGURES = TMP / 'history_gradient_complete_figures_20260908'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')

def announce(event, **fields):
    print(json.dumps(dict(at=datetime.now().astimezone().isoformat(), event=event, **fields)), flush=True)

def local_step(name, arguments):
    result = subprocess.run([sys.executable, '-X', 'utf8', *map(str, arguments)], cwd=REPO, capture_output=True)
    (OUTPUT / (name + '.stdout.log')).write_bytes(result.stdout)
    (OUTPUT / (name + '.stderr.log')).write_bytes(result.stderr)
    assert result.returncode == 0, (name, result.returncode)
    announce(name, exit_code=result.returncode)
    return result.stdout

assert not any(p.exists() for p in (OUTPUT, INTAKE, ANALYSIS, FIGURES))
OUTPUT.mkdir()
bindings = {
    'tools/verify_msvr_history_gradient_all_statistics.py': 'bb9c078f7800c2d5255c5922df68ed5072e6e93931347cb185b7dcce44987026',
    'tools/analyze_msvr_history_gradient_text.py': 'df34dcc9a922d574c8276d4673991db8b5c00d363fe70b8b7851eb3adb4cd81f',
    'tools/plot_msvr_history_candidate_gradients.py': 'cff9c2edf07c51f588b2566c1f4bb3141b35dbd4c8908379228c00ec9e9c092c',
}
assert all(sha(REPO / name) == digest for name, digest in bindings.items())
save('execution_binding.json', dict(script_sha256=sha(Path(__file__)), scripts=bindings,
    observer_sha256=sha(TMP / 'observe_history_gradient_source_20260908.py'),
    intake_sha256=sha(TMP / 'receive_history_gradient_complete_source_20260908.py'),
    source_run=RUN, python=sys.executable, started_at=datetime.now().astimezone().isoformat()))
while True:
    observation = json.loads(local_step('observe', [TMP / 'observe_history_gradient_source_20260908.py']))
    with (OUTPUT / 'observations.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(observation) + '\n')
    epochs = [json.loads(line) for line in observation['source_log_tail'].splitlines()
              if line.startswith('{') and 'candidate_gradient_epoch' in line]
    announce('original_run_observed', pipeline_status=observation['pipeline_status'],
             completed_states=len(observation['completed_states']),
             last_epoch=epochs[-1]['epoch'] if epochs else None,
             original_processes=observation['processes'], free_bytes=observation['free_bytes'])
    if observation['pipeline_status'] != 'RUNNING':
        assert observation['pipeline_status'] == 'COMPLETE_VERIFIED_SOURCE_ONLY'
        break
    time.sleep(300)

exec((TMP / 'trifusion_ssh_session_recovery_20260908.py').read_text(encoding='utf-8'), globals())
remote_tool = '/root/autodl-tmp/trifusion-v2/transport/verify_history_gradient_all_statistics_20260908.py'
remote_output = RUN + '/source/all_statistics_postcheck.json'
remote_code = (
    'from pathlib import Path\nimport hashlib,json,subprocess\n'
    + 'tool=Path(' + repr(remote_tool) + ')\n'
    + 'assert hashlib.sha256(tool.read_bytes()).hexdigest()==' + repr(bindings['tools/verify_msvr_history_gradient_all_statistics.py']) + '\n'
    + 'command=' + repr([REMOTE_PYTHON, remote_tool, '--input-dir', RUN + '/source', '--config',
        REMOTE_REPO + '/configs/MSVR310/TriFusion-history-candidate-gradient-v1.json', '--protocol',
        REMOTE_REPO + '/protocols/msvr310_train_oof_v1.json', '--output', remote_output]) + '\n'
    + 'subprocess.run(command,check=True)\n'
    + 'p=Path(' + repr(remote_output) + ')\n'
    + 'print(json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))\n'
)
_, stdout, stderr = c.exec_command(REMOTE_PYTHON + ' -c ' + shlex.quote(remote_code))
raw, errors = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(OUTPUT / 'all_statistics.stdout.log').write_bytes(raw)
(OUTPUT / 'all_statistics.stderr.log').write_bytes(errors)
assert exit_code == 0, ('all_statistics', exit_code)
binding = json.loads(raw.splitlines()[-1])
sftp = c.open_sftp()
with sftp.open(remote_output, 'rb') as stream:
    result = stream.read()
sftp.close()
c.close()
assert len(result) == binding['bytes'] and hashlib.sha256(result).hexdigest() == binding['sha256']
(OUTPUT / 'all_statistics_postcheck.json').write_bytes(result)
save('all_statistics_remote_binding.json', binding)
announce('all_statistics', exit_code=exit_code)

local_step('intake', [TMP / 'receive_history_gradient_complete_source_20260908.py'])
local_step('analysis', [REPO / 'tools/analyze_msvr_history_gradient_text.py',
    '--run-dir', INTAKE / 'source', '--output-dir', ANALYSIS])
local_step('plot', [REPO / 'tools/plot_msvr_history_candidate_gradients.py',
    '--run-dir', INTAKE / 'source', '--analysis-dir', ANALYSIS, '--output-dir', FIGURES])
save('completion.json', dict(status='COMPLETE_SOURCE_POSTPROCESSING_PENDING_REVIEW',
    finished_at=datetime.now().astimezone().isoformat(), intake=str(INTAKE), analysis=str(ANALYSIS),
    figures=str(FIGURES), optimizer_updates=0, new_training_launched=False))
announce('postprocessing_complete_review_required')
