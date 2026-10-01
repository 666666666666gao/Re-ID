"""Materialize the actually excluded first-wave evidence directory after completed fast-forward."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OLD = 'a451711156319d7923b517761a40ad1cb0a771eb'
NEW = '62a7f36ea6aaa3d7cd725232ea73f20887e75f27'
DOC = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
CAMPAIGN = ROOT / 'logs/role_global_tokens_20261001_v1'
PROOF = ROOT / 'logs/role_global_tokens_first_wave_sync688_20261001.json'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert git('rev-parse', 'HEAD').decode().strip() == NEW
assert not git('status', '--porcelain', '--untracked-files=no').strip()
assert not PROOF.exists()
assert not (ROOT / 'logs/role_global_tokens_first_wave_688_20261001/INTAKE_MANIFEST.json').exists()
failed_source = ROOT / '.codex_tmp/sync_role_global_tokens_688_20261001.py'
failure = {'observed_at': datetime.now().astimezone().isoformat(), 'head_after_fast_forward': NEW,
           'failed_source_sha256': sha(failed_source), 'actual_executor_exit_code': 1,
           'failure': 'FileNotFoundError: excluded logs/role_global_tokens_first_wave_688_20261001/INTAKE_MANIFEST.json after git merge completed',
           'scope': 'First-wave publication materialization only; no training retry or runtime modification.'}
failure_path = ROOT / 'logs/role_global_tokens_first_wave_sync688_failure_20261001.json'
assert not failure_path.exists()
failure_path.write_text(json.dumps(failure, indent=2) + '\n')
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 215
assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
snapshot_sha = sha(CAMPAIGN / 'first_wave_snapshot.json')
previous_sparse = git('sparse-checkout', 'list').decode().splitlines()
git('sparse-checkout', 'add', 'logs/role_global_tokens_first_wave_688_20261001')
changed = git('diff', '--name-only', OLD, NEW).decode().splitlines()
assert len(changed) == 40
assert all((ROOT / path).read_bytes() == git('show', NEW + ':' + path) for path in changed)
assert sha(ROOT / DOC) == '9fcbf109b1216618cdb29c470b52e02ded1628411e8c7a88380963ccfdd90714'
assert (ROOT / DOC).read_bytes().startswith(git('show', OLD + ':' + DOC))
assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
assert sha(CAMPAIGN / 'first_wave_snapshot.json') == snapshot_sha
assert not git('status', '--porcelain', '--untracked-files=no').strip()
launch = json.loads((ROOT / 'logs/role_global_tokens_launch_20261001_v1.json').read_text())
controller = Path('/proc') / str(launch['pid'])
assert controller.exists() and (controller / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
record = {'synced_at': datetime.now().astimezone().isoformat(), 'head': NEW, 'previous_head': OLD,
          'changed_blobs_verified': 40, 'doc_sha256': sha(ROOT / DOC),
          'active_source_count_verified': 215, 'first_wave_snapshot_sha256': snapshot_sha,
          'controller_pid': launch['pid'], 'controller_proc_live': True, 'tracked_worktree_clean': True,
          'prior_materialization_failure': failure, 'previous_sparse_directories': previous_sparse,
          'added_sparse_directory': 'logs/role_global_tokens_first_wave_688_20261001',
          'scope': 'Materialized missing text evidence, all 40 blobs verified; no runtime source or training schedule edited.'}
PROOF.write_text(json.dumps(record, indent=2) + '\n')
observer = json.loads((ROOT / 'logs/role_global_tokens_observer_0814_20261001.json').read_text())
observer_proc = Path('/proc') / str(observer['pid'])
assert observer_proc.exists() and (observer_proc / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
print(json.dumps({'sync': record, 'observer': observer, 'observer_proc_live': True}))