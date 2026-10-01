"""Publish first-wave evidence while keeping every active runtime source unchanged."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OLD = 'a451711156319d7923b517761a40ad1cb0a771eb'
DOC = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
CAMPAIGN = ROOT / 'logs/role_global_tokens_20261001_v1'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--new', required=True)
parser.add_argument('--doc-sha', required=True)
parser.add_argument('--changed-count', type=int, required=True)
args = parser.parse_args()
proof = ROOT / 'logs/role_global_tokens_first_wave_sync688_20261001.json'
assert not proof.exists()
assert git('rev-parse', 'HEAD').decode().strip() == OLD
assert not git('status', '--porcelain', '--untracked-files=no').strip()
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 215
assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
launch = json.loads((ROOT / 'logs/role_global_tokens_launch_20261001_v1.json').read_text())
assert (Path('/proc') / str(launch['pid'])).exists()
assert (Path('/proc') / str(launch['pid']) / 'stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
snapshot = CAMPAIGN / 'first_wave_snapshot.json'
snapshot_sha = sha(snapshot)
git('fetch', str(ROOT / '.git/role_global_tokens_first_wave_sync688_20261001.incremental.bundle'), 'main')
assert git('rev-parse', 'FETCH_HEAD').decode().strip() == args.new
changed = git('diff', '--name-only', OLD, args.new).decode().splitlines()
assert len(changed) == args.changed_count
assert DOC in changed and all(path == DOC or path.startswith('logs/role_global_tokens_first_wave_688_20261001/') for path in changed)
expected = {path: git('show', args.new + ':' + path) for path in changed}
assert hashlib.sha256(expected[DOC]).hexdigest() == args.doc_sha
assert expected[DOC].startswith(git('show', OLD + ':' + DOC))
assert not (ROOT / 'logs/role_global_tokens_first_wave_688_20261001').exists()
git('merge', '--ff-only', args.new)
assert all((ROOT / path).read_bytes() == content for path, content in expected.items())
assert all(sha(ROOT / name) == digest for name, digest in manifest['source_sha256'].items())
assert sha(snapshot) == snapshot_sha
assert not git('status', '--porcelain', '--untracked-files=no').strip()
record = {'synced_at': datetime.now().astimezone().isoformat(), 'head': args.new,
          'previous_head': OLD, 'changed_blobs_verified': len(changed), 'doc_sha256': sha(ROOT / DOC),
          'active_source_count_verified': 215, 'first_wave_snapshot_sha256': snapshot_sha,
          'controller_pid': launch['pid'], 'controller_proc_present': (Path('/proc') / str(launch['pid'])).exists(),
          'tracked_worktree_clean': True, 'scope': 'Text first-wave publication only; no active training source or schedule edited.'}
proof.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))