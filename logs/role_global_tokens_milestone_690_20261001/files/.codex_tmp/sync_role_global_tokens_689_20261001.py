"""Fast-forward one milestone publication without editing active runtime bindings."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OLD = '62a7f36ea6aaa3d7cd725232ea73f20887e75f27'
DOC = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
CAMPAIGN = ROOT / 'logs/role_global_tokens_20261001_v1'
PREFIX = 'logs/role_global_tokens_milestone_689_20261001'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--new', required=True)
parser.add_argument('--doc-sha', required=True)
parser.add_argument('--changed-count', type=int, required=True)
args = parser.parse_args()
proof = ROOT / 'logs/role_global_tokens_milestone_sync689_20261001.json'
backup = ROOT / '.codex_tmp/role_global_tokens_milestone_sync689_20261001_originals'
assert not proof.exists() and not backup.exists()
assert git('rev-parse', 'HEAD').decode().strip() == OLD
assert not git('status', '--porcelain', '--untracked-files=no').strip()
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 215
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
snapshot_path = CAMPAIGN / 'milestone_0814_snapshot.json'
snapshot_sha = sha(snapshot_path)
git('fetch', str(ROOT / '.git/role_global_tokens_milestone_sync689_20261001.incremental.bundle'), 'main')
assert git('rev-parse', 'FETCH_HEAD').decode().strip() == args.new
changed = git('diff', '--name-only', OLD, args.new).decode().splitlines()
assert len(changed) == args.changed_count
allowed = {DOC, 'tools/report_role_global_tokens_complete.py',
           'refine-logs/role_global_tokens_v1/ANALYSIS_PREPARATION_20261001.json',
           'logs/role_global_tokens_first_wave_sync688_20261001.json',
           'logs/role_global_tokens_first_wave_sync688_failure_20261001.json'}
assert all(path in allowed or path.startswith(PREFIX + '/') for path in changed)
expected = {path: git('show', args.new + ':' + path) for path in changed}
assert hashlib.sha256(expected[DOC]).hexdigest() == args.doc_sha
assert expected[DOC].startswith(git('show', OLD + ':' + DOC))
git('sparse-checkout', 'add', PREFIX)
tracked = set(git('ls-files').decode().splitlines())
moved = []
for path in changed:
    target = ROOT / path
    assert target.resolve().is_relative_to(ROOT.resolve())
    if path not in tracked and target.exists():
        assert target.read_bytes() == expected[path], path
        stored = backup / path
        assert stored.resolve().is_relative_to(backup.resolve())
        stored.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(stored))
        moved.append(path)
git('merge', '--ff-only', args.new)
assert all((ROOT / path).read_bytes() == content for path, content in expected.items())
assert all((backup / path).read_bytes() == (ROOT / path).read_bytes() for path in moved)
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
assert sha(snapshot_path) == snapshot_sha
assert not git('status', '--porcelain', '--untracked-files=no').strip()
record = {'synced_at': datetime.now().astimezone().isoformat(), 'previous_head': OLD, 'head': args.new,
          'doc_sha256': sha(ROOT / DOC), 'changed_blobs_verified': len(changed),
          'active_source_count_verified': 215, 'snapshot_sha256': snapshot_sha,
          'owned_untracked_originals_preserved': moved, 'tracked_worktree_clean': True,
          'scope': 'Text milestone and CPU report tool only; no active training source, dataset, loss or schedule edited.'}
proof.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))