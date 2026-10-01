"""Fast-forward one milestone publication without editing active runtime bindings."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OLD = '4ffeb990656e77f995c8b08eaa54860e8fb5888a'
DOC = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
PREFIX = 'logs/visual_start_launch_696_20261001'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--new', required=True)
parser.add_argument('--doc-sha', required=True)
parser.add_argument('--changed-count', type=int, required=True)
args = parser.parse_args()
proof = ROOT / 'logs/visual_start_launch_sync696_20261001.json'
backup = ROOT / '.codex_tmp/visual_start_launch_sync696_20261001_originals'
assert not proof.exists() and not backup.exists()
assert git('rev-parse', 'HEAD').decode().strip() == OLD
assert not git('status', '--porcelain', '--untracked-files=no').strip()
manifest = json.loads((CAMPAIGN / 'manifest.json').read_text())
assert len(manifest['source_sha256']) == 221
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
snapshot_path = CAMPAIGN / 'manifest.json'
snapshot_sha = sha(snapshot_path)
git('fetch', str(ROOT / '.git/visual_start_launch_sync696_20261001.incremental.bundle'), 'main')
assert git('rev-parse', 'FETCH_HEAD').decode().strip() == args.new
changed = git('diff', '--name-only', OLD, args.new).decode().splitlines()
assert len(changed) == args.changed_count
allowed = {DOC}
allowed.add('tools/report_visual_start_roles_complete.py')
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
          'active_source_count_verified': 221, 'manifest_sha256': snapshot_sha,
          'owned_untracked_originals_preserved': moved, 'tracked_worktree_clean': True,
          'scope': 'Real six-end launch/M0 evidence and prepared completion-only CPU source; all221 active runtime bindings and manifest unchanged. Four full50 jobs remain live; no restart.'}
proof.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))