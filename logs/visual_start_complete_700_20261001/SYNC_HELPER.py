"""Fast-forward milestone evidence while preserving all active runtime sources."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OLD = '05ffe0f1fe6d8b27b6826c95ea91cd80b079c95c'
DOC = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
CAMPAIGN = ROOT / 'logs/visual_start_roles_20261001_v1'
PREFIX = 'logs/visual_start_complete_700_20261001'

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--new', required=True)
parser.add_argument('--doc-sha', required=True)
parser.add_argument('--changed-count', type=int, required=True)
args = parser.parse_args()
proof = ROOT / 'logs/visual_start_complete_sync700_20261001.json'
backup = ROOT / '.codex_tmp/visual_start_complete_sync700_20261001_originals'
assert not proof.exists() and not backup.exists()
assert git('rev-parse', 'HEAD').decode().strip() == OLD
assert not git('status', '--porcelain', '--untracked-files=no').strip()
manifest_path = CAMPAIGN / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
assert len(manifest['source_sha256']) == 221
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
manifest_sha = sha(manifest_path)
git('fetch', str(ROOT / '.git/visual_start_complete_sync700_20261001.incremental.bundle'), 'main')
assert git('rev-parse', 'FETCH_HEAD').decode().strip() == args.new
changed = git('diff', '--name-only', OLD, args.new).decode().splitlines()
assert len(changed) == args.changed_count
assert all(path in (DOC, 'MANIFEST.md', 'tools/preflight_visual_update_control.py', 'tools/check_visual_update_initialization.py', 'tools/collect_visual_update_control.py', 'tools/queue_visual_update_control.py') or path.startswith(PREFIX + '/') or path.startswith('refine-logs/visual_update_control_v1/') for path in changed)
expected = {path: git('show', args.new + ':' + path) for path in changed}
assert hashlib.sha256(expected[DOC]).hexdigest() == args.doc_sha
assert expected[DOC].startswith(git('show', OLD + ':' + DOC))
assert expected['MANIFEST.md'].startswith(git('show', OLD + ':MANIFEST.md'))
git('sparse-checkout', 'add', PREFIX, 'refine-logs/visual_update_control_v1')
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
assert sha(manifest_path) == manifest_sha
assert not git('status', '--porcelain', '--untracked-files=no').strip()
record = {'synced_at': datetime.now().astimezone().isoformat(), 'previous_head': OLD, 'head': args.new,
          'doc_sha256': sha(ROOT / DOC), 'changed_blobs_verified': len(changed),
          'active_source_count_verified': 221, 'manifest_sha256': manifest_sha,
          'owned_untracked_originals_preserved': moved, 'tracked_worktree_clean': True,
          'scope': 'Full6 negative results, actual final audit/source review and reviewed queue helpers deployed. All221 original runtime bindings and manifest unchanged. New preflight/models/formal training remain unexecuted at publication. Original once-only report untouched.'}
proof.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
