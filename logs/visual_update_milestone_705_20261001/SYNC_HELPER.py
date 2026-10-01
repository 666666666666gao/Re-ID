"""Fast-forward dated milestone text without changing registered runtime sources."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OLD = '4cf144d2194ad6fd3b2a88bdb42c7b9ce7de3408'
DOC = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
PREFIX = 'logs/visual_update_milestone_705_20261001'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--new', required=True)
parser.add_argument('--doc-sha', required=True)
parser.add_argument('--changed-count', type=int, required=True)
args = parser.parse_args()
proof = ROOT / 'logs/visual_update_milestone_sync705_20261001.json'
backup = ROOT / '.codex_tmp/visual_update_milestone_sync705_20261001_originals'
assert not proof.exists() and not backup.exists()
assert git('rev-parse', 'HEAD').decode().strip() == OLD
assert not git('status', '--porcelain', '--untracked-files=no').strip()
manifest_path = ROOT / 'logs/visual_update_control_20261001_v1/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest_sha = sha(manifest_path)
assert manifest_sha == '947b028e835a6c722fac47f5e6333f7f02838ec9b1fbd2beda35ec1885543e5c'
assert len(manifest['source_sha256']) == 222
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
assert sha(Path(manifest['preflight_path'])) == manifest['preflight_sha256']
old_manifest_path = ROOT / 'logs/visual_start_roles_20261001_v1/manifest.json'
old_manifest = json.loads(old_manifest_path.read_text())
assert sha(old_manifest_path) == 'c749e3089a775c5781435fe14bfb7a1b75228f59004c9a31ec26d2a9409706dc'
assert len(old_manifest['source_sha256']) == 221
assert all(sha(ROOT / path) == digest for path, digest in old_manifest['source_sha256'].items())
report_path = ROOT / 'tools/report_visual_update_control_complete.py'
assert sha(report_path) == '4c21b1cc6212ef0202512ef46afc8fc714e3b2329fd68f701e4a7ac24acd4765'
git('fetch', str(ROOT / '.git/visual_update_milestone_sync705_20261001.incremental.bundle'), 'main')
assert git('rev-parse', 'FETCH_HEAD').decode().strip() == args.new
changed = git('diff', '--name-only', OLD, args.new).decode().splitlines()
assert len(changed) == args.changed_count
assert all(path in (DOC, 'MANIFEST.md') or path.startswith(PREFIX + '/')
           or path.startswith('refine-logs/visual_update_control_v1/EXPERIMENT_TRACKER') for path in changed)
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
assert sha(manifest_path) == manifest_sha
assert all(sha(ROOT / path) == digest for path, digest in manifest['source_sha256'].items())
assert sha(Path(manifest['initialization_witness_path'])) == manifest['initialization_witness_sha256']
assert sha(Path(manifest['preflight_path'])) == manifest['preflight_sha256']
assert sha(old_manifest_path) == 'c749e3089a775c5781435fe14bfb7a1b75228f59004c9a31ec26d2a9409706dc'
assert all(sha(ROOT / path) == digest for path, digest in old_manifest['source_sha256'].items())
assert sha(report_path) == '4c21b1cc6212ef0202512ef46afc8fc714e3b2329fd68f701e4a7ac24acd4765'
assert not git('status', '--porcelain', '--untracked-files=no').strip()
record = {'synced_at': datetime.now().astimezone().isoformat(), 'previous_head': OLD, 'head': args.new,
          'doc_sha256': sha(ROOT / DOC), 'changed_blobs_verified': len(changed),
          'active_source_count_verified': 222, 'active_manifest_sha256': manifest_sha,
          'original_source_count_verified': 221, 'owned_untracked_originals_preserved': moved,
          'tracked_worktree_clean': True,
          'scope': 'Dated first formal endpoints and running queue/observer text. Both source bindings and registered witness/preflight/report unchanged. No queue/observer/report replay or interim-score configuration changes.'}
proof.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
