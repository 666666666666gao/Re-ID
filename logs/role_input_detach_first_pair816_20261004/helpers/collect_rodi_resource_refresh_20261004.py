from datetime import datetime
from pathlib import Path
import hashlib
import json
import urllib.request

packet = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/rodi_resource_refresh_20261004')
assert not packet.exists()
packet.mkdir()

def read(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'TriFusion-primary-source-readonly'})
    return urllib.request.urlopen(request, timeout=30).read()

commit_url = 'https://api.github.com/repos/lsh-ahu/RoDI/commits/main'
commit_bytes = read(commit_url)
(packet / 'commit.json').write_bytes(commit_bytes)
commit = json.loads(commit_bytes)
head = commit['sha']
tree_url = 'https://api.github.com/repos/lsh-ahu/RoDI/git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1'
tree_bytes = read(tree_url)
(packet / 'tree.json').write_bytes(tree_bytes)
tree = json.loads(tree_bytes)
assert not tree['truncated']
paths = [item['path'] for item in tree['tree'] if item['type'] == 'blob']
readme_url = 'https://raw.githubusercontent.com/lsh-ahu/RoDI/' + head + '/README.md'
readme = read(readme_url)
(packet / 'README.md').write_bytes(readme)
record = {
    'status': 'COMPLETE_READONLY_PRIMARY_SOURCE_REFRESH',
    'at': datetime.now().astimezone().isoformat(),
    'repository': 'https://github.com/lsh-ahu/RoDI',
    'head': head,
    'tree': tree['sha'],
    'truncated': tree['truncated'],
    'files': paths,
    'python_sources': [path for path in paths if path.endswith('.py')],
    'sources': [commit_url, tree_url, readme_url],
    'sha256': {name: hashlib.sha256((packet / name).read_bytes()).hexdigest() for name in ('commit.json', 'tree.json', 'README.md')},
    'boundary': 'Public default-branch source tree and README only. No model execution, PDF rescoring, training configuration change or independent reproduction. Absence of published evaluation code does not prove protocol mismatch. Published scores remain author-reported references with existing resource boundaries.'
}
(packet / 'REFRESH.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record))
