from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
proof = private / 'foundation_recipe_v1_20261002'
base = repo / 'refine-logs/incremental_role_objective_v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
previous = json.loads((proof / 'four_copy880_2025_pending.json').read_bytes())
old = json.loads((proof / 'publication880_local.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous['head']
assert not (proof / 'publication881_local.json').exists()
doc_name = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
doc = repo / doc_name
original_doc = subprocess.check_output(['git', 'show', 'HEAD:' + doc_name], cwd=repo)
assert hashlib.sha256(original_doc).hexdigest() == previous['doc_sha256']
assert doc.read_bytes().startswith(original_doc) and doc.read_text(encoding='utf-8').count('## 41.881 ') == 1
scope = base / 'PENDING_M0_SOURCE_SCOPE.json'
assert sha(scope) == old['source_scope_sha256']
sources = json.loads(scope.read_text())['source_sha256']
present = {name:digest for name,digest in sources.items() if (repo / name).is_file()}
changed = {name:digest for name,digest in present.items() if sha(repo / name) != digest}
assert len(sources) == 398 and len(sources) - len(present) == 107 and len(changed) == 27
assert all(hashlib.sha256((repo / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest for name,digest in changed.items())
actual = json.loads((private / 'independent_evidence_draft/incremental_pending_m0_complete881/REMOTE.json').read_bytes())
assert actual['source_count'] == 398 and actual['current45_inputs_unchanged'] and actual['fixed9_inputs_unchanged']
archive = repo / 'logs/incremental_pending_m0_complete881'
mapping = json.loads((archive / 'FILE_MAP.json').read_text())
assert all(sha(archive / name) == row['sha256'] for name,row in mapping.items())
incident = dict(status='PUBLICATION_PREPARATION_CONTINUATION_ONLY', at=datetime.now().astimezone().isoformat(),
    original_preparer='prepare_incremental_m0_closure881.py', original_exit_code=1,
    original_failure='Local raw-byte source assertion after owned archive/summary/doc append. Full398 runtime scope includes107 remote-only dependencies and27 CRLF local copies.',
    verified='All27 present differences are only CRLF;398 physical remote runtime source SHA already verified by terminal collector. The scope JSON itself is unchanged. No old runtime/model source rewritten.',
    continuation='Finish publication record from existing owned outputs without rerunning append/archive/training; remote publisher will assert all398 original physical source SHA again.',
    neural_updates=0, original_document_section_count=1)
incident_path = base / 'PUBLICATION_PREPARATION_CONTINUATION_20261007_881.json'
incident_path.write_text(json.dumps(incident, indent=2) + '\n', encoding='utf-8')
for source in (private / 'prepare_incremental_m0_closure881.py', private / 'finalize_incremental_m0_publication881.py'):
    dest = archive / f'{len(mapping):04d}_{source.name}'
    dest.write_bytes(source.read_bytes())
    mapping[dest.name] = dict(original_local_path=str(source), sha256=sha(dest))
(archive / 'FILE_MAP.json').write_text(json.dumps(mapping, indent=2) + '\n', encoding='utf-8')
owned = [doc_name, 'MANIFEST.md'] + ['refine-logs/incremental_role_objective_v1/' + name for name in (
    'M0_COMPLETE_MATRIX_20261007_881.json', 'M0_COMPLETE_MATRIX_20261007_881.csv',
    'NEXT_STEP_DECISION_20261007_881.md', 'EXPERIMENT_TRACKER_20261007_M0_COMPLETE881.md',
    'EXPERIMENT_TRACKER.md', 'PUBLICATION_PREPARATION_CONTINUATION_20261007_881.json')]
owned += [str(path.relative_to(repo)).replace('\\', '/') for path in archive.iterdir() if path.is_file()]
assert all(sha(repo / name) == digest for name,digest in old['protected_files'].items())
record = dict(previous_head=previous['head'], files=sorted(set(owned)), protected_files=old['protected_files'],
    source_count=398, source_scope_sha256=sha(scope), doc_sha256=sha(doc),
    original_manifest_sha256=hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:MANIFEST.md'], cwd=repo)).hexdigest(),
    remote_sparse_roots=['refine-logs/incremental_role_objective_v1', 'logs/incremental_pending_m0_complete881'],
    immutable_archive_roots=['logs/incremental_pending_m0_complete881'])
(proof / 'publication881_local.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
publisher = (private / 'publish_incremental_pending880_runtime.py').read_text(encoding='utf-8')
publisher = publisher.replace('four_copy879_2025_pending.json', '__PREVIOUS_PROOF__').replace('880', '881').replace('__PREVIOUS_PROOF__', 'four_copy880_2025_pending.json')
publisher = publisher.replace('Close fixed gradient comparison and register only pending M0 conditions', 'Close all six incremental-objective M0 gates and retain original failures')
publisher = publisher.replace('Fixed post-eight-state onebatch AMP scale1/256 comparison complete: bothsixQKnonzero,0updates,state restored,397sources/9inputs/current45 exact. Originalfirst-eightFAIL remains. FiveoriginalPENDINGM0 sourcesreviewed/prepared,398sources,NN NOT_RUN;same objective/gate/steps,no firstretry/no formalauto. Only26GPU0/1/no25/power-temp. GoalACTIVE_UNMET.',
    'Six conditions productionM0 PASS48updates but isolatedunscaledgate FAIL0/6; originalfirstfailure retained, fivepending executedonce/closed.398sources/current45/fixed9 exact,5engineeringprobes retired1786893304B. Formal0/ineligible; no retrieval claim. Initial-state diagnostic onlyproposed NOT_RUN. Only26GPU0/1/no25/power-temp. GoalACTIVE_UNMET.')
import ast
ast.parse(publisher)
target = private / 'publish_incremental_m0_complete881_runtime.py'
assert not target.exists()
target.write_text(publisher, encoding='utf-8')
print(json.dumps(dict(owned_files=len(record['files']), source_count=398, matrix='0/6', publication_prepared=True, publisher_executed=False)))
