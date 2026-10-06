from pathlib import Path
import json
base=Path('C:/Users/gb/.codex_tmp')
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
archive=repo/'logs/signal_selection_execution_failure866_20261006'
(archive/'LOCAL_PREPARATION_FAILURE.json').write_text(json.dumps(dict(stage='local_legacy_source_check',published=False,remote_modified=False,error='Local legacy sparse/CRLF files are not the canonical remote366 source bytes. All366 frozen canonical intake bytes match; no source normalization repair.'),indent=2)+'\n')
source=(base/'prepare_selection_reference866.py').read_text()
tail=source[source.index('for name in set(sources)-set(frozen):'):]
header='''from pathlib import Path
import hashlib,json,subprocess
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');base=Path('C:/Users/gb/.codex_tmp')
proof=base/'foundation_recipe_v1_20261002';packet=base/'independent_evidence_draft/signal_selection_reference_v1'
previous=json.loads((proof/'four_copy865_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
assert not (proof/'publication866_local.json').exists()
target=repo/'refine-logs/signal_selection_reference_v1'
archive=repo/'logs/signal_selection_execution_failure866_20261006'
at=json.loads((archive/'MINIMAL_CHANGE_CHECK.json').read_bytes())['at']
sources=json.loads((target/'SOURCE_SCOPE.json').read_bytes())['source_sha256']
frozen=json.loads((repo/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_bytes())['source_sha256']
canonical=base/'rt_complete862/received'
assert len(frozen)==366 and all(hashlib.sha256((canonical/n).read_bytes()).hexdigest()==d for n,d in frozen.items())
'''
result=header+tail
compile(result,'actual_preparation866_continuation','exec')
destination=base/'prepare_selection_reference866_continuation.py';assert not destination.exists();destination.write_text(result)
(archive/'create_selection_preparation866_continuation.py').write_bytes(Path(__file__).read_bytes())
(archive/'prepare_selection_reference866_continuation.py').write_bytes(destination.read_bytes())
print('Continuation only after known local sparse-byte check; old partial actions not repeated')
