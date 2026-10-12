"""Retire only the four closed900/901 MSVR M0 probes, after exact verification."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path('/data/gaob/Re-ID/Trifusion')
OUT = ROOT/'logs/text_semantic_prior_resume901/FAILED_MSVR_PROBE_RETIREMENT_902'
ROWS = [
 (900,'pretrained','f140e60e49e586e4ae98c61e9cdddf191e9d4f996262e0cf47ecb99ce7954c21','afcd1f27cc3b3e883bea1479fa571018fefa703f82098c0695014222645e1b91'),
 (900,'random','2602025991a9a765c1cb32f9274b22dc15078a5476c19edbbbc691d82340c273','2ee4ed46078e6275649bedf8fa6e802054c34d9ad30f82784039cae8eea38e8c'),
 (901,'pretrained','fb1dea7d3e7b851b4b840fffad04eb2bbdd84bef4ec1853c5c9339f55a67d884','b831eef2aa36cc8ad2cf5f2e542c913ecdbe35ec0e996284f663458723d715a7'),
 (901,'random','7592897978d10a063c27826855771a2a41fa5a96562f27dc1f95c58a33e71db9','0275d422cf96715de63702ed11424fd9d389f7120d6636b17beb439c07bf734e'),
]

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

assert not OUT.exists()
state=json.loads((ROOT/'logs/text_semantic_prior_v1_20261010_901/campaign.json').read_text())
assert state['status']=='COMPLETE_WITH_MISSING' and state['report_invocations']==1 and state['report_exit_code']==0
assert all(job['status'] in ('COMPLETE','FAILED') for job in state['jobs'])
processes=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
    command=path.read_bytes().replace(b'\0',b' ').decode(errors='replace') if path.exists() else ''
    if ('text_semantic_prior_v1_20261010_900' in command or 'text_semantic_prior_v1_20261010_901' in command) and 'tools/' in command:
        processes.append(dict(pid=int(path.parent.name),command=command))
assert processes==[], processes
seal_path=ROOT/'refine-logs/incremental_role_objective_v1/INPUT_SEAL.json'
seal=json.loads(seal_path.read_text())
matrix=json.loads((ROOT/'logs/text_semantic_prior_v1_20261010_901/accepted_matrix.json').read_text())
manifest=json.loads((ROOT/'logs/text_semantic_prior_v1_20261010_901/manifest.json').read_text())
assert sha(seal_path)==manifest['control_seal_sha256'] and len(seal['artifact_sha256'])==45
protected=dict(seal['artifact_sha256'])
for row in matrix['rows']:
    run=Path(row['run_dir'])
    metric=json.loads((run/'official_metrics.json').read_text())
    protected.update({str(run/'best_map.pth'):row['checkpoint_sha256'],
        str(run/'official_distances.pt'):row['distance_sha256'],
        str(run/'own_global_distances.pt'):row['own_global_distance_sha256'],
        str(run/'best_epoch_distances.pt'):metric['training_best_distance_sha256']})
for name,digest in protected.items(): assert sha(Path(name))==digest, name
verified=[]
for campaign,package,probe_pin,training_pin in ROWS:
    run=ROOT/f'trained-model/text_semantic_prior_v1_20261010_{campaign}_m0_{package}_MSVR310'
    probe=(run/'m0_reload_probe.pth').resolve(strict=True)
    assert probe.is_relative_to((ROOT/'trained-model').resolve()) and str(probe) not in protected
    assert sha(probe)==probe_pin and sha(run/'training.json')==training_pin
    training=json.loads((run/'training.json').read_text())
    assert training['status']=='M0_PASS' and len(training['history'])==1 and training['history'][0]['steps']==8
    verified.append(dict(path=str(probe),bytes=probe.stat().st_size,sha256=probe_pin,
                         training_path=str(run/'training.json'),training_sha256=training_pin))
OUT.mkdir()
before=shutil.disk_usage(ROOT).free
qualification=dict(qualified_at=datetime.now().astimezone().isoformat(),rows=verified,
    protected_artifact_sha256=protected,free_before_bytes=before,
    review_report_sha256='fa14f91de08fb7e86379527114f3174003ffa390d29c434ec6d8cb2cb2eb12e7',
    review_scope='Closed source/text auditor; no probe binary consumer.',processes=processes)
(OUT/'QUALIFIED.json').write_text(json.dumps(qualification,indent=2)+'\n')
with (OUT/'DELETIONS.jsonl').open('x') as stream:
    for row in verified:
        Path(row['path']).unlink()
        stream.write(json.dumps(dict(row,retired_at=datetime.now().astimezone().isoformat()))+'\n')
        stream.flush()
for name,digest in protected.items(): assert sha(Path(name))==digest, name
assert all(not Path(row['path']).exists() and sha(Path(row['training_path']))==row['training_sha256'] for row in verified)
result=dict(status='COMPLETE',completed_at=datetime.now().astimezone().isoformat(),
    retired_files=len(verified),retired_bytes=sum(row['bytes'] for row in verified),
    free_before_bytes=before,free_after_bytes=shutil.disk_usage(ROOT).free,
    protected_files_verified_before_and_after=len(protected),rows=verified,
    boundary='Exact four failed-acceptance M0 probes only. Formal best, all45 current controls and selected distance files unchanged. Original metadata/failure logs retained; direct binary replay of these four probes retired. No model, evaluation, source synchronization, other deletion or power/temperature action.')
(OUT/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:value for key,value in result.items() if key!='rows'}))
