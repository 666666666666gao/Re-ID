import hashlib
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

repo = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
queue = [repo/'configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json',
         repo/'tools/build_msvr310_train_oof_protocol.py']
files, bindings = {}, []
def bind(parent, child, expected):
    p = Path(child)
    p = p if p.is_absolute() else repo/p
    bindings.append(dict(parent=str(parent), child=str(p), expected=expected))
    queue.append(p)

while queue:
    p = queue.pop(0)
    if str(p) in files:
        continue
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    row = dict(size=p.stat().st_size, sha256=h.hexdigest())
    files[str(p)] = row
    if p.suffix not in ('.py','.json','.md','.txt','.log','.yml'):
        continue
    row['text'] = p.read_text(encoding='utf-8')
    if p.suffix != '.json':
        continue
    value = json.loads(row['text'])
    if '/configs/' not in str(p):
        continue
    for key in ('project_file_sha256','project_source_file_sha256','fixed_file_sha256'):
        for child, expected in value.get(key, {}).items():
            bind(p, child, expected)
    for block in ('BASELINE','DATA','SOURCE_METADATA'):
        v = value.get(block, {})
        for key in ('CONFIG','SUMMARY','PROTOCOL','PATH'):
            if key in v:
                digest = v.get(key+'_SHA256',v.get('SHA256'))
                bind(p, v[key], digest)
    for key in ('base_config','protocol','clip_weight'):
        if key in value:
            bind(p,value[key],value[key+'_sha256'])
    if 'signal_source_file_sha256' in value:
        source = Path(value['signal_source'])
        for child, expected in value['signal_source_file_sha256'].items():
            bind(p,source/child,expected)
        row['signal_commit_actual'] = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
        row['signal_diff_sha256_actual'] = hashlib.sha256(subprocess.check_output(['git','-C',str(source),'diff','--binary'])).hexdigest()

baseline_path = repo/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json'
baseline_spec = json.loads(files[str(baseline_path)]['text'])
baseline = json.loads(files[baseline_spec['BASELINE']['SUMMARY']]['text'])
baseline_bindings = []
for fold in baseline['folds']:
    for p, expected in ((Path(fold['checkpoint']), fold['checkpoint_sha256']),
                        (Path(fold['checkpoint']).parent/'retrieval_arrays.pt',fold['retrieval']['retrieval_arrays_sha256'])):
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                h.update(chunk)
        baseline_bindings.append(dict(fold=fold['fold'],path=str(p),size=p.stat().st_size,
                                      expected=expected, actual=h.hexdigest()))
print(json.dumps(dict(collected_at=datetime.now(timezone.utc).isoformat(), files=files,
                     bindings=bindings, baseline_bindings=baseline_bindings), ensure_ascii=False))
