"""Read-only Q1 audit intake. No project imports, model forwards, or file writes."""
from pathlib import Path
import hashlib
import json
import subprocess
import datetime

ROOT = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN = Path('/root/trifusion-storage/artifacts/msvr310_cross_scene_smooth_ap_v1_seed42_d35864d')

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(Path(p).read_bytes())

configs = {}
path = ROOT/'configs/MSVR310/TriFusion-cross-scene-smooth-ap-paired-v1.json'
while True:
    data = read(path)
    configs[str(path)] = data
    follow = data.get('previous_config') or data.get('coordinate_config') or data.get('memory_config') or data.get('base_config')
    if not follow:
        break
    path = ROOT/follow
base = data
signal_path = ROOT/base['BASELINE']['CONFIG']
signal = read(signal_path)
configs[str(signal_path)] = signal
protocol_path = ROOT/signal['protocol']
protocol = read(protocol_path)
meta_path = ROOT/base['SOURCE_METADATA']['PATH']
summary = read(RUN/'q1/summary.json')
inventory = []
for p in sorted(RUN.rglob('*')):
    if p.is_file():
        inventory.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
out = dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
           configs=configs, protocol_path=str(protocol_path), metadata_path=str(meta_path),
           baseline_summary_path=base['BASELINE']['SUMMARY'], pipeline=read(RUN/'pipeline.json'),
           q1_summary_keys=list(summary), inventory=inventory,
           protocol_keys=list(protocol), protocol_record_example=protocol['records'][0],
           folds=[dict(fold=f['fold'],keys=list(f), source_ids=len(f['source_ids']),
                       heldout_ids=len(f['heldout_ids']), source_records=len(f['source_record_indices']),
                       gallery_records=len(f['gallery_record_indices']),query_rows=len(f['query_rows']))
                  for f in protocol['folds']],
           metadata_keys=list(read(meta_path)),
           model_forwards=0,optimizer_updates=0,official_test_reads=0)
print(json.dumps(out,ensure_ascii=False))
