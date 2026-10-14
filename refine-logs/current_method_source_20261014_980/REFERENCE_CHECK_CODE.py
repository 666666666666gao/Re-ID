import json
from pathlib import Path
import subprocess

w=Path(__file__).resolve().parent
assert not (w/'MATCHED201_REFERENCE980.json').exists()
code=r'''
import hashlib,json
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for path in (root/'trained-model').glob('*bestmap_e50*/RGBNT201_SIGNAL_V8_seed42/official_metrics.json'):
    receipt=json.loads(path.read_text())
    if receipt['status']=='COMPLETE' and receipt['training_epochs']==50:
        train_path=path.parent/'training.json';training=json.loads(train_path.read_text())
        assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and training['training']['epochs']==50 and training['seed']==42
        history_path=path.parent/'epoch_official_metrics.jsonl';epochs=[json.loads(line) for line in history_path.read_text().splitlines()]
        assert [row['epoch'] for row in epochs]==list(range(1,51))
        best=max(epochs,key=lambda row:(row['metrics']['mAP'],row['epoch']))
        assert best['epoch']==training['selected_epoch']==receipt['selected_epoch']
        metrics=receipt['outputs']['fused']['metrics']
        assert all(abs(value-best['metrics'][key])<1e-5 for key,value in metrics.items())
        roles=Path(training['checkpoint']);assert sha(roles)==receipt['role_checkpoint_sha256']
        rows.append(dict(path=str(path),receipt_sha256=sha(path),training_sha256=sha(train_path),history_sha256=sha(history_path),
            selected_epoch=receipt['selected_epoch'],metrics=metrics,initializer=training['initializer'],
            protocol_sha256=receipt['protocol_sha256'],role_checkpoint_sha256=receipt['role_checkpoint_sha256'],role_checkpoint_bytes=roles.stat().st_size))
assert len(rows)==1,rows
print(json.dumps(dict(status='VERIFIED_CLOSED_MATCHED_201_SIGNAL_V8_FULL50_REFERENCE',rows=rows,nn_calls=0,gpu_queries=0,running_training_queries=0,
    boundary='Cold hashes and full saved50 history of the existing control only. No new training or retrieval. Shared initializer/budget is not bitwise paired optimization or isolation of one loss.')))
'''
run=subprocess.run(['ssh','-i','C:/Users/gb/.ssh/id_ed25519','-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-p','2026','gaob@172.19.12.138','/usr/bin/python3','-'],input=code.encode(),capture_output=True)
(w/'MATCHED201_REFERENCE980.stdout').write_bytes(run.stdout);(w/'MATCHED201_REFERENCE980.stderr').write_bytes(run.stderr)
assert run.returncode==0,run.stderr.decode()
value=json.loads(run.stdout);(w/'MATCHED201_REFERENCE980.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
print(json.dumps(value))
