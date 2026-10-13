"""Decode original completed transport locally, without another NN/SSH call."""
import base64,hashlib,json
from pathlib import Path
w=Path(__file__).resolve().parent
assert json.loads((w/'DIAGNOSIS_TRANSPORT_EXIT_948.json').read_bytes())['exit_code']==0
value=json.loads((w/'DIAGNOSIS_TRANSPORT_948.stdout').read_bytes())
assert value['exit_code']==0
destination=w/'diagnosis948';assert destination.is_dir() and not list(destination.iterdir())
for name,item in value['files'].items():
    path=destination/name;assert path.resolve().is_relative_to(destination.resolve()) and not path.exists()
    data=base64.b64decode(item['data']);assert hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
receipt=json.loads((destination/'logs/joint_spectral_diagnosis948/EXIT.json').read_bytes())
assert receipt['status']=='COMPLETE' and receipt['exit_code']==0
assert [row['aggregation'] for row in receipt['arms']]==['joint_delta','add_delta','joint_absolute']
assert all(row['exit_code']==0 and row['status']=='COMPLETE' for row in receipt['arms'])
assert not (w/'DIAGNOSIS_COLLECTED_948.json').exists()
(w/'DIAGNOSIS_COLLECTED_948.json').write_text(json.dumps(dict(exit_code=value['exit_code'],remote_free_bytes=value['remote_free_bytes'],
    local_collection_first_exit_code=1,local_collection_first_terminal='be4a05',
    first_failure='Local transport decoder omitted base64 import after all three remote diagnoses completed.',
    recovery='Decode original saved transport only; zero additional SSH/NN/update.',
    files={name:{k:v for k,v in item.items() if k!='data'} for name,item in value['files'].items()}),indent=2)+'\n')
for mode in ('joint_delta','add_delta','joint_absolute'):
    row=json.loads((destination/f'results/joint_spectral_diagnosis948/{mode}/DIAGNOSIS.json').read_bytes())
    assert row['status']=='COMPLETE' and row['optimizer_updates']==0
    assert row['model_state_before_sha256']==row['model_state_after_sha256']
    print(json.dumps(dict(mode=mode,epoch=row['selected_epoch'],distance_max_difference=row['fused_distance_max_abs_difference'],
        seconds=row['elapsed_seconds'],scores={name:value['metrics'] for name,value in row['scores'].items()},
        correction_norm_ratio=row['geometry']['query']['scaled_correction_global_norm_ratio']['mean'])))
