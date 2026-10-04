from pathlib import Path
import hashlib
import json
import paramiko

target = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/role_input_detach_complete_report')
assert not target.exists()
code = '''
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/role_input_detach_v1_20261004_813'
launch=root/'logs/role_input_detach_launch_20261004_813'
report_root=root/'results/role_input_detach_v1_complete_20261004_813'
state=json.loads((campaign/'campaign.json').read_text())
exit_record=json.loads((launch/'EXIT.json').read_text())
assert exit_record['exit_code']==0
assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==0
assert len(state['jobs'])==12 and all(job['status']=='COMPLETE' and job['exit_code']==0 for job in state['jobs'])
manifest=json.loads((campaign/'manifest.json').read_text())
matrix=json.loads((campaign/'accepted_matrix.json').read_text())
report=json.loads((report_root/'SUMMARY.json').read_text())
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()
assert len(manifest['source_sha256'])==322 and all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())
assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==6
assert report['status']=='COMPLETE' and report['accepted']==6 and report['formal_epochs']==300
assert len(report['rows'])==6 and len(report['pairs'])==12 and report['formal_steps']==12968
controls_path=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
controls=json.loads(controls_path.read_text())
assert len(controls['rows'])==9 and len(controls['artifact_sha256'])==61
assert all(sha(Path(name))==digest for name,digest in controls['artifact_sha256'].items())
assert matrix['schema']==report['schema']=='trifusion-role-input-detach-v1'
accepted={(row['dataset'],row['variant']):row for row in matrix['rows']}
assert set(accepted)=={(dataset,variant) for dataset in ('RGBNT201','MSVR310','RGBNT100') for variant in ('semantic','native')}
artifacts={}
orders={}
for row in report['rows']:
    key=(row['dataset'],row['variant'])
    assert all(row[name]==value for name,value in accepted[key].items())
    out=Path(row['run_dir'])
    receipt=json.loads((out/'official_metrics.json').read_text())
    training=json.loads((out/'training.json').read_text())
    assert receipt['status']=='COMPLETE' and receipt['independent_upstream_metrics_equal'] and not receipt['reranking']
    assert receipt['metrics']==row['metrics'] and receipt['selected_epoch']==row['best_epoch']
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(training['history'])==50
    assert [entry['epoch'] for entry in training['history']]==list(range(1,51))
    assert training['history'][row['best_epoch']-1]['official_fused']==row['metrics']
    assert [path.name for path in out.glob('*.pth')]==['best_map.pth']
    bound={}
    for name,field in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('best_epoch_distances.pt','training_best_distance_sha256')):
        path=out/name
        actual=sha(path)
        assert actual==receipt[field]
        bound[name]={'sha256':actual,'bytes':path.stat().st_size}
    assert sha(out/'official_metrics.json')==row['receipt_sha256']
    m0=Path(row['m0_dir'])
    m0_receipt=json.loads((m0/'training.json').read_text())
    assert m0_receipt['status']=='M0_PASS' and m0_receipt['m0']['reload_max_abs_difference']==0
    assert m0_receipt['production_m0_diagnostics']['effective_optimizer_updates']==8
    probe=m0/'m0_reload_probe.pth'
    actual=sha(probe)
    assert actual==m0_receipt['m0']['reload_probe_sha256']
    bound['m0_reload_probe.pth']={'sha256':actual,'bytes':probe.stat().st_size,'path':str(probe)}
    data=(out/'training_batch_order.jsonl').read_bytes()
    assert len(data.splitlines())==row['formal_steps']
    orders.setdefault(row['dataset'],{})[row['variant']]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'lines':len(data.splitlines())}
    original=next(item for item in controls['rows'] if (item['dataset'],item['variant'])==key)
    assert data==(Path(original['run_dir'])/'training_batch_order.jsonl').read_bytes()
    assert {k:v for k,v in row['initializer'].items() if k not in ('architecture','entry_sha256','role_input_gradient_policy')}=={k:v for k,v in original['initializer'].items() if k not in ('architecture','entry_sha256')}

    artifacts['/'.join(key)]=bound
common=('public_clip_sha256','protocol_sha256','visual_initial_sha256','camera_initial_sha256','shared_initializer_sha256','batch_size','num_instances','seed','foundation_option')
for dataset,bindings in orders.items():
    assert len(bindings)==2 and len({entry['sha256'] for entry in bindings.values()})==1
    selected=[row for row in matrix['rows'] if row['dataset']==dataset]
    assert all(selected[0]['initializer'][name]==row['initializer'][name] for row in selected for name in common)
for pair in report['pairs']:
    assert pair['actual_training_batch_order_equal']
    diagnosis=pair['paired_diagnosis']
    assert diagnosis['status']=='CPU_ARRAY_PARITY_AND_PAIRED_DIAGNOSIS_COMPLETE'
    assert len(diagnosis['query_changes'])==sum(identity['queries'] for identity in diagnosis['identity_changes'])
files={}
for path in (campaign/'manifest.json',campaign/'campaign.json',campaign/'accepted_matrix.json',campaign/'report.log',
             launch/'EXIT.json',report_root/'SUMMARY.json',report_root/'REPORT.md'):
    files[str(path.relative_to(root))]={'sha256':sha(path),'bytes':path.stat().st_size}
print(json.dumps({'at':datetime.now().astimezone().isoformat(),'status':'ORIGINAL_ONCE_ONLY_FINAL_CPU_REPORT_AND_ARTIFACTS_VERIFIED',
    'campaign_status':state['status'],'report_invocations':state['report_invocations'],'report_exit_code':state['report_exit_code'],
    'report_completed_at':state['report_completed_at'],'launch_exit':exit_record,'formal_completed':6,
    'formal_epochs':300,'formal_steps':12968,'source_files_verified_unchanged':322,'each_full_weight_count':1,
    'all_actual_batch_orders_equal':True,'original61control_artifacts_unchanged':True,'original_control_seal_sha256':sha(controls_path),'batch_orders':orders,'artifacts':artifacts,'m0_probe_dependencies_verified_before_cleanup':True,
    'controller_pid':state['controller_pid'],'controller_present':(Path('/proc')/str(state['controller_pid'])).exists(),
    'disk_free_bytes':shutil.disk_usage(root).free,'files':files,
    'boundary':'Text collection and SHA checks only; no torch import, model construction/inference or replay of original report. Only26GPU0/1,25textonly,no power/temp. CPU report is original automatic single invocation after all6 full50 endpoints, not a retraining or new inference. Goal remains unmet unless strongbaseline/SOTA, mechanism necessity and fullpipeline stability are independently established. M0 probes retained until this evidence is received and cleanup explicitly audited.'}))
'''
compile(code, '<collect-final-report>', 'exec')
target.mkdir()
client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
stdin, stdout, stderr = client.exec_command('/usr/bin/python3 -B -')
stdin.write(code)
stdin.channel.shutdown_write()
stdout.channel.settimeout(60)
data, error = stdout.read(), stderr.read()
exit_code = stdout.channel.recv_exit_status()
(target / 'stdout.json').write_bytes(data)
(target / 'stderr.txt').write_bytes(error)
(target / 'EXIT.json').write_text(json.dumps({'exit_code':exit_code})+'\n', encoding='utf-8')
assert exit_code == 0, error.decode()
record = json.loads(data)
texts = target / 'texts'
texts.mkdir()
sftp = client.open_sftp()
for name, info in record['files'].items():
    path = Path(name)
    destination = texts / (path.parent.name + '_' + path.name)
    sftp.get('/data/gaob/Re-ID/Trifusion/' + name, str(destination))
    assert destination.stat().st_size == info['bytes'] and hashlib.sha256(destination.read_bytes()).hexdigest() == info['sha256']
sftp.close()
client.close()
print(json.dumps({key:value for key,value in record.items() if key not in ('artifacts','files')}, indent=2))
