from pathlib import Path
import json
import paramiko

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865_v2')
assert not packet.exists();packet.mkdir()
source=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/selection_storage865/SOURCE.py').read_text()
start=source.index('names=');end=source.index('rows=[]',start)
source=source[:start]+'''targets=[
 'clean_clip_joint_20261002_v1_clean_clip_global_only_MSVR310_seed42_full',
 'clean_clip_joint_20261002_v1_clean_clip_roles_MSVR310_seed42_full',
 'clean_clip_joint_20261002_v1_clean_clip_global_only_RGBNT100_seed42_full',
 'clean_clip_joint_20261002_v1_clean_clip_roles_RGBNT100_seed42_full',
 'semantic_native_evidence_20261002_v1_clean_clip_semantic_RGBNT100_seed42_full',
 'semantic_native_evidence_20261002_v1_clean_clip_semantic_RGBNT201_seed42_full']
winners={'MSVR310':'metric_feature_scale_20261003_v1_full_metric_raw_MSVR310',
 'RGBNT100':'native_research_v6_20261003_794_full_global_only_RGBNT100',
 'RGBNT201':'region_reconstruction_v1_20261005_858_full_native_RGBNT201'}
names=targets+list(winners.values())
'''+source[end:]
start=source.index('winner=rows[-1]')
source=source[:start]+'''by_name={Path(r['directory']).name:r for r in rows}
qualified=[]
for name in targets:
 target=by_name[name]
 dataset=next(d for d in winners if d in name)
 winner=by_name[winners[dataset]]
 assert all(winner['metrics'][k]>=v for k,v in target['metrics'].items()) and winner['metrics']['mAP']>target['metrics']['mAP']
 assert not any(n.startswith(target['directory']+'/') for n in protected)
 qualified.append(dict(target=target,winner=winner))
print(json.dumps(dict(status='SIX_CLOSED_WEIGHTS_QUALIFIED_NO_DELETION',at=__import__('datetime').datetime.now().astimezone().isoformat(),free_bytes=shutil.disk_usage(root).free,gpu_memory_only=gpu,qualified=qualified,sources=366,protected=187)))
'''
compile(source,'qualification865_v2','exec');(packet/'SOURCE.py').write_text(source)
c=paramiko.SSHClient();c.load_host_keys('C:/Users/gb/.ssh/known_hosts')
c.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
i,o,e=c.exec_command('/usr/bin/python3 -B -');i.write(source);i.channel.shutdown_write();o.channel.settimeout(300)
data,error=o.read(),e.read();status=o.channel.recv_exit_status();c.close()
(packet/'QUALIFICATION.json').write_bytes(data);(packet/'STDERR.txt').write_bytes(error)
assert status==0,error.decode()
d=json.loads(data);print(json.dumps(dict(status=d['status'],free_bytes=d['free_bytes'],qualified=len(d['qualified']),bytes=sum(r['target']['bytes'] for r in d['qualified']))))
