
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT201_seed42_m0/training.json': '59fea7ae8e62211dabd740de0c8e482155e2d8575c5014885b1dc9282eedd4cc', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT201_seed42_full/training.json': 'd752a9d6c1946961b84206b86e297cac6b305687c8ff05f372a77de3b8e6deda', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT201_seed42_full/official_metrics.json': 'dfe69e96b28b47e654a4f801f0453cae9a1f6b6335cd1c6b2cdb6d1859ec7de2', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_reset_roles_RGBNT201/campaign.json': '24a0f0e2404fbe27b02ba189829a40b7ba08c7eed2c9b84e35d0ae44691d9266', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT100_seed42_m0/training.json': '891fd295b0830dc4fb8f9e58b7d84964995236e0cafa453fa5276793318d5888', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT100_seed42_full/training.json': '04751862a3f5d37ce9352c93e20de2091471288e0a9bd9223e4fadcb8ebaf5c8', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT100_seed42_full/official_metrics.json': '377ab6fdcaeb65c49bdd685de00c115237599b38c5e8efa4a8d55b3dba14cc58', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_reset_roles_RGBNT100/campaign.json': 'd2b7d3c6497ec14320dd64845a4fcc42097d0fcd84e7a06c91788b5e05a0155b', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_MSVR310_seed42_m0/training.json': '665c86c5b11158e83cdf731150b86308376e4954e89ebe49c40cf7e94d09dfe7', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_MSVR310_seed42_full/training.json': 'c0e99381375383bec0a3e16bba828a8e57b53fb68e9c0ba49fafc005ff6715fa', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_MSVR310_seed42_full/official_metrics.json': 'f36f9a2eef520079a87d51a5d3319eec799c5692a152a2996ee06999f7e53bef', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_reset_roles_MSVR310/campaign.json': 'aab7648c86562687f5f1181816c3387399b798aab5d694793ee65ee5b42af451', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT201_seed42_m0/training.json': '7921b0911ca14130afc5b8a58d4ba54d1115202779c385179e96862cc69db210', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT201_seed42_full/training.json': 'a9eb2e21f2e2cf70237f3f662436eff2db37ca035d33441afd90fcc8d31c2c3f', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT201_seed42_full/official_metrics.json': '99e011b492d371bcb314e5f8cab07308e1d09ea88efe29b2d45a2c6b8f882348', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_carry_roles_RGBNT201/campaign.json': 'a25448d29eca96e5b2a443f59df1dd7584dbd82af130ed1a8db48a843700efea', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT100_seed42_m0/training.json': 'ae5cfec72b60fc450cb31112e40e4240980c73c3dbae6b39b80a19a72d987615', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT100_seed42_full/training.json': '773c864b412d4a3b6368485c6e7527d87c56acf9ffdba99f7137785eab196d88', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT100_seed42_full/official_metrics.json': 'a0f7142ff4afd2c9d756eef47127ff372d1f2910675a87a18761b94dbd4a29db', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_carry_roles_RGBNT100/campaign.json': '493a5ef900d74b1d57d1e4862aa8db128052af43b8b83a220d59f5311f9c74bd', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_MSVR310_seed42_m0/training.json': '21c1c4d95c17884493eb7eb8ca5ff15d7849e9a0b048a356cd2b29651aaee399', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_MSVR310_seed42_full/training.json': 'ff63e57b82e0ca635edc7dc23e2edd281daac236a5cd20bc51b312d95f1f62fa', '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_MSVR310_seed42_full/official_metrics.json': '9be442198b84ebe484f8e00dd3ea2da63f8f20fab7c5ece54bab0a1c0ab63bb2', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_carry_roles_MSVR310/campaign.json': 'ac79d3649e14171f12e73d1a2afdbfa681302c7eb7748ba824c233e5e1beee5d', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/campaign.json': '979be2aa7e080c34c5067bb59c9b52e3ef43edf127219bb06180243d61deb417', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/accepted_matrix.json': '1405d73124e7a1776de521edefb14620457f7b6e0691787bb18e2f2a8e0af7f9', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/paired_RGBNT201.json': 'de81023ee03556cbf39e94c6c2ad2fc0d34d6bf9cf40cc6b8df094ad63afb1e6', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/paired_RGBNT100.json': '0cd53a3895742cd83e0eee87ebace3e9b68baefa9966d5b0c79aaa89f7151c39', '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/paired_MSVR310.json': 'a30d49d71beb43dd39469b3e6da897dff5b27d1f20defe59f03d97a3bbd901aa', '/data/gaob/Re-ID/Trifusion/results/PROMPT_ROLE_STATE_FULL_2026-09-30.md': '729d4ddbc7e54ba56ae81b1eb12da85a4d7f761942b4538c34685525696686dc'}
targets=[{'path': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT201_seed42_m0/m0_reload_probe.pth', 'sha256': '87829c8c977c989012969c08f72c8c6a18f52f6fe645a455caf5293894fb6ce8', 'receipt': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT201_seed42_m0/training.json', 'full_dir': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT201_seed42_full', 'campaign_dir': '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_reset_roles_RGBNT201', 'checkpoint_sha256': '0da20026517a7f9d0dec294e47b8551dd19d5d426b7abccd781b324c9b0150b2', 'distance_sha256': '6a68a2ae3aa699b636c8e4e342ce8b7d5b0cf8349d6b970b2b9e0bb610b36b4a', 'receipt_sha256': 'dfe69e96b28b47e654a4f801f0453cae9a1f6b6335cd1c6b2cdb6d1859ec7de2'}, {'path': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT100_seed42_m0/m0_reload_probe.pth', 'sha256': 'dde206e8166934f048fb0affbb471979c00b2860e0c620f8becbc093d7c9f53c', 'receipt': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT100_seed42_m0/training.json', 'full_dir': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_RGBNT100_seed42_full', 'campaign_dir': '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_reset_roles_RGBNT100', 'checkpoint_sha256': '22c078384d560e1d362694fb33f6332e27d7937f28c2166758e0c7e1e38c58d3', 'distance_sha256': '6d9450eb02f6276eb644322672ea0a502fb8e463f105f30975ceaef130e53eb3', 'receipt_sha256': '377ab6fdcaeb65c49bdd685de00c115237599b38c5e8efa4a8d55b3dba14cc58'}, {'path': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_MSVR310_seed42_m0/m0_reload_probe.pth', 'sha256': '90ead27a518cf120b8f93c4d732b5d36a6b38d95a5284ef10d785cd25d7df2f0', 'receipt': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_MSVR310_seed42_m0/training.json', 'full_dir': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_reset_roles_MSVR310_seed42_full', 'campaign_dir': '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_reset_roles_MSVR310', 'checkpoint_sha256': '3ac5ad6b6ffbaede1f7d54f04853e617ea7b162a4abee7b9f69faf398a211483', 'distance_sha256': '1058b1c39b47a63cbc656d1e9a9e230ccef4c3e6488e62215dd03ec9d9323080', 'receipt_sha256': 'f36f9a2eef520079a87d51a5d3319eec799c5692a152a2996ee06999f7e53bef'}, {'path': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT201_seed42_m0/m0_reload_probe.pth', 'sha256': 'b3d3f49ab48abdd43c5568abd3dadbd0aa681d39fb98392726f58029e27a9cf1', 'receipt': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT201_seed42_m0/training.json', 'full_dir': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT201_seed42_full', 'campaign_dir': '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_carry_roles_RGBNT201', 'checkpoint_sha256': '3d282b9d208c5c409c2eb8e05b44a5a0caa9567595daee3d4d28e0de05f286a1', 'distance_sha256': '1d31d5d772b6980920fa0075dda1be05535309735b0fd72be40699320ca3f806', 'receipt_sha256': '99e011b492d371bcb314e5f8cab07308e1d09ea88efe29b2d45a2c6b8f882348'}, {'path': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT100_seed42_m0/m0_reload_probe.pth', 'sha256': 'c24289bca7bebdb68673dafb6f0fd2bb9e8d7fe69222e6f37d3b78b86bf7f2e8', 'receipt': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT100_seed42_m0/training.json', 'full_dir': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_RGBNT100_seed42_full', 'campaign_dir': '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_carry_roles_RGBNT100', 'checkpoint_sha256': '1e2360adf8c5ef4ffbab04097ff0eb97dbe0408f41cdb2c7a018a9af0333779a', 'distance_sha256': 'b3f2821e20e17f469cc839e0499d039c37ea97bba8e5fcb39c0a25e13bb9b3c1', 'receipt_sha256': 'a0f7142ff4afd2c9d756eef47127ff372d1f2910675a87a18761b94dbd4a29db'}, {'path': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_MSVR310_seed42_m0/m0_reload_probe.pth', 'sha256': '8d83c690e0ebf4accd420058013087619a9ec36dfdcebf4c0b197f67c09c7b5d', 'receipt': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_MSVR310_seed42_m0/training.json', 'full_dir': '/data/gaob/Re-ID/Trifusion/trained-model/prompt_role_state_20260930_prompt_carry_roles_MSVR310_seed42_full', 'campaign_dir': '/data/gaob/Re-ID/Trifusion/logs/prompt_role_state_20260930/prompt_role_state_20260930_prompt_carry_roles_MSVR310', 'checkpoint_sha256': '34377c4a950706004c6180c95ed80664daa62899b5c3919d7b39c74dced074ed', 'distance_sha256': '74ea255d5c9521ae84d1ebc9cce3bed7df36a6c1127566e2cc8df887999935f4', 'receipt_sha256': '9be442198b84ebe484f8e00dd3ea2da63f8f20fab7c5ece54bab0a1c0ab63bb2'}]
assert len(targets)==6
report=root/'logs/prompt_role_state_20260930/accepted_matrix.json'
assert sha(report)=='1405d73124e7a1776de521edefb14620457f7b6e0691787bb18e2f2a8e0af7f9'
assert all(sha(Path(name))==digest for name,digest in expected.items())
old=root/'logs/prompt_role_state_20260930'
state=json.loads((old/'campaign.json').read_text())
matrix=json.loads(report.read_text())
assert state['status']=='COMPLETE' and len(state['jobs'])==6
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert matrix['verified_complete']==matrix['expected_endpoints']==6
assert len(matrix['rows'])==6 and all(r['status']=='VERIFIED_COMPLETE' for r in matrix['rows'])
for target in targets:
 child=json.loads((Path(target['campaign_dir'])/'campaign.json').read_text())
 assert child['status']=='COMPLETE' and len(child['jobs'])==3
 assert [j['mode'] for j in child['jobs']]==['m0','train','evaluate']
 assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in child['jobs'])

current=root/'logs/global_task_role_v1_20261004_824'
manifest=json.loads((current/'manifest.json').read_text())
assert len(manifest['source_sha256'])==330
assert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())
control=root/'refine-logs/role_input_detach_fixed_best_diagnosis_v1/INPUT_SEAL.json'
historical=root/'refine-logs/native_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(control)==manifest['control_seal_sha256']
assert sha(historical)==manifest['historical_control_seal_sha256']
protected=set(manifest['initialization_sha256'])
for p in (control,historical):
 protected.update(json.loads(p.read_text())['artifact_sha256'])
assert not protected.intersection(t['path'] for t in targets)
current_state=json.loads((current/'campaign.json').read_text())
probes={}
for j in current_state['jobs']:
 if j['phase']=='m0':
  assert j['status']=='COMPLETE' and j['exit_code']==0
  p=root/'trained-model'/('global_task_role_v1_20261004_824_m0_'+j['variant']+'_'+j['dataset'])/'m0_reload_probe.pth'
  digest=j['result']['m0']['reload_probe_sha256']
  assert sha(p)==digest
  probes[str(p)]=digest
assert len(probes)==6
process=Path('/proc/3215669/stat').read_text()
assert int(process[process.rfind(')')+2:].split()[19])==38149139
command=Path('/proc/3215669/cmdline').read_bytes().replace(b'\0',b' ')
assert b'run_global_task_role.py' in command and b'RGBNT100' in command and b'native' in command

retiring=[]
best={}
for t in targets:
 p=Path(t['path'])
 assert p.resolve().is_relative_to((root/'trained-model').resolve()) and p.resolve()==p
 assert p.name=='m0_reload_probe.pth' and p.parent.name.startswith('prompt_role_state_20260930_prompt_') and p.parent.name.endswith('_m0')
 assert sha(p)==t['sha256']
 receipt=json.loads(Path(t['receipt']).read_text())
 assert receipt['status']=='M0_PASS' and receipt['m0']['reload_probe_sha256']==t['sha256']
 full=Path(t['full_dir'])
 assert [p.name for p in full.glob('*.pth')]==['best_map.pth']
 training=json.loads((full/'training.json').read_text())
 assert [r['epoch'] for r in training['history']]==list(range(1,51))
 official=json.loads((full/'official_metrics.json').read_text())
 assert official['status']=='COMPLETE' and official['training_epochs']==50
 for name,key in (('best_map.pth','checkpoint_sha256'),('official_distances.pt','distance_sha256'),('official_metrics.json','receipt_sha256')):
  actual=sha(full/name)
  assert actual==t[key]
  best[str(full/name)]=actual
 retiring.append(dict(t,bytes=p.stat().st_size))
journal=root/'logs/prompt_m0_retirement834_20261005'
assert not journal.exists()
journal.mkdir()
free_before=shutil.disk_usage(root).free
plan=dict(status='CLOSED_SIX_ENDPOINT_M0_PROBES_VERIFIED_BEFORE_RETIREMENT',
 at=datetime.now().astimezone().isoformat(),targets=retiring,formal_artifacts=best,
 current_required_probes=probes,original_report_sha256=sha(report),
 protected_control_seal_sha256={str(p):sha(p) for p in (control,historical)},
 source_count=330,active_training_pid=3215669,active_training_start_ticks=38149139,
 boundary='Only six exact successful M0 reload probes; full50, firststrict and complete-gallery verification completed 2026-09-30. Formal best/receipts and current six required probes retained. No model forward, original verifier/report replay or power/temperature action.')
(journal/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
for t in retiring:
 p=Path(t['path'])
 assert sha(p)==t['sha256']
 p.unlink()
 assert not p.exists()
 with (journal/'RETIREMENT.jsonl').open('a') as stream:
  stream.write(json.dumps(dict(path=t['path'],bytes=t['bytes'],sha256=t['sha256'],retired_at=datetime.now().astimezone().isoformat()))+'\n')
assert all(sha(Path(name))==digest for name,digest in best.items())
assert all(sha(Path(name))==digest for name,digest in probes.items())
assert all(sha(Path(name))==digest for name,digest in expected.items())
assert all(sha(root/name)==digest for name,digest in manifest['source_sha256'].items())
assert sha(control)==manifest['control_seal_sha256'] and sha(historical)==manifest['historical_control_seal_sha256']
process_after=Path('/proc/3215669/stat').read_text()
assert int(process_after[process_after.rfind(')')+2:].split()[19])==38149139
result=dict(status='SIX_CLOSED_PROMPT_M0_PROBES_RETIRED',completed_at=datetime.now().astimezone().isoformat(),
 retired_files=6,retired_bytes=sum(t['bytes'] for t in retiring),disk_free_before=free_before,
 disk_free_after=shutil.disk_usage(root).free,formal_best_and_receipts_unchanged=True,
 current_six_probes_unchanged=True,current_scientific330_sources_unchanged=True,
 current_native_pid_still_same=True,
 boundary='Old probe binaries no longer available for direct M0-replay. Original successful receipts, formal best and full reports remain. Disk free delta includes concurrent training writes; retired_bytes is exact file-byte sum.')
(journal/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(result=result,plan=plan,journal={p.name:p.read_text() for p in journal.iterdir()})))
