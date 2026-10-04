
from datetime import datetime
from pathlib import Path
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected={'/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/accepted_matrix.json': 'a258accd4c4ab32d411af9da638655a85364a541a87df17b00e0a10b23c9e897', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/manifest.json': 'c749e3089a775c5781435fe14bfb7a1b75228f59004c9a31ec26d2a9409706dc', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/campaign.json': '219eb6c4617ab5100e2ea73171153c42058404bf48dae3b1efed1ed644eb335e', '/data/gaob/Re-ID/Trifusion/refine-logs/visual_start_roles_v1/INITIALIZATION_WITNESS.json': 'ccfe359b92ba76dd74f5fe25c1d79a864cd4b3905bb64b65c606d688da561a7a', '/data/gaob/Re-ID/Trifusion/pertrained-model/visual_start_reset_20261001/INPUTS.json': '6fb9d6a1876a67447ad0b126f154397b6af9fd48e1c3b885a2d13de102cd347a', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100/campaign.json': '660b8c00751332d94ce1771f98405f0e2064da2b3d7b3334f3f8debbe887ea40', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_full/training.json': '2eb3feccc92ec2fef04afe71bfcc12bcea28f1ca10670e8d2a0aba0ea98c8bd8', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_full/training_steps.jsonl': '61348d7405d1be536284c19562ba5e4da521bcb0d8bd49edcbbddf5824aee108', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_full/official_metrics.json': 'f1689cc099a942dbef6d7936bee9651e434c68198d253af8c7c6541686c744ba', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_m0/training.json': 'd5e823def4d6e2c44ae37159247ec4c5831c8602c1c246019dedb62185a26f2e', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_m0/training_steps.jsonl': 'dd233e0e1a353bf947af71dfd5518338e2fd4e53e6c2223f93e91d147525bf1b', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201/campaign.json': 'e5181e18227741dd9d3d485a64a76f19a0d9513015d9966a804ae982061ba2ba', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_full/training.json': 'ec284d70c3981f8645f781c2d0401529cee17f7bacd62d9be8e690b52def41f9', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_full/training_steps.jsonl': 'b6543d64c4996c19ffe079dc42042666639c162a1ac4433b2beeda5ab3a45bd2', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_full/official_metrics.json': '39dd2fc250fed19bc745dc3bf697b84c3c1d95d869443a19f81a1be10e334f3e', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_m0/training.json': 'b1ed2b291eda5710544d259c1790e148fc9736a4f29627cd9e62ee26e6b32621', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_m0/training_steps.jsonl': '465a5b7513ef44a08310175b4fee9b3fa67d9b4681c75cbb8e3e0e463cc0c098', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310/campaign.json': '9aa77f68b5394611bf4aa775b3ca2f5d00f3cdfb48f9c4e2590d73d0a598b47d', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_full/training.json': '52e3386f71421950cd36951696f8f3c1ac2da9dc2e712c44267e9f166b50f542', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_full/training_steps.jsonl': '8f3c5589032a7e7344ea6c9292bb4ae9efbd9dbda94a29cabe83955d1d2b8231', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_full/official_metrics.json': 'b9dd6b475f7a2f73b4633df0ceceb784a459a188a95cf51108b4f53a0b1bd2c0', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_m0/training.json': 'dea48584579a4c0d4839adade784e9e0b3b7004a9ed5ead8b7bc7676ef29956b', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_m0/training_steps.jsonl': '4746489d335a65ddc3d0a965f8120bd22abc8d146d2db4abe11039264e8ace30', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100/campaign.json': '97fd54d46b4e95b8c42048ec4daad116a4342a9f97973b9eb9388dc4dcbebb6e', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_full/training.json': '5c1ca8283259d14060fbe7a499b8d32bbce311b47181d26c12cf0486fec345d4', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_full/training_steps.jsonl': '4944c9654bdd15a435f3ae14dfff8f4e569cfa2bc2d8c1cefdac1fc325d8c4c1', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_full/official_metrics.json': '96cceb8db1ef445102611525890df44dd2a4ef130ce30443d190713be311bb3b', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_m0/training.json': '6ee8b002a3b9a71526b9b447ebf4efd02a2b0e94580a53b7f0f0201f7559af0b', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_m0/training_steps.jsonl': '7f097180478c6d4cd01c3322142297f501ac9336b367a5cc462761b3a8e1f337', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201/campaign.json': 'cef1c2edd95314596f6425c8d27974beaa15eecbba8296ebaf9012a0e42f058b', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_full/training.json': 'fb85670d32e5e261ec07aa410e48f91162c61ccdce4bbb954be0454f2b6f6e4d', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_full/training_steps.jsonl': '9477e3ca9c2b42c43557c757e556845d998fe14073273935603cff5815a5083d', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_full/official_metrics.json': '6048ed8709c0da0facd800f92f1b46e0d655a4849819d08dcf73a108d19ffc7e', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_m0/training.json': '38b191c87a56ea05af1ea7574f56b61203c90e251b8def643a148c17d0241631', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_m0/training_steps.jsonl': 'efb2b4241b1db013f40d51a1376ceb5dd3b4952f9b67993804dc1d2aa100dc3a', '/data/gaob/Re-ID/Trifusion/logs/visual_start_roles_20261001_v1/visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310/campaign.json': '5729715979d69dd25e493803cf8fc6f2db1f8ef4177ce1b3f35f07969b974504', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_full/training.json': '10a54e13c4916948123b8eab42faa2031585fc32ee7b051222cfc72c31bf9323', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_full/training_steps.jsonl': 'ecdd88ad129de638c00818a8125885e8fa7d1e63d0bc82aab250c04193cf2679', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_full/official_metrics.json': '6bf8ab311ef8d3a952cc0e7615264d702d5757b63d6d53c4210034ce88f0ddfc', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_m0/training.json': '6592466719c36b74f8372f2eb2636722c0651c9b899a70bd6836b43b9c77d93f', '/data/gaob/Re-ID/Trifusion/trained-model/visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_m0/training_steps.jsonl': '0abd41aefb8adb49ed2db844f755458920e78a184c8705304f4c9d89a3bc7fad'}
targets=[{'path': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_m0\\m0_reload_probe.pth', 'sha256': 'ceea28c10d5e87302a04feb74025a2af29756a0537052fcb63dcc54f4e8d2421', 'receipt': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_m0\\training.json', 'full_dir': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT100_seed42_full', 'checkpoint_sha256': '721d37fe3023b8789146bf6481f13209d99eda66fa2855d1ff63a71683256ab1', 'distance_sha256': '9f525c7cfd033fe44abdc8f2518c9bf19fdf0ea9a6686f71bcf04d26bbb53ecd', 'receipt_sha256': 'f1689cc099a942dbef6d7936bee9651e434c68198d253af8c7c6541686c744ba'}, {'path': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_m0\\m0_reload_probe.pth', 'sha256': 'fe84f2333e654d7e7e42e42fd112a9fb1e16cf91dbf49da53b1cf187889cb2fa', 'receipt': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_m0\\training.json', 'full_dir': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_RGBNT201_seed42_full', 'checkpoint_sha256': 'd6296303ce5860750464399fcf45565efaaf6547b39068ccf6b174c497d782ad', 'distance_sha256': '89ea925472d213b072ae8ac1cc09c29e201b47971d1436179b4d653013ee619e', 'receipt_sha256': '39dd2fc250fed19bc745dc3bf697b84c3c1d95d869443a19f81a1be10e334f3e'}, {'path': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_m0\\m0_reload_probe.pth', 'sha256': '3d239fb1ac379543fa51f18cb620f0c9a9185ffc6cfce47d70606b224576dfb6', 'receipt': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_m0\\training.json', 'full_dir': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_public_visual_MSVR310_seed42_full', 'checkpoint_sha256': '852865ea374a02843d3040039aaa28e712269fb81d82a6fddd4c03183c7b8448', 'distance_sha256': 'b3416c7238e8ce57ba391a363938186a3d2d98969a6396b794321afdc5b5beb0', 'receipt_sha256': 'b9dd6b475f7a2f73b4633df0ceceb784a459a188a95cf51108b4f53a0b1bd2c0'}, {'path': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_m0\\m0_reload_probe.pth', 'sha256': '984d7f152bfd99efd41632a87223c150f7d15adf1377df7c23ba72d61c711a1f', 'receipt': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_m0\\training.json', 'full_dir': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT100_seed42_full', 'checkpoint_sha256': '5be14e04dc2e9f953df08567650432c8ea36ea91c350e64f114df04897dd5c73', 'distance_sha256': '3c51f52ceeb43e98e24feec2e869101267a2e14c261bc79ccd96d16fb4598ac1', 'receipt_sha256': '96cceb8db1ef445102611525890df44dd2a4ef130ce30443d190713be311bb3b'}, {'path': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_m0\\m0_reload_probe.pth', 'sha256': 'ffaa1f38fbc9ccf544ac74ef8bad7284d93d443a8939f41ee18ebac5b76f6eb2', 'receipt': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_m0\\training.json', 'full_dir': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_RGBNT201_seed42_full', 'checkpoint_sha256': 'e860ac57ffb49faeb65236592a06b344eca65780d83710b1871bfd500da2827b', 'distance_sha256': '129a8b8819ba6e77511a91ea16507f87e2cc0f6dbb30f86e3aab3a8b210798a2', 'receipt_sha256': '6048ed8709c0da0facd800f92f1b46e0d655a4849819d08dcf73a108d19ffc7e'}, {'path': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_m0\\m0_reload_probe.pth', 'sha256': '2084c2b9381aeb803a6f3107874b9b4102568319200663438feb88a6974a64ad', 'receipt': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_m0\\training.json', 'full_dir': '\\data\\gaob\\Re-ID\\Trifusion\\trained-model\\visual_start_roles_20261001_v1_visual_start_reid_visual_MSVR310_seed42_full', 'checkpoint_sha256': '4d5324bde8918b9c29de56ebeeeef49e68fe2e1b3bf1946532a1e7a6a761b101', 'distance_sha256': '7fc7150fa5457fb53a8e72f2b52f1c9d9967b8debf6c9e3a639e11af6f80d0ed', 'receipt_sha256': '6bf8ab311ef8d3a952cc0e7615264d702d5757b63d6d53c4210034ce88f0ddfc'}]
assert len(targets)==6
report=root/'results/visual_start_roles_complete_20261001/SUMMARY.json'
assert sha(report)=='214ea13646e218bb3a1f771ba5bebd9255c6065f05b99cda37d37680f2563567'
assert all(sha(Path(name))==digest for name,digest in expected.items())
old=root/'logs/visual_start_roles_20261001_v1'
state=json.loads((old/'campaign.json').read_text())
analysis=json.loads((old/'analysis_waiter_status.json').read_text())
assert state['status']=='COMPLETE' and len(state['jobs'])==6
assert all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert analysis['status']=='CPU_REPORT_COMPLETE' and analysis['exit_code']==0
assert analysis['complete_parent_jobs']==analysis['expected_jobs']==6

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
  p=root/'trained-model'/('global_task_role_v1_20261004_824_m0_'+j['variant']+'_'+j['dataset'])/'probe_checkpoint.pth'
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
 assert p.name=='m0_reload_probe.pth' and p.parent.name.startswith('visual_start_roles_20261001_v1_visual_start_') and p.parent.name.endswith('_m0')
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
journal=root/'logs/visual_start_m0_retirement832_20261004'
assert not journal.exists()
journal.mkdir()
free_before=shutil.disk_usage(root).free
plan=dict(status='CLOSED_SIX_ENDPOINT_M0_PROBES_VERIFIED_BEFORE_RETIREMENT',
 at=datetime.now().astimezone().isoformat(),targets=retiring,formal_artifacts=best,
 current_required_probes=probes,original_report_sha256=sha(report),
 protected_control_seal_sha256={str(p):sha(p) for p in (control,historical)},
 source_count=330,active_training_pid=3215669,active_training_start_ticks=38149139,
 boundary='Only six exact successful M0 reload probes; full50, firststrict and CPU report completed 2026-10-01. Formal best/receipts and current six required probes retained. No model forward, original verifier/report replay or power/temperature action.')
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
result=dict(status='SIX_CLOSED_VISUAL_START_M0_PROBES_RETIRED',completed_at=datetime.now().astimezone().isoformat(),
 retired_files=6,retired_bytes=sum(t['bytes'] for t in retiring),disk_free_before=free_before,
 disk_free_after=shutil.disk_usage(root).free,formal_best_and_receipts_unchanged=True,
 current_six_probes_unchanged=True,current_scientific330_sources_unchanged=True,
 current_native_pid_still_same=True,
 boundary='Old probe binaries no longer available for direct M0-replay. Original successful receipts, formal best and full reports remain. Disk free delta includes concurrent training writes; retired_bytes is exact file-byte sum.')
(journal/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(result=result,plan=plan,journal={p.name:p.read_text() for p in journal.iterdir()})))
