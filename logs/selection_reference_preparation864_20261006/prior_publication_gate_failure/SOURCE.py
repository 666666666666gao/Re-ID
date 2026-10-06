from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp')
source=(base/'publish_row_transport860.py').read_text(encoding='utf-8')
source=source.replace('publication860_local','publication863_local').replace('four_copy859','four_copy861')
source=source.replace('commit860','commit863').replace('target860','target863')
source=source.replace("four_copy860_2025_pending.json').write_text", "four_copy863_2025_pending.json').write_text")
source=source.replace('range(739,861)','range(739,862)')
source=source.replace("files=sorted(files);blobs_exact", "files.update(publication['files'])\nfiles=sorted(files);blobs_exact")
source=source.replace("section='41.860'", "section='41.863'")
source=source.replace('trifusion_row_transport860_20261006.bundle','trifusion_row_transport_closeout863_20261006.bundle')
source=source.replace('Register row-null role transport controls, preserve OT component failure and retire closed redundant weights',
                      'Close six row-transport experiments and fixed-best diagnosis; preserve failures and retire dominated weights')
source=source.replace('OT CPU failure retained; row-null CPU component PASS and fixed six-endpoint RAW study registered. No real M0/formal score yet. Seven actual closed weights retired, protected dependencies unchanged. 2025 no probe; Goal ACTIVE/UNMET.',
                      'Six complete50 first-strict endpoints, primary0/3 and all0/15. Fixed-best6/18/24 closed after preserved cfg failure; only missing100 resumed. Five qualified own weights retired, 201slot/RAW187 retained. Goal ACTIVE/UNMET; no new experiment launched; 2025 mirror pending.')
start=source.index("gate='''")
end=source.index("script(c,gate)",start)
gate="""gate='''from pathlib import Path
import hashlib,json
root=Path('/data/gaob/Re-ID/Trifusion')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert json.loads((root/'logs/row_mass_role_transport_v2_launch_20261006_861/EXIT.json').read_text())['exit_code']==0
assert json.loads((root/'logs/fixed_best_row_transport_resume_20261006_863/EXIT.json').read_text())['exit_code']==0
assert json.loads((root/'logs/fixed_best_row_transport_launch_20261006_862/EXIT.json').read_text())['exit_code']==1
summary=json.loads((root/'results/row_mass_role_transport_v2_complete_20261006_860/SUMMARY.json').read_text())
diagnosis=json.loads((root/'results/fixed_best_row_transport_diagnosis_20261006_862/SUMMARY.json').read_text())
assert summary['accepted']==6 and summary['formal_epochs']==300 and summary['formal_steps']==12968
assert diagnosis['fixed_models']==6 and diagnosis['deployment_modes']==18 and diagnosis['all_query_comparisons']==24
retired=json.loads((root/'logs/row_transport_closed_weight_retirement_20261006_863_r2/RETIREMENT.json').read_text())
assert retired['removed_count']==5
assert all(not (Path(r['target']['directory'])/'best_map.pth').exists() for r in retired['rows'])
assert sha(root/'trained-model/row_mass_role_transport_v2_20261006_860_full_semantic_RGBNT201/best_map.pth')==retired['retained_row_201_slot_sha256']
sources=json.loads((root/'refine-logs/row_mass_role_transport_v2/SOURCE_SCOPE.json').read_text())['source_sha256']
controls=json.loads((root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())['artifact_sha256']
assert len(sources)==366 and len(controls)==187
assert all(sha(root/n)==d for n,d in sources.items()) and all(sha(Path(n))==d for n,d in controls.items())
active=[]
for directory in Path('/proc').iterdir():
 if directory.name.isdigit() and (directory/'cmdline').is_file():
  cmd=(directory/'cmdline').read_bytes().replace(b'\\0',b' ').decode(errors='replace')
  if '/data/gaob/Re-ID/Trifusion/tools/' in cmd:active.append(dict(pid=directory.name,command=cmd))
assert not active,active
print('ALL_OWN_PRODUCERS_TERMINAL_366_AND_187_UNCHANGED_QUALIFIED_RETIREMENT_VERIFIED')
'''
"""
source=source[:start]+gate+source[end:]
compile(source,'publish_row_transport_closeout863','exec')
exec(compile(source,'publish_row_transport_closeout863','exec'))
