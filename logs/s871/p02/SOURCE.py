from pathlib import Path
from datetime import datetime
import json,hashlib,subprocess,os,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_signal_selection_reference as panel
campaign=root/'logs/signal_selection_reference_v1_20261006_870';launch=root/'logs/signal_selection_reference_launch_20261006_870';journal=root/'logs/selection_report_completion_20261006_871';output=root/'results/signal_selection_reference_complete_20261006_871'
assert not journal.exists() and not output.exists() and not (root/'results/signal_selection_reference_complete_20261006_870').exists()
state=json.loads((campaign/'campaign.json').read_text());assert state['status']=='COMPLETE' and state['report_invocations']==1 and state['report_exit_code']==1
assert len(state['jobs'])==18 and all(j['status']=='COMPLETE' and j['exit_code']==0 for j in state['jobs'])
assert json.loads((launch/'EXIT.json').read_text())['exit_code']==1
for n in ('LAUNCH.json','CHILD.json'):
 r=json.loads((launch/n).read_text());p=Path('/proc')/str(r['pid']);assert not p.exists() or int((p/'stat').read_text().split()[21])!=r['start_ticks']
log=(campaign/'report.log').read_text();assert 'line 23, in main' in log and "assert state['status']=='COMPLETE' and state['report_invocations']==1" in log
matrix=json.loads((campaign/'accepted_matrix.json').read_text());assert matrix['accepted']==matrix['expected']==len(matrix['rows'])==9 and sum(r['formal_steps'] for r in matrix['rows'])==19452
sources=panel.require_sources();panel.require_protected()
assert panel.sha(root/'tools/report_signal_selection_reference.py')=='fd7fdbae1271289b67c65892af7ece17da8153d5723c19b103b93b2e737ee74a'
originals={str(p):panel.sha(p) for p in (campaign/'campaign.json',campaign/'accepted_matrix.json',campaign/'report.log',launch/'EXIT.json')}
command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/report_signal_selection_reference.py'),'--campaign',str(campaign),'--origin-campaigns',str(campaign/'endpoint_origins.json'),'--batch-metadata-paths',str(campaign/'batch_metadata_paths.json'),'--output-dir',str(output)]
qualification=dict(status='REGISTERED_CPU_ONLY_REPORT_CONTINUATION',at=datetime.now().astimezone().isoformat(),original_failure='Controller updated state to COMPLETE/report_invocations1 in memory but did not write it before spawning reader. After report gate failed, final controller write persisted it. No new full-query analysis occurred before line23 failure. Existing completed state now satisfies the unchanged report gate.',accepted=9,formal_epochs=450,formal_steps=19452,original_immutable_sha256=originals,report_source_sha256=panel.sha(root/'tools/report_signal_selection_reference.py'),source_count=len(sources),command=command,boundary='One new invocation of unchanged CPU report over nine existing saved-distance inputs. No model/source/recipe/gate/tolerance/state mutation and no NN/evaluator replay. Oldreport attempt1/EXIT1 remain sealed; this distinct continuation is not retroactive PASS or trainingseed/SOTA evidence.')
journal.mkdir();(journal/'QUALIFIED.json').write_text(json.dumps(qualification,indent=2)+'\n')
supervisor='from pathlib import Path\nfrom datetime import datetime\nimport subprocess,json,os\nfolder=Path('+repr(str(journal))+')\ncommand='+repr(command)+'\nwith (folder/"stdout.txt").open("x") as out,(folder/"stderr.txt").open("x") as err:\n p=subprocess.Popen(command,cwd='+repr(str(root))+',env=dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1"),stdout=out,stderr=err)\n (folder/"CHILD.json").write_text(json.dumps(dict(pid=p.pid,start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().split()[21]),started_at=datetime.now().astimezone().isoformat()))+"\\n")\n code=p.wait()\n(folder/"EXIT.json").write_text(json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+"\\n")\nraise SystemExit(code)\n'
compile(supervisor,'report871_supervisor','exec');(journal/'supervisor.py').write_text(supervisor)
with (journal/'supervisor.stdout.txt').open('x') as out,(journal/'supervisor.stderr.txt').open('x') as err:
 p=subprocess.Popen(['/usr/bin/python3','-B',str(journal/'supervisor.py')],cwd=root,stdout=out,stderr=err,start_new_session=True)
assert all(panel.sha(n)==d for n,d in originals.items())
result=dict(status='CPU_REPORT_SUPERVISOR_STARTED',at=datetime.now().astimezone().isoformat(),pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]),journal=str(journal),output=str(output),qualification=qualification)
(journal/'LAUNCH.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
