from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,subprocess
root=Path('/data/gaob/Re-ID/Trifusion')
campaign=root/'logs/semantic_capacity_first_eval_completion_20261005_857'
launch=root/'logs/semantic_capacity_first_eval_launch_20261005_857'
original=root/'logs/semantic_capacity_control_v1_20261005_856'
assert not campaign.exists() and not launch.exists()
assert not (root/'results/semantic_capacity_control_v1_complete_20261005_856').exists()
assert not Path('/proc/642951').exists() and not Path('/proc/642952').exists()
assert shutil.disk_usage(root).free>=2*1024**3
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for data in iter(lambda:stream.read(1024*1024),b''):h.update(data)
 return h.hexdigest()
failure={'/data/gaob/Re-ID/Trifusion/logs/semantic_capacity_control_v1_20261005_856/campaign.json': '7520956155145c8adde04affd538fea9d99b5799d34b89de002fb4cac8d0c174', '/data/gaob/Re-ID/Trifusion/logs/semantic_capacity_launch_20261005_856/EXIT.json': '4515887a6f349d7691e98ae6724e0c98d7bf14cc7a4620ed1d69f96d2b542cba', '/data/gaob/Re-ID/Trifusion/logs/semantic_capacity_launch_20261005_856/console.log': 'ba4d3944936cccbcfb898627dbdf7c087e883904917fc66e5b9f1e5a90067bfe'}
assert all(sha(Path(n))==d for n,d in failure.items())
state=json.loads((original/'campaign.json').read_text())
full=next(r for r in state['jobs'] if (r['dataset'],r['phase'])==('RGBNT100','full'))
assert len(full['steps'])==1 and full['steps'][0]['mode']=='train' and full['steps'][0]['exit_code']==0
assert state['report_invocations']==0
run=root/'trained-model/semantic_capacity_control_v1_20261005_856_full_native_RGBNT100'
assert not (run/'official_metrics.json').exists() and not (original/'RGBNT100_evaluate.log').exists()
training=json.loads((run/'training.json').read_text())
assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and len(training['history'])==50
campaign.mkdir();launch.mkdir()
entry=campaign/'ADMIN_COMPLETE857.py';entry.write_bytes(b'"""Complete the never-started strict evaluation/report after the recorded disk stop."""\nfrom datetime import datetime\nimport json\nimport os\nfrom pathlib import Path\nimport shutil\nimport subprocess\nimport sys\n\nROOT = Path(\'/data/gaob/Re-ID/Trifusion\')\nsys.path.insert(0, str(ROOT))\nfrom tools import queue_semantic_capacity as panel\n\noriginal = ROOT / \'logs/semantic_capacity_control_v1_20261005_856\'\ncampaign = Path(__file__).resolve().parent\nreport_dir = ROOT / \'results/semantic_capacity_control_v1_complete_20261005_856\'\nplan = json.loads((campaign / \'PLAN.json\').read_text())\npanel.configure()\nassert os.environ[\'CUDA_VISIBLE_DEVICES\'] == \'0,1\'\nassert all(panel.base.sha(Path(n)) == d for n, d in plan[\'original_failure_sha256\'].items())\nassert not Path(\'/proc/642951\').exists() and not Path(\'/proc/642952\').exists()\nassert not report_dir.exists()\nold_state = json.loads((original / \'campaign.json\').read_text())\nassert old_state[\'report_invocations\'] == 0\nfull = next(r for r in old_state[\'jobs\'] if (r[\'dataset\'], r[\'phase\']) == (\'RGBNT100\', \'full\'))\nassert len(full[\'steps\']) == 1 and full[\'steps\'][0][\'mode\'] == \'train\' and full[\'steps\'][0][\'exit_code\'] == 0\nrun = panel.base.output_dir(original, \'full\', \'RGBNT100\', \'native\')\nassert not (run / \'official_metrics.json\').exists() and not (original / \'RGBNT100_evaluate.log\').exists()\nassert panel.base.sha(run / \'training.json\') == plan[\'training_sha256\']\nassert panel.base.sha(run / \'best_map.pth\') == plan[\'best_sha256\']\nassert panel.base.sha(campaign / Path(__file__).name) == plan[\'coordinator_sha256\']\npanel.base.require_sources(original)\nmanifest = json.loads((original / \'manifest.json\').read_text())\nmanifest[\'administrative_completion\'] = dict(original_campaign=str(original), plan_sha256=panel.base.sha(campaign / \'PLAN.json\'),\n    boundary=\'Inherited original initializers and all three completed trainings; only first strict100 and first CPU report run here.\')\npanel.base.queue.write(campaign / \'manifest.json\', manifest)\nshutil.copytree(original / \'initialization\', campaign / \'initialization\')\nstate = json.loads(json.dumps(old_state))\nstate.update(status=\'RUNNING\', controller_pid=os.getpid(), started_at=panel.base.queue.stamp(),\n             original_campaign=str(original), original_parent_exit_code=1,\n             inherited_training_endpoints=3, new_training_invocations=0, report_invocations=0)\npanel.base.queue.write(campaign / \'campaign.json\', state)\nstep = dict(mode=\'evaluate\', command=panel.command(\'RGBNT100\', \'native\', \'evaluate\', original, run),\n            origin=\'FIRST_STRICT_NOT_PREVIOUSLY_STARTED\')\nfull = next(r for r in state[\'jobs\'] if (r[\'dataset\'], r[\'phase\']) == (\'RGBNT100\', \'full\'))\nfull[\'steps\'].append(step)\ncode = panel.previous.previous.previous.run_logged(campaign, state, step, \'RGBNT100_first_evaluate.log\')\nif code:\n    raise SystemExit(code)\nassert all(panel.base.sha(Path(n)) == d for n, d in plan[\'original_failure_sha256\'].items())\nrow = panel.previous.accept_and_retire_probe(original, \'RGBNT100\', \'native\')\nfull.update(status=\'COMPLETE\', exit_code=0, completed_at=panel.base.queue.stamp(), result=row,\n            completion_origin=\'ORIGINAL_FULL50_PLUS_FIRST_STRICT_ADMINISTRATIVE_COMPLETION\')\nshutil.copytree(original / \'acceptance\', campaign / \'acceptance\')\nshutil.copyfile(original / \'probe_retirement.jsonl\', campaign / \'probe_retirement.jsonl\')\nrows = [panel.previous.accepted_row(campaign, dataset, \'native\') for dataset in panel.DATASETS]\npanel.previous.require_controls()\npanel.base.queue.write(campaign / \'accepted_matrix.json\', dict(schema=panel.SCHEMA, accepted=3, expected=3, rows=rows))\nstate.update(status=\'COMPLETE\', completed_at=panel.base.queue.stamp(), report_invocations=1)\npanel.base.queue.write(campaign / \'campaign.json\', state)\nassert shutil.disk_usage(ROOT).free >= 2 * 1024**3\nwith (campaign / \'report.log\').open(\'x\') as log:\n    result = subprocess.run([sys.executable, \'-B\', str(ROOT / \'tools/report_semantic_capacity.py\'),\n        \'--campaign\', str(campaign), \'--output-dir\', str(report_dir)], cwd=ROOT,\n        env=dict(os.environ, CUDA_VISIBLE_DEVICES=\'\'), stdout=log, stderr=subprocess.STDOUT)\nstate.update(report_exit_code=result.returncode, report_completed_at=panel.base.queue.stamp())\npanel.base.queue.write(campaign / \'campaign.json\', state)\nassert all(panel.base.sha(Path(n)) == d for n, d in plan[\'original_failure_sha256\'].items())\nprint(json.dumps(dict(status=\'FIRST_STRICT_AND_SINGLE_REPORT_FINISHED\', at=datetime.now().astimezone().isoformat(),\n                     first_evaluation_exit=0, report_exit=result.returncode, original_parent_exit=1,\n                     new_training_invocations=0)), flush=True)\nraise SystemExit(result.returncode)\n')
assert sha(entry)=='ab50dadaf5e6bd73dea2e308c13ed651a291f30bba88479bf4824b9a7994e1a1'
plan=dict(status='FIRST_EVALUATION_AND_FIRST_REPORT_ONLY',registered_at=datetime.now().astimezone().isoformat(),
 original_failure_sha256=failure,original_parent_exit_code=1,original_campaign=str(original),
 training_sha256=sha(run/'training.json'),best_sha256=sha(run/'best_map.pth'),
 coordinator_sha256=sha(entry),science_source_count=345,raw_control_artifacts=187,
 new_training_invocations=0,first_strict100_previously_invoked=False,
 report_previously_invoked=False,physical_gpus=[0,1],disk_reserve_bytes=2*1024**3,
 boundary='Finish existing registered capacity study without rerunning training, changing weights/selection/recipe/source, or modifying original failure state/EXIT/log.100 M0 probe is kept until strict acceptance. No power/temperature action. New administrative state has inherited original training provenance.')
(campaign/'PLAN.json').write_bytes((json.dumps(plan,indent=2)+'\n').encode())
script=launch/'SUPERVISOR.py';script.write_bytes(b"from datetime import datetime\nimport json,os\nfrom pathlib import Path\nimport subprocess,sys\nroot=Path('/data/gaob/Re-ID/Trifusion')\nlaunch=Path(__file__).resolve().parent\ncommand=[sys.executable,'-B',str(root/'logs/semantic_capacity_first_eval_completion_20261005_857/ADMIN_COMPLETE857.py')]\nwith (launch/'console.log').open('x') as output:\n code=subprocess.run(command,cwd=root,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=output,stderr=subprocess.STDOUT).returncode\n(launch/'EXIT.json').write_bytes((json.dumps(dict(exit_code=code,completed_at=datetime.now().astimezone().isoformat()))+'\\n').encode())\nraise SystemExit(code)\n")
with (launch/'supervisor.log').open('x') as output:
 process=subprocess.Popen(['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(script)],cwd=root,
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1'),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
ticks=Path('/proc/'+str(process.pid)+'/stat').read_text().split()[21]
value=dict(status='ADMINISTRATIVE_FIRST_STRICT_AND_REPORT_LAUNCHED',at=datetime.now().astimezone().isoformat(),
 pid=process.pid,start_ticks=ticks,campaign=str(campaign),launch=str(launch),
 coordinator_sha256=sha(entry),plan_sha256=sha(campaign/'PLAN.json'),
 original_failure_sha256=failure,new_training_invocations=0,disk_free_bytes=shutil.disk_usage(root).free)
(launch/'LAUNCH.json').write_bytes((json.dumps(value,indent=2)+'\n').encode())
print(json.dumps(value))
