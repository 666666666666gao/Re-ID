"""Single new two-arm prepare/M0 stage. No formal training or retry."""
import hashlib,json,subprocess
from datetime import datetime
from pathlib import Path
w=Path(__file__).resolve().parent
receipt=json.loads((w/'EVIDENCE_SOURCE_SYNC_925.json').read_text())
assert receipt['status']=='EVIDENCE925_STATIC_EXACT'
request={'source_sha256':receipt['runtime_source_sha256'],
 'prior_seal_sha256':receipt['prior_seal_sha256'],
 'original_initializer':receipt['original_initializer'],
 'source_audit_sha256':receipt['source_audit_sha256']}
code='pins='+repr(request)+'\n'
code+=r'''
import hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime
from pathlib import Path
root=Path('/data/gaob/Re-ID/Trifusion')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stamp=lambda:datetime.now().astimezone().isoformat()
assert all(sha(root/n)==h for n,h in pins['source_sha256'].items())
prior=root/'refine-logs/region_use_fixed_best_diagnosis918/INPUT_SEAL.json'
assert sha(prior)==pins['prior_seal_sha256']
assert sha(root/'refine-logs/global_conditioned_evidence_20261013_924/SOURCE_AUDIT.json')==pins['source_audit_sha256']
protocol=root/'logs/training_feature_scale_protocols_20261002/RGBNT201.json'
clip=root/'pertrained-model/ViT-B-16.pt'
assert sha(protocol)==pins['original_initializer']['protocol_sha256']
assert sha(clip)==pins['original_initializer']['public_clip_sha256']
campaign=root/'logs/global_conditioned_evidence_m0_20261013_925'
assert not campaign.exists()
outputs={kind:root/f'trained-model/global_conditioned_evidence_20261013_925_m0_{kind}_RGBNT201' for kind in ('gradient','random')}
assert all(not p.exists() for p in outputs.values())
query=['nvidia-smi','--id=0,1','--query-gpu=index,memory.used','--format=csv,noheader,nounits']
devices=[tuple(int(v.strip()) for v in row.split(',')) for row in subprocess.check_output(query,text=True).splitlines()]
assert {x[0] for x in devices}=={0,1} and all(x[1]<500 for x in devices),devices
required=(4*384+384+128+512)*1024**2
free=shutil.disk_usage(root).free
assert free>=required,(free,required)
campaign.mkdir()
(campaign/'initialization').mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
stages=[]
keys=('public_clip_sha256','protocol_sha256','visual_initial_sha256','camera_initial_sha256',
 'shared_initializer_sha256','initial_model_state_sha256','trainable_parameters','trainable_parameter_tensors',
 'batch_size','num_instances','head_names','cfg_yaml','visual_placement','parameter_placement')
status={'status':'PREPARE_M0_RUNNING','started_at':stamp(),'supervisor_pid':os.getpid(),
 'physical_gpus':[0,1],'gpu_admission':devices,'disk_free_bytes':free,'required_bytes':required,
 'new_formal_training_started':False,'stages':stages}
(campaign/'campaign.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'status':status['status'],'pid':os.getpid(),'campaign':str(campaign)}),flush=True)
started=time.perf_counter()
for kind in ('gradient','random'):
 init=campaign/'initialization'/f'RGBNT201_{kind}.json'
 for mode in ('prepare','m0'):
  assert all(sha(root/n)==h for n,h in pins['source_sha256'].items())
  command=['/data/gaob/Re-ID/conda-envs/tri_reid/bin/python','-B',str(root/'tools/run_global_conditioned_evidence.py'),
   '--dataset','RGBNT201','--exposure',kind,'--mode',mode,'--protocol',str(protocol),
   '--signal-source',str(root/'comparators/Signal-cd1b0a6'),'--clip-weight',str(clip),
   '--initialization',str(init),'--output-dir',str(outputs[kind])]
  log_path=campaign/f'{kind}_{mode}.log'
  with log_path.open('x') as log:
   child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
   stage={'exposure':kind,'mode':mode,'started_at':stamp(),'child_pid':child.pid,'command':command,'log':str(log_path)}
   stages.append(stage)
   (campaign/'campaign.json').write_text(json.dumps(status,indent=2)+'\n')
   exit_code=child.wait()
  stage.update(exit_code=exit_code,completed_at=stamp())
  (campaign/f'{kind}_{mode}_EXIT.json').write_text(json.dumps(stage,indent=2)+'\n')
  print(json.dumps(stage),flush=True)
  if exit_code!=0:
   status.update(status='ORIGINAL_STAGE_FAILED',failed_exposure=kind,failed_mode=mode,completed_at=stamp(),elapsed_seconds=time.perf_counter()-started)
   (campaign/'campaign.json').write_text(json.dumps(status,indent=2)+'\n')
   sys.exit(exit_code)
  value=json.loads(init.read_text())
  binding=value['binding']
  assert value['schema']=='trifusion-global-conditioned-evidence-v1' and value['status']=='INITIALIZATION_VERIFIED'
  assert all(binding[k]==pins['original_initializer'][k] for k in keys)
  assert binding['evidence_exposure']==kind and binding['added_model_parameters']==0
  if mode=='m0':
   value=json.loads((outputs[kind]/'training.json').read_text())
   assert value['status']=='M0_PASS' and value['initializer']==binding
   assert value['m0']['nonzero_gradient_parameters']==value['m0']['trainable_parameters']==binding['trainable_parameter_tensors']
   assert value['m0']['reload_max_abs_difference']<=1e-5
   assert value['frozen_parameters_unchanged'] and value['visual_parameters_changed'] and value['fresh_camera_parameters_changed']
   assert value['production_m0_diagnostics']['effective_optimizer_updates']==8
   assert all(v==8 for v in value['production_m0_diagnostics']['author_bn_batches_tracked'].values())
   steps=[json.loads(line) for line in (outputs[kind]/'training_steps.jsonl').read_text().splitlines()]
   assert len(steps)==8
   supported=0
   for step in steps:
    activity=step['evidence_exposure']
    assert activity['parameter_gradients_absent_after_exposure_vjp']
    assert step['role_task_shared_gradients_absent'] and step['role_task_checked_shared_parameter_tensors']>0
    assert activity['excluded_patches']==32*activity['active_sample_modalities']
    assert activity['available_sample_modalities']==192
    supported+=int(activity['support']['supported_relations']>0 and activity['active_sample_modalities']>0)
   assert supported>0
   stage['specific_activity_acceptance']={'supported_active_steps':supported,'checked_steps':8,'old_semantic_initial_state_equal':True,
    'exposure_vjp_parameter_gradients_absent':True,'role_task_shared_gradients_absent':True,'equal_count_exclusion_checked':True,
    'training_receipt_sha256':sha(outputs[kind]/'training.json'),'step_log_sha256':sha(outputs[kind]/'training_steps.jsonl')}
  (campaign/'campaign.json').write_text(json.dumps(status,indent=2)+'\n')
assert all(sha(root/n)==h for n,h in pins['source_sha256'].items())
status.update(status='BOTH_PREPARE_M0_ACCEPTED_NO_FORMAL_TRAINING',completed_at=stamp(),elapsed_seconds=time.perf_counter()-started)
(campaign/'campaign.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'status':status['status'],'elapsed_seconds':status['elapsed_seconds'],'campaign':str(campaign)}),flush=True)
'''
opts=['-o','ProxyCommand=none','-o','ProxyJump=none','-o','BatchMode=yes','-i','C:/Users/gb/.ssh/id_ed25519']
cmd=['ssh',*opts,'-p','2026','gaob@172.19.12.138','/usr/bin/python3','-']
(w/'EVIDENCE_M0_LAUNCH_COMMAND_925.json').write_text(json.dumps({'at':datetime.now().astimezone().isoformat(),'command':cmd,'remote_code_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2)+'\n')
run=subprocess.run(cmd,input=code,text=True,capture_output=True)
for suffix,data in [('stdout',run.stdout),('stderr',run.stderr)]:
 (w/('EVIDENCE_M0_LAUNCH_925.'+suffix)).write_text(data)
(w/'EVIDENCE_M0_LAUNCH_EXIT_925.json').write_text(json.dumps({'at':datetime.now().astimezone().isoformat(),'exit_code':run.returncode},indent=2)+'\n')
print(run.stdout,end='')
print(run.stderr,end='')
raise SystemExit(run.returncode)
