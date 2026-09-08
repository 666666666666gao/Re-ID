import pathlib,hashlib,json,subprocess,datetime
R=pathlib.Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID');RUN=pathlib.Path('/root/trifusion-storage/artifacts/msvr310_role_set_v1_seed42_26c9739')
def j(p):return json.loads(p.read_text())
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
style=j(R/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json');basepath=R/style['BASELINE']['CONFIG'];base=j(basepath)
assert h(basepath)==style['BASELINE']['CONFIG_SHA256']
assert h(R/style['DATA']['PROTOCOL'])==style['DATA']['PROTOCOL_SHA256']==base['protocol_sha256']
source=pathlib.Path(base['signal_source']);commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip();diff=hashlib.sha256(subprocess.check_output(['git','-C',str(source),'diff','--binary'])).hexdigest()
assert commit==base['signal_commit'] and diff==base['signal_diff_sha256']
pipeline=j(RUN/'pipeline.json');summary=j(RUN/'q1/summary.json')
assert [x['stage'] for x in pipeline['stages']]==['t0','m0','m0_cpu','q1','q1_cpu']
assert all(x['exit_code']==0 for x in pipeline['stages'])
assert summary['m0_receipt_sha256']==h(RUN/'m0/summary.json') and summary['m0_verification_sha256']==h(RUN/'m0_cpu.json')
assert datetime.datetime.fromisoformat(pipeline['stages'][2]['ended_at'])<datetime.datetime.fromisoformat(pipeline['stages'][3]['started_at'])
pid_state={str(p):pathlib.Path('/proc') .joinpath(str(p)).exists() for p in [pipeline['wrapper_pid'],*[s['original_pid'] for s in pipeline['stages']]]}
print(json.dumps(dict(status='PASS_RECURSIVE_BASE_CONFIG_SIGNAL_COMMIT_DIFF_AND_STAGE_ORDER',base_config_sha256=h(basepath),protocol_sha256=h(R/style['DATA']['PROTOCOL']),signal_commit=commit,signal_diff_sha256=diff,original_pid_exists=pid_state,m0_receipt_binding=True,m0_cpu_before_q1=True,all_stage_exit_codes_zero=True,model_forwards=0,optimizer_updates=0,checked_at=datetime.datetime.now().astimezone().isoformat())))
