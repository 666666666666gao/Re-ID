"""Independent provenance hashes/text capture. No GPU or model execution."""
from pathlib import Path
import ast, hashlib, json, os, subprocess, time
R = Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
RUN = Path('/root/trifusion-storage/artifacts/msvr310_smooth_ap_v1_seed42_2e947a4')
result = {'generated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':{},'bindings':[], 'texts':{},'configs':{},'processes':{}}
def capture(p, text=False):
    p = Path(p)
    assert '/msvr310_smooth_ap_v1_seed42_2e947a4/q1/' not in str(p)
    if str(p) not in result['files']:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):h.update(b)
        result['files'][str(p)]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    if text:result['texts'][str(p)]=p.read_text(encoding='utf-8')
    return result['files'][str(p)]
def bind(p,digest,origin):
    actual=capture(p)
    result['bindings'].append({'path':str(p),'expected':digest,'actual':actual['sha256'],'match':actual['sha256']==digest,'origin':origin})
def visit(rel):
    p=R/rel
    if str(p) in result['configs'] or p.suffix!='.json' or not p.is_file():return
    capture(p,True);obj=json.loads(p.read_bytes());result['configs'][str(p)]=obj
    for key in ('project_file_sha256','project_source_file_sha256','project_files','fixed_file_sha256','fixed_files','signal_source_file_sha256'):
        for name,h in obj.get(key,{}).items():
            target=Path(obj['signal_source'])/name if key=='signal_source_file_sha256' else Path(name) if name.startswith('/') else R/name
            bind(target,h,str(p)+':'+key)
            if target.suffix in ('.py','.yml','.yaml'):capture(target,True)
    def walk(value):
        if isinstance(value,dict):
            for k,v in value.items():walk(k);walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
        elif isinstance(value,str) and value.startswith('configs/'):visit(value)
    walk(obj)
visit('configs/MSVR310/TriFusion-smooth-ap-paired-v1.json')
style=result['configs'][str(R/'configs/MSVR310/TriFusion-source-style-paired-v1-r2.json')]
base=result['configs'][str(R/style['BASELINE']['CONFIG'])]
for p,h in [(R/style['BASELINE']['CONFIG'],style['BASELINE']['CONFIG_SHA256']), (style['BASELINE']['SUMMARY'],style['BASELINE']['SUMMARY_SHA256']), (R/base['protocol'],base['protocol_sha256']), (R/style['SOURCE_METADATA']['PATH'],style['SOURCE_METADATA']['SHA256']), (base['clip_weight'],base['clip_weight_sha256'])]:
    bind(p,h,'explicit context binding')
for p in (R/base['protocol'],R/style['SOURCE_METADATA']['PATH'],Path(style['BASELINE']['SUMMARY'])):capture(p,True)
baseline=json.loads(Path(style['BASELINE']['SUMMARY']).read_bytes())
for fold in baseline['folds']:
    bind(fold['checkpoint'],fold['checkpoint_sha256'],'baseline checkpoint')
    bind(Path(fold['checkpoint']).parent/'retrieval_arrays.pt',fold['retrieval']['retrieval_arrays_sha256'],'baseline retrieval file hash only')
    for name in ('training.json','receipt.json'):
        p=Path(fold['checkpoint']).parent/name
        if p.is_file():capture(p,True)
for p in sorted((RUN/'m0').rglob('*')):
    if p.is_file():capture(p,p.suffix in ('.json','.jsonl','.log'))
for name in ('t0.json','t0.log','m0.log','m0_cpu.json','m0_cpu.log','pipeline.json'):capture(RUN/name,True)
pipeline=json.loads((RUN/'pipeline.json').read_bytes())
for pid in [pipeline['wrapper_pid']]+[r['original_pid'] for r in pipeline['stages']]:
    proc=Path('/proc')/str(pid)
    result['processes'][str(pid)]={'exists':proc.is_dir()}
    if proc.is_dir():
        result['processes'][str(pid)]['stat']= (proc/'stat').read_text()
        result['processes'][str(pid)]['cmdline']=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        result['processes'][str(pid)]['cwd']=str((proc/'cwd').resolve())
result['remote_head']=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
result['signal_head']=subprocess.check_output(['git','-C',base['signal_source'],'rev-parse','HEAD'],text=True).strip()
result['signal_diff_sha256']=hashlib.sha256(subprocess.check_output(['git','-C',base['signal_source'],'diff','--binary'])).hexdigest()
result['signal_commit_match']=result['signal_head']==base['signal_commit']
result['signal_diff_match']=result['signal_diff_sha256']==base['signal_diff_sha256']
for p in ('refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md','refine-logs/msvr310_smooth_ap_v1/EXPERIMENT_TRACKER.md','tools/build_msvr310_train_oof_protocol.py'):
    capture(R/p,True)
result['boundary']={'model_forwards':0,'optimizer_updates':0,'official_image_reads':0,'q1_files_read':0,'file_mutations':0,'process_mutations':0}
print(json.dumps(result))
