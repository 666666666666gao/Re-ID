from pathlib import Path
import json,collections
root=Path(__file__).parent
intake=root/'snapshots/intake'
summary=json.loads((intake/'m0/summary.json').read_bytes())
output={k:v for k,v in summary.items() if k not in ('folds','overfit')}
output['endpoints']=[]
for folder in sorted((intake/'m0').iterdir()):
    if not folder.is_dir():continue
    tr=json.loads((folder/'training.json').read_bytes())
    audits=[json.loads(s) for s in (folder/'memory_steps.jsonl').read_text().splitlines()]
    counter=collections.Counter()
    refs=[];scalars={}
    for a in audits:
        eligible=a['support']['eligible_anchors']
        counter['steps']+=1;counter['warmup']+=not a['replacement_active']
        counter['supported_active']+=a['replacement_active'] and eligible>0
        counter['unsupported_active']+=a['replacement_active'] and eligible==0
        counter['zero_eligible_any']+=eligible==0
        counter['eligible_anchors']+=eligible
        counter['history_steps']+=bool(a['memory']);counter['candidates']+=len(a['memory'])
        counter['distance_elements']+=a['distance_float_count']
        counter['history_vjp_groups']+=len(a['history_vjp_groups'])
        counter['fresh_forwards']+=a['fresh_role_record_forwards']
        for r,parts in a['rank_auxiliary_reference_checks'].items():
            for k,v in parts.items():refs.append(dict(step=a['step'],role=r,part=k,relative=v['relative_l2_error'],difference=v['difference_norm'],passed=v['passed']))
    row=dict(folder=folder.name,counts=dict(counter),training={k:v for k,v in tr.items() if k not in ('steps','audit_files','historical_parameter_gradient_witness')},reference_rows=refs)
    if folder.name.startswith('overfit_'):row['gate']=summary['overfit'][folder.name.split('_')[1]]['gate']
    else:row['preflight']=json.loads((folder/'receipt.json').read_bytes())['preflight']
    output['endpoints'].append(row)
t0=json.loads((intake/'t0.json').read_bytes())
output['t0']={k:v for k,v in t0.items() if k!='folds'}
output['t0']['counts']=[dict(fold=f['fold'],batches=len(f['steps']),zero_eligible=sum(not x['eligible_anchors'] for x in f['steps']),zero_eligible_active=sum(not x['eligible_anchors'] and x['step']>65 for x in f['steps'])) for f in t0['folds']]
cpu=json.loads((intake/'m0_cpu.json').read_bytes())
output['executor_cpu_receipt']={k:v for k,v in cpu.items() if k!='files'}
destination=root/'local_evidence_structure.json'
destination.write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in output.items() if k not in ('endpoints','executor_cpu_receipt')},indent=2))
for r in output['endpoints']:
    print(json.dumps(dict(folder=r['folder'],counts=r['counts'],gate=r.get('gate'),reference_count=len(r['reference_rows']),reference_max=max((x['relative'] or 0 for x in r['reference_rows']),default=0),history=r['training']['history'],state=r['training']['gradient_balance_state'])))
