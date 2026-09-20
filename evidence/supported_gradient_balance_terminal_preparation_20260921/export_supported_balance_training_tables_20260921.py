"""Export every recorded training step/epoch; do not create epoch retrieval scores."""
from pathlib import Path
import csv,hashlib,json,sys

source=Path(sys.argv[1]);mode=sys.argv[2];output=Path(sys.argv[3])
assert mode in ('m0','q1') and not output.exists()
inventory=json.loads((source/'inventory.json').read_bytes())
for item in inventory['files']:
    data=(source/item['path']).read_bytes()
    assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
receipt=json.loads((source/(mode+'_cpu.json')).read_bytes())
assert receipt['status']=='PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_'+mode.upper()
summary_path=source/mode/'summary.json'
assert hashlib.sha256(summary_path.read_bytes()).hexdigest()==receipt['summary_sha256']
steps=[];epochs=[]
for path in sorted((source/mode).glob('*/training.json')):
    training=json.loads(path.read_bytes());name=path.parent.name
    assert len(training['steps'])==training['optimizer_steps']
    for step in training['steps']:
        assert len(step['components'])==14
        steps.append(dict(endpoint=name,step=step['step'],epoch=step['epoch'],loss=step['loss'],
            active_fused_metric=step['active_fused_metric'],amp_scale_before=step['amp_scale_before'],
            amp_scale_after=step['amp_scale_after'],**step['components']))
    for epoch in training['history']:
        selected=[s for s in training['steps'] if s['epoch']==epoch['epoch']]
        assert len(selected)==epoch['optimizer_steps']
        means={key:sum(s['components'][key] for s in selected)/len(selected) for key in selected[0]['components']}
        assert abs(epoch['mean_loss']-sum(s['loss'] for s in selected)/len(selected))<1e-10
        epochs.append(dict(endpoint=name,**epoch,active_fused_metrics=','.join(sorted({s['active_fused_metric'] for s in selected})),**means))
assert len(steps)==(248 if mode=='m0' else 1560)
assert len(epochs)==(8 if mode=='m0' else 120)
output.mkdir(parents=True)
for filename,rows in [('all_training_steps.csv',steps),('all_training_epochs.csv',epochs)]:
    with (output/filename).open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
result=dict(status='COMPLETE_RECORDED_TRAINING_TABLES',mode=mode,steps=len(steps),epochs=len(epochs),
    input_summary_sha256=receipt['summary_sha256'],model_forwards=0,optimizer_updates=0,
    scope='All recorded training scalar objectives. Epoch component columns are step means. No epoch heldout retrieval was performed or inferred; final retrieval remains a separate protocol.')
(output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
