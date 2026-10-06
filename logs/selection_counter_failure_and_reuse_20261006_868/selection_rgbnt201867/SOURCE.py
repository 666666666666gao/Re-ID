from pathlib import Path
from datetime import datetime
import json,shutil,sys
root=Path('/data/gaob/Re-ID/Trifusion');sys.path.insert(0,str(root))
from tools import queue_signal_selection_reference as panel
from tools import continue_signal_selection_reference as continuation
campaign=root/'logs/signal_selection_reference_v1_20261006_867'
sources=panel.require_sources();panel.require_protected();assert len(sources)==373
state=json.loads((campaign/'campaign.json').read_text())
assert state['status']=='RUNNING' and state['report_invocations']==0
rows=[];files=[];training={};m0_results=[]
old_retired=[json.loads(line) for line in (continuation.ORIGINAL/'retired_m0.jsonl').read_text().splitlines()]
current_retired=[json.loads(line) for line in (campaign/'retired_m0.jsonl').read_text().splitlines()]
for selection in panel.SELECTIONS:
    job=next(j for j in state['jobs'] if (j['dataset'],j['selection'],j['phase'])==('RGBNT201',selection,'full'))
    assert job['status']=='COMPLETE' and job['exit_code']==0
    source=continuation.origin(campaign,'RGBNT201',selection,'full')
    row=panel.accepted_row(source,'RGBNT201',selection);assert row==job['result']
    folder=Path(row['run_dir']);rows.append(row)
    training[selection]=json.loads((folder/'training.json').read_text())
    assert row['formal_steps']==2649
    files.extend(folder/name for name in ('training.json','training_steps.jsonl','training_batch_metadata.jsonl','official_metrics.json'))
    m0_source=continuation.origin(campaign,'RGBNT201',selection,'m0')
    m0folder=panel.output_dir(m0_source,'RGBNT201',selection,'m0')
    m0=json.loads((m0folder/'training.json').read_text())
    assert m0['status']=='M0_PASS' and len(m0['history'])==1 and m0['history'][0]['steps']==8
    assert m0['initializer']==panel.binding(m0_source,'RGBNT201',selection)
    assert len((m0folder/'training_batch_metadata.jsonl').read_text().splitlines())==8
    assert m0['m0']['nonzero_gradient_parameters']==m0['m0']['trainable_parameters']
    assert m0['m0']['reload_max_abs_difference']<=1e-5
    assert all(v==8 for v in m0['selection_reference']['m0_bn_counts'].values())
    probe=m0folder/'m0_reload_probe.pth'
    retirement=next(r for r in old_retired+current_retired if r['path']==str(probe))
    assert not probe.exists() and retirement['sha256']==m0['m0']['reload_probe_sha256']
    assert retirement['formal_receipt_sha256']==row['receipt_sha256']
    m0_results.append(dict(selection=selection,m0=m0,retirement=retirement))
    files.extend(m0folder/name for name in ('training.json','training_steps.jsonl','training_batch_metadata.jsonl'))
    files.append(campaign/f'initialization/RGBNT201_{selection}.json')
masked,all_patch=(panel.binding(campaign,'RGBNT201',s) for s in ('masked','all_patch'))
for key in ('initial_model_state_sha256','trainable_parameters','trainable_parameter_tensors','cfg_yaml','selection_topk','feature_width'):
    assert masked[key]==all_patch[key],key
assert masked['feature_width']==3072
batch=[(Path(r['run_dir'])/'training_batch_metadata.jsonl').read_bytes() for r in rows]
assert batch[0]==batch[1]==batch[2]
selected={r['variant']:r for r in rows}
pairs=[dict(candidate=candidate,control=control,delta_metrics={k:selected[candidate]['metrics'][k]-selected[control]['metrics'][k] for k in selected[control]['metrics']}) for candidate,control in (('masked','all_patch'),('masked','global_only'),('all_patch','global_only'))]
files.extend(campaign/name for name in ('paired_initialization_RGBNT201.json','initializer_reuse_proof.json','endpoint_origins.json','failed_first_epoch_batch_reuse.json','RGBNT201_masked_train.log','RGBNT201_masked_evaluate.log','RGBNT201_all_patch_m0.log','RGBNT201_all_patch_train.log','RGBNT201_all_patch_evaluate.log'))
assert all(p.is_file() for p in files)
result=dict(status='RGBNT201_THREE_FORMAL_ENDPOINTS_ACCEPTED',at=datetime.now().astimezone().isoformat(),
    rows=rows,training=training,pairs=pairs,m0_results=m0_results,source_count=len(sources),actual_batch_metadata_equal=True,
    free_bytes=shutil.disk_usage(root).free,text_files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=panel.sha(p)) for p in files},
    boundary='Closed endpoint texts plus actualPTH/distance/receiptSHA. Primarymasked-allpatch matched3072; global1536 mixesdimension/head/capacity/objectives. No NN/evaluation/queryreport replay, no activecampaignfiles transferred. First201pair alone notthree-datasetstability, notTriFusioninnovation orSOTA. Bestweightsstillconsumed byall9report.')
print(json.dumps(result))
