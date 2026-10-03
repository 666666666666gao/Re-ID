"""Arithmetic/integrity checks of received M0 text; no imports or scoring."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math

root=Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755/m0_intake757')
intake=json.loads((root/'INTAKE.json').read_bytes())
assert intake['status']=='ALL_SIX_M0_PASS_FULL_PHASE_INTAKE'
campaign='logs/metric_feature_scale_20261003_v1'
manifest=json.loads((root/'raw'/campaign/'manifest.json').read_bytes())
rows=[];pairs=[]
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
    witness=json.loads((root/'raw'/campaign/f'initial_forward_pair_{dataset}.json').read_bytes())
    assert witness['status']=='INITIAL_FORWARD_PAIR_PASS'
    assert witness['deployment_max_abs_difference']==0 and witness['ce_logits_max_abs_difference']==0
    assert len(witness['records'])==2
    batches=[]
    for variant in ('normalized','metric_raw'):
        output=root/'raw/trained-model'/f'metric_feature_scale_20261003_v1_m0_{variant}_{dataset}'
        value=json.loads((output/'training.json').read_bytes())
        lines=[json.loads(line) for line in (output/'training_steps.jsonl').read_text().splitlines()]
        assert value['status']=='M0_PASS' and len(lines)==8
        assert [(line['epoch'],line['batch']) for line in lines]==[(1,index) for index in range(8)]
        assert value['history'][0]['steps']==8 and value['m0']['nonzero_gradient_parameters']==value['m0']['trainable_parameters']==155
        assert value['m0']['reload_max_abs_difference']==0
        assert value['initializer']['ce_training_feature_scaling']=='normalized'
        assert value['initializer']['metric_training_feature_scaling']==('normalized' if variant=='normalized' else 'raw')
        assert value['initializer']['initial_model_state_sha256']==witness['initial_model_state_sha256']
        assert value['frozen_parameters_unchanged'] and value['visual_parameters_changed'] and value['fresh_camera_parameters_changed']
        scalar_keys=('loss','id','triplet','ce_feature_norm_min','ce_feature_norm_max','training_feature_norm_min','training_feature_norm_max')
        assert all(math.isfinite(line[key]) for line in lines for key in scalar_keys)
        order=(output/'training_batch_order.jsonl').read_bytes();batches.append(order)
        assert len(order.splitlines())==8
        rows.append({'dataset':dataset,'variant':variant,'optimizer_updates':8,
                     'nonzero_gradient_parameters':155,'strict_reload_max_abs_difference':0,
                     'completed_at':value['completed_at'],'peak_allocated_bytes':value['peak_allocated_bytes'],
                     'ce_norm_range':[min(line['ce_feature_norm_min'] for line in lines),max(line['ce_feature_norm_max'] for line in lines)],
                     'metric_norm_range':[min(line['training_feature_norm_min'] for line in lines),max(line['training_feature_norm_max'] for line in lines)],
                     'triplet_nonzero_steps':sum(line['triplet']>0 for line in lines),
                     'max_total_loss_arithmetic_error':max(abs(line['loss']-line['id']-line['triplet']) for line in lines),
                     'batch_order_sha256':hashlib.sha256(order).hexdigest(),
                     'probe_sha256':value['m0']['reload_probe_sha256']})
    assert batches[0]==batches[1]
    pairs.append({'dataset':dataset,'initial_model_state_sha256':witness['initial_model_state_sha256'],
                  'deployment_max_abs_difference':0,'ce_logits_max_abs_difference':0,
                  'eight_update_batch_order_bytes_equal':True,'witness_scope':'Two real training records in eval mode only.'})
result={'status':'ORIGINAL_SIX_M0_TEXT_ACCOUNTING_PASS','recorded_at':datetime.now().astimezone().isoformat(),
        'total_optimizer_updates':48,'source_count':len(manifest['source_sha256']),'rows':rows,'pairs':pairs,
        'original_intake_sha256':hashlib.sha256((root/'INTAKE.json').read_bytes()).hexdigest(),
        'boundary':'Engineering and recorded batch equality only; no evaluation/scoring/model work, no full50 result, no training-seed or SOTA claim. Six probe SHAs are remote attestations.'}
path=root/'M0_ACCOUNTING.json';assert not path.exists()
path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
