from pathlib import Path
from datetime import datetime
import hashlib,json,shutil
root=Path('/data/gaob/Re-ID/Trifusion');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((root/'refine-logs/independent_role_heads_fixed_best_diagnosis_v1/INPUT_SEAL.json').read_text())
targets=['/data/gaob/Re-ID/Trifusion/trained-model/deployment_metric_role_pending100_20261005_842_full_semantic_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/deployment_metric_storage_continuation_20261005_848_full_native_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_native_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/global_task_role_v1_20261004_824_full_semantic_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/metric_feature_scale_20261003_v1_full_metric_raw_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/metric_feature_scale_20261003_v1_full_normalized_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/native_research_v6_20261003_794_full_global_only_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/native_research_v6_20261003_794_full_native_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/native_research_v6_20261003_794_full_semantic_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/region_reconstruction_v1_20261005_858_full_native_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/region_reconstruction_v1_20261005_858_full_semantic_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/role_input_detach_v1_20261004_813_full_native_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/role_input_detach_v1_20261004_813_full_semantic_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/semantic_capacity_control_v1_20261005_856_full_native_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/signal_selection_reference_v1_20261006_868_full_global_only_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/signal_selection_reference_v1_20261006_870_full_all_patch_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/signal_selection_reference_v1_20261006_870_full_masked_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/training_feature_scale_20261003_v2_full_normalized_RGBNT100/last_epoch_distances.pt', '/data/gaob/Re-ID/Trifusion/trained-model/training_feature_scale_20261003_v2_full_raw_RGBNT100/last_epoch_distances.pt'];rows=[];texts={}
for n in targets:
    p=Path(n);folder=p.parent
    assert p.resolve().is_relative_to(root/'trained-model') and not p.is_symlink()
    assert str(p) not in seal['artifact_sha256'] and p.name=='last_epoch_distances.pt'
    assert p.stat().st_nlink==1 and p.stat().st_size==59074461
    training=folder/'training.json';receipt=folder/'official_metrics.json'
    t=json.loads(training.read_text());r=json.loads(receipt.read_text())
    assert t['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and r['status']=='COMPLETE'
    assert [v['epoch'] for v in t['history']]==list(range(1,51))
    assert t['best_epoch']==r['selected_epoch'] and r['selected_epoch']!=50
    assert sha(folder/'best_epoch_distances.pt')==r['training_best_distance_sha256']
    assert sha(folder/'official_distances.pt')==r['distance_sha256']
    best=max(t['history'],key=lambda v:(v['official_fused']['mAP'],v['epoch']))
    assert best['epoch']==r['selected_epoch']
    assert all(abs(best['official_fused'][k]-v)<1e-5 for k,v in r['metrics'].items())
    rows.append(dict(path=n,sha256=sha(p),bytes=p.stat().st_size,
        training_sha256=sha(training),receipt_sha256=sha(receipt),
        best_distance_sha256=r['training_best_distance_sha256'],official_distance_sha256=r['distance_sha256'],
        selected_epoch=r['selected_epoch'],best_metrics=r['metrics'],
        last_epoch=50,last_metrics=t['history'][-1]['official_fused'],
        reason='Closed full50 temporary last-epoch matrix, not mAP-best; all best/official distance artifacts and full trajectories preserved. No current241-seal dependency; no remaining planned query-level last-epoch use.'))
    texts[str(training)]=sha(training);texts[str(receipt)]=sha(receipt)
print(json.dumps(dict(status='QUALIFIED_NOT_RETIRED',at=datetime.now().astimezone().isoformat(),
    rows=rows,text_files=texts,total_bytes=sum(v['bytes'] for v in rows),
    free_bytes=shutil.disk_usage(root).free,
    boundary='Qualification only; no deletion. Preserve every PTH/public/author/current dependency and all best/official distances. Retiring these optional last matrices limits old last-epoch query-level replay; E50 aggregate metrics and complete curves remain.')))
