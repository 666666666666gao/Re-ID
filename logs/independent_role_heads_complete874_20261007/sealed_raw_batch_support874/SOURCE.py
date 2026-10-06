from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime
import hashlib,json,statistics

root=Path('/data/gaob/Re-ID/Trifusion')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
seal_path=root/'refine-logs/global_task_role_fixed_best_diagnosis_v1/INPUT_SEAL.json'
assert sha(seal_path)=='9f55f45278dbe3c03f5b5cc78b007e7a49b5693019ccba05cea3eea3bdc8256e'
seal=json.loads(seal_path.read_text());results=[];input_sha={str(seal_path):sha(seal_path)}
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
    control=next(r for r in seal['rows'] if (r['dataset'],r['variant'])==(dataset,'semantic'))
    folder=Path(control['run_dir']);batch_path=folder/'training_batch_order.jsonl';training_path=folder/'training.json'
    protocol_path=root/'logs/training_feature_scale_protocols_20261002'/f'{dataset}.json'
    for p in (batch_path,training_path):
        digest=sha(p);assert digest==seal['artifact_sha256'][str(p)];input_sha[str(p)]=digest
    assert sha(protocol_path)==control['initializer']['protocol_sha256'];input_sha[str(protocol_path)]=sha(protocol_path)
    protocol=json.loads(protocol_path.read_text());training=json.loads(training_path.read_text())
    assert training['status']=='BEST_OFFICIAL_MAP_TRAINING_COMPLETE' and training['epochs']==50 and len(training['history'])==50
    environment=protocol['environment_key'];assert environment==('scene' if dataset=='MSVR310' else 'camera')
    records=protocol['records']['train'];by_name={Path(r['paths'][0]).name:r for r in records}
    assert len(by_name)==len(records)
    grouped=defaultdict(list)
    for r in records:
        assert protocol['train_label_map'][str(r['identity'])]==r['label'];grouped[r['label']].append(r)
    assert len(grouped)==len(protocol['train_label_map'])
    inventory_cross={label for label,items in grouped.items() if len({r[environment] for r in items})>=2}
    totals=Counter();positive_hist=Counter();unique_hist=Counter();per_epoch=defaultdict(Counter)
    per_identity=defaultdict(Counter);batch_counts=[];batch_keys=set();observed_records=set()
    for line in batch_path.read_text().splitlines():
        batch=json.loads(line);key=(batch['epoch'],batch['batch']);assert key not in batch_keys;batch_keys.add(key)
        rows=[by_name[name] for name in batch['paths']]
        labels=batch['labels'];cameras=batch['cameras'];size=len(rows)
        assert size==len(labels)==len(cameras)==control['initializer']['batch_size']
        assert all(r['label']==label and r['camera']==camera for r,label,camera in zip(rows,labels,cameras))
        id_counts=Counter(labels);env_counts=Counter((r['label'],r[environment]) for r in rows)
        camera_counts=Counter((r['label'],r['camera']) for r in rows)
        assert set(id_counts.values())=={control['initializer']['num_instances']}
        unique_by_identity=defaultdict(dict)
        for name,r in zip(batch['paths'],rows):unique_by_identity[r['label']][name]=r[environment]
        batch_total=Counter(batches=1,anchor_positions=size,duplicate_record_positions=size-len(set(batch['paths'])))
        for name,r in zip(batch['paths'],rows):
            label,env=r['label'],r[environment]
            positives=id_counts[label]-env_counts[(label,env)]
            distinct=sum(e!=env for e in unique_by_identity[label].values())
            negatives=size-id_counts[label];camera_positives=id_counts[label]-camera_counts[(label,r['camera'])]
            ordinary=id_counts[label]-1;assert 0<=distinct<=positives<=ordinary and negatives>0
            value=Counter(anchor_positions=1,ordinary_positive_positions=ordinary,
                legal_positive_positions=positives,legal_negative_positions=negatives,
                legal_directed_triplets=positives*negatives,
                anchors_with_legal_positive=int(positives>0),anchors_with_multiple_positive_positions=int(positives>=2),
                anchors_with_multiple_distinct_positive_records=int(distinct>=2),
                inventory_supported_anchor_positions=int(label in inventory_cross),
                camera_proxy_supported_anchor_positions=int(camera_positives>0),
                camera_proxy_false_support=int(camera_positives>0 and positives==0),
                camera_proxy_missed_support=int(camera_positives==0 and positives>0))
            positive_hist[positives]+=1;unique_hist[distinct]+=1
            totals.update(value);per_epoch[batch['epoch']].update(value);per_identity[label].update(value);batch_total.update(value)
            observed_records.add(name)
        batch_counts.append(dict(epoch=batch['epoch'],batch=batch['batch'],
            anchors_with_legal_positive=batch_total['anchors_with_legal_positive'],
            anchors_with_multiple_distinct_positive_records=batch_total['anchors_with_multiple_distinct_positive_records']))
        totals.update({k:v for k,v in batch_total.items() if k in ('batches','duplicate_record_positions')})
    steps=sum(r['steps'] for r in training['history']);assert len(batch_keys)==steps and set(per_epoch)==set(range(1,51))
    assert all(sum(k[0]==h['epoch'] for k in batch_keys)==h['steps'] for h in training['history'])
    anchors=totals['anchor_positions'];assert anchors==steps*control['initializer']['batch_size']
    eligible_counts=[b['anchors_with_legal_positive'] for b in batch_counts]
    observed_supported={label for label,value in per_identity.items() if value['anchors_with_legal_positive']>0}
    assert observed_supported<=inventory_cross
    summary=dict(dataset=dataset,environment=environment,formal_updates=steps,
        inventory_records=len(records),inventory_identities=len(grouped),
        inventory_cross_environment_identities=len(inventory_cross),
        observed_distinct_records=len(observed_records),observed_identities=len(per_identity),
        observed_supported_identities=len(observed_supported),
        supported_anchor_fraction=totals['anchors_with_legal_positive']/anchors,
        multiple_distinct_positive_anchor_fraction=totals['anchors_with_multiple_distinct_positive_records']/anchors,
        supported_among_inventory_supported_anchor_fraction=totals['anchors_with_legal_positive']/totals['inventory_supported_anchor_positions'],
        cross_environment_fraction_of_ordinary_positive_positions=totals['legal_positive_positions']/totals['ordinary_positive_positions'],
        zero_support_batches=sum(v==0 for v in eligible_counts),
        legal_anchor_count_per_batch=dict(min=min(eligible_counts),median=statistics.median(eligible_counts),max=max(eligible_counts)),
        totals=dict(totals),legal_positive_position_histogram=dict(sorted(positive_hist.items())),
        distinct_legal_positive_record_histogram=dict(sorted(unique_hist.items())),
        epoch_supported_anchor_fractions={e:v['anchors_with_legal_positive']/v['anchor_positions'] for e,v in sorted(per_epoch.items())},
        per_identity={label:dict(v) for label,v in sorted(per_identity.items())})
    results.append(summary)
    for p in (batch_path,training_path,protocol_path):assert sha(p)==input_sha[str(p)]
assert sha(seal_path)==input_sha[str(seal_path)]
print(json.dumps(dict(status='COMPLETE_SEALED_RAW50_LABEL_CENSUS',at=datetime.now().astimezone().isoformat(),
    input_sha256=input_sha,rows=results,model_forwards=0,optimizer_updates=0,
    boundary='All50 historical RAW semantic batches, label/protocol-only feasibility census. Directed position pairs and distinct positive records reported separately; all different identities remain legal negatives. MSVR uses scene from protocol, not logged camera/view. No loss, hard-negative ordering, repair/keep partition or causal effect measured; no current training/GPU/source/weights/evaluation changes. Current independent-head batch-order identity is only established upon its completed report, not assumed here.')))
