"""Offline comparison of already closed foundation and new global references; no SSH."""
from pathlib import Path
from datetime import datetime
import csv,hashlib,json,subprocess

base=Path('C:/Users/gb/.codex_tmp')
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
output=base/'independent_evidence_draft/selection_global_fidelity867'
assert not output.exists();output.mkdir()
old=repo/'logs/foundation_complete739_20261002/raw/trained-model/foundation_recipe_20261002_v1_full_author_RGBNT201'
intake=base/'independent_evidence_draft/signal_selection_campaign_failure866'
new=intake/'received/trained-model/signal_selection_reference_v1_20261006_866_full_global_only_RGBNT201'
old_protocol=base/'foundation_recipe_v1_20261002/source_intake737/logs/official_three_dataset_protocols_20260923/RGBNT201.json'
new_protocol=repo/'logs/training_feature_scale_protocols_20261002/RGBNT201.json'
observed=base/'independent_evidence_draft/signal_selection_launch867/OBSERVATION.json'
failed=intake/'received/trained-model/signal_selection_reference_v1_20261006_866_full_masked_RGBNT201/training_steps.jsonl'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert head=='858c59868c8c830840eb5836725b7774d87f9e63'
for name in ('training.json','training_steps.jsonl','official_metrics.json'):
    p=old/name;relative=str(p.relative_to(repo)).replace('\\','/')
    assert subprocess.check_output(['git','show',head+':'+relative],cwd=repo)==p.read_bytes()
    p=new/name;relative=str(p.relative_to(intake/'received')).replace('\\','/')
    record=json.loads((intake/'REMOTE.json').read_text())['text_files'][relative]
    assert p.stat().st_size==record['bytes'] and sha(p)==record['sha256']
a=json.loads((old/'training.json').read_text());b=json.loads((new/'training.json').read_text())
pa=json.loads(old_protocol.read_text());pb=json.loads(new_protocol.read_text())
assert sha(old_protocol)==a['initializer']['protocol_sha256']
assert sha(new_protocol)==b['initializer']['protocol_sha256']
assert [k for k in pa if pa[k]!=pb[k]]==['dataset_root']
assert pa['dataset_root']=='/data2/gb/Re-ID/dataset/RGBNT201'
assert pb['dataset_root']=='/data/gaob/Re-ID/dataset/RGBNT201'
assert {k:v for k,v in pa.items() if k!='dataset_root'}=={k:v for k,v in pb.items() if k!='dataset_root'}
initializer_keys=('dataset','recipe','seed','public_clip_sha256','author_source_commit',
    'visual_initial_sha256','camera_initial_sha256','current_head_initial_sha256',
    'initial_model_state_sha256','trainable_parameters','trainable_parameter_tensors','batch_size','num_instances')
assert all(a['initializer'][k]==b['initializer']['plain_foundation_binding'][k] for k in initializer_keys)
sa=[json.loads(line) for line in (old/'training_steps.jsonl').read_text().splitlines()]
sb=[json.loads(line) for line in (new/'training_steps.jsonl').read_text().splitlines()]
assert len(sa)==len(sb)==2649 and sa==sb
assert [r['epoch'] for r in a['history']]==[r['epoch'] for r in b['history']]==list(range(1,51))
assert [{k:v for k,v in r.items() if k!='seconds'} for r in a['history']]==[{k:v for k,v in r.items() if k!='seconds'} for r in b['history']]
ra=json.loads((old/'official_metrics.json').read_text());rb=json.loads((new/'official_metrics.json').read_text())
assert ra['status']==rb['status']=='COMPLETE' and ra['metrics']==rb['metrics']
assert ra['selected_epoch']==rb['selected_epoch']==27
assert ra['distance_sha256']==rb['distance_sha256']
first=json.loads(observed.read_text())
assert first['at']=='2026-10-06T12:11:56.056271+08:00'
rows=[json.loads(line) for line in failed.read_text().splitlines()]
assert len(rows)==53 and all(r['epoch']==1 for r in rows)
failed_mean=sum(r['loss'] for r in rows)/len(rows)
assert failed_mean==first['training'][0]['last_epoch']['mean_loss']
inputs=[old/name for name in ('training.json','training_steps.jsonl','official_metrics.json')]+[new/name for name in ('training.json','training_steps.jsonl','official_metrics.json')]+[old_protocol,new_protocol,observed,failed]
result=dict(status='OFFLINE_LOG_FIDELITY_VERIFIED',completed_at=datetime.now().astimezone().isoformat(),
    input_sha256={str(p):sha(p) for p in inputs},published_head=head,dataset='RGBNT201',
    same_seed=42,common_initializer_fields_equal=list(initializer_keys),
    protocol_difference_only_dataset_root=True,ordered_records_counts_label_map_filter_query_rows_exact=True,
    all_step_fields_exact=2649,all_epoch_fields_except_seconds_exact=50,
    selected_epoch=27,best_metrics=rb['metrics'],recorded_official_distance_sha256_equal=rb['distance_sha256'],
    checkpoint_file_sha256_different=ra['checkpoint_sha256']!=rb['checkpoint_sha256'],
    failed_masked_first53_mean_equals_corrected_epoch1=failed_mean,
    boundary='Already received text only. Old protocol differs solely in dataset_root; does not prove current image bytes or access old2025 files. Logged initial-state hashes and complete losses/LRs/epoch rankings match; old checkpoint tensors not reloaded, so differing file SHA not attributed solely to metadata. Official distance equality is recorded digest consistency, actual old binary not rehashed now. Masked equality only firstepoch aggregate, not per-step gradients/augmentations or full50. Same seed, not independent seeds or original contribution/SOTA.')
(output/'FACTS.json').write_text(json.dumps(result,indent=2)+'\n')
with (output/'COMPLETE50.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=['epoch','steps','mean_loss','mAP','Rank-1','Rank-5','Rank-10','old_training_seconds','new_training_seconds'])
    writer.writeheader()
    for x,y in zip(a['history'],b['history']):
        writer.writerow(dict(epoch=y['epoch'],steps=y['steps'],mean_loss=y['mean_loss'],**y['official_fused'],old_training_seconds=x['seconds'],new_training_seconds=y['seconds']))
(output/'NOTE.md').write_text('''# RGBNT201基础参照的离线全轨迹核对

新26 global-only与历史F1作者来源配方，2649步所有记录字段（epoch、batch、loss、head_losses、lr）及完整50轮除计时外的字段精确相同。共同视觉/camera/heads/model初始状态记录相同；协议仅dataset_root从/data2/gb映射到/data/gaob，其余包含有序records、query_rows、label_map、counts、filter等全部字段相同。

两端同为第27轮mAP-best，73.4728/77.1531/85.8852/89.9522；正式距离的原回执SHA相同，当前26二进制此前实际核验过，旧25仅使用已接收回执而没有重新访问。checkpoint文件SHA不同，未重载旧binary，不能唯一归因于schema/provenance差别。

原masked退出前53步loss的均值4.445760803402595与修后已观察的第1轮均值精确相同。这只核对聚合量，修后逐步记录尚未接收，不能当成梯度/增强字节或完整训练等价性。

这是同seed的基础配方和修订保真证据，不是新增模块增益、多种子、泛化或SOTA。所有分析仅消费本地已有文本，无SSH/NN/新的检索评价，不修改活动实验。公开归档待当前NN/报告闭合后执行。
''',encoding='utf-8')
(output/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','boundary','common_initializer_fields_equal')},indent=2))
