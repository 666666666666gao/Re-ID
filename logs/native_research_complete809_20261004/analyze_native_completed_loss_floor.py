"""Analyze recorded author losses; no model imports or new experimental scores."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import re
import statistics
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
target = private / 'native_research_completed_loss_floor'
assert not target.exists()
scope = json.loads((repo / 'refine-logs/native_research_v6/SOURCE_SCOPE.json').read_bytes())['source_sha256']
cached = private / 'native_original_control785_terminal_intake/_source'
source_paths = {
    'softmax': cached / 'comparators/Signal-cd1b0a6/layers/softmax_loss.py',
    'triplet': cached / 'comparators/Signal-cd1b0a6/layers/triplet_loss.py',
    'make_loss': cached / 'comparators/Signal-cd1b0a6/layers/make_loss.py',
    'sum_heads': repo / 'tools/run_foundation_recipe.py',
    'loss_wrapper': repo / 'tools/run_independent_native_evidence.py',
}
source_bindings = {}
texts = {}
for name, path in source_paths.items():
    relative = path.relative_to(cached if path.is_relative_to(cached) else repo).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == scope[relative]
    source_bindings[relative] = {'path':path.as_posix(),'sha256':digest}
    texts[name] = path.read_text(encoding='utf-8')
assert 'epsilon=0.1' in texts['softmax']
assert 'targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes' in texts['softmax']
assert 'loss = (- targets * log_probs).mean(0).sum()' in texts['softmax']
assert 'triplet = TripletLoss()' in texts['make_loss'] and 'xent = CrossEntropyLabelSmooth(num_classes=num_classes)' in texts['make_loss']
assert 'self.ranking_loss = nn.SoftMarginLoss()' in texts['triplet']
assert "return sum(components), {'head_losses': [float(item.detach()) for item in components]}" in texts['sum_heads']
trace = json.loads((private / 'native_research_closed_eight_training_traces/ANALYSIS.json').read_bytes())
head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert trace['publication_head'] == head == '10b77166c655b550cfd951c38b6e93d9d1a64324'
archives = {
    ('RGBNT201','global_only'): 'native_research_first_full800_20261003',
    ('RGBNT201','semantic'): 'native_research_semantic801_20261004',
    ('RGBNT201','native'): 'native_research_first_dataset802_20261004',
    ('MSVR310','global_only'): 'native_research_msvr_global803_20261004',
    ('MSVR310','semantic'): 'native_research_msvr_semantic804_20261004',
    ('MSVR310','native'): 'native_research_second_dataset805_20261004',
    ('RGBNT100','global_only'): 'native_research_rgb100_global807_20261004',
    ('RGBNT100','semantic'): 'native_research_rgb100_semantic808_20261004',
}
def scalar(cfg, key):
    values = re.findall(r'^\s+' + re.escape(key) + r':\s*(.+)$',cfg,re.MULTILINE)
    assert len(values) == 1, key
    return values[0]
rows = []
for completed in trace['rows']:
    dataset, variant = completed['dataset'], completed['variant']
    archive = repo / 'logs' / archives[dataset,variant]
    prefix = f'native_research_v6_20261003_794_full_{variant}_{dataset}'
    paths = {name:archive/f'{prefix}_{name}' for name in ('training.json','training_steps.jsonl','official_metrics.json')}
    assert all(hashlib.sha256(path.read_bytes()).hexdigest()==completed['source_text_sha256'][name] for name,path in paths.items())
    training = json.loads(paths['training.json'].read_bytes())
    binding = training['initializer']
    cfg = binding['cfg_yaml']
    assert scalar(cfg,'IF_LABELSMOOTH') == "'on'" and scalar(cfg,'NO_MARGIN') == 'true'
    assert scalar(cfg,'SAMPLER') == 'softmax_triplet' and scalar(cfg,'METRIC_LOSS_TYPE') == 'triplet'
    id_weight, triplet_weight = float(scalar(cfg,'ID_LOSS_WEIGHT')), float(scalar(cfg,'TRIPLET_LOSS_WEIGHT'))
    assert id_weight > 0 and triplet_weight > 0
    protocol_path = repo / f'logs/training_feature_scale_protocols_20261002/{dataset}.json'
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == binding['protocol_sha256']
    protocol = json.loads(protocol_path.read_bytes())
    classes = len(protocol['train_label_map'])
    assert sorted(protocol['train_label_map'].values()) == list(range(classes)) and classes > 1
    head_count = len(binding['head_names'])
    epsilon = 0.1
    positive = 1-epsilon+epsilon/classes
    negative = epsilon/classes
    entropy = -positive*math.log(positive)-(classes-1)*negative*math.log(negative)
    ce_floor = head_count*id_weight*entropy
    steps = [json.loads(line) for line in paths['training_steps.jsonl'].read_bytes().splitlines()]
    epochs = []
    for epoch in training['history']:
        selected = [step for step in steps if step['epoch']==epoch['epoch']]
        assert len(selected)==epoch['steps'] and all(len(step['head_losses'])==head_count for step in selected)
        assert all(math.isfinite(value) for step in selected for value in step['head_losses'])
        residual = epoch['mean_loss']-ce_floor
        epochs.append({'epoch':epoch['epoch'],'mean_recorded_loss':epoch['mean_loss'],
            'mean_recorded_head_totals':[statistics.mean(step['head_losses'][index] for step in selected) for index in range(head_count)],
            'ideal_weighted_smoothed_ce_floor':ce_floor,
            'loss_above_ideal_floor':residual,
            'sum_unweighted_triplet_loss_upper_bound_in_real_arithmetic':residual/triplet_weight,
            'official_fused':epoch['official_fused']})
    assert len(epochs)==50 and len(steps)==sum(epoch['steps'] for epoch in training['history'])
    rows.append({'dataset':dataset,'variant':variant,'classes':classes,'head_count':head_count,
        'epsilon':epsilon,'id_weight':id_weight,'triplet_weight':triplet_weight,'target_entropy':entropy,
        'ideal_weighted_smoothed_ce_floor':ce_floor,'source_text_sha256':completed['source_text_sha256'],
        'protocol_sha256':binding['protocol_sha256'],'epochs':epochs,'first_epoch':epochs[0],
        'selected_best_epoch':epochs[training['best_epoch']-1],'last_epoch':epochs[-1],
        'best_to_last_map_drop':completed['best_to_last_mAP_drop']})
report = {'status':'EIGHT_COMPLETED_AUTHOR_LOSS_FLOORS_ANALYZED','at':datetime.now().astimezone().isoformat(),
    'publication_head':head,'source_bindings':source_bindings,'rows':rows,
    'identity': 'In exact real arithmetic, L - n_heads*w_ID*H(q) = w_ID*sum KL(q||p_head) + w_triplet*sum soft-margin triplet. Both terms are nonnegative.',
    'boundary': 'Analytical lower bound from actual epsilon/heads/class counts/weights and existing all50-epoch logs, not a new metric or model evaluation. Recorded head totals contain CE plus Triplet; this does not identify either component separately or show Triplet is exactly zero. Finite-precision loss can differ from an ideal mathematical floor at roundoff. No causal scale/style/overfitting mechanism, seed robustness, SOTA, new gate or recipe change; RGBNT100 native remains excluded until formal completion.'}
target.mkdir()
(target/'ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'at':report['at'],'rows':[{'dataset':row['dataset'],'variant':row['variant'],'classes':row['classes'],
    'heads':row['head_count'],'ideal_ce_floor':row['ideal_weighted_smoothed_ce_floor'],
    'best_loss':row['selected_best_epoch']['mean_recorded_loss'],'best_residual':row['selected_best_epoch']['loss_above_ideal_floor'],
    'last_loss':row['last_epoch']['mean_recorded_loss'],'last_residual':row['last_epoch']['loss_above_ideal_floor'],
    'best_to_last_map_drop':row['best_to_last_map_drop']} for row in rows],'boundary':report['boundary']}))
