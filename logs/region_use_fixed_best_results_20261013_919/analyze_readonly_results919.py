import csv
import hashlib
import json
from pathlib import Path

work=Path(__file__).resolve().parent
intake=work/'readonly_results919'
result=work/'readonly_summary919'
assert not result.exists()
result.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((intake/'INTAKE_MANIFEST_919.json').read_text(encoding='utf-8'))
assert all(sha(intake/name)==h for name,h in manifest['file_sha256'].items())
histories={json.loads(p.read_text(encoding='utf-8'))['dataset']:json.loads(p.read_text(encoding='utf-8')) for p in intake.rglob('training.json')}
keys=('mAP','Rank-1','Rank-5','Rank-10')
rows=[]
for dataset in ('RGBNT201','RGBNT100','MSVR310'):
 for condition in ('uniform','ambiguity'):
  name=f'logs/region_use_fixed_best_diagnosis_20261013_918/{dataset}_{condition}/DIAGNOSIS.json'
  d=json.loads((intake/name).read_text(encoding='utf-8'))
  assert d['status']=='COMPLETE' and d['optimizer_updates']==0
  assert d['model_state_before_sha256']==d['model_state_after_sha256']
  assert d['original_fused_distance_max_abs_difference']<=1e-5
  h=histories[dataset]
  same_epoch=next(e for e in h['history'] if e['epoch']==d['selected_epoch'])['official_fused']
  best_epoch=next(e for e in h['history'] if e['epoch']==h['best_epoch'])['official_fused']
  g=d['scores']['global']['metrics']
  row=dict(dataset=dataset,condition=condition,epoch=d['selected_epoch'],diagnosis_path=name,diagnosis_sha256=sha(intake/name),
   metrics={n:s['metrics'] for n,s in d['scores'].items()},
   same_model_global_minus_independent_same_epoch={k:g[k]-same_epoch[k] for k in keys},
   independent_global_best_epoch=h['best_epoch'],independent_global_same_epoch=same_epoch,independent_global_best=best_epoch,
   comparisons=d['comparisons'],query_count=len(d['scores']['global']['average_precision']),
   query_correction_global_ratio=d['geometry']['query']['scaled_correction_global_ratio']['mean'],
   query_global_fused_angle=d['geometry']['query']['global_fused_angle_degrees']['mean'],
   model_state_equal=True,fused_distance_difference=d['original_fused_distance_max_abs_difference'],distance_artifact=d['distance_artifact'])
  rows.append(row)
value=dict(status='SIX_ORIGINAL_BEST_FIVE_SCORE_DIAGNOSIS_COMPLETE',rows=rows,
 optimizer_updates=0,training_stopped=True,goal_status='ACTIVE_UNMET',
 boundary='Five descriptive fixed-best scores; original seed42 development benchmarks. No fitting, sweep, teacher, deployment or compression claim. Independent same-epoch history is text-metric comparison, not a newly replayed independent checkpoint or feature equality.')
(result/'SUMMARY.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
with (result/'SCORES.csv').open('w',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['dataset','condition','epoch','score',*keys,'delta_mAP_vs_same_global','delta_R1_vs_same_global'])
 writer.writeheader()
 for r in rows:
  for n,m in r['metrics'].items():
   g=r['metrics']['global']
   writer.writerow(dict(dataset=r['dataset'],condition=r['condition'],epoch=r['epoch'],score=n,**m,
     delta_mAP_vs_same_global=m['mAP']-g['mAP'],delta_R1_vs_same_global=m['Rank-1']-g['Rank-1']))
lines=['# Frozen best: five-score evidence-use diagnosis','',
 'All six original selected bests were evaluated once, with no optimizer updates or model/buffer changes. Training remains stopped. The six original fused distance matrices replayed with maximum absolute difference 0; original metrics agreed within the registered tolerance. Only physical GPU0/1 on2026 were used. The controller completed at2026-10-13T06:42:30.084846+08:00 and the actual supervisor EXIT was0 at06:42:30.530957. Both controller and supervisor, and all six children, were absent at the06:47 observation.', '',
 'Scores below use the same best and the same complete legal query/gallery. G: same-model global1536 cosine. C: correction1536 cosine. F: original1536 additive deployment. R: symmetric meanMaxSim over the actual final16 regions, separately in all nine role/modality spaces. J: the fixed existing training score0.5*(G cosine+R); its coefficient was not fitted.', '',
 '| Dataset | Condition | Epoch | G mAP | C mAP | F mAP | R mAP | J mAP | F−G | J−G |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 m=r['metrics'];g=m['global']['mAP']
 lines.append(f"| {r['dataset']} | {r['condition']} | {r['epoch']} | "+' | '.join(f"{m[k]['mAP']:.4f}" for k in ('global','correction','fused','regions','training_pairwise'))+f" | {m['fused']['mAP']-g:+.4f} | {m['training_pairwise']['mAP']-g:+.4f} |")
lines+=['','Full mAP/R1/R5/R10 values for all30 scores are in SCORES.csv; rows never mix epochs.', '',
 '| Dataset | Condition | F−G R1 | F first repairs/new errors | Correction/G norm ratio | G→F degrees | Independent global epoch | Same-G minus independent same-epoch mAP |',
 '|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
 c=r['comparisons']['fused']
 lines.append(f"| {r['dataset']} | {r['condition']} | {c['delta_metrics']['Rank-1']:+.4f} | {c['rank1_repairs']}/{c['rank1_new_errors']} | {r['query_correction_global_ratio']:.6f} | {r['query_global_fused_angle']:.4f} | {r['epoch']} | {r['same_model_global_minus_independent_same_epoch']['mAP']:+.10f} |")
lines+=['', '1. In all six cases the existing fixed training-pairwise comparator underperforms the same-model global in mAP and Rank-1. This diagnosis does not establish a stronger local teacher or evidence that additive compression is the primary bottleneck. It constrains this exact region/comparator combination; it does not exclude every local comparison or prove that the regions contain no complementary information.',
 '2. RGBNT201 has a small positive additive mAP increment, RGBNT100 loses mAP and Rank-1, and MSVR310 is almost unchanged in mAP while losing one first-rank query. No stable three-dataset deployed benefit has been established.',
 '3. The current same-model global metrics match the independent global history at the selected epoch up to the original floating-point reporting tolerance. In RGBNT100 the relevant comparison is independent global epoch26, not its independently selected epoch7 best. Therefore the gap to independent best must not all be described as shared-backbone harm. Text metric agreement is not proof that all features or weights are identical.',
 '4. Correction and regional standalone discrimination do not guarantee useful complement. Their lower standalone mAP is also insufficient to prove that all added evidence is useless. The report includes same-model paired repairs, new errors and AP contributions so net benefit is assessed directly.', '',
 '| Dataset | Condition | Repair AP contribution | New-error contribution | Kept-correct contribution | Kept-wrong contribution | F−G total |',
 '|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 groups=r['comparisons']['fused']['rank1_groups']
 lines.append(f"| {r['dataset']} | {r['condition']} | "+' | '.join(f"{groups[k]['contribution_to_total_mAP_points']:+.6f}" for k in ('repair','new_error','kept_correct','kept_wrong'))+f" | {r['comparisons']['fused']['delta_metrics']['mAP']:+.6f} |")
lines+=['', 'Research direction: study whether added evidence changes actual final identity relations usefully. External methods address one demonstrated gap at a time. These results do not justify automatically launching late interaction, distillation, a new gate, gain/seed searches, or another training queue. Current training and NN diagnosis are complete; retain the original bests, first strict results, pinned diagnostic distances and historical failures.', '',
 'Scope: single seed42, consumed development benchmarks. The pairwise R/J scores require pairwise local comparison and are not the1536 deployment result or a new main result. Source and terminal receipts establish execution checks; raw image labels and distance sorting have not been independently re-audited by this summarizer. A fresh same-family provisional integrity review is separate.']
(result/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(status=value['status'],rows=len(rows),score_rows=30,
 max_same_epoch_global_metric_difference=max(abs(v) for r in rows for v in r['same_model_global_minus_independent_same_epoch'].values()),
 files={p.name:sha(p) for p in result.iterdir()})))
