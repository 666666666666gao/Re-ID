from datetime import datetime
from pathlib import Path
import hashlib
import json

packet=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/native_fixed_best812_complete')
record=json.loads((packet/'stdout.json').read_bytes())
assert record['status']=='SIX_FIXED_BEST_READ_ONLY_DIAGNOSES_VERIFIED'
rows=[]
for dataset in ('RGBNT201','MSVR310','RGBNT100'):
 for variant in ('semantic','native'):
  source=packet/'texts'/f'{dataset}_{variant}_DIAGNOSIS.json'
  report=json.loads(source.read_bytes())
  assert report['status']=='COMPLETE'
  scores=report['scores'];comparisons=report['comparisons'];query=report['diagnostic']['query']
  rows.append({'dataset':dataset,'variant':variant,'epoch':report['selected_epoch'],
   'metrics':{key:value['metrics'] for key,value in scores.items()},
   'same_model_correction':comparisons['same_model_global_to_fused']['delta_metrics'],
   'same_model_global_vs_independent':comparisons['independent_global_only_to_same_model_global']['delta_metrics'],
   'total_vs_independent':comparisons['independent_global_only_to_fused']['delta_metrics'],
   'correction_rank1_repairs':comparisons['same_model_global_to_fused']['rank1_repairs'],
   'correction_rank1_new_errors':comparisons['same_model_global_to_fused']['rank1_new_errors'],
   'gain':report['readout_gain'],
   'scaled_correction_global_norm_ratio_mean':query['actual_scaled_correction_to_global_norm_ratio']['mean'],
   'scaled_correction_global_norm_ratio_median':query['actual_scaled_correction_to_global_norm_ratio']['percentiles_0_25_50_75_95_100'][2],
   'global_fused_angle_mean_degrees':query['global_to_fused_angle_degrees']['mean'],
   'native_reader_norm_mean':query['actual_reader_output_norm']['mean'] if variant=='native' else None,
   'native_reader_semantic_anchor_ratio_mean':query['reader_to_derived_semantic_plus_anchor_norm_ratio']['mean'] if variant=='native' else None,
   'input_checkpoint_sha256':report['input_checkpoint_sha256'],'diagnosis_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
result={'status':'COMPLETE_FIXED_BEST_DESCRIPTIVE_ANALYSIS','at':datetime.now().astimezone().isoformat(),
 'rows':rows,'input_observer_at':record['at'],'completed_at':record['campaign']['completed_at'],
 'source_files_verified':record['source_files_verified'],'original_artifacts_verified':record['original_artifacts_verified'],
 'boundary':'Six selected seed42 models on already-consumed official benchmarks. Within-model correction/global comparisons are descriptive, not independently trained ablations or causal attribution. No optimizer updates, new checkpoint selection, hyperparameter tuning, environment change or parity repair.'}
target=packet/'ANALYSIS.json';assert not target.exists()
target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
table='| 数据集 | 条件 | best轮 | 独立global-only mAP/R1 | 同模型global mAP/R1 | fused mAP/R1 | 修正ΔmAP/R1 | global差ΔmAP/R1 |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
fmt=lambda metrics:f'{metrics["mAP"]:.4f}/{metrics["Rank-1"]:.4f}'
for row in rows:
 table+=f'| {row["dataset"]} | {row["variant"]} | {row["epoch"]} | {fmt(row["metrics"]["independent_global_only"])} | {fmt(row["metrics"]["global"])} | {fmt(row["metrics"]["fused"])} | {fmt(row["same_model_correction"])} | {fmt(row["same_model_global_vs_independent"])} |\n'
text='# 固定 best 全量诊断\n\n'+table+'\n'
text+='每行全部CMC跟随原同一mAP-best；原模型、buffer和61项输入依赖SHA前后不变，六端fused复现原计分。完整四项、所有query AP/首正例、身份变化及query/gallery分布保留在原DIAGNOSIS.json。\n\n'
text+='| 数据集/条件 | 实际gain | mean ||gain*c||/||g|| | g→f角度均值(度) | 修正首位修复/新增错误 | native出口范数均值 | detail/(语义+anchor)范数比均值 |\n|---|---:|---:|---:|---:|---:|---:|\n'
for row in rows:
 native=lambda key:'—' if row[key] is None else f'{row[key]:.6f}'
 text+=f'| {row["dataset"]}/{row["variant"]} | {row["gain"]:.6f} | {row["scaled_correction_global_norm_ratio_mean"]:.6f} | {row["global_fused_angle_mean_degrees"]:.6f} | {row["correction_rank1_repairs"]}/{row["correction_rank1_new_errors"]} | {native("native_reader_norm_mean")} | {native("native_reader_semantic_anchor_ratio_mean")} |\n'
text+='\n边界：两项mAP差相加等于总差，是算术关系，不是梯度因果分解。同模型global不等于独立global-only。幅度不证明信息质量，correction单独分数不证明互补，单个best不是种子稳定性。无测试时模型更新、M0重放、参数/精度/环境/选择规则更改。\n'
(packet/'ANALYSIS.md').write_text(text,encoding='utf-8')
print(json.dumps(result))
