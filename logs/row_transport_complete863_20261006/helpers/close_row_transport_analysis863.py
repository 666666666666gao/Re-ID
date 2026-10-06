from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import math

base = Path('C:/Users/gb/.codex_tmp')
formal = json.loads((base/'rt_complete862/remote_stdout.json').read_bytes())
fixed = json.loads((base/'fb_complete863/remote_stdout.json').read_bytes())
out = base/'independent_evidence_draft/row_transport_complete_analysis863'
assert not out.exists()
assert formal['supervisor_exit']['exit_code'] == 0
assert formal['report']['accepted'] == 6
assert formal['report']['formal_epochs'] == 300
assert formal['report']['formal_steps'] == 12968
assert fixed['exit']['exit_code'] == 0
assert fixed['nn_summary']['accepted_models'] == 6
assert fixed['cpu_summary']['all_query_comparisons'] == 24
assert all(not p['phase_progress'] and p['actual_training_batch_order_equal'] for p in formal['report']['pairs'])
out.mkdir()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def write_csv(name, rows):
    with (out/name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

rows = []
pairs = []
comparisons = []
for row in formal['report']['rows']:
    key = (row['dataset'], row['variant'])
    diagnostic = next(r for r in fixed['cpu_summary']['rows'] if (r['dataset'], r['variant']) == key)
    nn = next(r for r in fixed['nn_summary']['rows'] if (r['dataset'], r['variant']) == key)
    original = next(m for m in nn['modes'] if m['mode'] == 'original')
    assert original['metrics'] == row['metrics']
    stats = diagnostic['full_query_gallery_measurements']
    own = diagnostic['pairs']['own_global__original']
    opposite = diagnostic['pairs']['original__opposite']
    peer_zero = diagnostic['pairs']['original__self_only']
    correction = diagnostic['pairs']['own_global__own_correction']
    training = diagnostic['training']
    # Both formal receipts and exact CPU array scorings are retained separately.
    item = dict(dataset=key[0], mode=nn['original_mass_mode'], best_epoch=row['best_epoch'],
                **row['metrics'], own_global_mAP=own['control_metrics']['mAP'],
                correction_only_mAP=correction['candidate_metrics']['mAP'],
                fused_minus_own_global_mAP=own['delta_metrics']['mAP'],
                fused_minus_own_global_R1=own['delta_metrics']['Rank-1'],
                fused_rank1_repairs=own['rank1_repairs'], fused_rank1_new_errors=own['rank1_new_errors'],
                opposite_minus_original_mAP=opposite['delta_metrics']['mAP'],
                peer_zero_minus_original_mAP=peer_zero['delta_metrics']['mAP'],
                r_mean=stats['real_mass_mean']['mean'], r_std_mean=stats['real_mass_std']['mean'],
                conditional_entropy_mean=stats['conditional_entropy']['mean'],
                conditional_entropy_uniform=math.log(16),
                conditional_max_column_share_mean=stats['conditional_max_column_share']['mean'],
                effective_peer_write_norm_mean=stats['effective_peer_write_slot_norm_mean']['mean'],
                private_slot_norm_mean=stats['private_slot_norm_mean']['mean'],
                scaled_correction_global_ratio_mean=stats['actual_scaled_correction_global_ratio']['mean'],
                global_fused_angle_degrees_mean=stats['global_fused_angle_degrees']['mean'],
                **nn['weight_statistics'])
    rows.append(item)
    for name, comparison in diagnostic['pairs'].items():
        comparisons.append(dict(dataset=key[0], mode=nn['original_mass_mode'], pair=name,
                                **comparison['delta_metrics'],
                                rank1_repairs=comparison['rank1_repairs'],
                                rank1_new_errors=comparison['rank1_new_errors'],
                                identity_macro_delta_ap= comparison['identity_macro_delta_ap_points']['mean']))
for pair in formal['report']['pairs']:
    d = pair['paired_diagnosis']
    pairs.append(dict(dataset=pair['dataset'], candidate=pair['candidate'], control=pair['control'],
                      **d['delta_metrics'], rank1_repairs=d['rank1_repairs'],
                      rank1_new_errors=d['rank1_new_errors'],
                      identity_macro_delta_ap=d['identity_macro_mean_delta_ap_points'],
                      batch_order_equal=pair['actual_training_batch_order_equal'],
                      phase_progress=pair['phase_progress']))
write_csv('six_formal_and_fixed_best.csv', rows)
write_csv('fifteen_formal_comparisons.csv', pairs)
write_csv('twenty_four_fixed_best_comparisons.csv', comparisons)
assert len(rows) == 6 and len(pairs) == 15 and len(comparisons) == 24
opposites = [r for r in comparisons if r['pair'] == 'original__opposite']
assert all(r[k] == 0 for r in opposites for k in ('Rank-1','Rank-5','Rank-10'))
facts = dict(schema='row-transport-complete-analysis-v1', at=datetime.now().astimezone().isoformat(),
             formal_accepted=6, formal_epochs=300, formal_updates=12968,
             primary_pass=0, all_fifteen_pass=0, fixed_models=6, fixed_modes=18,
             fixed_comparisons=24, max_abs_opposite_mAP=max(abs(r['mAP']) for r in opposites),
             all_opposite_CMC_deltas_zero=True,
             max_abs_peer_zero_mAP=max(abs(r['mAP']) for r in comparisons if r['pair']=='original__self_only'),
             original_formal_summary_sha256=hashlib.sha256((base/'rt_complete862/received/results/row_mass_role_transport_v2_complete_20261006_860/SUMMARY.json').read_bytes()).hexdigest(),
             fixed_summary_sha256=fixed['cpu_summary']['diagnosis_summary_sha256'],
             boundary='One consumed-benchmark seed42 campaign; fixed interventions do not retrain or remove complete roles. No calibration, semantic correspondence truth, seed variance, or SOTA claim.')
write_json(out/'FACTS.json', facts)
write_json(out/'FULL_FIXED_MEASUREMENTS.json', fixed['cpu_summary'])
formal_table = '\n'.join('| '+r['dataset']+' | '+r['mode']+' | '+str(r['best_epoch'])+' | '+
                          ' / '.join(f'{r[k]:.6f}' for k in ('mAP','Rank-1','Rank-5','Rank-10'))+' |' for r in rows)
read_table = '\n'.join('| '+r['dataset']+' | '+r['mode']+' | '+
                       ' | '.join(f'{r[k]:.6f}' for k in ('r_std_mean','conditional_entropy_mean','scaled_correction_global_ratio_mean','fused_minus_own_global_mAP'))+' |' for r in rows)
report = f'''# Row transport complete evidence — 2026-10-06

All six fresh 50-epoch endpoints and first strict reloads completed; 300 epochs / 12,968 updates. All 15 paired training batch orders match. The slot-versus-uniform primary gate passes 0/3, all comparisons 0/15. The fixed-best diagnosis is complete: six models, 18 deployment modes, 24 full-query comparisons.

| Dataset | Mode | Best epoch | mAP / R1 / R5 / R10 |
|---|---|---:|---|
{formal_table}

## Observations and limits

1. Switching allocation in the same best model changes mAP by at most {facts['max_abs_opposite_mAP']:.9f} percentage points; all six R1/R5/R10 values stay unchanged. This does not explain the separately trained RGBNT100 gap of 0.477150 mAP.
2. Removing only the effective peer write changes mAP by at most {facts['max_abs_peer_zero_mAP']:.9f} points. It retains private Mamba, CNN/Transformer, readout and existing shared context. This is not a retrained deletion of all cross-modal computation.
3. RGBNT201 conditional assignments are nearly uniform (entropy approximately ln(16)=2.772589), and receiving-slot mass variation is tiny. RGBNT100 has more varied masses and stronger writes, yet its fused representation remains worse than its own global: -0.547380/-0.070231 mAP for slot/uniform. Neither branch activity nor correction-only recognition proves useful complementarity.
4. Matching-matrix norms shrink sharply in RGBNT201/RGBNT100. The records do not isolate task gradients from weight decay across training, so no unique causal explanation is claimed.
5. RGBNT100 own globals are 83.965452 mAP, below the independent global-only 84.533784. Duty separation does not establish identical global trajectories; the cause of this difference is not identified here.

| Dataset | Mode | Mean within-object mass std | Conditional entropy | Scaled correction/global | Fused − own global mAP |
|---|---|---:|---:|---:|---:|
{read_table}

## Execution provenance

The initial fixed-best process exited 1 at 09:31:57 after four models/12 modes. Its shared YACS configuration carried MSVR steps (20,40) into RGBNT100, whose YAML omits that key. A CPU replay shows fresh RGBNT100 uses (40,70) and exactly matches its initializer dump. Original formal training used fresh processes and is unaffected. Only the two missing RGBNT100 models were resumed, each in a fresh process with unchanged scientific source and original strict gates. Both exited 0; the sole CPU report exited 0 at 10:00:00. Original failure receipts and the old four-model progress file remain historical, not rewritten as successful.

The original 50-epoch traces, all query/identity changes, full measurement distributions, checkpoint/distance/source hashes and failure provenance are retained. Arrays and weights remain remote. Fixed-model identity summaries cannot replace complete training seeds; official test data has participated in development.

## Next question

Do not promote row allocation as an effective main module or tune null/gain/LR/seed from these official results. Prioritize the formation of identity-relevant regional content. A next intervention must have a direct active control and preserve the matched raw-duty recipe; no simultaneous N2/N3 stack or claim of three validated contributions.

Overall Goal remains ACTIVE/UNMET.
'''
(out/'ANALYSIS.md').write_text(report, encoding='utf-8')
live_path=base/'independent_evidence_draft/fixed_best_row_transport_live862.json'
live=json.loads(live_path.read_bytes())
live.update(status='COMPLETE', phase='SIX_FIXED_BESTS_EIGHTEEN_MODES_AND_SOLE_CPU_REPORT_TERMINAL_EXIT0',
            active_nn=None, next_observation_due=None, completed_diagnostic_models=6, completed_modes=18,
            terminal_intake=str(base/'fb_complete863'), closed_at=fixed['exit']['completed_at'],
            last_consumed_handle=dict(session=25987,exec_cell=235,exit_code=0),
            analysis=str(out), observation_policy='No live observer. Do not rerun old deploy/resume/collectors or four accepted models.')
write_json(live_path,live)
with Path('C:/Users/gb/memory/2026-10-06.md').open('a',encoding='utf-8') as stream:
    stream.write('\n\nTriFusion '+facts['at']+': original row-mass six fresh50/300epochs/12968updates/first strict and sole15-pair report closed EXIT0; primary0/3/all0/15. Fixed-best diagnostic old process EXIT1 cfg carryover preserved; CPU reproduction proves steps20,40 carryover vs fresh10040,70. Only missing100slot/uniform fresh processes completed0 and sole CPU report0 at10:00;6models/18modes/24pairs closed. Original/diagnostic intakes complete once467+156 texts/no arrays locally; free3,854,245,888B. Same-weight opposite maxabs mAP.001059942/allCMC0; peerzero maxabs.030482853, not complete role deletion. 201 nearuniformQ;100 stronger varied messages still damages ownG. All handles consumed/no live observer/NN; publication and qualified retirement pending. OverallGoal ACTIVE_UNMET,26physical0/1 only/no power-temperature/25 actions; foreign10 untouched.\n')
print(json.dumps(facts, ensure_ascii=False, indent=2))
