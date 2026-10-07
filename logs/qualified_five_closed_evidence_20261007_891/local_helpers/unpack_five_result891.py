from pathlib import Path
from datetime import datetime
import hashlib
import json

b = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
packet = b / 'qualified_five_full_complete889'
r = json.loads((packet / 'REMOTE.json').read_bytes())
assert r['formal_completed'] == 5 and r['formal_epochs'] == 250 and r['formal_steps'] == 9839
assert len(r['pairs']) == 12 and not any(p['phase_progress'] for p in r['pairs'])
folder = packet / 'primary_text'
assert not folder.exists()
folder.mkdir()
mapping = {}
for index, (name, item) in enumerate(r['files'].items()):
    p = folder / f'{index:04d}_{Path(name).name}'
    p.write_bytes(item['text'].encode('utf-8'))
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], name
    mapping[name] = {'local_file': str(p), 'sha256': item['sha256'], 'bytes': p.stat().st_size}
(packet / 'FILE_MAP.json').write_bytes((json.dumps(mapping, indent=2) + '\n').encode('utf-8'))
summary_path = next(Path(v['local_file']) for n, v in mapping.items() if n.endswith('/SUMMARY.json') and '/results/' in n)
summary = json.loads(summary_path.read_bytes())
rows = []
for row in r['rows']:
    pairs = {p['control']: p for p in summary['pairs'] if p['dataset'] == row['dataset'] and p['objective'] == row['variant']}
    rows.append({'dataset': row['dataset'], 'objective': row['variant'], 'best_epoch': row['best_epoch'],
        'metrics': row['metrics'], 'last_epoch_metrics': row['last_epoch_metrics'],
        'best_to_last_map_drop': row['best_to_last_map_drop'], 'formal_steps': row['formal_steps'],
        'delta_semantic': pairs['raw_semantic']['paired_diagnosis']['delta_metrics'],
        'delta_global': pairs['raw_global_only']['paired_diagnosis']['delta_metrics'],
        'pair_diagnosis_keys': list(pairs['raw_semantic']['paired_diagnosis'])})
v = {'status': 'FIVE_FORMAL_AND_TWELVE_FULL_QUERY_PAIRS_RECEIVED', 'rows': rows, 'primary_comparisons': 5,
    'primary_advances': sum(p['phase_progress'] for p in summary['pairs'] if p['control'] == 'raw_semantic'),
    'all_comparisons': 12, 'all_advances': sum(p['phase_progress'] for p in summary['pairs']), 'missing': r['missing'],
    'free_bytes_observed': r['free_bytes'], 'primary_summary': str(summary_path), 'source_sha_physically_verified_count': 420}
(packet / 'LOCAL_RECEIPT_SUMMARY.json').write_bytes((json.dumps(v, indent=2) + '\n').encode('utf-8'))
state_path = b / 'qualified_five_full_state888.json'
state = json.loads(state_path.read_bytes())
state['status'] = 'FIVE_FORMAL_COMPLETE_EXIT0_PRIMARY_RECEIVED'
state['phase'] = 'Five formal results and12pairs complete; audit and claim review pending;100repair remains missing'
state['last_turn_classification'] = 'ACTUAL_TERMINAL_PROGRESS'
state['original_terminal'] = r['actual_parent_exit']
state['terminal_primary_packet'] = str(packet)
state['exact_source_packet'] = str(b / 'qualified_five_audit_sources891')
state['last_progress_recorded_at'] = datetime.now().astimezone().isoformat()
state['next_action'] = 'Fresh same-family provisional integrity audit, then result-to-claim review. Publish main handoff plus full primary text, CPU M0support889 and literature890 after review; preserve420source bytes and10unrelated protectedfiles. No NN rerun or numerical rescue. GoalACTIVE_UNMET.'
state_path.write_bytes((json.dumps(state, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
print(json.dumps(v, indent=2))
