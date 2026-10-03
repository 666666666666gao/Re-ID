"""Retain the small observed strict-metric difference; no rerun or rescore."""
from pathlib import Path
from datetime import datetime
import json

base = Path('C:/Users/gb/.codex_tmp')
new = base/'training_feature_scale_split755/first_full_intake758'
intake = json.loads((new/'INTAKE.json').read_bytes())
old = base/'training_feature_scale_complete754/raw/trained-model'/f"training_feature_scale_20261003_v2_full_normalized_{intake['dataset']}"/'official_metrics.json'
original = json.loads(old.read_bytes())
delta = {name:intake['metrics'][name]-value for name,value in original['metrics'].items()}
record = {'status':'ORIGINAL_STRICT_METRICS_CONTEXT_COMPARED','recorded_at':datetime.now().astimezone().isoformat(),
          'dataset':intake['dataset'],'current_strict_metrics':intake['metrics'],
          'historical_strict_metrics':original['metrics'],'current_minus_historical':delta,
          'max_abs_difference':max(abs(value) for value in delta.values()),
          'boundary':'Separate original strict receipts, not the equal training-history fields. Small differences preserved without score replay, tolerance changes, checkpoint substitution or unsupported unique numerical-cause claim.'}
output = new/'STRICT_METRIC_COMPARISON.json'
assert not output.exists()
output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(record, indent=2))
