"""Retain and clarify the inherited scope sentence in the third intake."""
from pathlib import Path
from datetime import datetime
import hashlib
import json

private = Path('C:/Users/gb/.codex_tmp/training_feature_scale_split755')
path = private/'third_full_intake759/INTAKE.json'
original = json.loads(path.read_bytes())
assert 'Paired other condition/final report unavailable' in original['boundary']
assert (private/'second_full_intake759/INTAKE.json').is_file()
correction = {'status':'INTAKE_SCOPE_SENTENCE_CLARIFIED','recorded_at':datetime.now().astimezone().isoformat(),
              'original_intake_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
              'original_sentence':'Paired other condition/final report unavailable',
              'correct_scope':'Both RGBNT201 normalized and metric_raw original formal50/strict results have now been received; the other two dataset pairs and the sole six-arm report remain incomplete.',
              'boundary':'Editorial inherited-template error only. Original intake/text/hash/model/metrics unchanged; no replay, rescore, tolerance change or scientific gate decision.'}
output = private/'INTAKE_SCOPE_CORRECTION759.json'
assert not output.exists()
output.write_text(json.dumps(correction, indent=2)+'\n', encoding='utf-8')
print(json.dumps(correction, indent=2))
