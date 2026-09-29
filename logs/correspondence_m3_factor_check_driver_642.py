import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import traceback

root = Path('/data/gaob/Re-ID/Trifusion')
source = root / 'tools/analyze_correspondence_role_prediction.py'
compile(source.read_text(), str(source), 'exec')
spec = importlib.util.spec_from_file_location('m3_analysis_check', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
checks = []
for values, expected in (
    ((2, 5, 7, 13), (4.5, 6.5, 3)),
    ((2, 5, 7, 10), (3, 5, 0)),
):
    rows = [{'variant': variant, 'metrics': {name: value * (i + 1) for i, name in enumerate(module.METRICS)}}
            for variant, value in zip(module.CONDITIONS, values)]
    actual = module.factor_metrics(list(reversed(rows)))
    for field, value in zip(('mean_address_effect', 'mean_predictor_effect', 'address_predictor_interaction'), expected):
        assert actual[field] == {name: value * (i + 1) for i, name in enumerate(module.METRICS)}
    checks.append({'input_values': values, 'expected_address_predictor_interaction': expected,
                   'computed': actual, 'reversed_input_rows': True})
fixture = root / '.git/correspondence_m3_analysis_642_fixture'
assert not fixture.exists()
fixture.mkdir()
matrix = {'schema': 'trifusion-correspondence-m3-panel-verification-v1', 'expected_endpoints': 12,
          'rows': [{'phase': 'm3', 'dataset': dataset, 'variant': variant,
                    'status': 'UNACCEPTED' if dataset == 'RGBNT201' and variant == 'matched_predictor'
                    else 'VERIFIED_COMPLETE'} for variant in module.CONDITIONS for dataset in module.DATASETS]}
prior = {'expected_endpoints': 18, 'verified_complete': 18, 'rows': [{'status': 'VERIFIED_COMPLETE'}] * 18}
original = {'expected_endpoints': 24, 'verified_complete': 24, 'rows': [{'status': 'VERIFIED_COMPLETE'}] * 24}
for name, value in (('matrix', matrix), ('prior', prior), ('original', original)):
    (fixture / f'{name}.json').write_text(json.dumps(value) + '\n')
sys.argv = [str(source), '--matrix', str(fixture / 'matrix.json'), '--prior-matrix', str(fixture / 'prior.json'),
            '--original-matrix', str(fixture / 'original.json'), '--dataset', 'RGBNT201',
            '--output-dir', str(fixture / 'must_not_be_created')]
try:
    module.main()
except AssertionError as error:
    refusal = traceback.extract_tb(error.__traceback__)[-1]
else:
    raise AssertionError('Analyzer incorrectly accepted a dataset with one unaccepted cell')
assert not (fixture / 'must_not_be_created').exists()
assert refusal.filename == str(source)
report = {'status': 'M3_FACTOR_ARITHMETIC_AND_INCOMPLETE_CELL_REFUSAL_PASS',
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'arithmetic_checks': checks, 'incomplete_cell_refusal_line': refusal.line,
          'incomplete_cell_refusal_lineno': refusal.lineno, 'created_result_directory': False,
          'boundary': 'Synthetic metric arithmetic and one unaccepted-cell gate only. No experiment distance, '
                      'checkpoint, prediction, GPU, bootstrap or real completed M3 endpoint was analyzed.'}
out = root / '.git/correspondence_m3_analysis_cpu_642_20260929.json'
assert not out.exists()
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
