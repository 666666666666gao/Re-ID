"""Prepare the fresh file-based audit request after original F3 text intake."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

root = Path('C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1')
intake = json.loads((root/'INTAKE.json').read_bytes())
assert intake['status'] == 'ALL6_FORMAL_AND_ONCE_CPU_REPORT_COMPLETE'
assert not (root/'AUDIT_REQUEST.txt').exists()
sources = []
for name, expected in intake['source_sha256'].items():
    path = root/'source'/name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, name
    sources.append({'path': str(path), 'sha256': actual})
primary = []
for name, expected in intake['text'].items():
    path = root/'raw'/name
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    assert actual == expected['sha256'] and len(data) == expected['bytes'], name
    primary.append({'path': str(path), 'sha256': actual, 'bytes': len(data)})
assert len(sources) == 265 and len(intake['binary']) == 24
catalog = {'terminal_intake': str(root/'INTAKE.json'),
           'sealed_source': sources, 'primary_text': primary,
           'remote_binary_attestations': intake['binary'],
           'prepared_at': datetime.now().astimezone().isoformat()}
(root/'AUDIT_INPUT_PATHS.json').write_text(json.dumps(catalog, indent=2)+'\n', encoding='utf-8')
output = root/'reviewer_audit766'
output.mkdir()
old = Path('C:/Users/gb/.codex_tmp/training_feature_scale_complete754/AUDIT_REQUEST.txt').read_text(encoding='utf-8')
prompt = old.replace('C:/Users/gb/.codex_tmp/training_feature_scale_complete754', root.as_posix())
prompt = prompt.replace('C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/source', (root/'source').as_posix())
prompt = prompt.replace('- C:/Users/gb/.codex_tmp/training_feature_scale_source_intake749/SOURCE_INTAKE.json\n', '')
prompt = prompt.replace('refine-logs/training_feature_scale_v1/EXPERIMENT_PLAN.md', 'refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md')
prompt = prompt.replace('training_feature_scale_20261003_v2', 'metric_feature_scale_20261003_v1')
prompt = prompt.replace('results/training_feature_scale_complete_20261003', 'results/metric_feature_scale_complete_20261003')
prompt = prompt.replace('reviewer_audit754', 'reviewer_audit766')
prompt = prompt.replace('- '+(root/'source/tools/run_training_feature_scale.py').as_posix()+'\n',
                        '- '+(root/'source/tools/run_metric_feature_scale.py').as_posix()+'\n'
                        '- '+(root/'source/tools/run_training_feature_scale.py').as_posix()+'\n')
prompt = prompt.replace('- '+(root/'source/tools/queue_training_feature_scale.py').as_posix()+'\n',
                        '- '+(root/'source/tools/queue_metric_feature_scale.py').as_posix()+'\n'
                        '- '+(root/'source/tools/queue_training_feature_scale.py').as_posix()+'\n')
prompt = prompt.replace('- '+(root/'source/tools/check_training_feature_scale_pair.py').as_posix()+'\n',
                        '- '+(root/'source/tools/check_metric_feature_scale_pair.py').as_posix()+'\n'
                        '- '+(root/'source/tools/check_training_feature_scale_pair.py').as_posix()+'\n')
prompt = prompt.replace('- '+(root/'source/tools/report_training_feature_scale.py').as_posix()+'\n',
                        '- '+(root/'source/tools/report_metric_feature_scale.py').as_posix()+'\n'
                        '- '+(root/'source/tools/report_training_feature_scale.py').as_posix()+'\n')
assert 'complete754' not in prompt and 'intake749' not in prompt
prompt += '''

ACTUAL F3 INTERVENTION — audit the new entry, not the inherited F2 label:
Both F3 variants use normalized classification input and unchanged normalized
L2_1536 deployment. Only Triplet receives normalized versus raw global features.
Read run_metric_feature_scale.py and its caller monkey-patch execution order.
F2 changed both classification and metric inputs, so cross-panel differences
are not independent seed replicates or automatic isolated CE causality.

ACTUAL STORAGE-ONLY FINISH PROVENANCE — mandatory audit scope:
The original controller failed after all six complete50 trainings, before the
RGBNT100 metric_raw independent evaluate. It hit the unchanged10GiB disk guard.
The original FAILED parent, original train-only child and worker exit1 traceback
are archived in raw/logs/metric_feature_scale_storage_finish766_20261003/.
The exact24 historical closed M0 engineering probes were retired only after real
path/size/SHA/owner/closed-summary/protected-best/currentF3 guards. Their remote
receipt is raw/logs/closed_m0_retirement766_20261003/RETIREMENT.json. This ends
historical direct M0 binary replay; formalbest/public/author/currentF3 remain.
Review the separately attributed finish helper/spec/reviews/LAUNCH/status in
raw/logs/metric_storage_finish_source766_v2_20261003/ and the finish directory.
The first transport failed before Popen because a Windows-rewritten CRLF JSON
receipt was compared to exact remote LF bytes. Original source766 asset folder
is retained; revision only uses actual received remote bytes and a new asset
directory. No training restart or model/science/tolerance/source265 change.
The original worker is NOT retrospectively successful: current completed job
means original train exit0 plus a separately recorded first evaluate exit0.
Verify nested original_failure, immutable archived FAILED digest, both PIDs,
the actual first evaluation invocation/strict result and first original report.
Do not manufacture a completed original process or hide the disk failure.
No SSH, model import, scoring, report replay, deletion or evidence edits.
Source-only and primary-text audit; recorded binary SHA attestations are not
local binary replays. Report all real integrity blockers and these limits.
'''
(root/'AUDIT_REQUEST.txt').write_text(prompt, encoding='utf-8')
print(json.dumps({'status':'AUDIT_REQUEST_READY_NOT_YET_REVIEWED',
                  'source_files':len(sources), 'primary_text_files':len(primary),
                  'request':str(root/'AUDIT_REQUEST.txt'), 'output':str(output)}, indent=2))
