import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path

work = Path(__file__).resolve().parent
private = work / 'frozen_region_use_draft912'
raw_binding_path = work / 'DIAGNOSTIC_BINDINGS_916.json'
binding = json.loads(raw_binding_path.read_text(encoding='utf-8'))
for key in ('remaining_campaign', 'original_failed_remaining_campaign'):
    path = Path(binding[key])
    assert path.name == 'campaign.json'
    binding[key + '_file'] = binding[key]
    binding[key] = path.parent.as_posix()
binding_path = work / 'DIAGNOSTIC_INPUT_BINDING_916.json'
assert not binding_path.exists()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_pins = {
    'tools/diagnose_region_use_best.py': 'bc73f6f2d8dc399048dca58ba25d99c059f6cb80bae35e5ab01b130c5b438ff4',
    'tools/queue_region_use_best_diagnosis.py': '8999fcc3b3f80f9a44b0c882462a141af068c6e32bdcd3266cfe3e2e610f2294',
    'EXPERIMENT_PLAN.md': 'eebd8c9548ab7dc879d19e72940447466378f7b96b6c27db9a70fbd4a3ed2633',
}
for name, h in source_pins.items():
    p = private / name
    assert sha(p) == h
    if p.suffix == '.py':
        ast.parse(p.read_text(encoding='utf-8'))
assert binding['status'] == 'PRIVATE_UNADMITTED_SIX_BEST_ADMINISTRATIVE_BINDINGS'
assert len(binding['rows']) == 6
assert {(r['dataset'], r['comparison']) for r in binding['rows']} == {
    (d, c) for d in ('RGBNT201', 'MSVR310', 'RGBNT100') for c in ('uniform', 'ambiguity')}
assert binding['remaining_campaign'].endswith('region_use_remaining_strict_20261013_912/region_use_remaining_20261013_911')
assert binding['remaining_exit'].endswith('region_use_remaining_strict_admin912/EXIT.json')
assert binding['original_failed_remaining_campaign'].endswith('logs/region_use_remaining_20261013_911')
assert binding['original_failed_remaining_exit'].endswith('region_use_remaining_admin911/EXIT.json')
assert binding['text_artifact_sha256'][binding['original_failed_remaining_campaign'] + '/campaign.json'] == 'a32e91b4929ef914a10e12c41ae45a4f40a17386dc3236e73e83e357d0f04b51'
assert binding['text_artifact_sha256'][binding['remaining_campaign'] + '/campaign.json'] == '7c3de93015f00aa864023535fbef737aafcc7e77dcc7063b37c20fccb03b1c2b'
payload = sum(5 * 4 * r['query_count'] * r['gallery_count'] for r in binding['rows'])
assert payload == binding['diagnostic_payload_bytes'] == 641141040
assert binding['storage_required_bytes'] == 2 * 1024**3 + 64 * 1024**2 + payload == 2855733552
assert not binding['admission'] and binding['neural_calls'] == binding['optimizer_updates'] == binding['gpu_reads'] == 0
aggregate = json.loads((work / 'ALL_SIX_RESULTS_912.json').read_text(encoding='utf-8'))
for r in binding['rows']:
    accepted = next(a for a in aggregate['rows'] if (a['dataset'], a['comparison']) == (r['dataset'], r['comparison']))
    assert r['metrics'] == accepted['metrics'] and r['best_epoch'] == accepted['best_epoch']
    assert r['run_dir'] == accepted['candidate']['run_dir']
    assert r['checkpoint_sha256'] == accepted['candidate']['checkpoint_sha256']
    assert r['distance_sha256'] == accepted['candidate']['distance_sha256']
    assert r['original50_history_validated'] and r['initializer']['seed'] == 42
    assert r['initializer']['architecture'] == 'trifusion-region-use-supervision-v1'
    assert all(b['exists'] and b['bytes'] > 0 and not b['bytes_sha256_rechecked'] for b in r['binaries'].values())

binding_path.write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8')
plan = '''# Post-terminal fixed-best diagnostic contract 916 — private, not admitted

All six original seed42/full50/first-strict results are accepted under sections913–915. The old draft912 assumed original remaining911 would exit0; it instead exited1 before first strict evaluation. Its original failed campaign/EXIT remain immutable. The separate first-unattempted strict completion912 exited0 and is the acceptance reference. This correction changes administrative binding, not training, scoring, model or checkpoint selection.

DIAGNOSTIC_BINDINGS_916.json is the immutable cold metadata collection. Its campaign fields contain actual JSON file paths. DIAGNOSTIC_INPUT_BINDING_916.json explicitly maps those two fields to directories required by the unchanged consumer, retaining the original file paths separately. The first static check failed on this file/directory mismatch; original failed helper and receipt remain preserved. This is not a fallback accepting both types.

The corrected binding is not an executable full seal. It binds six original bests, current strict metric receipts, original50 histories, original initialization/initializer, protocol Q/G counts and the accepted administrative remaining campaign. It preserves the original failed remaining campaign separately. Current binary presence/size were observed; their hashes remain the hashes of the accepted strict receipts, not a fresh binary re-read.

The previously reviewed inference consumer and sequential controller remain byte-identical at the SHA pins in CONTRACT_CHECK_916.json. Their old draft plan remains preserved. No new module, head, loss, scorer, coefficient, seed or checkpoint is introduced. Before any approved execution, a new full seal still must pin/check original463 producer sources, original50 dependencies, the unchanged consumer/controller and all exact six current artifacts. This contract does not substitute metadata checks for those full SHA gates.

Five fixed descriptive scores: same-model global1536, correction1536, original fused1536, nine role/modality symmetric regional meanMaxSim, and exactly the existing auxiliary target0.5*(global cosine+regional meanMaxSim). One original best per condition; full original legal Q/G; no fitting/sweep/best reselection. Exact fused replay retains original1e-5 bounds, model/buffer state fixed and gradients absent. The learned joint target is a pairwise diagnosis; a higher score would not certify teacher superiority, compression causality or a new deployable method.

Cold current protocol counts give all-six distance payload641,141,040B. Existing2GiB reserve plus64MiB metadata gives2,855,733,552B. Observed free3,689,156,608B at2026-10-13T05:25:40.75742+08:00 fits at that observation only. Recheck storage and actual GPU0/1 idle at any future admission; no GPU read was performed here. Keep currentbest, Signal, dependencies and local archives; no additional cleanup was needed.

User finish-current-and-stop remains effective; next-scope clarification is pending. No input seal/launcher deployment or NN execution is admitted by this file. It does not resume training. Only26 physicalGPU0/1 if a read-only diagnostic is subsequently selected. The full scientific objective all3>Signal remains unmet. Sole handoff remains Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md.
'''
(work / 'CONDITIONAL_DIAGNOSTIC_PLAN_916.md').write_text(plan, encoding='utf-8')
receipt = dict(status='POST_TERMINAL_ADMINISTRATIVE_BINDINGS_CHECKED_NOT_ADMITTED', at=datetime.now().astimezone().isoformat(),
               binding_sha256=sha(binding_path), raw_binding_sha256=sha(raw_binding_path), unchanged_private_source_sha256=source_pins,
               rows=6, binary_hashes_rechecked=False, full_execution_seal=False, gpu_idle_checked=False,
               original_failed_exit_retained=True, accepted_strict_reference_corrected=True,
               neural_calls=0, optimizer_updates=0, training_stopped=True, goal_status='ACTIVE_UNMET',
               cold_collection=dict(tool='exec_command', chunk_id='8e4313', exit_code=0, wall_time_seconds=3.7835207),
               preceding_cold_check=dict(tool='exec_command', chunk_id='0689eb', exit_code=0,
                   original_terminal_condition=False, accepted_terminal_condition=True, source_seals_unchanged=True),
               scope='Root static text/AST/receipt/count checks; no new semantic experiment audit or full raw/binary/NN verification.')
(work / 'CONTRACT_CHECK_916.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
