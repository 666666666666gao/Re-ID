import hashlib
import json
from datetime import datetime
from pathlib import Path

work = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
package_path = work / 'PREPARED_DIAGNOSTIC_INPUTS_917.json'
package = json.loads(package_path.read_text(encoding='utf-8'))
assert package['scope_counts']['root_sources'] == 463 and package['scope_counts']['artifacts'] == 89
assert package['execution_admission'] is False
source_review = work / 'review/current_region_draft912'
delta = json.loads((source_review / 'DELTA_AUDIT.json').read_text(encoding='utf-8'))
controller = json.loads((source_review / 'CONTROLLER_C1_DELTA_AUDIT.json').read_text(encoding='utf-8'))
assert delta['implementation_blockers'] == [] and delta['source_sha256'] == 'bc73f6f2d8dc399048dca58ba25d99c059f6cb80bae35e5ab01b130c5b438ff4'
assert controller['remaining_implementation_blockers'] == [] and controller['source_usable_within_reviewed_scope']
for name, h in package['staged_source_checks'].items():
    assert sha(work / 'frozen_region_use_draft912' / name) == h['sha256']
root = '/data/gaob/Re-ID/Trifusion'
seal = dict(schema=package['schema'], registered_at=datetime.now().astimezone().isoformat(),
            source_sha256=package['source_sha256'], artifact_sha256=dict(package['artifact_sha256']),
            rows=package['rows'], remaining_campaign=package['remaining_campaign'], remaining_exit=package['remaining_exit'],
            original_failed_remaining_campaign=package['original_failed_remaining_campaign'],
            original_failed_remaining_exit=package['original_failed_remaining_exit'],
            storage_required_bytes=package['storage_required_bytes'], physical_gpus=[0, 1], max_parallel_jobs=1,
            execution_scope='READ_ONLY_ORIGINAL_BEST_INFERENCE_NO_TRAINING',
            runtime_admission='SEPARATE_GPU_IDLE_STORAGE_FULL_SHA_RECEIPT_REQUIRED',
            objective_authorization='Existing user objective explicitly requires diagnosing failed results; user finish-current-and-stop governs training. No answer or automatic continuation is treated as new permission. Agent-added NN confirmation barrier is corrected; no new training authorized.',
            boundaries='Five original descriptive scores, original six bests/full QG, same source, strict fused parity and fixed model state; no optimizer, fitting, new checkpoint, seed, gain, teacher or compression claim.')
seal['artifact_sha256'][root + '/logs/region_use_prepared_inputs_20261013_917/PREPARED_DIAGNOSTIC_INPUTS_917.json'] = sha(package_path)
directory = work / 'readonly_diagnosis918'
directory.mkdir()
(directory / 'INPUT_SEAL.json').write_text(json.dumps(seal, indent=2) + '\n', encoding='utf-8')
plan = '''# Read-only current-best diagnosis918

Authority: the existing user objective requires diagnostics of failed results. The user's finish-current-and-stop instruction continues to stop training. Fixed-best eval is read-only. The earlier agent priority incorrectly required another scope answer for this already authorized diagnostic; it is superseded for inference only. No new user answer or elapsed time is represented as approval, and new training remains stopped.

Use the unchanged reviewed five-score consumer bc73f6f2… and sequential controller8999fcc3… on26 physicalGPU0/1 and the existing tri_reid environment. Warm reuse; no package/env rebuild. Six original bests in prepared917/accepted913, seed42/full50 histories, exact initialization and source bindings. Compare same-model global/correction/fused, nine-group regional symmetric meanMaxSim, and existing0.5*(globalcos+regionalMaxSim) pairwise target. No score sweep, weight/optimizer update, checkpoint replacement or best/seed/gain choice.

Runtime admission must verify465 deployedsource SHA and90 artifact SHA, bothGPU0/1 below500MiB, disk at least2,855,733,552B. After verified archive-to-root copy, freeze INPUT_SEAL.json. Controller repeats required gates, stops first failure without retry, records each child and actualexit. Fused full-gallery distance and metric parity retain the original1e-5 boundaries; model/buffers fixed andgradNone. Future diagnostics cannot silently narrow to fewer than6 cases.

Run once using /data/gaob/Re-ID/conda-envs/tri_reid/bin/python -B tools/queue_region_use_best_diagnosis.py --seal /data/gaob/Re-ID/Trifusion/refine-logs/region_use_fixed_best_diagnosis918/INPUT_SEAL.json --output-dir /data/gaob/Re-ID/Trifusion/logs/region_use_fixed_best_diagnosis_20261013_918 withCUDA_VISIBLE_DEVICES=0,1. A separate stdlib supervisor records launch/log/realEXIT in logs/region_use_fixed_best_diagnosis_admin918. No training/M0/prepare replay. Estimate15–25min; one own observer scheduled near estimated completion, not repeated GPU/progress polling. Only actual terminal state and process handles establish completion.

Source-only reviews and priorfailures retained; after terminal, collect original diagnosis JSON/distances once and bounded fresh result audit before mechanism claims. Lower branchstandalone mAP does not prove no complement; higher pairwise mAP does not prove causal compression or stronger teacher. All benchmarks are development-consumed. Full all3>Signal objective remainsunmet, even if a pairwise diagnostic performswell. Sole handoff remains Desktop/document/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md.
'''
(directory / 'EXPERIMENT_PLAN.md').write_text(plan, encoding='utf-8')
for name in ('EXPERIMENT_AUDIT.json', 'DELTA_AUDIT.json', 'CONTROLLER_C1_DELTA_AUDIT.json'):
    (directory / name).write_bytes((source_review / name).read_bytes())
print(json.dumps(dict(status='READ_ONLY_DIAGNOSIS918_REGISTERED_PENDING_RUNTIME_ADMISSION',
                     source_count=len(seal['source_sha256']), artifact_count=len(seal['artifact_sha256']),
                     seal_sha256=sha(directory / 'INPUT_SEAL.json'), training_stopped=True)))
