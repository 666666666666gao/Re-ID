from pathlib import Path
from datetime import datetime
import json,os

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1')
intake=json.loads((base/'INTAKE.json').read_bytes())
assert intake['status']=='ALL6_FORMAL_AND_ONCE_CPU_REPORT_COMPLETE'
assert not (base/'CLAIM_REQUEST.txt').exists()
manifest=repo/'.aris/installed-skills-codex.txt'
aris_root=os.environ.get('ARIS_REPO')
if not aris_root and manifest.is_file():
 for line in manifest.read_text(encoding='utf-8').splitlines():
  fields=line.split('\t')
  if fields[0]=='repo_root':aris_root=fields[1];break
paths=([str(Path(aris_root)/'tools/evidence_check.py')] if aris_root else [])+[str(repo/'tools/evidence_check.py')]
actual=[path for path in paths if Path(path).is_file()]
assert actual==[],actual
precheck={'status':'WARN_EVIDENCE_CHECK_UNRESOLVED_SEMANTIC_REVIEW_STILL_REQUIRED',
 'recorded_at':datetime.now().astimezone().isoformat(),'aris_repo_configured':bool(aris_root),
 'installed_manifest_exists':manifest.is_file(),'checked_helper_paths':paths,'resolved_helpers':actual,
 'boundary':'Resolver only; not an evidence-check PASS or a semantic claim verdict. Actual intake hashes/strict receipts are separately recorded.'}
(base/'EVIDENCE_PRECHECK.json').write_bytes((json.dumps(precheck,indent=2)+'\n').encode())
out=base/'reviewer_claim768';out.mkdir()
request='''RESULT-TO-CLAIM EVALUATION — F3 normalized CE / raw versus normalized Triplet.
You are a fresh claim reviewer. Read actual files, not the executor's conclusions:
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/EVIDENCE_PRECHECK.json
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/INTAKE.json
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/AUDIT_INPUT_PATHS.json
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/refine-logs/metric_feature_scale_v1/EXPERIMENT_PLAN.md
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_metric_feature_scale.py
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_training_feature_scale.py
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/source/tools/run_foundation_recipe.py
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/results/metric_feature_scale_complete_20261003/SUMMARY.json
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/raw/results/metric_feature_scale_complete_20261003/REPORT.md
- The six full training/strict metrics/batchorder/steps and all three pair analyses in the catalog.
- Original archived FAILED parent/child/worker and actual storage-finish evaluation/report receipts.

Intended question: in the fixed fresh current recipe, does keeping normalized
classification input but giving Triplet raw global features improve retrieval
across all three datasets? Only Triplet input changes; normalized deployment,
architecture/capacity/initialization/order/budget/protocol/seed42 are held fixed.
Advance gate per dataset: mAP gain at least0.5pp with no Rank1 drop. Recompute
all three outcomes from exact original values; do not round before the gate.
This is a finite foundation diagnostic, not a new model/module or SOTA claim.
The original all6 trains/full50 completed; only the unstarted last evaluate
was executed after a disk-guard failure, then the first unmodified CPU report.
Retired24 old closed M0 probes limit historical binary replay, but current
F3 M0/best/distance binaries remain remote and were actually SHA-attested.

Known boundaries to examine: single fixed seed and officially consumed benchmark;
mAP-best/latest exact tie selection, same-checkpoint CMC; fixed-model identity
bootstrap is not independent training-seed uncertainty. F2 changed CE and
Triplet inputs together; F1 author/current differed in head/optimizer/sampler/
schedule and other recipe details. Do not infer unique CE, feature norm, domain
shift or architecture cause from cross-panel subtraction. Do not tune margin,
scale, coefficients, seeds or number of epochs to rescue F3. The next direction
is an independently read semantic/native-detail method on a matched foundation,
still unregistered/unexecuted. No promise of10pp/SOTA or all-three-role efficacy.
If helpful for foundation interpretation, read historical actual reports:
- C:/Users/gb/.codex_tmp/training_feature_scale_complete754/raw/results/training_feature_scale_complete_20261003/SUMMARY.json
- C:/Users/gb/.trifusion_github_publish_22c3bee/logs/foundation_complete739_20261002/raw/results/foundation_recipe_complete_20261002/SUMMARY.json
Check exact existing historical paths rather than treating these packages as
single-factor or resource-matched claims. F3 integrity review is running in
parallel. If reviewer_audit766/EXPERIMENT_AUDIT.json exists, inspect it; otherwise
label integrity pending. The parent will attach the actual audit before final
acceptance, with fail downgrading confidence; no fabricated audit PASS.

Evaluate claim_supported yes|partial|no; what_results_support;
what_results_dont_support; missing_evidence; suggested_claim_revision;
next_experiments_needed; confidence high|medium|low and its precise scope.
Directly cite absolute file:line evidence and exact deltas/gates. Explicitly
distinguish descriptive paired results from broad causal/generalization claims.
Same-family/provisional. Deterministic source receipt existence is not a claim.
Local read/hash/JSON/arithmetic only. No SSH, model imports, evaluation/report
replay, binary download, deletion or input/source/evidence edits.
Write only these outputs (parent owns private traces):
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/reviewer_claim768/CLAIM_ASSESSMENT.md
- C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1/reviewer_claim768/CLAIM_ASSESSMENT.json
Include claim_supported, confidence, integrity_status, input_hashes,
review_independence same-family and acceptance_status provisional. Return the
full verdict, concrete limitations, output paths and SHA256s.
'''
(base/'CLAIM_REQUEST.txt').write_bytes(request.encode())
print('Actual complete F3 claim request prepared; helper unresolved WARN, no reviewer verdict yet.')
