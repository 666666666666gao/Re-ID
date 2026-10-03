"""Publish actual F3 review closure and the independently reviewed next plan."""
from datetime import datetime
from pathlib import Path
import ast
import hashlib
import json
import shutil
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
f3 = Path('C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1')
native = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
previous = json.loads((proof / 'five_copy768.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous['head']
assert not (proof / 'publication769_local.json').exists()
protected = json.loads((proof / 'publication768_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest for name, digest in protected.items())
audit = json.loads((f3 / 'reviewer_audit766/EXPERIMENT_AUDIT.json').read_bytes())
claim = json.loads((f3 / 'reviewer_claim768/CLAIM_ASSESSMENT.json').read_bytes())
review = json.loads((native / 'review_source769/EXPERIMENT_CODE_REVIEW.json').read_bytes())
assert audit['integrity_status'] in ('pass', 'warn') and not audit['blocking_findings']
assert claim['claim_supported'] == 'no' and claim['confidence'] == 'medium'
assert review['verdict'].upper() in ('PASS', 'WARN') and not review['blocking_findings']
target = repo / 'logs/metric_feature_scale_closeout769_20261003'
assert not target.exists()
target.mkdir()
files = []

def copy(source, destination):
    path = repo / destination
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    assert path.read_bytes() == source.read_bytes()
    files.append(path.relative_to(repo).as_posix())

for directory, names, output in (
    (f3 / 'reviewer_audit766', ('EXPERIMENT_AUDIT.md', 'EXPERIMENT_AUDIT.json'), 'refine-logs/metric_feature_scale_v1'),
    (f3 / 'reviewer_claim768', ('CLAIM_ASSESSMENT.md', 'CLAIM_ASSESSMENT.json'), 'refine-logs/metric_feature_scale_v1'),
    (native / 'review_source769', ('EXPERIMENT_CODE_REVIEW.md', 'EXPERIMENT_CODE_REVIEW.json'), 'refine-logs/independent_native_evidence_v1'),
):
    for name in names:
        copy(directory / name, output + '/' + name)
attached = {key: claim[key] for key in ('claim_supported', 'confidence', 'what_results_support',
    'what_results_dont_support', 'missing_evidence', 'suggested_claim_revision', 'next_experiments_needed')}
attached.update(integrity_status=audit['integrity_status'],
    review_independence='same-family', acceptance_status='provisional',
    original_claim_integrity_status=claim['integrity_status'],
    audit_sha256=hashlib.sha256((f3 / 'reviewer_audit766/EXPERIMENT_AUDIT.json').read_bytes()).hexdigest(),
    claim_sha256=hashlib.sha256((f3 / 'reviewer_claim768/CLAIM_ASSESSMENT.json').read_bytes()).hexdigest(),
    routing='Close finite F3 diagnostics; no scale/margin/seed/coefficient rescue. Select matched author-package foundation and registered independent native evidence plan.',
    recorded_at=datetime.now().astimezone().isoformat())
claim_path = repo / 'refine-logs/metric_feature_scale_v1/CLAIMS_FROM_RESULTS.json'
assert not claim_path.exists()
claim_path.write_bytes((json.dumps(attached, ensure_ascii=False, indent=2) + '\n').encode())
files.append(claim_path.relative_to(repo).as_posix())
findings = repo / 'refine-logs/metric_feature_scale_v1/findings.md'
assert not findings.exists()
findings.write_bytes(('''# F3 findings and next action

Fresh same-family/provisional review: the intended all-three claim is **no**, with medium confidence and actual integrity **''' + audit['integrity_status'] + '''**. Only MSVR310 passes the preregistered +0.5mAP/noR1drop development gate;201 and100 lose mAP. Original worker exit1 and separately attributed first missing evaluation / first report remain disclosed. No scale/margin/seed/coefficient rescue.

The arithmetic supports a fixed42 MSVR observation, not an all-dataset mechanism, new architecture, training-seed stability or SOTA. Historical F1 compares whole packages; F2 changes CE and Triplet together. They do not isolate a unique cause.

Next use the pinned author training package consistently for all three datasets. Rebuild the shared-adapter global-only control and original semantic roles from publicCLIP/freshcamera, then add the independently read image-native evidence. This is the fixed nine-arm plan in ../independent_native_evidence_v1/EXPERIMENT_PLAN.md. Do not use the old69.6415 or weaker current F3 as a fixed denominator. Added159296 parameters remain a capacity confound. Source review is not real M0 or performance acceptance.
''').encode())
files.append(findings.relative_to(repo).as_posix())
for name in ('stdout.json', 'stderr.txt', 'EXIT.json'):
    copy(native / 'resources769' / name, 'logs/metric_feature_scale_closeout769_20261003/resources/' + name)
for name in ('initial_pair_schema_before_fix.py', 'entry_before_import_fix.py', 'plan_before_lr_wording_fix.md'):
    copy(native / 'review_source769' / name, 'logs/metric_feature_scale_closeout769_20261003/source_review_history/' + name)
for name in ('prepare_independent_native_sources769.py', 'observe_native_resources769.py',
             'deploy_independent_native_evidence769.py', 'prepare_metric_closeout_and_native769.py'):
    copy(Path('C:/Users/gb/.codex_tmp') / name, 'logs/metric_feature_scale_closeout769_20261003/' + name)
new_source = ('modeling/trifusion/image_native_evidence.py', 'modeling/trifusion/independent_native_roles.py',
    'modeling/trifusion/evidence_author_heads.py', 'tools/run_independent_native_evidence.py',
    'tools/check_independent_native_pair.py', 'tools/queue_independent_native_evidence.py',
    'tools/report_independent_native_evidence.py', 'refine-logs/independent_native_evidence_v1/EXPERIMENT_PLAN.md')
for name in new_source:
    if name.endswith('.py'):
        ast.parse((repo / name).read_text(encoding='utf-8'))
    files.append(name)
for folder, rules in (
    ('refine-logs/metric_feature_scale_v1', 'EXPERIMENT_AUDIT.md -whitespace\nCLAIM_ASSESSMENT.md -whitespace\n'),
    ('refine-logs/independent_native_evidence_v1', 'EXPERIMENT_CODE_REVIEW.md -whitespace\n'),
):
    attribute = repo / folder / '.gitattributes'
    assert not attribute.exists()
    attribute.write_bytes(rules.encode())
    files.append(attribute.relative_to(repo).as_posix())
status = {'recorded_at':datetime.now().astimezone().isoformat(), 'status':'F3_REVIEWS_CLOSED_NATIVE_PLAN_REGISTERED',
    'claim_supported': 'no', 'confidence': 'medium', 'integrity_status': audit['integrity_status'],
    'source_review_verdict': review['verdict'], 'review_independence':'same-family', 'acceptance_status':'provisional',
    'foundation_selected':True, 'foundation':'pinned author package, fresh publicCLIP, original raw heads; shared adaptation controlled separately',
    'new_plan_registered':True, 'new_initializer_executed':False, 'new_m0_executed':False,
    'new_training_started':False, 'training_ports':[2026], 'physical_gpu_scope':[0,1,2,3],
    'max_parallel':4, 'goal':'ACTIVE_UNMET',
    'new_source_sha256':{name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in new_source}}
(target/'STATUS.json').write_bytes((json.dumps(status,indent=2)+'\n').encode())
files.append((target/'STATUS.json').relative_to(repo).as_posix())
doc = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
path = repo/doc
old = path.read_bytes()
assert hashlib.sha256(old).hexdigest()==previous['doc_sha256']
text = old.decode('utf-8')
assert '## 41.769 ' not in text
headings=[line for line in text.splitlines() if line.startswith('**') and '41.768' in line]
assert len(headings)==1
text=text.replace(headings[0], '**当前进度（§41.769，2026-10-03）：** F3六端完成后的独立复核已收齐，integrity '+audit['integrity_status']+'、全三集claim no；封存FAIL/FAIL/PASS，不继续尺度救分。下一项固定作者训练配方、共享global／语义roles／独立原生细节九端计划已登记并完成源码复核，尚未执行真实初始化、M0或训练。仅2026 GPU0–3/max4；2025不启动训练。Goal ACTIVE_UNMET。')
text += f'''

## 41.769 F3复核收束与独立原生细节九端登记（{status['recorded_at']}）

两位fresh gpt-6-astra/max/forknone复核员已给出实际完整报告。完整性`{audit['integrity_status']}`，无阻塞发现；claim_supported=`no`、confidence=`medium`。同模型家族、独立上下文、provisional，不是跨模型家族验收。原始claim报告保留其当时integrity pending；另由`CLAIMS_FROM_RESULTS.json`附加现已完成的实际完整性回执，未覆盖reviewer原文或改成partial/yes。

严格回执的mAP/R1百分点差为：201 −2.846207561661925/−2.15311050415039；100 −2.13230723621369/+0.58308839797974；MSVR +2.402812986499136/+3.89170646667480。仍为FAIL/FAIL/PASS。只支持固定42及本配方下MSVR描述性改善；不支持全三集有效、新机制、角色必要性、多seed或SOTA。原worker磁盘exit1、CRLF运输失败及后续各一次补评价/report的复合执行来源均保留。二进制仍远端，复核只读源码、文本和实存SHA回执；24个历史M0退役造成的历史二进制重放损失仍披露。

下一项不再修改F3尺度、margin、seed或系数。选择统一的作者训练package，重新训练匹配的共享适配global-only、原语义roles、语义roles加独立原生细节三个条件×三数据集；每端50轮。全局控制含共享适配，不等于F1无adapter的plain baseline，也不继续使用旧69.6415作固定分母。新CNN从原图stride8得到512候选，独立Q/K/V读取，再以单个零出口加到保留的128-token语义CNN区域；原桥接和1536维部署接口不变。新增159296参数/14张量，若正增益还需同容量语义控制，不能先宣称纯细节因果贡献。

源码复核在运行前找到并修正两项具体阻塞：独立校验子进程未初始化新schema，以及入口提前缓存DeMo的modeling包。分别只加入panel.configure()、改用项目已有trifusion导入；原草稿字节和反馈保留。MSVR作者学习率明确base5e-6、bias×2、classifier×100，未调LR救分。实际作者head/optimizer/loader/scheduler按三条件匹配。没有添加新损失、teacher、文本、掩码或外部数据。

12:22:27只读实查2026四张3090无compute任务、可用磁盘11,176,480,768字节。新队列依据实际旧state345–347MB与最大distance59MB，预估18份384MiB状态、600MiB距离、256MiB文本并保留2GiB；原F3的10GiB断言不改写。启动前仍要实查空卡与磁盘。新的九个真实M0全部通过前不启动正式训练，检查完整初始化／真实作者batch／semantic-native初始完全等价／所有参数梯度与更新／作者BN模式／全状态重载。此刻只是计划登记和source复核，尚无新GPU运行或性能。所有后续GPU只用2026物理0–3，最多四端；2025仅继续已授权文本同步。
'''
path.write_bytes(text.encode())
shutil.copyfile(path, Path('C:/Users/gb/Desktop/document')/path.name)
files.append(doc)
publication={'previous_head':previous['head'],'files':sorted(files),'section':'41.769',
    'doc_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'doc_bytes':path.stat().st_size,
    'status':status['status'],'training_ports':[2026],'max_parallel':4,'protected_files':protected}
(proof/'publication769_local.json').write_bytes((json.dumps(publication,indent=2)+'\n').encode())
publisher=Path('C:/Users/gb/.codex_tmp/publish_metric_closeout_and_native769.py')
assert not publisher.exists()
source=Path('C:/Users/gb/.codex_tmp/publish_metric_scale_received768_v2.py').read_text(encoding='utf-8')
source=source.replace('768','769').replace('five_copy767.json','five_copy768.json').replace('range(739,769)','range(739,770)')
source=source.replace("'logs/metric_feature_scale_complete769_20261003','refine-logs/metric_feature_scale_v1'", "'logs/metric_feature_scale_closeout769_20261003','refine-logs/metric_feature_scale_v1','refine-logs/independent_native_evidence_v1','modeling/trifusion','tools'")
source=source.replace('Receive all six F3 strict results with original failure provenance', 'Close F3 reviews and register matched independent native evidence controls')
source=source.replace('all-six primary text/source intake with fresh reviews pending;', 'actual F3 review closure and registered reviewed native plan;')
publisher.write_bytes(source.encode())
ast.parse(source)
print(json.dumps({k:v for k,v in publication.items() if k not in ('files','protected_files')},indent=2))
