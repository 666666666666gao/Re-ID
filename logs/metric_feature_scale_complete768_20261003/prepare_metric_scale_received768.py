"""Publish all actual F3 primary text/source intake, with reviews pending."""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess,ast

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_complete_v1')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
previous=json.loads((proof/'five_copy767.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
protected=json.loads((proof/'publication767_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==value for name,value in protected.items())
intake=json.loads((base/'INTAKE.json').read_bytes())
assert intake['status']=='ALL6_FORMAL_AND_ONCE_CPU_REPORT_COMPLETE'
summary=json.loads((base/'raw/results/metric_feature_scale_complete_20261003/SUMMARY.json').read_bytes())
assert summary['status']=='COMPLETE' and summary['accepted']==6 and summary['formal_epochs']==300 and summary['formal_steps']==20416
observer=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766/finish_observer/STATUS.json')
observed=json.loads(observer.read_bytes())
assert observed['status']=='STORAGE_FINISH_TERMINAL_OBSERVED' and not observed['controller_live']
assert observed['finish']['status']=='ALL6_STRICT_AND_FIRST_REPORT_COMPLETE'
target=repo/'logs/metric_feature_scale_complete768_20261003'
assert not target.exists() and not (proof/'publication768_local.json').exists();target.mkdir()
files=[]
def copy(source,name):
 path=target/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
 assert source.read_bytes()==path.read_bytes();files.append(path.relative_to(repo).as_posix())
for name,expected in intake['source_sha256'].items():
 source=base/'source'/name;assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
 copy(source,'source/'+name)
for name,expected in intake['text'].items():
 source=base/'raw'/name;data=source.read_bytes()
 assert len(data)==expected['bytes'] and hashlib.sha256(data).hexdigest()==expected['sha256']
 copy(source,'raw/'+name)
for name in ('INTAKE.json','AUDIT_INPUT_PATHS.json','AUDIT_REQUEST.txt','CLAIM_REQUEST.txt','EVIDENCE_PRECHECK.json'):
 copy(base/name,name)
for name in ('collect_metric_scale_complete_v1.py','prepare_metric_scale_complete_audit766.py','prepare_metric_scale_claim768.py'):
 source=Path('C:/Users/gb/.codex_tmp')/name;ast.parse(source.read_text(encoding='utf-8'));copy(source,name)
copy(observer,'finish_observer/STATUS.json')
copy(Path(observed['last_receipt']),'finish_observer/0001.stdout.json')
copy(Path(__file__),Path(__file__).name)
copy(Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002/publication767_git_format_failure.json'),'PUBLICATION767_FORMAT_FAILURE.json')
copy(Path('C:/Users/gb/.codex_tmp/prepare_metric_publication_resume767.py'),'prepare_metric_publication_resume767.py')
attribute=target/'.gitattributes';attribute.write_bytes(b'raw/logs/metric_storage_finish_source766_v2_20261003/SOURCE_REVIEW.md -whitespace\n')
files.append(attribute.relative_to(repo).as_posix())
status={'recorded_at':datetime.now().astimezone().isoformat(),'status':'ALL6_PRIMARY_TEXT_RECEIVED_FRESH_REVIEWS_RUNNING',
 'accepted':6,'expected':6,'formal_epochs':300,'formal_steps':20416,'source_files':265,
 'primary_text_files':126,'remote_binary_records':24,'first_evaluation_exit_code':0,'original_report_invocations':1,
 'original_report_exit_code':0,'original_failure_preserved':True,
 'integrity_reviewer':'/root/audit_metric_scale_complete768','claim_reviewer':'/root/claim_metric_scale_complete768',
 'review_independence':'same-family','acceptance_status':'provisional','foundation_selected':False,
 'new_plan_registered':False,'new_training_started':False,'training_ports':[2026],
 'physical_gpu_scope':[0,1,2,3],'max_parallel':4,'goal':'ACTIVE_UNMET'}
(target/'STATUS.json').write_bytes((json.dumps(status,indent=2)+'\n').encode());files.append((target/'STATUS.json').relative_to(repo).as_posix())
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';path=repo/doc
old=path.read_bytes();assert hashlib.sha256(old).hexdigest()==previous['doc_sha256']
text=old.decode('utf-8');assert '## 41.768 ' not in text
headings=[line for line in text.splitlines() if line.startswith('**') and '41.767' in line];assert len(headings)==1
text=text.replace(headings[0],
 '**当前进度：§41.768（2026-10-03）** F3六端全部完整50轮/300epochs/20416steps，并已严格验收。原磁盘门失败保留；单独storage-only finish仅首次补缺失评价和首次原CPU report，均exit0。已一次接收126原文本、265源码、24远端二进制SHA记录；fresh完整性与claim审计进行中，尚未选择下一foundation或启动新训练。仅2026 GPU0–3/max4，2025不训练；Goal ACTIVE_UNMET。')
text+=f'''

## 41.768 F3全部严格终态一次接收，fresh审计进行中（{status['recorded_at']}）

新finish controller3070294于11:51:59退出0；11:53:24唯一新observer首次里程碑确认PID结束、六端strict及原CPU report第一次exit0，随即自然退出。未重启原controller2691826，未重新训练、评分重试或重选checkpoint；原exit1/FAILED和CRLF transport失败全部保留。随后一次collector接收126份原文本共49,731,550字节、265份实际源码和24项实存二进制size/SHA回执，六端300epochs、20416steps核对通过。原图、模型与距离留远端，未下载二进制或重跑report。

本轮仅改变Triplet的输入，CE保持normalized，部署仍L2_1536。以下各行都是各自同一mAP-best权重的正式评价；201同时列四项，车辆列mAP/R1。不是跨seed/epoch拼列。

| 数据集 | 条件 | best轮 | mAP | R1 | R5 | R10 |
|---|---|---:|---:|---:|---:|---:|
| RGBNT201 | normalized | 26 | 62.5806 | 62.4402 | 75.8373 | 83.0144 |
| RGBNT201 | metric_raw | 38 | 59.7343 | 60.2871 | 72.9665 | 82.4163 |
| RGBNT100 | normalized | 12 | 77.5843 | 94.1691 | — | — |
| RGBNT100 | metric_raw | 48 | 75.4520 | 94.7522 | — | — |
| MSVR310 | normalized | 24 | 50.3567 | 67.5127 | — | — |
| MSVR310 | metric_raw | 23 | 52.7596 | 71.4044 | — | — |

原CPU诊断配对差：201 −2.8462 mAP/−2.1531 R1，100 −2.1323/+0.5831，MSVR +2.4028/+3.8917。按登记mAP至少+0.5且R1不下降门，仅MSVR推进，201/100不通过；这是当前固定recipe单seed的描述性证据，不是通用原生细节贡献、三个角色证明或SOTA。100虽第一名略好，mAP仍明显下降，不能只报告R1。

所有source/主回执/完整step与batch-order/strict/原report/失败与finish归属在`logs/metric_feature_scale_complete768_20261003/`。两位fresh gpt-6-astra/max/forknone reviewer已实际启动：完整性与claim分别只读审计；现在尚无其正式终态，不能预填PASS。辅助evidence_check resolver实际未找到脚本，明确WARN，不能把它写成检验通过；实际collector与hash核对也不替代语义claim。

公开benchmark已用于逐轮mAP-best和研发选择；identity bootstrap不等于训练seed不确定性。F2同时改CE和Triplet，F1作者与当前配方还有head/optimizer/sampler/scheduler差异，不能用跨panel简单相减确定唯一失分原因。停止输入倍率、margin、seed、loss系数救F3；完整审计后才选择可信matched foundation，登记独立语义/细节读取对照并执行全模型M0。当前草稿尚未形成新的GPU实验，目标仍未达到。
'''
path.write_bytes(text.encode());shutil.copyfile(path,Path('C:/Users/gb/Desktop/document')/path.name);files.append(doc)
publication={'previous_head':previous['head'],'files':sorted(files),'section':'41.768',
 'doc_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'doc_bytes':path.stat().st_size,
 'status':status['status'],'training_ports':[2026],'max_parallel':4,'protected_files':protected}
(proof/'publication768_local.json').write_bytes((json.dumps(publication,indent=2)+'\n').encode())
publisher=Path('C:/Users/gb/.codex_tmp/publish_metric_scale_received768.py');assert not publisher.exists()
source=Path('C:/Users/gb/.codex_tmp/publish_metric_finish_launch767.py').read_text(encoding='utf-8')
source=source.replace('767','768').replace('five_copy766.json','five_copy767.json').replace('range(739,768)','range(739,769)')
source=source.replace('metric_scale_storage_finish768_20261003','metric_feature_scale_complete768_20261003')
source=source.replace('Record exact M0 retirement and first evaluation-only finish launch','Receive all six F3 strict results with original failure provenance')
source=source.replace('exact retirement and separately attributed first-evaluation finish launch;','all-six primary text/source intake with fresh reviews pending;')
publisher.write_bytes(source.encode());ast.parse(source)
print(json.dumps({key:value for key,value in publication.items() if key not in ('files','protected_files')},indent=2))
