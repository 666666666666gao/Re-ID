"""Append the actual F3 disk-guard failure and pending finish preparation."""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess,ast

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
previous=json.loads((proof/'five_copy765.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
protected=json.loads((proof/'publication765_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==value for name,value in protected.items())
intake=json.loads((base/'INTAKE.json').read_bytes())
assert intake['controller_state']['status']=='FAILED' and intake['controller_state']['report_invocations']==0
target=repo/'logs/metric_scale_disk_failure766_20261003'
assert not target.exists() and not (proof/'publication766_local.json').exists()
target.mkdir()
files=[]
sources={
 'INTAKE.json':base/'INTAKE.json',
 'STORAGE_FINISH_PLAN.md':base/'STORAGE_FINISH_PLAN.md',
 'FINISH_SPEC.json':base/'FINISH_SPEC.json',
 'retirement/SPEC.json':base/'retirement/SPEC.json',
 'storage_inventory/INVENTORY.json':base/'storage_inventory/INVENTORY.json',
 'original_failed_campaign.json':base/'raw/logs/metric_feature_scale_20261003_v1/campaign.json',
 'original_failed_worker.log':base/'raw/logs/metric_feature_scale_20261003_v1/full_metric_raw_RGBNT100.log',
 'original_train_only_child.json':base/'raw/logs/metric_feature_scale_20261003_v1/metric_feature_scale_20261003_v1_full_metric_raw_RGBNT100/campaign.json',
 'SOURCE_REVIEW_REQUEST.txt':base/'SOURCE_REVIEW_REQUEST.txt',
 'retire_closed_m0_766.py':Path('C:/Users/gb/.codex_tmp/retire_closed_m0_766.py'),
 'finish_metric_feature_scale766.py':Path('C:/Users/gb/.codex_tmp/finish_metric_feature_scale766.py'),
 'deploy_metric_storage_finish766.py':Path('C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766.py'),
 'prepare_metric_failure766.py':Path(__file__),
}
for name,source in sources.items():
    if source.suffix=='.py':ast.parse(source.read_text(encoding='utf-8'),filename=str(source))
    path=target/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
    assert path.read_bytes()==source.read_bytes()
    files.append(path.relative_to(repo).as_posix())
status={'recorded_at':datetime.now().astimezone().isoformat(),
 'status':'ORIGINAL_F3_DISK_GUARD_FAILURE_FINISH_PREPARED_REVIEW_PENDING',
 'original_controller_pid':2691826,'original_controller_ended':True,
 'formal_training_complete':6,'formal_training_epochs':300,'strict_accepted':5,'expected':6,
 'original_report_invocations':0,'retirement_executed':False,'finish_launched':False,
 'source_review':'RUNNING_NOT_RUNTIME_ACCEPTANCE','training_ports':[2026],
 'physical_gpu_scope':[0,1,2,3],'max_parallel':4,'goal':'ACTIVE_UNMET'}
(target/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf-8')
files.append((target/'STATUS.json').relative_to(repo).as_posix())
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
path=repo/doc;old=path.read_bytes();assert hashlib.sha256(old).hexdigest()==previous['doc_sha256']
text=old.decode('utf-8');assert '## 41.766 ' not in text
headings=[line for line in text.splitlines() if line.startswith('**') and '41.765' in line]
assert len(headings)==1
text=text.replace(headings[0],
 '**当前进度：§41.766（2026-10-03）** 原F3六端均完成50轮，但RGBNT100 metric_raw在评价前触发10 GiB磁盘检查，原controller已FAILED退出。正式验收仍5/6，原CPU report调用0次。已收齐原失败文本与实际源码，24份旧已结项M0权重退役及仅补齐未启动评价的脚本正接受fresh源码复核，尚未执行。仅2026 GPU0–3/max4；2025不训练。独立原生细节方法仍是未登记草稿；Goal ACTIVE_UNMET。')
append=f'''

## 41.766 原F3磁盘门失败与仅补齐未启动评价的准备（{status['recorded_at']}）

原六端训练均跑满50轮；最后RGBNT100 metric_raw共6559步，best为第48轮，训练记录mAP75.45198460760588、R1 94.75218653678894。这些是训练时的记录，尚未通过独立重载验收，不能提前填为第六项正式结果。

原训练于10:59:28结束、训练子进程exit0。原worker于11:02:54 exit1，原controller2691826写入FAILED；traceback停在`queue_foundation_recipe.py:130`的`shutil.disk_usage(ROOT).free >= 10 * 1024**3`。独立evaluate未调用：没有evaluate.log、official_metrics.json或official_distances.pt，accepted_matrix及report目录也不存在。11:06原sole observer观察到terminal-without-report后自然退出，不重启旧controller或observer。11:11实际空间为4,519,247,872字节。

已一次接收103份原文本、265份实际源码及24项二进制存在/SHA记录，其中23项存在；唯一缺项是尚未生成的RGBNT100 metric_raw正式距离。原FAILED parent、train-only child、worker traceback全部保留。不存在重新训练、重选checkpoint或放宽原1e-5容差的授权变更。

清理范围为24份已完成并发布的旧工程M0 `m0_reload_probe.pth`，清单合计8,444,122,668字节：旧视觉更新12份、干净起点6份、旧F2六份。它们的正式best、文本、距离、作者和公共权重及当前F3全部23项依赖保留；F1在2025的六份基础权重不属于清单。旧M0二进制退役会限制历史工程重载，必须披露，不能称历史全部仍可从二进制重放。

storage-only finish单独归属：待fresh源码复核无阻断、实存SHA及旧campaign终态确认、实际释放空间后，只第一次启动原缺失evaluate，严格通过后核齐六端，第一次调用未经修改的原CPU report。保存原FAILED字节和真实exit1，不伪造原worker成功。使用2026的一张空闲卡，所有新实验仍限定其物理0–3四张卡、最多4并行；2025只同步文本，不启动训练。

本节是失败事实与准备记录：fresh review仍在运行，权重尚未退役，finish尚未启动，六端完整汇总及fresh完整性/claim审计均未完成。准备源码、清单、原失败回执和INTAKE在`logs/metric_scale_disk_failure766_20261003/`；新的独立双来源方法尚未选择foundation、登记plan、执行全模型M0或正式训练。总体性能目标仍未达到。
'''
path.write_bytes((text+append).encode('utf-8'))
shutil.copyfile(path,Path('C:/Users/gb/Desktop/document')/path.name);files.append(doc)
publication={'previous_head':previous['head'],'files':sorted(files),'section':'41.766',
 'doc_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'doc_bytes':path.stat().st_size,
 'status':status['status'],'training_ports':[2026],'max_parallel':4,'protected_files':protected}
(proof/'publication766_local.json').write_text(json.dumps(publication,indent=2)+'\n',encoding='utf-8')
publisher=Path('C:/Users/gb/.codex_tmp/publish_metric_failure766.py');assert not publisher.exists()
source=Path('C:/Users/gb/.codex_tmp/publish_native_author_entry_option765.py').read_text(encoding='utf-8')
source=source.replace('765','766').replace('five_copy764.json','five_copy765.json').replace('range(739,766)','range(739,767)')
source=source.replace('native_evidence_author_entry_option766_20261003','metric_scale_disk_failure766_20261003')
source=source.replace('Prepare full50 author-option entry for independent native evidence','Record original F3 disk-guard failure and pending storage-only finish')
source=source.replace('unregistered full50 entry source option and AST check only;','original disk-guard failure and pending storage-only finish;')
publisher.write_text(source,encoding='utf-8');ast.parse(source)
print(json.dumps({key:value for key,value in publication.items() if key not in ('files','protected_files')},indent=2))
