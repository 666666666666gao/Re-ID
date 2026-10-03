"""Publish real retirement and separately attributed evaluation-only launch."""
from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess,ast

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp/metric_feature_scale_failed_closeout766')
proof=Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
previous=json.loads((proof/'five_copy766.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
protected=json.loads((proof/'publication766_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest()==value for name,value in protected.items())
launch=json.loads((base/'deploy_v2/LAUNCH.json').read_bytes())
assert launch['port']==2026 and launch['gpu']==1
retirement=json.loads((base/'retirement/REMOTE_RETIREMENT_BYTES.json').read_bytes())
assert retirement['status']=='RETIRED_EXACT_24_CLOSED_M0' and retirement['retired_bytes']==8444122668
review=json.loads((base/'launcher_revision/SOURCE_REVIEW.json').read_bytes())
assert review['verdict'] in ('PASS','WARN') and review['blocking_findings']==[]
target=repo/'logs/metric_scale_storage_finish767_20261003'
assert not target.exists() and not (proof/'publication767_local.json').exists();target.mkdir()
sources={
 'original_source_review/SOURCE_REVIEW.md':base/'reviewer_source/SOURCE_REVIEW.md',
 'original_source_review/SOURCE_REVIEW.json':base/'reviewer_source/SOURCE_REVIEW.json',
 'retirement/REMOTE_RETIREMENT_BYTES.json':base/'retirement/REMOTE_RETIREMENT_BYTES.json',
 'retirement/TRANSPORT_DIAGNOSIS.json':base/'retirement/TRANSPORT_DIAGNOSIS.json',
 'retirement/EXIT.json':base/'retirement/EXIT.json',
 'failed_transport/EXIT.json':base/'deploy/EXIT.json',
 'failed_transport/stderr.txt':base/'deploy/stderr.txt',
 'launcher_revision/SOURCE_REVIEW.md':base/'launcher_revision/SOURCE_REVIEW.md',
 'launcher_revision/SOURCE_REVIEW.json':base/'launcher_revision/SOURCE_REVIEW.json',
 'deploy_v2/LAUNCH.json':base/'deploy_v2/LAUNCH.json',
 'deploy_v2/EXIT.json':base/'deploy_v2/EXIT.json',
 'deploy_metric_storage_finish766_v2.py':Path('C:/Users/gb/.codex_tmp/deploy_metric_storage_finish766_v2.py'),
 'collect_retirement_bytes766.py':Path('C:/Users/gb/.codex_tmp/collect_retirement_bytes766.py'),
 'observe_metric_storage_finish766.py':Path('C:/Users/gb/.codex_tmp/observe_metric_storage_finish766.py'),
 'prepare_metric_finish_launch767.py':Path(__file__),
}
files=[]
for name,source in sources.items():
 if source.suffix=='.py':ast.parse(source.read_text(encoding='utf-8'),filename=str(source))
 path=target/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
 assert path.read_bytes()==source.read_bytes();files.append(path.relative_to(repo).as_posix())
doc='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
path=repo/doc;old=path.read_bytes();assert hashlib.sha256(old).hexdigest()==previous['doc_sha256']
text=old.decode('utf-8');assert '## 41.767 ' not in text
headings=[line for line in text.splitlines() if line.startswith('**') and '41.766' in line];assert len(headings)==1
text=text.replace(headings[0],
 f"**当前进度：§41.767（2026-10-03）** 原F3磁盘门失败已封存，24份旧已结项M0权重实退役8.44GB，当前F3与正式best保留。fresh源码复核无阻断后，storage-only finish已在2026 GPU1单次启动，PID{launch['controller_pid']}；只补齐原未启动评价与首次CPU report，完成及六端审计尚未接收。仅2026 GPU0–3/max4；2025不训练。Goal ACTIVE_UNMET。")
text+=f'''

## 41.767 旧M0实退役及storage-only finish实际启动（{datetime.now().astimezone().isoformat()}）

fresh gpt-6-astra/max源码复核为WARN、无阻断项，same-family/provisional，未执行模型。实际退役于{retirement['completed_at']}结束：严格按24份清单逐一核对path/size/SHA、旧campaign终态和owner结束、正式best/currentF3保护；实删8,444,122,668字节，free由{retirement['free_bytes_before']}到{retirement['free_bytes_after']}字节。没有扩大删除范围；原F1在2025的基础权重未动。旧M0不再能直接二进制重放，但其原回执、正式best与文本保留。

首次transport在Popen之前失败：本地`RETIREMENT.json`写成CRLF，远端原回执为LF，同一JSON对象但字节摘要不同。原stderr/exit1保留，11:41:38实查finish目录、launcher.log、LAUNCH均不存在。一次接收实际远端132376字节，SHA`63956ff16251c212b37ec1264c58878e4ccab46ffed1721f6f8da499ebba4645`，未改原记录、未重新清理或重试原失败脚本。

新v2 transport仅改为绑定实接收的远端字节，并使用独立asset目录和deploy_v2输出；相同reviewer重新复核{review['verdict']}、无阻断。原科学source265、finish helper、checkpoint、协议、seed、loss及strict1e-5容差不变。真实launch时间{launch['observed_at']}，新controller PID{launch['controller_pid']}，GPU1，空闲与10GiB空间门实际通过。这是首次缺失evaluate和首次原CPU report的单独归属controller，不是重启原2691826，原exit1/FAILED仍归档。启动不等于评价或汇总已通过。

launch/原失败transport/两轮源码review/真实退役回执在`logs/metric_scale_storage_finish767_20261003/`，源码与原FAILED见§766。新只读observer按240秒节奏观察新controller，旧observer不重启。完整结果应在实际评价exit0、原report第一次exit0、finish PID结束后一次接收，再做fresh完整性与claim审计；当前尚未据此选择新foundation或登记新训练。所有新训练限制2026物理0–3四张卡/max4；2025仅文本同步。
'''
path.write_bytes(text.encode('utf-8'));shutil.copyfile(path,Path('C:/Users/gb/Desktop/document')/path.name);files.append(doc)
publication={'previous_head':previous['head'],'files':sorted(files),'section':'41.767',
 'doc_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'doc_bytes':path.stat().st_size,
 'status':'STORAGE_ONLY_FINISH_ACTUALLY_STARTED_COMPLETION_NOT_YET_RECEIVED',
 'training_ports':[2026],'max_parallel':4,'protected_files':protected}
(proof/'publication767_local.json').write_bytes((json.dumps(publication,indent=2)+'\n').encode())
publisher=Path('C:/Users/gb/.codex_tmp/publish_metric_finish_launch767.py');assert not publisher.exists()
source=Path('C:/Users/gb/.codex_tmp/publish_metric_failure766.py').read_text(encoding='utf-8')
source=source.replace('766','767').replace('five_copy765.json','five_copy766.json').replace('range(739,767)','range(739,768)')
source=source.replace('metric_scale_disk_failure767_20261003','metric_scale_storage_finish767_20261003')
source=source.replace('Record original F3 disk-guard failure and pending storage-only finish','Record exact M0 retirement and first evaluation-only finish launch')
source=source.replace('original disk-guard failure and pending storage-only finish;','exact retirement and separately attributed first-evaluation finish launch;')
publisher.write_bytes(source.encode());ast.parse(source)
print(json.dumps({key:value for key,value in publication.items() if key not in ('files','protected_files')},indent=2))
