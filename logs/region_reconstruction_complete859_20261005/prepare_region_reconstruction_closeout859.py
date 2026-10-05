"""Prepare owned terminal closeout after intake and fresh claim review."""
from datetime import datetime
import csv, hashlib, json, shutil, subprocess
from pathlib import Path
repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private=Path('C:/Users/gb/.codex_tmp')
base=private/'independent_evidence_draft'
proof=private/'foundation_recipe_v1_20261002'
intake=base/'region_reconstruction_completed_admin_intake858'
analysis=base/'region_reconstruction_complete_analysis859'
review_dir=base/'region_reconstruction_claim_review859'
record=json.loads((intake/'stdout.json').read_bytes())
report=record['summary']
visual=json.loads((analysis/'SUMMARY.json').read_bytes())
review=json.loads((review_dir/'REVIEW.json').read_bytes())
assert record['accepted']==6 and record['formal_epochs']==300 and record['formal_steps']==12968
assert record['report_invocations']==1 and record['completion_launch_exit']['exit_code']==0 and record['original_parent_exit_code']==1
assert visual['curve_rows']==750 and visual['pair_rows']==15 and visual['all_pair_progress_count']==0
assert review['claim_supported'] in ('yes','partial','no')
assert review['review_independence']=='same-family' and review['acceptance_status']=='provisional'
previous=json.loads((proof/'four_copy858_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==previous['head']
assert subprocess.check_output(['git','diff','--cached','--name-only'],cwd=repo)==b''
publication_file=proof/'publication859_local.json'
assert not publication_file.exists()
archive_relative='logs/region_reconstruction_complete859_20261005'
archive=repo/archive_relative
assert not archive.exists()
source_note='refine-logs/strong_reference_protocol_check_20261005/PRIMARY_SOURCE_CHECK.md'
assert hashlib.sha256((repo/source_note).read_bytes()).hexdigest()=='67a58b579b13f0d7dd715f903cdc685d46134796b830b2488f12c0ab5f333d39'
storage_root='logs/region_reconstruction_storage_completion858_20261005'
assert len([p for p in (repo/storage_root).rglob('*') if p.is_file()])==22
own_roots=(archive_relative+'/',storage_root+'/',source_note)
status=subprocess.check_output(['git','status','--porcelain=v1','-z','--untracked-files=all'],cwd=repo).decode()
foreign=[r[3:] for r in status.split('\0') if r and not r[3:].startswith(own_roots)]
protected={n:hashlib.sha256((repo/n).read_bytes()).hexdigest() for n in foreign}
for n,d in record['text_sha256'].items():
 assert hashlib.sha256((intake/'received'/n).read_bytes()).hexdigest()==d,n
for n,info in visual['artifacts'].items():
 assert hashlib.sha256((analysis/n).read_bytes()).hexdigest()==info['sha256'],n
archive.mkdir()
shutil.copytree(intake,archive/'intake')
shutil.copytree(analysis,archive/'analysis')
shutil.copytree(review_dir,archive/'claim_review')
for name in ('collect_region_reconstruction_completed_admin858.py','analyze_region_reconstruction_completed859.py','prepare_region_reconstruction_closeout859.py','region_reconstruction_final_training858_ADMIN.py'):
 shutil.copyfile(private/name,archive/name)
selected=list(csv.DictReader((analysis/'selected_checkpoints.csv').open(encoding='utf-8')))
chosen={(r['dataset'],r['variant']):r for r in selected}
metrics=['| 数据集 | 查询 | best轮 | mAP | R1 | R5 | R10 | best→末轮mAP下降 |','|---|---|---:|---:|---:|---:|---:|---:|']
pairs=['| 数据集 | 候选−对照 | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP | 推进门 |','|---|---|---:|---:|---:|---:|---|']
cost=['| 数据集 | 查询 | training.json训练+逐轮评价秒数 |','|---|---|---:|']
for r in report['rows']:
 c=chosen[r['dataset'],r['query_mode']];m=r['metrics']
 metrics.append(f"| {r['dataset']} | {r['query_mode']} | {r['best_epoch']} | {m['mAP']:.6f} | {m['Rank-1']:.6f} | {m['Rank-5']:.6f} | {m['Rank-10']:.6f} | {float(c['best_to_last_mAP_drop']):.6f} |")
 cost.append(f"| {r['dataset']} | {r['query_mode']} | {r['training_and_epoch_evaluation_seconds']:.6f} |")
for p in report['pairs']:
 d=p['paired_diagnosis'];delta=d['delta_metrics']
 pairs.append(f"| {p['dataset']} | {p['candidate']}−{p['control']} | {delta['mAP']:+.6f} | {delta['Rank-1']:+.6f} | {d['rank1_repairs']}/{d['rank1_new_errors']} | {d['identity_macro_mean_delta_ap_points']:+.6f} | {'PASS' if p['phase_progress'] else 'FAIL'} |")
at=datetime.now().astimezone().isoformat()
section=f"""

## 41.859 区域候选重建六端正式闭合：patch路由未取得净增益；原磁盘停机与行政补完分别保留

{at}接收全部六端、首次strict及原唯一15配对CPU报告。每端真实初始化/8M0/fresh50/单一mAP-best/首次重载全部验收，共300正式轮、12,968更新；六端batch顺序与各自数据集三个sealed RAW控制逐字节一致。执行科学来源354项、RAW九行187件实存SHA未变。仅26物理GPU0/1，无功率温度动作、25探测、新种子或旧五端重训。

本轮是原CNN语义patch在16区域读取前残差查询视觉anchor，patch与mean均105,232活动参数/15张量、完整128个query位置；mean将每模态图像均值广播，不是原生图像CNN。两者均有逐槽位图像条件anchor，因此不能据本轮证明条件anchor必要。RAW职责分离、作者head、旧桥接/gain、1536维L2部署、完整gallery与camera/scene过滤不变；没有文本、FFN、额外loss或N2/N3。

{chr(10).join(metrics)}

所有列跟随各端同一mAP-best，车辆R5/R10仅展开原保存CMC，不替代mAP/R1主指标。原保存距离报告15配对如下；CPU配对CMC浮点表示遵从原报告，不覆盖正式receipt四项。

{chr(10).join(pairs)}

三个patch−mean主配对及全部15门均0通过。ΔmAP≥0.5且R1不降是事前项目推进线，不是显著性。201 patch−mean为−0.338730 mAP、−0.119617 R1；MSVR仅+0.000801且CMC相同，不构成等价或有效证据；100为−0.059234 mAP、+0.874636 R1。100相对RAWsemantic虽然46首位修复/33新增、R1提高0.758017，mAP仍−0.591102；相对独立global mAP−1.034582，首位39修复/39新增。mean在201比RAWsemantic多0.303619、比global多0.381817，仍未达原线且未跨数据集成立。不得选择有利R1覆盖mAP、将微小差值写成因果/等价/来源无用，或把一次seed42写成稳定机制成功。

完整15模型/750轮文本曲线、15,710逐query差异、660身份配对行、全部best与末轮以及成本图表见`{archive_relative}/analysis/`，实际PNG已查看。100本轮best→末轮仅1.142334/1.027495，不复现旧joint-L2二十多点下降；这是不同固定目标的行为观察，不是单因素根因证明。201下降2.790450/3.285612，MSVR约0.06。epoch与固定模型身份bootstrap都不等于独立训练种子。

{chr(10).join(cost)}

上表是training.json记录训练及逐轮评价区间，不包含构造、最终strict、补完等待与清理，更不能用旧history.seconds训练循环字段代替完整成本。

原父在20:15:09最后100-mean已M0通过、正式训练尚未启动时触发原2GiB磁盘门，原EXIT1/campaign/console三个SHA不改。已有5接受/250轮/9839更新保留。20:47:07另立行政补完，只首次启动原最后fresh50与首次strict，并执行原15配对报告一次；22:50:51新监督退出0，22:51:02独立观察确认终态。新训练次数1、继承五接受；不是重训、评价重试、报告重跑、挑权重或把原停机改成成功。全部原始失败、补完计划/代码/启动与164终态文本SHA见本归档和`{storage_root}/`。

为恢复固定存储预算，20:41:34只退役三份已完成且被保留权重四指标支配的旧capacity best，1,074,550,093B；20:46:03只退役F2/F3四份重复normalized201/100 best，1,384,184,748B。合计7份/2,458,734,841B。先核各自完整50轮、首次回执/距离、实际weightSHA及保留winner，不属于当前354/187依赖；F1原始、作者/public初始化、RAW控制及当前必要best保持。原分数、历史和距离不改；这七个旧binary的直接重载能力已退役。最初本地准备脚本仅f-string插值失败、SSH/删除尚未执行，失败副本仍留私有记录，不能算NN失败；两个实际退役journal原样保留。

已追加RoDI与DEEP原论文/当前官方树只读核查：24项主表数值与既有记录一致，RoDI的CLIP/DINOv3资源分列，DEEP为60轮且推理仍有图像条件prompt与冻结文本分支。当前官方树没有可复核训练/评价入口；精确过滤、reranking、测试更新与成本未知处不补猜。见`{source_note}`；论文报告不等于本地复现或同协议SOTA。

result-to-claim新鲜上下文复核 verdict=`{review['claim_supported']}`，same-family/provisional，见本归档claim_review。ARIS evidence_check.py规范helper未解析已记录；不把终态SHA存在性检查当语义成功，运行后端也未独立证明。原整体验证可重放范围与旧失败口径保持。

本轮patch候选重建不晋级有效主模块，mean广播也没有跨三集净增益支持。下一结构尚未登记或启动，不按官方分数继续调LR/gain/margin/query数/种子，不自动在当前失败结构上叠N2/N3。先依据完整对照锁定下一个单一问题及直接活动控制，再实现；完整Goal ACTIVE/UNMET，同匹配强基线净增益、三个必要机制、完整流程多种子与同资源SOTA仍缺。仅在所有NN/首次评价/CPU报告终态后更新并同步本节；2025原I/O pending继续保留，不探测、不宣称镜像一致。
"""
doc_name='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
doc=repo/doc_name
assert hashlib.sha256(doc.read_bytes()).hexdigest()==previous['doc_sha256']
original=doc.read_text(encoding='utf-8');assert '## 41.859 ' not in original
doc.write_bytes((original+section).encode('utf-8'))
desktop=Path('C:/Users/gb/Desktop/document')/doc.name
shutil.copyfile(doc,desktop);assert doc.read_bytes()==desktop.read_bytes()
goal=repo/'refine-logs/CURRENT_GOAL.md'
lines=goal.read_text(encoding='utf-8').splitlines();assert lines[2].startswith('更新：')
lines[2]=f'更新：{at} §41.859。区域patch/mean六端fresh50/首次strict/唯一15配对报告闭合，300轮/12968更新，主门0/3、全部0/15；不晋级当前候选重建。原父磁盘EXIT1保持，新行政只补原最后首次训练/strict/report后EXIT0。当前无活动NN，后继尚未登记；不以gain/LR/margin/seed救分或自动叠N2/N3。仅26 GPU0/1；无功率温度动作；2025原I/O pending。Goal ACTIVE/UNMET。'
goal.write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
tracker=repo/'refine-logs/region_evidence_reconstruction_v1/EXPERIMENT_TRACKER.md'
old=tracker.read_text(encoding='utf-8');assert '§41.859' not in old
tracker.write_bytes((old+'\n\n## §41.859 终态（覆盖上文准备状态）\n\n'+'\n'.join(metrics)+'\n\n六端fresh50/首次strict接受、原15配对一次完成；推进0/15，候选重建不晋级。原父磁盘EXIT1与最后首次训练的行政补完EXIT0分别保留。\n').encode('utf-8'))
findings=repo/'refine-logs/region_evidence_reconstruction_v1/findings.md'
assert not findings.exists()
findings.write_bytes(('本轮主张复核：'+review['claim_supported']+'；same-family/provisional。\n\n'+(review_dir/'CLAIMS_FROM_RESULTS.md').read_text(encoding='utf-8')+'\n\n约束：本轮局部query不获净增益；不按已消费官方成绩调LR/gain/margin/seed；下一项不得用更多anchor或残差能量替代互补证据检验。\n').encode('utf-8'))
owned=[doc_name,'refine-logs/CURRENT_GOAL.md',tracker.relative_to(repo).as_posix(),findings.relative_to(repo).as_posix(),source_note]
for directory in (archive,repo/storage_root):
 owned.extend(p.relative_to(repo).as_posix() for p in directory.rglob('*') if p.is_file())
manifest=repo/'MANIFEST.md'
with manifest.open('a',encoding='utf-8',newline='\n') as stream:
 stream.write('\n\n## 2026-10-05 §41.859 六端候选重建终态\n\n| Output | Provenance |\n|---|---|\n')
 for name in owned:stream.write(f'| `{name}` | 原完整终态/保存文本分析/只读复核；无模型重跑 |\n')
owned.append('MANIFEST.md')
assert all(hashlib.sha256((repo/n).read_bytes()).hexdigest()==d for n,d in protected.items())
publication=dict(previous_head=previous['head'],section='41.859',files=sorted(set(owned)),protected_files=protected,
 immutable_archive_roots=[archive_relative,storage_root],remote_sparse_roots=[archive_relative,storage_root,'refine-logs/strong_reference_protocol_check_20261005'],doc_sha256=hashlib.sha256(doc.read_bytes()).hexdigest())
publication_file.write_bytes((json.dumps(publication,indent=2)+'\n').encode())
print(json.dumps(dict(status='LOCAL_REGION_CLOSEOUT_PREPARED_NOT_PUBLISHED',owned_files=len(publication['files']),protected_unrelated_files=len(protected),claim_supported=review['claim_supported'],doc_sha256=publication['doc_sha256'])))
