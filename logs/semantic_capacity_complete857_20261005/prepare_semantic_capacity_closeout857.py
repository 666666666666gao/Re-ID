"""Prepare local closeout from the accepted original campaign; run only after intake."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
base = private / 'independent_evidence_draft'
proof = private / 'foundation_recipe_v1_20261002'
intake = base / 'semantic_capacity_complete_intake856'
analysis = base / 'semantic_capacity_text_analysis857'
record = json.loads((intake / 'stdout.json').read_bytes())
summary = json.loads((analysis / 'SUMMARY.json').read_bytes())
assert record['status'] == 'CAPACITY_COMPLETE_WITH_FIRST_EVAL_CONTINUATION_VERIFIED'
assert record['formal_completed'] == 3 and record['formal_epochs'] == 150
assert record['formal_steps'] == 6484 and record['report_invocations'] == 1
assert record['completion_launch_exit']['exit_code'] == 0
assert record['original_launch_exit']['exit_code'] == 1 and record['new_training_invocations'] == 0
assert summary['status'] == 'ACCEPTED_REPORT_TEXT_VISUALIZATION_COMPLETE'
assert summary['curve_rows'] == 600 and summary['selected_rows'] == 12
previous_head = '60e1c4dec1f21a772cde7c703cb29ad2b5ee61bd'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous_head
assert subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo) == b''
publication_file = proof / 'publication857_local.json'
assert not publication_file.exists()
archive_relative = 'logs/semantic_capacity_complete857_20261005'
archive = repo / archive_relative
assert not archive.exists()
doc_relative = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
doc = repo / doc_relative
assert hashlib.sha256(doc.read_bytes()).hexdigest() == '040b9015acee6f5548fb96c61f23882ef7c54820d2c52ab1b74b40fe859f6eb6'
original_doc = doc.read_text(encoding='utf-8')
assert '## 41.857 ' not in original_doc

foreign = [repo / '.aris/meta/events.jsonl', repo / 'tools/run_trifusion_experiment.py']
foreign.extend(p for p in (repo / 'refine-logs/cross_depth_role_state_v1/initialization_source_check_666_20260929').rglob('*') if p.is_file())
assert len(foreign) == 13
protected = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in foreign}
source_note = repo / 'refine-logs/region_evidence_reconstruction_v1/PRIMARY_SOURCE_CHECK.md'
assert hashlib.sha256(source_note.read_bytes()).hexdigest() == '24b44e38a928b4c5f7f87ca04780ec39c0457d3bc53dd1c722c60bb5512dde9c'

for name, info in record['files'].items():
    body = (intake / 'received' / name).read_bytes()
    assert len(body) == info['bytes'] and hashlib.sha256(body).hexdigest() == info['sha256']
for name, info in summary['artifacts'].items():
    body = (analysis / name).read_bytes()
    assert len(body) == info['bytes'] and hashlib.sha256(body).hexdigest() == info['sha256']

archive.mkdir()
shutil.copytree(intake, archive / 'complete_intake')
shutil.copytree(analysis, archive / 'text_analysis')
partial_directory = base / 'semantic_capacity_first_two_saved_trajectory857'
partial = json.loads((partial_directory / 'SUMMARY.json').read_bytes())
assert partial['status'] == 'FIRST_TWO_ACCEPTED_SAVED_TEXT_TRAJECTORIES_ANALYZED'
assert partial['underlying_epoch_observations'] == 300 and partial['paired_epoch_rows'] == 200
for name, info in partial['inputs'].items():
    body = Path(name).read_bytes()
    assert len(body) == info['bytes'] and hashlib.sha256(body).hexdigest() == info['sha256']
shutil.copytree(partial_directory, archive / 'first_two_saved_trajectory')
for alias, name in (
    ('first201', 'semantic_capacity_first_full_intake856'),
    ('secondMSVR', 'semantic_capacity_msvr_full_intake856'),
    ('start100', 'semantic_capacity_rgb100_start_milestone856'),
    ('originalStop', 'semantic_capacity_campaign_complete_milestone856'),
    ('scaleCleanup', 'semantic_capacity_obsolete_scale_weight_retirement856'),
    ('evalLaunch', 'capacity_first_eval_completion_launch857'),
    ('evalTerminal', 'capacity_first_eval_completion_milestone857'),
    ('metricCleanup', 'capacity_first_eval_closed_metric_retirement857')):
    shutil.copytree(base / name, archive / 'steps' / alias)
helpers = ('collect_semantic_capacity_complete856.py', 'analyze_semantic_capacity_text857.py',
           'prepare_semantic_capacity_closeout857.py', 'publish_semantic_capacity_complete857.py',
           'retire_obsolete_scale_weights856.py', 'wait_semantic_capacity_complete856.py')
for name in helpers:
    shutil.copyfile(private / name, archive / name)
shutil.copyfile(private / 'analyze_semantic_capacity_first_two_saved857.py',
                archive / 'analyze_semantic_capacity_first_two_saved857.py')
for name in ('capacity_first_eval_completion857_ADMIN.py', 'launch_capacity_first_eval_completion857.py',
             'wait_capacity_first_eval_completion857.py', 'retire_closed_joint_metric_bests857.py'):
    shutil.copyfile(private / name, archive / name)

datasets = ('RGBNT201', 'RGBNT100', 'MSVR310')
rows = {r['dataset']: r for r in record['rows']}
pairs = {(p['dataset'], p['control']): p for p in record['pairs']}
assert set(rows) == set(datasets) and len(pairs) == 9
gate_count = sum(p['phase_progress'] for p in record['pairs'])
metric_table = ['| 数据集 | best轮次 | mAP | R1 | R5 | R10 | best→末轮ΔmAP |',
                '|---|---:|---:|---:|---:|---:|---:|']
pair_table = ['| 数据集 | 对照 | ΔmAP | ΔR1 | 首位修复/新增错误 | 身份宏平均ΔAP | 推进门 |',
              '|---|---|---:|---:|---:|---:|---|']
cost_table = ['| 数据集 | 记录的训练+逐轮评价秒数 |', '|---|---:|']
for dataset in datasets:
    row = rows[dataset]
    metrics = row['metrics']
    cost_table.append(f"| {dataset} | {row['complete_training_and_epoch_eval_seconds']:.6f} |")
    metric_table.append(f"| {dataset} | {row['best_epoch']} | {metrics['mAP']:.6f} | {metrics['Rank-1']:.6f} | {metrics['Rank-5']:.6f} | {metrics['Rank-10']:.6f} | {row['best_to_last_map_drop']:.6f} |")
    for control in ('raw_semantic', 'raw_native', 'raw_global_only'):
        pair = pairs[dataset, control]
        delta = pair['delta']
        pair_table.append(f"| {dataset} | {control} | {delta['mAP']:+.6f} | {delta['Rank-1']:+.6f} | {pair['rank1_repairs']}/{pair['rank1_new_errors']} | {pair['identity_macro_mean_delta_ap_points']:+.6f} | {'PASS' if pair['phase_progress'] else 'FAIL'} |")

at = datetime.now().astimezone().isoformat()
claim_boundary = (
    f'九项预登记推进门中{gate_count}/9通过；推进门为ΔmAP≥0.5且ΔR1≥0，不是统计显著性。'
    '全部是seed42、各自同一mAP-best的开发结果，不能据微小差值证明等价、原生信息完全无用、'
    '唯一因果、跨种子稳定性或SOTA。近容量控制差200参数约0.126%，FLOPs、输入与梯度路径不完全匹配。'
)
next_boundary = (
    '容量对照到此闭合，当前原生/近容量额外读取均不晋级为主模块。下一项优先选原raw-semantic的CNN语义patch，'
    '不保留未获净收益的原生CNN或额外MLP；在原16区域读取前研究patch查询视觉anchor的残差候选重建。'
    '必须加入同参数、实际运行的整图均值query重复到各patch的控制，分清局部路由与广播上下文/容量。'
    '共享anchor的图像条件偏移须逐槽位输出，不把公共平移当局部选择；仅新增一个零输出出口保留初始语义行为。'
    '保持raw职责目标、原读取/桥接、1536维、配方、采样、seed42/fresh50和单一best；不改gain/梯度切断/Triplet尺度/'
    'LR/margin/seed，不同时加N2/N3。此为视觉-only项目改写，非完整SAGA复现。后继代码和正式合同尚未登记或启动。'
)
section = f'''\n\n## 41.857 近容量语义控制三端正式闭合：分清来源与额外容量；后继区域证据组织尚未启动

{at}接收原三端训练及明确首次评价补完的终态证据。原队列13:59:42在100的fresh50正常EXIT0后、首次evaluate启动前触发2GiB磁盘门；14:01:14原观察器确认父进程EXIT1、仅两端接受、report次数0。原campaign.json、EXIT1及console日志SHA不改。14:19:04另行启动仅首次strict100/首次报告的行政补完，14:22:33退出0，14:22:56原新观察器确认三接受/report1/EXIT0；新增训练次数0。三个数据集各自完成真实初始化匹配、8更新M0、原fresh50、第一次严格重载及验收，共150轮/6484正式更新。M0探针仅在各自验收后退役；没有重训、评价重试、报告重跑或重新选权重。NN科学来源仍为`60e1c4de`的345项；9行/187件raw控制SHA未变。各新端训练batch顺序与该数据集三个旧控制逐字节相同。

新增来源为CNN角色语义patch→128/202/202/128 MLP→512候选独立读取，共159,096参数/14张量。它没有增加原生图像内容，插值不生成新的高分辨率信息；内部native factory名称不改变该来源事实。raw全局/角色职责目标及作者配方、1536维L2部署接口不变。下表使用原正式同一best回执；车辆R5/R10作为已保存诊断同时列出，主结果仍报告mAP/R1。

{chr(10).join(metric_table)}

所有source/control配对均来自原唯一CPU保存距离报告，首位修复和身份收益按全query/全身份统计。CPU配对CMC的浮点表示不覆盖原正式重载CMC；表中Δ遵从原报告。

{chr(10).join(pair_table)}

{claim_boundary}

科学结论：目前没有证据支持原生图像来源的必要性，也不能说额外读取容量已成为有效解。201/MSVR的近容量控制与native差分别−0.023333/−0.002430 mAP；100则比native高0.427230，但R1下降0.466472，仍低于原semantic 0.499840和独立global 0.943319 mAP。100相对semantic虽首位30修复/22新增、R1正，全部正例排序和身份宏平均AP却下降；不能选择有利CMC覆盖mAP。100的cap-best E9、semantic E5、native E26、global E7不同，因此独立方法差包含选点/训练结果差，不能全部归为同权重内的角色效应。薄单种子差值不证明来源等价或永久无用；当前只是不晋级这两条额外读取设计，转向更直接的候选内容组织。

完整12模型/600轮文本轨迹、九配对、全部query/身份CSV和SVG/PNG图见`{archive_relative}/text_analysis/`。原训练、首strict回执、输入SHA、batch顺序以及观测/清理执行来源保存在同一归档。训练循环时间、训练加逐轮评价时间、完整监督进程墙钟应按各字段原定义分开，不能混称完整成本。

{chr(10).join(cost_table)}

上述为training.json记录区间及逐轮评价，排除构造/最终strict；不是完整端到端成本，更不包括磁盘中断后的等待和清理。图表经实际查看，展示全部50轮及所有九配对，不从不同epoch拼列、不以epoch数当种子数。

训练期间只读既有两端的6模型/300轮文本，并与四份已发布raw控制CSV核对：四个配对的50轮global loss和global范数均值最大差均0。201 capacity相对raw-native的逐轮mAP有38高/12低，但其原best仍略低；相对raw-semantic有18高/32低。MSVR对应15高/35低和44高/6低，差值很小。这些是相关轮次，不是训练重复或显著性；相同标量均值不证明完整state或query特征相同。额外支路已活动，仍不能据此宣布有用互补。旧下载文本由Windows写入CRLF，第一次本地字节比较因此失败；已验证仅换行不同，按原捕获正文核对并保留两种字节SHA，没有改变文件、数值或模型结果。详细四对逐轮数据见`first_two_saved_trajectory/`。

11:38:36仅退役四份已完成负向F2/F3的201/100自训best，共1,384,184,748字节；先核50轮、原回执/距离及实存SHA，且均不属于当前345/187依赖。原日志、指标和距离保留；这些权重的直接重载能力已经退役。MSVR正向尺度实验、作者/公开初始化、当前raw控制及新三端best保留。原清理journal字节原样归档，不修写历史记录，也不再次删除。

磁盘停止后，14:13:45又按既有清理授权退役已闭合、不再选择的四份joint-metric候选best，1,429,428,514字节；保留该阶段最高201-native best。删除前核原五模型seal中的权重/回执/历史SHA、完整50轮、两份原距离及不属于当前345/187依赖；删除后核原失败三文件SHA不变。余量858,386,432→2,287,820,800字节，2GiB门未放宽。旧四权重及原五模型seal中相关binary直接重放已退役；原成绩、距离及文本证据保留。100自己的M0探针在首次strict通过之前保留，之后依原规则退役。

已补实际SAGA论文/作者实现的只读来源核查，见`refine-logs/region_evidence_reconstruction_v1/PRIMARY_SOURCE_CHECK.md`。作者实现是patch查询anchor、残差attention加FFN、共享文本anchor与动态anchor拼接；与视觉-only/shared-plus-delta提案不同。没有复制作者实现、执行SAGA模型或迁移其成绩。共同偏移在纯仿射key路径的softmax中抵消仅是设计代数边界，不是已定位的TriFusion错误原因。

{next_boundary}

只有原训练及首次评价补完/CPU报告均终态后才准备本次文本/图表发布与远端同步。原失败仍为EXIT1/RUNNING状态字节，不把行政补完计作原父进程成功；新状态显式继承三端训练来源。仅2026物理GPU0/1，功率/温度不查询、设置或监控；2025原I/O pending不探测、不假称其镜像一致。旧parity/M0/诊断失败不改判。完整Goal ACTIVE / UNMET；强基线、机制必要性、完整流程多种子与同资源SOTA仍未达到。
'''
doc.write_bytes((original_doc + section).encode('utf-8'))
desktop = Path('C:/Users/gb/Desktop/document') / doc.name
shutil.copyfile(doc, desktop)
assert desktop.read_bytes() == doc.read_bytes()

goal = repo / 'refine-logs/CURRENT_GOAL.md'
goal_text = goal.read_text(encoding='utf-8')
lines = goal_text.splitlines()
assert lines[0].startswith('# ')
new_top = f'更新：{at} §41.857。近容量语义三端fresh50及原一次报告已闭合，150轮/6484更新，九配对推进{gate_count}/9。raw控制187及执行来源345不变；没有活动训练或新后继启动。{next_boundary}只用26 GPU0/1；无功率温度动作；2025 I/O pending。Goal ACTIVE / UNMET。'
assert lines[2].startswith('更新：')
lines[2] = new_top
goal.write_bytes(('\n'.join(lines) + '\n\n§41.857当前边界：本节顶部覆盖旧PREPARED及旧观察状态；原capacity训练/报告终态，下一机制须先明确登记。\n').encode('utf-8'))

tracker = repo / 'refine-logs/semantic_capacity_control_v1/EXPERIMENT_TRACKER.md'
tracker_text = tracker.read_text(encoding='utf-8')
assert '§41.857' not in tracker_text
tracker.write_bytes((tracker_text + '\n\n## §41.857 终态更新（覆盖上文准备状态）\n\n' + '\n'.join(metric_table) + '\n\n' + claim_boundary + '\n\n三个正式端、各自M0/首次strict及原唯一九配对报告已完成。没有新增后继训练。\n').encode('utf-8'))
result = dict(status='LOCAL_ACCEPTED_CAPACITY_CLOSEOUT_PREPARED_NOT_PUBLISHED', at=at,
              completed=3, formal_epochs=150, formal_steps=6484, gate_count=gate_count,
              doc_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(), protected_files=protected,
              local_preparation_provenance='First attempt stopped before writes: assumed7 foreign files, actual13. Second attempt hit Windows path length while copying a milestone subtree; original doc/proof remained unchanged, partial owned archive moved intact to private capacity_closeout_failed_windows_path857. Short step aliases now preserve complete packets without a compatibility layer. All13 foreign files protected; no NN/report replay or metric change.')
(archive / 'LOCAL_CLOSEOUT.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
owned = [doc_relative, 'refine-logs/CURRENT_GOAL.md', 'refine-logs/semantic_capacity_control_v1/EXPERIMENT_TRACKER.md',
         source_note.relative_to(repo).as_posix()]
owned.extend(p.relative_to(repo).as_posix() for p in archive.rglob('*') if p.is_file())
manifest = repo / 'MANIFEST.md'
with manifest.open('a', encoding='utf-8', newline='\n') as stream:
    stream.write(f'\n\n## 2026-10-05 §41.857 原capacity终态与来源核查\n\n')
    stream.write('| Output | Provenance |\n|---|---|\n')
    for name in owned:
        stream.write(f'| `{name}` | 原三端/一次报告闭合后的文本与图表归档；无模型重跑 |\n')
owned.append('MANIFEST.md')
assert all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest for name, digest in protected.items())
publication = dict(previous_head=previous_head, section='41.857', files=sorted(owned),
                   protected_files=protected, immutable_archive_roots=[archive_relative],
                   remote_sparse_roots=[archive_relative, 'refine-logs/region_evidence_reconstruction_v1'],
                   doc_sha256=result['doc_sha256'])
publication_file.write_bytes((json.dumps(publication, indent=2) + '\n').encode())
print(json.dumps(dict(status=result['status'], owned_files=len(owned), gate_count=gate_count,
                     doc_sha256=result['doc_sha256'], published=False)))
