"""Summarize saved diagnostic text; no tensor loading or ranking replay."""
import hashlib
import json
from pathlib import Path

w = Path(__file__).resolve().parent
r = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
observation = json.loads((w / 'TERMINAL_OBSERVATION.json').read_text())
assert observation['controller_exit']['exit_code'] == 0
assert not observation['supervisor_live'] and not observation['worker_live']
assert observation['campaign']['status'] == 'COMPLETE_TWO_FIXED_BEST_DIAGNOSES'
for name, row in observation['files'].items():
    assert sha(w / 'results' / name) == row['sha256']
rows = {n: json.loads((w / 'results' / n / 'DIAGNOSIS.json').read_text()) for n in ('gradient', 'random')}
keys = ('mAP', 'Rank-1', 'Rank-5', 'Rank-10')
summary = {'schema': 'gcf930-saved-text-summary-v1', 'source_sha256': sha(Path(__file__)),
           'input_sha256': {name: row['sha256'] for name, row in observation['files'].items()},
           'status': 'TWO_FIXED_BEST_TEXT_SUMMARIZED', 'dataset': 'RGBNT201', 'seed': 42,
           'optimizer_updates': 0, 'new_training_started': False, 'ranking_replays': 0,
           'goal_status': 'ACTIVE_UNMET', 'arms': {}}
lines = ['# 固定best证据净用途：global / correction / fused', '',
         '原两端各50轮、2649更新；本次只诊断各自同一E8 mAP-best，未训练、改选或生成新权重。', '',
         '| 条件 / 表示 | mAP | R1 | R5 | R10 |', '|---|---:|---:|---:|---:|']
for name, d in rows.items():
    assert d['status'] == 'COMPLETE' and d['optimizer_updates'] == 0
    assert d['model_state_before_sha256'] == d['model_state_after_sha256']
    assert d['fused_distance_max_abs_difference'] == 0
    for feature in ('global', 'correction', 'fused'):
        lines.append('| ' + name + ' / ' + feature + ' | ' + ' | '.join(f'{d["scores"][feature]["metrics"][k]:.6f}' for k in keys) + ' |')
    pair = d['comparisons']['same_model_global_to_fused']
    changes = pair['query_changes']
    assert len(changes) == 836
    groups = {'repair': [], 'new_error': [], 'kept_correct': [], 'kept_wrong': []}
    delta_ap = []
    for q in changes:
        a, b = q['control_first_rank'] == 1, q['candidate_first_rank'] == 1
        kind = 'kept_correct' if a and b else 'kept_wrong' if not a and not b else 'repair' if b else 'new_error'
        value = 100 * (q['candidate_ap'] - q['control_ap'])
        delta_ap.append(value)
        groups[kind].append(value)
    grouped = {k: {'queries': len(v), 'contribution_to_total_map_points': sum(v) / len(changes)} for k, v in groups.items()}
    assert abs(sum(delta_ap) / len(changes) - pair['delta_metrics']['mAP']) < 1e-10
    independent = d['comparisons']['independent_global_to_same_model_global']
    assert all(v == 0 for v in independent['delta_metrics'].values())
    assert all(q['control_ap'] == q['candidate_ap'] and q['control_first_rank'] == q['candidate_first_rank'] for q in independent['query_changes'])
    summary['arms'][name] = {
        'metrics': {k: v['metrics'] for k, v in d['scores'].items()},
        'same_model_global_to_fused': {k: v for k, v in pair.items() if k not in ('query_changes', 'identity_changes')},
        'rank1_group_ap_contributions': grouped,
        'query_ap_improved': sum(v > 0 for v in delta_ap), 'query_ap_worsened': sum(v < 0 for v in delta_ap),
        'same_model_global_matches_independent_saved_ap_and_first_rank': True,
        'geometry': d['geometry']['query'], 'readout_gain': d['readout_gain'],
        'query_gallery_extraction_seconds': d['query_gallery_extraction_seconds'],
        'diagnosis_process_seconds': next(j['elapsed_seconds'] for j in observation['campaign']['jobs'] if j['exposure'] == name)}
lines += ['', '## 对自己global的实际增量', '',
          '| 条件 | ΔmAP | ΔR1 | 首位修复 / 新增错误 | query AP提高 / 下降 | 身份宏AP变化 |',
          '|---|---:|---:|---:|---:|---:|']
for name, d in summary['arms'].items():
    p = d['same_model_global_to_fused']
    lines.append(f"| {name} | {p['delta_metrics']['mAP']:+.6f} | {p['delta_metrics']['Rank-1']:+.6f} | {p['rank1_repairs']} / {p['rank1_new_errors']} | {d['query_ap_improved']} / {d['query_ap_worsened']} | {p['identity_macro_delta_ap_points']:+.6f} |")
lines += ['', '| 条件 | 首位组 | query数 | 对总mAP变化贡献 |', '|---|---|---:|---:|']
for name, d in summary['arms'].items():
    for group, values in d['rank1_group_ap_contributions'].items():
        lines.append(f"| {name} | {group} | {values['queries']} | {values['contribution_to_total_map_points']:+.6f} |")
delta = {k: rows['gradient']['scores']['fused']['metrics'][k] - rows['random']['scores']['fused']['metrics'][k] for k in keys}
summary['gradient_minus_random_metrics'] = delta
summary['primary_mechanism_advance'] = False
lines += ['', '## 可支持的判断与边界', '',
          '两端自身global的四项分数、836个query的AP与first-rank均与已保存独立global一致。本次测量支持“global检索能力保留”的窄结论；没有测量global特征逐位相等、全部排名相等或global距离逐位相等。', '',
          f"主配对gradient−random仍为{delta['mAP']:+.6f} mAP、{delta['Rank-1']:+.6f} R1，原机制不晋级。本次补测不改变原first-strict指标、选轮或失败判断。两端修正都对自己global提供很薄的正增量；梯度排除没有超过同数量随机排除。它未证明global敏感的token排除后就能学到有用的非冗余身份证据。", '',
          'correction单独mAP只是固定best描述，不能当成融合失败的唯一原因；有效残差也不一定需要高独立分数。修正幅度、角度、独立判别性与最终互补用途必须分别记账，不据本次分数搜索gain/层号/比例/margin/seed。', '',
          '本轮没有测量实际被读取区域的语义、局部成对比较或新比较器；不能将结果解释为“所有失败都是单向量压缩”“背景/错配已确认”或任何因果结论。旧五分数队列不重放，原源码469/M0来源15均不变；新增诊断472 source pin含一个新工具、controller及登记计划。', '',
          '每个query/gallery batch一次原模型forward，同时保存g/c/h/f；推理不mask、不求VJP、不用身份标签。参数buffer摘要前后相同、grad均None、0优化器更新；两端F距离与原数组max abs差均0，四项指标在预设1e−5内。新特征与距离留26远端，没有新权重。']
for name, d in summary['arms'].items():
    geom = d['geometry']
    lines += ['', f"{name}：修正/global平均范数比{geom['scaled_correction_global_norm_ratio']['mean']:.6f}、平均夹角{geom['global_fused_angle_degrees']['mean']:.6f}°；提取{d['query_gallery_extraction_seconds']:.6f}s，真实诊断子进程{d['diagnosis_process_seconds']:.6f}s。提取时间含dataloading/encoding/CPU传输，不含构造/文件校验/评分；共享服务器单次观察，不是隔离速度基准。"]
lines += ['', f"原新诊断supervisor{observation['supervisor_pid']}/worker{observation['worker_pid']}在{observation['controller_exit']['completed_at']} EXIT0；{observation['at']}一次延迟终态观察二者均不存在，两个child各EXIT0。只26物理GPU0/1，未查询/设置功率温度，不用25/2/3，不抢占。", '',
          '源码与文本证据审计同家族provisional，模型/effort实际未attest；审查者没有远端访问或重读tensor。实际输入/hash/state/评分门由原一次执行记录支持，具体文本审查范围见审计附件。', '',
          '当前训练及补充固定best诊断均结束，停止启动新训练。全三集超过完整Signal仍未实现，主Goal ACTIVE_UNMET；没有因为有限诊断结束而标记complete/paused/blocked。只保留best和必要来源/证据，低分对照并非自动无用。', '',
          '论文主线：不再追求每个分支都更强，而是检验每条新增证据在最终身份比较中是否有净用途。外领域方法只回答其中一环，不能用它的名称替代项目自身的机制证据。']
base = 'refine-logs/global_conditioned_evidence_best930/'
copies = {name: w / 'results' / name for name in observation['files']}
copies.update({'TERMINAL_OBSERVATION.json': w / 'TERMINAL_OBSERVATION.json',
               'COLLECT_EXIT.json': w / 'COLLECT_EXIT.json', 'SOURCE_AUDIT.json': w / 'review/SOURCE_AUDIT.json',
               'SOURCE_AUDIT.md': w / 'review/SOURCE_AUDIT.md', 'controller.py': w / 'controller.py',
               'deploy_and_launch.py': w / 'deploy_and_launch.py', 'collect_terminal.py': w / 'collect_terminal.py',
               'prepare_publication.py': Path(__file__)})
for name, p in copies.items():
    target = r / base / name
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(p.read_bytes())
(r / base / 'SUMMARY.json').write_bytes((json.dumps(summary, indent=2)+'\n').encode())
(r / base / 'RESULT_REPORT.md').write_bytes(('\n'.join(lines)+'\n').encode())
(r / base / 'EXPERIMENT_TRACKER.md').write_bytes(('# 930固定best诊断终态\n\n| 项目 | 状态 | 边界 |\n|---|---|---|\n| 两个固定E8 best分解 | DONE | 各一次，无训练/优化器/改选 |\n| 原机制晋级 | FAIL | gradient−random mAP/R1均负 |\n| 新训练 | STOPPED | 原两端及本次有限诊断均已终止 |\n| 区域内容/新比较器 | NOT_RUN | 无新增测量 |\n| 主目标 | ACTIVE_UNMET | 三集超过完整Signal未完成 |\n').encode())
names = [base + n for n in copies] + [base + n for n in ('SUMMARY.json','RESULT_REPORT.md','EXPERIMENT_TRACKER.md','EXPERIMENT_PLAN.md')]
names.append('tools/diagnose_global_conditioned_evidence_best.py')
(w / 'PREPARED_PUBLICATION.json').write_text(json.dumps({'file_sha256': {n: sha(r/n) for n in names},
          'summary_source_sha256': sha(Path(__file__)), 'neural_calls': 0, 'ranking_replays': 0}, indent=2)+'\n')
print(json.dumps({'prepared_files': len(names), 'new_training_started': False, 'ranking_replays': 0,
                  'gradient_minus_random_metrics': delta, 'same_model_global_measured': True}))
