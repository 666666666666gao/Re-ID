"""Summarize verified fixed-state source gradients, never held-out efficacy."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import statistics
import sys


def distribution(values):
    valid = [v for v in values if v is not None]
    assert all(math.isfinite(v) for v in valid)
    if not valid:
        return dict(count=0, undefined=len(values), zeros=0, mean=None,
                    median=None, p10=None, p90=None, minimum=None, maximum=None)
    ordered = sorted(valid)
    def quantile(q):
        index = (len(ordered)-1)*q
        low = math.floor(index); high = math.ceil(index)
        return ordered[low] + (ordered[high]-ordered[low])*(index-low)
    return dict(count=len(valid), undefined=len(values)-len(valid),
                zeros=sum(v == 0 for v in valid), mean=statistics.mean(valid),
                median=quantile(.5), p10=quantile(.1), p90=quantile(.9),
                minimum=ordered[0], maximum=ordered[-1])


def run(root, output):
    summary_path = root/'summary.json'
    summary = json.loads(summary_path.read_bytes())
    verified = json.loads((root/'source_verification.json').read_bytes())
    assert summary['status'] == 'COMPLETE_SOURCE_OBJECTIVE_GRADIENTS'
    assert verified['status'] == 'PASS_COMPLETE_SOURCE_OBJECTIVE_GRADIENT_LEDGER'
    assert verified['steps'] == 1560 and verified['role_rows'] == 4680
    assert verified['summary_sha256'] == hashlib.sha256(summary_path.read_bytes()).hexdigest()
    assert not output.exists()
    ends = ('control', 'smooth_ap'); roles = ('cnn', 'transformer', 'mamba')
    assert [c['directory'] for c in summary['conditions']] == [f'fold_{f}_{e}' for f in range(3) for e in ends]
    raw, costs, groups, source_rows = [], [], [], {}
    for index, condition in enumerate(summary['conditions']):
        rows_path = root/condition['directory']/'steps.jsonl'
        assert hashlib.sha256(rows_path.read_bytes()).hexdigest() == condition['files']['steps.jsonl']['sha256']
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
        assert len(rows) == 260
        fold, end = index//2, ends[index%2]
        source_rows[(fold, end)] = rows
        count = Counter(r['active_fused_metric'] for r in rows)
        assert count == (Counter(control=260) if end == 'control' else Counter(control=65, smooth_ap=195))
        measured_extra = sum(r['refresh_record_forwards'] + r['history_vjp_record_forwards'] + r['direct_check_record_forwards'] for r in rows)+64
        assert measured_extra == condition['extra_record_forwards']
        costs.append(dict(fold=fold, endpoint=end, current_record_forwards=260*64,
                          extra_record_forwards=measured_extra,
                          refresh_record_forwards=sum(r['refresh_record_forwards'] for r in rows),
                          history_vjp_record_forwards=sum(r['history_vjp_record_forwards'] for r in rows),
                          direct_check_record_forwards=sum(r['direct_check_record_forwards'] for r in rows),
                          initial_reencode_record_forwards=64,
                          elapsed_seconds=condition['elapsed_seconds'], peak_allocated_mib=condition['peak_allocated_mib']))
        for row in rows:
            for role in roles:
                metric = row['roles'][role]
                fo, ft = metric['fused_vs_other'], metric['fused_vs_full']
                assert fo['first_norm'] == ft['first_norm']
                raw.append(dict(fold=fold, endpoint=end, step=row['step'], epoch=row['epoch'],
                                role=role, active_fused_metric=row['active_fused_metric'],
                                fused_norm=fo['first_norm'], other_norm=fo['second_norm'], full_norm=ft['second_norm'],
                                fused_other_ratio=fo['first_norm']/fo['second_norm'] if fo['second_norm'] else None,
                                fused_full_ratio=ft['first_norm']/ft['second_norm'] if ft['second_norm'] else None,
                                fused_other_cosine=fo['cosine'], fused_full_cosine=ft['cosine'],
                                current_repeat_difference=metric['current_repeat']['difference_norm'],
                                history_repeat_difference=metric['history_repeat']['difference_norm'],
                                history_repeat_norm=metric['history_repeat']['first_norm'],
                                fused_loss=row['fused_loss'], other_loss=row['other_loss'],
                                current_identity_error=row['current_decomposition']['relative_to_sum_of_component_norms'],
                                full_identity_error=row['full_decomposition']['relative_to_sum_of_component_norms']))
    for fold in range(3):
        for a,b in zip(source_rows[(fold,'control')],source_rows[(fold,'smooth_ap')],strict=True):
            assert a['record_indices'] == b['record_indices']
            assert a['pixel_sha256'] == b['pixel_sha256']
            assert a['memory'] == b['memory']
    metrics = ('fused_norm','other_norm','full_norm','fused_other_ratio','fused_full_ratio',
               'fused_other_cosine','fused_full_cosine','current_repeat_difference',
               'history_repeat_difference','history_repeat_norm','fused_loss','other_loss',
               'current_identity_error','full_identity_error')
    windows = dict(all=(1,260),warmup=(1,65),active=(66,260),last65=(196,260))
    for fold in range(3):
        for end in ends:
            for role in roles:
                for window,(first,last) in windows.items():
                    selected = [r for r in raw if r['fold']==fold and r['endpoint']==end and r['role']==role and first<=r['step']<=last]
                    assert len(selected)==last-first+1
                    group=dict(fold=fold,endpoint=end,role=role,window=window,first_step=first,last_step=last,rows=len(selected))
                    group['statistics']={k:distribution([r[k] for r in selected]) for k in metrics}
                    group['negative_fused_other_cosines']=sum(r['fused_other_cosine'] is not None and r['fused_other_cosine']<0 for r in selected)
                    group['negative_fused_full_cosines']=sum(r['fused_full_cosine'] is not None and r['fused_full_cosine']<0 for r in selected)
                    groups.append(group)
    assert len(raw)==4680 and len(groups)==72 and len(costs)==6
    result=dict(status='COMPLETE_VERIFIED_SOURCE_GRADIENT_ANALYSIS',role_rows=len(raw),groups=len(groups),
                summary_sha256=verified['summary_sha256'],verification_sha256=hashlib.sha256((root/'source_verification.json').read_bytes()).hexdigest(),
                fixed_state=True,optimizer_updates=0,heldout_evaluation=False,window_overlap='all/active/last65 overlap; do not sum windows',
                limitations=['Fixed final models and restored buffers, not the original optimization trajectory.',
                             'Runtime vector norms/dots; full-source parameter vectors were not persisted.',
                             'Different ranking objectives; scalar loss magnitudes are not directly comparable.',
                             'Undefined ratios and cosines retained and counted; no epsilon replacement.',
                             'Gradient direction and magnitude do not establish the cause of held-out generalization.'],
                costs=costs,statistics=groups)
    output.mkdir()
    (output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'all_role_steps.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    with (output/'all_group_statistics.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['fold','endpoint','role','window','metric','count','undefined','zeros','mean','median','p10','p90','minimum','maximum'])
        for group in groups:
            for metric,values in group['statistics'].items():
                writer.writerow([group[k] for k in ('fold','endpoint','role','window')]+[metric]+[values[k] for k in ('count','undefined','zeros','mean','median','p10','p90','minimum','maximum')])
    lines=['# 固定来源目标梯度测量完整汇总','','仅报告固定终态来源诊断，不是新训练或检索成绩。完整范围：1560批、4680角色记录、72组；所有组见JSON/CSV。',
           '', '| fold | endpoint | role | active F/O norm median | active F/O cosine median | negative cosine / 195 | last65 F/O norm median |',
           '|---|---|---|---:|---:|---:|---:|']
    for group in groups:
        if group['window']!='active':continue
        late=next(g for g in groups if all(g[k]==group[k] for k in ('fold','endpoint','role')) and g['window']=='last65')
        stats=group['statistics']
        lines.append('| '+ ' | '.join(str(v) for v in [group['fold'],group['endpoint'],group['role'],stats['fused_other_ratio']['median'],stats['fused_other_cosine']['median'],group['negative_fused_other_cosines'],late['statistics']['fused_other_ratio']['median']])+' |')
    lines+=['','F/O为完整fused目标梯度与其余13项梯度之比；它不是loss权重，也不是泛化贡献百分比。零分母留空并单列，负余弦不自动构成应使用梯度投影的证据。',
            '',f"当前记录前向{sum(c['current_record_forwards'] for c in costs)}，额外角色记录前向{sum(c['extra_record_forwards'] for c in costs)}；后者包含重复梯度见证/直接检查，不能当作正常训练成本。总端点测量秒数{sum(c['elapsed_seconds'] for c in costs)}。",'']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=result['status'],role_rows=len(raw),groups=len(groups),output=str(output))))


if __name__=='__main__':
    run(Path(sys.argv[1]),Path(sys.argv[2]))
