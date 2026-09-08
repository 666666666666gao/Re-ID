from pathlib import Path
from collections import Counter
import csv,hashlib,json

t=Path('C:/Users/gb/.codex_tmp');out=t/'role_set_q1_executor_report_20260908';assert not out.exists();out.mkdir()
source=t/'role_set_q1_source_analysis_20260908.json';ranking=t/'role_set_q1_rankings_20260908/ranking_replay.json'
s=json.loads(source.read_bytes());r=json.loads(ranking.read_bytes());root=t/'role_set_q1_complete_20260908'
pipe=json.loads((root/'pipeline.json').read_bytes());intake=json.loads((root/'intake_complete.json').read_bytes());cpu=json.loads((root/'q1_cpu.json').read_bytes())
assert r['status']=='PASS_COMPLETE_TERMINAL_RANKING_TEXT' and s['status']=='COMPLETE_DESCRIPTIVE_SOURCE_LOG_ANALYSIS'
assert r['scientific_status']=='Q1_FAIL' and len(s['endpoints'])==6 and cpu['checked_training_steps']==1560
agg={}
for end in ('control','role_set'):
    rows=[x for x in s['endpoints'] if x['endpoint']==end];assert len(rows)==3
    agg[end]={}
    for phase in ('post_warmup_steps_66_260','last_five_epochs_steps_196_260'):
        ps=[x['phases'][phase] for x in rows];counts=Counter();identities=Counter();roles={e:Counter() for e in ('cnn','transformer','mamba')}
        for x in ps:
            counts.update(x['counts']);identities.update(x['unique_negative_identities_histogram'])
            for e in roles:roles[e].update(x['role_proposal_exposures'][e])
        assert len({x['counts']['steps'] for x in ps})==1
        agg[end][phase]=dict(counts=dict(counts),negative_identity_histogram=dict(identities),roles={e:dict(c) for e,c in roles.items()},mean_losses={k:sum(x['mean_losses'][k] for x in ps)/3 for k in ps[0]['mean_losses']})
    agg[end]['cost']={k:sum(x['cost'][k] for x in rows) for k in rows[0]['cost'] if k!='peak_allocated_mib'}
    agg[end]['cost']['max_peak_allocated_mib']=max(x['cost']['peak_allocated_mib'] for x in rows)
identity_path=t/'role_set_q1_rankings_20260908/all300_identity_output_changes.csv'
ids=[x for x in csv.DictReader(identity_path.open(encoding='utf-8')) if x['output']=='fused'];assert len(ids)==60
delta=[float(x['delta_mAP_pp']) for x in ids]
identity_summary=dict(improved=sum(x>0 for x in delta),declined=sum(x<0 for x in delta),unchanged=sum(x==0 for x in delta),queries=sum(int(x['query_count']) for x in ids))
assert identity_summary['queries']==600
proof=dict(source_analysis_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),ranking_replay_sha256=hashlib.sha256(ranking.read_bytes()).hexdigest(),identity_csv_sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest(),aggregation=agg,fused_identity_summary=identity_summary,scope='Executor descriptive text aggregation only; no new model forwards, updates or independent semantic judgment.')
(out/'source_aggregate.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
metrics=r['endpoints'];counts=agg['role_set']['post_warmup_steps_66_260']['counts'];changes=r['paired_changes']['fused']
metric_table='\n'.join(f"| {e} | {metrics['control']['metrics'][e]['mAP']:.6f} | {metrics['role_set']['metrics'][e]['mAP']:.6f} | {r['paired_gains'][e]:+.6f} | {metrics['control']['metrics'][e]['Rank-1']:.6f} | {metrics['role_set']['metrics'][e]['Rank-1']:.6f} |" for e in ('baseline_only','fused','cnn','transformer','mamba'))
costs=[agg[e]['cost'] for e in ('control','role_set')]
loss_rows=[]
for phase,label in (('post_warmup_steps_66_260','预热后'),('last_five_epochs_steps_196_260','最后5epoch')):
    for key in ('actual_total','batch_hard','expanded_hard','role_set'):
        a=agg['control'][phase]['mean_losses'][key];b=agg['role_set'][phase]['mean_losses'][key]
        loss_rows.append(f'| {label} | {key} | {a:.9f} | {b:.9f} | {b-a:+.9f} |')
text=f'''# MSVR310 role-set v1：完整Q1终态与执行侧分析

**状态：Q1_FAIL；执行侧全量文本/排名复算通过；独立终态审计进行中。** 本报告的描述性结论不是独立审计意见，后续需附其实际报告与限制。

执行代码26c97390704c629237687d263b4381f5584cbe97，固定seed42，三折双端。Q1于{pipe['stages'][3]['ended_at']}结束，CPU于{pipe['stages'][4]['ended_at']}结束；所有阶段退出0，终态接收要求全部原PID已不存在。接收{intake['files']}份文本/{intake['bytes']}B，全部大小与SHA核对。summary SHA为{cpu['summary_sha256']}。

两端均使用新鲜历史坐标和完整历史候选导数，当前64个anchor、512容量/8步年龄/65步预热、原V8三角色与固定融合不变。唯一主要干预是fused与三个完整角色最近负例的位置去重并集，采用既有欧氏hinge均值；最难正例不变，其余13项损失不变。控制端也计算四空间提议，但没有使用候选集合标量更新。集合增加覆盖并同时改变最难关系权重，不能只归因为角色多样性。

## 全部输出与原条件

以下为训练内部完整路径身份隔离、完整图库的Q1：600合法query/60身份，非官方结果。

| 输出 | control mAP | role_set mAP | 配对差值pp | control R1 | role_set R1 |
|---|---:|---:|---:|---:|---:|
{metric_table}

配对fused三折差值为{', '.join(f'{x:+.6f}' for x in r['paired_fold_gains'])}pp，身份bootstrap下界{r['paired_lower_bound']:+.6f}pp。配对五条件0/5；候选相对Signal五条件0/5。候选fused高于三个角色，但低于Signal{-metrics['role_set']['gains_over_signal']['fused']:.6f}pp，因此包含Signal的严格最高条件仍失败。原门槛和FAIL保持封存。

全600query中，fused AP改善{changes['ap_improved']}、下降{changes['ap_declined']}、不变{changes['ap_unchanged']}；Rank-1修复{changes['rank1_repaired']}、新增错误{changes['rank1_new_errors']}。60身份均值AP改善{identity_summary['improved']}、下降{identity_summary['declined']}、不变{identity_summary['unchanged']}。完整3000条query-output和300条identity-output表保留，未选择个别身份替代全量结果。身份bootstrap不代表多训练种子，也不消除反复开发选择偏差。

## 来源关系确有扩展，但不能由总loss下降推出泛化

候选端预热后第66–260步、三折共585更新/{counts['anchor_exposures']}次anchor曝光，实际选择{counts['selected_position_exposures']}个负位置曝光，平均{counts['selected_position_exposures']/counts['anchor_exposures']:.6f}/anchor。相对fused最近负例多{counts['extra_position_exposures']}个位置，其中{counts['extra_historical_positions']}为历史候选。{counts['anchors_with_extra_negative_identity']}次anchor曝光引入额外负身份（{counts['anchors_with_extra_negative_identity']/counts['anchor_exposures']*100:.4f}%）；去重后的额外负身份曝光{counts['extra_negative_identity_exposures']}。这些是训练曝光，并非独立新图片或新增身份总数。

额外active hinge曝光{counts['extra_active_hinge_exposures']}，涉及{counts['anchors_with_extra_active_hinge']}次anchor曝光。当前文本只保存每anchor active数量，不逐项标出是哪一角色独有提议被激活，因此不将提议独有计数写成独有有效梯度贡献。预热阶段虽计算提议，候选目标尚未启用；上述用于说明训练干预的计数仅使用预热后。

| 范围 | 标量定义 | control | role_set | 差值 |
|---|---|---:|---:|---:|
{chr(10).join(loss_rows)}

候选actual_total较低，但它采用不同的fused项，不能直接据此声称优化更好。按相同定义复算的expanded_hard及role_set均值，在候选末段略高；因此本轮不支持“已经解决更多来源困难关系，只差检索融合”的结论。也不能仅凭这些聚合数断言某个梯度或某项损失是唯一失败原因。

三折两端每步采样记录和三模态增强像素SHA一致；原65步预热中总loss均从step2开始存在细微数值差异，最大差分别{', '.join(f"{x['warmup']['max_abs_total_loss_difference']:.9f}" for x in s['paired_warmup'])}。候选目标从step66启用，而首个历史候选为step67，故“历史生效前”不等于“干预前”。不得将微小配对增益单独解释成严格隔离的因果效应。

## 完整核验与成本边界

原CPU复算1560更新、{cpu['checked_memory_distance_elements']}个保存的训练距离元素、{cpu['checked_retrieval_distance_and_rank_elements']}个检索距离/排名元素，核对{cpu['checked_history_vjp_record_forwards']}次历史VJP record-forward计数。执行侧文本工具另逐条复算全部2069520个排序位置、scene排除、AP/CMC和两组门槛；不重跑模型或重新计算距离。逐步参数梯度仍是运行见证，不能把这些算术核验写成完整重放训练反传。

| 成本（三折合计；峰值取最大） | control | role_set |
|---|---:|---:|
| 新鲜角色record-forward | {costs[0]['fresh_role_record_forwards']} | {costs[1]['fresh_role_record_forwards']} |
| 历史VJP record-forward | {costs[0]['historical_vjp_record_forwards']} | {costs[1]['historical_vjp_record_forwards']} |
| fit epoch秒数之和 | {costs[0]['fit_epoch_seconds']:.3f} | {costs[1]['fit_epoch_seconds']:.3f} |
| 最大分配显存MiB | {costs[0]['max_peak_allocated_mib']:.3f} | {costs[1]['max_peak_allocated_mib']:.3f} |

候选历史VJP record-forward增加{(costs[1]['historical_vjp_record_forwards']/costs[0]['historical_vjp_record_forwards']-1)*100:.4f}%，fit时间增加{(costs[1]['fit_epoch_seconds']/costs[0]['fit_epoch_seconds']-1)*100:.4f}%。fit时间不含全部构建/检索/CPU开销；Q1实际阶段总耗时{pipe['stages'][3]['elapsed_seconds']:.3f}秒。没有新增推理参数，不等于训练无额外成本。

当前保留：角色确能提出不同身份关系、完整当前坐标和两侧导数的实现能力。当前不成立：该固定集合均值目标稳定提高未知身份检索或超越Signal。后继实验尚未登记；先完成独立审计，再据完整来源证据确定单一新假设，不扫权重/候选数/终点挽救本版。三个核心数据集的完整Goal仍未达到，正式测试表没有新增结果。
'''
(out/'MSVR310_ROLE_SET_V1_Q1_2026-09-08.md').write_text(text,encoding='utf-8')
print(json.dumps(dict(output=str(out),scientific_status=r['scientific_status'],paired_gain=r['paired_gains']['fused'],independent_audit='PENDING')))
