from pathlib import Path
import csv,json,hashlib,math
from collections import Counter

temp=Path('D:/Program Files/UserCache/gb/codex/tmp')
root=temp/'trifusion_cross_scene_q1_complete_20260921'
out=temp/'trifusion_cross_scene_q1_analysis_20260921';out.mkdir(exist_ok=False)
summary=json.loads((root/'q1/summary.json').read_bytes());comp=summary['comparison']
source=json.loads((temp/'trifusion_cross_scene_q1_source_log_analysis_20260921.json').read_bytes())
cpu=json.loads((root/'q1_cpu.json').read_bytes())
assert cpu['checked_training_steps']==1560 and summary['status']=='Q1_FAIL'
assert cpu['summary_sha256']==hashlib.sha256((root/'q1/summary.json').read_bytes()).hexdigest()
outputs=('baseline_only','fused','cnn','transformer','mamba')
query_rows=[];fold_rows=[];epochs=[];step_rows=[];zero_rows=[]
for fold in range(3):
    receipts={e:json.loads((root/'q1'/f'fold_{fold}_{e}'/'receipt.json').read_bytes()) for e in ('control','cross_scene')}
    ranks={e:json.loads((root/'q1'/f'fold_{fold}_{e}'/'rankings.json').read_bytes()) for e in receipts}
    a,b=(receipts[e]['retrieval'] for e in ('control','cross_scene'))
    assert a['query_rows']==b['query_rows'] and a['gallery_manifest']==b['gallery_manifest']
    for name in outputs:
        for e,x in (('control',a),('cross_scene',b)):
            fold_rows.append(dict(fold=fold,endpoint=e,output=name,queries=len(x['query_rows']),gallery=len(x['gallery_manifest']),**x['outputs'][name]['metrics']))
        for i,q in enumerate(a['query_rows']):
            ca=a['outputs'][name]['average_precision'][i];cb=b['outputs'][name]['average_precision'][i]
            ra=a['outputs'][name]['first_match_rank'][i];rb=b['outputs'][name]['first_match_rank'][i]
            row=dict(fold=fold,output=name,query_record_index=q['record_index'],identity=q['identity'],scene=q['scene'],control_ap=ca,candidate_ap=cb,gain_ap_pp=100*(cb-ca),control_first_match_rank=ra,candidate_first_match_rank=rb,rank1_repaired=ra!=1 and rb==1,rank1_new_error=ra==1 and rb!=1)
            if row['rank1_new_error']:
                legal=[p for p in ranks['cross_scene'][name][q['gallery_position']] if not (b['gallery_manifest'][p]['identity']==q['identity'] and b['gallery_manifest'][p]['scene']==q['scene'])]
                first=b['gallery_manifest'][legal[0]]
                assert first['identity']!=q['identity']
                row.update(new_error_top_identity=first['identity'],new_error_top_scene=first['scene'],new_error_top_same_scene=first['scene']==q['scene'])
            query_rows.append(row)
    for e in receipts:
        tr=receipts[e]['training']
        for row in tr['history']:epochs.append(dict(fold=fold,endpoint=e,**row))
        for row in tr['steps']:
            step_rows.append(dict(fold=fold,endpoint=e,step=row['step'],active_fused_metric=row['active_fused_metric'],total_loss=row['loss'],**row['components']))
        memory=list(map(json.loads,(root/'q1'/f'fold_{fold}_{e}'/'memory_steps.jsonl').read_text().splitlines()))
        for row,step in zip(memory,tr['steps'],strict=True):
            if e=='cross_scene' and step['active_fused_metric']=='cross_scene_smooth_ap' and row['relation_objective']['cross_scene_eligible_anchors']==0:
                assert step['components']['triplet_fused']==0 and all(x==0 for x in row['historical_leaf_upstream_norms']) and not row['history_vjp_groups']
                zero_rows.append(dict(fold=fold,step=step['step'],rank_loss=step['components']['triplet_fused'],history_records=len(row['memory']),history_upstream_all_zero=True,vjp_groups=0))
assert len(query_rows)==3000 and len(epochs)==120 and len(step_rows)==1560 and len(zero_rows)==4
def save_csv(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
save_csv('all_paired_queries.csv',query_rows)
save_csv('all_fold_output_metrics.csv',fold_rows)
save_csv('all_training_epochs.csv',epochs)
save_csv('all_training_step_components.csv',step_rows)
identity_rows=[dict(identity=x['identity'],query_count=x['query_count'],**x['gains_mAP']) for x in comp['paired_per_identity']]
save_csv('all_paired_identities.csv',identity_rows)
changes={}
for name in outputs:
    rows=[r for r in query_rows if r['output']==name]
    assert len(rows)==600
    gain=sum(r['gain_ap_pp'] for r in rows)/600
    assert abs(gain-comp['matched_gains_mAP'][name])<1e-10
    ids=[r[name] for r in identity_rows]
    changes[name]=dict(query_count=600,ap_improved=sum(r['gain_ap_pp']>0 for r in rows),ap_declined=sum(r['gain_ap_pp']<0 for r in rows),ap_unchanged=sum(r['gain_ap_pp']==0 for r in rows),rank1_repaired=sum(r['rank1_repaired'] for r in rows),rank1_new_errors=sum(r['rank1_new_error'] for r in rows),new_errors_same_scene=sum(r.get('new_error_top_same_scene',False) for r in rows),identities_improved=sum(x>0 for x in ids),identities_declined=sum(x<0 for x in ids),identities_unchanged=sum(x==0 for x in ids))
    a=comp['endpoints']['control']['metrics'][name]['Rank-1'];b=comp['endpoints']['cross_scene']['metrics'][name]['Rank-1']
    assert abs((changes[name]['rank1_repaired']-changes[name]['rank1_new_errors'])/6-(b-a))<1e-10
pooled={}
for e in ('control','cross_scene'):
    selected=[x for x in source['endpoints'] if x['endpoint']==e]
    pooled[e]=dict(last65_common_mean_losses={k:sum(x['phases']['last_five_epochs_steps_196_260']['mean_losses'][k] for x in selected)/3 for k in selected[0]['phases']['last_five_epochs_steps_196_260']['mean_losses']},fit_epoch_seconds=sum(x['cost']['fit_epoch_seconds'] for x in selected),fresh_role_record_forwards_including_zero_check=sum(x['cost']['fresh_role_record_forwards_including_zero_check'] for x in selected),history_vjp_record_forwards=sum(x['cost']['historical_vjp_record_forwards'] for x in selected),peak_allocated_mib=max(x['cost']['peak_allocated_mib'] for x in selected))
result=dict(status='COMPLETE_EXECUTOR_TERMINAL_ANALYSIS_AUDIT_PENDING',scientific_status=summary['status'],summary_sha256=cpu['summary_sha256'],source_analysis_sha256=hashlib.sha256((temp/'trifusion_cross_scene_q1_source_log_analysis_20260921.json').read_bytes()).hexdigest(),paired_query_changes=changes,pooled_source=pooled,actual_postwarmup_zero_support=zero_rows,paired_checks=comp['paired_checks'],signal_checks=comp['endpoints']['cross_scene']['scientific_checks'],paired_gains=comp['matched_gains_mAP'],fold_gains=comp['fold_fused_gains_mAP'],paired_bootstrap_lower_pp=comp['paired_bootstrap_lower_pp'],warmup=source['paired_warmup'],model_forwards=0,optimizer_updates=0)
(out/'terminal_analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
lines=['# MSVR310 跨场景 Smooth-AP 完整 Q1（2026-09-21）','', '**状态：原合同 Q1_FAIL；完整 CPU 核验通过；独立审查进行中。**','',
'执行 d35864d6411591e05c8ac3e5164ebae48063ad99；配置 SHA-256 e5326b52ebb12dced24ebfac788db2e2bb5bdca0b50c6c4dec64e596f1af05a1。仅 seed42。三折两端各20epoch/260更新，共1560更新。此为训练内部身份隔离、完整图库比较，不是官方测试。', '',
'唯一干预同时改变同身份跨scene正例集合和eligible-anchor均值分母；不能把效果单独归因于删除容易正例。两端均新鲜历史坐标、完整历史VJP、64当前anchor和0历史anchor，其他13项保持。', '',
'## 完整指标', '', '| 输出 | 控制 mAP | 候选 mAP | 配对增益pp | 控制R1 | 候选R1 |','|---|---:|---:|---:|---:|---:|']
for name in outputs:
    a=comp['endpoints']['control']['metrics'][name];b=comp['endpoints']['cross_scene']['metrics'][name]
    lines.append(f'| {name} | {a["mAP"]:.6f} | {b["mAP"]:.6f} | {comp["matched_gains_mAP"][name]:+.6f} | {a["Rank-1"]:.6f} | {b["Rank-1"]:.6f} |')
lines+=['',f'配对五项通过{sum(comp["paired_checks"].values())}/5；对Signal五项通过{sum(comp["endpoints"]["cross_scene"]["scientific_checks"].values())}/5。配对三折fused增益：'+', '.join(f'{x:+.6f}' for x in comp['fold_fused_gains_mAP'])+f'；身份bootstrap下界{comp["paired_bootstrap_lower_pp"]:+.6f}pp。', '',
'候选fused已高于本机Signal及三个角色，但增益不足1pp、CNN与Transformer配对下降、配对bootstrap下界非正；对Signal的各折与角色条件也未满足。保留真实正均值，同时保持未晋级，不改门槛或选择其他checkpoint。', '',
'## 全部查询与身份变化', '', '| 输出 | AP改善/下降/不变query | R1修复/新增错误 | 新错误同scene | 身份改善/下降/不变 |','|---|---|---|---:|---|']
for name,x in changes.items():lines.append(f'| {name} | {x["ap_improved"]}/{x["ap_declined"]}/{x["ap_unchanged"]} | {x["rank1_repaired"]}/{x["rank1_new_errors"]} | {x["new_errors_same_scene"]} | {x["identities_improved"]}/{x["identities_declined"]}/{x["identities_unchanged"]} |')
lines+=['', '配对变化均以本次control为参照；不同于summary中各端相对Signal的query_changes。全部600query×5输出和60身份见CSV，无删除不利行，不读取图片推断失败成因。', '',
'## 来源末段与监督支持', '', '| 同定义诊断，最后65步三折等步数均值 | control | cross_scene |','|---|---:|---:|']
for k in ('batch_hard','expanded_hard','smooth_ap','cross_scene_ap'):
    lines.append(f'| {k} | {pooled["control"]["last65_common_mean_losses"][k]:.9f} | {pooled["cross_scene"]["last65_common_mean_losses"][k]:.9f} |')
lines+=['', '跨scene和标准AP同定义诊断均下降，但批内与扩展hard间隔损失略升。实际total中的活动目标定义不同，不用total绝对值判断哪端优化更好；这些批次AP诊断也不是完整来源图库或官方mAP。', '',
'预热后每端37440次anchor曝光中，合法跨scene曝光15376次（约41.0684%）；各折5144/5176/5056。候选实际零支持发生fold0 step180、fold1 step221、fold2 step133/232，排名项和历史上游为零且VJP组跳过。其余监督/更新按原合同继续。完整训练账本与实际记录已检查，未重建全程模型梯度。', '',
'## 数值、成本与证据范围', '',
'两端记录与输入像素哈希逐步相同，但共同hard预热已有浮点差异：各折total最大绝对差见terminal_analysis.json，约0.001278/0.001623/0.002021；不能声称严格逐位训练轨迹相同。第66步已切换不同AP目标，before_first_history窗口包含它，不能把其差异全部归为预热噪声。', '',
'all_training_epochs.csv保留全部120个训练epoch；all_training_step_components.csv保留全部1560步的14项标量。原合同只检索固定epoch20，没有补造每epoch检索成绩。', '',
f'控制/候选三折fit epoch耗时{pooled["control"]["fit_epoch_seconds"]:.3f}/{pooled["cross_scene"]["fit_epoch_seconds"]:.3f}秒；冻结字段后额外fresh角色record前向各{pooled["control"]["fresh_role_record_forwards_including_zero_check"]}/{pooled["cross_scene"]["fresh_role_record_forwards_including_zero_check"]}次（含各端零更新一致性检查）；历史VJP record前向{pooled["control"]["history_vjp_record_forwards"]}/{pooled["cross_scene"]["history_vjp_record_forwards"]}次。它们是重复计算记录，不是独立图片。最大allocated约{max(x["peak_allocated_mib"] for x in pooled.values()):.3f}MiB。', '',
'CPU回执覆盖1560训练步、116501504距离元素、583168历史VJP record前向计数及2069520检索距离/排名元素；模型前向和更新均0。梯度记录仍为原运行见证，不能把CPU算术核验声称为全训练独立重演。此日志中的current/history梯度比较也不是排名F与其余监督O的比较。', '',
'summary.project_commit记录Q1开始时工作树HEAD fdc6305，wrapper.code_commit绑定执行d35864d；两者字段含义不同，应由文件哈希与独立审查核对，不把文档提交误当算法变化。', '',
'## 后续边界', '',
'本配置封存Q1_FAIL，等待独立审查关闭；不进入官方测试、不扫描温度/分母/终点，不将本次增益与旧版本相加。跨场景目标具有平均正证据，但还未证明三角色均获益或身份稳定性。后继支持感知优化仍为待登记候选，须结合完整证据和已核近邻再确定唯一干预。三数据集总Goal ACTIVE/UNMET。', '']
(out/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
manifest={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.iterdir() if p.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(out=str(out),paired_query_changes=changes,pooled_source=pooled,zero_support=zero_rows),ensure_ascii=False))
