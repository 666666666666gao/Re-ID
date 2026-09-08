from pathlib import Path
from collections import Counter
import hashlib,json

t=Path('C:/Users/gb/.codex_tmp')
src=t/'role_set_source_positive_coverage_20260908.json'
d=json.loads(src.read_bytes())
assert d['status']=='COMPLETE_SAVED_SOURCE_POSITIVE_COVERAGE'
assert d['checked_steps']==1560 and d['checked_distance_elements']==116501504
out=t/'role_set_positive_coverage_report_20260908'
assert not out.exists()
out.mkdir()
agg={}
for end in ('control','role_set'):
    agg[end]={}
    rows=[x for x in d['endpoints'] if x['endpoint']==end]
    assert sorted(x['fold'] for x in rows)==[0,1,2]
    for phase in ('all','warmup','post_warmup','last65'):
        total=Counter()
        for x in rows:total.update(x['phases'][phase])
        agg[end][phase]=dict(total)
        for pre in ('all_positive','cross_scene_positive'):
            assert total[pre+'_affected_positions']==total[pre+'_affected_nonmax_positions']+total[pre+'_affected_max_positions']
for phase in ('all','warmup','post_warmup','last65'):
    a,b=(agg[end][phase] for end in ('control','role_set'))
    for key in ('steps','anchor_exposures','all_positive_positions','cross_scene_positive_positions','all_positive_distinct_record_exposures','cross_scene_positive_distinct_record_exposures','same_record_positive_position_exposures'):
        assert a[key]==b[key]
receipt=dict(source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest(),aggregates=agg)
(out/'aggregate.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
lines=['# role-set 来源正例排序覆盖诊断（2026-09-08）','','这是完整六端训练过程中已保存的来源距离的描述性复算，独立审计尚未覆盖此补充。它不重新运行模型，不是固定终点的全来源检索，也不是新的 Q1 或官方结果。原 role-set Q1 FAIL 保持。','',f"实际完成：{d['completed_at']}；CPU {d['elapsed_seconds']:.3f} 秒。读取全部 1560 步、116501504 个距离元素；无模型前向、优化器更新、held-out 排名读取或远端写入。",'', '正例按真实身份定义，排除 anchor 自身位置；同图片的其他增强位置仍保留。负例是所有不同身份（包括同场景）。跨场景统计仅限制正例，不删除无此类正例的 anchor 或图库干扰项。所有数字均是重复训练曝光，不能视为独立图片、身份或样本量。','', '严格反序定义为 d(p)>d(n)。非最远正例定义为 d(p)<max_p d(p)，所有最大值并列均排除。等距正负关系另行计数，本次为零。']
keys=[('steps','来源更新数'),('anchor_exposures','当前 anchor 曝光'),('anchors_with_any_strict_inversion','存在反序的 anchor'),('anchors_with_nonmax_strict_inversion','存在非最远正例反序的 anchor'),('anchors_with_cross_scene_positive','有跨场景正例的 anchor'),('anchors_with_cross_scene_strict_inversion','存在跨场景正例反序的 anchor'),('all_positive_affected_positions','被至少一个负例超过的正例位置'),('all_positive_affected_nonmax_positions','其中非最远正例位置'),('cross_scene_positive_affected_positions','被超过的跨场景正例位置'),('cross_scene_positive_affected_nonmax_positions','其中非最远跨场景正例位置'),('all_positive_strict_inversion_pairs','全部严格正负反序对'),('all_positive_nonmax_strict_inversion_pairs','其中非最远正例的反序对'),('anchors_all_hardest_positives_same_scene_with_cross_available','有跨场景正例但所有最远正例均同场景的 anchor')]
for phase,title in (('post_warmup','预热后 step66–260，三折合计'),('last65','末65步 step196–260，三折合计')):
    lines+=['',f'## {title}','','|计数|control|role_set|','|---|---:|---:|']
    for key,label in keys:lines.append(f"|{label}|{agg['control'][phase][key]:,}|{agg['role_set'][phase][key]:,}|")
lines+=['','## 证据的含义与边界','','末65步 candidate 有 2272 次正例位置曝光被负例超过，其中 1310 次严格低于该 anchor 的最远正例距离；这 1310 次中，1060 次为跨场景正例。对应 control 为 2239、1298、1061。新角色负例集合并未消除这些训练过程中记录的关系，但该比较不是独立的泛化因果检验。','','非最远正例对本 anchor 的单最远正例 fused 项没有直接的正例距离导数；这不表示该图片或角色参数没有总梯度，也不表示其他 anchor、角色辅助目标无法间接改善它。最远正例收近也可能同时改善其他正例。因此不能仅凭这些计数宣布现有目标错误、标签有噪声，或多正例/AP 损失必然有效。','','统计保留全部最大值并列，故最大值受影响位置数偶尔可以多于受影响 anchor 数。位置计数还包括不同视图及重复曝光；不能用位置比例当成独立检索风险估计。训练后段计数不替代最终固定模型的完整来源普查。','','核验：18 份来源文件按已有 CPU 清单核对大小/SHA；记录身份、场景和索引逐项核对来源协议；每步反序 anchor 数匹配既有运行统计；六端首批共384个 anchor 的排序计数与显式正负广播比较一致；完整流偏移、EOF和计数分解均通过。输入哈希与所有折/阶段原始计数保存在原 JSON；本报告聚合不重新运行训练或推理。','','后继尚未登记。本诊断只为评估多正例关系目标是否有来源侧动机提供描述证据，不选择温度、间隔、候选预算或新增训练。']
(out/'SOURCE_POSITIVE_COVERAGE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(report=str(out/'SOURCE_POSITIVE_COVERAGE.md'),source_sha256=receipt['sha256'])))
