from pathlib import Path
from datetime import datetime
import ast,hashlib,json

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');base=Path('C:/Users/gb/.codex_tmp')
out=base/'independent_evidence_draft/selection_mask_semantics_and_closed_cost868'
assert not out.exists();out.mkdir()
upstream='comparators/Signal-cd1b0a6/modeling/AddModule/useA.py'
source=base/'rt_complete862/received'/upstream
scope=json.loads((repo/'refine-logs/signal_selection_reference_v1/SOURCE_SCOPE.json').read_bytes())['source_sha256']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(source)==scope[upstream]
tree=ast.parse(source.read_bytes())
classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
token=ast.unparse(classes['TokenSelection']);interactive=ast.unparse(classes['ModalInteractive'])
assert 'rgb_mask = (rgb_mask_c + rgb_mask_i > 0).float()' in token
assert 'rgb_selected = rgb_patches * rgb_mask' in token
assert 'self.cross_attn(queries, keys_values, keys_values)' in interactive
assert 'key_padding_mask' not in interactive and 'attn_mask' not in interactive
entry=repo/'tools/run_signal_selection_reference.py';entry_tree=ast.parse(entry.read_bytes())
all_patch=next(n for n in entry_tree.body if isinstance(n,ast.FunctionDef) and n.name=='all_patch')
assert len(all_patch.body)==1 and ast.unparse(all_patch.body[0])=='return (rgb, nir, tir)'
closed=base/'independent_evidence_draft/selection_rgbnt201_analysis867/FACTS.json'
panel=json.loads(closed.read_bytes());rows={r['arm']:r for r in panel['rows']}
pairs=[]
for candidate,control in (('masked','all_patch'),('masked','global_only'),('all_patch','global_only')):
    a,b=rows[candidate],rows[control]
    pairs.append(dict(candidate=candidate,control=control,delta_metrics={k:a['best_metrics'][k]-b['best_metrics'][k] for k in a['best_metrics']},
        training_parameters_added=a['trainable_parameters']-b['trainable_parameters'],
        train_cli_seconds_added=a['full_train_cli_seconds']-b['full_train_cli_seconds'],
        train_cli_ratio=a['full_train_cli_seconds']/b['full_train_cli_seconds'],
        training_loop_ratio=a['training_loop_seconds']/b['training_loop_seconds'],
        loop_epoch_evaluation_save_ratio=a['loop_with_epoch_evaluation_save_seconds']/b['loop_with_epoch_evaluation_save_seconds']))
facts=dict(status='CPU_ONLY_SOURCE_SEMANTICS_AND_CLOSED_COST_READOUT',created_at=datetime.now().astimezone().isoformat(),
    source_sha256={upstream:sha(source),'tools/run_signal_selection_reference.py':sha(entry)},closed_input_sha256=sha(closed),
    properties=dict(mask_is_intra_inter_union=True,topk_not_total_retained_count=True,masked_tokens_are_zeroed_not_removed=True,attention_sequence_length_unchanged=True,
        selected_and_unselected_tokens_share_same_MHA_call=True,explicit_attention_exclusion_absent=True,actual_retained_counts_not_saved=True),
    pairs=pairs,boundary='No NN/SSH/runtime-mask measurement/officialscoring/report rerun. Actualunchanged pinnedauthorcode, not a bug claim. Zeroinputtokens keep softmaxmass even ifprojectionbias iszero; trainedK/Vbias maygive them a common projectedvalue. Union count is not fixedk; currentretainedcounts unknown. Closed201 timing is single sequentialrun onsharedserver, not repeated isolatedlatency orcausal speed claim. No cross-dataset conclusion; do notchangeactive topk/mask/threshold/recipe.')
(out/'FACTS.json').write_text(json.dumps(facts,indent=2)+'\n')
note='''# 已封存SIM选择语义与RGBNT201成本解释

源码的两类mask取并集，再将未选patch乘为零；没有删除token，也没有给MHA传key_padding_mask/attn_mask。因此固定top-k不是最终保留位置数，也不是attention实际序列长度。零位置仍进入softmax归一化：即使投影bias为零也占概率质量；训练后还可能有共同的K/V bias输出。这是忠实作者实现的结构性质，不是这次指标下降已定位的根因。

当前没有保存真实mask覆盖率，不能声称RGBNT100实际全选或选择已失效。当前all-patch控制保留同一交互器和头，仅用原patch替换token_selection的输出。masked与global比较另外混入交互器、额外训练头和1536→3072部署宽度。

在已经闭合的RGBNT201上，masked/all-patch同容量，训练CLI比值与loss-loop比值见FACTS；mAP/R1跟随各自单一mAP-best。时长来自顺序单次共享服务器运行，不是多次孤立benchmark；阶段还需M0、初始化、firststrict及历史失败成本。所有三个数据集完成后仍只执行原登记全query报告，不据此改top-k或另起训练。
'''
(out/'NOTE.md').write_text(note,encoding='utf-8');(out/'SOURCE.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(status=facts['status'],masked_vs_all_patch=next(r for r in pairs if r['control']=='all_patch'),mask_semantics=facts['properties']),indent=2))
