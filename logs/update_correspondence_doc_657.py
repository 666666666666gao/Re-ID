from pathlib import Path
import hashlib
import json
import shutil

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
logs = root / 'logs'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
matrix_path = logs / 'context_identity_accepted_complete_670_20260929.json'
matrix = read(matrix_path)
prior = read(logs / 'context_identity_accepted_667_20260929.json')
rows = {(r['dataset'], r['variant']): r for r in matrix['rows']}
assert matrix['verified_complete'] == matrix['expected_endpoints'] == len(rows) == 15
assert all(rows[(r['dataset'], r['variant'])] == r for r in prior['rows'] if r['status'] == 'VERIFIED_COMPLETE')
campaign = read(logs / 'context_identity_campaign_complete_670_20260929.json')
assert campaign['status'] == 'COMPLETE' and all(r['status'] == 'COMPLETE' for r in campaign['jobs'])
factor_dir = logs / 'context_identity_factor_complete_657_20260929'
factor = read(factor_dir / 'summary.json')
assert factor['matrix_sha256'] == sha(matrix_path) and len(factor['pairs']) == 15
assert factor['source_sha256'] == sha(root / 'tools/analyze_correspondence_context_identity.py')
assert sum(p['origin'] == 'sealed_report_reused' for p in factor['pairs']) == 2
for pair in factor['pairs']:
    path = logs / Path(pair['report']).name if pair['origin'] == 'sealed_report_reused' else factor_dir / Path(pair['report']).name
    assert sha(path) == pair['report_sha256']
review_dir = root / 'refine-logs/correspondence_context_identity_v1'
reviews = ('REVIEW_FINAL_ANALYZER_657_20260929', 'REVIEW_ENERGY_DIAGNOSTIC_657_20260929',
           'REVIEW_ENERGY_RUNTIME_657_20260929', 'REVIEW_FINAL_PANEL_RUNTIME_657_20260929')
for name in reviews:
    record = read(review_dir / f'{name}.json')
    assert (review_dir / f'{name}.md').is_file()
    assert record['review_independence'] == 'same-family'
    assert record['acceptance_status'] == 'provisional'
energy = {dataset: read(logs / f'context_energy_{dataset}_657_20260929/summary.json')
          for dataset in ('RGBNT201', 'MSVR310')}
for dataset, summary in energy.items():
    assert summary['status'] == 'FIVE_ACCEPTED_BEST_ENERGY_DIAGNOSTICS_COMPLETE'
    assert summary['source_sha256'] == sha(root / 'tools/diagnose_correspondence_context_energy.py')
    for cell in summary['rows']:
        row = rows[(dataset, cell['variant'])]
        assert all(cell[name] == row[name] for name in ('best_epoch', 'checkpoint_sha256', 'receipt_sha256', 'distance_sha256'))
        assert cell['maximum_forward_reconstruction_error'] < 1e-5
        assert max(cell['maximum_saved_distance_difference'].values()) < 1e-5
archives = {}
for index in (667, 670):
    data = read(logs / f'context_identity_archive_{index}_20260929.json')
    for record in data['archived']:
        folder = Path(record['directory'])
        assert all(sha(folder / name) == digest for name, digest in record['files'].items())
        archives[folder.name] = {'run_dir': rows[(record['dataset'], record['variant'])]['run_dir'], 'files': record['files']}
archive_path = Path('C:/Users/gb/.codex_tmp/context_closure_archive_657.json')
assert not archive_path.exists()
archive_path.write_text(json.dumps(archives, indent=2) + '\n', encoding='utf-8')
shutil.copyfile(archive_path, logs / 'context_closure_archive_657_20260929.json')
sources = {**read(Path('C:/Users/gb/.codex_tmp/m3_source_sha_640.json')),
           **read(logs / 'context_identity_manifest_653_20260929.json')['source_sha256']}
assert all(sha(root / name) == digest for name, digest in sources.items())
doc = root / 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
assert sha(doc) == 'ab47348df07435cd03a9964ec78910fb1c121c1a95012f53450dd9fa49c16b5f'
raw = doc.read_bytes()
key = '### 0.1 历史阶段摘要'.encode()
front, history = raw[:raw.index(key)].decode(), raw[raw.index(key):]
start, end = front.index('**本页更新至'), front.index('** 历史规则') + 2
front = front[:start] + '**本页更新至§41.657：新context/local五条件×三集15/15全50／best／重载／CPU验收闭合。查询平均ΔmAP三集仅+0.0210/+0.0023/+0.0214；局部身份监督201/100为−0.5915/−0.6152，MSVR+0.1819且与query负交互。10份已选权重的完整Q/G诊断通过原forward及三路距离一致性：局部监督提高local-alone，却使角色缩放修正能量比201约6%→1.1%、MSVR约49%→3.6%。没有据此调倍率、选seed或取消负端；下一项跨层持续角色结构仅源码准备，未训练。**' + front[end:]
table = '\n'.join('| ' + variant + ' | ' + ' | '.join(
    f"E{rows[(dataset, variant)]['best_epoch']} " + '/'.join(f"{rows[(dataset, variant)]['metrics'][name]:.4f}"
       for name in (('mAP','Rank-1','Rank-5','Rank-10') if dataset == 'RGBNT201' else ('mAP','Rank-1')))
    for dataset in ('RGBNT201','RGBNT100','MSVR310')) + ' |'
    for variant in ('static_none','context_none','static_local','context_local','context_global'))
factors = '\n'.join('| ' + dataset + ' | ' + ' | '.join(
    f"{group['factor_metrics_pp'][field]['mAP']:+.4f}/{group['factor_metrics_pp'][field]['Rank-1']:+.4f}"
    for field in ('mean_query_effect','mean_local_identity_effect','query_local_identity_interaction','local_vs_global_identity')) + ' |'
    for dataset, group in factor['datasets'].items())
pairs = '\n'.join(f"| {p['dataset']} | {p['control']}→{p['candidate']} | {p['delta_metrics']['mAP']:+.4f}/{p['delta_metrics']['Rank-1']:+.4f} | {p['rank1_repairs']}/{p['rank1_new_errors']} | {p['identity_macro_mean_delta_ap_points']:+.4f} |" for p in factor['pairs'])
energy_table = '\n'.join(f"| {dataset} | {cell['variant']} | {cell['readout_gain']:.5f} | " + ' | '.join(
    f"{cell['statistics'][split]['scaled_correction_to_global_norm']['mean']:.5f}"
    for split in ('query','gallery')) + f" | {cell['statistics']['query']['global_correction_cosine']['mean']:+.5f} |"
    for dataset, summary in energy.items() for cell in summary['rows'])
steps = sum(r['task_scalars']['logged_steps'] for r in matrix['rows'])
maximum_loss_error = max(r['task_scalars']['maximum_loss_reconstruction_error'] for r in matrix['rows'])
append = f'''

### 41.657 全局条件查询×联合局部身份职责完整15端闭合与实际融合能量诊断（2026-09-29）

父队列实际{campaign['completed_at']}正常COMPLETE：15 COMPLETE/0 RUNNING/0 PENDING/0 FAILED。父队列自己的collector于{matrix['collected_at']}完整验收15/15；本次摄取该accepted_matrix原始字节、重核checkpoint/receipt SHA与已有CPU三路全图库差值，不重复旧13端的训练或评价。此前13条VERIFIED_COMPLETE对象完全不变，最后两条100的training/official/steps文本另行归档。当前只是这批实验闭合，统一研究目标仍ACTIVE/UNMET。

#### 同协议、同初始化、full50单mAP-best的完整五条件

| 条件 | RGBNT201 mAP/R1/R5/R10 | RGBNT100 mAP/R1 | MSVR310 mAP/R1 |
|---|---|---|---|
{table}

五条件全部M1/M2开、M3关，输出1536D，固定纯CLIP ReID起点和seed42；每数据集五端初始完整state SHA相同。static/context无aux两端参数相同，三种有aux端参数相同；同存储参数不等于同有效函数容量。不同数据集分类头宽度由来源身份数决定，不能跨数据集比较参数数或绝对分数判断难度。新结果不继承旧Signal增强路线的83分，不将其减去纯baseline69.64冒算新模块贡献。

最后100 context_local与context_global均选E1；local的shared_global 84.5111/94.9271，joint_local 74.3882/87.6968，fused84.5125/94.9271。global辅助监督对应global85.3237/95.6851、local59.1912/79.5918、fused85.3363/95.5685；单模型global/local分解不是独立训练角色消融。15端总{steps:,}步骤标量重构、最大误差{maximum_loss_error:.12g}；100五端Triplet全部0，所有有aux端其aux ID非零。任务标量不等于梯度或AdamW更新份额。

#### 全因素和合法首位关系：没有“查询＋局部职责”稳定的大幅增量

| 数据集 | query平均ΔmAP/R1 | local ID平均ΔmAP/R1 | query×local交互ΔmAP/R1 | local vs global额外ID ΔmAP/R1 |
|---|---:|---:|---:|---:|
{factors}

这是两个因素四条件的平均作用与差中差，不是可顺次叠加的模块贡献。201/100条件查询作用很薄，局部监督在两种查询下都降低mAP；MSVR查询在无aux时+.7198 mAP，而有local ID时−.6770，交互−1.3968。MSVR static_local53.1429/69.8816虽改善其同结构控制，仍低于发布Signal53.2424/72.4196；context_local在201/100低于同结构static_none，并未解决global-only之外的可靠角色增量。不同已选best的条件比较不是同训练步状态的唯一因果分离。

| 完整距离配对 | 条件 | ΔmAP/R1 | 首位修复/新增 | 身份等权ΔAP百分点 |
|---|---|---:|---:|---:|
{pairs}

统一分析工具SHA{factor['source_sha256']}，于{factor['at']}开始、实际CLI退出0，15配对和三个因素表完整生成；只用保存的完整距离与合法camera/时间段标签，不新推理、不重加权。13份新配对JSON＋summary一次归档；之前201§655、MSVR§656两份static_none→context_none报告显式--sealed-pair复用，原字节与绑定SHA保留。单训练seed，官方基准逐轮选点及历史选择，固定权重的身份bootstrap不是训练seed方差或未消费测试上的泛化保证。

#### 首次读取实际修正能量与方向，完整原forward及三路距离通过

诊断于17:50:21在空GPU1/2启动，201/MSVR各固定五份已接受的best，正常eval/inference_mode全query/gallery，禁止使用能量结果挑倍率、位置、身份或权重。仅hook读取backbone global与role evidence，角色区域读出额外计算一次；教师与分类头不运行。10个模型原forward重构最大差0，三个完整距离最大差5.960464477539062e-7（201 static_local的joint_local），其它均0，阈值1e-5。201 summary实际{energy['RGBNT201']['at']}，MSVR为{energy['MSVR310']['at']}；保存remote逐样本标量和协议绑定，local只归档12份JSON。

| 数据集 | 条件 | 学习gain | query平均abs(gain)·norm(c)/norm(g) | gallery同量 | query平均cos(g,c) |
|---|---|---:|---:|---:|---:|
{energy_table}

local CE令角色单独检索更可辨识，却没有令它在最终表示中承担更大作用：201缩放修正范数比由约6%降为1.1%，MSVR由约48%–49%降为3.6%–3.7%。这同时伴随global成绩变化，不能据此宣布范数是唯一原因或放大倍率就会涨分；局部空间的较好mAP也不保证与global在加性融合中形成有效判别互补。余弦近0不是身份互补的证明，MSVR其它端负余弦也不自动代表有害。测量仅针对每端已选best，不能冒充全训练轨迹或参数更新比例。

代码可确定：aux local直接监督Normalize(c)，不经过gain；其直接gain导数为0，但仍经角色、M1共享适配和context查询回传。fused身份梯度包含gain与归一化Jacobian，却不能从该系数换算角色梯度或实际更新份额。保留此边界，不把“global捷径”当成已经唯一证明的根因。

#### 审查、执行证据与后继边界

final analyzer和energy source均经fresh native Codex Astra max/fork none只读审查，无具体源码启动阻塞；source审查时只有13/15或没有runtime，原WARN报告保留，不改为事后PASS。后续runtime审查读取完整15和10份诊断JSON，仍same-family/provisional，远端原张量不在local独立重放范围，九份runtime manifest也不是全部依赖的传递锁。能量诊断的OS退出码没有由持久父进程记录，不能从成功summary和PID消失补造exit0；仅完整分析CLI退出0是实际工具返回。原GPU/队列启动条件由17:50 snapshot及启动器核对支持。报告见refine-logs/correspondence_context_identity_v1/REVIEW_*657_20260929，caller调用转录见logs/context_identity_review_invocations_657_20260929.json，不独立证明backend身份。

观察更正：原本计划15:31的本地等待句柄丢失且没有写回执，确认后17:33才执行observer666一次，不能把其filename当成15:31实查。后续17:50/17:57使用remote持久一次observer；两次在预定时间之前取文件出现不存在，不代表训练失败，未重启队列。能量启动器的嵌套换行在local py_compile时发现并修正，原诊断还未启动；不涉及正式训练失败或阈值变更。

下一项回到角色证据形成：当前三层快照先混合再执行角色，新的跨层角色状态源码已开始准备。拟比较mixed_once、三层独立角色处理后平均、角色状态按第4→8→12层持续传递；共享CLIP/M1路径、采样地址、query/读出与角色参数保持明确控制，以区分更多层证据/计算与持续状态。不开M3和已失败的local辅助头，不扫描倍率。当前只有新增模型源码语法检查，没有完整入口、生产M0、队列或成绩，不能写为已实现成功方法；原15端和原M3全12结果封存不改。
'''
new_bytes = (front.encode() + history + append.encode())
assert new_bytes.count('### 41.657 '.encode()) == 1
assert new_bytes[new_bytes.index(key):new_bytes.index(key)+len(history)] == history
doc.write_bytes(new_bytes)
shutil.copyfile(doc, Path('C:/Users/gb/Desktop/document') / doc.name)
tracker = review_dir / 'EXPERIMENT_TRACKER.md'
text = tracker.read_text(encoding='utf-8')
start, end = text.index('| 条件 |'), text.index('\n\n15:06实际')
tracker_table = '| 条件 | RGBNT201 | RGBNT100 | MSVR310 |\n|---|---|---|---|\n' + '\n'.join(
    '| ' + variant + ' | ' + ' | '.join(f"VERIFIED_COMPLETE E{rows[(dataset,variant)]['best_epoch']} {rows[(dataset,variant)]['metrics']['mAP']:.4f}/{rows[(dataset,variant)]['metrics']['Rank-1']:.4f}"
    for dataset in ('RGBNT201','RGBNT100','MSVR310')) + ' |' for variant in ('static_none','context_none','static_local','context_local','context_global'))
text = text[:start] + tracker_table + text[end:]
text += f'\n2026-09-29 18:00完整15/15：父队列{campaign["completed_at"]} COMPLETE，原13正式行不变。统一CPU条件分析15配对完成、显式复用2封存报告；10权重能量诊断完整数组一致性通过。完整正负结果和限制均保留；不能称三数据集目标完成。跨层角色结构仅source准备、未训练。\n'
for name in ('EXPERIMENT_TRACKER_20260929_1800.md','EXPERIMENT_TRACKER.md'):
    (review_dir / name).write_text(text, encoding='utf-8')
print(json.dumps({'status':'DOCUMENT657_COMPLETE15_AND_ENERGY_UPDATED','document_sha256':sha(doc),'steps':steps,'new_archived_endpoints':len(archives)}))
