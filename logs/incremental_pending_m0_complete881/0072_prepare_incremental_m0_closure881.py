from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
base = repo / 'refine-logs/incremental_role_objective_v1'
proof_root = private / 'foundation_recipe_v1_20261002'
previous = json.loads((proof_root / 'four_copy880_2025_pending.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous['head']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
actual = json.loads((private / 'independent_evidence_draft/incremental_pending_m0_complete881/REMOTE.json').read_bytes())
assert actual['status'] == 'SIX_CONDITION_M0_MATRIX_PHYSICAL_SHA_VERIFIED'
assert actual['qualified'] == 0 and actual['failed'] == 6 and actual['total_updates'] == 48
assert not actual['formal_eligible'] and actual['formal_runs'] == 0
archive = repo / 'logs/incremental_pending_m0_complete881'
assert not archive.exists()
archive.mkdir()
mapping = {}
for name in ('incremental_pending_m0_launch880', 'incremental_pending_m0_observer880',
             'incremental_pending_m0_complete880', 'incremental_pending_m0_complete881'):
    folder = private / 'independent_evidence_draft' / name
    for source in sorted(folder.iterdir()):
        if source.is_file():
            dest = archive / f'{len(mapping):04d}_{source.name}'
            shutil.copyfile(source, dest)
            mapping[dest.name] = dict(original_local_path=str(source), sha256=sha(dest))
for name, artifact in actual['files'].items():
    dest = archive / f'{len(mapping):04d}_{Path(name).name}'
    dest.write_bytes(artifact['text'].encode('utf-8'))
    assert sha(dest) == artifact['sha256'], name
    mapping[dest.name] = dict(original_remote_path=name, sha256=artifact['sha256'])
(archive / 'FILE_MAP.json').write_text(json.dumps(mapping, indent=2) + '\n', encoding='utf-8')
summary = {key:value for key,value in actual.items() if key != 'files'}
support = {}
for row in actual['rows']:
    dataset, objective = row['dataset'], row['objective']
    if objective == 'repair_keep':
        name = next(name for name in actual['files'] if name.endswith(f'm0_{objective}_{dataset}/training_steps.jsonl'))
        steps = [json.loads(line) for line in actual['files'][name]['text'].splitlines()]
        support[dataset] = dict(steps=8, legal_positive_pair_exposures=sum(step['legal_positive_pairs'] for step in steps),
            legal_triplet_exposures=sum(step['legal_triplets'] for step in steps),
            eligible_query_exposures=sum(step['eligible_queries'] for step in steps),
            multiple_positive_query_exposures=sum(step['multiple_positive_queries'] for step in steps),
            steps_with_legal_triplets=sum(step['legal_triplets'] > 0 for step in steps))
summary.update(repair_keep_actual_source_support=support, archive_file_count=len(mapping),
    collector_calls=2, successful_collector_calls=1,
    collector_failure='First CPU collector incorrectly asserted the RGBNT201-specific 281 parameter tensors for every dataset. Independent continuation uses each verified initializer binding (201281; vehicles285), preserving first collector failure. No neural rerun or qualification change.',
    interpretation='All six production M0s passed, including complete combined-task parameter activity. All isolated unscaled auxiliary first-eight gates failed: c gradients nonzero; CNN Q/K active only in 201 and100 batch-ratio conditions; remaining Q/K sums zero. This does not prove absent scaled training signal, permanent graph disconnection, underflow root cause, or retrieval failure.')
result_path = base / 'M0_COMPLETE_MATRIX_20261007_881.json'
result_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
csv_path = base / 'M0_COMPLETE_MATRIX_20261007_881.csv'
with csv_path.open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['dataset', 'objective', 'production_status', 'gradient_tensors',
        'optimizer_updates', 'reload_max_abs_difference', 'correction_gradient_norm_sum', 'active_query_key_tensors', 'gate_status'])
    writer.writeheader()
    for row in actual['rows']:
        writer.writerow({**{name:row[name] for name in writer.fieldnames if name != 'active_query_key_tensors'},
            'active_query_key_tensors':sum(value > 0 for value in row['query_key_gradient_norm_sums'].values())})
next_path = base / 'NEXT_STEP_DECISION_20261007_881.md'
next_path.write_text('''# Next decision after the complete six-condition M0 matrix

The two registered final-vector objectives have no formal retrieval result. Preserve all six failed first-eight isolated unscaled activity gates and the successful production M0 receipts. Do not rerun the original first condition, lower the gate, increase loss/gain, change seed or treat zero measurements as retrieval failures.

The captured first-eight correction gradients are nonzero for every condition, while Transformer/Mamba Q/K measurements are zero. The previous fixed post-eight one-forward test found all six Q/K active at both scale1 and256. Therefore neither a permanent graph disconnection nor scale-related root cause is established. The actual production AMP backward is scaled; the isolated auxiliary observation was unscaled. Those are different numerical paths.

The next bounded decision should establish where that specific initial-state numerical difference occurs, using one freshly verified public initialization, one actual source batch, one AMP forward and scale1/production256 VJPs on the same graph, recording unused flags and restoring buffers; zero optimizer updates and no official evaluation. This is a proposed diagnostic, not an accepted gate revision or a registered deployment. It needs a concrete source and review before execution. It does not reconstruct the old eight augmented batches.

If no adequate initial auxiliary path is observed, prefer a controlled change in the location/content of supervision (region evidence before the residual readout), with a matched uniform task control. Avoid another gain, margin or free-query search. A regional objective would be a new hypothesis, not proof that current P1/P2/P3 work; prior local-ID, V24/V26/R2/CIRC precedents remain necessary controls.

Full-stage source readiness is not execution eligibility. The original six-end formal stage remains ineligible. Goal ACTIVE / UNMET.
''', encoding='utf-8')
tracker = f'''# Six-condition incremental-objective M0 closed

- Original first201 MD gate failure retained; five never-run conditions executed once, terminal08:13:25, sole observer93450 confirmed08:16:19.
- Production M0:6/6 PASS,8 effective updates each,48 total,BN8,strict reload0; gradient tensors201281/vehicles285.
- Isolated unscaled auxiliary activity:0/6 qualified. Every correction gradient is nonzero; MD201/100 only CNN Q/K active; all repair_keep Q/K cumulative norms zero. No actual scaled inactivity or retrieval-failure claim.
- Six initializers and first-eight source batches match controls;398 source bytes/current45/fixed9 inputs physically unchanged.
- Five closed engineering probes retired:{actual['retired_probe_bytes']} bytes; original diagnostic probe retained; no formal PTH deleted in this round.
- CPU collection2 calls/1 successful continuation; original parameter-count assertion failure preserved; no neural rerun.
- Formal50 runs0; original full-stage eligibility false. Next bounded initial-state numerical boundary diagnosis is only a proposal, not deployed. No rate/gain/margin/seed/gate rescue.
- Only26physicalGPU0/1,max1; no25/GPU2/3/power-temp actions. Goal ACTIVE / UNMET.
'''
tracker_path = base / 'EXPERIMENT_TRACKER_20261007_M0_COMPLETE881.md'
tracker_path.write_text(tracker, encoding='utf-8')
shutil.copyfile(tracker_path, base / 'EXPERIMENT_TRACKER.md')
doc_name = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
doc = repo / doc_name
assert sha(doc) == previous['doc_sha256']
table = ['| 数据集 | 目标 | 生产梯度张量 | 修正梯度累计范数 | 活动Q/K张量 | 原独立活动门 |',
         '|---|---|---:|---:|---:|---|']
for row in actual['rows']:
    table.append(f"| {row['dataset']} | {row['objective']} | {row['gradient_tensors']}/{row['gradient_tensors']} | {row['correction_gradient_norm_sum']:.10g} | {sum(value > 0 for value in row['query_key_gradient_norm_sums'].values())}/6 | FAIL |")
section = '''

## 41.881 2026-10-07：增量目标六端M0资格矩阵收齐；生产通过与独立活动失败分账

仅五个原PENDING条件续接，supervisor2290677于08:06:15启动、controller2290684于08:13:25完成exit0。唯一observer93450在08:12:18观察前四条件已分类、末条件尚未完成，08:16:19确认五端终态；原首端201 md_batch_ratio未重跑、原EXIT1仍保留。两目标×三数据集共六次生产M0、每端8次真实更新，累计48次，不是六端50轮。

六端生产训练均M0_PASS，全部应训练参数有累计梯度、BN计数8、严格重载最大差0、旧来源首八batch及匹配公开初始化不变；201为281个张量，车辆为285个张量。另一方面，原登记的未缩放新增损失独立VJP门全部FAIL，资格0/6，formal_eligible=false，正式训练0。表内范数属于八步独立观测累计，不能当实际生产GradScaler反向的同一观测。

''' + '\n'.join(table) + f'''

每端修正向量c均有非零独立梯度，MD201/100的CNN Q/K累计活动，其余Q/K未达到非零门；repair_keep三端全部六个Q/K累计为0。实际合法关系统计保存在M0_COMPLETE_MATRIX_20261007_881.json，不能解释成所有训练关系为空。先前post-eight固定状态的scale1/256测试又都能传到sixQK，因此既不能宣称永久断路，也未证明当前原八步零值的唯一原因是underflow或实现遗漏。没有新增mAP，不能写成算法检索失败或新方法成功。

CPU收集有两次独立调用：原59711因将201的281张量硬编码到车辆而失败，18684续接按各自已核验initializer计数成功；原失败源码/退出保存，不宣称一次报告或重跑神经训练。08:18:06实存核对398来源、current45与fixed9输入不变，全部生产/独立活动/批次/清理原始文本有SHA映射。

按已授权无用权重清理，五份新增工程probe在严格回执与活动分类落盘后退役，共{actual['retired_probe_bytes']:,}字节（约1.79GB），原首端诊断probe、作者/公开初始化、正式最佳权重、当前45依赖与正式距离证据保留。本轮未退役任何正式PTH。历史工程probe已不可直接重载；原始回执和SHA保留，不能称所有二进制仍完整。

下一步需先把具体初始状态的数值边界问题收束成有限的一次forward/同图两VJP/0优化诊断，源代码与复核尚未登记，不能宣称已经执行或借此改判原门。随后才决定是否把新增责任直接作用到读出前区域证据，配匹配普通任务控制；不是继续加倍率、margin、自由query或重复旧队列。原六端full源码准备不等于执行资格，不自动启动。三个模块与三数据集SOTA仍未完成；Goal ACTIVE_UNMET。仅26物理GPU0/1/max1，25与GPU2/3、功率温度不查询不操作。
'''
assert '\ufffd' not in section
with doc.open('ab') as stream:
    stream.write(section.encode('utf-8'))
manifest = repo / 'MANIFEST.md'
manifest_original = sha(manifest)
with manifest.open('a', encoding='utf-8') as stream:
    stream.write('\n- 2026-10-07 §41.881：两目标×三集M0资格矩阵收齐，生产6/6通过、独立活动0/6，48次更新/正式0；五工程probe退役1,786,893,304字节，原失败与CPU收集失败保留。\n')
owned = [doc_name, 'MANIFEST.md', str(result_path.relative_to(repo)).replace('\\', '/'),
    str(csv_path.relative_to(repo)).replace('\\', '/'), str(next_path.relative_to(repo)).replace('\\', '/'),
    str(tracker_path.relative_to(repo)).replace('\\', '/'), 'refine-logs/incremental_role_objective_v1/EXPERIMENT_TRACKER.md']
owned += [str(path.relative_to(repo)).replace('\\', '/') for path in archive.iterdir() if path.is_file()]
old = json.loads((proof_root / 'publication880_local.json').read_bytes())
assert all(sha(repo / name) == digest for name, digest in old['protected_files'].items())
scope = base / 'PENDING_M0_SOURCE_SCOPE.json'
assert all(sha(repo / name) == digest for name, digest in json.loads(scope.read_text())['source_sha256'].items())
record = dict(previous_head=previous['head'], files=sorted(set(owned)), protected_files=old['protected_files'],
    source_count=398, source_scope_sha256=sha(scope), doc_sha256=sha(doc), original_manifest_sha256=manifest_original,
    remote_sparse_roots=['refine-logs/incremental_role_objective_v1', 'logs/incremental_pending_m0_complete881'],
    immutable_archive_roots=['logs/incremental_pending_m0_complete881'])
(proof_root / 'publication881_local.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(owned_files=len(record['files']), matrix='0/6', updates=48, retired_probe_bytes=actual['retired_probe_bytes'], formal_runs=0)))
