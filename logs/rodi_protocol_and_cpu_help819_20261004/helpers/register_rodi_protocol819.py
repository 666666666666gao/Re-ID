from datetime import datetime
from pathlib import Path
import ast
import hashlib
import json
import re
import shutil
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
proof = private / 'foundation_recipe_v1_20261002'
previous = json.loads((proof / 'five_copy818.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous['head']
assert subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo) == b''
assert not (proof / 'publication819_local.json').exists()
paper = private / 'independent_evidence_draft/rodi_primary_pdf_20261004'
download = json.loads((paper / 'DOWNLOAD.json').read_bytes())
assert download['git_blob_sha1'] == 'dcfca845758fdb072688c5077679f10ee6f92cc3'
assert download['sha256'] == hashlib.sha256((paper / 'RoDI.pdf').read_bytes()).hexdigest()
assert download['pages'] == 13
help_packet = private / 'independent_evidence_draft/role_input_detach_diagnosis818_cpu_help'
help_record = json.loads((help_packet / 'EXIT.json').read_bytes())
assert help_record['status'] == 'CPU_ONLY_ENTRY_IMPORT_AND_ARGUMENT_HELP_PASS'
assert help_record['exit_code'] == 0 and help_record['scope']['formal_completed'] == 4
observed = private / 'independent_evidence_draft/role_input_detach818_rgbnt100_observations/122733_411058'
observation = json.loads((observed / 'stdout.json').read_bytes())
assert json.loads((observed / 'EXIT.json').read_bytes())['exit_code'] == 0
assert observation['source_sha_unchanged'] and observation['exit'] is None
assert observation['receipts']['full']['epochs_completed'] == 6
assert observation['campaign']['active_command']['pid'] == 1716550
assert observation['campaign']['active_command']['gpus'] == [0, 1]
facts = {
    'status': 'AUTHOR_PDF_READ_AND_RELEVANT_TABLES_VISUALLY_CHECKED',
    'checked_at': datetime.now().astimezone().isoformat(),
    'source': download, 'visual_pages': [6, 7, 12, 13],
    'full_modality_table1_pdf_page': 6,
    'table1_metrics': {
        'CLIP_ViT_B16': {'RGBNT201': [84.1, 87.2, 92.0, 93.2], 'MSVR310': [64.1, 77.2], 'RGBNT100': [88.5, 97.6]},
        'DINOv3_ViT_B16_distilled': {'RGBNT201': [85.3, 87.9, 93.0, 94.8], 'MSVR310': [71.8, 84.8], 'RGBNT100': [89.0, 99.1]}},
    'table2_rgbnt201_mAP': {'CLIP': [76.0, 79.3, 82.7, 84.1], 'DINOv3': [81.0, 82.3, 84.0, 85.3]},
    'paper_training_details_pdf_page6': {'batch': 64, 'K': 8, 'optimizer': 'Adam', 'learning_rate': 0.00035, 'weight_decay': 0.0001, 'warmup_epochs': 10, 'GPU': 'RTX4090'},
    'unverified_exact_protocol': ['query/gallery membership and counts', 'camera/scene filtering implementation', 'checkpoint selection rule', 'total training epochs', 'exact pretrained weight hash'],
    'repository_tree_read_earlier': 'README.md and paper/poster only, no runnable training or evaluation code at fixed commit',
    'boundary': 'Published full-modality reference, not local reproduction or proof of exact protocol parity. No training change, new benchmark, model probe, inference or hardware power/temperature action.'
}
(paper / 'CHECK.json').write_text(json.dumps(facts, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(paper / 'NOTES.md').write_text('''# RoDI作者PDF：资源与协议边界

固定作者提交 `2f38911c49d42d4ca259d440a851b8d77dddccbe` 的[PDF](https://github.com/lsh-ahu/RoDI/blob/2f38911c49d42d4ca259d440a851b8d77dddccbe/assets/RoDI.pdf)已取得；13页，文件与Git blob哈希核验。目视复核PDF页6、7、12、13。二进制、页面图及全文提取留本地私有目录，不再分发。

| 主表1版本 | RGBNT201 mAP/R1/R5/R10 | MSVR310 mAP/R1 | RGBNT100 mAP/R1 |
|---|---|---|---|
| CLIP ViT-B/16 | 84.1/87.2/92.0/93.2 | 64.1/77.2 | 88.5/97.6 |
| DINOv3 ViT-B/16 distilled | 85.3/87.9/93.0/94.8 | 71.8/84.8 | 89.0/99.1 |

页7表2的201累计消融mAP：CLIP 76.0→79.3→82.7→84.1；DINOv3 81.0→82.3→84.0→85.3。不同预训练版本分列。

页6说明B64/K8、Adam、学习率3.5e-4、weight decay1e-4、warmup10轮、单RTX4090。该说明不足以核实精确query/gallery成员、camera/scene过滤、选checkpoint规则、总训练轮数和预训练权重哈希。固定仓库仍无可执行训练/评价代码。这是论文参照，不是同协议本地复现或SOTA已达成证明；不修改当前实验。
''', encoding='utf-8')
archive_name = 'logs/rodi_protocol_and_cpu_help819_20261004'
archive = repo / archive_name
assert not archive.exists()
(archive / 'paper').mkdir(parents=True)
for name in ('DOWNLOAD.json', 'CHECK.json', 'NOTES.md'):
    shutil.copyfile(paper / name, archive / 'paper' / name)
shutil.copytree(help_packet, archive / 'cpu_help')
shutil.copytree(observed, archive / 'RGBNT100_observation_1227')
(archive / 'helpers').mkdir()
for name in ('read_rodi_primary_pdf_20261004.py', 'render_rodi_primary_pages_20261004.py', 'check_role_input_detach_diagnosis818_cpu_help.py', 'register_rodi_protocol819.py'):
    shutil.copyfile(private / name, archive / 'helpers' / name)
intro = 'RoDI作者PDF的主表/资源已实际读取并目视复核，仍未建立精确评价协议等价；新固定best诊断仅CPU import/help通过，尚未封存输入或执行模型诊断。12:27观察为正式4/6、100 semantic 6/50轮，按原队列继续，预计14:10左右结束、14:05近末轮观察。仅26GPU0/1，无功率温度操作；Goal active/unmet。'
docname = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
docpath = repo / docname
assert hashlib.sha256(docpath.read_bytes()).hexdigest() == previous['doc_sha256']
doc = docpath.read_text(encoding='utf-8')
header = next(line for line in doc.splitlines() if line.startswith('**当前进度：'))
doc = doc.replace(header, '**当前进度：§41.819（2026-10-04更新）：** ' + intro)
doc += '\n\n## 41.819 RoDI作者PDF核读、固定best入口CPU检查与原队列等待边界\n\n' + intro + '\n\n'
doc += '### 文献参照重新建立原文来源\n\n'
doc += (paper / 'NOTES.md').read_text(encoding='utf-8').replace('# RoDI作者PDF：资源与协议边界\n', '')
doc += '\n### 执行不变与诊断前提\n\n新入口 `tools/diagnose_role_input_detach_best.py` 于12:19:05在空CUDA_VISIBLE_DEVICES下仅执行一次 `--help`，退出0；324项源文件SHA不变，输入封存不存在。此项不构建模型、不运行forward/M0、无新分数；只能证明CPU导入和参数解析。必须等六端各自完整50轮/首次严格评价及原一次性CPU全量报告退出0后，才能封存并执行固定best g/c/h/f诊断。\n\n'
doc += '12:27:35观察仍为原RGBNT100 semantic PID1716550/ticks34238497，正式4/6，自己的M0已通过，训练6/50轮完成、最新E7 batch61；源322不变，controller未退出。训练step c/g不是固定best检索效用证据，不据此调gain/LR。预计14:10–14:15完成，下一次近末轮观察约14:05，实际终态以回执为准。剩余native由原队列接续；不重启、不提前运行诊断、不退役仍为原报告依赖的新M0探针。原九端/旧固定best/M0退役历史边界不变。\n'
docpath.write_text(doc, encoding='utf-8')
goalpath = repo / 'refine-logs/CURRENT_GOAL.md'
goal = goalpath.read_text(encoding='utf-8')
updated = next(line for line in goal.splitlines() if line.startswith('更新：'))
goal = goal.replace(updated, '更新：2026-10-04 §41.819。' + intro)
goal += '\n\n§41.819：RoDI论文实际核读与CPU help/12:27观察证据见 logs/rodi_protocol_and_cpu_help819_20261004。文献主表分预训练资源，不冒称同协议复现；不借此变更当前实验。正式4/6，100两端仍待原完整回执。新324源诊断仍未封存/执行。完成目标仍需三集稳定净收益/强参照/必要性/完整多种子，Goal active/unmet。\n'
goalpath.write_text(goal, encoding='utf-8')
desktop = Path('C:/Users/gb/Desktop/document') / docpath.name
assert hashlib.sha256(desktop.read_bytes()).hexdigest() == previous['doc_sha256']
desktop.write_bytes(docpath.read_bytes())
owned = [docname, 'refine-logs/CURRENT_GOAL.md'] + [path.relative_to(repo).as_posix() for path in archive.rglob('*') if path.is_file()]
assert len(owned) == len(set(owned))
protected = json.loads((proof / 'publication818_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest for name, digest in protected.items())
publication = {'previous_head': previous['head'], 'section': '41.819', 'files': owned, 'protected_files': protected, 'immutable_archive_roots': [archive_name], 'doc_sha256': hashlib.sha256(docpath.read_bytes()).hexdigest()}
(proof / 'publication819_local.json').write_text(json.dumps(publication, indent=2) + '\n', encoding='utf-8')
source = (private / 'publish_role_input_detach818_msvr_pair.py').read_text(encoding='utf-8')
replacements = {
    'publication818_local.json': 'publication819_local.json', 'five_copy817.json': 'five_copy818.json',
    'commit818.json': 'commit819.json', 'target818.bundle': 'target819.bundle',
    'Record MSVR role gradient pair and read-only diagnosis plan': 'Verify RoDI primary paper protocol boundaries and CPU diagnosis entry',
    'range(739,819)': 'range(739,820)',
    "'sparse-checkout','add','logs/role_input_detach_msvr_pair818_20261004','refine-logs/role_input_detach_fixed_best_diagnosis_v1'": "'sparse-checkout','add','" + archive_name + "'",
    '/tmp/trifusion_target818_20261004.bundle': '/tmp/trifusion_target819_20261004.bundle',
    "'section':'41.818'": "'section':'41.819'", "five_copy818.json').write_text": "five_copy819.json').write_text"}
assert all(key in source for key in replacements)
source = re.sub('|'.join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)), lambda match: replacements[match.group()], source)
source = re.sub(r"'boundary':'Two datasets own full50[^']+'", "'boundary':'Primary RoDI paper resource/protocol review and CPU-only diagnosis help archived. Formal4/6 at12:27 historical observation, RGBNT100 queue unchanged. New diagnosis324 not sealed/executed. Source322/controls/init/recipe unchanged,26GPU0/1 only,25textonly,no power/temp. Scientific Goal active/unmet.'", source)
ast.parse(source)
target = private / 'publish_rodi_protocol819.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
print(json.dumps({'status': 'READ_ONLY_PAPER_AND_CPU_HELP_PUBLICATION_PREPARED', 'files': len(owned), 'doc_sha256': publication['doc_sha256']}))
