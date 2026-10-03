"""Record and publish an unexecuted author-head integration option."""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import shutil
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
draft = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
previous = json.loads((proof/'five_copy763.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() == previous['head']
protected = json.loads((proof/'publication763_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest() == value for name, value in protected.items())
assert hashlib.sha256((draft/'independent_native_roles.py').read_bytes()).hexdigest() == 'bd895d3d27812de4db6bbfd9aa569966ca5ad6f3793c411513dc3049cd871a41'
assert hashlib.sha256((draft/'ImageNativeEvidenceReader.py').read_bytes()).hexdigest() == '14c7ffef0d3e540a24542c5b6f8d9ed68c5d2410dcaed40b2225fe547d547db8'
source = draft/'evidence_author_heads.py'
tree = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
status = {'status': 'UNREGISTERED_AUTHOR_HEAD_OPTION_AST_ONLY',
          'recorded_at': datetime.now().astimezone().isoformat(),
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'classes': [node.name for node in tree.body if isinstance(node, ast.ClassDef)],
          'training_ports': [2026], 'physical_gpus': [0,1,2,3], 'max_parallel': 4,
          'foundation_selected': False, 'plan_registered': False,
          'source_review_passed': False, 'model_imported': False,
          'optimizer_executed': False, 'm0_executed': False, 'formal_training_started': False,
          'boundary': 'Source option and AST parsing only. Actual optimizer coverage, BN modes, paired features, gradients and full reload are not verified. Original F3 sources/controller/report remain untouched.'}
status_path = draft/'AUTHOR_HEAD_OPTION_STATUS.json'
assert not status_path.exists()
status_path.write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
target = repo/'logs/native_evidence_author_head_option764_20261003'
assert not target.exists() and not (proof/'publication764_local.json').exists()
target.mkdir()
files = []
for name in ('evidence_author_heads.py','AUTHOR_HEAD_OPTION_NOTES.md','AUTHOR_HEAD_OPTION_STATUS.json'):
    shutil.copyfile(draft/name, target/name)
    assert (target/name).read_bytes() == (draft/name).read_bytes()
    files.append((target/name).relative_to(repo).as_posix())
shutil.copyfile(Path(__file__), target/Path(__file__).name)
files.append((target/Path(__file__).name).relative_to(repo).as_posix())
doc = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
path = repo/doc
old = path.read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['doc_sha256']
text = old.decode('utf-8')
assert '## 41.764 ' not in text
headings = [line for line in text.splitlines() if line.startswith('**') and '41.763' in line]
assert len(headings) == 1
text = text.replace(headings[0],
    '**首页当前进度：§41.764（2026-10-03）** 原F3仍按既有任务接续，最新远端实查10:18为5/6完整50轮端及RGBNT100 metric_raw在26 GPU2实活，原报告调用0；final observer37344的首次11:02:04.284968与240秒节奏保持。新细节组件有限CPU检查通过；完整接入稿提供raw feature接口，并准备author单／三head及完整optimizer ownership选项，仅AST检查，未选择配方／注册／执行新M0或训练。仅26 GPU0–3/max4；25不训练。Goal ACTIVE／UNMET。')
append = f'''

## 41.764 作者训练head与全模型optimizer ownership接入选项（{status['recorded_at']}）

在§763明确raw feature接口后，继续实现未登记author-head选项，未据未完成F3中间成绩选择配方。SharedGlobalRawFeatures独立控制只注册共享backbone及历史neck/classifier，不注册roles和其读出；AuthorHeadEvidence对global／semantic／native均可以调用forward_features一次，取raw_fused后使用Signal已有DIRECT=1单1536 head或DIRECT=0三个512 head，推理仍返回L2的1536向量。没有重新调用Signal.forward、detach raw特征或新增分类参数。

旧normalized neck/classifier在这个选项中冻结且不参与forward。包装器train/eval先沿继承链执行，再显式恢复Signal对应模式，包括作者BN heads与visual encoder，避免继承的强制eval阻止BN正常训练。该模式合同须在三个新配对控制中一致；不能冒称与旧role训练模式相同。全模型初始化、BN实际运行统计及对应输出尚无运行证据。

优化器选项使用封存作者make_loss／make_optimizer，传入整个包装模型而非仅Signal；保留已有按名称分组的LR／decay，加入intended requires_grad参数ID与optimizer groups的恰好一次覆盖断言。这里只证明源码中有这个检查，没有实际构造优化器或验证梯度。具体采样、增强、head loss汇总、scheduler和全state保存仍须在选定训练配方及完整入口中落实。

源文件`evidence_author_heads.py`的实际SHA为`{status['source_sha256']}`，AST解析通过；代码及notes/status归档`logs/native_evidence_author_head_option764_20261003/`。没有导入模型、执行optimizer、运行真实paired前向／AMP-CUDA M0或训练；当前author只是准备选项，没有选作foundation。§762组件CPU通过不接受本包装器，§763 integration源码／detail组件字节未改，原F3受保护源码／计划也未改。

下一门仍是原six端全50＋strict＋once report及fresh audit／claim，随后登记可信匹配基座与有限证据对照，获得完整source review后才能实际M0。既有final observer在10:31实际PID37344实活；其11:02:04.284968首次观察与240秒间隔不变，无重复观察器、report或训练启动。总体三数据集性能与SOTA目标仍未满足。
'''
path.write_bytes((text+append).encode('utf-8'))
shutil.copyfile(path, Path('C:/Users/gb/Desktop/document')/path.name)
files.append(doc)
publication = {'previous_head': previous['head'], 'files': sorted(files), 'section': '41.764',
               'doc_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'doc_bytes': path.stat().st_size,
               'status': status['status'], 'training_ports': [2026], 'max_parallel': 4,
               'protected_files': protected}
(proof/'publication764_local.json').write_text(json.dumps(publication, indent=2)+'\n', encoding='utf-8')
publisher_source = Path('C:/Users/gb/.codex_tmp/publish_native_feature_interface763.py').read_text(encoding='utf-8')
publisher_source = publisher_source.replace('763','764').replace('five_copy762.json','five_copy763.json')
publisher_source = publisher_source.replace('range(739,764)','range(739,765)')
publisher_source = publisher_source.replace('native_evidence_feature_interface764_20261003','native_evidence_author_head_option764_20261003')
publisher_source = publisher_source.replace('Prepare raw feature interface for matched native evidence training',
    'Prepare explicit author heads and whole-model optimizer ownership')
publisher_source = publisher_source.replace('unregistered raw feature source preparation and actual running observation only;',
    'unregistered author-head source option and AST check only;')
publisher = Path('C:/Users/gb/.codex_tmp/publish_native_author_head_option764.py')
assert not publisher.exists()
publisher.write_text(publisher_source, encoding='utf-8')
print(json.dumps({key:value for key,value in publication.items() if key not in ('files','protected_files')}, indent=2))
