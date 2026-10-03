"""Publish the unregistered complete training-entry option, not a launch."""
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
previous = json.loads((proof/'five_copy764.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip() == previous['head']
protected = json.loads((proof/'publication764_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo/name).read_bytes()).hexdigest() == value for name, value in protected.items())
assert hashlib.sha256((draft/'independent_native_roles.py').read_bytes()).hexdigest() == 'bd895d3d27812de4db6bbfd9aa569966ca5ad6f3793c411513dc3049cd871a41'
assert hashlib.sha256((draft/'evidence_author_heads.py').read_bytes()).hexdigest() == 'b0a3a20c1ca5124f761d6bae01f88429a1d1c48cb0066926af7b82dd6f11f218'
source = draft/'run_independent_native_author_option.py'
ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
status = {'status': 'UNREGISTERED_FULL50_AUTHOR_ENTRY_OPTION_AST_ONLY',
          'recorded_at': datetime.now().astimezone().isoformat(),
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'variants': ['global_only','semantic','native'],
          'training_ports': [2026], 'physical_gpus': [0,1,2,3], 'max_parallel': 4,
          'foundation_selected': False, 'plan_registered': False,
          'source_review_passed': False, 'runtime_imported': False,
          'initialization_executed': False, 'm0_executed': False, 'formal_training_started': False,
          'boundary': 'AST/source preparation only. This is an author-option entry, not a selected campaign. Real initialization, paired forward, optimizer coverage, AMP updates, batch order and strict full-model reload are still unverified.'}
status_path = draft/'AUTHOR_ENTRY_OPTION_STATUS.json'
assert not status_path.exists()
status_path.write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
target = repo/'logs/native_evidence_author_entry_option765_20261003'
assert not target.exists() and not (proof/'publication765_local.json').exists()
target.mkdir()
files = []
for name in ('run_independent_native_author_option.py','AUTHOR_ENTRY_OPTION_NOTES.md','AUTHOR_ENTRY_OPTION_STATUS.json'):
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
assert '## 41.765 ' not in text
headings = [line for line in text.splitlines() if line.startswith('**') and '41.764' in line]
assert len(headings) == 1
text = text.replace(headings[0],
    '**首页当前进度：§41.765（2026-10-03）** 原F3尚待最后RGBNT100 metric_raw及原once report，既有final observer37344按首次11:02:04.284968／240秒观察，不重启。新独立细节组件有限CPU检查通过；raw feature、author heads及复用全50／strict／batch-order的完整入口草稿已形成，仅AST检查。author仍只是选项，未选择配方／注册计划／执行完整模型、初始化或M0／训练。仅26 GPU0–3/max4；25不训练。Goal ACTIVE／UNMET。')
append = f'''

## 41.765 完整50轮作者选项训练入口草稿（{status['recorded_at']}）

已将§763 raw feature及§764 author heads选项接到完整训练入口草稿`run_independent_native_author_option.py`，variants为shared global-only、semantic roles、semantic+independent native detail。仍未选择author作为最终foundation，未登记队列或科学计划；本入口未导入／执行，只有AST解析。global-only含共享适配器，不能等同F1无adapter纯baseline。

该入口复用原F1全50循环／官方完整图库评价／mAP-best及latest exact tie／完整state save及strict reload；用原F2 BatchOrderLoader给作者loader增加实际labels/cameras/paths顺序记录。保留author loss、dataset-specific sampler／增强／scheduler／LR，整个模型交给§764 optimizer option，旧normalized heads冻结。native构造在既有CPU RNG fork内进行，将原semantic完整state复制到带detail的state，再在全部variants构造完成后恢复visual及camera梯度；新增backbone构造会重新冻结Signal，因此此恢复必须放在variant replacement之后。

源码绑定记录public CLIP、fresh camera／视觉／common initializer SHA、实际head names、cfg、参数量及batch/K。损失仍是author head汇总，没有新增loss；部署L2_1536。继承8步M0的累计梯度、有限AMP更新及full-state reload，但目前未执行任何一门，不能把继承检查逻辑的存在写成PASS。后续实际paired witness还须核对全部common role张量、真实同批输入与semantic/native输出、BN模式和optimizer groups；全50后核对所有批次顺序。

entry实际SHA为`{status['source_sha256']}`，源码／notes／AST status存于`logs/native_evidence_author_entry_option765_20261003/`。§762组件CPU证据不接受本trainer，head/integration草稿源码也尚未获得完整所选plan的source review。原F3受保护source／plan字节保持；没有新模型、initializer、M0、训练、评价或额外report。继承history.seconds是训练循环耗时，不能以其求和推断含epoch评分／checkpoint保存的墙钟ETA。

下一步先完整接收原F3最后一端及原once report，经fresh完整审计／claim后选择可信配方并登记有限比较，再进行完整source review和真实全模型M0。停止尺度变体或margin／seed／系数救分；当前准备不能替代三个数据集上的实际净收益与资源注明SOTA要求。
'''
path.write_bytes((text+append).encode('utf-8'))
shutil.copyfile(path, Path('C:/Users/gb/Desktop/document')/path.name)
files.append(doc)
publication = {'previous_head': previous['head'], 'files': sorted(files), 'section': '41.765',
               'doc_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'doc_bytes': path.stat().st_size,
               'status': status['status'], 'training_ports': [2026], 'max_parallel': 4,
               'protected_files': protected}
(proof/'publication765_local.json').write_text(json.dumps(publication, indent=2)+'\n', encoding='utf-8')
publisher_source = Path('C:/Users/gb/.codex_tmp/publish_native_author_head_option764.py').read_text(encoding='utf-8')
publisher_source = publisher_source.replace('764','765').replace('five_copy763.json','five_copy764.json')
publisher_source = publisher_source.replace('range(739,765)','range(739,766)')
publisher_source = publisher_source.replace('native_evidence_author_head_option765_20261003','native_evidence_author_entry_option765_20261003')
publisher_source = publisher_source.replace('Prepare explicit author heads and whole-model optimizer ownership',
    'Prepare full50 author-option entry for independent native evidence')
publisher_source = publisher_source.replace('unregistered author-head source option and AST check only;',
    'unregistered full50 entry source option and AST check only;')
publisher = Path('C:/Users/gb/.codex_tmp/publish_native_author_entry_option765.py')
assert not publisher.exists()
publisher.write_text(publisher_source, encoding='utf-8')
print(json.dumps({key:value for key,value in publication.items() if key not in ('files','protected_files')}, indent=2))
