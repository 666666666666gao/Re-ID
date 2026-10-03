"""Archive actual launch/retirement receipts and the first timed observation."""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import shutil
import subprocess
import paramiko

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
native = private / 'independent_evidence_draft'
proof = private / 'foundation_recipe_v1_20261002'
previous = json.loads((proof / 'five_copy769.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous['head']
assert not (proof / 'publication770_local.json').exists()
observation_path = native / 'observer769/observation_0001.stdout.json'
observed = json.loads(observation_path.read_bytes())
launch = json.loads((native / 'deploy770/stdout.json').read_bytes())
assert launch['port'] == 2026 and launch['max_parallel'] == 4
target = repo / 'logs/independent_native_launch770_20261003'
assert not target.exists()
target.mkdir()
files = []

def copy(source, relative):
    path = repo / relative
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, path)
    assert path.read_bytes() == source.read_bytes()
    files.append(relative)

for group in ('deploy769', 'deploy770', 'launch_failure_resources769', 'storage770'):
    for source in sorted((native / group).iterdir()):
        assert source.suffix in ('.json', '.txt')
        copy(source, 'logs/independent_native_launch770_20261003/' + group + '/' + source.name)
for name in ('observation_0001.stdout.json', 'observation_0001.stderr.txt', 'observation_0001.EXIT.json'):
    copy(native / 'observer769' / name, 'logs/independent_native_launch770_20261003/observation/' + name)
for name in ('prepare_closed_shared_private_m0_770.py', 'retire_closed_shared_private_m0_770.py',
             'deploy_independent_native_evidence770.py', 'observe_native_launch_failure769.py',
             'observe_independent_native769.py', 'prepare_native_launch770_publication.py'):
    copy(private / name, 'logs/independent_native_launch770_20261003/' + name)

client = paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138', port=2026, username='gaob', key_filename='C:/Users/gb/.ssh/id_ed25519', timeout=20)
receipt_relative = 'logs/closed_shared_private_m0_retirement770_20261003/RETIREMENT.json'
receipt_local = target / 'raw_retirement770.json'
with client.open_sftp() as sftp:
    sftp.get('/data/gaob/Re-ID/Trifusion/' + receipt_relative, str(receipt_local))
client.close()
retired = json.loads(receipt_local.read_bytes())
assert retired['status'] == 'RETIRED_EXACT_9_CLOSED_SHARED_PRIVATE_M0'
assert len(retired['candidates']) == 9 and retired['retired_bytes'] == 3216709812
assert all(row['deleted'] for row in retired['candidates'])
files.append(receipt_local.relative_to(repo).as_posix())
formal_started = observed['phase'] == 'full' or any(observed['counts']['full'][k] for k in ('RUNNING', 'COMPLETE', 'FAILED'))
status = {'status': 'NATIVE_QUEUE_LAUNCHED_OBSERVED', 'recorded_at': datetime.now().astimezone().isoformat(),
    'launch': launch, 'observation': observed, 'original_launch_failed_before_popen': True,
    'retired_closed_m0': 9, 'retired_bytes': retired['retired_bytes'],
    'retirement_raw_sha256': hashlib.sha256(receipt_local.read_bytes()).hexdigest(),
    'new_formal_training_started': formal_started, 'training_ports': [2026], 'max_parallel': 4,
    'goal': 'ACTIVE_UNMET', 'boundary': 'Real launch and timed read-only observation; not performance or completed M0 acceptance. Original disk failure kept. No2025 training.'}
status_path = target / 'STATUS.json'
status_path.write_bytes((json.dumps(status, indent=2) + '\n').encode())
files.append(status_path.relative_to(repo).as_posix())
doc = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
path = repo / doc
old = path.read_bytes()
assert hashlib.sha256(old).hexdigest() == previous['doc_sha256']
text = old.decode('utf-8')
assert '## 41.770 ' not in text
headings = [line for line in text.splitlines() if line.startswith('**') and '41.769' in line]
assert len(headings) == 1
text = text.replace(headings[0], f"**当前进度（§41.770，2026-10-03）：** 独立原生细节九端队列已在2026启动；{observed['at']}实查status={observed['status']}、phase={observed['phase']}、M0完成{observed['counts']['m0']['COMPLETE']}/9、正式完成{observed['counts']['full']['COMPLETE']}/9。仅GPU0–3/max4。首次启动因磁盘预算未过、在Popen前退出；核验退役九份早已完成的shared/private M0后，保持所有代码/预算/实验合同不变，12:57:34启动PID3160856。2025不训练。Goal ACTIVE_UNMET。")
text += f'''

## 41.770 独立原生细节队列实际启动与磁盘阻塞收尾（{status['recorded_at']}）

§769已同步提交0c1dff450d74b82b6a93f7614124fa601d4a3a32，1377份累计自有文本与五份主文档一致。首次部署未越过磁盘预算检查，remote stdin line11 assertion失败；无新campaign或launcher log，未创建训练进程。12:53:05实查可用9,140,363,264字节，低于固定10,292,822,016字节全批预算；不修改预算或旧F3保护断言。四张3090仍无计算任务。

随后只退役已完成shared_private_evidence_20261001_v2正式9/9验收的九份M0检查权重。逐文件核对实际SHA/大小、M0_PASS原回执、闭合summary和退出进程，并保护其九份正式best、全部当前F3二进制及公共CLIP。12:57:16实删3,216,709,812字节，可用空间变为12,357,050,368字节；原文本、distance和正式权重未改。此九份历史M0二进制从现在起不能直接重放，不覆盖此前24份退役或任何失败记录。精确原始远端receipt保存在本节日志raw_retirement770.json。

第二次部署只改变本地尝试回执目录以保留首次失败，远端命令、代码、计划、预算和checkpoint规则完全不变。12:57:34.912972实启动持久队列PID3160856，port2026、物理GPU0–3、最多四个单卡任务；2025不启动任何模型或训练。初始化在GPU0顺序执行，九个prepare和三组真实batch/prediction配对通过后才运行全部九个M0，全部M0通过才进入正式50轮。没有重跑既有训练、评分或F3报告。

首个定时观察{observed['at']}：controller_present={observed['controller_present']}，status={observed['status']}，phase={observed['phase']}；初始化文件{observed['initialization_count']}/9、配对文件{observed['initial_pair_count']}/3；M0计数{observed['counts']['m0']}，正式计数{observed['counts']['full']}。这些是当前运行事实，不是性能或完整M0通过的替代证据。只读观察器PID25216，初始化/M0按240秒观察，正式50轮按预计耗时降低频率；观察器不重启/控制进程、不执行模型或评分。source-only PASS和F3 integrity warn仍不等于新方案已经成功。
'''
path.write_bytes(text.encode())
shutil.copyfile(path, Path('C:/Users/gb/Desktop/document') / path.name)
files.append(doc)
protected = json.loads((proof / 'publication769_local.json').read_bytes())['protected_files']
assert all(hashlib.sha256((repo / n).read_bytes()).hexdigest() == digest for n, digest in protected.items())
publication = {'previous_head': previous['head'], 'files': sorted(files), 'section': '41.770',
    'doc_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'doc_bytes': path.stat().st_size,
    'status': status['status'], 'training_ports': [2026], 'max_parallel': 4, 'protected_files': protected}
(proof / 'publication770_local.json').write_bytes((json.dumps(publication, indent=2) + '\n').encode())
publisher = private / 'publish_native_launch770.py'
assert not publisher.exists()
source = (private / 'publish_metric_scale_received768_v2.py').read_text(encoding='utf-8')
source = source.replace('768', '770').replace('five_copy767.json', 'five_copy769.json').replace('range(739,769)', 'range(739,771)')
source = source.replace("'logs/metric_feature_scale_complete770_20261003','refine-logs/metric_feature_scale_v1'", "'logs/independent_native_launch770_20261003','refine-logs/independent_native_evidence_v1'")
source = source.replace('Receive all six F3 strict results with original failure provenance', 'Record only2026 native queue launch and exact closed M0 retirement')
source = source.replace('all-six primary text/source intake with fresh reviews pending;', 'actual only2026 native launch, first timed observation and preserved storage failure;')
publisher.write_bytes(source.encode())
ast.parse(source)
print(json.dumps({k:v for k,v in publication.items() if k not in ('files', 'protected_files')}, indent=2))
