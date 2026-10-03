from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import shutil
import subprocess
import unittest

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
proof = Path('C:/Users/gb/.codex_tmp/foundation_recipe_v1_20261002')
previous = json.loads((proof / 'five_copy791.json').read_bytes())
protected = json.loads((proof / 'publication791_local.json').read_bytes())['protected_files']
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous['head']
assert not (proof / 'publication792_local.json').exists()
queue_name = 'tools/queue_independent_native_evidence.py'
queue_source = (repo / queue_name).read_text(encoding='utf-8')
tree = ast.parse(queue_source)
compile(queue_source, queue_name, 'exec')
archive_root = 'logs/native_resource_scope792_20261003'
archive = repo / archive_root
assert archive.is_dir() and sorted(p.name for p in archive.iterdir()) == ['INITIAL_CPU_CHECK_FAILURE.json', 'SECOND_CPU_CHECK_FAILURE.json', 'initial_failed_cpu_check.py', 'second_failed_cpu_check.py']
simulation = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/resource_scope792_simulation_v3')
assert not simulation.exists()
simulation.mkdir()

calls, sleeps, pending_history = [], [], []
class Process:
    def __init__(self, args, **_kwargs):
        calls.append(args)
        self.pid = 1000 + len(calls)
    def poll(self):
        return 0

def require_sources(_campaign):
    raise ValueError('CPU sentinel before any source or model execution')

fake_base = SimpleNamespace(
    require_sources=require_sources,
    queue=SimpleNamespace(
        stamp=lambda: 'CPU_SIMULATION',
        write=lambda _path, state: pending_history.append(sum(job['status'] == 'RUNNING' for job in state['jobs'])),
        child_campaign=lambda campaign, phase, dataset, variant: campaign / f'{phase}_{dataset}_{variant}',
    ),
    require_complete=lambda _child, _dataset: {'status': 'CPU_SIMULATION_COMPLETE'},
)
environment = {
    'ROOT': simulation, 'RESERVE_BYTES': 2 * 1024**3, 'base': fake_base,
    'subprocess': SimpleNamespace(check_output=lambda _args, **_kwargs: '0, 0\n1, 0\n2, 0\n3, 0\n', Popen=Process, STDOUT=-2),
    'shutil': SimpleNamespace(disk_usage=lambda _path: SimpleNamespace(free=100 * 1024**3)),
    'time': SimpleNamespace(sleep=lambda value: sleeps.append(value)),
    'start_command': lambda _campaign, job, gpu: ['CPU_ONLY_FAKE_WORKER', '--gpu', str(gpu)],
}
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('worker', 'coordinate', 'run_phase')]
assert len(functions) == 3
exec(compile(ast.Module(body=functions, type_ignores=[]), queue_name, 'exec'), environment)
checks = unittest.TestCase()
for name in ('worker', 'coordinate'):
    for gpu in (2, 3):
        with checks.assertRaises(AssertionError):
            environment[name](SimpleNamespace(gpu=gpu))
for gpu in (0, 1):
    with checks.assertRaisesRegex(ValueError, 'CPU sentinel'):
        environment['worker'](SimpleNamespace(gpu=gpu, campaign=simulation))
jobs = [{'phase': 'full', 'dataset': 'CPU_DATASET', 'variant': str(i), 'status': 'PENDING'} for i in range(4)]
result = environment['run_phase'](simulation, {'jobs': jobs}, 'full')
assert result == 0 and [int(command[-1]) for command in calls] == [0, 1, 0, 1]
assert all(job['status'] == 'COMPLETE' for job in jobs)
assert max(pending_history) == 2 and sleeps == [240, 240]
runtime_check = {
    'at': datetime.now().astimezone().isoformat(), 'queue_sha256': hashlib.sha256((repo / queue_name).read_bytes()).hexdigest(),
    'free_gpu_simulation': [0, 1, 2, 3], 'assigned_physical_gpus': [int(command[-1]) for command in calls],
    'maximum_active_jobs': max(pending_history), 'worker_and_coordinator_reject_gpu2_3_before_work': True,
    'worker_gpu0_1_reaches_source_check': True, 'poll_seconds': sleeps,
    'model_imports': 0, 'gpu_queries': 0, 'cuda_execution': 0, 'training_started': False,
    'boundary': 'Isolated CPU scheduler simulation only; no M0, memory, gradient, model or training acceptance.',
}
(archive / 'RESOURCE_SCOPE_CHECK.json').write_text(json.dumps(runtime_check, indent=2) + '\n', encoding='utf-8')
shutil.copyfile(Path(__file__), archive / 'prepare_resource_scope_publication.py')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plan_name = 'refine-logs/independent_native_evidence_v1/EXPERIMENT_PLAN.md'
plan_path = repo / plan_name
plan = plan_path.read_text(encoding='utf-8')
old = 'physical GPU0–3; maximum four single-GPU jobs'
assert plan.count(old) == 1
plan = plan.replace(old, 'physical GPU0/1; maximum two single-GPU jobs')
notice = '\nCurrent resource amendment §41.792: latest user instruction permits only2026 physicalGPU0/1. Queue coordinator and direct workers now rejectGPU2/3 and cap active single-GPU jobs at2. This updates the resource scope only. Old FAIL/STOP and the all-nine M0 prerequisite remain unchanged; execution-policy clarification is pending, so this amendment does not start or authorize training.\n'
plan = plan.replace('\n## Question and fixed foundation', notice + '\n## Question and fixed foundation', 1)
versioned_plan = plan_path.with_name(f'EXPERIMENT_PLAN_{stamp}.md')
assert not versioned_plan.exists()
versioned_plan.write_text(plan, encoding='utf-8')
shutil.copyfile(versioned_plan, plan_path)

tracker_name = 'refine-logs/independent_native_evidence_v1/EXPERIMENT_TRACKER.md'
tracker_path = repo / tracker_name
tracker = tracker_path.read_text(encoding='utf-8') + '\nSection792 resource execution scope aligned with latest user instruction: coordinator/direct workers permit only2026physicalGPU0/1, max2single-GPU jobs, poll240seconds. CPU simulation with allfourcardsfree assigns0/1/0/1 and rejects directGPU2/3; no GPU/model/M0/training executed. Original all-nine prerequisite and historical FAIL/STOP unchanged; requested execution-policy clarification remains pending. This is resource-scope synchronization, not backward/memory repair or scientific efficacy evidence.\n'
versioned_tracker = tracker_path.with_name(f'EXPERIMENT_TRACKER_{stamp}.md')
assert not versioned_tracker.exists()
versioned_tracker.write_text(tracker, encoding='utf-8')
shutil.copyfile(versioned_tracker, tracker_path)

manifest_name = 'MANIFEST.md'
with (repo / manifest_name).open('a', encoding='utf-8', newline='') as stream:
    for name in (str(versioned_plan.relative_to(repo)).replace('\\', '/'), plan_name,
                 str(versioned_tracker.relative_to(repo)).replace('\\', '/'), tracker_name, queue_name,
                 archive_root + '/RESOURCE_SCOPE_CHECK.json'):
        stream.write(f'\n| {datetime.now().astimezone().isoformat()} | /experiment-plan | {name} | implementation | §41.792 仅GPU0/1资源范围同步；CPU调度检查；旧M0/STOP不改、训练未启动 |\n')

doc_name = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
doc_path = repo / doc_name
assert hashlib.sha256(doc_path.read_bytes()).hexdigest() == previous['doc_sha256']
doc = doc_path.read_bytes().decode('utf-8')
headers = [line for line in doc.splitlines() if line.startswith('**当前进度')]
assert len(headers) == 1 and '## 41.792 ' not in doc
doc = doc.replace(headers[0], '**当前进度（§41.792，2026-10-03晚间）：** 当前Goal继续执行独立证据提取—可靠对应协作—判别信息保留。旧队列仍允许GPU0–3的范围冲突已同步为仅2026物理GPU0/1、最多2个单卡任务；CPU调度检查通过，没有GPU/M0/模型/训练执行。停止工程修复；旧FAIL/STOP与九端全部M0前置条件不改，执行口径问题等待用户答复。新独立读取仍无正式50轮结果，Goal ACTIVE_UNMET。')
doc += '''

## 41.792 将当前Goal的GPU0/1范围落实到队列，训练口径仍待明确（2026-10-03晚间）

上一轮§791完成了真实新科学证据；本轮核对当前源码时发现，原九端队列仍按GPU0–3挑选空卡、最多启动4个单卡任务，且直接worker入口继承了0–3参数范围。这与用户后来明确“只用26的0、1号显卡”及§790当前Goal冲突。最小修改仅为队列在选择空卡、并发上限、coordinator和直接worker入口限定0/1；模型、作者配方、batch、精度、种子、梯度容差、九端M0及正式报告条件均未改变。计划只同步资源范围，历史已封存源码/manifest/失败回执不修改。

CPU隔离检查的前两次分别因模拟subprocess缺少STDOUT常量、轮询次数预期错误而失败，均保留原脚本与实际错误；只修正检查脚本、不改生产训练路径。最终检查把四张卡全部模拟为空闲，四个待执行任务实际分配为0/1/0/1、最多2个活动任务，GPU2/3在coordinator与直接worker入口均先拒绝；轮询仍240秒。检查未导入模型/torch、未调用nvidia-smi/CUDA、未创建真实子进程或训练，不能证明B128容量、反向一致性、M0通过或识别效果。记录为logs/native_resource_scope792_20261003/RESOURCE_SCOPE_CHECK.json。当前生产资源范围明确，但并未启动队列。

用户要求停止工程修复仍优先；本次只是落实已有资源指令，不恢复已撤回的边界诊断、显存或算子修复。训练前置条件的问答尚无回复：旧“九个M0全部通过才开始任何正式端”仍有效，是否改为逐端研究训练及保留有限值/真实更新/完整保存重载/评分检查，不能由等待时间或预选项代替决定。新native正式结果仍0，科学Goal未达成；也不能把这次调度范围通过写成新方法有效。
'''
doc_path.write_bytes(doc.encode('utf-8'))
shutil.copyfile(doc_path, Path('C:/Users/gb/Desktop/document') / doc_path.name)
assert all(hashlib.sha256((repo / name).read_bytes()).hexdigest() == digest for name, digest in protected.items())
files = [queue_name, plan_name, tracker_name, manifest_name, doc_name,
         str(versioned_plan.relative_to(repo)).replace('\\', '/'), str(versioned_tracker.relative_to(repo)).replace('\\', '/')]
files += sorted(str(p.relative_to(repo)).replace('\\', '/') for p in archive.iterdir())
publication = {'previous_head': previous['head'], 'section': '41.792', 'files': sorted(files),
               'protected_files': protected, 'immutable_archive_roots': [archive_root],
               'doc_sha256': hashlib.sha256(doc_path.read_bytes()).hexdigest()}
(proof / 'publication792_local.json').write_text(json.dumps(publication, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'section': '41.792', 'files': len(files), 'assigned_cpu_simulation_gpus': runtime_check['assigned_physical_gpus'], 'training_started': False}))
