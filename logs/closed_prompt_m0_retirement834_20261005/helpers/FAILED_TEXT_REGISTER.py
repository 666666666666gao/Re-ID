"""Archive the completed storage retirement, preserving current scientific files."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

repo = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
private = Path('C:/Users/gb/.codex_tmp')
base = private / 'independent_evidence_draft'
proof = private / 'foundation_recipe_v1_20261002'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
previous = json.loads((proof / 'four_copy833_2025_pending.json').read_bytes())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() == previous['head']
assert subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=repo) == b''
assert not (proof / 'publication834_local.json').exists()
protected = json.loads((proof / 'publication833_local.json').read_bytes())['protected_files']
assert all(sha(repo / name) == digest for name, digest in protected.items())
packet = base / 'closed_prompt_m0_retirement834'
assert json.loads((packet / 'EXIT.json').read_bytes())['exit_code'] == 0
result = json.loads((packet / 'RESULT.json').read_bytes())
plan = json.loads((packet / 'PLAN.json').read_bytes())
assert result['status'] == 'SIX_CLOSED_PROMPT_M0_PROBES_RETIRED'
assert result['retired_files'] == len(plan['targets']) == 6 and result['retired_bytes'] == 66536808
assert result['formal_best_and_receipts_unchanged'] and result['current_six_probes_unchanged']
assert result['current_scientific330_sources_unchanged'] and result['current_native_pid_still_same']
assert plan['active_training_pid'] == 3215669 and plan['active_training_start_ticks'] == 38149139
assert len(plan['current_required_probes']) == 6
archive_name = 'logs/closed_prompt_m0_retirement834_20261005'
archive = repo / archive_name
assert not archive.exists()
shutil.copytree(packet, archive / 'completed_retirement')
(archive / 'helpers').mkdir()
for name in ('prepare_closed_prompt_m0_retirement834.py', 'retire_closed_prompt_m0_834.py'):
    shutil.copyfile(private / name, archive / 'helpers' / name)
shutil.copyfile(proof / 'four_copy833_2025_pending.json', archive / 'publication833_four_copy_proof.json')
manifest = {p.relative_to(archive).as_posix(): sha(p) for p in archive.rglob('*') if p.is_file()}
(archive / 'MANIFEST.json').write_text(json.dumps(dict(status='CLOSED_PROMPT_STORAGE_RETIREMENT_COMPLETE',
    at=datetime.now().astimezone().isoformat(), files=manifest), indent=2) + '\n', encoding='utf-8')
intro = ('§41.834：已按确切路径、回执与实存SHA退役旧prompt六份M0探针，释放66,536,808B。'
         '正式best/距离/回执、当前六份M0与330科学source保持不变。'
         '00:24:05再次核对原RGBNT100 native PID3215669/ticks38149139存活；正式仍5/6，原01:00观察不变。'
         '只26GPU0/1，无功率/温度操作；Goal active/unmet。')
section = '\n\n## 41.834 已闭合prompt实验的临时权重退役\n\n' + intro + '\n\n'
section += f'''本次限定旧`logs/prompt_role_state_20260930`的reset/carry×三数据集六端。原父任务6/6和每个子任务M0、fresh50、首次evaluate都已COMPLETE/exit0，原accepted矩阵包含完整合法图库重算及任务标量验证；本次只核对已保存文本/模型SHA，不调用旧verify/report、不做模型前向。

删除前实际核对六端M0成功回执、reload为0、非零梯度覆盖、probe实存SHA、正式50轮/唯一best/距离/评价回执，以及原配对结果和完整报告文本SHA。当前两个control seal的正式artifact集合和初始化依赖不含这些候选；330科学源与当前六份M0依赖均实际核对。Linux绝对路径使用PurePosixPath准备，删除范围逐一限定本项目trained-model下的六个`prompt_role_state_20260930_prompt_*_seed42_m0/m0_reload_probe.pth`。

远端先写PLAN，再逐文件记录原SHA/bytes/退役时间，最后复核旧正式artifact、当前六份探针、源代码和控制seal。原native训练PID及start ticks在删除前后均相同，没有重启、重新选择best或改变实验条件。

实际完成：{result['completed_at']}。退役6文件/66,536,808B，磁盘可用从{result['disk_free_before']}B到{result['disk_free_after']}B。精确释放量取文件大小之和；并行训练及同盘其他写入使free变化不能直接全部归到本次删除。没有删除作者权重、必要初始化、旧正式best或当前六端探针。

这些旧M0二进制已不可直接重放，原M0成功记录、训练轨迹和首次评价仍保留；不能因删除后旧验证器需要探针而重跑或追认失败。这是存储收尾，不是算法创新或新增性能。固定best v2候选仍未登记/上传/执行，当前原六端及唯一CPU报告未全闭合前不启动诊断。原01:00里程碑观察保持，2025已记录的I/O镜像问题仍pending，不访问其GPU或尝试恢复；只同步本地/Desktop/GitHub/2026文本。
'''
section += f'\n证据：`{archive_name}`。\n'
docname = 'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
docpath = repo / docname
desktop = Path('C:/Users/gb/Desktop/document') / docpath.name
assert sha(docpath) == sha(desktop) == previous['doc_sha256']
doc = docpath.read_text(encoding='utf-8')
assert '## 41.834 ' not in doc
docpath.write_text(doc + section, encoding='utf-8')
goalpath = repo / 'refine-logs/CURRENT_GOAL.md'
goal = goalpath.read_text(encoding='utf-8')
old = next(line for line in goal.splitlines() if line.startswith('更新：'))
goal = goal.replace(old, '更新：2026-10-05 ' + intro, 1)
old_disk = '23:32:52按合同退休六个已闭合视觉起点M0共72,616,692B，实测空闲3,667,312,640B；其正式best与当前六份必需探针不变，原队列大于2GiB前置检查保持。两次删除前文件名/Windows路径准备失败已保留，均未删除；没有模型或环境修订。'
assert old_disk in goal
goal = goal.replace(old_disk, '00:24:05按合同再退役旧prompt六份M0共66,536,808B，实测空闲5,255,360,512B。其正式best/回执、当前六份必需探针与330科学源不变，原native PID/start ticks相同。原队列大于2GiB前置检查保持；原两次视觉起点准备失败及其成功退役记录均保留。没有模型、环境或功温修订。', 1)
goalpath.write_text(goal, encoding='utf-8')
trackername = 'refine-logs/global_task_role_v1/EXPERIMENT_TRACKER.md'
with (repo / trackername).open('a', encoding='utf-8') as stream:
    stream.write('\n\n' + intro + '\n当前六端M0仍保留至原唯一CPU报告与全部接收闭合。\n')
desktop.write_bytes(docpath.read_bytes())
owned = [docname, 'refine-logs/CURRENT_GOAL.md', trackername] + [p.relative_to(repo).as_posix() for p in archive.rglob('*') if p.is_file()]
assert all(sha(repo / name) == digest for name, digest in protected.items())
(proof / 'publication834_local.json').write_text(json.dumps(dict(previous_head=previous['head'], section='41.834', files=owned,
    protected_files=protected, immutable_archive_roots=[archive_name], doc_sha256=sha(docpath)), indent=2) + '\n', encoding='utf-8')
source = (private / 'publish_sealed_role_moments833.py').read_text(encoding='utf-8')
old_boundary = 'Closed twelve-model CPU feature moments completed24splits unchangedSHA;accepted RGBNT100 semantic original50curve archived. Descriptive only,no new forward/ranking/test-statistics deployment/training decision. Current formal5/6,M0 6/6;original native queue and01:00milestone unchanged.330scientificsources unchanged;332fixedbest remains local/unexecuted.'
new_boundary = 'Six closed prompt M0 probes retired66536808B with actual receipts/path/SHA;old formal best/receipts,current six probes and330scientificsources unchanged. Original native PID3215669/ticks38149139 confirmed before/after00:24:05. Formal5/6,M0 6/6;01:00milestone unchanged.332fixedbest remains local/unexecuted. Old probe direct replay unavailable;original results preserved.'
for old, new in (
    ('publication833_local.json', 'publication834_local.json'), ('four_copy832_2025_pending.json', 'four_copy833_2025_pending.json'),
    ('commit833.json', 'commit834.json'), ('target833.bundle', 'target834.bundle'), ('range(739,834)', 'range(739,835)'),
    ('Describe sealed role correction moments and accepted RGBNT100 trajectory', 'Retire six closed prompt M0 probes with preserved formal evidence'),
    ('logs/sealed_role_correction_moments833_20261005', archive_name), ("'section':'41.833'", "'section':'41.834'"),
    ("four_copy833_2025_pending.json').write_text", "four_copy834_2025_pending.json').write_text"),
    ('/tmp/trifusion_target833_20261005.bundle', '/tmp/trifusion_target834_20261005.bundle'),
    (old_boundary, new_boundary)):
    assert old in source, old
    source = source.replace(old, new)
ast.parse(source)
target = private / 'publish_closed_prompt_retirement834.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-05.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n' + datetime.now().astimezone().isoformat() + ' ' + intro + '\nPublication834 prepared; root only actual SHA-guarded six old prompt M0 removal. No current inference/training modification.\n')
print(json.dumps(dict(status='CLOSED_PROMPT_RETIREMENT834_REGISTERED_FOR_PUBLICATION', files=len(owned), at=datetime.now().astimezone().isoformat())))
