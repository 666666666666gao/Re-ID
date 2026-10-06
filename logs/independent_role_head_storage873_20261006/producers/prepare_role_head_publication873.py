from pathlib import Path
from datetime import datetime
import hashlib,json,shutil,subprocess

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
base=Path('C:/Users/gb/.codex_tmp');proof=base/'foundation_recipe_v1_20261002';evidence=base/'independent_evidence_draft'
prior=json.loads((proof/'four_copy872_2025_pending.json').read_bytes())
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==prior['head']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protected_names=json.loads((proof/'publication872_local.json').read_bytes())['protected_files']
protected={n:sha(repo/n) for n in protected_names};assert len(protected)==10
scope=repo/'refine-logs/independent_role_heads_v1/SOURCE_SCOPE.json';scope_sha=sha(scope)
sources=json.loads(scope.read_text())['source_sha256'];assert len(sources)==385
assert all(sha(repo/n)==d for n,d in sources.items())
launch_exit=json.loads((evidence/'role_head_launch872/EXIT.json').read_bytes());assert launch_exit['exit_code']==1
retired=json.loads((evidence/'role_head_storage_retirement873_r2/REMOTE.json').read_bytes())
transport=json.loads((evidence/'role_head_transport_retirement873/REMOTE.json').read_bytes())
extra=json.loads((evidence/'role_head_unused_f2_retirement873/REMOTE.json').read_bytes())
assert retired['retired_bytes']==704086974 and extra['retired_bytes']==346600965
assert extra['free_after']>=extra['required_start_bytes']==3758096384
archive=repo/'logs/independent_role_head_storage873_20261006';assert not archive.exists()
changed=set()
names=('role_head_launch872','role_head_storage_preflight873','role_head_storage_retirement873',
       'role_head_storage_retirement873_r2','role_head_transport_retirement873','role_head_unused_f2_retirement873')
for name in names:
    for p in (evidence/name).rglob('*'):
        if p.is_file():
            assert p.suffix.lower() not in ('.pth','.pt','.npy','.npz')
            dest=archive/name/p.relative_to(evidence/name);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest);changed.add(dest.relative_to(repo).as_posix())
for name in ('launch_independent_role_heads872.py','inspect_head_storage873.py','retire_head_storage873.py',
             'retire_head_storage873_r2.py','retire_used_transport873.py','retire_unused_f2_best873.py'):
    dest=archive/'producers'/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(base/name,dest);changed.add(dest.relative_to(repo).as_posix())
at=datetime.now().astimezone().isoformat()
doc=repo/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
body=doc.read_text(encoding='utf-8');assert '## §41.873 ' not in body
body+=f'''\n\n## §41.873 — 独立训练头队列第一次启动前空间门失败；旧二进制清理后重新具备启动空间（{at}）

§872首次启动检查在固定3,758,096,384B空间断言处退出1；尚未创建campaign/journal、初始化模型、M0或正式训练。原失败保留，不称训练失败或指标负结果。23:23:58复查free3,055,337,472B，比23:10记录少约840MB；变化来源未定位，不归咎其他项目。

23:32:29退役两份自己的闭合非当前依赖PTH，共704,086,974B：来源SIM的MSVR global-best由保留F3 metric-raw的四项指标支配；原N1-low MSVR完成50轮但首次数值验收失败、无正式receipt，作为已停用失败二进制退役，不补正式分数或改判。原训练史、首次失败评价距离与记录保留。首次清理资格代码误要求旧N1从未保存的training-best距离文件，在创建journal/删除前失败；后续根据实际旧保存入口，读取CPU checkpoint metadata与原失败official距离校验，原错误保留。旧N1没有training-best距离不能写成该数组仍在，退役后这份权重不可直接重放。

已用的117份自己Git传输bundle字节匹配本地留存副本后清理，共882,883,245B；它们位于/tmp另一设备，未增加/data训练盘空间，不能计入有效训练盘释放。仓库提交及本地bundle保留。

随后按用户“只保留指标最好的权重”的授权，退役不再参与当前研究的F2 raw-MSVR best二进制346,600,965B；保留F3 metric-raw best。后者mAP/R1/R10较高，R5较低，这不是四项支配，也不是跨配方单因素因果比较；原50轮、正式receipt、训练best与official距离均留存。本次三份训练盘清理共1,050,687,939B；{extra['at']}实际free{extra['free_after']:,}B，超过固定启动门，未放宽空间或数值门槛。

独立fused训练头科学计划、385份来源文件与RAW187当前输入全不变。初值CPU见证通过仍不代表真实pair/M0或性能通过；目前新三端尚未正式开始。保持201→MSVR→100，每端8M0/fresh50/first-strict、同一mAP-best全部CMC。只26GPU0/1，不探测25、其他卡、功率或温度；NN期间不改source/Git。空间证据与全部失败/清理journal归档于logs/independent_role_head_storage873_20261006，目标ACTIVE/UNMET。
'''
doc.write_text(body,encoding='utf-8');changed.add(doc.relative_to(repo).as_posix())
(Path('C:/Users/gb/Desktop/document')/doc.name).write_bytes(doc.read_bytes())
manifest=repo/'MANIFEST.md';original_manifest_sha=sha(manifest)
with manifest.open('a',encoding='utf-8') as f:
    f.write(f'| {at} | /run-experiment | logs/independent_role_head_storage873_20261006/ | preparation | 独立训练头启动前空间失败与授权清理记录；未产生训练结果 |\n')
changed.add('MANIFEST.md')
assert sha(scope)==scope_sha and all(sha(repo/n)==d for n,d in sources.items())
assert all(sha(repo/n)==d for n,d in protected.items())
publication=dict(previous_head=prior['head'],files=sorted(changed),protected_files=protected,
    original_manifest_sha256=original_manifest_sha,source_count=385,source_scope_sha256=scope_sha,
    remote_sparse_roots=['logs/independent_role_head_storage873_20261006'],
    immutable_archive_roots=['logs/independent_role_head_storage873_20261006'],
    doc_sha256=sha(doc),section='41.873',at=at)
out=proof/'publication873_local.json';assert not out.exists();out.write_text(json.dumps(publication,indent=2)+'\n')
print(json.dumps(dict(files=len(changed),doc_sha256=sha(doc),free_after=extra['free_after'])))
