from pathlib import Path
import ast,hashlib,json,subprocess
t=Path('D:/Program Files/UserCache/gb/codex/tmp');p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
archive='evidence/transport_cleanup_20260921';(p/archive).mkdir()
paths=[]
for name in ('inventory_trifusion_transport_20260921.json','plan_transport_cleanup_20260921.py','trifusion_transport_retention_20260921.json','cleanup_verified_trifusion_transport_20260921.py','trifusion_transport_cleanup_receipt_20260921.json'):
    (p/archive/name).write_bytes((t/name).read_bytes());paths.append(archive+'/'+name)
inv='inventory_trifusion_transport_20260909.py';(p/archive/inv).write_bytes((p/'evidence/transport_cleanup_20260909'/inv).read_bytes());paths.append(archive+'/'+inv)
rel='evidence/supported_gradient_balance_terminal_preparation_20260921/intake_trifusion_supported_balance_phase_20260921.py'
b=(p/rel).read_bytes();assert b==(t/Path(rel).name).read_bytes();tree=ast.parse(b)
# Parse both the local helper and its embedded remote Python payload.
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='script' for z in n.targets))
payload=node.value.func.value.value
ast.parse(payload.replace('MODE',repr('q1')))
assert "if mode=='q1':paths.append(root/'pipeline.json')" in payload
paths.append(rel)
workflow='evidence/supported_gradient_balance_terminal_preparation_20260921/supported_balance_terminal_workflow_20260921.md'
w=(p/workflow).read_bytes()
w+=b'\n## Terminal intake correction (2026-09-21 07:10)\n\nThe Q1 intake now also copies the byte-exact terminal `pipeline.json`; the generic all-ranking replay requires that file and its terminal summary/CPU hashes. Previously only its parsed object was embedded in the inventory. Both helper and embedded payload parse successfully; no incomplete Q1 intake or ranking replay was run.\n\nAfter complete intake, run the unchanged `tools/audit_msvr_paired_ranking_text.py` with `--candidate balanced --cpu-status PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_Q1`, the actual complete summary SHA, the intake root, and a new output directory. It checks all 600 queries/60 identities, 2,069,520 ranking positions, both registered gate groups, and writes 3000 query-output plus 300 identity-output rows. This text replay does not load arrays/models and is executor verification, not independent review.\n'
(p/workflow).write_bytes(w);paths.append(workflow)
proof=dict(status='SYNTAX_AND_INTERFACE_CHECK_ONLY',helper_sha256=hashlib.sha256(b).hexdigest(),local_and_embedded_remote_ast=True,terminal_pipeline_copied=True,candidate='balanced',cpu_status='PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_Q1',q1_intake_executed=False,q1_ranking_executed=False,training_changes=False)
rel='evidence/supported_gradient_balance_terminal_preparation_20260921/terminal_interface_check_20260921.json';(p/rel).write_bytes((json.dumps(proof,indent=2)+'\n').encode());paths.append(rel)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='e01d6c2dcba7d566564f92805b4dc4195064202ab20b52c5e40b8507df74fffa'
b+='''

#### §41.234 接续：终态接口与磁盘维护（07:10）

07:08:31原wrapper42758/Q1 44804持续运行，首候选完成9/20epoch、1/6完整端点，GPU7578MiB/100%，输出盘5207883776B。未改变训练。检查终态工具发现文本接收清单仅嵌入pipeline对象，通用排名复算却要求独立pipeline.json；已补接收该原始文件并解析本地/嵌入远端Python语法。原排名复算可直接指定candidate=balanced及PASS_COMPLETE_SUPPORTED_GRADIENT_BALANCE_Q1，等待完整终态后执行，不重复实现指标，不把接口准备当成核验通过。

磁盘盘点48个已同步Git bundle，共104067074B；全部具备本地相同SHA副本、引用提交在远端HEAD11678ca祖先链内，GitHub HEAD相同。删除前再次核对精确绝对路径位于transport、非符号链接、大小/SHA、提交存在与祖先关系，仅删除这48个明确文件。07:10:33完成，主卷1472622592→1576755200B；权重删除0、训练输出未动。逐文件计划/回执及执行脚本归档evidence/transport_cleanup_20260921。三数据集Goal保持ACTIVE/UNMET。
'''.encode()
(p/master).write_bytes(b);paths.append(master)
rel=archive+'/publish_supported_balance_terminal_fix_20260921.py';(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
j=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
(t/'trifusion_supported_balance_terminal_fix_publication_20260921.json').write_bytes((json.dumps(j,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),master_sha256=j['new_master_sha256'],syntax_checked=True)))
