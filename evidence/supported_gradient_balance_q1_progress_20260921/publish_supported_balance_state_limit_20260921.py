from pathlib import Path
import hashlib,json,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp')
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
assert prior=='b54a9b923a754cc724e33f0a569d4077444cf099292d005c35703b2034241cc7'
b+='''

#### §41.236 优化器历史可恢复性边界补核

本轮读取R2实际checkpoint调用及保存函数，保存函数与执行1381639 Git blob逐字一致。终点只保存role模型状态、baseline别名和绑定/身份/配置信息，不保存AdamW state_dict或一/二阶矩；gradient_balance_state是范数EMA，actual_parameter_updates是比较统计。因而当前合同支持终点检索重载及已登记更新统计，不支持仅从这些产物精确恢复原AdamW历史或分解任务更新。未额外读取模型、未更改运行中的保存规则、未重跑已结束端点。后继如需真实状态的反事实分析，必须在新来源流程预先登记捕获；不能用新建空优化器代替原状态。详见refine-logs/msvr310_supported_gradient_balance_v1/ROLE_LOSS_DEPENDENCY_NOTE_20260921.md。该限制不改变当前Q1合同及已完成终点检索的有效性。
'''.encode();(p/master).write_bytes(b)
rel='evidence/supported_gradient_balance_q1_progress_20260921/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes())
paths=[master,'refine-logs/msvr310_supported_gradient_balance_v1/ROLE_LOSS_DEPENDENCY_NOTE_20260921.md',rel]
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
(t/'trifusion_supported_balance_state_limit_publication_20260921.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
print(json.dumps(dict(paths=len(paths),master_sha256=m['new_master_sha256'])))
