from pathlib import Path
import hashlib,json,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp')
folder='evidence/supported_gradient_balance_q1_cpu_failure_20260921';paths=[]
for name in ['launch_supported_balance_cpu_bound_recheck_20260921.py','trifusion_supported_balance_cpu_bound_recheck_launch_20260921.json','observe_supported_balance_cpu_bound_recheck_20260921.py','trifusion_supported_balance_cpu_bound_recheck_observation_1_20260921.json','diagnose_supported_balance_scalar_equations_20260921.py','trifusion_supported_balance_scalar_equations_diagnosis_20260921.json']:
    rel=folder+'/'+name;(p/rel).write_bytes((t/name).read_bytes());paths.append(rel)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
b+='''

#### §41.237 加权梯度范数标量检查未闭环

独立入口q1_cpu_bound_sqrt_recheck通过原配置绑定与ratio精确重放，随后停在原stats第77行：balanced加权后范数平方与wR²||R||²+wA²||A||²+2wRwA<R,A>比较。全六端1560步标量诊断执行95343次close比较，唯一超限为fold0 balanced第71步Mamba，5.0530119215775136与5.05301246766921，差5.460916963073714e-7，原阈值5.053012467669209e-7。诊断仅收集原失败，不是验证PASS；所有失败保留。已由experiment-audit要求的新鲜同族审查agent audit_supported_balance_cpu_replay_failure_20260921独立检查实际FP32运算与标量等式、允许的修正范围；尚无最终意见。没有放宽容差、没有训练重启、没有宣告完整Q1科学结论。原CPU与后续复核分别记录，不改写原pipeline。
'''.encode();(p/master).write_bytes(b);paths.append(master)
tracker='refine-logs/msvr310_supported_gradient_balance_v1/EXPERIMENT_TRACKER.md';b=(p/tracker).read_bytes();old='10:29六端训练完整退出0；CPU ratio精确比较失败，已定位sqrt与幂运算1ULP差异；核验器最小修复，完整重核及审查待完成'
new='六端训练完整；独立sqrt复核通过执行绑定，现加权范数恒等式1/95343标量比较超限；全量诊断归档、独立审查中，未宣告CPU通过'
assert b.count(old.encode())==1;(p/tracker).write_bytes(b.replace(old.encode(),new.encode()));paths.append(tracker)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256((p/master).read_bytes()).hexdigest())
(t/'trifusion_supported_balance_scalar_failure_publication_20260921.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(json.dumps(m))
