from pathlib import Path
import json,hashlib,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp')
paths=['tools/verify_msvr_supported_gradient_balance_stats.py','tools/recheck_msvr_supported_balance_sqrt.py']
folder='evidence/supported_gradient_balance_q1_cpu_failure_20260921'
for name in ['launch_supported_balance_cpu_recheck_20260921.py','trifusion_supported_balance_cpu_recheck_launch_20260921.json','trifusion_supported_balance_cpu_recheck_result_20260921.json']:
    rel=folder+'/'+name;(p/rel).write_bytes((t/name).read_bytes());paths.append(rel)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
b+='''

#### §41.237 执行绑定完整保留的复核入口

第一次修复复核q1_cpu_sqrt_recheck在context被原project_file_sha256拦截，未进入完整统计核验，失败记录保留。恢复原verify_msvr_supported_gradient_balance_stats.py精确执行字节；新增tools/recheck_msvr_supported_balance_sqrt.py独立事后核验入口，先验证原文件SHA，再在进程内仅替换x**0.5为math.sqrt并记录修改前后源码SHA，保留精确比较和所有原检查。原训练配置、全部绑定文件和原pipeline保持；不将失败stage重写为PASS。新的复核仍待完成。
'''.encode();(p/master).write_bytes(b);paths.append(master)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
(t/'trifusion_supported_balance_bound_recheck_publication_20260921.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(json.dumps(m))
