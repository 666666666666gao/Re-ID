from pathlib import Path
import json,hashlib,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp');folder='evidence/transport_cleanup_postq1_20260921';(p/folder).mkdir();paths=[]
for name in ['inventory_trifusion_transport_postq1_20260921.py','inventory_trifusion_transport_postq1_20260921.json','plan_transport_cleanup_postq1_20260921.py','trifusion_transport_retention_postq1_20260921.json','cleanup_verified_trifusion_transport_postq1_20260921.py','trifusion_transport_cleanup_receipt_postq1_20260921.json']:
    rel=folder+'/'+name;(p/rel).write_bytes((t/name).read_bytes());paths.append(rel)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
b+='''

#### §41.238 11:07归档后磁盘整理

在远端HEAD与GitHub9ea4025一致、本地同名副本SHA/字节一致且bundle全部引用已合入当前HEAD的条件下，删除/root/autodl-tmp/trifusion-v2/transport内13个冗余.bundle，逻辑17721502B，目录无符号链接越界；原逐文件清单和删除回执保存evidence/transport_cleanup_postq1_20260921。项目卷空闲1405218816→1422946304B；输出卷盘点4430852096B。权重删除0，初始化、终点、数组和审查证据保留。完整Q1独立审查仍进行中，不启动后继训练。
'''.encode();(p/master).write_bytes(b);paths.append(master)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
(t/'trifusion_transport_cleanup_postq1_publication_20260921.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(paths=len(paths))))
