from pathlib import Path
import hashlib,json,subprocess
p=Path('C:/Users/gb/.trifusion_github_publish_22c3bee');t=Path('D:/Program Files/UserCache/gb/codex/tmp');a=t/'trifusion_supported_balance_cpu_equations_audit_20260921'
folder='evidence/supported_gradient_balance_cpu_equations_audit_20260921';(p/folder).mkdir()
paths=['tools/recheck_msvr_supported_balance_sqrt.py']
for f in sorted(a.iterdir()):
    assert f.is_file() and f.suffix in ('.py','.json','.stderr','.md')
    f.read_bytes().decode('utf-8');rel=folder+'/'+f.name;(p/rel).write_bytes(f.read_bytes());paths.append(rel)
rel=folder+'/REPAIR_SCOPE.md';(p/rel).write_text('Fresh same-family gpt-6-astra/max review approved bounded verifier-only sqrt and FP32 coefficient arithmetic repair. Full report pending; no full Q1 or scientific pass claimed. All original thresholds and execution bindings remain unchanged. See primary scalar audit and minimum repair replay for 1560 steps, 4680 role rows and 9360 actual PyTorch CPU scalar conversion checks. Candidate transformed source SHA 4f21a5ec2ace07ed446d56c39881b3e3ce7c3783ff6edfa0bad26bc0b0e85231. Saved parameter-gradient witnesses are not independent gradient regeneration.\n',encoding='utf-8');paths.append(rel)
master='docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md';b=(p/master).read_bytes();prior=hashlib.sha256(b).hexdigest()
b+='''

#### §41.237 独立算术审查与最小修正

新鲜gpt-6-astra/max同族审查支持仅修事后核验：开方使用math.sqrt；第77行加权范数预期值使用实际张量乘法转换后的FP32系数。出错行wR由1.0421102637890658转为1.0421102046966553，wA由0.9578897362109341转为0.9578897356987，修正后范数平方残差4.67142147e-9，小于原阈值。PyTorch2.5.1+cu121 CPU实际转换检查9360次；完整六端1560步的原verify_balance断言在两处修正后全部通过；close容差、控制器精确检查与M0参考门槛均未放宽。合成单元素反例仅验证运算语义，不冒充模型实验。证据evidence/supported_gradient_balance_cpu_equations_audit_20260921。独立审查为same-family/provisional，只批准算术核验修正；完整CPU/Q1审查仍待完成。原执行文件、配置、训练、检查点与pipeline保持不变。
'''.encode();(p/master).write_bytes(b);paths.append(master)
rel=folder+'/'+Path(__file__).name;(p/rel).write_bytes(Path(__file__).read_bytes());paths.append(rel)
m=dict(old_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),paths=paths,prior_master_sha256=prior,new_master_sha256=hashlib.sha256(b).hexdigest())
(t/'trifusion_supported_balance_fp32_recheck_publication_20260921.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(files=len(paths))))
