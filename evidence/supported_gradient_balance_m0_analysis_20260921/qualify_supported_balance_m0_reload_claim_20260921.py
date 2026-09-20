from pathlib import Path
import hashlib,json

repo=Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
path=repo/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
data=path.read_bytes();before=hashlib.sha256(data).hexdigest()
assert before=='9bd89fa5e541b3b3d5267d8a22301655d93622934d99bfb309d889edc9fe9de3'
old='6容量端各8步及2过拟合端各100步全部完成，共248更新；各端203/203累计非零、overflow0、冻结/Signal状态不变、角色更新/严格重载通过。'
new='6容量端各8步及2过拟合端各100步全部完成，共248更新；八个运行均203/203累计非零、overflow0、冻结/Signal状态不变及角色更新检查通过。六个容量端检查点严格重载通过；两个过拟合运行未保存终点检查点，也没有执行严格重载。'
assert data.count(old.encode('utf-8'))==1
data=data.replace(old.encode('utf-8'),new.encode('utf-8'),1)
# Correct the current overview using the already sealed full cross-scene Q1 result.
old2='两项车辆训练比较均已完成；RGBNT100内部完整比较支持三角色增益，MSVR310尚未超过Signal。'
new2='两项原V8车辆训练比较均已完成：RGBNT100内部完整比较支持三角色增益，MSVR310原V8比较未超过Signal；后续跨scene Smooth-AP的完整内部Q1 fused已比Signal高0.2761pp，但仍未通过原晋级条件。'
assert data.count(old2.encode('utf-8'))==1
data=data.replace(old2.encode('utf-8'),new2.encode('utf-8'),1)
path.write_bytes(data)
result=dict(path=str(path),before_sha256=before,after_sha256=hashlib.sha256(data).hexdigest(),
    reload_qualifier_before=old,reload_qualifier_after=new,current_overview_before=old2,current_overview_after=new2,
    source_or_gate_changes=False)
output=Path('D:/Program Files/UserCache/gb/codex/tmp/supported_balance_m0_claim_qualification_20260921.json')
assert not output.exists();output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
