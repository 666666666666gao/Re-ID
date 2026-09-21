from pathlib import Path
import json,math
root=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/q1')
files=[];differences=[];checked=0
for path in sorted(root.rglob('*.jsonl')):
    count=0
    for line in path.open():
        row=json.loads(line)
        if 'gradient_balance' not in row: continue
        count+=1
        for role,b in row['gradient_balance'].items():
            if not b['supported']: continue
            checked+=1
            s=b['after'];x=(s['auxiliary']+1e-12)/(s['rank']+1e-12)
            power=min(4.,max(.25,x**.5));sqrt=min(4.,max(.25,math.sqrt(x)))
            assert b['ratio']==sqrt,(str(path),count,role,b['ratio'],sqrt)
            if b['ratio']!=power:
                differences.append(dict(file=str(path.relative_to(root)),row=count,role=role,recorded=b['ratio'],power=power,sqrt=sqrt,absolute_difference=abs(power-sqrt),ulps=abs(power-sqrt)/math.ulp(sqrt)))
    if count:files.append(dict(file=str(path.relative_to(root)),steps=count))
print(json.dumps(dict(files=files,supported_role_rows_checked=checked,all_recorded_ratios_exactly_match_runtime_math_sqrt=True,differences=differences),indent=2))
